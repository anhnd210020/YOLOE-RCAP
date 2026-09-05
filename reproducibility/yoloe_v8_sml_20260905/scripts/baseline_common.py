"""Run metadata and immutable configuration; all generated artifacts stay in baseline_logs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/workspace/YOLOE-RCAP')
LOGS = ROOT / 'baseline_logs'
STATE = LOGS / 'baseline_state.json'
EXPECTED_COMMIT = '40cd606cabdbe2b566d6f14a6b162c89206e9a1b'
CHECKPOINT = ROOT / 'pretrain/yoloe-v8s-seg.pt'
DATASET = Path('/workspace/datasets/lvis')
ANNOTATIONS = DATASET / 'annotations/lvis_v1_minival.json'
PREDICTIONS = LOGS / 'official_fixed_ap/predictions.json'
EXPECTED_AP = 27.9
# An absolute tolerance of 0.5 AP points is a declared reporting criterion only.
AP_TOLERANCE = 0.5
CONFIG = dict(
    data='ultralytics/cfg/datasets/lvis.yaml', imgsz=640, batch=1,
    split='minival', rect=False, conf=0.001, iou=0.7, max_det=1000,
    save_json=True, half=False, load_vp=False, device=0,
    project=str(LOGS), name='official_fixed_ap', exist_ok=False,
)

def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}

def write_state(**changes):
    state = read_state()
    state.update(changes)
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2, default=str) + '\n')
    temp.replace(STATE)
    return state

def verify_repository():
    assert git('rev-parse', 'HEAD').stdout.strip() == EXPECTED_COMMIT, 'Wrong git HEAD'
    assert git('diff', '--exit-code').returncode == 0, 'Tracked working-tree changes'
    assert git('diff', '--cached', '--exit-code').returncode == 0, 'Staged changes'
    extras = git('ls-files', '--others', '--exclude-standard').stdout.splitlines()
    assert all(p.startswith('baseline_logs/') for p in extras), f'Unexpected untracked files: {extras}'

def protect_inputs():
    """Deny Python writes to source, model, environment and dataset paths."""
    import sys
    roots = (ROOT, DATASET, Path('/workspace/yoloe-env'), Path('/workspace/miniconda3'))
    def forbidden(value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        p = Path(os.fsdecode(value)).resolve()
        return not p.is_relative_to(LOGS) and any(p.is_relative_to(r) for r in roots)
    def audit(event, args):
        targets = []
        if event == 'open':
            path, mode, flags = args
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing:
                targets = [path]
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.truncate', 'os.utime'):
            targets = [args[0]]
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            targets = [args[0], args[1]]
        if any(forbidden(p) for p in targets):
            raise PermissionError(f'Baseline input protection blocked {event}: {targets}')
    sys.addaudithook(audit)
