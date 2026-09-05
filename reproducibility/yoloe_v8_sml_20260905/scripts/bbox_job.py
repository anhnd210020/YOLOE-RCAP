"""One smoke image, one full inference, and the unchanged official bbox Fixed AP evaluator."""
import datetime
import importlib.metadata
import json
import math
import platform
import re
import shlex
import subprocess
import sys
import time
import traceback
from bbox_common import *


def preflight():
    import torch
    import pycocotools.mask
    import lvis
    import ultralytics
    import yaml
    from ultralytics.utils import AUTOINSTALL, DATASETS_DIR
    verify_repository()
    assert sys.prefix == '/workspace/yoloe-env'
    assert platform.python_version() == '3.10.21'
    assert torch.__version__ == '2.5.1+cu121' and torch.version.cuda == '12.1'
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 3090'
    assert ultralytics.__version__ == '8.3.39'
    assert Path(ultralytics.__file__).resolve() == ROOT / 'ultralytics/__init__.py'
    assert importlib.metadata.version('pycocotools') == '2.0.11' and not AUTOINSTALL
    hashes = {}
    for path, expected in (
        (CHECKPOINT, 'ac2b90ed23011495a3e86d89caeb3432a15129cac8d849ba121293c8fc1e0536'),
        (ROOT / 'mobileclip_blt.pt', '670844f7a886dd6eff7a9285adfc53f3d3c889c03bfc8354010cb5c6bf27441a'),
        (DATASET / 'minival.txt', '1b7c59904861001d0c0d71a024cee80617a4a74f565b38e9ab6c60d63123c4df'),
        (ANNOTATIONS, '02301f6ccd89d1ee3d35112cb57d000c3396f34e4073066c90b2c1fbf47b55ce'),
    ):
        hashes[str(path)] = digest(path)
        assert hashes[str(path)] == expected, f'Input hash changed: {path}'
    data = yaml.safe_load((ROOT / CONFIG['data']).read_text())
    assert (DATASETS_DIR / data['path']).resolve() == DATASET
    paths = [(DATASET / line).resolve() for line in (DATASET / data['minival']).read_text().splitlines()]
    gt = json.loads(ANNOTATIONS.read_text())
    assert len(paths) == len(set(paths)) == len(gt['images']) == 4809
    assert {int(p.stem) for p in paths} == {i['id'] for i in gt['images']}
    assert len(gt['annotations']) == 50672 and len(gt['categories']) == len(data['names']) == 1203
    for path in paths:
        assert path.is_file()
        with path.open('rb') as f:
            f.seek(-2, 2)
            assert f.read() == b'\xff\xd9', f'JPEG would require repair: {path}'
    metadata = dict(python=platform.python_version(), pytorch=torch.__version__, torch_cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0), git_commit=EXPECTED_COMMIT,
                    input_sha256=hashes, preflight_passed=True)
    write_state(**metadata)
    print('Bbox workaround preflight PASS:\n' + json.dumps(metadata, indent=2), flush=True)


def main():
    # This new authorization is separate from the previous failed segmentation run.
    assert not STATE.exists(), 'Bbox workaround job state already exists; refusing automatic retry'
    write_state(start_time=datetime.datetime.now(datetime.timezone.utc).isoformat(), start_epoch=time.time(),
                phase='preflight', overall_status='BLOCKED', images_evaluated=0, workaround=WORKAROUND)
    print('\n=== APPROVED BBOX FIXED AP WORKAROUND RUN ===', flush=True)
    print('Start time:', read_state()['start_time'])
    print(DISCLOSURE)
    print('Git status before:\n' + (git('status','--short').stdout or 'CLEAN'))
    print('Exact configuration:\n' + json.dumps(CONFIG, indent=2))
    print('Fixed AP reporting tolerance:', AP_TOLERANCE, 'AP points from', EXPECTED_AP)
    print('No retries: separate exclusive guards allow one smoke invocation and one full inference invocation.')
    preflight()
    for name in ('bbox_common.py', 'bbox_inference.py', 'bbox_job.py', 'run_bbox_baseline.sh'):
        print(f'\nExact runner source {LOGS / name}:\n' + (LOGS / name).read_text())
    sources = {name: digest(LOGS / name) for name in ('bbox_common.py','bbox_inference.py','bbox_job.py')}
    smoke_cmd = [sys.executable, '-u', str(LOGS / 'bbox_inference.py'), '--smoke']
    full_cmd = [sys.executable, '-u', str(LOGS / 'bbox_inference.py')]
    fixed_cmd = [sys.executable, '-u', 'tools/eval_fixed_ap.py', str(ANNOTATIONS), str(PREDICTIONS), '--type', 'bbox']
    print('Smoke command:', shlex.join(smoke_cmd))
    print('Full inference command:', shlex.join(full_cmd))
    print('Fixed AP command:', shlex.join(fixed_cmd), flush=True)
    write_state(smoke_command=shlex.join(smoke_cmd), inference_command=shlex.join(full_cmd),
                fixed_ap_command=shlex.join(fixed_cmd), runner_source_sha256=sources, phase='one-image smoke test')
    with (LOGS / 'bbox_smoke_started.lock').open('x') as f:
        f.write(read_state()['start_time'] + '\n')
    result = subprocess.run(smoke_cmd, cwd=ROOT)
    write_state(smoke_exit_code=result.returncode)
    assert result.returncode == 0, f'Smoke failed with exit {result.returncode}; full inference will not start'
    smoke = json.loads(SMOKE_RESULT.read_text())
    assert smoke['status'] == 'PASS' and smoke['images_evaluated'] == 1
    assert smoke['mask_path_calls'] == 0 and smoke['values_finite'] and smoke['required_json_fields']
    assert smoke['exact_match_to_original_diagnostic_boxes']
    assert smoke['git_diff_exit_code'] == smoke['git_cached_diff_exit_code'] == 0
    verify_repository()
    assert all(digest(LOGS / name) == sha for name, sha in sources.items()), 'Runner changed after smoke test'
    write_state(smoke_result=smoke, smoke_passed=True, phase='launching one full inference')
    print('\nSmoke PASS confirmed. Launching the one authorized full inference immediately.', flush=True)
    with (LOGS / 'bbox_inference_started.lock').open('x') as f:
        f.write(datetime.datetime.now(datetime.timezone.utc).isoformat() + '\n')
    result = subprocess.run(full_cmd, cwd=ROOT)
    write_state(inference_exit_code=result.returncode)
    if result.returncode != 0:
        raise RuntimeError(f'Full inference exited {result.returncode}; no retry will be attempted')
    state = read_state()
    assert state.get('inference_complete') and state.get('images_evaluated') == 4809
    assert state.get('image_sequence_verified') and PREDICTIONS.is_file()
    verify_repository()
    write_state(phase='official bbox Fixed AP evaluation')
    print('\n=== OFFICIAL FIXED AP OUTPUT (bbox) ===', flush=True)
    print('Command:', shlex.join(fixed_cmd), flush=True)
    process = subprocess.Popen(fixed_cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    ap, keys = None, None
    for line in process.stdout:
        print(line, end='', flush=True)
        clean = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
        if 'copypaste:' in clean:
            payload = clean.split('copypaste:', 1)[1].strip().split(',')
            if payload[0] == 'AP':
                keys = payload
            elif keys:
                ap = dict(zip(keys, map(float, payload)))['AP']
    code = process.wait()
    write_state(fixed_ap_exit_code=code, fixed_ap=ap)
    assert code == 0, f'Official Fixed AP evaluator exited {code}'
    assert ap is not None and math.isfinite(ap), 'Unable to parse official Fixed AP output'
    difference = ap - EXPECTED_AP
    status = 'PASS' if abs(difference) <= AP_TOLERANCE else 'FAIL'
    write_state(difference=difference, overall_status=status, phase='completed')
    return 0 if status == 'PASS' else 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        traceback.print_exc()
        write_state(error=str(exc), overall_status='FAIL' if (LOGS / 'bbox_inference_started.lock').exists() else 'BLOCKED')
        sys.exit(1)
