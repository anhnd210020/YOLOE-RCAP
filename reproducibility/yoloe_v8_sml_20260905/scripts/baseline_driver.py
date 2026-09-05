"""Preflight, exactly one inference process, then the official Fixed AP program."""
import datetime
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import traceback
from baseline_common import *


def preflight():
    import torch
    import pycocotools.mask
    import lvis
    import ultralytics.nn.text_model
    from ultralytics import YOLOE
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import AUTOINSTALL
    verify_repository()
    assert sys.prefix == '/workspace/yoloe-env', sys.prefix
    assert platform.python_version() == '3.10.21'
    assert torch.__version__ == '2.5.1+cu121'
    assert torch.version.cuda == '12.1' and torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 3090'
    assert importlib.metadata.version('pycocotools') == '2.0.11'
    assert not AUTOINSTALL
    assert CHECKPOINT.is_file()
    assert digest(CHECKPOINT) == 'ac2b90ed23011495a3e86d89caeb3432a15129cac8d849ba121293c8fc1e0536'
    assert digest(ROOT / 'mobileclip_blt.pt') == '670844f7a886dd6eff7a9285adfc53f3d3c889c03bfc8354010cb5c6bf27441a'
    assert digest(DATASET / 'minival.txt') == '1b7c59904861001d0c0d71a024cee80617a4a74f565b38e9ab6c60d63123c4df'
    assert digest(ANNOTATIONS) == '02301f6ccd89d1ee3d35112cb57d000c3396f34e4073066c90b2c1fbf47b55ce'
    data = check_det_dataset(CONFIG['data'], autodownload=False)
    assert data['path'].resolve() == DATASET and len(data['names']) == 1203
    paths = [(DATASET / line).resolve() for line in Path(data['minival']).read_text().splitlines()]
    annotation_data = json.loads(ANNOTATIONS.read_text())
    assert len(paths) == len(set(paths)) == len(annotation_data['images']) == 4809
    assert {int(p.stem) for p in paths} == {a['id'] for a in annotation_data['images']}
    assert len(annotation_data['annotations']) == 50672
    assert len(annotation_data['categories']) == 1203
    # Avoid the official loader's optional corrupt-JPEG repair touching the dataset.
    missing_labels = []
    for p in paths:
        assert p.is_file(), str(p)
        with p.open('rb') as f:
            f.seek(-2, 2)
            assert f.read() == b'\xff\xd9', f'JPEG requires repair: {p}'
        if not Path(str(p).replace('/images/', '/labels/')).with_suffix('.txt').is_file():
            missing_labels.append(int(p.stem))
    annotated_ids = {a['image_id'] for a in annotation_data['annotations']}
    print(f'Images without label files: {len(missing_labels)}; official loader treats missing labels as backgrounds.')
    print(f'Of these, minival JSON images with positive annotations: {len(set(missing_labels) & annotated_ids)}')
    print('All images are retained; official Fixed AP uses the unchanged annotation JSON.')
    model = YOLOE(str(CHECKPOINT))
    assert model.task == 'segment' and model.model.args['text_model'] == 'mobileclip:blt'
    assert not hasattr(model.model, 'pe')
    args = {**model.overrides, 'rect': True, **CONFIG, 'mode': 'val'}
    # Constructor resolves the official defaults without invoking inference.
    validator = model._smart_load('validator')(args=args, save_dir=LOGS / 'config_inspection')
    resolved = vars(validator.args).copy()
    print('Official validator:', type(validator).__name__)
    print('Preflight resolved validation configuration:\n' + json.dumps(resolved, indent=2, default=str), flush=True)
    print('Official class prompts: first slash-separated alias from each of 1203 YAML names.')
    print('Official NMS: multi_label=True; max_nms=30000; max_wh=7680; max_time_img=0.05.')
    print('Official Fixed AP defaults: bbox, top 10000 detections/category, max_dets=-1, IoUs 0.50:0.05:0.95.')
    print('Built-in standard LVIS box/mask AP is preserved; its AP is separate from the subsequent Fixed AP.')
    print(f'Reporting consistency threshold: abs(Fixed AP - {EXPECTED_AP}) <= {AP_TOLERANCE} AP points.')
    write_state(python=platform.python_version(), pytorch=torch.__version__, torch_cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), git_commit=EXPECTED_COMMIT,
                preflight_config=resolved, preflight_passed=True, error=None)
    print('BASELINE_PREFLIGHT_PASS: no dataset inference performed.', flush=True)


def main():
    if '--preflight' in sys.argv:
        preflight()
        return 0
    write_state(start_time=datetime.datetime.now(datetime.timezone.utc).isoformat(), start_epoch=time.time(),
                images_evaluated=0, phase='preflight', overall_status='BLOCKED')
    print('\nLong job start time:', read_state()['start_time'], flush=True)
    print('Git status before long job:\n' + (git('status', '--short', '--untracked-files=all').stdout or 'CLEAN'))
    preflight()
    assert not PREDICTIONS.parent.exists(), f'Output directory already exists: {PREDICTIONS.parent}'
    cmd = [sys.executable, '-u', str(LOGS / 'infer_baseline.py')]
    fixed_cmd = [sys.executable, '-u', 'tools/eval_fixed_ap.py', str(ANNOTATIONS), str(PREDICTIONS)]
    print('Exact inference command:', shlex.join(cmd))
    print('Exact inference code (and reporting/storage hooks):\n' + (LOGS / 'infer_baseline.py').read_text())
    print('Exact model.val kwargs:\n' + json.dumps(CONFIG, indent=2))
    print('Fixed AP evaluation command:', shlex.join(fixed_cmd), flush=True)
    write_state(inference_command=shlex.join(cmd), fixed_ap_command=shlex.join(fixed_cmd), phase='launching inference')
    # Exclusive, persistent guard: an attempted inference is never automatically retried.
    with (LOGS / 'inference_started.lock').open('x') as f:
        f.write(read_state()['start_time'] + '\n')
    result = subprocess.run(cmd, cwd=ROOT)
    write_state(inference_exit_code=result.returncode)
    if result.returncode != 0:
        raise RuntimeError(f'Inference exited {result.returncode}; no retry will be attempted')
    state = read_state()
    assert state.get('inference_complete') and state.get('images_evaluated') == 4809
    assert PREDICTIONS.is_file()
    write_state(phase='official Fixed AP evaluation')
    print('\n=== OFFICIAL FIXED AP OUTPUT ===', flush=True)
    # Stream the unmodified official evaluator's output into the same result file.
    process = subprocess.Popen(fixed_cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    fixed_ap = None
    keys = None
    for line in process.stdout:
        print(line, end='', flush=True)
        clean = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
        if 'copypaste:' in clean:
            payload = clean.split('copypaste:', 1)[1].strip().split(',')
            if payload[0] == 'AP':
                keys = payload
            elif keys:
                values = dict(zip(keys, map(float, payload)))
                fixed_ap = values['AP']
    code = process.wait()
    write_state(fixed_ap_exit_code=code, fixed_ap=fixed_ap)
    assert code == 0, f'Official Fixed AP evaluator exited {code}'
    assert fixed_ap is not None and math.isfinite(fixed_ap), 'Fixed AP could not be parsed from official output'
    difference = fixed_ap - EXPECTED_AP
    write_state(difference=difference, overall_status='PASS' if abs(difference) <= AP_TOLERANCE else 'FAIL', phase='completed')
    return 0 if abs(difference) <= AP_TOLERANCE else 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            raise
        traceback.print_exc()
        write_state(error=str(exc), overall_status='FAIL' if (LOGS / 'inference_started.lock').exists() else 'BLOCKED')
        sys.exit(1)
