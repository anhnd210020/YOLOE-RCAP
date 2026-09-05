"""Always append the requested final report, including failures and source integrity."""
import datetime
import json
import sys
import time
from baseline_common import *

state = read_state()
exit_code = int(sys.argv[1]) if len(sys.argv) > 1 else 1
end = state.get('end_time', datetime.datetime.now(datetime.timezone.utc).isoformat())
runtime = state.get('total_runtime_seconds', time.time() - state.get('start_epoch', time.time()))
head = git('rev-parse', 'HEAD').stdout.strip()
work_diff = git('diff', '--exit-code')
index_diff = git('diff', '--cached', '--exit-code')
official_diff = git('diff', '--exit-code', EXPECTED_COMMIT, '--')
modified = head != EXPECTED_COMMIT or any(r.returncode != 0 for r in (work_diff, index_diff, official_diff))
status = state.get('overall_status', 'BLOCKED')
if exit_code != 0 and status == 'PASS':
    status = 'FAIL'
if modified:
    status = 'FAIL'
if status == 'PASS' and not (state.get('images_evaluated') == 4809 and state.get('inference_complete') and state.get('fixed_ap_exit_code') == 0 and state.get('fixed_ap') is not None and abs(state['fixed_ap'] - EXPECTED_AP) <= AP_TOLERANCE):
    status = 'FAIL'

print('\nEnd time:', end)
print('Runner exit code:', exit_code)
print('Git status after:\n' + (git('status', '--short').stdout or 'CLEAN'))
print('git diff --exit-code result:', work_diff.returncode)
print('git diff --cached --exit-code result:', index_diff.returncode)
print(f'git diff --exit-code {EXPECTED_COMMIT} -- result:', official_diff.returncode)
for result in (work_diff, index_diff, official_diff):
    if result.returncode:
        print(result.stdout, result.stderr)

integrity = {}
for key, path, expected in (
    ('YOLOE checkpoint', CHECKPOINT, 'ac2b90ed23011495a3e86d89caeb3432a15129cac8d849ba121293c8fc1e0536'),
    ('minival list', DATASET / 'minival.txt', '1b7c59904861001d0c0d71a024cee80617a4a74f565b38e9ab6c60d63123c4df'),
    ('minival annotation JSON', ANNOTATIONS, '02301f6ccd89d1ee3d35112cb57d000c3396f34e4073066c90b2c1fbf47b55ce'),
):
    try:
        actual = digest(path)
        integrity[key] = actual == expected
        print(f'{key} SHA256 after: {actual}; unchanged: {integrity[key]}')
    except Exception as exc:
        integrity[key] = False
        print(f'{key} integrity check error: {exc}')
if not all(integrity.values()):
    status = 'FAIL'
print('Important warnings/errors:')
if state.get('error'):
    print(state['error'])
log = LOGS / 'baseline_result.txt'
warnings = []
with log.open(errors='replace') as f:
    for line in f:
        stripped = line.strip()
        if ('WARNING' in stripped or stripped.startswith(('Traceback ', 'RuntimeError:', 'AssertionError:', 'FileNotFoundError:', 'PermissionError:', 'cv2.error:', 'ERROR:')) or 'unable to run:' in stripped):
            if len(stripped) < 4000 and stripped not in warnings:
                warnings.append(stripped)
for line in warnings[:30]:
    print(line)
print(f'{len(warnings)} distinct warning/error lines in the complete output above.')
print('Automatic package installation was disabled; no package-management commands were run.')
print('Label-cache writes were redirected only to baseline_logs using the unchanged official cache implementation.')
print('Inference launch attempts:', 1 if (LOGS / 'inference_started.lock').exists() else 0)
print('Completed full 4809-image inferences:', 1 if state.get('inference_complete') else 0)
print('Images attempted:', state.get('images_attempted', state.get('images_evaluated', 0)))
write_state(end_time=end, total_runtime_seconds=runtime, runner_exit_code=exit_code,
            tracked_source_modified=modified, integrity_checks=integrity, overall_status=status)
config = state.get('resolved_config', state.get('preflight_config', CONFIG))
relevant_keys = ('task', 'mode', 'data', 'imgsz', 'batch', 'split', 'rect', 'conf', 'iou', 'max_det', 'save_json', 'half', 'load_vp', 'text_model', 'device', 'workers', 'single_cls', 'agnostic_nms', 'classes', 'augment', 'cache', 'plots', 'overlap_mask', 'mask_ratio', 'save_hybrid', 'save_txt', 'save_conf', 'dnn', 'project', 'name')
relevant = {k: config[k] for k in relevant_keys if k in config}
relevant.update(prompt='text; first alias per YAML class', classes_count=1203, fixed_ap_type='bbox', dets_per_cat=10000, fixed_ap_max_dets=-1)
ap = state.get('fixed_ap')
difference = f'{ap - EXPECTED_AP:+.2f} AP points' if ap is not None else 'N/A'
print('\n=== YOLOE OFFICIAL BASELINE REPORT ===')
print('Git commit:', head)
print('Python:', state.get('python', 'unavailable'))
print('PyTorch:', state.get('pytorch', 'unavailable'))
print('Torch CUDA:', state.get('torch_cuda', 'unavailable'))
print('GPU:', state.get('gpu', 'unavailable'))
print('Checkpoint:', CHECKPOINT)
print('Dataset:', DATASET)
print('Images evaluated:', state.get('images_evaluated', 0))
print('Inference configuration:', json.dumps(relevant, sort_keys=True, default=str))
print('Predictions JSON:', str(PREDICTIONS) if PREDICTIONS.is_file() else 'NOT GENERATED (planned: ' + str(PREDICTIONS) + ')')
print('Fixed AP:', f'{ap:.2f}' if ap is not None else 'NOT COMPUTED')
print('Expected Fixed AP: 27.9')
print('Difference from expected:', difference)
print('Total runtime:', str(datetime.timedelta(seconds=round(runtime))), f'({runtime:.2f} seconds)')
print('Tracked source modified:', 'YES' if modified else 'NO')
print('Overall status:', status)
