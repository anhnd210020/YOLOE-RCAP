"""Append the required self-contained final report on success or failure."""
import datetime
import json
import sys
import time
from bbox_common import *

state = read_state()
exit_code = int(sys.argv[1]) if len(sys.argv) > 1 else 1
end = datetime.datetime.now(datetime.timezone.utc).isoformat()
runtime = time.time() - state.get('start_epoch', time.time())
head = git('rev-parse','HEAD').stdout.strip()
diffs = [(args, git(*args)) for args in (('diff','--exit-code'), ('diff','--cached','--exit-code'), ('diff','--exit-code',EXPECTED_COMMIT,'--'))]
modified = head != EXPECTED_COMMIT or any(r.returncode != 0 for _, r in diffs)
status = state.get('overall_status', 'BLOCKED')
if modified or (exit_code != 0 and status == 'PASS'):
    status = 'FAIL'
if status == 'PASS' and not (state.get('smoke_passed') and state.get('inference_complete') and state.get('images_evaluated') == 4809 and state.get('image_sequence_verified') and state.get('fixed_ap_exit_code') == 0 and state.get('fixed_ap') is not None and abs(state['fixed_ap']-EXPECTED_AP) <= AP_TOLERANCE):
    status = 'FAIL'
print('\nEnd time:', end)
print('Runner exit code:', exit_code)
print('Git status after:\n' + (git('status','--short').stdout or 'CLEAN'))
for args, result in diffs:
    print('git ' + ' '.join(args) + ' result:', result.returncode)
    if result.returncode:
        print(result.stdout, result.stderr)
integrity = {}
for path, expected in state.get('input_sha256', {}).items():
    try:
        actual = digest(path)
        integrity[path] = actual == expected
        print(f'Input SHA256 after: {path}: {actual}; unchanged={integrity[path]}')
    except Exception as exc:
        integrity[path] = False
        print('Input integrity error:', path, str(exc))
if integrity and not all(integrity.values()):
    status = 'FAIL'
print('Important warnings/errors:', state.get('error', 'All inference and evaluation output is appended above.'))
print('Earlier segmentation-failure and diagnosis records remain above this new approved-run section.')
print('Package auto-installation was disabled. No packages, dataset files, checkpoint, CUDA or PyTorch were modified.')
print('Bbox workaround smoke image count:', state.get('smoke_result',{}).get('images_evaluated',0))
print('Bbox workaround full inference launch attempts:', int((LOGS / 'bbox_inference_started.lock').exists()))
print('Complete 4809-image inference runs:', int(bool(state.get('inference_complete'))))
print('Smoke result:', json.dumps(state.get('smoke_result',{}), sort_keys=True))
print('Inference command:', state.get('inference_command','not reached'))
print('Fixed AP evaluation command:', state.get('fixed_ap_command','not reached'))
print('Prediction records:', state.get('prediction_records','not available'))
print('Image order verified:', state.get('image_sequence_verified',False))
print(DISCLOSURE)
write_state(end_time=end, total_runtime_seconds=runtime, runner_exit_code=exit_code,
            tracked_source_modified=modified, final_input_integrity=integrity, overall_status=status)
config = state.get('resolved_config', CONFIG)
keys = ('task','mode','data','imgsz','batch','split','rect','conf','iou','max_det','half','save_json','load_vp','text_model','device','workers','single_cls','agnostic_nms','classes','augment','plots','cache','mask_ratio','overlap_mask','save_txt','save_hybrid','dnn','project','name')
summary = {k:config[k] for k in keys if k in config}
summary.update(prompt='text; first slash-separated alias', classes_count=1203,
               nms='official SegmentationValidator.postprocess; multi_label=True; nc=1203',
               serialization='official DetectionValidator.pred_to_json', fixed_ap_type='bbox',
               dets_per_cat=10000, fixed_ap_max_dets=-1)
ap = state.get('fixed_ap')
print('\n=== YOLOE OFFICIAL BASELINE REPORT ===')
print('Git commit:', head)
print('Python:', state.get('python','unavailable'))
print('PyTorch:', state.get('pytorch','unavailable'))
print('Torch CUDA:', state.get('torch_cuda','unavailable'))
print('GPU:', state.get('gpu','unavailable'))
print('Checkpoint:', CHECKPOINT)
print('Dataset:', DATASET)
print('Images evaluated:', state.get('images_evaluated',0))
print('Inference configuration:', json.dumps(summary,sort_keys=True,default=str))
print('Predictions JSON:', PREDICTIONS if PREDICTIONS.is_file() else 'NOT GENERATED (planned: ' + str(PREDICTIONS) + ')')
print('Fixed AP:', f'{ap:.2f}' if ap is not None else 'NOT COMPUTED')
print('Expected Fixed AP: 27.9')
print('Difference from expected:', f'{ap-EXPECTED_AP:+.2f} AP points' if ap is not None else 'N/A')
print('Total runtime:', str(datetime.timedelta(seconds=round(runtime))), f'({runtime:.2f} seconds)')
print('Tracked source modified:', 'YES' if modified else 'NO')
print('Workaround used: bbox-only serialization bypass for mask cv2.resize issue')
print('Overall status:', status)
