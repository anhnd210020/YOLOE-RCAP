"""Diagnosis only: one dataset image, unchanged failing path, read-only frame observation."""
import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys
import traceback
from baseline_common import ROOT, LOGS, DATASET, CHECKPOINT, CONFIG, protect_inputs, verify_repository, git

protect_inputs()
import cv2
import numpy as np
import torch
from ultralytics import YOLOE
from ultralytics.data import dataset as dataset_module

verify_repository()
OUTPUT = LOGS / 'mask_diagnosis_smoke.json'
report = dict(start_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              git_commit=git('rev-parse', 'HEAD').stdout.strip(), opencv=cv2.__version__,
              dataset_batches_started=0, fix_applied=False)

# Small synthetic OpenCV arrays; these are not model outputs or dataset images.
report['synthetic_opencv_channel_checks'] = {}
for channels in (1, 300, 512, 513, 1000):
    try:
        resized = cv2.resize(np.zeros((2, 3, channels), dtype=np.uint8), (6, 4))
        report['synthetic_opencv_channel_checks'][str(channels)] = dict(success=True, output_shape=list(resized.shape))
    except cv2.error as exc:
        report['synthetic_opencv_channel_checks'][str(channels)] = dict(success=False, error=str(exc))
print('Synthetic OpenCV channel checks:', json.dumps(report['synthetic_opencv_channel_checks']), flush=True)

# Inspect, rather than execute or replace, both official JSON dictionary expressions.
def bbox_dict_ast(path, cls):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'pred_to_json')
    for n in ast.walk(method):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'append':
            d = copy.deepcopy(n.args[0])
            pairs = [(k, v) for k, v in zip(d.keys, d.values) if k.value != 'segmentation']
            d.keys, d.values = map(list, zip(*pairs))
            return ast.dump(d, include_attributes=False)
    raise AssertionError('JSON append dictionary not found')
report['official_bbox_field_expressions_identical'] = bbox_dict_ast(ROOT / 'ultralytics/models/yolo/detect/val.py', 'DetectionValidator') == bbox_dict_ast(ROOT / 'ultralytics/models/yolo/segment/val.py', 'SegmentationValidator')
assert report['official_bbox_field_expressions_identical']

# Storage-only redirect, identical to the baseline's existing cache handling.
original_save_cache = dataset_module.save_dataset_cache_file

def store_cache(prefix, path, cache, version):
    destination = LOGS / 'mask_diagnosis_cache' / Path(path).name
    destination.parent.mkdir(exist_ok=True, parents=True)
    return original_save_cache(prefix, destination, cache, version)

dataset_module.save_dataset_cache_file = store_cache

class OneBatch:
    def __init__(self, original):
        self.original = original
        self.dataset = original.dataset
    def __len__(self):
        return 1
    def __iter__(self):
        yield next(iter(self.original))

ops_file = str(ROOT / 'ultralytics/utils/ops.py')

def trace_scale(frame, event, arg):
    if event == 'call':
        if frame.f_code.co_filename == ops_file and frame.f_code.co_name == 'scale_image':
            report['scale_image_input_shape'] = list(frame.f_locals['masks'].shape)
            return trace_scale
        return None
    if event == 'line' and frame.f_lineno == 384 and 'failing_resize_array_shape' not in report:
        loc = frame.f_locals
        arr = loc['masks']
        caller = frame.f_back.f_locals
        pred = caller['pred']
        predn = caller['predn']
        report.update(
            failing_resize_array_shape=list(arr.shape), failing_resize_array_dtype=str(arr.dtype),
            failing_resize_array_strides=list(arr.strides),
            resize_target_wh=[int(loc['im0_shape'][1]), int(loc['im0_shape'][0])],
            original_image_shape=list(loc['im0_shape']), ratio_pad=loc['ratio_pad'],
            processed_mask_tensor_shape=list(caller['pred_masks'].shape),
            post_nms_prediction_shape=list(pred.shape), scaled_prediction_shape=list(predn.shape),
            bbox_tensor_shape=list(predn[:, :6].shape), number_of_detections=len(predn),
            bbox_values_finite=bool(torch.isfinite(predn[:, :6]).all()),
            scores_classes_unchanged_by_rescaling=bool(torch.equal(pred[:, 4:6], predn[:, 4:6])),
            bbox_sample=predn[:3, :6].detach().cpu().tolist(),
            bbox_tensor_sha256=hashlib.sha256(predn[:, :6].detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
            image_file=caller['batch']['im_file'][caller['si']],
            mask_processing_function=caller['self'].process.__name__,
            save_json=caller['self'].args.save_json,
            json_entries_before_failure=len(caller['self'].jdict),
        )
        print('Captured original cv2.resize inputs and completed bbox predictions:', json.dumps(report, default=str), flush=True)
    if event == 'exception':
        report['observed_scale_image_exception'] = str(arg[1])
    return trace_scale

class DiagnosisComplete(Exception):
    pass

def on_start(validator):
    assert validator.args.batch == 1
    assert len(validator.dataloader.dataset) == 4809
    validator.dataloader = OneBatch(validator.dataloader)
    report['validator'] = type(validator).__name__
    report['configuration'] = vars(validator.args).copy()
    print('Diagnosis guard: exactly ONE dataset batch; official backend warmup is preserved.', flush=True)
    sys.settrace(trace_scale)

def on_batch_start(validator):
    report['dataset_batches_started'] += 1
    assert report['dataset_batches_started'] == 1, 'More than one dataset batch requested'

def on_end(validator):
    # Even if the failure disappears, never enter JSON saving or an LVIS evaluator.
    raise DiagnosisComplete('One-image loop completed without reproducing the error')

try:
    model = YOLOE(str(CHECKPOINT))
    model.add_callback('on_val_start', on_start)
    model.add_callback('on_val_batch_start', on_batch_start)
    model.add_callback('on_val_end', on_end)
    cfg = dict(CONFIG, name='mask_diagnosis_one_image')
    print('Diagnosis call: model.val(**' + repr(cfg) + ')', flush=True)
    model.val(**cfg)
except cv2.error as exc:
    sys.settrace(None)
    report['error_reproduced'] = 'm->dims <= 2' in str(exc)
    report['exception'] = str(exc)
    traceback.print_exc()
except DiagnosisComplete as exc:
    sys.settrace(None)
    report['error_reproduced'] = False
    report['exception'] = str(exc)
finally:
    sys.settrace(None)
    report['git_diff_exit_code'] = git('diff', '--exit-code').returncode
    report['git_cached_diff_exit_code'] = git('diff', '--cached', '--exit-code').returncode
    report['end_time'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print('\nDIAGNOSIS OBSERVATIONS:\n' + json.dumps(report, indent=2, default=str), flush=True)

assert report['dataset_batches_started'] == 1
assert report.get('error_reproduced'), 'Expected failure not reproduced; inspect observations'
assert report['git_diff_exit_code'] == report['git_cached_diff_exit_code'] == 0
print('Diagnosis complete. No fix, bbox export, or full LVIS evaluation was performed.', flush=True)
