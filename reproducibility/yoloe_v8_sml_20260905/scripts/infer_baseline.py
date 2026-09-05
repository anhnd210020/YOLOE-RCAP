"""One invocation of the unchanged official model.val, plus storage and reporting hooks."""
import datetime
import json
from pathlib import Path
import time
from baseline_common import *

protect_inputs()
import torch
from ultralytics import YOLOE
from ultralytics.data import dataset as dataset_module
from ultralytics.data import utils as data_utils

# Change only the destination of an optional derived label cache. The official
# cache function, label verification, dataset, transforms and validator are used.
_original_cache_save = dataset_module.save_dataset_cache_file

def save_cache_under_logs(prefix, path, data, version):
    target = LOGS / 'label_cache' / Path(path).name
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f'Label-cache storage redirect: {path} -> {target}', flush=True)
    return _original_cache_save(prefix, target, data, version)

dataset_module.save_dataset_cache_file = save_cache_under_logs

model = YOLOE(str(CHECKPOINT))
assert model.task == 'segment'
assert model.model.args['text_model'] == 'mobileclip:blt'
assert not hasattr(model.model, 'pe'), 'Checkpoint unexpectedly has fixed prompt embeddings'
expected_paths = sorted(str((DATASET / p).resolve()) for p in (DATASET / 'minival.txt').read_text().splitlines())
started = time.monotonic()

def on_start(validator):
    paths = sorted(str(Path(p).resolve()) for p in validator.dataloader.dataset.im_files)
    assert paths == expected_paths and len(paths) == 4809, 'Dataloader omitted/changed images'
    assert len(validator.data['names']) == 1203
    assert validator.args.task == 'segment' and validator.args.load_vp is False
    assert validator.device.type == 'cuda' and validator.args.half is False
    for key in ('imgsz', 'batch', 'split', 'rect', 'conf', 'iou', 'max_det', 'save_json', 'half', 'load_vp'):
        assert getattr(validator.args, key) == CONFIG[key], f'Configuration changed: {key}'
    assert (validator.save_dir / 'predictions.json').resolve() == PREDICTIONS
    actual = vars(validator.args).copy()
    print('Complete resolved validation configuration:\n' + json.dumps(actual, indent=2, default=str), flush=True)
    print('Dataloader verified: 4809 unique minival images; 1203 classes; task=segment; prompt=text', flush=True)
    write_state(resolved_config=actual, dataloader_images=len(paths), predictions=str(PREDICTIONS), phase='inference')

def on_batch_end(validator):
    assert validator.seen == validator.batch_i + 1, 'Unexpected image count for batch=1'
    if validator.seen % 100 == 0 or validator.seen == 4809:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        print(f'\nBASELINE_PROGRESS {validator.seen}/4809 images at {now}', flush=True)
        write_state(images_evaluated=validator.seen)

def on_end(validator):
    assert validator.seen == 4809
    write_state(images_evaluated=int(validator.seen), inference_loop_complete=True,
                phase='official built-in LVIS evaluation', inference_loop_seconds=time.monotonic()-started)
    print('\nSingle inference loop completed: 4809/4809 images. Official JSON saving and built-in evaluation follow.', flush=True)

model.add_callback('on_val_start', on_start)
model.add_callback('on_val_batch_end', on_batch_end)
model.add_callback('on_val_end', on_end)
print('Starting the one authorized full inference via model.val.', flush=True)
model.val(**CONFIG)
assert PREDICTIONS.is_file() and PREDICTIONS.stat().st_size > 0, 'Official predictions.json missing'
write_state(inference_complete=True, inference_seconds=time.monotonic()-started,
            predictions_bytes=PREDICTIONS.stat().st_size, phase='inference complete')
print(f'Inference and built-in evaluation finished. Predictions: {PREDICTIONS}', flush=True)
