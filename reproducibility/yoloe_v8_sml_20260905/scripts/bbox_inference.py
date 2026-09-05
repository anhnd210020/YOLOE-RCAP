"""Approved bbox-only serializer using the official detection methods verbatim."""
import argparse
import datetime
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from bbox_common import *

protect_inputs()
import torch
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.val import YOLOESegValidator
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils import ops
from ultralytics.data import dataset as dataset_module

class BBoxFixedAPValidator(YOLOESegValidator):
    """Keep segmentation inference/NMS; dispatch box work to official detection code."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.args.task == 'segment'
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.visited_image_ids = []

    # Aliases reference the official implementations, without copying their logic.
    init_metrics = DetectionValidator.init_metrics
    _prepare_batch = DetectionValidator._prepare_batch
    _prepare_pred = DetectionValidator._prepare_pred
    _process_batch = DetectionValidator._process_batch
    pred_to_json = DetectionValidator.pred_to_json
    get_stats = DetectionValidator.get_stats
    finalize_metrics = DetectionValidator.finalize_metrics
    print_results = DetectionValidator.print_results
    get_desc = DetectionValidator.get_desc
    eval_json = DetectionValidator.eval_json

    def update_metrics(self, preds, batch):
        # SegmentationValidator.postprocess is inherited unchanged. Its first
        # result contains the original post-NMS rows, including mask coefficients.
        DetectionValidator.update_metrics(self, preds[0], batch)
        self.visited_image_ids.extend(int(Path(p).stem) for p in batch['im_file'])

class SmokeValidator(BBoxFixedAPValidator):
    """Observation-only additions for the one authorized smoke image."""
    def _prepare_pred(self, pred, pbatch):
        native = DetectionValidator._prepare_pred(self, pred, pbatch)
        assert torch.isfinite(native[:, :6]).all(), 'Non-finite box/score/class'
        assert torch.equal(pred[:, 4:6], native[:, 4:6]), 'Score/class changed by scaling'
        self.smoke_native_hash = hashlib.sha256(native[:, :6].detach().contiguous().cpu().numpy().tobytes()).hexdigest()
        self.smoke_detections = len(native)
        return native

    def eval_json(self, stats):
        # The smoke test must not evaluate a one-image file against full LVIS.
        print('Smoke: standalone LVIS evaluation intentionally not invoked.', flush=True)
        return stats

assert BBoxFixedAPValidator.postprocess is SegmentationValidator.postprocess
assert BBoxFixedAPValidator.preprocess is YOLOESegValidator.preprocess
assert BBoxFixedAPValidator._prepare_pred is DetectionValidator._prepare_pred
assert BBoxFixedAPValidator.pred_to_json is DetectionValidator.pred_to_json

# Keep all derived cache writes within baseline_logs; preserve official cache code.
_cache_load = dataset_module.load_dataset_cache_file
_cache_save = dataset_module.save_dataset_cache_file

def load_cache(path):
    existing = LOGS / 'label_cache' / Path(path).name
    return _cache_load(existing if existing.is_file() else path)

def save_cache(prefix, path, data, version):
    target = LOGS / 'bbox_label_cache' / Path(path).name
    target.parent.mkdir(parents=True, exist_ok=True)
    return _cache_save(prefix, target, data, version)

dataset_module.load_dataset_cache_file = load_cache
dataset_module.save_dataset_cache_file = save_cache

class OneBatch:
    def __init__(self, original):
        self.original = original
        self.dataset = original.dataset
    def __len__(self):
        return 1
    def __iter__(self):
        yield next(iter(self.original))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    mode = parser.parse_args()
    verify_repository()
    expected_count = 1 if mode.smoke else 4809
    cfg = dict(CONFIG, name='bbox_smoke' if mode.smoke else CONFIG['name'])
    output = LOGS / cfg['name'] / 'predictions.json'
    assert not output.parent.exists(), f'Refusing existing output directory: {output.parent}'
    ordered_paths = sorted(str((DATASET / line).resolve()) for line in (DATASET / 'minival.txt').read_text().splitlines())
    assert len(ordered_paths) == len(set(ordered_paths)) == 4809
    expected_ids = [int(Path(p).stem) for p in ordered_paths[:expected_count]]
    observations = dict(dataset_batches_started=0, mask_path_calls=0)
    active = {}
    started = time.monotonic()

    def on_start(validator):
        actual = [str(Path(p).resolve()) for p in validator.dataloader.dataset.im_files]
        assert actual == ordered_paths, 'Dataset contents or ordering changed'
        assert len(validator.data['names']) == 1203
        assert validator.args.task == 'segment' and validator.device.type == 'cuda'
        for key in ('imgsz','batch','split','rect','conf','iou','max_det','half','save_json','load_vp'):
            assert getattr(validator.args, key) == CONFIG[key], f'Unexpected configuration: {key}'
        assert (validator.save_dir / 'predictions.json').resolve() == output
        if mode.smoke:
            validator.dataloader = OneBatch(validator.dataloader)
        active['validator'] = validator
        print('Complete effective inference configuration:\n' + json.dumps(vars(validator.args), indent=2, default=str), flush=True)
        print(f'Image sequence verified. This invocation is limited to {expected_count} dataset image(s).', flush=True)
        if not mode.smoke:
            write_state(resolved_config=vars(validator.args), dataloader_images=4809, phase='full inference')

    def on_batch_start(validator):
        observations['dataset_batches_started'] += 1
        assert observations['dataset_batches_started'] <= expected_count

    def on_batch_end(validator):
        assert validator.seen == len(validator.visited_image_ids) == observations['dataset_batches_started']
        if not mode.smoke and (validator.seen % 100 == 0 or validator.seen == 4809):
            print(f'\nBBOX_BASELINE_PROGRESS {validator.seen}/4809 at {datetime.datetime.now(datetime.timezone.utc).isoformat()}', flush=True)
            write_state(images_evaluated=int(validator.seen), prediction_records=len(validator.jdict))

    def on_end(validator):
        assert validator.seen == expected_count
        assert validator.visited_image_ids == expected_ids, 'Image sequence/count mismatch'
        print(f'Inference loop complete: {validator.seen}/{expected_count} images, {len(validator.jdict)} bbox records.', flush=True)
        if not mode.smoke:
            write_state(images_evaluated=int(validator.seen), inference_loop_complete=True,
                        image_sequence_verified=True, prediction_records=len(validator.jdict),
                        phase='saving JSON and official standard bbox evaluation')

    # Fail the smoke test if any predicted-mask decoding or serializer is called.
    forbidden = {ops.process_mask.__code__, ops.process_mask_native.__code__, ops.scale_image.__code__,
                 SegmentationValidator._prepare_pred.__code__, SegmentationValidator.pred_to_json.__code__}
    def mask_guard(frame, event, arg):
        if event == 'call' and frame.f_code in forbidden:
            observations['mask_path_calls'] += 1
            raise AssertionError(f'Mask path unexpectedly invoked: {frame.f_code.co_name}')

    model = YOLOE(str(CHECKPOINT))
    assert model.task == 'segment' and model.model.args['text_model'] == 'mobileclip:blt'
    assert not hasattr(model.model, 'pe')
    for event, callback in (('on_val_start', on_start), ('on_val_batch_start', on_batch_start),
                            ('on_val_batch_end', on_batch_end), ('on_val_end', on_end)):
        model.add_callback(event, callback)
    print(DISCLOSURE, flush=True)
    print('Exact call: YOLOE(' + repr(str(CHECKPOINT)) + ').val(validator=' + ('SmokeValidator' if mode.smoke else 'BBoxFixedAPValidator') + ', **' + repr(cfg) + ')', flush=True)
    try:
        if mode.smoke:
            sys.setprofile(mask_guard)
        model.val(validator=SmokeValidator if mode.smoke else BBoxFixedAPValidator, **cfg)
    finally:
        sys.setprofile(None)
    validator = active['validator']
    assert output.is_file() and output.stat().st_size > 0
    assert validator.seen == expected_count and validator.visited_image_ids == expected_ids
    verify_repository()
    if mode.smoke:
        records = json.loads(output.read_text())
        assert records and len(records) == validator.smoke_detections
        for row in records:
            assert set(row) == {'image_id', 'category_id', 'bbox', 'score'}
            assert row['image_id'] == expected_ids[0]
            assert type(row['category_id']) is int and 1 <= row['category_id'] <= 1203
            assert len(row['bbox']) == 4
            assert all(math.isfinite(x) for x in [*row['bbox'], row['score'], row['category_id']])
        diagnosis = json.loads((LOGS / 'mask_diagnosis_smoke.json').read_text())
        exact_match = validator.smoke_native_hash == diagnosis['bbox_tensor_sha256']
        assert exact_match, 'Smoke native boxes differ from the captured original segmentation path'
        assert observations['mask_path_calls'] == 0
        result = dict(status='PASS', images_evaluated=1, predictions=str(output), detections=len(records),
                      mask_path_calls=0, values_finite=True, required_json_fields=True,
                      native_bbox_sha256=validator.smoke_native_hash,
                      exact_match_to_original_diagnostic_boxes=exact_match,
                      git_diff_exit_code=git('diff','--exit-code').returncode,
                      git_cached_diff_exit_code=git('diff','--cached','--exit-code').returncode,
                      runtime_seconds=time.monotonic()-started)
        SMOKE_RESULT.write_text(json.dumps(result, indent=2) + '\n')
        print('BBOX_SMOKE_PASS:\n' + json.dumps(result, indent=2), flush=True)
    else:
        write_state(inference_complete=True, inference_seconds=time.monotonic()-started,
                    predictions=str(output), predictions_bytes=output.stat().st_size, phase='inference complete')
        print('Full bbox inference and standard bbox evaluation complete:', output, flush=True)

if __name__ == '__main__':
    main()
