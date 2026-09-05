"""One M or L smoke/full inference using the unchanged validated bbox exporter class."""
import argparse
import datetime
import json
import math
import sys
import time
from pathlib import Path
from v8_ml_common import *
# Importing these classes does not execute the S runner's main() or load an S model.
from bbox_inference import BBoxFixedAPValidator,SmokeValidator,OneBatch
import torch
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.val import YOLOESegValidator
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops

class RecordedBBoxValidator(BBoxFixedAPValidator):
    def eval_json(self,stats):
        # Observation only: retain the official built-in standard bbox evaluator.
        result=DetectionValidator.eval_json(self,stats)
        self.standard_bbox_stats=dict(result)
        print('Official standard LVIS bbox stats (fractions): '+json.dumps(result,default=str),flush=True)
        return result

class RecordedSmokeValidator(RecordedBBoxValidator):
    _prepare_pred=SmokeValidator._prepare_pred
    eval_json=SmokeValidator.eval_json

assert RecordedBBoxValidator.postprocess is SegmentationValidator.postprocess
assert RecordedBBoxValidator.preprocess is YOLOESegValidator.preprocess
assert RecordedBBoxValidator._prepare_pred is DetectionValidator._prepare_pred
assert RecordedBBoxValidator.pred_to_json is DetectionValidator.pred_to_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scale',required=True,choices=['m','l'])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    scale=args.scale
    verify_repository()
    imported_lvis=verify_lvis()
    md=metadata(scale)
    assert digest(md['path'])==md['sha256']
    if not args.smoke:
        assert all(state(s).get('smoke',{}).get('status')=='PASS' for s in ('m','l')), 'Both smoke tests must pass first'
    output=predictions(scale,args.smoke)
    assert not output.parent.exists(),f'Refusing existing output directory {output.parent}'
    cfg=config(scale,args.smoke)
    count=1 if args.smoke else 4809
    paths=sorted(str((DATASET/p).resolve()) for p in (DATASET/'minival.txt').read_text().splitlines())
    assert len(paths)==len(set(paths))==4809
    ids=[int(Path(p).stem) for p in paths[:count]]
    observations={'batches':0,'mask_calls':0}
    active={}
    started=time.monotonic()
    model=YOLOE(str(checkpoint(scale)))
    ident=identity(model,scale)
    assert ident==state(scale)['identity'],'Checkpoint identity changed since preflight'
    print(f'\nYOLOE-v8-{scale.upper()} '+('one-image smoke' if args.smoke else 'FULL 4809-image inference'),flush=True)
    print('Checkpoint identity: '+json.dumps(ident),flush=True)
    print('Vendored LVIS: '+imported_lvis,flush=True)
    print(DISCLOSURE,flush=True)
    print('Exact model.val configuration: '+json.dumps(cfg,indent=2),flush=True)

    def on_start(v):
        assert [str(Path(p).resolve()) for p in v.dataloader.dataset.im_files]==paths
        assert len(v.data['names'])==1203 and v.device.type=='cuda' and v.args.task=='segment'
        for k in ('imgsz','batch','split','rect','conf','iou','max_det','half','load_vp','save_json'):
            assert getattr(v.args,k)==cfg[k],k
        for k in ('augment','single_cls','agnostic_nms','save_hybrid','save_txt','dnn'):
            assert getattr(v.args,k) is False,k
        assert v.args.classes is None and v.args.workers==8
        assert (v.save_dir/'predictions.json').resolve()==output
        if args.smoke:
            v.dataloader=OneBatch(v.dataloader)
        active['validator']=v
        observations['loop_start']=time.monotonic()
        print('Complete resolved validation configuration:\n'+json.dumps(vars(v.args),indent=2,default=str),flush=True)
        if not args.smoke:
            update(scale,phase='inference',configuration=vars(v.args),images_evaluated=0)

    def on_batch_start(v):
        observations['batches']+=1
        assert observations['batches']<=count
        assert v.nc==1203 and v.class_map==list(range(1203))

    def on_batch_end(v):
        assert v.seen==len(v.visited_image_ids)==observations['batches']
        if not args.smoke and (v.seen%100==0 or v.seen==4809):
            print(f'\nV8-{scale.upper()} PROGRESS {v.seen}/4809 at {datetime.datetime.now(datetime.timezone.utc).isoformat()}',flush=True)
            update(scale,images_evaluated=int(v.seen),prediction_records=len(v.jdict))

    def on_end(v):
        assert v.seen==count and v.visited_image_ids==ids
        observations['loop_seconds']=time.monotonic()-observations['loop_start']
        print(f'V8-{scale.upper()} image loop complete: {count}; records={len(v.jdict)}; seconds={observations["loop_seconds"]:.3f}',flush=True)
        if not args.smoke:
            update(scale,images_evaluated=4809,image_sequence_verified=True,inference_loop_complete=True,
                   prediction_records=len(v.jdict),image_loop_seconds=observations['loop_seconds'],
                   phase='JSON save and standard bbox evaluation')

    forbidden={ops.process_mask.__code__,ops.process_mask_native.__code__,ops.scale_image.__code__,
               SegmentationValidator._prepare_pred.__code__,SegmentationValidator.pred_to_json.__code__}
    def guard(frame,event,arg):
        if event=='call' and frame.f_code in forbidden:
            observations['mask_calls']+=1
            raise AssertionError(f'Unexpected mask path: {frame.f_code.co_name}')
    for event,func in (('on_val_start',on_start),('on_val_batch_start',on_batch_start),('on_val_batch_end',on_batch_end),('on_val_end',on_end)):
        model.add_callback(event,func)
    try:
        if args.smoke:
            sys.setprofile(guard)
        model.val(validator=RecordedSmokeValidator if args.smoke else RecordedBBoxValidator,**cfg)
    finally:
        sys.setprofile(None)
    v=active['validator']
    assert v.seen==count and v.visited_image_ids==ids
    assert output.is_file() and output.stat().st_size>0
    verify_repository()
    if args.smoke:
        records=json.loads(output.read_text())
        assert len(records)==v.smoke_detections and records
        for r in records:
            assert set(r)=={'image_id','category_id','bbox','score'}
            assert r['image_id']==ids[0] and type(r['category_id']) is int and 1<=r['category_id']<=1203
            assert len(r['bbox'])==4 and all(math.isfinite(x) for x in [*r['bbox'],r['score'],r['category_id']])
        assert observations['mask_calls']==0
        result=dict(status='PASS',images_evaluated=1,prediction_records=len(records),predictions=str(output),
                    values_finite=True,required_fields=True,mask_path_calls=0,native_bbox_sha256=v.smoke_native_hash,
                    official_box_serializer=True,git_diff_exit_code=git('diff','--exit-code').returncode,
                    runtime_seconds=time.monotonic()-started,configuration=vars(v.args))
        update(scale,smoke=result,phase='smoke passed')
        print(f'V8-{scale.upper()} SMOKE PASS:\n'+json.dumps(result,indent=2,default=str),flush=True)
    else:
        standard=getattr(v,'standard_bbox_stats',{})
        standard_ap=standard.get('metrics/mAP50-95(B)') if 'metrics/APr(B)' in standard else None
        update(scale,inference_complete=True,inference_seconds=time.monotonic()-started,
               predictions=str(output),predictions_bytes=output.stat().st_size,
               standard_bbox_stats=standard,standard_lvis_bbox_ap=standard_ap*100 if standard_ap is not None else None,
               phase='inference complete')
        print(f'V8-{scale.upper()} inference stage complete; predictions: {output}',flush=True)

if __name__=='__main__':
    main()
