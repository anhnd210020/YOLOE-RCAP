"""Shared M/L configuration and state; no v8-S inference is reachable from this runner."""
import json
from pathlib import Path
from baseline_common import ROOT, LOGS, DATASET, ANNOTATIONS, EXPECTED_COMMIT, digest, git, protect_inputs, verify_repository

REPORT=LOGS/'baseline_v8_ml_result.txt'
MANIFEST=LOGS/'v8_ml_checkpoints.json'
VENDORED_LVIS=ROOT/'third_party/lvis-api'
PAPER={'m':{'AP':32.6,'APr':26.9,'APc':31.9,'APf':34.4},
       'l':{'AP':35.9,'APr':33.2,'APc':34.8,'APf':37.3}}
TOLERANCE=0.5
BASE_CONFIG=dict(data='ultralytics/cfg/datasets/lvis.yaml',imgsz=640,batch=1,split='minival',
                 rect=False,conf=0.001,iou=0.7,max_det=1000,half=False,device=0,
                 load_vp=False,save_json=True,plots=False,project=str(LOGS),exist_ok=False)
DISCLOSURE=('Official-equivalent bbox Fixed AP workaround, not the unmodified end-to-end segmentation JSON path. '
            'Reuses the v8-S-validated BBoxFixedAPValidator: full official segmentation model and YOLOE text prompts, '
            'unchanged image preprocessing and SegmentationValidator postprocess/multi-label NMS, '
            'official DetectionValidator native-box preparation and bbox JSON serialization. '
            'Only predicted-mask decoding/serialization and dependent mask work are bypassed; official box metrics are retained.')

def metadata(scale):
    assert scale in ('m','l'), 'Only v8-M/v8-L are authorized'
    return json.loads(MANIFEST.read_text())[scale]

def checkpoint(scale):
    return Path(metadata(scale)['path'])

def config(scale,smoke=False):
    assert scale in ('m','l')
    return dict(BASE_CONFIG,name=f'v8{scale}_smoke' if smoke else f'v8{scale}_fixed_ap')

def predictions(scale,smoke=False):
    return LOGS/config(scale,smoke)['name']/'predictions.json'

def state_path(scale):
    assert scale in ('m','l')
    return LOGS/f'v8{scale}_state.json'

def state(scale):
    p=state_path(scale)
    return json.loads(p.read_text()) if p.exists() else {}

def update(scale,**changes):
    d=state(scale)
    d.update(changes)
    p=state_path(scale)
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d,indent=2,default=str)+'\n')
    tmp.replace(p)
    return d

def verify_lvis():
    import lvis
    from lvis import LVIS,LVISResults,LVISEval
    actual=Path(lvis.__file__).resolve()
    assert actual==VENDORED_LVIS/'lvis/__init__.py',f'Wrong LVIS import: {actual}'
    return str(actual)

def identity(model,scale):
    import yaml
    from ultralytics.nn.tasks import YOLOESegModel
    from ultralytics.nn.modules.head import YOLOESegment
    y=model.model.yaml
    official=yaml.safe_load((ROOT/'ultralytics/cfg/models/v8/yoloe-v8-seg.yaml').read_text())
    d=dict(scale=y.get('scale'),yaml_file=y.get('yaml_file'),task=model.task,
           model_class=type(model.model).__name__,head_class=type(model.model.model[-1]).__name__,
           text_model=model.model.args.get('text_model'),parameters=sum(p.numel() for p in model.model.parameters()),
           first_conv_out_channels=model.model.model[0].conv.out_channels,
           first_c2f_repeats=len(model.model.model[2].m),strides=model.model.stride.tolist(),
           scale_multipliers=y.get('scales',{}).get(scale),has_cached_pe=hasattr(model.model,'pe'))
    assert isinstance(model.model,YOLOESegModel) and isinstance(model.model.model[-1],YOLOESegment)
    assert model.task=='segment' and y.get('scale')==scale, d
    assert d['text_model']=='mobileclip:blt' and not d['has_cached_pe'],d
    assert d['scale_multipliers']==official['scales'][scale],d
    assert d['first_conv_out_channels']=={'m':48,'l':64}[scale],d
    assert d['first_c2f_repeats']=={'m':2,'l':3}[scale],d
    assert d['strides']==[8.0,16.0,32.0],d
    # Compare the official architecture definitions, independently of file naming.
    assert y['backbone']==official['backbone'] and y['head']==official['head'], 'Architecture YAML differs from official v8 segmentation definition'
    return d
