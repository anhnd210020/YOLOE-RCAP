"""Independent state for the newly approved bbox workaround; prior reports remain intact."""
import json
from pathlib import Path
from baseline_common import ROOT, LOGS, DATASET, CHECKPOINT, ANNOTATIONS, EXPECTED_COMMIT, EXPECTED_AP, AP_TOLERANCE, digest, git, protect_inputs, verify_repository

STATE = LOGS / 'bbox_run_state.json'
SMOKE_RESULT = LOGS / 'bbox_smoke_result.json'
PREDICTIONS = LOGS / 'bbox_fixed_ap/predictions.json'
CONFIG = dict(data='ultralytics/cfg/datasets/lvis.yaml', imgsz=640, batch=1,
              split='minival', rect=False, conf=0.001, iou=0.7, max_det=1000,
              half=False, save_json=True, load_vp=False, device=0,
              project=str(LOGS), name='bbox_fixed_ap', exist_ok=False, plots=False)
WORKAROUND = 'bbox-only serialization bypass for mask cv2.resize issue'
DISCLOSURE = ('This is an official-equivalent bbox Fixed AP workaround, not an unmodified '
              'end-to-end execution of the original segmentation JSON serialization path. '
              'The full segmentation model, text prompts, preprocessing, segmentation NMS, '
              'native box rescaling, category mapping, ordering, and bbox/score rounding are preserved. '
              'Predicted mask decoding, mask JSON, and dependent mask metrics/plots are bypassed. '
              'Official box metrics and built-in standard bbox evaluation are retained for the full run; '
              'Fixed AP is separately computed by the unchanged tools/eval_fixed_ap.py --type bbox.')

def read_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}

def write_state(**updates):
    state = read_state()
    state.update(updates)
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2, default=str) + '\n')
    tmp.replace(STATE)
    return state
