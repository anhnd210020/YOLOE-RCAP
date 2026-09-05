# YOLOE v8 S/M/L Baseline Reproduction

Official YOLOE commit:

`40cd606cabdbe2b566d6f14a6b162c89206e9a1b`

## Evaluation

LVIS minival, zero-shot text prompts, 1203 categories, Fixed AP.

| Model | Paper AP | Reproduced AP | Paper APr/APc/APf | Reproduced APr/APc/APf |
|---|---:|---:|---|---|
| YOLOE-v8-S | 27.9 | 27.90 | 22.3 / 27.8 / 29.0 | 22.32 / 27.81 / 29.00 |
| YOLOE-v8-M | 32.6 | 32.64 | 26.9 / 31.9 / 34.4 | 26.87 / 31.87 / 34.35 |
| YOLOE-v8-L | 35.9 | 35.87 | 33.2 / 34.8 / 37.3 | 33.24 / 34.78 / 37.31 |

All three models match the paper after rounding to one decimal place.

## Important evaluation note

The official segmentation JSON serialization path encountered an OpenCV
mask-resize limitation when serializing up to 1000 predicted masks.

For detection Fixed AP reproduction, an official-equivalent bbox-only
serialization workaround was used.

Preserved:

- official YOLOE segmentation checkpoints
- MobileCLIP-B(LT) text prompts
- official preprocessing
- YOLOE SegmentationValidator postprocess
- multi-label NMS
- native box preparation
- category mapping
- DetectionValidator bbox JSON serialization
- official `tools/eval_fixed_ap.py`

Bypassed:

- predicted-mask decoding
- mask JSON serialization

No tracked official YOLOE source file was modified.

See `MANIFEST.json` and the reports/states directories for exact hashes,
configuration and execution evidence.
