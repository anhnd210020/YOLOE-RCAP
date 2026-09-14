# A1/A3 robust-vocabulary experiment package

Status at the 2026-09-14 resume audit: baseline training completed 30/30 epochs with `TRAIN_EXIT_CODE=0`; its `last.pt` exists. The baseline worktree is clean at `9a7065c3e8c75b358e023d84473cbd78de24e56f`. The robust worktree remains uncommitted. Existing CPU correctness suite: **PASS (11 tests in the preceding session; unchanged source at this audit, not rerun here)**. GPU smoke: **PENDING**. A1 long run: **NOT STARTED**. A3 long run: **NOT STARTED**. No LVIS evaluation has been performed for A1/A3.

## Initialization and locked settings

The pilot calls `YOLOE("yoloe-v8s-seg.yaml")`, then `YOLOE.train` constructs `PilotTrainOnlyTrainer` and calls its `get_model` with the YAML configuration. `BaseTrainer.__init__` calls `init_seeds(seed + 1 + RANK, deterministic=True)` before `get_model`; with this single-GPU run, `seed=0` and `RANK=-1`, so the effective seed is 0. `YOLOESegTrainer.get_model` constructs a fresh `YOLOESegModel` and loads weights only if a weight object is supplied. Here the YAML model has no checkpoint, `model.ckpt` is false, and the default `pretrained=True` is a boolean, not a detector-weight path. The initial `YOLOE` wrapper's model is discarded as training weights; A1/A3 each get a newly seeded detector. Neither starts from the baseline `last.pt` nor from the other arm.

The text model is official `mobileclip:blt`, loading the lock's `/workspace/YOLOE-RCAP/pilot/v8s_10pct_v1/data/text/checkpoints/mobileclip_blt.pt` (SHA256 `670844f7a886dd6eff7a9285adfc53f3d3c889c03bfc8354010cb5c6bf27441a`). The lock also pins train-label embeddings and global-negative categories/embeddings. The launcher hashes the complete freshly constructed detector `state_dict`, including named parameters and buffers, immediately after `get_model` and before the first optimizer update. A1 and A3 initial fingerprints must match exactly before either result is compared. The locked MobileCLIP and text-artifact hashes must also match.

The launcher reads the baseline pilot's existing, absolute-path dataset lock and calls its original complete input preflight. It does not copy datasets or rewrite the lock. The pilot preflight's own output-name check receives an unused positive sentinel batch because the completed baseline `main_batch_32` already exists; all actual training arguments and the manifest retain physical batch 32. Run outputs are reserved atomically under `robust_vocab/runs/`, separate from the pilot's `runs/main_batch_32`. There is no resume, automatic deletion, or overwrite.

Exact source-derived training settings: `yoloe-v8s-seg.yaml`, scale `s` override `scale=0.9`, `mixup=0.05`, `copy_paste=0.15`; segment task and text prompts; seed 0, deterministic true; 30 epochs; 640 pixels; physical batch 32; `nbs=128` and initial accumulation 4 (interpolated during warmup); AdamW; `lr0=0.002`; weight-decay argument 0.05 (scaled optimizer decay 0.05); momentum 0.9; linear scheduler (`cos_lr=False`, `lrf=0.01`); 3.0 warmup epochs, warmup momentum 0.8, warmup bias LR 0; close mosaic 2; 4 workers; AMP true; ModelEMA enabled; `save_period=-1`, normal `last.pt` after each epoch. The pilot's `PilotTrainOnlyTrainer` has no validation dataloader and returns placeholder fitness, while preserving normal checkpoint saving and EMA. Its `best.pt` is not a scientific selection criterion.

Training data are the same deterministic 10% locked Objects365v1, GQA, and Flickr30k subsets. The lock records respective subset hashes `2b413d71…e72ab9bb02746127ad48722527667da9003dd`, `023afe46…783f83a93dabe`, and `cf82081f…5e357dd04`; the full hashes are in the lock and each run manifest. The baseline Objects365 subset covers 364/365 classes (missing category 361), a locked warning that must not be repaired post hoc. The model gets `C=80` prompt slots in this pilot; the robust loss itself handles variable `C`.

## Launcher and manifest

CPU/file-only examples (review commands, not a launch):

```bash
cd /workspace/YOLOE-RCAP-robust-vocab
/workspace/yoloe-pilot-conda/bin/python robust_vocab/scripts/run_experiment.py A1 --dry-run
/workspace/yoloe-pilot-conda/bin/python robust_vocab/scripts/run_experiment.py A3 --tau 0.3 --dry-run
```

After separate run authorization, remove `--dry-run` for each arm. A1 forces `tau=0.0` and rejects any supplied `--tau`. A3 requires an explicit finite positive tau. Example names: `A1_tau_0p0_seed_0`, `A3_tau_0p3_seed_0`. The launcher refuses an existing directory, creates a fresh exact run directory, and writes `experiment_manifest.json` there. Its `git_dirty` and `git_status_porcelain` record the uncommitted method files; the full Git SHA remains the parent commit until a later commit. `dataset_lock_sha256`, three subset/manifest identities, model-config hash, initialization policy and fingerprint, MobileCLIP and text hashes, all principal training settings, output path, start/end UTC timestamps, status, last checkpoint path/hash, peak allocated CUDA bytes, and duration are recorded. `final_metrics` stays null until a separate locked evaluation. `best.pt` fitness is never copied into that field.

## Tau development plan

Do not tune or run tau during this preparation. A provisional positive grid is `0.1`, `0.3`, `1.0`: for a one-unit gap in negative BCE, it changes relative exponential weights by approximately 1.11, 1.35, and 2.72; for a five-unit gap, by approximately 1.65, 4.48, and 148.4. This probes weak, moderate, and aggressive tilting while retaining the same per-anchor `B_i` scale. The per-entry negative BCE distribution has **not** yet been measured, so inspect its quantiles on training/development batches before accepting this grid. No candidate is called optimal. A1 at tau zero is the mandatory control. Never choose tau using final LVIS minival AP.

## Locked final detection evaluation

The scientific checkpoint is **epoch-30 `weights/last.pt`**, with its hash in the manifest. Use the validated bbox Fixed-AP path on the **full 4,809-image LVIS v1 minival**, text prompt, `imgsz=640`, `conf=0.001`, `iou=0.7`, `max_det=1000`, `batch=1`, `rect=False`, `half=False`, `load_vp=False`, and `save_json=True`. Primary results are bbox AP, APr, APc, APf. Preserve any emitted AP50, AP75, APs, APm, APl, and AR as additional bbox metrics; **APm means medium-size bbox AP, not mask AP**. The current pilot evaluator's parser persists only the four primary AP values, so extend the robust-arm result recording to retain additional emitted metrics before either final evaluation. Do not select `best.pt` by the synthetic in-training fitness. Do not claim segmentation AP.

## Prepared GPU smoke (not executed)

`robust_vocab/scripts/gpu_smoke.py` is a future command only. After separate GPU authorization, run A1 and an explicitly chosen A3 tau in distinct smoke directories. It uses the real YOLOE model, locked datasets, pilot trainer, and a small training fraction. It asserts a head forward signature, unchanged parameter count, finite gradients and all four loss components (box, seg, cls, DFL), at least one optimizer step, robust branch use only for A3, and checkpoint argument tau save/reload. Compare the two recorded head signatures and parameter counts after both smokes. This procedure has **not** been executed or validated on CUDA in this task.

## Fairness contract

| Setting | A1 | A3 | Same? |
| --- | --- | --- | --- |
| Architecture | YOLOE-v8-S segmentation, text prompt | Same | Yes |
| Detector initialization | Fresh YAML model, seed 0; verify fingerprint | Same policy; must match fingerprint | Required |
| MobileCLIP | Locked BLT weights and text artifacts | Same hashes | Yes |
| Dataset lock | Complete baseline pilot lock | Same SHA256 | Yes |
| Objects365 subset | Locked 10%, 60,860 images | Same hash | Yes |
| GQA subset | Locked 10%, 62,114 image records | Same hash | Yes |
| Flickr30k subset | Locked 10%, 14,891 image records | Same hash | Yes |
| Seed / deterministic | 0 / true | 0 / true | Yes |
| Epochs / imgsz | 30 / 640 | 30 / 640 | Yes |
| Physical batch / nbs / accumulation | 32 / 128 / initially 4; warmup interpolation | Same | Yes |
| Optimizer / lr0 | AdamW / 0.002 | Same | Yes |
| Weight decay / momentum | argument 0.05 / 0.9 | Same | Yes |
| Scheduler / warmup / close mosaic | linear, lrf .01; 3 epochs; 2 | Same | Yes |
| Augmentation | Pinned defaults plus scale-s mixup .05, copy-paste .15 | Same | Yes |
| Workers / AMP / EMA | 4 / true / enabled | Same | Yes |
| Validation bypass | Pilot train-only adapter | Same | Yes |
| Final checkpoint | Epoch-30 `last.pt` | Epoch-30 `last.pt` | Yes |
| LVIS evaluator | Full minival bbox Fixed AP | Same locked protocol | Yes |
| `robust_vocab_tau` / classification negatives | `0.0` / original BCE | explicit positive / per-anchor robust aggregation | Intended difference |
