# Region-wise Robust Vocabulary Training for YOLOE

This directory contains the controlled training and evaluation pipeline used to study **Region-wise Robust Vocabulary Training** on top of YOLOE-v8-S.

The completed experiments in this branch are:

- **A1**: paired BCE control with `robust_vocab_tau = 0.0`
- **A3**: robust vocabulary training with `robust_vocab_tau = 0.3`

A1 and A3 use the same model architecture, dataset split, initialization policy, training schedule, random seed, and evaluation protocol. The intended experimental difference is only the robust-vocabulary aggregation controlled by `robust_vocab_tau`.

## Problem

Open-vocabulary object detectors classify each region against a large text vocabulary. During training, most region-category pairs are negatives. Treating all negative entries independently with ordinary BCE can make the classification objective sensitive to the size and composition of the negative vocabulary.

This project investigates a region-wise robust aggregation of negative vocabulary losses while preserving the original positive-target supervision.

The method changes the **training classification objective only**. It does not add a new detection head, inference-time parameters, vocabulary re-ranking, or post-processing.

## Method

For a negative logit \(z_{ij}\), the BCE term is

\[
\ell^-_{ij} = \operatorname{softplus}(z_{ij}).
\]

For region \(i\), let \(w_{ij}\) be the negative-entry weight and

\[
B_i = \sum_{j \in \mathcal{N}_i} w_{ij}, \qquad
q_{ij} = \frac{w_{ij}}{B_i}.
\]

For \(\tau > 0\), the negative losses are aggregated as

\[
R_{\tau,i}
=
\frac{1}{\tau}
\log
\left(
\sum_{j \in \mathcal{N}_i}
q_{ij}\exp(\tau \ell^-_{ij})
\right).
\]

The negative contribution of region \(i\) becomes \(B_iR_{\tau,i}\). Positive soft-target BCE terms are kept unchanged.

The robust implementation is in:

```text
ultralytics/utils/robust_vocab_loss.py
```

and is integrated into YOLOE through:

```text
ultralytics/utils/loss.py
```

When `robust_vocab_tau = 0.0`, the code dispatches to the original YOLOE BCE path exactly. Therefore A1 is the paired control.

## Experimental protocol

Both A1 and A3 use:

| Setting | Value |
|---|---|
| Model | YOLOE-v8-S segmentation |
| Training stage | Stage-1 text-prompt training |
| Training data | deterministic 10% Objects365v1 + GQA + Flickr30k pilot |
| Image size | 640 |
| Physical batch size | 32 |
| Nominal batch size | 128 |
| Epochs | 30 |
| Optimizer | AdamW |
| Learning rate | 0.002 |
| Momentum | 0.9 |
| Workers | 4 |
| Seed | 0 |
| Deterministic mode | enabled |
| Text encoder | MobileCLIP-BLT |
| Device | 1 GPU |
| Final evaluation | full LVIS v1 minival Fixed AP |

The exact dataset and text artifacts are pinned by:

```text
pilot/v8s_10pct_v1/PILOT_DATASET_LOCK.json
```

Dataset-lock SHA256:

```text
8b7fad638f8b16dc97cca3ad9640d0d12b0b9413ca69be98afa148ea6175f058
```

Raw Objects365v1, GQA, Flickr30k, and LVIS datasets are not redistributed.

## Environment used for the completed runs

```text
Python 3.10
PyTorch 2.5.1 + CUDA 12.1
torchvision 0.20.1
1 x NVIDIA RTX 3090 24 GB
```

Reference interpreter:

```bash
/workspace/yoloe-pilot-conda/bin/python3.10
```

Select the experiment branch:

```bash
git checkout research/robust-vocab-v1
```

Run all commands below from the repository root.

## Optional GPU smoke test

A1:

```bash
/workspace/yoloe-pilot-conda/bin/python3.10 \
  robust_vocab/scripts/gpu_smoke.py A1
```

A3:

```bash
/workspace/yoloe-pilot-conda/bin/python3.10 \
  robust_vocab/scripts/gpu_smoke.py A3 --tau 0.3
```

The smoke test exercises a real forward/backward/optimizer update and checks AMP behavior, initialization information, cache isolation, and persistence of `robust_vocab_tau`.

## Dry-run validation

A1:

```bash
/workspace/yoloe-pilot-conda/bin/python3.10 \
  robust_vocab/scripts/run_experiment.py A1 --dry-run
```

A3:

```bash
/workspace/yoloe-pilot-conda/bin/python3.10 \
  robust_vocab/scripts/run_experiment.py A3 --tau 0.3 --dry-run
```

For A1, `--tau` must be omitted. For A3, a finite positive `--tau` must be supplied explicitly.

## Full training

A1:

```bash
/workspace/yoloe-pilot-conda/bin/python3.10 \
  robust_vocab/scripts/run_experiment.py A1
```

Expected output:

```text
robust_vocab/runs/A1_tau_0p0_seed_0/
```

A3:

```bash
/workspace/yoloe-pilot-conda/bin/python3.10 \
  robust_vocab/scripts/run_experiment.py A3 --tau 0.3
```

Expected output:

```text
robust_vocab/runs/A3_tau_0p3_seed_0/
```

Both runs start from a fresh YOLOE-v8-S detector initialization. A3 is not continued from A1.

On the reference RTX 3090 server, each 30-epoch run took approximately 12.5 hours.

## LVIS Fixed AP evaluation

A1:

```bash
/workspace/yoloe-pilot-conda/bin/python3.10 \
  pilot/v8s_10pct_v1/scripts/eval_pilot.py \
  --checkpoint robust_vocab/runs/A1_tau_0p0_seed_0/weights/last.pt \
  --lock pilot/v8s_10pct_v1/PILOT_DATASET_LOCK.json \
  --name A1_tau0_epoch30_fixed_ap
```

A3:

```bash
/workspace/yoloe-pilot-conda/bin/python3.10 \
  pilot/v8s_10pct_v1/scripts/eval_pilot.py \
  --checkpoint robust_vocab/runs/A3_tau_0p3_seed_0/weights/last.pt \
  --lock pilot/v8s_10pct_v1/PILOT_DATASET_LOCK.json \
  --name A3_tau0p3_epoch30_fixed_ap
```

Evaluation uses the full LVIS v1 minival set with 4,809 images. The main result file is `fixed_ap_result.json` inside the corresponding directory under:

```text
pilot/v8s_10pct_v1/runs/eval/
```

## Completed results

| Metric | A1: tau=0.0 | A3: tau=0.3 | A3 - A1 |
|---|---:|---:|---:|
| AP | 19.35 | 19.04 | -0.31 |
| AP50 | 27.67 | 27.28 | -0.39 |
| AP75 | 20.66 | 20.27 | -0.39 |
| APs | 12.50 | 12.32 | -0.18 |
| APm | 26.11 | 26.02 | -0.09 |
| APl | 37.79 | 38.46 | +0.67 |
| APr | 14.50 | 14.32 | -0.18 |
| APc | 18.04 | 16.98 | -1.06 |
| APf | 21.38 | 21.71 | +0.33 |
| AR | 34.60 | 34.30 | -0.30 |
| ARs | 18.40 | 18.00 | -0.40 |
| ARm | 42.00 | 41.90 | -0.10 |
| ARl | 56.80 | 58.30 | +1.50 |

At `tau = 0.3`, the robust objective does not improve overall AP over the paired BCE control. A3 improves APl, APf, and ARl, but overall AP and APc are lower. These results are reported as an experimental finding, not as evidence of an overall improvement.

## Checkpoints and provenance

A1 checkpoint SHA256:

```text
bbc77e0927366a6b333d58274cb661b6a8c97862d1b55f01521633877fa6dd44
```

A3 checkpoint SHA256:

```text
a58af4c66fae9c5ed41081e7d86b6528503e9fc0fa810ba659615e820cf7a31d
```

Evaluation implementation commit:

```text
f2375e7eefa577044dc8d979a38438c8be63e1bd
```

A1 prediction SHA256:

```text
31e97a1d0397550f33d7b1ce6783056adb226cc6102144f783cb189449afc5ef
```

A3 prediction SHA256:

```text
2deb255c5a9b6fb9bfa0d1a76e5baa7e2482d54c772936b6affd16fcc94711bf
```

## Reproducibility artifacts

Large experiment artifacts are stored separately from Git.

Hugging Face repository:

```text
anhnd210020/yoloe-pilot-repro-assets
```

Archive:

```text
artifacts/robust_vocab/yoloe_robust_vocab_A1_A3_full_v2_20260915.tar.gz
```

Archive SHA256:

```text
22ae71bb2bebc980ef340356de37ca8bd1717f52f35991c85ade98693dd87b0c
```

The archive contains the A1/A3 final checkpoints, experiment manifests, training CSV files, LVIS Fixed AP result JSON files, prediction JSON files, dataset lock, training logs, and evaluation logs.

## Main source files

```text
ultralytics/utils/robust_vocab_loss.py
ultralytics/utils/loss.py
robust_vocab/scripts/run_experiment.py
robust_vocab/scripts/gpu_smoke.py
robust_vocab/scripts/smoke_utils.py
pilot/v8s_10pct_v1/scripts/eval_pilot.py
```

The implementation is intentionally narrow: the training classification loss changes while the YOLOE inference architecture remains unchanged.
