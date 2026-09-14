"""Focused CPU checks for region-wise robust vocabulary training."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, yaml_load, yaml_save
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.robust_vocab_loss import robust_vocab_classification_loss


BCE = nn.BCEWithLogitsLoss(reduction="none")
TAUS = (1e-8, 1e-5, 0.01, 0.1, 0.7, 1.0, 10.0, 100.0)
SCALAR_ATOL = 1e-5
GRAD_ATOL = 1e-6


def reference_fp64(logits, targets, normalizer, tau):
    """Direct, test-only definition using explicit q0 and exp in FP64."""
    logits64 = logits.double()
    targets64 = targets.double()
    positive = BCE(logits64, targets64)[targets64 > 0].sum()
    negative = logits64.new_zeros(())
    for image in range(logits64.shape[0]):
        for anchor in range(logits64.shape[1]):
            losses = F.softplus(logits64[image, anchor][targets64[image, anchor] == 0])
            b = losses.numel()
            if b:
                q0 = torch.full_like(losses, 1.0 / b)
                risk = torch.log((q0 * torch.exp(tau * losses)).sum()) / tau
                negative = negative + b * risk
    return (positive + negative) / torch.as_tensor(normalizer, dtype=torch.float64)


def logits_for_losses(losses):
    """Inverse softplus, used only to construct test examples."""
    losses = torch.as_tensor(losses, dtype=torch.float64)
    return torch.log(torch.expm1(losses)).float()


class CaptureBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = None

    def forward(self, scores, targets):
        self.scores = scores
        return F.binary_cross_entropy_with_logits(scores, targets, reduction="none")


class TinyHead:
    def __init__(self, classes):
        self.nc = classes
        self.reg_max = 2
        self.stride = torch.tensor([8.0])


class TinyModel(nn.Module):
    def __init__(self, classes, tau):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.model = [TinyHead(classes)]
        self.args = get_cfg(overrides={"robust_vocab_tau": tau})


class FixedAssigner:
    """Supply deterministic soft targets; leave real box/DFL/mask losses active."""

    def __init__(self, target_scores):
        self.target_scores = target_scores

    def __call__(self, scores, *args):
        batch, anchors, _ = scores.shape
        foreground = self.target_scores.sum(dim=-1) > 0
        boxes = torch.zeros(batch, anchors, 4)
        boxes[foreground] = torch.tensor([4.0, 4.0, 12.0, 12.0])
        return (
            torch.zeros(batch, anchors),
            boxes,
            self.target_scores,
            foreground,
            torch.zeros(batch, anchors, dtype=torch.long),
        )


def criterion_fixture(tau, target_scores):
    """Run the actual segmentation criterion with a tiny fixed-assignment batch."""
    assert target_scores.shape[0:2] == (1, 4)
    classes = target_scores.shape[-1]
    criterion = v8SegmentationLoss(TinyModel(classes, tau))
    criterion.assigner = FixedAssigner(target_scores)
    criterion.bce = CaptureBCE()
    channels = 8 + classes
    features = torch.linspace(-1.0, 1.0, channels * 4).reshape(1, channels, 2, 2).requires_grad_()
    coefficients = torch.linspace(-0.3, 0.4, 8).reshape(1, 2, 4).requires_grad_()
    prototypes = torch.linspace(-0.5, 0.6, 32).reshape(1, 2, 4, 4).requires_grad_()
    has_labels = bool((target_scores > 0).any())
    batch = {
        "batch_idx": torch.tensor([0.0]) if has_labels else torch.empty(0),
        "cls": torch.tensor([[0.0]]) if has_labels else torch.empty(0, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]) if has_labels else torch.empty(0, 4),
        "masks": torch.ones(1, 4, 4) if has_labels else torch.empty(0, 4, 4),
    }
    total, items = criterion(([features], coefficients, prototypes), batch)
    return criterion, total, items


class RobustVocabLossTests(unittest.TestCase):
    def test_tau_zero_actual_criterion_matches_baseline_scalar_and_gradient(self):
        max_scalar = max_gradient = 0.0
        cases = []
        for classes in (1, 4, 7):
            targets = torch.zeros(1, 4, classes)
            targets[0, 0, 0] = 0.6
            cases.append(targets)
        cases.append(torch.zeros(1, 4, 7))  # empty-positive/all-background batch
        for targets in cases:
            with self.subTest(classes=targets.shape[-1], empty_positive=not bool(targets.any())):
                with patch("ultralytics.utils.loss.robust_vocab_classification_loss", side_effect=AssertionError("tau=0 called helper")):
                    criterion, total, items = criterion_fixture(0.0, targets)
                scores = criterion.bce.scores
                self.assertEqual(tuple(scores.shape), tuple(targets.shape))
                normalizer = max(targets.sum(), 1)
                baseline = BCE(scores, targets.to(scores.dtype)).sum() / normalizer
                scalar_difference = abs(float(items[2] - (baseline * criterion.hyp.cls).detach()))
                actual_gradient = torch.autograd.grad(total, scores, retain_graph=True)[0]
                baseline_gradient = torch.autograd.grad(baseline * criterion.hyp.cls, scores)[0]
                gradient_difference = float((actual_gradient - baseline_gradient).abs().max())
                max_scalar = max(max_scalar, scalar_difference)
                max_gradient = max(max_gradient, gradient_difference)
                self.assertLessEqual(scalar_difference, 0.0)
                self.assertLessEqual(gradient_difference, 0.0)
                self.assertEqual(items.shape, (4,))
        print(f"tau=0 max scalar difference={max_scalar:.9g}; max pred_scores gradient difference={max_gradient:.9g}; tolerance=0")

    def test_actual_criterion_other_loss_items_and_aggregation_unchanged(self):
        targets = torch.zeros(1, 4, 4)
        targets[0, 0, 0] = 0.6
        baseline_criterion, baseline_total, baseline_items = criterion_fixture(0.0, targets)
        robust_criterion, robust_total, robust_items = criterion_fixture(0.7, targets)
        for index in (0, 1, 3):  # real box, segmentation, and DFL paths
            self.assertEqual(float(baseline_items[index]), float(robust_items[index]))
            self.assertGreater(float(baseline_items[index]), 0.0)
        self.assertNotEqual(float(baseline_items[2]), float(robust_items[2]))
        self.assertAlmostEqual(float(baseline_total), float(baseline_items.sum()), places=5)
        self.assertAlmostEqual(float(robust_total), float(robust_items.sum()), places=5)
        self.assertEqual(baseline_criterion.hyp.box, robust_criterion.hyp.box)
        self.assertEqual(baseline_criterion.hyp.dfl, robust_criterion.hyp.dfl)

    def test_fp32_production_matches_direct_fp64_reference(self):
        generator = torch.Generator().manual_seed(20260913)
        max_scalar = max_gradient = 0.0
        per_tau = {tau: 0.0 for tau in TAUS}
        for batch, anchors, classes in ((1, 2, 1), (1, 3, 4), (2, 4, 7), (2, 2, 13)):
            scores = torch.randn(batch, anchors, classes, generator=generator) * 1.2
            targets = torch.zeros_like(scores)
            targets[0, 0, 0] = 0.6
            if classes > 1:
                targets[-1, -1, :] = 0.2  # B_i = 0 row
            normalizer = max(targets.sum(), 1)
            for tau in TAUS:
                with self.subTest(shape=scores.shape, tau=tau):
                    production_scores = scores.detach().clone().requires_grad_()
                    reference_scores = scores.double().detach().requires_grad_()
                    actual = robust_vocab_classification_loss(production_scores, targets, normalizer, BCE, tau)
                    expected = reference_fp64(reference_scores, targets, normalizer, tau)
                    actual_gradient = torch.autograd.grad(actual, production_scores)[0]
                    expected_gradient = torch.autograd.grad(expected, reference_scores)[0]
                    scalar_error = abs(float(actual) - float(expected))
                    gradient_error = float((actual_gradient.double() - expected_gradient).abs().max())
                    max_scalar = max(max_scalar, scalar_error)
                    max_gradient = max(max_gradient, gradient_error)
                    per_tau[tau] = max(per_tau[tau], scalar_error)
                    self.assertLessEqual(scalar_error, SCALAR_ATOL)
                    self.assertLessEqual(gradient_error, GRAD_ATOL)
                    self.assertEqual(actual.dtype, torch.float32)
        print(f"FP32 vs direct FP64 max scalar error={max_scalar:.9g} (atol={SCALAR_ATOL}); max gradient error={max_gradient:.9g} (atol={GRAD_ATOL})")
        print("FP32 vs FP64 max scalar error by tau: " + ", ".join(f"{tau:g}={per_tau[tau]:.9g}" for tau in TAUS))

    def test_required_three_loss_example_and_detached_tilt_diagnostic(self):
        losses = torch.tensor([0.01, 0.02, 1.00], dtype=torch.float64)
        scores = logits_for_losses(losses).reshape(1, 1, 3)
        targets = torch.zeros_like(scores)
        production_risk = robust_vocab_classification_loss(scores, targets, 1, BCE, 1.0) / 3
        reference_risk = reference_fp64(scores.double(), targets.double(), 1, 1.0) / 3
        q_star = torch.softmax(losses.detach(), dim=0)
        self.assertAlmostEqual(float(losses.mean()), 0.3433333333333333, places=12)
        self.assertAlmostEqual(float(reference_risk), 0.45922351023976415, delta=1e-6)
        self.assertAlmostEqual(float(production_risk), float(reference_risk), delta=1e-6)
        torch.testing.assert_close(q_star, torch.tensor([0.21270782, 0.21484557, 0.57244661], dtype=torch.float64), atol=1e-8, rtol=0)

    def test_risk_bounds_equal_losses_and_monotonic_tilt(self):
        losses = torch.tensor([0.1, 0.7, 2.0])
        scores = logits_for_losses(losses).reshape(1, 1, 3)
        targets = torch.zeros_like(scores)
        risks = [float(robust_vocab_classification_loss(scores, targets, 1, BCE, tau) / 3) for tau in TAUS]
        for risk in risks:
            self.assertGreaterEqual(risk + 2e-6, float(losses.mean()))
            self.assertLessEqual(risk, float(losses.max()) + 2e-6)
        for earlier, later in zip(risks, risks[1:]):
            self.assertLessEqual(earlier, later + 2e-6)
        equal_loss = 1.25
        equal_scores = logits_for_losses([equal_loss] * 5).reshape(1, 1, 5)
        for tau in TAUS:
            risk = robust_vocab_classification_loss(equal_scores, torch.zeros_like(equal_scores), 1, BCE, tau) / 5
            self.assertAlmostEqual(float(risk), equal_loss, delta=2e-6)

    def test_one_negative_keeps_its_original_bce(self):
        scores = torch.tensor([[[0.4, -1.0, 1.2]]])
        targets = torch.tensor([[[0.2, 0.7, 0.0]]])
        positive = BCE(scores, targets)[targets > 0].sum()
        original_negative = F.softplus(scores[targets == 0]).sum()
        for tau in TAUS:
            result = robust_vocab_classification_loss(scores, targets, max(targets.sum(), 1), BCE, tau)
            self.assertAlmostEqual(float(result), float(positive + original_negative), delta=2e-6)

    def test_soft_positive_uses_complete_bce(self):
        scores = torch.tensor([[[1.2, -0.5]]])
        targets = torch.tensor([[[0.3, 0.7]]])  # B_i = 0
        actual = robust_vocab_classification_loss(scores, targets, max(targets.sum(), 1), BCE, 0.7)
        complete = BCE(scores, targets).sum()
        hard_positive = BCE(scores, torch.ones_like(targets)).sum()
        simplified = (-targets * F.logsigmoid(scores)).sum()
        self.assertAlmostEqual(float(actual), float(complete), delta=1e-7)
        self.assertGreater(abs(float(actual - hard_positive)), 0.1)
        self.assertGreater(abs(float(actual - simplified)), 0.1)

    def test_region_wise_axis_and_background_anchor(self):
        losses = torch.tensor([[[0.01, 0.02, 3.0], [1.0, 1.1, 1.2]], [[0.2, 0.4, 0.9], [0.3, 0.6, 1.5]]])
        scores = logits_for_losses(losses).reshape(2, 2, 3)
        targets = torch.zeros_like(scores)
        targets[0, 1, 0] = 0.5  # a soft-positive foreground anchor beside a background anchor
        self.assertEqual(int((targets[0, 0] == 0).sum()), scores.shape[-1])  # background B_i = C
        actual = robust_vocab_classification_loss(scores, targets, 1, BCE, 1.0)
        expected = reference_fp64(scores.double(), targets.double(), 1, 1.0)
        negative_losses = F.softplus(scores.double())[targets == 0]
        positive_bce = BCE(scores.double(), targets.double())[targets > 0].sum()
        pooled_wrong = positive_bce + negative_losses.numel() * torch.log(torch.exp(negative_losses).mean())
        self.assertAlmostEqual(float(actual), float(expected), delta=3e-6)
        self.assertGreater(abs(float(actual - pooled_wrong)), 0.05)
        without_background = reference_fp64(scores.reshape(-1, 3)[1:].reshape(1, 3, 3),
                                            targets.reshape(-1, 3)[1:].reshape(1, 3, 3), 1, 1.0)
        self.assertGreater(float(actual), float(without_background))

    def test_zero_negative_rows_and_empty_positive_clamp(self):
        all_positive = torch.tensor([[[0.3, 0.5, 0.2]]])
        scores = torch.tensor([[[0.2, -0.7, 1.0]]], requires_grad=True)
        actual = robust_vocab_classification_loss(scores, all_positive, max(all_positive.sum(), 1), BCE, 0.7)
        self.assertAlmostEqual(float(actual), float(BCE(scores, all_positive).sum()), delta=1e-7)
        actual.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        all_background = torch.zeros(2, 3, 7)
        self.assertEqual(max(all_background.sum(), 1), 1)
        background_scores = torch.randn(2, 3, 7, requires_grad=True)
        background = robust_vocab_classification_loss(background_scores, all_background, max(all_background.sum(), 1), BCE, 0.7)
        self.assertAlmostEqual(float(background), float(reference_fp64(background_scores, all_background, 1, 0.7)), delta=3e-6)
        background.backward()
        self.assertTrue(torch.isfinite(background_scores.grad).all())

    def test_large_logits_and_low_precision_input(self):
        targets = torch.zeros(1, 2, 4)
        targets[0, 0, 0] = 0.6
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                scores = torch.tensor([[[80.0, -80.0, 40.0, -40.0], [-70.0, 70.0, -30.0, 30.0]]], dtype=dtype, requires_grad=True)
                result = robust_vocab_classification_loss(scores, targets, max(targets.sum(), 1), BCE, 100.0)
                self.assertEqual(result.dtype, torch.float32)
                self.assertTrue(torch.isfinite(result))
                result.backward()
                self.assertTrue(torch.isfinite(scores.grad).all())

    def test_config_validation_and_yaml_serialization(self):
        self.assertIn("robust_vocab_tau", DEFAULT_CFG_DICT)
        self.assertEqual(DEFAULT_CFG.robust_vocab_tau, 0.0)
        self.assertEqual(get_cfg().robust_vocab_tau, 0.0)
        for valid in (0, 0.0, 0.7):
            self.assertEqual(get_cfg(overrides={"robust_vocab_tau": valid}).robust_vocab_tau, valid)
        for invalid in (-1.0, float("nan"), float("inf"), -float("inf"), True, False, None):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                get_cfg(overrides={"robust_vocab_tau": invalid})
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "args.yaml"
            yaml_save(path, vars(get_cfg(overrides={"robust_vocab_tau": 0.7})))
            self.assertEqual(yaml_load(path)["robust_vocab_tau"], 0.7)

if __name__ == "__main__":
    unittest.main(verbosity=2)
