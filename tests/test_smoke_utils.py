"""CPU-only checks of the smoke diagnostic order and cache redirection."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from robust_vocab.scripts.smoke_utils import (StopAfterUpdateLoader, checked_optimizer_step,
                                               ordered_file_identity, route_dataset_cache)


class FakeScaler:
    def __init__(self, events):
        self.events = events
        self.scale = 4.0
        self.found_inf = False

    def get_scale(self):
        return self.scale

    def unscale_(self, optimizer):
        self.events.append("unscale")
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(4)
                    self.found_inf |= not bool(torch.isfinite(parameter.grad).all())

    def step(self, optimizer):
        if self.found_inf:
            self.events.append("skipped")
        else:
            self.events.append("step")
            optimizer.step()

    def update(self):
        self.events.append("update")
        if self.found_inf:
            self.scale /= 2


def empty_report():
    return {"optimizer_attempts": 0, "actual_optimizer_updates": 0,
            "amp_skipped_updates": 0, "optimizer_steps": 0,
            "box_seg_cls_dfl_finite": True, "attempt_diagnostics": []}


class SmokeUtilsTests(unittest.TestCase):
    def test_ordered_batch_identity_accepts_paths_and_preserves_order(self):
        first = [Path("/data/a.jpg"), Path("/data/b.jpg")]
        self.assertEqual(ordered_file_identity(first), ordered_file_identity([str(p) for p in first]))
        self.assertNotEqual(ordered_file_identity(first), ordered_file_identity(list(reversed(first))))

    def test_loader_ends_epoch_after_first_update_without_extra_batch(self):
        class Loader:
            num_workers = 4

            def __init__(self):
                self.yielded = 0

            def __len__(self):
                return 20

            def __iter__(self):
                for value in range(20):
                    self.yielded += 1
                    yield value

        underlying = Loader()
        report = {"actual_optimizer_updates": 0}
        loader = StopAfterUpdateLoader(underlying, report)
        self.assertEqual(len(loader), 20)
        self.assertEqual(loader.num_workers, 4)
        iterator = iter(loader)
        self.assertEqual(next(iterator), 0)
        self.assertEqual(next(iterator), 1)
        report["actual_optimizer_updates"] = 1
        with self.assertRaises(StopIteration):
            next(iterator)
        self.assertEqual(underlying.yielded, 2)

    def trainer(self, value):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([value])
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        events = []
        original_zero_grad = optimizer.zero_grad

        def zero_grad():
            events.append("zero_grad")
            original_zero_grad()

        optimizer.zero_grad = zero_grad
        ema = SimpleNamespace(update=lambda model: events.append("ema"))
        model = torch.nn.Linear(1, 1)
        model.parameters = lambda: iter([parameter])
        model.named_parameters = lambda: iter([("weight", parameter)])
        trainer = SimpleNamespace(model=model, optimizer=optimizer, scaler=FakeScaler(events),
                                  ema=ema, tloss=torch.ones(4), loss_items=torch.ones(4))
        return trainer, parameter, events

    def test_diagnostics_follow_unscale_and_preserve_step_order(self):
        trainer, parameter, events = self.trainer(8.0)
        report = empty_report()
        checked_optimizer_step(trainer, report)
        self.assertEqual(events, ["unscale", "step", "update", "zero_grad", "ema"])
        self.assertEqual(report["parameters_with_grad"], 1)
        self.assertEqual(report["finite_gradient_tensors"], 1)
        self.assertEqual(report["nonfinite_gradient_tensors"], 0)
        self.assertEqual(report["max_abs_finite_unscaled_grad"], 2.0)
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertEqual(report["optimizer_attempts"], 1)
        self.assertEqual(report["actual_optimizer_updates"], 1)
        self.assertEqual(report["amp_skipped_updates"], 0)
        self.assertEqual(report["attempt_diagnostics"][0]["scaler_scale_before"], 4.0)
        self.assertEqual(report["attempt_diagnostics"][0]["scaler_scale_after"], 4.0)
        self.assertAlmostEqual(float(parameter), 0.8)

    def test_nonfinite_after_unscale_allows_scaler_skip_and_scale_reduction(self):
        trainer, _, events = self.trainer(float("inf"))
        report = empty_report()
        checked_optimizer_step(trainer, report)
        self.assertEqual(events, ["unscale", "skipped", "update", "zero_grad", "ema"])
        self.assertEqual(report["nonfinite_gradient_tensors"], 1)
        self.assertEqual(report["optimizer_attempts"], 1)
        self.assertEqual(report["actual_optimizer_updates"], 0)
        self.assertEqual(report["amp_skipped_updates"], 1)
        self.assertEqual(report["attempt_diagnostics"][0]["scaler_scale_after"], 2.0)

    def test_no_gradients_is_distinct_failure(self):
        trainer, parameter, events = self.trainer(1.0)
        parameter.grad = None
        report = empty_report()
        with self.assertRaisesRegex(AssertionError, "No gradients after GradScaler unscale"):
            checked_optimizer_step(trainer, report)
        self.assertEqual(events, ["unscale"])
        self.assertEqual(report["parameters_with_grad"], 0)

    def test_real_cpu_gradscaler_skip_then_actual_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        model = torch.nn.Linear(1, 1)
        model.parameters = lambda: iter([parameter])
        model.named_parameters = lambda: iter([("weight", parameter)])
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=4.0)
        trainer = SimpleNamespace(model=model, optimizer=optimizer, scaler=scaler,
                                  ema=None, loss_items=torch.ones(4))
        report = empty_report()
        scaler.scale(parameter.sum() * float("inf")).backward()
        checked_optimizer_step(trainer, report)
        self.assertEqual(report["optimizer_attempts"], 1)
        self.assertEqual(report["actual_optimizer_updates"], 0)
        self.assertEqual(report["amp_skipped_updates"], 1)
        self.assertEqual(report["attempt_diagnostics"][0]["scaler_scale_after"], 2.0)
        self.assertEqual(float(parameter), 1.0)
        scaler.scale(parameter.sum() * 2).backward()
        checked_optimizer_step(trainer, report)
        self.assertEqual(report["optimizer_attempts"], 2)
        self.assertEqual(report["actual_optimizer_updates"], 1)
        self.assertEqual(report["amp_skipped_updates"], 1)
        self.assertAlmostEqual(float(parameter), 0.8)

    def test_cache_saves_only_under_smoke_output_and_restores_functions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pilot = root / "YOLOE-RCAP/pilot/v8s_10pct_v1"
            baseline_cache = pilot / "data/prepared/objects365/labels/train.cache"
            output = root / "robust/smoke_A3"
            output.mkdir(parents=True)
            saved = []

            def original_load(path):
                return Path(path).read_text(encoding="utf-8")

            def original_save(prefix, path, contents, version):
                Path(path).write_text(contents, encoding="utf-8")

            module = SimpleNamespace(load_dataset_cache_file=original_load,
                                     save_dataset_cache_file=original_save)
            with route_dataset_cache(output, pilot, saved, dataset_module=module):
                module.save_dataset_cache_file("train", baseline_cache, "full", "1")
                self.assertEqual(module.load_dataset_cache_file(baseline_cache), "full")
            self.assertIs(module.load_dataset_cache_file, original_load)
            self.assertIs(module.save_dataset_cache_file, original_save)
            self.assertFalse(baseline_cache.exists())
            self.assertEqual(len(saved), 1)
            self.assertTrue(Path(saved[0]).is_relative_to(output / "dataset_cache"))


if __name__ == "__main__":
    unittest.main()
