"""Explicitly invoked YOLOE A1/A3 GPU smoke with AMP-step diagnostics.

Runs the real pilot trainer on a tiny fraction of the locked datasets, checks a
training forward/backward/optimizer step, and writes a separate smoke report.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import json
import os

from run_experiment import (PILOT, RUNS, model_fingerprint, paths_from_lock, read_lock, validate_baseline,
                            run_name, settings, tau_for)
from smoke_utils import StopAfterUpdateLoader, checked_optimizer_step, ordered_file_identity, route_dataset_cache


def signature(value):
    if isinstance(value, (tuple, list)):
        return [signature(item) for item in value]
    if isinstance(value, dict):
        return {key: signature(item) for key, item in sorted(value.items())}
    if hasattr(value, "shape"):
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    return type(value).__name__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("A1", "A3"))
    parser.add_argument("--tau", help="Required for A3; omit for A1")
    args = parser.parse_args()
    tau, tau_text = tau_for(args.experiment, args.tau)
    validate_baseline()
    lock_path, lock = read_lock()
    output = RUNS / ("smoke_" + run_name(args.experiment, tau_text, 0))
    if output.exists():
        raise FileExistsError(output)
    baseline_cache = PILOT / "data/prepared/objects365/labels/train.cache"
    if baseline_cache.exists():
        raise FileExistsError(f"Baseline dataset cache must remain absent: {baseline_cache}")

    # Same immutable-input preflight as the main experiment launcher. The sentinel
    # bypasses only the completed baseline's own output-name collision check.
    from train_pilot import preflight
    preflight(SimpleNamespace(**paths_from_lock(lock), physical_batch=1_000_000_000, smoke=False))
    import torch
    from ultralytics import YOLOE
    from ultralytics.utils import loss as loss_module
    from pilot_text import install_pilot_text
    from pilot_trainer import PilotSmokeTrainer, PilotYOLOESegTrainerFromScratch

    install_pilot_text(lock["text_artifacts"])
    PilotYOLOESegTrainerFromScratch.lock_path = str(lock_path)
    model_path, kwargs, _ = settings(tau, 32, lock)
    kwargs.update(epochs=1, fraction=0.005, val=False, save=False,
                  project=str(RUNS), name=output.name, exist_ok=True)
    report = {"experiment": args.experiment, "tau": tau, "seed": 0,
              "physical_batch": 32, "fraction": 0.005, "epochs": 1,
              "initialization_fingerprint": None, "first_batch_identity_sha256": None,
              "initial_gradscaler_scale": None, "available_batches": None,
              "actual_training_batch": False,
              "robust_branch_calls": 0, "optimizer_attempts": 0,
              "actual_optimizer_updates": 0, "amp_skipped_updates": 0,
              "optimizer_steps": 0, "attempt_diagnostics": [],
              "scaler_scale_before": None, "scaler_scale_after": None,
              "loss_items": None, "loss_items_finite": False,
              "head_output_signature": None,
              "parameter_count": None, "box_seg_cls_dfl_finite": True,
              "parameters_with_grad": 0, "finite_gradient_tensors": 0,
              "nonfinite_gradient_tensors": 0, "max_abs_finite_unscaled_grad": None,
              "generated_dataset_cache_paths": [], "baseline_dataset_cache_absent": None,
              "checkpoint_tau_roundtrip": False, "passed": False}
    original_robust = loss_module.robust_vocab_classification_loss

    def counted_robust(*a, **kw):
        report["robust_branch_calls"] += 1
        return original_robust(*a, **kw)

    class SmokeTrainer(PilotSmokeTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is not None:
                raise ValueError("Smoke must initialize a fresh detector")
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            report["parameter_count"] = sum(p.numel() for p in model.parameters())
            report["initialization_fingerprint"] = model_fingerprint(model)
            return model

        def _setup_train(self, world_size):
            super()._setup_train(world_size)
            report["initial_gradscaler_scale"] = float(self.scaler.get_scale())
            report["available_batches"] = len(self.train_loader)
            self.train_loader = StopAfterUpdateLoader(self.train_loader, report)
            head = self.model.model[-1]

            def capture(_module, _inputs, output):
                report["head_output_signature"] = signature(output)

            head.register_forward_hook(capture)

        def preprocess_batch(self, batch):
            if report["first_batch_identity_sha256"] is None:
                names = batch.get("im_file")
                if names:
                    report["first_batch_identity_sha256"] = ordered_file_identity(names)
            return super().preprocess_batch(batch)

        def optimizer_step(self):
            checked_optimizer_step(self, report)

    RUNS.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    old_cwd = Path.cwd()
    loss_module.robust_vocab_classification_loss = counted_robust
    try:
        os.chdir(Path(lock["text_artifacts"]["mobileclip_checkpoint"]).parent)
        model = YOLOE(model_path)
        initial_count = sum(p.numel() for p in model.model.parameters())
        overrides = dict(kwargs)
        overrides.update(model=model.overrides["model"], task=model.task, mode="train")
        with route_dataset_cache(output, PILOT, report["generated_dataset_cache_paths"]):
            trainer = SmokeTrainer(overrides=overrides, _callbacks=model.callbacks)
            trainer.model = trainer.get_model(weights=model.model if model.ckpt else None, cfg=model.model.yaml)
            trainer.train()
        if baseline_cache.exists():
            raise AssertionError(f"Smoke created a baseline dataset cache: {baseline_cache}")
        if report["parameter_count"] != initial_count:
            raise AssertionError("Tau changed model parameter count")
        if not report["actual_training_batch"] or report["actual_optimizer_updates"] < 1:
            raise AssertionError("Smoke completed without an actual optimizer update")
        if not report["box_seg_cls_dfl_finite"]:
            raise AssertionError("A training loss item was nonfinite")
        if report["head_output_signature"] is None:
            raise AssertionError("YOLOE head did not execute")
        if (args.experiment == "A3") != (report["robust_branch_calls"] > 0):
            raise AssertionError("Wrong classification loss branch executed")
        # The real trainer's smoke adapter intentionally skips ordinary checkpoints;
        # test checkpoint train_args persistence explicitly here.
        checkpoint = output / "smoke_checkpoint.pt"
        torch.save({"train_args": vars(trainer.args), "model": trainer.model.state_dict()}, checkpoint)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        report["checkpoint_tau_roundtrip"] = loaded["train_args"]["robust_vocab_tau"] == tau
        if not report["checkpoint_tau_roundtrip"]:
            raise AssertionError("Tau did not survive checkpoint save/reload")
        report["passed"] = True
    finally:
        loss_module.robust_vocab_classification_loss = original_robust
        os.chdir(old_cwd)
        report["baseline_dataset_cache_absent"] = not baseline_cache.exists()
        with (output / "smoke_report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
