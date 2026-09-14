"""Smoke-only diagnostics and YOLO cache routing; no normal-training patching."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


def ordered_file_identity(names):
    """Hash the ordered first-batch image paths without altering the batch."""
    payload = json.dumps([str(name) for name in names], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class StopAfterUpdateLoader:
    """End one smoke epoch normally once the first real optimizer update occurs."""

    def __init__(self, loader, report):
        self.loader = loader
        self.report = report

    def __iter__(self):
        iterator = iter(self.loader)
        while self.report["actual_optimizer_updates"] < 1:
            try:
                yield next(iterator)
            except StopIteration:
                return

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)


def gradient_diagnostics(named_parameters):
    """Read detached, already-unscaled gradients without changing them."""
    import torch

    result = {"parameters_with_grad": 0, "finite_gradient_tensors": 0,
              "nonfinite_gradient_tensors": 0, "max_abs_finite_unscaled_grad": None,
              "nonfinite_gradient_parameters": []}
    maximum = None
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        result["parameters_with_grad"] += 1
        grad = parameter.grad.detach()
        finite = torch.isfinite(grad)
        if bool(finite.all()):
            result["finite_gradient_tensors"] += 1
        else:
            result["nonfinite_gradient_tensors"] += 1
            result["nonfinite_gradient_parameters"].append({
                "name": name,
                "nan": int(torch.isnan(grad).sum().item()),
                "posinf": int(torch.isposinf(grad).sum().item()),
                "neginf": int(torch.isneginf(grad).sum().item()),
            })
        if bool(finite.any()):
            value = float(grad[finite].abs().max().item())
            maximum = value if maximum is None else max(maximum, value)
    result["max_abs_finite_unscaled_grad"] = maximum
    return result


def checked_optimizer_step(trainer, report):
    """Exact pinned BaseTrainer step with detached per-attempt AMP diagnostics."""
    import torch

    report["optimizer_attempts"] += 1
    scale_before = float(trainer.scaler.get_scale())
    trainer.scaler.unscale_(trainer.optimizer)
    gradients = gradient_diagnostics(trainer.model.named_parameters())
    report.update(gradients)
    if gradients["parameters_with_grad"] == 0:
        raise AssertionError("No gradients after GradScaler unscale")
    loss_items = None if trainer.loss_items is None else trainer.loss_items.detach().float().cpu().tolist()
    loss_items_finite = (isinstance(loss_items, list) and len(loss_items) == 4
                         and all(torch.isfinite(torch.tensor(value)).item() for value in loss_items))
    report["box_seg_cls_dfl_finite"] = report["box_seg_cls_dfl_finite"] and loss_items_finite
    report["actual_training_batch"] = True
    attempt = {"attempt": report["optimizer_attempts"], "scaler_scale_before": scale_before,
               "scaler_scale_after": None, "loss_items": loss_items,
               "loss_items_finite": loss_items_finite, **gradients,
               "actual_optimizer_update": False}
    # A post-hook runs only when the underlying optimizer.step() is actually
    # called; GradScaler's skipped step never invokes it. It observes without
    # changing the optimizer, gradients, or the pinned step order.
    actual_calls = 0

    def on_optimizer_step(_optimizer, _args, _kwargs):
        nonlocal actual_calls
        actual_calls += 1

    hook = trainer.optimizer.register_step_post_hook(on_optimizer_step)
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=10.0)
    try:
        trainer.scaler.step(trainer.optimizer)
    finally:
        hook.remove()
    trainer.scaler.update()
    trainer.optimizer.zero_grad()
    if trainer.ema:
        trainer.ema.update(trainer.model)
    attempt["scaler_scale_after"] = float(trainer.scaler.get_scale())
    attempt["actual_optimizer_update"] = actual_calls > 0
    report["attempt_diagnostics"].append(attempt)
    report["actual_optimizer_updates"] += actual_calls
    report["amp_skipped_updates"] += int(actual_calls == 0)
    report["optimizer_steps"] = report["actual_optimizer_updates"]
    for key in ("parameters_with_grad", "finite_gradient_tensors", "nonfinite_gradient_tensors",
                "max_abs_finite_unscaled_grad"):
        report[key] = attempt[key]
    report["scaler_scale_before"] = scale_before
    report["scaler_scale_after"] = attempt["scaler_scale_after"]
    report["loss_items_finite"] = loss_items_finite
    report["loss_items"] = loss_items


@contextmanager
def route_dataset_cache(output, baseline_pilot, generated_paths, dataset_module=None):
    """Redirect YOLODataset cache load/save calls to one smoke output directory."""
    if dataset_module is None:
        from ultralytics.data import dataset as dataset_module
    old_load = dataset_module.load_dataset_cache_file
    old_save = dataset_module.save_dataset_cache_file
    cache_root = Path(output).resolve() / "dataset_cache"
    pilot_data = Path(baseline_pilot).resolve() / "data"

    def routed(path):
        source = Path(path).resolve()
        try:
            relative = source.relative_to(pilot_data)
            return cache_root / "pilot_data" / relative
        except ValueError:
            digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
            return cache_root / "external" / digest / source.name

    def load(path):
        target = routed(path)
        return old_load(target if target.is_file() else path)

    def save(prefix, path, contents, version):
        target = routed(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        old_save(prefix, target, contents, version)
        if target.is_file():
            generated_paths.append(str(target))

    dataset_module.load_dataset_cache_file = load
    dataset_module.save_dataset_cache_file = save
    try:
        yield
    finally:
        dataset_module.load_dataset_cache_file = old_load
        dataset_module.save_dataset_cache_file = old_save
