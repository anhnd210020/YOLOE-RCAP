"""CPU regression coverage for pilot-only validation bypass and real checkpoint I/O."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.torch_utils import EarlyStopping
from pilot_trainer import PilotSmokeTrainer, PilotTrainOnlyTrainer, PilotYOLOESegTrainerFromScratch

ROOT = Path(__file__).resolve().parents[3]
PINNED = "40cd606cabdbe2b566d6f14a6b162c89206e9a1b"


class TrainOnlyTests(unittest.TestCase):
    def test_main_bypasses_validation_loader_and_validator(self):
        trainer = object.__new__(PilotTrainOnlyTrainer)
        trainer.metrics = {"metrics/mAP50-95(M)": 0.0}
        trainer.validator = Mock(side_effect=AssertionError("mask validator called"))
        with patch.object(PilotYOLOESegTrainerFromScratch, "get_dataloader", return_value="train loader") as loader:
            self.assertIsNone(trainer.get_dataloader("lvis", mode="val"))
            loader.assert_not_called()
            self.assertEqual(trainer.get_dataloader("locked train", 32, 0, "train"), "train loader")
            loader.assert_called_once_with("locked train", batch_size=32, rank=0, mode="train")
        stopper = EarlyStopping(patience=100)
        for epoch in range(30):
            trainer.epoch = epoch
            metrics, fitness = trainer.validate()
            self.assertIs(metrics, trainer.metrics)
            self.assertEqual((fitness, trainer.best_fitness), (0.0, 0.0))
            self.assertFalse(stopper(epoch + 1, fitness))
        trainer.validator.assert_not_called()

    def test_main_inherits_normal_checkpoint_serialization(self):
        self.assertIs(PilotTrainOnlyTrainer.save_model, BaseTrainer.save_model)
        self.assertIs(PilotTrainOnlyTrainer.run_callbacks, BaseTrainer.run_callbacks)
        with tempfile.TemporaryDirectory(prefix="pilot_checkpoint_test_") as directory:
            trainer = object.__new__(PilotTrainOnlyTrainer)
            trainer.wdir = Path(directory)
            trainer.last, trainer.best = trainer.wdir / "last.pt", trainer.wdir / "best.pt"
            network = torch.nn.Linear(2, 1)
            trainer.ema = SimpleNamespace(ema=network, updates=1)
            trainer.optimizer = torch.optim.AdamW(network.parameters(), lr=0.002)
            network(torch.ones(1, 2)).sum().backward()
            trainer.optimizer.step()
            trainer.args = SimpleNamespace(close_mosaic=2, epochs=30, batch=32, nbs=128)
            trainer.epochs, trainer.save_period = 30, -1
            trainer.metrics = {}
            trainer.read_results_csv = lambda: {"epoch": [trainer.epoch + 1]}
            trainer.validator = Mock(side_effect=AssertionError("mask validator called"))
            for epoch in (0, 29):
                trainer.epoch = epoch
                trainer.metrics, trainer.fitness = trainer.validate()
                trainer.save_model()
                self.assertGreater(trainer.last.stat().st_size, 0)
                saved = torch.load(trainer.last, map_location="cpu", weights_only=False)
                self.assertEqual(saved["epoch"], epoch)
                self.assertEqual(saved["train_results"]["epoch"], [epoch + 1])
                self.assertEqual(saved["updates"], 1)
                self.assertTrue(saved["optimizer"]["state"])
                self.assertEqual(saved["ema"].weight.dtype, torch.float16)
                self.assertTrue(torch.equal(saved["ema"].weight, network.weight.half()))
                before = trainer.last.read_bytes()
                trainer.final_eval()
                self.assertEqual(trainer.last.read_bytes(), before)
            trainer.validator.assert_not_called()

    def test_final_eval_cannot_delegate_to_official_evaluator(self):
        trainer = object.__new__(PilotTrainOnlyTrainer)
        trainer.validator = Mock(side_effect=AssertionError("mask validator called"))
        with patch.object(PilotYOLOESegTrainerFromScratch, "final_eval", side_effect=AssertionError("official final_eval called")):
            self.assertIsNone(trainer.final_eval())
        trainer.validator.assert_not_called()

    def test_smoke_behavior_unchanged(self):
        trainer = object.__new__(PilotSmokeTrainer)
        trainer.loss = torch.tensor(2.0)
        trainer.metrics = {}
        trainer.validator = Mock(side_effect=AssertionError("mask validator called"))
        self.assertIsNone(trainer.get_dataloader("lvis", mode="val"))
        self.assertEqual(trainer.validate(), ({}, -2.0))
        self.assertEqual(trainer.best_fitness, -2.0)
        with patch.object(BaseTrainer, "save_model", side_effect=AssertionError("smoke saved checkpoint")):
            self.assertIsNone(trainer.save_model())
        self.assertIsNone(trainer.final_eval())
        with patch.object(BaseTrainer, "run_callbacks", return_value="called") as callback:
            self.assertIsNone(trainer.run_callbacks("on_model_save"))
            callback.assert_not_called()
            self.assertEqual(trainer.run_callbacks("on_train_end"), "called")
        trainer.validator.assert_not_called()

    def test_official_source_integrity_nine_hashes_and_zero_git_diff(self):
        manifest = ROOT / "pilot/v8s_10pct_v1/manifests/OFFICIAL_SOURCE_SHA256.txt"
        entries = [line.split() for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        self.assertEqual(len(entries), 9)
        for expected, name in entries:
            with self.subTest(path=name):
                data = (ROOT / name.replace("\\", "/")).read_bytes()
                canonical_crlf = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
                self.assertEqual(hashlib.sha256(canonical_crlf).hexdigest(), expected)
        result = subprocess.run(
            ["git", "diff", "--exit-code", PINNED, "--", "ultralytics", "train_seg.py", "tools"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
