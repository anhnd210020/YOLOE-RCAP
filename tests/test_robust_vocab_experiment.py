"""CPU-only checks for the A1/A3 experiment package; never launches training."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robust_vocab.scripts import run_experiment as run


class ExperimentPackageTests(unittest.TestCase):
    def test_arm_tau_rules_and_identity(self):
        self.assertEqual(run.tau_for("A1", None), (0.0, "0.0"))
        for supplied in ("0", "0.0", "0.1"):
            with self.assertRaises(ValueError):
                run.tau_for("A1", supplied)
        with self.assertRaises(ValueError):
            run.tau_for("A3", None)
        for supplied in ("0", "-0.1", "nan", "inf", "1e999", "bad"):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                run.tau_for("A3", supplied)
        self.assertEqual(run.tau_for("A3", "0.300"), (0.3, "0.3"))
        self.assertNotEqual(run.run_name("A1", "0.0", 0), run.run_name("A3", "0.3", 0))

    def test_manifest_dry_run_and_output_collision(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, "RUNS", Path(tmp)):
            stream = StringIO()
            with redirect_stdout(stream):
                self.assertEqual(run.main(["A1", "--dry-run"]), 0)
            record = json.loads(stream.getvalue())
            self.assertEqual(record["arm"], "A1")
            self.assertEqual(record["robust_vocab_tau"], 0.0)
            self.assertEqual(record["physical_batch"], 32)
            self.assertEqual(record["initial_accumulation"], 4)
            self.assertEqual(record["exit_status"], "not_started")
            for key in ("initialization_fingerprint", "start_timestamp", "end_timestamp",
                        "last_pt_sha256", "peak_vram_bytes", "training_duration_seconds", "final_metrics"):
                self.assertIsNone(record[key])
            self.assertFalse(list(Path(tmp).iterdir()))
            (Path(tmp) / record["experiment_id"]).mkdir()
            with self.assertRaises(FileExistsError):
                run.main(["A1", "--dry-run"])

    def test_only_tau_differs_in_training_settings(self):
        _, lock = run.read_lock()
        _, a1, _ = run.settings(0.0, 32, lock)
        _, a3, _ = run.settings(0.3, 32, lock)
        self.assertEqual({k: v for k, v in a1.items() if k != "robust_vocab_tau"},
                         {k: v for k, v in a3.items() if k != "robust_vocab_tau"})
        self.assertEqual((a1["robust_vocab_tau"], a3["robust_vocab_tau"]), (0.0, 0.3))


if __name__ == "__main__":
    unittest.main()
