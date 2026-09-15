"""CPU-only regression checks for final laptop preparation; no model is loaded."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import eval_pilot
import verify_gqa_official_images
import make_subset
import pilot_grounding
import pilot_text
import smoke_batch
import train_pilot
import verify_subset

PILOT = Path(__file__).resolve().parents[1]
FIXED_AP_OUTPUT = """\
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.197
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= -1 catIds=all] = 0.277
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= -1 catIds=all] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.125
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.270
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.379
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  r] = 0.133
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  c] = 0.188
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  f] = 0.216
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.184
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.569
copypaste: AP,AP50,AP75,APs,APm,APl,APr,APc,APf
copypaste: 19.66,27.70,21.09,12.53,26.98,37.86,13.26,18.81,21.57
"""


class PilotPreparationTests(unittest.TestCase):
    def test_known_gqa_dimensions_warn_only_for_exact_official_record(self):
        classify = verify_gqa_official_images.classify_dimensions
        image = {"id": 770291, "file_name": "61564.jpg", "width": 1024, "height": 768}
        warning = classify(image, (1024, 1022))
        self.assertEqual(warning["status"], "WARN_UPSTREAM_OFFICIAL")
        self.assertEqual((warning["json_height"], warning["jpeg_height"]), (768, 1022))
        with self.assertRaises(ValueError):
            classify({**image, "height": 1022}, (1024, 1022))
        with self.assertRaises(ValueError):
            classify(image, (1024, 1021))
        with self.assertRaises(ValueError):
            classify({**image, "id": 12345}, (1024, 1022))

    def test_subset_and_manifest_with_synthetic_images(self):
        with tempfile.TemporaryDirectory(prefix="pilot_test_", dir=PILOT) as directory:
            root = Path(directory)
            source = root / "source.json"
            images = [{"id": index, "file_name": f"{index}.jpg"} for index in range(20)]
            anns = [{"id": index, "image_id": index, "category_id": 1} for index in range(20)]
            source.write_text(json.dumps({"images": images, "annotations": anns, "categories": []}), encoding="utf-8")
            subset = root / "gqa_pilot10_v1.json"
            manifest = root / "manifest.json"

            def json_items(path, prefix):
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                value = data.get(prefix.removesuffix(".item"), [])
                return iter(value if prefix.endswith(".item") else [value])

            args = SimpleNamespace(input_json=str(source), output_json=str(subset), manifest_out=str(manifest),
                                   source_name="GQA", fraction=0.1, seed=0, pilot_id="yoloe-v8s-10pct-v1", dry_run=False)
            with patch.object(make_subset, "items", json_items), redirect_stdout(io.StringIO()):
                report = make_subset.build(args)
            self.assertEqual(report["selected_image_count"], 2)
            with patch.object(verify_subset, "items", json_items):
                checked = verify_subset.verify(SimpleNamespace(subset_json=str(subset), manifest=str(manifest),
                                                               source_json=str(source), image_root=None, flat_images=False,
                                                               cache=None, lock_out=None))
            self.assertEqual((checked["images"], checked["annotations"]), (2, 2))
            tampered = json.loads(subset.read_text(encoding="utf-8"))
            tampered["images"][0]["file_name"] = "changed.jpg"
            subset.write_text(json.dumps(tampered), encoding="utf-8")
            with patch.object(verify_subset, "items", json_items), self.assertRaises(ValueError):
                verify_subset.verify(SimpleNamespace(subset_json=str(subset), manifest=str(manifest),
                                                     source_json=str(source), image_root=None, flat_images=False,
                                                     cache=None, lock_out=None))

    def test_cache_and_lock_reject_wrong_counts(self):
        self.assertEqual(pilot_grounding.load_then_reduce_for_smoke(lambda: list(range(10)),
                         SimpleNamespace(im_files=list(range(10))), 0.1), [0])
        with self.assertRaises(ValueError):
            pilot_grounding.verify_cache_labels([], 1, 0, False)
        with self.assertRaises(ValueError):
            train_pilot.require_lock_sections({"version": "v1", "pilot_id": "yoloe-v8s-10pct-v1"})

    def test_grounding_cache_shape_and_text_artifact_lock(self):
        import numpy as np
        with tempfile.TemporaryDirectory(prefix="pilot_test_", dir=PILOT) as directory:
            root = Path(directory)
            image = root / "sample.jpg"
            image.write_bytes(b"synthetic image presence only")
            label = {"im_file": str(image), "shape": (2, 2), "cls": np.zeros((1, 1)),
                     "bboxes": np.zeros((1, 4)), "segments": [np.zeros((3, 2))],
                     "normalized": True, "bbox_format": "xywh", "texts": [["sample"]]}
            self.assertEqual(pilot_grounding.verify_cache_labels([label], 1, 1, True, root, {str(image)}),
                             {"images": 1, "instances": 1})
            # Repeated JPEGs are distinct grounding records; preserve multiplicity.
            self.assertEqual(pilot_grounding.verify_cache_labels(
                [label, label], 2, 2, True, root,
                expected_file_counts={str(image): 2}), {"images": 2, "instances": 2})
            other = root / "other.jpg"
            other.write_bytes(b"synthetic image presence only")
            other_label = {**label, "im_file": str(other)}
            with self.assertRaises(ValueError):
                # Same total record count and file set, wrong per-file counts.
                pilot_grounding.verify_cache_labels(
                    [label, label, other_label], 3, 3, True, root,
                    expected_file_counts={str(image): 1, str(other): 2})
            label["bboxes"] = np.zeros((2, 4))
            with self.assertRaises(ValueError):
                pilot_grounding.verify_cache_labels([label], 1, 1, True)
            paths = {"mobileclip_checkpoint": root / "mobileclip_blt.pt",
                     "train_label_embeddings": root / "train.pt",
                     "global_negative_categories": root / "cats.json",
                     "global_negative_embeddings": root / "negative.pt"}
            for key, path in paths.items():
                path.write_text(json.dumps([f"cat_{i}" for i in range(80)]) if key == "global_negative_categories"
                                else "synthetic", encoding="utf-8")
            entry = {"text_model": "mobileclip:blt", "global_negative_threshold": 10,
                     "global_negative_count": 80}
            for key, path in paths.items():
                entry[key] = str(path)
                entry[key + "_sha256"] = make_subset.sha256_file(path)
            self.assertIs(pilot_text.verify_text_artifacts(entry), entry)
            entry["global_negative_categories_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                pilot_text.verify_text_artifacts(entry)

    def test_smoke_final_epoch_bypass_is_pilot_only(self):
        trainer_source = (eval_pilot.ROOT / "ultralytics/engine/trainer.py").read_text(encoding="utf-8")
        self.assertIn("or final_epoch or self.stopper.possible_stop", trainer_source)
        self.assertIn("if self.args.save or final_epoch:", trainer_source)
        self.assertIn("self.final_eval()", trainer_source)
        tree = ast.parse((PILOT / "scripts/pilot_trainer.py").read_text(encoding="utf-8"))
        smoke = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PilotSmokeTrainer")
        self.assertEqual({node.name for node in smoke.body if isinstance(node, ast.FunctionDef)},
                         {"get_dataloader", "validate", "save_model", "final_eval", "run_callbacks"})
        train_source = (PILOT / "scripts/train_pilot.py").read_text(encoding="utf-8")
        self.assertIn("run_smoke_trainer(YOLOE(model_path), kwargs)", train_source)
        self.assertIn("YOLOE(model_path).train(**kwargs)", train_source)

    def test_cuda_oom_exit_and_retry(self):
        self.assertTrue(train_pilot.is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate")))
        self.assertFalse(train_pilot.is_cuda_oom(RuntimeError("dataset missing")))
        self.assertFalse(train_pilot.is_cuda_oom(ValueError("CUDA out of memory")))
        self.assertTrue(smoke_batch.should_retry(75))
        self.assertFalse(smoke_batch.should_retry(1))
        for codes, expected in [([1], 1), ([75, 1], 2), ([75, 0], 2), ([75, 75, 0], 3)]:
            calls = []

            def fake_run(command, check):
                calls.append(command)
                return SimpleNamespace(returncode=codes[len(calls) - 1])

            with patch.object(sys, "argv", ["smoke_batch.py", "--execute", "--"]), \
                    patch.object(smoke_batch.subprocess, "run", fake_run), redirect_stdout(io.StringIO()):
                if codes[-1] == 1:
                    with self.assertRaises(SystemExit):
                        smoke_batch.main()
                else:
                    smoke_batch.main()
            self.assertEqual(len(calls), expected)

    def test_eval_preflight_and_hashes_without_model(self):
        with tempfile.TemporaryDirectory(prefix="pilot_test_", dir=PILOT) as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"synthetic checkpoint: never loaded")
            lock = {"version": "v1", "pilot_id": "yoloe-v8s-10pct-v1", "text_artifacts": {},
                    "evaluation": {"name": "LVIS v1 full minival", "expected_image_count": 4809,
                                   "minival_list": str(eval_pilot.LVIS / "minival.txt"),
                                   "minival_list_sha256": eval_pilot.verify_evaluation.__globals__["LVIS_LIST_SHA256"],
                                   "minival_annotation": str(eval_pilot.LVIS / "annotations/lvis_v1_minival.json"),
                                   "minival_annotation_sha256": eval_pilot.verify_evaluation.__globals__["LVIS_JSON_SHA256"]}}
            args = SimpleNamespace(checkpoint=str(checkpoint), lock=str(PILOT / "PILOT_DATASET_LOCK.json"),
                                   name="pass3_cpu_test")
            with patch.object(eval_pilot, "read_lock", return_value=lock), \
                    patch.object(eval_pilot, "verify_text_artifacts", return_value={}):
                result = eval_pilot.preflight(args)
                self.assertEqual((result[1], len(result[4]), len(result[5])),
                                 (make_subset.sha256_file(checkpoint), 4809, 4809))
                lock["evaluation"]["minival_annotation_sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    eval_pilot.preflight(args)
            args.checkpoint = str(Path(directory) / "missing.pt")
            with self.assertRaises(FileNotFoundError):
                eval_pilot.preflight(args)

    def test_fixed_ap_uses_module_invocation(self):
        annotation = Path("/tmp/annotations.json")
        prediction = Path("/tmp/predictions.json")
        command = eval_pilot.fixed_ap_command(annotation, prediction)
        self.assertEqual(command, [sys.executable, "-m", "tools.eval_fixed_ap", str(annotation),
                                   str(prediction), "--type", "bbox"])
        self.assertNotIn(str(eval_pilot.FIXED_AP), command)

    def test_fixed_ap_parser_extracts_complete_actual_summary(self):
        metrics = eval_pilot.parse_fixed_ap(FIXED_AP_OUTPUT)
        self.assertEqual(metrics, {
            "AP": 19.66, "AP50": 27.7, "AP75": 21.09, "APs": 12.53,
            "APm": 26.98, "APl": 37.86, "APr": 13.26, "APc": 18.81,
            "APf": 21.57, "AR": 34.2, "ARs": 18.4, "ARm": 41.3, "ARl": 56.9,
        })
        self.assertTrue(set(eval_pilot.PRIMARY_AP_METRICS) <= metrics.keys())

    def test_fixed_ap_parser_rejects_malformed_missing_nonfinite_and_out_of_range(self):
        invalid_outputs = {
            "malformed": FIXED_AP_OUTPUT.replace(
                "19.66,27.70,21.09,12.53,26.98,37.86,13.26,18.81,21.57",
                "19.66,27.70"),
            "missing": FIXED_AP_OUTPUT.replace(
                "copypaste: AP,AP50,AP75,APs,APm,APl,APr,APc,APf",
                "copypaste: AP,AP50,AP75,APs,APm,APl,APr,APc"),
            "nonfinite_ap": FIXED_AP_OUTPUT.replace("19.66,27.70", "nan,27.70"),
            "nonfinite_ar": FIXED_AP_OUTPUT.replace("= 0.342", "= inf"),
            "out_of_range": FIXED_AP_OUTPUT.replace("19.66,27.70", "100.01,27.70"),
        }
        for name, output in invalid_outputs.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                eval_pilot.parse_fixed_ap(output)

    def test_validated_bbox_class_is_reused(self):
        tree = ast.parse(eval_pilot.BASELINE_VALIDATOR.read_text(encoding="utf-8"))
        bbox = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "BBoxFixedAPValidator")
        aliases = {node.targets[0].id for node in bbox.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
        self.assertTrue({"pred_to_json", "_prepare_pred", "get_stats", "eval_json"} <= aliases)
        wrapper = (PILOT / "scripts/eval_pilot.py").read_text(encoding="utf-8")
        self.assertIn("return module.BBoxFixedAPValidator", wrapper)
        self.assertIn("model.val(validator=validator_class, **config)", wrapper)

    def test_gitignore_keeps_lock_trackable(self):
        lines = (PILOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertTrue({"data/", "runs/", "temp/", "__pycache__/", "*.pyc", "*_review*.zip"} <= set(lines))
        self.assertFalse(any("PILOT_DATASET_LOCK.json" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
