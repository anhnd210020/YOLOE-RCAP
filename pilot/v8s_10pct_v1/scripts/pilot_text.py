"""Runtime-only pilot artifact routing for the official RandomLoadText sampler."""
import json
from pathlib import Path

from make_subset import inside_pilot, sha256_file


def verify_text_artifacts(entry, checkpoint_path=None):
    if entry.get("text_model") != "mobileclip:blt":
        raise ValueError("Pilot text model must be mobileclip:blt")
    if entry.get("global_negative_threshold") != 10:
        raise ValueError("Pilot global negative threshold must be 10")
    pairs = (
        ("mobileclip_checkpoint", "mobileclip_checkpoint_sha256", False),
        ("train_label_embeddings", "train_label_embeddings_sha256", True),
        ("global_negative_categories", "global_negative_categories_sha256", True),
        ("global_negative_embeddings", "global_negative_embeddings_sha256", True),
    )
    for path_key, hash_key, pilot_only in pairs:
        raw = entry.get(path_key)
        if not raw:
            raise ValueError(f"Text lock missing {path_key}")
        path = Path(raw)
        if not path.is_absolute():
            raise ValueError(f"Text lock path must be absolute: {path_key}")
        if pilot_only:
            inside_pilot(path)
        if not path.is_file():
            raise FileNotFoundError(f"Required pilot text artifact or MobileCLIP weights missing: {path}")
        if sha256_file(path) != entry.get(hash_key):
            raise ValueError(f"Text artifact SHA256 mismatch: {path}")
    if Path(entry["mobileclip_checkpoint"]).name != "mobileclip_blt.pt":
        raise ValueError("Official MobileCLIP expects checkpoint basename mobileclip_blt.pt")
    if checkpoint_path and Path(checkpoint_path).resolve() != Path(entry["mobileclip_checkpoint"]).resolve():
        raise ValueError("MobileCLIP checkpoint path differs from locked path")
    cats = json.loads(Path(entry["global_negative_categories"]).read_text(encoding="utf-8"))
    if not isinstance(cats, list) or len(cats) != entry.get("global_negative_count") or len(cats) < 80:
        raise ValueError("Pilot global negative categories must match lock and contain at least 80 names")
    return entry


try:
    from ultralytics.data.augment import RandomLoadText
except ImportError:
    RandomLoadText = object


class PilotRandomLoadText(RandomLoadText):
    """Keep the official __call__ sampling behavior; change only constructor paths."""

    artifacts = None

    def __init__(self, text_model, prompt_format="{}", neg_samples=(80, 80),
                 max_samples=80, padding=False, padding_value=""):
        if self.artifacts is None or text_model != self.artifacts["text_model"]:
            raise ValueError("Pilot text artifacts were not installed for this text model")
        import numpy as np
        import torch
        self.prompt_format = prompt_format
        self.neg_samples = neg_samples
        self.max_samples = max_samples
        self.padding = padding
        self.padding_value = padding_value
        with open(self.artifacts["global_negative_categories"], encoding="utf-8") as stream:
            self.global_grounding_neg_cats = np.array(json.load(stream))
        self.global_grounding_neg_embeddings = torch.load(self.artifacts["global_negative_embeddings"])
        self.train_label_embeddings = torch.load(self.artifacts["train_label_embeddings"])


def install_pilot_text(entry):
    """Route official dataset transform construction to the pilot subclass in this process."""
    verify_text_artifacts(entry)
    from ultralytics.data import dataset as dataset_module
    PilotRandomLoadText.artifacts = entry
    dataset_module.RandomLoadText = PilotRandomLoadText
