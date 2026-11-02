"""Fast checks that run on every push (CI) and before any change to the decision code.

    python -m pytest -q            # fast tests (vegetable model only, no download)
    FULL_TESTS=1 python -m pytest  # also the full answer with CLIP (downloads about 600 MB once)
"""

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
GOLDEN = json.load(open(REPO / "tests/golden.json"))

from model_utils import (  # noqa: E402  (tests/conftest.py puts src on the path)
    CLASS_NAMES, clip_head_logits, clip_unknown_score, combined_logits, combined_probabilities, is_unknown_combined,
    new_vegetable_decision, read_image, make_transform, input_size, load_model, calibrated_probabilities,
)
from train import read_manifest  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint():
    return torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)


# ---------- data ----------
def test_manifest_splits_are_clean():
    """read_manifest checks class ids, files on disk, and that no photo, group or SHA-256 sits in two splits."""
    splits = read_manifest(REPO / "dataset/manifest.csv", CLASS_NAMES[:10])
    assert {k: len(v) for k, v in splits.items() if v} == {"train": 1260, "val": 300, "test": 300}


def test_manifest_hashes_match_files():
    rows = list(csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8")))
    for r in rows[::97]:  # a spread-out sample, so the test stays fast
        assert hashlib.sha256((REPO / "dataset" / r["image_path"]).read_bytes()).hexdigest() == r["sha256"], r["image_path"]


# ---------- the adopted checkpoint ----------
def test_checkpoint_is_tv12(checkpoint):
    assert checkpoint["version"] == "TV-12"
    assert checkpoint["class_names"] == CLASS_NAMES[:10]
    head = checkpoint["clip_head"]
    assert head["other_class"] and "text_evidence" in head and "alpha" not in head
    assert checkpoint["new_vegetable_head"]["switch_probability"] == 0.8


# ---------- decision functions on made-up CLIP features ----------
def _features(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, 512, generator=g), dim=-1)


def test_clip_head_without_text_mix_is_linear(checkpoint):
    f = _features(); h = checkpoint["clip_head"]
    assert torch.allclose(clip_head_logits(f, checkpoint), f @ h["coef"].T + h["intercept"])


def test_text_mix_changes_only_class_columns(checkpoint):
    f = _features(); mixed = {**checkpoint, "clip_head": {**checkpoint["clip_head"], "alpha": 0.5,
                                                         "text_embeddings": checkpoint["new_vegetable_head"]["text_embeddings"][:10]}}
    a, b = clip_head_logits(f, checkpoint), clip_head_logits(f, mixed)
    assert torch.equal(a[:, 10], b[:, 10]) and not torch.allclose(a[:, :10], b[:, :10])


def test_full_decision_shapes(checkpoint):
    f = _features(); logits = torch.randn(len(f), 10)
    clip_logits = clip_head_logits(f, checkpoint)
    p = combined_probabilities(combined_logits(logits, clip_logits, checkpoint), checkpoint)
    assert p.shape == (len(f), 10) and torch.allclose(p.sum(1), torch.ones(len(f)), atol=1e-5)
    unknown = is_unknown_combined(logits, clip_logits, checkpoint, f)
    assert unknown.dtype == torch.bool and unknown.shape == (len(f),)
    assert clip_unknown_score(clip_logits, checkpoint, f).shape == (len(f),)
    out, index, still_unknown, switched = new_vegetable_decision(p, unknown, f, checkpoint)
    assert out.shape == (len(f), 16) and index.shape == still_unknown.shape == switched.shape == (len(f),)
    assert not (still_unknown & switched).any()


# ---------- the same answers as when the model was adopted ----------
def test_vegetable_model_golden_answers(checkpoint):
    model, ck = load_model(REPO / "model.pth", torch.device("cpu")); tf = make_transform(input_size(ck))
    with torch.inference_mode():
        x = torch.stack([read_image(REPO / "dataset" / p, tf) for p in GOLDEN["photos"]])
        p = calibrated_probabilities(model(x), ck)
    for path, row in zip(GOLDEN["photos"], p):
        want = GOLDEN["no_clip"][Path(path).name]
        assert CLASS_NAMES[int(row.argmax())] == want["predicted_class"], path
        assert abs(float(row.max()) - want["probability"]) < 2e-3, path


def _predict(extra: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "in"; folder.mkdir()
        for p in GOLDEN["photos"]:
            (folder / Path(p).name).symlink_to(REPO / "dataset" / p)
        out = Path(tmp) / "out.csv"
        subprocess.run([sys.executable, str(REPO / "src/predict.py"), "--input", str(folder), "--model", str(REPO / "model.pth"),
                        "--output", str(out), "--threads", "2", *extra], check=True, capture_output=True)
        return {Path(r["image"]).name: r for r in csv.DictReader(open(out, encoding="utf-8"))}


def test_predict_cli_without_clip():
    rows = _predict(["--no-clip"])
    for name, want in GOLDEN["no_clip"].items():
        assert rows[name]["predicted_class"] == want["predicted_class"] and rows[name]["error"] == ""


@pytest.mark.skipif(not os.environ.get("FULL_TESTS"), reason="needs open_clip and the CLIP weights (about 600 MB); set FULL_TESTS=1")
def test_predict_cli_full_system():
    rows = _predict([])
    for name, want in GOLDEN["full"].items():
        assert rows[name]["predicted_class"] == want["predicted_class"], name
        assert abs(float(rows[name]["probability"]) - want["probability"]) < 2e-3, name
