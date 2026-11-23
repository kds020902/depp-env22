"""train.read_manifest (also used by calibrate.py and train_clip_head.py) and the training command's defaults."""

import csv
import shutil
import subprocess
import sys

import pytest

from conftest import REPO

from labels import CLASS_NAMES  # noqa: E402
from train import parse_args, read_manifest  # noqa: E402


@pytest.fixture()
def small_dataset(tmp_path):
    """One photo of every class in every split, copied from the repository's dataset, with its manifest rows."""
    rows = list(csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8")))
    picked, seen = [], set()
    for r in rows:
        if (r["split"], r["class_name"]) not in seen:
            seen.add((r["split"], r["class_name"]))
            picked.append(r)
            (tmp_path / r["image_path"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / "dataset" / r["image_path"], tmp_path / r["image_path"])
    with open(tmp_path / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(picked)
    return tmp_path, picked


def test_a_correct_manifest_is_read(small_dataset):
    root, picked = small_dataset
    splits = read_manifest(root / "manifest.csv", CLASS_NAMES[:10])
    assert {s: len(v) for s, v in splits.items()} == {"train": 10, "val": 10, "test": 10}
    assert {s: len(v) for s, v in read_manifest(REPO / "dataset/manifest.csv", CLASS_NAMES[:10]).items()} == {"train": 1260, "val": 300, "test": 300}


def test_a_train_photo_copied_over_a_val_photo_is_refused(small_dataset):
    """Codex's case: the file under a Val path is a Train photo while the CSV still lists the old hash."""
    root, picked = small_dataset
    train = next(r for r in picked if r["split"] == "train")
    val = next(r for r in picked if r["split"] == "val" and r["class_name"] == train["class_name"])
    shutil.copy(root / train["image_path"], root / val["image_path"])
    with pytest.raises(ValueError) as error:
        read_manifest(root / "manifest.csv", CLASS_NAMES[:10])
    message = str(error.value)
    assert "SHA-256" in message and f"val: {root / val['image_path']}" in message and "train:" not in message


def test_group_and_duplicate_checks_still_hold(small_dataset):
    root, picked = small_dataset
    rows = [dict(r) for r in picked]
    train = next(r for r in rows if r["split"] == "train")
    val = next(r for r in rows if r["split"] == "val")
    val["group_id"] = train["group_id"]                                   # one near-duplicate group in two splits
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="leakage in group_id"):
        read_manifest(root / "manifest.csv", CLASS_NAMES[:10])


def test_calibrate_and_clip_head_commands_stop_on_a_changed_photo(small_dataset):
    """Both commands read the manifest through read_manifest before they fit anything or load CLIP."""
    root, picked = small_dataset
    val = next(r for r in picked if r["split"] == "val")
    other = next(r for r in picked if r["split"] == "test")
    shutil.copy(root / other["image_path"], root / val["image_path"])
    model = root / "candidate.pth"
    shutil.copy(REPO / "model.pth", model)
    before = model.read_bytes()
    (root / "extra").mkdir()
    for script, more in (("src/calibrate.py", []), ("src/train_clip_head.py", ["--extra-photos", str(root / "extra")])):
        done = subprocess.run([sys.executable, str(REPO / script), "--manifest", str(root / "manifest.csv"), "--model", str(model), *more],
                              capture_output=True, text=True, cwd=REPO)
        assert done.returncode != 0 and "do not match the SHA-256" in done.stderr, script + done.stderr[-400:]
    assert model.read_bytes() == before


def test_training_command_defaults_to_the_ten_classes_of_the_repository_manifest(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--manifest", "dataset/manifest.csv", "--out", "x"])
    assert parse_args().num_classes == 10
    for n in (6, 10, 16):
        monkeypatch.setattr(sys, "argv", ["train.py", "--manifest", "m.csv", "--out", "x", "--num-classes", str(n)])
        assert parse_args().num_classes == n
    monkeypatch.setattr(sys, "argv", ["train.py", "--manifest", "m.csv", "--out", "x", "--num-classes", "12"])
    with pytest.raises(SystemExit):
        parse_args()
    assert set(read_manifest(REPO / "dataset/manifest.csv", CLASS_NAMES[:6])) == {"train", "val", "test"}


def test_tracked_files_are_checked_out_byte_for_byte():
    """.gitattributes: no line-ending conversion for any file, so the recorded SHA-256 values hold under core.autocrlf=true."""
    lines = [line.strip() for line in (REPO / ".gitattributes").read_text(encoding="utf-8").splitlines()]
    assert "* -text" in lines
    done = subprocess.run(["git", "check-attr", "text", "--", "requirements.txt", "results/t3_test/scores_once.json",
                           "dataset/manifest.csv", "results/t9_test/t9_main.csv"], cwd=REPO, capture_output=True, text=True)
    if done.returncode == 0 and done.stdout:
        assert all(line.endswith("text: unset") for line in done.stdout.splitlines()), done.stdout
