"""tools/compare_dataset_builds.py: the cases a before/after check of dataset/tools must not miss."""

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools/compare_dataset_builds.py"
COLUMNS = ["image_id", "image_path", "class_name", "split", "source_dataset", "source_archive_path", "sha256"]


def build(folder: Path, photos: dict[str, bytes | None], extra_column: bool = False) -> Path:
    """A tiny build folder: one manifest row per photo; content None = listed in the manifest but not written."""
    rows = []
    for k, (path, content) in enumerate(photos.items()):
        if content is not None:
            (folder / path).parent.mkdir(parents=True, exist_ok=True)
            (folder / path).write_bytes(content)
        row = dict(image_id=f"p{k}", image_path=path, class_name="potato_fresh", split=path.split("/")[0].replace("valid", "val"),
                   source_dataset="s", source_archive_path=f"src/{k}.jpg", sha256=hashlib.sha256(content or b"gone").hexdigest())
        rows.append({**row, "extra": "x"} if extra_column else row)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS + (["extra"] if extra_column else []))
        writer.writeheader()
        writer.writerows(rows)
    return folder


def run(a: Path, b: Path, *args: str) -> int:
    return subprocess.run([sys.executable, str(TOOL), str(a), str(b), *args], capture_output=True).returncode


PHOTOS = {"train/P_fresh/a.jpg": b"one", "test/P_fresh/b.jpg": b"two"}


def test_same_builds_pass(tmp_path):
    assert run(build(tmp_path / "a", PHOTOS), build(tmp_path / "b", PHOTOS)) == 0
    assert run(build(tmp_path / "c", PHOTOS), build(tmp_path / "d", PHOTOS), "--rows", "1") == 0


def test_different_photo_content_fails(tmp_path):
    assert run(build(tmp_path / "a", PHOTOS), build(tmp_path / "b", {**PHOTOS, "test/P_fresh/b.jpg": b"other"})) == 1


def test_photo_missing_on_both_sides_fails(tmp_path):
    gone = {**PHOTOS, "test/P_fresh/b.jpg": None}
    assert run(build(tmp_path / "a", gone), build(tmp_path / "b", gone)) == 1
    assert run(build(tmp_path / "c", gone), build(tmp_path / "d", gone), "--rows", "2") == 1


def test_column_only_in_b_fails_with_rows(tmp_path):
    assert run(build(tmp_path / "a", PHOTOS), build(tmp_path / "b", PHOTOS, extra_column=True), "--rows", "1") == 1
    assert run(build(tmp_path / "c", PHOTOS, extra_column=True), build(tmp_path / "d", PHOTOS), "--rows", "1") == 1


def test_photo_not_matching_its_manifest_hash_fails(tmp_path):
    a, b = build(tmp_path / "a", PHOTOS), build(tmp_path / "b", PHOTOS)
    for folder in (a, b):  # the same wrong file on both sides
        (folder / "train/P_fresh/a.jpg").write_bytes(b"changed after the manifest was written")
    assert run(a, b) == 1
