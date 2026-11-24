"""predict.py outputs and the CLIP feature cache: the cases of the 2026-11-20 review (docs/codex_handoff.md)."""

import csv
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]

from predict import heatmap_names, report_notes  # noqa: E402  (tests/conftest.py puts src on the path)
from train_clip_head import clip_features, encoder_tag  # noqa: E402

PHOTO = REPO / "dataset/test/N_rotten/tomato_rotten_1188.jpg"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "src/predict.py"), "--model", str(REPO / "model.pth"),
                           "--threads", "2", *args], capture_output=True, text=True)


# a run of predict.py is the whole system: it needs open_clip and the CLIP weights (about 600 MB)
NEEDS_CLIP = pytest.mark.skipif(not os.environ.get("FULL_TESTS"), reason="predict.py runs the whole system (CLIP); set FULL_TESTS=1")


def test_heatmap_names_are_unique(tmp_path):
    photos = [tmp_path / "a.jpg", tmp_path / "x/a.jpg", tmp_path / "y/a.jpg", tmp_path / "y/a.png", tmp_path / "y/b.jpg"]
    want = ["a_heatmap.jpg", "x/a_heatmap.jpg", "y/a.jpg_heatmap.jpg", "y/a.png_heatmap.jpg", "y/b_heatmap.jpg"]
    assert [heatmap_names(photos, tmp_path)[p].as_posix() for p in photos] == want
    single = tmp_path / "x/a.jpg"
    single.parent.mkdir()
    single.write_bytes(b"")
    assert heatmap_names([single], single) == {single: Path("a_heatmap.jpg")}   # --input was one photo


def test_heatmap_names_resolve_collisions_with_full_names_and_folders(tmp_path):
    paths = ["a.jpg", "a.png", "a.jpg.png", "b.jpg", "b_heatmap.jpg/photo.jpg"]
    photos = [tmp_path / path for path in paths]
    outputs = set(heatmap_names(photos, tmp_path).values())

    assert len(outputs) == len(photos)
    assert not any(parent in outputs for output in outputs for parent in output.parents)


def test_heatmap_names_differ_by_more_than_letter_case(tmp_path):
    paths = ["A.jpg", "a.png", "IMG.JPG", "img.jpg.png", "Sub/x.jpg", "sub/X.png"]
    names = heatmap_names([tmp_path / path for path in paths], tmp_path)
    assert len({name.as_posix().casefold() for name in names.values()}) == len(paths)
    plain = heatmap_names([tmp_path / "Photo.JPG"], tmp_path)
    assert plain[tmp_path / "Photo.JPG"] == Path("Photo_heatmap.jpg")   # an ordinary name keeps its letters


@NEEDS_CLIP
def test_heatmaps_of_same_named_photos_do_not_overwrite(tmp_path):
    for sub, src in (("a", PHOTO), ("b", REPO / "dataset/test/N_rotten/tomato_rotten_1201.jpg")):
        (tmp_path / "in" / sub).mkdir(parents=True)
        shutil.copy(src, tmp_path / "in" / sub / "photo.jpg")
    shutil.copy(PHOTO, tmp_path / "in/top.jpg")
    done = _run(["--input", str(tmp_path / "in"), "--output", str(tmp_path / "out.csv"), "--heatmap", str(tmp_path / "heat")])
    assert done.returncode == 0, done.stderr
    rows = list(csv.DictReader(open(tmp_path / "out.csv", encoding="utf-8")))
    maps = [r["heatmap"] for r in rows if r["heatmap"]]
    assert len(maps) == 3 and len(set(maps)) == 3 and all(Path(m).is_file() for m in maps)
    assert Path(maps[0]).read_bytes() != Path(maps[1]).read_bytes()
    assert (tmp_path / "heat/top_heatmap.jpg").is_file()   # a photo directly in the input folder keeps the old name


@NEEDS_CLIP
def test_all_photos_failing_is_an_error(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in/broken.jpg").write_bytes(b"not a photo")
    done = _run(["--input", str(tmp_path / "in"), "--output", str(tmp_path / "out.csv"), "--report", str(tmp_path / "r.html")])
    assert done.returncode == 1
    rows = list(csv.DictReader(open(tmp_path / "out.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["error"] and (tmp_path / "r.html").is_file()
    shutil.copy(PHOTO, tmp_path / "in/good.jpg")   # one photo that works: success
    assert _run(["--input", str(tmp_path / "in"), "--output", str(tmp_path / "out2.csv")]).returncode == 0


def test_report_notes_follow_the_options():
    system = types.SimpleNamespace(calibrated=True)
    assert "90% 미만" in report_notes(system, 0.9)[0] and "보정한 값" in report_notes(system, 0.9)[0]
    assert "75% 미만" in report_notes(system, 0.75)[0]
    off = report_notes(system, 0)[0]
    assert "미만이면" not in off and "판단 보류를 쓰지 않아서" in off
    raw = report_notes(types.SimpleNamespace(calibrated=False), 0.9)
    assert "보정하지 않은 값" in raw[0] and "보정한 값입니다" not in raw[0]
    assert any("양파·고구마·대파" in note for note in raw)        # the notes always describe the one eight-vegetable system


class _FakeClip:
    def __init__(self, value: float, width: int = 4):
        self.value, self.width, self.calls = value, width, 0

    def image_features(self, images):
        self.calls += len(images)
        return torch.full((len(images), self.width), self.value)


def test_feature_cache_keeps_image_models_apart(tmp_path):
    cache, paths, load = tmp_path / "f.npz", [Path("p1.jpg"), Path("p2.jpg")], (lambda i: None)
    default, other = encoder_tag("ViT-B-16", "datacomp_xl_s13b_b90k"), encoder_tag("ViT-L-16-SigLIP2-256", "webli")
    assert default == "" and other
    a, b = _FakeClip(1.0), _FakeClip(2.0, width=6)
    assert float(clip_features(a, paths, cache=cache, load=load, encoder=default).mean()) == 1.0
    alternate = clip_features(b, paths, cache=cache, load=load, encoder=other)
    assert alternate.shape == (2, 6) and float(alternate.mean()) == 2.0
    assert float(clip_features(_FakeClip(9.0), paths, cache=cache, load=load, encoder=default).mean()) == 1.0   # still cached
    again = _FakeClip(9.0)
    assert float(clip_features(again, paths, cache=cache, load=load, encoder=other).mean()) == 2.0 and again.calls == 0


def test_feature_cache_reads_caches_written_before_the_model_prefix(tmp_path):
    cache = tmp_path / "old.npz"
    np.savez(cache, keys=np.array(["p1.jpg"]), features=np.full((1, 4), 5.0, np.float32))
    clip = _FakeClip(9.0)
    out = clip_features(clip, [Path("p1.jpg")], cache=cache, load=lambda i: None, encoder=encoder_tag("ViT-B-16", "datacomp_xl_s13b_b90k"))
    assert float(out.mean()) == 5.0 and clip.calls == 0


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_output_or_report_on_an_input_photo_is_refused_and_the_photo_survives(tmp_path):
    """--output / --report naming an input photo (relative or absolute spelling) stops before anything is written."""
    folder = tmp_path / "in"
    folder.mkdir()
    photo = folder / "a.jpg"
    shutil.copy(PHOTO, photo)
    before = _sha(photo)
    relative = Path("in") / "sub" / ".." / "a.jpg"       # another spelling of the same file, from cwd = tmp_path
    (folder / "sub").mkdir()
    for option, target in (("--output", str(photo)), ("--output", str(relative)), ("--report", str(photo)), ("--report", str(relative))):
        extra = [option, target] + ([] if option == "--output" else ["--output", str(tmp_path / "ok.csv")])
        done = subprocess.run([sys.executable, str(REPO / "src/predict.py"), "--model", str(REPO / "model.pth"),
                               "--input", str(folder), *extra], capture_output=True, text=True, cwd=tmp_path)
        assert done.returncode != 0 and "is the input photo" in done.stderr and option in done.stderr, done.stderr
        assert _sha(photo) == before and not (tmp_path / "ok.csv").exists()
    single = subprocess.run([sys.executable, str(REPO / "src/predict.py"), "--model", str(REPO / "model.pth"),
                             "--input", str(photo), "--output", "in/a.jpg"], capture_output=True, text=True, cwd=tmp_path)
    assert single.returncode != 0 and _sha(photo) == before


@NEEDS_CLIP
def test_heatmap_folder_inside_the_input_folder_is_not_read_as_input(tmp_path):
    """The same command twice: the second run sees the same photos, not the heat maps the first one wrote."""
    folder = tmp_path / "in"
    (folder / "x").mkdir(parents=True)
    shutil.copy(PHOTO, folder / "a.jpg")
    shutil.copy(REPO / "dataset/test/N_rotten/tomato_rotten_1201.jpg", folder / "x" / "b.jpg")
    hashes = {p: _sha(p) for p in (folder / "a.jpg", folder / "x" / "b.jpg")}
    listed = []
    for run in (1, 2):
        out = tmp_path / f"run{run}.csv"
        done = _run(["--input", str(folder), "--output", str(out), "--heatmap", str(folder / "heat")])
        assert done.returncode == 0, done.stderr
        listed.append(sorted(row["image"] for row in csv.DictReader(open(out, encoding="utf-8"))))
    assert listed[0] == listed[1] and len(listed[0]) == 2
    assert sorted(p.relative_to(folder / "heat").as_posix() for p in (folder / "heat").rglob("*.jpg")) == ["a_heatmap.jpg", "x/b_heatmap.jpg"]
    assert all(_sha(p) == h for p, h in hashes.items())
    same = _run(["--input", str(folder), "--output", str(tmp_path / "no.csv"), "--heatmap", str(folder)])
    assert same.returncode != 0 and "is the input folder" in same.stderr and not (tmp_path / "no.csv").exists()
