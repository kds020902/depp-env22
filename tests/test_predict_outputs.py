"""predict.py outputs and the CLIP feature cache: the cases of the 2026-11-20 review (docs/codex_handoff.md)."""

import csv
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]

from predict import heatmap_names, report_notes  # noqa: E402  (tests/conftest.py puts src on the path)
from train_clip_head import clip_features, encoder_tag  # noqa: E402

PHOTO = REPO / "dataset/test/N_rotten/tomato_rotten_1188.jpg"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "src/predict.py"), "--model", str(REPO / "model.pth"), "--no-clip",
                           "--threads", "2", *args], capture_output=True, text=True)


def test_heatmap_names_are_unique(tmp_path):
    photos = [tmp_path / "a.jpg", tmp_path / "x/a.jpg", tmp_path / "y/a.jpg", tmp_path / "y/a.png", tmp_path / "y/b.jpg"]
    want = ["a_heatmap.jpg", "x/a_heatmap.jpg", "y/a.jpg_heatmap.jpg", "y/a.png_heatmap.jpg", "y/b_heatmap.jpg"]
    assert [heatmap_names(photos, tmp_path)[p].as_posix() for p in photos] == want
    single = tmp_path / "x/a.jpg"
    single.parent.mkdir()
    single.write_bytes(b"")
    assert heatmap_names([single], single) == {single: Path("a_heatmap.jpg")}   # --input was one photo


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
    system = types.SimpleNamespace(calibrated=True, new_vegetables=False)
    assert "90% 미만" in report_notes(system, 0.9)[0] and "보정한 값" in report_notes(system, 0.9)[0]
    assert "75% 미만" in report_notes(system, 0.75)[0]
    off = report_notes(system, 0)[0]
    assert "미만이면" not in off and "판단 보류를 쓰지 않아서" in off
    raw = report_notes(types.SimpleNamespace(calibrated=False, new_vegetables=True), 0.9)
    assert "보정하지 않은 값" in raw[0] and "보정한 값입니다" not in raw[0]


class _FakeClip:
    def __init__(self, value: float):
        self.value, self.calls = value, 0

    def image_features(self, images):
        self.calls += len(images)
        return torch.full((len(images), 4), self.value)


def test_feature_cache_keeps_image_models_apart(tmp_path):
    cache, paths, load = tmp_path / "f.npz", [Path("p1.jpg"), Path("p2.jpg")], (lambda i: None)
    default, other = encoder_tag("ViT-B-16", "datacomp_xl_s13b_b90k"), encoder_tag("ViT-L-16-SigLIP2-256", "webli")
    assert default == "" and other
    a, b = _FakeClip(1.0), _FakeClip(2.0)
    assert float(clip_features(a, paths, cache=cache, load=load, encoder=default).mean()) == 1.0
    assert float(clip_features(b, paths, cache=cache, load=load, encoder=other).mean()) == 2.0   # not the cached 1.0
    assert float(clip_features(_FakeClip(9.0), paths, cache=cache, load=load, encoder=default).mean()) == 1.0   # still cached
    again = _FakeClip(9.0)
    assert float(clip_features(again, paths, cache=cache, load=load, encoder=other).mean()) == 2.0 and again.calls == 0


def test_feature_cache_reads_caches_written_before_the_model_prefix(tmp_path):
    cache = tmp_path / "old.npz"
    np.savez(cache, keys=np.array(["p1.jpg"]), features=np.full((1, 4), 5.0, np.float32))
    clip = _FakeClip(9.0)
    out = clip_features(clip, [Path("p1.jpg")], cache=cache, load=lambda i: None, encoder=encoder_tag("ViT-B-16", "datacomp_xl_s13b_b90k"))
    assert float(out.mean()) == 5.0 and clip.calls == 0
