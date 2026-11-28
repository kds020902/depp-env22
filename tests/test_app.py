"""The web page (app.py) without Gradio or CLIP: what it shows for each kind of answer, and the Space folder."""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import app  # noqa: E402
import build_space  # noqa: E402

PHOTO = REPO / "dataset/test/P_fresh/potato_fresh_0201.jpg"


def result(index=2, probability=0.97, unknown=False, held=False, message="", rot_map=None):
    probabilities = torch.full((16,), (1 - probability) / 15)
    probabilities[index] = probability
    return {"index": index, "probability": probability, "unknown": unknown, "held": held, "message": message,
            "probabilities": probabilities, "rot_map": rot_map, "scale": 1.0}


class Fake:
    def __init__(self, r):
        self.r, self.seen = r, []

    def classify(self, photo, hold_back_below):
        self.seen.append((photo.mode, hold_back_below))
        return self.r


def test_a_shown_answer_names_the_vegetable_its_condition_and_the_probability():
    head, bars, heat = app.answer(Fake(result(rot_map=torch.ones(7, 7))), PHOTO)
    assert head == "## 감자 정상\n확률 97%"
    assert list(bars)[0] == "감자 정상" and len(bars) == 5
    assert heat is not None and heat.mode == "RGB"


def test_the_photo_is_opened_as_predict_py_opens_it_and_held_back_below_the_adopted_threshold():
    fake = Fake(result())
    app.answer(fake, PHOTO)
    assert fake.seen == [("RGB", app.HOLD_BACK_BELOW)] and app.HOLD_BACK_BELOW == 0.9


def test_a_held_back_answer_is_not_shown_as_an_answer():
    message = "판단 보류: 가장 가까운 답은 감자 비정상(확률 62%)입니다. 정상인지 상했는지 헷갈립니다."
    head, bars, _ = app.answer(Fake(result(3, 0.62, held=True, message=message)), PHOTO)
    assert head.startswith("## 판단 보류\n가장 가까운 답은 감자 비정상") and bars


def test_a_turned_away_photo_shows_no_class_probabilities_and_no_heat_map():
    r = result(unknown=True, message=f"{app.UNKNOWN_MESSAGE} (고양이로 보입니다.)", rot_map=None)
    assert app.answer(Fake(r), PHOTO) == (f"## {app.UNKNOWN_MESSAGE}\n고양이로 보입니다.", {}, None)


def test_green_onion_has_one_bar_because_its_condition_is_not_judged():
    bars = app.shown_probabilities(result(14, 0.6))
    assert bars["대파 상태 판정 안 함"] == pytest.approx(0.6 + 0.4 / 15)


def test_no_file_a_file_that_is_no_photo_and_an_oversized_photo_do_not_reach_the_model(tmp_path, monkeypatch):
    fake = Fake(result())
    assert app.answer(fake, None)[0] == "사진을 올려 주세요."
    assert app.answer(fake, REPO / "README.md")[0].startswith("## 판정하지 못했습니다")
    monkeypatch.setattr(app, "MAX_PIXELS", 100)
    assert "너무 큽니다" in app.answer(fake, PHOTO)[0]
    assert fake.seen == []


def test_the_space_folder_has_what_app_py_imports_and_refuses_a_folder_in_use(tmp_path):
    copied = build_space.build(tmp_path / "space")
    assert {"app.py", "web.css", "model.pth", "README.md", "requirements.txt", "src/predict.py", "src/clip_model.py"} <= set(copied)
    imported = {line.split()[1] for name in build_space.SRC + ("../app.py",)
                for line in (REPO / "src" / name).read_text(encoding="utf-8").splitlines()
                if line.startswith(("from ", "import ")) or line.startswith("    from ")
                for line in [line.strip()]}
    local = {p.stem for p in (REPO / "src").glob("*.py")}
    assert imported & local <= {Path(name).stem for name in build_space.SRC}
    with pytest.raises(SystemExit):
        build_space.build(tmp_path / "space")


def test_the_space_installs_the_versions_the_recorded_results_were_made_with():
    pins = lambda path: {l.split("==")[0]: l.split("==")[1] for l in path.read_text(encoding="utf-8").split() if "==" in l}  # noqa: E731
    space, project = pins(REPO / "space/requirements.txt"), pins(REPO / "requirements.txt")
    assert space and all(project[name] == version for name, version in space.items())
    assert {"torch", "torchvision", "Pillow", "numpy", "open_clip_torch"} <= set(space)


def test_unexpected_error_unlocks_photo_and_button():
    """A failed run (also a failed ZeroGPU call) must not leave the page locked: a failure handler re-enables both."""
    import app

    class Broken:
        def classify(self, *args, **kwargs):
            raise KeyError("unexpected")

    page = app.build(Broken())
    fns = page.fns.values() if hasattr(page.fns, "values") else page.fns
    failure = [f for f in fns if getattr(f, "trigger_only_on_failure", False)]
    assert len(failure) == 1
    photo, button, message, *_ = failure[0].fn("photo.jpg")
    assert photo["interactive"] is True and button["interactive"] is True
    assert message.startswith("### 판정하지 못했어요")


def test_greenonion_hint_never_says_fresh_and_marks_experimental():
    """Experimental green onion condition: only 'sign to check' or 'held', always labelled as not validated."""
    import numpy as np
    import app

    hint = np.load(app.GREENONION_HINT)
    spoiled = hint["spoiled"].mean(0)
    fresh = hint["fresh"].mean(0)
    unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
    near_spoiled = app.greenonion_hint(unit(spoiled - fresh))
    near_fresh = app.greenonion_hint(unit(fresh - spoiled))
    assert "이상 징후 후보" in near_spoiled
    assert "판단 보류" in near_fresh
    for text in (near_spoiled, near_fresh):
        assert "실험 기능" in text and "정상" not in text


def test_greenonion_hint_only_for_confident_green_onion_answers(monkeypatch, tmp_path):
    """No condition guess for a held answer, a turned-away photo or another vegetable (Codex review)."""
    import numpy as np
    import app
    from labels import SPECIES_NAMES

    green = SPECIES_NAMES.index("greenonion") * 2
    calls = []
    monkeypatch.setattr(app, "check_size", lambda path: None)
    monkeypatch.setattr(app, "open_image", lambda path: object())
    monkeypatch.setattr(app, "greenonion_hint", lambda f: calls.append(1) or "\n\nHINT")

    class Clip:
        def image_features(self, photos):
            import torch
            return torch.zeros(1, 512)

    def system(**r):
        base = {"index": green, "unknown": False, "held": False, "probability": 0.95, "probabilities": np.zeros(16),
                "message": "", "rot_map": None, "scale": 1.0}
        base.update(r)
        return type("S", (), {"clip": Clip(), "classify": lambda self, photo, hold: base})()

    assert app.answer(system(), "p.jpg")[0].endswith("HINT")
    assert not app.answer(system(held=True, message="판단 보류: 가장 가까운 답"), "p.jpg")[0].endswith("HINT")
    assert not app.answer(system(unknown=True, message="학습된 데이터가 아닙니다 (x)"), "p.jpg")[0].endswith("HINT")
    assert not app.answer(system(index=0), "p.jpg")[0].endswith("HINT")
    assert len(calls) == 1
