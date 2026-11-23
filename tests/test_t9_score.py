"""The counting of results/t9_test/score.py on made-up answers (no photo, no model)."""

import sys

from conftest import REPO

sys.path.insert(0, str(REPO / "results/t9_test"))
import pytest  # noqa: E402
import score  # noqa: E402
from score import claim_the_one_run, conditions, paired, report, right  # noqa: E402

LABELS = ["sweetpotato_rotten"] * 4 + ["potato_rotten"] * 3 + ["potato_fresh"] * 3
SEVERITY = ["heavy", "heavy", "slight", "sprout"] + ["heavy"] * 3 + ["dirty"] * 3


def test_report_counts():
    answers = ["sweetpotato_rotten", "sweetpotato_fresh", "potato_rotten", "turned_away",
               "potato_rotten", "potato_fresh", "sweetpotato_rotten", "potato_fresh", "potato_rotten", "greenonion_fresh"]
    held = [False, True, False, False, False, False, False, True, False, False]
    r = report(LABELS, SEVERITY, answers, held)
    sp = r["sweetpotato_rotten"]
    assert (sp["right"], sp["class_right"], sp["answered_potato"], sp["turned_away"], sp["called_fresh"]) == (2, 1, 1, 1, 1)
    assert sp["right_heavy"] == [2, 2] and sp["right_slight"] == [0, 1] and sp["right_sprout"] == [0, 1]
    assert (r["potato_rotten"]["right"], r["potato_rotten"]["called_fresh"], r["potato_rotten"]["answered_sweet_potato"]) == (1, 1, 1)
    assert (r["potato_fresh"]["right"], r["potato_fresh"]["wrong_condition_same_kind"]) == (1, 1)
    assert r["_all"] == {"n": 10, "right": 4, "rotten_n": 7, "rotten_called_fresh": 2, "rotten_called_fresh_shown": 1, "potato_n": 6,
                         "potato_answered_sweet_potato": 1, "turned_away": 1, "held_back": 2, "shown": 7, "shown_right": 2}
    assert right("sweetpotato_rotten", "sweetpotato_fresh") and not right("potato_rotten", "potato_fresh")


def test_conditions_and_paired():
    def system(dirty_right, rotten_potato_right, called_fresh, called_fresh_shown, total):
        return {"potato_fresh": {"right": dirty_right}, "potato_rotten": {"right": rotten_potato_right},
                "_all": {"rotten_called_fresh": called_fresh, "rotten_called_fresh_shown": called_fresh_shown, "right": total}}
    base = system(5, 25, 3, 1, 40)
    assert all(conditions(base, system(11, 23, 5, 1, 43)).values())
    for worse in (system(10, 23, 5, 1, 43), system(11, 22, 5, 1, 43), system(11, 23, 6, 1, 43), system(11, 23, 5, 2, 43), system(11, 23, 5, 1, 42)):
        assert not all(conditions(base, worse).values())
    assert paired(["potato_rotten", "sweetpotato_rotten"], ["potato_rotten", "potato_rotten"], ["sweetpotato_rotten", "sweetpotato_fresh"]) == {
        "potato_rotten": {"right_to_wrong": 1, "wrong_to_right": 0}, "sweetpotato_rotten": {"right_to_wrong": 0, "wrong_to_right": 1}}


def test_potato_check_changes_only_confident_clip_fresh_potato_answers():
    """decision.apply_potato_check: only "rotten potato" answers of the five-vegetable system, only when the CLIP head alone
    gives fresh potato at least the threshold; TV-12 (no key) is untouched."""
    import torch
    from decision import Answers, apply_potato_check
    from labels import CLASS_NAMES
    fresh, rotten = CLASS_NAMES.index("potato_fresh"), CLASS_NAMES.index("potato_rotten")
    n = 5
    clip_logits = torch.zeros(n, 11)
    clip_logits[:, fresh] = torch.tensor([3.0, -3.0, 3.0, 3.0, 3.0])            # the head says fresh, except photo 1
    index = torch.tensor([rotten, rotten, rotten, rotten, 0])                  # photo 4 is another answer
    unknown = torch.tensor([False, False, True, False, False])                 # photo 2 was turned away
    switched = torch.tensor([False, False, False, True, False])                # photo 3 was answered by the new-vegetable head
    a = Answers(torch.zeros(n, 16), index, torch.full((n,), 0.95), unknown, switched, torch.zeros(n, 10), unknown)
    out = apply_potato_check(a, {"potato_check": {"threshold": 0.5}}, clip_logits)
    assert out.index.tolist() == [fresh, rotten, rotten, rotten, 0]
    assert abs(float(out.probability[0]) - float(torch.sigmoid(torch.tensor(3.0)))) < 1e-6 and torch.allclose(out.probability[1:], torch.full((4,), 0.95))


def test_flipped_potato_answer_and_condition_summary_agree():
    """Codex's case: the answer became fresh potato at 95.3% while predict.py's condition summary still said rotten 99%."""
    import torch
    from decision import Answers, apply_potato_check
    from labels import CLASS_NAMES
    from model_utils import marginal_probabilities
    fresh, rotten = CLASS_NAMES.index("potato_fresh"), CLASS_NAMES.index("potato_rotten")
    shown = torch.zeros(2, 16)
    shown[:, rotten], shown[:, fresh] = 0.99, 0.01
    clip_logits = torch.zeros(2, 11)
    clip_logits[0, fresh] = 3.0                                  # photo 0 is flipped, photo 1 is not
    clip_logits[1, rotten] = 3.0
    index = torch.tensor([rotten, rotten])
    no = torch.zeros(2, dtype=torch.bool)
    a = apply_potato_check(Answers(shown, index, torch.tensor([0.99, 0.99]), no, no, torch.zeros(2, 10), no), {"potato_check": {"threshold": 0.5}}, clip_logits)
    species, condition = marginal_probabilities(a.probabilities)
    assert int(a.index[0]) == fresh and int(condition[0].argmax()) == 0                      # condition 0 = fresh
    assert abs(float(condition[0].max()) - float(a.probability[0])) < 1e-6 and abs(float(a.probability[0]) - 0.9526) < 1e-3
    assert abs(float(species[0].max()) - 1.0) < 1e-6 and abs(float(a.probabilities[0].sum()) - 1.0) < 1e-6
    assert torch.equal(a.probabilities[1], shown[1]) and int(a.index[1]) == rotten            # the other photo is untouched


def test_scoring_stops_before_reading_when_a_seal_is_wrong_or_a_run_exists(tmp_path, monkeypatch):
    """Explicit exits (not assert, so python -O cannot skip them); a second or interrupted run is refused."""
    for name in ("t9_test.csv", "t9_main.csv", "RULES.md"):
        (tmp_path / name).write_bytes((score.HERE / name).read_bytes())
    monkeypatch.setattr(score, "HERE", tmp_path)
    assert set(score.seals()) == {"t9_test.csv", "t9_main.csv"}
    (tmp_path / "t9_main.csv").write_text((tmp_path / "t9_main.csv").read_text(encoding="utf-8") + "x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match its seal"):
        score.seals()
    result, marker = tmp_path / "scores_once.json", tmp_path / "scores_once.running"
    claim_the_one_run(result, marker)
    assert marker.exists()
    with pytest.raises(SystemExit, match="under way or was interrupted"):
        claim_the_one_run(result, marker)
    marker.unlink()
    result.write_text("{}")
    with pytest.raises(SystemExit, match="scored once already"):
        claim_the_one_run(result, marker)
    assert not marker.exists()


def test_t9_has_not_been_scored_without_a_record():
    """scores_once.json and RULES.md go together: a result file needs the addendum that reports it."""
    if (score.HERE / "scores_once.json").exists():
        assert "scores_once.json" in (score.HERE / "RULES.md").read_text(encoding="utf-8")
    assert not (score.HERE / "scores_once.running").exists()
