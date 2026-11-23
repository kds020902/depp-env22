"""The counting of results/t9_test/score.py on made-up answers (no photo, no model)."""

import sys

from conftest import REPO

sys.path.insert(0, str(REPO / "results/t9_test"))
from score import conditions, paired, report, right  # noqa: E402

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
