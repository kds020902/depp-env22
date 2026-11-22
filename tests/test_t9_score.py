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
    def system(sweet_right, potato_as_sweet, rotten_potato_right, called_fresh, total):
        return {"sweetpotato_rotten": {"right": sweet_right}, "potato_rotten": {"right": rotten_potato_right},
                "_all": {"potato_answered_sweet_potato": potato_as_sweet, "rotten_called_fresh": called_fresh, "right": total}}
    base = system(10, 1, 25, 3, 60)
    assert all(conditions(base, system(16, 3, 24, 3, 60)).values())
    for worse in (system(15, 3, 24, 3, 60), system(16, 4, 24, 3, 60), system(16, 3, 23, 3, 60), system(16, 3, 24, 4, 60), system(16, 3, 24, 3, 59)):
        assert not all(conditions(base, worse).values())
    assert paired(["potato_rotten", "sweetpotato_rotten"], ["potato_rotten", "potato_rotten"], ["sweetpotato_rotten", "sweetpotato_fresh"]) == {
        "potato_rotten": {"right_to_wrong": 1, "wrong_to_right": 0}, "sweetpotato_rotten": {"right_to_wrong": 0, "wrong_to_right": 1}}
