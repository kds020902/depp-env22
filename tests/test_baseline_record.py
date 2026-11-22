"""The baseline record (results/baseline) still describes model.pth and the stored development sets. No CLIP weights needed."""

import json
import subprocess
import sys

import torch

from conftest import REPO

sys.path.insert(0, str(REPO / "tools"))
import copy  # noqa: E402

import pytest  # noqa: E402
from compare_on_dev import cached_inputs_apply, development_sets, evaluate, parse_candidates, right, tally  # noqa: E402


def test_baseline_card_matches_the_files():
    done = subprocess.run([sys.executable, "tools/baseline_card.py", "--check"], cwd=REPO, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def test_development_baseline_is_reproduced():
    stored = json.load(open(REPO / "results/baseline/tv12_dev.json"))["systems"]["TV-12"]
    again = evaluate(torch.load(REPO / "model.pth", map_location="cpu", weights_only=True), development_sets())
    assert json.loads(json.dumps(again)) == stored


def test_tally_counts_each_kind_of_outcome():
    labels = ["potato_rotten", "potato_rotten", "potato_fresh", "greenonion_fresh", "sweetpotato_rotten"]
    answers = ["potato_rotten", "potato_fresh", "turned_away", "greenonion_rotten", "potato_rotten"]
    held = [False, False, False, True, False]
    assert tally(labels, answers, held) == {"n": 5, "right": 2, "rotten_n": 3, "rotten_called_fresh": 1, "rotten_called_fresh_same_kind": 1,
                                            "rotten_called_fresh_shown": 1, "turned_away": 1, "held_back": 1, "shown": 3, "shown_right": 1}
    assert right("greenonion_fresh", "greenonion_rotten") and not right("potato_fresh", "potato_rotten")


def test_rotten_called_fresh_counts_other_vegetables_too():
    """Codex's case: a rotten sweet potato answered "fresh potato" is a rotten photo called fresh, as in results/scorecard."""
    labels = ["sweetpotato_rotten", "potato_rotten", "onion_rotten", "tomato_rotten"]
    answers = ["potato_fresh", "potato_fresh", "greenonion_fresh", "carrot_fresh"]   # a green onion answer claims no condition
    t = tally(labels, answers, [False, False, False, True])
    assert (t["rotten_called_fresh"], t["rotten_called_fresh_same_kind"], t["rotten_called_fresh_shown"]) == (3, 1, 2)


def test_stored_inputs_are_refused_when_the_candidate_was_not_made_with_them():
    base = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    assert cached_inputs_apply(base, base) == []
    text_only = copy.deepcopy(base)                      # heads, thresholds and text may change
    text_only["new_vegetable_head"]["text_embeddings"] += 0.01
    text_only["clip_head"]["coef"] *= 0.5
    text_only["new_vegetable_head"]["switch_probability"] = 0.5
    assert cached_inputs_apply(base, text_only) == []
    for change in (lambda c: c["clip_head"]["clip"].update(pretrained="laion2b_s34b_b88k"),
                   lambda c: c["new_vegetable_head"]["clip"].update(arch="ViT-L-14"),
                   lambda c: c["preprocess"].update(resize=[256, 256]),
                   lambda c: c["preprocess"].update(interpolation="bicubic"),
                   lambda c: c["state_dict"][next(iter(c["state_dict"]))].add_(1)):
        other = copy.deepcopy(base)
        change(other)
        assert cached_inputs_apply(base, other), "this change must be refused"


def test_candidate_names_cannot_replace_the_baseline_or_repeat():
    assert parse_candidates(["A=a.pth", "B=dir/b=1.pth"]) == {"A": REPO.__class__("a.pth"), "B": REPO.__class__("dir/b=1.pth")}
    for bad in (["TV-12=a.pth"], ["A=a.pth", "A=b.pth"], ["a.pth"], ["=a.pth"], ["A="]):
        with pytest.raises(SystemExit):
            parse_candidates(bad)
