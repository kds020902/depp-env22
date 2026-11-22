"""The baseline record (results/baseline) still describes model.pth and the stored development sets. No CLIP weights needed."""

import json
import subprocess
import sys

import torch

from conftest import REPO

sys.path.insert(0, str(REPO / "tools"))
from compare_on_dev import development_sets, evaluate, right, tally  # noqa: E402


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
    assert tally(labels, answers, held) == {"n": 5, "right": 2, "rotten_n": 3, "rotten_called_fresh": 1, "turned_away": 1,
                                            "held_back": 1, "shown": 3, "shown_right": 1}
    assert right("greenonion_fresh", "greenonion_rotten") and not right("potato_fresh", "potato_rotten")
