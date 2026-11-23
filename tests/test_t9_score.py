"""The counting of results/t9_test/score.py on made-up answers (no photo, no model)."""

import sys

from conftest import REPO

sys.path.insert(0, str(REPO / "results/t9_test"))
import pytest  # noqa: E402
import score  # noqa: E402
from score import claim_the_one_run, conditions, paired, parse_command, report, right, write_result  # noqa: E402

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


def test_a_run_that_finishes_between_the_first_look_and_the_marker_is_seen(tmp_path, monkeypatch):
    """Codex's order: B has seen no result; A saves its result and removes its marker; B takes the marker. B must look
    again, remove only its own marker and stop; A's result is untouched."""
    result, marker = tmp_path / "scores_once.json", tmp_path / "scores_once.running"
    real_open = score.os.open

    def a_finishes_first(path, flags, *more):
        if str(path) == str(marker):
            result.write_text('{"run": "A"}', encoding="utf-8")      # A wrote its result (and its marker is already gone)
        return real_open(path, flags, *more)

    monkeypatch.setattr(score.os, "open", a_finishes_first)
    with pytest.raises(SystemExit, match="another run scored T9"):
        claim_the_one_run(result, marker)
    assert not marker.exists() and result.read_text(encoding="utf-8") == '{"run": "A"}'
    monkeypatch.setattr(score.os, "open", real_open)
    result.unlink()
    marker.write_text("left by another run")                         # an interrupted run's marker is never removed
    with pytest.raises(SystemExit, match="under way or was interrupted"):
        claim_the_one_run(result, marker)
    assert marker.read_text() == "left by another run"


def test_only_the_two_command_forms_are_accepted(tmp_path):
    assert parse_command([]) == (False, score.RESULT)
    assert parse_command(["--rehearsal", str(tmp_path / "o.json")]) == (True, tmp_path / "o.json")
    for bad in (["a.json"], ["a.json", "b.json"], ["--rehearsal"], ["--rehearsal", "--rehearsal"], ["--rehearsal", "a.json", "b.json"],
                ["a.json", "--rehearsal"], ["--rehearsal", "a.json", "--rehearsal"], ["--unknown"], ["--rehearsal", "--out"]):
        with pytest.raises(SystemExit, match="usage"):
            parse_command(bad)
    for reserved in ("scores_once.json", "scores_once.running", "scores_once.partial"):
        with pytest.raises(SystemExit, match="may not write"):
            parse_command(["--rehearsal", str(score.HERE / reserved)])
        with pytest.raises(SystemExit, match="may not write"):
            parse_command(["--rehearsal", str(score.HERE / "." / "x" / ".." / reserved)])
    assert parse_command(["--rehearsal", str(score.HERE / "scores_once.dev")])[0]


def test_wrong_arguments_stop_before_models_photos_or_the_marker():
    """The command itself: no DEPP_WORK, no model is loaded, nothing is created in results/t9_test."""
    import os
    import subprocess
    before = sorted(p.name for p in score.HERE.iterdir() if p.name != "__pycache__")
    environment = {k: v for k, v in os.environ.items() if k != "DEPP_WORK"}
    for bad in (["one.json", "two.json"], ["--rehearsal"], ["--rehearsal", "--rehearsal"], ["one.json"]):
        done = subprocess.run([sys.executable, str(score.HERE / "score.py"), *bad], capture_output=True, text=True, env=environment)
        assert done.returncode != 0 and "usage" in done.stderr and "DEPP_WORK" not in done.stderr, done.stderr
    assert sorted(p.name for p in score.HERE.iterdir() if p.name != "__pycache__") == before


def test_each_run_writes_through_its_own_temporary_file(tmp_path):
    """A rehearsal writing scores_once.dev next to a scoring run's unfinished file leaves that file alone; a failed write
    removes only its own temporary file."""
    unfinished = tmp_path / "scores_once.partial"
    unfinished.write_text("the one scoring, half written", encoding="utf-8")
    write_result({"rehearsal": True}, tmp_path / "scores_once.dev")
    assert unfinished.read_text(encoding="utf-8") == "the one scoring, half written"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scores_once.dev", "scores_once.partial"]
    with pytest.raises(TypeError):
        write_result({"not json": object()}, tmp_path / "broken.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scores_once.dev", "scores_once.partial"]


def test_rehearsal_names_windows_would_read_as_reserved_are_refused_before_the_files_exist():
    """Codex's Windows case: with a dot or a space at the end, "scores_once.json." is written to scores_once.json there.
    None of the three reserved files exists yet; every spelling is refused with the usage text and nothing is created."""
    import os
    import subprocess
    reserved = ("scores_once.json", "scores_once.running", "scores_once.partial")
    assert not any((score.HERE / name).exists() for name in reserved)
    before = sorted(p.name for p in score.HERE.iterdir() if p.name != "__pycache__")
    environment = {k: v for k, v in os.environ.items() if k != "DEPP_WORK"}
    for name in reserved:
        for ending in (".", " ", ". .", "  "):
            with pytest.raises(SystemExit, match="usage") as stopped:
                parse_command(["--rehearsal", str(score.HERE / (name + ending))])
            assert "may not write" in str(stopped.value)
        with pytest.raises(SystemExit, match="may not write"):
            parse_command(["--rehearsal", str(score.HERE / name.upper())])              # letter case (Windows, macOS)
        for ending in (".", " "):                                                       # the command itself
            done = subprocess.run([sys.executable, str(score.HERE / "score.py"), "--rehearsal", str(score.HERE / (name + ending))],
                                  capture_output=True, text=True, env=environment)
            assert done.returncode != 0 and "usage" in done.stderr and "may not write" in done.stderr, done.stderr
    assert sorted(p.name for p in score.HERE.iterdir() if p.name != "__pycache__") == before
    for fine in ("scores_once.dev", "scores_once.json.bak", "rehearsal.json"):
        assert parse_command(["--rehearsal", str(score.HERE / fine)]) == (True, score.HERE / fine)
