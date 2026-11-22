"""The training / T9 assignment of experiments/sweetpotato_photos (PREREG addendum 2). No photos, no model."""

import sys
from collections import Counter

from conftest import REPO

sys.path.insert(0, str(REPO / "experiments/sweetpotato_photos"))
from split import choose_exact, group_order, split  # noqa: E402


def photos(group: str, **counts: int) -> list[dict]:
    return [{"path": f"{group}/{c}_{i}.jpg", "class_name": c, "source_group": group, "severity": ""} for c, n in counts.items() for i in range(n)]


def potatoes(n_groups: int = 15) -> list[dict]:
    return [r for i in range(n_groups) for r in photos(f"potato-{i}", potato_rotten=2, potato_fresh=2)]


def by_hash(names: list[str]) -> list[str]:
    return sorted(names, key=group_order)


def test_exact_choice_is_the_earliest_that_can_be_completed():
    assert choose_exact([2] * 14 + [1, 2], 30) == list(range(14)) + [15]
    assert choose_exact([1, 2, 2], 4) == [1, 2] and choose_exact([2, 2], 3) is None and choose_exact([], 0) == []


def test_codex_case_gives_30_and_20_not_31_and_19():
    """In hash order: fourteen 2-photo groups, a 1-photo group, a 2-photo group; 19 photos in one training-only group."""
    names = by_hash([f"small-{i}" for i in range(16)])
    rows = [r for i, g in enumerate(names) for r in photos(g, sweetpotato_rotten=1 if i == 14 else 2)]
    result = split(rows + photos("large", sweetpotato_rotten=19) + potatoes())
    assert result["t9"]["sweetpotato_rotten"] == 30 and result["train_rotten_sweet_potato"] == 20
    assert result["status"] == "ready" and result["group_side"][names[14]] == "train" and result["group_side"][names[15]] == "t9"


def test_a_group_is_on_one_side_over_all_classes():
    """Three rotten sweet potatoes and a potato by one photographer: the group trains, the potato is not in T9."""
    result = split(photos("mixed", sweetpotato_rotten=3, potato_rotten=1) + photos("other", potato_fresh=1))
    sides = Counter(result["assignment"].values())
    assert result["group_side"] == {"mixed": "train", "other": "t9"} and sides["train"] == 3 and sides["t9"] == 1
    assert result["assignment"]["mixed/potato_rotten_0.jpg"].startswith("left out")
    for group, side in result["group_side"].items():
        assert {a for p, a in result["assignment"].items() if p.startswith(group + "/") and a in ("train", "t9")} <= {side}


def test_counts_after_every_exclusion_decide_hold():
    small = [r for i in range(15) for r in photos(f"s-{i}", sweetpotato_rotten=2)]
    ready = split(small + photos("large", sweetpotato_rotten=20) + potatoes())
    assert ready["status"] == "ready" and ready["t9"] == {"sweetpotato_rotten": 30, "potato_rotten": 30, "potato_fresh": 30}
    assert split(small + photos("large", sweetpotato_rotten=19) + potatoes())["status"] == "hold"       # training 19
    assert split(small + photos("large", sweetpotato_rotten=20) + potatoes(14))["status"] == "hold"     # potatoes 28
    one_group = split(photos("s", sweetpotato_rotten=2, potato_fresh=5) + photos("large", sweetpotato_rotten=30))
    assert one_group["t9"]["potato_fresh"] == 0 and one_group["status"] == "hold"   # 2 < 30: the small group trains too
    capped = split(small + photos("large", sweetpotato_rotten=20) + potatoes(20))
    assert capped["t9"]["potato_rotten"] == 30 and capped["status"] == "ready"      # each potato class is cut to 30


def test_codex_case_three_classes_are_chosen_together():
    """Sixteen 2-photo sweet potato groups, 18 training-only photos, fourteen potato-only groups (2 + 2); the last sweet
    potato group in hash order also has 2 + 2 potatoes. Choosing for sweet potatoes alone gave T9 30 / 28 / 28."""
    names = by_hash([f"small-{i}" for i in range(16)])
    rows = [r for g in names[:15] for r in photos(g, sweetpotato_rotten=2)]
    rows += photos(names[15], sweetpotato_rotten=2, potato_rotten=2, potato_fresh=2)
    result = split(rows + photos("large", sweetpotato_rotten=18) + potatoes(14))
    assert result["t9"] == {"sweetpotato_rotten": 30, "potato_rotten": 30, "potato_fresh": 30}
    assert result["train_rotten_sweet_potato"] == 20 and result["status"] == "ready"
    assert result["group_side"][names[15]] == "t9" and result["group_side"][names[14]] == "train"   # the last of the others gives way
    for group, side in result["group_side"].items():
        assert {a for p, a in result["assignment"].items() if p.startswith(group + "/") and a in ("train", "t9")} <= {side}


def test_only_sprouted_photos_are_at_most_a_third_of_a_class():
    def sprouted(group, **counts):
        return [{**r, "severity": "싹"} for r in photos(group, **counts)]
    plain = [r for i in range(10) for r in photos(f"p-{i}", sweetpotato_rotten=2)]
    sprout = [r for i in range(8) for r in sprouted(f"s-{i}", sweetpotato_rotten=2)]
    result = split(plain + sprout + photos("large", sweetpotato_rotten=20) + potatoes())
    assert result["t9"]["sweetpotato_rotten"] == 30 and result["t9_only_sprouted"]["sweetpotato_rotten"] == 10
    assert result["status"] == "ready" and result["train_rotten_sweet_potato"] == 26
    too_few_plain = split(plain[:18] + sprout + photos("large", sweetpotato_rotten=20) + potatoes())   # 18 plain + 16 sprouted: 28 at most
    assert too_few_plain["status"] == "hold"
    potato_sprouts = [r for i in range(15) for r in sprouted(f"q-{i}", potato_rotten=2)]
    result = split(plain + sprout + photos("large", sweetpotato_rotten=20) + potatoes(10) + potato_sprouts)
    assert result["t9"]["potato_rotten"] == 30 and result["t9_only_sprouted"]["potato_rotten"] == 10


def test_current_list_waits():
    import csv
    rows = [r for r in csv.DictReader(open(REPO / "experiments/sweetpotato_photos/collected.csv", encoding="utf-8")) if r["status"] == "kept"]
    result = split(rows)
    assert result["status"] == "hold" and result["t9"]["sweetpotato_rotten"] == 0
