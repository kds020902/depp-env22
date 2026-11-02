"""Development check of PREREG.md: TV-12 against the candidates on the development photos only (T4 is not touched).

    python experiments/t3_weak_spots/evaluate.py K1 [K2]   -> dev_results.json

Uses the cached CLIP features / vegetable-model logits of the earlier development sets (record script) and the
development share of the newly reviewed photos (t4/lists/dev.csv in the working folder).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "watermark"))
from common import REPO, S, W, dev_rules, dev_score  # noqa: E402  (the scoring itself lives in common.py)

LISTS = Path(os.environ.get("T3WS_LISTS", S / "t4/lists"))  # round 2: t5/lists
OUT = os.environ.get("T3WS_OUT", "dev_results.json")


def main():
    res = {"TV-12": dev_score(REPO / "model.pth", "WC", LISTS)}
    for tag in sys.argv[1:]:
        res[tag] = dev_score(W / f"{tag}.pth", "WC" if tag in ("K1", "K3", "L1") else ("K2" if tag in ("K2", "K4") else tag), LISTS)
    base = res["TV-12"]; rules = {}
    for tag in sys.argv[1:]:
        checks = dev_rules(res[tag], base)
        rules[tag] = {"checks": checks, "pass": all(checks.values()), "weak_gain": res[tag]["weak_total"] - base["weak_total"]}
    ok = [t for t in rules if rules[t]["pass"]]
    best = max(ok, key=lambda t: rules[t]["weak_gain"]) if ok else None
    chosen = best if best is not None and rules[best]["weak_gain"] >= 10 else None
    res.update(rules=rules, chosen=chosen)
    json.dump(res, open(HERE / OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
