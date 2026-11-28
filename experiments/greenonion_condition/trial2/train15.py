"""Run src/train_new_veg_head.py unchanged, except that greenonion_rotten counts as a trained class (trial2/PREREG.md).

    python experiments/greenonion_condition/trial2/train15.py --model ... --data ... --out ... [--tau 0.8]
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
import train_new_veg_head  # noqa: E402

train_new_veg_head.UNTRAINED_CLASSES = set()   # the head records no untrained class, so class 15 gets its zero-shot score

if __name__ == "__main__":
    train_new_veg_head.main()
