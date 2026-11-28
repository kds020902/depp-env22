"""Trial 3 trainer (PREREG.md): trial 2's train15.py plus one change - greenonion_rotten (15) rows are weighted so that
their total weight equals that of greenonion_fresh (14) rows; every other row keeps weight 1.

    python experiments/greenonion_condition/trial3/train15w.py --model ... --data ... --out ... [--tau 0.8]
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
import train_new_veg_head  # noqa: E402

GO_FRESH, GO_ROTTEN = 14, 15
train_new_veg_head.UNTRAINED_CLASSES = set()   # as trial 2


def fit_head(x, y, k):
    from sklearn.linear_model import LogisticRegression
    w = np.ones(len(y))
    n14, n15 = int((y == GO_FRESH).sum()), int((y == GO_ROTTEN).sum())
    w[y == GO_ROTTEN] = n14 / n15
    lr = LogisticRegression(C=10, max_iter=5000).fit(x, y, sample_weight=w)
    coef = np.zeros((k + 1, x.shape[1])); intercept = np.full(k + 1, lr.intercept_.min() - 50.0)
    coef[lr.classes_] = lr.coef_; intercept[lr.classes_] = lr.intercept_
    return coef, intercept


train_new_veg_head.fit_head = fit_head

if __name__ == "__main__":
    train_new_veg_head.main()
