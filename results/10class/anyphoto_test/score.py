"""Score a predict.py CSV against anyphoto_test.csv: how many photos got a good answer, per kind of photo.

    python results/10class/anyphoto_test/score.py results.csv

Good answer: turned away for fruit, other vegetables, food, drawings/toys and everyday photos; right kind and state for one of the
five vegetables in an everyday scene; either for cut vegetables and photos with several of the five vegetables.
The photos must be named <id>.jpg (results/10class/outside_test/evaluate.py fetch --csv ... saves them that way).
"""
import csv, sys, json, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
NAMES = ['cucumber_fresh','cucumber_rotten','potato_fresh','potato_rotten','tomato_fresh','tomato_rotten','bellpepper_fresh','bellpepper_rotten','carrot_fresh','carrot_rotten']
REJECT = ('rotten_fruit', 'fresh_fruit', 'other_veg', 'food', 'drawing_toy', 'everyday')
truth = {r['id']: r for r in csv.DictReader(open(HERE / 'anyphoto_test.csv', encoding='utf-8'))}
pred = {Path(r['image']).stem: r for r in csv.DictReader(open(sys.argv[1], encoding='utf-8'))}
assert set(pred) == set(truth), (len(pred), len(truth))
by = collections.defaultdict(lambda: [0, 0])
for i, t in truth.items():
    p = pred[i]; turned_away = p['known'] == 'no'; guess = max(NAMES, key=lambda n: float(p['prob_' + n]))
    labels = t['label'].split('|') if t['label'] else []
    if t['category'] in REJECT: good = turned_away
    elif t['category'] == 'five_daily': good = not turned_away and guess == labels[0]
    else: good = turned_away or guess in labels
    by[t['category']][0] += 1; by[t['category']][1] += good
total = sum(v[1] for v in by.values())
print(json.dumps({'n': len(truth), 'good': total, 'good_pct': round(100 * total / len(truth), 1),
                  'by_category': {k: f'{v[1]}/{v[0]}' for k, v in by.items()}}, indent=1, ensure_ascii=False))
