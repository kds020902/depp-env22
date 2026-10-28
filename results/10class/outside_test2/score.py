"""Score a predict.py CSV against outside_test2.csv: accuracy (most likely class), kept-correct (not turned away and right), per class.

    python results/10class/outside_test2/score.py results.csv

The photos must be named <id>.jpg (results/10class/outside_test/evaluate.py fetch --csv ... saves them that way).
"""
import csv, sys, json, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
NAMES = ['cucumber_fresh','cucumber_rotten','potato_fresh','potato_rotten','tomato_fresh','tomato_rotten','bellpepper_fresh','bellpepper_rotten','carrot_fresh','carrot_rotten']
truth = {r['id']: r for r in csv.DictReader(open(HERE / 'outside_test2.csv', encoding='utf-8'))}
pred = {Path(r['image']).stem: r for r in csv.DictReader(open(sys.argv[1], encoding='utf-8'))}
assert set(pred) == set(truth), (len(pred), len(truth))
out = {'n': len(truth), 'accuracy': 0, 'kept_correct': 0, 'rejected': 0, 'species_right': 0, 'condition_right': 0}
by = collections.defaultdict(lambda: [0, 0, 0]); src = collections.defaultdict(lambda: [0, 0])
for i, t in truth.items():
    p = pred[i]; guess = max(NAMES, key=lambda n: float(p['prob_' + n])); right = guess == t['class_name']
    out['accuracy'] += right; out['rejected'] += p['known'] == 'no'; out['kept_correct'] += right and p['known'] == 'yes'
    out['species_right'] += guess.split('_')[0] == t['class_name'].split('_')[0]; out['condition_right'] += guess.split('_')[1] == t['class_name'].split('_')[1]
    b = by[t['class_name']]; b[0] += 1; b[1] += right; b[2] += right and p['known'] == 'yes'
    s = src[t['source']]; s[0] += 1; s[1] += right
out['accuracy_pct'] = round(100 * out['accuracy'] / out['n'], 1)
out['by_class'] = {n: f'{by[n][1]}/{by[n][0]} (kept {by[n][2]})' for n in NAMES if n in by}
out['by_source'] = {k: f'{v[1]}/{v[0]}' for k, v in src.items()}
print(json.dumps(out, indent=1, ensure_ascii=False))
