# Pre-registered (weak/PREREG.md): CLIP head retrained with the weak-spot photos (W1) or with them plus the earlier reviewed photos (W2).
# Development checks only; the new sealed test set and Test are scored later, once, by final.py.
import sys, csv, json, pickle, collections, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from scipy.special import log_softmax, logsumexp
sys.path.insert(0, '/home/user/depp-env22/src')
from model_utils import fit_temperature
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; E = S + 'ext3/'; REPO = '/home/user/depp-env22/'
NAMES = ['cucumber_fresh','cucumber_rotten','potato_fresh','potato_rotten','tomato_fresh','tomato_rotten','bellpepper_fresh','bellpepper_rotten','carrot_fresh','carrot_rotten']
sets = json.load(open(S + 'clip/sets.json')); F = dict(np.load(S + 'clip/feat_datacomp_xl_s13b_b90k.npz'))
feat_of = {p: F['probe_train'][i] for i, (p, _) in enumerate(sets['probe_train'])}

# Train 1,260 (same as the current head)
rows = list(csv.DictReader(open(REPO + 'dataset/manifest.csv', encoding='utf-8')))
lab = {REPO + 'dataset/' + r['image_path']: r['class_name'] for r in rows if r['split'] == 'train'}
idx = [i for i, (p, _) in enumerate(sets['probe_train']) if p in lab]
Xt = F['probe_train'][idx]; yt = np.array([NAMES.index(lab[sets['probe_train'][i][0]]) for i in idx]); assert len(yt) == 1260

# Weak-spot photos (434): jimpax reviewed + Openverse train side reviewed
jf = np.load(E + 'jimpax_features.npz'); jidx = {str(n): i for i, n in enumerate(jf['names'])}
rev = json.load(open(E + 'jimpax_reviewed.json'))
weak = [(jf['f'][jidx[p.split('/')[-1]]], c, 'jimpax') for c in ('tomato_rotten', 'tomato_fresh') for p in rev[c]]
of = np.load(E + 'features.npz'); oidx = {str(n): i for i, n in enumerate(of['ids'])}
for i, c in json.load(open(E + 'train_labels.json')).items():
    if c: weak.append((of['f'][oidx[i]], c, 'openverse'))
print('weak-spot photos', len(weak), collections.Counter((s, c) for _, c, s in weak))

# Earlier reviewed photos (1,168), exactly as in experiments/clip_head_more_data
main = {r['image_path'] for r in rows}
pool = []
for r in csv.DictReader(open(S + 'aug2/ds6/manifest_notest.csv', encoding='utf-8')):
    if r['split'] == 'train' and r['image_path'] not in main: pool.append((S + 'aug2/ds6/' + r['image_path'], r['class_name']))
for r in csv.DictReader(open(S + 'fix/dsK/manifest_notest.csv', encoding='utf-8')):
    if r['split'] == 'train' and int(r['image_id'].rsplit('_', 1)[1]) >= 5000 and r['source_dataset'] != 'internet':
        pool.append((S + 'fix/dsK/' + r['image_path'], r['class_name']))
Xp = np.stack([feat_of[p] for p, _ in pool]); yp = np.array([NAMES.index(c) for _, c in pool])

# No copies of any check photo (Valid, Test, internet 221, 160 exam, Asian cucumbers, new sealed test set)
T = np.load(S + 'clipcls/test_features.npz')['clip']
t2 = {r['id'] for r in csv.DictReader(open(E + 'outside_test2/outside_test2.csv', encoding='utf-8'))}
inf = np.load(E + 'inat/features.npz')
new_test = np.concatenate([of['f'][[oidx[i] for i in t2 if not i.startswith('inat-')]], inf['f'][[i for i, p in enumerate(inf['ids']) if 'inat-' + str(p) in t2]]])
assert len(new_test) == len(t2)
checks = np.concatenate([F['valid'], T, F['odev'], F['o160'], F['asian37'], new_test])
Xw = np.stack([f for f, _, _ in weak]); yw = np.array([NAMES.index(c) for _, c, _ in weak])
near_w = (Xw @ checks.T).max(1) >= 0.95; near_p = (Xp @ checks.T).max(1) >= 0.95
print('copies of check photos: weak', int(near_w.sum()), 'earlier', int(near_p.sum()))
Xw, yw, Xp, yp = Xw[~near_w], yw[~near_w], Xp[~near_p], yp[~near_p]

ck = torch.load(REPO + 'model.pth', map_location='cpu', weights_only=True)
TB, EB = ck['calibration']['temperature'], ck['unknown_check']['threshold']
C = np.load(S + 'post/cache.npz'); Y = json.load(open(S + 'post/labels.json'))
key = {'valid': ('valid', 'valid'), 'lvis': ('lvis192', 'lvis_veg'), 'odev': ('odev', 'odev'), 'o160': ('o160', 'o160'),
       'asian': ('asian37', 'asian37'), 'fruit': ('fruit374', 'fruit374'), 'coco': ('coco400', 'coco400')}
energy = lambda z: -logsumexp(z, 1)
yv = np.array([NAMES.index(c) for c in Y['valid']])

def run(name, X, y):
    lr = LogisticRegression(C=10, max_iter=5000).fit(X, y)
    cl = lambda k: lr.decision_function(F[key[k][1]])
    EC = float(np.quantile(np.concatenate([energy(cl('valid')), energy(cl('lvis'))]), 0.95))
    TC = fit_temperature(torch.tensor(log_softmax(C['B/valid/logit'] / TB, 1) + log_softmax(cl('valid'), 1)), torch.tensor(yv))
    out = {'train_photos': int(len(y)), 'clip_threshold': EC, 'combined_temperature': TC}
    for k, (bk, _) in key.items():
        bl = C[f'B/{bk}/logit']; c = cl(k); comb = log_softmax(bl / TB, 1) + log_softmax(c, 1)
        rej = (energy(bl) > EB) | (energy(c) > EC)
        if k in ('fruit', 'coco', 'lvis'): out[k] = {'rejected': int(rej.sum())}
        else:
            yy = np.array([NAMES.index(v) for v in Y[bk]]); right = comb.argmax(1) == yy
            out[k] = {'accuracy': int(right.sum()), 'kept_correct': int((right & ~rej).sum()), 'rejected': int(rej.sum())}
            if k == 'odev': out['odev_by_class'] = {n: f'{int(right[yy == i].sum())}/{int((yy == i).sum())}' for i, n in enumerate(NAMES) if (yy == i).any()}
    out['checks_pass'] = (out['valid']['kept_correct'] >= 283 and out['o160']['kept_correct'] >= 143 and out['asian']['kept_correct'] >= 29
                          and out['fruit']['rejected'] >= 234 and out['coco']['rejected'] >= 379 and out['odev']['accuracy'] > 190)
    pickle.dump({'coef': lr.coef_, 'intercept': lr.intercept_, 'EC': EC, 'TC': TC, 'n': int(len(y))}, open(S + f'weak/head_{name}.pkl', 'wb'))
    return out

res = {'W1': run('W1', np.concatenate([Xt, Xw]), np.concatenate([yt, yw])),
       'W2': run('W2', np.concatenate([Xt, Xp, Xw]), np.concatenate([yt, yp, yw]))}
ok = [k for k in ('W1', 'W2') if res[k]['checks_pass']]
res['chosen'] = max(ok, key=lambda k: (res[k]['odev']['accuracy'], res[k]['odev']['kept_correct'], k == 'W1')) if ok else None
json.dump(res, open(S + 'weak/check_results.json', 'w'), indent=1); print(json.dumps(res, indent=1))
