# Pre-registered (negx/PREREG.md): CLIP head with an 11th "not one of the five vegetables" class.
# The CLIP-side rejection score becomes logit(other) - logsumexp(vegetable logits); its threshold keeps the same rule (95% of Valid + 192 real-scene vegetables pass).
# Classification still uses only the 10 vegetable logits, combined with the vegetable model (B) as now. Development checks only.
import sys, csv, json, pickle, collections, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from scipy.special import log_softmax, logsumexp
sys.path.insert(0, '/home/user/depp-env22/src')
from model_utils import fit_temperature
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; E = S + 'ext3/'; REPO = '/home/user/depp-env22/'
NAMES = ['cucumber_fresh','cucumber_rotten','potato_fresh','potato_rotten','tomato_fresh','tomato_rotten','bellpepper_fresh','bellpepper_rotten','carrot_fresh','carrot_rotten']
OTHER = 10
sets = json.load(open(S + 'clip/sets.json')); F = dict(np.load(S + 'clip/feat_datacomp_xl_s13b_b90k.npz'))
feat_of = {p: F['probe_train'][i] for i, (p, _) in enumerate(sets['probe_train'])}
rows = list(csv.DictReader(open(REPO + 'dataset/manifest.csv', encoding='utf-8')))
lab = {REPO + 'dataset/' + r['image_path']: r['class_name'] for r in rows if r['split'] == 'train'}
idx = [i for i, (p, _) in enumerate(sets['probe_train']) if p in lab]
Xt = F['probe_train'][idx]; yt = np.array([NAMES.index(lab[sets['probe_train'][i][0]]) for i in idx]); assert len(yt) == 1260

# Earlier reviewed photos (1,168) and weak-spot photos (434), exactly as in the W2 candidate
main = {r['image_path'] for r in rows}
pool = []
for r in csv.DictReader(open(S + 'aug2/ds6/manifest_notest.csv', encoding='utf-8')):
    if r['split'] == 'train' and r['image_path'] not in main: pool.append((S + 'aug2/ds6/' + r['image_path'], r['class_name']))
for r in csv.DictReader(open(S + 'fix/dsK/manifest_notest.csv', encoding='utf-8')):
    if r['split'] == 'train' and int(r['image_id'].rsplit('_', 1)[1]) >= 5000 and r['source_dataset'] != 'internet':
        pool.append((S + 'fix/dsK/' + r['image_path'], r['class_name']))
Xp = np.stack([feat_of[p] for p, _ in pool]); yp = np.array([NAMES.index(c) for _, c in pool])
jf = np.load(E + 'jimpax_features.npz'); jidx = {str(n): i for i, n in enumerate(jf['names'])}; rev = json.load(open(E + 'jimpax_reviewed.json'))
wk = [(jf['f'][jidx[p.split('/')[-1]]], c) for c in ('tomato_rotten', 'tomato_fresh') for p in rev[c]]
of = np.load(E + 'features.npz'); oidx = {str(n): i for i, n in enumerate(of['ids'])}
wk += [(of['f'][oidx[i]], c) for i, c in json.load(open(E + 'train_labels.json')).items() if c]
# Photos from the user's query pattern ("rotten/fresh asian <vegetable>" ...), reviewed by eye
f4 = np.load(S + 'ext4/features.npz'); i4 = {str(n): i for i, n in enumerate(f4['ids'])}
wk += [(f4['f'][i4[i]], c) for i, c in json.load(open(S + 'ext4/labels.json')).items() if c]
Xw = np.stack([f for f, _ in wk]); yw = np.array([NAMES.index(c) for _, c in wk])
# N3 only: 140 East Asian fresh cucumber crops (experiments/asian_cucumber J2, Zenodo 10081197)
Xa = np.stack([feat_of[p] for p, _ in sets['probe_train'] if '/asia/zen/' in p]); assert len(Xa) == 140

# "Not one of the five" photos: fruit, everyday objects and scenes, other vegetables
Xn = np.concatenate([np.load(S + f'neg/{k}_feats.npz')['f'] for k in ('fruit', 'everyday', 'otherveg')])

# No copies of any check photo (Valid, Test, internet 221, 160 exam, Asian cucumbers, internet fruit 374, everyday 400, new sealed test set)
T = np.load(S + 'clipcls/test_features.npz')['clip']
t2 = {r['id'] for r in csv.DictReader(open(E + 'outside_test2/outside_test2.csv', encoding='utf-8'))}
inf = np.load(E + 'inat/features.npz')
new_test = np.concatenate([of['f'][[oidx[i] for i in t2 if not i.startswith('inat-')]], inf['f'][[i for i, p in enumerate(inf['ids']) if 'inat-' + str(p) in t2]]]); assert len(new_test) == len(t2)
checks = np.concatenate([F['valid'], T, F['odev'], F['o160'], F['asian37'], F['fruit374'], F['coco400'], new_test])
drop = lambda X: (X @ checks.T).max(1) >= 0.95
dp, dw, dn, da = drop(Xp), drop(Xw), drop(Xn), drop(Xa)
print('copies of check photos: earlier', int(dp.sum()), 'weak', int(dw.sum()), 'not-five', int(dn.sum()), 'asian', int(da.sum()))
Xp, yp, Xw, yw, Xn, Xa = Xp[~dp], yp[~dp], Xw[~dw], yw[~dw], Xn[~dn], Xa[~da]
print('earlier', len(yp), 'weak', len(yw), collections.Counter(NAMES[c] for c in yw), 'not-five', len(Xn))

ck = torch.load(REPO + 'model.pth', map_location='cpu', weights_only=True)
TB, EB = ck['calibration']['temperature'], ck['unknown_check']['threshold']
C = np.load(S + 'post/cache.npz'); Y = json.load(open(S + 'post/labels.json'))
key = {'valid': ('valid', 'valid'), 'lvis': ('lvis192', 'lvis_veg'), 'odev': ('odev', 'odev'), 'o160': ('o160', 'o160'),
       'asian': ('asian37', 'asian37'), 'fruit': ('fruit374', 'fruit374'), 'coco': ('coco400', 'coco400')}
energy = lambda z: -logsumexp(z, 1)
other_score = lambda z: z[:, OTHER] - logsumexp(z[:, :OTHER], 1)
yv = np.array([NAMES.index(c) for c in Y['valid']])

def run(name, X, y):
    lr = LogisticRegression(C=10, max_iter=5000).fit(X, y)
    cl = lambda k: lr.decision_function(F[key[k][1]])
    EC = float(np.quantile(np.concatenate([other_score(cl('valid')), other_score(cl('lvis'))]), 0.95))
    TC = fit_temperature(torch.tensor(log_softmax(C['B/valid/logit'] / TB, 1) + log_softmax(cl('valid')[:, :OTHER], 1)), torch.tensor(yv))
    out = {'train_photos': {'vegetable': int((y < OTHER).sum()), 'not_five': int((y == OTHER).sum())}, 'clip_threshold': EC, 'combined_temperature': TC}
    for k, (bk, _) in key.items():
        bl = C[f'B/{bk}/logit']; c = cl(k); comb = log_softmax(bl / TB, 1) + log_softmax(c[:, :OTHER], 1)
        rej = (energy(bl) > EB) | (other_score(c) > EC)
        if k in ('fruit', 'coco', 'lvis'): out[k] = {'rejected': int(rej.sum())}
        else:
            yy = np.array([NAMES.index(v) for v in Y[bk]]); right = comb.argmax(1) == yy
            out[k] = {'accuracy': int(right.sum()), 'kept_correct': int((right & ~rej).sum()), 'rejected': int(rej.sum())}
            if k == 'odev': out['odev_by_class'] = {n: f'{int(right[yy == i].sum())}/{int((yy == i).sum())}' for i, n in enumerate(NAMES) if (yy == i).any()}
    out['checks_pass'] = (out['valid']['kept_correct'] >= 283 and out['o160']['kept_correct'] >= 143 and out['asian']['kept_correct'] >= 29
                          and out['fruit']['rejected'] >= 234 and out['coco']['rejected'] >= 379 and out['odev']['accuracy'] > 190)
    pickle.dump({'coef': lr.coef_, 'intercept': lr.intercept_, 'EC': EC, 'TC': TC}, open(S + f'negx/head_{name}.pkl', 'wb'))
    return out

yN = np.full(len(Xn), OTHER)
import sys as _sys
if 'N3' in _sys.argv:
    res = json.load(open(S + 'negx/check_results_N1_N2.json'))
    ya = np.full(len(Xa), NAMES.index('cucumber_fresh'))
    res['N3'] = run('N3', np.concatenate([Xt, Xp, Xw, Xa, Xn]), np.concatenate([yt, yp, yw, ya, yN]))
    res['chosen'] = 'N3' if res['N3']['checks_pass'] else None
else:
    res = {'N1': run('N1', np.concatenate([Xt, Xn]), np.concatenate([yt, yN])),
           'N2': run('N2', np.concatenate([Xt, Xp, Xw, Xn]), np.concatenate([yt, yp, yw, yN]))}
    ok = [k for k in ('N1', 'N2') if res[k]['checks_pass']]
    res['chosen'] = max(ok, key=lambda k: (res[k]['odev']['accuracy'], res[k]['odev']['kept_correct'], k == 'N1')) if ok else None
json.dump(res, open(S + 'negx/check_results.json', 'w'), indent=1); print(json.dumps(res.get('N3', res), indent=1))
