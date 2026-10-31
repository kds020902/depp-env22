# Diagnostic only (development photos): accuracy of each check set as the vegetable model's weight w changes. Rejection unchanged (R1).
import json, pickle, numpy as np
from scipy.special import log_softmax, logsumexp
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'
NAMES = ['cucumber_fresh','cucumber_rotten','potato_fresh','potato_rotten','tomato_fresh','tomato_rotten','bellpepper_fresh','bellpepper_rotten','carrot_fresh','carrot_rotten']
C = np.load(S + 'post/cache.npz'); Y = json.load(open(S + 'post/labels.json')); F = np.load(S + 'clip/feat_datacomp_xl_s13b_b90k.npz')
h = pickle.load(open(S + 'negx/head_N3.pkl', 'rb')); TB = 0.5283377766609192
r = json.load(open(S + 'tom/check_results.json'))
print('fusion-set NLL by w:', {w: round(v[1], 3) for w, v in r['nll_grid'].items() if float(w) in (0, .05, .1, .2, .3, .5, .7, 1, 1.5)})
rows = {}
for name in ('valid', 'odev', 'o160', 'asian37'):
    y = np.array([NAMES.index(v) for v in Y[name]]); bl = log_softmax(C[f'B/{name}/logit'] / TB, 1); z = F[name] @ h['coef'].T + h['intercept']
    lc = log_softmax(z[:, :10], 1); rej = z[:, 10] - logsumexp(z[:, :10], 1) > h['EC']
    for w in (0, .05, .1, .2, .3, .4, .5, .6, .7, .8, 1.0):
        right = (w * bl + lc).argmax(1) == y
        rows.setdefault(w, {})[name] = (int(right.sum()), int((right & ~rej).sum()), int(right[y == 5].sum()), int(right[y == 4].sum()))
print('w     valid   odev(rotT/freshT)   o160(rotT/freshT)   asian')
for w, d in rows.items():
    print(f"{w:<5} {d['valid'][0]:>4}   {d['odev'][0]:>4} ({d['odev'][2]}/{d['odev'][3]})       {d['o160'][0]:>4} ({d['o160'][2]}/{d['o160'][3]})        {d['asian37'][0]}")
