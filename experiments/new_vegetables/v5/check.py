# Development check record for V5-H (tau 0.7) and R5 (v5/PREREG.md). No sealed set is touched.
import sys, json, pickle, collections, numpy as np, torch
from scipy.special import log_softmax, logsumexp, softmax
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; V = S + 'v5/'; REPO = '/home/user/depp-env22/'
XSRC = open(V + 'explore.py').read()
exec(XSRC[:XSRC.index('lr = LogisticRegression')])
hd = pickle.load(open(V + 'head_v5.pkl', 'rb')); coef, icp, EC = hd['coef'], hd['intercept'], hd['EC']
exec(XSRC[XSRC.index('hz = lambda'):XSRC.index('res = {')].replace("EC = float(np.quantile(np.concatenate([nfo(hz(D['valid'][1])), nfo(hz(D['newval'][1])), nfo(hz(D['lvis'][1]))]), 0.95))", ''))
TAU = 0.7
def dec(name, b, f):
    p0, r0 = now(b, f)
    if name == 'now': return p0, r0
    sp, pn, rn = newhead(f)
    if name == 'R5': return np.where(sp < 5, p0, pn), rn
    q = softmax(hz(f)[:, :K], 1)[:, 10:].sum(1)
    take = (r0 & ~rn & (sp >= 5)) | (~r0 & (q >= TAU) & (sp >= 5))
    return np.where(take, pn, p0), r0 & ~take
res = {'tau': TAU, 'EC_new': EC}
for name in ('now', 'V5-H', 'R5'):
    out = {}
    for k, (b, f, yy) in D.items():
        p, rej = dec(name, b, f)
        if yy is None: out[k] = {'rejected': int(rej.sum()), 'of': len(rej), 'answered_new_species': int(((p >= 10) & ~rej).sum())}; continue
        right = p == yy
        if k == 'newval':
            out[k] = {'kind_right_kept': int(((p // 2 == yy // 2) & ~rej).sum()), 'of': len(yy),
                      'misses': collections.Counter(f'{CLASS_NAMES[a]}->{"REJ" if j else CLASS_NAMES[q]}' for a, q, j in zip(yy, p, rej) if not (q // 2 == a // 2 and not j)).most_common()}
        else: out[k] = {'accuracy': int(right.sum()), 'kept_correct': int((right & ~rej).sum()), 'rejected': int(rej.sum()), 'answered_new_species': int(((p >= 10) & ~rej).sum()), 'of': len(yy)}
    res[name] = out
h = res['V5-H']
res['V5-H_checks_pass'] = bool(h['valid']['accuracy'] >= 288 and h['valid']['kept_correct'] >= 284 and h['o160']['accuracy'] >= 147 and h['odev']['accuracy'] >= 195
                               and h['odev']['kept_correct'] >= 186 and h['asian37']['accuracy'] >= 31 and h['fruit374']['rejected'] >= 320 and h['coco400']['rejected'] >= 395)
json.dump(res, open(V + 'check_results.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: res[k] for k in ('V5-H_checks_pass',)}), json.dumps(res['V5-H'], ensure_ascii=False))
