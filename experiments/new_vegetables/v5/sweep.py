# Development only. Hybrid H(tau): the current system first; a photo goes to the new head's answer when
#   (a) the current system turns it away and the new head says onion / sweet potato / green onion (as cascade C), or
#   (b) the current system answers but the new head gives the three new kinds together probability >= tau.
import sys, json, pickle, numpy as np, torch
from scipy.special import log_softmax, logsumexp, softmax
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; V = S + 'v5/'; REPO = '/home/user/depp-env22/'
XSRC = open(V + 'explore.py').read()
# reuse explore.py's data loading but not its head training: load the saved head
pre = XSRC[:XSRC.index('lr = LogisticRegression')]
exec(pre)
hd = pickle.load(open(V + 'head_v5.pkl', 'rb')); coef, icp, EC = hd['coef'], hd['intercept'], hd['EC']
post = XSRC[XSRC.index('hz = lambda'):XSRC.index('res = {')]
post = post.replace("EC = float(np.quantile(np.concatenate([nfo(hz(D['valid'][1])), nfo(hz(D['newval'][1])), nfo(hz(D['lvis'][1]))]), 0.95))", '')
exec(post)
def H(b, f, tau):
    p0, r0 = now(b, f); sp, pn, rn = newhead(f)
    q = softmax(hz(f)[:, :K], 1)[:, 10:].sum(1)
    take = (r0 & ~rn & (sp >= 5)) | (~r0 & (q >= tau) & (sp >= 5))
    return np.where(take, pn, p0), r0 & ~take
for tau in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.01):
    line = [f'tau {tau}:']
    for k, (b, f, yy) in D.items():
        p, rej = H(b, f, tau)
        if yy is None: line.append(f"{k} rej {int(rej.sum())} new {int(((p >= 10) & ~rej).sum())}"); continue
        if k == 'newval': line.append(f"newval {int(((p // 2 == yy // 2) & ~rej).sum())}/58"); continue
        right = p == yy; line.append(f"{k} {int(right.sum())}/{int((right & ~rej).sum())} new {int(((p >= 10) & ~rej).sum())}")
    print(' | '.join(line))
