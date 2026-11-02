# Development only: where do abnormal sweet potato (and green onion) Train photos go in grouped CV (variant zs0.25 and base)?
import os
import sys, collections, numpy as np
XS = open(os.environ["DEPP_WORK"] + "/v7/explore.py").read()
exec(XS[:XS.index("VARIANTS = {")])
groups = [pool[i]["group"] for i in tr]; fold = np.array([int(hashlib.sha1(("cv:" + g).encode()).hexdigest()[:8], 16) % 5 for g in groups])
SPN = ['cucumber','potato','tomato','bellpepper','carrot','onion','sweetpotato','greenonion']
for name, a, tau in (('base', 0, 0.7), ('zs0.25', 0.25, 0.8)):
    Xall = np.concatenate([X0, pf[tr]]); yall = np.concatenate([y0, ytr]); coef, icp = fit(Xall, yall); EC, _, _ = calibrate(coef, icp, a)
    rows = collections.Counter(); per = collections.defaultdict(collections.Counter)
    for k in range(5):
        m = fold != k; ck_, ik_ = fit(np.concatenate([X0, pf[tr][m]]), np.concatenate([y0, ytr[m]]))
        idx = np.where(~m)[0]; ii = np.array(tr)[idx]; b, f = bpool[ii], pf[ii]
        p0, r0 = now(b, f); z = head_logits(f, ck_, ik_, a); n = len(f)
        spm = np.exp(log_softmax(z[:, :K], 1)).reshape(n, 8, 2).sum(2); sp = spm.argmax(1); rn = z[:, K] - logsumexp(z[:, :K], 1) > EC
        q = softmax(z[:, :K], 1)[:, 10:].sum(1); p, rej = decide(b, f, ck_, ik_, a, EC, tau)
        for j, t in enumerate(ytr[idx]):
            if t not in (13, 14): continue
            route = ('five REJ' if r0[j] else 'five ' + CLASS_NAMES[p0[j]]) + ' | head ' + ('NOT8' if rn[j] else SPN[sp[j]]) + f' q{"≥" if q[j] >= tau else "<"}tau'
            final = 'REJ' if rej[j] else CLASS_NAMES[p[j]]
            per[CLASS_NAMES[t]][(route, final)] += 1
    print('==', name)
    for c, cnt in per.items():
        print(' ', c)
        for (route, final), v in cnt.most_common(): print('    ', v, route, '->', final)
