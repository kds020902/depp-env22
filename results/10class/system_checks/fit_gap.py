# Over/under-fitting check: accuracy of each part on its own training photos vs. Valid, Test and outside photos,
# and how the CLIP head behaves with weaker/stronger regularisation (C). Analysis only; the adopted model is not changed.
import sys, json, pickle, csv, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from scipy.special import log_softmax, logsumexp
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'
src = open(S + 'negx/evaluate.py').read()
exec(src[:src.index("def run(")])
Xa_ = Xa
X = np.concatenate([Xt, Xp, Xw, Xa_, Xn]); y = np.concatenate([yt, yp, yw, np.full(len(Xa_), 0), np.full(len(Xn), OTHER)])
h = pickle.load(open(S + 'negx/head_N3.pkl', 'rb'))
T = np.load(S + 'clipcls/test_features.npz')
rows = [r for r in csv.DictReader(open(REPO + 'dataset/manifest.csv', encoding='utf-8')) if r['split'] == 'test']; yT = np.array([NAMES.index(r['class_name']) for r in rows])
ytr = np.array([NAMES.index(lab[sets['probe_train'][i][0]]) for i in idx])
def system(bl, f, yy, coef, inter):
    z = f @ coef.T + inter
    comb = log_softmax(bl / TB, 1) + log_softmax(z[:, :OTHER], 1)
    return round(100 * float((comb.argmax(1) == yy).mean()), 1)
sets_eval = {'Train 1,260': (C['B/train/logit'], Xt, ytr), 'Valid 300': (C['B/valid/logit'], F['valid'], yv), 'Test 300': (T['b_logits'], T['clip'], yT),
             'internet 221': (C['B/odev/logit'], F['odev'], np.array([NAMES.index(v) for v in Y['odev']])),
             '160 exam': (C['B/o160/logit'], F['o160'], np.array([NAMES.index(v) for v in Y['o160']]))}
out = {'vegetable_model_B': {k: round(100 * float((bl.argmax(1) == yy).mean()), 1) for k, (bl, f, yy) in sets_eval.items()},
       'clip_head_N3_alone': {k: round(100 * float(((f @ h['coef'].T + h['intercept'])[:, :OTHER].argmax(1) == yy).mean()), 1) for k, (bl, f, yy) in sets_eval.items()},
       'combined_system': {k: system(bl, f, yy, h['coef'], h['intercept']) for k, (bl, f, yy) in sets_eval.items()}}
zt = X @ h['coef'].T + h['intercept']; out['clip_head_N3_own_training_photos_11_classes'] = round(100 * float((zt.argmax(1) == y).mean()), 1)
sweep = {}
for c in (0.01, 0.1, 1, 10, 100):
    lr = LogisticRegression(C=c, max_iter=5000).fit(X, y)
    zv = lr.decision_function(F['valid'])[:, :OTHER]; zo = lr.decision_function(F['odev'])[:, :OTHER]; yo = np.array([NAMES.index(v) for v in Y['odev']])
    sweep[f'C={c}'] = {'train_acc_11': round(100 * lr.score(X, y), 1), 'valid_acc_head': round(100 * float((zv.argmax(1) == yv).mean()), 1),
                       'valid_nll_head': round(float(-(log_softmax(zv, 1)[np.arange(len(yv)), yv]).mean()), 3),
                       'internet221_acc_head': round(100 * float((zo.argmax(1) == yo).mean()), 1)}
    print(c, sweep[f'C={c}'], flush=True)
out['clip_head_C_sweep'] = sweep
json.dump(out, open(S + 'checks/fit_gap.json', 'w'), indent=1); print(json.dumps({k: v for k, v in out.items() if k != 'clip_head_C_sweep'}, indent=1))
