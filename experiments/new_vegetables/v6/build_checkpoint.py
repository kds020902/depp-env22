# Put the adopted V6-H head into model.pth as "new_vegetable_head" (the vegetable model and the five-vegetable CLIP head are
# unchanged). The display temperature (probabilities shown for switched photos only; decisions do not use it) is fitted on
# Valid: the 300 old Valid photos and the 73 new-vegetable Valid photos.
import sys, json, pickle, numpy as np, torch
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; V = S + 'v6/'; REPO = '/home/user/depp-env22/'
sys.path.insert(0, REPO + 'src')
from model_utils import CLASS_NAMES, fit_temperature
hv = pickle.load(open(V + 'head_v6_noaug.pkl', 'rb')); fin = json.load(open(V + 'final_once.json')); chk = json.load(open(V + 'check_noaug.json'))
assert fin['adopt_V6-H'] and hv['tau'] == 0.7
F = dict(np.load(S + 'clip/feat_datacomp_xl_s13b_b90k.npz')); Y = json.load(open(S + 'post/labels.json'))
pool = json.load(open(V + 'pool.json')); pf = np.load(V + 'pool_feats.npz')['f']; va = [i for i, r in enumerate(pool) if r['split'] == 'val']
X = np.concatenate([F['valid'], pf[va]]); y = np.array([CLASS_NAMES.index(c) for c in Y['valid']] + [CLASS_NAMES.index(pool[i]['class_name']) for i in va])
z = X @ hv['coef'].T + hv['intercept']
T = fit_temperature(torch.tensor(z[:, :16], dtype=torch.float32), torch.tensor(y))
ck = torch.load(REPO + 'model.pth', map_location='cpu', weights_only=True)
tr = {c: sum(1 for r in pool if r['split'] == 'train' and r['class_name'] == c) for c in CLASS_NAMES[10:]}
ck['new_vegetable_head'] = {
    'clip': dict(ck['clip_head']['clip']), 'class_names': list(CLASS_NAMES), 'first_new_species': 5,
    'coef': torch.tensor(hv['coef'], dtype=torch.float32), 'intercept': torch.tensor(hv['intercept'], dtype=torch.float32),
    'other_name': 'not one of the eight vegetables', 'unknown_threshold': float(hv['EC']), 'switch_probability': float(hv['tau']),
    'temperature': float(T), 'temperature_fitted_on': 'val (300 old + 73 new-vegetable photos), display only',
    'untrained_classes': ['greenonion_rotten'], 'no_condition_species': ['greenonion'],
    'head': f"multinomial logistic regression, C=10, 17 outputs: the five-vegetable head's photos + new-vegetable Train {tr} "
            "(experiments/new_vegetables v6); greenonion_rotten has no photos (zero row, intercept min - 50)",
    'unknown_fitted_on': '95% of Valid (300 old + 73 new) + 192 real-scene vegetable crops pass',
    'switch_rule': 'tau: smallest of 0.5/0.7/0.8/0.9/0.95 moving at most 1 of 910 old-vegetable development photos the five-vegetable system answered',
    'notes': ('experiments/new_vegetables (V6-H). Test 300: {}/{}; new internet test 241: {}/{}; new-vegetable tests 1+2+3: {}/{} kind right and kept.'
              .format(fin['test_300']['V6-H']['accuracy'], fin['test_300']['V6-H']['kept_correct'], fin['new_test_241']['V6-H']['accuracy'],
                      fin['new_test_241']['V6-H']['kept_correct'], fin['new_veg_total']['V6-H'], fin['new_veg_total']['of'])),
}
torch.save(ck, REPO + 'model.pth')
print('temperature', T, 'train', tr)
