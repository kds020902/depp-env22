# Pre-registered (v5/PREREG.md): V5-H (adoptable) and R5 (record only), scored once together with the current system on
# Test 300, the new internet test 241, the any-photo set v2, new-vegetable test 1 (97) and new-vegetable test 2 (54).
import sys, csv, json, pickle, collections, numpy as np, torch, open_clip
from scipy.special import log_softmax, logsumexp, softmax
sys.path.insert(0, '/home/user/depp-env22/src')
from model_utils import load_model, make_transform, open_image, input_size, CLASS_NAMES
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; V = S + 'v5/'; REPO = '/home/user/depp-env22/'
assert json.load(open(V + 'check_results.json'))['V5-H_checks_pass']
REJECT = ('rotten_fruit', 'fresh_fruit', 'other_veg', 'food', 'drawing_toy', 'everyday'); K = 16; TAU = 0.7
torch.set_num_threads(4)
hv = pickle.load(open(V + 'head_v5.pkl', 'rb'))
B0, ck0 = load_model(REPO + 'model.pth', torch.device('cpu')); tf0 = make_transform(input_size(ck0)); h0 = ck0['clip_head']; TB0 = ck0['calibration']['temperature']
clip, _, prep = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k'); clip.eval()
def run(paths):
    b0, cf = [], []
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            ims = [open_image(p) for p in paths[i:i + 32]]
            b0.append(B0(torch.stack([tf0(im) for im in ims]))); cf.append(torch.nn.functional.normalize(clip.encode_image(torch.stack([prep(im) for im in ims])), dim=-1))
    return torch.cat(b0).numpy(), torch.cat(cf).numpy()
def decide(name, b0, f):
    z0 = f @ h0['coef'].numpy().T + h0['intercept'].numpy()
    p0 = (log_softmax(b0 / TB0, 1) + log_softmax(z0[:, :10], 1)).argmax(1); r0 = z0[:, 10] - logsumexp(z0[:, :10], 1) > h0['unknown_threshold']
    if name == 'now': return p0, r0
    z = f @ hv['coef'].T + hv['intercept']; n = len(f)
    sp = np.exp(log_softmax(z[:, :K], 1)).reshape(n, 8, 2).sum(2).argmax(1)
    pn = sp * 2 + z[:, :K].reshape(n, 8, 2)[np.arange(n), sp].argmax(1); rn = z[:, K] - logsumexp(z[:, :K], 1) > hv['EC']
    if name == 'R5': return np.where(sp < 5, p0, pn), rn
    q = softmax(z[:, :K], 1)[:, 10:].sum(1)
    take = (r0 & ~rn & (sp >= 5)) | (~r0 & (q >= TAU) & (sp >= 5))
    return np.where(take, pn, p0), r0 & ~take
NAMES = ('now', 'V5-H', 'R5')
def veg_scores(b0, f, y):
    out = {}
    for name in NAMES:
        p, rej = decide(name, b0, f); right = p == y
        out[name] = {'accuracy': int(right.sum()), 'kept_correct': int((right & ~rej).sum()), 'rejected': int(rej.sum()), 'kept_wrong': int((~right & ~rej).sum()),
                     'answered_new_species': int(((p >= 10) & ~rej).sum()),
                     'new_species_answers': collections.Counter(f'{CLASS_NAMES[a]}->{CLASS_NAMES[q]}' for a, q, j in zip(y, p, rej) if q >= 10 and not j).most_common(),
                     'by_class': {n: f'{int(right[y == i].sum())}/{int((y == i).sum())}' for i, n in enumerate(CLASS_NAMES) if (y == i).any()}}
    return out
res = {}
test = [r for r in csv.DictReader(open(REPO + 'dataset/manifest.csv', encoding='utf-8')) if r['split'] == 'test']
T = np.load(S + 'clipcls/test_features.npz')
res['test_300'] = veg_scores(T['b_logits'], T['clip'], np.array([CLASS_NAMES.index(r['class_name']) for r in test]))
assert (res['test_300']['now']['accuracy'], res['test_300']['now']['kept_correct']) == (288, 285), res['test_300']['now']
t2 = list(csv.DictReader(open(REPO + 'results/10class/outside_test2/outside_test2.csv', encoding='utf-8')))
res['new_test_241'] = veg_scores(*run([S + f"weak/t2img/{r['id']}.jpg" for r in t2]), np.array([CLASS_NAMES.index(r['class_name']) for r in t2]))
assert (res['new_test_241']['now']['accuracy'], res['new_test_241']['now']['kept_correct']) == (203, 193), res['new_test_241']['now']
rows = list(csv.DictReader(open(S + 'arb/anyphoto_test_v2.csv', encoding='utf-8'))); files = json.load(open(S + 'arb/local_files.json'))
b0, f = run([files[r['id']] for r in rows]); res['any_photo_v2'] = {}
for name in NAMES:
    p, rej = decide(name, b0, f); good, cats = [], []
    for r, pp, j in zip(rows, p, rej):
        c = r['category']; lab = r['label'].split('|') if r['label'] else []; pred = CLASS_NAMES[pp]
        if c == 'excluded_v2': continue
        if c in REJECT: g = bool(j)
        elif c == 'five_daily': g = (not j) and pred == lab[0]
        elif c == 'new_veg': g = (not j) and (pred == lab[0] or (lab[0].startswith('greenonion') and pred.startswith('greenonion')))
        elif c == 'cut_five': g = bool(j) or pred == lab[0]
        else: g = bool(j) or pred in lab or (pred.startswith('greenonion') and any(l.startswith('greenonion') for l in lab))
        good.append(g); cats.append(c)
    good = np.array(good); cats = np.array(cats); old = cats != 'new_veg'
    res['any_photo_v2'][name] = {'good_old_categories': int(good[old].sum()), 'of_old': int(old.sum()), 'good_new_veg': int(good[~old].sum()), 'of_new_veg': int((~old).sum()),
                                 'by_category': {c: f'{int(good[cats == c].sum())}/{int((cats == c).sum())}' for c in dict.fromkeys(cats)}}
assert res['any_photo_v2']['now']['good_old_categories'] == 205, res['any_photo_v2']['now']
pool = json.load(open(V + 'pool.json')); NEW = ('onion', 'sweetpotato', 'greenonion')
for key, rnd in (('new_veg_test1_97', 'v4'), ('new_veg_test2_54', 'v5')):
    te = [r for r in pool if r['round'] == rnd and r['split'] == 'test' and r['class_name'].startswith(NEW)]
    b0, f = run([r['path'] for r in te]); y = np.array([CLASS_NAMES.index(r['class_name']) for r in te]); res[key] = {'n': len(y)}
    for name in ('V5-H', 'R5'):
        p, rej = decide(name, b0, f); kind = (p // 2 == y // 2) & ~rej
        res[key][name] = {'kind_right_and_kept': int(kind.sum()), 'rejected': int(rej.sum()),
                          'by_class': {CLASS_NAMES[c]: {'n': int((y == c).sum()), 'kind_kept': int(kind[y == c].sum()), 'class_right_kept': int(((p == y) & ~rej)[y == c].sum())} for c in sorted(set(y))},
                          'misses': collections.Counter(f'{CLASS_NAMES[a]}->{"REJ" if j else CLASS_NAMES[q]}' for a, q, j in zip(y, p, rej) if not (q // 2 == a // 2 and not j)).most_common()}
assert res['new_veg_test1_97']['n'] == 97 and res['new_veg_test2_54']['n'] == 54
t, n, a = res['test_300']['V5-H'], res['new_test_241']['V5-H'], res['any_photo_v2']
res['adopt_V5-H'] = bool(t['accuracy'] >= 284 and t['kept_correct'] >= 281 and n['accuracy'] >= 199 and n['kept_correct'] >= 189
                         and a['V5-H']['good_old_categories'] >= a['now']['good_old_categories'] - 5
                         and res['new_veg_test1_97']['V5-H']['kind_right_and_kept'] >= 73 and res['new_veg_test2_54']['V5-H']['kind_right_and_kept'] >= 41)
json.dump(res, open(V + 'final_once.json', 'w'), indent=1, ensure_ascii=False); print(json.dumps(res, indent=1, ensure_ascii=False))
