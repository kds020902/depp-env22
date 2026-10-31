# Pre-registered (v4/PREREG.md): V4 (8 vegetables, 16 slots) and the current system, scored once on the sealed sets:
# Test 300, the new internet test 241, the any-photo set relabelled for 8 vegetables (v2), and the new-vegetable test 97.
import sys, csv, json, pickle, collections, numpy as np, torch, open_clip
from scipy.special import log_softmax, logsumexp
sys.path.insert(0, '/home/user/depp-env22/src')
from model_utils import load_model, make_transform, open_image, input_size, CLASS_NAMES
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; V = S + 'v4/'; REPO = '/home/user/depp-env22/'
REJECT = ('rotten_fruit', 'fresh_fruit', 'other_veg', 'food', 'drawing_toy', 'everyday')
torch.set_num_threads(4)
B4, ck4 = load_model(sys.argv[1], torch.device('cpu')); tf4 = make_transform(input_size(ck4)); hv = pickle.load(open(V + 'head_v4.pkl', 'rb'))   # retrained model: run for the record only, V4-R does not use it
B0, ck0 = load_model(REPO + 'model.pth', torch.device('cpu')); tf0 = make_transform(input_size(ck0)); h0 = ck0['clip_head']
clip, _, prep = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k'); clip.eval()
def run(paths, need_clip=True):
    b4, b0, cf = [], [], []
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            ims = [open_image(p) for p in paths[i:i + 32]]
            b4.append(B4(torch.stack([tf4(im) for im in ims]))); b0.append(B0(torch.stack([tf0(im) for im in ims])))
            if need_clip: cf.append(torch.nn.functional.normalize(clip.encode_image(torch.stack([prep(im) for im in ims])), dim=-1))
    return torch.cat(b4).numpy(), torch.cat(b0).numpy(), (torch.cat(cf).numpy() if need_clip else None)
def decide(name, b4, b0, f):
    if name == 'v4':   # V4-R: the new head picks the kind; the five old vegetables keep the current system's answer
        zv = f @ hv['coef'].T + hv['intercept']; n = len(f)
        sp = np.exp(log_softmax(zv[:, :16], 1)).reshape(n, 8, 2).sum(2).argmax(1)
        newp = sp * 2 + zv[:, :16].reshape(n, 8, 2)[np.arange(n), sp].argmax(1)
        z0 = f @ h0['coef'].numpy().T + h0['intercept'].numpy()
        oldp = (log_softmax(b0 / ck0['calibration']['temperature'], 1) + log_softmax(z0[:, :10], 1)).argmax(1)
        return np.where(sp < 5, oldp, newp), zv[:, 16] - logsumexp(zv[:, :16], 1) > hv['EC']
    if name == 'v4_unused':
        z = f @ hv['coef'].T + hv['intercept']; K = 16; b, TB = b4, ck4['calibration']['temperature']; EC = hv['EC']
    else:
        z = f @ h0['coef'].numpy().T + h0['intercept'].numpy(); K = 10; b, TB = b0, ck0['calibration']['temperature']; EC = h0['unknown_threshold']
    return (log_softmax(b / TB, 1) + log_softmax(z[:, :K], 1)).argmax(1), z[:, K] - logsumexp(z[:, :K], 1) > EC
def veg_scores(b4, b0, f, y):
    out = {}
    for name in ('now', 'v4'):
        p, rej = decide(name, b4, b0, f); right = p == y
        out[name] = {'accuracy': int(right.sum()), 'kept_correct': int((right & ~rej).sum()), 'rejected': int(rej.sum()), 'kept_wrong': int((~right & ~rej).sum()),
                     'answered_new_species': int((p >= 10).sum()), 'by_class': {n: f'{int(right[y == i].sum())}/{int((y == i).sum())}' for i, n in enumerate(CLASS_NAMES) if (y == i).any()}}
    return out
res = {'vegetable_model': sys.argv[1]}
test = [r for r in csv.DictReader(open(REPO + 'dataset/manifest.csv', encoding='utf-8')) if r['split'] == 'test']
T = np.load(S + 'clipcls/test_features.npz')
b4, _, _ = run([REPO + 'dataset/' + r['image_path'] for r in test], need_clip=False)
res['test_300'] = veg_scores(b4, T['b_logits'], T['clip'], np.array([CLASS_NAMES.index(r['class_name']) for r in test]))
assert (res['test_300']['now']['accuracy'], res['test_300']['now']['kept_correct']) == (288, 285), res['test_300']['now']
t2 = list(csv.DictReader(open(REPO + 'results/10class/outside_test2/outside_test2.csv', encoding='utf-8')))
res['new_test_241'] = veg_scores(*run([S + f"weak/t2img/{r['id']}.jpg" for r in t2]), np.array([CLASS_NAMES.index(r['class_name']) for r in t2]))
assert (res['new_test_241']['now']['accuracy'], res['new_test_241']['now']['kept_correct']) == (203, 193), res['new_test_241']['now']
# any-photo v2
rows = list(csv.DictReader(open(S + 'arb/anyphoto_test_v2.csv', encoding='utf-8'))); files = json.load(open(S + 'arb/local_files.json'))
b4, b0, f = run([files[r['id']] for r in rows])
res['any_photo_v2'] = {}
for name in ('now', 'v4'):
    p, rej = decide(name, b4, b0, f); good, cats = [], []
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
# new-vegetable test (97)
pool = json.load(open(V + 'pool_with_ids.json')); te = [r for r in pool if r['split'] == 'test' and r['class_name'].split('_')[0] in ('onion', 'sweetpotato', 'greenonion')]
b4, b0, f = run([V + 'ds/' + r['image_path'] for r in te]); y = np.array([CLASS_NAMES.index(r['class_name']) for r in te])
p, rej = decide('v4', b4, b0, f)
kind = (p // 2 == y // 2) & ~rej
res['new_veg_test'] = {'n': len(y), 'kind_right_and_kept': int(kind.sum()), 'rejected': int(rej.sum()),
                       'by_class': {CLASS_NAMES[c]: {'n': int((y == c).sum()), 'kind_kept': int(kind[y == c].sum()), 'class_right_kept': int(((p == y) & ~rej)[y == c].sum())} for c in sorted(set(y))},
                       'answers': collections.Counter(f'{CLASS_NAMES[a]}->{"REJ" if j else CLASS_NAMES[q]}' for a, q, j in zip(y, p, rej)).most_common()}
t, n, a = res['test_300']['v4'], res['new_test_241']['v4'], res['any_photo_v2']
res['adopt'] = bool(t['accuracy'] >= 284 and t['kept_correct'] >= 281 and n['accuracy'] >= 199 and n['kept_correct'] >= 189
                    and a['v4']['good_old_categories'] >= a['now']['good_old_categories'] - 5 and res['new_veg_test']['kind_right_and_kept'] >= 73)
json.dump(res, open(V + 'final_once.json', 'w'), indent=1, ensure_ascii=False); print(json.dumps(res, indent=1, ensure_ascii=False))
