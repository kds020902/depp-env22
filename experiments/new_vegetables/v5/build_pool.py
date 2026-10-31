# v5 pool: the v4 pool (splits unchanged) + photos reviewed in this round (iNaturalist, Openverse, Wikimedia Commons, more LVIS crops).
# Same rules as v4: copies (CLIP cosine >= 0.95) of any earlier photo, or of each other, are dropped; split by uploader / observer /
# source photo; a group already in the v4 pool keeps its v4 split; uploaders of sealed check sets go to the test side; otherwise
# h('t:'+group) % 3 == 0 -> test; at most 1/3 of each new class goes to test (non-held groups moved back in hash order);
# Valid = h('v:'+group) % 5 == 0 among the rest. Photos of the old five (weak spots) are Train only.
import json, csv, re, math, hashlib, collections, sys, glob, numpy as np, torch, open_clip
sys.path.insert(0, '/home/user/depp-env22/src')
from model_utils import open_image
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; V4 = S + 'v4/'; V = S + 'v5/'; R = '/home/user/depp-env22/'
h = lambda s: int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)
NEW = ('onion', 'sweetpotato', 'greenonion')
def uploader(o):
    m = re.search(r'flickr\.com/photos/([^/]+)/', o.get('page') or '')
    if m: return 'flickr:' + m.group(1).lower()
    c = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', o.get('creator') or '')).strip().lower()
    return (o['provider'] + ':' + c) if c else 'id:' + o['id']
held = set()
for path in ('results/10class/outside_test/outside_test.csv', 'experiments/asian_cucumber/asian_check.csv', 'experiments/fruit_other/fruit_check.csv'):
    for r in csv.DictReader(open(R + path, encoding='utf-8')): held.add(uploader({'page': r.get('page'), 'creator': r.get('creator'), 'provider': r.get('provider'), 'id': r['id']}))
for path in ('results/10class/outside_test2/outside_test2.csv', 'results/10class/anyphoto_test/anyphoto_test.csv'):
    for r in csv.DictReader(open(R + path, encoding='utf-8')):
        if r['source'] == 'inaturalist': held.add('inat:' + r['creator'].lower())
        else: held.add(uploader({'page': r['page'], 'creator': r['creator'], 'provider': 'openverse:' + r['provider'], 'id': r['id']}))
gk = lambda g: g[2:] if g.startswith('u:') else g

v4 = json.load(open(V4 + 'pool.json'))
v4_split = {r['group']: r['split'] for r in v4}
v4_paths = {r['path'] for r in v4}
rows = []
ov = {o['id']: o for o in json.load(open(V + 'candidates_search3.json'))}
for i, v in json.load(open(V + 'ov_labels.json')).items():
    if not v: continue
    o = ov[i]
    rows.append(dict(path=o['path'], class_name=v['label'], source='openverse:' + o['provider'].split(':', 1)[1], license=f"{o['license']} {o['license_version'] or ''}".strip(),
                     reference=o['page'] or o['url'], group='u:' + o['uploader'], round='v5'))
inat = {str(o['obs_id']): o for o in json.load(open(V + 'inat/inat_obs.json'))}
for i, v in json.load(open(V + 'labels.json')).items():
    if not v: continue
    o = inat[i]; u = (o['user'] or '').lower()
    rows.append(dict(path=V + f'inat/img/{i}.jpg', class_name=v, source='inaturalist', license=o['photos'][0]['license'] or o['obs_license'] or '',
                     reference=o['page'], group='inat:' + u, round='v5'))
cm = {o['path']: o for o in json.load(open(V + 'commons/files.json'))}
for p, v in json.load(open(V + 'commons_labels.json')).items():
    if not v: continue
    o = cm[p]; u = (o['user'] or '').lower()
    rows.append(dict(path=p, class_name=v, source='wikimedia_commons', license=o['license'], reference=o['page'], group='commons:' + u, round='v5'))
for p, v in json.load(open(V4 + 'file_labels.json')).items():
    if not v or '/lvis/' not in p or p in v4_paths: continue
    img = p.split('/')[-1].split('__')[1]
    rows.append(dict(path=p, class_name=v, source='kaggle:henningheyen/lvis-fruits-and-vegetables-dataset (COCO crop)', license='MIT (LVIS) / COCO image licence',
                     reference=p.split('/')[-1], group='coco:' + img, round='v5'))
print('reviewed this round', len(rows), dict(collections.Counter(r['class_name'] for r in rows)))

torch.set_num_threads(4)
model, _, prep = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k'); model.eval()
def feats(paths):
    out = []
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            out.append(torch.nn.functional.normalize(model.encode_image(torch.stack([prep(open_image(p)) for p in paths[i:i + 32]])), dim=-1))
    return torch.cat(out).numpy()
f = feats([r['path'] for r in rows])
t2 = sorted(glob.glob(S + 'weak/t2img/*.jpg')); ft2 = feats(t2)       # the sealed new internet test 241, to be sure
F = dict(np.load(S + 'clip/feat_datacomp_xl_s13b_b90k.npz'))
old = [F[k] for k in F if k != 'demo'] + [np.load(S + 'clipcls/test_features.npz')['clip'], np.load(S + 'arb/anyphoto_features.npz')['f'], ft2, np.load(V4 + 'pool_feats.npz')['f']]
old += [np.load(p)['f'] for p in (S + 'ext3/features.npz', S + 'ext3/inat/features.npz', S + 'ext4/features.npz', S + 'arb/pool/feats.npz')]
old += [np.load(S + f'neg/{k}_feats.npz')['f'] for k in ('fruit', 'everyday')]
copy_old = (f @ np.concatenate(old).T).max(1) >= 0.95
sim = f @ f.T; n = len(rows); parent = list(range(n))
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
for a in range(n):
    for b in range(a + 1, n):
        if sim[a, b] >= 0.95: parent[find(a)] = find(b)
keep = {}
for i in range(n):
    if not copy_old[i]: keep.setdefault(find(i), i)
kept = sorted(keep.values()); new = [dict(rows[i]) for i in kept]
print('copies of earlier photos', int(copy_old.sum()), 'near-copies inside', n - int(copy_old.sum()) - len(kept), 'kept', len(new))
for r in new:
    g = r['group']
    if not r['class_name'].startswith(NEW): r['split'] = 'train'
    elif g in v4_split: r['split'] = v4_split[g]
    elif gk(g) in held or h('t:' + g) % 3 == 0: r['split'] = 'test'
    else: r['split'] = 'val' if h('v:' + g) % 5 == 0 else 'train'
for c in ('onion_fresh', 'onion_rotten', 'sweetpotato_fresh', 'sweetpotato_rotten', 'greenonion_fresh'):
    rs = [r for r in new if r['class_name'] == c]; cap = math.ceil(len(rs) / 3)
    groups = sorted({r['group'] for r in rs if r['split'] == 'test' and gk(r['group']) not in held and r['group'] not in v4_split}, key=lambda g: h('m:' + g))
    while sum(r['split'] == 'test' for r in rs) > cap and groups:
        g = groups.pop()
        for r in new:
            if r['group'] == g and r['class_name'].startswith(NEW): r['split'] = 'val' if h('v:' + g) % 5 == 0 else 'train'
pool = [dict(r, round='v4') for r in v4] + new
gs = collections.defaultdict(set)
for r in pool: gs[r['group']].add(r['split'] if r['class_name'].startswith(NEW) else 'old5')
bad = [g for g, v in gs.items() if len(v - {'old5'}) > 1]
print('groups in two splits:', len(bad), bad[:5])
np.savez(V + 'pool_feats.npz', f=np.concatenate([np.load(V4 + 'pool_feats.npz')['f'], f[kept]]))
json.dump(pool, open(V + 'pool.json', 'w'), ensure_ascii=False, indent=0)
t = collections.Counter((r['class_name'], r['split'], r['round']) for r in pool)
print(f"{'class':20s} | v4 train/val/test | v5 new train/val/test | total train/val/test")
for c in sorted({r['class_name'] for r in pool}):
    a = [t[(c, s, 'v4')] for s in ('train', 'val', 'test')]; b = [t[(c, s, 'v5')] for s in ('train', 'val', 'test')]
    print(f"{c:20s} | {a} | {b} | {[x + y for x, y in zip(a, b)]}")
