# All reviewed photos for v4 (new vegetables + a few weak-spot photos), with source, licence and a group (uploader /
# observer / source photo) for splitting; CLIP features; copies of earlier photos and near-copies inside the pool dropped.
# Split: Openverse / iNaturalist / Commons by uploader (sha1 parity, as the sealed sets: even = test side);
# Kaggle (kritikseth) and LVIS crops by group hash (1 in 4 to test). Test side -> "new vegetable" sealed test;
# the rest -> 80% Train, 20% Valid by group hash.
import json, csv, re, hashlib, collections, sys, numpy as np, torch, open_clip
sys.path.insert(0, '/home/user/depp-env22/src')
from model_utils import open_image
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; V = S + 'v4/'
h = lambda s: int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)
rows = []
ov = {o['id']: o for f in ('candidates_search.json', 'candidates_search2.json') for o in json.load(open(V + f))}
for i, v in json.load(open(V + 'ov_labels.json')).items():
    if not v: continue
    o = ov[i]
    rows.append(dict(path=o['path'], class_name=v['label'], source='openverse:' + o['provider'].split(':', 1)[1], license=f"{o['license']} {o['license_version'] or ''}".strip(),
                     reference=o['page'] or o['url'], group='u:' + o['uploader'], side=v['side']))
inat = {str(o['obs_id']): o for o in json.load(open(V + 'inat/inat_obs.json'))}
held = {r['creator'].lower() for r in csv.DictReader(open('/home/user/depp-env22/results/10class/outside_test2/outside_test2.csv', encoding='utf-8')) if r['id'].startswith('inat-')}
for i, v in json.load(open(V + 'inat_labels.json')).items():
    if not v: continue
    o = inat[i]; u = (o['user'] or '').lower()
    rows.append(dict(path=V + f'inat/img/{i}.jpg', class_name=v, source='inaturalist', license=o['photos'][0]['license'] or o['obs_license'] or '',
                     reference=o['page'], group='inat:' + u, side='test' if (u in held or h('inat:' + u) % 2 == 0) else 'train'))
cm = {o['path']: o for o in json.load(open(V + 'commons/files.json'))}
for p, v in json.load(open(V + 'file_labels.json')).items():
    if not v: continue
    if '/commons/' in p:
        o = cm[p]; u = (o['user'] or '').lower()
        rows.append(dict(path=p, class_name=v, source='wikimedia_commons', license=o['license'], reference=o['page'], group='commons:' + u,
                         side='test' if h('commons:' + u) % 2 == 0 else 'train'))
    elif '/krit/' in p:
        name = p.split('/')[-1]
        rows.append(dict(path=p, class_name=v, source='kaggle:kritikseth/fruit-and-vegetable-image-recognition', license='CC0-1.0', reference=name, group='krit:' + name, side=None))
    elif '/lvis/' in p:
        img = p.split('/')[-1].split('__')[1]
        rows.append(dict(path=p, class_name=v, source='kaggle:henningheyen/lvis-fruits-and-vegetables-dataset (COCO crop)', license='MIT (LVIS) / COCO image licence',
                         reference=p.split('/')[-1], group='coco:' + img, side=None))
print('reviewed', len(rows), collections.Counter(r['class_name'] for r in rows))
torch.set_num_threads(4)
model, _, prep = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k'); model.eval()
fs = []
with torch.inference_mode():
    for i in range(0, len(rows), 32):
        fs.append(torch.nn.functional.normalize(model.encode_image(torch.stack([prep(open_image(r['path'])) for r in rows[i:i + 32]])), dim=-1))
f = torch.cat(fs).numpy()
F = dict(np.load(S + 'clip/feat_datacomp_xl_s13b_b90k.npz'))
old = [F[k] for k in F if k != 'demo'] + [np.load(S + 'clipcls/test_features.npz')['clip'], np.load(S + 'arb/anyphoto_features.npz')['f']]
old += [np.load(p)['f'] for p in (S + 'ext3/features.npz', S + 'ext3/inat/features.npz', S + 'ext4/features.npz', S + 'arb/pool/feats.npz')]
old += [np.load(S + f'neg/{k}_feats.npz')['f'] for k in ('fruit', 'everyday')]   # otherveg (incl. kritikseth onion/sweet potato) left out on purpose
copy_old = (f @ np.concatenate(old).T).max(1) >= 0.95
# near-copies inside the pool: union groups (so a copy can never sit in two splits), keep one photo per near-copy cluster
sim = f @ f.T; n = len(rows); parent = list(range(n))
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
for a in range(n):
    for b in range(a + 1, n):
        if sim[a, b] >= 0.95: parent[find(a)] = find(b)
keep = {}
for i in range(n):
    if copy_old[i]: continue
    keep.setdefault(find(i), i)
kept = sorted(keep.values())
out = []
for i in kept:
    r = dict(rows[i]); side = r['side']
    if side is None: side = 'test' if h(r['group']) % 4 == 0 else 'train'
    r['split'] = 'test' if side == 'test' else ('val' if h('v:' + r['group']) % 5 == 0 else 'train')
    out.append(r)
np.savez(V + 'pool_feats.npz', f=f[kept])
json.dump(out, open(V + 'pool.json', 'w'), ensure_ascii=False, indent=0)
print('copies of earlier photos', int(copy_old.sum()), 'near-copies inside', n - int(copy_old.sum()) - len(kept), 'kept', len(out))
t = collections.Counter((r['class_name'], r['split']) for r in out)
for c in sorted({r['class_name'] for r in out}): print(f"{c:20s} train {t[(c,'train')]:4d} val {t[(c,'val')]:3d} test {t[(c,'test')]:3d}")
