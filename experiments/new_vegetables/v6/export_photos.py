# Write the photo lists for the repository: every reviewed new-vegetable photo (v4, v5, v6 rounds) with its split, source,
# licence, attribution and download address, and the three sealed new-vegetable tests. Photo files are not committed.
import json, csv, glob, os, hashlib, collections, sys
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; R = '/home/user/depp-env22/'
out_dir, test_dir = sys.argv[1], sys.argv[2]
ov = {}
for f in glob.glob(S + 'v4/candidates_search*.json') + [S + 'v5/candidates_search3.json'] + glob.glob(S + 'v6/candidates_search_ov*.json'):
    for o in json.load(open(f)): ov[o['path']] = o
inat = {}
for v in ('v4', 'v5', 'v6'):
    for o in json.load(open(S + f'{v}/inat/inat_obs.json')): inat[S + f"{v}/inat/img/{o['obs_id']}.jpg"] = o
cm = {}
for f in (S + 'v4/commons/files.json', S + 'v5/commons/files.json', S + 'v6/commons/files2.json'):
    for o in json.load(open(f)): cm[o['path']] = o
pool = json.load(open(S + 'v6/pool.json'))
rows = []
for r in pool:
    p = r['path']; name = os.path.basename(p)
    if p in ov:
        o = ov[p]; rid, url, att = 'openverse-' + o['id'], o['url'], o.get('creator') or ''
    elif p in inat:
        o = inat[p]; ph = o['photos'][0]; rid, url, att = f"inat-{o['obs_id']}", ph['url'], ph.get('attribution') or o.get('user') or ''
    elif p in cm:
        o = cm[p]; rid, url, att = o['id'], o['url'], o.get('artist') or o.get('user') or ''
    elif '/krit/' in p:
        rid, url, att = 'kritikseth/' + name, '', ''
    elif '/lvis/' in p:
        rid, url, att = 'lvis/' + name, '', ''
    else: raise SystemExit('unknown source ' + p)
    rows.append(dict(id=rid, round=r['round'], split=r['split'], class_name=r['class_name'], source=r['source'], license=r['license'],
                     attribution=att.replace('\n', ' ')[:150], page=r['reference'], image_url=url, group=r['group']))
assert len({x['id'] for x in rows}) == len(rows)
F = ['id', 'round', 'split', 'class_name', 'source', 'license', 'attribution', 'page', 'image_url', 'group']
with open(out_dir + '/photos.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=F); w.writeheader(); w.writerows(rows)
NEW = ('onion', 'sweetpotato', 'greenonion')
test = [dict(x, test={'v4': 1, 'v5': 2, 'v6': 3}[x['round']]) for x in rows if x['split'] == 'test' and x['class_name'].startswith(NEW)]
with open(test_dir + '/new_veg_test.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'test', 'class_name', 'source', 'license', 'attribution', 'page', 'image_url']); w.writeheader()
    w.writerows({k: x[k] for k in ['id', 'test', 'class_name', 'source', 'license', 'attribution', 'page', 'image_url']} for x in test)
print(len(rows), collections.Counter((x['round'], x['split']) for x in rows))
print('tests', collections.Counter(x['test'] for x in test), hashlib.sha256(open(test_dir + '/new_veg_test.csv', 'rb').read()).hexdigest())
print(collections.Counter(x['source'].split(':')[0] for x in rows))
