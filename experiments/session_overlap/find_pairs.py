# Record of the review sheets (paths point into the working folder; photos of the CLIP-head extra set are not in the repository).
# Make review sheets: each row = Test photo + its two most similar training photos (Train 1260 + CLIP-head extra 4623).
import numpy as np, csv, json, sys, random
from PIL import Image, ImageDraw, ImageFont, ImageOps
S='/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; R='/home/user/depp-env22/'
F=np.load(S+'sess/feats.npz')
rows=list(csv.DictReader(open(R+'dataset/manifest.csv',encoding='utf-8')))
tr=[(R+'dataset/'+r['image_path'], 'Train '+r['class_name']) for r in rows if r['split']=='train']
te=[(R+'dataset/'+r['image_path'], r['class_name']) for r in rows if r['split']=='test']
ex=[(r['path'], 'extra '+r['class_name']) for r in csv.DictReader(open(S+'negx/extra_train_N3.csv'))]
pool=tr+ex; P=np.concatenate([F['train'],F['extra']])
C=F['test']@P.T; order=np.argsort(-C,1)[:,:2]; top=C[np.arange(len(te)),order[:,0]]
cand=[i for i in np.argsort(-top) if top[i]>=0.88]
mid=[i for i in range(len(te)) if 0.85<=top[i]<0.88]; random.Random(0).shuffle(mid); spot=sorted(mid[:20], key=lambda i:-top[i])
json.dump({'cand':[int(i) for i in cand],'spot':[int(i) for i in spot],'top':top.tolist(),'nn':order.tolist(),'pool':[p for p,_ in pool],'test':[p for p,_ in te]}, open(S+'sess/cand.json','w'))
try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
except Exception: font=ImageFont.load_default()
def cell(p, size=230):
    im=ImageOps.exif_transpose(Image.open(p)).convert('RGB'); im.thumbnail((size,size)); c=Image.new('RGB',(size,size),'white'); c.paste(im,((size-im.width)//2,(size-im.height)//2)); return c
def sheet(idx, name, per=8):
    for s in range(0,len(idx),per):
        part=idx[s:s+per]; W=3*240+260; H=len(part)*250
        out=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(out)
        for k,i in enumerate(part):
            y=k*250; out.paste(cell(te[i][0]),(0,y))
            for j in range(2):
                n=order[i,j]; out.paste(cell(pool[n][0]),(240*(j+1),y))
            d.text((725,y+10),f'#{s+k}  test {i}',fill='black',font=font)
            d.text((725,y+30),te[i][1],fill='black',font=font)
            for j in range(2):
                n=order[i,j]; d.text((725,y+60+40*j),f'{j+1}: {pool[n][1]}',fill='black',font=font); d.text((725,y+78+40*j),f'cos {C[i,n]:.3f}',fill='red',font=font)
        out.save(S+f'sess/{name}_{s:03d}.jpg',quality=85)
sheet(cand,'cand'); sheet(spot,'spot')
print(len(cand), len(spot))
