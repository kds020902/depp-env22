"""Check overlap in immutable manifests, saved development features, and recorded CNN outputs."""

import argparse
import csv
import hashlib
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"tools"))
from compare_on_dev import development_sets, HEAD_DATA, ROOT_DATA
from evaluate import wilson
from decision import decide
from labels import CLASS_NAMES


def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def check(left,right,field):
    x={r[field] for r in left if r.get(field)}
    y={r[field] for r in right if r.get(field)}
    keys=x&y
    return {"keys":len(keys),"left":sum(r.get(field) in keys for r in left),
            "right":sum(r.get(field) in keys for r in right)}


def fingerprint(row):
    p=REPO/"dataset"/row["image_path"]
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    with Image.open(p) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        pixel = hashlib.sha256(rgb.tobytes()).hexdigest()
    return sha,pixel


def near_hash(left,right):
    ix,iy,pairs=set(),set(),0
    for i,a in enumerate(left):
        p,d=[int(a[k],16) for k in ("phash","dhash")]
        for j,b in enumerate(right):
            bp,bd=[int(b[k],16) for k in ("phash","dhash")]
            if (p^bp).bit_count()<=6 and (d^bd).bit_count()<=8:
                ix.add(i)
                iy.add(j)
                pairs+=1
    return {"pairs":pairs,"left":len(ix),"right":len(iy)}


def overlaps(X,Y):
    def norm(x):
        return x/np.linalg.norm(x,axis=1,keepdims=True).clip(min=1e-12)
    sim=norm(X)@norm(Y).T
    top=sim.max(1)
    exact={hashlib.sha256(x.tobytes()).digest() for x in Y}
    return {"bit_exact_left":sum(hashlib.sha256(x.tobytes()).digest() in exact for x in X),
            "near95_left": int((top>=.95).sum()), "near99_left":int((top>=.99).sum()),
            "near95_right":int((sim.max(0)>=.95).sum()),"max":float(top.max())},top


def accuracy_after(predictions,excluded):
    return {"all":wilson(sum(predictions),len(predictions)),
            "without":wilson(sum(v for i,v in enumerate(predictions) if i not in excluded),len(predictions)-len(excluded))}


def main(embedding_file):
    torch.set_num_threads(2)
    data=rows(REPO/"dataset/manifest.csv")
    file_hashes={}
    pixel_bad=0
    for row in data:
        sha,pixel=fingerprint(row)
        assert row["sha256"]==sha,row["image_id"]
        pixel_bad+=pixel!=row["pixel_sha256"]
        file_hashes[row["image_id"]]=pixel
    digest_rows=[{**r,"decoded_pixel_sha256":file_hashes[r["image_id"]]} for r in data]
    by={split:[r for r in digest_rows if r["split"]==split] for split in ("train","val","test")}
    main_compare={}
    for a,b in combinations(by,2):
        main_compare[a+"/"+b]={f:check(by[a],by[b],f) for f in
                               ("sha256","pixel_sha256","decoded_pixel_sha256","group_id","session_id","source_dataset")}
        main_compare[a+"/"+b]["perceptual"] = near_hash(by[a],by[b])
    scores={}
    train_ses={r["session_id"] for r in by["train"]}
    base=REPO/"results/10class/efficientnet_b0_finetune"
    for split in ("val","test"):
        pred=rows(base/(split+"_predictions.csv"))
        assert {r["image_id"] for r in pred}=={r["image_id"] for r in by[split]}
        drop={r["image_id"] for r in by[split] if r["session_id"] in train_ses}
        scores[split+"_recorded_CNN_without_shared_session"]=accuracy_after(
            [r["correct"]=="True" for r in pred],{i for i,r in enumerate(pred) if r["image_id"] in drop})
    head=np.load(HEAD_DATA)
    X=head["train_X"]
    base_train=X[head["train_tag"]=="train1260"]
    checkpoint=torch.load(REPO/"model.pth",map_location="cpu",weights_only=True)
    per_set={}
    stored=development_sets()
    root_ids=np.load(ROOT_DATA)["ids"].tolist()
    inputs={name:x.numpy() for name,(_,x,_) in stored.items()}
    if embedding_file:
        baked=np.load(embedding_file)
        assert baked["ids"].tolist()==[r["image_id"] for r in by["test"]]
        inputs["base Test 300 (embeddings only)"]=baked["features"]
    pooled_ok=[]
    pooled_dropped=set()
    for name,x in inputs.items():
        full,top=overlaps(x,X)
        common,_=overlaps(x,base_train)
        per_set[name]={"new_head_train":full,"base_train_subset_shared_by_both":common}
        exclude=set(np.flatnonzero(top>=.95).tolist())
        if name not in stored:
            ids={by["test"][i]["image_id"] for i in exclude}
            session_ids={r["image_id"] for r in by["test"] if r["session_id"] in train_ses}
            prediction_rows=rows(base/"test_predictions.csv")
            for suffix,drop in (("clip95",ids),("session_or_clip95",ids | session_ids)):
                scores["test_recorded_CNN_without_"+suffix]=accuracy_after(
                    [r["correct"]=="True" for r in prediction_rows],
                    {i for i,r in enumerate(prediction_rows) if r["image_id"] in drop})
            per_set[name]["clip95_also_shared_session"]=len(ids & session_ids)
            continue
        logits,features,labels=stored[name]
        with torch.inference_mode():
            a=decide(logits,checkpoint,features)
        if labels is None:
            expected_to_reject=not name.startswith("real-scene")
            scores[name+"_without_new_head_near95"]=accuracy_after(
                [u==expected_to_reject for u in a.unknown.tolist()],exclude)
            continue
        is_right=[not u and y==CLASS_NAMES[i] for y,i,u in zip(labels,a.index.tolist(),a.unknown.tolist())]
        scores[name+"_without_new_head_near95"]=accuracy_after(is_right,exclude)
        pooled_dropped.update(len(pooled_ok)+i for i in exclude)
        pooled_ok.extend(is_right)
        if name.startswith("potato /"):
            per_set[name]["near95_root_ids"]=[{"id":str(root_ids[i]),"label":labels[i],"cosine":float(top[i])} for i in sorted(exclude)]
    scores["all_labelled_dev_without_new_head_near95"]=accuracy_after(pooled_ok,pooled_dropped)
    pairwise={}
    for (name,x),(other,y) in combinations(inputs.items(),2):
        pairwise[name+" / "+other]=overlaps(x,y)[0]
    first_manifest=rows(REPO/"experiments/not_five_class/added_photos.csv")
    new_manifest=rows(REPO/"experiments/new_vegetables/photos.csv")
    new_train=[r for r in new_manifest if r["split"]=="train"]
    new_val=[r for r in new_manifest if r["split"]=="val"]
    provenance={
        "first_list_rows":len(first_manifest),
        "first_list_source_groups_are_collection_tags":dict(Counter(r["group"] for r in first_manifest)),
        "new_vegetables_photo_train_rows":len(new_train),
        "new_vegetables_photo_val_rows":len(new_val),
        "new_vegetables_train_val_source_group":check(new_train,new_val,"group"),
        "new_vegetables_train_val_page":check(new_train,new_val,"page"),
        "new_vegetables_train_val_image_url":check(new_train,new_val,"image_url"),
    }
    return {"verified_file_hashes":len(data),"pixel_digest_differences_from_manifest":pixel_bad,
            "base_manifest":main_compare,"dev_or_test_to_heads":per_set,"development_pairwise":pairwise,
            "metadata_provenance":provenance, "scores_without_overlap":scores,
            "first_head_complete":"missing original 4623 extra images / features and complete row mapping"}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--test-embeddings",type=Path)
    args=parser.parse_args()
    args.out.write_text(json.dumps(main(args.test_embeddings),indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
