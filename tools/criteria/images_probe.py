"""Use only the packaged dataset; no T3–T9 images or scoring entry points."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes, label
import torch
from torch.utils.data import DataLoader

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
import predict
from images import open_image
from networks import input_size
from train import CachedImages, predict_logits
from evaluate import wilson


def manifest(split):
    with (REPO/"dataset/manifest.csv").open(encoding="utf-8",newline="") as handle:
        return [r for r in csv.DictReader(handle) if r["split"]==split]


def valid_selection():
    grouped=defaultdict(list)
    for r in manifest("val"):
        grouped[r["class_name"]].append(r)
    return [r for key in sorted(grouped) for r in sorted(grouped[key],key=lambda x:x["image_id"])[:2]]


def shortcut_images(photo):
    p=np.array(photo.resize((224,224),Image.Resampling.BILINEAR))
    border=np.concatenate([p[:10].reshape(-1,3),p[-10:].reshape(-1,3),p[:,:10].reshape(-1,3),p[:,-10:].reshape(-1,3)])
    neutral=np.median(border,axis=0)
    delta=np.linalg.norm(p.astype(float)-neutral,axis=2)
    border_delta=np.linalg.norm(border-neutral,axis=1)
    threshold=max(25,float(np.quantile(border_delta,.90))+12)
    candidates,n=label(binary_fill_holes(binary_closing(delta>threshold,iterations=2)))
    counts=np.bincount(candidates.ravel())
    if n:
        index=int(counts[1:].argmax())+1
        fg=candidates==index
    else:
        fg=np.zeros(p.shape[:2],dtype=bool)
    fill=neutral.round().clip(0,255).astype(np.uint8)
    center=p.copy()
    center[56:168,56:168]=fill
    bg=p.copy()
    bg[fg]=fill
    area=float(fg.mean())
    return Image.fromarray(p),Image.fromarray(center),Image.fromarray(bg),fg if .05<area<.85 else None


def key(result):
    return (("unknown",) if result["unknown"] else (result["index"],))


def probe(model):
    root=REPO/"dataset"
    selected=valid_selection()
    size=input_size(model.checkpoint)
    valid=CachedImages(selected,root,train=False,size=size)
    tensors=[]
    outputs=[]
    checks=[]
    center_same,background_same,background_n,heat_on=[],[],0,[]
    masks=[]
    times=[]
    for i,row in enumerate(selected):
        photo=open_image(root/row["image_path"])
        tensors.clear()
        outputs.clear()
        start=time.perf_counter()
        pre=model.model.register_forward_pre_hook(lambda module,args: tensors.append(args[0].detach().cpu().clone()))
        post=model.model.register_forward_hook(lambda module,args,out: outputs.append(out.detach().cpu().clone()))
        try:
            original=model.classify(photo,.9)
        finally:
            pre.remove()
            post.remove()
        times.append(time.perf_counter()-start)
        expected_x,expected_y=valid[i]
        expected_z,_=predict_logits(model.model,DataLoader([(expected_x,expected_y)],batch_size=1),torch.device("cpu"))
        checks.append({"id":row["image_id"],"tensor_max_abs":float((tensors[0][0]-expected_x).abs().max()),
                       "logit_max_abs":float((outputs[0]-expected_z).abs().max())})
        resized,center,bg,fg=shortcut_images(photo)
        base=model.classify(resized,.9)
        c=model.classify(center,.9)
        center_same.append(key(c)==key(base))
        if fg is not None:
            background_n+=1
            q=model.classify(bg,.9)
            same=key(q)==key(base)
            background_same.append(same)
            entry={"id":row["image_id"],"mask_fraction":float(fg.mean()),"background_same":same}
            h=base["rot_map"]
            if h is not None:
                h=torch.nn.functional.interpolate(h[None,None].float(),size=fg.shape,mode="bilinear",align_corners=False)[0,0].numpy()
                if h.sum()>0:
                    entry["heat_mass_fraction"]=float(h[fg].sum()/h.sum())
                    entry["heat_peak_inside"]=bool(fg[np.unravel_index(h.argmax(),h.shape)])
                    heat_on.append(entry["heat_peak_inside"])
            masks.append(entry)
    return {"valid_checks":checks,"max_tensor_abs":max(x["tensor_max_abs"] for x in checks),
            "max_logit_abs":max(x["logit_max_abs"] for x in checks),"predict_seconds_median":float(np.median(times)),
            "center_covered":wilson(sum(center_same),len(center_same)),
            "background_proxy":wilson(sum(background_same),background_n),"heat_peak_in_proxy_mask":wilson(sum(heat_on),len(heat_on)),
            "proxy_area_and_heat_samples":masks}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["probe","test-embeddings"])
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    start=time.perf_counter()
    model=predict.Classifier(REPO/"model.pth",torch.device("cpu"))
    if a.mode=="probe":
        result=probe(model)
        result["including_model_load_seconds"]=time.perf_counter()-start
        a.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    else:
        test_rows=manifest("test")
        features=[]
        with torch.inference_mode():
            for n in range(0,len(test_rows),16):
                photo=[open_image(REPO/"dataset"/r["image_path"]) for r in test_rows[n:n+16]]
                features.append(model.clip.image_features(photo).cpu().numpy())
        np.savez(a.out,ids=[r["image_id"] for r in test_rows],features=np.concatenate(features))
