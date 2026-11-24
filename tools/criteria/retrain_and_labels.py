"""Train audit checkpoints in a temporary directory; diagnose label issues with out-of-fold scores."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from cleanlab.filter import find_label_issues
import torch

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"tools"))
from compare_on_dev import development_sets, evaluate, HEAD_DATA, ROOT_DATA
from calibration import fit_temperature
from decision import clip_head_logits, clip_unknown_score, combined_logits
from labels import CLASS_NAMES
from networks import set_seed, state_checksum

SEEDS=(42,43,44)


def assess(checkpoint,sets):
    x=evaluate(checkpoint,sets)
    y=x["_all_labelled"]
    return {"right": y["right"],"n":y["n"],"shown_right":y["shown_right"],"shown":y["shown"],
            "rotten_called_fresh":y["rotten_called_fresh"],"held":y["held_back"],
            "fruit_rejected":x["fruit 374"]["turned_away"],"lvis_rejected":x["real-scene vegetable crops 192"]["turned_away"],
            "answers":{k:v.get("answers") for k,v in x.items() if k!="_all_labelled"}}


def compare(head,other):
    return {k: {"equal":torch.equal(head[k],other[k]),"max_abs":float((head[k]-other[k]).abs().max())}
            for k in ("coef","intercept")}


def diff_answers(x,y):
    return sum(a!=b for name in x["answers"] for a,b in zip(x["answers"][name],y["answers"][name]))


def averaged(runs):
    out={}
    for key in ("right","shown_right","shown","rotten_called_fresh","fruit_rejected","lvis_rejected"):
        v=[r["dev"][key] for r in runs]
        out[key]={"mean":float(np.mean(v)),"sample_sd":float(np.std(v,ddof=1))}
    return out


def retrain():
    checkpoint=torch.load(REPO/"model.pth",map_location="cpu",weights_only=True)
    sets=development_sets()
    reference=assess(checkpoint,sets)
    head_data=np.load(HEAD_DATA)
    label=head_data["train_y"].astype(int)
    take=(label<10)|(label==16)
    seed_runs, proxy_runs=[],[]
    argv_seed=("import random,sys,runpy,numpy as np,torch; from pathlib import Path; "
               "sys.argv.pop(0); s=int(sys.argv.pop(0)); sys.path.insert(0,str(Path(sys.argv[0]).parent)); "
               "random.seed(s); np.random.seed(s); torch.manual_seed(s); "
               "torch.set_num_threads(2); runpy.run_path(sys.argv[0],run_name='__main__')")
    tools=os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="depp-criteria-") as folder:
        for seed in SEEDS:
            output=Path(folder)/f"candidate-{seed}.pth"
            cmd=[sys.executable,"-c",argv_seed,str(seed),"src/train_new_veg_head.py","--model","model.pth",
                 "--data","experiments/new_vegetables/v7/head_data.npz","--tau","0.8","--out",str(output)]
            subprocess.run(cmd,cwd=REPO,env=tools,check=True)
            new=torch.load(output,map_location="cpu",weights_only=True)
            assert state_checksum(new["state_dict"])==state_checksum(checkpoint["state_dict"])
            result=assess(new,sets)
            new_h=new["new_vegetable_head"]
            seed_runs.append({"seed":seed,"weights":compare(new_h,checkpoint["new_vegetable_head"]),
                              "params":{k:[checkpoint["new_vegetable_head"][k],new_h[k]] for k in ("temperature","unknown_threshold","switch_probability")},
                              "answers_changed":diff_answers(result,reference),"dev":result})
            # First head surrogate: available V7 rows with labels 0-9 or not-eight.
            # Missing not-five-only rows, original watermark features and original order.
            set_seed(seed,2)
            fitted=LogisticRegression(C=10,max_iter=5000,random_state=seed).fit(head_data["train_X"][take],np.where(label[take]==16,10,label[take]))
            h={**checkpoint["clip_head"],"coef":torch.tensor(fitted.coef_,dtype=torch.float32),
               "intercept":torch.tensor(fitted.intercept_,dtype=torch.float32)}
            copied={**checkpoint,"clip_head":h}
            b,x,y=sets["Valid 300"]
            def unknown(f):
                return clip_unknown_score(clip_head_logits(f,copied),copied,f)
            h["unknown_threshold"]=float(torch.quantile(torch.cat([unknown(x),unknown(sets["real-scene vegetable crops 192"][1])]),.95))
            h["temperature"]=fit_temperature(combined_logits(b,clip_head_logits(x,copied),copied),
                                              torch.tensor([CLASS_NAMES.index(name) for name in y]))
            trial=assess(copied,sets)
            proxy_runs.append({"seed":seed,"weights":compare(h,checkpoint["clip_head"]),"answers_changed":diff_answers(trial,reference),"dev":trial})
    def compact(entries):
        return [{**e,"dev":{k:v for k,v in e["dev"].items() if k!="answers"}} for e in entries]
    return {"ids":list(SEEDS),"reference":{k:v for k,v in reference.items() if k!="answers"},
            "second_head_cli":compact(seed_runs),"second_head_seed_summary":averaged(seed_runs),
            "first_head_original":"not measured: missing external original extra-train photos, watermark features, and required LVIS photo folder",
            "first_head_proxy_n":int(take.sum()),"first_head_proxy":compact(proxy_runs),"first_head_proxy_seed_summary":averaged(proxy_runs)}


def label_issues():
    names,data,rows,labels,features=[],[],[],[],[]
    for name,(b,f,y) in development_sets().items():
        if y is None:continue
        names.extend([name]*len(y))
        labels.extend(y)
        data.append(np.c_[b.numpy(),f.numpy()])
        features.append(f.numpy())
        if name.startswith("potato /"):
            rows.extend(map(str,np.load(ROOT_DATA)["ids"]))
        else:
            rows.extend(str(i+1) for i in range(len(y)))
    values=np.concatenate(data)
    clip=np.concatenate(features)
    clip=clip/np.linalg.norm(clip,axis=1,keepdims=True).clip(min=1e-12)
    parent=list(range(len(clip)))
    def root(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]]
            index=parent[index]
        return index
    for i,j in zip(*np.where(np.triu(clip@clip.T>=.95,k=1))):
        parent[root(int(i))]=root(int(j))
    groups=np.array([root(i) for i in range(len(clip))])
    classes,lab=np.unique(labels,return_inverse=True)
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=421)
    folds=list(splitter.split(values,lab,groups))
    for a,b in folds:
        assert len(set(lab[a]))==len(classes) and not set(groups[a])&set(groups[b])
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2500))
    probabilities=cross_val_predict(model,values,lab,cv=folds,method="predict_proba")
    issues=find_label_issues(lab,probabilities,return_indices_ranked_by="self_confidence",n_jobs=1)
    result=[]
    for i in issues.tolist():
        best=probabilities[i].argmax()
        result.append({"group":names[i],"input_row_1_based_or_image_id":rows[i],"given":labels[i],"suspected_class":str(classes[best]),
                       "self_confidence":float(probabilities[i,lab[i]]),"alternative_confidence":float(probabilities[i,best])})
    return {"n":len(labels),"groups":len(set(groups)), "folds":[len(b) for a,b in folds],
            "oof_accuracy":float(np.mean(probabilities.argmax(1)==lab)), "flagged":len(result),"finite":bool(np.isfinite(probabilities).all()),
            "flags":result}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["retrain","labels"])
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=retrain() if args.mode=="retrain" else label_issues()
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
