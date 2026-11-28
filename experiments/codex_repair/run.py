"""Fixed, CPU-only constrained update. Never loads T10/T11 and never replaces model.pth."""
from __future__ import annotations
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools'))
from decision import new_vegetable_logits
from compare_on_dev import development_sets,evaluate,answers_of,right,cached_inputs_apply
WORK=HERE/'work'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def fit(teacher,oldX,newX,newy,weight):
    head=teacher['new_vegetable_head']
    with torch.no_grad():
        oldz=new_vegetable_logits(oldX,head)
        newz=new_vegetable_logits(newX,head)
        targets=oldz.softmax(1)
    dw=torch.zeros((2,oldX.shape[1]),requires_grad=True)
    db=torch.zeros(2,requires_grad=True)
    optimizer=torch.optim.LBFGS([dw,db],max_iter=150,line_search_fn='strong_wolfe')
    calls=0
    def components():
        zo=oldz.clone();zn=newz.clone()
        zo[:,12:14]=zo[:,12:14]+oldX@dw.T+db
        zn[:,12:14]=zn[:,12:14]+newX@dw.T+db
        ce=F.cross_entropy(zn,newy)
        kl=F.kl_div(zo.log_softmax(1),targets,reduction='batchmean')
        l2=.001*(dw.square().sum()+db.square().sum())
        return ce,kl,l2
    def closure():
        nonlocal calls
        optimizer.zero_grad()
        ce,kl,l2=components();loss=ce+weight*kl+l2
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
        loss.backward();calls+=1
        return loss
    optimizer.step(closure)
    ce,kl,l2=components()
    student=copy.deepcopy(teacher)
    student['new_vegetable_head']['coef'][12:14]+=dw.detach()
    student['new_vegetable_head']['intercept'][12:14]+=db.detach()
    return student,{'ce_new':float(ce.detach()),'kl_old':float(kl.detach()),'l2':float(l2.detach()),'closure_calls':calls,'optimizer_iterations':optimizer.state[dw].get('n_iter'), 'max_iter':150}

def gates(base,cand):
    b=base['_all_labelled'];c=cand['_all_labelled']
    rejected=lambda s:sum(s[n]['turned_away'] for n in ('fruit 374','everyday things 400'))
    return {
        'total_correct_not_down':c['right']>=b['right'],
        'strict_development_gain':c['right']>b['right'],
        'fresh_potato_not_down':c['classes']['potato_fresh']['right']>=b['classes']['potato_fresh']['right'],
        'rotten_potato_not_down':c['classes']['potato_rotten']['right']>=b['classes']['potato_rotten']['right'],
        'rotten_called_fresh_not_up':c['rotten_called_fresh']<=b['rotten_called_fresh'],
        'shown_rotten_called_fresh_not_up':c['rotten_called_fresh_shown']<=b['rotten_called_fresh_shown'],
        'fruit_and_everyday_rejection_not_down':rejected(cand)>=rejected(base),
        'lvis_retention_not_down':cand['real-scene vegetable crops 192']['turned_away']<=base['real-scene vegetable crops 192']['turned_away'],
    }

def main():
    torch.set_num_threads(4);torch.manual_seed(0)
    WORK.mkdir(exist_ok=True)
    result_path=HERE/'result.json'
    if result_path.exists():raise SystemExit('Experiment already has a result; refusing automatic rerun')
    model=ROOT/'model.pth'
    teacher=torch.load(model,map_location='cpu',weights_only=True)
    if teacher.get('model_version',teacher.get('version'))!='TV-13':raise SystemExit('Expected TV-13 baseline')
    orig=ROOT/'experiments/new_vegetables/v7/head_data.npz'
    added=ROOT/'experiments/sweetpotato_more/retrain_b/work/head_data_b.npz'
    o=np.load(orig);a=np.load(added);n=len(o['train_y'])
    if not np.array_equal(o['train_X'],a['train_X'][:n]) or not np.array_equal(o['train_y'],a['train_y'][:n]):
        raise SystemExit('Original training prefix mismatch')
    newy=a['train_y'][n:].astype(int)
    if len(newy)!=259 or not np.isin(newy,[12,13]).all():raise SystemExit('Unexpected new-training data')
    oldX=torch.tensor(o['train_X'],dtype=torch.float32)
    newX=torch.tensor(a['train_X'][n:],dtype=torch.float32)
    y=torch.tensor(newy,dtype=torch.long)
    sets=development_sets();base=evaluate(teacher,sets)
    report={'inputs':{str(p.relative_to(ROOT)):digest(p) for p in (model,orig,added)},'cpu_only':True,'new_training_rows':len(newy),'baseline':base,'candidates':{},'selected':None,'blind_tests_read':False}
    for weight in (1,4,16):
        start=time.monotonic();name=f'C{weight}'
        candidate,training=fit(teacher,oldX,newX,y,weight)
        reasons=cached_inputs_apply(teacher,candidate)
        if reasons:raise RuntimeError(reasons)
        metrics=evaluate(candidate,sets);checks=gates(base,metrics)
        changes=[]
        for dataset,(b,X,labels) in sets.items():
            if labels is None:continue
            before,_=answers_of(teacher,b,X);after,_=answers_of(candidate,b,X)
            for i,(label,p,q) in enumerate(zip(labels,before,after)):
                if p!=q:changes.append({'dataset':dataset,'row':i,'truth':label,'before':p,'after':q,'lost':right(label,p) and not right(label,q),'gained':not right(label,p) and right(label,q)})
        ck=WORK/f'{name}.pth';torch.save(candidate,ck)
        report['candidates'][name]={'lambda':weight,'training':training,'seconds':round(time.monotonic()-start,2),'gates':checks,'pass':all(checks.values()),'metrics':metrics,'changes':changes,'checkpoint_sha256':digest(ck)}
        c=metrics['_all_labelled']
        print(json.dumps({'candidate':name,'right':c['right'],'potato_fresh':c['classes']['potato_fresh']['right'],'potato_rotten':c['classes']['potato_rotten']['right'],'rotten_called_fresh':c['rotten_called_fresh'],'gates':checks,'training':training},ensure_ascii=True),flush=True)
        temp=result_path.with_suffix('.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(result_path)
    passing=[(name,r) for name,r in report['candidates'].items() if r['pass']]
    if passing:
        report['selected']=max(passing,key=lambda nr:(nr[1]['metrics']['_all_labelled']['right'],-nr[1]['metrics']['_all_labelled']['rotten_called_fresh'],nr[1]['lambda']))[0]
    if digest(model)!=report['inputs']['model.pth']:raise RuntimeError('Baseline changed during experiment')
    result_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SELECTED',report['selected'],flush=True)
if __name__=='__main__':main()
