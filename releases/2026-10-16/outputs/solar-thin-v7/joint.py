"""Joint precision budget: remove weak baseline instances while adding confirmed thin ones."""
import sys
from pathlib import Path
import numpy as np,joblib
from scipy import ndimage as ndi
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'solar-thin-v6'))
import experiment as e
from supplement import proposals
import component_followup as cf
r=e.r;HERE=Path(__file__).resolve().parent/'joint';HERE.mkdir(exist_ok=True)
def record(d,name,model):
    labels,x=cf.candidate(d,name);quality=model.predict(x) if len(x) else np.zeros(0)
    keep=np.flatnonzero(quality>=.2)+1;mm=np.zeros(labels.max()+1,np.int32);mm[keep]=np.arange(1,len(keep)+1);base=mm[labels]
    _,_,disk=d.arrays(name);forest=r.probabilities(d,name,'rf');unet=r.probabilities(d,name,'unet900')
    proposed,items=proposals(unet,disk,base,.7);low=proposed[::2,::2];ids=[k for k,elong,width in items if elong>=2 and width<=4]
    support=np.array([forest[low==k].mean() for k in ids],np.float32);mapping=np.zeros(proposed.max()+1,np.int32);mapping[ids]=np.arange(1,len(ids)+1)+base.max()
    return name,quality[keep-1],support,e.matrices(d,name,base+mapping[proposed])

def score(records,q,s):
    out=[]
    for name,quality,support,ms in records:
        keep=np.r_[quality>=q,np.zeros(len(support),bool) if s is None else support>=s]
        for a,m,thin in ms:
            selected=m[:,keep];hit=(selected>.5).any(axis=1)
            out.append({'file':name,'annotation_set':a,**e.stats_from_iou(selected),'thin_tp':int((hit&thin).sum()),'thin_fn':int((~hit&thin).sum())})
    return out

def main():
    d=r.Dataset();old=r.read(r.HERE/'component-protocol.json');model=joblib.load(r.HERE/'component-filter.joblib')
    protocol={'tune':old['tune'],'assessment':d.protocol['split']['assessment'],'baseline_quality':[.2,.205,.21,.225,.25,.275,.3],'thin_rf_support':[.6,.7,.8],
      'selection':'Require tune thin TP > baseline, FP <= baseline, total TP >= baseline and PQ >= baseline. Max PQ, tie higher component quality.',
      'reason':'New user requests no additional false detections. Jointly calibrate rejection of weak baseline instances and fixed thin candidate additions; no assessment outcomes from this turn used.',
      'limitations':'Repeated tuning and assessment on previously inspected development data; no independent guarantee of unchanged FP.', 'source_sha256':r.sha(__file__)}
    if (HERE/'protocol.json').exists():raise FileExistsError('Preserve run')
    r.write(HERE/'protocol.json',protocol)
    def records(names):
        out=[]
        for i,name in enumerate(names,1):out.append(record(d,name,model));print('Joint',i,len(names),flush=True)
        return out
    rec=records(protocol['tune']);base=e.totals(score(rec,.2,None));trials=[]
    for q in protocol['baseline_quality']:
        for s in protocol['thin_rf_support']:trials.append({'quality':q,'support':s,**e.totals(score(rec,q,s))})
    valid=[v for v in trials if v['thin_tp']>base['thin_tp'] and v['fp']<=base['fp'] and v['tp']>=base['tp'] and v['pq']>=base['pq']]
    best=max(valid,key=lambda v:(v['pq'],v['quality'])) if valid else None
    r.write(HERE/'frozen.json',{'baseline':base,'trials':trials,'selected':best,'protocol_sha256':r.sha(HERE/'protocol.json')});print('JOINT FROZEN',best,flush=True)
    if best:
        rec=records(protocol['assessment']);joblib.dump(rec,HERE.parents[2].parent/'work/solar-thin-v7-cache/joint-assessment.joblib',compress=3)
        rows={'baseline':score(rec,.2,None),'candidate':score(rec,best['quality'],best['support'])};result={'results':{k:e.totals(v) for k,v in rows.items()},'rows':rows}
        r.write(HERE/'assessment.json',result);print('JOINT RESULT',result['results'],flush=True)
if __name__=='__main__':main()
