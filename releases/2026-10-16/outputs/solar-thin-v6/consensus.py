"""Exploratory third hypothesis: require forest support for thin U-Net additions."""
import experiment as e
from supplement import proposals
import component_followup as cf
import joblib,numpy as np
r=e.r;HERE=e.HERE/'consensus';HERE.mkdir(exist_ok=True)

def candidate(d,name,model):
    labels,x=cf.candidate(d,name);keep=np.flatnonzero(model.predict(x)>=.2)+1 if len(x) else []
    mapping=np.zeros(labels.max()+1,np.int32);mapping[keep]=np.arange(1,len(keep)+1);base=mapping[labels]
    _,_,disk=d.arrays(name);forest=r.probabilities(d,name,'rf');unet=r.probabilities(d,name,'unet900')
    proposed,items=proposals(unet,disk,base,.7);low=proposed[::2,::2]
    ids=[k for k,elong,width in items if elong>=2 and width<=4]
    scores=np.array([forest[low==k].mean() for k in ids],np.float32)
    mm=np.zeros(proposed.max()+1,np.int32);mm[ids]=np.arange(1,len(ids)+1)+base.max()
    return base+mm[proposed],np.r_[np.ones(base.max()),scores],int(base.max())

def evaluate(records,t):
    out=[]
    for name,values,nb,ms in records:
        keep=np.arange(len(values))<nb if t is None else values>=t
        for a,m,thin in ms:
            selected=m[:,keep];hit=(selected>.5).any(axis=1)
            out.append({'file':name,'annotation_set':a,**e.stats_from_iou(selected),'thin_tp':int((hit&thin).sum()),'thin_fn':int((~hit&thin).sum())})
    return out

def main():
    d=r.Dataset();old=r.read(r.HERE/'component-protocol.json');model=joblib.load(r.HERE/'component-filter.joblib')
    protocol={'tune':old['tune'],'assessment':d.protocol['split']['assessment'],'forest_mean_thresholds':[.1,.2,.3,.4,.5,.6,.7],
       'selection':'Maximum tune PQ strictly above baseline with thin recall strictly above baseline. Prefer larger support threshold on ties. Otherwise baseline.',
       'reason':'Third exploratory hypothesis after unconfirmed additions yielded +1 thin TP and +16 FP on already-used assessment. No individual assessment masks used to choose support thresholds.',
       'limitations':'Assessment reused after two hypotheses; strongly exploratory development, not independent validation.',
       'source_sha256':r.sha(__file__),'supplement_params':[.7,2,4]}
    if (HERE/'protocol.json').exists():raise FileExistsError('Preserve experiment')
    r.write(HERE/'protocol.json',protocol)
    def records(names):
        out=[]
        for i,name in enumerate(names,1):
            labels,values,nb=candidate(d,name,model);out.append((name,values,nb,e.matrices(d,name,labels)));print('Consensus',i,len(names),flush=True)
        return out
    tune=records(protocol['tune']);baseline=e.totals(evaluate(tune,None));trials=[{'threshold':t,**e.totals(evaluate(tune,t))} for t in protocol['forest_mean_thresholds']]
    valid=[x for x in trials if x['pq']>baseline['pq'] and x['thin_recall']>baseline['thin_recall']]
    best=max(valid,key=lambda x:(x['pq'],x['threshold'])) if valid else None
    r.write(HERE/'frozen.json',{'baseline':baseline,'trials':trials,'selected':best,'protocol_sha256':r.sha(HERE/'protocol.json')});print('FROZEN CONSENSUS',best,flush=True)
    if best:
        rec=records(protocol['assessment']);rows={'baseline':evaluate(rec,None),'consensus':evaluate(rec,best['threshold'])}
        result={'results':{k:e.totals(v) for k,v in rows.items()},'rows':rows}
        r.write(HERE/'assessment.json',result);print('CONSENSUS RESULT',result['results'],flush=True)
if __name__=='__main__':main()
