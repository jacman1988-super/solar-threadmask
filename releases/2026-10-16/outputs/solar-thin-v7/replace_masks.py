"""Learned mask substitution: change geometry, retain instance count."""
import precision as p
import joblib,numpy as np,itertools
r=p.r;e=p.e;HERE=p.HERE/'replacement';HERE.mkdir(exist_ok=True)

def choose(pair,qbase,qunet,gain,quality,min_iou):
    # Every proposed mask may touch exactly one original instance.
    unique=(pair>0).sum(axis=0)==1
    mapping=np.full(len(qbase),-1,int)
    for k in range(len(qbase)):
        ids=np.flatnonzero(unique&(pair[k]>=min_iou)&(qunet>=quality)&(qunet-qbase[k]>=gain))
        if len(ids):mapping[k]=ids[np.argmax(qunet[ids])]
    return mapping

def score(records,params):
    rows=[]
    for rec in records:
        name,pair,qbase,qunet,ms,bms=rec
        switch=choose(pair,qbase,qunet,*params) if params is not None else np.full(len(qbase),-1,int)
        for (a,m,thin),(a2,b,thin2) in zip(ms,bms):
            assert a==a2 and np.array_equal(thin,thin2)
            iou=b.copy()
            for k,j in enumerate(switch):
                if j>=0:iou[:,k]=m[:,j]
            hit=(iou>.5).any(axis=1);rows.append({'file':name,'annotation_set':a,**e.stats_from_iou(iou),'thin_tp':int((hit&thin).sum()),'thin_fn':int((~hit&thin).sum()),'replaced':int((switch>=0).sum())})
    return rows

def main():
    d=r.Dataset();original=r.read(r.HERE/'component-protocol.json');baseline_model=joblib.load(r.HERE/'component-filter.joblib');model=joblib.load(p.HERE/'precision-filter.joblib')
    tuning_cache=p.CACHE/'tune.joblib';assessment_cache=p.CACHE/'assessment.joblib'
    protocol={'tune':original['tune'],'assessment':d.protocol['split']['assessment'],'grid':list(itertools.product([0,.05,.1,.2],[.2,.4,.6],[.25,.5])),
      'selection':'Tune thin TP > baseline, FP <= baseline, total TP >= baseline, PQ >= baseline; max PQ. Otherwise baseline.',
      'rule':'U-Net .7 area8; use trained 22-feature precision model on both masks. Require gain, absolute quality, overlap >= minimum IoU and zero overlap with any other baseline instance. At most one replacement per original instance; no new instance IDs.',
      'reason':'Aggregate diagnosis on previously used assessment: 17 missed thin objects matched by U-Net, only 1 allowed by disjoint thin-addition rules. Substitution removes the restrictive geometry/novelty rules without increasing instance count.',
      'limitations':'Fifth exploratory hypothesis this turn; assessment diagnostic informed design. Repeated development evaluation, not independent.',
      'source_sha256':r.sha(__file__),'model_sha256':r.sha(p.HERE/'precision-filter.joblib'),'tune_cache_sha256':r.sha(tuning_cache),'assessment_cache_sha256':r.sha(assessment_cache)}
    if (HERE/'protocol.json').exists():raise FileExistsError('Preserve run')
    r.write(HERE/'protocol.json',protocol)
    def records(cached):
        out=[]
        for i,rec in enumerate(cached,1):
            name=rec['name'];base=p.baseline(d,name,baseline_model);im,_,disk=d.arrays(name);rf=r.probabilities(d,name,'rf');unet=r.probabilities(d,name,'unet900');labels=r.instances(unet,disk,.7,8,0)
            bx=p.features(base,rf,unet,im,disk);qbase=model.predict(bx) if len(bx) else np.zeros(0);qunet=model.predict(rec['x']) if len(rec['x']) else np.zeros(0)
            encbase=r.encode_labels(base);encunet=r.encode_labels(labels)
            pair=np.asarray(r.coco.iou(encbase,encunet,[0]*len(encunet)),np.float32) if encbase and encunet else np.zeros((len(encbase),len(encunet)),np.float32)
            assert all(b.shape[1]==len(qbase) and m.shape[1]==len(qunet) for (_,m,_),(_,b,_) in zip(rec['ms'],rec['base_ms']))
            out.append((name,pair,qbase,qunet,rec['ms'],rec['base_ms']));print('Replace',i,len(cached),flush=True)
        return out
    tune=records(joblib.load(tuning_cache)[.7]);base=e.totals(score(tune,None));trials=[{'params':g,**e.totals(score(tune,g))} for g in protocol['grid']]
    valid=[v for v in trials if v['thin_tp']>base['thin_tp'] and v['fp']<=base['fp'] and v['tp']>=base['tp'] and v['pq']>=base['pq']]
    best=max(valid,key=lambda v:v['pq']) if valid else None
    r.write(HERE/'frozen.json',{'baseline':base,'trials':trials,'selected':best,'protocol_sha256':r.sha(HERE/'protocol.json')});print('REPLACE FROZEN',best,flush=True)
    if best:
        rec=records(joblib.load(assessment_cache));joblib.dump(rec,p.CACHE/'replacement-assessment.joblib',compress=3)
        rows={'baseline':score(rec,None),'candidate':score(rec,best['params'])};result={'results':{k:e.totals(v) for k,v in rows.items()},'rows':rows};r.write(HERE/'assessment.json',result);print('REPLACE RESULT',result['results'],flush=True)
if __name__=='__main__':main()
