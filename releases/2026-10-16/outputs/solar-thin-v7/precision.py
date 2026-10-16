"""Learn precision of thin U-Net proposals; require no extra tuning FP."""
import sys,json,itertools
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import joblib
from sklearn.ensemble import ExtraTreesRegressor
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'solar-thin-v6'))
import experiment as e
from runtime import component_features
import component_followup as cf
r=e.r;CACHE=ROOT/'work/solar-thin-v7-cache';CACHE.mkdir(exist_ok=True)

def features(labels,rf,unet,im,disk):
    x=component_features(labels,rf,im,disk);low=labels[::2,::2];extra=[]
    for k,box in enumerate(ndi.find_objects(low),1):
        mask=low[box]==k;u=unet[box][mask];f=rf[box][mask]
        extra.append([u.mean(),u.min(),np.percentile(u,10),u.max(),u.std(),(f>=.5).mean(),(f>=.7).mean(),(u>=.8).mean()])
    return np.c_[x,np.asarray(extra,np.float32).reshape(-1,8)]

def baseline(d,name,model):
    labels,x=cf.candidate(d,name);keep=np.flatnonzero(model.predict(x)>=.2)+1 if len(x) else []
    mapping=np.zeros(labels.max()+1,np.int32);mapping[keep]=np.arange(1,len(keep)+1)
    return mapping[labels]

def shape_indices(labels):
    low=labels[::2,::2];keep=[]
    for k,box in enumerate(ndi.find_objects(low),1):
        mask=low[box]==k;y,x=np.nonzero(mask);eig=np.maximum(np.linalg.eigvalsh(np.cov(np.stack([y,x]),bias=True)),0)
        elong=np.sqrt((eig[-1]+1)/(eig[0]+1));per=(mask&~ndi.binary_erosion(mask)).sum();width=2*mask.sum()/max(per,1)
        if elong>=2 and width<=4 and max(np.ptp(x),np.ptp(y))>=20:keep.append(k-1)
    return np.asarray(keep,int)

def make(d,name,t,base):
    im,_,disk=d.arrays(name);rf=r.probabilities(d,name,'rf');unet=r.probabilities(d,name,'unet900')
    labels=r.instances(unet,disk,t,8,0);x=features(labels,rf,unet,im,disk);ms=e.matrices(d,name,labels)
    occupied=ndi.binary_dilation(base[::2,::2]>0,iterations=2);low=labels[::2,::2];eligible=shape_indices(labels)
    eligible=np.array([k for k in eligible if not (occupied&(low==k+1)).any()],int)
    base_ms=e.matrices(d,name,base)
    return {'name':name,'x':x,'ms':ms,'eligible':eligible,'base_ms':base_ms}

def rows(records,model,quality):
    out=[]
    for rec in records:
        x=rec['x'];chosen=rec['eligible']
        if len(chosen):chosen=chosen[model.predict(x[chosen])>=quality]
        for (a,m,thin),(a2,b,thin2) in zip(rec['ms'],rec['base_ms']):
            assert a==a2 and np.array_equal(thin,thin2)
            iou=np.c_[b,m[:,chosen]];hit=(iou>.5).any(axis=1)
            out.append({'file':rec['name'],'annotation_set':a,**e.stats_from_iou(iou),'thin_tp':int((hit&thin).sum()),'thin_fn':int((~hit&thin).sum())})
    return out

def baseline_rows(records):
    out=[]
    for rec in records:
        for a,m,thin in rec['base_ms']:
            hit=(m>.5).any(axis=1);out.append({'file':rec['name'],'annotation_set':a,**e.stats_from_iou(m),'thin_tp':int((hit&thin).sum()),'thin_fn':int((~hit&thin).sum())})
    return out

def main():
    d=r.Dataset();old=r.read(r.HERE/'component-protocol.json');base_model=joblib.load(r.HERE/'component-filter.joblib')
    protocol={'train':old['train'],'tune':old['tune'],'assessment':d.protocol['split']['assessment'],
      'pixel_thresholds':[.55,.7],'quality_thresholds':[.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9,.95],
      'target':'Every annotation set must have a GT match at IoU > .5. One row per component, no annotation metadata as features.',
      'model':{'n_estimators':128,'max_depth':8,'min_samples_leaf':2,'random_state':20261016,'n_jobs':4},
      'selection':'Tune thin TP strictly increases, FP <= baseline, total TP >= baseline, PQ >= baseline. Max PQ, then higher quality threshold. Otherwise retain baseline.',
      'features':cf.NAMES+['unet_mean','unet_min','unet_p10','unet_max','unet_std','rf_fraction_05','rf_fraction_07','unet_fraction_08'],
      'limitations':'Reuses 36 training and 19 tuning images from original tuning pool and previously inspected 106-image assessment. Exploratory development; not blind.',
      'source_sha256':r.sha(__file__),'no_publication_or_submission':True}
    if (HERE/'protocol.json').exists():raise FileExistsError('Preserve run')
    r.write(HERE/'protocol.json',protocol);xx=[];yy=[];tune={}
    for t in protocol['pixel_thresholds']:
        for i,name in enumerate(protocol['train'],1):
            rec=make(d,name,t,baseline(d,name,base_model));x=rec['x'];targets=[]
            for _,m,_ in rec['ms']:targets.append((m>.5).any(axis=0))
            target=np.all(targets,axis=0).astype(float) if targets else np.zeros(len(x))
            xx.extend(x);yy.extend(target);print('Train',t,i,len(protocol['train']),flush=True)
        tune[t]=[]
        for i,name in enumerate(protocol['tune'],1):
            tune[t].append(make(d,name,t,baseline(d,name,base_model)));print('Tune',t,i,len(protocol['tune']),flush=True)
    model=ExtraTreesRegressor(**protocol['model']);model.fit(np.asarray(xx),np.asarray(yy));joblib.dump(model,HERE/'precision-filter.joblib',compress=3)
    joblib.dump(tune,CACHE/'tune.joblib',compress=3)
    base=e.totals(baseline_rows(next(iter(tune.values()))));trials=[]
    for t,q in itertools.product(protocol['pixel_thresholds'],protocol['quality_thresholds']):trials.append({'pixel_threshold':t,'quality':q,**e.totals(rows(tune[t],model,q))})
    valid=[v for v in trials if v['thin_tp']>base['thin_tp'] and v['fp']<=base['fp'] and v['tp']>=base['tp'] and v['pq']>=base['pq']]
    best=max(valid,key=lambda v:(v['pq'],v['quality'])) if valid else None
    frozen={'baseline':base,'trials':trials,'selected':best,'training_components':len(yy),'training_positives':int(sum(yy)),'model_sha256':r.sha(HERE/'precision-filter.joblib'),'protocol_sha256':r.sha(HERE/'protocol.json')}
    r.write(HERE/'frozen.json',frozen);print('FROZEN',frozen,flush=True)
    if best:
        recs=[]
        for i,name in enumerate(protocol['assessment'],1):
            recs.append(make(d,name,best['pixel_threshold'],baseline(d,name,base_model)));print('Assessment',i,len(protocol['assessment']),flush=True)
        joblib.dump(recs,CACHE/'assessment.joblib',compress=3)
        rr={'baseline':baseline_rows(recs),'candidate':rows(recs,model,best['quality'])};result={'results':{k:e.totals(v) for k,v in rr.items()},'rows':rr}
        r.write(HERE/'assessment.json',result);print('RESULT',result['results'],flush=True)
if __name__=='__main__':main()
