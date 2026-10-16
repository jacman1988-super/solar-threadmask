"""Train from segmentation masks, freeze tuning choice, then development assessment."""
import ridge as g
import numpy as np,joblib,itertools,hashlib
from sklearn.ensemble import ExtraTreesRegressor
from scipy import ndimage as ndi
from pathlib import Path
import contour as metrics
r=g.r;e=g.e;H=g.HERE;CACHE=r.ROOT/'work/solar-ridge-v8-cache';CACHE.mkdir(exist_ok=True)

def target(d,name):
    vote=np.zeros((2048,2048),np.float32);thin=np.zeros(vote.shape,bool);sets=d.truth(name)
    for _,gt in sets:
        foreground=np.zeros(vote.shape,bool)
        for encoded in gt:
            mask=r.coco.decode(encoded).astype(bool);foreground|=mask;y,x=np.nonzero(mask);boundary=(mask&~ndi.binary_erosion(mask)).sum()
            if len(x) and 2*len(x)/max(boundary,1)<=8 and max(np.ptp(x),np.ptp(y))>=40:thin|=mask
        vote+=foreground
    return vote/max(len(sets),1),thin

def main():
    d=r.Dataset();split=r.read(r.HERE/'component-protocol.json');train=sorted(d.protocol['split']['train'],key=lambda n:hashlib.sha256(('ridge-v8-'+n).encode()).hexdigest())[:60]
    protocol={'train':train,'tune':split['tune'],'assessment':d.protocol['split']['assessment'],
      'model':{'n_estimators':48,'max_depth':18,'min_samples_leaf':8,'max_features':.8,'random_state':20261016,'n_jobs':4},
      'features':'10 full-2048 image features: normalized/raw intensity, contrast sigma3/7, positive anisotropic Hessian ridge and positive-direction eigenvalue at sigma1/2/4. No low-resolution probability maps as training input.',
      'sampling':'Per image: up to 1200 thin foreground, 1200 other foreground, 1200 near-negative, 2400 other-negative pixels; target mean annotator foreground vote.',
      'grid':list(itertools.product([8,16,32],[.2,.4,.6],[0,2,4])),
      'selection':'More tune thin TP; FP <= baseline, total TP >= baseline, PQ >= baseline. Max PQ, otherwise baseline. No new instance IDs, only learned full-resolution contour correction.',
      'limitations':'Repeated development data and hypotheses; not blind or an independent generalization estimate.',
      'feature_source_sha256':r.sha(H/'ridge.py'),'harness_sha256':r.sha(__file__),'no_publication_or_submission':True}
    if (H/'protocol.json').exists():raise FileExistsError('Preserve run')
    r.write(H/'protocol.json',protocol);rng=np.random.default_rng(20261016);xx=[];yy=[]
    for i,name in enumerate(train,1):
        image=r.load_image(r.DATA/'train/train_images'/name);_,_,lowdisk=d.arrays(name);disk=np.repeat(np.repeat(lowdisk,2,0),2,1);truth,thin=target(d,name);fg=truth>0
        near=ndi.binary_dilation(fg,iterations=8)&~fg&disk
        pools=[(thin&fg,1200),(fg&~thin,1200),(near,1200),(disk&~fg&~near,2400)];indices=[]
        for mask,n in pools:
            candidates=np.flatnonzero(mask);indices.extend(rng.choice(candidates,min(n,len(candidates)),replace=False))
        indices=np.asarray(indices,int);features=g.features(image,disk).reshape(-1,10);xx.append(features[indices]);yy.append(truth.ravel()[indices]);print('Ridge train',i,len(train),flush=True)
    model=ExtraTreesRegressor(**protocol['model']);model.fit(np.concatenate(xx),np.concatenate(yy));joblib.dump(model,H/'ridge-forest.joblib',compress=3)
    r.write(H/'training.json',{'pixels':sum(len(v) for v in yy),'images':len(train),'model_sha256':r.sha(H/'ridge-forest.joblib')})
    baseline_model=joblib.load(r.HERE/'component-filter.joblib')
    def run(names,grid):
        out={'baseline':[]};out.update({str(tuple(v)):[] for v in grid})
        for i,name in enumerate(names,1):
            labels=g.old.baseline(d,name,baseline_model);widths,distance,nearest=geo=g.geometry(labels)
            image=r.load_image(r.DATA/'train/train_images'/name);_,_,lowdisk=d.arrays(name);disk=np.repeat(np.repeat(lowdisk,2,0),2,1)
            region=(widths[nearest]<=32)&(distance<=4);features=g.features(image,disk);prob=np.zeros(labels.shape,np.float32)
            if region.any():prob[region]=model.predict(features[region])
            np.save(CACHE/(name+'.npy'),prob.astype(np.float16));gt=metrics.truth(d,name);out['baseline'].extend(metrics.metrics(name,labels,gt))
            for width,t,rad in grid:out[str((width,t,rad))].extend(metrics.metrics(name,g.refine(labels,prob,width,t,rad,geo),gt))
            print('Ridge evaluate',i,len(names),flush=True)
        return out
    tune=run(protocol['tune'],[tuple(v) for v in protocol['grid']]);base=e.totals(tune['baseline']);trials=[{'params':v,**e.totals(tune[str(tuple(v))])} for v in protocol['grid']]
    valid=[v for v in trials if v['thin_tp']>base['thin_tp'] and v['fp']<=base['fp'] and v['tp']>=base['tp'] and v['pq']>=base['pq']]
    best=max(valid,key=lambda v:v['pq']) if valid else None
    r.write(H/'frozen.json',{'baseline':base,'trials':trials,'selected':best,'protocol_sha256':r.sha(H/'protocol.json'),'model_sha256':r.sha(H/'ridge-forest.joblib')});print('RIDGE FROZEN',best,flush=True)
    if best:
        rows=run(protocol['assessment'],[tuple(best['params'])]);result={'results':{k:e.totals(v) for k,v in rows.items()},'rows':rows};r.write(H/'assessment.json',result);print('RIDGE RESULT',result['results'],flush=True)
if __name__=='__main__':main()
