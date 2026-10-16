"""Local thin-filament development experiment; never submits artifacts."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import joblib
from sklearn.ensemble import ExtraTreesRegressor
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'solar-research-v5'))
import research as r
from runtime import component_features
from organizer_metric import stats_from_iou,pool

def hysteresis(prob,disk,low,high=.65):
    """Keep low-confidence connected structures only when they contain a seed."""
    allowed=(prob>=low)&(ndi.distance_transform_edt(disk)>8)
    labels,n=ndi.label(allowed,np.ones((3,3)))
    sizes=np.bincount(labels.ravel(),minlength=n+1)
    seeded=np.unique(labels[(prob>=high)&allowed])
    keep=seeded[(sizes[seeded]>=8)&(sizes[seeded]<disk.sum()*.05)]
    keep=keep[keep!=0];mapping=np.zeros(n+1,np.int32)
    mapping[keep]=np.arange(1,len(keep)+1)
    return np.repeat(np.repeat(mapping[labels],2,0),2,1)

def matrices(d,name,labels):
    pred=r.encode_labels(labels);out=[]
    for a,gt in d.truth(name):
        m=np.asarray(r.coco.iou(gt,pred,[0]*len(pred)),np.float32) if gt and pred else np.zeros((len(gt),len(pred)),np.float32)
        thin=[]
        for g in gt:
            mask=r.coco.decode(g).astype(bool)
            yy,xx=np.nonzero(mask);per=(mask&~ndi.binary_erosion(mask)).sum()
            thin.append(bool(len(xx) and 2*len(xx)/max(per,1)<=8 and max(np.ptp(xx),np.ptp(yy))>=40))
        out.append((a,m,np.array(thin,bool)))
    return out

def score(records,model,t):
    rows=[]
    for name,x,ms in records:
        keep=model.predict(x)>=t if len(x) else np.zeros(0,bool)
        for a,m,thin in ms:
            selected=m[:,keep];hit=(selected>.5).any(axis=1)
            rows.append({'file':name,'annotation_set':a,**stats_from_iou(selected),
                         'thin_tp':int((hit&thin).sum()),'thin_fn':int((~hit&thin).sum())})
    return rows

def totals(rows):
    ans=pool(rows);tp=sum(x['thin_tp'] for x in rows);fn=sum(x['thin_fn'] for x in rows)
    return {**ans,'recall':ans['tp']/max(ans['tp']+ans['fn'],1),'thin_tp':tp,'thin_fn':fn,'thin_recall':tp/max(tp+fn,1)}

def records(d,names,low):
    result=[]
    for i,name in enumerate(names,1):
        im,_,disk=d.arrays(name);p=r.probabilities(d,name,'rf')
        labels=hysteresis(p,disk,low)
        result.append((name,component_features(labels,p,im,disk),matrices(d,name,labels)))
        print(low,i,len(names),flush=True)
    return result

def main():
    d=r.Dataset();old=r.read(r.HERE/'component-protocol.json')
    protocol={'train':old['train'],'tune':old['tune'],'assessment':d.protocol['split']['assessment'],
      'low_thresholds':[.35,.45,.5],'seed_threshold':.65,'min_area_1024':8,
      'quality_thresholds':[.05,.1,.15,.2,.3,.4,.5],
      'thin_definition':'Evaluation only: 2*area/inner-boundary-pixels <= 8 and bounding-box longest span >= 40, at 2048 resolution. Width proxy, not anatomical truth.',
      'selection':'Maximum tuning PQ among trials with thin recall >= baseline and overall recall >= baseline; otherwise retain baseline.',
      'limitations':'Reuses development data previously inspected; not independent or blind. No leaderboard result.',
      'source_sha256':r.sha(__file__),'pixel_model_sha256':r.sha(r.HERE/'pixel-forest.joblib')}
    if (HERE/'protocol.json').exists():raise FileExistsError('Preserve experiment')
    r.write(HERE/'protocol.json',protocol)
    import component_followup as cf
    baseline_model=joblib.load(r.HERE/'component-filter.joblib')
    def baseline(names):
        rr=[]
        for name in names:
            labels,x=cf.candidate(d,name);rr.append((name,x,matrices(d,name,labels)))
        return score(rr,baseline_model,.2)
    base_tune=totals(baseline(protocol['tune']));trials=[]
    for low in protocol['low_thresholds']:
        train=records(d,protocol['train'],low);xx=[];yy=[]
        for name,x,ms in train:
            for _,m,_ in ms:
                if len(m):xx.extend(x);yy.extend((m.max(axis=0)>.5).astype(float))
        model=ExtraTreesRegressor(n_estimators=64,max_depth=10,min_samples_leaf=5,random_state=20261016,n_jobs=4)
        model.fit(np.asarray(xx),np.asarray(yy));joblib.dump(model,HERE/f'filter-{low}.joblib',compress=3)
        tune=records(d,protocol['tune'],low)
        for t in protocol['quality_thresholds']:
            trials.append({'low':low,'quality':t,**totals(score(tune,model,t))})
    eligible=[x for x in trials if x['thin_recall']>=base_tune['thin_recall'] and x['recall']>=base_tune['recall'] and x['pq']>base_tune['pq']]
    best=max(eligible,key=lambda x:x['pq']) if eligible else None
    r.write(HERE/'frozen.json',{'baseline_tune':base_tune,'trials':trials,'selected':best,'protocol_sha256':r.sha(HERE/'protocol.json')})
    print('FROZEN',best,flush=True)
    base_rows=baseline(protocol['assessment']);out={'baseline':totals(base_rows),'baseline_rows':base_rows,'selected':best}
    if best:
        model=joblib.load(HERE/f"filter-{best['low']}.joblib")
        rr=score(records(d,protocol['assessment'],best['low']),model,best['quality'])
        out.update(candidate=totals(rr),candidate_rows=rr)
    r.write(HERE/'assessment.json',out);print('RESULT',out['baseline'],out.get('candidate'),flush=True)

if __name__=='__main__':main()
