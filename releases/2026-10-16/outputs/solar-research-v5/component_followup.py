"""Tuning-only learned instance rejection over high-recall pixel-forest masks."""
import sys,itertools,time
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import ExtraTreesRegressor
import joblib
from research import *
from organizer_metric import stats_from_iou

NAMES=['log_area','elongation','extent','compactness','mean_probability','p90_probability','min_probability','std_probability','mean_normalized_intensity','p10_normalized_intensity','std_intensity','disk_gap','mean_fine_contrast','max_fine_contrast']
def candidate(d,name):
    im,_,disk=d.arrays(name);prob=probabilities(d,name,'rf');labels=instances(prob,disk,.55,20,1)
    low=labels[::2,::2];ch=channels(im,disk);gap=ndi.distance_transform_edt(disk)/1024;rows=[]
    for k,box in enumerate(ndi.find_objects(low),1):
        mask=low[box]==k;y,x=np.nonzero(mask);area=len(x);eigen=np.maximum(np.linalg.eigvalsh(np.cov(np.stack([y,x]),bias=True)),0)
        elong=np.sqrt((eigen[-1]+1)/(eigen[0]+1));perimeter=(mask&~ndi.binary_erosion(mask)).sum();h,w=mask.shape
        q=prob[box][mask];v=ch[0][box][mask];fine=ch[1][box][mask]
        rows.append([np.log1p(area),np.log(elong),area/(h*w),np.log1p(perimeter**2/(4*np.pi*area)),q.mean(),np.percentile(q,90),q.min(),q.std(),v.mean(),np.percentile(v,10),v.std(),gap[box][mask].mean(),fine.mean(),fine.max()])
    return labels,np.asarray(rows,np.float32).reshape(-1,len(NAMES))

def record(d,name):
    labels,x=candidate(d,name);pred=encode_labels(labels);matrices=[]
    for a,gt in d.truth(name):
        if gt:
            matrix=np.asarray(coco.iou(gt,pred,[0]*len(pred)),np.float32) if pred else np.zeros((len(gt),0),np.float32)
            matrices.append((a,matrix))
    return x,matrices

def scores(records,model,t):
    rows=[]
    for name,x,matrices in records:
        keep=model.predict(x)>=t if len(x) else np.zeros(0,bool)
        for a,m in matrices:rows.append({'file':name,'annotation_set':a,**stats_from_iou(m[:,keep])})
    return rows

def fit():
    d=Dataset();names=d.protocol['split']['tune'];days=sorted({n[:8] for n in names},key=lambda v:hashlib.sha256(('solar-v5-instance-followup-'+v).encode()).hexdigest())
    train=[n for n in names if n[:8] in set(days[:32])];valid=[n for n in names if n[:8] not in set(days[:32])]
    assert not (HERE/'assessment.json').exists(),'Freeze this follow-up before opening assessment'
    protocol={'reason':'Forest improves tuning PQ but still produces many false instances. Add a learned instance-quality filter before assessment.','train':train,'tune':valid,'origin':'Original 55-image tuning pool reused and repartitioned for development. Pixel forest settings were selected on that entire pool; this is NOT independent validation.','candidate':{'threshold':.55,'min_area':20,'close':1},'features':NAMES,'model':{'n_estimators':64,'max_depth':10,'min_samples_leaf':5,'random_state':20261015},'thresholds':[0,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9],'source_sha256':sha(__file__),'pixel_model_sha256':sha(HERE/'pixel-forest.joblib'),'assessment_used_for_training_or_selection':False}
    if (HERE/'component-protocol.json').exists():raise FileExistsError('Keep existing follow-up')
    write(HERE/'component-protocol.json',protocol);xx=[];yy=[]
    for k,n in enumerate(train,1):
        x,ms=record(d,n)
        for a,m in ms:
            xx.extend(x);yy.extend((m.max(axis=0)>.5).astype(np.float32))
        print('Component train',k,flush=True)
    model=ExtraTreesRegressor(**protocol['model'],n_jobs=4);model.fit(np.asarray(xx),np.asarray(yy));joblib.dump(model,HERE/'component-filter.joblib',compress=3)
    records=[]
    for k,n in enumerate(valid,1):
        x,ms=record(d,n);records.append((n,x,ms));print('Component tune',k,flush=True)
    trials=[{'threshold':t,**pool(scores(records,model,t))} for t in protocol['thresholds']]
    best=max(trials,key=lambda r:r['pq']);rows=scores(records,model,best['threshold'])
    reference={m:pool(r for r in read(HERE/(m+'-tuning.json'))['rows'] if r['file'] in valid) for m in ['rf']}
    write(HERE/'component-frozen.json',{'threshold':best['threshold'],'trials':trials,'selected':best,'same_validation_rf_reference':reference,'protocol_sha256':sha(HERE/'component-protocol.json'),'model_sha256':sha(HERE/'component-filter.joblib'),'rows':rows})
    print('Component selected',best,'RF reference',reference,flush=True)

def assess_extra():
    d=Dataset();frozen=read(HERE/'component-frozen.json');model=joblib.load(HERE/'component-filter.joblib');rows=[]
    assert frozen['model_sha256']==sha(HERE/'component-filter.joblib')
    for k,name in enumerate(d.protocol['split']['assessment'],1):
        x,ms=record(d,name);rows.extend(scores([(name,x,ms)],model,frozen['threshold']));print('Component assessment',k,flush=True)
    t=pool(rows);t.update(precision=t['tp']/max(t['tp']+t['fp'],1),recall=t['tp']/max(t['tp']+t['fn'],1))
    base=read(HERE/'assessment.json');baseline=base['rows']['v3'];days=sorted({r['file'][:8] for r in rows});rng=np.random.default_rng(20261015);boot=[]
    groups={day:[r for r in rows if r['file'][:8]==day] for day in days};bg={day:[r for r in baseline if r['file'][:8]==day] for day in days}
    for _ in range(2000):
        draw=rng.choice(days,len(days),replace=True);boot.append(pool(r for day in draw for r in groups[day])['pq']-pool(r for day in draw for r in bg[day])['pq'])
    write(HERE/'component-assessment.json',{'results':t,'rows':rows,'paired_gain_95_date_cluster_interval':np.quantile(boot,[.025,.975]).tolist(),'frozen':frozen,'independent_validation':False,'submitted':False})
    print('Component result',t,flush=True)

if __name__=='__main__':
    if sys.argv[1]=='fit':fit()
    elif sys.argv[1]=='assess':assess_extra()
