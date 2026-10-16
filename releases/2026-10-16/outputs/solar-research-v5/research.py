"""Local-only supervised pixel segmentation experiment; no submission operations."""
import argparse, hashlib, itertools, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=HERE.parent/'solar-threadmask'
sys.path.insert(0,str(OLD))
from pipeline import load_image, encode_annotation, encode_labels
from organizer_metric import instance_stats, pool
from pycocotools import mask as coco
DATA=ROOT/'work/solar-data/MAGFiLO_1.0_Kaggle_2026'
CACHE=ROOT/'work/solar-v5-cache'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x): Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')

def freeze():
    if (HERE/'protocol.json').exists(): raise FileExistsError('Protocol already frozen')
    old=read(OLD/'results/v3/protocol.json')['split']
    split={'train':old['train'],'tune':old['tune'],'assessment':old['unused']}
    for a,b in itertools.combinations(split,2):
        assert not {n[:8] for n in split[a]} & {n[:8] for n in split[b]}
    write(HERE/'protocol.json',{
        'split':split,'seed':20261015,'resolution':1024,
        'source_sha256':sha(__file__),
        'annotations_sha256':sha(DATA/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
        'status':'Development assessment: dates excluded from training/tuning, but previously inspected in recall-v4 and precision-guard. NOT a fresh blind test.',
        'training_target':'Mean foreground vote across annotation sets for the same image; uses only segmentation masks.',
        'rf':'ExtraTreesRegressor 48 trees, max_depth=20, min_samples_leaf=8, max_features=.8; 1200 positive, 1200 near-negative, 2400 other disk pixels/image; no class-prior correction; probability threshold selected only on tune.',
        'unet':'Small GroupNorm U-Net from scratch, base width 12, 3 input channels, 192 pixel crops; balanced foreground/background crops, rotations/flips; AdamW lr .001; BCE + soft Dice; snapshots at 300,600,900 steps; select on tune.',
        'grid':{'threshold':[.25,.4,.55,.7],'min_area':[20,60,120],'close':[0,1]},
        'metric':'Public organizer v6 pooled PQ at 2048 resolution, per annotator. Not private Kaggle score.',
        'selection':'Best tune PQ; tie follows grid order. Freeze settings before assessment. Compare both prospective model families and baseline, report all.',
        'external_actions':'No uploads, publications, paid services or leaderboard calls.'})

class Dataset:
    def __init__(self):
        self.protocol=read(HERE/'protocol.json')
        assert self.protocol['source_sha256']==sha(__file__),'Source changed after freeze'
        assert self.protocol['annotations_sha256']==sha(DATA/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
        raw=read(DATA/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
        self.infos={}; self.anns={}
        for i in raw['images']: self.infos.setdefault(i['file_name'],[]).append(i)
        for a in raw['annotations']: self.anns.setdefault(a['image_id'],[]).append(a)
        CACHE.mkdir(exist_ok=True)
    def truth(self,name):
        return [(i['id'],[encode_annotation(a,2048,2048) for a in self.anns.get(i['id'],[])]) for i in self.infos[name]]
    def arrays(self,name):
        p=CACHE/(name+'.npz')
        if p.exists():
            with np.load(p) as a: return a['image'].astype(np.float32),a['target'].astype(np.float32),a['disk'].astype(bool)
        im=np.asarray(Image.fromarray(load_image(DATA/'train/train_images'/name)).resize((1024,1024),Image.Resampling.BILINEAR),dtype=np.float32)
        lab,n=ndi.label(im>max(np.percentile(im,90)*.22,1))
        sizes=np.bincount(lab.ravel());sizes[0]=0
        disk=ndi.binary_fill_holes(lab==sizes.argmax()) if n else np.zeros(im.shape,bool)
        target=np.zeros((1024,1024),np.float32)
        for i in self.infos[name]:
            fg=np.zeros((1024,1024),bool)
            for a in self.anns.get(i['id'],[]):
                seg=a['segmentation']
                if isinstance(seg,list):
                    scaled=[[float(v)*.5 for v in polygon] for polygon in seg]
                    rle=coco.merge(coco.frPyObjects(scaled,1024,1024))
                    fg |= coco.decode(rle).astype(bool)
                else:
                    fg |= np.asarray(Image.fromarray(coco.decode(encode_annotation(a,2048,2048))).resize((1024,1024),Image.Resampling.NEAREST),bool)
            target+=fg
        target/=len(self.infos[name])
        np.savez_compressed(p,image=im.astype(np.uint8),target=target.astype(np.float16),disk=disk)
        return im,target,disk

def channels(im,disk):
    occ=ndi.gaussian_filter(disk.astype(np.float32),24)
    bg=ndi.gaussian_filter(im*disk,24)/np.maximum(occ,1e-4)
    normalized=np.clip(im/np.maximum(bg,1),0,2)
    local=ndi.gaussian_filter(im,6)
    contrast=(local-im)/np.maximum(local,1)
    return np.stack([normalized-1,np.clip(contrast,-.5,.5)*2,im/255],axis=0).astype(np.float32)

def pixel_features(im,disk):
    ch=channels(im,disk); norm=ch[0]
    fs=[*ch,ndi.distance_transform_edt(disk).astype(np.float32)/512]
    for s in [1.,2.,4.,8.,16.]:
        smooth=ndi.gaussian_filter(norm,s)
        variance=np.maximum(ndi.gaussian_filter(norm*norm,s)-smooth*smooth,0)
        hxx=ndi.gaussian_filter(norm,s,order=(0,2))*s*s
        hyy=ndi.gaussian_filter(norm,s,order=(2,0))*s*s
        hxy=ndi.gaussian_filter(norm,s,order=(1,1))*s*s
        delta=np.sqrt((hxx-hyy)**2+4*hxy*hxy)
        fs.extend([smooth,np.sqrt(variance),.5*(hxx+hyy+delta),.5*(hxx+hyy-delta)])
    return np.stack(fs,axis=-1).astype(np.float32)

def instances(prob,disk,threshold,min_area,close):
    if not 0<threshold<1 or min_area<1 or close not in [0,1]: raise ValueError('Invalid settings')
    binary=(prob>=threshold)&(ndi.distance_transform_edt(disk)>8)
    if close: binary=ndi.binary_closing(binary,iterations=close)
    labels,n=ndi.label(binary,np.ones((3,3)))
    sizes=np.bincount(labels.ravel())
    keep=np.flatnonzero((sizes>=min_area)&(sizes<disk.sum()*.05));keep=keep[keep!=0]
    mapping=np.zeros(n+1,np.int32);mapping[keep]=np.arange(1,len(keep)+1)
    full=np.asarray(Image.fromarray(mapping[labels]).resize((2048,2048),Image.Resampling.NEAREST),np.int32)
    return full

def encoded_grid(prob,disk,params):
    """Reuse geometry for identical threshold/closing pairs, without changing masks."""
    inside=ndi.distance_transform_edt(disk)>8
    cache={};limit=disk.sum()*.05
    for p in params:
        key=(p['threshold'],p['close'])
        if key not in cache:
            binary=(prob>=key[0])&inside
            if key[1]:binary=ndi.binary_closing(binary,iterations=key[1])
            labels,n=ndi.label(binary,np.ones((3,3)))
            cache[key]=(labels,n,np.bincount(labels.ravel()))
        labels,n,sizes=cache[key]
        keep=np.flatnonzero((sizes>=p['min_area'])&(sizes<limit));keep=keep[keep!=0]
        mapping=np.zeros(n+1,np.int32);mapping[keep]=np.arange(1,len(keep)+1)
        full=np.asarray(Image.fromarray(mapping[labels]).resize((2048,2048),Image.Resampling.NEAREST),np.int32)
        yield encode_labels(full)

def train_rf(d):
    from sklearn.ensemble import ExtraTreesRegressor
    import joblib
    if (HERE/'pixel-forest.joblib').exists(): raise FileExistsError('Keep fitted model')
    rng=np.random.default_rng(20261015);xx=[];yy=[];inputs=[]
    for k,name in enumerate(d.protocol['split']['train'],1):
        im,target,disk=d.arrays(name);f=pixel_features(im,disk)
        pos=target>0; near=ndi.binary_dilation(pos,iterations=12)&~pos&disk
        neg=disk&~pos&~near
        indices=[]
        for region,n in [(pos,1200),(near,1200),(neg,2400)]:
            loc=np.flatnonzero(region)
            if len(loc): indices.extend(rng.choice(loc,min(n,len(loc)),replace=False))
        idx=np.asarray(indices);xx.append(f.reshape(-1,f.shape[-1])[idx]);yy.append(target.ravel()[idx])
        inputs.append({'file':name,'sha256':sha(DATA/'train/train_images'/name)})
        print('RF sample',k,name,flush=True)
    x=np.concatenate(xx);y=np.concatenate(yy)
    model=ExtraTreesRegressor(n_estimators=48,max_depth=20,min_samples_leaf=8,max_features=.8,random_state=20261015,n_jobs=4)
    started=time.perf_counter();model.fit(x,y)
    joblib.dump(model,HERE/'pixel-forest.joblib',compress=3)
    write(HERE/'rf-training.json',{'samples':len(y),'features':x.shape[1],'fit_seconds':time.perf_counter()-started,'inputs':inputs,'protocol_sha256':sha(HERE/'protocol.json')})
    print('RF fitted',len(y),flush=True)

def probabilities(d,name,method):
    p=CACHE/(name+'.'+method+'.npy')
    if p.exists(): return np.load(p).astype(np.float32)
    im,_,disk=d.arrays(name)
    if method=='rf':
        import joblib
        model=joblib.load(HERE/'pixel-forest.joblib');f=pixel_features(im,disk)
        prob=model.predict(f.reshape(-1,f.shape[-1])).reshape(im.shape)
    else:
        from neural import predict
        prob=predict(channels(im,disk),HERE/(method+'.pt'))
    # Apply the declared cache precision on the first run too, so cached and
    # uncached threshold decisions are identical at probability boundaries.
    quantized=prob.astype(np.float16)
    np.save(p,quantized);return quantized.astype(np.float32)

def grid(d,method):
    params=[dict(zip(['threshold','min_area','close'],p)) for p in itertools.product(*d.protocol['grid'].values())]
    rows=[[] for _ in params]
    for k,name in enumerate(d.protocol['split']['tune'],1):
        _,_,disk=d.arrays(name);prob=probabilities(d,name,method);truth=d.truth(name)
        for j,pred in enumerate(encoded_grid(prob,disk,params)):
            rows[j].extend({'file':name,'annotation_set':a,**instance_stats(gt,pred)} for a,gt in truth if gt)
        print('Tune',method,k,flush=True)
    totals=[{'params':p,**pool(rs)} for p,rs in zip(params,rows)]
    best=max(range(len(totals)),key=lambda j:totals[j]['pq'])
    write(HERE/(method+'-tuning.json'),{'trials':totals,'rows':rows[best]})
    write(HERE/(method+'-frozen.json'),{'method':method,'params':params[best],'tune':totals[best],'protocol_sha256':sha(HERE/'protocol.json'),'model_sha256':sha(HERE/('pixel-forest.joblib' if method=='rf' else method+'.pt'))})
    print('Selected',method,totals[best],flush=True)

def assess(d,methods):
    from solar_v3 import probabilities as old_prob
    oldmodel=read(OLD/'results/v3/frozen-model.json');rows={m:[] for m in ['v3']+methods}
    frozen={m:read(HERE/(m+'-frozen.json')) for m in methods}
    for k,name in enumerate(d.protocol['split']['assessment'],1):
        rec=read(ROOT/'work/solar-v3-cache'/(name+'.json'))
        from organizer_metric import stats_from_iou
        keep=old_prob(np.asarray(rec['features']),oldmodel)>=oldmodel['threshold']
        for entry in rec['matrices']:
            mat=np.asarray(entry['iou'],np.float32).reshape(entry['shape'])
            rows['v3'].append({'file':name,'annotation_set':entry['annotation_set'],**stats_from_iou(mat[:,keep])})
        _,_,disk=d.arrays(name);truth=d.truth(name)
        for method in methods:
            started=time.perf_counter();pr=probabilities(d,name,method)
            pred=encode_labels(instances(pr,disk,**frozen[method]['params']))
            elapsed=time.perf_counter()-started
            rows[method].extend({'file':name,'annotation_set':a,'seconds':elapsed,**instance_stats(gt,pred)} for a,gt in truth if gt)
        print('Assess',k,flush=True)
    totals={m:pool(rs) for m,rs in rows.items()}
    for t in totals.values():
        t.update(precision=t['tp']/max(t['tp']+t['fp'],1),recall=t['tp']/max(t['tp']+t['fn'],1))
    days=sorted({n[:8] for n in d.protocol['split']['assessment']});rng=np.random.default_rng(20261015)
    groups={m:{day:[r for r in rs if r['file'][:8]==day] for day in days} for m,rs in rows.items()}
    boot={m:[] for m in methods}
    for _ in range(2000):
        draw=rng.choice(days,len(days),replace=True)
        b=pool(r for day in draw for r in groups['v3'][day])['pq']
        for m in methods:boot[m].append(pool(r for day in draw for r in groups[m][day])['pq']-b)
    result={'status':d.protocol['status'],'images':len(d.protocol['split']['assessment']),'days':len(days),'results':totals,'paired_gain_95_date_cluster_interval':{m:np.quantile(v,[.025,.975]).tolist() for m,v in boot.items()},'rows':rows,'frozen':frozen,'submitted':False,'published':False}
    target=HERE/'assessment.json'
    if target.exists():raise FileExistsError('Assessment already exists')
    write(target,result);print(json.dumps(totals,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','rf','tune','assess']);p.add_argument('--method',default='rf');p.add_argument('--methods',nargs='+',default=['rf']);a=p.parse_args()
    if a.action=='freeze':freeze()
    else:
        d=Dataset()
        if a.action=='rf':train_rf(d)
        if a.action=='tune':grid(d,a.method)
        if a.action=='assess':assess(d,a.methods)
