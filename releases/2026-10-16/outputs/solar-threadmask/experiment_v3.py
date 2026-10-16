"""Freeze, train, tune and once-only evaluate a component rejection model."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_api
from pipeline import load_image,encode_labels,encode_annotation
from solar_v2 import ConfigV2
from solar_v3 import candidates,fit_filter,probabilities
from organizer_metric import stats_from_iou,pool

PROJECT=Path(__file__).resolve().parent
THRESHOLDS=[0,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')
def sources():return {n:sha(PROJECT/n) for n in ['solar_v2.py','solar_v3.py','experiment_v3.py','organizer_metric.py']}

def freeze(data,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('Preserve prior protocol')
    coco=read(data/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    names=sorted({i['file_name'] for i in coco['images']})
    old=read(PROJECT/'results/v2/prospective-split.json')
    used={n[:8] for k in ['blind','tune','old_adaptive'] for n in old[k]}
    fresh=[n for n in names if n[:8] not in used]
    days=sorted({n[:8] for n in fresh},key=lambda d:hashlib.sha256(('threadmask-v3-component-filter-'+d).encode()).hexdigest())
    assert len(days)>=304
    groups={'blind':set(days[:96]),'tune':set(days[96:144]),'train':set(days[144:304]),'unused':set(days[304:])}
    split={k:[n for n in fresh if n[:8] in g] for k,g in groups.items()}
    for a in groups:
        for b in groups:
            if a!=b:assert not groups[a]&groups[b]
    protocol={'split':split,'excluded_prior_days':sorted(used),
      'rule':'SHA256(threadmask-v3-component-filter- + date): first 96 days blind, next 48 tune, next 160 train, remainder unused. Exclude every date from all old splits.',
      'source_sha256':sources(),'annotations_sha256':sha(data/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
      'baseline_config':read(PROJECT/'results/v2/frozen-config.json'),'thresholds':THRESHOLDS,
      'training':'Unweighted logistic loss on per-annotator candidate labels: any GT IoU > .5 within that annotator entry. Fixed L2=.01, standardized features, no coefficient/model-family search.',
      'selection':'Highest pooled public-v6 PQ on tune; tie prefers lower rejection threshold. Includes threshold zero, recovering baseline.',
      'promotion':'Use v3 as recommended candidate only if paired date-cluster bootstrap 95% lower bound of v3-v2 on new blind is > 0; otherwise retain v2.',
      'bootstrap':{'seed':20261014,'replicates':2000},
      'scope':'Fresh dates relative to all recorded prior splits. Date groups can remain physically correlated across nearby days. Blind opened once after training and threshold freeze. No leaderboard calls.'}
    write(out/'protocol.json',protocol)
    print('Frozen images:',{k:len(v) for k,v in split.items()},flush=True)

class Experiment:
    def __init__(self,data,out,cache):
        self.data,self.out,self.cache=data,out,cache
        self.protocol=read(out/'protocol.json')
        assert self.protocol['source_sha256']==sources(),'Code changed after protocol freeze'
        assert self.protocol['annotations_sha256']==sha(data/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
        self.config=ConfigV2(**self.protocol['baseline_config'])
        coco=read(data/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
        self.infos={};self.anns={}
        for i in coco['images']:self.infos.setdefault(i['file_name'],[]).append(i)
        for a in coco['annotations']:self.anns.setdefault(a['image_id'],[]).append(a)
        cache.mkdir(parents=True,exist_ok=True)

    def record(self,name):
        image_path=self.data/'train/train_images'/name
        key={'image_sha256':sha(image_path),'source_sha256':self.protocol['source_sha256'],
             'annotations_sha256':self.protocol['annotations_sha256']}
        target=self.cache/(name+'.json')
        if target.exists():
            result=read(target);assert result['input']==key;return result
        image=load_image(image_path);labels,x=candidates(image,self.config)
        pred=encode_labels(labels);matrices=[]
        assert len(pred)==len(x)
        for info in self.infos[name]:
            truth=[encode_annotation(a,info['height'],info['width']) for a in self.anns.get(info['id'],[])]
            if not truth:continue # public notebook enumerates annotation entries with GT
            matrix=(np.asarray(mask_api.iou(truth,pred,[0]*len(pred)),dtype=np.float32)
                    if pred else np.zeros((len(truth),0),dtype=np.float32))
            matrices.append({'annotation_set':info['id'],'shape':list(matrix.shape),'iou':matrix.tolist()})
        result={'file':name,'input':key,'features':x.tolist(),'instances':len(pred),'matrices':matrices}
        write(target,result);return result

    def records(self,stage):
        for i,name in enumerate(self.protocol['split'][stage],1):
            yield self.record(name)
            print(stage,i,'/',len(self.protocol['split'][stage]),flush=True)

    def fit(self):
        if (self.out/'model.json').exists():raise FileExistsError('Keep previous fit')
        x=[];y=[];inputs=[]
        for record in self.records('train'):
            inputs.append({'file':record['file'],'sha256':record['input']['image_sha256']})
            for entry in record['matrices']:
                matrix=np.asarray(entry['iou'],dtype=np.float32).reshape(entry['shape'])
                x.extend(record['features']);y.extend((matrix.max(axis=0)>.5).astype(int).tolist())
        model=fit_filter(x,y,.01)
        model.update({'threshold':None,'protocol_sha256':sha(self.out/'protocol.json'),
            'training_rows':len(y),'positive_rows':int(sum(y)),'negative_rows':int(len(y)-sum(y))})
        write(self.out/'model.json',model);write(self.out/'training-inputs.json',inputs)
        print('Trained:',len(y),'rows;',sum(y),'positive',flush=True)

    def stats(self,record,model,threshold):
        x=np.asarray(record['features'],dtype=np.float64).reshape(record['instances'],-1) if record['instances'] else np.zeros((0,len(model['feature_names'])))
        keep=probabilities(x,model)>=threshold
        rows=[]
        for entry in record['matrices']:
            matrix=np.asarray(entry['iou'],dtype=np.float32).reshape(entry['shape'])
            rows.append({'file':record['file'],'annotation_set':entry['annotation_set'],**stats_from_iou(matrix[:,keep])})
        return rows

    def tune(self):
        if (self.out/'frozen-model.json').exists():raise FileExistsError('Threshold already frozen')
        model=read(self.out/'model.json');trials={str(t):[] for t in self.protocol['thresholds']}
        for record in self.records('tune'):
            for threshold in self.protocol['thresholds']:
                trials[str(threshold)].extend(self.stats(record,model,threshold))
        totals=[{'threshold':t,**pool(trials[str(t)])} for t in self.protocol['thresholds']]
        best=min(totals,key=lambda r:(-r['pq'],r['threshold']))
        model['threshold']=best['threshold'];write(self.out/'frozen-model.json',model)
        write(self.out/'tuning.json',{'trials':totals,'selected_threshold':best['threshold'],'rows':trials})
        print('Frozen threshold:',best,flush=True)

    def evaluate(self):
        if (self.out/'blind-evaluation.json').exists():raise FileExistsError('Blind already inspected; do not retune')
        model=read(self.out/'frozen-model.json');rows=[]
        assert model['protocol_sha256']==sha(self.out/'protocol.json')
        for record in self.records('blind'):
            for method,threshold in [('v2',0),('v3',model['threshold'])]:
                rows.extend({'method':method,**r} for r in self.stats(record,model,threshold))
        totals={m:pool(r for r in rows if r['method']==m) for m in ['v2','v3']}
        days=sorted({r['file'][:8] for r in rows});grouped={d:[r for r in rows if r['file'][:8]==d] for d in days}
        rng=np.random.default_rng(20261014);boot=[]
        for _ in range(2000):
            sample=[r for d in rng.choice(days,len(days),replace=True) for r in grouped[d]]
            boot.append([pool(r for r in sample if r['method']==m)['pq'] for m in ['v2','v3']])
        boot=np.asarray(boot);interval=np.quantile(boot[:,1]-boot[:,0],[.025,.975]).tolist()
        result={'observations':len(self.protocol['split']['blind']),'days':len(days),'results':totals,
            'v3_95_date_cluster_interval':np.quantile(boot[:,1],[.025,.975]).tolist(),
            'paired_gain_95_interval':interval,'promote_v3':bool(interval[0]>0),
            'selected_threshold':model['threshold'],'rows':rows,
            'frozen_model_sha256':sha(self.out/'frozen-model.json'),
            'metric':'Public self-evaluation v6 pooled arithmetic; not private scorer or leaderboard',
            'submitted':False}
        write(self.out/'blind-evaluation.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','fit','tune','evaluate'])
    p.add_argument('data',type=Path);p.add_argument('cache',type=Path)
    p.add_argument('--output',type=Path,default=PROJECT/'results/v3')
    a=p.parse_args();data=a.data.resolve();out=a.output.resolve()
    if a.action=='freeze':freeze(data,out)
    else:getattr(Experiment(data,out,a.cache.resolve()),a.action)()
