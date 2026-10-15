"""Freeze prospective split, tune on development only, then evaluate once."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

PROJECT=Path(__file__).resolve().parent
BASE=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT))
from pipeline import holdout,load_image,encode_annotation,encode_labels,pq_encoded,encoded_predictions
from solar_v2 import ConfigV2,features,from_features,segment_v2
from PIL import Image

DATA=BASE/'work/solar-data/MAGFiLO_1.0_Kaggle_2026'
OUT=PROJECT/'results/v2'
OUT.mkdir(exist_ok=True)


def digest(s):return hashlib.sha256(s.encode()).hexdigest()


def dataset():
    raw=(DATA/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_bytes()
    coco=json.loads(raw);infos={};annotations={}
    for i in coco['images']:infos.setdefault(i['file_name'],[]).append(i)
    for a in coco['annotations']:annotations.setdefault(a['image_id'],[]).append(a)
    return raw,infos,annotations


def freeze():
    raw,infos,_=dataset()
    development=sorted(n for n in infos if not holdout(n))
    days=sorted({n[:8] for n in development},key=lambda d:digest('threadmask-v2-prospective-'+d))
    blind_days=set(days[:96])
    remaining=days[96:]
    tune_days=set(remaining[:32])
    # All observations for a day remain together. Other development data stays unused.
    manifest={'rule':'SHA256(threadmask-v2-prospective- + day), first 96 days blind, next 32 tune',
              'blind':[n for n in development if n[:8] in blind_days],
              'tune':[n for n in development if n[:8] in tune_days],
              'old_adaptive':[n for n in infos if holdout(n)],
              'annotations_sha256':hashlib.sha256(raw).hexdigest(),
              'limits':'Old 135-image holdout was inspected; new blind is drawn from previously uninspected development dates. Use once after frozen tuning.'}
    target=OUT/'prospective-split.json'
    if target.exists():assert json.loads(target.read_text())==manifest
    else:target.write_text(json.dumps(manifest,indent=2))
    print({k:len(manifest[k]) for k in ['blind','tune','old_adaptive']},flush=True)


def ground_truth(name,infos,annotations):
    return [[encode_annotation(a,i['height'],i['width']) for a in annotations.get(i['id'],[])] for i in infos[name]]


def score(predicted,truth):
    return float(np.mean([pq_encoded(gt,predicted)['pq'] or 0.0 for gt in truth]))


def tune():
    _,infos,annotations=dataset();split=json.loads((OUT/'prospective-split.json').read_text())
    configs=[ConfigV2(sigma=s,low=l,min_area=a) for s in [12.,24.,48.] for l in [.025,.04,.06,.09] for a in [20,60]]
    rows=[{'config':asdict(c),'scores':[]} for c in configs]
    for index,name in enumerate(split['tune'],1):
        image=load_image(DATA/'train/train_images'/name)
        truth=ground_truth(name,infos,annotations)
        cache={s:features(image,s,1024,24) for s in [12.,24.,48.]}
        for row,c in zip(rows,configs):
            labels=from_features(*cache[c.sigma],c)
            full=np.asarray(Image.fromarray(labels).resize((2048,2048),Image.Resampling.NEAREST))
            row['scores'].append({'file':name,'pq':score(encode_labels(full),truth)})
        print(f'Tuned {index}/{len(split["tune"])} observations',flush=True)
    for row in rows:row['mean_observation_pq']=float(np.mean([r['pq'] for r in row['scores']]))
    rows.sort(key=lambda r:(-r['mean_observation_pq'],json.dumps(r['config'],sort_keys=True)))
    result={'kind':'development_parameter_search','trials':rows,'selected_config':rows[0]['config'],
            'source_sha256':hashlib.sha256((PROJECT/'solar_v2.py').read_bytes()).hexdigest(),
            'split_sha256':hashlib.sha256((OUT/'prospective-split.json').read_bytes()).hexdigest()}
    (OUT/'tuning.json').write_text(json.dumps(result,indent=2))
    (OUT/'frozen-config.json').write_text(json.dumps(result['selected_config'],indent=2))
    print('Selected:',rows[0]['config'],'development PQ:',rows[0]['mean_observation_pq'],flush=True)


def evaluate():
    _,infos,annotations=dataset();split=json.loads((OUT/'prospective-split.json').read_text())
    config=ConfigV2(**json.loads((OUT/'frozen-config.json').read_text()))
    source_hash=hashlib.sha256((PROJECT/'solar_v2.py').read_bytes()).hexdigest()
    assert source_hash==json.loads((OUT/'tuning.json').read_text())['source_sha256']
    rows=[]
    for index,name in enumerate(split['blind'],1):
        image=load_image(DATA/'train/train_images'/name)
        truth=ground_truth(name,infos,annotations)
        started=time.perf_counter();v2=encode_labels(segment_v2(image,config));elapsed=time.perf_counter()-started
        baseline_started=time.perf_counter();v1=encoded_predictions(image);baseline_seconds=time.perf_counter()-baseline_started
        rows.append({'file':name,'baseline_pq':score(v1,truth),'v2_pq':score(v2,truth),
                     'v2_instances':len(v2),'baseline_instances':len(v1),'v2_seconds':elapsed,'baseline_seconds':baseline_seconds,
                     'image_sha256':hashlib.sha256((DATA/'train/train_images'/name).read_bytes()).hexdigest()})
        print(f'Blind evaluation {index}/{len(split["blind"])}',flush=True)
    result={'kind':'prospective_once_only_local_validation','config':asdict(config),'rows':rows,
            'baseline_mean_pq':float(np.mean([r['baseline_pq'] for r in rows])),
            'v2_mean_pq':float(np.mean([r['v2_pq'] for r in rows])),
            'source_sha256':source_hash,'metric':'local diagnostic PQ, organizer parity unverified',
            'warning':'Do not tune on this blind set after opening these results.'}
    (OUT/'blind-validation.json').write_text(json.dumps(result,indent=2))
    print('BASELINE:',result['baseline_mean_pq'],'V2:',result['v2_mean_pq'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','tune','evaluate']);p.add_argument('--data',type=Path,default=DATA);a=p.parse_args();DATA=a.data
    {'freeze':freeze,'tune':tune,'evaluate':evaluate}[a.action]()
