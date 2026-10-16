"""Frozen v8 model on additional dates excluded from current model splits."""
import sys,hashlib,itertools
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'solar-ridge-v8'))
import ridge as g
import runtime as base
import contour as metric
r=g.r;e=g.e;MODEL=H.parent/'solar-ridge-v8'

def main():
    model_protocol=r.read(MODEL/'protocol.json');frozen=r.read(MODEL/'frozen.json');assert frozen['selected']
    old=r.read(r.HERE/'protocol.json');excluded={n[:8] for values in old['split'].values() for n in values}
    files=list((r.DATA/'train/train_images').glob('*.jpeg'));available={}
    for p in files:
        if p.name[:8] not in excluded:available.setdefault(p.name[:8],[]).append(p.name)
    days=sorted(available,key=lambda day:hashlib.sha256(('ridge-extra-v9-'+day).encode()).hexdigest())[:40]
    names=[sorted(available[day],key=lambda n:hashlib.sha256(n.encode()).hexdigest())[0] for day in days]
    protocol={'images':names,'date_count':len(days),'selection':'40 deterministically hashed dates outside ALL current v5 train/tune/assessment dates; one image per date. Frozen before evaluation.',
      'parameters':frozen['selected']['params'],'model_sha256':frozen['model_sha256'],'model_protocol_sha256':r.sha(MODEL/'protocol.json'),'model_feature_sha256':r.sha(MODEL/'ridge.py'),
      'baseline_component_model_sha256':r.sha(r.HERE/'component-filter.joblib'),'source_sha256':r.sha(__file__),
      'limitations':'Additional dates were historically available/used by older baseline experiments. Not a fresh blind test. Never train, tune or select settings from these results.',
      'eligible_images':sum(map(len,available.values())),'excluded_current_split_dates':len(excluded),'no_publication_or_submission':True}
    assert len(names)==40 and not set(days)&excluded
    if (H/'protocol.json').exists():raise FileExistsError('Preserve existing run')
    r.write(H/'protocol.json',protocol)
    assert r.sha(MODEL/'protocol.json')==frozen['protocol_sha256'] and r.sha(MODEL/'ridge.py')==model_protocol['feature_source_sha256']
    path=base.verified_artifact(MODEL/'ridge-forest.joblib',frozen['model_sha256']);model=base._forest(path,frozen['model_sha256'])
    dataset=r.Dataset();rows={'baseline':[],'ridge':[]};hashes=[]
    for i,name in enumerate(names,1):
        image=r.load_image(r.DATA/'train/train_images'/name);labels=base.segment(image,'rf_filter');_,disk_low=base.prepare(image);disk=np.repeat(np.repeat(disk_low,2,0),2,1)
        geo=g.geometry(labels);widths,distance,nearest=geo;region=(widths[nearest]<=32)&(distance<=4);prob=np.zeros(labels.shape,np.float32)
        if region.any():prob[region]=model.predict(g.features(image,disk)[region])
        predicted=g.refine(labels,prob,*frozen['selected']['params'],geo=geo)
        assert set(np.unique(labels))==set(np.unique(predicted))
        truth=metric.truth(dataset,name);rows['baseline'].extend(metric.metrics(name,labels,truth));rows['ridge'].extend(metric.metrics(name,predicted,truth))
        hashes.append({'file':name,'baseline_sha256':hashlib.sha256(labels.tobytes()).hexdigest(),'ridge_sha256':hashlib.sha256(predicted.tobytes()).hexdigest()})
        result={'complete':i==len(names),'images_evaluated':i,'results':{k:e.totals(v) for k,v in rows.items()},'rows':rows,'mask_hashes':hashes}
        r.write(H/'assessment.json',result);print('Additional dates',i,len(names),flush=True)
    print('EXTRA RESULT',result['results'],flush=True)
if __name__=='__main__':main()
