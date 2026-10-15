"""Re-score frozen v1/v2 masks using public self-evaluation v6 logic."""
import argparse
from dataclasses import asdict
import hashlib,json
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_api
from pipeline import load_image,encoded_predictions,encode_annotation,encode_labels
from solar_v2 import ConfigV2,segment_v2
from organizer_metric import stats_from_iou,pool
from reference.organizer_pq import get_pq_score
from reference_adapter import Table


def run(data,project):
    out=project/'results/public-v6';out.mkdir(exist_ok=True)
    split=json.loads((project/'results/v2/prospective-split.json').read_text())
    config=ConfigV2(**json.loads((project/'results/v2/frozen-config.json').read_text()))
    coco=json.loads((data/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_bytes())
    infos={};anns={}
    for i in coco['images']:infos.setdefault(i['file_name'],[]).append(i)
    for a in coco['annotations']:anns.setdefault(a['image_id'],[]).append(a)
    rows=[];matrices={'v1':[],'v2':[]};overlaps=[]
    for index,name in enumerate(split['blind'],1):
        image=load_image(data/'train/train_images'/name)
        preds={'v1':encoded_predictions(image),'v2':encode_labels(segment_v2(image,config))}
        for i in infos[name]:
            truth=[encode_annotation(a,2048,2048) for a in anns.get(i['id'],[])]
            if not truth:continue  # public get_overlap_df enumerates IDs from GT rows
            for method,pr in preds.items():
                matrix=(np.asarray(mask_api.iou(truth,pr,[0]*len(pr)),dtype=np.float32)
                        if pr else np.zeros((len(truth),0),dtype=np.float32))
                matrices[method].append(matrix)
                rows.append({'file':name,'annotation_set':i['id'],'method':method,**stats_from_iou(matrix)})
                overlaps.append({'file':name,'annotation_set':i['id'],'method':method,'shape':list(matrix.shape),'iou':matrix.tolist()})
        print(f'Public-v6 scoring {index}/{len(split["blind"])}',flush=True)
    totals={m:pool(r for r in rows if r['method']==m) for m in ['v1','v2']}
    parity={m:abs(get_pq_score(Table(matrices[m]))-totals[m]['pq']) for m in totals}
    assert max(parity.values())<1e-12
    days=sorted({r['file'][:8] for r in rows});by_day={day:[r for r in rows if r['file'][:8]==day] for day in days}
    rng=np.random.default_rng(20261014);boot=[]
    for _ in range(2000):
        sampled=[r for day in rng.choice(days,len(days),replace=True) for r in by_day[day]]
        boot.append([pool(r for r in sampled if r['method']==m)['pq'] for m in ['v1','v2']])
    boot=np.asarray(boot)
    result={'metric':'Public self-evaluation notebook v6, pooled per-annotator counts, every IoU > .5 pair',
            'scope':'Function parity verified using NumPy adapter; not executed through Torch or private Kaggle scoring service.',
            'source':'https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook',
            'kernel_version_id':341270459,'observations':len(split['blind']),'days':len(days),
            'annotation_sets':len(matrices['v2']),'results':totals,'reference_absolute_errors':parity,
            'v2_95_date_cluster_interval':np.quantile(boot[:,1],[.025,.975]).tolist(),
            'paired_improvement_95_interval':np.quantile(boot[:,1]-boot[:,0],[.025,.975]).tolist(),
            'configuration':asdict(config),'rows':rows,'submitted':False,'leaderboard_score':None,
            'bootstrap_replicates':2000,'bootstrap_seed':20261014,
            'scorer_sha256':hashlib.sha256((project/'organizer_metric.py').read_bytes()).hexdigest()}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    (out/'overlap-matrices.json').write_text(json.dumps(overlaps))
    print(json.dumps({'results':totals,'parity_errors':parity},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);a=p.parse_args()
    run(a.data,Path(__file__).resolve().parent)
