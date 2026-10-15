"""Frozen guard on earlier inspected v3 assessment; no selection or retuning."""
from pathlib import Path
import numpy as np
from experiment_v3 import Experiment,read,write,sha
from solar_v3 import FEATURE_NAMES,probabilities
from solar_precision_guard import guard_mask
from organizer_metric import stats_from_iou,pool

def run():
    root=Path(__file__).resolve().parent;base=root.parents[1];out=root/'results/guard-previous-assessment'
    out.mkdir(parents=True,exist_ok=False)
    rule=read(root/'results/precision-guard/frozen-rule.json')['selected']['rule'];model=read(root/'results/v3/frozen-model.json')
    write(out/'protocol.json',{'scope':__doc__,'rule':rule,'rule_file_sha256':sha(root/'results/precision-guard/frozen-rule.json'),
                              'source_sha256':sha(__file__),'independent_validation':False,'no_retuning':True})
    e=Experiment(base/'work/solar-data/MAGFiLO_1.0_Kaggle_2026',root/'results/v3',base/'work/solar-v3-cache');rows=[]
    for rec in e.records('blind'):
        x=np.asarray(rec['features']).reshape(rec['instances'],len(FEATURE_NAMES))
        for method,keep in [('recall_v4',probabilities(x,model)>=.075),('guard',guard_mask(x,model,rule))]:
            for entry in rec['matrices']:
                matrix=np.asarray(entry['iou'],dtype=np.float32).reshape(entry['shape'])
                rows.append({'method':method,'file':rec['file'],'annotation_set':entry['annotation_set'],**stats_from_iou(matrix[:,keep])})
    totals={m:pool(r for r in rows if r['method']==m) for m in ['recall_v4','guard']}
    for t in totals.values():t['recall']=t['tp']/(t['tp']+t['fn'])
    old=[r for r in rows if r['method']=='recall_v4'];new=[r for r in rows if r['method']=='guard']
    losses=[{'file':a['file'],'annotation_set':a['annotation_set'],'lost_tp':a['tp']-b['tp']} for a,b in zip(old,new) if a['tp']>b['tp']]
    result={'images':len(e.protocol['split']['blind']),'results':totals,'lost_tp_entries':losses,'rows':rows,'independent_validation':False,'retuned':False}
    write(out/'evaluation.json',result);print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':run()
