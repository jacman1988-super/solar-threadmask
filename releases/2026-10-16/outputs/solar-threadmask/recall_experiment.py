"""Recall-oriented threshold selected on old tuning dates, new unused evaluation."""
import hashlib,json
from pathlib import Path
import numpy as np
from experiment_v3 import Experiment,read,write,sha
from organizer_metric import pool

def run():
    root=Path(__file__).resolve().parent;base=root.parents[1];out=root/'results/recall-v4'
    out.mkdir(parents=True,exist_ok=False)
    old=root/'results/v3';protocol=read(old/'protocol.json')
    rule={'scope':__doc__,'thresholds':[0,.025,.05,.075,.1,.125,.15],
          'selection':'Maximize pooled recall on old tune subject to PQ >= 90% of v3 tune PQ; tie highest PQ then highest threshold.',
          'new_evaluation':protocol['split']['unused'],'comparison':'v3 threshold .15; v2 threshold 0',
          'promotion':'No automatic candidate replacement. Report recall/PQ tradeoff and paired date bootstrap.',
          'source_sha256':sha(__file__),'old_protocol_sha256':sha(old/'protocol.json'),'model_sha256':sha(old/'frozen-model.json')}
    write(out/'protocol.json',rule)
    e=Experiment(base/'work/solar-data/MAGFiLO_1.0_Kaggle_2026',old,base/'work/solar-v3-cache')
    model=read(old/'frozen-model.json');trials=[]
    records=list(e.records('tune'))
    for t in rule['thresholds']:
        counts=pool(r for rec in records for r in e.stats(rec,model,t));counts.update(threshold=t,recall=counts['tp']/(counts['tp']+counts['fn']))
        trials.append(counts)
    floor=next(r['pq'] for r in trials if r['threshold']==.15)*.9
    best=min((r for r in trials if r['pq']>=floor),key=lambda r:(-r['recall'],-r['pq'],-r['threshold']))
    write(out/'frozen-threshold.json',{'selected':best,'pq_floor':floor,'trials':trials,'protocol_sha256':sha(out/'protocol.json')})
    rows=[]
    for rec in e.records('unused'):
        for method,t in [('v2',0),('v3',.15),('recall_v4',best['threshold'])]:
            rows.extend(dict(method=method,**r) for r in e.stats(rec,model,t))
    def metrics(rs):
        p=pool(rs);p['recall']=p['tp']/(p['tp']+p['fn']) if p['tp']+p['fn'] else 0;return p
    totals={m:metrics([r for r in rows if r['method']==m]) for m in ['v2','v3','recall_v4']}
    days=sorted({r['file'][:8] for r in rows});grouped={d:[r for r in rows if r['file'][:8]==d] for d in days}
    rng=np.random.default_rng(20261014);deltas=[]
    for _ in range(2000):
        sample=[r for d in rng.choice(days,len(days),replace=True) for r in grouped[d]]
        a=metrics([r for r in sample if r['method']=='v3']);b=metrics([r for r in sample if r['method']=='recall_v4'])
        deltas.append([b['recall']-a['recall'],b['pq']-a['pq']])
    result={'images':len(rule['new_evaluation']),'days':len(days),'threshold':best['threshold'],'results':totals,'paired_95_intervals_recall_then_pq':np.quantile(deltas,[.025,.975],axis=0).tolist(),'rows':rows,'candidate_replaced':False,'submitted':False}
    write(out/'evaluation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':run()
