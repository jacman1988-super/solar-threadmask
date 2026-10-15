"""Single-feature rejection search with zero training/tuning TP-loss constraint."""
import json
from pathlib import Path
import numpy as np
from experiment_v3 import Experiment,read,write,sha
from solar_v3 import probabilities,FEATURE_NAMES
from organizer_metric import pool,stats_from_iou

def run():
    root=Path(__file__).resolve().parent;base=root.parents[1];out=root/'results/precision-guard'
    out.mkdir(parents=True,exist_ok=False)
    protocol={'scope':'Development only: evaluation dates already inspected in recall-v4; no independent validation or guaranteed recall preservation.',
              'source_sha256':sha(__file__),'model_sha256':sha(root/'results/v3/frozen-model.json'),
              'baseline_threshold':.075,'quantiles':[.01,.025,.05,.1,.2,.3,.5,.7,.8,.9,.95,.975,.99],
              'rule':'Single feature >= or <= training quantile. Require zero TP loss on both training and tuning. Choose highest tune PQ, then fewest tune FP, then rule order. Include no-op. No retuning on evaluation.',
              'evaluation':'All 106 previously unused dates now inspected in recall-v4. Original candidate unchanged.'}
    write(out/'protocol.json',protocol)
    e=Experiment(base/'work/solar-data/MAGFiLO_1.0_Kaggle_2026',root/'results/v3',base/'work/solar-v3-cache');model=read(root/'results/v3/frozen-model.json')
    train=list(e.records('train'));tune=list(e.records('tune'))
    x=np.concatenate([np.asarray(r['features']).reshape(r['instances'],len(FEATURE_NAMES)) for r in train if r['instances']])
    rules=[None]+[{'feature':i,'name':FEATURE_NAMES[i],'direction':direction,'cutoff':float(np.quantile(x[:,i],q)),'quantile':q} for i in range(len(FEATURE_NAMES)) for q in protocol['quantiles'] for direction in ['ge','le']]
    def stats(records,rule):
        rows=[]
        for rec in records:
            features=np.asarray(rec['features']).reshape(rec['instances'],len(FEATURE_NAMES));keep=probabilities(features,model)>=.075
            if rule:
                v=features[:,rule['feature']];keep &= (v>=rule['cutoff'] if rule['direction']=='ge' else v<=rule['cutoff'])
            for entry in rec['matrices']:
                matrix=np.asarray(entry['iou'],dtype=np.float32).reshape(entry['shape'])
                rows.append({'file':rec['file'],'annotation_set':entry['annotation_set'],**stats_from_iou(matrix[:,keep])})
        return rows
    base_train=pool(stats(train,None));base_tune=pool(stats(tune,None));eligible=[]
    for index,rule in enumerate(rules):
        a=pool(stats(train,rule))
        if a['tp']!=base_train['tp']:continue
        b=pool(stats(tune,rule))
        if b['tp']==base_tune['tp']:eligible.append({'index':index,'rule':rule,'train':a,'tune':b})
    chosen=min(eligible,key=lambda r:(-r['tune']['pq'],r['tune']['fp'],r['index']))
    write(out/'frozen-rule.json',{'selected':chosen,'eligible':eligible,'trials':len(rules),'protocol_sha256':sha(out/'protocol.json')})
    evaluation=list(e.records('unused'));rows={m:stats(evaluation,r) for m,r in [('recall_v4',None),('precision_guard',chosen['rule'])]}
    totals={m:pool(rs) for m,rs in rows.items()}
    for t in totals.values():t['recall']=t['tp']/(t['tp']+t['fn'])
    result={'results':totals,'selected_rule':chosen['rule'],'rows':rows,'independent_validation':False,'candidate_replaced':False,'submitted':False}
    write(out/'evaluation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':run()
