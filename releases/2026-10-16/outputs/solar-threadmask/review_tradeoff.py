"""Post-hoc descriptive review of existing evaluation; no model retuning."""
import json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent
    data=json.loads((root/'results/v3/blind-evaluation.json').read_text())
    grouped={}
    for row in data['rows']:
        acc=grouped.setdefault(row['file'],{}).setdefault(row['method'],dict(tp=0,fp=0,fn=0,iou_sum=0))
        for key in acc:acc[key]+=row[key]
    rows=[]
    for name,methods in grouped.items():
        for counts in methods.values():
            denominator=counts['tp']+.5*(counts['fp']+counts['fn'])
            counts['pq']=counts['iou_sum']/denominator if denominator else 0
        rows.append({'file':name,'v2':methods['v2'],'v3':methods['v3'],'pq_delta':methods['v3']['pq']-methods['v2']['pq']})
    rows.sort(key=lambda row:(row['pq_delta'],row['file']))
    totals=data['results'];metrics={}
    for method,t in totals.items():
        metrics[method]={'precision':t['tp']/(t['tp']+t['fp']),'recall':t['tp']/(t['tp']+t['fn']),
                         'recognition_quality':t['tp']/(t['tp']+.5*(t['fp']+t['fn'])),
                         'segmentation_quality':t['iou_sum']/t['tp']}
    result={'scope':__doc__,'annotation_handling':'Pool all annotator entries within each image; not consensus labels',
            'images':len(rows),'improved':sum(r['pq_delta']>1e-12 for r in rows),
            'worse':sum(r['pq_delta'] < -1e-12 for r in rows),'unchanged':sum(abs(r['pq_delta'])<=1e-12 for r in rows),
            'pooled_metrics':metrics,'worst_five':rows[:5],'per_image':rows}
    (root/'results/v3/tradeoff-review.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_image','worst_five']},indent=2))

if __name__=='__main__':main()
