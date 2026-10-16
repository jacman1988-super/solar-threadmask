"""Three deterministic evaluation filenames: fresh image inference versus cache."""
from pathlib import Path
import numpy as np
from experiment_v3 import Experiment,read,write
from pipeline import load_image,encode_labels,encode_annotation
from solar_v3 import candidates,filter_labels,probabilities
from solar_precision_guard import segment_precision_guard
from organizer_metric import instance_stats

def run():
    root=Path(__file__).resolve().parent;base=root.parents[1]
    e=Experiment(base/'work/solar-data/MAGFiLO_1.0_Kaggle_2026',root/'results/v3',base/'work/solar-v3-cache')
    model=read(root/'results/v3/frozen-model.json');rule=read(root/'results/precision-guard/frozen-rule.json')['selected']['rule']
    recorded=read(root/'results/precision-guard/evaluation.json');rows=[]
    for name in sorted(e.protocol['split']['unused'])[:3]:
        image=load_image(e.data/'train/train_images'/name);labels,x=candidates(image,e.config);cached=e.record(name)
        assert np.allclose(x,np.asarray(cached['features']),rtol=0,atol=1e-12)
        for method,predicted in [('recall_v4',filter_labels(labels,probabilities(x,model),.075)),('precision_guard',segment_precision_guard(image,e.config,model,rule))]:
            prediction=encode_labels(predicted)
            for info in e.infos[name]:
                truth=[encode_annotation(a,info['height'],info['width']) for a in e.anns.get(info['id'],[])]
                if not truth:continue
                actual=instance_stats(truth,prediction)
                expected=next(r for r in recorded['rows'][method] if r['file']==name and r['annotation_set']==info['id'])
                for key in ['tp','fp','fn','iou_sum']:assert abs(actual[key]-expected[key])<1e-6,(name,key)
                rows.append({'file':name,'method':method,'annotation_set':info['id'],**actual})
    write(root/'results/precision-guard/inference-check.json',{'images':3,'all_checks_passed':True,'selection':'First three lexicographic filenames; deterministic, inspected evaluation data','rows':rows})
    print('Fresh image inference agrees with cached experiment for all three images')

if __name__=='__main__':run()
