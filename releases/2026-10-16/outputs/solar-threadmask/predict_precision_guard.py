"""Generate a separate experimental precision-guard candidate with exact mask/RLE checks."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_api
from pipeline import load_image,encode_labels
from solar_v2 import ConfigV2
from solar_v3 import candidates,probabilities,filter_labels
from solar_precision_guard import guard_mask
from experiment_v3 import sources,sha,read,write

def predict(data,output,model_dir):
    protocol=read(model_dir/'protocol.json');model=read(model_dir/'frozen-model.json')
    guard_file=Path(__file__).resolve().parent/'results/precision-guard/frozen-rule.json'
    guard=read(guard_file)['selected']['rule']
    assert protocol['source_sha256']==sources()
    assert model['protocol_sha256']==sha(model_dir/'protocol.json')
    assert model['threshold'] is not None
    config=ConfigV2(**protocol['baseline_config'])
    paths=sorted((data/'test/test_images').glob('*.jp*g'))
    if not paths:raise ValueError('No test observations')
    output.parent.mkdir(parents=True,exist_ok=True);counts={};before={};inputs=[];start=time.perf_counter()
    with output.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['filament_id','segmentation_rle'])
        for index,path in enumerate(paths,1):
            labels,x=candidates(load_image(path),config);before[path.stem]=len(x)
            filtered=filter_labels(labels,guard_mask(x,model,guard).astype(float),.5)
            masks=encode_labels(filtered);counts[path.stem]=len(masks)
            for identifier,rle in enumerate(masks,1):
                assert np.array_equal(mask_api.decode(rle).astype(bool),filtered==identifier)
                writer.writerow([f'{path.stem}_{identifier}',rle['counts'].decode('ascii')])
            inputs.append({'file':path.name,'sha256':sha(path)})
            print('guard prediction',index,'/',len(paths),flush=True)
    audit={'method':'experimental-recall-precision-guard','threshold':.075,'guard_rule':guard,'guard_rule_sha256':sha(guard_file),'guard_source_sha256':sha(Path(__file__).resolve().parent/'solar_precision_guard.py'),'recommended_candidate':False,
        'frozen_model_sha256':sha(model_dir/'frozen-model.json'),'protocol_sha256':sha(model_dir/'protocol.json'),
        'source_sha256':sources(),'predictor_sha256':sha(__file__),
        'counts':counts,'unfiltered_counts':before,'images':inputs,'csv_sha256':sha(output),
        'zero_prediction_images':[n for n,c in counts.items() if c==0],
        'exact_rle_roundtrip_for_every_instance':True,'seconds':time.perf_counter()-start,
        'submitted':False,'private_validator_tested':False,
        'format':'One row per detected object; zero detections means zero object rows, as in inspected public evaluator.'}
    write(output.with_suffix('.audit.json'),audit)
    print('Images:',len(paths),'instance rows:',sum(counts.values()),'empty images:',len(audit['zero_prediction_images']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--model-dir',type=Path,default=Path(__file__).resolve().parent/'results/v3')
    a=p.parse_args();predict(a.data.resolve(),a.output.resolve(),a.model_dir.resolve())
