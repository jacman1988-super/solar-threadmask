"""Generate a local Solar ridge candidate; never contact Kaggle or overwrite a CSV."""
import argparse,csv,json,hashlib,sys,time,importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from pycocotools import mask as coco

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
PACKAGE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,default=PACKAGE.parent/'releases/2026-10-16');p.add_argument('--images',type=Path,required=True);p.add_argument('--weights',type=Path,default=PACKAGE/'weights');p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if a.output.exists():p.error('Output CSV already exists')
    release=a.release.resolve();images=sorted(a.images.glob('*.jp*g'))
    import subprocess
    subprocess.run([sys.executable,str(PACKAGE/'verify_materials.py')],check=True)
    subprocess.run([sys.executable,str(release/'verify_release.py')],check=True)
    if not images:p.error('No input images')
    a.output.parent.mkdir(parents=True,exist_ok=True);cache=a.output.parent/'prediction-records';cache.mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('release_adapter',release/'predict.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    (release/'work').mkdir(exist_ok=True);sys.path.insert(0,str(release/'outputs/solar-ridge-v8'));import image_only as ridge
    adapter.install_weights(ridge.base,a.weights.resolve())
    protocol={'image_files':[x.name for x in images],'weights':{n:sha(a.weights/n) for n in ['pixel-forest.joblib','component-filter.joblib','ridge-forest.joblib']},'ridge_protocol_sha256':sha(release/'outputs/solar-ridge-v8/protocol.json'),'ridge_frozen_sha256':sha(release/'outputs/solar-ridge-v8/frozen.json'),'script_sha256':sha(Path(__file__)),'workers':a.workers,'submitted':False,'private_score_known':False}
    frozen=a.output.parent/'prediction-protocol.json'
    if frozen.exists():assert json.loads(frozen.read_text())==protocol,'Do not mix protocols'
    else:frozen.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    # Load and verify the two reference forest artifacts once before concurrent inference.
    for n,key in [('pixel-forest.joblib','rf-frozen.json'),('component-filter.joblib','component-frozen.json')]:
        f=json.loads((release/'outputs/solar-research-v5'/key).read_text());path=ridge.base.verified_artifact(a.weights/n,f['model_sha256']);ridge.base._forest(path,f['model_sha256'])
    frozen_r=json.loads((release/'outputs/solar-ridge-v8/frozen.json').read_text());path=ridge.base.verified_artifact(a.weights/'ridge-forest.joblib',frozen_r['model_sha256']);ridge.base._forest(path,frozen_r['model_sha256'])
    def one(path):
        record=cache/(path.stem+'.json');digest=sha(path)
        if record.exists():
            data=json.loads(record.read_text());assert data['input_sha256']==digest;return data
        start=time.perf_counter()
        with Image.open(path) as im:image=np.asarray(im)
        labels=ridge.segment(image);rows=[]
        for k in range(1,int(labels.max())+1):
            mask=labels==k;assert mask.any();encoded=coco.encode(np.asfortranarray(mask.astype(np.uint8)));assert np.array_equal(coco.decode(encoded).astype(bool),mask)
            rows.append({'filament_id':f'{path.stem}_{k}','segmentation_rle':encoded['counts'].decode('ascii')})
        data={'file':path.name,'input_sha256':digest,'labels_sha256':hashlib.sha256(labels.tobytes()).hexdigest(),'rows':rows,'seconds':time.perf_counter()-start}
        record.write_text(json.dumps(data),encoding='utf-8');return data
    results=[];start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for k,result in enumerate(pool.map(one,images),1):results.append(result);print(k,'/',len(images),result['file'],'instances',len(result['rows']),flush=True)
    with a.output.open('x',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=['filament_id','segmentation_rle']);writer.writeheader()
        for result in results:writer.writerows(result['rows'])
    audit={'counts':{Path(x['file']).stem:len(x['rows']) for x in results},'images':[{'file':x['file'],'sha256':x['input_sha256'],'labels_sha256':x['labels_sha256']} for x in results],'csv_sha256':sha(a.output),'exact_rle_roundtrip_for_every_instance':True,'seconds':time.perf_counter()-start,'submitted':False,'private_validator_tested':False,'new_kaggle_score':None}
    a.output.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print('Complete CSV:',a.output,flush=True)
if __name__=='__main__':main()
