"""Generate LOCAL v2 predictions from a frozen configuration; never submit."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from pycocotools import mask as mask_api
from pipeline import load_image,encode_labels
from solar_v2 import ConfigV2,segment_v2


def predict(data,output,config_path):
    config=ConfigV2(**json.loads(config_path.read_text()))
    files=sorted((data/'test/test_images').glob('*.jp*g'))
    if not files:raise ValueError('No test images')
    counts={};manifest=[];output.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
        for index,path in enumerate(files,1):
            labels=segment_v2(load_image(path),config)
            encoded=encode_labels(labels)
            counts[path.stem]=len(encoded)
            # Verify decoded pixels against exact label membership, one mask at a time.
            for i,rle in enumerate(encoded,1):
                if not np.array_equal(mask_api.decode(rle).astype(bool),labels==i):
                    raise ValueError('RLE exact pixel verification failed')
                writer.writerow([f'{path.stem}_{i}',rle['counts'].decode('ascii')])
            manifest.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            print(f'Predicted {index}/{len(files)}',flush=True)
    audit={'method':'disk-aware-v2','configuration':asdict(config),'counts':counts,'images':manifest,
           'csv_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
           'zero_prediction_images':[n for n,c in counts.items() if not c],
           'seconds':time.perf_counter()-start,'submitted':False,
           'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'model_source_sha256':hashlib.sha256(Path(__file__).with_name('solar_v2.py').read_bytes()).hexdigest(),
           'status':'LOCAL candidate; organizer metric/format parity remains to be checked'}
    output.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'results/v2/frozen-config.json')
    a=p.parse_args();predict(a.data,a.output,a.config)
