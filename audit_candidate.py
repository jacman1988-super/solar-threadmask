"""Independent schema, instance-geometry and input audit. No upload or network."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_api

def validate_rows(rows,counts,shape=(2048,2048)):
    observed={key:[] for key in counts};encoded={key:[] for key in counts};seen=set()
    for row in rows:
        if set(row)!={'filament_id','segmentation_rle'} or any(not isinstance(row[k],str) or not row[k] for k in ['filament_id','segmentation_rle']):
            raise ValueError('Every CSV row must contain exactly two nonempty string fields')
        identifier=row['filament_id']
        if identifier in seen:raise ValueError('Duplicate filament ID')
        seen.add(identifier)
        try:key,suffix=identifier.rsplit('_',1);number=int(suffix)
        except (ValueError,AttributeError):raise ValueError('Invalid filament ID')
        if key not in observed or number<1 or str(number)!=suffix:raise ValueError('Unknown image ID or invalid object suffix')
        rle={'size':list(shape),'counts':row['segmentation_rle'].encode('ascii')}
        decoded=mask_api.decode(rle)
        if decoded.shape!=shape or not decoded.any():raise ValueError('Empty or invalid instance mask')
        observed[key].append(number);encoded[key].append(rle)
    for key,expected in counts.items():
        if sorted(observed[key])!=list(range(1,expected+1)):raise ValueError('Per-image object count/suffix mismatch: '+key)
        masks=encoded[key]
        if masks:
            overlap=np.asarray(mask_api.iou(masks,masks,[0]*len(masks)))
            if np.any(np.triu(overlap>0,k=1)):raise ValueError('Predicted instances overlap: '+key)
    return {'decoded_instances':len(rows),'images':len(counts),'zero_detection_images':sum(v==0 for v in counts.values()),'overlapping_instance_pairs':0}

def audit(project,data=None):
    read=lambda name:json.loads((project/name).read_text(encoding='utf-8'))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    metadata=read('results/v3/candidate.audit.json');candidate=project/'results/v3/candidate.csv'
    if digest(candidate)!=metadata['csv_sha256']:raise ValueError('CSV checksum differs from recorded export')
    with candidate.open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('Invalid CSV header')
        rows=list(reader)
    expected={Path(row['file']).stem for row in metadata['images']}
    if set(metadata['counts'])!=expected:raise ValueError('Image registry mismatch')
    result=validate_rows(rows,metadata['counts'])
    empty={key for key,value in metadata['counts'].items() if value==0}
    if empty!=set(metadata['zero_prediction_images']):raise ValueError('Empty-image audit differs')
    checked=0
    if data:
        files={p.name:p for p in (data/'test/test_images').glob('*.jp*g')}
        if set(files)!={row['file'] for row in metadata['images']}:raise ValueError('Test input set differs')
        for row in metadata['images']:
            if digest(files[row['file']])!=row['sha256']:raise ValueError('Test image checksum differs')
            checked+=1
    result.update({'source_input_hashes_checked':checked,'csv_sha256':digest(candidate),'submitted':False,'private_validator_tested':False})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path);a=p.parse_args()
    print(json.dumps(audit(Path(__file__).resolve().parent,a.data.resolve() if a.data else None),indent=2))
