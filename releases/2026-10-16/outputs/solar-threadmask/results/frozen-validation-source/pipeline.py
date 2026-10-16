"""Read authorized local COCO data, evaluate a frozen holdout, or write predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
from dataclasses import asdict
import numpy as np
from PIL import Image
from pycocotools import mask as mask_api
from threadmask import Config, masks, pq, segment, match_pq


def encoded_predictions(image):
    labels = segment(image)
    return [mask_api.encode(np.asfortranarray((labels == i).astype(np.uint8)))
            for i in np.unique(labels) if i != 0]


def encode_annotation(annotation, h, w):
    obj = annotation['segmentation']
    if isinstance(obj,list):
        return mask_api.merge(mask_api.frPyObjects(obj,h,w))
    if isinstance(obj.get('counts'),list):
        return mask_api.frPyObjects(obj,h,w)
    return obj


def pq_encoded(truth,predicted):
    pairs = []
    if truth and predicted:
        matrix = mask_api.iou(truth,predicted,[0]*len(predicted))
        for i,j in np.argwhere(matrix > 0.5):
            pairs.append((float(matrix[i,j]),int(i),int(j)))
    return match_pq(len(truth),len(predicted),pairs)


def holdout(file_name):
    # Same observing day always stays together, across annotators/instruments.
    day = Path(file_name).stem[:8]
    if len(day) != 8 or not day.isdigit():
        raise ValueError('Expected YYYYMMDD observation filename')
    return int(hashlib.sha256(day.encode()).hexdigest()[:8],16) % 5 == 0


def load_image(path):
    with Image.open(path) as im:
        if im.mode != 'L' or im.size != (2048,2048):
            raise ValueError(f'Expected grayscale 2048x2048: {path}')
        return np.asarray(im)


def decode(annotation, h, w):
    obj = annotation['segmentation']
    if isinstance(obj, list):
        obj = mask_api.merge(mask_api.frPyObjects(obj,h,w))
    elif isinstance(obj.get('counts'), list):
        obj = mask_api.frPyObjects(obj,h,w)
    return mask_api.decode(obj).astype(bool)


def evaluate(data, output):
    annotation_path = data/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    raw = annotation_path.read_bytes()
    coco = json.loads(raw)
    by_image = {}
    for ann in coco['annotations']:
        by_image.setdefault(ann['image_id'],[]).append(ann)
    rows, grouped = [], {}
    for info in coco['images']:
        filename = Path(info['file_name']).name
        if not holdout(filename):
            continue
        grouped.setdefault(filename,[]).append(info)
    for index,(filename,infos) in enumerate(sorted(grouped.items()),1):
        path = data/'train/train_images'/filename
        started = time.perf_counter()
        predicted = encoded_predictions(load_image(path))
        seconds = time.perf_counter()-started
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for info in infos:
            gt = [encode_annotation(a,info['height'],info['width']) for a in by_image.get(info['id'],[])]
            rows.append({'annotation_set':info['id'], 'file':filename, 'image_sha256':digest,
                         'seconds':seconds, 'predicted_instances':len(predicted), **pq_encoded(gt,predicted)})
        print(f'Validated {index}/{len(grouped)} observations',flush=True)
    if not rows:
        raise ValueError('No holdout observations found')
    # Average annotation sets within an observation before averaging observations.
    groups = {}
    for row in rows:
        if row['pq'] is not None:
            groups.setdefault(row['file'],[]).append(row['pq'])
    means = [float(np.mean(v)) for v in groups.values()]
    result = {'kind':'local_real_data_validation', 'metric':'diagnostic PQ; organizer parity unverified',
              'configuration':asdict(Config()), 'split':'sha256(YYYYMMDD) first8 modulo5 == 0',
              'mean_observation_pq':float(np.mean(means)) if means else None, 'rows':rows,
              'annotations_sha256':hashlib.sha256(raw).hexdigest(), 'python':platform.python_version()}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')


def predict(data, output):
    files = sorted((data/'test/test_images').glob('*.jp*g'))
    if not files:
        raise ValueError('No test images found')
    counts, manifest = {}, []
    with output.open('w',newline='',encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['filament_id','segmentation_rle'])
        for path in files:
            predicted = encoded_predictions(load_image(path))
            counts[path.stem] = len(predicted)
            for i, rle in enumerate(predicted,1):
                decoded = mask_api.decode(rle)
                if mask_api.encode(np.asfortranarray(decoded))['counts'] != rle['counts']:
                    raise ValueError('RLE round-trip failed')
                writer.writerow([f'{path.stem}_{i}',rle['counts'].decode('ascii')])
            manifest.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    sidecar = {'counts':counts,'images':manifest,'configuration':asdict(Config()),
               'csv_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
               'zero_prediction_images':[key for key,value in counts.items() if not value],
               'status':'local candidate only; zero-prediction image handling needs organizer verification'}
    output.with_suffix('.audit.json').write_text(json.dumps(sidecar,indent=2),encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['evaluate','predict'])
    p.add_argument('data',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    (evaluate if a.command == 'evaluate' else predict)(a.data,a.output)
