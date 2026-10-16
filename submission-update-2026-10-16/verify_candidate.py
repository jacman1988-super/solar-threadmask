"""Verify exact candidate bytes, schema, masks, counts and authorized input hashes."""
import argparse,json,hashlib,csv,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--images',type=Path,required=True);a=p.parse_args()
    expected=json.loads((ROOT/'candidate-ridge-v8.audit.json').read_text())
    if sha(a.candidate)!=expected['csv_sha256']:raise ValueError('CSV differs from the frozen candidate')
    with a.candidate.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('CSV columns differ')
        rows=list(reader)
    module=ROOT.parent/'releases/2026-10-16/outputs/solar-threadmask/audit_candidate.py'
    spec=importlib.util.spec_from_file_location('independent_auditor',module);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    result=m.validate_rows(rows,expected['counts'])
    files={p.name:p for p in a.images.glob('*.jp*g')}
    if set(files)!={r['file'] for r in expected['images']}:raise ValueError('Input registry differs')
    for row in expected['images']:
        if sha(files[row['file']])!=row['sha256']:raise ValueError('Input hash differs: '+row['file'])
    print(json.dumps({**result,'exact_csv_sha256':sha(a.candidate),'input_hashes_verified':len(files),'submitted':False},indent=2))
if __name__=='__main__':main()
