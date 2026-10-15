"""Verify the local package; never performs network calls or submissions."""
import csv,hashlib,json
from pathlib import Path
from check_release import check
from audit_candidate import validate_rows

def verify(root):
    original=check(root);release=json.loads((root/'final-release.json').read_text())
    for name,expected in release['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Release checksum mismatch: '+name)
    alternative=root/'results/precision-guard/candidate.csv'
    metadata=json.loads(alternative.with_suffix('.audit.json').read_text())
    with alternative.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('Invalid alternative header')
        audit=validate_rows(list(reader),metadata['counts'])
    if len(metadata['images'])!=180 or audit['images']!=180:raise ValueError('Incomplete alternative inference')
    if hashlib.sha256(alternative.read_bytes()).hexdigest()!=metadata['csv_sha256']:raise ValueError('Alternative checksum mismatch')
    return {'recommended_candidate':release['recommended_candidate'],'original_checks':original,'alternative_audit':audit,'ready_for_local_review':True,'submitted':False}

if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent),indent=2))
