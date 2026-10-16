"""Offline integrity verification for this prepared source release."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def verify():
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    omitted={'RELEASE_MANIFEST.json','VALIDATION.json'}
    def ignored(p):return any(x in p.relative_to(ROOT).parts for x in ['work','weights','.git','__pycache__','.pytest_cache','.venv'])
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and not ignored(p) and p.name not in omitted}
    if actual!=set(manifest['files']):raise ValueError('Release file inventory differs')
    for name,digest in manifest['files'].items():
        p=ROOT/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Checksum differs: '+name)
        if p.suffix.lower() in {'.joblib','.pt','.gguf','.jpeg','.jpg','.png','.csv','.zip'}:raise ValueError('Excluded asset in public source release: '+name)
        if p.suffix.lower() in {'.py','.json','.txt','.md','.cff'} and ('jacman1988'+'@'+'gmail.com') in p.read_text(encoding='utf-8-sig',errors='replace'):raise ValueError('Account email in public package: '+name)
    return {'integrity_verified':True,'files':len(actual),'weights_or_competition_inputs_distributed':False}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
