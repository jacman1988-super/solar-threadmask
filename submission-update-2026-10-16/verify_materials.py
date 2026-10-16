"""Verify the dated public update without network access."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def verify():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    files={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts}
    if files!=set(manifest['files']):raise ValueError('Material inventory differs')
    for name,digest in manifest['files'].items():
        if '.private.' in name or Path(name).suffix in {'.gguf','.csv'}:raise ValueError('Unexpected private/model/prediction asset: '+name)
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('SHA256 differs: '+name)
    return {'verified':True,'files':len(files),'fitted_weights':sum(n.endswith('.joblib') for n in files),'competition_submitted':False}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
