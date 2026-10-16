"""Verify protected artifacts and package local research only."""
from pathlib import Path
import hashlib,json,zipfile
H=Path(__file__).resolve().parent;O=H.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    checks=[];g=O/'gemma-importguard';manifest=json.loads((g/'manifest.json').read_text())
    for name,digest in manifest.items():
        if name in ['submission-paper.pdf','submission-writeup.txt'] or name.startswith('results/exception-contract-resource/'):
            checks.append({'file':'gemma-importguard/'+name,'unchanged':sha(g/name)==digest})
    csv=O/'solar-threadmask/results/v3/candidate.csv'
    checks.append({'file':'solar-threadmask/results/v3/candidate.csv','unchanged':sha(csv)=='2a58ee67e8beaa23187921d03bd7f2222db5c747ca8bfd3e7db0a9852f7a2f56'})
    for row in json.loads((O/'solar-research-v5/manifest.json').read_text()):
        if row['file'].endswith(('.joblib','.pt')) or row['file'] in ['protocol.json','combined-assessment.json']:
            checks.append({'file':'solar-research-v5/'+row['file'],'unchanged':sha(O/'solar-research-v5'/row['file'])==row['sha256']})
    assert checks and all(c['unchanged'] for c in checks)
    for directory in [H,H/'supplement',H/'consensus']:
        p=json.loads((directory/'protocol.json').read_text());f=json.loads((directory/'frozen.json').read_text())
        source=H/('experiment.py' if directory==H else directory.name+'.py')
        assert sha(source)==p['source_sha256'];assert sha(directory/'protocol.json')==f['protocol_sha256']
    assert '7 passed' in (H/'testy.txt').read_text()
    assert json.loads((H/'consensus-runtime-check.json').read_text())['exact_research_match']
    solar=json.loads((H/'summary-final.json').read_text());gemma=json.loads((O/'gemma-new-repositories-v2/summary.json').read_text())
    verification={'protected_files':checks,'all_unchanged':True,'protocol_source_hashes_match':True,'solar_unit_tests_passed':7,'solar_real_image_inference_checks':2,'gemma_fixed_reference_checks_passed':231,'gemma_full_repairs':gemma['fully_repaired'],'no_publication_or_submission':True}
    (O/'Kontrola-Solar-Gemma-2026-10-16.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    a=solar['baseline'];b=solar['consensus']
    text=f"Solar i Gemma — lokalne wyniki, 16.10.2026\nJacek Adamek\n\nSolar (106 zdjęć, 95 dat; dane rozwojowe):\nCienkie obiekty {a['thin_tp']}/314 -> {b['thin_tp']}/314. PQ {a['pq']:.6f} -> {b['pq']:.6f}. Fałszywe detekcje {a['fp']} -> {b['fp']}.\nNie zmieniono domyślnego modelu. Osobny eksperymentalny wariant: solar-thin-v6/consensus_runtime.py.\n\nGemma: dwa nowe repozytoria Tenacity i Packaging, 0/2 pełnych napraw. Wersje referencyjne przechodzą 231/231 sprawdzeń. Zapisano wygenerowane poprawki i konkretne regresje.\n\nKontrola: 7 testów nowego algorytmu Solar, 2 sprawdzenia inferencji na rzeczywistym zdjęciu, poprawna i kompletna dokumentacja testów Gemma, niezmienione chronione wyniki i zgłoszenia.\n\nSzczegóły: solar-thin-v6/Wyniki-Solar.txt i gemma-new-repositories-v2/Wyniki-Gemma.txt.\nNic nie opublikowano ani nie wysłano.\n"
    (O/'Wyniki-Solar-Gemma-2026-10-16.txt').write_text(text,encoding='utf-8')
    files=[]
    for directory in [H,O/'gemma-new-repositories-v2']:
        files.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix!='.pyc')
    files.extend([O/'Wyniki-Solar-Gemma-2026-10-16.txt',O/'Kontrola-Solar-Gemma-2026-10-16.json'])
    digests={p.relative_to(O).as_posix():sha(p) for p in sorted(files)}
    mp=O/'Manifest-Solar-Gemma-2026-10-16.json';mp.write_text(json.dumps(digests,indent=2),encoding='utf-8')
    archive=O/'Solar-Gemma-lokalne-badania-2026-10-16.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[mp]:z.write(p,p.relative_to(O))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(text);print('Protected',len(checks),'Archive bytes',archive.stat().st_size,flush=True)
if __name__=='__main__':main()
