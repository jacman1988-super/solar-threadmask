"""Package local follow-up experiments and verify unchanged historical artifacts."""
from pathlib import Path
import hashlib,json,zipfile
H=Path(__file__).resolve().parent;O=H.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=json.loads((O/'Manifest-Solar-Gemma-2026-10-16.json').read_text());unchanged={name:sha(O/name)==digest for name,digest in old.items()};assert all(unchanged.values())
    protected=[];g=O/'gemma-importguard';gm=json.loads((g/'manifest.json').read_text())
    for name,digest in gm.items():
        if name in ['submission-paper.pdf','submission-writeup.txt'] or name.startswith('results/exception-contract-resource/'):
            assert sha(g/name)==digest;protected.append('gemma-importguard/'+name)
    csv=O/'solar-threadmask/results/v3/candidate.csv';assert sha(csv)=='2a58ee67e8beaa23187921d03bd7f2222db5c747ca8bfd3e7db0a9852f7a2f56'
    for directory,source in [(H,'precision.py'),(H/'joint','joint.py'),(H/'contour','contour.py'),(H/'shrink','shrink.py'),(H/'replacement','replace_masks.py')]:
        p=json.loads((directory/'protocol.json').read_text());f=json.loads((directory/'frozen.json').read_text());assert sha(H/source)==p['source_sha256'];assert sha(directory/'protocol.json')==f['protocol_sha256']
    gf=O/'gemma-feedback-v3'
    for directory,source in [(gf,gf/'run.py'),(gf/'round2',gf/'round2.py')]:assert sha(source)==json.loads((directory/'protocol.json').read_text())['source_sha256']
    assert '15 passed' in (H/'testy.txt').read_text();runtime=json.loads((H/'runtime-checks.json').read_text());assert len(runtime)==2 and all(r['exact_cached_research_match'] for r in runtime)
    gemma=json.loads((gf/'summary.json').read_text());assert gemma['fixed_reference_checks_passed']==249
    verification={'previous_research_files_checked':len(unchanged),'all_previous_research_unchanged':True,'protected_gemma_artifacts_checked':len(protected),'submitted_solar_csv_unchanged':True,'frozen_sources_and_protocols_match':True,'new_unit_checks_passed':15,'actual_image_inference_checks_passed':2,'gemma_fixed_reference_checks_passed':249,'no_publication_or_submission':True}
    check=O/'Kontrola-dalszych-prob-Solar-Gemma-2026-10-16.json';check.write_text(json.dumps(verification,indent=2),encoding='utf-8')
    text='Solar i Gemma — dalsze lokalne próby, 16.10.2026\nAutor: Jacek Adamek\n\nNie uzyskano jeszcze wymaganej poprawy Solar ani pełnej naprawy obu przypadków Gemma. Wszystkie wyniki, także nieudane, zachowano.\n\nSolar: pięć nowych hipotez, 105 ustawień na danych strojenia. Dwa wybrane warianty oceniono na 106 zdjęciach. Wyuczony filtr: cienkie 82/314, FP 550, PQ 0.271485 — bez zmiany względem bazowego modelu. Kalibracja: cienkie 82/314, FP 546, PQ 0.269002 — mniej FP, ale niższa ogólna czułość i jakość. Pozostałe trzy hipotezy odrzucono na strojeniu. Nie zastąpiono obecnego modelu. Dane oceny były używane wcześniej; brak niezależnego testu.\n\nGemma: dwie rundy informacji zwrotnej, cztery nowe generacje, nadal 0/2 pełnych napraw. Tenacity: niekompletna odpowiedź, następnie brak zmiany kodu. Packaging: przywrócenie błędu, następnie nadal niepoprawna obsługa końcowego znaku nowej linii. Referencje upstream przechodzą 249/249 sprawdzeń.\n\nKontrola: 15 testów nowych funkcji i walidacji poprawek, dwa rzeczywiste sprawdzenia inferencji Solar. Poprzednie wyniki i wysłane pliki są niezmienione. Niczego nie opublikowano ani nie wysłano.\n\nSzczegóły: solar-thin-v7/Wyniki-Solar.txt oraz gemma-feedback-v3/Wyniki-Gemma.txt.\n'
    summary=O/'Dalsze-proby-Solar-Gemma-2026-10-16.txt';summary.write_text(text,encoding='utf-8')
    files=[]
    for folder in [H,gf]:files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix!='.pyc')
    files.extend([check,summary]);manifest=O/'Manifest-dalszych-prob-Solar-Gemma-2026-10-16.json';manifest.write_text(json.dumps({p.relative_to(O).as_posix():sha(p) for p in sorted(files)},indent=2),encoding='utf-8')
    archive=O/'Solar-Gemma-dalsze-proby-2026-10-16.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[manifest]:z.write(p,p.relative_to(O))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(verification);print('Archive bytes',archive.stat().st_size,flush=True)
if __name__=='__main__':main()
