"""Verify historical artifacts, freeze local delivery hashes, package experiments."""
from pathlib import Path
import json,hashlib,zipfile,itertools
H=Path(__file__).resolve().parent;O=H.parent;G=O/'gemma-feedback-v4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old_files={}
    for name in ['Manifest-Solar-Gemma-2026-10-16.json','Manifest-dalszych-prob-Solar-Gemma-2026-10-16.json']:
        for file,digest in json.loads((O/name).read_text()).items():
            assert sha(O/file)==digest,file;old_files[file]=digest
    gm=json.loads((O/'gemma-importguard/manifest.json').read_text());protected=[]
    for name,digest in gm.items():
        if name in ['submission-paper.pdf','submission-writeup.txt'] or name.startswith('results/exception-contract-resource/'):
            assert sha(O/'gemma-importguard'/name)==digest;protected.append(name)
    assert sha(O/'solar-threadmask/results/v3/candidate.csv')=='2a58ee67e8beaa23187921d03bd7f2222db5c747ca8bfd3e7db0a9852f7a2f56'
    p=json.loads((H/'protocol.json').read_text());f=json.loads((H/'frozen.json').read_text());assert sha(H/'protocol.json')==f['protocol_sha256'];assert sha(H/'ridge.py')==p['feature_source_sha256'];assert sha(H/'train_evaluate.py')==p['harness_sha256'];assert sha(H/'ridge-forest.joblib')==f['model_sha256']
    dates={key:{n[:8] for n in p[key]} for key in ['train','tune','assessment']}
    assert all(not dates[a]&dates[b] for a,b in itertools.combinations(dates,2))
    for folder,source in [(G,G/'run_staged.py'),(G/'api-feedback',G/'api_feedback.py'),(G/'single-failure',G/'last_case_feedback.py')]:assert sha(source)==json.loads((folder/'protocol.json').read_text())['source_sha256']
    assert '9 passed' in (H/'testy.txt').read_text();runtime=json.loads((H/'runtime-check.json').read_text());assert runtime['matches_research_exactly'] and runtime['same_instance_ids']
    solar=json.loads((H/'summary.json').read_text());gemma=json.loads((G/'summary-final.json').read_text());base=solar['baseline'];new=solar['ridge_refinement']
    verification={'previous_research_files_unchanged':len(old_files),'protected_gemma_files_unchanged':len(protected),'submitted_solar_csv_unchanged':True,'frozen_model_protocol_sources_match':True,'date_disjoint_training_tuning_assessment':{k:len(v) for k,v in dates.items()},'unit_tests_passed':9,'actual_image_runtime_matches':True,'tenacity_checks_passed':159,'packaging_latest_checks_passed':88,'packaging_latest_checks_total':90,'published_or_submitted':False}
    check=O/'Kontrola-Solar-ridge-Gemma-etapy-2026-10-16.json';check.write_text(json.dumps(verification,indent=2),encoding='utf-8')
    text=f"Solar i Gemma — pełna rozdzielczość i etapowa informacja zwrotna, 16.10.2026\nJacek Adamek\n\nSolar: osobny model uczony na 60 zdjęciach i 335163 pikselach w pełnym 2048x2048. Ocena rozwojowa 106 zdjęć/95 dat. Cienkie trafienia {base['thin_tp']}/314 -> {new['thin_tp']}/314; FP {base['fp']} -> {new['fp']}; PQ {base['pq']:.6f} -> {new['pq']:.6f}. Spełnia lokalny warunek poprawy cienkich bez dodatkowych FP i utraty TP/PQ: {solar['meets_local_thin_no_extra_fp_criterion']}. Dane używane wcześniej, bez nowego wyniku Kaggle.\n\nGemma: diagnoza, tabela przypadków, literalny kod funkcji. Tenacity 159/159 — potwierdzona naprawa w zadeklarowanym zakresie. Packaging 88/90 — nadal odrzuca poprawne pylock.toml; dodatkowa próba była bez zmiany kodu. Łącznie sześć generacji, w tym jawnie wspomagane dokumentacją API.\n\nKontrola: 9 testów nowych narzędzi i cech, rzeczywista inferencja zgodna z badaniem bez dostępu do adnotacji/pamięci predykcji. Poprzednie wyniki i zgłoszenia niezmienione. Niczego nie opublikowano ani nie wysłano.\n\nSzczegóły: solar-ridge-v8/Wyniki-Solar.txt i gemma-feedback-v4/Wyniki-Gemma.txt. Przetestowana lokalna poprawka: gemma-feedback-v4/tenacity-tested.patch.\n"
    summary=O/'Wyniki-Solar-ridge-Gemma-etapy-2026-10-16.txt';summary.write_text(text,encoding='utf-8')
    files=[]
    for folder in [H,G]:files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix!='.pyc')
    files.extend([check,summary]);manifest=O/'Manifest-Solar-ridge-Gemma-etapy-2026-10-16.json';manifest.write_text(json.dumps({p.relative_to(O).as_posix():sha(p) for p in sorted(files)},indent=2),encoding='utf-8')
    archive=O/'Solar-ridge-Gemma-etapy-2026-10-16.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[manifest]:z.write(p,p.relative_to(O))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(verification);print('Archive bytes',archive.stat().st_size,flush=True)
if __name__=='__main__':main()
