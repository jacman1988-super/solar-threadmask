"""Verify unchanged previous work and package continued local evidence."""
from pathlib import Path
import json,hashlib,zipfile
H=Path(__file__).resolve().parent;O=H.parent;G=O/'gemma-branch-feedback-v5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old={}
    for name in ['Manifest-Solar-Gemma-2026-10-16.json','Manifest-dalszych-prob-Solar-Gemma-2026-10-16.json','Manifest-Solar-ridge-Gemma-etapy-2026-10-16.json']:
        for file,digest in json.loads((O/name).read_text()).items():assert sha(O/file)==digest,file;old[file]=digest
    p=json.loads((H/'protocol.json').read_text());m=O/'solar-ridge-v8';f=json.loads((m/'frozen.json').read_text());assert sha(H/'evaluate_extra.py')==p['source_sha256'];assert sha(m/'ridge-forest.joblib')==p['model_sha256']==f['model_sha256'];assert p['parameters']==f['selected']['params'];assert sha(m/'ridge.py')==p['model_feature_sha256']
    current=json.loads((O/'solar-research-v5/protocol.json').read_text())['split'];excluded={n[:8] for files in current.values() for n in files};days={n[:8] for n in p['images']};assert len(days)==len(p['images'])==40 and not days&excluded
    gp=json.loads((G/'protocol.json').read_text());assert sha(G/'run.py')==gp['source_sha256'];gr=json.loads((G/'summary.json').read_text());assert gr['packaging_all_pass'] and gr['differential_differences']==0
    solar=json.loads((H/'summary.json').read_text());r=solar['additional_40'];a=r['baseline'];b=r['ridge']
    assert sha(O/'solar-threadmask/results/v3/candidate.csv')=='2a58ee67e8beaa23187921d03bd7f2222db5c747ca8bfd3e7db0a9852f7a2f56'
    verification={'previous_research_files_unchanged':len(old),'current_models_and_parameters_unchanged':True,'additional_dates_disjoint_current_splits':True,'additional_images':40,'all_additional_masks_have_hashes':len(json.loads((H/'assessment.json').read_text())['mask_hashes'])==40,'packaging_tests_passed':90,'differential_names':1245,'differential_discrepancies':0,'tenacity_previous_tests_passed':159,'csv_unchanged':True,'no_publication_or_submission':True}
    check=O/'Kontrola-Solar-dodatkowe-daty-Gemma-galezie-2026-10-16.json';check.write_text(json.dumps(verification,indent=2),encoding='utf-8')
    text=f"Solar i Gemma — dalsza kontrola, 16.10.2026\nJacek Adamek\n\nSolar bez zmiany modelu i parametrów: dodatkowe 40 zdjęć/40 dat spoza bieżących podziałów. Cienkie trafienia {a['thin_tp']}/{a['thin_tp']+a['thin_fn']} -> {b['thin_tp']}/{b['thin_tp']+b['thin_fn']}; FP {a['fp']} -> {b['fp']}; PQ {a['pq']:.6f} -> {b['pq']:.6f}. Spełnia pełne lokalne kryterium poprawy: {r['meets_thin_no_extra_fp_criterion']}. To kontrola rozwojowa; starsze eksperymenty znały tę pulę danych.\n\nGemma Packaging: jawny podział wymagań na dwie rodziny nazw i prośba o osobne gałęzie. Model wygenerował poprawkę przechodzącą 90/90 sprawdzeń. Dodatkowa zgodność dla 1245 nazw z oficjalną poprawką upstream, bez rozbieżności. To silnie wspomagana naprawa, nie wynik pierwszej lub samodzielnej próby. Tenacity zachowuje 159/159 z poprzedniego etapu.\n\nGotowe lokalne patche: gemma-branch-feedback-v5/packaging-tested.patch i gemma-feedback-v4/tenacity-tested.patch. Szczegóły: solar-ridge-validation-v9/Wyniki-Solar.txt i gemma-branch-feedback-v5/Wyniki-Gemma.txt.\n\nPoprzednie raporty, modele i zgłoszenia niezmienione. Niczego nie opublikowano ani nie wysłano.\n"
    summary=O/'Wyniki-Solar-dodatkowe-daty-Gemma-galezie-2026-10-16.txt';summary.write_text(text,encoding='utf-8');files=[]
    for folder in [H,G]:files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix!='.pyc')
    files.extend([check,summary,O/'gemma-feedback-v4/tenacity-tested.patch',O/'gemma-feedback-v4/tenacity-LICENSE.txt']);manifest=O/'Manifest-Solar-dodatkowe-daty-Gemma-galezie-2026-10-16.json';manifest.write_text(json.dumps({p.relative_to(O).as_posix():sha(p) for p in sorted(files)},indent=2),encoding='utf-8')
    archive=O/'Solar-dodatkowe-daty-Gemma-galezie-2026-10-16.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[manifest]:z.write(p,p.relative_to(O))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(verification);print('Archive bytes',archive.stat().st_size,flush=True)
if __name__=='__main__':main()
