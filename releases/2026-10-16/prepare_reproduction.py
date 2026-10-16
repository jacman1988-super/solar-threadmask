"""Prepare a new, path-compatible model-reproduction workspace without altering frozen evidence."""
import argparse,shutil,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);a=p.parse_args()
    if a.destination.exists():p.error('Destination must be new')
    a.destination.mkdir(parents=True)
    for name in ['solar-threadmask','solar-research-v5','solar-thin-v6','solar-thin-v7','solar-ridge-v8']:
        src=ROOT/'outputs'/name;dst=a.destination/'outputs'/name;dst.mkdir(parents=True)
        for item in src.glob('*.py'):shutil.copy2(item,dst/item.name)
    v5=a.destination/'outputs/solar-research-v5'
    shutil.copy2(ROOT/'outputs/solar-research-v5/protocol.json',v5/'protocol.json')
    # Fixed model settings/protocols; outcome files are deliberately not copied.
    for name in ['rf-frozen.json','rf-refinement-protocol.json','selection-followup-protocol.json','model-selection-followup.json']:
        source=ROOT/'outputs/solar-research-v5'/name
        if source.exists():shutil.copy2(source,v5/name)
    shutil.copy2(ROOT/'predict.py',a.destination/'predict.py')
    (a.destination/'work').mkdir();(a.destination/'PREPARED.json').write_text(json.dumps({'prepared_only':True,'data_or_weights_copied':False,'protocol':'Original pixel-model split retained; new fits/results must be kept separate from published evidence.'},indent=2),encoding='utf-8')
    print('Prepared clean workspace. Supply the separately licensed dataset and follow REPRODUCE.md.')
if __name__=='__main__':main()
