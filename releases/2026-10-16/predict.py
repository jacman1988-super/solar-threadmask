"""Run the frozen baseline or ridge refinement with separately supplied reference weights."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def install_weights(runtime,folder):
    original=runtime.verified_artifact;allowed={'pixel-forest.joblib','component-filter.joblib','ridge-forest.joblib'}
    def resolve(path,digest):
        if path.name not in allowed:raise ValueError('Unknown model artifact')
        return original(folder/path.name,digest)
    runtime.verified_artifact=resolve
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('image',type=Path);p.add_argument('output',type=Path);p.add_argument('--weights',type=Path,required=True);p.add_argument('--model',choices=['baseline','ridge'],default='ridge');a=p.parse_args()
    required=['pixel-forest.joblib','component-filter.joblib']+(['ridge-forest.joblib'] if a.model=='ridge' else [])
    if not a.image.is_file():p.error('Input image does not exist')
    if a.output.exists():p.error('Output exists; choose a new file')
    missing=[name for name in required if not (a.weights/name).is_file()]
    if missing:p.error('Reference weights are distributed separately from source. See REPRODUCE.md. Missing: '+', '.join(missing))
    (ROOT/'work').mkdir(exist_ok=True)
    sys.path.insert(0,str(ROOT/'outputs/solar-ridge-v8'))
    import image_only as ridge
    from PIL import Image
    import numpy as np
    install_weights(ridge.base,a.weights.resolve())
    try:
        with Image.open(a.image) as im:image=np.asarray(im)
        result=ridge.segment(image) if a.model=='ridge' else ridge.base.segment(image,'rf_filter')
        with a.output.open('xb') as f:np.save(f,result)
    except (ValueError,OSError) as exc:p.error(str(exc))
    print('Local instance labels saved:',int(result.max()))
if __name__=='__main__':main()
