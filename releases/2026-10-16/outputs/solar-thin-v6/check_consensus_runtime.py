"""Image-only integrity check on an actual consensus addition."""
import experiment as e
import consensus_runtime as t
import component_followup as cf
import numpy as np,joblib,hashlib
def main():
    d=e.r.Dataset();p=e.r.read(t.HERE/'protocol.json');threshold=e.r.read(t.HERE/'frozen.json')['selected']['threshold'];model=joblib.load(e.r.HERE/'component-filter.joblib')
    choice=None
    for name in p['tune']:
        labels,x=cf.candidate(d,name);keep=np.flatnonzero(model.predict(x)>=.2)+1 if len(x) else []
        mm=np.zeros(labels.max()+1,np.int32);mm[keep]=np.arange(1,len(keep)+1);baseline=mm[labels];_,_,disk=d.arrays(name)
        expected=t.combine(baseline,e.r.probabilities(d,name,'unet900'),e.r.probabilities(d,name,'rf'),disk,threshold)
        if expected.max()>baseline.max():choice=(name,baseline,expected);break
    assert choice
    name,baseline,expected=choice;image=e.r.load_image(e.r.DATA/'train/train_images'/name)
    def forbidden(*a,**kw):raise AssertionError('Inference read annotations or image cache')
    e.r.Dataset.__init__=forbidden;e.r.probabilities=forbidden
    actual=t.segment(image);assert np.array_equal(expected,actual)
    assert np.array_equal(actual[baseline>0],baseline[baseline>0])
    e.r.write(e.HERE/'consensus-runtime-check.json',{'file':name,'baseline_instances':int(baseline.max()),'consensus_instances':int(actual.max()),'exact_research_match':True,'baseline_masks_unchanged':True,'annotations_and_image_cache_forbidden':True,'labels_sha256':hashlib.sha256(actual.tobytes()).hexdigest()});print('Consensus runtime exact match',name,flush=True)
if __name__=='__main__':main()
