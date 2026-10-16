"""Compare annotation-free inference with cached research inference."""
import experiment as e
import thin_runtime as t
import component_followup as cf
import joblib,numpy as np

def main():
    d=e.r.Dataset();protocol=e.r.read(t.HERE/'protocol.json');selected=e.r.read(t.HERE/'frozen.json')['selected'];assert selected
    model=joblib.load(e.r.HERE/'component-filter.joblib');choice=None
    for name in protocol['tune']:
        labels,x=cf.candidate(d,name);keep=np.flatnonzero(model.predict(x)>=.2)+1 if len(x) else []
        mapping=np.zeros(labels.max()+1,np.int32);mapping[keep]=np.arange(1,len(keep)+1);baseline=mapping[labels]
        _,_,disk=d.arrays(name);expected=t.combine(baseline,e.r.probabilities(d,name,'unet900'),disk,selected['params'])
        if expected.max()>baseline.max():choice=(name,baseline,expected);break
    assert choice,'No actual supplemented image to check'
    name,baseline,expected=choice;image=e.r.load_image(e.r.DATA/'train/train_images'/name)
    def forbidden(*a,**kw):raise AssertionError('Inference read training annotations/cache')
    e.r.Dataset.__init__=forbidden;e.r.probabilities=forbidden
    actual=t.segment(image)
    assert np.array_equal(expected,actual),'Image-only output differs from research'
    assert np.array_equal(actual[baseline>0],baseline[baseline>0]),'Baseline instance changed'
    import hashlib
    e.r.write(e.HERE/'runtime-check.json',{'file':name,'baseline_instances':int(baseline.max()),'supplemented_instances':int(actual.max()),'exact_research_match':True,'baseline_masks_unchanged':True,'annotations_and_image_cache_forbidden':True,'labels_sha256':hashlib.sha256(actual.tobytes()).hexdigest()})
    print('Runtime exact match',name,baseline.max(),actual.max(),flush=True)
if __name__=='__main__':main()
