"""Check both experimental image-only paths against cached development inference."""
import inference as inf
import component_followup as cf
import numpy as np,joblib,hashlib
r=inf.r;H=inf.HERE
def main():
    d=r.Dataset();component=joblib.load(r.HERE/'component-filter.joblib');precision=joblib.load(H/'precision-filter.joblib');protocol=r.read(H/'protocol.json');selected=r.read(H/'frozen.json')['selected'];joint=r.read(H/'joint/frozen.json')['selected'];assert selected and joint
    examples={}
    for name in protocol['tune']:
        labels,x=cf.candidate(d,name);qualities=component.predict(x) if len(x) else np.zeros(0);ids=np.flatnonzero(qualities>=.2)+1;original=inf.remap(labels,ids);quality=qualities[ids-1]
        im,_,disk=d.arrays(name);rf=r.probabilities(d,name,'rf');unet=r.probabilities(d,name,'unet900')
        expected={'learned':inf.learned(original,im,disk,rf,unet,precision,selected['pixel_threshold'],selected['quality']),'joint':inf.joint(original,quality,rf,unet,disk,joint['quality'],joint['support'])}
        for method,output in expected.items():
            if method not in examples and not np.array_equal(output,original):examples[method]=(name,output)
        if len(examples)==2:break
    assert len(examples)==2
    def forbidden(*a,**kw):raise AssertionError('Inference read annotations or image probability cache')
    r.Dataset.__init__=forbidden;r.probabilities=forbidden;results=[]
    for method,(name,expected) in examples.items():
        image=r.load_image(r.DATA/'train/train_images'/name);actual=inf.segment(image,method);assert np.array_equal(actual,expected)
        results.append({'method':method,'file':name,'exact_cached_research_match':True,'annotation_and_image_cache_access_forbidden':True,'instances':int(actual.max()),'labels_sha256':hashlib.sha256(actual.tobytes()).hexdigest()});print('Exact runtime match',method,name,flush=True)
    r.write(H/'runtime-checks.json',results)
if __name__=='__main__':main()
