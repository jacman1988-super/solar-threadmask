"""Actual image inference with annotation/cache access forbidden."""
import image_only as inf
import ridge as g
import numpy as np,joblib,hashlib
r=g.r;H=g.HERE
def main():
    d=r.Dataset();protocol=r.read(H/'protocol.json');frozen=r.read(H/'frozen.json');name=protocol['tune'][0]
    labels=g.old.baseline(d,name,joblib.load(r.HERE/'component-filter.joblib'));image=r.load_image(r.DATA/'train/train_images'/name);expected=labels
    if frozen['selected']:
        model=joblib.load(H/'ridge-forest.joblib');_,_,lowdisk=d.arrays(name);disk=np.repeat(np.repeat(lowdisk,2,0),2,1);geo=g.geometry(labels);widths,distance,nearest=geo;region=(widths[nearest]<=32)&(distance<=4);prob=np.zeros(labels.shape,np.float32)
        if region.any():prob[region]=model.predict(g.features(image,disk)[region])
        expected=g.refine(labels,prob,*frozen['selected']['params'],geo=geo)
    def forbidden(*a,**kw):raise AssertionError('Inference used annotations or cached probability maps')
    r.Dataset.__init__=forbidden;r.probabilities=forbidden
    actual=inf.segment(image);assert np.array_equal(actual,expected)
    assert set(np.unique(actual))==set(np.unique(labels))
    r.write(H/'runtime-check.json',{'file':name,'matches_research_exactly':True,'annotations_and_probability_cache_forbidden':True,'same_instance_ids':True,'labels_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'selected':frozen['selected']});print('Ridge runtime exact match',name,flush=True)
if __name__=='__main__':main()
