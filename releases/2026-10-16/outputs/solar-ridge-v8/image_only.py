"""Image-only inference for the frozen full-resolution ridge experiment."""
import ridge as g
import runtime as base
import numpy as np
r=g.r;H=g.HERE
def segment(image):
    protocol=r.read(H/'protocol.json');frozen=r.read(H/'frozen.json')
    if r.sha(H/'protocol.json')!=frozen['protocol_sha256'] or r.sha(H/'ridge.py')!=protocol['feature_source_sha256']:raise ValueError('Ridge protocol/features changed')
    labels=base.segment(image,'rf_filter');selected=frozen['selected']
    if selected is None:return labels
    _,lowdisk=base.prepare(image);disk=np.repeat(np.repeat(lowdisk,2,0),2,1)
    path=base.verified_artifact(H/'ridge-forest.joblib',frozen['model_sha256']);model=base._forest(path,frozen['model_sha256'])
    geo=g.geometry(labels);widths,distance,nearest=geo;region=(widths[nearest]<=32)&(distance<=4);prob=np.zeros(labels.shape,np.float32)
    if region.any():prob[region]=model.predict(g.features(image,disk)[region])
    return g.refine(labels,prob,*selected['params'],geo=geo)
