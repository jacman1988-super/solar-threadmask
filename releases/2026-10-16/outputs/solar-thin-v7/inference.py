"""Separate image-only experimental inference; never submits or changes defaults."""
import precision as p
from supplement import proposals
import runtime as base
import numpy as np
from pathlib import Path
r=p.r;HERE=p.HERE

def remap(labels,keep):
    mapping=np.zeros(int(labels.max())+1,np.int32)
    mapping[np.asarray(keep,int)]=np.arange(1,len(keep)+1)
    return mapping[labels]

def prepared(image):
    im,disk=base.prepare(image);rf=base.probability_map(im,disk,'rf');unet=base.probability_map(im,disk,'unet900')
    frozen=r.read(r.HERE/'component-frozen.json');protocol=r.read(r.HERE/'component-protocol.json')
    if r.sha(r.HERE/'component-protocol.json')!=frozen['protocol_sha256']:raise ValueError('Component protocol changed')
    if protocol['pixel_model_sha256']!=r.read(r.HERE/'rf-frozen.json')['model_sha256']:raise ValueError('Pixel model mismatch')
    model_path=base.verified_artifact(r.HERE/'component-filter.joblib',frozen['model_sha256']);model=base._forest(model_path,frozen['model_sha256'])
    labels=r.instances(rf,disk,**protocol['candidate']);x=base.component_features(labels,rf,im,disk)
    quality=model.predict(x) if len(x) else np.zeros(0)
    ids=np.flatnonzero(quality>=frozen['threshold'])+1
    return im,disk,rf,unet,remap(labels,ids),quality[ids-1]

def learned(base_labels,im,disk,rf,unet,model,pixel_threshold,quality):
    labels=r.instances(unet,disk,pixel_threshold,8,0);eligible=p.shape_indices(labels)
    from scipy import ndimage as ndi
    occupied=ndi.binary_dilation(base_labels[::2,::2]>0,iterations=2);low=labels[::2,::2]
    eligible=np.array([k for k in eligible if not (occupied&(low==k+1)).any()],int)
    x=p.features(labels,rf,unet,im,disk)
    keep=eligible[model.predict(x[eligible])>=quality]+1 if len(eligible) else []
    addition=remap(labels,keep);addition=np.where(addition>0,addition+base_labels.max(),0)
    if np.any((base_labels>0)&(addition>0)):raise ValueError('Overlapping additions')
    return base_labels+addition

def joint(original,quality,rf,unet,disk,q,s):
    labels,items=proposals(unet,disk,original,.7);low=labels[::2,::2]
    keep=[k for k,elong,width in items if elong>=2 and width<=4 and rf[low==k].mean()>=s]
    baseline=remap(original,np.flatnonzero(quality>=q)+1);addition=remap(labels,keep)
    addition=np.where(addition>0,addition+baseline.max(),0)
    if np.any((baseline>0)&(addition>0)):raise ValueError('Overlapping additions')
    return baseline+addition

def substitute(original,proposed,mapping):
    if len(mapping)!=int(original.max()):raise ValueError('Replacement mapping size')
    selected=np.flatnonzero(np.asarray(mapping)>=0)
    values=np.asarray(mapping)[selected]
    if len(np.unique(values))!=len(values):raise ValueError('A proposed instance cannot replace two instances')
    result=original.copy()
    for k in selected:result[original==k+1]=0
    for k in selected:
        mask=proposed==mapping[k]+1
        if not mask.any() or (result[mask]>0).any():raise ValueError('Empty or overlapping replacement')
        result[mask]=k+1
    return result

def segment(image,method):
    if method=='replacement':
        import replace_masks as rep
        protocol=r.read(HERE/'replacement/protocol.json');frozen=r.read(HERE/'replacement/frozen.json')
        if r.sha(HERE/'replacement/protocol.json')!=frozen['protocol_sha256'] or r.sha(HERE/'replace_masks.py')!=protocol['source_sha256']:raise ValueError('Replacement experiment changed')
        if frozen['selected'] is None:return base.segment(image,'rf_filter')
        pf=r.read(HERE/'frozen.json')
        if pf['model_sha256']!=protocol['model_sha256']:raise ValueError('Precision model differs')
        path=base.verified_artifact(HERE/'precision-filter.joblib',protocol['model_sha256']);model=base._forest(path,protocol['model_sha256'])
        im,disk,rf,unet,original,_=prepared(image);proposed=r.instances(unet,disk,.7,8,0)
        bx=p.features(original,rf,unet,im,disk);ux=p.features(proposed,rf,unet,im,disk)
        qb=model.predict(bx) if len(bx) else np.zeros(0);qu=model.predict(ux) if len(ux) else np.zeros(0)
        b=r.encode_labels(original);u=r.encode_labels(proposed)
        pair=np.asarray(r.coco.iou(b,u,[0]*len(u)),np.float32) if b and u else np.zeros((len(b),len(u)),np.float32)
        return substitute(original,proposed,rep.choose(pair,qb,qu,*frozen['selected']['params']))
    if method=='shrink':
        import shrink as sh
        protocol=r.read(HERE/'shrink/protocol.json');frozen=r.read(HERE/'shrink/frozen.json')
        if r.sha(HERE/'shrink/protocol.json')!=frozen['protocol_sha256'] or r.sha(HERE/'shrink.py')!=protocol['source_sha256'] or r.sha(HERE/'contour.py')!=protocol['geometry_source_sha256']:raise ValueError('Shrink experiment changed')
        baseline=base.segment(image,'rf_filter');chosen=frozen['selected']
        return baseline if chosen is None else sh.shrink(baseline,*chosen['params'])
    if method not in ['learned','joint']:raise ValueError('Expected learned, joint or shrink method')
    folder=HERE if method=='learned' else HERE/'joint';source=HERE/('precision.py' if method=='learned' else 'joint.py')
    protocol=r.read(folder/'protocol.json');frozen=r.read(folder/'frozen.json')
    if r.sha(folder/'protocol.json')!=frozen['protocol_sha256'] or r.sha(source)!=protocol['source_sha256']:raise ValueError('Experiment protocol/source changed')
    chosen=frozen['selected']
    if chosen is None:return base.segment(image,'rf_filter')
    im,disk,rf,unet,original,quality=prepared(image)
    if method=='joint':return joint(original,quality,rf,unet,disk,chosen['quality'],chosen['support'])
    path=base.verified_artifact(HERE/'precision-filter.joblib',frozen['model_sha256']);model=base._forest(path,frozen['model_sha256'])
    return learned(original,im,disk,rf,unet,model,chosen['pixel_threshold'],chosen['quality'])
