"""Image-only thin proposal supplement; does not read annotations or caches."""
import experiment as e
import runtime as base
from supplement import proposals
import numpy as np
HERE=e.HERE/'supplement'
def combine(baseline,prob,disk,params):
    threshold,elongation,width=params
    proposed,items=proposals(prob,disk,baseline,threshold)
    keep=[k for k,elong,w in items if elong>=elongation and w<=width]
    mapping=np.zeros(proposed.max()+1,np.int32)
    mapping[keep]=np.arange(1,len(keep)+1)+baseline.max()
    addition=mapping[proposed]
    if np.any((baseline>0)&(addition>0)):raise ValueError('Supplement overlaps baseline')
    return baseline+addition

def segment(image):
    frozen=e.r.read(HERE/'frozen.json');protocol=e.r.read(HERE/'protocol.json')
    if e.r.sha(HERE/'protocol.json')!=frozen['protocol_sha256']:raise ValueError('Supplement protocol changed')
    if e.r.sha(e.HERE/'supplement.py')!=protocol['source_sha256']:raise ValueError('Supplement source changed')
    selected=frozen['selected']
    baseline=base.segment(image,'rf_filter')
    if selected is None:return baseline
    im,disk=base.prepare(image)
    return combine(baseline,base.probability_map(im,disk,'unet900'),disk,selected['params'])
