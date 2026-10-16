"""Image-only experimental inference requiring agreement of RF and U-Net."""
import experiment as e
from supplement import proposals
import runtime as base
import numpy as np
HERE=e.HERE/'consensus'
def combine(baseline,unet,forest,disk,threshold):
    proposed,items=proposals(unet,disk,baseline,.7);low=proposed[::2,::2]
    keep=[k for k,elong,width in items if elong>=2 and width<=4 and forest[low==k].mean()>=threshold]
    mapping=np.zeros(proposed.max()+1,np.int32);mapping[keep]=np.arange(1,len(keep)+1)+baseline.max()
    addition=mapping[proposed]
    if np.any((addition>0)&(baseline>0)):raise ValueError('Supplement overlaps baseline')
    return baseline+addition

def segment(image):
    frozen=e.r.read(HERE/'frozen.json');protocol=e.r.read(HERE/'protocol.json')
    if e.r.sha(HERE/'protocol.json')!=frozen['protocol_sha256']:raise ValueError('Consensus protocol changed')
    if e.r.sha(e.HERE/'consensus.py')!=protocol['source_sha256']:raise ValueError('Consensus source changed')
    original_protocol=e.r.read(e.HERE/'supplement/protocol.json')
    if e.r.sha(e.HERE/'supplement.py')!=original_protocol['source_sha256']:raise ValueError('Proposal source changed')
    baseline=base.segment(image,'rf_filter')
    if frozen['selected'] is None:return baseline
    im,disk=base.prepare(image)
    return combine(baseline,base.probability_map(im,disk,'unet900'),base.probability_map(im,disk,'rf'),disk,frozen['selected']['threshold'])
