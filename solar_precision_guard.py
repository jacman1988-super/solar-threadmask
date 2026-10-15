"""Experimental image-only precision guard; original candidate unchanged."""
import numpy as np
from solar_v3 import candidates,probabilities,filter_labels,FEATURE_NAMES

def guard_mask(features,model,rule,threshold=.075):
    features=np.asarray(features,dtype=np.float64)
    if features.ndim!=2 or features.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(features).all():raise ValueError('Invalid features')
    keep=probabilities(features,model)>=threshold
    if rule:
        index=rule['feature'];cutoff=rule['cutoff'];direction=rule['direction']
        if not isinstance(index,int) or not 0<=index<len(FEATURE_NAMES) or direction not in ['ge','le'] or not np.isfinite(cutoff):raise ValueError('Invalid guard rule')
        if rule['name']!=FEATURE_NAMES[index]:raise ValueError('Feature schema differs')
        keep &= features[:,index]>=cutoff if direction=='ge' else features[:,index]<=cutoff
    return keep

def segment_precision_guard(image,config,model,rule):
    labels,features=candidates(image,config)
    return filter_labels(labels,guard_mask(features,model,rule).astype(float),.5)
