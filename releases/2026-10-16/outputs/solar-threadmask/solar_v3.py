"""Learned component rejection over frozen v2 masks. CPU, image-only inference."""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.special import expit
from solar_v2 import features,from_features

FEATURE_NAMES=['log_area','log_elongation','extent','log_compactness',
 'mean_contrast','p90_contrast','std_contrast','min_disk_gap','mean_disk_gap',
 'log_bbox_aspect','strong_fraction']

def candidates(image,config):
    score,inside=features(image,config.sigma,config.resolution,config.limb_margin)
    labels=from_features(score,inside,config)
    gap=ndi.distance_transform_edt(inside)/config.resolution
    rows=[]
    for identifier,box in enumerate(ndi.find_objects(labels),1):
        if box is None:raise ValueError('Candidate labels must be contiguous')
        mask=labels[box]==identifier;y,x=np.nonzero(mask);area=len(x)
        cov=np.cov(np.stack([y,x]),bias=True)
        eigen=np.maximum(np.linalg.eigvalsh(cov),0)
        elongation=np.sqrt((eigen[-1]+1)/(eigen[0]+1))
        boundary=mask&~ndi.binary_erosion(mask)
        perimeter=int(boundary.sum());height,width=mask.shape
        contrast=score[box][mask];distance=gap[box][mask]
        rows.append([np.log1p(area),np.log(elongation),area/(height*width),
          np.log1p(perimeter**2/(4*np.pi*area)),float(contrast.mean()),
          float(np.percentile(contrast,90)),float(contrast.std()),float(distance.min()),
          float(distance.mean()),np.log(max(height,width)/min(height,width)),
          float(np.mean(contrast>=config.low*config.high_ratio))])
    full=np.asarray(Image.fromarray(labels).resize((image.shape[1],image.shape[0]),Image.Resampling.NEAREST),dtype=np.int32)
    return full,np.asarray(rows,dtype=np.float64).reshape(-1,len(FEATURE_NAMES))

def fit_filter(x,y,regularization=.01):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES) or len(x)!=len(y) or not len(x):
        raise ValueError('Invalid training shapes')
    if not np.isfinite(x).all() or not np.isin(y,[0,1]).all() or len(np.unique(y))!=2:
        raise ValueError('Finite features and both binary classes required')
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-6);z=(x-mean)/scale
    def objective(w):
        logits=z@w[1:]+w[0];error=expit(logits)-y
        loss=np.mean(np.logaddexp(0,logits)-y*logits)+regularization*np.dot(w[1:],w[1:])/2
        gradient=np.r_[error.mean(),z.T@error/len(y)+regularization*w[1:]]
        return loss,gradient
    initial=np.zeros(z.shape[1]+1);initial[0]=np.log(y.mean()/(1-y.mean()))
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':500,'gtol':1e-7,'ftol':1e-12})
    if not result.success:raise RuntimeError('Filter optimization failed: '+str(result.message))
    return {'feature_names':FEATURE_NAMES,'mean':mean.tolist(),'scale':scale.tolist(),
       'weights':result.x.tolist(),'regularization':regularization,'objective':float(result.fun),
       'optimizer_success':bool(result.success),'optimizer_message':str(result.message)}

def probabilities(x,model):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(x).all():raise ValueError('Invalid component features')
    if model['feature_names']!=FEATURE_NAMES:raise ValueError('Feature schema mismatch')
    w=np.asarray(model['weights']);z=(x-np.asarray(model['mean']))/np.asarray(model['scale'])
    return expit(w[0]+z@w[1:])

def filter_labels(labels,probability,threshold):
    labels=np.asarray(labels);probability=np.asarray(probability)
    if not 0<=threshold<=1 or labels.ndim!=2 or int(labels.max(initial=0))!=len(probability):raise ValueError('Invalid filter input')
    mapping=np.zeros(len(probability)+1,dtype=np.int32)
    keep=np.flatnonzero(probability>=threshold)+1;mapping[keep]=np.arange(1,len(keep)+1)
    return mapping[labels]

def segment_v3(image,config,model):
    labels,x=candidates(image,config)
    return filter_labels(labels,probabilities(x,model),model['threshold'])
