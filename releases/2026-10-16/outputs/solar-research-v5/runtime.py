"""Image-only inference with verified models and no stale image cache."""
from functools import lru_cache
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import joblib
from research import HERE,read,sha,channels,pixel_features,instances

METHODS=('rf','rf_filter','unet300','unet600','unet900')

def prepare(image):
    image=np.asarray(image)
    if image.ndim!=2 or image.shape!=(2048,2048) or image.dtype.kind not in 'uif':
        raise ValueError('Expected 2048x2048 grayscale image with integer pixels in 0..255')
    if not np.isfinite(image).all() or image.min()<0 or image.max()>255 or np.any(image!=np.floor(image)):
        raise ValueError('Pixels must be finite integers in 0..255; normalize other formats explicitly')
    im=np.asarray(Image.fromarray(image.astype(np.uint8)).resize((1024,1024),Image.Resampling.BILINEAR),np.float32)
    labels,n=ndi.label(im>max(np.percentile(im,90)*.22,1));sizes=np.bincount(labels.ravel());sizes[0]=0
    disk=ndi.binary_fill_holes(labels==sizes.argmax()) if n else np.zeros(im.shape,bool)
    return im,disk

def verified_artifact(path,digest):
    path=path.resolve()
    if not isinstance(digest,str) or len(digest)!=64 or sha(path)!=digest:
        raise ValueError(f'Model integrity check failed: {path.name}')
    return path

@lru_cache(maxsize=4)
def _forest(path,digest):
    verified_artifact(path,digest)
    return joblib.load(path)

def probability_map(im,disk,method):
    if method=='rf':
        frozen=read(HERE/'rf-frozen.json');digest=frozen['model_sha256']
        path=verified_artifact(HERE/'pixel-forest.joblib',digest)
        features=pixel_features(im,disk)
        prob=_forest(path,digest).predict(features.reshape(-1,features.shape[-1])).reshape(im.shape)
    elif method in ('unet300','unet600','unet900'):
        from neural import predict
        frozen=read(HERE/(method+'-frozen.json'))
        path=verified_artifact(HERE/(method+'.pt'),frozen['model_sha256'])
        prob=predict(channels(im,disk),path)
    else:raise ValueError('Unknown probability model')
    return np.asarray(prob,dtype=np.float16).astype(np.float32)

def component_features(labels,prob,im,disk):
    low=labels[::2,::2];ch=channels(im,disk);gap=ndi.distance_transform_edt(disk)/1024;rows=[]
    for k,box in enumerate(ndi.find_objects(low),1):
        if box is None:raise ValueError('Expected contiguous instance IDs')
        mask=low[box]==k;y,x=np.nonzero(mask);area=len(x)
        eigen=np.maximum(np.linalg.eigvalsh(np.cov(np.stack([y,x]),bias=True)),0)
        elong=np.sqrt((eigen[-1]+1)/(eigen[0]+1));perimeter=(mask&~ndi.binary_erosion(mask)).sum();h,w=mask.shape
        q=prob[box][mask];v=ch[0][box][mask];fine=ch[1][box][mask]
        rows.append([np.log1p(area),np.log(elong),area/(h*w),np.log1p(perimeter**2/(4*np.pi*area)),q.mean(),np.percentile(q,90),q.min(),q.std(),v.mean(),np.percentile(v,10),v.std(),gap[box][mask].mean(),fine.mean(),fine.max()])
    return np.asarray(rows,np.float32).reshape(-1,14)

def segment(image,method=None):
    method=method or read(HERE/'model-selection-followup.json')['method']
    if method not in METHODS:raise ValueError(f'Unknown method: {method}')
    im,disk=prepare(image)
    if method=='rf_filter':
        frozen=read(HERE/'component-frozen.json')
        protocol_path=HERE/'component-protocol.json'
        if sha(protocol_path)!=frozen['protocol_sha256']:
            raise ValueError('Component protocol integrity check failed')
        protocol=read(protocol_path)
        if protocol['pixel_model_sha256']!=read(HERE/'rf-frozen.json')['model_sha256']:
            raise ValueError('Component filter and pixel model versions differ')
        path=verified_artifact(HERE/'component-filter.joblib',frozen['model_sha256'])
        prob=probability_map(im,disk,'rf');labels=instances(prob,disk,**protocol['candidate'])
        features=component_features(labels,prob,im,disk);model=_forest(path,frozen['model_sha256'])
        threshold=frozen['threshold']
        if not 0<=threshold<=1:raise ValueError('Invalid component threshold')
        keep=np.flatnonzero(model.predict(features)>=threshold)+1 if len(features) else np.zeros(0,int)
        mapping=np.zeros(int(labels.max())+1,np.int32);mapping[keep]=np.arange(1,len(keep)+1)
        return mapping[labels]
    frozen=read(HERE/(method+'-frozen.json'))
    return instances(probability_map(im,disk,method),disk,**frozen['params'])
