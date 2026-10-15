"""Version 2: image-derived disk and normalized dark contrast, CPU only."""
from dataclasses import dataclass
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


@dataclass(frozen=True)
class ConfigV2:
    resolution: int = 1024
    sigma: float = 24.0
    low: float = 0.04
    high_ratio: float = 1.5
    min_area: int = 20
    limb_margin: int = 24
    close_iterations: int = 1


def features(image, sigma=24.0, resolution=1024, limb_margin=24):
    image=np.asarray(image)
    if image.ndim!=2 or not np.isfinite(image).all() or min(image.shape)<16:
        raise ValueError('Expected finite grayscale image of at least 16x16')
    if resolution<32 or sigma<=0 or limb_margin<0:
        raise ValueError('Invalid feature configuration')
    resized=np.asarray(Image.fromarray(image.astype(np.float32)).resize(
        (resolution,resolution),Image.Resampling.BILINEAR),dtype=np.float32)
    threshold=max(float(np.percentile(resized,90))*0.22,1.0)
    labels,count=ndi.label(resized>threshold)
    if not count:
        return np.zeros(resized.shape,dtype=np.float32),np.zeros(resized.shape,dtype=bool)
    sizes=np.bincount(labels.ravel());sizes[0]=0
    disk=ndi.binary_fill_holes(labels==sizes.argmax())
    inside=ndi.distance_transform_edt(disk)>limb_margin
    # Normalize the blur by disk occupancy, avoiding dark outside-disk contamination.
    occupancy=ndi.gaussian_filter(disk.astype(np.float32),sigma)
    background=ndi.gaussian_filter(resized*disk,sigma)/np.maximum(occupancy,1e-4)
    score=(background-resized)/np.maximum(background,1.0)
    return score,inside


def from_features(score,inside,config):
    if config.low<=0 or config.high_ratio<1 or config.min_area<1 or config.close_iterations<0:
        raise ValueError('Invalid component configuration')
    candidates=(score>=config.low)&inside
    if config.close_iterations:
        candidates=ndi.binary_closing(candidates,iterations=config.close_iterations)&inside
    labels,count=ndi.label(candidates,structure=np.ones((3,3)))
    areas=np.bincount(labels.ravel())
    strong=set(np.unique(labels[(score>=config.low*config.high_ratio)&inside]))-{0}
    keep=[i for i in range(1,count+1) if i in strong and config.min_area<=areas[i]<=inside.sum()*0.03]
    mapping=np.zeros(count+1,dtype=np.int32);mapping[keep]=np.arange(1,len(keep)+1)
    return mapping[labels]


def segment_v2(image,config=ConfigV2()):
    score,inside=features(image,config.sigma,config.resolution,config.limb_margin)
    labels=from_features(score,inside,config)
    return np.asarray(Image.fromarray(labels).resize(
        (image.shape[1],image.shape[0]),Image.Resampling.NEAREST),dtype=np.int32)
