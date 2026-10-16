"""Full-resolution image ridge features and learned contour refinement."""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'solar-thin-v7'))
import precision as old
r=old.r;e=old.e

def features(image,disk):
    image=np.asarray(image,np.float32);disk=np.asarray(disk,bool)
    coverage=ndi.gaussian_filter(disk.astype(np.float32),48)
    bg=ndi.gaussian_filter(image*disk,48)/np.maximum(coverage,1e-4)
    norm=np.clip(image/np.maximum(bg,1),0,2)-1
    fs=[norm,image/255]
    for s in [3,7]:
        smooth=ndi.gaussian_filter(image,s);fs.append(np.clip((smooth-image)/np.maximum(smooth,1),-.5,.5))
    for s in [1,2,4]:
        xx=ndi.gaussian_filter(norm,s,order=(0,2))*s*s;yy=ndi.gaussian_filter(norm,s,order=(2,0))*s*s;xy=ndi.gaussian_filter(norm,s,order=(1,1))*s*s
        delta=np.sqrt((xx-yy)**2+4*xy*xy);large=(xx+yy+delta)/2;small=(xx+yy-delta)/2
        ridge=np.maximum(large,0)*np.exp(-.5*(small/np.maximum(np.abs(large),1e-5)/.5)**2)
        fs.extend([ridge,large])
    return np.stack(fs,axis=-1).astype(np.float32)

def geometry(labels):
    widths=np.full(int(labels.max())+1,np.inf)
    for k,box in enumerate(ndi.find_objects(labels),1):
        mask=labels[box]==k;boundary=(mask&~ndi.binary_erosion(mask)).sum();widths[k]=2*mask.sum()/max(boundary,1)
    distance,index=ndi.distance_transform_edt(labels==0,return_indices=True);nearest=labels[index[0],index[1]]
    return widths,distance,nearest

def refine(labels,prob,width,threshold,radius,geo=None):
    if not labels.any():return labels.copy()
    widths,distance,nearest=geometry(labels) if geo is None else geo;eligible=widths<=width;eligible[0]=False
    selected=eligible[nearest]&(distance<=radius)&(prob>=threshold)
    candidate=np.where(selected,nearest,0).astype(np.int32)
    original_size=np.bincount(labels.ravel(),minlength=len(widths));new_size=np.bincount(candidate.ravel(),minlength=len(widths))
    eligible&=(new_size>=8)&(new_size>=original_size*.25)&(new_size<=original_size*2)
    result=labels.copy();result[eligible[labels]]=0
    add=eligible[candidate]&(candidate>0)
    if np.any((result[add]>0)&(result[add]!=candidate[add])):raise ValueError('Refinement overlaps preserved instance')
    result[add]=candidate[add]
    return result
