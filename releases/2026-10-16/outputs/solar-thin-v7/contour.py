"""Refine existing thin contours without adding instance IDs."""
import precision as p
import numpy as np
from scipy import ndimage as ndi
import joblib,itertools
r=p.r;e=p.e;HERE=p.HERE/'contour';HERE.mkdir(exist_ok=True)

def geometry(labels):
    widths=np.full(int(labels.max())+1,np.inf);eligible=np.zeros(len(widths),bool)
    for k,box in enumerate(ndi.find_objects(labels),1):
        mask=labels[box]==k;y,x=np.nonzero(mask);per=(mask&~ndi.binary_erosion(mask)).sum();widths[k]=2*len(x)/max(per,1)
        eig=np.maximum(np.linalg.eigvalsh(np.cov(np.stack([y,x]),bias=True)),0);elong=np.sqrt((eig[-1]+1)/(eig[0]+1))
        eligible[k]=elong>=2 and max(np.ptp(x),np.ptp(y))>=40
    distance,index=ndi.distance_transform_edt(labels==0,return_indices=True);nearest=labels[index[0],index[1]]
    return widths,eligible,distance,nearest

def refine(labels,prob,width,radius,support,geo=None):
    if not labels.any():return labels.copy()
    widths,eligible,distance,nearest=geometry(labels) if geo is None else geo
    selected=eligible&(widths<=width)
    add=(labels==0)&(distance<=radius)&selected[nearest]&(prob>=support)
    result=labels.copy();result[add]=nearest[add]
    return result

def truth(d,name):
    result=[]
    for a,gt in d.truth(name):
        thin=[]
        for g in gt:
            mask=r.coco.decode(g).astype(bool);y,x=np.nonzero(mask);per=(mask&~ndi.binary_erosion(mask)).sum()
            thin.append(bool(len(x) and 2*len(x)/max(per,1)<=8 and max(np.ptp(x),np.ptp(y))>=40))
        result.append((a,gt,np.asarray(thin,bool)))
    return result

def metrics(name,labels,gt):
    pred=r.encode_labels(labels);rows=[]
    for a,truth_masks,thin in gt:
        m=np.asarray(r.coco.iou(truth_masks,pred,[0]*len(pred)),np.float32) if truth_masks and pred else np.zeros((len(truth_masks),len(pred)),np.float32)
        hit=(m>.5).any(axis=1)
        rows.append({'file':name,'annotation_set':a,**e.stats_from_iou(m),'thin_tp':int((hit&thin).sum()),'thin_fn':int((~hit&thin).sum())})
    return rows

def main():
    d=r.Dataset();old=r.read(r.HERE/'component-protocol.json');model=joblib.load(r.HERE/'component-filter.joblib')
    protocol={'tune':old['tune'],'assessment':d.protocol['split']['assessment'],'grid':list(itertools.product([4,8,12],[1,2,3],[0,.25,.5])),
      'rule':'Grow existing elongated masks only, at full 2048 resolution. Assign background to nearest original instance, never overwrite any original instance pixel. No new instance IDs. Width is 2*area/inner-boundary pixels; elongation >=2 and max span >=40.',
      'selection':'Tune thin TP > baseline, FP <= baseline, total TP >= baseline and PQ >= baseline; max PQ, tie first grid entry. Otherwise baseline.',
      'limitations':'Third experiment in this turn, using repeatedly inspected tuning and assessment data. Exploratory development; no independent guarantee.', 'source_sha256':r.sha(__file__)}
    if (HERE/'protocol.json').exists():raise FileExistsError('Preserve run')
    r.write(HERE/'protocol.json',protocol)
    def run(names,grid):
        rows={'baseline':[]};rows.update({str(tuple(g)):[] for g in grid})
        for i,name in enumerate(names,1):
            labels=p.baseline(d,name,model);prob=np.repeat(np.repeat(r.probabilities(d,name,'rf'),2,0),2,1);gt=truth(d,name);geo=geometry(labels) if labels.any() else None
            rows['baseline'].extend(metrics(name,labels,gt))
            for w,rad,s in grid:rows[str((w,rad,s))].extend(metrics(name,refine(labels,prob,w,rad,s,geo),gt))
            print('Contour',i,len(names),flush=True)
        return rows
    tuning=run(protocol['tune'],[tuple(g) for g in protocol['grid']]);base=e.totals(tuning['baseline']);trials=[{'params':g,**e.totals(tuning[str(tuple(g))])} for g in protocol['grid']]
    valid=[v for v in trials if v['thin_tp']>base['thin_tp'] and v['fp']<=base['fp'] and v['tp']>=base['tp'] and v['pq']>=base['pq']]
    best=max(valid,key=lambda v:v['pq']) if valid else None
    r.write(HERE/'frozen.json',{'baseline':base,'trials':trials,'selected':best,'protocol_sha256':r.sha(HERE/'protocol.json')});print('CONTOUR FROZEN',best,flush=True)
    if best:
        rows=run(protocol['assessment'],[tuple(best['params'])]);result={'results':{k:e.totals(v) for k,v in rows.items()},'rows':rows}
        r.write(HERE/'assessment.json',result);print('CONTOUR RESULT',result['results'],flush=True)
if __name__=='__main__':main()
