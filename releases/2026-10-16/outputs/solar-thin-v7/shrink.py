"""Conservative erosion of elongated predicted contours; no new instances."""
import contour as c
import numpy as np
from scipy import ndimage as ndi
import joblib,itertools
r=c.r;e=c.e;p=c.p;HERE=p.HERE/'shrink';HERE.mkdir(exist_ok=True)

def shrink(labels,width,radius,geo=None,inside=None):
    if not labels.any():return labels.copy()
    widths,eligible,_,_=c.geometry(labels) if geo is None else geo
    distance=ndi.distance_transform_edt(labels>0) if inside is None else inside
    selected=eligible&(widths<=width);remove=selected[labels]&(distance<=radius)
    trial=labels.copy();trial[remove]=0
    size=np.bincount(labels.ravel(),minlength=len(selected));remaining=np.bincount(trial.ravel(),minlength=len(selected))
    restore=np.flatnonzero(selected&((remaining<8)|(remaining<size*.25)))
    if len(restore):mask=np.isin(labels,restore);trial[mask]=labels[mask]
    return trial

def main():
    d=r.Dataset();old=r.read(r.HERE/'component-protocol.json');model=joblib.load(r.HERE/'component-filter.joblib')
    protocol={'tune':old['tune'],'assessment':d.protocol['split']['assessment'],'grid':list(itertools.product([8,16,32],[1,2,3])),
      'rule':'Erode elongated existing instances at full resolution; keep original instance if fewer than 8 pixels or 25% of original area would remain. No new IDs, no added pixels.',
      'selection':'Thin TP strictly increases, FP <= baseline, total TP >= baseline, PQ >= baseline on tune. Max PQ, tie first grid entry.',
      'limitations':'Fourth exploratory hypothesis this turn after learned filter, joint calibration, and growth. Reused development data, not blind.',
      'source_sha256':r.sha(__file__),'geometry_source_sha256':r.sha(p.HERE/'contour.py')}
    if (HERE/'protocol.json').exists():raise FileExistsError('Preserve run')
    r.write(HERE/'protocol.json',protocol)
    def run(names,grid):
        rows={'baseline':[]};rows.update({str(tuple(g)):[] for g in grid})
        for i,name in enumerate(names,1):
            labels=p.baseline(d,name,model);gt=c.truth(d,name);geo=c.geometry(labels) if labels.any() else None;inside=ndi.distance_transform_edt(labels>0)
            rows['baseline'].extend(c.metrics(name,labels,gt))
            for w,rad in grid:rows[str((w,rad))].extend(c.metrics(name,shrink(labels,w,rad,geo,inside),gt))
            print('Shrink',i,len(names),flush=True)
        return rows
    tune=run(protocol['tune'],[tuple(g) for g in protocol['grid']]);base=e.totals(tune['baseline']);trials=[{'params':g,**e.totals(tune[str(tuple(g))])} for g in protocol['grid']]
    valid=[v for v in trials if v['thin_tp']>base['thin_tp'] and v['fp']<=base['fp'] and v['tp']>=base['tp'] and v['pq']>=base['pq']]
    best=max(valid,key=lambda v:v['pq']) if valid else None
    r.write(HERE/'frozen.json',{'baseline':base,'trials':trials,'selected':best,'protocol_sha256':r.sha(HERE/'protocol.json')});print('SHRINK FROZEN',best,flush=True)
    if best:
        rows=run(protocol['assessment'],[tuple(best['params'])]);result={'results':{k:e.totals(v) for k,v in rows.items()},'rows':rows};r.write(HERE/'assessment.json',result);print('SHRINK RESULT',result['results'],flush=True)
if __name__=='__main__':main()
