"""Second development hypothesis: add disjoint thin U-Net proposals to RF baseline."""
import experiment as e
import itertools
from scipy import ndimage as ndi
import numpy as np
import joblib
r=e.r;HERE=e.HERE/'supplement';HERE.mkdir(exist_ok=True)

def proposals(prob,disk,baseline,threshold):
    labels=r.instances(prob,disk,threshold,8,0)[::2,::2]
    occupied=ndi.binary_dilation(baseline[::2,::2]>0,iterations=2);items=[]
    for k,box in enumerate(ndi.find_objects(labels),1):
        mask=labels[box]==k
        if (mask&occupied[box]).any():continue
        y,x=np.nonzero(mask);eig=np.maximum(np.linalg.eigvalsh(np.cov(np.stack([y,x]),bias=True)),0)
        elong=np.sqrt((eig[-1]+1)/(eig[0]+1));per=(mask&~ndi.binary_erosion(mask)).sum()
        width=2*mask.sum()/max(per,1)
        if max(np.ptp(x),np.ptp(y))<20:continue
        items.append((k,float(elong),float(width)))
    return np.repeat(np.repeat(labels,2,0),2,1),items

def main():
    d=r.Dataset();old=r.read(r.HERE/'component-protocol.json');model=joblib.load(r.HERE/'component-filter.joblib')
    protocol={'tune':old['tune'],'assessment':d.protocol['split']['assessment'],
       'grid':list(itertools.product([.4,.55,.7],[2,3,5],[4,8])),
       'selection':'Max tune PQ with strictly greater thin recall and PQ than baseline; fallback baseline. All additions disjoint including 2-pixel margin at 1024.',
       'limitations':'Second hypothesis after hysteresis failed on tune. Reuses 19 tune images and previously inspected assessment data; exploratory development, not blind.',
       'source_sha256':r.sha(__file__)}
    if (HERE/'protocol.json').exists():raise FileExistsError('Preserve results')
    r.write(HERE/'protocol.json',protocol)
    import component_followup as cf
    def run(names,grid):
        out={'baseline':[]};out.update({str(g):[] for g in grid})
        for index,name in enumerate(names,1):
            labels,x=cf.candidate(d,name);keep=np.flatnonzero(model.predict(x)>=.2)+1 if len(x) else []
            mapping=np.zeros(labels.max()+1,np.int32);mapping[keep]=np.arange(1,len(keep)+1);base=mapping[labels]
            im,_,disk=d.arrays(name);prob=r.probabilities(d,name,'unet900')
            def add(key,lab):
                ms=e.matrices(d,name,lab)
                out[key].extend(e.score([(name,np.zeros((lab.max(),1)),ms)],type('All',(),{'predict':lambda self,x:np.ones(len(x))})(),0))
            add('baseline',base)
            for threshold in sorted({g[0] for g in grid}):
                proposed,items=proposals(prob,disk,base,threshold)
                for g in [g for g in grid if g[0]==threshold]:
                    ids=[k for k,elong,width in items if elong>=g[1] and width<=g[2]]
                    mm=np.zeros(proposed.max()+1,np.int32);mm[ids]=np.arange(1,len(ids)+1)+base.max()
                    lab=base+mm[proposed];add(str(g),lab)
            print('Supplement',index,len(names),flush=True)
        return out
    tune=run(protocol['tune'],[tuple(x) for x in protocol['grid']]);baseline=e.totals(tune['baseline']);trials=[{'params':list(g),**e.totals(tune[str(tuple(g))])} for g in protocol['grid']]
    eligible=[x for x in trials if x['pq']>baseline['pq'] and x['thin_recall']>baseline['thin_recall']]
    best=max(eligible,key=lambda x:x['pq']) if eligible else None
    r.write(HERE/'frozen.json',{'baseline':baseline,'trials':trials,'selected':best,'protocol_sha256':r.sha(HERE/'protocol.json')});print('FROZEN SUPPLEMENT',best,flush=True)
    if best:
        rows=run(protocol['assessment'],[tuple(best['params'])]);r.write(HERE/'assessment.json',{'results':{k:e.totals(v) for k,v in rows.items()},'rows':rows})
        print('SUPPLEMENT RESULT',{k:e.totals(v) for k,v in rows.items()},flush=True)
if __name__=='__main__':main()
