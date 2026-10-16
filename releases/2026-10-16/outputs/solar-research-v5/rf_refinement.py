"""Tuning-only boundary follow-up, before opening assessment outcomes."""
import sys,itertools,shutil
from pathlib import Path
base=Path.cwd();p=base/'outputs/solar-research-v5';sys.path.insert(0,str(p))
from research import *
d=Dataset();initial=read(p/'rf-frozen.json')
assert not (p/'assessment.json').exists()
assert not (p/'rf-refinement-protocol.json').exists()
params=[dict(zip(['threshold','min_area','close'],x)) for x in itertools.product([.7,.8,.9],[20,60,120,240],[0,1])]
write(p/'rf-refinement-protocol.json',{'reason':'Initial tune optimum was at highest threshold and area boundary; extend this development search before assessment.','grid':params,'initial':initial,'selection':'Highest pooled tune PQ across initial best and follow-up; initial best wins ties.','source_sha256':sha(__file__),'assessment_accessed':False})
rows=[[] for _ in params]
for k,name in enumerate(d.protocol['split']['tune'],1):
    _,_,disk=d.arrays(name);prob=probabilities(d,name,'rf');truth=d.truth(name)
    for j,pred in enumerate(encoded_grid(prob,disk,params)):
        rows[j].extend({'file':name,'annotation_set':a,**instance_stats(gt,pred)} for a,gt in truth if gt)
    print('RF refinement',k,flush=True)
trials=[{'params':pa,**pool(rs)} for pa,rs in zip(params,rows)]
best=max(range(len(trials)),key=lambda j:trials[j]['pq'])
shutil.copyfile(p/'rf-frozen.json',p/'rf-frozen-initial.json')
if trials[best]['pq']>initial['tune']['pq']:
    new=dict(initial);new.update(params=params[best],tune=trials[best],refinement_protocol_sha256=sha(p/'rf-refinement-protocol.json'))
    write(p/'rf-frozen.json',new)
write(p/'rf-refinement.json',{'trials':trials,'selected':read(p/'rf-frozen.json'),'rows':rows[best]})
print('Final RF',read(p/'rf-frozen.json')['tune'],flush=True)
