import sys,time,json,subprocess,os
from pathlib import Path
base=Path.cwd();p=base/'outputs/solar-research-v5';sys.path.insert(0,str(p))
from organizer_metric import pool
def read(path):return json.loads(path.read_text())
def write(path,x):path.write_text(json.dumps(x,indent=2))
write(p/'selection-followup-protocol.json',{'rule':'Select among RF, best CNN snapshot (chosen on original 55 tune images), and RF plus component filter, by PQ on the same component validation subset. Tie favors RF then CNN then component filter. No assessment metrics used.','limitation':'Development subset reused after initial parameter search; not an independent validation set. Component filter is trained on the complementary portion of the former tune set.','frozen_before_assessment_results':not (p/'assessment.json').exists()})
while not (p/'neural-selected.json').exists() or not (p/'component-frozen.json').exists():time.sleep(3)
net=read(p/'neural-selected.json')['selected'];valid=set(read(p/'component-protocol.json')['tune'])
values={m:pool(r for r in read(p/(m+'-tuning.json'))['rows'] if r['file'] in valid)['pq'] for m in ['rf',net]}
values['rf_filter']=read(p/'component-frozen.json')['selected']['pq']
selected=max(values,key=values.get)
write(p/'model-selection-followup.json',{'method':selected,'same_validation_pq':values,'images':len(valid),'rule':'Frozen in selection-followup-protocol.json; assessment not consulted','assessment_results_present_at_selection':(p/'assessment.json').exists()})
print('Selected',selected,values,flush=True)
while not (p/'assessment.json').exists():time.sleep(3)
env=os.environ.copy()
with (base/'work/solar-v5-component-assessment.log').open('w',encoding='utf-8') as f:
    subprocess.run([sys.executable,'outputs/solar-research-v5/component_followup.py','assess'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
print('COMPLETE',flush=True)
