"""Finish already authorized local experiments, never submit or publish."""
import os,sys,subprocess,time,json
from pathlib import Path
base=Path.cwd();out=base/'outputs/solar-research-v5'
env=os.environ.copy();env['PYTHONPATH']=str(base/'work/python-packages')+';'+str(base/'work/solar-ml-packages');env['OMP_NUM_THREADS']='4'
def wait_for(path):
    while not path.exists(): time.sleep(3)
def run(args,log):
    print('Starting',args,flush=True)
    with (base/'work'/log).open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('Finished',args,flush=True)
wait_for(out/'rf-training.json')
run(['outputs/solar-research-v5/research.py','tune','--method','rf'],'solar-v5-rf-tune.log')
wait_for(out/'neural-loss.json')
for method in ['unet300','unet600','unet900']:
    run(['outputs/solar-research-v5/research.py','tune','--method',method],'solar-v5-'+method+'-tune.log')
scores={m:json.loads((out/(m+'-frozen.json')).read_text())['tune']['pq'] for m in ['unet300','unet600','unet900']}
best=max(scores,key=scores.get)
(out/'neural-selected.json').write_text(json.dumps({'tune_pq_by_checkpoint':scores,'selected':best,'rule':'Highest tuning PQ, earliest snapshot on a tie; no assessment outcomes accessed.'},indent=2))
run(['outputs/solar-research-v5/research.py','assess','--methods','rf',best],'solar-v5-assessment.log')
print('COMPLETE',flush=True)
