"""Execute this package's reviewed notebook without installing a Jupyter frontend."""
import argparse,contextlib,io,json,os,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,help='Authorized MAGFiLO directory (Solar only)');p.add_argument('--full-export',action='store_true',help='Solar: execute all 180 images; output must not already exist');p.add_argument('--output',type=Path,required=True,help='New executed-notebook output path')
    a=p.parse_args();package=Path(__file__).resolve().parent;solar=(package/'pipeline-update.ipynb').exists();source=package/('pipeline-update.ipynb' if solar else 'review-update.ipynb')
    output=a.output.resolve()
    if output.exists():p.error('Output already exists')
    if a.full_export and not solar:p.error('--full-export is Solar-only')
    if solar:
        if not a.data or not (a.data/'test/test_images').is_dir():p.error('--data must contain authorized test/test_images')
        os.environ['SOLAR_DATA']=str(a.data.resolve());os.environ['SOLAR_EXPORT_ALL']='1' if a.full_export else '0'
    for name in ['SOLAR_RELEASE','SOLAR_WEIGHTS','GEMMA_RELEASE']:os.environ.pop(name,None)
    os.chdir(package.parent);nb=json.loads(source.read_text(encoding='utf-8'));namespace={};count=0;start=time.perf_counter()
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        count+=1;stream=io.StringIO()
        with contextlib.redirect_stdout(stream):exec(compile(''.join(cell['source']),'<public-notebook-cell>','exec'),namespace)
        cell['execution_count']=count;cell['outputs']=[{'output_type':'stream','name':'stdout','text':stream.getvalue().splitlines(True)}];print(stream.getvalue(),flush=True)
    nb['metadata']['execution_engine']='Plain CPython code-cell execution, not a Jupyter server';nb['metadata']['full_test_set_export_executed']=solar and a.full_export
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as file:json.dump(nb,file,indent=2)
    print('Executed notebook code cells:',count,'seconds:',round(time.perf_counter()-start,2))
if __name__=='__main__':main()
