"""Read-only local release checks; does not submit or call a network service."""
import csv,hashlib,json
from pathlib import Path
from audit_candidate import audit as audit_instances

def check(project):
    read=lambda name:json.loads((project/name).read_text(encoding='utf-8'))
    sha=lambda name:hashlib.sha256((project/name).read_bytes()).hexdigest()
    author=read('author.json');form=read('submission-draft.json')
    assert author['full_name']==form['fields']['full_names']
    model=read('results/v3/frozen-model.json');assessment=read('results/v3/blind-evaluation.json')
    protocol=read('results/v3/protocol.json');audit=read('results/v3/candidate.audit.json')
    for name,digest in protocol['source_sha256'].items():assert sha(name)==digest,name
    assert model['protocol_sha256']==sha('results/v3/protocol.json')
    assert audit['csv_sha256']==sha('results/v3/candidate.csv')
    assert audit['frozen_model_sha256']==assessment['frozen_model_sha256']==sha('results/v3/frozen-model.json')
    with (project/'results/v3/candidate.csv').open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    assert len({r['filament_id'] for r in rows})==len(rows)==950
    assert len(audit['counts'])==180 and sum(audit['counts'].values())==len(rows)
    for name,metadata in form['local_files'].items():
        assert (project/name).stat().st_size==metadata['bytes'] and sha(name)==metadata['sha256']
    reproduction=read('results/v3/reproduction-audit.json')
    assert reproduction['all_checks_passed']
    assert reproduction['reproduced_csv_sha256']==sha('results/v3/candidate.csv')
    assert reproduction['reproduced_model_sha256']==sha('results/v3/frozen-model.json')
    instance_review=audit_instances(project)
    return {'author':author['full_name'],'local_checks_passed':True,'csv_rows':len(rows),
        'instance_review':instance_review,
        'images':180,'reproduction_confirmed':True,'published':False,'submitted':False,
        'remaining_external_steps':['Participant review','Approved public repository URL','Explicit approval for any submission','Private platform validation']}

if __name__=='__main__':print(json.dumps(check(Path(__file__).resolve().parent),indent=2))
