import hashlib, json, os, sys, tempfile
from pathlib import Path
import yaml, pytest, coverage
root=Path.cwd()
names=['AGENTS.md','CLAUDE.md','CODEX.md','ops/e2e-test-plan.md','openspec/capabilities/research-reporting/spec.md','openspec/capabilities/verification/spec.md','scripts/experiment_template.py','python/carnot/reporting/primary_publication.py','ops/exclusion_manifest.yaml','openspec/change-proposals/research-roadmap-vNEXT.md','python/carnot/reporting/roadmap_contract.py','python/carnot/reporting/v685_authority_lifecycle.py','python/carnot/reporting/v708_contract_custody.py','python/carnot/reporting/v708_capstone_inputs.py','python/carnot/reporting/v708_capstone.py','results/experiment_8192_v708_contract_custody.json','results/experiment_8204_v708_capstone.json','openspec/change-proposals/research-roadmap-v708-preserved-20261006.md','research-roadmap.yaml','research-complete.yaml','ops/conductor-log.md']
rows=[]
for name in names+['research-roadmap-next.yaml']:
 p=root/name
 rows.append(dict(path=name,exists=p.is_file(),required=name!='research-roadmap-next.yaml',sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
for name in ['python','pytest','coverage','ruff','mypy']:
 p=root/'.venv/bin'/name
 rows.append(dict(path=str(p),exists=p.is_file(),executable=os.access(p,os.X_OK),required=True))
for name in ['results/experiment_8192_v708_contract_custody.json','results/experiment_8204_v708_capstone.json']:
 v=json.loads((root/name).read_bytes())
 print(json.dumps(dict(path=name,schema=v.get('schema'),verdict=v['honest_verdict'],code_shape=type(v['code_config_hashes']).__name__,terminal=v.get('terminal_validation_sidecar_path'),dispositions=len(v.get('task_dispositions',[])),historical_failures=len(v.get('historical_hash_failures',[])),failed_receipts=[r.get('name') for r in v.get('validation_receipts',[]) if not r.get('passed')])))
for name in ['research-roadmap.yaml','research-roadmap-next.yaml','research-complete.yaml']:
 p=root/name
 if p.is_file():
  v=yaml.safe_load(p.read_bytes()); print(json.dumps(dict(path=name,type=type(v).__name__,keys=list(v)[:5] if isinstance(v,dict) else [],milestone=v.get('milestone') if isinstance(v,dict) else None)))
with tempfile.TemporaryDirectory(prefix='carnot8205-preflight-') as temp:
 p=Path(temp); (p/'probe').write_bytes(b'private actual write')
 scratch=dict(path=str(p),mode=oct(p.stat().st_mode & 0o777),passed=(p/'probe').read_bytes()==b'private actual write')
print(json.dumps(dict(python=sys.version,executable=sys.executable,paths=rows,private_scratch=scratch),indent=2))
raise SystemExit(any(r['required'] and not r['exists'] for r in rows))
