"""Run preserved assertions with the producer's sealed aliases, without editing history."""
import json, os, sys
from pathlib import Path
from carnot.reporting.v721_capstone_evidence import frozen_inputs
refs=json.loads(Path('results/experiment_8374_v722_contract_methods.json').read_bytes())['source_artifact_hashes']
private=Path('/home/ianblenke/.cache/carnot-exp8388-private/historical-bootstrap');private.mkdir(parents=True,exist_ok=True)
(private/'sitecustomize.py').write_text('import json, os\nfrom pathlib import Path\nfrom carnot.reporting.v721_capstone_evidence import frozen_inputs\nrefs=json.loads(Path(os.environ["V723_HISTORY_REFS"]).read_bytes())\ncontext=frozen_inputs(refs,Path(os.environ["V723_HISTORY_SCRATCH"]))\ncontext.__enter__()\nfrom carnot.reporting import v709_execution\nv709_execution.ROOT=Path(os.environ["V723_HISTORY_SCRATCH"]).parent\n')
(private/'refs.json').write_text(json.dumps(refs))
os.environ['V723_HISTORY_REFS']=str(private/'refs.json');os.environ['V723_HISTORY_SCRATCH']=str(private/'writes');os.environ['PYTHONPATH']=str(private)+':'+str(Path('python').resolve())+':'+str(Path.cwd())
from unittest.mock import patch
from carnot.reporting import v709_execution as execution
for name in ['python','scripts','tests','.venv','pyproject.toml']:
 target=private/name
 if not target.exists():target.symlink_to(Path(name).resolve())
(private/'test_v722_preserved.py').write_bytes(Path('tests/python/test_v722_contract_methods_8374.py').read_bytes())
import pytest
with frozen_inputs(refs,private/'writes'), patch.object(execution, 'ROOT', private):
 raise SystemExit(pytest.main(['-c','/dev/null','--confcutdir='+str(private),'-n','0','-o','addopts=','--no-cov','-q',str(private/'test_v722_preserved.py'),'--basetemp='+str(private/'pytest')]))
