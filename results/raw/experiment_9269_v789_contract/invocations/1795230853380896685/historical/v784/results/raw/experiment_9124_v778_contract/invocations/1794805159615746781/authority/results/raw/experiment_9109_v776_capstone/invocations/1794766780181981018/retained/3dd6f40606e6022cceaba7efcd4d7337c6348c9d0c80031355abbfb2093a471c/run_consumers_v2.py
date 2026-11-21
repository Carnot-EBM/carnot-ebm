from pathlib import Path
import json
import sys
import pytest
root=Path('/home/ianblenke/github.com/ianblenke/carnot')
sys.path[:0]=[str(root/'python'),str(root)]
from carnot.reporting import historical_consumer_roots_8402 as history
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting import human_methods_runner_9084 as runner
primary=root/'results/experiment_8403_v724_released_question_identity.json'
assert sha256_file(primary)=='sha256:13afe888c9f7f516d63adff8eab9eea25e011745db6530d41cd836925661235b'
p=json.loads(primary.read_bytes())
assert sha256_file(Path(p['work_reference']['path']))==p['work_reference']['sha256']
w=json.loads(Path(p['work_reference']['path']).read_bytes())
refs=[]
for role,label in [('design',history.DESIGN),('active',history.ACTIVE),('staged',history.STAGED)]:
 s=w['authority']['authority_snapshots'][role]
 assert sha256_file(Path(s['snapshot_path']))==s['sha256']
 refs.append(dict(s,path=s['snapshot_path'],source_path=str(root/label)))
class OriginalAuthority:
 @pytest.fixture(autouse=True)
 def bind(self,request,tmp_path):
  if request.module.__name__ not in {'test_released_question_identity_8403','test_retrospective_label_criterion_8409'}:
   yield
   return
  with history.inject(dict(refs=refs),tmp_path/'original-authority'):
   yield
print('phase=unchanged_consumers_sealed_original_authority_before',flush=True)
raise SystemExit(pytest.main(['-n','0','-o','addopts=','--no-cov','-q',*runner.DIRECT],plugins=[OriginalAuthority()]))
