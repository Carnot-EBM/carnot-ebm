"""Finish qualification after correcting a test double; preserve measured science."""

from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

ROOT = Path('/home/ianblenke/github.com/ianblenke/carnot')
sys.path[:0] = [str(ROOT/'python'),str(ROOT)]
from carnot.reporting import selective_methods_runner_8829 as r
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting.primary_publication import publish_primary

raw = ROOT/'results/raw'/r.m.NAME/'invocations/1793860196624396502'
old_work_ref = reference(raw/'work.json')
old_validation_ref = reference(raw/'validation.json')
work = load(old_work_ref)
failed_validation = load(old_validation_ref)
assert not r.qualified(failed_validation)
for ref in work['code_config_hashes']:
    if Path(ref['path']).relative_to(ROOT).as_posix() not in r.m.TESTS:
        assert sha256_file(Path(ref['path'])) == ref['sha256'], ref['path']
failed_candidate = raw/'failed_qualification_candidate.json'
shutil.copyfile(ROOT/'results/raw'/r.m.NAME/'terminal_candidate.json', failed_candidate)
retained_helper = raw/'qualification_continuation.py'
shutil.copyfile(__file__,retained_helper)
assert sha256_file(retained_helper) == sha256_file(Path(__file__))
r.m.progress('before_corrected_owned_qualification')
with TemporaryDirectory(prefix='exp8829-requalification-') as directory:
    validation = r.validate(raw/'requalification',Path(directory))
atomic_json(raw/'requalification/validation.json',validation)
r.m.progress('after_corrected_owned_qualification')
print({'qualified':r.qualified(validation),'owned_statement_coverage':validation['coverage']},flush=True)
assert r.qualified(validation), 'corrected_owned_qualification_failed'
work['qualification_repair_lineage'] = dict(
    failed_work=old_work_ref, failed_validation=old_validation_ref,
    failed_candidate=reference(failed_candidate),
    reason='Test double omitted the directory creation guaranteed by real preconditions.',
    measurement_code_changed=False, scientific_inputs_or_decisions_changed=False)
work['code_config_hashes'] = [reference(Path(ref['path'])) for ref in work['code_config_hashes']]
work['code_config_hashes'].append(reference(retained_helper))
work['raw_shard_hashes'] += [old_work_ref,old_validation_ref,reference(failed_candidate)]
work['raw_shard_hashes'] += failed_validation['raw_references']
atomic_json(raw/'qualified_work.json',work)
value = r.sealed_value(work,validation,'20261105',reference(raw/'qualified_work.json'),
                       reference(raw/'requalification/validation.json'))
output = ROOT/'results'/(r.m.NAME+'.json')
publication = publish_primary(output,value,lambda candidate:r.checker.terminal(candidate,raw/'requalification'))
atomic_json(raw/'requalification/publication_receipt.json',publication)
print({'artifact':str(output),'verdict':value['honest_verdict'],
       'independent_worlds':value['independent_count'],'rows':len(value['rows']),
       'new_statement_coverage_percent':value['owned_statement_coverage']['percent'],
       'readiness':[value[f] for f in ['evidence_protocol_ready_score',
                                     'selective_protocol_ready_score','canary_protocol_ready_score']]},flush=True)
