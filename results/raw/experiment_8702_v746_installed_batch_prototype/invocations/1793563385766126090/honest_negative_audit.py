"""Publish an authenticated failed qualification without granting readiness."""
from pathlib import Path
import json
import sys
from carnot.reporting import installed_batch_8702 as e
from carnot.reporting import installed_batch_runner_8702 as r
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child


def audit(path, pin):
    try:
        value = json.loads(path.read_bytes())
        assert value['primitive_reference'] == pin
        work = e.load(pin)
        assert e.load(work['seal_reference']) == {k:v for k,v in work.items() if k != 'seal_reference'}
        refs = work['sources'] + work['inputs']['references'] + work['measurement']['traces']
        refs += [e.load(ref)['stderr_reference'] for ref in work['measurement']['traces']]
        refs += [work['measurement'][key] for key in ['installed_wheel_reference','loaded_extension_reference','config_reference','installed_identity_reference']]
        assert all(Path(ref['path']).is_file() and e.reference(Path(ref['path']))['sha256'] == ref['sha256'] for ref in refs)
        children = work['validation']['children']
        nodes = [node for ref in children for node in e.load(ref)['node_ids']]
        custody = r.retained.aggregate(children, nodes)
        assert not custody['passed'] and 'failed_child' in custody['errors']
        assert value['verdict_class'] == 'disqualified'
        assert value['required_checks_passed'] is False and value['flagged_adversarial'] is True
        assert value['batch_service_ready_score'] == 0
        assert not r.replay(path, pin)
        assert e.build(work, Path(work['output'])) == value
        print('honest_disqualification_reduction_verified', flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        print('replay_rejected', flush=True)
        return 1


if sys.argv[1] == 'audit':
    raise SystemExit(audit(Path(sys.argv[2]), dict(path=sys.argv[3], sha256=sys.argv[4])))
raw = Path(sys.argv[1]).absolute()
work_path = raw / 'work.json'
pin = e.reference(work_path)
work = e.load(pin)
output = e.ROOT / 'results' / (e.NAME + '.json')
value = e.build(work, output)
logs = raw / 'honest_negative_publication'
logs.mkdir()

def validate(candidate):
    changed = dict(value, completed_count=value['completed_count'] + 1)
    changed['reproducibility_checksum'] = e.canonical_hash(changed)
    bad = logs / 'rehashed_tamper.json'
    e.atomic_json(bad, changed)
    receipts = []
    for name, path, expected in [('cold_disqualified', candidate, 0), ('missing_operand', logs/'missing.json', 1), ('rehashed_tamper', bad, 1)]:
        receipts.append(child(name, [sys.executable, '-u', __file__, 'audit', str(path), pin['path'], pin['sha256']], logs, expected=expected, deadline=120))
    for name, argv in [('adversarial', ['scripts/adversarial_verify.py','--json',str(candidate)]), ('strict_rows', ['scripts/verdict_row_consistency_lint.py','--strict',str(candidate)])]:
        receipts.append(child(name, [sys.executable,'-u',*argv], logs, deadline=120))
    findings = json.loads(Path(receipts[-2]['stdout_path']).read_text())
    original_cold = json.loads((raw/'terminal/cold.receipt.json').read_text())
    return dict(passed=all(x['passed'] for x in receipts) and findings['flagged_count']==0, receipts=receipts,
                actual_findings=findings, publication_scope='authenticated disqualification only; qualification remains failed',
                qualification_replay_receipt=original_cold, qualification_passed=False,
                audit_source_reference=e.reference(Path(__file__)))
publication = publish_primary(output, value, validate)
e.atomic_json(Path(value['terminal_validation_sidecar_path']), dict(publication=publication, qualification_passed=False))
print(json.dumps(dict(publication=publication, verdict=value['honest_verdict'], ready=value['batch_service_ready_score'])), flush=True)
