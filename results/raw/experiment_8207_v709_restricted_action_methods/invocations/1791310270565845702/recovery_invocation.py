"""Recover owned validation after a real log-custody failure; keep prior health clocks."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import shutil
import time
from unittest.mock import patch

from carnot.verify import restricted_action_methods_8207 as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

output = e.ROOT / 'results' / (e.NAME + '.json')
base = output.parent / 'raw' / e.NAME
prior = base / 'invocations' / '1791309784811194196'
prior_receipt = prior / 'global_health/repository_full_suite.receipt.json'
health = json.loads(prior_receipt.read_bytes())
assert health['argv'] == [str(e.ROOT / '.venv/bin/pytest'), 'tests/python', '-q']
assert health['timed_out'] and not health['passed']
raw = base / 'invocations' / str(time.time_ns())
raw.mkdir(parents=True)
shutil.copyfile(__file__, raw / 'recovery_invocation.py')
with TemporaryDirectory(prefix='carnot8207-recovery-') as directory:
    private = Path(directory)
    candidate = base / 'terminal_candidate.json'
    plan = e.manifest(private, candidate)
    plan['repository_health'].update(disposition='preserve earlier same-task unrelated timeout; do not rerun full suite', prior_receipt_path=str(prior_receipt), prior_receipt_sha256=sha256_file(prior_receipt))
    plan['measurement'] = dict(name='measurement', argv=[str(e.ROOT / '.venv/bin/python'), '-u', str(e.ROOT / e.CLI), '--date', e.RUN_DATE, '--worker-output', str(raw / 'measurement.json')], deadline_s=180, expected_exit=0, classification='required')
    atomic_json(raw / 'validation_commands.json', plan)
    e.progress('recovery_measurement_before', 0, 1)
    receipts = [e.run_check(e.ROOT, plan['measurement'], private, raw / 'logs')]
    assert receipts[0]['passed']
    work = json.loads((raw / 'measurement.json').read_bytes())
    for index, spec in enumerate(plan['commands']):
        e.progress('recovery_owned_validation', index, len(plan['commands'])-index)
        receipt = e.run_check(e.ROOT, spec, private, raw / 'logs')
        receipts.append(receipt)
        print(spec['name'], receipt['actual_exit'], flush=True)
    assert all(r['passed'] for r in receipts), [(r['name'],r['actual_exit']) for r in receipts]
    # These are the original health command, streams and clocks from the first invocation.
    work['global_health'] = dict(health, evidence_scope='earlier same task before log-custody fix; unrelated diagnostic only', reused_receipt_reference=e.old.fit.reference(prior_receipt))
    work['refs'] += [e.old.fit.reference(p) for p in [prior_receipt, Path(health['stdout_path']), Path(health['stderr_path'])]]
    for source in [private/'coverage.ini', private/'coverage.json', private/'.coverage']:
        assert source.is_file()
        saved = raw / source.name
        shutil.copyfile(source, saved)
        work['raw_shard_hashes'].append(e.old.fit.reference(saved))
    work['code_config_hashes'].append(e.old.fit.reference(raw / 'recovery_invocation.py'))
    atomic_json(raw / 'measurement.json', work)
    atomic_json(raw / 'validation_receipts.json', dict(rows=receipts))
    value = e.build(work, raw, receipts)
    with patch.object(e.execution, 'e', e), patch.object(e.execution, 'run_check', e.run_check):
        e.execution.publish(value, output, private, raw, plan['terminal_commands'], False)
    print('PUBLISHED', output, value['honest_verdict'], flush=True)
