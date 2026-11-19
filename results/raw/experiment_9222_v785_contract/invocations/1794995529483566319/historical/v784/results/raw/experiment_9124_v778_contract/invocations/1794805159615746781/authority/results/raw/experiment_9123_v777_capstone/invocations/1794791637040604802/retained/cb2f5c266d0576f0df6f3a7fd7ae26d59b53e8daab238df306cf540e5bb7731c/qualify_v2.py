"""Freeze exact owned validation, then execute under bounded heartbeat supervision."""
import json
from pathlib import Path
import time
from carnot.reporting import human_authority_9097 as d
from carnot.reporting import human_authority_runner_9097 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child
raw = d.DEFAULT_INPUT / ('qualified-' + str(time.time_ns()))
raw.mkdir(mode=0o700)
(raw / 'coverage').mkdir()
config = raw / 'coverage.ini'
config.write_text('[run]\nparallel = true\ndata_file = ' + str(raw / 'coverage/.coverage') + '\ninclude =\n' + ''.join('    ' + str(d.ROOT / p) + '\n' for p in d.OWNED) + '[report]\nexclude_lines =\n')
python = str(d.ROOT / '.venv/bin/python')
pytest = str(d.ROOT / '.venv/bin/pytest')
base = [pytest, '-n', '0', '-o', 'addopts=', '--no-cov', '-q']
paths = [*d.OWNED, d.TEST]
commands = [
 dict(name='owned_tests', argv=['env', 'COVERAGE_RCFILE=' + str(config), python, '-m', 'coverage', 'run', '--rcfile=' + str(config), '-m', 'pytest', '-n', '0', '-o', 'addopts=', '--no-cov', '-q', d.TEST]),
 dict(name='direct_consumers', argv=[python, str(d.DEFAULT_INPUT / 'run_consumers.py')]),
 dict(name='e2e018', argv=base + ['tests/python/test_experiment_7891_v685_authority_lifecycle.py']),
 dict(name='e2e021', argv=base + ['tests/python/test_restricted_decision_audit_8210.py']),
 dict(name='ruff_check', argv=[str(d.ROOT / '.venv/bin/ruff'), 'check', *paths]),
 dict(name='ruff_format', argv=[str(d.ROOT / '.venv/bin/ruff'), 'format', '--check', *paths]),
 dict(name='strict_mypy', argv=[str(d.ROOT / '.venv/bin/mypy'), '--strict', '--follow-imports=silent', *d.OWNED]),
 dict(name='spec_coverage', argv=[python, 'scripts/check_spec_coverage.py', d.TEST]),
 dict(name='owned_coverage', argv=[python, str(d.DEFAULT_INPUT / 'finish_coverage.py'), str(raw)]),
]
scope = r.scope_manifest(commands)
scope['code_test_hashes'] += [d.ref(p) for p in [config, d.DEFAULT_INPUT / 'finish_coverage.py', d.DEFAULT_INPUT / 'qualify_v2.py', d.DEFAULT_INPUT / 'run_consumers.py', d.ROOT / 'python/carnot/reporting/historical_consumer_roots_8402.py']]
d.seal(raw / 'scope.json', scope)
atomic_json(d.DEFAULT_INPUT / 'validation_scope_manifest.json', scope)
checks = []
for spec in commands:
    receipt = child(spec['name'], spec['argv'], raw / 'logs', deadline=600, heartbeat=20)
    receipt['scope_sha256'] = canonical_hash(scope)
    checks.append(receipt)
    atomic_json(raw / 'progress.json', dict(checks=checks))
    if not receipt['passed']:
        print('FAILED ' + spec['name'] + ' logs=' + str(raw / 'logs'), flush=True)
        print(Path(receipt['stdout_path']).read_text()[-12000:], flush=True)
        print(Path(receipt['stderr_path']).read_text()[-4000:], flush=True)
validation = dict(checks=checks, owned_statement_coverage=json.loads((raw / 'owned_coverage.json').read_bytes()) if (raw / 'owned_coverage.json').is_file() else {})
atomic_json(raw / 'validation.json', validation)
atomic_json(d.DEFAULT_INPUT / 'validation.json', validation)
print(json.dumps(dict(raw=str(raw), passed=r.receipts_pass(scope, validation))), flush=True)
