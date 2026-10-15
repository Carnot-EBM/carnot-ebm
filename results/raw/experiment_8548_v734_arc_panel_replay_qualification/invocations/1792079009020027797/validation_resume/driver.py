"""Resume measured Exp8548 validation without repeating original test nodes."""
from pathlib import Path
import ast
import json
import shutil
import signal
import sys
import time

sys.path[:0] = ['/home/ianblenke/github.com/ianblenke/carnot/python', '/home/ianblenke/github.com/ianblenke/carnot']
from carnot.reporting import arc_panel_replay_8548 as e
from carnot.reporting import arc_panel_replay_runner_8548 as r
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

signal.setitimer(signal.ITIMER_REAL, 900)
e.progress('validation_resume_no_new_measurements')
raw = e.ROOT / 'results/raw' / e.NAME / 'invocations/1792079009020027797'
old_path = raw / 'work.json'
w = json.loads(old_path.read_bytes())
resume = raw / 'validation_resume'
resume.mkdir(exist_ok=True)
shutil.copyfile(old_path, resume / 'original_checkpoint.json')
old_code = dict(w['code_hashes'])
old_runner = raw / 'prior_execution_sources' / (old_code[str(e.ROOT / e.OWNED[1])][7:] + '.py')
original_tree = ast.parse(old_runner.read_text())
current_tree = ast.parse((e.ROOT / e.OWNED[1]).read_text())
for name in ['historical_suite', 'validation', 'measurement', 'main']:
    before = next(n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    after = next(n for n in current_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False), name
for path in (e.ROOT / e.OWNED[0], e.ROOT / e.CLI):
    assert sha256_file(path) == old_code[str(path)]
assert e.exact_union(w['union']['collected_node_ids'], w['family_receipts']) == w['union']
for receipt in w['checks']:
    for channel in ('stdout', 'stderr'):
        if channel + '_path' in receipt:
            assert sha256_file(Path(receipt[channel + '_path'])) == receipt[channel + '_sha256']
private = Path('/var/tmp/exp8548-final-validation')
private.mkdir(exist_ok=True)
data = private / 'coverage_data'
data.mkdir(exist_ok=True)
config = private / 'coverage.ini'
config.write_text('[run]\ncore=sysmon\nparallel=true\npatch=subprocess,_exit\ndata_file=' + str(data / '.coverage') + '\ninclude=\n' + ''.join('    ' + str(e.ROOT / p) + '\n' for p in e.OWNED))
driver = resume / 'driver.py'
shutil.copyfile(__file__, driver)
w['preconditions']['validation_resume'] = dict(original_checkpoint_sha256=sha256_file(resume / 'original_checkpoint.json'), original_union_reused_without_rerun=True, unchanged_functions=['historical_suite', 'validation', 'measurement', 'main'], prior_execution_source_hashes=old_code, driver_path=str(driver), driver_sha256=sha256_file(driver), resources=r.resources(private))
names = {'owned_tests', 'ruff_check', 'ruff_format', 'strict_mypy', 'spec_coverage'}
w['preconditions']['prior_owned_validation_receipts'] = [row for row in w['checks'] if row['name'] in names]
w['checks'] = [row for row in w['checks'] if row['name'] not in names]
w['code_hashes'] = {str(e.ROOT / p): sha256_file(e.ROOT / p) for p in e.OWNED}
for source in (raw / 'prior_execution_sources').glob('*.py'):
    w['code_hashes'][str(source)] = sha256_file(source)
w['code_hashes'][str(driver)] = sha256_file(driver)
commands = [
    ('owned_tests', [sys.executable, '-m', 'coverage', 'run', '--rcfile=' + str(config), '-m', 'pytest', *r.COMMON, '-q', e.TEST, '--basetemp=' + str(private / 'owned')]),
    ('ruff_check', [str(e.ROOT / '.venv/bin/ruff'), 'check', *e.OWNED, e.TEST]),
    ('ruff_format', [str(e.ROOT / '.venv/bin/ruff'), 'format', '--check', *e.OWNED, e.TEST]),
    ('strict_mypy', [str(e.ROOT / '.venv/bin/mypy'), '--strict', '--follow-imports=silent', *e.OWNED]),
    ('spec_coverage', [sys.executable, 'scripts/check_spec_coverage.py', '--files', e.TEST]),
]
for name, argv in commands:
    w['checks'].append(r.invoke(name, argv, resume / 'checks', deadline=360))
preserved = resume / 'coverage_before_reporting'
shutil.copytree(data, preserved)
for name, args in [('coverage_combine', ['combine', '--keep']), ('coverage_report', ['report', '--fail-under=100'])]:
    w['checks'].append(r.invoke(name, [sys.executable, '-m', 'coverage', args[0], '--rcfile=' + str(config), *args[1:]], resume / 'checks', deadline=60))
for shard in preserved.glob('.coverage.*'):
    shutil.copyfile(shard, data / shard.name)
stored = resume / 'coverage_data'
shutil.copytree(data, stored)
stored_config = resume / 'coverage.ini'
stored_config.write_text(config.read_text().replace(str(data), str(stored)))
w['coverage_config'] = str(stored_config)
w['coverage'] = e.coverage_receipt(stored_config, w['code_hashes'])
w['checks'].append(dict(name='exact_owned_coverage', passed=w['coverage']['passed']))
w['raw_hashes'] = {str(p): sha256_file(p) for p in stored.glob('.coverage*') if p.is_file()}
w['raw_hashes'].update({str(raw / name): sha256_file(raw / name) for name in ['replay_rows.json', 'fixtures.json']})
path = resume / 'work.json'
output = e.ROOT / 'results' / (e.NAME + '.json')
atomic_json(path, w)
changed = json.loads(path.read_bytes())
changed['replay_rows'][0]['observation']['actual_disabled_attempts'] += 1
atomic_json(resume / 'tampered_work.json', changed)
atomic_json(resume / 'tampered.json', e.build(resume / 'tampered_work.json', output))
for name, args, expected in [('missing', ['--cold-replay', str(resume / 'absent')], 1), ('intentional', ['--deliberate-error'], 7), ('rehashed_tamper', ['--cold-replay', str(resume / 'tampered.json')], 1)]:
    w['checks'].append(r.invoke(name, [sys.executable, '-u', str(e.ROOT / e.CLI), *args], resume / 'controls', deadline=240, expected=expected))
w['duration_s'] = (time.time_ns() - int(raw.name)) / 1e9
w['phase_spans'] = [w['phase_spans'][0], dict(phase='validation_including_source_bound_resume', duration_s=w['duration_s'] - w['phase_spans'][0]['duration_s'])]
atomic_json(path, w)
value = e.build(path, output)
e.progress('publication_before', value['completed_count'], 0)
side = r.publish(value, output, resume)
e.progress('publication_after', value['completed_count'], 0)
print(json.dumps(dict(passed=side['passed'], readiness=value['arc_panel_execution_ready_score'], verdict=value['verdict_class'], coverage=w['coverage']['passed'])), flush=True)
raise SystemExit(int(not side['passed']))
