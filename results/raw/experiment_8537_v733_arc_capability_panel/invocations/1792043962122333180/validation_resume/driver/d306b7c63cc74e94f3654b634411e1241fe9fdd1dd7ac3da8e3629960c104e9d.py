"""Finish the honest failed attempt without collecting any episode again."""
import json
from pathlib import Path
import sys
import time

from carnot.reporting import arc_capability_panel_8537 as e
from carnot.reporting import arc_capability_panel_runner_8537 as runner
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import child

started = time.monotonic()
raw = sorted((e.ROOT / 'results/raw' / e.NAME / 'invocations').glob('*'))[-1]
final = raw / 'validation_resume'
final.mkdir(exist_ok=True)
config = Path('/var/tmp/exp8537-replay-budget/coverage.ini')
receipts = []
commands = [
    ('budget_regression', [sys.executable, '-m', 'coverage', 'run', '--rcfile=' + str(config), '-m', 'pytest', '-n', '0', '-o', 'addopts=', '--no-cov', '-q', e.TESTS[1] + '::test_full_primitive_replay_uses_validation_reserve'], 90),
    ('budget_ruff_check', [str(e.ROOT / '.venv/bin/ruff'), 'check', *e.OWNED, *e.TESTS], 60),
    ('budget_ruff_format', [str(e.ROOT / '.venv/bin/ruff'), 'format', '--check', *e.OWNED, *e.TESTS], 60),
    ('budget_strict_mypy', [str(e.ROOT / '.venv/bin/mypy'), '--strict', '--follow-imports=skip', *e.OWNED], 120),
    ('budget_spec_coverage', [sys.executable, 'scripts/check_spec_coverage.py', '--files', *e.TESTS], 60),
    ('budget_coverage_combine', [sys.executable, '-m', 'coverage', 'combine', '--rcfile=' + str(config), '--keep', str(raw / 'validation/coverage_shards'), str(config.parent)], 60),
    ('budget_coverage_report', [sys.executable, '-m', 'coverage', 'report', '--rcfile=' + str(config), '--fail-under=100'], 60),
    ('budget_coverage_json', [sys.executable, '-m', 'coverage', 'json', '--rcfile=' + str(config), '-o', str(final / 'owned_coverage.json')], 60),
]
for name, argv, deadline in commands:
    receipts.append(child(name, argv, final / 'checks', deadline=deadline))
assert all(r['passed'] for r in receipts), [(r['name'], r['passed']) for r in receipts]
original = json.loads((raw / 'before_replay_budget_fix').glob('*.json').__iter__().__next__().read_bytes())
candidate = json.loads((raw.parent.parent / 'terminal_candidate.json').read_bytes())
work = json.loads((raw / 'work.json').read_bytes())
failed = json.loads((raw / 'initial_failed_terminal_report.json').read_bytes())
old_runner = next((raw / 'measurement_sources').glob('*.py'))
runner_path = e.ROOT / e.OWNED[1]
assert sha256_file(old_runner) == work['code_config_hashes'][str(runner_path)]
work['code_config_hashes'][str(old_runner)] = sha256_file(old_runner)
work['code_config_hashes'][str(runner_path)] = sha256_file(runner_path)
work['validation_budget_correction'] = dict(original_terminal_report=str(raw / 'initial_failed_terminal_report.json'), original_measurement_runner=str(old_runner), corrected_runner=str(runner_path), previous_deadline_s=90, full_replay_deadline_s=240, episodes_rerun=0, original_failure_retained=True)
work['phase_spans'].append(dict(phase='failed_initial_terminal_validation', duration_s=sum(r['duration_s'] for r in failed['checks'])))
work['phase_spans'].append(dict(phase='replay_budget_regression_and_checks', duration_s=time.monotonic() - started))
work['duration_s'] += sum(r['duration_s'] for r in failed['checks']) + time.monotonic() - started
path = final / 'work.json'
atomic_json(path, work)
failed_receipt = dict(failed['checks'][0], name='failed_initial_terminal_replay')
all_receipts = [*candidate['validation_receipts'], failed_receipt, *receipts]
output = e.ROOT / 'results' / (e.NAME + '.json')
e.progress('final_disqualified_build_no_episode_reruns')
value = e.build(path, all_receipts, output)
assert value['verdict_class'] == 'disqualified'
assert value['arc_capability_panel_ready_score'] == value['capability_efficiency_score'] == 0
atomic_json(final / 'candidate.json', value)
e.progress('final_disqualified_publication_before')
side = runner.publish(value, output, final)
e.progress('final_disqualified_publication_after', value['completed_count'], 0)
print(json.dumps(dict(passed=side['passed'], experiment_id=value['experiment_id'], verdict=value['verdict_class'], completed=value['completed_count'], primary_sha256=side['publication']['primary_sha256']), sort_keys=True), flush=True)
raise SystemExit(int(not side['passed']))
