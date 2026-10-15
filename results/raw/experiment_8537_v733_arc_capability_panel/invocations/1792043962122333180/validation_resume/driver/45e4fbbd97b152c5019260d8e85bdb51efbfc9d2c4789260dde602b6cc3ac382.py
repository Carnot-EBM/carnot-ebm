"""Publish the disqualified attempt while retaining all original failures."""
import json
from pathlib import Path
import sys
import time

from carnot.reporting import arc_capability_panel_8537 as e
from carnot.reporting import arc_capability_panel_runner_8537 as runner
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v721_capstone_evidence import freeze

started = time.monotonic()
raw = sorted((e.ROOT / 'results/raw' / e.NAME / 'invocations').glob('*'))[-1]
final = raw / 'validation_resume'
receipts = [json.loads(p.read_bytes()) for p in sorted((final / 'checks').glob('*.receipt.json'))]
required_final = {'complete_coverage_combine', 'complete_coverage_report', 'complete_coverage_json', 'budget_regression', 'budget_ruff_check', 'budget_ruff_format', 'budget_strict_mypy', 'budget_spec_coverage'}
assert required_final <= {r['name'] for r in receipts if r['passed']}
candidate = json.loads((raw.parent.parent / 'terminal_candidate.json').read_bytes())
work = json.loads((raw / 'work.json').read_bytes())
failed = json.loads((raw / 'initial_failed_terminal_report.json').read_bytes())
old_runner = next((raw / 'measurement_sources').glob('*.py'))
runner_path = e.ROOT / e.OWNED[1]
assert sha256_file(old_runner) == work['code_config_hashes'][str(runner_path)]
work['code_config_hashes'][str(old_runner)] = sha256_file(old_runner)
work['code_config_hashes'][str(runner_path)] = sha256_file(runner_path)
driver = freeze(Path(__file__), final / 'driver')
work['code_config_hashes'][driver['snapshot_path']] = driver['sha256']
work['validation_budget_correction'] = dict(original_terminal_report=str(raw / 'initial_failed_terminal_report.json'), original_measurement_runner=str(old_runner), corrected_runner=str(runner_path), previous_deadline_s=90, full_replay_deadline_s=240, episodes_rerun=0, original_failure_retained=True, publication_resume_invocation=list(sys.argv), driver_reference=driver, coverage_assembly_failures_retained=True)
failed_time = sum(r['duration_s'] for r in failed['checks'])
check_time = sum(r['duration_s'] for r in receipts)
work['phase_spans'].append(dict(phase='failed_initial_terminal_validation', duration_s=failed_time))
work['phase_spans'].append(dict(phase='replay_budget_correction_checks', duration_s=check_time))
work['duration_s'] += failed_time + check_time + time.monotonic() - started
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
