"""Run Exp6681 test functions under coverage without importing optional JAX.

The host JAX build aborts when imported under Python tracing. Exp6681 does not
use JAX, so this runner selects Carnot's supported no-JAX import path and calls
the same pytest test functions used by the focused suite.
"""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import tempfile


REAL_IMPORT = builtins.__import__


def _no_jax_import(name, *args, **kwargs):
    if name == "jax" or name.startswith("jax."):
        error = ModuleNotFoundError("Exp6681 coverage excludes unused optional JAX")
        error.name = "jax"
        raise error
    return REAL_IMPORT(name, *args, **kwargs)


builtins.__import__ = _no_jax_import
spec = importlib.util.spec_from_file_location(
    "test_exp6681", "tests/python/test_experiment_6681_arc_post_redirect_outcomes.py"
)
tests = importlib.util.module_from_spec(spec)
if spec.loader is None:
    raise RuntimeError("Exp6681 test loader unavailable")
spec.loader.exec_module(tests)

tests.test_scenario_6681_lineage_joins_after_reordering()
tests.test_scenario_6681_exact_return_keeps_absent_and_present_rewards()
tests.test_scenario_6681_missing_outcome_and_duplicate_ids_fail_closed()
tests.test_scenario_6681_attacks_cover_required_ambiguities()
tests.test_scenario_6681_step_timeout_and_error_are_terminal_rows()
tests.test_scenario_6681_artifact_blocks_missing_redirect_and_validator_mutations()
tests.test_scenario_6681_normalizers_and_transport_defenses_fail_closed()

with tempfile.TemporaryDirectory() as directory:
    tmp_path = Path(directory)
    tests.test_scenario_6681_artifact_recomputes_ready_rows_and_no_solve(tmp_path)
    monkeypatch = tests.pytest.MonkeyPatch()
    try:
        tests.test_scenario_6681_canonical_agent_step_joins_redirect_and_control(monkeypatch)
        for failure, status in (
            (TimeoutError("late"), "timeout"),
            (RuntimeError("broken"), "environment_error"),
        ):
            tests.test_scenario_6681_canonical_step_records_failure_before_reraise(
                monkeypatch, failure, status
            )
        tests.test_scenario_6681_helpers_handle_atomic_paths_and_live_runner_failure(
            tmp_path, monkeypatch
        )
        tests.test_scenario_6681_blocked_reductions_and_host_fallbacks(tmp_path, monkeypatch)
    finally:
        monkeypatch.undo()

print("13 scoped Exp6681 test cases passed")
