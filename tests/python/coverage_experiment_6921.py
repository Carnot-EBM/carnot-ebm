"""Run the focused Exp6921 tests under coverage without importing JAX."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import tempfile


REAL_IMPORT = builtins.__import__


def _no_jax_import(name, *args, **kwargs):
    if name == "jax" or name.startswith("jax."):
        error = ModuleNotFoundError("Exp6921 does not use optional JAX")
        error.name = "jax"
        raise error
    return REAL_IMPORT(name, *args, **kwargs)


try:
    builtins.__import__ = _no_jax_import
    spec = importlib.util.spec_from_file_location(
        "test_exp6921", "tests/python/test_experiment_6921_arc_dynamic_supervisor_banked_credit.py"
    )
    tests = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Exp6921 test loader unavailable")
    spec.loader.exec_module(tests)

    tests.test_req_arc_wmte_6921_spec_owns_required_contract()
    tests.test_req_arc_wmte_6921_registers_exact_no_llm_audit_substrate()
    temporary_tests = (
        tests.test_scenario_6921_hard_coded_source_regression,
        tests.test_scenario_6921_nested_clone_and_copied_row_dedupe,
        tests.test_scenario_6921_shadow_and_error_admission,
        tests.test_scenario_6921_transient_progress_has_no_banked_credit,
        tests.test_scenario_6921_banked_level_credit,
        tests.test_scenario_6921_actions_mismatch_and_post_redirect_order,
        tests.test_scenario_6921_competing_redirects_are_explicit,
        tests.test_scenario_6921_no_new_row_state,
        tests.test_scenario_6921_frozen_floor_controls_eligibility,
        tests.test_scenario_6921_automatic_arm_mutation_is_forbidden,
        tests.test_req_arc_wmte_6921_precondition_failure_is_complete_blocked,
        tests.test_scenario_6921_artifact_schema_checksum_and_atomic_writer,
        tests.test_req_arc_wmte_6921_helper_defenses_and_frame_fallback,
        tests.test_req_arc_wmte_6921_new_arm_receipt_stays_recommendation_only,
        tests.test_req_arc_wmte_6921_validator_rejects_each_contract_mutation,
        tests.test_req_arc_wmte_6921_main_writes_and_reports_validation_errors,
    )
    for test in temporary_tests:
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))
finally:
    builtins.__import__ = REAL_IMPORT

print("18 scoped Exp6921 test cases passed")
