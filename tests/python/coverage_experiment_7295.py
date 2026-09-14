"""Run the exact Exp7295 pytest file under coverage without optional JAX.

This CPU-only experiment does not use JAX. The host JAX build aborts under
Python tracing, so this process selects Carnot's supported no-JAX import path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile


class _NoJaxFinder:
    """Select Carnot's optional no-JAX import path without wrapping NumPy imports."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jax" or fullname.startswith("jax."):
            error = ModuleNotFoundError("Exp7295 coverage excludes unused optional JAX")
            error.name = "jax"
            raise error
        return None


finder = _NoJaxFinder()
try:
    sys.meta_path.insert(0, finder)
    import numpy  # noqa: F401

    spec = importlib.util.spec_from_file_location(
        "test_exp7295", "tests/python/test_experiment_7295_v641_mixture_prototype.py"
    )
    tests = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Exp7295 test loader unavailable")
    spec.loader.exec_module(tests)

    tests.test_req_cl_7295_freezes_contract_and_no_model_work()
    tests.test_scenario_cl_7295_update_uses_exact_loss_share_and_ties_abstain()
    tests.test_scenario_cl_7295_nominee_birth_follows_sixteenth_release()
    tests.test_scenario_cl_7295_nominee_evicts_lowest_then_oldest()
    tests.test_scenario_cl_7295_feedback_replay_is_causal_and_common()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tests.test_scenario_cl_7295_memory_rejection_and_restart_preserve_bytes(root / "memory")
        tests.test_scenario_cl_7295_streams_are_fresh_balanced_and_scorer_only(root / "streams")
        tests.test_scenario_cl_7295_controls_cover_chronology_bytes_and_restart(root / "controls")
        paths = tests.exp.ExperimentPaths.under(root / "artifact")
        artifact = tests.exp.build_and_seal(tests.exp.REPO_ROOT, paths, progress=True)
        built = (paths, artifact)
        tests.test_scenario_cl_7295_terminal_builds_and_cold_reduces(built)
        tests.test_req_cl_7295_validation_rejects_mutated_rows_and_writes_atomically(
            built, root / "writer"
        )
        monkeypatch = tests.pytest.MonkeyPatch()
        try:
            tests.test_scenario_cl_7295_preconditions_block_external_absence(
                root / "blocked", monkeypatch
            )
        finally:
            monkeypatch.undo()
        monkeypatch = tests.pytest.MonkeyPatch()
        try:
            tests.test_req_cl_7295_cli_helpers_and_thin_entrypoint(built, monkeypatch)
        finally:
            monkeypatch.undo()
        monkeypatch = tests.pytest.MonkeyPatch()
        try:
            tests.test_req_cl_7295_controller_and_raw_defenses(
                root / "controller-defenses", monkeypatch
            )
        finally:
            monkeypatch.undo()
        monkeypatch = tests.pytest.MonkeyPatch()
        try:
            tests.test_req_cl_7295_precondition_and_build_defenses(
                root / "build-defenses", monkeypatch
            )
        finally:
            monkeypatch.undo()
        monkeypatch = tests.pytest.MonkeyPatch()
        try:
            tests.test_req_cl_7295_validator_subprocess_and_main_defenses(
                built, root / "validator-defenses", monkeypatch
            )
        finally:
            monkeypatch.undo()
finally:
    sys.meta_path.remove(finder)

print("15 scoped Exp7295 test cases passed")
