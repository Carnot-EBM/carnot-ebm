#!/usr/bin/env python3
"""Experiment 590 — CARNOT_FORCE_LIVE Assertion Module.

**Context (RETRO-062):**
    Three consecutive milestone failures (.42, .43, .44) blocked the Live 50q A
    benchmark because CARNOT_FORCE_LIVE was not set at session start.
    EnvironmentAutoFix (env_autofix.py) injects the variable when a GPU is detected,
    but if a live GPU script forgot to call apply_env_autofix() first, it could
    silently fall back to synthetic mode.

    This experiment verifies that live_assertion.py exists and that both
    assert_live_gpu_available() and assert_live_or_ci_skip() behave correctly,
    making silent fallback structurally impossible.

Spec: REQ-INFRA-082, SCENARIO-INFRA-089, SCENARIO-INFRA-090
"""

from __future__ import annotations

# apply_env_autofix MUST be called before any JAX or CUDA import.
from carnot.pipeline.env_autofix import apply_env_autofix

_env_result = apply_env_autofix()

import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_RESULT_PATH = "results/experiment_590_live_assertion.json"


def verify_module_importable() -> bool:
    """Return True iff live_assertion.py is importable from carnot.pipeline."""
    try:
        import carnot.pipeline.live_assertion  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def verify_functions_importable() -> tuple[bool, bool]:
    """Return (assert_live_imported, assert_ci_imported) booleans."""
    assert_live_ok = False
    assert_ci_ok = False
    try:
        from carnot.pipeline.live_assertion import assert_live_gpu_available  # noqa: PLC0415
        assert_live_ok = callable(assert_live_gpu_available)
    except (ImportError, AttributeError):
        pass
    try:
        from carnot.pipeline.live_assertion import assert_live_or_ci_skip  # noqa: PLC0415
        assert_ci_ok = callable(assert_live_or_ci_skip)
    except (ImportError, AttributeError):
        pass
    return assert_live_ok, assert_ci_ok


def test_no_raise_when_force_live_set() -> bool:
    """Return True iff assert_live_gpu_available() does NOT raise when CARNOT_FORCE_LIVE='1'.

    Mocks torch.cuda.is_available=True so the GPU branch is exercised.
    """
    from carnot.pipeline.live_assertion import assert_live_gpu_available  # noqa: PLC0415

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    old_val = os.environ.get("CARNOT_FORCE_LIVE")
    try:
        os.environ["CARNOT_FORCE_LIVE"] = "1"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            assert_live_gpu_available()
        return True
    except RuntimeError:
        return False
    finally:
        if old_val is None:
            os.environ.pop("CARNOT_FORCE_LIVE", None)
        else:
            os.environ["CARNOT_FORCE_LIVE"] = old_val


def test_raises_when_force_live_not_set() -> bool:
    """Return True iff assert_live_gpu_available() raises RuntimeError when CARNOT_FORCE_LIVE='0'.

    Mocks torch.cuda.is_available=True so the GPU branch is exercised.
    """
    from carnot.pipeline.live_assertion import assert_live_gpu_available  # noqa: PLC0415

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    old_val = os.environ.get("CARNOT_FORCE_LIVE")
    try:
        os.environ["CARNOT_FORCE_LIVE"] = "0"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            assert_live_gpu_available()
        # Should have raised — if we get here, the test failed.
        return False
    except RuntimeError:
        return True
    finally:
        if old_val is None:
            os.environ.pop("CARNOT_FORCE_LIVE", None)
        else:
            os.environ["CARNOT_FORCE_LIVE"] = old_val


def verify_pipeline_exports() -> tuple[bool, bool]:
    """Return (assert_live_exported, assert_ci_exported) from carnot.pipeline."""
    import carnot.pipeline as pipeline  # noqa: PLC0415

    assert_live_ok = hasattr(pipeline, "assert_live_gpu_available") and callable(
        pipeline.assert_live_gpu_available
    )
    assert_ci_ok = hasattr(pipeline, "assert_live_or_ci_skip") and callable(
        pipeline.assert_live_or_ci_skip
    )
    return assert_live_ok, assert_ci_ok


def run_experiment() -> dict:
    """Run all assertion module checks and return the artifact payload."""
    module_created = verify_module_importable()
    assert_live_ok, assert_ci_ok = verify_functions_importable()
    no_raise_ok = test_no_raise_when_force_live_set()
    raises_ok = test_raises_when_force_live_not_set()
    export_live_ok, export_ci_ok = verify_pipeline_exports()

    all_checks_passed = all([
        module_created,
        assert_live_ok,
        assert_ci_ok,
        no_raise_ok,
        raises_ok,
        export_live_ok,
        export_ci_ok,
    ])

    return {
        "schema": "carnot.live_assertion.v1",
        "module_created": module_created,
        "assert_live_gpu_available_exported": export_live_ok,
        "assert_live_or_ci_skip_exported": export_ci_ok,
        "assert_live_importable": assert_live_ok,
        "assert_ci_importable": assert_ci_ok,
        "no_raise_when_force_live_1": no_raise_ok,
        "raises_when_force_live_0": raises_ok,
        "all_checks_passed": all_checks_passed,
        "retro_062_prevention_mechanism": "import_time_assertion_raises_before_model_load",
        "honest_verdict": "assertion_module_ready" if all_checks_passed else "assertion_module_checks_failed",
    }


def main() -> None:
    """Entry point: run experiment under watchdog, write result JSON."""
    import json  # noqa: PLC0415

    tmpl = ExperimentTemplate(
        590,
        "CARNOT_FORCE_LIVE Assertion Module",
        _RESULT_PATH,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(590, timeout_minutes=10, result_path=str(_REPO_ROOT / _RESULT_PATH)):
        payload = run_experiment()

    artifact = tmpl.build_result(payload, status="success")

    output_path = _REPO_ROOT / _RESULT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(artifact, fh, indent=2)

    print(f"\nResult: {output_path}")
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
