"""Tests for Exp 3438 FoVer G2 clean-room root-cause-and-fix.

Spec: REQ-FOVER-G2-ROOTCAUSE
      SCENARIO-FOVER-G2-ROOTCAUSE-FIXED — fresh-env recompute lands in CI, the
        artifact reports the root cause + fix + a complete: verdict.
      SCENARIO-FOVER-G2-ROOTCAUSE-FALLBACK — when the recompute does NOT land in
        CI, the artifact honestly reports cleanroom_still_failing.

These tests exercise the pure logic of the experiment module (classification,
artifact assembly, precondition gating, pyproject fix detection) using synthetic
harness results and a fake clock, so they do not spawn the heavy git-worktree /
venv install path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "experiment_3438_fover_g2_cleanroom_rootcause_and_fix_v2.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("exp3438_rc", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp = _load_module()


# A harness result that lands inside the published CI (the post-fix happy path).
_GOOD_HARNESS = {
    "condition_a_production_auroc_mean": 0.9131,
    "learning_contribution_ci95": {"mean": 0.0185, "low": 0.0125, "high": 0.0245},
    "reproducibility_checksum": "deadbeef",
    "live_model_invoked": False,
}
# A harness result that errored (the pre-fix clean-room failure shape).
_ERROR_HARNESS = {"error": "ModuleNotFoundError: No module named 'sklearn'", "returncode": 1}

_ENV_FRESH = {
    "isolation_level": "fresh_worktree",
    "install_returncode": 0,
    "install_transcript_hash": "abc",
    "carnot_importable_in_isolated_env": True,
}


def test_classify_ci_both_in_range_reproduces():
    # SCENARIO-FOVER-G2-ROOTCAUSE-FIXED: both numbers inside CI -> reproduced.
    cond_a_in, lc_in, repro = exp.classify_ci(0.9131, 0.0185)
    assert cond_a_in is True
    assert lc_in is True
    assert repro is True


def test_classify_ci_condition_a_out_of_range():
    # Condition A above the CI -> not reproduced even if LC is fine.
    _, _, repro = exp.classify_ci(0.95, 0.0185)
    assert repro is False


def test_classify_ci_none_values_are_not_in_ci():
    # The exact exp3430 failure shape: condition_a is None (an error).
    cond_a_in, lc_in, repro = exp.classify_ci(None, None)
    assert cond_a_in is False
    assert lc_in is False
    assert repro is False


def test_extract_numbers_from_ci95_dict():
    cond_a, lc = exp._extract_numbers(_GOOD_HARNESS)
    assert cond_a == 0.9131
    assert lc == 0.0185


def test_extract_numbers_falls_back_to_flat_learning_contribution():
    harness = {
        "condition_a_production_auroc_mean": 0.91,
        "learning_contribution": 0.02,
    }
    cond_a, lc = exp._extract_numbers(harness)
    assert cond_a == 0.91
    assert lc == 0.02


def test_pyproject_declares_sklearn_true_in_repo():
    # The fix is committed to the working tree's pyproject.
    assert exp.pyproject_declares_sklearn(REPO_ROOT) is True


def test_pyproject_declares_sklearn_false_when_absent(tmp_path):
    (tmp_path / "pyproject.toml").write_text('dependencies = ["numpy>=1.26"]\n')
    assert exp.pyproject_declares_sklearn(tmp_path) is False


def test_pyproject_declares_sklearn_false_when_missing_file(tmp_path):
    assert exp.pyproject_declares_sklearn(tmp_path) is False


def test_build_artifact_fixed_path_has_all_required_fields():
    # SCENARIO-FOVER-G2-ROOTCAUSE-FIXED end-to-end artifact assembly.
    artifact = exp.build_artifact(
        start_time=100.0,
        preconditions={"ok": True, "head_sha": "abc123"},
        env_info=_ENV_FRESH,
        harness_result=_GOOD_HARNESS,
        isolated_versions={"python": "3.14.5", "scikit_learn": "1.8.0"},
        fix_present=True,
        clock=lambda: 142.0,
    )
    required = {
        "honest_verdict",
        "inference_substrate",
        "cleanroom_failure_traceback",
        "root_cause",
        "fix_applied",
        "isolation_level",
        "condition_a_auroc_reproduced",
        "learning_contribution_reproduced",
        "reproduced_in_ci",
        "isolated_env_versions",
        "g2_status",
        "reproducibility_checksum",
        "random_seed",
        "duration_s",
    }
    assert required.issubset(artifact.keys())
    # Terminal-prefix + substrate discipline.
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    # Reproduction outcome.
    assert artifact["reproduced_in_ci"] is True
    assert artifact["condition_a_auroc_reproduced"] == 0.9131
    assert artifact["learning_contribution_reproduced"] == 0.0185
    assert artifact["g2_status"] == "cleanroom_reproducible_internal_external_run_pending"
    # Never claim G2 met.
    assert artifact["g2_independent_reproducer"] is False
    # Duration comes from the injected clock.
    assert artifact["duration_s"] == 42.0
    # Seeds are the published ones.
    assert artifact["random_seed"] == [42, 137, 271, 314, 1729]
    # Every required field carries a principle.
    assert required.issubset(set(artifact["field_principles"].keys()))


def test_build_artifact_records_real_failure_traceback():
    artifact = exp.build_artifact(
        start_time=0.0,
        preconditions={"ok": True},
        env_info=_ENV_FRESH,
        harness_result=_GOOD_HARNESS,
        isolated_versions={},
        fix_present=True,
        clock=lambda: 1.0,
    )
    tb = artifact["cleanroom_failure_traceback"]
    assert tb
    assert "ModuleNotFoundError: No module named 'sklearn'" in tb
    assert "tier0g_semantic_energy" in tb
    assert artifact["root_cause"] == "other_undeclared_sklearn_dependency"


def test_build_artifact_failing_path_reports_still_failing():
    # SCENARIO-FOVER-G2-ROOTCAUSE-FALLBACK: harness errored -> honest failure.
    artifact = exp.build_artifact(
        start_time=0.0,
        preconditions={"ok": True},
        env_info=_ENV_FRESH,
        harness_result=_ERROR_HARNESS,
        isolated_versions={},
        fix_present=False,
        clock=lambda: 5.0,
    )
    assert artifact["reproduced_in_ci"] is False
    assert artifact["condition_a_auroc_reproduced"] is None
    assert artifact["g2_status"].startswith("cleanroom_still_failing_")
    assert artifact["honest_verdict"].startswith("complete:")
    assert "fix_pending" in artifact["honest_verdict"]
    assert artifact["harness_error_if_any"] == _ERROR_HARNESS["error"]


def test_check_preconditions_passes_in_repo():
    result = exp.check_preconditions(REPO_ROOT)
    assert result["ok"] is True
    assert "head_sha" in result


def test_check_preconditions_blocks_on_missing_harness(tmp_path):
    # No scripts/reproduce_fover_headline.py and no data/ -> harness block first.
    result = exp.check_preconditions(tmp_path)
    assert result["ok"] is False
    assert result["blocked_reason"] == "blocked_fover_harness_missing"


def test_check_preconditions_blocks_on_missing_corpus(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "reproduce_fover_headline.py").write_text("# stub\n")
    result = exp.check_preconditions(tmp_path)
    assert result["ok"] is False
    assert result["blocked_reason"] == "blocked_fover_corpus_missing"


def test_emitted_artifact_on_disk_is_valid():
    # The deliverable JSON exists and carries the fixed-path verdict + numbers.
    import json

    path = REPO_ROOT / "results" / (
        "experiment_3438_fover_g2_cleanroom_rootcause_and_fix_v2.json"
    )
    assert path.exists(), "experiment must have written its artifact"
    data = json.loads(path.read_text())
    assert data["honest_verdict"].startswith("complete:")
    assert data["root_cause"] == "other_undeclared_sklearn_dependency"
    assert data["reproduced_in_ci"] is True
    assert 0.9027 <= float(data["condition_a_auroc_reproduced"]) <= 0.9235
    assert 0.0125 <= float(data["learning_contribution_reproduced"]) <= 0.0245
    assert data["isolation_level"] in ("fresh_worktree", "fresh_clone")
    assert data["g2_independent_reproducer"] is False
