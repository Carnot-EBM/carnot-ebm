"""Tests for Exp 3430: FoVer G2 clean-room validation via fresh git worktree.

References:
  REQ-FOVER-G2-CLEANROOM
  SCENARIO-FOVER-G2-CLEANROOM-PASS
  SCENARIO-FOVER-G2-CLEANROOM-BLOCKED
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import importlib.util

import pytest

# ---------------------------------------------------------------------------
# Import the module under test via importlib to avoid sys.path pollution
# ---------------------------------------------------------------------------

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "experiment_3430_fover_g2_cleanroom_validation_v1.py"
)
_spec = importlib.util.spec_from_file_location(
    "experiment_3430_fover_g2_cleanroom_validation_v1", _SCRIPT_PATH
)
exp3430 = importlib.util.module_from_spec(_spec)
sys.modules["experiment_3430_fover_g2_cleanroom_validation_v1"] = exp3430
_spec.loader.exec_module(exp3430)

REPO_ROOT = exp3430.REPO_ROOT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_harness_result(
    cond_a: float = 0.9131,
    lc: float = 0.0185,
    verdict: str = "complete: fover g2 pass",
) -> dict:
    """Minimal harness result that mimics reproduce_fover_headline output."""
    return {
        "honest_verdict": verdict,
        "condition_a_production_auroc_mean": cond_a,
        "condition_b_architecture_only_auroc_mean": 0.8947,
        "learning_contribution_ci95": {"mean": lc, "low": 0.012, "high": 0.024},
        "reproducibility_checksum": "abc123",
        "per_seed_results": [],
        "condition_a_auroc_ci95": {"mean": cond_a, "low": 0.90, "high": 0.93},
        "live_model_invoked": False,
    }


# ---------------------------------------------------------------------------
# REQ-FOVER-G2-CLEANROOM: check_preconditions
# ---------------------------------------------------------------------------


class TestCheckPreconditions:
    """SCENARIO-FOVER-G2-CLEANROOM-BLOCKED: preconditions block when artefacts missing."""

    def test_ok_when_all_present(self, tmp_path):
        """SCENARIO-FOVER-G2-CLEANROOM-PASS: all preconditions satisfied."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "reproduce_fover_headline.py").write_text("")
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "fover_corpus.jsonl").write_text("")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")
            result = exp3430.check_preconditions(tmp_path)

        assert result["ok"] is True
        assert result["head_sha"] == "abc123"

    def test_blocked_when_harness_missing(self, tmp_path):
        """SCENARIO-FOVER-G2-CLEANROOM-BLOCKED: harness absent → blocked_fover_harness_missing."""
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "fover_corpus.jsonl").write_text("")
        result = exp3430.check_preconditions(tmp_path)
        assert result["ok"] is False
        assert result["blocked_reason"] == "blocked_fover_harness_missing"

    def test_blocked_when_corpus_missing(self, tmp_path):
        """SCENARIO-FOVER-G2-CLEANROOM-BLOCKED: corpus absent → blocked_fover_corpus_missing."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "reproduce_fover_headline.py").write_text("")
        result = exp3430.check_preconditions(tmp_path)
        assert result["ok"] is False
        assert result["blocked_reason"] == "blocked_fover_corpus_missing"

    def test_blocked_when_git_unavailable(self, tmp_path):
        """SCENARIO-FOVER-G2-CLEANROOM-BLOCKED: git absent → blocked_git_unavailable."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "reproduce_fover_headline.py").write_text("")
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "fover_corpus.jsonl").write_text("")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=128, stdout="", stderr="not a git repo")
            result = exp3430.check_preconditions(tmp_path)

        assert result["ok"] is False
        assert result["blocked_reason"] == "blocked_git_unavailable"


# ---------------------------------------------------------------------------
# REQ-FOVER-G2-CLEANROOM: build_artifact
# ---------------------------------------------------------------------------


class TestBuildArtifact:
    """SCENARIO-FOVER-G2-CLEANROOM-PASS: artifact shape and gate logic."""

    def _call(self, harness_result, isolation_level="fresh_worktree"):
        return exp3430.build_artifact(
            start_time=0.0,
            preconditions={"ok": True},
            env_info={
                "isolation_level": isolation_level,
                "install_transcript_hash": "deadbeef",
                "carnot_importable_in_isolated_env": True,
            },
            harness_result=harness_result,
            isolated_versions={"python": "3.14.0"},
        )

    def test_pass_worktree_in_ci(self):
        """Acceptance gate passes for fresh_worktree + in-CI numbers."""
        art = self._call(_make_harness_result(cond_a=0.9131, lc=0.0185))
        assert art["reproduced_in_ci"] is True
        assert art["acceptance_gates_passed"] is True
        assert art["isolation_level"] == "fresh_worktree"
        assert art["g2_status"] == "cleanroom_validated_internal_external_run_pending"
        assert art["honest_verdict"].startswith("complete:")
        assert art["g2_independent_reproducer"] is False

    def test_pass_fresh_clone_in_ci(self):
        """Acceptance gate also passes for fresh_clone isolation."""
        art = self._call(_make_harness_result(cond_a=0.9131, lc=0.0185), "fresh_clone")
        assert art["acceptance_gates_passed"] is True
        assert art["g2_status"] == "cleanroom_validated_internal_external_run_pending"

    def test_fail_ci_gate_cond_a_low(self):
        """Out-of-CI condition-A → ci_gate_failed."""
        art = self._call(_make_harness_result(cond_a=0.85, lc=0.0185))
        assert art["reproduced_in_ci"] is False
        assert art["acceptance_gates_passed"] is False
        assert art["g2_status"] == "ci_gate_failed"

    def test_fail_ci_gate_lc_low(self):
        """Out-of-CI learning contribution → ci_gate_failed."""
        art = self._call(_make_harness_result(cond_a=0.9131, lc=0.005))
        assert art["reproduced_in_ci"] is False
        assert art["g2_status"] == "ci_gate_failed"

    def test_in_place_fallback_not_accepted(self):
        """in_place_fallback with in-CI numbers → not clean-room accepted."""
        art = self._call(_make_harness_result(cond_a=0.9131, lc=0.0185), "in_place_fallback")
        assert art["reproduced_in_ci"] is True
        assert art["acceptance_gates_passed"] is False
        assert art["g2_status"] == "in_place_reproduced_isolation_not_achieved"

    def test_blocked_harness(self):
        """Blocked harness result → blocked verdict."""
        art = self._call({"honest_verdict": "blocked_fr11_state_files"})
        assert "blocked" in art["honest_verdict"]
        assert art["g2_status"] == "blocked_harness_failed"

    def test_required_fields_present(self):
        """All schema-required fields are present in the artifact."""
        art = self._call(_make_harness_result())
        required = [
            "honest_verdict",
            "inference_substrate",
            "isolation_level",
            "condition_a_auroc_reproduced",
            "learning_contribution_reproduced",
            "reproduced_in_ci",
            "isolated_env_versions",
            "g2_status",
            "reproducibility_checksum",
            "random_seed",
            "duration_s",
            "field_principles",
        ]
        for field in required:
            assert field in art, f"missing required field: {field}"

    def test_inference_substrate_value(self):
        """inference_substrate must equal the verifier-scoring declaration."""
        art = self._call(_make_harness_result())
        assert art["inference_substrate"] == "verifier_ensemble_against_cached_candidates"

    def test_random_seeds(self):
        """random_seed records the published five seeds."""
        art = self._call(_make_harness_result())
        assert art["random_seed"] == [42, 137, 271, 314, 1729]

    def test_g2_independent_reproducer_never_true(self):
        """g2_independent_reproducer must never be True (no external run yet)."""
        art = self._call(_make_harness_result(cond_a=0.9131, lc=0.0185))
        assert art["g2_independent_reproducer"] is False

    def test_verdict_terminal_prefix(self):
        """honest_verdict must start with one of the terminal prefixes."""
        art = self._call(_make_harness_result(cond_a=0.9131, lc=0.0185))
        prefixes = ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped:", "shipped_")
        assert art["honest_verdict"].startswith(prefixes), (
            f"verdict lacks terminal prefix: {art['honest_verdict']!r}"
        )


# ---------------------------------------------------------------------------
# REQ-FOVER-G2-CLEANROOM: run_harness_in_isolated_env
# ---------------------------------------------------------------------------


class TestRunHarnessInIsolatedEnv:
    """SCENARIO-FOVER-G2-CLEANROOM-PASS: harness subprocess output parsed correctly."""

    def test_parses_json_from_stdout(self, tmp_path):
        """JSON on the last line of stdout is parsed as the harness result."""
        expected = {"honest_verdict": "complete: ok", "condition_a_production_auroc_mean": 0.9131}
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="log line\n" + json.dumps(expected) + "\n",
                stderr="",
            )
            result = exp3430.run_harness_in_isolated_env(tmp_path, tmp_path / "python")
        assert result["condition_a_production_auroc_mean"] == 0.9131

    def test_returns_error_on_nonzero_returncode(self, tmp_path):
        """Non-zero returncode → error dict returned."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="import error")
            result = exp3430.run_harness_in_isolated_env(tmp_path, tmp_path / "python")
        assert "error" in result

    def test_returns_error_when_no_json_line(self, tmp_path):
        """stdout with no JSON line → error dict with no_json_line_found."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="just text output\n", stderr=""
            )
            result = exp3430.run_harness_in_isolated_env(tmp_path, tmp_path / "python")
        assert result.get("error") == "no_json_line_found"


# ---------------------------------------------------------------------------
# REQ-FOVER-G2-CLEANROOM: cleanup_isolated_env
# ---------------------------------------------------------------------------


class TestCleanupIsolatedEnv:
    """SCENARIO-FOVER-G2-CLEANROOM-PASS: worktree removed after run."""

    def test_removes_worktree_when_fresh(self, tmp_path):
        """fresh_worktree → git worktree remove is called."""
        with patch("subprocess.run") as mock_run, patch("shutil.rmtree") as mock_rm:
            mock_run.return_value = MagicMock(returncode=0)
            exp3430.cleanup_isolated_env(tmp_path, "fresh_worktree", tmp_path / "wt", tmp_path)
            # git worktree remove was called
            assert any("worktree" in str(c) for c in mock_run.call_args_list)
            mock_rm.assert_called_once_with(tmp_path, ignore_errors=True)

    def test_skips_git_remove_for_clone(self, tmp_path):
        """fresh_clone → git worktree remove is NOT called."""
        with patch("subprocess.run") as mock_run, patch("shutil.rmtree") as mock_rm:
            exp3430.cleanup_isolated_env(tmp_path, "fresh_clone", tmp_path / "wt", tmp_path)
            mock_run.assert_not_called()
            mock_rm.assert_called_once()

    def test_skips_git_remove_for_in_place(self, tmp_path):
        """in_place_fallback → git worktree remove is NOT called."""
        with patch("subprocess.run") as mock_run, patch("shutil.rmtree") as mock_rm:
            exp3430.cleanup_isolated_env(tmp_path, "in_place_fallback", tmp_path / "wt", tmp_path)
            mock_run.assert_not_called()
            mock_rm.assert_called_once()


# ---------------------------------------------------------------------------
# REQ-FOVER-G2-CLEANROOM: run_experiment integration (mocked)
# ---------------------------------------------------------------------------


class TestRunExperiment:
    """SCENARIO-FOVER-G2-CLEANROOM-PASS / BLOCKED: full run_experiment path."""

    def _mock_env_info(self, isolation_level="fresh_worktree", tmp_path=None):
        from pathlib import Path
        tmpdir = tmp_path or Path("/tmp/mock_cleanroom")
        return {
            "isolation_level": isolation_level,
            "isolated_root": tmpdir / "worktree",
            "venv_python": tmpdir / "venv" / "bin" / "python",
            "tmpdir": tmpdir,
            "install_transcript": "",
            "install_transcript_hash": "abc",
            "carnot_importable_in_isolated_env": True,
        }

    def test_blocked_when_preconditions_fail(self):
        """Precondition failure short-circuits with blocked verdict."""
        with patch.object(
            exp3430, "check_preconditions", return_value={"ok": False, "blocked_reason": "blocked_fover_corpus_missing"}
        ):
            art = exp3430.run_experiment()
        assert art["honest_verdict"] == "blocked_fover_corpus_missing"

    def test_full_pass_path(self, tmp_path):
        """Happy path: worktree created, harness in CI, artifact produced."""
        harness_res = _make_harness_result(cond_a=0.9131, lc=0.0185)
        env_info = self._mock_env_info("fresh_worktree", tmp_path)

        with (
            patch.object(exp3430, "check_preconditions", return_value={"ok": True, "head_sha": "abc"}),
            patch.object(exp3430, "create_isolated_env", return_value=env_info),
            patch.object(exp3430, "get_isolated_env_versions", return_value={"python": "3.14"}),
            patch.object(exp3430, "run_harness_in_isolated_env", return_value=harness_res),
            patch.object(exp3430, "cleanup_isolated_env"),
        ):
            art = exp3430.run_experiment()

        assert art["honest_verdict"].startswith("complete:")
        assert art["reproduced_in_ci"] is True
        assert art["g2_status"] == "cleanroom_validated_internal_external_run_pending"
        assert art["g2_independent_reproducer"] is False

    def test_cleanup_called_even_on_harness_error(self, tmp_path):
        """cleanup_isolated_env is called in the finally block."""
        env_info = self._mock_env_info("fresh_worktree", tmp_path)

        with (
            patch.object(exp3430, "check_preconditions", return_value={"ok": True, "head_sha": "x"}),
            patch.object(exp3430, "create_isolated_env", return_value=env_info),
            patch.object(exp3430, "get_isolated_env_versions", return_value={}),
            patch.object(exp3430, "run_harness_in_isolated_env", return_value={"error": "boom"}),
            patch.object(exp3430, "cleanup_isolated_env") as mock_cleanup,
        ):
            exp3430.run_experiment()

        mock_cleanup.assert_called_once()
