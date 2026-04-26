"""Tests for experiment_881_code_repair_v8_gemma4.py.

Covers: gate check, signed_improvement computation, honest_verdict mapping,
        model-load failure path, _exec_humaneval_test helper.

Spec: REQ-VR-020 (verify-repair live), SCENARIO-VR-030 (HumanEval live),
      REQ-CODE-010, SCENARIO-CODE-009
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root on path so the module can be imported without install.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_881_code_repair_v8_gemma4 as exp881  # noqa: E402


# ---------------------------------------------------------------------------
# _exec_humaneval_test
# ---------------------------------------------------------------------------


def test_exec_humaneval_test_passing() -> None:
    """A trivially correct implementation passes its own test."""
    code = "def add(a, b):\n    return a + b\n"
    test = "def check(fn):\n    assert fn(1, 2) == 3\n"
    assert exp881._exec_humaneval_test(code, test, "add") is True


def test_exec_humaneval_test_failing() -> None:
    """A buggy implementation returns False."""
    code = "def add(a, b):\n    return a - b\n"
    test = "def check(fn):\n    assert fn(1, 2) == 3\n"
    assert exp881._exec_humaneval_test(code, test, "add") is False


def test_exec_humaneval_test_syntax_error() -> None:
    """Syntactically invalid code returns False rather than raising."""
    assert exp881._exec_humaneval_test("def f(", "def check(fn): pass", "f") is False


# ---------------------------------------------------------------------------
# _check_gate
# ---------------------------------------------------------------------------


def test_check_gate_missing_env(tmp_path: Path) -> None:
    """Gate fails when CARNOT_FORCE_LIVE is absent."""
    env_without = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
    with patch.dict(os.environ, env_without, clear=True):
        ok, reason = exp881._check_gate()
    assert not ok
    assert "CARNOT_FORCE_LIVE" in reason


def test_check_gate_missing_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate fails when the preflight artifact file does not exist."""
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    monkeypatch.setattr(exp881, "_PREFLIGHT_PATH", tmp_path / "nonexistent.json")
    ok, reason = exp881._check_gate()
    assert not ok
    assert "missing" in reason


def test_check_gate_live_env_fixed_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate fails when live_env_fixed is False in the preflight artifact."""
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"live_env_fixed": False}))
    monkeypatch.setattr(exp881, "_PREFLIGHT_PATH", preflight)
    ok, reason = exp881._check_gate()
    assert not ok
    assert "live_env_fixed" in reason


def test_check_gate_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate passes when CARNOT_FORCE_LIVE is set and live_env_fixed is True."""
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"live_env_fixed": True}))
    monkeypatch.setattr(exp881, "_PREFLIGHT_PATH", preflight)
    ok, reason = exp881._check_gate()
    assert ok
    assert reason == "gate passed"


# ---------------------------------------------------------------------------
# signed_improvement computation and honest_verdict mapping
# ---------------------------------------------------------------------------


def _signed_improvement_and_verdict(
    results: list[dict],
    n_constraints_total: int,
    inference_mode: str = "live_gpu",
) -> tuple[float, str]:
    """Replicate the metric + verdict logic from main() for parametric testing."""
    n = len(results)
    baseline = sum(r["passed_baseline"] for r in results) / n
    carnot = sum(r["passed_repaired"] for r in results) / n
    si = round(carnot - baseline, 4)
    if n_constraints_total == 0:
        verdict = "zero_constraints"
    elif inference_mode != "live_gpu":
        verdict = "simulation_fallback"
    elif si > 0:
        verdict = "positive_repair"
    else:
        verdict = "live_no_improvement"
    return si, verdict


@pytest.mark.parametrize(
    "pass_base, pass_rep, n_constraints, inference_mode, expected_verdict, expected_si",
    [
        # Positive repair — repair improved a failing answer
        (
            [False, False],
            [True, False],
            5,
            "live_gpu",
            "positive_repair",
            0.5,
        ),
        # Live no improvement — repair made no difference
        (
            [True, False],
            [True, False],
            5,
            "live_gpu",
            "live_no_improvement",
            0.0,
        ),
        # Zero constraints — extractor found nothing
        (
            [False, False],
            [False, False],
            0,
            "live_gpu",
            "zero_constraints",
            0.0,
        ),
        # Simulation fallback — inference_mode was not live_gpu
        (
            [False, False],
            [True, True],
            5,
            "simulated",
            "simulation_fallback",
            1.0,
        ),
    ],
)
def test_verdict_mapping(
    pass_base: list[bool],
    pass_rep: list[bool],
    n_constraints: int,
    inference_mode: str,
    expected_verdict: str,
    expected_si: float,
) -> None:
    """Parametric test covering all four honest_verdict branches."""
    results = [{"passed_baseline": b, "passed_repaired": r} for b, r in zip(pass_base, pass_rep)]
    si, verdict = _signed_improvement_and_verdict(results, n_constraints, inference_mode)
    assert verdict == expected_verdict
    assert abs(si - expected_si) < 1e-6


# ---------------------------------------------------------------------------
# main() integration path: gate-blocked
# ---------------------------------------------------------------------------


def test_main_gate_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() writes a blocked artifact when the gate fails (no CARNOT_FORCE_LIVE)."""
    deliverable = tmp_path / "experiment_881_code_repair_v8_gemma4.json"
    monkeypatch.setattr(exp881, "_DELIVERABLE", str(deliverable))

    # Patch _check_gate to return failure
    monkeypatch.setattr(exp881, "_check_gate", lambda: (False, "CARNOT_FORCE_LIVE not set"))

    # Patch ExperimentTemplate so we don't hit filesystem side-effects
    mock_tmpl = MagicMock()
    # Use configure_mock to set assert_deliverable_written as a regular callable
    # (not an assertion). MagicMock raises AttributeError for assert_* names unless
    # we explicitly configure them.
    mock_tmpl.configure_mock(**{"assert_deliverable_written": MagicMock(return_value=None)})
    mock_tmpl.build_result.return_value = {
        "experiment": 881,
        "schema": [],
        "run_date": "2026-04-25",
        "started_at": "T",
        "finished_at": "T",
        "duration_s": 0.0,
        "status": "blocked",
        "title": "t",
        "honest_verdict": "blocked",
    }
    with patch(
        "scripts.experiment_881_code_repair_v8_gemma4.ExperimentTemplate", return_value=mock_tmpl
    ):
        exp881.main()

    assert deliverable.exists()
    data = json.loads(deliverable.read_text())
    assert data["status"] == "blocked"


# ---------------------------------------------------------------------------
# main() integration path: model-load failure
# ---------------------------------------------------------------------------


def test_main_model_load_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() writes a blocked artifact when model loading raises."""
    deliverable = tmp_path / "experiment_881_code_repair_v8_gemma4.json"
    monkeypatch.setattr(exp881, "_DELIVERABLE", str(deliverable))
    monkeypatch.setattr(exp881, "_check_gate", lambda: (True, "gate passed"))

    mock_tmpl = MagicMock()
    mock_tmpl.configure_mock(**{"assert_deliverable_written": MagicMock(return_value=None)})
    blocked_artifact = {
        "experiment": 881,
        "schema": [],
        "run_date": "2026-04-25",
        "started_at": "T",
        "finished_at": "T",
        "duration_s": 0.0,
        "status": "blocked",
        "title": "t",
        "honest_verdict": "blocked",
        "model_load_error": "CUDA OOM",
    }
    mock_tmpl.build_result.return_value = blocked_artifact

    # Torch must be importable for the main() function body to reach the load step
    mock_torch = MagicMock()

    def _raise(*args: object, **kwargs: object) -> None:
        raise RuntimeError("CUDA OOM")

    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.side_effect = _raise
    mock_auto_tokenizer = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = MagicMock()

    with (
        patch(
            "scripts.experiment_881_code_repair_v8_gemma4.ExperimentTemplate",
            return_value=mock_tmpl,
        ),
        patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": MagicMock(
                    AutoModelForCausalLM=mock_auto_model,
                    AutoTokenizer=mock_auto_tokenizer,
                ),
            },
        ),
    ):
        exp881.main()

    assert deliverable.exists()
    data = json.loads(deliverable.read_text())
    assert data["status"] == "blocked"
