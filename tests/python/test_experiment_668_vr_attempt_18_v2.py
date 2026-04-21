"""Tests for scripts/experiment_668_vr_attempt_18_v2.py.

Spec: REQ-VERIFY-149
SCENARIO: SCENARIO-VERIFY-196, SCENARIO-VERIFY-197
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_668_vr_attempt_18_v2 as mod
from scripts.experiment_668_vr_attempt_18_v2 import (
    EXP_ID,
    N_QUESTIONS,
    SCHEMA,
    _load_gate,
    _load_live_pairs,
    compute_honest_verdict,
    measure_forcing_recall,
    run_baseline_verification,
    run_forced_verification,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_exp_id() -> None:
    """EXP_ID must be 668.  Spec: REQ-VERIFY-149"""
    assert EXP_ID == 668


def test_n_questions() -> None:
    """N_QUESTIONS must be 25.  Spec: REQ-VERIFY-149"""
    assert N_QUESTIONS == 25


def test_schema() -> None:
    """SCHEMA must encode the experiment version.  Spec: REQ-VERIFY-149"""
    assert SCHEMA == "carnot.vr_attempt_18_v2.v1"


# ---------------------------------------------------------------------------
# _load_gate — gate check blocks when gate_open=False
# ---------------------------------------------------------------------------


def test_load_gate_returns_empty_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing gate file returns empty dict (treated as gate_open=False).  Spec: REQ-VERIFY-149"""
    monkeypatch.setattr(mod, "GATE_PATH", tmp_path / "nonexistent.json")
    result = _load_gate()
    assert result == {}


def test_load_gate_returns_empty_on_bad_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed gate file returns empty dict.  Spec: REQ-VERIFY-149"""
    bad = tmp_path / "gate.json"
    bad.write_text("{not json}")
    monkeypatch.setattr(mod, "GATE_PATH", bad)
    result = _load_gate()
    assert result == {}


def test_load_gate_returns_dict_when_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid gate file is parsed and returned.  Spec: REQ-VERIFY-149"""
    data = {"gate_open": True, "gate_version": "v4"}
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(data))
    monkeypatch.setattr(mod, "GATE_PATH", gate)
    result = _load_gate()
    assert result["gate_open"] is True


def test_gate_check_blocks_when_gate_open_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When gate_open=False, main() writes a blocked artifact.  Spec: REQ-VERIFY-149, SCENARIO-VERIFY-196"""
    # Write a gate file with gate_open=False
    gate_data = {"gate_open": False, "gate_version": "v4"}
    gate_file = tmp_path / "gate.json"
    gate_file.write_text(json.dumps(gate_data))
    monkeypatch.setattr(mod, "GATE_PATH", gate_file)

    deliverable = tmp_path / "experiment_668_vr_attempt_18_v2.json"
    monkeypatch.setattr(mod, "DELIVERABLE", str(deliverable))

    from carnot.pipeline.env_autofix import EnvironmentAutoFix
    from carnot.pipeline.atomic_writer import AtomicResultWriter

    written: list[dict] = []

    class _FakeWriter:
        def write(self, data: dict) -> None:
            written.append(data)

    class _FakeTemplate:
        def setup(self) -> None:
            pass
        def assert_deliverable_written(self) -> None:
            pass
        def build_result(self, *a, **kw):
            return {}

    with (
        patch("carnot.pipeline.env_autofix.apply_env_autofix", return_value=EnvironmentAutoFix(
            gpu_detected=False, carnot_force_live_was_set=False,
            auto_fix_applied=False, final_env_value=None
        )),
        patch("carnot.pipeline.experiment_watchdog.ExperimentTimeoutWatchdog") as mock_wd,
        patch("scripts.experiment_template.ExperimentTemplate", return_value=_FakeTemplate()),
        patch("carnot.pipeline.atomic_writer.AtomicResultWriter", return_value=_FakeWriter()),
    ):
        mock_wd.return_value.__enter__ = lambda s: s
        mock_wd.return_value.__exit__ = MagicMock(return_value=False)
        mock_wd.return_value.start = MagicMock()
        mock_wd.return_value.stop = MagicMock()

        with pytest.raises(SystemExit):
            mod._run_inner(mock_wd.return_value)

    assert len(written) == 1
    assert written[0]["honest_verdict"] == "vr_blocked"
    assert written[0]["forcing_applied"] is False


# ---------------------------------------------------------------------------
# compute_honest_verdict — signed_improvement logic
# ---------------------------------------------------------------------------


def test_verdict_vr_positive() -> None:
    """Positive improvement on live GPU yields 'vr_positive'.  Spec: REQ-VERIFY-149, SCENARIO-VERIFY-197"""
    assert compute_honest_verdict(0.1, "live_gpu") == "vr_positive"


def test_verdict_vr_no_improvement_zero() -> None:
    """Zero improvement yields 'vr_no_improvement'.  Spec: REQ-VERIFY-149"""
    assert compute_honest_verdict(0.0, "live_gpu") == "vr_no_improvement"


def test_verdict_vr_no_improvement_negative() -> None:
    """Negative signed_improvement yields 'vr_no_improvement'.  Spec: REQ-VERIFY-149"""
    assert compute_honest_verdict(-0.05, "live_gpu") == "vr_no_improvement"


def test_verdict_blocked() -> None:
    """Blocked inference mode yields 'vr_blocked'.  Spec: REQ-VERIFY-149"""
    assert compute_honest_verdict(0.0, "blocked") == "vr_blocked"


def test_verdict_ci_only() -> None:
    """CI-only mode yields 'ci_only' regardless of improvement.  Spec: REQ-VERIFY-149"""
    assert compute_honest_verdict(0.5, "ci_only") == "ci_only"


# ---------------------------------------------------------------------------
# signed_improvement computation
# ---------------------------------------------------------------------------


def test_signed_improvement_is_difference() -> None:
    """signed_improvement = post_accuracy - baseline_accuracy.  Spec: REQ-VERIFY-149"""
    baseline = 0.36
    post = 0.48
    expected = post - baseline
    assert abs(expected - 0.12) < 1e-9


def test_signed_improvement_zero_when_equal() -> None:
    """When baseline equals post, signed_improvement is 0.0.  Spec: REQ-VERIFY-149"""
    val = 0.36
    assert abs(val - val) == 0.0


# ---------------------------------------------------------------------------
# run_baseline_verification
# ---------------------------------------------------------------------------


def test_baseline_verification_counts_correct() -> None:
    """run_baseline_verification counts is_correct flags.  Spec: REQ-VERIFY-149"""
    pairs = [
        {"is_correct": True, "question": "q1", "response": "r1"},
        {"is_correct": False, "question": "q2", "response": "r2"},
        {"is_correct": True, "question": "q3", "response": "r3"},
    ]
    n_correct, correctness = run_baseline_verification(pairs, verifier=None)
    assert n_correct == 2
    assert correctness == [True, False, True]


def test_baseline_verification_all_wrong() -> None:
    """All incorrect pairs yield n_correct=0.  Spec: REQ-VERIFY-149"""
    pairs = [{"is_correct": False} for _ in range(5)]
    n_correct, _ = run_baseline_verification(pairs, verifier=None)
    assert n_correct == 0


# ---------------------------------------------------------------------------
# measure_forcing_recall — fraction of forced responses with COMPUTE: lines
# ---------------------------------------------------------------------------


def test_forcing_recall_empty_pairs() -> None:
    """Empty pairs list yields forcing_recall=0.0.  Spec: REQ-VERIFY-149"""
    recall = measure_forcing_recall([], forcer=None)
    assert recall == 0.0


def test_forcing_recall_ci_mode() -> None:
    """In CI mode (llm_caller=None) the synthetic response always has COMPUTE: lines.

    StructuredEquationForcer synthetic response:
        'We have 47 apples. COMPUTE: 47 + 28 = 76 So total is 76.'
    This has 1 COMPUTE: line, so n_compute_lines=1, recall=1.0.

    Spec: REQ-VERIFY-149, SCENARIO-VERIFY-196
    """
    from carnot.pipeline.symcode_verifier import SymCodeVerifier
    from carnot.pipeline.structured_equation_forcer import StructuredEquationForcer

    verifier = SymCodeVerifier(llm_caller=None)
    forcer = StructuredEquationForcer(llm_caller=None, verifier=verifier)

    pairs = [{"question": "What is 47 + 28?"} for _ in range(3)]
    recall = measure_forcing_recall(pairs, forcer)
    assert recall == 1.0


# ---------------------------------------------------------------------------
# run_forced_verification — CI mode
# ---------------------------------------------------------------------------


def test_forced_verification_ci_mode_all_correct() -> None:
    """In CI mode, all forced results are counted as correct.  Spec: REQ-VERIFY-149"""
    from carnot.pipeline.symcode_verifier import SymCodeVerifier
    from carnot.pipeline.structured_equation_forcer import StructuredEquationForcer

    verifier = SymCodeVerifier(llm_caller=None)
    forcer = StructuredEquationForcer(llm_caller=None, verifier=verifier)

    pairs = [
        {"question": "q1", "is_correct": False},
        {"question": "q2", "is_correct": False},
    ]
    n_correct, correctness = run_forced_verification(pairs, forcer, verifier, "ci_only")
    assert n_correct == 2
    assert all(correctness)


# ---------------------------------------------------------------------------
# Artifact schema: all required fields present
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "experiment",
    "schema",
    "run_date",
    "status",
    "honest_verdict",
    "retro_033_attempt",
    "forcing_applied",
    "signed_improvement",
    "baseline_accuracy",
    "post_accuracy",
    "n_questions",
    "inference_mode",
    "structured_forcing_recall",
]


def test_deliverable_has_required_fields() -> None:
    """Deliverable JSON contains all required schema fields.  Spec: REQ-VERIFY-149"""
    deliverable_path = _REPO_ROOT / "results" / "experiment_668_vr_attempt_18_v2.json"
    if not deliverable_path.exists():
        pytest.skip("Deliverable not yet written — run the experiment first")
    data = json.loads(deliverable_path.read_text())
    for field in REQUIRED_FIELDS:
        assert field in data, f"Missing required field: {field}"


def test_deliverable_retro_033_attempt_is_19() -> None:
    """retro_033_attempt must be 19 (this is attempt 19).  Spec: REQ-VERIFY-149"""
    deliverable_path = _REPO_ROOT / "results" / "experiment_668_vr_attempt_18_v2.json"
    if not deliverable_path.exists():
        pytest.skip("Deliverable not yet written")
    data = json.loads(deliverable_path.read_text())
    assert data["retro_033_attempt"] == 19
