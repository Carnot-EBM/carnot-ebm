"""Tests for Exp 561 helper functions — Tier 1 relay real-data relay.

Covers all new helper functions introduced in
scripts/experiment_561_tier1_relay_real.py:
  - load_exp554_fp_patterns()
  - load_response_corpus()
  - run_session()

Spec: REQ-SELFLEARN-013, SCENARIO-SELFLEARN-013, SCENARIO-SELFLEARN-014,
      SCENARIO-SELFLEARN-015
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import helpers from the experiment module under test.
# ---------------------------------------------------------------------------
import sys
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.experiment_561_tier1_relay_real import (
    load_exp554_fp_patterns,
    load_response_corpus,
    run_session,
)
from carnot.pipeline.constraint_addition import ViolationPattern


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_exp554_data(n_fn_vericot: int = 17, n_fn_vprm: int = 17,
                      n_fp_vericot: int = 0, n_fp_vprm: int = 0) -> dict:
    """Build a minimal Exp 554 diagnostic JSON matching the real schema."""
    def _flags(n_fn: int, n_fp: int) -> list[dict]:
        flags = []
        for _ in range(n_fn):
            flags.append({"is_correct": False, "violation_found": False, "cell": "FN"})
        for _ in range(n_fp):
            flags.append({"is_correct": True, "violation_found": True, "cell": "FP"})
        return flags

    return {
        "experiment": 554,
        "status": "success",
        "root_cause_hypothesis": "low_tp_extraction",
        "vericot_result": {
            "extractor_name": "VeriCoTStepValidator",
            "fp_rate": n_fp_vericot / 8 if n_fp_vericot else 0.0,
            "tp_rate": 0.0,
            "per_question_flags": _flags(n_fn_vericot, n_fp_vericot),
        },
        "vprm_result": {
            "extractor_name": "VPRMArithmeticVerifier",
            "fp_rate": n_fp_vprm / 8 if n_fp_vprm else 0.0,
            "tp_rate": 0.0,
            "per_question_flags": _flags(n_fn_vprm, n_fp_vprm),
        },
    }


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data) + "\n")


# ---------------------------------------------------------------------------
# Tests: load_exp554_fp_patterns
# ---------------------------------------------------------------------------


def test_load_exp554_fp_patterns_missing_file(tmp_path):
    """SCENARIO-SELFLEARN-015: missing file returns empty list."""
    missing = tmp_path / "does_not_exist.json"
    result = load_exp554_fp_patterns(missing)
    assert result == []


def test_load_exp554_fp_patterns_fn_only(tmp_path):
    """SCENARIO-SELFLEARN-013: 17 FN from VeriCoT + 17 FN from VPRM → low_tp_extraction."""
    data = _make_exp554_data(n_fn_vericot=17, n_fn_vprm=17, n_fp_vericot=0, n_fp_vprm=0)
    path = tmp_path / "exp554.json"
    _write_json(path, data)

    patterns = load_exp554_fp_patterns(path)

    assert len(patterns) == 1
    p = patterns[0]
    assert p.type == "low_tp_extraction"
    # 17 FN from each extractor = 34 total step observations
    assert p.count == 34
    # example_steps is capped at 5
    assert len(p.example_steps) <= 5


def test_load_exp554_fp_patterns_fp_present(tmp_path):
    """When FPs are present, a second pattern with type 'it_format_false_positive' is returned."""
    data = _make_exp554_data(n_fn_vericot=3, n_fn_vprm=3, n_fp_vericot=2, n_fp_vprm=1)
    path = tmp_path / "exp554_fp.json"
    _write_json(path, data)

    patterns = load_exp554_fp_patterns(path)
    types = {p.type for p in patterns}

    assert "low_tp_extraction" in types
    assert "it_format_false_positive" in types


def test_load_exp554_fp_patterns_no_violations(tmp_path):
    """All TN cells (no FN, no FP) → empty list returned."""
    data = {
        "experiment": 554,
        "status": "success",
        "vericot_result": {
            "extractor_name": "VeriCoTStepValidator",
            "per_question_flags": [
                {"is_correct": True, "violation_found": False, "cell": "TN"}
            ],
        },
        "vprm_result": {
            "extractor_name": "VPRMArithmeticVerifier",
            "per_question_flags": [],
        },
    }
    path = tmp_path / "exp554_clean.json"
    _write_json(path, data)

    patterns = load_exp554_fp_patterns(path)
    assert patterns == []


def test_load_exp554_fp_patterns_malformed_json(tmp_path):
    """Malformed JSON returns empty list without raising."""
    path = tmp_path / "bad.json"
    path.write_text("NOT JSON")
    result = load_exp554_fp_patterns(path)
    assert result == []


# ---------------------------------------------------------------------------
# Tests: load_response_corpus
# ---------------------------------------------------------------------------


def _make_corpus(n: int) -> list[dict]:
    return [
        {
            "question": f"q{i}",
            "response": f"response {i}",
            "is_correct": (i % 3 == 0),
        }
        for i in range(n)
    ]


def test_load_response_corpus_returns_n_items(tmp_path):
    """Returns exactly n items when corpus has >= n entries."""
    data = _make_corpus(50)
    path = tmp_path / "corpus.json"
    _write_json(path, data)

    result = load_response_corpus(path, 25)
    assert len(result) == 25
    assert result[0]["question"] == "q0"


def test_load_response_corpus_fewer_than_n(tmp_path):
    """Returns all items when corpus has < n entries."""
    data = _make_corpus(10)
    path = tmp_path / "small_corpus.json"
    _write_json(path, data)

    result = load_response_corpus(path, 25)
    assert len(result) == 10


def test_load_response_corpus_missing_file(tmp_path):
    """Missing file returns empty list."""
    result = load_response_corpus(tmp_path / "missing.json", 25)
    assert result == []


def test_load_response_corpus_malformed_json(tmp_path):
    """Malformed JSON returns empty list without raising."""
    path = tmp_path / "bad.json"
    path.write_text("[[not valid")
    result = load_response_corpus(path, 25)
    assert result == []


# ---------------------------------------------------------------------------
# Tests: run_session
# ---------------------------------------------------------------------------


def _make_pipeline_stub(violation_on_response: str | None = None) -> MagicMock:
    """Build a pipeline stub where verify() returns violation_found based on response."""
    from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline

    pipeline = MagicMock(spec=VerifyRepairPipeline)

    def _verify(question, response, domain="general"):
        # Fire a violation only when response matches the configured target.
        verified = not (violation_on_response and response == violation_on_response)
        return VerificationResult(verified=verified, constraints=[], energy=0.0, violations=[])

    pipeline.verify.side_effect = _verify
    return pipeline


def test_run_session_all_correct_no_violations():
    """SCENARIO-SELFLEARN-014: all TN → fp_rate=0.0, tp_rate=0.0."""
    corpus = [
        {"question": "q1", "response": "r1", "is_correct": True},
        {"question": "q2", "response": "r2", "is_correct": True},
    ]
    pipeline = _make_pipeline_stub(violation_on_response=None)

    fp_rate, tp_rate, details = run_session(corpus, pipeline)

    assert fp_rate == 0.0
    assert tp_rate == 0.0
    assert len(details) == 2
    assert all(d["cell"] == "TN" for d in details)


def test_run_session_fp_rate_computed_correctly():
    """Pipeline flags a correct response → fp_rate = 1/n_correct."""
    corpus = [
        {"question": "q1", "response": "trigger", "is_correct": True},
        {"question": "q2", "response": "safe", "is_correct": True},
        {"question": "q3", "response": "safe2", "is_correct": False},
    ]
    pipeline = _make_pipeline_stub(violation_on_response="trigger")

    fp_rate, tp_rate, details = run_session(corpus, pipeline)

    # 1 FP out of 2 correct
    assert abs(fp_rate - 0.5) < 1e-9
    # 0 TP out of 1 incorrect ("trigger" is correct, not incorrect here)
    assert tp_rate == 0.0


def test_run_session_tp_rate_computed_correctly():
    """Pipeline correctly flags an incorrect response → tp_rate = 1/n_incorrect."""
    corpus = [
        {"question": "q1", "response": "wrong_resp", "is_correct": False},
        {"question": "q2", "response": "correct_resp", "is_correct": True},
    ]
    pipeline = _make_pipeline_stub(violation_on_response="wrong_resp")

    fp_rate, tp_rate, details = run_session(corpus, pipeline)

    assert fp_rate == 0.0
    assert tp_rate == 1.0
    assert details[0]["cell"] == "TP"
    assert details[1]["cell"] == "TN"


def test_run_session_detail_cells_classified_correctly():
    """All four cell types (FP, TP, FN, TN) are assigned correctly."""
    corpus = [
        {"question": "q1", "response": "fp_resp", "is_correct": True},   # FP
        {"question": "q2", "response": "tp_resp", "is_correct": False},  # TP
        {"question": "q3", "response": "fn_resp", "is_correct": False},  # FN
        {"question": "q4", "response": "tn_resp", "is_correct": True},   # TN
    ]

    def _verify(question, response, domain="general"):
        from carnot.pipeline.verify_repair import VerificationResult
        # flag fp_resp and tp_resp, not the others
        verified = response not in ("fp_resp", "tp_resp")
        return VerificationResult(verified=verified, constraints=[], energy=0.0, violations=[])

    pipeline = MagicMock()
    pipeline.verify.side_effect = _verify

    fp_rate, tp_rate, details = run_session(corpus, pipeline)

    cells = [d["cell"] for d in details]
    assert cells == ["FP", "TP", "FN", "TN"]
    assert abs(fp_rate - 0.5) < 1e-9  # 1 FP / 2 correct
    assert tp_rate == 0.5              # 1 TP / 2 incorrect


def test_run_session_pipeline_exception_is_graceful():
    """If pipeline.verify() raises, the item is treated as no-violation (FN or TN)."""
    corpus = [
        {"question": "q1", "response": "r1", "is_correct": False},
    ]
    pipeline = MagicMock()
    pipeline.verify.side_effect = RuntimeError("broken")

    fp_rate, tp_rate, details = run_session(corpus, pipeline)

    # No violation found due to exception → FN
    assert details[0]["cell"] == "FN"
    assert fp_rate == 0.0
    assert tp_rate == 0.0


def test_run_session_empty_corpus():
    """Empty corpus returns (0.0, 0.0, [])."""
    pipeline = MagicMock()
    fp_rate, tp_rate, details = run_session([], pipeline)
    assert fp_rate == 0.0
    assert tp_rate == 0.0
    assert details == []
