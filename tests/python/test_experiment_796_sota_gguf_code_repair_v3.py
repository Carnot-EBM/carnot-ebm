"""Tests for Exp 796: SOTA GGUF Code Repair v3 — MARS margin gate.

Covers:
    - MARSMarginGate.decide() skip when margin exceeds threshold (REQ-BENCH-061)
    - MARSMarginGate.decide() ci_logits_unavailable when logits=None (REQ-BENCH-061)
    - compute_logit_margin returns top1 - top2 (REQ-BENCH-061)
    - check_retro028_gate returns False when file missing (REQ-BENCH-060)
    - check_retro028_gate returns False when retro_028_closed=False (REQ-BENCH-060)
    - check_retro028_gate returns True when retro_028_closed=True (REQ-BENCH-060)

Spec: REQ-BENCH-060, REQ-BENCH-061, SCENARIO-BENCH-084, SCENARIO-BENCH-085
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from carnot.pipeline.mars_margin_gate import (
    MARSMarginGate,
    MARSMarginResult,
    compute_logit_margin,
)
from scripts.experiment_796_sota_gguf_code_repair_v3 import check_retro028_gate


# ---------------------------------------------------------------------------
# compute_logit_margin — REQ-BENCH-061-1
# ---------------------------------------------------------------------------


def test_compute_logit_margin_returns_top1_minus_top2() -> None:
    """compute_logit_margin must return the difference of the two largest logits.

    Spec: REQ-BENCH-061-1
    """
    logits = [1.0, 3.5, 2.0, -0.5]
    # top1 = 3.5, top2 = 2.0 → margin = 1.5
    assert compute_logit_margin(logits) == pytest.approx(1.5)


def test_compute_logit_margin_single_element_returns_zero() -> None:
    """Single-element logit list has no top2 → margin is 0.0.

    Spec: REQ-BENCH-061-1
    """
    assert compute_logit_margin([5.0]) == 0.0


def test_compute_logit_margin_two_equal_elements() -> None:
    """Equal top1/top2 logits produce zero margin.

    Spec: REQ-BENCH-061-1
    """
    assert compute_logit_margin([2.0, 2.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MARSMarginGate.decide() with logits — REQ-BENCH-061, SCENARIO-BENCH-085
# ---------------------------------------------------------------------------


def test_mars_gate_skip_oracle_when_margin_above_threshold() -> None:
    """Gate must return skip_oracle=True and verdict='margin_skip' when margin > threshold.

    Spec: REQ-BENCH-061, SCENARIO-BENCH-085
    """
    gate = MARSMarginGate(threshold=2.0)
    # margin = 5.0 - 1.0 = 4.0 > 2.0 → skip
    result = gate.decide([5.0, 1.0, 0.5])
    assert isinstance(result, MARSMarginResult)
    assert result.skip_oracle is True
    assert result.honest_verdict == "margin_skip"
    assert result.logit_margin == pytest.approx(4.0)


def test_mars_gate_run_oracle_when_margin_at_threshold() -> None:
    """Gate must NOT skip oracle when margin equals the threshold (strictly greater required).

    Spec: REQ-BENCH-061
    """
    gate = MARSMarginGate(threshold=2.0)
    # margin = 3.0 - 1.0 = 2.0 == threshold → do NOT skip
    result = gate.decide([3.0, 1.0])
    assert result.skip_oracle is False
    assert result.honest_verdict == "margin_run_oracle"


def test_mars_gate_run_oracle_when_margin_below_threshold() -> None:
    """Gate must NOT skip oracle when margin is below threshold.

    Spec: REQ-BENCH-061
    """
    gate = MARSMarginGate(threshold=2.0)
    result = gate.decide([2.0, 1.5])
    assert result.skip_oracle is False
    assert result.honest_verdict == "margin_run_oracle"


# ---------------------------------------------------------------------------
# MARSMarginGate.decide() CI mode — REQ-BENCH-061-2, SCENARIO-BENCH-084
# ---------------------------------------------------------------------------


def test_mars_gate_ci_mode_when_logits_none() -> None:
    """Gate must return skip_oracle=False and verdict='ci_logits_unavailable' when logits=None.

    Spec: REQ-BENCH-061-2
    """
    gate = MARSMarginGate(threshold=2.0)
    result = gate.decide(None)
    assert result.skip_oracle is False
    assert result.honest_verdict == "ci_logits_unavailable"
    assert result.logit_margin == 0.0


# ---------------------------------------------------------------------------
# check_retro028_gate — REQ-BENCH-060-3
# ---------------------------------------------------------------------------


def test_retro028_gate_missing_file() -> None:
    """Gate returns False when the Exp 795 result file does not exist.

    Spec: REQ-BENCH-060-3
    """
    assert check_retro028_gate(Path("/nonexistent/path/experiment_795.json")) is False


def test_retro028_gate_false_when_not_closed() -> None:
    """Gate returns False when retro_028_closed is False in the artifact.

    Spec: REQ-BENCH-060-3
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump({"retro_028_closed": False, "honest_verdict": "partial_success"}, fh)
        tmp = Path(fh.name)
    try:
        assert check_retro028_gate(tmp) is False
    finally:
        tmp.unlink(missing_ok=True)


def test_retro028_gate_true_when_closed() -> None:
    """Gate returns True when retro_028_closed is True in the artifact.

    Spec: REQ-BENCH-060-3
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump({"retro_028_closed": True, "honest_verdict": "success"}, fh)
        tmp = Path(fh.name)
    try:
        assert check_retro028_gate(tmp) is True
    finally:
        tmp.unlink(missing_ok=True)


def test_retro028_gate_false_on_corrupt_json() -> None:
    """Gate returns False when the artifact file contains invalid JSON.

    Spec: REQ-BENCH-060-3
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        fh.write("not valid json {{{")
        tmp = Path(fh.name)
    try:
        assert check_retro028_gate(tmp) is False
    finally:
        tmp.unlink(missing_ok=True)
