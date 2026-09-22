"""Tests for the REQ-ARC-WMTE-7530 B2 pure evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/experiments/semif_arc_readout_eval.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("semif_arc_readout_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _opportunity(index: int, *, fired: bool = False) -> dict[str, Any]:
    return {
        "record_type": "decision",
        "seam": "induction_timing",
        "gate_decision": "induce_now" if fired else "continue_explore",
        "attempt_id": f"attempt-{index}" if fired else None,
    }


def _attempt(
    index: int,
    *,
    progress: bool,
    planned: bool = False,
    verifier: str = "reject",
    tokens: int = 10,
) -> dict[str, Any]:
    return {
        "record_type": "induction_attempt",
        "seam": "induction_timing",
        "attempt_id": f"attempt-{index}",
        "progress_within_window": progress,
        "planned": planned,
        "verifier_result": verifier,
        "completion_tokens": tokens,
    }


def test_below_floor_reports_feasibility_only_and_keeps_censored_rows() -> None:
    """SCENARIO-ARC-WMTE-7530-FEASIBILITY forbids a numeric gate claim."""

    module = _load()
    rows = [_opportunity(index, fired=index < 2) for index in range(10)]
    rows.extend([_attempt(0, progress=True), _attempt(1, progress=False)])
    rows[-1]["progress_window_censored"] = True
    artifact = module.build_measurement(rows, metadata={"experiment_id": 7531})

    assert artifact["gate_opportunity_count"] == 10
    assert artifact["induction_attempt_count"] == 2
    assert artifact["sample_floor"]["met"] is False
    assert artifact["publication_mode"] == "feasibility_only"
    assert artifact["numeric_gate_quality_claim"] is False
    assert artifact["gate_ready_to_ship"] is False
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["per_attempt_rows"][-1]["progress_window_censored"] is True


def test_oracle_saves_only_useless_attempt_tokens_without_losing_progress() -> None:
    """SCENARIO-ARC-WMTE-7530-POSITIVE-CONTROL measures oracle headroom."""

    module = _load()
    attempts = [
        _attempt(0, progress=True, tokens=20),
        _attempt(1, progress=False, planned=True, verifier="accept", tokens=30),
        _attempt(2, progress=False, tokens=40),
    ]
    control = module.oracle_positive_control(attempts)

    assert control["headroom_exists"] is True
    assert control["completion_tokens_saved"] == 40
    assert control["progress_attempts_suppressed"] == 0
    assert control["progress_recall"] == 1.0
    assert control["suppressed_attempt_ids"] == ["attempt-2"]


def test_floor_met_with_no_suppressible_attempt_triggers_kill_verdict() -> None:
    """SCENARIO-ARC-WMTE-7530-POSITIVE-CONTROL exposes absent headroom."""

    module = _load()
    rows = [_opportunity(index, fired=index < 100) for index in range(1_000)]
    rows.extend(_attempt(index, progress=True) for index in range(100))
    artifact = module.build_measurement(rows, metadata={"experiment_id": 7531})

    assert artifact["sample_floor"]["met"] is True
    assert artifact["positive_control_headroom_exists"] is False
    assert artifact["honest_verdict"] == "complete_b2_kill_no_oracle_headroom"
    assert artifact["numeric_gate_quality_claim"] is False
