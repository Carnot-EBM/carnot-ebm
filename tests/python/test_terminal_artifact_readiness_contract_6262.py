"""Tests for the Exp6262 terminal-artifact readiness boundary.

Spec refs: REQ-INFRA-6262, SCENARIO-INFRA-6262-1,
SCENARIO-INFRA-6262-2, SCENARIO-INFRA-6262-3,
SCENARIO-INFRA-6262-4, SCENARIO-INFRA-6262-5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av
from carnot import terminal_artifacts as ta


REPO = Path(__file__).resolve().parents[2]
EXP6228_PATH = REPO / "results/experiment_6228_supervised_three_family_runtime_endurance.json"


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _readiness_flags(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [flag for flag in report.get("flags", []) if flag["kind"] == av.NONTERMINAL_FLAG_KIND]


def _declared_path(tmp_path: Path, name: str) -> Path:
    return tmp_path / f"experiment_6262_{name}.json"


def test_scenario_infra_6262_exp6228_preconditions_receipt_is_critical() -> None:
    """SCENARIO-INFRA-6262-1: Exp6228's current exact artifact is nonterminal."""

    classification = ta.classify_artifact_path(EXP6228_PATH)
    report = av.verify_artifact(EXP6228_PATH)
    flags = _readiness_flags(report)

    assert classification.terminal is False
    assert classification.classification == "unknown"
    assert flags == [
        {
            "kind": av.NONTERMINAL_FLAG_KIND,
            "severity": "critical",
            "detail": (
                f"declared artifact is nonterminal: path={EXP6228_PATH}; "
                "classification=unknown; reason=status or honest_verdict is absent or unknown"
            ),
        }
    ]


def test_scenario_infra_6262_nonterminal_declared_artifacts_get_critical_flags(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6262-2: nonterminal declared classes all fail closed."""

    cases = {
        "running": (
            {"status": "running", "honest_verdict": "running"},
            "running",
        ),
        "bootstrap": (
            {"status": "bootstrap_only", "honest_verdict": "blocked: bootstrap only"},
            "bootstrap_only",
        ),
        "partial": (
            {"status": "complete_partial", "honest_verdict": "complete_partial: partial"},
            "partial",
        ),
        "contradictory": (
            {"status": "complete_ready", "honest_verdict": "blocked_precondition"},
            "contradictory",
        ),
        "unknown": (
            {"status": "preconditions_recorded", "honest_verdict": None},
            "unknown",
        ),
        "non_object": (
            ["not", "an", "object"],
            "malformed",
        ),
    }

    for name, (payload, expected_class) in cases.items():
        report = av.verify_artifact(_write_json(_declared_path(tmp_path, name), payload))
        flags = _readiness_flags(report)
        assert len(flags) == 1, name
        assert flags[0]["severity"] == "critical", name
        assert f"classification={expected_class}" in flags[0]["detail"], name

    missing_report = av.verify_artifact(_declared_path(tmp_path, "missing"))
    malformed = _declared_path(tmp_path, "malformed")
    malformed.write_text("{not json", encoding="utf-8")
    malformed_report = av.verify_artifact(malformed)

    missing_detail = _readiness_flags(missing_report)[0]["detail"]
    malformed_detail = _readiness_flags(malformed_report)[0]["detail"]
    assert "classification=missing" in missing_detail
    assert "classification=malformed" in malformed_detail
    assert "artifact path is missing" in missing_detail
    assert "artifact JSON could not be loaded" in malformed_detail
    assert missing_detail != malformed_detail


def test_scenario_infra_6262_receipts_cannot_override_exact_paths(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6262-3: completion receipts do not terminalize bad paths."""

    path = _write_json(
        _declared_path(tmp_path, "receipt_negative"),
        {"status": "running", "honest_verdict": "running", "ready_score": 1},
    )

    classification = ta.classify_artifact_path(path, conductor_receipt={"status": "OK"})
    eligibility = ta.gate_field_eligibility_for_path(
        path,
        "ready_score",
        conductor_receipt={"status": "OK", "ready_score": 1},
    )

    assert classification.terminal is False
    assert classification.receipt_override_attempted is True
    assert classification.receipt_overrode is False
    assert eligibility.eligible is False
    assert eligibility.classification.classification == "running"
    assert eligibility.classification.receipt_override_attempted is True


def test_scenario_infra_6262_gate_fields_require_terminal_exact_bare_field() -> None:
    """SCENARIO-INFRA-6262-4: only terminal artifacts expose exact bare fields."""

    terminal = {"status": "complete_ready", "honest_verdict": "complete_ready: ok"}

    eligible = ta.gate_field_eligibility({**terminal, "ready_score": 1}, "ready_score")
    assert eligible.eligible is True
    assert eligible.value == 1
    assert eligible.field_present is True
    assert eligible.field_is_bare is True

    nonterminal = ta.gate_field_eligibility(
        {"status": "running", "honest_verdict": "running", "ready_score": 1},
        "ready_score",
    )
    assert nonterminal.eligible is False
    assert "nonterminal" in nonterminal.reason

    nested = ta.gate_field_eligibility({**terminal, "metrics": {"ready_score": 1}}, "ready_score")
    assert nested.eligible is False
    assert nested.field_present is False
    assert "absent" in nested.reason

    wrapped = ta.gate_field_eligibility(
        {**terminal, "ready_score": {"value": 1, "principle": "fixture"}},
        "ready_score",
    )
    assert wrapped.eligible is False
    assert wrapped.field_present is True
    assert wrapped.field_is_bare is False
    assert "not bare" in wrapped.reason

    receipt_only = ta.gate_field_eligibility(
        terminal,
        "ready_score",
        conductor_receipt={"status": "OK", "ready_score": 1},
    )
    assert receipt_only.eligible is False
    assert receipt_only.field_present is False


def test_scenario_infra_6262_honest_terminal_controls_stay_clean(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6262-5: complete, null, blocked, and skipped controls pass."""

    controls = {
        "complete": {"status": "complete", "honest_verdict": "complete: clean"},
        "null": {"status": "complete_null", "honest_verdict": "complete_null: clean null"},
        "blocked": {"status": "blocked", "honest_verdict": "blocked_precondition"},
        "gate_skip": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [{"passed": False}],
        },
    }

    for name, payload in controls.items():
        report = av.verify_artifact(_write_json(_declared_path(tmp_path, name), payload))
        assert _readiness_flags(report) == [], name

