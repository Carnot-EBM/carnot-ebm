"""Tests for Exp 1449 LTLZinc temporal continual-learning adapter.

Spec: REQ-LEARN-1449, SCENARIO-LEARN-1449.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import ltlzinc_temporal_continual_learning_adapter as mod


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_req_learn_1449_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1449-1: bootstrap artifact exposes required fields first."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    mod.validate_artifact(artifact)
    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert artifact["ltlzinc_adapter_ready"] is False
    assert artifact["temporal_cases_generated"] == 0
    assert artifact["verifier_available"] is False
    assert artifact["accepted_case_count"] == 0
    assert artifact["rejected_case_count"] == 0
    assert artifact["dataset_path"] is None
    assert artifact["commands_run"] == []
    assert artifact["honest_verdict"] == "in_progress"


def test_scenario_learn_1449_generated_cases_are_balanced_and_verified() -> None:
    """SCENARIO-LEARN-1449: every temporal family has SAT and repair-hint rows."""

    cases = mod.generate_temporal_cases()

    assert len(cases) >= 20
    families: dict[str, set[str]] = {}
    for case in cases:
        mod.validate_case_schema(case)
        satisfied = mod.verify_temporal_case(case)
        expected_satisfied = bool(case["expected_satisfied"])
        assert satisfied is expected_satisfied
        assert case["certificate_state"] == ("SAT" if expected_satisfied else "REPAIR_HINT")
        assert case["dvi_label"] == (0 if expected_satisfied else 1)
        families.setdefault(str(case["constraint_family"]), set()).add(
            str(case["certificate_state"])
        )

    assert set(families) == set(mod.SUPPORTED_OPERATORS)
    assert all(states == {"SAT", "REPAIR_HINT"} for states in families.values())


def test_req_learn_1449_verifier_distinguishes_supported_operators() -> None:
    """REQ-LEARN-1449-3: local temporal checks separate accepted and rejected traces."""

    accepted = [
        mod.make_case("always-ok", "always", "ok", [{"ok": True}, {"ok": True}], True),
        mod.make_case(
            "eventually-ready",
            "eventually",
            "ready",
            [{"ready": False}, {"ready": True}],
            True,
        ),
        mod.make_case("next-armed", "next", "armed", [{"armed": False}, {"armed": True}], True),
        mod.make_case(
            "until-ready",
            "until",
            "ready",
            [{"waiting": True, "ready": False}, {"waiting": False, "ready": True}],
            True,
            guard_signal="waiting",
        ),
    ]
    rejected = [
        mod.make_case("always-bad", "always", "ok", [{"ok": True}, {"ok": False}], False),
        mod.make_case(
            "eventually-bad",
            "eventually",
            "ready",
            [{"ready": False}, {"ready": False}],
            False,
        ),
        mod.make_case(
            "next-bad",
            "next",
            "armed",
            [{"armed": True}, {"armed": False}],
            False,
        ),
        mod.make_case(
            "until-bad",
            "until",
            "ready",
            [{"waiting": False, "ready": False}, {"waiting": True, "ready": True}],
            False,
            guard_signal="waiting",
        ),
    ]

    assert [mod.verify_temporal_case(case) for case in accepted] == [True] * 4
    assert [mod.verify_temporal_case(case) for case in rejected] == [False] * 4


def test_req_learn_1449_run_writes_dataset_and_complete_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-1449-2/4/5: run writes verified JSONL and the terminal artifact."""

    out_path = tmp_path / "results" / mod.OUTPUT_FILE
    dataset_path = tmp_path / "data" / "ltlzinc_temporal_cases_1449.jsonl"

    artifact = mod.run(
        out_path=out_path,
        dataset_path=dataset_path,
        project_root=tmp_path,
        commands_run=["pytest targeted"],
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(dataset_path)
    assert written == artifact
    mod.validate_artifact(artifact)
    assert len(rows) == artifact["temporal_cases_generated"]
    assert artifact["status"] == "complete"
    assert artifact["ltlzinc_adapter_ready"] is True
    assert artifact["verifier_available"] is True
    assert artifact["accepted_case_count"] > 0
    assert artifact["rejected_case_count"] > 0
    assert artifact["accepted_case_count"] + artifact["rejected_case_count"] == len(rows)
    assert artifact["dataset_path"] == str(dataset_path)
    assert artifact["commands_run"] == ["pytest targeted"]
    assert "FR-11" in artifact["later_milestone_feed"]["fr11"]
    assert "DVI" in artifact["later_milestone_feed"]["dvi"]
    assert artifact["honest_verdict"] == (
        "ltlzinc_temporal_adapter_ready_verified_cases_only_no_training"
    )


def test_req_learn_1449_validation_rejects_mismatched_counts(tmp_path: Path) -> None:
    """REQ-LEARN-1449-4: artifact validation protects terminal count fields."""

    artifact = mod.build_artifact(
        cases=mod.generate_temporal_cases(),
        dataset_path=tmp_path / "cases.jsonl",
        project_root=tmp_path,
    )
    artifact["accepted_case_count"] = 0

    with pytest.raises(AssertionError, match="accepted/rejected counts"):
        mod.validate_artifact(artifact)
