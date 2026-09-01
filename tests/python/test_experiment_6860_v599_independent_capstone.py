"""Focused tests for the V599 independent capstone.

Spec refs: REQ-REPORT-6860 and SCENARIO-REPORT-6860-*.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6860_v599_independent_capstone as exp


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"


@pytest.fixture(scope="module")
def payloads() -> dict[str, dict]:
    """REQ-REPORT-6860 loads producers only as raw data sources."""

    return exp.load_source_payloads(REPO)


@pytest.fixture(scope="module")
def artifact() -> dict:
    """REQ-REPORT-6860 builds without writing to the research record."""

    return exp.build_artifact(REPO, "20260901")


def test_req_report_6860_spec_precedes_implementation() -> None:
    """REQ-REPORT-6860 owns all requested adversarial scenarios."""

    text = SPEC.read_text(encoding="utf-8").split("REQ-REPORT-6860", 1)[1]
    for marker in (
        "SCENARIO-REPORT-6860-INVENTORY",
        "SCENARIO-REPORT-6860-IDENTITY-HASH",
        "SCENARIO-REPORT-6860-ROW-AUTHORITY",
        "SCENARIO-REPORT-6860-CIRCULARITY",
        "SCENARIO-REPORT-6860-ARC-PROVENANCE",
        "SCENARIO-REPORT-6860-RETIREMENT",
    ):
        assert marker in text


def test_scenario_report_6860_inventory_missing_and_gate_skip() -> None:
    """SCENARIO-REPORT-6860-INVENTORY keeps missing distinct from zero."""

    missing = exp.classify_task_state("exp6849", None, [])
    assert missing["verdict_class"] == "null"
    assert missing["observed_value"] is None
    skip = exp.classify_task_state(
        "exp6858",
        None,
        [{"status": "GATE_BLOCK", "detail": "headroom actual=0 expected=1"}],
    )
    assert skip["verdict_class"] == "blocked"
    assert skip["observed_value"] == "headroom actual=0 expected=1"


def test_scenario_report_6860_inventory_duplicate_task_id() -> None:
    """SCENARIO-REPORT-6860-IDENTITY-HASH rejects duplicate task IDs."""

    tasks = [{"id": "exp6848-a"}, {"id": "exp6848-a"}]
    assert exp.duplicate_task_ids(tasks) == ["exp6848-a"]
    assert exp.duplicate_task_ids(exp.load_roadmap_tasks(REPO)) == []


def test_scenario_report_6860_stale_hash(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6860-IDENTITY-HASH exposes stale source bytes."""

    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    declared = {"upstream": {"path": "source.json", "sha256": "sha256:bad"}}
    results = exp.check_declared_hashes(tmp_path, declared)
    assert results == [
        {
            "source_id": "upstream",
            "path": "source.json",
            "expected_sha256": "sha256:bad",
            "observed_sha256": exp.sha256_path(source),
            "passed": False,
        }
    ]
    source.unlink()
    assert exp.check_declared_hashes(tmp_path, declared)[0]["observed_sha256"] is None


def test_req_report_6860_conductor_inventory() -> None:
    """REQ-REPORT-6860 records retries, flags, and the structured gate block."""

    events = exp.parse_conductor_events((REPO / "ops/conductor-log.md").read_text(encoding="utf-8"))
    assert [row["task_id"] for row in events if row["status"] == "GATE_BLOCK"] == [
        "exp6858-supervisor-counterfactual-credit-audit"
    ]
    assert {row["task_id"] for row in events if row["status"] == "FLAGGED"} == {
        "exp6853-risk-sensitive-memory-opportunity-fixture",
        "exp6856-sealed-risk-sensitive-learning-audit",
        "exp6857-dynamic-live-arc-receipt-router",
    }
    assert {row["task_id"] for row in exp.retry_events(events)} == {
        "exp6851-three-family-isomorphic-compatibility-stream",
        "exp6855-counterfactual-memory-credit-audit",
    }


def test_req_report_6860_fresh_typed_reduction(payloads: dict[str, dict]) -> None:
    """REQ-REPORT-6860 recomputes typed science apart from GPU readiness."""

    result = exp.reduce_typed(payloads)
    assert result["authority_ready"] is True
    assert result["resource_admission_ready"] is True
    assert result["resource_rows_support_science"] is False
    assert result["compatibility_row_count"] == 168
    assert result["model_metrics"]["unsloth/Qwen3.6-35B-A3B-GGUF"]["wins"] == 14
    assert result["model_metrics"]["unsloth/gemma-4-31B-it-GGUF"]["mean_margin"] == pytest.approx(
        0.026394844642857167
    )
    assert result["shortcut_explanation_count"] > 0
    assert result["isomorphic_direction_failure_count"] > 0
    assert result["claim_eligible"] is False


def test_scenario_report_6860_aggregate_row_contradiction(
    payloads: dict[str, dict],
) -> None:
    """SCENARIO-REPORT-6860-ROW-AUTHORITY makes raw rows control aggregates."""

    changed = dict(payloads)
    changed["exp6851"] = copy.deepcopy(payloads["exp6851"])
    changed["exp6851"]["positive_margin_models"] = [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
    ]
    consistency = exp.aggregate_consistency(changed, exp.reduce_typed(changed), None)
    mismatch = next(row for row in consistency if row["field"] == "positive_margin_models")
    assert mismatch["passed"] is False
    assert mismatch["recomputed"] == ["unsloth/gemma-4-31B-it-GGUF"]


def test_req_report_6860_fresh_memory_reduction(payloads: dict[str, dict]) -> None:
    """REQ-REPORT-6860 does not let persistence replace safe learning."""

    result = exp.reduce_memory(payloads)
    assert result["held_future_effect"] == pytest.approx(0.054901960784313995)
    assert result["false_positive_injection_rate"] == 0.0
    assert result["abstention_rate"] == 1.0
    assert result["helpful_write_count"] == 16
    assert result["harmful_write_count"] == 13
    assert result["unsupported_write_count"] == 0
    assert result["durability_passed"] is True
    assert result["family_portability_passed"] is True
    assert result["order_portability_passed"] is True
    assert result["self_learning_ready"] is False


def test_scenario_report_6860_circular_verifier(payloads: dict[str, dict]) -> None:
    """SCENARIO-REPORT-6860-CIRCULARITY downgrades self-authorization."""

    changed = dict(payloads)
    changed["exp6854"] = copy.deepcopy(payloads["exp6854"])
    changed["exp6854"]["verifier_is_oracle"] = True
    changed["exp6854"]["verdict_class"] = "positive"
    rows = exp.circularity_checks(changed)
    finding = next(row for row in rows if row["task_id"].startswith("exp6854"))
    assert finding["supported_verdict_class"] == "circular_positive"
    assert finding["passed"] is False


def test_scenario_report_6860_arc_provenance(payloads: dict[str, dict]) -> None:
    """SCENARIO-REPORT-6860-ARC-PROVENANCE quarantines fixture effects."""

    result = exp.reduce_arc(payloads)
    assert result == {
        "qualified_source_count": 1,
        "invalid_source_count": 39,
        "supervisor_headroom_row_count": 0,
        "supervisor_action_credit": None,
        "supervisor_effect_eligible": False,
        "tool_gap_chain_count": 2,
        "tool_gap_complete_chain_count": 1,
        "tool_gap_authentic_live_chain_count": 0,
        "tool_gap_live_effect_eligible_count": 0,
        "tool_gap_contract_ready": True,
    }
    changed = dict(payloads)
    changed["exp6859"] = copy.deepcopy(payloads["exp6859"])
    changed["exp6859"]["rows"][0]["provenance_class"] = "development_proxy"
    assert exp.reduce_arc(changed)["tool_gap_authentic_live_chain_count"] == 0


def test_scenario_report_6860_retirement() -> None:
    """SCENARIO-REPORT-6860-RETIREMENT stops an unchanged blocked audit."""

    prior = {
        "experiment_id": "exp6844-supervisor-action-outcome-credit-audit",
        "verdict": "complete_blocked_supervisor_outcome_credit_audit",
        "retire_if_same_verdict": True,
    }
    decision = exp.retirement_decision("arc_supervisor", prior, "blocked", mechanism_changed=False)
    assert decision["retire"] is True
    assert decision["rerun_recommended"] is False
    assert (
        exp.retirement_decision("memory", prior, "null", mechanism_changed=True)["retire"] is False
    )


def test_req_report_6860_terminal_artifact_schema(artifact: dict) -> None:
    """REQ-REPORT-6860 emits every required field and one row per task."""

    required = {
        "field_principles",
        "preconditions_checked",
        "inference_substrate",
        "duration_s",
        "source_artifact_hashes",
        "task_state_manifest",
        "conductor_skip_manifest",
        "retry_manifest",
        "flag_manifest",
        "rows",
        "fresh_reducer_manifest",
        "aggregate_row_consistency_results",
        "typed_compatibility_disposition",
        "continuous_self_learning_disposition",
        "arc_supervisor_disposition",
        "tool_gap_receipt_disposition",
        "circularity_results",
        "retirement_decisions",
        "next_action_by_branch",
        "positive_claims",
        "null_claims",
        "blocked_claims",
        "partial_claims",
        "disqualified_claims",
        "v599_milestone_disposition_complete_score",
        "solve_claimed",
        "game_level_solve_count",
        "hardware_speedup_claimed",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
    }
    assert required <= artifact.keys()
    assert len(artifact["task_state_manifest"]) == 13
    assert len({row["task_id"] for row in artifact["task_state_manifest"]}) == 13
    assert artifact["typed_compatibility_disposition"]["verdict_class"] == "null"
    assert artifact["continuous_self_learning_disposition"]["verdict_class"] == "disqualified"
    assert artifact["arc_supervisor_disposition"]["verdict_class"] == "blocked"
    assert artifact["tool_gap_receipt_disposition"]["verdict_class"] == "partial"
    assert artifact["solve_claimed"] is False
    assert artifact["game_level_solve_count"] == 0
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["v599_milestone_disposition_complete_score"] == 1
    # The reducer completed its ungated disposition job even though the
    # preserved scientific branch outcomes include partial and blocked states.
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert exp.validate_artifact(artifact) == []


def test_req_report_6860_validator_failures(artifact: dict) -> None:
    """REQ-REPORT-6860 validation rejects unsafe or incomplete output."""

    changed = copy.deepcopy(artifact)
    changed["solve_claimed"] = True
    changed["task_state_manifest"].append(changed["task_state_manifest"][0])
    changed["honest_verdict"] = "partial_without_prefix"
    changed["field_principles"].pop("rows")
    findings = exp.validate_artifact(changed)
    assert "solve_claimed must be false" in findings
    assert "task_state_manifest task IDs must be unique" in findings
    assert "honest_verdict must start with complete_" in findings
    assert "field_principles must explain every artifact field" in findings
    assert exp.validate_artifact({})


def test_req_report_6860_atomic_cli(tmp_path: Path) -> None:
    """REQ-REPORT-6860 writes and validates only the selected output."""

    target = tmp_path / "capstone.json"
    assert exp.main(["--repo-root", str(REPO), "--date", "20260901", "--output", str(target)]) == 0
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["honest_verdict"].startswith("complete_")
    assert not list(tmp_path.glob("*.tmp"))
    assert exp.main(["--validate", "--output", str(target)]) == 0
    target.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", "--output", str(target)]) == 1
    assert exp.main(["--validate", "--output", str(tmp_path / "missing.json")]) == 1


def test_req_report_6860_defensive_input_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6860 turns unreadable and malformed sources into evidence."""

    assert exp.read_json(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert exp.read_json(bad) is None
    assert exp.classify_task_state("exp6848", {"status": "running"}, [])["verdict_class"] == (
        "partial"
    )
    assert (
        exp.classify_task_state("exp6848", {"honest_verdict": "complete_disqualified_bad"}, [])[
            "verdict_class"
        ]
        == "disqualified"
    )
    assert (
        exp.classify_task_state(
            "exp6848", {"honest_verdict": "complete_circular_positive_bad"}, []
        )["verdict_class"]
        == "circular_positive"
    )
    assert (
        exp.classify_task_state("exp6848", {"honest_verdict": "complete_positive_ready"}, [])[
            "verdict_class"
        ]
        == "positive"
    )
    conductor = "\n".join(
        [
            "before activation",
            "| 2026-09-01 | ignored | OK | detail |",
            "| Milestone 2026.09.599 activated |",
            "| short |",
            "| 2026-09-01 | unrelated task | OK | detail |",
        ]
    )
    assert exp.parse_conductor_events(conductor) == []
    assert (
        exp.check_declared_hashes(
            tmp_path,
            {"scalar": "bad", "missing_path": {}, "missing_hash": {"path": "bad.json"}},
        )
        == []
    )
    malformed_group = {
        "exp6851": {
            "rows": [
                {
                    "model_hf_id": "model",
                    "semantic_pair_identity": "pair",
                    "repeat": 0,
                    "transform_kind": "atom_rename",
                    "scalar_compatibility_margin": 1.0,
                    "row_identity": "row",
                }
            ]
        }
    }
    assert exp.reduce_typed(malformed_group)["isomorphic_direction_failure_count"] == 1
    assert exp.circularity_checks({}) == []
    monkeypatch.setattr(exp, "build_artifact", lambda _root, _date: {})
    assert exp.main(["--repo-root", str(tmp_path), "--output", str(tmp_path / "out.json")]) == 1
