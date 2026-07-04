"""Tests for Exp 5227 typed multi-head verifier memory.

Spec refs: REQ-LEARN-5227, SCENARIO-LEARN-5227.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline import multihead_verifier_memory as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
MEMORY_PATH = REPO / mod.MEMORY_RELATIVE_PATH
CONSUMER_PATH = REPO / mod.CONSUMER_RELATIVE_PATH


def test_req_learn_5227_spec_declares_multihead_memory_contract() -> None:
    """REQ-LEARN-5227: OpenSpec declares heads, evidence gates, and consumer output."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5227") :]

    for marker in (
        "REQ-LEARN-5227",
        "SCENARIO-LEARN-5227",
        "constraints",
        "provenance",
        "failures",
        "skills_rubrics",
        "entry_id",
        "promotion_state",
        "invalidated_by",
        mod.RESULT_RELATIVE_PATH,
        "ARC rubric setup",
    ):
        assert marker in section


def test_req_learn_5227_promotion_and_rollback_rules_are_evidence_gated() -> None:
    """REQ-LEARN-5227-1/2: promoted and rolled-back entries cite evidence."""

    with pytest.raises(ValueError, match="promoted entry requires evidence"):
        mod.make_typed_memory_entry(
            head="constraints",
            subject="hardware no-speedup claim boundary",
            promotion_state="promoted",
            payload={
                "constraint_id": "hardware_no_speedup_without_transcript",
                "scope": "hardware_reporting",
                "must_enforce": True,
                "forbidden_claim": "speedup_without_authenticated_transcript",
                "action": "block_speedup_claim",
            },
        )

    promoted = mod.make_typed_memory_entry(
        head="constraints",
        subject="hardware no-speedup claim boundary",
        promotion_state="promoted",
        evidence=["results/experiment_5217_hardware_continuity_v477.json"],
        payload={
            "constraint_id": "hardware_no_speedup_without_transcript",
            "scope": "hardware_reporting",
            "must_enforce": True,
            "forbidden_claim": "speedup_without_authenticated_transcript",
            "action": "block_speedup_claim",
        },
    )
    assert promoted["entry_id"].startswith("typed-memory:")
    assert promoted["head"] == "constraints"
    assert promoted["payload"]["forbidden_claim"] == "speedup_without_authenticated_transcript"

    with pytest.raises(ValueError, match="rolled_back entry requires invalidating evidence"):
        mod.make_typed_memory_entry(
            head="failures",
            subject="GAP-4 clean validation null",
            promotion_state="rolled_back",
            evidence=["results/experiment_5225_gap4_clean_scale_validation_gated_v478.json"],
            payload={
                "failure_id": "gap4_clean_null",
                "failure_mode": "canonical rows did not cross the six-win floor",
                "retirement_status": "quarantined",
                "avoid_until": "new_positive_validation",
                "replacement_gate": "six_discordant_wins_zero_losses_exact_p_lt_0_05",
            },
        )

    rolled_back = mod.make_typed_memory_entry(
        head="failures",
        subject="GAP-4 clean validation null",
        promotion_state="rolled_back",
        evidence=["results/experiment_5225_gap4_clean_scale_validation_gated_v478.json"],
        invalidated_by={
            "artifact": "results/experiment_5225_gap4_clean_scale_validation_gated_v478.json"
        },
        rollback_reason="gap4_clean_null",
        payload={
            "failure_id": "gap4_clean_null",
            "failure_mode": "canonical rows did not cross the six-win floor",
            "retirement_status": "quarantined",
            "avoid_until": "new_positive_validation",
            "replacement_gate": "six_discordant_wins_zero_losses_exact_p_lt_0_05",
        },
    )
    assert rolled_back["promotion_state"] == "rolled_back"
    assert rolled_back["invalidated_by"]["artifact"].endswith("v478.json")
    assert rolled_back["rollback_reason"] == "gap4_clean_null"

    with pytest.raises(ValueError, match="unknown memory head"):
        mod.make_typed_memory_entry(
            head="free_text",
            subject="bad",
            promotion_state="promoted",
            evidence=["results/source.json"],
            payload={"note": "bad"},
        )
    with pytest.raises(ValueError, match="payload missing required keys"):
        mod.make_typed_memory_entry(
            head="provenance",
            subject="bad",
            promotion_state="promoted",
            evidence=["results/source.json"],
            payload={"source_scope": "gap1"},
        )
    with pytest.raises(ValueError, match="unknown promotion_state"):
        mod.make_typed_memory_entry(
            head="constraints",
            subject="bad state",
            promotion_state="unknown",
            evidence=["results/source.json"],
            payload={
                "constraint_id": "x",
                "scope": "test",
                "must_enforce": True,
                "forbidden_claim": "bad",
                "action": "reject",
            },
        )


def test_req_learn_5227_validation_reports_schema_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5227: malformed typed memory is rejected with concrete errors."""

    valid_constraint_payload = {
        "constraint_id": "x",
        "scope": "test",
        "must_enforce": True,
        "forbidden_claim": "bad",
        "action": "reject",
    }
    bad_memory = {
        "heads": ["bad"],
        "entries": [
            {
                "head": "constraints",
                "promotion_state": "unknown",
                "payload": valid_constraint_payload,
                "evidence": ["results/source.json"],
            },
            {
                "head": "constraints",
                "promotion_state": "promoted",
                "payload": valid_constraint_payload,
                "evidence": [],
            },
            {
                "head": "constraints",
                "promotion_state": "rolled_back",
                "payload": valid_constraint_payload,
                "evidence": ["results/source.json"],
                "invalidated_by": {},
            },
            {
                "head": "constraints",
                "promotion_state": "promoted",
                "payload": valid_constraint_payload,
                "evidence": ["results/source.json"],
                "test_output": [[1]],
            },
        ],
    }

    errors = mod.validate_memory(bad_memory)

    assert any("typed heads mismatch" in error for error in errors)
    assert any("unknown promotion_state" in error for error in errors)
    assert any("promoted entry requires evidence" in error for error in errors)
    assert any("rolled_back entry requires invalidating evidence" in error for error in errors)
    assert any("test_output" in error for error in errors)
    assert mod.query_memory({"entries": []}, "and the") == []
    assert mod._find_v477_entry({"entries": []}, "GAP-X") == {}
    assert mod._output_root(tmp_path / "artifact.json") == tmp_path
    assert mod._relative_to_output(tmp_path / "elsewhere.json", tmp_path / "nested") == str(
        tmp_path / "elsewhere.json"
    )

    monkeypatch.setattr(mod, "validate_memory", lambda _memory: ["fixture schema error"])
    with pytest.raises(ValueError, match="fixture schema error"):
        mod.build_memory_bundle(
            inputs=mod.load_source_artifacts(REPO),
            tests_run=[{"command": "fixture", "passed": True}],
        )


def test_req_learn_5227_builds_heads_imports_v477_and_adds_outcomes() -> None:
    """REQ-LEARN-5227-3: the bundle imports Exp 5214 and adds V477/V478 outcomes."""

    inputs = mod.load_source_artifacts(REPO)
    bundle = mod.build_memory_bundle(
        inputs=inputs, tests_run=[{"command": "fixture", "passed": True}]
    )
    memory = bundle["memory"]
    result = bundle["result"]

    assert memory["heads"] == list(mod.TYPED_MEMORY_HEADS)
    assert result["typed_memory_heads"]["value"] == list(mod.TYPED_MEMORY_HEADS)
    assert result["memory_entries_written"]["value"] == 6
    assert result["promotions"]["value"] == 2
    assert result["rollbacks"]["value"] == 4
    assert mod.validate_memory(memory) == []

    entries = {entry["subject"]: entry for entry in memory["entries"]}
    gap1_memory = entries["GAP-1 orientation discriminator memory-only promotion"]
    assert gap1_memory["promotion_state"] == "promoted"
    assert gap1_memory["payload"]["allowed_use"] == "memory_only"
    assert gap1_memory["payload"]["blocked_use"] == "registry_promotion"
    assert gap1_memory["payload"]["imported_from_memory_id"] == "verifier-memory:fdd0d952dbf7f33e"

    gap1_registry = entries["GAP-1 registry promotion blocked by subset instability"]
    assert gap1_registry["promotion_state"] == "rolled_back"
    assert gap1_registry["invalidated_by"]["artifact"] == mod.EXP5222_RELATIVE_PATH
    assert gap1_registry["payload"]["registry_state"] == "blocked_instability"

    gap4 = entries["GAP-4 candidate-pool validation null/quarantine"]
    assert gap4["promotion_state"] == "rolled_back"
    assert gap4["payload"]["retirement_status"] == "quarantined"
    assert gap4["invalidated_by"]["artifact"] == mod.EXP5225_RELATIVE_PATH

    mmlu = entries["MMLU hidden-state verifier path retired"]
    assert mmlu["payload"]["retirement_status"] == "retired"
    assert mmlu["invalidated_by"]["artifact"] == mod.EXP5213_RELATIVE_PATH

    arc = entries["ARC live-path zero-level delta retained for rubric setup"]
    assert arc["head"] == "skills_rubrics"
    assert arc["payload"]["known_nulls"]["reproducible_total_levels_delta"] == 0
    assert (
        arc["payload"]["recommended_consumer_action"] == "build_process_rubric_before_level_patch"
    )

    hardware = entries["Hardware speedup claim boundary"]
    assert hardware["promotion_state"] == "promoted"
    assert (
        hardware["payload"]["forbidden_claim"]
        == "hardware_speedup_without_authenticated_transcript"
    )


def test_req_learn_5227_retention_check_returns_older_critical_retirements() -> None:
    """REQ-LEARN-5227-4: relevant queries still retrieve older retirements/nulls."""

    memory = mod.build_memory_bundle(
        inputs=mod.load_source_artifacts(REPO),
        tests_run=[{"command": "fixture", "passed": True}],
    )["memory"]

    mmlu_results = mod.query_memory(memory, "hidden-state MMLU path should stay retired")
    assert [entry["subject"] for entry in mmlu_results] == [
        "MMLU hidden-state verifier path retired"
    ]

    gap4_results = mod.query_memory(memory, "GAP-4 clean validation null quarantine")
    assert [entry["subject"] for entry in gap4_results] == [
        "GAP-4 candidate-pool validation null/quarantine"
    ]

    assert mod.query_memory(memory, "unrelated verifier topic") == []
    retention = mod.run_retention_check(memory)
    assert retention == {"passed": True, "queries": mod.DEFAULT_RETENTION_QUERIES}

    missing_mmlu = {
        **memory,
        "entries": [
            entry
            for entry in memory["entries"]
            if entry["subject"] != "MMLU hidden-state verifier path retired"
        ],
    }
    assert mod.run_retention_check(missing_mmlu)["passed"] is False


def test_scenario_learn_5227_consumer_summary_is_arc_rubric_ready() -> None:
    """SCENARIO-LEARN-5227: Exp 5228 can consume ARC rubric setup directly."""

    consumer = mod.build_memory_bundle(
        inputs=mod.load_source_artifacts(REPO),
        tests_run=[{"command": "fixture", "passed": True}],
    )["consumer_summary"]

    assert consumer["consumer_ready"] is True
    assert consumer["next_task"] == "exp5228-arc-provenance-skill-rubric-gate-v478"
    assert consumer["rubric_fields"] == [
        "skill_selection",
        "skill_following",
        "skill_composition",
        "reflection_retry_quality",
        "provenance_validity",
    ]
    assert consumer["known_arc_nulls"]["new_levels_banked"] == []
    assert consumer["known_arc_nulls"]["reproducible_total_levels_delta"] == 0
    assert consumer["known_arc_nulls"]["paw_amortization_viable"] is False
    assert consumer["provenance_requirements"]["accepted"] == ["live_agent_self_discovery"]
    assert "development_proxy" in consumer["provenance_requirements"]["blocked"]
    assert (
        consumer["memory_pointers"]["gap1"]
        == "GAP-1 orientation discriminator memory-only promotion"
    )
    assert consumer["memory_pointers"]["gap4"] == "GAP-4 candidate-pool validation null/quarantine"
    assert (
        consumer["memory_pointers"]["arc"]
        == "ARC live-path zero-level delta retained for rubric setup"
    )


def test_scenario_learn_5227_run_writes_consumer_ready_files(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5227: run() writes result, memory, and consumer files."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    memory_path = tmp_path / mod.MEMORY_RELATIVE_PATH
    consumer_path = tmp_path / mod.CONSUMER_RELATIVE_PATH
    tests_run = [{"command": "fixture command", "passed": True}]

    artifact = mod.run(
        root=REPO,
        result_path=result_path,
        memory_path=memory_path,
        consumer_path=consumer_path,
        tests_run=tests_run,
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    consumer = json.loads(consumer_path.read_text(encoding="utf-8"))
    assert artifact["continuous_self_learning_task"]["value"] is True
    assert artifact["consumer_ready_path"]["value"] == str(consumer_path.relative_to(tmp_path))
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["retention_check_passed"]["value"] is True
    assert artifact["inference_substrate"]["value"] == "verified_typed_memory_no_model_training"
    assert artifact["broad_self_distillation_used"]["value"] is False
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert memory["summary"]["consumer_ready_path"] == str(consumer_path.relative_to(tmp_path))
    assert consumer["consumer_ready"] is True


def test_req_learn_5227_repository_artifact_is_consumer_ready() -> None:
    """REQ-LEARN-5227-5: checked-in Exp 5227 artifact has required schema fields."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    consumer = json.loads(CONSUMER_PATH.read_text(encoding="utf-8"))

    assert result["continuous_self_learning_task"]["value"] is True
    assert result["typed_memory_heads"]["value"] == list(mod.TYPED_MEMORY_HEADS)
    assert result["memory_entries_written"]["value"] == len(memory["entries"])
    assert result["promotions"]["value"] == memory["summary"]["promotions"]
    assert result["rollbacks"]["value"] == memory["summary"]["rollbacks"]
    assert result["retention_check_passed"]["value"] is True
    assert result["consumer_ready_path"]["value"] == mod.CONSUMER_RELATIVE_PATH
    assert result["broad_self_distillation_used"]["value"] is False
    assert result["inference_substrate"]["value"] == "verified_typed_memory_no_model_training"
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert "consumer-ready" in result["honest_verdict"]["value"]
    assert consumer["consumer_ready"] is True
    assert mod.validate_memory(memory) == []
