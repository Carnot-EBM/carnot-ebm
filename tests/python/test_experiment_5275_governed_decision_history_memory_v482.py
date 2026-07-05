"""Tests for Exp 5275 governed decision-history memory.

Spec refs: REQ-LEARN-5275, SCENARIO-LEARN-5275.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.pipeline import governed_decision_history_memory as mod
from carnot.pipeline.verifier_memory import make_memory_entry


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_learn_5275_spec_declares_governed_decision_history_contract() -> None:
    """REQ-LEARN-5275: OpenSpec anchors the governed memory contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5275") :]

    for marker in (
        "REQ-LEARN-5275",
        "SCENARIO-LEARN-5275",
        "source artifact",
        "task scope",
        "evidence checksum",
        "promoted decision",
        "rejected alternatives",
        "verifier outcome",
        "conflict status",
        "poisoning flags",
        "scope flags",
        "rollback status",
        "aggregation_from_upstream_artifacts",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in section
    normalized_section = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_learn_5275_migrates_legacy_memory_with_safe_defaults() -> None:
    """REQ-LEARN-5275-1: legacy typed memory gains governed defaults."""

    legacy = make_memory_entry(
        failure_signature="gap1-orientation",
        candidate_predicate_or_set={"decision": "memory_only_orientation"},
        provenance={"experiment": "exp5261", "scope": "verifier"},
        deterministic_guard_result={"passed": True, "checks": {"fixture": True}},
        heldout_delta=0.041797,
        source_artifacts=["results/experiment_5261_typed_memory_interference_audit_v481.json"],
    )

    row = mod.migrate_legacy_memory_entry(
        legacy,
        task_scope="verifier/gap1_orientation",
        promoted_decision="use_gap1_orientation_discriminator_as_memory_only",
        rejected_alternatives=["promote_gap1_registry_now"],
        verifier_outcome={"heldout_delta": 0.041797, "verifier": "cached_fixture"},
    )

    assert row["memory_id"] == legacy["memory_id"]
    assert row["decision_id"].startswith("decision-history:")
    assert row["source_artifact"] == legacy["source_artifacts"][0]
    assert row["source_artifacts"] == legacy["source_artifacts"]
    assert row["evidence_checksum"].startswith("sha256:")
    assert row["task_scope"] == "verifier/gap1_orientation"
    assert row["promoted_decision"] == "use_gap1_orientation_discriminator_as_memory_only"
    assert row["rejected_alternatives"] == ["promote_gap1_registry_now"]
    assert row["verifier_outcome"]["heldout_delta"] == 0.041797
    assert row["conflict_status"] == "none"
    assert row["poisoning_flags"] == []
    assert row["scope_flags"] == ["in_scope"]
    assert row["rollback_status"] == "not_required"
    assert row["promotion_state"] == "promoted"

    minimal = mod.migrate_legacy_memory_entry(
        {"memory_id": "legacy:minimal", "promotion_state": "promoted"},
        task_scope="verifier/minimal",
        promoted_decision="minimal_decision",
        rejected_alternatives=[],
        verifier_outcome={},
    )

    assert minimal["source_artifact"] == "legacy:unknown"
    assert minimal["evidence_checksum"].startswith("sha256:")
    assert minimal["conflict_status"] == "none"
    assert minimal["poisoning_flags"] == []
    assert minimal["scope_flags"] == ["in_scope"]
    assert minimal["rollback_status"] == "not_required"


def test_req_learn_5275_fixtures_cover_required_governance_cases() -> None:
    """REQ-LEARN-5275-2/3/4/5/6: fixtures cover all governance gates."""

    fixtures = mod.build_deterministic_fixtures()
    rows = fixtures.rows

    assert fixtures.fixture_kinds == (
        "out_of_scope",
        "poisoning_like",
        "promotion",
        "rollback",
        "stale_conflict",
    )
    assert {row["fixture_kind"] for row in rows} == set(fixtures.fixture_kinds)
    assert any(row["conflict_status"] == "stale_conflict" for row in rows)
    assert any("out_of_scope" in row["scope_flags"] for row in rows)
    assert any(row["poisoning_flags"] for row in rows)
    assert any(row["rollback_status"] == "rolled_back_harmful" for row in rows)

    for row in rows:
        for field in mod.REQUIRED_DECISION_HISTORY_FIELDS:
            assert field in row
        assert row["evidence_checksum"].startswith("sha256:")
        assert isinstance(row["rejected_alternatives"], list)
        assert isinstance(row["verifier_outcome"], dict)


def test_req_learn_5275_governance_promotes_evicts_rejects_and_rolls_back() -> None:
    """REQ-LEARN-5275-2/3/4/5/6: governance decisions enforce memory safety."""

    audit = mod.evaluate_decision_history(mod.build_deterministic_fixtures())
    by_kind = {row["fixture_kind"]: row for row in audit["governance_rows"]}

    assert audit["provenance_fields_present"] is True
    assert audit["scope_enforcement_passed"] is True
    assert audit["stale_conflict_eviction_passed"] is True
    assert audit["harmful_memory_rollback_passed"] is True
    assert audit["unsafe_false_accepts"] == 0
    assert audit["memory_decision_history_ready"] is True

    assert by_kind["promotion"]["governance_action"] == "promote"
    assert by_kind["promotion"]["active"] is True
    assert by_kind["stale_conflict"]["governance_action"] == "evict_stale_conflict"
    assert by_kind["stale_conflict"]["active"] is False
    assert by_kind["out_of_scope"]["governance_action"] == "reject_out_of_scope"
    assert by_kind["poisoning_like"]["governance_action"] == "reject_poisoning"
    assert by_kind["rollback"]["governance_action"] == "rollback_harmful"
    assert by_kind["rollback"]["safe_action_selected"] is True

    blocked = {**audit, "memory_decision_history_ready": False}
    assert mod._honest_verdict(blocked).startswith("blocked_")


def test_scenario_learn_5275_run_writes_required_artifact_schema(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5275: run() writes the governed decision-history artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "fixture command", "outcome": "passed"}]

    artifact = mod.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["memory_decision_history_ready"] is True
    assert artifact["memory_decision_history_ready_principle"]
    assert artifact["tests_run"] == tests_run
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "decision-history memory is ready" in artifact["honest_verdict"]["value"]
    assert artifact["provenance_fields_present"]["value"] is True
    assert artifact["scope_enforcement_passed"]["value"] is True
    assert artifact["stale_conflict_eviction_passed"]["value"] is True
    assert artifact["harmful_memory_rollback_passed"]["value"] is True
    assert artifact["unsafe_false_accepts"]["value"] == 0
    assert artifact["fixture_checksums"]["value"]["fixture_set_sha256"]

    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert "value" in artifact[field]
        assert "principle" in artifact[field]
    mod.validate_artifact(artifact)


def test_req_learn_5275_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5275: checked-in artifact is stable under cached replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_result_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == "aggregation_from_upstream_artifacts"
    assert result["memory_decision_history_ready"] is True
    assert result["provenance_fields_present"]["value"] is True
    assert result["scope_enforcement_passed"]["value"] is True
    assert result["stale_conflict_eviction_passed"]["value"] is True
    assert result["harmful_memory_rollback_passed"]["value"] is True
    assert result["unsafe_false_accepts"]["value"] == 0
    mod.validate_artifact(result)
