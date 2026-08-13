"""Tests for Exp6383 dependency-guided factor rollback stress.

Spec refs: REQ-LEARN-6383, SCENARIO-LEARN-6383-SCHEMA,
SCENARIO-LEARN-6383-SELECTIVE, SCENARIO-LEARN-6383-CONTROLS,
SCENARIO-LEARN-6383-JOURNAL, SCENARIO-LEARN-6383-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6383_dependency_guided_factor_rollback_stress as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        write=write,
    )


def _refresh(artifact: dict[str, Any]) -> dict[str, Any]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_learn_6383_spec_declares_fields_and_scenarios() -> None:
    """REQ-LEARN-6383: OpenSpec owns the rollback contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6383") :]
    for token in (
        "SCENARIO-LEARN-6383-SCHEMA",
        "SCENARIO-LEARN-6383-SELECTIVE",
        "SCENARIO-LEARN-6383-CONTROLS",
        "SCENARIO-LEARN-6383-JOURNAL",
        "SCENARIO-LEARN-6383-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6383_schema_and_bad_fixtures_fail_closed() -> None:
    """SCENARIO-LEARN-6383-SCHEMA: invalid lineage never promotes."""

    schema = mod.typed_dependency_schema()
    clean = mod.build_clean_graph()
    injected = mod.build_injected_graph()

    assert schema["schema_version"] == mod.TYPED_DEPENDENCY_SCHEMA_VERSION
    assert set(schema["node_types"]) == set(mod.NODE_TYPES)
    assert set(schema["edge_types"]) == set(mod.EDGE_TYPES)
    assert mod.validate_graph(clean)["valid"] is True
    assert mod.validate_graph(injected)["valid"] is True
    assert mod.validate_graph_rows(
        list(clean["nodes"].values()),
        clean["edges"],
    )["valid"] is True
    assert mod.attack_fixture_result("clean")["accepted"] is True

    for fixture_name in (
        "duplicated",
        "misattributed",
        "cyclic",
        "missing_evidence",
        "edge_tampering",
        "orphaned_nodes",
        "corrupted_lineage",
    ):
        receipt = mod.attack_fixture_result(fixture_name)
        assert receipt["fixture"] == fixture_name
        assert receipt["fail_closed"] is True
        assert receipt["accepted"] is False
        assert receipt["reason"]

    with pytest.raises(ValueError, match="cycle_detected"):
        mod.validate_graph(mod.build_attack_graph("cyclic"))
    with pytest.raises(ValueError, match="unsupported_edge"):
        mod.validate_graph(mod.build_attack_graph("edge_tampering"))
    with pytest.raises(ValueError, match="missing_node"):
        mod.validate_graph(mod.build_attack_graph("orphaned_nodes"))
    with pytest.raises(ValueError, match="duplicate_node_id"):
        mod.validate_graph_rows(
            [
                {"id": "dup", "type": "source_event", "trusted": True},
                {"id": "dup", "type": "source_event", "trusted": True},
            ],
            [],
        )
    with pytest.raises(ValueError, match="unknown_fixture"):
        mod.build_attack_graph("not_a_fixture")


def test_scenario_learn_6383_selective_removes_only_unsupported_descendants(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6383-SELECTIVE: bad descendants roll back."""

    artifact = _artifact(tmp_path)
    diagnosis = artifact["diagnosis_receipts"]
    results = artifact["selective_full_reset_and_no_rollback_results"]
    selective = results["selective_descendant_rollback"]
    preservation = artifact["independently_supported_state_preserved"]
    removed = artifact["harmful_descendants_removed"]

    assert diagnosis["bad_source_node_id"] == mod.BAD_SOURCE_ID
    assert diagnosis["bad_source_found"] is True
    assert set(diagnosis["unsupported_descendant_node_ids"]) == set(
        mod.EXPECTED_UNSUPPORTED_DESCENDANTS
    )
    assert removed["removed_all_harmful_descendants"] is True
    assert set(removed["removed_node_ids"]) == set(mod.EXPECTED_HARMFUL_STATE_NODES)
    assert set(preservation["preserved_node_ids"]) == set(mod.EXPECTED_PRESERVED_STATE_NODES)
    assert preservation["all_independently_supported_state_preserved"] is True
    assert selective["unsafe_survivor_count"] == 0
    assert selective["overrollback_count"] == 0
    assert selective["underrollback_count"] == 0
    assert "decision_mixed_active" in selective["invalidated_active_consumer_decisions"]
    assert "factor_shared_guard" in selective["active_node_ids_after"]
    assert "decision_shared_guard" in selective["active_node_ids_after"]
    assert "factor_repair_bad" not in selective["active_node_ids_after"]


def test_scenario_learn_6383_controls_share_replay_work(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6383-CONTROLS: controls are matched."""

    artifact = _artifact(tmp_path)
    results = artifact["selective_full_reset_and_no_rollback_results"]
    selective = results["selective_descendant_rollback"]
    full_reset = results["full_registry_reset"]
    no_rollback = results["no_rollback"]

    assert len({row["initial_graph_root"] for row in results.values()}) == 1
    assert len({row["injection_order_hash"] for row in results.values()}) == 1
    assert len({row["exact_replay_work_hash"] for row in results.values()}) == 1
    assert selective["exact_replay_call_count"] == full_reset["exact_replay_call_count"]
    assert selective["valid_state_preserved_count"] > full_reset["valid_state_preserved_count"]
    assert no_rollback["unsafe_survivor_count"] > 0
    assert full_reset["overrollback_count"] > selective["overrollback_count"]
    assert artifact["overrollback_underrollback_and_unsafe_survivor_counts"] == {
        "selective_overrollback_count": 0,
        "selective_underrollback_count": 0,
        "selective_unsafe_survivor_count": 0,
        "full_reset_overrollback_count": full_reset["overrollback_count"],
        "no_rollback_underrollback_count": no_rollback["underrollback_count"],
        "no_rollback_unsafe_survivor_count": no_rollback["unsafe_survivor_count"],
    }


def test_scenario_learn_6383_journal_restart_and_idempotence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6383-JOURNAL: one terminal root is exact-valid."""

    artifact = _artifact(tmp_path)
    journal = artifact["journal_restart_and_idempotence_receipts"]
    failures = artifact["cycle_missing_edge_corruption_and_interruption_results"]
    roots = artifact["terminal_registry_roots"]

    assert journal["interrupted_journal_restart"]["restart_completed"] is True
    assert journal["interrupted_journal_restart"]["terminal_root"] == roots["selective_terminal_root"]
    assert journal["double_rollback"]["idempotent"] is True
    assert journal["double_rollback"]["second_terminal_root"] == roots["selective_terminal_root"]
    assert journal["root_mismatch"]["fail_closed"] is True
    assert journal["rollback_of_active_consumer_decision"]["decision_id"] == "decision_mixed_active"
    assert journal["rollback_of_active_consumer_decision"]["invalidated"] is True
    assert roots["idempotent_exact_valid_terminal_root"] is True
    assert failures["all_fail_closed"] is True
    assert failures["journal_interruption"]["restart_completed"] is True

    graph = mod.build_injected_graph()
    diagnosis = mod.diagnose_bad_source(graph)
    first = mod.apply_selective_rollback(graph, diagnosis)
    second = mod.apply_selective_rollback(
        graph,
        diagnosis,
        starting_active_node_ids=first["active_node_ids_after"],
    )
    assert first["terminal_root"] == second["terminal_root"]
    with pytest.raises(ValueError, match="journal_root_mismatch"):
        mod.restart_from_journal(graph, diagnosis, {"pre_root": "sha256:bad"})


def test_scenario_learn_6383_cli_checksum_ready_and_negative_gates(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6383-READY: readiness is fully conjunctive."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260813", "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    schema_receipt = artifact["typed_dependency_schema_path_hash_and_version"]

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["dependency_guided_rollback_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["no_live_utility_claim"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert schema_receipt["sha256"] == mod.sha256_file(Path(schema_receipt["path"]))
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is None

    utility_claim = deepcopy(artifact)
    utility_claim["no_live_utility_claim"] = False
    _refresh(utility_claim)
    assert utility_claim["dependency_guided_rollback_ready_score"] == 0.0
    with pytest.raises(ValueError, match="no_live_utility_claim"):
        mod.validate_artifact(utility_claim)

    failed_test = deepcopy(artifact)
    failed_test["tests_run"]["exit_codes"][mod.DEFAULT_TEST_COMMANDS[0]] = 1
    _refresh(failed_test)
    assert failed_test["dependency_guided_rollback_ready_score"] == 0.0

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_learn_6383_helpers_and_error_paths(tmp_path: Path) -> None:
    """REQ-LEARN-6383: helpers expose deterministic fail-closed paths."""

    artifact = _artifact(tmp_path, write=False)

    assert mod.sha256_json({"ok": True}).startswith("sha256:")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.as_mapping([]) == {}
    assert mod.relative_or_absolute(REPO / "AGENTS.md") == "AGENTS.md"
    assert mod.relative_or_absolute(tmp_path / "outside.json").endswith("outside.json")
    with pytest.raises(ValueError, match="forced"):
        mod.require(False, "forced")

    clean_summary = mod.graph_summary(mod.build_clean_graph())
    injected = mod.build_injected_graph()
    injected_summary = mod.graph_summary(injected)
    assert clean_summary["node_count"] < injected_summary["node_count"]
    assert injected_summary["active_state_count"] > clean_summary["active_state_count"]
    duplicate_incoming = deepcopy(injected)
    duplicate_incoming["edges"].append(mod._edge("obligation_route", "evidence_route_exact", "checked_by"))
    assert "source_clean_route" in mod._source_ancestors(
        "evidence_route_exact",
        duplicate_incoming["nodes"],
        mod._incoming(duplicate_incoming["edges"]),
    )
    duplicate_descendant = deepcopy(injected)
    duplicate_descendant["edges"].append(
        mod._edge(mod.BAD_SOURCE_ID, "obligation_bad_stale", "declares_obligation")
    )
    assert "obligation_bad_stale" in mod._descendants(duplicate_descendant, mod.BAD_SOURCE_ID)
    assert mod.tests_run(None)["exit_codes"][mod.DEFAULT_TEST_COMMANDS[0]] == 0

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    blocked = tmp_path / "blocked.json"
    blocked.write_text(
        json.dumps({"status": "blocked", "honest_verdict": "blocked: fixture"}),
        encoding="utf-8",
    )
    terminal = tmp_path / "terminal.json"
    terminal.write_text(
        json.dumps({"status": "complete_positive", "honest_verdict": "complete_positive: fixture"}),
        encoding="utf-8",
    )
    unqualified = tmp_path / "unqualified.json"
    unqualified.write_text(json.dumps({"status": "draft"}), encoding="utf-8")
    assert mod.classify_upstream_learning_context(tmp_path / "missing.json")[
        "context_class"
    ] == "absent"
    assert mod.classify_upstream_learning_context(malformed)["context_class"] == "malformed"
    assert mod.classify_upstream_learning_context(blocked)["context_class"] == "blocked"
    assert mod.classify_upstream_learning_context(terminal)["context_class"] == "terminal"
    assert mod.classify_upstream_learning_context(unqualified)[
        "context_class"
    ] == "present_unqualified"

    invalidated = deepcopy(artifact)
    invalidated["harmful_descendants_removed"]["removed_all_harmful_descendants"] = False
    _refresh(invalidated)
    assert invalidated["dependency_guided_rollback_ready_score"] == 0.0

    for field in (
        "harmful_descendants_removed",
        "independently_supported_state_preserved",
        "overrollback_underrollback_and_unsafe_survivor_counts",
        "cycle_missing_edge_corruption_and_interruption_results",
        "journal_restart_and_idempotence_receipts",
        "protected_files_unchanged",
        "tests_run",
    ):
        malformed = deepcopy(artifact)
        malformed[field] = []
        _refresh(malformed)
        assert malformed["dependency_guided_rollback_ready_score"] == 0.0

    missing_field = deepcopy(artifact)
    missing_field.pop("status")
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(missing_field)
