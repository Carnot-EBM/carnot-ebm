"""Tests for Exp6407 provenance tiered factor memory protocol.

Spec refs: REQ-LEARN-6407, SCENARIO-LEARN-6407-RAW-COMPILED,
SCENARIO-LEARN-6407-REPLAY, SCENARIO-LEARN-6407-ESCALATION,
SCENARIO-LEARN-6407-CONTAMINATION, SCENARIO-LEARN-6407-ATTACKS,
SCENARIO-LEARN-6407-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6407_provenance_tiered_factor_memory_protocol as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data_6407",
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        write=write,
    )


def _refresh(artifact: dict[str, Any]) -> dict[str, Any]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_learn_6407_spec_declares_protocol_fields() -> None:
    """REQ-LEARN-6407: OpenSpec owns the tiered-memory contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6407") : text.index("REQ-LEARN-6383")]
    for token in (
        "SCENARIO-LEARN-6407-RAW-COMPILED",
        "SCENARIO-LEARN-6407-REPLAY",
        "SCENARIO-LEARN-6407-ESCALATION",
        "SCENARIO-LEARN-6407-CONTAMINATION",
        "SCENARIO-LEARN-6407-ATTACKS",
        "SCENARIO-LEARN-6407-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_6407_raw_compiled_links_and_schemas(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6407-RAW-COMPILED: cache rows link to raw rows."""

    artifact = _artifact(tmp_path)
    raw_schema = artifact["raw_record_schema_path_hash_and_required_fields"]
    graph_schema = artifact["compiled_typed_graph_schema_path_hash_node_and_edge_types"]
    links = artifact["raw_to_compiled_provenance_link_receipts"]

    assert set(mod.RAW_REQUIRED_FIELDS) <= set(raw_schema["required_fields"])
    assert raw_schema["schema_path"].endswith(".raw_record_schema.json")
    assert raw_schema["schema_sha256"].startswith("sha256:")
    assert set(mod.COMPILED_NODE_TYPES) == set(graph_schema["node_types"])
    assert set(mod.REQUIRED_EDGE_TYPES) <= set(graph_schema["edge_types"])
    assert graph_schema["schema_path"].endswith(".compiled_typed_graph_schema.json")
    assert links["raw_ledger"]["present"] is True
    assert links["compiled_graph"]["present"] is True
    assert links["compiled_row_count"] > 0
    assert links["missing_raw_link_count"] == 0
    assert links["forged_raw_link_count"] == 0
    assert links["all_compiled_rows_trace_to_raw"] is True
    assert artifact["compiled_cache_authority_claimed"] is False

    raw_hashes = {row["raw_row_hash"] for row in links["raw_rows"]}
    for row in links["compiled_rows"]:
        assert set(row["raw_hashes"]) <= raw_hashes
        assert row["compiled_row_hash"].startswith("sha256:")


def test_scenario_learn_6407_replay_escalation_and_contamination(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6407-REPLAY: local replay and full replay agree."""

    artifact = _artifact(tmp_path)
    equations = artifact["affected_neighborhood_equations_and_receipts"]
    replay = artifact["local_vs_full_replay_equivalence_results"]
    escalation = artifact["raw_tier_escalation_rules_and_tests"]
    manifest = artifact["contamination_manifest_path_hash_counts_classes_and_partition_seals"]
    diagnostics = artifact["diagnostic_admission_feature_contract"]
    fixture_results = artifact[
        "supported_contradicted_implicit_stale_duplicate_replay_supersession_poison_and_negative_fixture_results"
    ]

    assert set(equations["operations"]) == {"addition", "revocation", "expiry", "supersession"}
    assert all(row["local_equals_full"] for row in equations["receipts"].values())
    assert replay["all_equivalent"] is True
    assert replay["mismatch_count"] == 0
    assert set(escalation["conditions"]) == set(mod.ESCALATION_CONDITIONS)
    assert escalation["all_conditions_tested"] is True
    assert escalation["all_fail_closed"] is True
    assert manifest["event_count"] >= 48
    assert set(manifest["class_counts"]) == set(mod.EVENT_CLASSES)
    assert all(count > 0 for count in manifest["class_counts"].values())
    assert set(manifest["partition_seals"]) == set(mod.PARTITIONS)
    assert manifest["partitions_sealed"] is True
    assert diagnostics["feature_names"] == list(mod.DIAGNOSTIC_FEATURES)
    assert diagnostics["weighted_diagnostic_authority"] is False
    assert artifact["exact_veto_override_count"] == 0
    assert fixture_results["all_fixture_classes_present"] is True
    assert fixture_results["poison_propagation_count"] == 0

    denied = mod.diagnostic_admission_decision(
        {
            "utility": 1.0,
            "exact_confidence": 1.0,
            "novelty": 1.0,
            "recency": 1.0,
            "content_type": "factor",
        },
        exact_veto=True,
    )
    allowed = mod.diagnostic_admission_decision(
        {
            "utility": 0.8,
            "exact_confidence": 0.9,
            "novelty": 0.7,
            "recency": 0.6,
            "content_type": "factor",
        },
        exact_veto=False,
    )
    assert denied["admitted"] is False
    assert denied["exact_veto_overridden"] is False
    assert allowed["admitted"] is True


def test_scenario_learn_6407_attacks_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6407-ATTACKS: cache attacks cannot promote authority."""

    artifact = _artifact(tmp_path)
    matrix = artifact[
        "orphan_forgery_cycle_neighborhood_head_atomic_duplicate_expiry_and_restart_attack_matrix"
    ]

    assert set(matrix["attacks"]) == set(mod.ATTACK_IDS)
    assert matrix["all_fail_closed"] is True
    assert matrix["cache_authority_claim_count"] == 0
    assert matrix["learning_utility_claim_count"] == 0
    for attack_id, row in matrix["attacks"].items():
        assert row == mod.evaluate_cache_attack(attack_id)
        assert row["failed_closed"] is True
        assert row["compiled_cache_authority_claimed"] is False
        assert row["learning_utility_claimed"] is False

    with pytest.raises(ValueError, match="unknown_attack"):
        mod.evaluate_cache_attack("not_registered")


def test_scenario_learn_6407_cli_checksum_ready_and_negative_gates(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6407-READY: readiness is fully conjunctive."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260813", "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["provenance_tiered_memory_protocol_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verifier_is_oracle"] is True
    assert artifact["learning_utility_claimed"] is False
    assert artifact["compiled_cache_authority_claimed"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is None

    negative_cases = {
        "missing_raw_link": lambda row: row["raw_to_compiled_provenance_link_receipts"].update(
            {"all_compiled_rows_trace_to_raw": False}
        ),
        "replay_mismatch": lambda row: row["local_vs_full_replay_equivalence_results"].update(
            {"all_equivalent": False}
        ),
        "escalation_gap": lambda row: row["raw_tier_escalation_rules_and_tests"].update(
            {"all_conditions_tested": False}
        ),
        "partition_open": lambda row: row[
            "contamination_manifest_path_hash_counts_classes_and_partition_seals"
        ].update({"partitions_sealed": False}),
        "exact_veto": lambda row: row.update({"exact_veto_override_count": 1}),
        "attack_success": lambda row: row[
            "orphan_forgery_cycle_neighborhood_head_atomic_duplicate_expiry_and_restart_attack_matrix"
        ].update({"all_fail_closed": False}),
        "cache_authority": lambda row: row.update({"compiled_cache_authority_claimed": True}),
        "learning_utility": lambda row: row.update({"learning_utility_claimed": True}),
        "protected_change": lambda row: row["protected_files_unchanged"].update(
            {"unchanged": False}
        ),
        "failed_test": lambda row: row["tests_run"]["exit_codes"].update(
            {mod.DEFAULT_TEST_COMMANDS[0]: 1}
        ),
    }
    for mutate in negative_cases.values():
        candidate = deepcopy(artifact)
        mutate(candidate)
        _refresh(candidate)
        assert candidate["provenance_tiered_memory_protocol_ready_score"] == 0.0

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_learn_6407_helpers_and_fail_closed_paths(tmp_path: Path) -> None:
    """REQ-LEARN-6407: helpers expose deterministic fail-closed paths."""

    assert mod.sha256_json({"ok": True}).startswith("sha256:")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.as_mapping([]) == {}
    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    with pytest.raises(ValueError, match="forced"):
        mod.require(False, "forced")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        mod.read_json(malformed)

    raw_rows = mod.build_raw_records()[:8]
    graph = mod.compile_typed_graph(raw_rows)
    assert mod.raw_record_schema()["required_fields"] == list(mod.RAW_REQUIRED_FIELDS)
    assert set(mod.compiled_typed_graph_schema()["node_types"]) == set(
        mod.COMPILED_NODE_TYPES
    )
    assert mod.raw_to_compiled_provenance_link_receipts(
        raw_rows,
        graph,
        raw_ledger_path=tmp_path / "missing_raw.jsonl",
        compiled_graph_path=tmp_path / "missing_graph.json",
    )["all_compiled_rows_trace_to_raw"] is True
    forged_graph = deepcopy(graph)
    forged_graph["nodes"][0]["raw_hashes"] = ["sha256:forged"]
    forged_receipt = mod.raw_to_compiled_provenance_link_receipts(
        raw_rows,
        forged_graph,
        raw_ledger_path=tmp_path / "missing_raw.jsonl",
        compiled_graph_path=tmp_path / "missing_graph.json",
    )
    assert forged_receipt["forged_raw_link_count"] == 1
    assert forged_receipt["all_compiled_rows_trace_to_raw"] is False

    bad_preconditions = mod.preconditions_checked(
        date="20260101",
        upstream={"all_hashes_present": False, "license_hash_count": 0},
        raw_schema={"schema_sha256": None, "required_fields_complete": False},
        compiled_schema={"schema_sha256": None, "node_types_complete": False},
        manifest={"event_count": 0, "partitions_sealed": False},
        protected_before={"missing": None},
        source_before={"missing": None},
    )
    assert {
        "wrong_planning_date",
        "upstream_hash_missing",
        "license_hash_missing",
        "raw_schema_incomplete",
        "compiled_schema_incomplete",
        "contamination_manifest_too_short",
        "partition_seal_missing",
        "protected_hash_missing",
        "source_hash_missing",
    } == set(bad_preconditions["blocked_reasons"])
    blocked_status = mod.status({"preconditions_checked": bad_preconditions})
    assert blocked_status == "blocked_precondition"
    assert mod.honest_verdict({"status": blocked_status}).startswith("complete_null:")

    unwritten = _artifact(tmp_path, write=False)
    assert unwritten["raw_record_schema_path_hash_and_required_fields"]["schema_sha256"].startswith(
        "sha256:"
    )
