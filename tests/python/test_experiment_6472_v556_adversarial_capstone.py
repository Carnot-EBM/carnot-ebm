"""Tests for Exp6472 V556 adversarial capstone.

Spec refs: REQ-CAPSTONE-6472,
SCENARIO-CAPSTONE-6472-INVENTORY,
SCENARIO-CAPSTONE-6472-RECOMPUTATION,
SCENARIO-CAPSTONE-6472-CLAIM-ELIGIBILITY,
SCENARIO-CAPSTONE-6472-ATTACKS,
SCENARIO-CAPSTONE-6472-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6472_v556_adversarial_capstone as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
_ARTIFACT_CACHE: dict[str, Any] | None = None


def _artifact() -> dict[str, Any]:
    global _ARTIFACT_CACHE
    if _ARTIFACT_CACHE is None:
        _ARTIFACT_CACHE = mod.build_artifact(
            repo_root=REPO,
            date="20260819",
            result_path=Path("/tmp/experiment_6472_test_result.json"),
            write=False,
            run_current_checks=False,
            duration_s=1.0,
            tests_run=[{"command": "focused", "exit_code": 0}],
        )
    return copy.deepcopy(_ARTIFACT_CACHE)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_req_capstone_6472_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-6472: OpenSpec owns the Exp6472 contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CAPSTONE-6472") :]

    for marker in (
        "SCENARIO-CAPSTONE-6472-INVENTORY",
        "SCENARIO-CAPSTONE-6472-RECOMPUTATION",
        "SCENARIO-CAPSTONE-6472-CLAIM-ELIGIBILITY",
        "SCENARIO-CAPSTONE-6472-ATTACKS",
        "SCENARIO-CAPSTONE-6472-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES
        assert field in mod.FIELD_PROVENANCE


def test_scenario_capstone_6472_inventory_preserves_terminal_states() -> None:
    """SCENARIO-CAPSTONE-6472-INVENTORY: each task state remains visible."""

    artifact = _artifact()
    inventory = {
        row["task_id"]: row for row in artifact["upstream_artifact_inventory"]["rows"]
    }

    assert mod.validate_artifact(artifact) == []
    assert artifact["upstream_artifact_inventory"]["expected_task_count"] == 14
    assert inventory["exp6464-fixed-slot-grounding-exact-logic-ab"]["artifact_state"] == "blocked"
    assert inventory["exp6466-held-verifier-budget-allocation-v2"]["artifact_state"] == "blocked"
    assert inventory["exp6465-representation-objective-causal-ab-v2"]["artifact_state"] == "missing"
    assert inventory["exp6467-held-exact-constraint-energy-selection-v2"]["artifact_state"] == "missing"
    assert inventory["exp6462-sota-raw-persistence-uniqueness-canary"]["readiness_fields"][
        "raw_persistence_canary_ready_score"
    ] == 1.0
    assert inventory["exp6463-sota-fixed-policy-candidate-corpus-v2"]["readiness_fields"][
        "sota_corpus_ready_score"
    ] == 0.0
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml_present"] is False


def test_scenario_capstone_6472_recomputes_raw_hashes_and_gate_contracts() -> None:
    """SCENARIO-CAPSTONE-6472-RECOMPUTATION: files and gates are recomputed."""

    artifact = _artifact()

    raw = artifact["raw_file_and_event_identity_recomputation"]
    assert raw["raw_reference_count"] > 0
    assert raw["missing_raw_file_count"] == 0
    assert raw["zero_byte_raw_file_count"] > 0
    assert raw["sha256_mismatch_count"] == 0
    assert raw["duplicate_raw_path_count"] == 0
    assert raw["duplicate_event_id_count"] == 0
    assert raw["all_raw_contracts_passed"] is False

    gates = artifact["gate_contract_recomputation"]
    assert gates["passed"] is False
    assert {
        row["task_id"] for row in gates["rows"] if row["independent_gate_passed"] is False
    } >= {
        "exp6464-fixed-slot-grounding-exact-logic-ab",
        "exp6466-held-verifier-budget-allocation-v2",
    }
    assert all(row["missing_field"] is False for row in gates["rows"])


def test_scenario_capstone_6472_independent_reducers_match_current_rows() -> None:
    """SCENARIO-CAPSTONE-6472-RECOMPUTATION: rows drive all aggregates."""

    artifact = _artifact()

    grounding = artifact[
        "independent_grounding_objective_allocation_and_energy_recomputation"
    ]
    assert grounding["sota_corpus"]["matches_reported"] is True
    assert grounding["sota_corpus"]["candidate_headroom_by_partition"]["audit_held"][
        "success"
    ] == 9
    assert grounding["grounding_exact_logic"]["state"] == "blocked"
    assert grounding["objective_causal"]["state"] == "missing"
    assert grounding["allocation"]["state"] == "blocked"
    assert grounding["energy_selection"]["state"] == "missing"

    csl = artifact["independent_csl_recomputation"]
    assert csl["exp6468_unique_event_csl"]["matches_reported"] is True
    assert csl["exp6468_unique_event_csl"]["future_held_verifier_minus_frozen"] == 1.0
    assert csl["exp6469_corruption_restart"]["matches_reported"] is True
    assert csl["exp6470_independent_audit"]["eligible_score"] == 1.0

    arc = artifact["independent_arc_recomputation"]
    assert arc["matches_reported"] is True
    assert arc["no_solve_claim"] is True
    assert arc["source_access_count"] == 0
    assert arc["per_game_adapter_count"] == 0


def test_scenario_capstone_6472_claims_and_attacks_fail_closed() -> None:
    """SCENARIO-CAPSTONE-6472-CLAIM-ELIGIBILITY: branches stay separate."""

    artifact = _artifact()

    assert artifact["v556_capstone_ready_score"] == 1.0
    assert artifact["science_claim_eligible"]["eligible"] is False
    assert artifact["continuous_learning_claim_eligible"]["eligible"] is True
    assert artifact["arc_claim_eligible"]["eligible"] is True
    assert artifact["hardware_claim_eligible"]["eligible"] is False
    assert "readiness_only" in artifact["science_claim_eligible"]["reason"]
    assert artifact["blocked_reason"] is None
    assert artifact["gate_check_summary"]["capstone_audit_complete"] is True

    attacks = {row["attack_id"]: row for row in artifact["attack_matrix"]["rows"]}
    for attack_id in mod.ATTACK_IDS:
        assert attack_id in attacks
        assert attacks[attack_id]["detected"] is True
        assert attacks[attack_id]["fail_closed"] is True
        assert attacks[attack_id]["promoted_claim"] is False
    assert artifact["repeated_prior_verdict_retirements"]["retired_count"] >= 1
    assert artifact["determination_preservation"]["v555_preserved"] is True


def test_scenario_capstone_6472_fixture_attacks_and_schema_edges(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6472-ATTACKS: corrupt fixtures fail validation."""

    raw = tmp_path / "raw.json"
    raw.write_text("payload", encoding="utf-8")
    raw_sha = mod.sha256_file(raw)
    raw_row = {
        "event_id": "evt-1",
        "raw_output_path": str(raw),
        "raw_output_sha256": raw_sha,
        "raw_byte_length": raw.stat().st_size,
        "cpu_fallback": False,
    }
    raw_report = mod.recompute_raw_identity({"fixture": {"per_unit_rows": [raw_row]}})
    assert raw_report["all_raw_contracts_passed"] is True

    bad_hash = dict(raw_row)
    bad_hash["raw_output_sha256"] = "sha256:" + "0" * 64
    assert mod.recompute_raw_identity({"fixture": {"per_unit_rows": [bad_hash]}})[
        "sha256_mismatch_count"
    ] == 1
    assert mod.device_and_model_receipts({"fixture": {"per_unit_rows": [raw_row]}})[
        "cpu_fallback_count"
    ] == 0
    cpu_row = dict(raw_row)
    cpu_row["cpu_fallback"] = True
    assert mod.device_and_model_receipts({"fixture": {"per_unit_rows": [cpu_row]}})[
        "cpu_fallback_count"
    ] == 1

    gate_payload = {"sota_corpus_ready_score": 0.0}
    assert mod.recompute_gate_contracts(
        {"exp6463": gate_payload},
        [{"task_id": "consumer", "upstream_key": "exp6463", "field": "sota_corpus_ready_score", "expected": 1.0}],
    )["passed"] is False

    artifact = _artifact()
    _write_json(tmp_path / "artifact.json", artifact)
    assert mod.validate_artifact(tmp_path / "artifact.json") == []
    assert mod.load_json(tmp_path / "artifact.json")["status"] == artifact["status"]

    bad = copy.deepcopy(artifact)
    bad["verifier_is_oracle"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "verifier_is_oracle must be true" in mod.validate_artifact(bad)

    bad = copy.deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:" + "1" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)


def test_scenario_capstone_6472_defensive_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-6472-FIELD-PRINCIPLES: defensive paths are explicit."""

    zero = tmp_path / "zero.json"
    zero.touch()
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    partial = tmp_path / "partial.json"
    _write_json(partial, {"status": "complete_partial"})
    flagged = tmp_path / "flagged.json"
    _write_json(flagged, {"status": "flagged"})

    assert mod._artifact_state(zero, None) == "zero_byte"
    assert mod._artifact_state(malformed, None) == "malformed"
    assert mod._artifact_state(partial, {"status": "complete_partial"}) == "partial"
    assert mod._artifact_state(flagged, {"status": "flagged"}) == "flagged"
    assert mod._readiness_fields(None) == {}

    monkeypatch.setattr(
        mod,
        "TASKS",
        (
            ("zero-task", "zero", Path("zero.json")),
            ("malformed-task", "bad", Path("malformed.json")),
        ),
    )
    payloads, inventory = mod.load_expected_payloads(tmp_path)
    assert payloads == {}
    assert inventory["zero_byte_task_ids"] == ["zero-task"]
    assert inventory["rows"][1]["artifact_state"] == "malformed"
    assert "JSONDecodeError" in inventory["rows"][1]["load_error"]

    raw_payload = {
        "per_unit_rows": [
            {"row_kind": "attack", "raw_output_path": "x", "raw_output_sha256": "sha256:x"},
            {"event_id": "missing_sha", "raw_output_path": "x"},
            {
                "event_id": "receipt",
                "raw_output_path": str(partial),
                "raw_output_sha256": mod.sha256_file(partial),
                "atomic_write_receipt": {"durable_byte_count": partial.stat().st_size},
            },
        ]
    }
    raw_report = mod.recompute_raw_identity({"fixture": raw_payload})
    assert raw_report["raw_reference_count"] == 1
    assert raw_report["all_raw_contracts_passed"] is True

    assert mod._raw_size({}) is None
    assert mod._bool_from_nested({"nested": [{"cpu_fallback": True}]}, "cpu_fallback") is True
    assert mod._row_exact_success({"exact_success": True}) is True
    assert mod._row_exact_success({"exact_result": {"exact_success": True}}) is True
    assert mod._row_exact_success({}) is False
    skipped = mod.reduce_exp6468_csl(
        {"per_unit_rows": [{"schema": "not_per_unit", "arm": "x", "interval": "i"}]}
    )
    assert skipped["row_count"] == 1
    assert skipped["effect_by_arm_and_interval"] == {}
    assert mod.independent_arc({})["reason"] == "missing_exp6471"
    assert mod._value_at({"x": 1}, "x.y") is None
    assert mod._critical_from_payload("x", {"current_adversarial_findings": {}}) == []
    assert mod.tests_run_receipts(None)[0]["exit_code"] is None

    assert "unloadable artifact" in mod.validate_artifact(malformed)[0]
    artifact = _artifact()
    bad = copy.deepcopy(artifact)
    del bad["status"]
    bad["unexpected"] = True
    bad["field_principles"] = {}
    bad["field_provenance"] = {}
    bad["honest_verdict"] = "ok"
    bad["v556_capstone_ready_score"] = 0.0
    bad["blocked_reason"] = None
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    errors = mod.validate_artifact(bad)
    assert any("missing required fields" in error for error in errors)
    assert any("unexpected fields" in error for error in errors)
    assert "field_principles must cover exactly required fields" in errors
    assert "field_provenance must cover exactly required fields" in errors
    assert "honest_verdict lacks terminal prefix" in errors
    assert "blocked capstone must set blocked_reason" in errors

    bad = copy.deepcopy(artifact)
    bad["blocked_reason"] = "not blocked"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "ready capstone must not set blocked_reason" in mod.validate_artifact(bad)

    bad_roadmap = tmp_path / "research-roadmap.yaml"
    bad_roadmap.write_text(":", encoding="utf-8")
    assert "error" in mod.repeated_prior_verdict_retirements(tmp_path, {"rows": []})
    _write_json(
        bad_roadmap,
        {
            "tasks": [
                "not-a-mapping",
                {"id": "x", "prior_failures": ["bad", {"retire_if_same_verdict": False}]},
            ]
        },
    )
    assert mod.repeated_prior_verdict_retirements(tmp_path, {"rows": []})["retired_count"] == 0

    artifact_root = tmp_path / "artifact-root"
    artifact_root.mkdir()
    monkeypatch.setenv("CARNOT_EXPERIMENT_ARTIFACT_ROOT", str(artifact_root))
    written = mod.build_artifact(
        repo_root=REPO,
        result_path=Path("results/experiment_6472_tmp_test.json"),
        write=True,
        duration_s=1.0,
        tests_run=[{"command": "focused", "exit_code": 0}],
    )
    assert written["status"] == "complete_v556_adversarial_capstone_audit"
    assert (artifact_root / "experiment_6472_tmp_test.json").is_file()

    monkeypatch.setattr(mod, "validate_artifact", lambda _value: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        mod.build_artifact(
            repo_root=REPO,
            write=False,
            duration_s=1.0,
            tests_run=[{"command": "focused", "exit_code": 0}],
        )
