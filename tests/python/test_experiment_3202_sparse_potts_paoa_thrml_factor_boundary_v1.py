"""Tests for Exp 3202 sparse Potts/PAOA/THRML factor boundary v1.

Spec refs: REQ-HW-100, SCENARIO-HW-100.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import sparse_potts_paoa_thrml_factor_boundary_3202 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "source_artifacts",
    "factor_record_schema",
    "factor_record_count",
    "q_state_count_summary",
    "graph_density_summary",
    "paoa_metadata_schema",
    "thrml_local_api_checked",
    "authenticated_hardware_transcript_present",
    "speedup_claim_allowed",
    "hardware_claims_denied",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_common_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text(
        "No speedup claims without authenticated hardware receipts\n",
        encoding="utf-8",
    )
    (root / "research-references.md").write_text(
        "Potts MFC and PAOA motivate sparse multi-state factor records only.\n",
        encoding="utf-8",
    )
    (root / "research-hardware-wishlist.md").write_text(
        "KV260, GateMate, PolarFire, THRML, TSU, Z1, XTR-0, and Kona need transcripts.\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops/exclusion_manifest.yaml").write_text(
        "retired_experiments:\n"
        "  - id: thrml-scaling-sweep-retired-lineage\n"
        "    reason: no speedup without authenticated hardware\n",
        encoding="utf-8",
    )
    spec = root / mod.SPEC_REL_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-HW-100\nSCENARIO-HW-100\n"
        "results/experiment_3202_sparse_potts_paoa_thrml_factor_boundary_v1.json\n",
        encoding="utf-8",
    )
    (root / "hardware/kv260").mkdir(parents=True, exist_ok=True)
    (root / "hardware/kv260/potts_sampler_v1.v").write_text(
        "// q-state Potts boundary reference\n",
        encoding="utf-8",
    )
    _write_json(
        root,
        mod.EXP3188_REL_PATH,
        {
            "schema": "carnot.thrml_factor_graph_api_boundary.v1",
            "thrml_import_available": True,
            "local_api_smoke_passed": True,
            "selected_exact_rows": [
                {
                    "row_id": "row-valid",
                    "source_artifact": "results/exact_rows.json",
                    "exact_label": "VALID",
                    "candidate_answers": ["VALID"],
                    "known_false_accept": False,
                    "exact_authority_decision": "accept",
                },
                {
                    "row_id": "row-invalid",
                    "source_artifact": "results/exact_rows.json",
                    "exact_label": "INVALID",
                    "candidate_answers": ["INVALID", "VALID"],
                    "known_false_accept": True,
                    "exact_authority_decision": "reject",
                },
            ],
            "factor_graph_translation_records": [
                {
                    "row_id": "row-valid",
                    "thrml_mapping": {"construction_check": "passed"},
                },
                {
                    "row_id": "row-invalid",
                    "thrml_mapping": {"construction_check": "passed"},
                },
            ],
            "inference_substrate": {
                "executes_hardware": False,
                "hardware_commands_run": [],
                "sampler_speedup_reported": False,
            },
        },
    )
    _write_json(
        root,
        mod.EXP3197_REL_PATH,
        {
            "schema_version": "carnot.exverus_inductive_certificate_expansion.v1",
            "invariant_records": [
                {
                    "record_id": "inv-row-invalid",
                    "row_id": "row-invalid",
                    "source_artifact": "exp3180.exact_rows_evaluated",
                    "exact_label": "INVALID",
                    "row_family": "known_false_accept:arithmetic_code_assertions",
                    "observed_counterexample": {
                        "candidate_answers": ["INVALID", "VALID"],
                        "canonical_answer": "INVALID",
                        "certificate_type": "ast_execution",
                    },
                    "generalized_invariant": {
                        "statement": "claimed_value == computed_value",
                        "unsat_core": ["computed_value", "claimed_value"],
                    },
                    "exact_guard": {
                        "guard_id": "guard-row-invalid",
                        "required_exact_label": "INVALID",
                        "canonical_answer": "INVALID",
                        "preview_candidate_domain": ["INVALID"],
                    },
                    "anti_overfit_test": {"test_id": "anti-row-invalid"},
                },
                {
                    "record_id": "inv-json",
                    "row_id": "row-json",
                    "source_artifact": "exp3180.exact_rows_evaluated",
                    "exact_label": "REPAIRABLE",
                    "row_family": "fragment_code:parser_repair",
                    "observed_counterexample": {
                        "candidate_answers": ["REPAIRABLE"],
                        "canonical_answer": "REPAIRABLE",
                        "certificate_type": "solver_mcs",
                    },
                    "generalized_invariant": {
                        "statement": "JSON parses after delimiter insertion",
                        "unsat_core": [],
                    },
                    "exact_guard": {
                        "guard_id": "guard-json",
                        "required_exact_label": "REPAIRABLE",
                        "canonical_answer": "REPAIRABLE",
                        "preview_candidate_domain": ["REPAIRABLE"],
                    },
                    "anti_overfit_test": {"test_id": "anti-json"},
                },
            ],
        },
    )


def test_req_hw_100_spec_anchor_exists() -> None:
    """REQ-HW-100: OpenSpec declares the sparse Potts/PAOA boundary artifact."""
    spec = (mod.REPO_ROOT / mod.SPEC_REL_PATH).read_text(encoding="utf-8")

    assert "REQ-HW-100" in spec
    assert "SCENARIO-HW-100" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "hardware_claims_denied" in spec


def test_scenario_hw_100_builds_sparse_factor_boundary_without_speedup_claims(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-100: exact rows and invariants become sparse q-state records."""
    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, tests_run=("pytest targeted",))

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3202"
    assert artifact["factor_record_count"] == 4
    assert len(artifact["factor_records"]) == 4
    assert artifact["factor_record_schema"]["required"] == [
        "record_id",
        "source_kind",
        "row_id",
        "q_state_count",
        "state_labels",
        "variables",
        "sparse_scope",
        "coupling_entries",
        "paoa_metadata",
    ]
    assert artifact["paoa_metadata_schema"]["properties"]["coupling_format"] == (
        "sparse categorical energy triplets"
    )
    assert artifact["q_state_count_summary"]["record_count"] == 4
    assert artifact["q_state_count_summary"]["max_q_state_count"] == 2
    assert artifact["q_state_count_summary"]["unique_state_labels"] == [
        "INVALID",
        "REPAIRABLE",
        "VALID",
    ]
    assert artifact["graph_density_summary"]["factor_record_count"] == 4
    assert artifact["graph_density_summary"]["dense_energy_slot_count"] > (
        artifact["graph_density_summary"]["sparse_nonzero_energy_entry_count"]
    )
    assert artifact["graph_density_summary"]["sparse_vs_dense_slot_delta"] > 0
    assert artifact["thrml_local_api_checked"] is True
    assert artifact["authenticated_hardware_transcript_present"] is False
    assert artifact["speedup_claim_allowed"] is False
    denied = {row["claim"] for row in artifact["hardware_claims_denied"]}
    assert denied == set(mod.DENIED_HARDWARE_CLAIMS)
    assert all(row["denied"] is True for row in artifact["hardware_claims_denied"])
    assert artifact["inference_substrate"]["hardware_commands_run"] == []
    assert artifact["inference_substrate"]["executes_hardware"] is False
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_hw_100_factor_records_preserve_lineage_and_paoa_metadata(tmp_path: Path) -> None:
    """REQ-HW-100: factor records carry exact/invariant lineage and PAOA metadata."""
    _write_common_sources(tmp_path)

    records = mod.build_factor_records(
        mod.read_json_object(tmp_path / mod.EXP3188_REL_PATH),
        mod.read_json_object(tmp_path / mod.EXP3197_REL_PATH),
    )

    exact_record = next(record for record in records if record["record_id"] == "potts-exact:row-invalid")
    invariant_record = next(
        record for record in records if record["record_id"] == "potts-invariant:inv-row-invalid"
    )

    assert exact_record["source_kind"] == "exact_row"
    assert exact_record["q_state_count"] == 2
    assert exact_record["state_labels"] == ["INVALID", "VALID"]
    assert {
        "source_variable": exact_record["coupling_entries"][0]["source_variable"],
        "target_variable": exact_record["coupling_entries"][0]["target_variable"],
    } == {"source_variable": "candidate_label", "target_variable": "exact_label"}
    assert exact_record["paoa_metadata"]["boundary_only"] is True
    assert exact_record["paoa_metadata"]["coupling_format"] == "sparse_categorical_triplets"
    assert invariant_record["source_kind"] == "invariant_certificate"
    assert invariant_record["lineage"]["invariant_record_id"] == "inv-row-invalid"
    assert invariant_record["invariant_certificate"]["guard_id"] == "guard-row-invalid"
    assert invariant_record["paoa_metadata"]["constraint_family"] == (
        "known_false_accept:arithmetic_code_assertions"
    )


def test_req_hw_100_missing_sources_block_without_fabricating_claims(tmp_path: Path) -> None:
    """REQ-HW-100: missing source artifacts produce a blocked artifact and no claims."""
    _write_common_sources(tmp_path)
    (tmp_path / mod.EXP3197_REL_PATH).unlink()

    artifact = mod.build_artifact(tmp_path)

    assert artifact["factor_record_count"] == 0
    assert artifact["source_errors"] == [
        {
            "path": mod.EXP3197_REL_PATH.as_posix(),
            "reason": "missing_required_source",
        }
    ]
    assert artifact["authenticated_hardware_transcript_present"] is False
    assert artifact["speedup_claim_allowed"] is False
    assert artifact["honest_verdict"].startswith("blocked_missing_source:")


def test_req_hw_100_hardware_transcript_detection_controls_speedup_gate(tmp_path: Path) -> None:
    """REQ-HW-100: speedup stays denied unless authenticated transcript evidence exists."""
    _write_common_sources(tmp_path)
    exp3188 = mod.read_json_object(tmp_path / mod.EXP3188_REL_PATH)
    exp3197 = mod.read_json_object(tmp_path / mod.EXP3197_REL_PATH)

    assert mod.authenticated_hardware_transcript_present(exp3188, exp3197) is False
    exp3188["authenticated_hardware_transcript"] = {
        "present": True,
        "sha256": "a" * 64,
        "device": "KV260",
    }
    assert mod.authenticated_hardware_transcript_present(exp3188, exp3197) is True


def test_req_hw_100_writer_and_validation_are_deterministic(tmp_path: Path) -> None:
    """REQ-HW-100: writer, JSON helper, and schema validation fail closed."""
    _write_common_sources(tmp_path)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")

    output = mod.write_artifact(tmp_path, tests_run=("pytest targeted",))
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["factor_record_count"] == 4
    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(scalar_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.source_errors(
        [
            {
                "path": "bad.json",
                "required": True,
                "present": True,
                "readable_structured_source": False,
                "source_type": "json",
            }
        ]
    ) == [{"path": "bad.json", "reason": "malformed_required_source"}]
    assert mod.honest_verdict([], [], False).startswith("blocked_empty_scope:")
    try:
        mod.validate_artifact({})
    except ValueError as exc:
        assert "missing required Exp 3202 artifact fields" in str(exc)
    else:  # pragma: no cover - assertion guard.
        raise AssertionError("validate_artifact should reject missing fields")
