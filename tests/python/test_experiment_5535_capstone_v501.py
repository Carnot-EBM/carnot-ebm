"""Tests for Exp5535 V501 capstone reconciliation.

Spec refs: REQ-REPORT-5535, SCENARIO-REPORT-5535,
SCENARIO-REPORT-5535-MISSING-INPUT, SCENARIO-REPORT-5535-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5535_capstone_v501 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: JsonDict) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _context(root: Path, *, include_next: bool = False) -> None:
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        if rel_path == mod.ROADMAP_NEXT_RELATIVE_PATH and not include_next:
            continue
        if rel_path == mod.ROADMAP_RELATIVE_PATH:
            _write_text(
                root,
                rel_path,
                yaml.safe_dump(
                    {"milestone": mod.MILESTONE, "tasks": [{"id": "exp5535-v501-capstone"}]},
                    sort_keys=False,
                ),
            )
        else:
            _write_text(root, rel_path)
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH)


def _artifact_payloads() -> dict[Path, JsonDict]:
    return {
        Path("results/experiment_5523_transition_v501.json"): {
            "honest_verdict": "complete: transition ready",
            "milestone": mod.MILESTONE,
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        Path("results/experiment_5524_v501_source_delta_ingestion.json"): {
            "honest_verdict": "complete: no new source deltas",
            "milestone": mod.MILESTONE,
            "research_references_updated": False,
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        Path("results/experiment_5525_sota_schema_failure_taxonomy.json"): {
            "honest_verdict": "complete: taxonomy ready",
            "milestone": mod.MILESTONE,
            "sota_schema_failure_taxonomy_ready": True,
            "schema_validity_rate": 0.0,
            "exact_validator_handoff_ready": False,
        },
        Path("results/experiment_5526_sota_structured_repair_loop.json"): {
            "honest_verdict": "complete: repair loop ready",
            "milestone": mod.MILESTONE,
            "sota_structured_repair_loop_ready": True,
            "exact_validator_handoff_ready": True,
            "schema_validity_after": 1.0,
            "missing_candidate_rows_after": 0,
        },
        Path("results/experiment_5527_sota_hard_soft_panel_v2.json"): {
            "honest_verdict": "complete: panel says claim allowed but is flagged",
            "milestone": mod.MILESTONE,
            "sota_hard_soft_claim_allowed": True,
            "sota_structured_panel_ready": True,
            "exact_validator_accuracy": 1.0,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
        Path("results/experiment_5528_csl_canonical_gate_artifact.json"): {
            "honest_verdict": "complete: canonical gate visible",
            "milestone": mod.MILESTONE,
            "continuous_self_learning_evidence": True,
            "csl_gate_fields_conductor_visible": True,
            "conductor_gate_probe_passed": True,
            "heldout_delta": 1.0,
        },
        Path("results/experiment_5529_csl_event_topic_residue_stress.json"): {
            "honest_verdict": "complete: residue stress says ready but is flagged",
            "milestone": mod.MILESTONE,
            "csl_residue_stress_ready": True,
            "heldout_delta": 1.0,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
        Path("results/experiment_5530_sota_csl_memory_panel_v2.json"): {
            "honest_verdict": "complete: bounded csl panel claim allowed",
            "milestone": mod.MILESTONE,
            "continuous_self_learning_evidence": True,
            "csl_claim_allowed": True,
            "heldout_delta": 0.6666666667,
            "negative_transfer_rate": 0.0,
            "stale_evidence_rejection_rate": 1.0,
            "upstream_gate_evidence": {
                "exp5529": {
                    "flagged_adversarial": True,
                    "path": "results/experiment_5529_csl_event_topic_residue_stress.json",
                }
            },
        },
        Path("results/experiment_5531_sparse_repair_scaleup_ci.json"): {
            "honest_verdict": "complete: sparse scale ready no speedup",
            "milestone": mod.MILESTONE,
            "active_constraint_sparse_repair_ready": True,
            "sparse_repair_success_rate": 1.0,
            "matched_timing_available": False,
            "speedup_claim_allowed": False,
        },
        Path("results/experiment_5532_hardware_receipt_parser_repeatability.json"): {
            "honest_verdict": "complete: receipts no speedup but flagged",
            "milestone": mod.MILESTONE,
            "hardware_speedup_claim": False,
            "hardware_speedup_claim_allowed": False,
            "matched_timing_available": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
        Path("results/experiment_5533_arc_strategy_routing_precheck.json"): {
            "honest_verdict": "complete: strategy precheck ready but flagged",
            "milestone": mod.MILESTONE,
            "selected_game": "g50t",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "arc_sge_candidate_ready": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
        Path("results/experiment_5534_arc_strategy_routed_levelup.json"): {
            "honest_verdict": "honest_null: no target level reproduction",
            "milestone": mod.MILESTONE,
            "selected_game": "g50t",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "registry_delta": 0,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
        Path("results/experiment_5534_arc_strategy_routed_levelup_trajectory.json"): {
            "schema": "trajectory",
            "selected_game": "g50t",
        },
    }


def _make_root(root: Path, *, omit: Path | None = None) -> None:
    _context(root)
    for rel_path, payload in _artifact_payloads().items():
        if rel_path != omit:
            _write_json(root, rel_path, payload)


def test_req_report_5535_spec_declares_capstone_contract() -> None:
    """REQ-REPORT-5535: OpenSpec declares the V501 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5535") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for rel_path in mod.PRIMARY_ARTIFACT_PATHS:
        assert rel_path.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5535_aggregates_clean_claims_and_skips_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5535: flagged positives stay visible but unpromoted."""

    _make_root(tmp_path)

    artifact = mod.run_capstone(
        root=tmp_path,
        commands_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["artifact_paths_read"] == [
        rel_path.as_posix()
        for rel_path in (*mod.PRIMARY_ARTIFACT_PATHS, *mod.AUXILIARY_ARTIFACT_PATHS)
    ]
    assert artifact["missing_artifacts"] == [mod.ROADMAP_NEXT_RELATIVE_PATH.as_posix()]
    assert [row["artifact_path"] for row in artifact["skipped_by_gates"]] == [
        "results/experiment_5527_sota_hard_soft_panel_v2.json",
        "results/experiment_5529_csl_event_topic_residue_stress.json",
        "results/experiment_5532_hardware_receipt_parser_repeatability.json",
        "results/experiment_5533_arc_strategy_routing_precheck.json",
        "results/experiment_5534_arc_strategy_routed_levelup.json",
    ]
    assert artifact["structured_sota_claim_allowed"] is True
    assert artifact["sota_hard_soft_claim_allowed"] is False
    assert artifact["continuous_self_learning_evidence"] is True
    assert artifact["csl_claim_allowed"] is False
    assert artifact["sparse_repair_claim_allowed"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["arc_registry_delta"] == 0
    assert artifact["reproduced_levels"] == 0
    assert artifact["roadmap_yaml_unchanged"] is True
    assert artifact["conductor_unchanged"] is True
    assert artifact["docs_updated"] == ["openspec/capabilities/research-reporting/spec.md"]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["solve_provenance_summary"][-1]["flagged_adversarial"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5535_missing_primary_blocks_without_fabrication(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5535-MISSING-INPUT: missing primary inputs fail closed."""

    missing = Path("results/experiment_5526_sota_structured_repair_loop.json")
    _make_root(tmp_path, omit=missing)

    artifact = mod.run_capstone(
        root=tmp_path,
        commands_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("blocked:")
    assert missing.as_posix() in artifact["missing_artifacts"]
    assert artifact["structured_sota_claim_allowed"] is False
    assert artifact["sota_hard_soft_claim_allowed"] is False
    assert "missing_artifacts" not in mod.validate_artifact(artifact)


def test_scenario_report_5535_clean_arc_numbers_are_imported(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5535: clean ARC live-path rows can carry registry deltas."""

    _make_root(tmp_path)
    arc_path = Path("results/experiment_5534_arc_strategy_routed_levelup.json")
    arc_payload = _artifact_payloads()[arc_path]
    arc_payload.pop("flagged_adversarial")
    arc_payload.pop("corrigendum_pending")
    arc_payload["offline_reproduced"] = True
    arc_payload["reproduced_levels"] = 1
    arc_payload["registry_delta"] = 1
    _write_json(tmp_path, arc_path, arc_payload)

    artifact = mod.run_capstone(
        root=tmp_path,
        commands_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["arc_registry_delta"] == 1
    assert artifact["reproduced_levels"] == 1


def test_scenario_report_5535_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5535-FIELD-PRINCIPLES: malformed capstones are rejected."""

    _make_root(tmp_path)
    artifact = mod.run_capstone(
        root=tmp_path,
        commands_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert "sota_hard_soft_claim_allowed" in mod.validate_artifact(
        {**artifact, "sota_hard_soft_claim_allowed": "false"}
    )
    assert "arc_registry_delta" in mod.validate_artifact({**artifact, "arc_registry_delta": "0"})
    assert "commands_run" in mod.validate_artifact({**artifact, "commands_run": "unit"})
    assert "hardware_speedup_claim" in mod.validate_artifact(
        {**artifact, "hardware_speedup_claim": True}
    )
    assert "field_principles" in mod.validate_artifact(
        {**artifact, "field_principles": {"milestone": mod.FIELD_PRINCIPLES["milestone"]}}
    )
    assert "inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "aggregation_from_upstream_artifacts"}
    )
    assert "milestone" in mod.validate_artifact({**artifact, "milestone": "2026.07.500"})
    assert "honest_verdict" in mod.validate_artifact({**artifact, "honest_verdict": "maybe"})
    assert "schema" in mod.validate_artifact({k: v for k, v in artifact.items() if k != "schema"})
    assert mod._has_flagged_upstream({}) is False


def test_scenario_report_5535_write_helper_emits_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5535: writer persists the validated deliverable."""

    _make_root(tmp_path)

    artifact = mod.write_capstone(
        root=tmp_path,
        commands_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert mod.validate_artifact(written) == []
