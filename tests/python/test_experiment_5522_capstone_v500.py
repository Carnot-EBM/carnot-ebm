"""Tests for REQ-CAPSTONE-5522 / SCENARIO-CAPSTONE-5522."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5522_capstone_v500 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: JsonDict) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _primary_artifacts() -> dict[Path, JsonDict]:
    return {
        Path("results/experiment_5510_transition_v500.json"): {
            "honest_verdict": "complete: transition ready",
            "source_context_missing": ["research-roadmap-next.yaml"],
        },
        Path("results/experiment_5511_v500_source_delta_ingestion.json"): {
            "honest_verdict": "complete: no new actionable deltas",
            "research_references_updated": False,
        },
        Path("results/experiment_5512_structured_output_positive_control.json"): {
            "honest_verdict": "complete: structured_output_positive_control_ready",
            "structured_output_positive_control_ready": True,
        },
        Path("results/experiment_5513_sota_hard_soft_structured_panel.json"): {
            "honest_verdict": "blocked: sota panel not ready",
            "structured_positive_control_ready": True,
            "sota_rows_emitted": 1,
            "sota_structured_panel_ready": False,
            "candidate_rows": [{"parseable": False, "schema_valid": False}],
        },
        Path("results/experiment_5514_energy_spill_sidecar_diagnostic.json"): {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 2 gate(s) failed",
            "gates_evaluated": [
                {
                    "upstream": "exp5513-sota-hard-soft-structured-panel",
                    "artifact_field": "sota_structured_panel_ready",
                    "expected": True,
                    "actual": False,
                    "passed": False,
                },
                {
                    "upstream": "exp5513-sota-hard-soft-structured-panel",
                    "artifact_field": "sota_rows_emitted",
                    "expected": 0,
                    "actual": 1,
                    "passed": True,
                },
            ],
        },
        Path("results/experiment_5515_csl_independent_outcome_gate_repair.json"): {
            "honest_verdict": "complete: independent outcome graph memory ready",
            "metric_independence_clean": True,
            "csl_experience_graph_ready": True,
            "csl_gate_fields_resolvable": True,
            "continuous_self_learning_evidence": True,
            "heldout_delta": 1.0,
            "negative_transfer_rate": 0.0,
            "stale_evidence_rejection_rate": 1.0,
        },
        Path("results/experiment_5516_sota_csl_memory_panel.json"): {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "2 of 2 gate(s) failed",
            "gates_evaluated": [
                {
                    "upstream": "exp5515-csl-independent-outcome-gate-repair",
                    "artifact_field": "metric_independence_clean",
                    "expected": True,
                    "actual": None,
                    "passed": False,
                },
                {
                    "upstream": "exp5515-csl-independent-outcome-gate-repair",
                    "artifact_field": "csl_gate_fields_resolvable",
                    "expected": True,
                    "actual": None,
                    "passed": False,
                },
            ],
        },
        Path("results/experiment_5517_csl_memory_residue_stress.json"): {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 1 gate(s) failed",
            "gates_evaluated": [
                {
                    "upstream": "exp5515-csl-independent-outcome-gate-repair",
                    "artifact_field": "csl_experience_graph_ready",
                    "expected": True,
                    "actual": None,
                    "passed": False,
                }
            ],
        },
        Path("results/experiment_5518_block_gibbs_sparse_repair_descriptors.json"): {
            "honest_verdict": "complete: exact checked sparse repair ready",
            "active_constraint_sparse_repair_ready": True,
            "all_candidates_exact_checked": True,
            "exact_fallback_used": True,
            "sparse_repair_success_rate": 1.0,
            "speedup_claim_allowed": False,
            "readiness_blockers": [],
        },
        Path("results/experiment_5519_hardware_continuity_methodology_receipts.json"): {
            "honest_verdict": "complete: receipts no speedup",
            "matched_timing_available": False,
            "hardware_speedup_claim": False,
            "hardware_speedup_claim_allowed": False,
            "blocked_devices": [
                {"device": "kv260", "status": "blocked_identity"},
                {"device": "gatemate", "status": "blocked_identity"},
            ],
        },
        Path("results/experiment_5520_arc_action_diversity_target_precheck.json"): {
            "honest_verdict": "complete: sb26 L3 precheck ready",
            "arc_levelup_candidate_ready": True,
            "selected_game": "sb26",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "action_entropy": 3.0,
            "repeated_coordinate_rate": 0.0,
        },
        Path("results/experiment_5521_arc_live_action_diverse_levelup.json"): {
            "honest_verdict": "honest_null: sb26 L3 bounded budget no reproduction",
            "arc_live_levelup_ready": True,
            "selected_game": "sb26",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "banking_gate": False,
            "registry_delta": 0,
            "live_attempts": 47,
            "action_entropy": 2.48,
            "repeated_coordinate_rate": 0.526,
            "trajectory_log_path": "results/experiment_5521_arc_live_action_diverse_levelup_trajectory.json",
        },
    }


def _make_root(root: Path, *, omit: Path | None = None) -> None:
    for rel_path, payload in _primary_artifacts().items():
        if rel_path != omit:
            _write_json(root, rel_path, payload)
    _write_json(
        root,
        "results/experiment_5515_csl_independent_outcome_stream_fixture.json",
        {"schema": "fixture", "heldout_labels": ["independent"]},
    )
    _write_json(
        root,
        "results/experiment_5521_arc_live_action_diverse_levelup_trajectory.json",
        {"schema": "trajectory", "selected_game": "sb26"},
    )
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        if rel_path.name == "research-roadmap-next.yaml":
            continue
        _write_text(root, rel_path)


def test_req_capstone_5522_spec_declares_v500_reconciliation_contract() -> None:
    """REQ-CAPSTONE-5522: OpenSpec declares the capstone artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5522") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for rel_path in mod.PRIMARY_ARTIFACT_PATHS:
        assert rel_path.as_posix() in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_capstone_5522_default_gates_preserve_bounded_claims(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5522: .500 gates are aggregated without overclaiming."""

    _make_root(tmp_path)

    artifact = mod.run_capstone(
        root=tmp_path,
        commands_run=["unit"],
        modification_overrides={mod.CONDUCTOR_RELATIVE_PATH: False},
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["structured_sota_claim_allowed"] is False
    assert artifact["energy_sidecar_headline_allowed"] is False
    assert artifact["continuous_self_learning_evidence"] is True
    assert artifact["csl_claim_allowed"] is False
    assert artifact["sparse_repair_claim_allowed"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["arc_registry_delta"] == 0
    assert artifact["reproduced_levels"] == 0
    assert artifact["missing_artifacts"] == []
    assert [row["artifact_path"] for row in artifact["skipped_by_gates"]] == [
        "results/experiment_5514_energy_spill_sidecar_diagnostic.json",
        "results/experiment_5516_sota_csl_memory_panel.json",
        "results/experiment_5517_csl_memory_residue_stress.json",
    ]
    assert artifact["solve_provenance_summary"] == [
        {
            "artifact_path": "results/experiment_5520_arc_action_diversity_target_precheck.json",
            "selected_game": "sb26",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "registry_delta": None,
            "reproduced_levels": None,
            "honest_verdict": "complete: sb26 L3 precheck ready",
        },
        {
            "artifact_path": "results/experiment_5521_arc_live_action_diverse_levelup.json",
            "selected_game": "sb26",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "registry_delta": 0,
            "reproduced_levels": 0,
            "honest_verdict": "honest_null: sb26 L3 bounded budget no reproduction",
        },
    ]
    assert artifact["docs_updated"] == ["openspec/capabilities/capstone/spec.md"]
    assert artifact["conductor_unchanged"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5522_missing_primary_blocks_without_fabrication(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5522-MISSING-SKIPPED-GATES: missing inputs fail closed."""

    missing = Path("results/experiment_5513_sota_hard_soft_structured_panel.json")
    _make_root(tmp_path, omit=missing)

    artifact = mod.run_capstone(
        root=tmp_path,
        commands_run=["unit"],
        modification_overrides={mod.CONDUCTOR_RELATIVE_PATH: False},
    )

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["missing_artifacts"] == [missing.as_posix()]
    assert artifact["structured_sota_claim_allowed"] is False
    assert "missing_artifacts" not in mod.validate_artifact(artifact)


def test_scenario_capstone_5522_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5522-FIELD-PRINCIPLES: malformed capstones are rejected."""

    _make_root(tmp_path)
    artifact = mod.run_capstone(
        root=tmp_path,
        commands_run=["unit"],
        modification_overrides={mod.CONDUCTOR_RELATIVE_PATH: False},
    )

    assert "structured_sota_claim_allowed" in mod.validate_artifact(
        {**artifact, "structured_sota_claim_allowed": "false"}
    )
    assert "arc_registry_delta" in mod.validate_artifact({**artifact, "arc_registry_delta": "0"})
    assert "commands_run" in mod.validate_artifact({**artifact, "commands_run": "unit"})
    assert "schema" in mod.validate_artifact({k: v for k, v in artifact.items() if k != "schema"})
    assert "milestone" in mod.validate_artifact({**artifact, "milestone": "2026.07.499"})
    assert "inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "aggregation_from_upstream_artifacts"}
    )
    assert "honest_verdict" in mod.validate_artifact({**artifact, "honest_verdict": "maybe"})


def test_scenario_capstone_5522_write_helper_emits_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5522: writer persists the validated deliverable."""

    _make_root(tmp_path)

    artifact = mod.write_capstone(root=tmp_path, commands_run=["unit"])

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert mod.validate_artifact(written) == []
