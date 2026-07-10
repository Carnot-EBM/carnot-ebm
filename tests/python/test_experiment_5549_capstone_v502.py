"""Tests for Exp5549 V502 capstone reconciliation.

Spec refs: REQ-REPORT-5549, SCENARIO-REPORT-5549,
SCENARIO-REPORT-5549-MISSING-INPUT, SCENARIO-REPORT-5549-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5549_capstone_v502 as mod


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


def _model_specs() -> list[JsonDict]:
    return [
        {
            "hf_id": mod.QWEN_HF_ID,
            "model_path": "/models/qwen.gguf",
            "model_filename": "qwen.gguf",
            "local_model_present": True,
            "preferred_quant": "Q4_K_M",
        },
        {
            "hf_id": mod.GEMMA_26_HF_ID,
            "model_path": "/models/gemma26.gguf",
            "model_filename": "gemma26.gguf",
            "local_model_present": True,
            "preferred_quant": "Q4_K_M",
        },
        {
            "hf_id": mod.GEMMA_31_HF_ID,
            "model_path": "/models/gemma31.gguf",
            "model_filename": "gemma31.gguf",
            "local_model_present": True,
            "preferred_quant": "Q4_K_M",
        },
    ]


def _artifact_payloads() -> dict[Path, JsonDict]:
    return {
        Path("results/experiment_5536_transition_v502.json"): {
            "experiment": "experiment_5536_transition_v502",
            "milestone": mod.MILESTONE,
            "status": "complete",
            "honest_verdict": "complete: transition ready",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        Path("results/experiment_5537_v502_source_delta_ingestion.json"): {
            "milestone": mod.MILESTONE,
            "status": "complete",
            "honest_verdict": "complete: source delta",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        Path("results/experiment_5538_sota_panel_duration_substrate_corrigendum.json"): {
            "experiment": 5538,
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: duration corrigendum downgraded quality",
            "inference_substrate": "live_local_sota_gguf_panel_or_claim_downgrade",
            "adversarial_clean": True,
            "live_model_invoked": True,
            "sota_panel_duration_corrigendum_ready": True,
            "quality_claim_allowed": False,
            "model_specs": _model_specs(),
        },
        Path("results/experiment_5539_gram2token_grammar_table_preflight.json"): {
            "experiment": 5539,
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: grammar ready",
            "inference_substrate": "deterministic_grammar_table_preflight_no_llm",
            "grammar_table_preflight_ready": True,
            "no_model_specs_required": True,
        },
        Path("results/experiment_5540_sota_hard_soft_live_panel_v3.json"): {
            "experiment": 5540,
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: sota_hard_soft_live_panel_v3_honest_null_no_claim",
            "inference_substrate": "live_local_sota_gguf_exact_validated_panel",
            "gates_clean": True,
            "adversarial_clean": True,
            "rows_requested": 6,
            "rows_emitted": 2,
            "schema_valid_rows": 2,
            "schema_validity_rate": 0.333333,
            "exact_validator_accuracy": 1.0,
            "missing_candidate_rows": 4,
            "sota_hard_soft_claim_allowed": False,
            "model_specs": _model_specs(),
        },
        Path("results/experiment_5541_llm_fsm_exact_fixture.json"): {
            "experiment": 5541,
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: exact fsm fixture",
            "inference_substrate": "deterministic_fsm_exact_fixture_no_llm",
            "exact_fsm_fixture_ready": True,
        },
        Path("results/experiment_5542_csl_residue_metric_independence_corrigendum.json"): {
            "experiment": 5542,
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: csl residue ready",
            "inference_substrate": "deterministic_csl_residue_corrigendum_no_llm",
            "csl_residue_tautology_resolved": True,
            "nonidentical_metric_evidence": True,
            "csl_residue_stress_ready": True,
        },
        Path("results/experiment_5543_retrieval_warmed_csl_five_arm_ablation.json"): {
            "experiment": 5543,
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: five arm ready but flagged",
            "inference_substrate": "deterministic_retrieval_warmed_csl_no_llm",
            "csl_five_arm_ready": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
        Path("results/experiment_5544_cross_model_sota_csl_transfer.json"): {
            "experiment": 5544,
            "milestone": mod.MILESTONE,
            "honest_verdict": "blocked: cross model transfer not allowed",
            "inference_substrate": "live_local_sota_gguf_cross_model_csl",
            "csl_claim_allowed": False,
            "no_weight_mutation": True,
            "cross_family_delta_over_shuffled": 0.0,
            "model_specs": _model_specs(),
        },
        Path("results/experiment_5545_sparse_repair_fsm_descriptor_scale.json"): {
            "experiment": 5545,
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: sparse repair ready no speedup",
            "inference_substrate": "exact_checked_sparse_repair_fsm_no_llm",
            "sparse_repair_fsm_ready": True,
            "exact_validator_all_repairs_checked": True,
            "unchecked_repair_count": 0,
            "speedup_claim_allowed": False,
        },
        Path("results/experiment_5546_hardware_receipt_substrate_corrigendum.json"): {
            "experiment": 5546,
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: hardware receipt clean no speedup",
            "inference_substrate": "hardware_receipt_methodology_no_llm",
            "hardware_receipt_corrigendum_clean": True,
            "matched_timing_available": False,
            "hardware_speedup_claim": False,
            "no_model_specs_required": True,
        },
        Path("results/experiment_5547_arc_no_llm_substrate_precheck.json"): {
            "experiment": "experiment_5547_arc_no_llm_substrate_precheck",
            "milestone": mod.MILESTONE,
            "status": "complete",
            "honest_verdict": "complete: arc precheck ready no solve claimed",
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
            "arc_clean_precheck_ready": True,
            "selected_game": "g50t",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "no_model_specs_required": True,
        },
        Path("results/experiment_5548_arc_clean_live_levelup.json"): {
            "experiment": "experiment_5548_arc_clean_live_levelup",
            "milestone": mod.MILESTONE,
            "status": "honest_null",
            "honest_verdict": "honest_null: g50t L3 no reproduction",
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
            "arc_live_levelup_ready": True,
            "selected_game": "g50t",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "offline_reproduced": False,
            "registry_delta": 0,
            "reproduced_levels": 0,
            "no_model_specs_required": True,
        },
        Path("results/experiment_5548_arc_clean_live_levelup_trajectory.json"): {
            "experiment": "experiment_5548_arc_clean_live_levelup",
            "schema": "trajectory",
            "selected_game": "g50t",
        },
    }


def _context(root: Path) -> None:
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        if rel_path == mod.ROADMAP_NEXT_RELATIVE_PATH:
            continue
        _write_text(root, rel_path)


def _make_root(root: Path, *, omit: Path | None = None) -> None:
    _context(root)
    for rel_path, payload in _artifact_payloads().items():
        if rel_path != omit:
            _write_json(root, rel_path, payload)


def test_req_report_5549_spec_declares_capstone_contract() -> None:
    """REQ-REPORT-5549: OpenSpec anchors the V502 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5549") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for rel_path in mod.EXPECTED_ARTIFACT_PATHS:
        assert rel_path.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5549_classifies_and_bounds_live_repo_artifacts() -> None:
    """SCENARIO-REPORT-5549: actual V502 flags and nulls are not promoted."""

    artifact = mod.run_capstone(
        root=REPO,
        checks_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["task_range"] == mod.TASK_RANGE
    assert artifact["artifacts_expected"] == len(mod.EXPECTED_ARTIFACT_PATHS)
    assert artifact["artifacts_read"] == len(mod.EXPECTED_ARTIFACT_PATHS)
    assert artifact["missing_artifacts"] == []
    assert [row["artifact_path"] for row in artifact["flagged_artifacts"]] == [
        "results/experiment_5543_retrieval_warmed_csl_five_arm_ablation.json"
    ]
    assert [row["artifact_path"] for row in artifact["skipped_by_gates"]] == [
        "results/experiment_5544_cross_model_sota_csl_transfer.json"
    ]
    assert [row["artifact_path"] for row in artifact["honest_nulls"]] == [
        "results/experiment_5540_sota_hard_soft_live_panel_v3.json",
        "results/experiment_5548_arc_clean_live_levelup.json",
    ]
    assert "results/experiment_5545_sparse_repair_fsm_descriptor_scale.json" in (
        row["artifact_path"] for row in artifact["clean_artifacts"]
    )
    assert artifact["structured_sota_claim_allowed"] is False
    assert artifact["sota_hard_soft_claim_allowed"] is False
    assert artifact["continuous_self_learning_evidence"] is True
    assert artifact["csl_claim_allowed"] is False
    assert artifact["sparse_repair_claim_allowed"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["arc_live_levelup_claim_allowed"] is False
    assert artifact["arc_registry_delta"] == 0
    assert artifact["reproduced_levels"] == 0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["protected_files_unchanged"] == {
        "research-roadmap.yaml": True,
        "scripts/research_conductor.py": True,
    }
    assert artifact["docs_updated"] == ["openspec/capabilities/research-reporting/spec.md"]
    assert artifact["llm_model_spec_audit"]["all_mandated_specs_present"] is True
    assert artifact["arc_audit"]["capstone_counted_as_levelup_attempt"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5549_missing_primary_blocks_without_fabrication(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5549-MISSING-INPUT: missing inputs fail closed."""

    missing = Path("results/experiment_5540_sota_hard_soft_live_panel_v3.json")
    _make_root(tmp_path, omit=missing)

    artifact = mod.run_capstone(
        root=tmp_path,
        checks_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("blocked:")
    assert missing.as_posix() in artifact["missing_artifacts"]
    assert artifact["artifacts_read"] == len(mod.EXPECTED_ARTIFACT_PATHS) - 1
    assert artifact["structured_sota_claim_allowed"] is False
    assert artifact["sota_hard_soft_claim_allowed"] is False
    assert artifact["csl_claim_allowed"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert "missing_artifacts" not in mod.validate_artifact(artifact)


def test_scenario_report_5549_clean_positive_claims_require_all_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5549: clean upstream positives can become bounded claims."""

    _make_root(tmp_path)
    payloads = _artifact_payloads()
    sota = payloads[Path("results/experiment_5540_sota_hard_soft_live_panel_v3.json")]
    sota.update(
        {
            "honest_verdict": "complete: hard soft claim allowed",
            "rows_requested": 6,
            "rows_emitted": 6,
            "schema_valid_rows": 6,
            "schema_validity_rate": 1.0,
            "missing_candidate_rows": 0,
            "sota_hard_soft_claim_allowed": True,
        }
    )
    _write_json(tmp_path, "results/experiment_5540_sota_hard_soft_live_panel_v3.json", sota)

    five_arm = payloads[Path("results/experiment_5543_retrieval_warmed_csl_five_arm_ablation.json")]
    five_arm.pop("flagged_adversarial")
    five_arm.pop("corrigendum_pending")
    _write_json(
        tmp_path,
        "results/experiment_5543_retrieval_warmed_csl_five_arm_ablation.json",
        five_arm,
    )
    transfer = payloads[Path("results/experiment_5544_cross_model_sota_csl_transfer.json")]
    transfer.update(
        {
            "honest_verdict": "complete: cross model csl claim allowed",
            "csl_claim_allowed": True,
            "cross_family_delta_over_shuffled": 0.25,
            "heldout_delta": 0.25,
        }
    )
    _write_json(tmp_path, "results/experiment_5544_cross_model_sota_csl_transfer.json", transfer)
    arc = payloads[Path("results/experiment_5548_arc_clean_live_levelup.json")]
    arc.update(
        {
            "status": "complete",
            "honest_verdict": "complete: g50t L3 reproduced",
            "offline_reproduced": True,
            "registry_delta": 1,
            "reproduced_levels": 1,
        }
    )
    _write_json(tmp_path, "results/experiment_5548_arc_clean_live_levelup.json", arc)

    artifact = mod.run_capstone(
        root=tmp_path,
        checks_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["structured_sota_claim_allowed"] is True
    assert artifact["sota_hard_soft_claim_allowed"] is True
    assert artifact["csl_claim_allowed"] is True
    assert artifact["arc_live_levelup_claim_allowed"] is True
    assert artifact["arc_registry_delta"] == 1
    assert artifact["reproduced_levels"] == 1
    assert artifact["flagged_artifacts"] == []
    assert artifact["skipped_by_gates"] == []


def test_scenario_report_5549_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5549-FIELD-PRINCIPLES: malformed capstones fail validation."""

    _make_root(tmp_path)
    artifact = mod.run_capstone(
        root=tmp_path,
        checks_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert "structured_sota_claim_allowed" in mod.validate_artifact(
        {**artifact, "structured_sota_claim_allowed": "false"}
    )
    assert "artifacts_expected" in mod.validate_artifact({**artifact, "artifacts_expected": "14"})
    assert "missing_artifacts" in mod.validate_artifact({**artifact, "missing_artifacts": "none"})
    assert "field_principles" in mod.validate_artifact(
        {**artifact, "field_principles": {"milestone": mod.FIELD_PRINCIPLES["milestone"]}}
    )
    assert "llm_model_spec_audit" in mod.validate_artifact(
        {**artifact, "llm_model_spec_audit": {"all_mandated_specs_present": False}}
    )
    assert "protected_files_unchanged" in mod.validate_artifact(
        {**artifact, "protected_files_unchanged": []}
    )
    assert "hardware_speedup_claim" in mod.validate_artifact(
        {**artifact, "hardware_speedup_claim": True}
    )
    assert "milestone" in mod.validate_artifact({**artifact, "milestone": "2026.07.501"})
    assert "task_range" in mod.validate_artifact({**artifact, "task_range": "exp5536-exp5548"})
    assert "inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm_inference"}
    )
    assert "honest_verdict" in mod.validate_artifact({**artifact, "honest_verdict": "maybe"})
    assert "schema" in mod.validate_artifact({k: v for k, v in artifact.items() if k != "schema"})

    first_path = mod.EXPECTED_ARTIFACT_PATHS[0].as_posix()
    _, _, failed, _, _ = mod.classify_artifacts(
        {first_path: {"honest_verdict": "failed: synthetic failure"}},
        {},
    )
    assert failed == [
        {
            "artifact_path": first_path,
            "status": "failed",
            "honest_verdict": "failed: synthetic failure",
            "flagged_adversarial": False,
            "inference_substrate": None,
            "sha256": None,
            "failure_reason": "failed_terminal_status",
        }
    ]


def test_scenario_report_5549_write_helper_emits_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5549: writer persists the validated deliverable."""

    _make_root(tmp_path)

    artifact = mod.write_capstone(
        root=tmp_path,
        checks_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert mod.validate_artifact(written) == []
