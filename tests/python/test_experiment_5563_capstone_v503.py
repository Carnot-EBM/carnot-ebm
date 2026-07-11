"""Tests for Exp5563 V503 capstone reconciliation.

Spec refs: REQ-REPORT-5563, SCENARIO-REPORT-5563,
SCENARIO-REPORT-5563-MISSING-INPUT, SCENARIO-REPORT-5563-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5563_capstone_v503 as mod


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


def _artifact_payloads(*, include_panel: bool = False) -> dict[Path, JsonDict]:
    payloads: dict[Path, JsonDict] = {
        Path("results/experiment_5550_transition_v503.json"): {
            "milestone": mod.MILESTONE,
            "status": "complete",
            "honest_verdict": "complete: transition",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "roadmap_yaml_unchanged": True,
            "conductor_unchanged": True,
        },
        Path("results/experiment_5551_v503_source_delta_ingestion.json"): {
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: source delta",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        Path("results/experiment_5552_automaton_schema_row_completion_receipt.json"): {
            "milestone": mod.MILESTONE,
            "honest_verdict": "blocked: automaton row completion failed",
            "inference_substrate": "deterministic_automaton_no_llm",
            "automaton_row_completion_ready": False,
        },
        Path("results/experiment_5553_gated_gbnf_forced_sota_row_smoke.json"): {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "schema": "blocked_gate_check_v1",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "exp5552 automaton_row_completion_ready failed",
            "gates_evaluated": [{"passed": False}],
        },
        Path("results/experiment_5555_asp_fsm_nonmonotonic_fixture.json"): {
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: exact ASP/FSM fixture ready",
            "inference_substrate": "deterministic_asp_fsm_exact_fixture_no_llm",
            "exact_asp_validator_ready": True,
            "exact_fsm_fixture_extended_ready": True,
        },
        Path("results/experiment_5556_asp_fsm_sparse_repair_scale.json"): {
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: asp sparse repair ready no speedup",
            "inference_substrate": "deterministic_asp_fsm_sparse_repair_no_llm",
            "asp_sparse_repair_claim_allowed": True,
            "exact_asp_validator_ready": True,
            "stable_model_checked_rate": 1.0,
            "unchecked_repair_count": 0,
            "matched_timing_available": False,
            "speedup_claim_allowed": False,
        },
        Path("results/experiment_5557_csl_five_arm_tautology_corrigendum_v2.json"): {
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: csl five arm clean",
            "inference_substrate": "deterministic_csl_ablation_no_llm",
            "csl_five_arm_clean": True,
            "adversarial_clean": True,
            "tautology_resolved": True,
            "aligned_delta_over_shuffled": 0.5,
            "duplicated_metric_pairs": [],
        },
        Path("results/experiment_5558_causal_write_manage_read_csl_memory.json"): {
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: causal csl memory ready",
            "inference_substrate": "deterministic_online_memory_fixture_no_llm",
            "csl_memory_ready": True,
            "csl_claim_allowed": True,
            "no_weight_mutation": True,
            "quality_delta_vs_shuffled_memory": 1.0,
            "action_impact_delta_vs_no_memory": 0.5,
            "action_selection_changed_count": 3,
        },
        Path("results/experiment_5559_cross_model_sota_csl_transfer_v2.json"): {
            "milestone": mod.MILESTONE,
            "honest_verdict": "blocked: cross model transfer zero delta",
            "inference_substrate": "live_local_sota_gguf_cross_model_csl_transfer_or_gate_skip",
            "flagged_adversarial": True,
            "csl_claim_allowed": False,
            "cross_family_delta_over_shuffled": 0.0,
            "negative_transfer_rate": 0.8,
            "no_weight_mutation": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
        },
        Path("results/experiment_5560_hardware_and_timing_receipt_hygiene.json"): {
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: hardware hygiene no speedup",
            "inference_substrate": "hardware_receipt_and_timing_hygiene_no_llm",
            "hardware_speedup_claim": False,
            "matched_timing_available": False,
            "conductor_modified": False,
            "roadmap_yaml_unchanged": True,
        },
        Path("results/experiment_5561_arc_fsm_target_rotation_precheck.json"): {
            "milestone": mod.MILESTONE,
            "status": "complete",
            "honest_verdict": "complete: arc target ready",
            "inference_substrate": "arc_live_path_precheck_no_llm",
            "solve_provenance": "live_agent_self_discovery",
            "selected_game": "r11l",
            "selected_level": "L3",
        },
        Path("results/experiment_5562_arc_fsm_live_levelup.json"): {
            "milestone": mod.MILESTONE,
            "status": "honest_null",
            "honest_verdict": "honest_null: no target level reproduction",
            "inference_substrate": "arc_live_agent_self_discovery_no_llm",
            "solve_provenance": "live_agent_self_discovery",
            "offline_reproduced": False,
            "registry_delta": 0,
            "reproduced_levels": 0,
        },
        Path("results/experiment_5562_arc_fsm_live_levelup_trajectory.json"): {
            "experiment": "experiment_5562_arc_fsm_live_levelup",
            "schema": "trajectory",
            "selected_game": "r11l",
            "selected_level": "L3",
        },
    }
    if include_panel:
        payloads[Path("results/experiment_5554_sota_hard_soft_panel_v4.json")] = {
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: hard soft panel v4 claim allowed",
            "inference_substrate": "live_local_sota_gguf_exact_validated_panel",
            "gates_clean": True,
            "adversarial_clean": True,
            "rows_requested": 6,
            "rows_emitted": 6,
            "schema_valid_rows": 6,
            "missing_candidate_rows": 0,
            "exact_validator_accuracy": 1.0,
            "sota_hard_soft_claim_allowed": True,
        }
    return payloads


def _context(root: Path) -> None:
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        if rel_path == mod.CONDUCTOR_LOG_RELATIVE_PATH:
            continue
        _write_text(root, rel_path)
    _write_text(
        root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        "| 2026-07-10 23:28 UTC | Gated SOTA hard-soft panel v4 | GATE_BLOCK | "
        "Pre-emptive skip: upstream retired (exp5553-gated-gbnf-forced-sota-row-smoke) |\n",
    )


def _make_root(root: Path, *, omit: Path | None = None, include_panel: bool = False) -> None:
    _context(root)
    for rel_path, payload in _artifact_payloads(include_panel=include_panel).items():
        if rel_path != omit:
            _write_json(root, rel_path, payload)


def test_req_report_5563_spec_declares_capstone_contract() -> None:
    """REQ-REPORT-5563: OpenSpec anchors the V503 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5563") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5563_classifies_and_bounds_live_repo_artifacts() -> None:
    """SCENARIO-REPORT-5563: actual V503 flags and nulls are not promoted."""

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
    assert artifact["artifacts_read"] == len(mod.EXPECTED_ARTIFACT_PATHS) - 1
    assert artifact["missing_artifacts"] == []
    assert [row["artifact_path"] for row in artifact["flagged_artifacts"]] == [
        "results/experiment_5559_cross_model_sota_csl_transfer_v2.json"
    ]
    assert {
        row["artifact_path"] for row in artifact["blocked_artifacts"]
    } == {
        "results/experiment_5552_automaton_schema_row_completion_receipt.json",
        "results/experiment_5559_cross_model_sota_csl_transfer_v2.json",
    }
    assert [row["artifact_path"] for row in artifact["skipped_by_gates"]] == [
        "results/experiment_5553_gated_gbnf_forced_sota_row_smoke.json",
        "results/experiment_5554_sota_hard_soft_panel_v4.json",
    ]
    assert [row["artifact_path"] for row in artifact["honest_nulls"]] == [
        "results/experiment_5562_arc_fsm_live_levelup.json"
    ]
    assert "results/experiment_5556_asp_fsm_sparse_repair_scale.json" in (
        row["artifact_path"] for row in artifact["clean_artifacts"]
    )
    assert artifact["structured_sota_claim_allowed"] is False
    assert artifact["sota_hard_soft_claim_allowed"] is False
    assert artifact["continuous_self_learning_evidence"] is True
    assert artifact["csl_claim_allowed"] is False
    assert artifact["cross_model_csl_claim_allowed"] is False
    assert artifact["asp_sparse_repair_claim_allowed"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["arc_live_levelup_claim_allowed"] is False
    assert artifact["arc_registry_delta"] == 0
    assert artifact["docs_updated"] == ["openspec/capabilities/research-reporting/spec.md"]
    assert artifact["roadmap_yaml_unchanged"] is True
    assert artifact["conductor_unchanged"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5563_missing_primary_blocks_without_fabrication(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5563-MISSING-INPUT: missing inputs fail closed."""

    missing = Path("results/experiment_5558_causal_write_manage_read_csl_memory.json")
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
    assert artifact["missing_artifacts"] == [missing.as_posix()]
    assert artifact["artifacts_read"] == len(mod.EXPECTED_ARTIFACT_PATHS) - 2
    assert artifact["continuous_self_learning_evidence"] is False
    assert artifact["csl_claim_allowed"] is False
    assert artifact["cross_model_csl_claim_allowed"] is False
    assert artifact["hardware_speedup_claim"] is False


def test_scenario_report_5563_clean_positive_claims_require_all_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5563: clean upstream positives can become bounded claims."""

    _make_root(tmp_path, include_panel=True)
    row_completion = _artifact_payloads()[Path("results/experiment_5552_automaton_schema_row_completion_receipt.json")]
    row_completion.update(
        {
            "honest_verdict": "complete: automaton row completion ready",
            "automaton_row_completion_ready": True,
            "row_completion_support_rate": 1.0,
        }
    )
    _write_json(tmp_path, "results/experiment_5552_automaton_schema_row_completion_receipt.json", row_completion)
    _write_json(
        tmp_path,
        "results/experiment_5553_gated_gbnf_forced_sota_row_smoke.json",
        {
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: grammar forced rows complete",
            "inference_substrate": "live_local_sota_gguf_row_smoke",
            "grammar_forced_rows_complete": True,
            "rows_requested": 6,
            "rows_emitted": 6,
            "schema_valid_rows": 6,
        },
    )
    transfer = _artifact_payloads()[Path("results/experiment_5559_cross_model_sota_csl_transfer_v2.json")]
    transfer.pop("flagged_adversarial")
    transfer.pop("corrigendum_pending")
    transfer.update(
        {
            "honest_verdict": "complete: cross model transfer positive",
            "csl_claim_allowed": True,
            "cross_family_delta_over_shuffled": 0.25,
            "negative_transfer_rate": 0.0,
        }
    )
    _write_json(tmp_path, "results/experiment_5559_cross_model_sota_csl_transfer_v2.json", transfer)
    arc = _artifact_payloads()[Path("results/experiment_5562_arc_fsm_live_levelup.json")]
    arc.update(
        {
            "status": "complete",
            "honest_verdict": "complete: arc level reproduced",
            "offline_reproduced": True,
            "registry_delta": 1,
            "reproduced_levels": 1,
        }
    )
    _write_json(tmp_path, "results/experiment_5562_arc_fsm_live_levelup.json", arc)

    artifact = mod.run_capstone(
        root=tmp_path,
        checks_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["artifacts_read"] == len(mod.EXPECTED_ARTIFACT_PATHS)
    assert artifact["structured_sota_claim_allowed"] is True
    assert artifact["sota_hard_soft_claim_allowed"] is True
    assert artifact["continuous_self_learning_evidence"] is True
    assert artifact["csl_claim_allowed"] is True
    assert artifact["cross_model_csl_claim_allowed"] is True
    assert artifact["asp_sparse_repair_claim_allowed"] is True
    assert artifact["arc_live_levelup_claim_allowed"] is True
    assert artifact["arc_registry_delta"] == 1
    assert artifact["skipped_by_gates"] == []
    assert artifact["flagged_artifacts"] == []
    assert artifact["blocked_artifacts"] == []
    assert artifact["honest_nulls"] == []


def test_scenario_report_5563_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5563-FIELD-PRINCIPLES: malformed capstones fail validation."""

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
    assert "flagged_artifacts" in mod.validate_artifact({**artifact, "flagged_artifacts": "none"})
    assert "field_principles" in mod.validate_artifact(
        {**artifact, "field_principles": {"milestone": mod.FIELD_PRINCIPLES["milestone"]}}
    )
    assert "hardware_speedup_claim" in mod.validate_artifact(
        {**artifact, "hardware_speedup_claim": True}
    )
    assert "arc_live_levelup_claim_allowed" in mod.validate_artifact(
        {**artifact, "arc_live_levelup_claim_allowed": True, "arc_registry_delta": 0}
    )
    assert "csl_claim_allowed" in mod.validate_artifact(
        {**artifact, "csl_claim_allowed": True, "cross_model_csl_claim_allowed": False}
    )
    assert "sota_hard_soft_claim_allowed" in mod.validate_artifact(
        {**artifact, "sota_hard_soft_claim_allowed": True, "structured_sota_claim_allowed": False}
    )
    assert "roadmap_yaml_unchanged" in mod.validate_artifact({**artifact, "roadmap_yaml_unchanged": False})
    assert "conductor_unchanged" in mod.validate_artifact({**artifact, "conductor_unchanged": False})
    assert "milestone" in mod.validate_artifact({**artifact, "milestone": "2026.07.502"})
    assert "task_range" in mod.validate_artifact({**artifact, "task_range": "exp5550-exp5562"})
    assert "inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm_inference"}
    )
    assert "honest_verdict" in mod.validate_artifact({**artifact, "honest_verdict": "maybe"})
    assert "schema" in mod.validate_artifact({k: v for k, v in artifact.items() if k != "schema"})
    assert "artifacts_read" in mod.validate_artifact(
        {
            **artifact,
            "artifacts_expected": 1,
            "artifacts_read": 2,
        }
    )

    first_path = mod.EXPECTED_ARTIFACT_PATHS[0].as_posix()
    flagged, blocked, failed, skipped, honest_nulls, clean = mod.classify_artifacts(
        {first_path: {"honest_verdict": "failed: synthetic failure"}},
        {},
    )
    assert flagged == []
    assert blocked == []
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
    assert skipped == []
    assert honest_nulls == []
    assert clean == []
    assert mod.classify_artifacts(
        {first_path: {"honest_verdict": "honest_null: synthetic null"}},
        {},
    )[4][0]["null_reason"] == "clean_honest_null"
    assert mod._int({"value": True}, "value") == 1
    assert mod._int({"value": "7"}, "value") == 7
    assert mod._int({"value": "not-an-int"}, "value") == 0
    assert mod._float({"value": "1.25"}, "value") == 1.25
    assert mod._float({"value": "not-a-float"}, "value") == 0.0
    assert mod._float({"value": None}, "value") == 0.0
    assert mod._conductor_gate_skips(tmp_path / "missing-log-root", {}) == ([], set())


def test_scenario_report_5563_write_helper_emits_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5563: writer persists the validated deliverable."""

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
