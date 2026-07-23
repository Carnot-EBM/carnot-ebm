"""Tests for Exp5862 V521 capstone reconciliation.

Spec refs: REQ-REPORT-5862, SCENARIO-REPORT-5862-GATE-REPLAY,
SCENARIO-REPORT-5862-FLAGS-AND-RETIREMENTS,
SCENARIO-REPORT-5862-MODEL-AUTHORITY, SCENARIO-REPORT-5862-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5862_v521_capstone_reconciliation as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _artifact(task_id: str) -> JsonDict:
    payloads: dict[str, JsonDict] = {
        "exp5849-transition-v521": {
            "status": "complete",
            "honest_verdict": "complete: archived .520 into .521",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "next_range_collision_count": 0,
            "research_complete_append_count": 0,
            "test_exit_codes": {"focused": 0},
        },
        "exp5850-v521-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete: no accepted post-V521 source deltas",
            "inference_substrate": "aggregation_from_external_primary_sources",
            "accepted_finding_count": 0,
            "references_modified": False,
            "test_exit_codes": {"focused": 0},
        },
        "exp5851-deterministic-replay-provenance-contract": {
            "status": "ready",
            "honest_verdict": "ready: deterministic_replay_provenance_contract_clean",
            "inference_substrate": "deterministic_exact_verifier_and_replay_no_llm",
            "deterministic_replay_contract_ready_score": 1.0,
            "test_exit_codes": {"focused": 0},
        },
        "exp5852-three-family-paired-embeddings": {
            "status": "complete",
            "honest_verdict": "ready: paired_embedding_corpus_complete_all_three_models",
            "inference_substrate": "live_llm_embedding_extraction",
            "paired_embedding_corpus_ready_score": 1.0,
            "models_used": list(mod.MANDATED_EMBEDDING_MODEL_IDS),
            "model_specs": [
                {"hf_id": model_id, "headline_eligible": True, "quantization": "Q4_K_M"}
                for model_id in mod.MANDATED_EMBEDDING_MODEL_IDS
            ],
            "legacy_tiny_models": [
                {"hf_id": "Qwen/Qwen3.5-0.8B", "readiness_eligible": False}
            ],
            "model_file_and_tokenizer_receipts": {
                "all_embedded_tokenizers_loadable": True,
                "all_mandated_files_present": True,
                "receipts": {
                    model_id: {
                        "tokenizer_receipt": {"source": "embedded_gguf_llama_cpp_vocab_only"}
                    }
                    for model_id in mod.MANDATED_EMBEDDING_MODEL_IDS
                },
            },
            "row_file_receipt": {
                "path": mod.ROW_ARTIFACT_PATHS[
                    "exp5852-three-family-paired-embeddings"
                ].as_posix(),
                "row_count": 12,
            },
            "test_exit_codes": {"focused": 0},
        },
        "exp5853-paired-embedding-integrity-audit": {
            "status": "disqualified",
            "honest_verdict": "disqualified: raw_model_dimension_identity_shortcut",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "paired_embedding_integrity_ready_score": 0.0,
            "surviving_shortcuts": ["raw_model_dimension_identity_shortcut"],
            "test_exit_codes": {"focused": 0},
        },
        "exp5854-portable-comparative-energy-controls": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp5853-paired-embedding-integrity-audit.paired_embedding_integrity_ready_score"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp5853-paired-embedding-integrity-audit",
                    "artifact_field": "paired_embedding_integrity_ready_score",
                    "op": "==",
                    "expected": 1.0,
                    "actual": 0.0,
                    "passed": False,
                    "reason": "actual=0.0 == expected=1.0",
                }
            ],
        },
        "exp5856-provenance-correct-lifecycle": {
            "status": "complete",
            "honest_verdict": "complete: provenance_correct_adaptive_memory_lifecycle_credited",
            "inference_substrate": "deterministic_exact_verifier_and_replay_no_llm",
            "adaptive_memory_lifecycle_ready_score": 1.0,
            "no_model_weight_mutation": True,
            "verifier_is_oracle": True,
            "row_file_receipt": {
                "path": mod.ROW_ARTIFACT_PATHS[
                    "exp5856-provenance-correct-lifecycle"
                ].as_posix(),
                "row_count": 4,
            },
            "prospective_row_metrics": {
                "adaptive_accuracy": 1.0,
                "frozen_accuracy": 0.25,
                "adaptive_minus_frozen": {"mean": 0.75, "ci95": [0.5, 1.0]},
            },
            "test_exit_codes": {"focused": 0},
        },
        "exp5857-clean-transfer-selective-replay": {
            "status": "qualified",
            "honest_verdict": "qualified: clean_lifecycle_signature_compatible_replay",
            "inference_substrate": "deterministic_exact_verifier_and_replay_no_llm",
            "selective_replay_qualified_score": 1.0,
            "unsafe_transfer_count": 0,
            "test_exit_codes": {"focused": 0},
        },
        "exp5858-reduced-oracle-continuous-self-learning": {
            "status": "ready",
            "honest_verdict": "ready: reduced_oracle_continuous_self_learning",
            "inference_substrate": "deterministic_exact_verifier_and_replay_no_llm",
            "continuous_self_learning_task": True,
            "continuous_self_learning_ready_score": 1.0,
            "no_model_weight_mutation": True,
            "unsafe_accept_count": 0,
            "row_file_receipt": {
                "path": mod.ROW_ARTIFACT_PATHS[
                    "exp5858-reduced-oracle-continuous-self-learning"
                ].as_posix(),
                "row_count": 4,
            },
            "prospective_and_query_efficiency_metrics": {
                "lower_bounds_positive_over_controls": True,
                "full_oracle_lift_retained_fraction": 1.0,
                "reduced_query_fraction_of_full": 0.1,
                "reduced_minus_frozen": {"ci95": [0.2, 0.5]},
                "reduced_minus_random_query": {"ci95": [0.1, 0.4]},
            },
            "forward_transfer_recurrence_and_retention": {
                "no_retention_regression": True,
                "protected_prefix_retention": {"reduced_oracle": 1.0},
            },
            "rollback_restart_and_state_hashes": {
                "restart_equivalence": 1.0,
                "rollback_hash_mismatch_count": 0,
            },
            "memory_cap_accounting": {"cap_compliance": 1.0},
            "test_exit_codes": {"focused": 0},
        },
        "exp5859-adaptive-state-microkernel-parity": {
            "status": "blocked",
            "honest_verdict": "blocked: adaptive_state_microkernel_conformance_incomplete",
            "inference_substrate": "deterministic_cross_language_state_execution_no_llm",
            "adaptive_state_microkernel_ready_score": 0.0,
            "canonical_state_and_hash_parity": {"canonical_form_parity": True, "hash_parity": True},
            "cross_language_operation_parity": {"accept_reject_parity": True},
            "test_exit_codes": {".venv/bin/pytest tests/python -q": 2},
        },
        "exp5860-live-active-observation-ab": {
            "status": "complete_null",
            "honest_verdict": (
                "complete_null: active_observation_no_positive_preregistered_lower_bound"
            ),
            "inference_substrate": "live_llm_inference",
            "active_observation_ready_score": 0.0,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "fixture"}
            ],
            "models_used": [mod.MANDATED_ARC_MODEL_IDS[0]],
            "model_specs": [{"hf_id": model_id, "quantization": "Q4_K_M"} for model_id in mod.MANDATED_ARC_MODEL_IDS],
            "adapter_source_bfs_and_registry_exclusion_receipts": {
                "game_adapters_enabled": False,
                "public_source_read_enabled": False,
                "offline_ground_truth_bfs_enabled": False,
                "registry_trajectory_enabled": False,
                "per_game_model_enabled": False,
            },
            "solve_provenance": "live_agent_self_discovery",
            "registry_modified": False,
            "verifier_is_oracle": False,
            "test_exit_codes": {"focused": 0},
        },
        "exp5861-attached-board-state-receipts": {
            "status": "no_change_no_authenticated_state_operation_execution",
            "honest_verdict": "no-change: exp5859_not_ready no_speedup",
            "inference_substrate": "authenticated_hardware_state_execution_or_capability_receipt_no_llm",
            "authenticated_state_operation_parity_score": 0.0,
            "authenticated_physical_execution_receipts": [],
            "exp5859_input_receipt": {
                "present": True,
                "adaptive_state_microkernel_ready_score": 0.0,
                "mapping_allowed": False,
            },
            "same_input_state_and_hash_parity": {
                "physical_execution_observed": False,
                "parity_within_exact_tolerance": None,
            },
            "software_fallback_disclosed": {
                "cpu_reference_is_not_board_execution": True,
                "software_fallback_used_for_hardware_claim": False,
                "fallback_can_raise_parity_score": False,
            },
            "prohibited_claims_absent": {
                "all_absent": True,
                "speedup_claim_absent": True,
                "power_claim_absent": True,
                "energy_claim_absent": True,
            },
            "test_exit_codes": {"focused": 0},
        },
    }
    return payloads[task_id]


def _roadmap_payload() -> JsonDict:
    tasks: list[JsonDict] = []
    for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items():
        row: JsonDict = {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": mod.TASK_TITLES[task_id],
            "deliverable": rel_path.as_posix(),
        }
        if task_id in mod.GATE_DEFINITIONS:
            row["gated_on"] = mod.GATE_DEFINITIONS[task_id]
        tasks.append(row)
    return {"milestone": mod.MILESTONE, "tasks": tasks}


def _make_root(root: Path) -> None:
    for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items():
        if task_id in {"exp5855-exact-release-shadow-routing", "exp5862-v521-capstone-reconciliation"}:
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_text(root, mod.ROW_ARTIFACT_PATHS["exp5852-three-family-paired-embeddings"], "{}\n" * 12)
    _write_text(root, mod.ROW_ARTIFACT_PATHS["exp5856-provenance-correct-lifecycle"], "{}\n" * 4)
    _write_text(
        root,
        mod.ROW_ARTIFACT_PATHS["exp5858-reduced-oracle-continuous-self-learning"],
        "{}\n" * 4,
    )
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_roadmap_payload()))
    _write_text(root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: []\n")
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(
        root,
        mod.RESEARCH_REFERENCES_RELATIVE_PATH,
        "<!-- V521-PLANNER-REFRESH-20260723-END -->\n",
    )
    for rel_path in mod.SPEC_HASH_PATHS + mod.PROTECTED_FILE_PATHS:
        if not (root / rel_path).exists():
            _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\n")


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-07-23 12:05 UTC | Exact terminal-boundary handoff from .520 into .52 | OK | 87 passed |",
            "| 2026-07-23 13:58 UTC | Dated web evidence sweep after the V521 marker | OK | 117 passed |",
            "| 2026-07-23 14:40 UTC | Exact replay substrate contract and false-compute- | OK | 81 passed |",
            "| 2026-07-23 15:31 UTC | Current-SOTA causal-pair embedding extraction acro | OK | 87 passed |",
            "| 2026-07-23 16:02 UTC | Claim-flip, evaluator-swap, and identity-shortcut  | OK | 87 passed |",
            "| 2026-07-23 16:09 UTC | Held-model and held-constraint comparative energy  | GATE_BLOCK | 1 of 1 gate(s) failed |",
            "| 2026-07-23 16:53 UTC | Exact-authority shadow routing after a portable en | GATE_BLOCK | Pre-emptive skip |",
            "| 2026-07-23 16:30 UTC | Prospective adaptive-memory lifecycle on an honest | OK | 87 passed |",
            "| 2026-07-23 16:51 UTC | Clean-upstream selective replay with hard-case neg | OK | 87 passed |",
            "| 2026-07-23 17:12 UTC | Reduced-oracle versioned constraint memory on seal | OK | 87 passed |",
            "| 2026-07-23 19:27 UTC | Accepted adaptive operations ABI conformance | FAIL | artifact_not_updated_past_bootstrap |",
            "| 2026-07-23 20:05 UTC | Closed-loop visual probing under equal action budg | FLAGGED | adversarial_verify CRITICAL |",
            "| 2026-07-23 21:09 UTC | KV260 PolarFire GateMate physical capability ledge | FAIL | artifact_not_updated_past_bootstrap |",
        ]
    )


def _receipt(task_id: str, *, critical: bool = False) -> JsonDict:
    flags = (
        [{"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "fixture"}]
        if critical
        else []
    )
    report = {
        "reports": [
            {
                "path": mod.TASK_ARTIFACT_PATHS[task_id].as_posix(),
                "flag_count": len(flags),
                "flags": flags,
                "max_severity": 2 if flags else -1,
            }
        ],
        "flagged_count": 1 if flags else 0,
    }
    return {
        "task_id": task_id,
        "artifact_path": mod.TASK_ARTIFACT_PATHS[task_id].as_posix(),
        "command": (
            ".venv/bin/python scripts/adversarial_verify.py --json "
            f"{mod.TASK_ARTIFACT_PATHS[task_id].as_posix()}"
        ),
        "exit_code": 1 if flags else 0,
        "stdout_json": report,
        "stderr": "",
        "receipt_hash": mod.sha256_json(report),
    }


def _receipts() -> dict[str, JsonDict]:
    return {
        task_id: _receipt(task_id, critical=task_id == "exp5860-live-active-observation-ab")
        for task_id in mod.TASK_ARTIFACT_PATHS
        if task_id
        not in {"exp5855-exact-release-shadow-routing", "exp5862-v521-capstone-reconciliation"}
    }


def _publication_gate() -> JsonDict:
    return {"paper_ready": True, "unmet_gates": [], "gates": {"G1": {"pass": True}}}


def _build(root: Path) -> JsonDict:
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        publication_gate=_publication_gate(),
        tests_run=[
            {
                "command": (
                    ".venv/bin/pytest "
                    "tests/python/test_experiment_5862_v521_capstone_reconciliation.py -q"
                ),
                "exit_code": 0,
            },
            { "command": ".venv/bin/pytest tests/python -q", "exit_code": 0 },
        ],
        modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
        duration_s=1.5,
    )


def test_req_report_5862_spec_declares_capstone_contract() -> None:
    """REQ-REPORT-5862: OpenSpec names required fields, gates, and scenarios."""

    section = SPEC_PATH.read_text(encoding="utf-8")
    section = section[section.index("### REQ-REPORT-5862") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "SCENARIO-REPORT-5862-GATE-REPLAY" in section
    assert "SCENARIO-REPORT-5862-FLAGS-AND-RETIREMENTS" in section
    assert "SCENARIO-REPORT-5862-MODEL-AUTHORITY" in section
    assert "globbed numeric prefix" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_5862_gate_replay_and_exact_denominator(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5862-GATE-REPLAY: gates explain skipped and blocked branches."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert report["status"] == "mixed"
    assert report["honest_verdict"].startswith("mixed:")
    assert len(report["exact_task_and_deliverable_matrix"]) == 14
    assert report["exact_task_and_deliverable_matrix"]["exp5855-exact-release-shadow-routing"][
        "present"
    ] is False
    assert report["exact_task_and_deliverable_matrix"]["exp5855-exact-release-shadow-routing"][
        "selection_policy"
    ] == "exact_declared_deliverable"

    gates = report["structured_gate_replay"]
    assert gates["exp5854-portable-comparative-energy-controls"]["all_gates_passed"] is False
    assert gates["exp5854-portable-comparative-energy-controls"]["science_execution_allowed"] is False
    assert gates["exp5855-exact-release-shadow-routing"]["science_execution_allowed"] is False
    assert gates["exp5858-reduced-oracle-continuous-self-learning"]["all_gates_passed"] is True
    assert gates["exp5859-adaptive-state-microkernel-parity"]["all_gates_passed"] is True

    classes = report["outcome_classification"]
    assert "exp5854-portable-comparative-energy-controls" in classes["gated_skip"]
    assert "exp5855-exact-release-shadow-routing" in classes["gated_skip"]
    assert "exp5859-adaptive-state-microkernel-parity" in classes["blocked"]
    assert "exp5860-live-active-observation-ab" in classes["flagged"]
    assert classes["missing"] == []
    mod.validate_artifact(report)


def test_scenario_report_5862_decisions_and_retirement_predicates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5862-FLAGS-AND-RETIREMENTS: flags and nulls remain visible."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert report["transition_and_source_decision"]["transition_complete"] is True
    assert report["transition_and_source_decision"]["source_delta_accepted_count"] == 0
    assert report["comparative_energy_decision"]["portable_comparative_energy_ready"] is False
    assert report["comparative_energy_decision"]["blocking_task_id"] == (
        "exp5853-paired-embedding-integrity-audit"
    )
    assert report["lifecycle_and_replay_decision"]["adaptive_memory_lifecycle_promotable"] is True
    assert report["lifecycle_and_replay_decision"]["selective_replay_promotable"] is True
    assert report["continuous_self_learning_decision"]["continuous_self_learning_promotable"] is True
    assert report["microkernel_decision"]["microkernel_promotable"] is False
    assert report["arc_active_observation_decision"]["active_observation_promotable"] is False
    assert report["arc_active_observation_decision"]["verifier_clean"] is False
    assert report["arc_active_observation_decision"]["solve_credit"] is False
    assert report["hardware_capability_decision"]["board_claim_promotable"] is False
    assert report["hardware_capability_decision"]["speedup_claimed"] is False

    retirements = report["prior_failure_retirement_decisions"]
    assert retirements["lifecycle_replay"]["predicate_satisfied"] is False
    assert retirements["final_embedding_route"]["predicate_satisfied"] is False
    assert retirements["reduced_oracle_csl"]["predicate_satisfied"] is False
    assert retirements["active_observation"]["predicate_satisfied"] is False
    assert retirements["bounded_retirement_recommendations"] == []
    assert report["missing_or_flagged_evidence"]["flagged_task_ids"] == [
        "exp5860-live-active-observation-ab"
    ]
    mod.validate_artifact(report)


def test_scenario_report_5862_model_compliance_and_authority(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5862-MODEL-AUTHORITY: model and prohibited paths stay explicit."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    model = report["model_compliance_receipts"]
    assert model["exp5852"]["all_mandated_embedding_models_used"] is True
    assert model["exp5860"]["mandated_arc_model_used"] is True
    assert model["tiny_model_promoted"] is False
    assert model["auto_tokenizer_promoted"] is False
    assert set(model["exp5852"]["models_used"]) == set(mod.MANDATED_EMBEDDING_MODEL_IDS)

    authority = report["authority_and_prohibited_path_receipts"]
    assert authority["exact_validator_release_authority_preserved"] is True
    assert authority["arc_registry_modified"] is False
    assert authority["arc_forbidden_paths_excluded"] is True
    assert authority["hardware_software_fallback_promoted"] is False
    assert authority["requested_topology_promoted_as_execution"] is False
    assert authority["publication_action_taken"] is False
    mod.validate_artifact(report)


def test_scenario_report_5862_schema_checksum_and_protection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5862-SCHEMA: required fields and protected files are stable."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["paper_ready"] is True
    assert report["publication_action_taken"] is False
    for receipt in report["adversarial_verifier_receipts"].values():
        assert {"command", "exit_code", "stdout_json", "flag_count", "max_severity", "receipt_hash"} <= set(
            receipt
        )
    assert report["docs_reconciled"]["ops_status_md"] == "deferred_by_operator_stop_rule"
    assert report["docs_reconciled"]["ops_changelog_md"] == "deferred_by_operator_stop_rule"
    assert report["docs_reconciled"]["traceability_md"] == "deferred_by_operator_stop_rule"
    assert all(item["unchanged"] for item in report["protected_files_unchanged"].values())
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in report["field_provenance"]
        assert field in report["field_principles"]
    mod.validate_artifact(report)

    mutations = [
        (lambda a: a.pop("status"), "missing required field"),
        (lambda a: a.update(publication_action_taken=True), "publication_action_taken"),
        (lambda a: a.update(inference_substrate="live_llm_inference"), "inference_substrate"),
        (lambda a: a.update(honest_verdict="ambiguous"), "honest_verdict"),
        (
            lambda a: a["protected_files_unchanged"][mod.NORTH_STAR_RELATIVE_PATH.as_posix()].update(
                unchanged=False
            ),
            "protected file",
        ),
        (lambda a: a.update(field_provenance=[]), "field provenance"),
        (lambda a: a["field_provenance"].pop("status"), "field provenance missing"),
        (lambda a: a.update(outcome_classification=[]), "outcome_classification"),
        (
            lambda a: a["outcome_classification"]["clean_positive"].append(
                "exp5860-live-active-observation-ab"
            ),
            "flagged",
        ),
        (
            lambda a: a["model_compliance_receipts"].update(tiny_model_promoted=True),
            "tiny model",
        ),
        (
            lambda a: a["model_compliance_receipts"].update(auto_tokenizer_promoted=True),
            "AutoTokenizer",
        ),
        (
            lambda a: a["authority_and_prohibited_path_receipts"].update(
                hardware_software_fallback_promoted=True
            ),
            "software fallback",
        ),
        (
            lambda a: a["authority_and_prohibited_path_receipts"].update(
                publication_action_taken=True
            ),
            "publication_action_taken authority",
        ),
    ]
    for mutate, needle in mutations:
        artifact = deepcopy(report)
        mutate(artifact)
        artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
        with pytest.raises(ValueError, match=needle):
            mod.validate_artifact(artifact)

    checksum_drift = deepcopy(report)
    checksum_drift["status"] = "changed_after_checksum"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(checksum_drift)


def test_scenario_report_5862_ignores_prior_self_output_for_stability(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5862-SCHEMA: prior capstone JSON cannot perturb reruns."""

    _make_root(tmp_path)
    without_prior = _build(tmp_path)
    _write_json(
        tmp_path,
        mod.RESULT_RELATIVE_PATH,
        {
            "status": "stale",
            "honest_verdict": "mixed: stale fixture that must not become evidence",
            "reproducibility_checksum": "sha256:not-real",
        },
    )
    with_prior = _build(tmp_path)

    assert with_prior == without_prior
    matrix_row = with_prior["exact_task_and_deliverable_matrix"][
        "exp5862-v521-capstone-reconciliation"
    ]
    assert matrix_row["self_output_not_upstream_evidence"] is True
    assert matrix_row["present"] is False
    assert (
        "exp5862-v521-capstone-reconciliation"
        not in with_prior["preconditions_checked"]["declared_deliverable_hashes"]
    )
    mod.validate_artifact(with_prior)


def test_scenario_report_5862_failed_required_checks_block_closure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5862-SCHEMA: failed required checks cannot close the capstone."""

    _make_root(tmp_path)
    report = mod.build_report(
        tmp_path,
        adversarial_receipts=_receipts(),
        publication_gate=_publication_gate(),
        tests_run=[
            {"command": ".venv/bin/pytest tests/python -q", "exit_code": 2},
        ],
        modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
        duration_s=1.0,
    )

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["preconditions_checked"]["failed_required_test_commands"] == [
        ".venv/bin/pytest tests/python -q"
    ]
    mod.validate_artifact(report)


def test_req_report_5862_helpers_cover_defensive_and_complete_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5862: helper branches remain deterministic and auditable."""

    directory = tmp_path / "hashdir"
    _write_text(directory, "child.txt", "content\n")
    assert mod.path_sha256(directory).startswith("sha256:")

    output = tmp_path / "written.json"
    mod.write_json(output, {"b": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"b": 1}

    real_write_text = Path.write_text

    def broken_probe_write(path: Path, *args: Any, **kwargs: Any) -> int:
        if path.name.endswith(".tmp-probe"):
            raise OSError("fixture")
        return real_write_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", broken_probe_write)
        assert mod._atomic_output_receipt(tmp_path / "artifact.json")["error"].startswith("OSError:")

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    _, bad_meta = mod._read_json_mapping(bad_json)
    assert bad_meta["error"].startswith("json_error:")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    _, list_meta = mod._read_json_mapping(list_json)
    assert list_meta["error"] == "json_not_object"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("a: [\n", encoding="utf-8")
    _, yaml_meta = mod._read_yaml_mapping(bad_yaml)
    assert yaml_meta["error"].startswith("yaml_error:")
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- a\n", encoding="utf-8")
    _, yaml_list_meta = mod._read_yaml_mapping(list_yaml)
    assert yaml_list_meta["error"] == "yaml_not_mapping"

    assert mod._log_status("| x | UNKNOWN | y |") == "LOGGED"
    assert mod._compare(2, ">=", 1) is True
    assert mod._compare(2, ">", 1) is True
    assert mod._compare(1, "<=", 1) is True
    assert mod._compare("2", ">=", 1) is False
    assert mod._compare(2, "!=", 1) is False
    assert mod._roadmap_gates({"tasks": ["not-a-map"]})
    assert mod._normalize_receipts(None) == {}
    assert mod._normalize_receipts([{"task_id": "x", "ok": True}]) == {
        "x": {
            "task_id": "x",
            "ok": True,
            "flag_count": 0,
            "max_severity": -1,
            "flags": [],
            "receipt_hash": mod.sha256_json({}),
        }
    }
    assert mod._receipt_flag_count({"stdout_json": {"flagged_count": 3}}) == 3
    assert mod._receipt_flags({"stdout_json": {"reports": [{"flags": "not-list"}]}}) == []
    assert mod._tests_run_rows(None)[0]["status"] == "not_recorded"

    row_root = tmp_path / "rows"
    _write_text(row_root, mod.ROW_ARTIFACT_PATHS["exp5852-three-family-paired-embeddings"], "{}\n")
    _write_text(
        row_root,
        mod.ROW_ARTIFACT_PATHS["exp5856-provenance-correct-lifecycle"],
        '{bad json}\n{"adaptive_minus_frozen_delta": 0.5, "adaptive_accuracy": 1.0, "frozen_accuracy": 0.5}\n',
    )
    _write_text(
        row_root,
        mod.ROW_ARTIFACT_PATHS["exp5858-reduced-oracle-continuous-self-learning"],
        json.dumps(
            {
                "arms": {
                    "reduced_oracle": {"accuracy": 1.0, "exact_queries_used": 2},
                    "bad": "not-a-map",
                }
            }
        )
        + "\n",
    )
    rows = mod._row_file_receipts(row_root)
    assert rows["exp5856-provenance-correct-lifecycle"]["recomputed_means"][
        "adaptive_minus_frozen_delta"
    ] == pytest.approx(0.5)
    assert rows["exp5858-reduced-oracle-continuous-self-learning"]["recomputed_means"][
        "reduced_oracle_accuracy"
    ] == pytest.approx(1.0)

    payloads = {task_id: {"status": "complete"} for task_id in mod.EXPECTED_TASK_IDS}
    metadata = {task_id: {"present": True} for task_id in mod.EXPECTED_TASK_IDS}
    gates = {
        task_id: {"gates": [], "all_gates_passed": True}
        for task_id in mod.EXPECTED_TASK_IDS
    }
    receipts: dict[str, JsonDict] = {}
    payloads["exp5856-provenance-correct-lifecycle"]["unsafe_accept_count"] = 1
    metadata["exp5849-transition-v521"]["present"] = False
    payloads["exp5850-v521-source-delta-ingestion"] = {
        "status": "unknown",
        "honest_verdict": "",
    }
    classes = mod._classify_outcomes(payloads, metadata, gates, receipts)
    assert "exp5856-provenance-correct-lifecycle" in classes["unsafe"]
    assert "exp5849-transition-v521" in classes["missing"]
    assert "exp5850-v521-source-delta-ingestion" in classes["off_path"]

    _make_root(tmp_path)
    _write_json(
        tmp_path,
        mod.TASK_ARTIFACT_PATHS["exp5853-paired-embedding-integrity-audit"],
        {
            "status": "complete",
            "honest_verdict": "complete: integrity_ready",
            "paired_embedding_integrity_ready_score": 1.0,
        },
    )
    _write_json(
        tmp_path,
        mod.TASK_ARTIFACT_PATHS["exp5854-portable-comparative-energy-controls"],
        {
            "status": "complete",
            "honest_verdict": "complete: portable_energy_ready",
            "portable_comparative_energy_ready_score": 1.0,
        },
    )
    _write_json(
        tmp_path,
        mod.TASK_ARTIFACT_PATHS["exp5855-exact-release-shadow-routing"],
        {"status": "complete", "honest_verdict": "complete: shadow_routing_ready"},
    )
    _write_json(
        tmp_path,
        mod.TASK_ARTIFACT_PATHS["exp5859-adaptive-state-microkernel-parity"],
        {
            "status": "ready",
            "honest_verdict": "ready: microkernel",
            "adaptive_state_microkernel_ready_score": 1.0,
        },
    )
    arc = _artifact("exp5860-live-active-observation-ab")
    arc["flagged_adversarial"] = False
    arc["corrigendum_pending"] = []
    _write_json(tmp_path, mod.TASK_ARTIFACT_PATHS["exp5860-live-active-observation-ab"], arc)
    clean_receipts = {
        task_id: _receipt(task_id)
        for task_id in mod.TASK_ARTIFACT_PATHS
        if task_id != "exp5862-v521-capstone-reconciliation"
    }
    complete = mod.build_report(
        tmp_path,
        adversarial_receipts=clean_receipts,
        publication_gate=_publication_gate(),
        modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
        duration_s=1.0,
    )
    assert complete["status"] == "complete"
    assert complete["honest_verdict"].startswith("complete:")

    blocked = mod.build_report(
        tmp_path,
        adversarial_receipts=clean_receipts,
        publication_gate=_publication_gate(),
        modification_overrides={**{rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS}, mod.NORTH_STAR_RELATIVE_PATH: True},
        duration_s=1.0,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
