"""Tests for the Exp6168 V534 capstone reconciliation.

Spec refs: REQ-REPORT-6168,
SCENARIO-REPORT-6168-EXACT-PATH-TERMINALS,
SCENARIO-REPORT-6168-MANDATORY-CSL,
SCENARIO-REPORT-6168-QUARANTINE-AND-DECISION-GATES,
SCENARIO-REPORT-6168-SUBSTRATE-ARC-AND-STOCHASTIC,
SCENARIO-REPORT-6168-SCHEMA-HISTORY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_6168_v534_capstone_reconciliation as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_text(root: Path, rel_path: Path | str, text: str = "fixture\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    _write_text(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _artifact(task_id: str) -> JsonDict:
    payloads: dict[str, JsonDict] = {
        "exp6156-transition-v534": {
            "status": "complete_with_terminal_receipts",
            "honest_verdict": "complete: .534 activation mode=already_active",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "protected_files_unchanged": {"all_unchanged": True},
            "research_complete_append_count": 0,
            "duplicate_history_amplification_count": 0,
        },
        "exp6158-v534-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete_null: no accepted post-V534 source deltas",
            "inference_substrate": "literature_ingestion",
            "accepted_rejected_duplicate_completed_retired_and_abstained_findings": {
                "accepted": [],
                "accepted_count": 0,
            },
            "references_append_receipt": {"appended": False},
        },
        "exp6159-decision-calibrated-stream": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: fresh stream",
            "inference_substrate": "deterministic_verifier_plus_replay",
            "decision_calibrated_stream_ready_score": 1.0,
            "llm_invocation_count": 0,
            "event_template_family_partition_and_shift_counts": {
                "event_count": 240,
                "family_count": 6,
                "partition_counts": {"calibration": 96, "future_known": 64},
            },
            "exposed_fixture_overlap_counts": {
                "event_overlap_count": 0,
                "template_overlap_count": 0,
                "seed_overlap_count": 0,
            },
            "held_loader_one_shot_contract": {"held_access_count": 0},
            "exact_validator_agreement": {"disagreement_count": 0},
        },
        "exp6160-sota-decision-calibration-corpus": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: corpus complete",
            "inference_substrate": "live_local_sota_gguf_cuda",
            "sota_decision_corpus_ready_score": 1.0,
            "MODEL_SPECS": [
                {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "actual_use_count": 240},
                {"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "actual_use_count": 240},
            ],
            "model_specs": [
                {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "actual_use_count": 240},
                {"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "actual_use_count": 240},
            ],
            "gpu_offload_pid_lifecycle_and_cleanup_receipts": {
                "all_models_gpu_engaged": True,
                "all_models_release_ready": True,
                "orphan_task_owned_pid_count": 0,
            },
            "per_model_row_paths_hashes_and_counts": {
                "total_row_count": 480,
                "per_model": {
                    "qwen": {"row_count": 240},
                    "gemma": {"row_count": 240},
                },
            },
            "label_conditioned_retry_count": 0,
            "memory_read_and_write_counts": {"memory_read_count": 0, "memory_write_count": 0},
        },
        "exp6161-decision-calibrated-energy-policy": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: policy frozen",
            "inference_substrate": "cached_authentic_sota_rows_cpu_analysis",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "decision_calibrated_policy_ready_score": 1.0,
            "held_access_count": 0,
            "selected_policy_rationale_without_held_access": {
                "selected_arm": "decision_calibrated_task_energy",
                "policy_validly_frozen": True,
                "selection_uses_held_outcomes": False,
            },
            "per_model_calibration_cost_brier_ece_unsafe_safe_and_descriptive_ranking_metrics": {
                "by_model": {
                    "qwen": {
                        "arms": {
                            "decision_calibrated_task_energy": {
                                "utility_per_row": 1.18,
                                "brier": 0.01,
                                "ece": 0.02,
                            }
                        }
                    },
                    "gemma": {
                        "arms": {
                            "decision_calibrated_task_energy": {
                                "utility_per_row": 1.17,
                                "brier": 0.02,
                                "ece": 0.03,
                            }
                        }
                    },
                }
            },
        },
        "exp6162-prospective-admission-replication": {
            "status": "complete_positive",
            "honest_verdict": "complete_positive: both models pass",
            "inference_substrate": "sealed_cached_event_evaluation",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "prospective_admission_replication_ready_score": 1.0,
            "first_and_only_held_access_receipt": {
                "held_access_count_before": 0,
                "held_access_count_after": 1,
            },
            "selector_and_threshold_refit_counts": {
                "all_zero": True,
                "counts": {"selector_refit_count": 0, "threshold_refit_count": 0},
            },
            "per_model_and_conjunctive_gate_matrix": {
                "conjunctive_pass": True,
                "pooled_success_cannot_mask_model_or_partition_failure": True,
                "by_model": {"qwen": {"model_pass": True}, "gemma": {"model_pass": True}},
            },
            "unsafe_admission_and_known_family_noninferiority_gates": {"all_gates_pass": True},
            "brier_ece_and_descriptive_auroc_auprc_metrics": {
                "by_model": {"qwen": {"future_known": {}}, "gemma": {"future_known": {}}}
            },
            "per_model_future_known_and_shifted_decision_utility_intervals": {
                "by_model": {
                    "qwen": {
                        "future_known": {
                            "decision_calibrated_minus_global": {"lower_95_above_zero": True}
                        }
                    },
                    "gemma": {
                        "future_known": {
                            "decision_calibrated_minus_global": {"lower_95_above_zero": True}
                        }
                    },
                }
            },
            "retirement_triggered": False,
        },
        "exp6164-continuous-strategy-learning-ab": {
            "status": "blocked",
            "honest_verdict": "blocked: exp6163_not_ready; self-learning did not execute",
            "inference_substrate": "blocked_before_model_load_or_live_local_sota_gguf_cuda",
            "mandatory_artifact_written": True,
            "continuous_self_learning_task": True,
            "continuous_strategy_learning_ready_score": 0.0,
            "prerequisite_gate_receipts": {
                "all_passed": False,
                "blocked_reasons": ["exp6163_not_ready"],
                "exp6163": {"ready": False, "artifact_receipt": {"exists": False}},
            },
            "blocked_before_model_load_receipt": {
                "blocked": True,
                "blocked_reasons": ["exp6163_not_ready"],
                "all_invocation_counts_zero": True,
                "invocation_counts": {"model_load_count": 0, "gpu_worker_count": 0},
            },
            "model_weight_immutability_receipt": {
                "all_unchanged": True,
                "weight_update_count": 0,
            },
            "learning_speed_and_time_to_benefit": {
                "learning_executed": False,
                "time_to_benefit_event": None,
            },
            "per_model_family_partition_future_utility_accuracy_regret_and_grouped_intervals": {
                "by_model": {"qwen": {"future_known": {"row_count": 0}}}
            },
            "retirement_triggered": False,
        },
        "exp6166-mode-jumping-factor-thermalization": {
            "status": "blocked",
            "honest_verdict": "blocked: mode jumping improved but nonzero_test_commands",
            "inference_substrate": "jax_cpu_software_multimodal_factor_thermalization",
            "mode_jumping_factor_thermalization_ready_score": 0.0,
            "deliberately_nonzero_error_receipt": {
                "approximate_error_finite_and_strictly_positive": True,
                "identity_exact_table_zero_error": True,
                "local_only_joint_tv": 0.09,
                "mode_jump_joint_tv": 0.03,
            },
            "factor_and_joint_tv_kl_and_mode_mass_ratio_errors": {
                "arms": {
                    "local_only": {"joint_tv": 0.09},
                    "mode_jump": {"joint_tv": 0.03},
                }
            },
            "bound_slack_and_violation_counts": {"violation_count": 0},
            "hardware_execution_claimed": False,
            "latency_power_energy_and_speedup_claimed": False,
            "retirement_triggered": False,
        },
        "exp6167-arc-task-aware-multiseed-replication": {
            "status": "complete_positive",
            "honest_verdict": "complete_positive: no solve",
            "inference_substrate": "live_e3_adapter_disabled_runtime_transitions",
            "arc_task_aware_multiseed_replication_ready_score": 1.0,
            "game_seed_action_budget_and_arm_counts": {
                "game_count": 6,
                "seed_count": 3,
                "decision_count": 288,
                "live_row_count": 144,
            },
            "per_arm_triggered_decision_counts": {"global": 144, "task_aware": 144},
            "grouped_paired_intervals": {
                "lower_95": 0.04,
                "by_game": {"tu93": {"mean_task_aware_minus_global": 0.0}},
            },
            "solve_claimed": False,
            "level_credit_delta": 0,
            "registry_levels_unchanged": True,
            "offline_ground_truth_bfs": False,
            "used_game_source": False,
            "llm_invocation_count": 0,
        },
    }
    return payloads[task_id]


def _roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE,
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE,
                "title": title,
                "deliverable": rel_path.as_posix(),
                **(
                    {"gated_on": deepcopy(mod.GATED_ON[task_id])} if task_id in mod.GATED_ON else {}
                ),
            }
            for task_id, title, rel_path in mod.ACTIVATED_TASKS
        ]
        + [
            {
                "id": mod.EXPERIMENT_ID,
                "milestone": mod.MILESTONE,
                "title": "Branch-independent .534 capstone",
                "deliverable": mod.RESULT_RELATIVE_PATH.as_posix(),
            }
        ],
    }


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-08-06 08:32 UTC | Exact terminal-boundary handoff from .533 into .53 | OK | 86 passed |",
            "| 2026-08-06 12:43 UTC | Repository-wide test artifact-isolation compatibil | FAIL | hard wall |",
            "| 2026-08-06 13:17 UTC | Reliable dated evidence refresh after the V534 pla | OK | 115 passed |",
            "| 2026-08-06 13:57 UTC | Fresh chronological decision-calibration stream an | OK | 87 passed |",
            "| 2026-08-06 15:21 UTC | Gated on Exp6159 readiness: fresh flagship-GGUF de | OK | 87 passed |",
            "| 2026-08-06 15:44 UTC | Gated on Exp6160 readiness: freeze a decision-cali | FLAGGED | DURATION_TOO_SHORT |",
            "| 2026-08-06 16:10 UTC | Gated on Exp6161 readiness: one-shot decision-util | FLAGGED | DURATION_TOO_SHORT |",
            "| 2026-08-06 17:11 UTC | Gated on Exp6157 and Exp6159 readiness: certified | GATE_BLOCK | upstream retired |",
            "| 2026-08-06 17:44 UTC | Mandatory prospective continuous strategy-learning | FAIL | artifact_not_updated |",
            "| 2026-08-06 17:46 UTC | Gated on Exp6164 positive utility: default-off tra | GATE_BLOCK | upstream retired |",
            "| 2026-08-06 19:50 UTC | Mode-jumping CNCE for nonzero-error typed-factor t | FAIL | artifact_not_updated |",
            "| 2026-08-06 20:08 UTC | ARC live-path task-aware admission replication acr | OK | 86 passed |",
        ]
    )


def _make_root(root: Path) -> None:
    missing = {
        "exp6157-repo-wide-artifact-isolation-closure",
        "exp6163-certified-strategy-store-scaleup",
        "exp6165-strategy-memory-shadow-adapter",
    }
    for task_id, _title, rel_path in mod.ACTIVATED_TASKS:
        if task_id not in missing:
            _write_json(root, rel_path, _artifact(task_id))
    _write_json(
        root,
        "results/experiment_6157_same_number_alias.json",
        {"status": "complete_positive", "honest_verdict": "complete_positive: alias"},
    )
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_roadmap_payload()))
    _write_text(root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: []\n")
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    for rel_path in mod.PROTECTED_FILE_PATHS + mod.PRECONDITION_CONTEXT_PATHS:
        if not (root / rel_path).exists():
            _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\n")


def _receipt(task_id: str, rel_path: Path) -> JsonDict:
    report = {
        "artifact": rel_path.as_posix(),
        "loaded": True,
        "flag_count": 0,
        "max_severity": -1,
        "flags": [],
    }
    return {
        "task_id": task_id,
        "artifact_path": rel_path.as_posix(),
        "command": f".venv/bin/python scripts/adversarial_verify.py --json {rel_path.as_posix()}",
        "exit_code": 0,
        "stdout_json": {"reports": [report], "flagged_count": 0},
    }


def _receipts() -> dict[str, JsonDict]:
    missing = {
        "exp6157-repo-wide-artifact-isolation-closure",
        "exp6163-certified-strategy-store-scaleup",
        "exp6165-strategy-memory-shadow-adapter",
    }
    return {
        task_id: _receipt(task_id, rel_path)
        for task_id, _title, rel_path in mod.ACTIVATED_TASKS
        if task_id not in missing
    }


def _build(root: Path) -> JsonDict:
    _make_root(root)
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run=[
            {
                "command": ".venv/bin/pytest tests/python/test_experiment_6168_v534_capstone_reconciliation.py -q --no-cov -n 0",
                "exit_code": 0,
            },
            {"command": ".venv/bin/pytest tests/python -q", "exit_code": 0},
        ],
        duration_s=1.5,
    )


def test_req_report_6168_spec_declares_required_contract() -> None:
    """REQ-REPORT-6168: OpenSpec names exact-path capstone requirements."""

    section = SPEC_PATH.read_text(encoding="utf-8")
    section = section[section.index("### REQ-REPORT-6168") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "Exp6156 through Exp6167" in section
    assert "Exp6164 SHALL be treated as mandatory" in section
    assert "software simulation SHALL NOT be promoted to hardware" in section
    for scenario in (
        "SCENARIO-REPORT-6168-EXACT-PATH-TERMINALS",
        "SCENARIO-REPORT-6168-MANDATORY-CSL",
        "SCENARIO-REPORT-6168-QUARANTINE-AND-DECISION-GATES",
        "SCENARIO-REPORT-6168-SUBSTRATE-ARC-AND-STOCHASTIC",
        "SCENARIO-REPORT-6168-SCHEMA-HISTORY",
    ):
        assert scenario in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_6168_exact_path_terminals_and_counts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6168-EXACT-PATH-TERMINALS: each task is classified once."""

    report = _build(tmp_path)

    assert report["status"] == "complete_with_blocks_missing_skips_and_quarantine"
    assert report["honest_verdict"].startswith("complete:")
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE

    matrix = report["activated_task_and_declared_deliverable_matrix"]
    assert list(matrix) == [task_id for task_id, _title, _path in mod.ACTIVATED_TASKS]
    assert matrix["exp6157-repo-wide-artifact-isolation-closure"]["present"] is False
    assert matrix["exp6157-repo-wide-artifact-isolation-closure"]["terminal_class"] == "missing"
    assert matrix["exp6157-repo-wide-artifact-isolation-closure"]["same_number_alias_used"] is False
    assert matrix["exp6157-repo-wide-artifact-isolation-closure"][
        "same_number_alias_candidates_ignored"
    ] == ["results/experiment_6157_same_number_alias.json"]
    assert matrix["exp6163-certified-strategy-store-scaleup"]["terminal_class"] == "skipped"
    assert matrix["exp6165-strategy-memory-shadow-adapter"]["terminal_class"] == "skipped"

    classes = report["exact_terminal_classification"]["terminal_class_by_task_id"]
    assert classes == {
        "exp6156-transition-v534": "complete",
        "exp6157-repo-wide-artifact-isolation-closure": "missing",
        "exp6158-v534-source-delta-ingestion": "null",
        "exp6159-decision-calibrated-stream": "positive",
        "exp6160-sota-decision-calibration-corpus": "positive",
        "exp6161-decision-calibrated-energy-policy": "flagged",
        "exp6162-prospective-admission-replication": "flagged",
        "exp6163-certified-strategy-store-scaleup": "skipped",
        "exp6164-continuous-strategy-learning-ab": "internal_blocked",
        "exp6165-strategy-memory-shadow-adapter": "skipped",
        "exp6166-mode-jumping-factor-thermalization": "blocked",
        "exp6167-arc-task-aware-multiseed-replication": "positive",
    }

    counts = report[
        "present_missing_skipped_internal_blocked_null_retired_flagged_and_positive_counts"
    ]
    assert counts["present"] == 9
    assert counts["missing"] == 1
    assert counts["skipped"] == 2
    assert counts["internal_blocked"] == 1
    assert counts["blocked"] == 1
    assert counts["null"] == 1
    assert counts["flagged"] == 2
    assert counts["positive"] == 3
    assert counts["positive_aggregation_eligible"] == 3


def test_scenario_report_6168_mandatory_csl_internal_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6168-MANDATORY-CSL: Exp6164 is present but blocked inside."""

    report = _build(tmp_path)

    receipt = report["mandatory_continuous_learning_artifact_receipt"]
    assert receipt["task_id"] == "exp6164-continuous-strategy-learning-ab"
    assert receipt["present"] is True
    assert receipt["terminal_class"] == "internal_blocked"
    assert receipt["mandatory_artifact_written"] is True
    assert receipt["live_self_learning_executed"] is False
    assert receipt["blocked_reasons"] == ["exp6163_not_ready"]
    assert receipt["model_weight_immutability"]["all_unchanged"] is True

    csl = report["continuous_strategy_learning_and_shadow_summary"]
    assert csl["mandatory_csl_terminal_class"] == "internal_blocked"
    assert csl["mandatory_csl_ready_score"] == 0.0
    assert csl["shadow_adapter_terminal_class"] == "skipped"
    assert csl["shadow_adapter_artifact_present"] is False


def test_scenario_report_6168_quarantine_and_decision_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6168-QUARANTINE-AND-DECISION-GATES: flagged positives do not aggregate."""

    report = _build(tmp_path)
    quarantine = report["adversarial_verifier_and_quarantine_receipts"]

    assert quarantine["verified_present_artifact_count"] == 9
    assert quarantine["flagged_task_ids"] == [
        "exp6161-decision-calibrated-energy-policy",
        "exp6162-prospective-admission-replication",
    ]
    assert quarantine["positive_aggregation_eligible_task_ids"] == [
        "exp6159-decision-calibrated-stream",
        "exp6160-sota-decision-calibration-corpus",
        "exp6167-arc-task-aware-multiseed-replication",
    ]
    assert (
        quarantine["receipts_by_task_id"]["exp6162-prospective-admission-replication"][
            "excluded_from_positive_aggregation"
        ]
        is True
    )

    decision = report["decision_policy_and_one_shot_replication_summary"]
    assert decision["policy_held_access_count"] == 0
    assert decision["policy_validly_frozen"] is True
    assert decision["replication_held_access_before_after"] == {"before": 0, "after": 1}
    assert decision["selector_and_threshold_refit_all_zero"] is True
    assert decision["per_model_conjunctive_pass_raw"] is True
    assert decision["replication_positive_aggregation_eligible"] is False


def test_scenario_report_6168_substrate_arc_and_stochastic_boundaries(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6168-SUBSTRATE-ARC-AND-STOCHASTIC: raw boundaries remain bounded."""

    report = _build(tmp_path)

    stochastic = report["mode_jumping_factor_and_composition_summary"]
    assert stochastic["terminal_class"] == "blocked"
    assert stochastic["approximate_error_finite_and_strictly_positive"] is True
    assert stochastic["identity_exact_table_zero_error"] is True
    assert stochastic["mode_jump_joint_tv"] == 0.03
    assert stochastic["bound_violation_count"] == 0
    assert stochastic["hardware_execution_claimed"] is False
    assert stochastic["latency_power_energy_and_speedup_claimed"] is False

    arc = report["arc_multiseed_no_solve_summary"]
    assert arc["terminal_class"] == "positive"
    assert arc["game_count"] == 6
    assert arc["seed_count"] == 3
    assert arc["per_arm_triggered_decision_counts"] == {"global": 144, "task_aware": 144}
    assert arc["solve_claimed"] is False
    assert arc["level_credit_delta"] == 0
    assert arc["registry_levels_unchanged"] is True
    assert arc["offline_ground_truth_bfs"] is False
    assert arc["used_game_source"] is False

    substrate = report["oracle_distinctness_and_inference_substrate_matrix"]
    assert (
        substrate["rows_by_task_id"]["exp6166-mode-jumping-factor-thermalization"][
            "software_simulation_promoted_to_hardware"
        ]
        is False
    )
    assert (
        substrate["rows_by_task_id"]["exp6167-arc-task-aware-multiseed-replication"][
            "solve_claimed"
        ]
        is False
    )


def test_scenario_report_6168_schema_history_and_run(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6168-SCHEMA-HISTORY: schema, checksum, and history are stable."""

    report = _build(tmp_path)

    assert mod.validate_report(report) == []
    assert report["research_complete_append_count"] == 0
    assert report["duplicate_history_amplification_count"] == 0
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["preexisting_worktree_changes_preserved"]["preserved"] is True
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert (
        report["spec_bmad_ops_reference_and_completion_reconciliation"][
            "ops_status_changelog_traceability_update"
        ]
        == "deferred_to_conductor_per_stop_when_done"
    )

    written = mod.run(
        tmp_path,
        adversarial_receipts=_receipts(),
        tests_run=report["test_exit_codes"],
        duration_s=1.75,
    )
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert yaml.safe_load((tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text()) == {
        "milestones": []
    }
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)


def test_req_report_6168_defensive_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6168: malformed evidence does not become success."""

    missing_payload, missing_meta = mod._read_json_mapping(tmp_path / "missing.json")
    assert missing_payload == {}
    assert missing_meta["error"] == "missing"
    _write_text(tmp_path, "bad.json", "{")
    assert mod._read_json_mapping(tmp_path / "bad.json")[1]["error"].startswith("json_error:")
    _write_text(tmp_path, "array.json", "[]")
    assert mod._read_json_mapping(tmp_path / "array.json")[1]["error"] == "json_not_mapping"
    assert mod._read_yaml_mapping(tmp_path / "missing.yaml") == {}

    assert mod._latest_conductor_receipt("", "missing title") == {
        "present": False,
        "status": None,
        "line": None,
        "detail": None,
    }
    assert mod._ignored_same_number_aliases(tmp_path, "exp6157-any", Path("results/x.json")) == []
    assert mod._terminal_marker("retired: done") == "retired"
    assert mod._terminal_marker("complete_ready: ok") == "positive"
    assert mod._terminal_marker("unknown") is None
    assert mod._terminal_class({}, False, {"status": "GATE_BLOCK"}) == ("skipped", "skipped")
    assert mod._terminal_class({}, False, {"status": "FAIL"}) == ("missing", "missing")
    assert mod._terminal_class({"retirement_triggered": True}, True, {}) == (
        "retired",
        "retired",
    )
    assert mod._terminal_class({"status": "complete_partial"}, True, {}) == (
        "partial",
        "partial",
    )
    assert mod._normalize_tests(None)[0] == list(mod.DEFAULT_TEST_COMMANDS)
    assert mod._receipt_report({}) == {"flag_count": 0, "flags": [], "max_severity": -1}
    assert mod._receipt_report({"stdout_json": {"flagged_count": 2}})["flag_count"] == 2
    assert mod._normalize_adversarial_receipts([{"task_id": "expX"}])["expX"]["task_id"] == "expX"
    assert mod._mapping_or_empty({"a": 1}) == {"a": 1}
    assert mod._mapping_or_empty(["not", "mapping"]) == {}
    assert mod._history_duplicate_count(tmp_path, mod.MILESTONE) == 0
    _write_text(
        tmp_path,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump({"milestones": [{"id": mod.MILESTONE}, {"id": mod.MILESTONE}]}),
    )
    assert mod._history_duplicate_count(tmp_path, mod.MILESTONE) == 1

    (tmp_path / ".git").mkdir()

    class _Proc:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: _Proc(0, stdout=" M changed.py\n"),
    )
    assert mod._git_status_short(tmp_path) == [" M changed.py"]
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: _Proc(2, stderr="boom"),
    )
    assert mod._git_status_short(tmp_path) == ["git_status_error:boom"]

    _make_root(tmp_path)
    report = mod.build_report(
        tmp_path,
        adversarial_receipts=_receipts(),
        tests_run={".venv/bin/pytest tests/python -q": 0},
        duration_s=None,
    )
    assert report["fresh_stream_and_sota_corpus_summary"]["sota_model_count"] == 2
    assert "field_provenance:not_mapping" in mod.validate_report({})
    broken = dict(report)
    broken["field_provenance"] = dict(report["field_provenance"])
    broken["field_provenance"]["status"] = {"principle": "wrong"}
    broken["reproducibility_checksum"] = mod.payload_checksum(broken)
    assert "field_provenance:status" in mod.validate_report(broken)

    monkeypatch.setattr(mod, "validate_report", lambda _report: ["forced"])
    with pytest.raises(ValueError, match="invalid Exp6168 report"):
        mod.run(
            tmp_path,
            adversarial_receipts=_receipts(),
            tests_run={".venv/bin/pytest tests/python -q": 0},
            duration_s=1.0,
        )
