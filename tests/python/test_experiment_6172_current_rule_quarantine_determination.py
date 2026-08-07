"""Tests for Exp6172 current-rule quarantine companion determination.

Spec refs: REQ-REPORT-6172,
SCENARIO-REPORT-6172-IMMUTABLE-SOURCE,
SCENARIO-REPORT-6172-CURRENT-RULE-REPLAY,
SCENARIO-REPORT-6172-DURATION-PROVENANCE,
SCENARIO-REPORT-6172-OPERATOR-BOUNDARY,
SCENARIO-REPORT-6172-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6172_current_rule_quarantine_determination as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_text(root: Path, rel_path: str | Path, text: str = "fixture\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: str | Path, payload: object) -> None:
    _write_text(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _exp6161() -> dict[str, object]:
    return {
        "experiment_id": "experiment_6161_decision_calibrated_energy_policy",
        "status": "complete_ready",
        "honest_verdict": "complete_ready: decision policy frozen",
        "inference_substrate": "cached_authentic_sota_rows_cpu_analysis",
        "duration_s": 0.169107,
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": "duration_s=0.169107 but artifact references compute-bound markers; >=60.0s minimum",
            },
            {
                "kind": "METHODOLOGY_MISSING",
                "severity": "warn",
                "detail": "Compute-bound artifact missing: model_specs/target_model.",
            },
        ],
        "held_access_count": 0,
        "decision_calibrated_policy_ready_score": 1.0,
        "selected_policy_rationale_without_held_access": {
            "selected_arm": "decision_calibrated_task_energy",
            "selected_threshold": 0.13683932616272387,
            "selection_uses_held_outcomes": False,
            "policy_validly_frozen": True,
            "selected_cv_utility_per_row": 1.1875,
        },
        "score_threshold_abstention_and_cost_freeze_receipts": {
            "selected_arm": "decision_calibrated_task_energy",
            "threshold": 0.13683932616272387,
            "held_access_count_at_freeze": 0,
            "frozen_before_held_access": True,
        },
        "per_model_calibration_cost_brier_ece_unsafe_safe_and_descriptive_ranking_metrics": {
            "pooled_after_per_model": {
                "decision_calibrated_task_energy": {
                    "utility_per_row": 1.1875,
                    "brier": 0.0002509335390337641,
                    "ece": 0.015122811438360095,
                    "false_unsafe_admission_rate": 0.0,
                }
            }
        },
        "upstream_endpoint_row_and_control_hashes": {
            "exp6160": {
                "row_sidecars": {
                    "unsloth/Qwen3.6-35B-A3B-GGUF": {
                        "path": "results/experiment_6160_sota_decision_calibration_corpus.qwen3_6_35b_a3b.rows.jsonl",
                        "row_count": 240,
                        "sha256": "sha256:" + "1" * 64,
                        "partition_counts": {
                            "calibration": 96,
                            "future_known": 64,
                            "shifted_family_held": 80,
                        },
                    },
                    "unsloth/gemma-4-26B-A4B-it-GGUF": {
                        "path": "results/experiment_6160_sota_decision_calibration_corpus.gemma_4_26b_a4b_it.rows.jsonl",
                        "row_count": 240,
                        "sha256": "sha256:" + "2" * 64,
                        "partition_counts": {
                            "calibration": 96,
                            "future_known": 64,
                            "shifted_family_held": 80,
                        },
                    },
                }
            }
        },
        "protected_files_unchanged": {"unchanged": True},
        "field_provenance": {"duration_s": ["fixture"]},
        "test_commands": ["unit"],
        "test_exit_codes": {"unit": 0},
        "reproducibility_checksum": "sha256:" + "a" * 64,
    }


def _exp6162() -> dict[str, object]:
    return {
        "experiment_id": "experiment_6162_prospective_admission_replication",
        "status": "complete_positive",
        "honest_verdict": "complete_positive: both models pass",
        "inference_substrate": "sealed_cached_event_evaluation",
        "duration_s": 0.37409,
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": "duration_s=0.37409 but artifact references compute-bound markers; >=60.0s minimum",
            },
            {
                "kind": "METHODOLOGY_MISSING",
                "severity": "warn",
                "detail": "Compute-bound artifact missing: model_specs/target_model.",
            },
        ],
        "first_and_only_held_access_receipt": {
            "held_access_count_before": 0,
            "held_access_count_after": 1,
            "held_label_read_count": 288,
            "future_known_label_read_count": 128,
            "shifted_family_held_label_read_count": 160,
        },
        "prospective_admission_replication_ready_score": 1.0,
        "selector_and_threshold_refit_counts": {"all_zero": True},
        "per_model_and_conjunctive_gate_matrix": {
            "conjunctive_pass": True,
            "pooled_success_cannot_mask_model_or_partition_failure": True,
        },
        "unsafe_admission_and_known_family_noninferiority_gates": {
            "all_gates_pass": True,
            "unsafe_admission_margin": 0.02,
            "known_family_noninferiority_margin": 0.03,
        },
        "per_model_future_known_and_shifted_decision_utility_intervals": {
            "pooled_summary_after_per_model": {
                "future_known": {
                    "decision_calibrated_minus_global": {
                        "lower_95_above_zero": True,
                        "observed_per_row": 1.0,
                    }
                }
            }
        },
        "stream_rows_endpoint_policy_and_held_hashes": {
            "access_counters": {
                "held_access_count_before": 0,
                "held_access_count_after": 1,
            },
            "exp6161_policy": {
                "manifest": {
                    "manifest_matches_artifact": True,
                    "file_receipt": {
                        "path": "results/experiment_6161_decision_calibrated_energy_policy.manifest.json",
                        "sha256": "sha256:" + "3" * 64,
                    },
                }
            },
        },
        "retirement_triggered": False,
        "retirement_reason": "not_triggered_positive_replication",
        "protected_files_unchanged": {"unchanged": True},
        "field_provenance": {"duration_s": ["fixture"]},
        "test_commands": ["unit"],
        "test_exit_codes": {"unit": 0},
        "reproducibility_checksum": "sha256:" + "b" * 64,
    }


def _make_repo(root: Path) -> Path:
    for rel_path in (
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/known-issues.md",
        "ops/status.md",
        "scripts/determination_preservation_lint.py",
        "python/carnot/pipeline/deliverable_guard.py",
        "scripts/research_conductor.py",
    ):
        _write_text(root, rel_path)
    _write_text(
        root,
        "scripts/adversarial_verify.py",
        "TAUTOLOGY_DIGITS = 5\n"
        "COMPUTE_BOUND_MIN_DURATION_S = 60.0\n"
        "DETERMINISTIC_VERIFIER_MIN_DURATION_S = 0.0001\n"
        "HIGH_PRECISION_KINDS = ('DURATION_TOO_SHORT', 'GATE_PASSED_WITHOUT_DATA')\n"
        "def duration_floor_for_artifact(d):\n"
        "    return {'substrate': d.get('inference_substrate'), 'min_duration_s': 0.0001, 'reason': 'deterministic_verifier'}\n"
        "def check_duration_vs_claim(d, flags):\n"
        "    pass\n",
    )
    _write_json(root, mod.EXP6161_RELATIVE_PATH, _exp6161())
    _write_json(root, mod.EXP6162_RELATIVE_PATH, _exp6162())
    _write_json(
        root,
        mod.EXP6161_MANIFEST_RELATIVE_PATH,
        {"schema": "manifest", "selected_arm": "decision_calibrated_task_energy"},
    )
    _write_json(
        root,
        mod.EXP6160_RELATIVE_PATH,
        {
            "experiment_id": "experiment_6160_sota_decision_calibration_corpus",
            "status": "complete_ready",
            "honest_verdict": "complete_ready: live acquisition",
            "inference_substrate": "live_local_sota_gguf_cuda",
            "duration_s": 590.606661,
            "sota_decision_corpus_ready_score": 1.0,
            "per_model_row_paths_hashes_and_counts": {
                "total_row_count": 480,
                "per_model": {
                    "unsloth/Qwen3.6-35B-A3B-GGUF": {
                        "row_count": 240,
                        "sha256": "sha256:" + "1" * 64,
                    },
                    "unsloth/gemma-4-26B-A4B-it-GGUF": {
                        "row_count": 240,
                        "sha256": "sha256:" + "2" * 64,
                    },
                },
            },
            "gpu_offload_pid_lifecycle_and_cleanup_receipts": {
                "all_models_gpu_engaged": True,
                "all_models_release_ready": True,
                "orphan_task_owned_pid_count": 0,
            },
            "model_specs": [
                {
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "actual_use_count": 240,
                    "sha256": "sha256:" + "4" * 64,
                },
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "actual_use_count": 240,
                    "sha256": "sha256:" + "5" * 64,
                },
            ],
        },
    )
    _write_json(
        root,
        mod.EXP6159_RELATIVE_PATH,
        {
            "experiment_id": "experiment_6159_decision_calibrated_stream",
            "status": "complete_ready",
            "inference_substrate": "deterministic_verifier_plus_replay",
            "duration_s": 2.474093,
            "event_template_family_partition_and_shift_counts": {
                "event_count": 240,
                "partition_counts": {
                    "calibration": 96,
                    "future_known": 64,
                    "shifted_family_held": 80,
                },
            },
            "held_loader_one_shot_contract": {"held_access_count": 0},
        },
    )
    _write_json(
        root,
        mod.CAPSTONE_RELATIVE_PATH,
        {
            "status": "complete_with_blocks_missing_skips_and_quarantine",
            "honest_verdict": "complete: flagged decision artifacts preserved",
            "adversarial_verifier_and_quarantine_receipts": {
                "flagged_task_ids": [
                    "exp6161-decision-calibrated-energy-policy",
                    "exp6162-prospective-admission-replication",
                ]
            },
            "exact_terminal_classification": {
                "terminal_class_by_task_id": {
                    "exp6161-decision-calibrated-energy-policy": "flagged",
                    "exp6162-prospective-admission-replication": "flagged",
                },
                "underlying_terminal_class_by_task_id": {
                    "exp6161-decision-calibrated-energy-policy": "positive",
                    "exp6162-prospective-admission-replication": "positive",
                },
            },
        },
    )
    return root


def _clean_verifier_receipt() -> dict[str, object]:
    stdout_json = {
        "reports": [
            {
                "artifact": mod.EXP6161_RELATIVE_PATH.as_posix(),
                "loaded": True,
                "exp_id": "experiment_6161_decision_calibrated_energy_policy",
                "flag_count": 0,
                "max_severity": -1,
                "flags": [],
            },
            {
                "artifact": mod.EXP6162_RELATIVE_PATH.as_posix(),
                "loaded": True,
                "exp_id": "experiment_6162_prospective_admission_replication",
                "flag_count": 0,
                "max_severity": -1,
                "flags": [],
            },
        ],
        "flagged_count": 0,
    }
    stdout = json.dumps(stdout_json, sort_keys=True)
    return {
        "command": ".venv/bin/python scripts/adversarial_verify.py --json "
        f"{mod.EXP6161_RELATIVE_PATH.as_posix()} {mod.EXP6162_RELATIVE_PATH.as_posix()}",
        "started_at_utc": "2026-08-07T00:00:00Z",
        "finished_at_utc": "2026-08-07T00:00:01Z",
        "exit_code": 0,
        "stdout": stdout,
        "stderr": "",
        "stdout_sha256": mod.sha256_text(stdout),
        "parsed_json": stdout_json,
    }


def test_req_report_6172_spec_declares_companion_boundary() -> None:
    """REQ-REPORT-6172: OpenSpec names immutable companion requirements."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-6172") : spec.index("### REQ-REPORT-6168")]

    for marker in (
        "REQ-REPORT-6172",
        "SCENARIO-REPORT-6172-IMMUTABLE-SOURCE",
        "SCENARIO-REPORT-6172-CURRENT-RULE-REPLAY",
        "SCENARIO-REPORT-6172-DURATION-PROVENANCE",
        "SCENARIO-REPORT-6172-OPERATOR-BOUNDARY",
        "SCENARIO-REPORT-6172-SCHEMA",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6172_current_clean_does_not_unflag_source(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6172-CURRENT-RULE-REPLAY: clean replay is only current."""

    root = _make_repo(tmp_path)
    before_6161 = (root / mod.EXP6161_RELATIVE_PATH).read_bytes()
    before_6162 = (root / mod.EXP6162_RELATIVE_PATH).read_bytes()

    artifact = mod.build_artifact(
        root=root,
        verifier_receipt=_clean_verifier_receipt(),
        git_status_before="",
        git_status_after=" M results/experiment_6172_current_rule_quarantine_determination.json",
        git_head="fixture-head",
        duration_s=1.25,
        test_commands=["unit"],
        test_exit_codes={"unit": 0},
    )

    mod.validate_artifact(artifact)
    assert (root / mod.EXP6161_RELATIVE_PATH).read_bytes() == before_6161
    assert (root / mod.EXP6162_RELATIVE_PATH).read_bytes() == before_6162
    assert artifact["current_rule_clean"] is True
    assert artifact["historical_quarantine_preserved"] is True
    assert artifact["headline_promotion_authorized"] is False
    assert artifact["operator_reopen_required"] is True

    matrix = artifact["field_level_historical_vs_current_determination_matrix"]
    exp6161 = matrix["experiment_6161_decision_calibrated_energy_policy"]
    assert exp6161["source_flagged_adversarial"]["historical_value"] is True
    assert exp6161["source_flagged_adversarial"]["current_value"] is True
    assert exp6161["current_verifier_flag_count"]["current_value"] == 0
    assert exp6161["capstone_terminal_class"]["historical_value"] == "flagged"
    assert exp6161["capstone_underlying_class"]["historical_value"] == "positive"


def test_scenario_report_6172_duration_provenance_separates_acquisition_and_cache(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6172-DURATION-PROVENANCE: cached duration is not acquisition."""

    root = _make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=root,
        verifier_receipt=_clean_verifier_receipt(),
        git_status_before="",
        git_status_after="",
        git_head="fixture-head",
        duration_s=1.25,
    )

    duration = artifact["acquisition_duration_and_cached_analysis_duration_provenance"]
    assert duration["acquisition_receipt"]["source_experiment_id"] == (
        "experiment_6160_sota_decision_calibration_corpus"
    )
    assert duration["acquisition_receipt"]["duration_s"] == 590.606661
    assert (
        duration["cached_analysis_receipts"]["experiment_6161_decision_calibrated_energy_policy"][
            "duration_s"
        ]
        == 0.169107
    )
    assert (
        duration["cached_analysis_receipts"]["experiment_6162_prospective_admission_replication"][
            "current_duration_floor"
        ]["min_duration_s"]
        == 0.0001
    )
    assert duration["historical_duration_floor_s"] == 60.0
    assert duration["historical_rule_that_fired"] == "DURATION_TOO_SHORT"
    assert duration["current_rule_differs_because"] == (
        "top-level no-LLM cached substrates use deterministic-verifier duration floor"
    )

    model = artifact["model_lifecycle_and_held_access_receipts"]
    assert model["row_generation_receipts"]["total_row_count"] == 480
    assert model["model_lifecycle"]["all_models_gpu_engaged"] is True
    assert model["held_access_receipts"]["exp6161_policy_freeze_held_access_count"] == 0
    assert model["held_access_receipts"]["exp6162_held_access_count_after"] == 1


def test_scenario_report_6172_write_preserves_protected_source_hashes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6172-IMMUTABLE-SOURCE: companion write leaves sources fixed."""

    root = _make_repo(tmp_path)
    artifact = mod.build_and_write_artifact(
        root=root,
        verifier_receipt=_clean_verifier_receipt(),
        git_status_before="",
        git_head="fixture-head",
        duration_s=1.25,
        test_commands=["unit"],
        test_exit_codes={"unit": 0},
    )

    written = json.loads((root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert (
        artifact["source_hashes_and_git_status_before_after"]["all_source_hashes_unchanged"] is True
    )
    assert artifact["source_hashes_and_git_status_before_after"]["git_status_after"].startswith(
        "<git failed:"
    )
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["preexisting_worktree_changes_preserved"]["preserved"] is True


def test_scenario_report_6172_operator_boundary_validation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6172-OPERATOR-BOUNDARY: no-unflag booleans are enforced."""

    artifact = mod.build_artifact(
        root=_make_repo(tmp_path),
        verifier_receipt=_clean_verifier_receipt(),
        git_status_before="",
        git_status_after="",
        git_head="fixture-head",
        duration_s=1.25,
    )
    mod.validate_artifact(artifact)

    promoted = deepcopy(artifact)
    promoted["headline_promotion_authorized"] = True
    with pytest.raises(ValueError, match="headline_promotion_authorized"):
        mod.validate_artifact(promoted)

    aliased = deepcopy(artifact)
    aliased["historical_quarantine_preserved"] = False
    with pytest.raises(ValueError, match="historical_quarantine_preserved"):
        mod.validate_artifact(aliased)
