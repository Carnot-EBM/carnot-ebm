"""Tests for Exp6196 V536 capstone reconciliation.

Spec refs: REQ-CAPSTONE-6196, SCENARIO-CAPSTONE-6196,
SCENARIO-CAPSTONE-6196-BRANCH-INDEPENDENCE,
SCENARIO-CAPSTONE-6196-TERMINAL-CLASS-PRESERVATION,
SCENARIO-CAPSTONE-6196-ADVERSARIAL-VERIFY-AND-CHECKSUM,
SCENARIO-CAPSTONE-6196-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_6196_v536_capstone_reconciliation as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _write_text(root: Path, rel_path: Path | str, text: str = "fixture\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    _write_text(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _artifact(task_id: str) -> JsonDict:
    payloads: dict[str, JsonDict] = {
        "exp6183-v536-transition": {
            "status": "running_bootstrap",
            "honest_verdict": "blocked: bootstrap only; artifact survived initial write",
            "inference_substrate": "deterministic_repository_transition",
            "bootstrap_artifact_receipt": {"survived": True},
        },
        "exp6184-v536-evidence-isolation-preflight": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: task-scoped preflight ready_score=1",
            "inference_substrate": "deterministic_task_scoped_repository_test_isolation",
            "repository_wide_closure_claimed": False,
            "v536_task_artifact_isolation_ready_score": 1,
            "isolation_violation_count": 0,
        },
        "exp6185-v536-post-marker-source-delta": {
            "status": "complete",
            "honest_verdict": "complete_null: accepted_count=0; references unchanged",
            "inference_substrate": "dated_primary_secondary_source_ingestion",
            "zero_delta_accepted": True,
            "candidate_and_deduplicated_record_counts": {"accepted_count": 0},
            "reference_hash_before_after_and_append_count": {"append_count": 0},
        },
        "exp6186-livecodebench-bank-preregistration": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: bank unique_ids=120",
            "inference_substrate": "deterministic_cached_livecodebench_bank_preregistration",
            "bank_ready_score": 1,
            "candidate_and_model_access_count": 0,
            "private_test_access_control_receipt": {"prompts_contain_private_tests": False},
            "split_overlap_matrix": {"max_overlap": 0},
        },
        "exp6187-livecodebench-authentic-k8-pool": {
            "status": "complete_partial",
            "honest_verdict": "complete_partial: retained 576 samples",
            "inference_substrate": (
                "local_llama_cpp_cuda_gguf_plus_restricted_private_test_execution"
            ),
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "pool_integrity_ready_score": 0,
            "raw_before_label_checkpoint_paths_hashes_and_timestamps": {
                "raw_rows_complete_before_validation": True,
                "label_sidecar_write_count_before_raw_commit": 0,
                "private_test_open_count_before_raw_commit": 0,
            },
            "private_test_noninterference_receipt": {
                "generation_prompt_private_test_access_count": 0,
                "selector_input_private_test_access_count": 0,
                "private_material_found_in_generation_surfaces": False,
            },
            "model_cache_file_hash_revision_quantization_and_template": {
                "headline_model_id": "unsloth/gemma-4-31B-it-GGUF",
                "revision": "fixture-rev",
                "sha256": "sha256:6187",
                "quantization": "Q4_K_M",
            },
            "dual_gpu_utilization_memory_intervals": {"both_gpus_observed": True},
            "correctness_retry_count": 0,
            "verifier_is_oracle": {"generation_or_selection_inputs": False},
        },
        "exp6188-livecodebench-headroom-audit": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [
                {
                    "upstream": "exp6187-livecodebench-authentic-k8-pool",
                    "artifact_field": "pool_integrity_ready_score",
                    "expected": 1.0,
                    "actual": 0,
                    "passed": False,
                }
            ],
        },
        "exp6190-calibration-clue-linear-code-selector": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [
                {
                    "upstream": "exp6189-matching-base-code-hidden-state-surface",
                    "artifact_field": "surface_ready_score",
                    "expected": 1.0,
                    "actual": None,
                    "passed": False,
                }
            ],
        },
        "exp6192-live-strategy-seed-stream": {
            "status": "complete_partial",
            "honest_verdict": "complete_partial: seed stream sealed",
            "inference_substrate": (
                "local_dual_family_llama_cpp_cuda_live_generation_plus_restricted_execution"
            ),
            "seed_stream_ready_score": 0,
            "model_specs": [
                {
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "revision": "qwen-rev",
                    "sha256": "sha256:qwen",
                    "cuda_offload_authenticated": True,
                    "weight_mutation_allowed": False,
                },
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "revision": "gemma-rev",
                    "sha256": "sha256:gemma",
                    "cuda_offload_authenticated": True,
                    "weight_mutation_allowed": False,
                },
            ],
            "raw_before_label_checkpoint_hashes_and_timestamps": {
                "raw_rows_complete_before_validation": True,
                "label_sidecar_write_count_before_raw_commit": 0,
                "private_test_open_count_before_raw_commit": 0,
            },
            "private_test_noninterference_receipt": {
                "generation_prompt_private_test_access_count": 0,
                "strategy_choice_private_test_access_count": 0,
                "private_material_found_in_generation_surfaces": False,
            },
            "bounded_memory_schema_capacity_eviction_and_snapshot_receipt": {
                "bounded": True,
                "append_only_event_log": True,
                "snapshot_read_receipt": {"read_mutated_state": False},
                "post_outcome_commit_receipt": {"all_commits_after_outcome": True},
            },
            "poison_rollback_and_retention_fixture_receipts": {
                "poison_propagation_count": 0,
                "rollback_exact": True,
                "retention_probe_mutated_state": False,
            },
            "model_cache_hash_revision_quantization_template_and_cuda_receipts": {
                "model_weight_immutability_receipt": {
                    "all_unchanged": True,
                    "weight_update_count": 0,
                }
            },
            "correctness_retry_count": 0,
        },
        "exp6193-prospective-continuous-strategy-learning-ab": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [
                {
                    "upstream": "exp6192-live-strategy-seed-stream",
                    "artifact_field": "seed_stream_ready_score",
                    "expected": 1.0,
                    "actual": 0,
                    "passed": False,
                }
            ],
        },
        "exp6194-mode-jump-rust-pyo3-parity": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: exact short-chain parity true",
            "inference_substrate": "local_cpu_rust_pyo3_cross_runtime_sampler_parity",
            "mode_jump_rust_pyo3_ready_score": 1.0,
            "hardware_or_speedup_claimed": False,
            "exact_transition_fixture_hash_and_parity_matrix": {
                "all_fields_match": True,
                "mismatch_count": 0,
            },
            "distribution_frequency_tv_kl_metrics": {"distribution_pass": True},
            "nonzero_command_classification": [
                {"command": ".venv/bin/pytest tests/python -q", "exit_code": 2}
            ],
        },
        "exp6195-arc-task-aware-prospective-fresh-transition": {
            "status": "complete_positive",
            "honest_verdict": "complete_positive: fresh transitions no solve",
            "inference_substrate": (
                "submitted_live_agent_kernel_acquisition_plus_offline_frozen_policy_replay"
            ),
            "solve_claimed": False,
            "level_credit_claimed": False,
            "arc_solve_registry_delta": [],
            "solve_provenance": "live_agent_self_discovery",
            "fresh_live_agent_owned_transition_path_hash_count_and_provenance": {
                "all_rows_live_agent_owned": True,
                "transition_count": 48,
            },
            "forbidden_source_bfs_adapter_prior_game_hidden_state_access_counts": {
                "adapter_route_count": 0,
                "hidden_state_access_count": 0,
                "llm_invocation_count": 0,
                "offline_ground_truth_bfs_count": 0,
            },
            "global_and_task_aware_proposal_quality_metrics": {
                "task_aware_minus_global": 0.208333
            },
            "registry_precheck_and_hash": {"registry_update_permitted": False},
        },
    }
    return payloads[task_id]


def _roadmap_payload() -> JsonDict:
    tasks = []
    for task_id, title, rel_path in mod.DECLARED_TASKS:
        row: JsonDict = {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": title,
            "deliverable": rel_path.as_posix(),
        }
        if task_id in mod.GATED_ON:
            row["gated_on"] = deepcopy(mod.GATED_ON[task_id])
        tasks.append(row)
    tasks.append(
        {
            "id": mod.EXPERIMENT_ID,
            "milestone": mod.MILESTONE,
            "title": "V536 capstone",
            "deliverable": mod.RESULT_RELATIVE_PATH.as_posix(),
            "requires": [task_id for task_id, _title, _rel_path in mod.DECLARED_TASKS],
        }
    )
    return {"milestone": mod.MILESTONE, "tasks": tasks}


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-08-07 10:26 UTC | Minimal exact terminal-boundary handoff from .535  | OK | cache hit |",
            "| 2026-08-07 13:04 UTC | Task-scoped .536 evidence-isolation preflight with | OK | cache hit |",
            "| 2026-08-07 13:20 UTC | Reliable dated evidence refresh after the V536 pla | OK | 86 passed |",
            "| 2026-08-07 13:40 UTC | Frozen LiveCodeBench bank and private-test boundar | OK | 153 passed |",
            "| 2026-08-07 15:03 UTC | Authentic Gemma-4-31B executable K=8 code pool gat | FLAGGED | adversarial_verify CRITICAL |",
            "| 2026-08-07 15:09 UTC | Executable-code competence and oracle-headroom aud | GATE_BLOCK | pool_integrity_ready_score failed |",
            "| 2026-08-07 15:15 UTC | Matching-base code hidden-state surface gated on E | GATE_BLOCK | Pre-emptive skip |",
            "| 2026-08-07 15:15 UTC | Calibration-only CLUE and residualized linear code | GATE_BLOCK | surface_ready_score failed |",
            "| 2026-08-07 15:52 UTC | One-shot held executable-code internal-state selec | GATE_BLOCK | Pre-emptive skip |",
            "| 2026-08-07 15:48 UTC | Live dual-family strategy seed stream gated on Exp | OK | 87 passed |",
            "| 2026-08-07 15:54 UTC | Prospective retention-safe continuous strategy lea | GATE_BLOCK | seed_stream_ready_score failed |",
            "| 2026-08-07 16:33 UTC | Fixed mode-jump sampler Rust/PyO3 correctness and  | OK | 87 passed |",
            "| 2026-08-07 16:54 UTC | Prospective fresh-transition generalization of the | OK | 88 passed |",
        ]
    )


def _make_root(root: Path) -> None:
    missing = {
        "exp6189-matching-base-code-hidden-state-surface",
        "exp6191-held-code-internal-state-selection",
    }
    for task_id, _title, rel_path in mod.DECLARED_TASKS:
        if task_id not in missing:
            _write_json(root, rel_path, _artifact(task_id))
    _write_json(
        root,
        "results/experiment_6189_sidecar.json",
        {"status": "complete", "honest_verdict": "complete: must be ignored"},
    )
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_roadmap_payload()))
    _write_text(root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones:\n- id: 2026.08.535\n")
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    _write_text(root, mod.ARC_REGISTRY_RELATIVE_PATH, "schema_version: 1\n")
    for rel_path in mod.PROTECTED_FILE_PATHS + mod.PRECONDITION_CONTEXT_PATHS:
        if not (root / rel_path).exists():
            _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\n")


def _receipt(task_id: str, rel_path: Path, flag_count: int = 0) -> JsonDict:
    flags = [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}] if flag_count else []
    report = {
        "artifact": rel_path.as_posix(),
        "loaded": True,
        "flag_count": flag_count,
        "flags": flags,
        "max_severity": 5 if flag_count else -1,
    }
    return {
        "task_id": task_id,
        "artifact_path": rel_path.as_posix(),
        "command": f".venv/bin/python scripts/adversarial_verify.py --json {rel_path.as_posix()}",
        "exit_code": 1 if flag_count else 0,
        "stdout_json": {"reports": [report], "flagged_count": flag_count},
    }


def _receipts() -> dict[str, JsonDict]:
    out: dict[str, JsonDict] = {}
    for task_id, _title, rel_path in mod.DECLARED_TASKS:
        if task_id in {
            "exp6189-matching-base-code-hidden-state-surface",
            "exp6191-held-code-internal-state-selection",
        }:
            continue
        out[task_id] = _receipt(
            task_id,
            rel_path,
            flag_count=1 if task_id == "exp6187-livecodebench-authentic-k8-pool" else 0,
        )
    return out


def _build(root: Path) -> JsonDict:
    _make_root(root)
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        determination_receipt={"command": "det", "exit_code": 0, "violations": []},
        tests_run={".venv/bin/pytest tests/python/test_experiment_6196_v536_capstone_reconciliation.py -q --no-cov -n 0": 0},
        duration_s=2.0,
    )


def test_req_capstone_6196_spec_declares_exact_reconciliation_contract() -> None:
    """REQ-CAPSTONE-6196: OpenSpec names exact V536 capstone behavior."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-CAPSTONE-6196") :]

    for marker in (
        "REQ-CAPSTONE-6196",
        "SCENARIO-CAPSTONE-6196",
        "SCENARIO-CAPSTONE-6196-BRANCH-INDEPENDENCE",
        "SCENARIO-CAPSTONE-6196-TERMINAL-CLASS-PRESERVATION",
        "SCENARIO-CAPSTONE-6196-ADVERSARIAL-VERIFY-AND-CHECKSUM",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "Exp6183 through Exp6195",
        "deterministic_exact_path_capstone_reconciliation",
        "without modifying `scripts/research_conductor.py`",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_capstone_6196_preserves_exact_terminal_classes(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6196: exact paths produce the expected terminal matrix."""

    report = _build(tmp_path)

    assert report["status"] == "complete_partial_reconciliation"
    assert report["honest_verdict"].startswith("complete_partial:")
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE

    classes = {
        task_id: row["terminal_class"]
        for task_id, row in report["per_task_honest_verdict_and_terminal_class"].items()
    }
    assert classes == {
        "exp6183-v536-transition": "bootstrap_only",
        "exp6184-v536-evidence-isolation-preflight": "positive",
        "exp6185-v536-post-marker-source-delta": "null",
        "exp6186-livecodebench-bank-preregistration": "positive",
        "exp6187-livecodebench-authentic-k8-pool": "flagged",
        "exp6188-livecodebench-headroom-audit": "gated",
        "exp6189-matching-base-code-hidden-state-surface": "skipped",
        "exp6190-calibration-clue-linear-code-selector": "gated",
        "exp6191-held-code-internal-state-selection": "skipped",
        "exp6192-live-strategy-seed-stream": "partial",
        "exp6193-prospective-continuous-strategy-learning-ab": "gated",
        "exp6194-mode-jump-rust-pyo3-parity": "positive",
        "exp6195-arc-task-aware-prospective-fresh-transition": "positive",
    }
    exact = report["exact_path_existence_hash_and_conductor_receipt_matrix"]
    assert exact["exp6189-matching-base-code-hidden-state-surface"]["present"] is False
    assert exact["exp6189-matching-base-code-hidden-state-surface"]["same_number_alias_used"] is False
    assert exact["exp6189-matching-base-code-hidden-state-surface"]["same_number_alias_candidates_ignored"] == [
        "results/experiment_6189_sidecar.json"
    ]
    preservation = (
        report[
            "missing_bootstrap_null_partial_flagged_blocked_retired_gated_skipped_positive_software_proxy_and_no_solve_preservation_matrix"
        ]
    )
    assert preservation["exp6183-v536-transition"]["bootstrap_only"] is True
    assert preservation["exp6187-livecodebench-authentic-k8-pool"]["partial"] is True
    assert preservation["exp6187-livecodebench-authentic-k8-pool"]["flagged"] is True
    assert preservation["exp6194-mode-jump-rust-pyo3-parity"]["software_proxy"] is True
    assert preservation["exp6195-arc-task-aware-prospective-fresh-transition"]["no_solve"] is True


def test_scenario_capstone_6196_branch_independence_and_boundaries(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6196-BRANCH-INDEPENDENCE: gates do not bleed across branches."""

    report = _build(tmp_path)

    branch = report["branch_independence_receipt"]
    assert branch["principle"] == mod.FIELD_PRINCIPLES["branch_independence_receipt"]
    assert branch["code_selector_gate_suppresses_other_branches"] is False
    assert branch["source_branch_preserved"] is True
    assert branch["csl_branch_preserved"] is True
    assert branch["sampler_branch_preserved"] is True
    assert branch["arc_branch_preserved"] is True

    promotions = report["promotion_retirement_and_exclusion_matrix"]
    assert promotions["code_selector"]["outcome"] == "retired_or_skipped_only_code_selector_descendants"
    assert promotions["continuous_learning"]["outcome"] == "seed_partial_prospective_gated"
    assert promotions["sampler_parity"]["outcome"] == "software_parity_promoted_no_hardware"
    assert promotions["arc"]["outcome"] == "fresh_transition_positive_no_solve_no_registry_delta"
    assert promotions["source_delta"]["outcome"] == "complete_null_zero_delta"

    raw = report["raw_before_label_private_test_selector_freeze_and_transaction_order_audit"]
    assert raw["exp6187"]["raw_before_label"] is True
    assert raw["exp6187"]["private_test_leakage_detected"] is False
    assert raw["exp6192"]["transaction_order_preserved"] is True

    csl = report["continuous_learning_retention_lifecycle_and_immutable_weight_audit"]
    assert csl["exp6192_seed_stream_ready_score"] == 0
    assert csl["prospective_exp6193_terminal_class"] == "gated"
    assert csl["model_weights_immutable"] is True
    assert csl["poison_propagation_count"] == 0

    parity = report["rust_pyo3_parity_and_no_hardware_claim_audit"]
    assert parity["ready_score"] == 1.0
    assert parity["hardware_or_speedup_claimed"] is False

    arc = report["arc_live_path_solve_provenance_and_registry_delta_audit"]
    assert arc["solve_claimed"] is False
    assert arc["registry_delta_count"] == 0
    assert arc["no_solve_preserved"] is True


def test_scenario_capstone_6196_write_checksum_and_validate(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6196-ADVERSARIAL-VERIFY-AND-CHECKSUM: output is stable."""

    _make_root(tmp_path)
    report = mod.write_capstone(
        root=tmp_path,
        adversarial_receipts=_receipts(),
        determination_receipt={"command": "det", "exit_code": 0, "violations": []},
        tests_run={".venv/bin/pytest tests/python/test_experiment_6196_v536_capstone_reconciliation.py -q --no-cov -n 0": 0},
        duration_s=2.0,
    )

    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert mod.validate_report(report) == []
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert report["adversarial_verify_commands_exit_codes_and_flags"]["flagged_task_ids"] == [
        "exp6187-livecodebench-authentic-k8-pool"
    ]
    assert report["determination_preservation_receipt"]["exit_code"] == 0
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["preexisting_worktree_changes_preserved"]["preserved"] is True
    assert report["openspec_traceability_status_and_changelog_reconciliation"][
        "ops_status_changelog_traceability_modified"
    ] is False


def test_req_capstone_6196_defensive_validation_and_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-6196: defensive helpers do not launder broken evidence."""

    report = _build(tmp_path)
    missing_payload, missing_meta = mod._read_json_mapping(tmp_path / "missing.json")
    assert missing_payload == {}
    assert missing_meta["error"] == "missing"
    _write_text(tmp_path, "bad.json", "{")
    assert mod._read_json_mapping(tmp_path / "bad.json")[1]["error"].startswith("json_error:")
    _write_text(tmp_path, "array.json", "[]")
    assert mod._read_json_mapping(tmp_path / "array.json")[1]["error"] == "json_not_mapping"
    assert mod._read_yaml_mapping(tmp_path / "missing.yaml") == {}
    _write_text(tmp_path, "not_mapping.yaml", "[]\n")
    assert mod._read_yaml_mapping(tmp_path / "not_mapping.yaml") == {}
    assert mod._latest_conductor_receipt("", "not present") == {
        "present": False,
        "status": None,
        "line": None,
        "detail": None,
    }
    assert (
        mod._ignored_same_number_aliases(
            tmp_path / "no-results", "exp6189-any", Path("results/x.json")
        )
        == []
    )
    assert mod._sidecar_candidates(tmp_path / "no-results", Path("results/x.json")) == []
    _write_text(tmp_path, "results/experiment_6196_probe.extra.json", "{}")
    assert mod._sidecar_candidates(tmp_path, Path("results/experiment_6196_probe.json")) == [
        "results/experiment_6196_probe.extra.json"
    ]
    assert mod._terminal_marker("complete_partial: ok") == "partial"
    assert mod._terminal_marker("running_bootstrap") == "bootstrap_only"
    assert mod._terminal_marker("retired: ok") == "retired"
    assert mod._terminal_marker("complete: ok") == "positive"
    assert mod._terminal_marker("unknown") is None
    assert mod._terminal_class({}, False, {"status": "GATE_BLOCK"}) == "skipped"
    assert mod._terminal_class({"status": "blocked"}, True, {"status": "GATE_BLOCK"}) == "gated"
    assert mod._terminal_class({"retirement_triggered": True}, True, {}) == "retired"
    assert mod._terminal_class({"status": "blocked", "gates_evaluated": [1]}, True, {}) == "gated"
    assert mod._terminal_class({"honest_verdict": "complete_ready: ok"}, True, {}) == "positive"
    assert mod._terminal_class({}, True, {}) == "partial"
    assert mod._terminal_class({"status": "complete_ready"}, True, {}) == "positive"
    assert mod._normalize_tests(None)[1]
    assert mod._normalize_tests([{"command": "cmd", "exit_code": 7}]) == (["cmd"], {"cmd": 7})
    assert mod._receipt_report({}) == {"flag_count": 0, "flags": [], "max_severity": -1}
    assert mod._receipt_report({"stdout_json": {"flagged_count": 2}})["flag_count"] == 2
    assert mod._normalize_adversarial_receipts([{"task_id": "expX"}])["expX"]["task_id"] == (
        "expX"
    )

    monkeypatch.setattr(mod, "_run_live_adversarial_receipts", lambda _root, _paths: _receipts())
    live_report = mod.build_report(
        tmp_path,
        adversarial_receipts=None,
        determination_receipt={"command": "det", "exit_code": 0, "violations": []},
        tests_run={".venv/bin/pytest tests/python -q": 0},
        duration_s=2.0,
    )
    assert live_report["adversarial_verify_commands_exit_codes_and_flags"]["flagged_task_ids"] == [
        "exp6187-livecodebench-authentic-k8-pool"
    ]

    broken = deepcopy(report)
    broken.pop("status")
    assert "missing:status" in mod.validate_report(broken)

    bad_checksum = deepcopy(report)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum" in mod.validate_report(bad_checksum)

    bad_substrate = deepcopy(report)
    bad_substrate["inference_substrate"] = "live_gpu"
    bad_substrate["reproducibility_checksum"] = mod.payload_checksum(bad_substrate)
    assert "inference_substrate" in mod.validate_report(bad_substrate)

    bad_branch = deepcopy(report)
    bad_branch["branch_independence_receipt"]["code_selector_gate_suppresses_other_branches"] = True
    bad_branch["reproducibility_checksum"] = mod.payload_checksum(bad_branch)
    assert "branch_independence_receipt" in mod.validate_report(bad_branch)

    no_provenance = deepcopy(report)
    no_provenance["field_provenance"] = []
    no_provenance["reproducibility_checksum"] = mod.payload_checksum(no_provenance)
    assert "field_provenance:not_mapping" in mod.validate_report(no_provenance)

    bad_provenance = deepcopy(report)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.payload_checksum(bad_provenance)
    assert "field_provenance:status" in mod.validate_report(bad_provenance)

    bad_protected = deepcopy(report)
    bad_protected["protected_files_unchanged"]["all_unchanged"] = False
    bad_protected["reproducibility_checksum"] = mod.payload_checksum(bad_protected)
    assert "protected_files_unchanged" in mod.validate_report(bad_protected)

    bad_arc = deepcopy(report)
    bad_arc["arc_live_path_solve_provenance_and_registry_delta_audit"]["solve_claimed"] = True
    bad_arc["reproducibility_checksum"] = mod.payload_checksum(bad_arc)
    assert "arc_no_solve_preservation" in mod.validate_report(bad_arc)

    bad_docs = deepcopy(report)
    bad_docs["openspec_traceability_status_and_changelog_reconciliation"][
        "ops_status_changelog_traceability_modified"
    ] = True
    bad_docs["reproducibility_checksum"] = mod.payload_checksum(bad_docs)
    assert "ops_status_changelog_traceability_modified" in mod.validate_report(bad_docs)

    bad_verdict = deepcopy(report)
    bad_verdict["honest_verdict"] = "done"
    bad_verdict["reproducibility_checksum"] = mod.payload_checksum(bad_verdict)
    assert "honest_verdict_prefix" in mod.validate_report(bad_verdict)

    monkeypatch.setattr(mod, "validate_report", lambda _report: ["forced"])
    with pytest.raises(ValueError, match="invalid Exp6196 capstone"):
        mod.write_capstone(
            root=tmp_path,
            adversarial_receipts=_receipts(),
            determination_receipt={"command": "det", "exit_code": 0, "violations": []},
        )
