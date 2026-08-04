"""Tests for the Exp5973 V528 capstone reconciliation.

Spec refs: REQ-REPORT-5973,
SCENARIO-REPORT-5973-EXACT-MATRIX,
SCENARIO-REPORT-5973-GATES-AND-MISSING,
SCENARIO-REPORT-5973-BRANCH-INDEPENDENCE,
SCENARIO-REPORT-5973-VERIFIER-AND-SUBSTRATE,
SCENARIO-REPORT-5973-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5973_v528_capstone_reconciliation as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_text(root: Path, rel_path: Path | str, text: str = "fixture\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    _write_text(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _roadmap_payload() -> JsonDict:
    tasks: list[JsonDict] = []
    for task_id, title, rel_path in mod.UPSTREAM_TASKS:
        row: JsonDict = {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": title,
            "deliverable": rel_path.as_posix(),
        }
        if task_id == "exp5964-sota-atom-compatibility-corpus":
            row["gated_on"] = [
                {
                    "upstream": "exp5963-exact-atom-pair-fixture",
                    "artifact_field": "pair_fixture_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        if task_id == "exp5965-portable-atom-energy-ranker":
            row["gated_on"] = [
                {
                    "upstream": "exp5964-sota-atom-compatibility-corpus",
                    "artifact_field": "atom_compatibility_corpus_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        if task_id == "exp5966-discriminative-constraint-acquisition":
            row["gated_on"] = [
                {
                    "upstream": "exp5965-portable-atom-energy-ranker",
                    "artifact_field": "portable_atom_energy_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        if task_id == "exp5968-delayed-commit-csl-prospective":
            row["gated_on"] = [
                {
                    "upstream": "exp5967-delayed-commit-memory-fixture",
                    "artifact_field": "delayed_commit_fixture_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        if task_id == "exp5969-csl-poison-drift-abi-audit":
            row["gated_on"] = [
                {
                    "upstream": "exp5968-delayed-commit-csl-prospective",
                    "artifact_field": "prospective_csl_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        if task_id == "exp5971-arc-strip-swap-battery":
            row["gated_on"] = [
                {
                    "upstream": "exp5970-arc-strip-swap-sentinel",
                    "artifact_field": "strip_swap_sentinel_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        if task_id == "exp5962-v528-source-delta-ingestion":
            row["prior_failures"] = [
                {
                    "experiment_id": "exp5934-v527-source-delta-ingestion",
                    "verdict": "complete_null: no accepted post-V527 source deltas; references unchanged",
                    "retire_if_same_verdict": True,
                }
            ]
        tasks.append(row)
    tasks.append(
        {
            "id": mod.EXPERIMENT_ID,
            "milestone": mod.MILESTONE,
            "title": "Branch-independent .528 capstone and exact reconciliation",
            "deliverable": mod.RESULT_RELATIVE_PATH.as_posix(),
        }
    )
    return {
        "milestone": mod.MILESTONE,
        "milestone_title": mod.MILESTONE_TITLE,
        "milestone_doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "tasks": tasks,
    }


def _artifact(task_id: str) -> JsonDict:
    fixtures: dict[str, JsonDict] = {
        "exp5962-v528-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete_null: no accepted post-V528 source deltas; references unchanged",
            "inference_substrate": "aggregation_from_external_primary_sources",
            "accepted_rejected_abstained_findings": {"accepted": []},
        },
        "exp5963-exact-atom-pair-fixture": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: fixture ready",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "pair_fixture_ready_score": 1.0,
            "base_case_pair_and_class_counts": {"base_cases": 300, "pairs": 600},
        },
        "exp5964-sota-atom-compatibility-corpus": {
            "status": "blocked",
            "honest_verdict": "blocked: insufficient_free_vram",
            "inference_substrate": "live_llm_embedding_extraction",
            "atom_compatibility_corpus_ready_score": 0.0,
            "preconditions_checked": {"blocked_reasons": ["insufficient_free_vram"]},
            "model_specs": [{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}],
            "cuda_offload_vram_thermal_and_cleanup_receipts": {
                "cuda": {"available": True, "backend": "CUDA"},
                "all_models_cuda_offloaded": False,
            },
        },
        "exp5966-discriminative-constraint-acquisition": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "1 of 1 gate(s) failed",
            "gates_evaluated": [
                {
                    "upstream": "exp5965-portable-atom-energy-ranker",
                    "artifact_field": "portable_atom_energy_ready_score",
                    "op": "==",
                    "expected": 1.0,
                    "actual": None,
                    "passed": False,
                    "reason": "upstream artifact not found",
                }
            ],
        },
        "exp5967-delayed-commit-memory-fixture": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: delayed_commit_memory_fixture_ready",
            "inference_substrate": "deterministic_delayed_commit_transactional_replay_no_llm",
            "delayed_commit_fixture_ready_score": 1.0,
        },
        "exp5968-delayed-commit-csl-prospective": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: delayed_commit_prospective_csl_ready",
            "inference_substrate": "deterministic_delayed_commit_csl_prospective_no_llm",
            "prospective_csl_ready_score": 1.0,
            "gate_replay_receipt": {"gate_passed": True, "ready_score": 1.0},
            "paired_deltas_intervals_and_power": {"promotion_gate_passed": True},
            "unsafe_accept_count": 0,
            "immutable_model_weights_receipt": {"all_unchanged": True, "weight_update_count": 0},
        },
        "exp5969-csl-poison-drift-abi-audit": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: delayed_commit_csl_survives_poison_drift_abi_audit",
            "inference_substrate": "deterministic_csl_poison_drift_abi_audit_no_llm",
            "rollback_and_recovery_ready_score": 1.0,
            "unsafe_accept_count": 0,
            "gate_replay_receipt": {"gate_passed": True, "ready_score": 1.0},
        },
        "exp5970-arc-strip-swap-sentinel": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: strip-swap sentinel ready",
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
            "strip_swap_sentinel_ready_score": 1.0,
            "anchor_support_and_behavioral_validity": {"valid_live_support": True},
            "shipped_flag_and_registry_immutability": {
                "registry_unchanged": True,
                "policy_flags_modified_by_task": False,
            },
            "no_solve_credit_receipt": {"solve_credit_claimed": False},
        },
        "exp5971-arc-strip-swap-battery": {
            "status": "complete_null",
            "honest_verdict": "complete_null: original anchor support is empty for the shipped HUD contrast",
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
            "gate_replay_receipt": {"ready": True, "strip_swap_sentinel_ready_score": 1.0},
            "anchor_survival_and_discriminating_game_support": {
                "hud_given_frontier_on": {"discriminating_game_support": 0}
            },
            "convention_dependence_decision": {"status": "complete_null"},
            "no_solve_credit_receipt": {"solve_credit_claimed": False},
            "overall_hud_value_not_identified_receipt": {"flag_flip_recommended": False},
            "shipped_flag_and_registry_immutability": {
                "registry_unchanged": True,
                "policy_flags_modified_by_task": False,
            },
        },
        "exp5972-arc-llm-on-budget2000-feasibility": {
            "status": "complete_feasible",
            "honest_verdict": "complete_feasible: 25-game upper projection fits the 12-hour wall clock",
            "inference_substrate": "live_llm_inference",
            "model_specs": [{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}],
            "model_file_hash_embedded_tokenizer_llama_cpp_and_cuda_receipts": {
                "available": True,
                "embedded_tokenizer_ok": True,
            },
            "gpu_vram_thermal_process_port_and_cleanup_receipts": {
                "nvidia_smi": {"returncode": 0}
            },
            "twenty_five_game_twelve_hour_projection_and_interval": {
                "fits_12h_at_upper_bound": True,
                "upper_bound_s": 14101.318,
                "cap_s": 43200,
                "n_measured_games": 8,
            },
            "no_automatic_flag_change_receipt": {"feature_flags_changed": False},
            "no_new_solve_credit_receipt": {"registry_update_requested": False},
            "shipped_flag_and_registry_immutability": {
                "feature_flags_changed": False,
                "registry_unchanged": True,
            },
            "solve_provenance": "live_agent_self_discovery",
        },
    }
    return fixtures[task_id]


def _make_repo(root: Path) -> None:
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_roadmap_payload(), sort_keys=False))
    _write_text(root, mod.ROADMAP_DOC_RELATIVE_PATH, "# Research Roadmap vNEXT\n\nExp5961-Exp5973\n")
    _write_text(root, mod.RESEARCH_REFERENCES_RELATIVE_PATH, "<!-- V528-PLANNER-REFRESH-20260726-END -->\n")
    _write_text(root, mod.RESEARCH_STUDYING_RELATIVE_PATH, "studying fixture\n")
    _write_text(root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: []\n")
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "\n".join([
        "| 2026-08-03 13:37 UTC | Milestone 2026.07.528 activated | OK | 13 tasks queued |",
        "| 2026-08-03 15:01 UTC | Exact terminal-boundary handoff from .527 into .52 | FAIL | cap |",
        "| 2026-08-03 21:51 UTC | Gated on Exp5964 ready: portable exact-atom compat | GATE_BLOCK | skip |",
        "| 2026-08-03 21:51 UTC | Gated on Exp5965 ready: end-to-end discriminative | GATE_BLOCK | failed |",
    ]) + "\n")
    for rel_path in (
        mod.AGENTS_RELATIVE_PATH,
        mod.CODEX_RELATIVE_PATH,
        mod.CLAUDE_RELATIVE_PATH,
        mod.STATUS_RELATIVE_PATH,
        mod.CHANGELOG_RELATIVE_PATH,
        mod.TRACEABILITY_RELATIVE_PATH,
        mod.EXCLUSION_MANIFEST_RELATIVE_PATH,
        mod.KNOWN_ISSUES_RELATIVE_PATH,
        mod.ARC_REGISTRY_RELATIVE_PATH,
        mod.NORTH_STAR_RELATIVE_PATH,
        mod.ADVERSARIAL_VERIFY_RELATIVE_PATH,
        mod.DOC_RECONCILE_RELATIVE_PATH,
        mod.SPEC_RELATIVE_PATH,
        mod.EXP5933_CLASSIFIER_RELATIVE_PATH,
        mod.ARC_AGENT_RELATIVE_PATH,
    ):
        _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\nREQ-REPORT-5973\n")
    for task_id, _title, rel_path in mod.UPSTREAM_TASKS:
        if task_id in {"exp5961-transition-v528", "exp5965-portable-atom-energy-ranker"}:
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_json(root, "results/experiment_5961_gemma31b_placement_decision_corrected.json", {"status": "alias"})


def _receipt(task_id: str, rel_path: Path) -> JsonDict:
    stdout_json = {
        "reports": [
            {
                "artifact": rel_path.as_posix(),
                "loaded": True,
                "flag_count": 0,
                "max_severity": -1,
                "flags": [],
            }
        ],
        "flagged_count": 0,
    }
    return {
        "task_id": task_id,
        "artifact_path": rel_path.as_posix(),
        "command": f".venv/bin/python scripts/adversarial_verify.py --json {rel_path.as_posix()}",
        "exit_code": 0,
        "stdout_json": stdout_json,
        "receipt_hash": mod.sha256_json(stdout_json),
    }


def _receipts() -> list[JsonDict]:
    return [
        _receipt(task_id, rel_path)
        for task_id, _title, rel_path in mod.UPSTREAM_TASKS
        if task_id not in {"exp5961-transition-v528", "exp5965-portable-atom-energy-ranker"}
    ]


def _tests_run() -> list[JsonDict]:
    return [
        {"command": ".venv/bin/pytest tests/python/test_experiment_5973_v528_capstone_reconciliation.py -q", "exit_code": 0},
        {"command": ".venv/bin/coverage report --include=python/carnot/experiment_5973_v528_capstone_reconciliation.py --fail-under=100", "exit_code": 0},
    ]


def _build(root: Path) -> JsonDict:
    return mod.build_report(root, adversarial_receipts=_receipts(), tests_run=_tests_run(), duration_s=1.5)


def test_req_report_5973_spec_declares_capstone_contract() -> None:
    """REQ-REPORT-5973: OpenSpec declares exact identity and branch rules."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5973") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(task_id, declared_deliverable)" in section
    assert mod.INFERENCE_SUBSTRATE in section
    for scenario in (
        "SCENARIO-REPORT-5973-EXACT-MATRIX",
        "SCENARIO-REPORT-5973-GATES-AND-MISSING",
        "SCENARIO-REPORT-5973-BRANCH-INDEPENDENCE",
        "SCENARIO-REPORT-5973-VERIFIER-AND-SUBSTRATE",
        "SCENARIO-REPORT-5973-SCHEMA",
    ):
        assert scenario in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5973_exact_matrix_terminal_classes_and_verifier(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5973-EXACT-MATRIX: only declared deliverables are evidence."""

    _make_repo(tmp_path)
    report = _build(tmp_path)

    assert report["status"] == "complete_with_blocks"
    assert report["honest_verdict"].startswith("complete_with_blocks:")
    matrix = report["milestone_and_exact_task_deliverable_matrix"]
    assert matrix["selection_policy"] == "active_roadmap_declared_deliverable_only"
    assert matrix["upstream_task_count"] == 12
    assert matrix["numeric_prefix_aliases_ignored"] == [
        "results/experiment_5961_gemma31b_placement_decision_corrected.json"
    ]

    classes = report["per_task_path_hash_presence_and_terminal_class"]["terminal_class_by_task_id"]
    assert classes == {
        "exp5961-transition-v528": "missing",
        "exp5962-v528-source-delta-ingestion": "complete-null",
        "exp5963-exact-atom-pair-fixture": "complete-ready",
        "exp5964-sota-atom-compatibility-corpus": "blocked-precondition",
        "exp5965-portable-atom-energy-ranker": "gate-blocked",
        "exp5966-discriminative-constraint-acquisition": "gate-blocked",
        "exp5967-delayed-commit-memory-fixture": "complete-ready",
        "exp5968-delayed-commit-csl-prospective": "complete-ready",
        "exp5969-csl-poison-drift-abi-audit": "complete-ready",
        "exp5970-arc-strip-swap-sentinel": "complete-ready",
        "exp5971-arc-strip-swap-battery": "complete-null",
        "exp5972-arc-llm-on-budget2000-feasibility": "complete-feasible",
    }
    rows = report["per_task_path_hash_presence_and_terminal_class"]["tasks"]
    assert rows["exp5961-transition-v528"]["present"] is False
    assert rows["exp5965-portable-atom-energy-ranker"]["present"] is False
    assert rows["exp5965-portable-atom-energy-ranker"]["terminal_evidence_source"] == "recomputed_gate_and_conductor_skip"
    verifier = report["fresh_adversarial_verifier_receipts"]
    assert verifier["verified_present_declared_deliverable_count"] == 10
    assert verifier["missing_declared_deliverables_not_verified"] == [
        "results/experiment_5961_transition_v528.json",
        "results/experiment_5965_portable_atom_energy_ranker.json",
    ]
    assert verifier["flagged_count"] == 0
    mod.validate_artifact(report)


def test_scenario_report_5973_gates_cascade_and_branch_independence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5973-GATES-AND-MISSING: cascades do not rewrite branches."""

    _make_repo(tmp_path)
    report = _build(tmp_path)

    gates = report["gate_recomputation_and_cascade_receipts"]
    by_task = {row["task_id"]: row for row in gates["gate_rows"]}
    assert by_task["exp5964-sota-atom-compatibility-corpus"]["gate_passed"] is True
    assert by_task["exp5965-portable-atom-energy-ranker"]["gate_passed"] is False
    assert by_task["exp5965-portable-atom-energy-ranker"]["actual"] == 0.0
    assert by_task["exp5966-discriminative-constraint-acquisition"]["gate_passed"] is False
    assert by_task["exp5966-discriminative-constraint-acquisition"]["actual"] is None
    assert gates["cascade_skips"] == [
        {
            "task_id": "exp5965-portable-atom-energy-ranker",
            "reason": "upstream_gate_failed_or_retired",
            "not_executed_by_capstone": True,
        },
        {
            "task_id": "exp5966-discriminative-constraint-acquisition",
            "reason": "upstream_artifact_missing",
            "not_executed_by_capstone": True,
        },
    ]
    assert gates["title_yaml_gate_alignment_exact"] is True

    semantic = report["semantic_acquisition_branch_summary"]
    assert semantic["fixture"]["ready_score"] == 1.0
    assert semantic["corpus"]["terminal_class"] == "blocked-precondition"
    assert semantic["ranker"]["terminal_class"] == "gate-blocked"
    assert semantic["exact_acquisition"]["terminal_class"] == "gate-blocked"
    assert semantic["fixture_ready_does_not_imply_ranker_or_acquisition_quality"] is True

    csl = report["continuous_self_learning_branch_summary"]
    assert csl["fixture"]["ready_score"] == 1.0
    assert csl["prospective"]["ready_score"] == 1.0
    assert csl["poison_abi_audit"]["unsafe_accept_count"] == 0

    strip = report["arc_strip_swap_branch_summary"]
    assert strip["sentinel"]["ready_score"] == 1.0
    assert strip["battery"]["terminal_class"] == "complete-null"
    assert strip["hidden_transfer_claimed"] is False
    assert strip["new_solve_credit_claimed"] is False

    budget = report["arc_budget_feasibility_branch_summary"]
    assert budget["feasible_at_upper_bound"] is True
    assert budget["automatic_flag_change"] is False
    assert budget["new_solve_credit_claimed"] is False

    independence = report["branch_independence_receipt"]
    assert independence["branch_independence_preserved"] is True
    assert independence["borrowed_success_count"] == 0
    mod.validate_artifact(report)


def test_scenario_report_5973_policy_substrate_docs_and_checksum(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5973-SCHEMA: policy, docs, protection, and checksum hold."""

    _make_repo(tmp_path)
    report = _build(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["aggregation_substrate_classifier_receipt"] == {
        "exp5933_classifier_path": mod.EXP5933_CLASSIFIER_RELATIVE_PATH.as_posix(),
        "exp5933_classifier_present": True,
        "capstone_declared_substrate": mod.INFERENCE_SUBSTRATE,
        "nested_upstream_live_substrates_observed": [
            "live_llm_embedding_extraction",
            "live_llm_inference",
        ],
        "duration_rule_inherited_from_nested_upstream": False,
        "aggregation_classifier_ready": True,
        "principle": mod.FIELD_PRINCIPLES["aggregation_substrate_classifier_receipt"],
    }
    policy = report["model_and_hardware_policy_receipt"]
    assert policy["mandated_gguf_identities_observed"]
    assert policy["cuda_authenticity_checked"] is True
    assert policy["legacy_headline_claimed"] is False
    assert policy["unsupported_board_claim_count"] == 0
    arc = report["arc_provenance_registry_and_flag_immutability"]
    assert arc["registry_mutated"] is False
    assert arc["flag_flip_performed"] is False
    assert arc["new_solve_credit_claimed"] is False
    docs = report["docs_reconciled"]
    assert docs["openspec_research_reporting_req_5973_present"] is True
    assert docs["ops_status_deferred_to_conductor_stop_rule"] is True
    assert docs["references_and_studying_state_claims_reconciled_in_artifact"] is True
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert report["field_provenance"][field]["principle"] == mod.FIELD_PRINCIPLES[field]
    mod.validate_artifact(report)


def test_req_report_5973_validation_and_blocked_preconditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5973: invalid capstones fail closed with exact errors."""

    _make_repo(tmp_path)
    report = _build(tmp_path)

    mutations = [
        (lambda artifact: artifact.pop("status"), "missing required field"),
        (lambda artifact: artifact.update(inference_substrate="live_llm_inference"), "inference_substrate"),
        (lambda artifact: artifact.update(honest_verdict="complete_partial: bad"), "honest_verdict"),
        (
            lambda artifact: artifact["milestone_and_exact_task_deliverable_matrix"].update(upstream_task_count=11),
            "twelve upstream tasks",
        ),
        (
            lambda artifact: artifact["milestone_and_exact_task_deliverable_matrix"].update(upstream_task_ids=[]),
            "exact upstream task ids",
        ),
        (
            lambda artifact: artifact.update(per_task_path_hash_presence_and_terminal_class=[]),
            "terminal classes",
        ),
        (
            lambda artifact: artifact["per_task_path_hash_presence_and_terminal_class"]["terminal_class_by_task_id"].update({"exp5961-transition-v528": "complete"}),
            "missing handoff",
        ),
        (
            lambda artifact: artifact["fresh_adversarial_verifier_receipts"].update(verified_present_declared_deliverable_count=1),
            "adversarial verifier",
        ),
        (
            lambda artifact: artifact["fresh_adversarial_verifier_receipts"]["reports"][0].pop("receipt_hash"),
            "adversarial verifier receipt missing hash",
        ),
        (
            lambda artifact: artifact["fresh_adversarial_verifier_receipts"]["reports"][0].update(command="python other.py"),
            "adversarial verifier receipt command",
        ),
        (
            lambda artifact: artifact["gate_recomputation_and_cascade_receipts"].update(title_yaml_gate_alignment_exact=False),
            "gate alignment",
        ),
        (
            lambda artifact: artifact["branch_independence_receipt"].update(branch_independence_preserved=False),
            "branch independence",
        ),
        (
            lambda artifact: artifact["aggregation_substrate_classifier_receipt"].update(duration_rule_inherited_from_nested_upstream=True),
            "aggregation substrate",
        ),
        (
            lambda artifact: artifact["protected_files_unchanged"].update(all_unchanged=False),
            "protected file",
        ),
        (lambda artifact: artifact.update(field_provenance=[]), "field provenance"),
        (
            lambda artifact: artifact["field_provenance"]["status"].update(principle="wrong"),
            "field provenance missing for status",
        ),
    ]
    for mutate, needle in mutations:
        artifact = deepcopy(report)
        mutate(artifact)
        artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
        with pytest.raises(ValueError, match=needle):
            mod.validate_artifact(artifact)

    checksum_drift = deepcopy(report)
    checksum_drift["duration_s"] = 2.5
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(checksum_drift)

    bad_root = tmp_path / "bad"
    _make_repo(bad_root)
    bad_roadmap = _roadmap_payload()
    bad_roadmap["milestone"] = "2026.07.999"
    _write_text(bad_root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(bad_roadmap))
    blocked = mod.build_report(
        bad_root,
        adversarial_receipts=_receipts(),
        tests_run=_tests_run(),
        duration_s=1.5,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert "active_roadmap_milestone_mismatch" in blocked["preconditions_checked"]["failed_preconditions"]
    mod.validate_artifact(blocked)

    many_bad = tmp_path / "many_bad"
    _make_repo(many_bad)
    malformed = "a: [\n"
    _write_text(many_bad, mod.ROADMAP_RELATIVE_PATH, malformed)
    _write_text(many_bad, mod.CONDUCTOR_LOG_RELATIVE_PATH, "no activation\n")
    (many_bad / mod.ADVERSARIAL_VERIFY_RELATIVE_PATH).unlink()
    _write_text(many_bad, mod.SPEC_RELATIVE_PATH, "missing req\n")
    (many_bad / mod.EXP5933_CLASSIFIER_RELATIVE_PATH).unlink()
    rows = _receipts()
    rows = [row for row in rows if row["task_id"] != "exp5962-v528-source-delta-ingestion"]
    rows[0]["exit_code"] = 1
    rows[0]["stdout_json"]["reports"][0]["max_severity"] = 3
    rows[0]["stdout_json"]["reports"][0]["flag_count"] = 1
    monkeypatch.setattr(
        mod,
        "_protected_unchanged",
        lambda _root, _before: {
            "files": {"research-roadmap.yaml": {"unchanged": False}},
            "all_unchanged": False,
            "principle": mod.FIELD_PRINCIPLES["protected_files_unchanged"],
        },
    )
    monkeypatch.setattr(
        mod,
        "_resource_receipt",
        lambda _root: {
            "disk": {"ok": False},
            "ram": {"ok": True},
        },
    )
    monkeypatch.setattr(
        mod,
        "_atomic_output_receipt",
        lambda _root: {"ok": False},
    )
    many_blocked = mod.build_report(
        many_bad,
        adversarial_receipts=rows,
        tests_run=_tests_run(),
        duration_s=1.5,
    )
    failed = set(many_blocked["preconditions_checked"]["failed_preconditions"])
    assert {
        "active_roadmap_unloadable",
        "v528_activation_line_missing_or_not_thirteen",
        "adversarial_verifier_missing",
        "aggregation_classifier_missing_or_not_ready",
        "openspec_req_5973_missing",
        "protected_file_modified",
        "insufficient_resources",
        "atomic_output_unavailable",
    } <= failed

    receipt_bad = tmp_path / "receipt_bad"
    _make_repo(receipt_bad)
    receipt_rows = _receipts()
    receipt_rows = [
        row
        for row in receipt_rows
        if row["task_id"] != "exp5962-v528-source-delta-ingestion"
    ]
    receipt_rows[0]["exit_code"] = 1
    receipt_rows[0]["stdout_json"]["reports"][0]["max_severity"] = 3
    receipt_rows[0]["stdout_json"]["reports"][0]["flag_count"] = 1
    receipt_report = mod.build_report(
        receipt_bad,
        adversarial_receipts=receipt_rows,
        tests_run=_tests_run(),
        duration_s=1.5,
    )
    receipt_failed = set(
        receipt_report["preconditions_checked"]["failed_preconditions"]
    )
    assert {
        "missing_adversarial_receipts",
        "adversarial_verifier_failed",
    } <= receipt_failed

    mismatch = tmp_path / "mismatch"
    _make_repo(mismatch)
    mismatch_roadmap = _roadmap_payload()
    mismatch_roadmap["tasks"] = mismatch_roadmap["tasks"][:-2]
    mismatch_roadmap["tasks"][3]["title"] = "Gated on wrong upstream"
    _write_text(mismatch, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(mismatch_roadmap))
    mismatch_report = mod.build_report(
        mismatch,
        adversarial_receipts=_receipts(),
        tests_run=_tests_run(),
        duration_s=1.5,
    )
    assert "active_roadmap_task_ids_mismatch" in mismatch_report["preconditions_checked"]["failed_preconditions"]
    assert "gate_title_yaml_alignment_failed" in mismatch_report["preconditions_checked"]["failed_preconditions"]


def test_req_report_5973_io_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-5973: helper IO receipts are deterministic and defensive."""

    path = tmp_path / "artifact.json"
    mod.write_json(path, {"b": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"b": 1}
    assert mod.path_sha256(path).startswith("sha256:")
    assert mod.path_sha256(tmp_path / "missing") is None
    assert mod.sha256_json({"a": 1}) == mod.sha256_bytes(b'{"a":1}')

    _, missing_yaml_meta = mod._read_yaml_mapping(tmp_path / "missing.yaml")
    assert missing_yaml_meta["error"] == "missing"

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    _, bad_meta = mod._read_json_mapping(bad_json)
    assert bad_meta["error"].startswith("json_error:")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    _, list_meta = mod._read_json_mapping(list_json)
    assert list_meta["error"] == "json_not_mapping"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("a: [\n", encoding="utf-8")
    _, yaml_meta = mod._read_yaml_mapping(bad_yaml)
    assert yaml_meta["error"].startswith("yaml_error:")
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- a\n", encoding="utf-8")
    _, list_yaml_meta = mod._read_yaml_mapping(list_yaml)
    assert list_yaml_meta["error"] == "yaml_not_mapping"

    assert mod._task_number("exp5973-v528") == 5973
    assert mod._task_number("nope") is None
    assert mod._receipt_flags({"stdout_json": {"reports": [{"flags": "bad"}]}}) == []
    assert mod._receipt_flags({"flags": [{"kind": "WARN"}]}) == [{"kind": "WARN"}]
    assert mod._receipt_flag_count({"stdout_json": {"reports": [{"flags": [{"kind": "X"}]}]}}) == 1
    assert mod._receipt_flag_count({"stdout_json": {"flagged_count": 3}}) == 3
    assert mod._receipt_flag_count({"flag_count": 4}) == 4
    assert mod._receipt_max_severity({"max_severity": 2}) == 2
    assert mod._normalize_adversarial_receipts(None, {}) == {}
    assert mod._normalize_adversarial_receipts([{}, "bad", {"task_id": ""}], {}) == {}
    assert mod._conductor_status_from_line("| date | title | OK | detail |") == "OK"
    assert mod._conductor_status_from_line("bad") == ""

    assert mod._numeric_prefix_aliases(tmp_path / "no_results") == []
    gates = mod._gate_recomputation(
        [{"id": "exp9999", "title": "bad", "gated_on": ["not-a-gate"]}],
        {},
        {},
    )
    assert gates["gate_rows"] == []

    assert mod._terminal_class(
        "x",
        {"status": "retired", "honest_verdict": "retired: x"},
        {"present": True},
        {},
        [],
    ) == ("retired", "retire-if-same-verdict")
    assert mod._terminal_class(
        "x",
        {"status": "complete_underpowered", "honest_verdict": "complete_underpowered: x"},
        {"present": True},
        {},
        [],
    ) == ("underpowered", "underpowered")
    assert mod._terminal_class(
        "x",
        {"status": "complete", "honest_verdict": "complete: x"},
        {"present": True},
        {},
        [],
    ) == ("complete", "complete-receipt")
    assert mod._terminal_class(
        "x",
        {"status": "unknown", "honest_verdict": "unknown"},
        {"present": True},
        {},
        [],
    ) == ("missing", "unrecognized-terminal-treated-as-missing")

    terminal = {
        "tasks": {"x": {"present": True, "declared_deliverable": "x.json"}},
        "terminal_class_by_task_id": {"x": "complete"},
    }
    assert mod._fresh_verifier_receipts({}, terminal)["verified_present_declared_deliverable_count"] == 0
    prior = mod._prior_failure_receipt(
        [{"id": "x", "prior_failures": ["bad", {"verdict": "complete: old"}]}],
        {"x": {"honest_verdict": "complete: old"}},
        {"terminal_class_by_task_id": {"x": "complete"}},
        tmp_path,
    )
    assert prior["prior_failure_audit"][0]["same_verdict_recurred"] is True

    assert mod._status_verdict([], {"terminal_class_by_task_id": {"x": "complete-null"}})[0] == "complete_with_nulls"
    assert mod._status_verdict([], {"terminal_class_by_task_id": {"x": "complete"}})[0] == "complete"

    def _raise_oserror(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("atomic failed")

    monkeypatch.setattr(mod.os, "replace", _raise_oserror)
    assert mod._atomic_output_receipt(tmp_path)["ok"] is False
