"""Tests for the read-only V601 evidence and method-change contract.

Spec refs: REQ-REPORT-6865 and SCENARIO-REPORT-6865-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest
import yaml

from carnot import experiment_6865_v601_evidence_method_change_contract as mod


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_json(relative: str) -> dict[str, object]:
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def _source_hashes() -> dict[str, str]:
    return {
        name: mod.sha256_path(REPO_ROOT / relative)
        for name, relative in mod.DEFAULT_SOURCE_PATHS.items()
        if (REPO_ROOT / relative).is_file()
    }


def test_missing_artifact_writes_a_complete_blocked_contract(tmp_path: Path) -> None:
    # SCENARIO-REPORT-6865-MISSING-ARTIFACT
    artifact = mod.build_artifact(tmp_path, "20260902", live_reports={})

    assert artifact["v601_evidence_contract_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == (
        "complete_blocked_v601_evidence_method_change_contract"
    )
    assert artifact["gate_check_summary"]["failed_check"] == "required_source_readability"
    assert artifact["gate_check_summary"]["expected"] == "all required sources readable"
    assert mod.validate_artifact(artifact) == []


def test_synthetic_gate_artifact_is_terminal_blocked_evidence() -> None:
    # SCENARIO-REPORT-6865-SYNTHETIC-GATE
    gate = {
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "gates_evaluated": [
            {
                "upstream": "exp-upstream",
                "artifact_field": "ready_score",
                "op": "==",
                "expected": 1,
                "actual": 0,
                "passed": False,
            }
        ],
    }

    verdict = mod.read_terminal_verdict(gate)
    gate_rows = mod.normalized_gate_rows(gate)

    assert verdict == ("blocked_gate_check_failed", "blocked")
    assert gate_rows == [
        {
            "upstream": "exp-upstream",
            "artifact_field": "ready_score",
            "operator": "==",
            "expected": 1,
            "observed": 0,
            "passed": False,
        }
    ]
    assert mod.normalized_gate_rows({"gates_evaluated": ["not-a-gate"]}) == []


def test_adversarial_verifier_loader_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # REQ-REPORT-6865
    monkeypatch.setattr(mod.importlib.util, "spec_from_file_location", lambda *_args: None)

    with pytest.raises(RuntimeError, match="adversarial verifier import failed"):
        mod._load_validator(REPO_ROOT)


def test_principle_wrapped_fields_unwrap_without_destroying_dicts() -> None:
    # SCENARIO-REPORT-6865-WRAPPED-FIELD
    wrapped = {"principle": "The value stays auditable.", "value": ["a", "b"]}
    ordinary = {"value": 7, "checks": {"ready": True}}

    assert mod.unwrap_principle(wrapped) == ["a", "b"]
    assert mod.unwrap_principle(ordinary) is ordinary
    assert mod.unwrap_principle("plain") == "plain"


def test_stale_active_roadmap_blocks_with_exact_milestone() -> None:
    # SCENARIO-REPORT-6865-STALE-ROADMAP
    stale = {
        "milestone": "2026.09.600",
        "tasks": [
            {
                "id": mod.EXP6865_TASK_ID,
                "deliverable": mod.OUTPUT_PATH.as_posix(),
            }
        ],
    }

    check = mod.active_roadmap_check(stale)

    assert check["passed"] is False
    assert check["expected"] == mod.EXPECTED_ACTIVE_MILESTONE
    assert check["observed"] == "2026.09.600"

    current_but_missing = mod.active_roadmap_check(
        {"milestone": mod.EXPECTED_ACTIVE_MILESTONE, "tasks": []}
    )
    assert current_but_missing["observed"] == {
        "task_id": mod.EXP6865_TASK_ID,
        "deliverable": None,
    }


def test_source_reader_rejects_empty_and_non_mapping_documents(tmp_path: Path) -> None:
    # SCENARIO-REPORT-6865-MISSING-ARTIFACT
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    json_list = tmp_path / "list.json"
    json_list.write_text("[]", encoding="utf-8")
    yaml_list = tmp_path / "list.yaml"
    yaml_list.write_text("- value", encoding="utf-8")

    assert mod._read_document(empty) == (None, "empty")
    assert mod._read_document(json_list) == (None, "json_object_required")
    assert mod._read_document(yaml_list) == (None, "yaml_mapping_required")


def test_mismatched_tokenizer_hash_schemas_are_incomparable() -> None:
    # SCENARIO-REPORT-6865-HASH-SCHEMA
    exp6850 = {
        "tokenizer_receipts": [
            {"hf_id": "gemma", "probe_token_ids": [2, 7], "tokenizer_sha256": "sha256:old"}
        ]
    }
    exp6863 = {
        "tokenizer_receipts": [
            {
                "hf_id": "gemma",
                "probe_token_ids": [7],
                "tokenizer_sha256": "sha256:new",
                "special_tokens": {"bos": 2},
            }
        ]
    }

    diff = mod.tokenizer_receipt_schema_diff(exp6850, exp6863)

    assert diff["schemas_identical"] is False
    assert diff["only_in_exp6850"] == []
    assert diff["only_in_exp6863"] == ["special_tokens"]
    assert diff["hash_comparison_valid"] is False
    assert diff["tokenizer_relation"] == "undetermined_incomparable_schemas"


def test_retired_max_token_only_remedy_is_rejected() -> None:
    # SCENARIO-REPORT-6865-RETIRED-MAX-TOKEN
    exclusion = yaml.safe_load(
        (REPO_ROOT / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    )

    violations = mod.retired_scope_violations(
        ["raising CARNOT_ARC_INDUCE_MAX_TOKENS to fix induction quality"], exclusion
    )

    assert violations[0]["retired_scope_id"] == "arc-induce-completion-budget-raise"
    assert "CARNOT_ARC_INDUCE_MAX_TOKENS" in violations[0]["blocked_pattern"]
    assert (
        mod.retired_scope_violations(
            ["measure prompt, slot, n_ctx, VRAM, and shared-pool headroom"], exclusion
        )
        == []
    )
    assert mod.retired_scope_violations([], {"retired_extras": [None]}) == []


def test_real_v600_tasks_and_conductor_states_recompute_from_primary_records() -> None:
    # REQ-REPORT-6865
    completed = yaml.safe_load((REPO_ROOT / "research-complete.yaml").read_text(encoding="utf-8"))
    payloads = {
        task_id: _read_json(relative)
        for task_id, relative in mod.V600_ARTIFACT_PATHS.items()
    }
    conductor = (REPO_ROOT / "ops/conductor-log.md").read_text(encoding="utf-8")

    task_rows = mod.build_roadmap_task_rows(completed, payloads)
    conductor_rows = mod.build_conductor_gate_rows(conductor, task_rows)

    assert [row["task_id"] for row in task_rows] == list(mod.V600_TASK_IDS)
    assert task_rows[2]["roadmap_result"] == "OK"
    assert task_rows[2]["honest_verdict"].startswith("complete_blocked_")
    assert task_rows[2]["verdict_class"] == "blocked"
    assert task_rows[3]["roadmap_result"] == "OK_DELIVERABLE_ONLY"
    assert conductor_rows[3]["conductor_status"] == "GATE_BLOCK"
    assert conductor_rows[3]["gate_fields"][0]["artifact_field"] == (
        "semantic_contrast_preregistration_ready_score"
    )


def test_real_receipt_diff_records_exact_field_sets_without_semantic_claim() -> None:
    # SCENARIO-REPORT-6865-HASH-SCHEMA
    diff = mod.tokenizer_receipt_schema_diff(
        _read_json("results/experiment_6850_three_family_scoring_admission_canary.json"),
        _read_json(
            "results/experiment_6863_tokenizer_aware_semantic_contrast_preregistration.json"
        ),
    )

    assert diff["only_in_exp6850"] == ["detail"]
    assert diff["only_in_exp6863"] == [
        "chat_template_identity",
        "chat_template_present",
        "special_tokens",
        "tokenize_settings",
    ]
    assert len(diff["gemma_hash_witnesses"]) == 2
    assert all(row["comparison_valid"] is False for row in diff["gemma_hash_witnesses"])
    assert all(row["semantic_relation"] == "undetermined" for row in diff["gemma_hash_witnesses"])


def test_canonical_tokenizer_schema_and_probe_contract_are_versioned_and_hashed() -> None:
    # REQ-REPORT-6865
    hashes = _source_hashes()
    exp6850 = _read_json("results/experiment_6850_three_family_scoring_admission_canary.json")
    exp6863 = _read_json(
        "results/experiment_6863_tokenizer_aware_semantic_contrast_preregistration.json"
    )

    schema = mod.canonical_tokenizer_schema(hashes)
    probes = mod.frozen_semantic_probe_contract(exp6850, exp6863, hashes)

    assert schema["version"] == "v1"
    assert "semantic_vocabulary_metadata" in schema["required_payload_fields"]
    assert "special_token_ids" in schema["required_payload_fields"]
    assert "add_bos_behavior" in schema["required_payload_fields"]
    assert "paths" in schema["excluded_field_classes"]
    assert mod.contract_hash_valid(schema)
    assert probes["probe_text"] == " execution readiness is checked without a scientific label."
    assert len(probes["legacy_output_witnesses"]) == 6
    assert all(row["canonical_output"] is False for row in probes["legacy_output_witnesses"])
    assert mod.contract_hash_valid(probes)

    ignored = mod.frozen_semantic_probe_contract(
        {"tokenizer_receipts": [None]}, {}, hashes
    )
    assert ignored["legacy_output_witnesses"] == []


def test_contrast_bank_freezes_all_one_hundred_group_identities() -> None:
    # REQ-REPORT-6865
    exp6862 = _read_json("results/experiment_6862_dual_side_semantic_contrast_bank.json")
    source_sha = mod.sha256_path(
        REPO_ROOT / "results/experiment_6862_dual_side_semantic_contrast_bank.json"
    )

    manifest = mod.frozen_contrast_bank_manifest(exp6862, source_sha)

    assert manifest["accepted_group_count"] == 100
    assert len(manifest["group_identities"]) == 100
    assert len({row["group_id"] for row in manifest["group_identities"]}) == 100
    assert manifest["source_artifact_sha256"] == source_sha
    assert manifest["cell_filtering_allowed"] is True
    assert manifest["bank_regeneration_allowed"] is False
    assert mod.contract_hash_valid(manifest)


def test_memory_contract_freezes_observability_and_exact_quarantine_boundaries() -> None:
    # REQ-REPORT-6865
    hashes = _source_hashes()
    observability = mod.memory_observability_contract(
        _read_json("results/experiment_6856_sealed_risk_sensitive_learning_audit.json"),
        _read_json("results/experiment_6861_v600_branch_retirement_evidence_contract.json"),
        hashes,
    )
    quarantine = mod.memory_transition_quarantine_contract(hashes)

    assert "later_exact_outcome" in observability["offline_supervision_only_fields"]
    assert "prior_reliability_state" in observability["decision_time_observable_fields"]
    assert observability["same_event_outcome_may_select_action"] is False
    assert observability["preserved_harmful_write_counts"] == {
        "exp6856_fresh_reducer": 12,
        "exp6861_branch_contract": 13,
    }
    assert observability["harmful_write_count_disagreement_preserved"] is True
    assert mod.contract_hash_valid(observability)
    assert {row["check_id"] for row in quarantine["exact_transition_checks"]} == {
        "coverage",
        "preservation",
        "source_faithfulness",
        "provenance",
        "old_family_retention",
        "delayed_invalidation",
        "tombstone_non_reappearance",
        "replay_equivalence",
        "restart_equivalence",
        "rollback_byte_exact",
    }
    assert all(row["failure_action"] == "quarantine" for row in quarantine["exact_transition_checks"])
    assert mod.contract_hash_valid(quarantine)


def test_arc_contract_keeps_full_headroom_evidence_and_prohibits_budget_only_fix() -> None:
    # SCENARIO-REPORT-6865-RETIRED-MAX-TOKEN
    contract = mod.arc_context_headroom_contract(_source_hashes())

    assert contract["observed_evidence"]["actual_server_n_ctx_tokens"] == 49152
    assert contract["observed_evidence"]["slot_count"] == 4
    assert contract["observed_evidence"]["requested_completion_tokens"] == 26800
    assert contract["observed_evidence"]["generated_reasoning_token_counts"] == [
        18431,
        2996,
        4066,
    ]
    assert "full_limit_diagnostic" in contract["required_attempt_fields"]
    assert "measured_generation_headroom_tokens" in contract["required_attempt_fields"]
    assert contract["max_token_only_repair_allowed"] is False
    assert contract["slot_capacity_may_be_inferred_by_division"] is False
    assert mod.contract_hash_valid(contract)


def test_live_adversarial_recheck_preserves_the_exp6862_warn_disagreement() -> None:
    # REQ-REPORT-6865
    payloads = {
        task_id: _read_json(relative)
        for task_id, relative in mod.V600_ARTIFACT_PATHS.items()
    }
    reports = mod.collect_live_adversarial_reports(REPO_ROOT)
    rows = mod.stored_vs_live_adversarial_rows(payloads, reports)
    exp6862 = next(row for row in rows if row["task_id"].startswith("exp6862"))

    assert exp6862["stored_status"] == "unstamped"
    assert exp6862["live_status"] == "warn"
    assert exp6862["disagreement"] is True
    assert exp6862["live_flags"][0]["kind"] == "SUBSTRATE_HAS_NO_DURATION_FLOOR"


def test_real_artifact_is_ready_complete_and_reproducible() -> None:
    # REQ-REPORT-6865
    artifact = mod.build_artifact(REPO_ROOT, "20260902")

    assert artifact["v601_evidence_contract_ready_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"] == (
        "complete_positive_v601_evidence_method_change_contract_ready"
    )
    assert artifact["inference_substrate"] == "deterministic CPU read-only evidence reduction"
    assert artifact["verifier_is_oracle"] is False
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_validator_reports_contract_hash_and_terminal_errors() -> None:
    # REQ-REPORT-6865
    artifact = mod.build_artifact(REPO_ROOT, "20260902")
    broken = deepcopy(artifact)
    broken["canonical_tokenizer_payload_schema"]["version"] = "v2"
    broken["verifier_is_oracle"] = True
    broken["honest_verdict"] = "blocked"
    broken["reproducibility_checksum"] = "sha256:bad"

    errors = mod.validate_artifact(broken)

    assert "contract_hash_mismatch:canonical_tokenizer_payload_schema" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "honest_verdict_not_terminal" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_validator_rejects_every_readiness_shape_error() -> None:
    # REQ-REPORT-6865
    artifact = mod.build_artifact(REPO_ROOT, "20260902", live_reports={})

    malformed = deepcopy(artifact)
    del malformed["schema"]
    malformed["unexpected"] = True
    malformed["field_principles"] = {}
    malformed["inference_substrate"] = "model inference"
    malformed["verdict_class"] = "unknown"
    malformed["v601_evidence_contract_ready_score"] = 2
    malformed_errors = mod.validate_artifact(malformed)
    assert any(error.startswith("missing_required_fields:") for error in malformed_errors)
    assert any(error.startswith("unexpected_fields:") for error in malformed_errors)
    assert "field_principles_must_cover_every_field" in malformed_errors
    assert "invalid_inference_substrate" in malformed_errors
    assert "invalid_verdict_class" in malformed_errors
    assert "invalid_ready_score" in malformed_errors

    bad_ready = deepcopy(artifact)
    bad_ready["frozen_contrast_bank_identity_manifest"]["accepted_group_count"] = 99
    bad_ready["gate_check_summary"]["passed"] = False
    bad_ready["verdict_class"] = "partial"
    ready_errors = mod.validate_artifact(bad_ready)
    assert "ready_contract_requires_100_bank_groups" in ready_errors
    assert "ready_contract_has_failed_gate" in ready_errors
    assert "ready_contract_must_be_positive" in ready_errors

    bad_blocked = deepcopy(artifact)
    bad_blocked["v601_evidence_contract_ready_score"] = 0
    bad_blocked["gate_check_summary"] = {}
    bad_blocked["verdict_class"] = "positive"
    blocked_errors = mod.validate_artifact(bad_blocked)
    assert "blocked_contract_requires_exact_failed_check" in blocked_errors
    assert "blocked_contract_must_use_blocked_class" in blocked_errors


def test_atomic_writer_and_cli_use_the_requested_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-REPORT-6865
    artifact = mod.build_artifact(REPO_ROOT, "20260902")
    output = tmp_path / "exp6865.json"

    mod.write_atomic(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8"))["experiment_id"] == 6865

    monkeypatch.setattr(mod, "build_artifact", lambda _root, _date: artifact)
    assert mod.main(["--date", "20260902", "--output", str(output)]) == 0

    broken = deepcopy(artifact)
    broken["honest_verdict"] = "not-terminal"
    monkeypatch.setattr(mod, "build_artifact", lambda _root, _date: broken)
    with pytest.raises(ValueError, match="honest_verdict_not_terminal"):
        mod.main(["--date", "20260902", "--output", str(output)])


def test_module_entrypoint_writes_only_the_requested_temp_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-REPORT-6865
    output = tmp_path / "entrypoint.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [mod.__file__, "--date", "20260902", "--output", str(output)],
    )

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(mod.__file__, run_name="__main__")

    assert stopped.value.code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["experiment_id"] == 6865
