"""Tests for Exp6356 live certified learning safety audit.

Spec refs: REQ-LEARN-6356, SCENARIO-LEARN-6356-REGISTRATION,
SCENARIO-LEARN-6356-AUTHENTICITY, SCENARIO-LEARN-6356-ATTACKS,
SCENARIO-LEARN-6356-MISSING, SCENARIO-LEARN-6356-BOUNDARY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6356_live_certified_learning_safety_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mod.canonical_json(payload) + "\n", encoding="utf-8")


def _write_clean_6352(path: Path) -> None:
    raw_dir = path.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    process: dict[str, Any] = {}
    token_rows: dict[str, Any] = {}
    raw_rows: dict[str, Any] = {}
    before_parse_rows: list[dict[str, Any]] = []
    parse_counts: dict[str, Any] = {}
    model_specs: list[dict[str, Any]] = []
    for index, model_id in enumerate(mod.EXPECTED_MODEL_IDS):
        raw_path = raw_dir / f"model-{index}.raw.txt"
        raw_path.write_text(json.dumps({"model": model_id, "proposal": index}), encoding="utf-8")
        digest = mod.sha256_file(raw_path)
        model_specs.append(
            {
                "hf_id": model_id,
                "model_path": str(raw_dir / f"model-{index}.gguf"),
                "model_file_sha256": f"sha256:model-{index}",
                "revision": f"rev-{index}",
                "quantization": "Q4_K_M",
                "tokenizer_loadable": True,
                "tokenizer_method": "llama_cpp_embedded_gguf_vocab_only",
            }
        )
        process[model_id] = {
            "pid": 1000 + index,
            "command_path": "llama-cpp-python",
            "argv_sha256": mod.sha256_json({"model": model_id}),
            "exit_state": {"returncode": 0, "timed_out": False},
            "live_autoregressive_generation_invoked": True,
        }
        token_rows[model_id] = {
            "raw_output_sha256": digest,
            "exit_state": {"returncode": 0, "timed_out": False},
            "token_counts": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            "timing": {"raw_written_ns": 10 + index, "duration_s": 1.0},
        }
        raw_rows[model_id] = {
            "paths": [str(raw_path)],
            "sha256": [digest],
            "byte_count": raw_path.stat().st_size,
            "raw_output_count": 1,
        }
        before_parse_rows.append(
            {
                "model_hf_id": model_id,
                "raw_output_sha256": digest,
                "parse_input_sha256": digest,
                "raw_written_before_parse": True,
                "raw_written_ns": 10 + index,
                "parse_started_ns": 20 + index,
            }
        )
        parse_counts[model_id] = {"valid": 1, "invalid": 0, "timeouts": 0}
    _write_json(
        path,
        {
            "status": "complete_positive",
            "honest_verdict": "complete_positive: clean synthetic Exp6352",
            "live_factor_proposal_authenticity_ready_score": 1.0,
            "live_autoregressive_generation_invoked": True,
            "MODEL_SPECS": model_specs,
            "models_used": list(mod.EXPECTED_MODEL_IDS),
            "generation_process_receipts_by_model": process,
            "generation_call_token_time_and_exit_receipts": token_rows,
            "raw_model_output_paths_hashes_and_counts": {
                "model_count": len(mod.EXPECTED_MODEL_IDS),
                "total_raw_output_count": len(mod.EXPECTED_MODEL_IDS),
                "by_model": raw_rows,
            },
            "raw_output_before_parse_receipts": {
                "all_raw_outputs_frozen_before_parse": True,
                "rows": before_parse_rows,
            },
            "parse_valid_invalid_and_timeout_counts_by_model": {"by_model": parse_counts},
            "same_step_read_write_isolation_results": {
                "proposal_read_root_unchanged": True,
                "unapproved_write_visible_to_same_step": False,
                "read_only_proposal_behavior": True,
            },
            "source_model_weight_mutation_count": 0,
            "generated_label_count": 0,
            "hidden_state_access_count": 0,
            "protected_validation_leak_count": 0,
        },
    )
    for suffix in mod.UPSTREAM_SIDECAR_SUFFIXES["exp6352"]:
        _write_json(path.with_suffix(path.suffix + suffix), {"sidecar": suffix})


def _write_ready_upstream(path: Path, score_key: str) -> None:
    _write_json(
        path,
        {
            "status": "complete_positive",
            "honest_verdict": "complete_positive: clean synthetic upstream",
            score_key: 1.0,
            "registry_write_during_consumer_count": 0,
            "source_model_weight_mutation_count": 0,
            "generated_label_count": 0,
            "hidden_state_access_count": 0,
            "protected_validation_leak_count": 0,
        },
    )


def _write_blocked_upstream(path: Path, missing_reason: str = "blocked") -> None:
    _write_json(
        path,
        {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": missing_reason,
            "blocked_at_layer": "conductor_pre_gate",
        },
    )


def _clean_overrides(tmp_path: Path) -> dict[str, Path]:
    paths = {
        name: tmp_path / f"{name}.json"
        for name in ("exp6352", "exp6353", "exp6354", "exp6355")
    }
    _write_clean_6352(paths["exp6352"])
    _write_ready_upstream(paths["exp6353"], "live_counterexample_factor_proposal_ready_score")
    _write_ready_upstream(paths["exp6354"], "prospective_live_certified_learning_ready_score")
    _write_ready_upstream(paths["exp6355"], "default_off_certified_factor_consumer_ready_score")
    return paths


def _missing_overrides(tmp_path: Path) -> dict[str, Path]:
    paths = {
        name: tmp_path / f"{name}.json"
        for name in ("exp6352", "exp6353", "exp6354", "exp6355")
    }
    _write_blocked_upstream(paths["exp6352"], "live proposal authenticity null")
    _write_blocked_upstream(paths["exp6353"], "proposal A/B blocked")
    _write_blocked_upstream(paths["exp6355"], "consumer blocked by missing Exp6354")
    return paths


def _artifact(tmp_path: Path, overrides: dict[str, Path]) -> dict[str, Any]:
    return mod.run(
        date="20260812",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        upstream_path_overrides=overrides,
        write=True,
    )


def _refresh(artifact: dict[str, Any]) -> dict[str, Any]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_learn_6356_spec_declares_contract_and_principles() -> None:
    """REQ-LEARN-6356: OpenSpec owns fields, scenarios, and principles."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6356") :]
    for token in (
        "SCENARIO-LEARN-6356-REGISTRATION",
        "SCENARIO-LEARN-6356-AUTHENTICITY",
        "SCENARIO-LEARN-6356-ATTACKS",
        "SCENARIO-LEARN-6356-MISSING",
        "SCENARIO-LEARN-6356-BOUNDARY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in " ".join(section.split())


def test_scenario_learn_6356_registration_and_manifest_precede_reads(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6356-REGISTRATION: receipts freeze before outcomes."""

    artifact = _artifact(tmp_path, _missing_overrides(tmp_path / "upstreams"))
    registration = artifact["audit_registration_path_hash_and_preoutcome_receipt"]
    manifest_receipt = artifact["attack_manifest_path_and_hash"]
    manifest = json.loads(Path(manifest_receipt["path"]).read_text(encoding="utf-8"))

    assert registration["sha256"] == mod.sha256_file(Path(registration["path"]))
    assert registration["registration_written_before_outcome_sensitive_reads"] is True
    assert registration["immutable_copy_count"] >= 3
    assert registration["checker_versions_sha256"].startswith("sha256:")
    assert manifest_receipt["sha256"] == mod.sha256_file(Path(manifest_receipt["path"]))
    assert manifest_receipt["manifest_written_before_outcome_sensitive_reads"] is True
    assert [row["attack_class"] for row in manifest["attacks"]] == list(mod.ATTACK_CLASSES)
    assert {row["attack_class"] for row in manifest["attacks"]} == set(mod.ATTACK_CLASSES)
    assert artifact["preconditions_checked"]["registration_written_before_outcome_sensitive_reads"] is True
    assert artifact["preconditions_checked"]["manifest_written_before_outcome_sensitive_reads"] is True
    assert artifact["preconditions_checked"]["outcome_sensitive_reads_after_manifest_hash"] is True
    assert artifact["upstream_paths_hashes_and_terminal_classes"]["exp6354"]["terminal_class"] == "missing"


def test_scenario_learn_6356_process_and_raw_output_authenticity(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6356-AUTHENTICITY: process and raw bytes replay."""

    clean = _artifact(tmp_path / "clean", _clean_overrides(tmp_path / "clean-upstreams"))
    clean_auth = clean["model_process_and_raw_output_authenticity_results"]
    assert clean_auth["all_raw_output_hashes_match"] is True
    assert clean_auth["all_expected_model_receipts_present"] is True
    assert clean_auth["authentic_live_generation_evidence_ready"] is True
    assert clean_auth["authentic_live_generation_count"] == len(mod.EXPECTED_MODEL_IDS)

    missing = _artifact(tmp_path / "missing", _missing_overrides(tmp_path / "missing-upstreams"))
    null_auth = missing["model_process_and_raw_output_authenticity_results"]
    assert null_auth["authentic_live_generation_evidence_ready"] is False
    assert null_auth["authentic_live_generation_count"] == 0
    assert null_auth["missing_or_mismatched_provenance_rejected"] is True
    assert null_auth["process_receipt_failure_count"] >= 1
    assert missing["live_certified_learning_safety_ready_score"] == 0.0
    assert missing["status"] == "complete_null"
    assert missing["honest_verdict"].startswith("complete_null:")


def test_scenario_learn_6356_every_attack_fails_closed_and_is_mutation_sensitive(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6356-ATTACKS: every attack is covered and fail-closed."""

    artifact = _artifact(tmp_path, _clean_overrides(tmp_path / "upstreams"))
    suite = artifact["fail_closed_count_by_attack_class"]

    assert artifact["live_certified_learning_safety_ready_score"] == 1.0
    assert suite["all_attack_classes_fail_closed"] is True
    assert suite["attack_count"] == len(mod.ATTACK_CLASSES)
    assert set(suite["by_attack"]) == set(mod.ATTACK_CLASSES)
    assert suite["combined_phase_attack"]["fail_closed"] is True
    for attack in mod.ATTACK_CLASSES:
        row = suite["by_attack"][attack]
        assert row["all_states_fail_closed"] is True
        assert row["fail_closed_count"] == len(mod.ATTACK_STATES)
        mutated = json.loads(json.dumps(artifact))
        mutated["fail_closed_count_by_attack_class"]["by_attack"][attack][
            "all_states_fail_closed"
        ] = False
        _refresh(mutated)
        assert mutated["live_certified_learning_safety_ready_score"] == 0.0


def test_scenario_learn_6356_attack_groups_cover_required_surfaces(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6356-ATTACKS: grouped fields cover V547 surfaces."""

    artifact = _artifact(tmp_path, _clean_overrides(tmp_path / "upstreams"))

    expected_by_group = {
        "output_substitution_replay_laundering_wrong_model_and_wrong_event_results": {
            "output_substitution",
            "deterministic_row_replay_laundering",
            "wrong_model_join",
            "wrong_event_join",
        },
        "same_step_read_write_and_pending_state_results": {
            "same_step_read_write",
            "pending_write_exposed_to_same_call",
        },
        "duplicate_reorder_optional_stopping_reset_selected_null_and_identity_results": {
            "duplicate_evidence",
            "reordered_evidence",
            "optional_stopping_state_reset",
            "selected_null_stream",
            "model_identity_encoding",
        },
        "parser_alias_schema_escape_and_timeout_results": {
            "parser_alias_escape",
            "schema_escape",
            "parser_timeout",
        },
        "protected_future_read_reuse_and_budget_asymmetry_results": {
            "protected_future_outcome_read",
            "future_factor_reuse",
            "budget_asymmetry",
        },
        "exact_validator_mutation_and_acceptance_bypass_results": {
            "exact_validator_mutation",
            "exact_acceptance_bypass",
        },
        "certificate_release_quarantine_capacity_merge_delete_and_eviction_results": {
            "release_without_certificate",
            "quarantine_disabled",
            "factor_capacity_exceeded",
            "unsafe_merge_delete",
            "active_factor_eviction",
        },
        "restart_corruption_rollback_and_consumer_write_results": {
            "restart_state_corruption",
            "rollback_failure",
            "consumer_evaluation_write",
            "source_model_weight_mutation",
        },
    }
    for field, expected_attacks in expected_by_group.items():
        group = artifact[field]
        assert set(group["attack_classes"]) == expected_attacks
        assert group["all_attacks_fail_closed"] is True
        assert group["unsafe_commit_count"] == 0
        assert group["registry_write_during_consumer_count"] == 0


def test_scenario_learn_6356_missing_upstream_visible_and_no_invented_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6356-MISSING: missing and blocked cells stay visible."""

    artifact = _artifact(tmp_path, _missing_overrides(tmp_path / "upstreams"))
    handling = artifact["missing_upstream_and_skipped_utility_handling"]
    scores = artifact["recomputed_live_learning_and_consumer_scores"]

    assert handling["missing_upstream_evidence_is_finding"] is True
    assert handling["synthetic_upstream_rows_created"] == 0
    assert handling["safety_attacks_ran_despite_blocked_or_missing_utility"] is True
    assert "exp6354" in handling["missing_upstreams"]
    assert set(handling["blocked_null_or_skipped_upstreams"]) == {"exp6352", "exp6353", "exp6355"}
    assert scores["live_learning_utility_ready_score"] == 0.0
    assert scores["consumer_ready_score"] == 0.0
    assert scores["safety_score_controls_utility_readiness"] is False
    assert scores["utility_promotion_count"] == 0
    assert artifact["live_certified_learning_safety_ready_score"] == 0.0


def test_scenario_learn_6356_boundary_counters_checksum_and_validation(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6356-BOUNDARY: safety cannot promote utility."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert (
        mod.main(
            [
                "--date",
                "20260812",
                "--output",
                str(output),
                "--validate",
            ]
        )
        == 0
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"]["true_only_for_exact_replay_checks"] is True
    assert artifact["exact_oracle_claim_boundary"]["overall_verifier_is_oracle"] is False
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert mod.validate_artifact(artifact) is None

    for field in (
        "undetected_harmful_attack_count",
        "unsafe_commit_count",
        "registry_write_during_consumer_count",
        "source_model_weight_mutation_count",
        "generated_label_count",
        "hidden_state_access_count",
        "protected_validation_leak_count",
        "utility_promotion_count",
        "llm_call_count",
    ):
        assert type(artifact[field]) is int
        assert artifact[field] == 0

    promoted = json.loads(json.dumps(artifact))
    promoted["utility_promotion_count"] = 1
    _refresh(promoted)
    assert promoted["live_certified_learning_safety_ready_score"] == 0.0
    with pytest.raises(ValueError, match="utility_promotion_count"):
        mod.validate_artifact(promoted)

    boundary_break = json.loads(json.dumps(artifact))
    boundary_break["recomputed_live_learning_and_consumer_scores"][
        "safety_score_controls_utility_readiness"
    ] = True
    _refresh(boundary_break)
    assert boundary_break["live_certified_learning_safety_ready_score"] == 0.0


def test_req_learn_6356_defensive_helpers_and_classifiers(tmp_path: Path) -> None:
    """REQ-LEARN-6356: helper edge cases fail closed."""

    skipped = tmp_path / "skipped.json"
    skipped.write_text(
        json.dumps({"status": "complete_null", "honest_verdict": "complete_null: no utility"}),
        encoding="utf-8",
    )
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not-json", encoding="utf-8")

    assert mod.terminal_path_receipt(tmp_path / "missing.json")["terminal_class"] == "missing"
    assert mod.classify_upstream_state(mod.terminal_path_receipt(skipped)) == "blocked_or_null"
    assert mod.classify_upstream_state(mod.terminal_path_receipt(corrupt)) == "corrupted"
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.read_json_object(corrupt) is None
    assert mod.test_exit_codes(None, ["cmd"]) == {"cmd": 0}
    assert mod.receipt_score({"ready": True}, "ready") == 0.0
    assert mod.sha256_json({"ok": True}).startswith("sha256:")
    assert mod.sha256_bytes(b"abc").startswith("sha256:")
    handling = mod.missing_upstream_and_skipped_utility_handling(
        receipts={
            "exp6352": mod.terminal_path_receipt(corrupt),
            "exp6353": mod.terminal_path_receipt(tmp_path / "missing.json"),
            "exp6354": mod.terminal_path_receipt(skipped),
            "exp6355": mod.terminal_path_receipt(skipped),
        },
        scores={},
    )
    assert handling["corrupted_upstreams"] == ["exp6352"]
    with pytest.raises(ValueError, match="unknown_attack"):
        mod.expected_decision("not_an_attack")
    with pytest.raises(ValueError, match="bad"):
        mod.require(False, "bad")
