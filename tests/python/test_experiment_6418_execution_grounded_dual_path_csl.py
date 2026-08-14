"""Tests for Exp6418 execution-grounded dual-path CSL.

Spec refs: REQ-LEARN-6418, SCENARIO-LEARN-6418-GATES,
SCENARIO-LEARN-6418-CHRONOLOGY, SCENARIO-LEARN-6418-CAUSAL-PATHS,
SCENARIO-LEARN-6418-MATCHED-ARMS, SCENARIO-LEARN-6418-ATTACKS,
SCENARIO-LEARN-6418-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import runpy
from typing import Any

import pytest

from carnot import experiment_6418_execution_grounded_dual_path_csl as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_tests() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    return mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data_6418",
        duration_s=0.0,
        test_exit_codes=_passing_tests(),
        write=write,
    )


def _refresh(artifact: dict[str, Any]) -> dict[str, Any]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_learn_6418_spec_declares_fields_principles_and_scenarios() -> None:
    """REQ-LEARN-6418: OpenSpec owns the dual-path CSL contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6418") : text.index("REQ-LEARN-6409")]
    normalized = " ".join(section.split())

    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-LEARN-6418-GATES",
        "SCENARIO-LEARN-6418-CHRONOLOGY",
        "SCENARIO-LEARN-6418-CAUSAL-PATHS",
        "SCENARIO-LEARN-6418-MATCHED-ARMS",
        "SCENARIO-LEARN-6418-ATTACKS",
        "SCENARIO-LEARN-6418-READY",
        "execution_grounded_dual_path_csl_ready_score",
        "delta_proposal_coverage_over_frozen",
        "learning_path:proposal",
        "learning_path:selection",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section or field.startswith(("gate:", "learning_path:"))
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6418_gates_models_tokenizers_and_preconditions(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6418-GATES: authenticated preconditions revalidate."""

    artifact = _artifact(tmp_path)
    gates = artifact["exp6417_gate_receipts"]
    model_hashes = artifact["model_file_and_embedded_tokenizer_hashes"]
    cuda = artifact["cuda_offload_and_authenticated_process_receipts_by_model"]

    assert gates["all_gates_passed"] is True
    assert gates["exp6417"]["ready_score"] == 1.0
    assert gates["exp6413"]["ready_score"] == 1.0
    assert gates["exp6407"]["ready_score"] == 1.0
    assert gates["exp6397"]["ready_score"] == 1.0
    assert gates["raw_and_compiled_memory_schemas"]["gate_passed"] is True
    assert gates["rollback_receipts"]["gate_passed"] is True
    assert artifact["preconditions_checked"]["all_preconditions_passed"] is True

    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["autotokenizer_usage_count"] == 0
    assert model_hashes["model_count"] == 3
    assert model_hashes["all_model_files_present"] is True
    assert model_hashes["all_embedded_tokenizers_loadable"] is True
    assert all(row["autotokenizer_used"] is False for row in model_hashes["rows"])
    assert cuda["model_count"] == 3
    assert cuda["all_authenticated_process_receipts_present"] is True
    assert cuda["llama_cpp_cuda_offload_available"] is True
    assert artifact["cached_sota_pair_receipts"]["all_calls_made"] is True


def test_scenario_learn_6418_chronology_freeze_and_matched_arms(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6418-CHRONOLOGY: four sessions are sealed."""

    artifact = _artifact(tmp_path)
    manifest = artifact[
        "chronological_manifest_path_hash_session_drift_update_restart_expiry_supersession_counts_and_partition_seals"
    ]
    contract = artifact["preregistered_frozen_single_path_and_dual_path_arm_contract"]
    matched = artifact["matched_work_receipts"]
    freeze = artifact["raw_event_and_pre_outcome_proposal_freeze_records"]
    outcomes = artifact["exact_feasibility_and_consequence_outcome_receipts"]

    assert manifest["event_count"] == 96
    assert manifest["session_count"] == 4
    assert manifest["drift_regime_count"] == 3
    assert manifest["update_opportunity_count"] == 6
    assert manifest["process_restart_boundary_count"] == 4
    assert manifest["expiry_boundary_count"] == 2
    assert manifest["supersession_boundary_count"] == 2
    assert manifest["future_rows_sealed_before_generation"] is True
    assert manifest["partition_seals"]["future"]["used_for_training"] is False
    assert Path(manifest["path"]).is_file()

    assert set(contract["arms"]) == set(mod.ARMS)
    assert contract["future_labels_open_after_all_heads_freeze"] is True
    assert matched["all_matched"] is True
    assert len({row["event_order_sha256"] for row in matched["by_arm"].values()}) == 1
    assert len({row["model_call_count"] for row in matched["by_arm"].values()}) == 1
    assert len({row["prompt_token_count"] for row in matched["by_arm"].values()}) == 1
    assert len({row["checker_call_count"] for row in matched["by_arm"].values()}) == 1
    assert len({row["consumer_work_units"] for row in matched["by_arm"].values()}) == 1
    assert len({row["initial_heads_sha256"] for row in matched["by_arm"].values()}) == 1

    assert freeze["event_count"] == 96
    assert freeze["proposal_count"] == 288
    assert freeze["raw_bytes_frozen_before_proposals"] is True
    assert freeze["proposals_frozen_before_exact_outcomes"] is True
    assert freeze["future_label_visible_before_freeze_count"] == 0
    assert outcomes["causal_order_preserved"] is True
    assert outcomes["feasibility_label_count"] == 96
    assert outcomes["consequence_label_count"] == 96
    assert outcomes["label_opened_before_proposal_freeze_count"] == 0


def test_scenario_learn_6418_causal_paths_dispositions_and_bindings(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6418-CAUSAL-PATHS: exact labels update separate heads."""

    artifact = _artifact(tmp_path)
    proposal = artifact["proposal_memory_schema_head_and_transition_history"]
    selection = artifact["selection_memory_schema_head_and_transition_history"]
    bindings = artifact["predecessor_license_checker_expiry_and_supersession_bindings"]
    dispositions = artifact["atomic_disposition_records"]
    counts = artifact["commit_reject_quarantine_and_defer_counts_by_path_and_session"]

    assert proposal["schema"]["path_kind"] == "proposal_coverage_memory"
    assert proposal["update_source"] == "exact_feasibility_outcomes_only"
    assert proposal["consequence_label_update_count"] == 0
    assert proposal["commit_count"] > 0
    assert proposal["noncommit_head_change_count"] == 0
    assert selection["schema"]["path_kind"] == "selection_consequence_memory"
    assert selection["update_source"] == "exact_observed_consequences_only"
    assert selection["feasibility_label_update_count"] == 0
    assert selection["commit_count"] > 0
    assert selection["terminal_head_hash"] != proposal["terminal_head_hash"]

    assert bindings["binding_count"] == 96
    assert bindings["all_predecessors_fresh_or_deferred"] is True
    assert bindings["all_commits_license_valid"] is True
    assert bindings["all_commits_exact_supported"] is True
    assert bindings["expired_or_superseded_commit_count"] == 0
    assert dispositions["record_count"] == 192
    assert dispositions["all_have_single_atomic_disposition"] is True
    assert dispositions["exact_veto_override_count"] == 0
    assert counts["proposal"]["all_sessions_have_counts"] is True
    assert counts["selection"]["all_sessions_have_counts"] is True
    assert sum(row["Commit"] for row in counts["proposal"]["by_session"].values()) > 0
    assert sum(row["Commit"] for row in counts["selection"]["by_session"].values()) > 0


def test_scenario_learn_6418_metrics_ready_oracle_and_terminal_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6418-READY: exact-governed future gain gates readiness."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", mod.RUN_DATE, "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    results = artifact[
        "per_arm_session_model_and_family_proposal_coverage_selection_success_future_yield_transfer_retention_forgetting_negative_transfer_contamination_growth_escalation_restart_and_cost_results"
    ]

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert results["by_arm"][mod.DUAL_PATH_ARM]["future_exact_yield"] > results["by_arm"][
        mod.FROZEN_ARM
    ]["future_exact_yield"]
    assert artifact["delta_proposal_coverage_over_frozen"] > 0.0
    assert artifact["delta_selection_success_over_frozen"] > 0.0
    assert artifact["delta_future_exact_yield_over_frozen"] > 0.0
    assert all(math.isfinite(float(artifact[field])) for field in mod.BARE_FINITE_FIELDS)
    assert artifact["contamination_propagation_rate"] == 0.0
    assert artifact["forgetting_delta"] >= 0.0
    assert artifact["protected_leakage_count"] == 0
    assert artifact["same_step_write_count"] == 0
    assert artifact["exact_veto_override_count"] == 0
    assert artifact["model_weight_change_count"] == 0
    assert artifact["execution_grounded_dual_path_csl_ready_score"] == 1.0
    assert artifact["public_factor_claim_eligibility"]["eligible"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is True

    oracle = artifact["verifier_is_oracle"]
    assert oracle["value"] is True
    assert set(oracle["true_for"]) == {
        "exact_feasibility_checker",
        "exact_consequence_checker",
        "exact_release_checker",
        "exact_retention_checker",
    }
    for forbidden in ("proposal_memory", "selection_memory", "model_output"):
        assert oracle["false_for"][forbidden] is False


def test_scenario_learn_6418_attacks_and_negative_readiness_gates(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6418-ATTACKS: unsafe dual-path authority fails closed."""

    artifact = _artifact(tmp_path)
    attacks = artifact["attack_matrix"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_fail_closed"] is True
    assert attacks["committed_attack_count"] == 0
    assert attacks["readiness_promoted_attack_count"] == 0

    negative_cases = {
        "proposal_path_not_causal": lambda row: row[
            "proposal_memory_schema_head_and_transition_history"
        ].update({"causal_exact_outcome_count": 0}),
        "selection_path_not_causal": lambda row: row[
            "selection_memory_schema_head_and_transition_history"
        ].update({"causal_exact_outcome_count": 0}),
        "no_future_gain": lambda row: row.update(
            {"delta_future_exact_yield_over_frozen": 0.0}
        ),
        "contamination": lambda row: row.update({"contamination_propagation_rate": 0.1}),
        "forgetting": lambda row: row.update({"forgetting_delta": -0.1}),
        "unbounded_growth": lambda row: row[
            "per_arm_session_model_and_family_proposal_coverage_selection_success_future_yield_transfer_retention_forgetting_negative_transfer_contamination_growth_escalation_restart_and_cost_results"
        ].update({"growth_bounded": False}),
        "attack_survivor": lambda row: row["attack_matrix"].update({"all_fail_closed": False}),
        "failed_test": lambda row: row["tests_run"]["exit_codes"].update(
            {mod.DEFAULT_TEST_COMMANDS[0]: 1}
        ),
    }
    for mutate in negative_cases.values():
        candidate = deepcopy(artifact)
        mutate(candidate)
        _refresh(candidate)
        assert candidate["execution_grounded_dual_path_csl_ready_score"] == 0.0

    mutations = [
        ("required_fields", lambda data: data.pop("field_principles")),
        ("required_fields", lambda data: data.__setitem__("extra", True)),
        ("field_principles", lambda data: data["field_principles"].pop("status")),
        ("field_principles", lambda data: data["field_principles"].pop("gate:exp6417")),
        ("field_provenance", lambda data: data["field_provenance"].pop("status")),
        ("bare_finite", lambda data: data.__setitem__("delta_proposal_coverage_over_frozen", "bad")),
        ("contamination_propagation_rate", lambda data: data.__setitem__("contamination_propagation_rate", 1)),
        ("forgetting_delta", lambda data: data.__setitem__("forgetting_delta", -1)),
        ("protected_leakage_count", lambda data: data.__setitem__("protected_leakage_count", 1)),
        ("same_step_write_count", lambda data: data.__setitem__("same_step_write_count", 1)),
        ("exact_veto_override_count", lambda data: data.__setitem__("exact_veto_override_count", 1)),
        ("model_weight_change_count", lambda data: data.__setitem__("model_weight_change_count", 1)),
        ("attack_matrix", lambda data: data["attack_matrix"]["rows"][0].__setitem__("fail_closed", False)),
        ("verifier_is_oracle", lambda data: data["verifier_is_oracle"]["false_for"].__setitem__("proposal_memory", True)),
        ("readiness", lambda data: data.__setitem__("execution_grounded_dual_path_csl_ready_score", 0.0)),
        ("status", lambda data: data.__setitem__("status", "bad")),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        ("reproducibility_checksum", lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad")),
    ]
    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_learn_6418_helpers_and_fail_closed_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6418: helper failures are explicit and writes are stable."""

    assert mod.sha256_json({"ok": True}).startswith("sha256:")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod.as_mapping([]) == {}
    with pytest.raises(ValueError, match="forced"):
        mod.require(False, "forced")

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_top_level_not_object"):
        mod.read_json(non_object)

    context = mod.load_context(REPO)
    bad_context = deepcopy(context)
    bad_context["exp6417"]["authentic_write_time_admission_ready_score"] = 0.0
    bad_context["exp6413"]["authenticated_receipt_contract_ready_score"] = 0.0
    bad_context["exp6407"]["provenance_tiered_memory_protocol_ready_score"] = 0.0
    bad_context["exp6397"]["transactional_continuous_self_learning_ready_score"] = 0.0
    gates = mod.exp6417_gate_receipts(REPO, bad_context)
    assert gates["all_gates_passed"] is False
    assert {
        "exp6417_gate_failed",
        "exp6413_receipt_gate_failed",
        "exp6407_schema_gate_failed",
        "exp6397_transaction_gate_failed",
    } <= set(gates["blocked_reasons"])

    missing_schema_context = deepcopy(context)
    missing_schema_context["exp6407"]["raw_record_schema_path_hash_and_required_fields"][
        "schema_path"
    ] = str(tmp_path / "missing_raw_schema.json")
    missing_schema_context["exp6397"][
        "stale_duplicate_self_approval_concurrency_interrupt_and_restart_attack_matrix"
    ]["all_fail_closed"] = False
    missing_schema_gates = mod.exp6417_gate_receipts(REPO, missing_schema_context)
    assert {
        "memory_schema_missing",
        "rollback_gate_failed",
    } <= set(missing_schema_gates["blocked_reasons"])

    missing_model_context = deepcopy(context)
    missing_model_context["exp6413"]["MODEL_SPECS"][0]["model_file_sha256"] = "sha256:bad"
    model_hashes = mod.model_file_and_embedded_tokenizer_hashes(missing_model_context)
    assert model_hashes["all_model_hashes_match"] is False

    failing_preconditions = mod.preconditions_checked(
        date="20260101",
        gates={"all_gates_passed": False},
        model_hashes={"all_model_files_present": False, "all_embedded_tokenizers_loadable": False},
        cuda={"all_authenticated_process_receipts_present": False},
        manifest={"event_count": 0, "future_rows_sealed_before_generation": False},
        matched={"all_matched": False},
        protected_before={"missing": None},
        source_before={"missing": None},
    )
    assert {
        "wrong_planning_date",
        "upstream_gate_failed",
        "model_file_gate_failed",
        "embedded_tokenizer_gate_failed",
        "process_receipt_gate_failed",
        "chronological_manifest_too_short",
        "future_rows_not_sealed",
        "matched_work_failed",
        "protected_hash_missing",
        "source_hash_missing",
    } == set(failing_preconditions["blocked_reasons"])

    no_write = _artifact(tmp_path, write=False)
    assert no_write["reproducibility_checksum"] == mod.payload_checksum(no_write)
    blocked = deepcopy(no_write)
    blocked["preconditions_checked"]["all_preconditions_passed"] = False
    mod.refresh_terminal_fields(blocked)
    assert blocked["status"] == "blocked_precondition"
    assert blocked["honest_verdict"].startswith("blocked:")

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    written = mod.run(
        date=mod.RUN_DATE,
        result_path=output,
        data_dir=tmp_path / "written_data_6418",
        duration_s=0.0,
        test_exit_codes=_passing_tests(),
        write=True,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == written

    cli_output = tmp_path / "script_guard.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_6418_execution_grounded_dual_path_csl",
            "--date",
            mod.RUN_DATE,
            "--output",
            str(cli_output),
            "--data-dir",
            str(tmp_path / "script_guard_data"),
            "--validate",
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module(
            "carnot.experiment_6418_execution_grounded_dual_path_csl",
            run_name="__main__",
        )
    assert exit_info.value.code == 0
    assert cli_output.is_file()
