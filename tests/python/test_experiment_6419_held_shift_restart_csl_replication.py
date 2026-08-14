"""Tests for Exp6419 held-shift restart CSL replication.

Spec refs: REQ-LEARN-6419, SCENARIO-LEARN-6419-FREEZE,
SCENARIO-LEARN-6419-SHIFTS, SCENARIO-LEARN-6419-MATCHED-ARMS,
SCENARIO-LEARN-6419-NO-RETUNE, SCENARIO-LEARN-6419-ATTACKS,
SCENARIO-LEARN-6419-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import runpy
from typing import Any

import pytest

from carnot import experiment_6419_held_shift_restart_csl_replication as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_tests() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    return mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data_6419",
        duration_s=0.0,
        test_exit_codes=_passing_tests(),
        write=write,
    )


def _refresh(artifact: dict[str, Any]) -> dict[str, Any]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_learn_6419_spec_declares_fields_principles_and_scenarios() -> None:
    """REQ-LEARN-6419: OpenSpec owns the held-shift replication contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6419") : text.index("REQ-LEARN-6409")]
    normalized = " ".join(section.split())

    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-LEARN-6419-FREEZE",
        "SCENARIO-LEARN-6419-SHIFTS",
        "SCENARIO-LEARN-6419-MATCHED-ARMS",
        "SCENARIO-LEARN-6419-NO-RETUNE",
        "SCENARIO-LEARN-6419-ATTACKS",
        "SCENARIO-LEARN-6419-READY",
        "held_shift_csl_replication_ready_score",
        "held_delta_future_exact_yield_over_frozen",
        "shift:model_family",
        "shift:constraint_family",
        "shift:surface_form",
        "shift:temporal",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section or field.startswith(("gate:", "shift:"))
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6419_freeze_models_tokenizers_and_preconditions(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6419-FREEZE: the held run starts from a frozen learner."""

    artifact = _artifact(tmp_path)
    gates = artifact["exp6418_gate_receipts"]
    frozen = artifact["frozen_mechanism_config_checker_model_and_prompt_hashes"]
    tokenizers = artifact["embedded_gguf_tokenizer_receipts"]

    assert gates["all_gates_passed"] is True
    assert gates["exp6418"]["ready_score"] == 1.0
    assert gates["exp6414"]["ready_score"] == 1.0
    assert artifact["preconditions_checked"]["all_preconditions_passed"] is True
    assert frozen["held_outcomes_opened_after_freeze"] is True
    assert frozen["post_outcome_hashes_match_frozen_hashes"] is True
    assert frozen["exp6418_artifact_sha256"].startswith("sha256:")
    assert frozen["frozen_dual_path_head_hashes"]["proposal"].startswith("sha256:")
    assert frozen["frozen_dual_path_head_hashes"]["selection"].startswith("sha256:")

    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["cached_sota_pair_receipts"]["all_calls_made"] is True
    assert artifact["autotokenizer_usage_count"] == 0
    assert tokenizers["model_count"] == 3
    assert tokenizers["all_embedded_tokenizers_loadable"] is True
    assert all(row["autotokenizer_used"] is False for row in tokenizers["rows"])
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE


def test_scenario_learn_6419_shifts_manifest_absence_and_receipts(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6419-SHIFTS: held rows cover all declared shifts."""

    artifact = _artifact(tmp_path)
    manifest = artifact[
        "held_manifest_path_hash_shift_counts_restart_expiry_supersession_counts_and_partition_seals"
    ]
    absence = artifact["held_manifest_absence_before_freeze_receipt"]
    receipts = artifact["authenticated_process_and_raw_output_receipts_by_model"]

    assert manifest["event_count"] == 72
    assert manifest["chronological_order_preserved"] is True
    assert manifest["model_family_shift_count"] == 3
    assert manifest["constraint_family_shift_count"] == 4
    assert manifest["surface_form_shift_count"] == 4
    assert manifest["temporal_shift_count"] == 4
    assert manifest["process_restart_boundary_count"] >= 3
    assert manifest["expiry_boundary_count"] >= 2
    assert manifest["supersession_boundary_count"] >= 2
    assert manifest["partition_seals"]["future"]["used_for_training"] is False
    assert manifest["partition_seals"]["future"]["evaluated_once"] is True
    assert Path(manifest["path"]).is_file()

    assert absence["absent_during_mechanism_selection"] is True
    assert absence["held_manifest_hash_present_in_exp6418_artifact"] is False
    assert absence["held_manifest_path_present_in_exp6418_source"] is False
    assert absence["generation_started_after_mechanism_freeze"] is True

    assert receipts["model_count"] == 3
    assert receipts["row_count"] == 72
    assert receipts["all_process_receipts_accepted"] is True
    assert receipts["all_raw_outputs_frozen_before_outcomes"] is True
    assert receipts["raw_output_substitution_count"] == 0
    assert set(receipts["by_model"]) == set(mod.MANDATED_MODEL_IDS)


def test_scenario_learn_6419_matched_arms_no_retune_and_metrics(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6419-MATCHED-ARMS: held arms use equal work."""

    artifact = _artifact(tmp_path)
    matched = artifact["matched_arm_work_receipts"]
    no_retune = artifact["no_post_outcome_retuning_receipts"]
    results = artifact[
        "per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results"
    ]

    assert set(matched["arms"]) == set(mod.ARMS)
    assert matched["all_matched"] is True
    assert len({row["event_order_sha256"] for row in matched["by_arm"].values()}) == 1
    assert len({row["model_call_count"] for row in matched["by_arm"].values()}) == 1
    assert len({row["prompt_token_count"] for row in matched["by_arm"].values()}) == 1
    assert len({row["checker_call_count"] for row in matched["by_arm"].values()}) == 1
    assert len({row["raw_output_receipt_count"] for row in matched["by_arm"].values()}) == 1
    assert len({row["latency_surface_sha256"] for row in matched["by_arm"].values()}) == 1
    assert len({row["gpu_cost_surface_sha256"] for row in matched["by_arm"].values()}) == 1

    assert no_retune["retune_count"] == 0
    assert no_retune["held_outcome_evaluation_count"] == 1
    assert no_retune["all_hashes_match"] is True
    assert no_retune["incompatibility_policy"] == "record_as_harm_or_abstention"

    dual = results["by_arm"][mod.FROZEN_DUAL_PATH_ARM]
    frozen = results["by_arm"][mod.FROZEN_ARM]
    assert dual["future_exact_yield"] > frozen["future_exact_yield"]
    assert artifact["held_delta_future_exact_yield_over_frozen"] > 0.0
    assert math.isfinite(artifact["held_delta_future_exact_yield_over_frozen"])
    assert artifact["held_contamination_propagation_rate"] == 0.0
    assert artifact["held_forgetting_delta"] >= 0.0
    assert results["growth_bounded"] is True
    assert results["restart_recovery_success"] is True
    assert results["protected_retention_regression_count"] == 0
    for key in ("model_family", "constraint_family", "surface_form", "temporal"):
        assert results["by_shift"][key]
    assert set(results["by_model"]) == set(mod.MANDATED_MODEL_IDS)
    assert len(results["by_session"]) >= 4


def test_scenario_learn_6419_attacks_ready_oracle_and_terminal_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6419-READY: exact held gain gates readiness."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert (
        mod.main(
            [
                "--date",
                mod.RUN_DATE,
                "--output",
                str(output),
                "--data-dir",
                str(tmp_path / "cli_data_6419"),
                "--validate",
            ]
        )
        == 0
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    attacks = artifact["attack_matrix"]

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert set(mod.GATE_AND_SHIFT_PRINCIPLE_KEYS) <= set(artifact["field_principles"])
    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_fail_closed"] is True
    assert attacks["committed_attack_count"] == 0
    assert attacks["readiness_promoted_attack_count"] == 0
    assert artifact["protected_leakage_count"] == 0
    assert artifact["silent_fallback_count"] == 0
    assert artifact["held_shift_csl_replication_ready_score"] == 1.0
    assert artifact["public_factor_claim_eligibility"]["eligible"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is True

    oracle = artifact["verifier_is_oracle"]
    assert oracle["value"] is True
    assert set(oracle["true_for"]) == {"exact_outcome_checker", "exact_retention_checker"}
    assert oracle["false_for"] == {
        "model_output": False,
        "proposal_memory": False,
        "selection_memory": False,
    }


def test_req_learn_6419_helpers_fail_closed_paths_and_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6419: helper failures and readiness failures are explicit."""

    artifact = _artifact(tmp_path, write=False)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.sha256_json({"ok": True}).startswith("sha256:")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod.as_mapping([]) == {}
    with pytest.raises(ValueError, match="forced"):
        mod.require(False, "forced")

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_top_level_not_object"):
        mod.read_json(bad_json)

    context = mod.load_context(REPO)
    bad_context = deepcopy(context)
    bad_context["exp6418"]["execution_grounded_dual_path_csl_ready_score"] = 0.0
    bad_context["exp6418"]["delta_future_exact_yield_over_frozen"] = 0.0
    bad_context["exp6414"]["fresh_factor_event_corpus_ready_score"] = 0.0
    bad_context["exp6413"]["authenticated_receipt_contract_ready_score"] = 0.0
    gates = mod.exp6418_gate_receipts(REPO, bad_context)
    assert {
        "exp6418_gate_failed",
        "exp6418_no_prospective_improvement",
        "exp6414_held_source_gate_failed",
        "exp6413_receipt_gate_failed",
    } <= set(gates["blocked_reasons"])

    failing_preconditions = mod.preconditions_checked(
        date="20260101",
        gates={"all_gates_passed": False},
        tokenizers={"all_embedded_tokenizers_loadable": False},
        process_receipts={"all_process_receipts_accepted": False},
        manifest={
            "event_count": 0,
            "process_restart_boundary_count": 0,
            "partition_seals": {"future": {"used_for_training": True}},
        },
        absence={"absent_during_mechanism_selection": False},
        matched={"all_matched": False},
        no_retune={"all_hashes_match": False, "retune_count": 1},
        protected_before={"missing": None},
        source_before={"missing": None},
    )
    assert {
        "wrong_planning_date",
        "upstream_gate_failed",
        "embedded_tokenizer_gate_failed",
        "process_receipt_gate_failed",
        "held_manifest_too_short",
        "restart_boundary_missing",
        "future_partition_touched",
        "held_manifest_absence_failed",
        "matched_work_failed",
        "post_outcome_retune_detected",
        "protected_hash_missing",
        "source_hash_missing",
    } == set(failing_preconditions["blocked_reasons"])

    blocked = deepcopy(artifact)
    blocked["preconditions_checked"]["all_preconditions_passed"] = False
    mod.refresh_terminal_fields(blocked)
    assert blocked["status"] == "blocked_precondition"
    assert blocked["honest_verdict"].startswith("blocked:")

    negative_cases = {
        "no_future_gain": lambda row: row.update(
            {"held_delta_future_exact_yield_over_frozen": 0.0}
        ),
        "contamination": lambda row: row.update({"held_contamination_propagation_rate": 0.1}),
        "forgetting": lambda row: row.update({"held_forgetting_delta": -0.1}),
        "unbounded_growth": lambda row: row[
            "per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results"
        ].update({"growth_bounded": False}),
        "restart_failure": lambda row: row[
            "per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results"
        ].update({"restart_recovery_success": False}),
        "retune": lambda row: row["no_post_outcome_retuning_receipts"].update(
            {"retune_count": 1}
        ),
        "attack_survivor": lambda row: row["attack_matrix"].update({"all_fail_closed": False}),
        "protected_change": lambda row: row["protected_files_unchanged"].update(
            {"unchanged": False}
        ),
        "failed_test": lambda row: row["tests_run"]["exit_codes"].update(
            {mod.DEFAULT_TEST_COMMANDS[0]: 1}
        ),
        "leakage": lambda row: row.update({"protected_leakage_count": 1}),
        "fallback": lambda row: row.update({"silent_fallback_count": 1}),
    }
    for mutate in negative_cases.values():
        candidate = deepcopy(artifact)
        mutate(candidate)
        _refresh(candidate)
        assert candidate["held_shift_csl_replication_ready_score"] == 0.0

    mutations = [
        ("required_fields", lambda data: data.pop("field_principles")),
        ("required_fields", lambda data: data.__setitem__("extra", True)),
        ("field_principles", lambda data: data["field_principles"].pop("status")),
        ("field_principles", lambda data: data["field_principles"].pop("shift:temporal")),
        ("field_provenance", lambda data: data["field_provenance"].pop("status")),
        ("bare_finite", lambda data: data.__setitem__("held_delta_future_exact_yield_over_frozen", "bad")),
        ("held_contamination_propagation_rate", lambda data: data.__setitem__("held_contamination_propagation_rate", 1)),
        ("held_forgetting_delta", lambda data: data.__setitem__("held_forgetting_delta", -1)),
        ("protected_leakage_count", lambda data: data.__setitem__("protected_leakage_count", 1)),
        ("silent_fallback_count", lambda data: data.__setitem__("silent_fallback_count", 1)),
        ("attack_matrix", lambda data: data["attack_matrix"]["rows"][0].__setitem__("fail_closed", False)),
        ("verifier_is_oracle", lambda data: data["verifier_is_oracle"]["false_for"].__setitem__("model_output", True)),
        ("readiness", lambda data: data.__setitem__("held_shift_csl_replication_ready_score", 0.0)),
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

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    written = mod.run(
        date=mod.RUN_DATE,
        result_path=output,
        data_dir=tmp_path / "written_data_6419",
        duration_s=0.0,
        test_exit_codes=_passing_tests(),
        write=True,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == written

    cli_output = tmp_path / "script_guard.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_6419_held_shift_restart_csl_replication",
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
            "carnot.experiment_6419_held_shift_restart_csl_replication",
            run_name="__main__",
        )
    assert exit_info.value.code == 0
    assert cli_output.is_file()
