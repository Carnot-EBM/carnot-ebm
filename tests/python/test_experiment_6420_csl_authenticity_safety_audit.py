"""Tests for Exp6420 CSL authenticity and safety audit.

Spec refs: REQ-LEARN-6420, SCENARIO-LEARN-6420-MISSING,
SCENARIO-LEARN-6420-CAUSAL, SCENARIO-LEARN-6420-METRICS,
SCENARIO-LEARN-6420-ATTACKS, SCENARIO-LEARN-6420-ORACLE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6420_csl_authenticity_safety_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_tests() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = False) -> dict[str, Any]:
    return mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=0.0,
        test_exit_codes=_passing_tests(),
        write=write,
    )


def test_req_learn_6420_spec_declares_required_fields_and_scenarios() -> None:
    """REQ-LEARN-6420: OpenSpec owns the Exp6420 audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6420") : text.index("REQ-LEARN-6409")]

    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-LEARN-6420-MISSING",
        "SCENARIO-LEARN-6420-CAUSAL",
        "SCENARIO-LEARN-6420-METRICS",
        "SCENARIO-LEARN-6420-ATTACKS",
        "SCENARIO-LEARN-6420-ORACLE",
        "csl_authenticity_safety_audit_ready_score",
        "verifier_is_oracle",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section

    for attack_id in mod.ATTACK_IDS:
        assert attack_id.replace("_", " ") in section or attack_id in section


def test_scenario_learn_6420_current_chain_is_complete_but_not_ready(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6420-METRICS: row recompute mismatches lower readiness."""

    artifact = _artifact(tmp_path)

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.ATTACK_PRINCIPLE_KEYS) <= set(artifact["field_principles"])
    assert artifact["expected_and_available_upstream_inputs"]["missing_required_count"] == 0
    assert artifact["missing_input_findings"]["missing_required_count"] == 0
    assert artifact["process_and_raw_output_authenticity_rechecks"]["all_rechecks_passed"] is True
    assert artifact["reconstructed_event_time_order"]["causal_order_holds"] is True
    assert artifact["proposal_precedes_outcome_checks"]["all_proposals_precede_outcomes"] is True
    assert artifact["update_follows_exact_feedback_checks"]["all_updates_follow_exact_feedback"] is True
    assert artifact["untouched_future_partition_checks"]["future_partition_untouched"] is True
    assert artifact["proposal_memory_exact_feasibility_bindings"]["all_commits_have_exact_feasibility"] is True
    assert artifact["selection_memory_exact_consequence_bindings"]["all_commits_have_exact_consequence"] is True

    assert artifact["reported_vs_recomputed_deltas"]["all_reported_match_recomputed"] is False
    assert artifact["reported_vs_recomputed_deltas"]["mismatch_count"] > 0
    assert artifact["attack_matrix"]["all_critical_attacks_fail_closed"] is False
    assert "raw_output_reuse" in artifact["attack_matrix"]["open_critical_attack_ids"]
    assert "same_step_writes" not in artifact["attack_matrix"]["open_critical_attack_ids"]
    assert "model_identity_swap" not in artifact["attack_matrix"]["open_critical_attack_ids"]
    assert artifact["csl_authenticity_safety_audit_ready_score"] == 0.0
    assert artifact["prospective_csl_claim_eligibility"]["eligible"] is False
    assert artifact["public_factor_claim_eligibility"]["eligible"] is False
    assert artifact["status"] == "complete_null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["verifier_is_oracle"]["value"] is False
    assert artifact["verifier_is_oracle"]["audit_is_oracle"] is False
    assert "exp6412_historical" in artifact["adversarial_and_determination_preservation_findings"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is True

    blocker_probe = deepcopy(artifact)
    blocker_probe["exact_veto_override_count"] = 1
    blocker_probe["protected_leakage_count"] = 1
    blocker_probe["hidden_retuning_count"] = 1
    blockers = mod.prospective_csl_claim_eligibility(blocker_probe)["blockers"]
    assert "exact_veto_override" in blockers
    assert "protected_leakage" in blockers
    assert "hidden_retuning" in blockers


def test_scenario_learn_6420_missing_expected_input_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-6420-MISSING: missing expected inputs lower eligibility."""

    monkeypatch.setattr(
        mod,
        "EXPECTED_INPUTS",
        mod.EXPECTED_INPUTS
        + (
            mod.ExpectedInput(
                role="artifact",
                path=Path("results/experiment_6420_missing_required_fixture.json"),
                required=True,
                principle_key="missing_input:required_artifact",
            ),
        ),
    )
    artifact = _artifact(tmp_path)

    assert artifact["missing_input_findings"]["missing_required_count"] == 1
    assert artifact["preconditions_checked"]["all_required_inputs_available"] is False
    assert artifact["csl_authenticity_safety_audit_ready_score"] == 0.0
    assert "missing_required_inputs" in artifact["harm_underpowered_missing_and_flagged_cells"]["visible_harm_reasons"]
    assert mod.read_json(tmp_path / "missing.json") == {}
    with pytest.raises(ValueError, match="forced"):
        mod.require(False, "forced")

    model_file = tmp_path / "model.gguf"
    model_file.write_text("tiny model bytes", encoding="utf-8")
    digest, method = mod._model_byte_digest(model_file)
    assert digest == mod.sha256_file(model_file)
    assert method == "sha256_file"


def test_scenario_learn_6420_causal_and_update_checks_detect_tampering() -> None:
    """SCENARIO-LEARN-6420-CAUSAL: temporal tampering is detected."""

    context = mod.load_context(REPO)
    causal = mod.proposal_precedes_outcome_checks(context["exp6418"])
    updates = mod.update_follows_exact_feedback_checks(context["exp6418"])
    assert causal["all_proposals_precede_outcomes"] is True
    assert updates["all_updates_follow_exact_feedback"] is True

    tampered = deepcopy(context["exp6418"])
    tampered["raw_event_and_pre_outcome_proposal_freeze_records"]["rows"][0][
        "proposal_freeze_order"
    ] = 99
    tampered["proposal_memory_schema_head_and_transition_history"]["transitions"][0][
        "event_id"
    ] = "missing-event"

    assert mod.proposal_precedes_outcome_checks(tampered)["all_proposals_precede_outcomes"] is False
    assert mod.update_follows_exact_feedback_checks(tampered)[
        "all_updates_follow_exact_feedback"
    ] is False


def test_scenario_learn_6420_metric_recompute_and_attack_tampering() -> None:
    """SCENARIO-LEARN-6420-ATTACKS: attack and metric gates fail closed."""

    context = mod.load_context(REPO)
    recomputed = mod.recomputed_development_and_held_metrics(context)
    deltas = mod.reported_vs_recomputed_deltas(context, recomputed)
    attacks = mod.attack_matrix(context, mod.process_and_raw_output_authenticity_rechecks(context))

    assert recomputed["development"]["future_exact_yield"] == 0.0
    assert recomputed["held"]["future_exact_yield"] < context["exp6419"][
        "per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results"
    ]["by_arm"]["frozen_dual_path_execution_grounded"]["future_exact_yield"]
    assert deltas["all_reported_match_recomputed"] is False
    assert attacks["rows_by_attack"]["raw_output_reuse"]["fail_closed"] is False

    tampered = deepcopy(context)
    tampered["exp6419"]["no_post_outcome_retuning_receipts"]["retune_count"] = 1
    hidden = mod.hidden_retuning_count(tampered)
    tampered_attacks = mod.attack_matrix(
        tampered,
        mod.process_and_raw_output_authenticity_rechecks(tampered),
    )
    assert hidden == 1
    assert tampered_attacks["rows_by_attack"]["hidden_retuning"]["fail_closed"] is False


def test_scenario_learn_6420_cli_writes_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6420-ORACLE: CLI writes a non-oracle audit artifact."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", mod.RUN_DATE, "--output", str(output), "--validate"]) == 0
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["verifier_is_oracle"]["value"] is False
    assert saved["tests_run"]["all_passed"] is True
    assert saved["protected_files_unchanged"]["unchanged"] is True
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert mod.validate_artifact(saved) is True
