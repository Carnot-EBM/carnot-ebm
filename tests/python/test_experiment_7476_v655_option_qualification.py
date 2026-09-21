"""Tests for REQ-VERIFY-7476 and SCENARIO-VERIFY-7476-*.

The tests replay fixed artifacts and scripted logits. They do not load a model
or claim that the later option-energy selector has scientific value.
"""

from __future__ import annotations

from copy import deepcopy
import json

from carnot import experiment_7476_v655_option_qualification as exp


def _v654_artifact() -> dict[str, object]:
    """Read the immutable input whose failed receipt starts this qualification."""

    return json.loads((exp.REPO_ROOT / exp.V654_RESULT_PATH).read_text(encoding="utf-8"))


def test_req_verify_7476_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7476 and all V655 scenarios exist before implementation."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "### REQ-VERIFY-7476:" in text
    for name in ("HEALTH", "INTERFACE", "COHORT", "PLAN", "NO-MODEL", "E2E"):
        assert f"SCENARIO-VERIFY-7476-{name}" in text


def test_scenario_verify_7476_health_preserves_exact_failed_receipt() -> None:
    """SCENARIO-VERIFY-7476-HEALTH keeps the frozen broad failure unrelated."""

    artifact = _v654_artifact()
    receipt = next(
        row
        for row in artifact["validation_receipts"]
        if row["name"] == "all_python_tests_required_once"
    )
    health = exp.classify_repository_health([receipt])

    assert receipt["exit_code"] == 2
    assert receipt["passed"] is False
    assert receipt["log_sha256"] == (
        "sha256:46e530d225d2a13970316c369662682c551da67781d2572917a73c3b6420d4da"
    )
    assert health["status"] == "degraded_unrelated_baseline"
    assert health["controls_option_protocol_readiness"] is False
    assert health["observations"][0]["classification"] == "unrelated_repository_health"
    assert health["observations"][0]["required_for_exp7476"] is False
    assert health["observations"][0]["passed"] is False


def test_scenario_verify_7476_interface_replays_real_option_helper() -> None:
    """SCENARIO-VERIFY-7476-INTERFACE replays every V654 boundary mutation."""

    interface = exp.replay_option_interface()
    original = interface["original_order"]
    reverse = interface["reversed_order"]

    assert interface["passed"] is True
    assert interface["prompt_boundary_checked"] is True
    assert original["last_evaluated_prompt_position"] == original["prompt_token_count"] - 1
    assert original["score_buffer_rows"] > original["prompt_token_count"]
    assert all(value == value for value in original["raw_logits_by_option_id"].values())
    assert set(reverse["raw_logits_by_option_id"]) == set(exp.v654.OPTION_IDS)
    assert all(interface["mutation_controls"].values())
    assert {
        "missing_label",
        "nonfinite",
        "option_id_swap",
        "unused_buffer_scores_minus_one",
    } <= set(interface["mutation_controls"])


def test_scenario_verify_7476_cohort_reduces_exact_hash_bound_roles() -> None:
    """SCENARIO-VERIFY-7476-COHORT authenticates all five V654 raw shards."""

    reduced = exp.reduce_sealed_cohort(exp.REPO_ROOT / exp.V654_RAW_DIR)

    assert reduced["counts"] == exp.EXPECTED_COUNTS
    assert reduced["total_groups"] == 534
    assert reduced["planned_cells"] == 1228
    assert reduced["all_cells_unstarted"] is True
    assert reduced["group_and_source_role_disjoint"] is True
    assert reduced["predictor_identity_and_text_complete"] is True
    assert reduced["selected_response_ids_complete"] is True
    assert reduced["label_provenance_complete"] is True
    assert reduced["licenses_preserved"] is True
    assert reduced["shard_hashes_valid"] is True
    assert reduced["passed"] is True


def test_scenario_verify_7476_plan_preserves_statistics_and_defers_efficacy() -> None:
    """SCENARIO-VERIFY-7476-PLAN freezes fit, assessment, and cost choices."""

    plan = exp.v654.comparison_plan()
    qualified = exp.qualify_comparison_plan(plan)

    assert qualified["passed"] is True
    assert qualified["binary_brier"] is True
    assert qualified["group_level_uncertainty"] is True
    assert qualified["five_fitting_seeds"] is True
    assert qualified["calibration_only_model_choice"] is True
    assert qualified["external_assessment_untouched"] is True
    assert qualified["cost_grid_preserved"] is True
    assert qualified["scientific_benefit_measured"] is False


def test_scenario_verify_7476_health_mutation_only_current_failure_closes_readiness() -> None:
    """SCENARIO-VERIFY-7476-HEALTH fails closed on current, not historical, checks."""

    artifact = _v654_artifact()
    health = exp.classify_repository_health(artifact["validation_receipts"])
    baseline = exp.build_acceptance_gates(
        interface_passed=True,
        cohort_passed=True,
        plan_passed=True,
        current_checks_passed=True,
        source_hashes_passed=True,
    )
    ready = exp.reduce_qualification(baseline, health, flagged_adversarial=False)
    assert ready["option_protocol_ready_score"] == 1
    assert ready["verdict_class"] == "null"
    assert health["observations"][0]["passed"] is False

    mutated = deepcopy(baseline)
    current = next(row for row in mutated if row["check"] == "current_required_checks")
    current["observed"] = False
    current["passed"] = False
    closed = exp.reduce_qualification(mutated, health, flagged_adversarial=False)
    assert closed["option_protocol_ready_score"] == 0
    assert closed["verdict_class"] == "disqualified"
    assert closed["repository_health_status"] == "degraded_unrelated_baseline"
    assert health["observations"][0]["classification"] == "unrelated_repository_health"
    assert all(isinstance(row["principle"], str) and row["principle"] for row in mutated)


def test_scenario_verify_7476_no_model_and_schema_contract() -> None:
    """SCENARIO-VERIFY-7476-NO-MODEL keeps qualification work non-inferential."""

    assert exp.MODEL_SPECS == []
    assert exp.MODEL_SPECS_LOWER == []
    assert set(exp.INVOCATION_COUNTS.values()) == {0}
    assert exp.INFERENCE_SUBSTRATE_CLASS == "no_model_load"
    assert exp.EXECUTION_VENUE == "host"
    assert exp.SMALL_EBM_TRAINING["attempted"] is False
    assert exp.SMALL_EBM_TRAINING["fit_attempts"] == 0

    artifact = {field: None for field in exp.REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "schema": exp.SCHEMA,
            "experiment_id": exp.EXPERIMENT_ID,
            "milestone": exp.MILESTONE,
            "run_date": exp.RUN_DATE,
            "MODEL_SPECS": [],
            "model_specs": [],
            "model_invoked": False,
            "invocation_counts": deepcopy(exp.INVOCATION_COUNTS),
            "inference_substrate_class": "no_model_load",
            "execution_venue": "host",
            "option_protocol_ready_score": 1,
            "promotion_score": 0,
            "flagged_adversarial": False,
            "field_principles": {
                field: f"Principle for {field}." for field in exp.REQUIRED_ARTIFACT_FIELDS
            },
            "reproducibility_checksum": None,
        }
    )
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert exp.validate_artifact_shape(artifact) == []
    artifact["model_invoked"] = True
    assert "current_model_contract_invalid" in exp.validate_artifact_shape(artifact)
    assert exp.validate_artifact_shape({})[0].startswith("missing_field:")
