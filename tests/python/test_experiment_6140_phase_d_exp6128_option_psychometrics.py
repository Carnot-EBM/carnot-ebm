"""Tests for Exp6140 frozen Exp6128 option psychometrics.

Spec refs: REQ-VERIFY-6140, REQ-VERIFY-6140-1, REQ-VERIFY-6140-2,
REQ-VERIFY-6140-3, REQ-VERIFY-6140-4, REQ-VERIFY-6140-5,
REQ-VERIFY-6140-6, REQ-VERIFY-6140-7, REQ-VERIFY-6140-8,
SCENARIO-VERIFY-6140-CONSERVATION,
SCENARIO-VERIFY-6140-RECONCILIATION,
SCENARIO-VERIFY-6140-OPTION-DIAGNOSTICS,
SCENARIO-VERIFY-6140-UNCERTAINTY,
SCENARIO-VERIFY-6140-TRANSFORM-ISOLATION.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_6140_phase_d_exp6128_option_psychometrics as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6140_phase_d_exp6128_option_psychometrics.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6140_phase_d_exp6128_option_psychometrics.py "
    "-m pytest tests/python/test_experiment_6140_phase_d_exp6128_option_psychometrics.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6140_phase_d_exp6128_option_psychometrics.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6140_phase_d_exp6128_option_psychometrics.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6140_phase_d_exp6128_option_psychometrics.json"
)
EXCLUSION_COMMAND = (
    ".venv/bin/python scripts/exclusion_manifest_lint.py "
    "/tmp/experiment_6140_exclusion_probe.yaml"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    EXCLUSION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def test_req_verify_6140_spec_declares_option_psychometric_contract() -> None:
    """REQ-VERIFY-6140: OpenSpec names the Exp6140 fields and principles."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6140") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-6140",
        "REQ-VERIFY-6140-1",
        "REQ-VERIFY-6140-2",
        "REQ-VERIFY-6140-3",
        "REQ-VERIFY-6140-4",
        "REQ-VERIFY-6140-5",
        "REQ-VERIFY-6140-6",
        "REQ-VERIFY-6140-7",
        "REQ-VERIFY-6140-8",
        "SCENARIO-VERIFY-6140-CONSERVATION",
        "SCENARIO-VERIFY-6140-RECONCILIATION",
        "SCENARIO-VERIFY-6140-OPTION-DIAGNOSTICS",
        "SCENARIO-VERIFY-6140-UNCERTAINTY",
        "SCENARIO-VERIFY-6140-TRANSFORM-ISOLATION",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_6140_conserves_rows_and_reconciles_source_metrics(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6140-CONSERVATION/RECONCILIATION: rows and metrics match."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=6.14,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert artifact["status"] == "retired"
    assert artifact["honest_verdict"].startswith("retired:")
    assert artifact["duration_s"] == pytest.approx(6.14)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["empirical_item_bank_design_ready_score"] == pytest.approx(0.0)
    assert artifact["retirement_triggered"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact

    counts = artifact["expected_observed_duplicate_and_missing_row_counts"]
    assert counts["expected_row_count"] == 720
    assert counts["observed_row_count"] == 720
    assert counts["unique_candidate_row_id_count"] == 720
    assert counts["duplicate_row_count"] == 0
    assert counts["missing_row_count"] == 0
    assert counts["question_group_count"] == 90
    assert counts["candidate_rows_per_question_min"] == 8
    assert counts["candidate_rows_per_question_max"] == 8

    hashes = artifact["immutable_source_artifact_and_row_hashes"]
    for path in (
        mod.EXP6103_ARTIFACT_RELATIVE_PATH,
        mod.EXP6103_ROWS_RELATIVE_PATH,
        mod.EXP6127_ARTIFACT_RELATIVE_PATH,
        mod.EXP6128_ARTIFACT_RELATIVE_PATH,
        mod.EXP6128_ROWS_RELATIVE_PATH,
    ):
        assert path.as_posix() in hashes["path_hashes"]
        assert hashes["path_hashes"][path.as_posix()].startswith("sha256:")
    assert hashes["raw_row_identity_hash"].startswith("sha256:")
    assert artifact["protected_files_unchanged"]["unchanged"] is True

    reconciliation = artifact["rederived_source_metric_reconciliation"]
    assert reconciliation["all_reconciled"] is True
    assert reconciliation["overall"]["accuracy"]["observed"] == pytest.approx(0.723611)
    assert reconciliation["overall"]["oracle_at_k"]["observed"] == pytest.approx(0.9)
    assert reconciliation["overall"]["tuned_sc_accuracy"]["observed"] == pytest.approx(0.688889)
    assert reconciliation["overall"]["oracle_minus_tuned_sc"]["observed"] == pytest.approx(0.211111)
    assert reconciliation["by_family"]["typed_finite_choice"]["accuracy"]["observed"] == pytest.approx(
        0.170833
    )


def test_scenario_verify_6140_option_position_and_fallback_diagnostics() -> None:
    """SCENARIO-VERIFY-6140-OPTION-DIAGNOSTICS: options expose the mixture."""

    artifact = mod.run(duration_s=1.0, test_commands=TEST_COMMANDS, test_exit_codes=TEST_EXIT_CODES)
    metrics = artifact[
        "family_stratum_semantic_group_relabel_shortcut_and_position_metrics"
    ]
    diagnostics = artifact[
        "wrong_option_identity_position_fallback_and_response_cluster_diagnostics"
    ]
    attribution = artifact["saturation_and_below_chance_attribution"]

    assert metrics["by_family"]["finite_domain_scheduling"]["accuracy"] == pytest.approx(1.0)
    assert metrics["by_family"]["logic_grid"]["accuracy"] == pytest.approx(1.0)
    assert metrics["by_family"]["typed_finite_choice"]["accuracy"] == pytest.approx(0.170833)
    assert metrics["by_exact_answer_position"]["1"]["accuracy"] == pytest.approx(0.902574)
    assert metrics["by_response_position"]["1"]["response_count"] == 501
    assert metrics["by_response_position"]["1"]["accuracy"] == pytest.approx(0.98004)

    finite = diagnostics["family_position_confounding"]["finite_domain_scheduling"]
    logic = diagnostics["family_position_confounding"]["logic_grid"]
    typed = diagnostics["family_position_confounding"]["typed_finite_choice"]
    assert finite["exact_position_counts"] == {"1": 240}
    assert logic["exact_position_counts"] == {"1": 240}
    assert typed["exact_position_counts"] == {"1": 64, "2": 64, "3": 56, "4": 56}
    assert diagnostics["wrong_response_label_counts"] == {"A": 104, "C": 66, "D": 29}
    assert diagnostics["fallback_concentration"]["mean_max_response_cluster_share"] == pytest.approx(
        0.875
    )
    assert diagnostics["principle"] == mod.REQUIRED_FIELD_PRINCIPLES[
        "wrong_option_identity_position_fallback_and_response_cluster_diagnostics"
    ]

    assert attribution["family_states"]["finite_domain_scheduling"]["state"] == "saturated"
    assert attribution["family_states"]["logic_grid"]["state"] == "saturated"
    assert attribution["family_states"]["typed_finite_choice"]["state"] == "below_enumerated_floor"
    assert attribution["dominant_attribution"] == (
        "saturation_plus_position_confounding_with_typed_choice_below_floor"
    )


def test_scenario_verify_6140_question_clustered_uncertainty_is_the_unit() -> None:
    """SCENARIO-VERIFY-6140-UNCERTAINTY: uncertainty does not treat draws as questions."""

    artifact = mod.run(duration_s=1.0, test_commands=TEST_COMMANDS, test_exit_codes=TEST_EXIT_CODES)
    uncertainty = artifact["question_clustered_uncertainty_and_effective_information"]

    assert uncertainty["uncertainty_method"] == "deterministic_question_cluster_bootstrap"
    assert uncertainty["individual_draws_treated_as_independent_questions"] is False
    assert uncertainty["independent_question_group_count"] == 90
    assert uncertainty["candidate_draw_count"] == 720
    assert uncertainty["bootstrap_replicates"] == mod.BOOTSTRAP_REPLICATES
    assert uncertainty["metrics"]["clustered_accuracy"]["point"] == pytest.approx(0.723611)
    assert uncertainty["metrics"]["oracle_at_k"]["point"] == pytest.approx(0.9)
    assert uncertainty["metrics"]["typed_finite_choice_accuracy"]["point"] == pytest.approx(0.170833)
    assert uncertainty["metrics"]["option_identity_additional_entropy_bits"]["point"] > 0.0
    for receipt in uncertainty["metrics"].values():
        low, high = receipt["interval_95"]
        assert low <= receipt["point"] <= high


def test_scenario_verify_6140_transform_isolation_retires_unready_source_pool() -> None:
    """SCENARIO-VERIFY-6140-TRANSFORM-ISOLATION: no held leak or new rows."""

    artifact = mod.run(duration_s=1.0, test_commands=TEST_COMMANDS, test_exit_codes=TEST_EXIT_CODES)
    spec = artifact["candidate_transformation_specification"]
    isolation = artifact["label_blind_and_held_isolation_receipt"]
    methodology = artifact["top_level_model_specs_methodology_gap_noted"]

    assert spec["new_model_rows_generated"] is False
    assert spec["selected_transformation_class"] is None
    assert spec["specification_status"] == "retired_not_frozen_for_generation"
    assert {row["class_id"] for row in spec["candidate_classes"]} == {
        "balanced_option_permutations",
        "typed_choice_normalization",
        "controlled_distractors",
        "constraint_composition_depth",
        "proof_preserving_relabels",
        "templated_paraphrases",
    }
    assert all(row["label_blind"] is True for row in spec["candidate_classes"])
    assert all(row["held_dependence"] is False for row in spec["candidate_classes"])
    assert all(row["approved_for_exp6141"] is False for row in spec["candidate_classes"])

    assert isolation["held_test_access_count"] == 0
    assert isolation["new_model_rows_generated"] is False
    assert isolation["live_inference_invoked"] is False
    assert isolation["exact_labels_read_for_evaluation_only"] is True
    assert isolation["held_outcomes_used_for_selection"] is False
    assert isolation["source_labels_altered"] is False

    assert methodology["arxiv_id"] == "2608.02966"
    assert methodology["fitted_nominal_response_model_claimed"] is False
    assert methodology["transparent_count_diagnostics_used"] is True
    assert artifact["missing_verifier_gaps"] == [
        "Exp6140 evaluates frozen exact labels but is not an oracle and cannot distinguish typed-choice true inability from distractor or position effects without new pre-frozen transformed rows.",
        "No fitted multi-model LLM-NRM is claimed from one model's immutable Exp6128 row pool.",
    ]


def test_req_verify_6140_validation_and_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-6140-1/8: schema validation and tiny helpers fail closed."""

    artifact = mod.run(duration_s=1.0, test_commands=TEST_COMMANDS, test_exit_codes=TEST_EXIT_CODES)
    blank_jsonl = tmp_path / "rows.jsonl"
    blank_jsonl.write_text('\n{"ok": true}\n', encoding="utf-8")

    assert mod.read_jsonl(blank_jsonl) == [{"ok": True}]
    assert mod._rate(0, 0) == 0.0
    assert mod._entropy([]) == 0.0
    assert mod._percentile([3.0], 0.5) == 3.0
    assert mod._percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert mod._copy_json({"b": [2, 1]}) == {"b": [2, 1]}

    middle = mod._saturation_attribution(
        {
            "overall": {"accuracy": 0.5, "oracle_minus_tuned_sc": 0.1, "parseability": 1.0, "method_validity": 1.0},
            "by_family": {
                "mid_family": {"accuracy": 0.5, "oracle_at_k": 0.7, "tuned_sc_accuracy": 0.6}
            },
        },
        {
            "family_position_confounding": {
                "mid_family": {"exact_first_rate": 0.25, "response_first_rate": 0.25}
            }
        },
    )
    assert middle["family_states"]["mid_family"]["state"] == "middle_band"

    blockers = mod._preconditions(
        rows=[],
        counts={
            "observed_row_count": 1,
            "unique_candidate_row_id_count": 1,
            "question_group_count": 1,
            "candidate_rows_per_question_min": 1,
            "candidate_rows_per_question_max": 9,
            "all_rows_calibration_split": False,
        },
        immutable={"git_status_receipt": {}, "protected_file_hashes_before": {}},
    )["blocked_reasons"]
    assert blockers == [
        "row_count_mismatch",
        "row_identity_mismatch",
        "question_group_count_mismatch",
        "question_group_missing_candidates",
        "question_group_extra_candidates",
        "non_calibration_row_present",
    ]

    bad = json.loads(json.dumps(artifact))
    bad["verifier_is_oracle"] = True
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(artifact))
    bad["empirical_item_bank_design_ready_score"] = 1.0
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    with pytest.raises(ValueError, match="ready_score_requires_nonretired"):
        mod.validate_artifact(bad)
