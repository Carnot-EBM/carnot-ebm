"""Tests for Exp 3707 selection-diagnosis formal closure.

Spec: REQ-REPORT-3707, SCENARIO-REPORT-3707-CLOSED,
SCENARIO-REPORT-3707-OPEN.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import selection_diagnosis_formal_closure_3707 as exp


CLOSED_VERDICT = (
    "complete: selection_diagnosis_formally_closed_retirement_recommended_to_operator"
)
OPEN_VERDICT = "complete: selection_diagnosis_cannot_close_open_question"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _seed_repo(root: Path) -> None:
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired_questions: []\n",
        encoding="utf-8",
    )
    (root / "scripts" / "research_conductor.py").write_text(
        "# conductor fixture\n",
        encoding="utf-8",
    )
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.339\n"
        "notes:\n"
        "  - project_energy_selection_thesis_bounded records energy-selection "
        "as settled-bounded.\n"
        "  - project_orthogonality_stall is background only.\n",
        encoding="utf-8",
    )
    (root / "research-references.md").write_text(
        "- Reward Model Selection Crisis (arXiv:2512.23067): discrimination "
        "and selection utility decouple.\n"
        "- Reward Learning from Best-of-N (arXiv:2605.30619): margin-vs-"
        "connectivity tradeoff explains downstream selection failures.\n",
        encoding="utf-8",
    )

    _write_json(
        root / "results" / "experiment_3672_ensemble_selection_where_sc_weak.json",
        {
            "honest_verdict": (
                "complete: ensemble_no_selection_value_even_with_headroom_sc_weak_"
                "earned_negative"
            ),
            "ensemble_selection_accuracy": 0.344262,
            "confidence_selection_accuracy": 0.344262,
            "sc_accuracy": 0.459016,
            "oracle_bestofn_accuracy": 0.606557,
            "flip_count": 28,
            "positive_control_valid": True,
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "duration_s": 0.7,
        },
    )
    _write_json(
        root / "results" / "experiment_3682_discrimination_vs_selection_gap.json",
        {
            "honest_verdict": (
                "complete: selection_gap_fundamental_no_fix_beats_sc_"
                "discrimination_decoupled_as_2512_23067"
            ),
            "flagged_adversarial": True,
            "per_candidate_auroc": 0.555508,
            "ensemble_selection_accuracy": 0.344262,
            "selection_accuracy_per_question_normalized": 0.344262,
            "self_certainty_selection_accuracy": 0.344262,
            "selection_gap_closed": False,
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "duration_s": 0.8,
        },
    )
    _write_json(
        root / "results" / "experiment_3694_selection_gap_proper_rediagnosis.json",
        {
            "honest_verdict": "complete: blocked_no_multi_candidate_corpus",
            "adversarial_verify_clean": True,
            "block_reason": "cached per-candidate energy corpus unavailable",
            "n_examples": 0,
            "per_candidate_auroc": None,
            "positive_control_valid": False,
            "selection_gap_closed": False,
            "duration_s": 0.13,
        },
    )


@pytest.mark.parametrize(
    (
        "case_name",
        "mutate_inputs",
        "expected_closed",
        "expected_verdict",
    ),
    [
        pytest.param(
            "closure_recorded_retirement_recommended",
            lambda payloads, references_text, roadmap_text: (
                payloads,
                references_text,
                roadmap_text,
            ),
            True,
            CLOSED_VERDICT,
            id="closure_recorded_retirement_recommended",
        ),
        pytest.param(
            "cannot_close_open_question",
            lambda payloads, references_text, roadmap_text: (
                {
                    **payloads,
                    "exp3694": {
                        **payloads["exp3694"],
                        "honest_verdict": "complete: selection_gap_fixed",
                    },
                },
                references_text.replace("arXiv:2605.30619", "missing-reference"),
                roadmap_text,
            ),
            False,
            OPEN_VERDICT,
            id="cannot_close_open_question",
        ),
    ],
)
def test_scenario_report_3707_parametrized_honest_outcomes(
    case_name: str,
    mutate_inputs,
    expected_closed: bool,
    expected_verdict: str,
) -> None:
    """SCENARIO-REPORT-3707: synthetic outcomes cover closed and open."""

    payloads = {
        "exp3672": {
            "honest_verdict": (
                "complete: ensemble_no_selection_value_even_with_headroom_sc_weak_"
                "earned_negative"
            ),
            "ensemble_selection_accuracy": 0.34,
            "sc_accuracy": 0.46,
            "oracle_bestofn_accuracy": 0.61,
            "flip_count": 28,
            "positive_control_valid": True,
        },
        "exp3682": {
            "honest_verdict": "complete: selection_gap_fundamental_no_fix_beats_sc",
            "flagged_adversarial": True,
            "per_candidate_auroc": 0.555,
            "ensemble_selection_accuracy": 0.344262,
            "selection_accuracy_per_question_normalized": 0.344262,
            "self_certainty_selection_accuracy": 0.344262,
            "selection_gap_closed": False,
        },
        "exp3694": {
            "honest_verdict": "complete: blocked_no_multi_candidate_corpus",
            "block_reason": "cached per-candidate energy corpus unavailable",
            "n_examples": 0,
            "selection_gap_closed": False,
        },
    }
    references_text = (
        "Reward Model Selection Crisis (arXiv:2512.23067)\n"
        "Reward Learning from Best-of-N (arXiv:2605.30619)\n"
    )
    roadmap_text = "project_energy_selection_thesis_bounded settled-bounded\n"
    payloads, references_text, roadmap_text = mutate_inputs(
        payloads,
        references_text,
        roadmap_text,
    )

    artifact = exp.build_artifact_from_inputs(
        exp3672=payloads["exp3672"],
        exp3682=payloads["exp3682"],
        exp3694=payloads["exp3694"],
        references_text=references_text,
        roadmap_text=roadmap_text,
        manifest_hash_before="manifest",
        manifest_hash_after="manifest",
        conductor_hash_before="conductor",
        conductor_hash_after="conductor",
        started_s=1.0,
        now_s=2.25,
        adversarial_verify_clean=True,
        adversarial_verify_report={"flags": []},
    )

    exp.validate_artifact(artifact)
    assert case_name in {"closure_recorded_retirement_recommended", "cannot_close_open_question"}
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["question_closed"] is expected_closed
    assert type(artifact["question_closed"]) is bool
    assert type(artifact["manifest_unmodified_assert"]) is bool
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert "exp3672" in artifact["earned_negative_source"]
    assert "exp3682" in artifact["failed_diagnosis_attempts"]
    assert "exp3694" in artifact["failed_diagnosis_attempts"]
    assert "arXiv:2512.23067" in artifact["bounded_thesis_basis"]

    if expected_closed:
        assert "OPERATOR RECOMMENDATION" in artifact["operator_retirement_recommendation"]
        assert "ops/exclusion_manifest.yaml" in artifact["operator_retirement_recommendation"]
        assert "human-seeded thesis" in artifact["operator_retirement_recommendation"]
    else:
        assert "not recommended" in artifact["operator_retirement_recommendation"]


def test_req_report_3707_write_artifact_runs_adversarial_verify_and_preserves_files(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3707: writing persists the clean closure artifact."""

    _seed_repo(tmp_path)
    before_manifest = (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(
        encoding="utf-8"
    )
    before_conductor = (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    )

    output = exp.write_artifact(
        tmp_path,
        output_path="results/exp3707.json",
        started_s=0.0,
        now_s=1.0,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == CLOSED_VERDICT
    assert artifact["question_closed"] is True
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"]["flag_count"] == 0
    assert artifact["manifest_unmodified_assert"] is True
    assert artifact["scripts_research_conductor_modified"] is False
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(
        encoding="utf-8"
    ) == before_manifest
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor
    encoded = json.dumps(artifact)
    assert "model_specs" not in encoded
    assert "target_model" not in encoded
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded


def test_req_report_3707_validation_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3707: validation rejects schema drift and helpers are total."""

    artifact = exp.build_artifact_from_inputs(
        exp3672={},
        exp3682={},
        exp3694={},
        references_text="",
        roadmap_text="",
        manifest_hash_before="a",
        manifest_hash_after="b",
        conductor_hash_before="c",
        conductor_hash_after="d",
        started_s=3.0,
        now_s=2.0,
        adversarial_verify_clean=False,
        adversarial_verify_report={"flags": [{"severity": "critical"}]},
    )
    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == OPEN_VERDICT
    assert artifact["manifest_unmodified_assert"] is False
    assert artifact["duration_s"] == pytest.approx(0.0001)
    assert exp.adversarial_report_is_clean({"flags": [{"severity": "warn"}]}) is True
    assert exp.adversarial_report_is_clean({"flags": [{"severity": "critical"}]}) is False
    assert exp.adversarial_report_is_clean({"flags": "not-list"}) is False
    assert exp.compact_adversarial_report({"flags": [{"severity": "warn"}, "bad"]}) == {
        "flag_count": 1,
        "flags": [{"severity": "warn"}],
    }
    assert exp._read_json_object(tmp_path / "missing.json") == {}
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert exp._read_json_object(invalid) == {}
    text_file = tmp_path / "text.txt"
    text_file.write_text("abc", encoding="utf-8")
    assert len(exp._sha256_path(text_file)) == 64
    assert len(exp._sha256_path(tmp_path / "absent.txt")) == 64
    assert exp._coerce_float("not-a-number") is None

    missing = dict(artifact)
    missing.pop("question_closed")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="complete: unexpected")
    with pytest.raises(ValueError, match="terminal verdict"):
        exp.validate_artifact(bad_verdict)

    bad_bool = dict(artifact, question_closed={"value": False})
    with pytest.raises(ValueError, match="question_closed"):
        exp.validate_artifact(bad_bool)

    bad_manifest_bool = dict(artifact, manifest_unmodified_assert="true")
    with pytest.raises(ValueError, match="manifest_unmodified_assert"):
        exp.validate_artifact(bad_manifest_bool)

    bad_substrate = dict(artifact, inference_substrate="live_llm_inference")
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_duration = dict(artifact, duration_s=0.0)
    with pytest.raises(ValueError, match="duration_s"):
        exp.validate_artifact(bad_duration)

    bad_adv = dict(artifact, adversarial_verify_clean="clean")
    with pytest.raises(ValueError, match="adversarial_verify_clean"):
        exp.validate_artifact(bad_adv)

    saved = exp.importlib.util.spec_from_file_location
    try:
        exp.importlib.util.spec_from_file_location = lambda *args, **kwargs: None
        with pytest.raises(ImportError, match="adversarial_verify"):
            exp.run_adversarial_verify_report(text_file)
    finally:
        exp.importlib.util.spec_from_file_location = saved
