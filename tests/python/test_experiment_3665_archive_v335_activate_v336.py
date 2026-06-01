"""Tests for Exp 3665 v335 archive and v336 activation.

Spec: REQ-REPORT-3665, SCENARIO-REPORT-3665.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v335_activate_v336_3665 as exp3665


TERMINAL_VERDICT = (
    "complete: "
    "archived_v335_facts_domain_bound_on_synthetic_dependency_aware_lead_open_"
    "v336_active_paper_ready_true"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _metric(point: float) -> dict[str, object]:
    return {
        "point": point,
        "ci95": [round(point - 0.01, 6), round(point + 0.01, 6)],
        "n": 500,
    }


def _seed_repo(root: Path, *, active_milestone: str = "2026.06.336") -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{active_milestone}"\n'
        'milestone_title: "Advance the headline and stress facts on RAGTruth"\n'
        "tasks:\n"
        "  - id: exp3665-archive-v335-activate-v336\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "# completed\n\n"
        "milestones:\n"
        "- id: 2026.06.334\n"
        "  finding: previous archive\n"
        "- id: 2026.06.335\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3652-archive-v334-activate-v335\n"
        "    result: OK (conductor)\n",
        encoding="utf-8",
    )
    (root / "ops" / "north-star.md").write_text(
        "# Carnot North Star\n\npaper_ready := G1 and G2 and G3 and G4.\n",
        encoding="utf-8",
    )
    (root / "ops" / "status.md").write_text("status before\n", encoding="utf-8")
    (root / "ops" / "changelog.md").write_text("changelog before\n", encoding="utf-8")
    (root / "_bmad" / "traceability.md").write_text("trace before\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text(
        "# conductor unchanged\n",
        encoding="utf-8",
    )

    _write_json(
        root / "results" / "experiment_3655_facts_row_remeasurement_real_nli_v5.json",
        {
            "honest_verdict": "complete: facts_domain_bound_even_with_real_nli_334_negative_confirmed_earned",
            "inference_substrate": (
                "verifier_ensemble_against_cached_candidates "
                "(principle: scores the cached v3 corpus; no LLM load)."
            ),
            "corpus_path_used": "data/realistic_factual_corpus_v3.jsonl",
            "nli_substrate": "model_based_transformers_checkpoint: cross-encoder/nli-deberta-v3-small on cuda",
            "grounding_auroc_real_nli": _metric(0.743656),
            "confidence_baseline_auroc": _metric(0.744576),
            "grounding_minus_confidence_delta": {"point": -0.00092, "ci95": [-0.066269, 0.06274]},
            "facts_generalize_real_nli": False,
            "facts_conditional_catch_rate": {
                "point": 0.38191,
                "grounding_error_catch_rate": 0.36,
                "confidence_error_catch_rate": 0.204,
                "fixed_confidence_fpr": 0.1,
                "mcnemar": {
                    "p_value": 0.00031,
                    "grounding_only_error_catches": 76,
                    "confidence_only_error_catches": 37,
                },
            },
            "mcnemar_p_facts": 0.00031,
            "positive_control_valid": True,
            "sample_size_rigor_met": True,
            "random_seed": 3655,
            "duration_s": 14.758074,
        },
    )
    _write_json(
        root / "results" / "experiment_3656_correlation_aware_weighting_paradox_diagnosis.json",
        {
            "honest_verdict": "complete: paradox_resolved_naive_penalty_misspecified_dependency_aware_recovers",
            "flagged_adversarial": True,
            "correlation_harmless_or_penalty_misspecified": "dependency_aware_recovers",
            "ensemble_auroc_dependency_aware_proper": 0.932562,
            "ensemble_auroc_carnot": 0.919446,
            "dependency_aware_auroc_delta_vs_carnot": 0.013116,
            "ensemble_auroc_correlation_aware": 0.635312,
            "ensemble_auroc_naive_correlation_aware": 0.635312,
            "random_seed": 3656,
            "duration_s": 0.178347,
        },
    )
    _write_json(
        root / "results" / "experiment_3664_capstone_and_g_gate_v335.json",
        {
            "honest_verdict": (
                "complete: capstone_v335_facts_domain_bound_with_real_nli_"
                "verifier_value_math_plus_code_paper_ready_true"
            ),
            "paper_ready": True,
            "p01_status": "honest-negative",
            "verifier_value_scope": "math_plus_code",
            "facts_generalize_real_nli": False,
            "corrected_generalization_table": {
                "math": {"auroc": 0.9131, "generalizes": True, "ran_or_blocked": "ran"},
                "code": {
                    "auroc": 0.532222,
                    "generalizes": True,
                    "ran_or_blocked": "ran",
                    "second_corpus_balanced": True,
                },
                "facts": {
                    "auroc": 0.743656,
                    "confidence_auroc": 0.744576,
                    "generalizes": False,
                    "ran_or_blocked": "ran",
                    "real_nli_status": "domain_bound",
                },
            },
            "code_generalization_replicated": True,
            "second_pair_of_eyes_deployable": True,
            "correlation_paradox_resolution": {
                "reported_resolution": "H2_naive_penalty_misspecified_dependency_aware_recovers",
                "status": "excluded_flagged_adversarial",
                "usable_for_claims": False,
            },
            "trained_judge_real_substrate_result": {
                "status": "does_not_transfer_ood",
                "transfers_ood": False,
                "ood_judge_auroc": 0.572465,
                "confidence_only_baseline_auroc": 0.882162,
            },
            "fr11_continuous_self_learning_result": {
                "honest_verdict": "complete: fr11_v9_online_fusion_weighting_holds_no_collapse_quality_maintained",
                "quality_maintained": True,
            },
            "flagged_upstream_artifacts_excluded": [
                "results/experiment_3656_correlation_aware_weighting_paradox_diagnosis.json"
            ],
            "g1": True,
            "g2": True,
            "g3": True,
            "g4": True,
            "unmet_gates": [],
            "random_seed": 3664,
            "duration_s": 0.0001,
        },
    )


def test_req_report_3665_run_archives_v335_and_writes_required_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3665: v335 archive records the facts/dependency-aware state."""

    _seed_repo(tmp_path)
    before_conductor = (tmp_path / "scripts" / "research_conductor.py").read_text(encoding="utf-8")
    before_ops = {
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
    }

    out_path = exp3665.run(tmp_path)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    exp3665.validate_artifact(artifact)
    assert artifact["honest_verdict"] == TERMINAL_VERDICT
    assert set(exp3665.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3665.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3665.INFERENCE_SUBSTRATE
    assert artifact["v335_outcome_recorded_as"] == (
        "facts_domain_bound_real_nli_on_synthetic_v3_dependency_aware_lead_"
        "flagged_code_replicated_detector_math_wins_trained_judge_retired"
    )
    assert artifact["headline_advancement_lead_recorded"] == (
        "dependency_aware_weighting_beat_carnot_0.932562_vs_0.919446_"
        "but_exp3656_tautology_flag_false_positive_open_v336_lead"
    )
    assert artifact["facts_real_benchmark_gap_recorded"] == (
        "real_external_benchmark_gap_open_ragtruth_not_tried_v335_used_synthetic_v3"
    )
    assert artifact["paper_ready_preserved"] is True
    assert artifact["p01_status_preserved"] == "honest-negative"
    assert artifact["trained_judge_ood_retired_recorded"] is True
    assert artifact["n_tasks_archived"] == 14
    assert artifact["random_seed"] == 3665
    assert artifact["duration_s"] >= 0.0001
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["v336_active_confirmed"] is True
    assert artifact["facts_corpus_recorded_as"] == "synthetic_v3"
    assert artifact["facts_grounding_auroc_real_nli"] == 0.743656
    assert artifact["facts_confidence_auroc"] == 0.744576
    assert artifact["facts_mcnemar_p"] == 0.00031
    assert artifact["facts_conditional_catch_rate"] == 0.38191
    assert artifact["dependency_aware_auroc"] == 0.932562
    assert artifact["carnot_current_auroc"] == 0.919446
    assert artifact["dependency_aware_flagged_adversarial"] is True
    assert artifact["code_generalization_replicated"] is True
    assert artifact["second_pair_detector_math_wins"] is True
    assert artifact["fr11_v9_no_collapse_recorded"] is True

    complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    assert complete.count("- id: 2026.06.335") == 1
    assert "See conductor log" not in complete
    assert "FACTS DOMAIN-BOUND ON SYNTHETIC V3" in complete
    assert "RAGTruth real benchmark gap remains open" in complete
    assert "dependency-aware weighting BEAT Carnot" in complete
    assert "trained-judge-as-cross-domain-fix is RETIRED" in complete
    assert complete.count("OK (codex artifact landed)") == 14
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before_ops["status"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before_ops[
        "changelog"
    ]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before_ops[
        "trace"
    ]


def test_req_report_3665_research_complete_rewrite_is_idempotent(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3665: missing or existing v335 archive entries stay stable."""

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").write_text(
        "# completed\n\nmilestones:\n- id: 2026.06.334\n  finding: previous\n",
        encoding="utf-8",
    )

    first_path = exp3665.run(tmp_path)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    first_artifact = json.loads(first_path.read_text(encoding="utf-8"))

    second_path = exp3665.run(tmp_path)
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second_artifact = json.loads(second_path.read_text(encoding="utf-8"))

    assert first_complete == second_complete
    assert first_complete.count("- id: 2026.06.335") == 1
    assert first_artifact == second_artifact


def test_req_report_3665_validate_artifact_rejects_dishonest_fields(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3665: schema validation blocks silent regression."""

    _seed_repo(tmp_path)
    artifact = exp3665.build_artifact(tmp_path)
    exp3665.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("facts_real_benchmark_gap_recorded")
    with pytest.raises(ValueError, match="missing required"):
        exp3665.validate_artifact(missing)

    bad_principles_type = dict(artifact, field_principles=[])
    with pytest.raises(ValueError, match="field_principles"):
        exp3665.validate_artifact(bad_principles_type)

    missing_principle = dict(artifact)
    missing_principle["field_principles"] = dict(artifact["field_principles"])
    missing_principle["field_principles"].pop("duration_s")
    with pytest.raises(ValueError, match="missing field principles"):
        exp3665.validate_artifact(missing_principle)

    bad_verdict = dict(artifact, honest_verdict="complete: wrong")
    with pytest.raises(ValueError, match="terminal verdict"):
        exp3665.validate_artifact(bad_verdict)

    bad_substrate = dict(artifact, inference_substrate="live_model")
    with pytest.raises(ValueError, match="inference_substrate"):
        exp3665.validate_artifact(bad_substrate)

    bad_active = dict(artifact, v336_active_confirmed=False)
    with pytest.raises(ValueError, match="v336"):
        exp3665.validate_artifact(bad_active)

    bad_outcome = dict(artifact, v335_outcome_recorded_as="facts_positive")
    with pytest.raises(ValueError, match="v335 outcome"):
        exp3665.validate_artifact(bad_outcome)

    bad_headline = dict(artifact, headline_advancement_lead_recorded="no_lead")
    with pytest.raises(ValueError, match="headline advancement lead"):
        exp3665.validate_artifact(bad_headline)

    bad_paper = dict(artifact, paper_ready_preserved=False)
    with pytest.raises(ValueError, match="paper_ready"):
        exp3665.validate_artifact(bad_paper)

    bad_p01 = dict(artifact, p01_status_preserved="positive")
    with pytest.raises(ValueError, match="P0.1"):
        exp3665.validate_artifact(bad_p01)

    bad_tasks = dict(artifact, n_tasks_archived=13)
    with pytest.raises(ValueError, match="14"):
        exp3665.validate_artifact(bad_tasks)

    bad_gap = dict(artifact, facts_real_benchmark_gap_recorded="ragtruth_closed")
    with pytest.raises(ValueError, match="facts real benchmark gap"):
        exp3665.validate_artifact(bad_gap)

    bad_retired = dict(artifact, trained_judge_ood_retired_recorded=False)
    with pytest.raises(ValueError, match="trained judge"):
        exp3665.validate_artifact(bad_retired)

    bad_duration = dict(artifact, duration_s=0.0)
    with pytest.raises(ValueError, match="duration_s"):
        exp3665.validate_artifact(bad_duration)

    bad_checksum_shape = dict(artifact, reproducibility_checksum="short")
    with pytest.raises(ValueError, match="sha256"):
        exp3665.validate_artifact(bad_checksum_shape)

    bad_checksum_value = dict(artifact, reproducibility_checksum="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        exp3665.validate_artifact(bad_checksum_value)


def test_req_report_3665_requires_v336_to_be_active(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3665: the terminal archive cannot claim a wrong active milestone."""

    _seed_repo(tmp_path, active_milestone="2026.06.335")

    with pytest.raises(ValueError, match="v336"):
        exp3665.run(tmp_path)


def test_req_report_3665_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3665: malformed inputs do not produce a terminal archive."""

    _seed_repo(tmp_path)
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="v336"):
        exp3665.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    (tmp_path / "results" / "experiment_3656_correlation_aware_weighting_paradox_diagnosis.json").write_text(
        "[]",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected JSON object"):
        exp3665.build_artifact(tmp_path)

    assert exp3665._point({"point": 0.1234567}) == 0.123457
    assert exp3665._point("not-a-number") is None
