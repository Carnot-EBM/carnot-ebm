"""Tests for Exp 3652 v334 archive and v335 activation.

Spec: REQ-REPORT-3652, SCENARIO-REPORT-3652.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v334_activate_v335_3652 as exp3652


TERMINAL_VERDICT = (
    "complete: "
    "archived_v334_cross_domain_science_ran_math_only_was_artifact_"
    "verifier_value_math_plus_code_facts_gap_open_v335_active_paper_ready_true"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _metric(point: float) -> dict[str, object]:
    return {
        "point": point,
        "ci95": [round(point - 0.01, 6), round(point + 0.01, 6)],
    }


def _seed_repo(root: Path, *, active_milestone: str = "2026.06.335") -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{active_milestone}"\n'
        'milestone_title: "Make the FACTS row real"\n'
        "tasks:\n"
        "  - id: exp3652-archive-v334-activate-v335\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "# completed\n\n"
        "milestones:\n"
        "- id: 2026.06.333\n"
        "  finding: total infrastructure wipeout\n"
        "- id: 2026.06.334\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3638-archive-v333-activate-v334\n"
        "    result: OK (conductor)\n",
        encoding="utf-8",
    )
    (root / "ops" / "north-star.md").write_text(
        "# Carnot North Star\n\nG1-G4 are the publication gate.\n",
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
        root / "results" / "experiment_3642_corrected_cross_domain_remeasurement_v4.json",
        {
            "honest_verdict": "complete: verifier_value_generalizes_to_code_not_facts_partial_scope",
            "generalization_table": {
                "math": {
                    "ran_or_blocked": "ran",
                    "ensemble_auroc": _metric(0.9131),
                    "confidence_auroc": _metric(0.8947),
                    "delta": _metric(0.0185),
                    "domain_verdict": "generalizes",
                    "n_examples": 1000,
                },
                "code": {
                    "ran_or_blocked": "ran",
                    "ensemble_auroc": _metric(0.924831),
                    "confidence_auroc": _metric(0.362753),
                    "delta": _metric(0.562078),
                    "domain_verdict": "generalizes",
                    "n_examples": 320,
                },
                "facts": {
                    "ran_or_blocked": "ran",
                    "ensemble_auroc": _metric(0.64952),
                    "confidence_auroc": _metric(0.744576),
                    "delta": _metric(-0.095056),
                    "domain_verdict": "domain_bound",
                    "n_examples": 500,
                    "nli_substrate": (
                        "disclosed_text_statistical_proxy_token_support_no_gold_or_label_input"
                    ),
                },
            },
            "math_ensemble_auroc": 0.9131,
            "code_generalizes": True,
            "facts_generalize": False,
            "grounding_leak_free": True,
            "nli_substrate": "disclosed_text_statistical_proxy_token_support_no_gold_or_label_input",
        },
    )
    _write_json(
        root / "results" / "experiment_3643_additivity_second_pair_of_eyes_v4.json",
        {
            "honest_verdict": (
                "complete: ensemble_additive_to_confidence_second_pair_of_eyes_real_fusion_wins"
            ),
            "second_pair_of_eyes_real": True,
            "fused_detector_auroc": 0.822394,
            "confidence_alone_auroc": 0.536376,
        },
    )
    _write_json(
        root / "results" / "experiment_3644_weaver_peer_comparison_v3.json",
        {
            "honest_verdict": (
                "complete: weaver_compared_correlation_matters_carnot_differentiates_on_correlation_awareness"
            ),
            "ensemble_auroc_correlation_aware": 0.635312,
            "ensemble_auroc_carnot": 0.919446,
            "auroc_delta_correlation_aware_vs_weaver": -0.236268,
        },
    )
    _write_json(
        root / "results" / "experiment_3645_headroom_hybrid_verifier_vs_sc_v3.json",
        {
            "honest_verdict": "complete: verifier_beats_sc_on_headroom_corpus_hybrid_wins_under_budget",
            "verifier_beats_sc_where_headroom_exists": True,
            "sc_accuracy": 0.7,
            "verifier_reranked_accuracy": 0.7333333333333333,
        },
    )
    _write_json(
        root / "results" / "experiment_3646_trained_ebm_judge_ood_counterpoint_v2.json",
        {
            "honest_verdict": "complete: trained_ebm_judge_also_math_only_transfer_not_a_training_artifact",
            "trained_judge_transfers_ood": False,
            "ood_judge_auroc": 0.673554,
            "confidence_only_baseline_auroc": 0.882162,
        },
    )
    _write_json(
        root / "results" / "experiment_3651_capstone_and_g_gate_v334.json",
        {
            "honest_verdict": (
                "complete: capstone_v334_329_null_was_artifact_"
                "verifier_value_code_only_facts_code_rows_ran_paper_ready_true"
            ),
            "paper_ready": True,
            "p01_status": "honest-negative",
            "second_pair_of_eyes_real": True,
            "v329_null_was_artifact_or_confirmed": "artifact",
            "verifier_value_scope": "code_only",
        },
    )


def test_req_report_3652_run_archives_v334_and_writes_required_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3652: v334 archive records the corrected cross-domain state."""

    _seed_repo(tmp_path)
    before_conductor = (tmp_path / "scripts" / "research_conductor.py").read_text(encoding="utf-8")
    before_ops = {
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
    }

    out_path = exp3652.run(tmp_path)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    exp3652.validate_artifact(artifact)
    assert artifact["honest_verdict"] == TERMINAL_VERDICT
    assert set(exp3652.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3652.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3652.INFERENCE_SUBSTRATE
    assert artifact["v334_outcome_recorded_as"] == (
        "math_only_was_contamination_artifact_verifier_value_math_plus_code_"
        "not_facts_proxy_second_pair_of_eyes_real"
    )
    assert artifact["cross_domain_scope_recorded"] == (
        "math_plus_code_generalizes_facts_not_tested_with_real_model_based_nli"
    )
    assert artifact["facts_gap_recorded"] == (
        "facts_row_used_text_statistical_proxy_not_real_model_based_nli_grounding_verifier"
    )
    assert artifact["paper_ready_preserved"] is True
    assert artifact["p01_status_preserved"] == "honest-negative"
    assert artifact["n_tasks_archived"] == 14
    assert artifact["random_seed"] == 3652
    assert artifact["duration_s"] >= 0.0001
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["v335_active_confirmed"] is True
    assert artifact["facts_proxy_auroc"] == 0.64952
    assert artifact["facts_confidence_auroc"] == 0.744576
    assert artifact["second_pair_of_eyes_fused_auroc"] == 0.822394
    assert artifact["second_pair_of_eyes_confidence_auroc"] == 0.536376
    assert artifact["correlation_aware_weighting_hurt_delta"] == -0.236268
    assert artifact["verifier_beats_sc_where_headroom_exists"] is True
    assert artifact["trained_judge_solves_ood"] is False

    complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    assert complete.count("- id: 2026.06.334") == 1
    assert "See conductor log" not in complete
    assert "CONTAMINATION ARTIFACT" in complete
    assert "text-statistical PROXY" in complete
    assert "second-pair-of-eyes is REAL" in complete
    assert "OK (codex artifact landed)" in complete
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


def test_req_report_3652_research_complete_rewrite_is_idempotent(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3652: missing or existing v334 archive entries stay stable."""

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").write_text(
        "# completed\n\nmilestones:\n- id: 2026.06.333\n  finding: previous\n",
        encoding="utf-8",
    )

    first_path = exp3652.run(tmp_path)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    first_artifact = json.loads(first_path.read_text(encoding="utf-8"))

    second_path = exp3652.run(tmp_path)
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second_artifact = json.loads(second_path.read_text(encoding="utf-8"))

    assert first_complete == second_complete
    assert first_complete.count("- id: 2026.06.334") == 1
    assert first_artifact == second_artifact


def test_req_report_3652_validate_artifact_rejects_dishonest_fields(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3652: schema validation blocks silent regression."""

    _seed_repo(tmp_path)
    artifact = exp3652.build_artifact(tmp_path)
    exp3652.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("facts_gap_recorded")
    with pytest.raises(ValueError, match="missing required"):
        exp3652.validate_artifact(missing)

    bad_principles_type = dict(artifact, field_principles=[])
    with pytest.raises(ValueError, match="field_principles"):
        exp3652.validate_artifact(bad_principles_type)

    missing_principle = dict(artifact)
    missing_principle["field_principles"] = dict(artifact["field_principles"])
    missing_principle["field_principles"].pop("duration_s")
    with pytest.raises(ValueError, match="missing field principles"):
        exp3652.validate_artifact(missing_principle)

    bad_verdict = dict(artifact, honest_verdict="complete: wrong")
    with pytest.raises(ValueError, match="terminal verdict"):
        exp3652.validate_artifact(bad_verdict)

    bad_substrate = dict(artifact, inference_substrate="live_model")
    with pytest.raises(ValueError, match="inference_substrate"):
        exp3652.validate_artifact(bad_substrate)

    bad_paper = dict(artifact, paper_ready_preserved=False)
    with pytest.raises(ValueError, match="paper_ready"):
        exp3652.validate_artifact(bad_paper)

    bad_p01 = dict(artifact, p01_status_preserved="positive")
    with pytest.raises(ValueError, match="P0.1"):
        exp3652.validate_artifact(bad_p01)

    bad_tasks = dict(artifact, n_tasks_archived=13)
    with pytest.raises(ValueError, match="14"):
        exp3652.validate_artifact(bad_tasks)

    bad_gap = dict(artifact, facts_gap_recorded="facts_gap_closed")
    with pytest.raises(ValueError, match="facts gap"):
        exp3652.validate_artifact(bad_gap)

    bad_duration = dict(artifact, duration_s=0.0)
    with pytest.raises(ValueError, match="duration_s"):
        exp3652.validate_artifact(bad_duration)

    bad_checksum_shape = dict(artifact, reproducibility_checksum="short")
    with pytest.raises(ValueError, match="sha256"):
        exp3652.validate_artifact(bad_checksum_shape)

    bad_checksum_value = dict(artifact, reproducibility_checksum="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        exp3652.validate_artifact(bad_checksum_value)


def test_req_report_3652_requires_v335_to_be_active(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3652: the terminal archive cannot claim a wrong active milestone."""

    _seed_repo(tmp_path, active_milestone="2026.06.334")

    with pytest.raises(ValueError, match="v335"):
        exp3652.run(tmp_path)


def test_req_report_3652_defensive_read_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3652: malformed inputs do not produce a terminal archive."""

    _seed_repo(tmp_path)
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="v335"):
        exp3652.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    (tmp_path / "results" / "experiment_3643_additivity_second_pair_of_eyes_v4.json").write_text(
        "[]",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected JSON object"):
        exp3652.build_artifact(tmp_path)

    assert exp3652._point("not-a-number") is None
