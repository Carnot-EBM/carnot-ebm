"""Tests for Exp 3678 v336 archive and v337 activation.

Spec: REQ-REPORT-3678, SCENARIO-REPORT-3678.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v336_activate_v337_3678 as exp3678


TERMINAL_VERDICT = (
    "complete: "
    "archived_v336_dependency_aware_refreeze_candidate_facts_retired_"
    "selection_negative_v337_active_paper_ready_true"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _metric(point: float) -> dict[str, object]:
    return {"point": point, "ci95": [round(point - 0.01, 6), round(point + 0.01, 6)]}


def _seed_repo(root: Path, *, active_milestone: str = "2026.06.337") -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{active_milestone}"\n'
        "tasks:\n"
        "  - id: exp3678-archive-v336-activate-v337\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "# completed\n\n"
        "milestones:\n"
        "- id: 2026.06.335\n"
        "  finding: previous archive\n"
        "- id: 2026.06.336\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3666-backend-state-diagnostic-v2\n"
        "    result: OK (conductor)\n",
        encoding="utf-8",
    )
    (root / "ops" / "north-star.md").write_text(
        "# Carnot North Star\n\nFrozen FoVer headline AUROC: 0.9131.\n",
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
        root / "results" / "experiment_3668_dependency_aware_weighting_heldout.json",
        {
            "honest_verdict": (
                "complete: dependency_aware_weighting_generalizes_heldout_"
                "headline_re_freeze_candidate_for_v337"
            ),
            "heldout_auroc_dependency_aware": 0.933224,
            "heldout_auroc_carnot": 0.919964,
            "heldout_delong_p": 0.000072,
            "heldout_delta_ci": {"point": 0.01326, "ci95": [0.006932, 0.019857]},
            "dependency_aware_generalizes_heldout": True,
            "n_splits": 5,
            "random_seed": 3668,
            "duration_s": 0.94725,
        },
    )
    _write_json(
        root / "results" / "experiment_3670_facts_row_real_benchmark.json",
        {
            "honest_verdict": (
                "complete: facts_domain_bound_on_real_benchmark_335_negative_"
                "genuinely_earned"
            ),
            "grounding_auroc_real_corpus": _metric(0.642844),
            "confidence_baseline_auroc": _metric(0.708003),
            "grounding_minus_confidence_delta": {"point": -0.065158},
            "facts_generalize_or_adds_value_real": False,
            "grounding_leak_free": True,
            "positive_control_valid": True,
            "mcnemar_p_facts": 0.0,
            "n_examples": 17617,
            "corpus_path_used": "data/real_factual_corpus_ragtruth.jsonl",
            "random_seed": 3670,
            "duration_s": 1334.829451,
        },
    )
    _write_json(
        root / "results" / "experiment_3671_ship_second_pair_of_eyes_detector.json",
        {
            "honest_verdict": (
                "complete: second_pair_of_eyes_detector_shipped_math_only_"
                "code_weak_documented_e2e_green"
            ),
            "detector_shipped": True,
            "e2e_test_passed": True,
            "fused_detector_auroc_per_domain": {"math": 0.979656, "code": 0.5},
            "detector_module_path": "python/carnot/pipeline/second_pair_detector.py",
            "random_seed": 3671,
            "duration_s": 2.512606,
        },
    )
    _write_json(
        root / "results" / "experiment_3672_ensemble_selection_where_sc_weak.json",
        {
            "honest_verdict": (
                "complete: ensemble_no_selection_value_even_with_headroom_sc_weak_"
                "earned_negative"
            ),
            "ensemble_adds_selection_value_sc_weak": False,
            "ensemble_selection_accuracy": 0.3442622950819672,
            "sc_accuracy": 0.45901639344262296,
            "oracle_bestofn_accuracy": 0.6065573770491803,
            "ensemble_vs_sc_delta_ci": {
                "delta": -0.11475409836065575,
                "mcnemar_exact_p": 0.015625,
            },
            "positive_control_valid": True,
            "flip_count": 28,
            "n_examples": 61,
            "random_seed": 184757772,
            "duration_s": 0.7092676162719727,
        },
    )
    _write_json(
        root / "results" / "experiment_3673_fr11_continuous_self_learning_v10.json",
        {
            "honest_verdict": (
                "complete: fr11_v10_online_dependency_aware_weighting_holds_"
                "no_collapse_quality_maintained"
            ),
            "collapse_detected_deploy_arm": False,
            "collapse_detected_control": True,
            "quality_maintained": True,
            "online_dependency_aware_auroc_gain": 0.0018,
            "random_seed": 3673,
            "duration_s": 0.676938,
        },
    )
    _write_json(
        root / "results" / "experiment_3677_capstone_and_g_gate_v336.json",
        {
            "honest_verdict": (
                "complete: capstone_v336_dependency_aware_clean_and_heldout_"
                "validated_facts_real_domain_bound_real_earned_detector_shipped_"
                "true_paper_ready_true"
            ),
            "paper_ready": True,
            "p01_status": "honest-negative",
            "g1": True,
            "g2": True,
            "g3": True,
            "g4": True,
            "unmet_gates": [],
            "frozen_fover_headline_auroc": 0.9131,
            "dependency_aware_headline_candidate_status": "clean_and_heldout_validated",
            "facts_real_benchmark_verdict": "domain_bound_real_earned",
            "sc_weak_selection_direction_result": "no_value_with_headroom",
            "second_pair_of_eyes_shipped": True,
            "fr11_v10_result": "held_no_collapse_quality_maintained",
            "trained_judge_ood_retired": True,
            "random_seed": 3677,
            "duration_s": 0.362692,
        },
    )


def test_req_report_3678_run_archives_v336_and_writes_required_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3678: archive records the .336 headline candidate state."""

    _seed_repo(tmp_path)
    before_conductor = (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    )
    before_ops = {
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
    }

    out_path = exp3678.run(tmp_path)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    exp3678.validate_artifact(artifact)
    assert artifact["honest_verdict"] == TERMINAL_VERDICT
    assert set(exp3678.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3678.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3678.INFERENCE_SUBSTRATE
    assert artifact["v336_outcome_recorded_as"] == exp3678.V336_OUTCOME
    assert artifact["headline_refreeze_candidate_recorded"] == exp3678.HEADLINE_REFREEZE
    assert artifact["facts_generalization_retired_recorded"] is True
    assert artifact["selection_earned_negative_recorded"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["p01_status_preserved"] == "honest-negative"
    assert artifact["n_tasks_archived"] == 13
    assert artifact["random_seed"] == 3678
    assert artifact["duration_s"] >= 0.0001
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["v337_active_confirmed"] is True
    assert artifact["frozen_headline_auroc_preserved"] == 0.9131
    assert artifact["heldout_auroc_dependency_aware"] == 0.933224
    assert artifact["heldout_auroc_carnot"] == 0.919964
    assert artifact["facts_grounding_auroc_real_corpus"] == 0.642844
    assert artifact["facts_confidence_baseline_auroc"] == 0.708003
    assert artifact["ensemble_selection_accuracy"] == 0.344262
    assert artifact["sc_accuracy"] == 0.459016
    assert artifact["ensemble_vs_sc_delta"] == -0.114754
    assert artifact["detector_math_auroc"] == 0.979656
    assert artifact["detector_code_auroc"] == 0.5
    assert artifact["fr11_v10_no_collapse_recorded"] is True

    complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    assert complete.count("- id: 2026.06.336") == 1
    assert "See conductor log" not in complete
    assert "DEPENDENCY-AWARE RE-FREEZE CANDIDATE" in complete
    assert "facts-generalization is RETIRED" in complete
    assert "selection is an earned-negative" in complete
    assert "detector shipped math-strong/code-blind" in complete
    assert "frozen 0.9131 headline stayed frozen" in complete
    assert complete.count("OK (codex artifact landed)") == 13
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before_ops[
        "status"
    ]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before_ops[
        "changelog"
    ]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before_ops[
        "trace"
    ]


def test_req_report_3678_research_complete_rewrite_is_idempotent(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3678: missing or existing v336 archive entries stay stable."""

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").write_text(
        "# completed\n\nmilestones:\n- id: 2026.06.335\n  finding: previous\n",
        encoding="utf-8",
    )

    first_path = exp3678.run(tmp_path)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    first_artifact = json.loads(first_path.read_text(encoding="utf-8"))

    second_path = exp3678.run(tmp_path)
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second_artifact = json.loads(second_path.read_text(encoding="utf-8"))

    assert first_complete == second_complete
    assert first_complete.count("- id: 2026.06.336") == 1
    assert first_artifact == second_artifact


def test_req_report_3678_validate_artifact_rejects_dishonest_fields(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3678: schema validation blocks silent regression."""

    _seed_repo(tmp_path)
    artifact = exp3678.build_artifact(tmp_path)
    exp3678.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("selection_earned_negative_recorded")
    with pytest.raises(ValueError, match="missing required"):
        exp3678.validate_artifact(missing)

    bad_principles_type = dict(artifact, field_principles=[])
    with pytest.raises(ValueError, match="field_principles"):
        exp3678.validate_artifact(bad_principles_type)

    missing_principle = dict(artifact)
    missing_principle["field_principles"] = dict(artifact["field_principles"])
    missing_principle["field_principles"].pop("duration_s")
    with pytest.raises(ValueError, match="missing field principles"):
        exp3678.validate_artifact(missing_principle)

    bad_verdict = dict(artifact, honest_verdict="complete: wrong")
    with pytest.raises(ValueError, match="terminal verdict"):
        exp3678.validate_artifact(bad_verdict)

    bad_substrate = dict(artifact, inference_substrate="live_model")
    with pytest.raises(ValueError, match="inference_substrate"):
        exp3678.validate_artifact(bad_substrate)

    bad_active = dict(artifact, v337_active_confirmed=False)
    with pytest.raises(ValueError, match="v337"):
        exp3678.validate_artifact(bad_active)

    bad_outcome = dict(artifact, v336_outcome_recorded_as="facts_positive")
    with pytest.raises(ValueError, match="v336 outcome"):
        exp3678.validate_artifact(bad_outcome)

    bad_headline = dict(artifact, headline_refreeze_candidate_recorded="swap_headline")
    with pytest.raises(ValueError, match="headline re-freeze"):
        exp3678.validate_artifact(bad_headline)

    bad_facts = dict(artifact, facts_generalization_retired_recorded=False)
    with pytest.raises(ValueError, match="facts-generalization"):
        exp3678.validate_artifact(bad_facts)

    bad_selection = dict(artifact, selection_earned_negative_recorded=False)
    with pytest.raises(ValueError, match="selection earned-negative"):
        exp3678.validate_artifact(bad_selection)

    bad_paper = dict(artifact, paper_ready_preserved=False)
    with pytest.raises(ValueError, match="paper_ready"):
        exp3678.validate_artifact(bad_paper)

    bad_p01 = dict(artifact, p01_status_preserved="positive")
    with pytest.raises(ValueError, match="P0.1"):
        exp3678.validate_artifact(bad_p01)

    bad_tasks = dict(artifact, n_tasks_archived=12)
    with pytest.raises(ValueError, match="13"):
        exp3678.validate_artifact(bad_tasks)

    bad_duration = dict(artifact, duration_s=0.0)
    with pytest.raises(ValueError, match="duration_s"):
        exp3678.validate_artifact(bad_duration)

    bad_checksum_shape = dict(artifact, reproducibility_checksum="short")
    with pytest.raises(ValueError, match="sha256"):
        exp3678.validate_artifact(bad_checksum_shape)

    bad_checksum_value = dict(artifact, reproducibility_checksum="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        exp3678.validate_artifact(bad_checksum_value)


def test_req_report_3678_requires_v337_to_be_active(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3678: the archive cannot claim a wrong active milestone."""

    _seed_repo(tmp_path, active_milestone="2026.06.336")

    with pytest.raises(ValueError, match="v337"):
        exp3678.run(tmp_path)


def test_req_report_3678_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3678: malformed inputs do not produce a terminal archive."""

    _seed_repo(tmp_path)
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="v337"):
        exp3678.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    (
        tmp_path
        / "results"
        / "experiment_3672_ensemble_selection_where_sc_weak.json"
    ).write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        exp3678.build_artifact(tmp_path)

    assert exp3678._point({"point": 0.1234567}) == 0.123457
    assert exp3678._point("not-a-number") is None
