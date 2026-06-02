"""Tests for Exp 3690 v337 archive and v338 activation.

Spec: REQ-REPORT-3690, SCENARIO-REPORT-3690.
"""

from __future__ import annotations

import json
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from carnot.reporting import archive_v337_activate_v338_3690 as exp3690


TERMINAL_VERDICT = (
    "complete: "
    "archived_v337_dependency_aware_g1_candidate_clean_package_and_selection_"
    "to_redo_code_native_needed_v338_active_paper_ready_true_frozen_headline_"
    "unchanged"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _metric(point: float) -> dict[str, object]:
    return {"point": point, "ci95": [round(point - 0.01, 6), round(point + 0.01, 6)]}


def _seed_repo(root: Path, *, active_milestone: str = "2026.06.338") -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{active_milestone}"\n'
        "tasks:\n"
        "  - id: exp3690-archive-v337-activate-v338\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "# completed\n\n"
        "milestones:\n"
        "- id: 2026.06.336\n"
        "  finding: previous archive\n"
        "- id: 2026.06.337\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3680-dependency-aware-dual-condition-integrity-g1-rigor\n"
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
        root / "results" / "experiment_3680_dependency_aware_dual_condition_integrity.json",
        {
            "honest_verdict": (
                "complete: dependency_aware_g1_rigor_confirmed_headline_"
                "candidate_exceeds_frozen_0_9131"
            ),
            "dependency_aware_g1_rigor_confirmed": True,
            "adversarial_verify_clean": True,
            "leak_free": True,
            "n_seeds": 5,
            "n_examples": 1000,
            "frozen_headline_auroc": 0.9131,
            "production_auroc_dependency_aware": 0.925328,
            "production_auroc_carnot_current": 0.913134,
            "production_auroc_dependency_aware_vs_frozen_headline_delta": 0.012228,
            "production_auroc_ci": {"point": 0.924869, "ci95": [0.91699, 0.932891]},
            "dependency_vs_carnot_delta_ci": {"point": 0.011839, "ci95": [0.008643, 0.01515]},
            "learning_contribution_dependency_aware": 0.022149,
            "random_seed": 42,
            "duration_s": 21.229053,
        },
    )
    _write_json(
        root / "results" / "experiment_3681_g2_reproducer_prep_operator_refreeze_package.json",
        {
            "honest_verdict": "complete: refreeze_package_ready_for_operator_frozen_headline_unchanged",
            "candidate_reproduction_asserts_in_ci": True,
            "existing_0_9131_reproduction_still_green": True,
            "frozen_headline_unchanged_assert": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {
                    "kind": "DURATION_TOO_SHORT",
                    "severity": "critical",
                    "detail": "duration false-positive from vestigial compute marker",
                }
            ],
            "random_seed": 42,
            "duration_s": 25.454659,
        },
    )
    _write_json(
        root / "results" / "experiment_3682_discrimination_vs_selection_gap.json",
        {
            "honest_verdict": (
                "complete: selection_gap_fundamental_no_fix_beats_sc_"
                "discrimination_decoupled"
            ),
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "TAUTOLOGY", "severity": "critical", "detail": "same metrics"}
            ],
            "per_candidate_auroc": 0.5555075187969924,
            "selection_gap_closed": False,
            "ensemble_selection_accuracy": 0.3442622950819672,
            "selection_accuracy_per_question_normalized": 0.3442622950819672,
            "self_certainty_selection_accuracy": 0.3442622950819672,
            "selection_accuracy_ranking_calibrated": 0.15,
            "positive_control_valid": True,
            "flip_count": 28,
            "random_seed": 514504639,
            "duration_s": 0.801205,
        },
    )
    _write_json(
        root / "results" / "experiment_3683_detector_code_operating_point.json",
        {
            "honest_verdict": "complete: code_remains_math_only_detector_scoped_honestly",
            "code_operating_point_recovered": False,
            "code_auroc_baseline": {"ensemble": _metric(0.5)},
            "code_auroc_dependency_aware": _metric(0.463333),
            "code_auroc_recalibrated": _metric(0.506173),
            "code_calibration_brier_ece_after": {"brier": 0.249368, "ece": 0.034219},
            "e2e_test_passed": True,
            "n_examples_code": 60,
            "random_seed": 3683,
            "duration_s": 3.63492,
        },
    )
    _write_json(
        root / "results" / "experiment_3684_product_value_vs_self_certainty.json",
        {
            "honest_verdict": "complete: ensemble_adds_value_over_self_certainty_product_value_robust",
            "ensemble_adds_value_over_self_certainty": True,
            "material_win_per_domain": {"math": True, "code": False},
            "ensemble_minus_self_certainty_delta_ci_per_domain": {
                "math": {"point": 0.474931, "delta_ci_excludes_zero": True},
                "code": {"point": 0.0, "delta_ci_excludes_zero": False},
            },
            "random_seed": 3684,
            "duration_s": 7.554371,
        },
    )
    _write_json(
        root / "results" / "experiment_3685_fr11_continuous_self_learning_v11.json",
        {
            "honest_verdict": (
                "complete: fr11_v11_drift_aware_online_dependency_aware_"
                "recovers_no_collapse_quality_maintained"
            ),
            "drift_detected_deploy_arm": True,
            "collapse_detected_deploy_arm": False,
            "collapse_detected_control": True,
            "quality_maintained": True,
            "post_drift_auroc_gain_over_v10": 0.088142,
            "post_drift_auroc_gain_over_static_carnot": 0.014668,
            "random_seed": 3685,
            "duration_s": 0.753909,
        },
    )
    _write_json(
        root / "results" / "experiment_3689_capstone_and_g_gate_v337.json",
        {
            "honest_verdict": (
                "complete: capstone_v337_dependency_aware_g1_rigor_confirmed_"
                "package_blocked_selection_not_measured_detector_code_math_only_"
                "earned_paper_ready_true_frozen_headline_unchanged"
            ),
            "paper_ready": True,
            "p01_status": "honest-negative",
            "g1": True,
            "g2": True,
            "g3": True,
            "g4": True,
            "unmet_gates": [],
            "frozen_headline_unchanged": True,
            "frozen_fover_headline_auroc": 0.9131,
            "dependency_aware_g1_candidate_status": "g1_rigor_confirmed_package_blocked",
            "refreeze_package_status": "not_prepared_candidate_unconfirmed",
            "selection_gap_verdict": "not_measured",
            "detector_code_operating_point": "math_only_earned",
            "product_value_vs_self_certainty": "robust_beats_self_certainty",
            "fr11_v11_result": "drift_aware_online_dependency_aware_recovers_no_collapse_quality_maintained",
            "facts_generalization_retired": True,
            "trained_judge_ood_retired": True,
            "random_seed": 3689,
            "duration_s": 0.321147,
        },
    )


def test_req_report_3690_run_archives_v337_and_writes_clean_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3690: archive preserves candidate plus redo items."""

    _seed_repo(tmp_path)
    before_conductor = (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    )
    before_ops = {
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
    }

    out_path = exp3690.run(tmp_path)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    exp3690.validate_artifact(artifact)
    assert artifact["honest_verdict"] == TERMINAL_VERDICT
    assert set(exp3690.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3690.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3690.INFERENCE_SUBSTRATE
    assert artifact["v337_outcome_recorded_as"] == exp3690.V337_OUTCOME
    assert artifact["headline_refreeze_candidate_status"] == exp3690.HEADLINE_REFREEZE_STATUS
    assert artifact["refreeze_package_must_redo_recorded"] is True
    assert artifact["selection_diagnosis_still_open_recorded"] is True
    assert artifact["code_detector_blind_under_reweighting_recorded"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["p01_status_preserved"] == "honest-negative"
    assert artifact["n_tasks_archived"] == 12
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"]["max_severity"] < 2
    assert artifact["random_seed"] == 3690
    assert artifact["duration_s"] >= 0.0001
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["v338_active_confirmed"] is True
    assert artifact["production_auroc_dependency_aware"] == 0.925328
    assert artifact["production_auroc_carnot_current"] == 0.913134
    assert artifact["frozen_headline_auroc_preserved"] == 0.9131
    assert artifact["dependency_aware_vs_frozen_delta"] == 0.012228
    assert artifact["refreeze_package_flag_kinds"] == ["DURATION_TOO_SHORT"]
    assert artifact["selection_diagnosis_flag_kinds"] == ["TAUTOLOGY"]
    assert artifact["selection_per_candidate_auroc_recorded"] == 0.555508
    assert artifact["code_auroc_under_dependency_aware"] == 0.463333
    assert artifact["code_auroc_recalibrated"] == 0.506173
    assert artifact["product_value_robust_over_self_certainty_recorded"] is True
    assert artifact["fr11_v11_no_collapse_recovery_recorded"] is True
    assert artifact["fr11_v11_post_drift_gain_over_v10"] == 0.088142
    assert artifact["scripts_research_conductor_modified"] is False
    encoded = json.dumps(artifact)
    assert "model_specs" not in encoded
    assert "target_model" not in encoded

    complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    assert complete.count("- id: 2026.06.337") == 1
    assert "See conductor log" not in complete
    assert "DEPENDENCY-AWARE G1 CANDIDATE" in complete
    assert "re-freeze package must be re-emitted clean in .338" in complete
    assert "selection diagnosis remains OPEN" in complete
    assert "code-native signal is needed" in complete
    assert "paper_ready stayed TRUE" in complete
    assert "frozen FoVer 0.9131 stayed frozen" in complete
    assert complete.count("deliverable: results/experiment_") == 12
    assert "result: FLAGGED redo clean in .338" in complete
    assert "result: DEGENERATE redo properly in .338" in complete
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


def test_req_report_3690_research_complete_rewrite_is_idempotent(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3690: missing or existing v337 archive entries stay stable."""

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").write_text(
        "# completed\n\nmilestones:\n- id: 2026.06.336\n  finding: previous\n",
        encoding="utf-8",
    )

    first_path = exp3690.run(tmp_path)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    first_artifact = json.loads(first_path.read_text(encoding="utf-8"))

    second_path = exp3690.run(tmp_path)
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second_artifact = json.loads(second_path.read_text(encoding="utf-8"))

    assert first_complete == second_complete
    assert first_complete.count("- id: 2026.06.337") == 1
    assert first_artifact == second_artifact


def test_req_report_3690_validate_artifact_rejects_dishonest_fields(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3690: schema validation blocks silent regression."""

    _seed_repo(tmp_path)
    artifact = exp3690.run(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    exp3690.validate_artifact(payload)

    missing = dict(payload)
    missing.pop("selection_diagnosis_still_open_recorded")
    with pytest.raises(ValueError, match="missing required"):
        exp3690.validate_artifact(missing)

    bad_principles_type = dict(payload, field_principles=[])
    with pytest.raises(ValueError, match="field_principles"):
        exp3690.validate_artifact(bad_principles_type)

    missing_principle = dict(payload)
    missing_principle["field_principles"] = dict(payload["field_principles"])
    missing_principle["field_principles"].pop("adversarial_verify_clean")
    with pytest.raises(ValueError, match="missing field principles"):
        exp3690.validate_artifact(missing_principle)

    bad_verdict = dict(payload, honest_verdict="complete: wrong")
    with pytest.raises(ValueError, match="terminal verdict"):
        exp3690.validate_artifact(bad_verdict)

    bad_substrate = dict(payload, inference_substrate="live_inference")
    with pytest.raises(ValueError, match="inference_substrate"):
        exp3690.validate_artifact(bad_substrate)

    bad_active = dict(payload, v338_active_confirmed=False)
    with pytest.raises(ValueError, match="v338"):
        exp3690.validate_artifact(bad_active)

    bad_outcome = dict(payload, v337_outcome_recorded_as="all_clean")
    with pytest.raises(ValueError, match="v337 outcome"):
        exp3690.validate_artifact(bad_outcome)

    bad_headline = dict(payload, headline_refreeze_candidate_status="headline_swapped")
    with pytest.raises(ValueError, match="headline re-freeze"):
        exp3690.validate_artifact(bad_headline)

    bad_refreeze = dict(payload, refreeze_package_must_redo_recorded=False)
    with pytest.raises(ValueError, match="re-freeze package"):
        exp3690.validate_artifact(bad_refreeze)

    bad_selection = dict(payload, selection_diagnosis_still_open_recorded=False)
    with pytest.raises(ValueError, match="selection diagnosis"):
        exp3690.validate_artifact(bad_selection)

    bad_code = dict(payload, code_detector_blind_under_reweighting_recorded=False)
    with pytest.raises(ValueError, match="code detector"):
        exp3690.validate_artifact(bad_code)

    bad_paper = dict(payload, paper_ready_preserved=False)
    with pytest.raises(ValueError, match="paper_ready"):
        exp3690.validate_artifact(bad_paper)

    bad_p01 = dict(payload, p01_status_preserved="positive")
    with pytest.raises(ValueError, match="P0.1"):
        exp3690.validate_artifact(bad_p01)

    bad_tasks = dict(payload, n_tasks_archived=11)
    with pytest.raises(ValueError, match="12"):
        exp3690.validate_artifact(bad_tasks)

    bad_verify = dict(payload, adversarial_verify_clean=False)
    with pytest.raises(ValueError, match="adversarial_verify_clean"):
        exp3690.validate_artifact(bad_verify)

    bad_duration = dict(payload, duration_s=0.0)
    with pytest.raises(ValueError, match="duration_s"):
        exp3690.validate_artifact(bad_duration)

    bad_checksum_shape = dict(payload, reproducibility_checksum="short")
    with pytest.raises(ValueError, match="sha256"):
        exp3690.validate_artifact(bad_checksum_shape)

    bad_checksum_value = dict(payload, reproducibility_checksum="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        exp3690.validate_artifact(bad_checksum_value)

    bad_model_specs = dict(payload, model_specs={})
    with pytest.raises(ValueError, match="model_specs"):
        exp3690.validate_artifact(bad_model_specs)

    bad_target_model = dict(payload, target_model=None)
    with pytest.raises(ValueError, match="target_model"):
        exp3690.validate_artifact(bad_target_model)


def test_req_report_3690_requires_v338_to_be_active(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3690: the archive cannot claim a wrong active milestone."""

    _seed_repo(tmp_path, active_milestone="2026.06.337")

    with pytest.raises(ValueError, match="v338"):
        exp3690.run(tmp_path)


def test_req_report_3690_defensive_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-3690: malformed inputs do not produce a terminal archive."""

    _seed_repo(tmp_path)
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="v338"):
        exp3690.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    (
        tmp_path
        / "results"
        / "experiment_3680_dependency_aware_dual_condition_integrity.json"
    ).write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        exp3690.build_artifact(tmp_path)

    assert exp3690._point({"point": 0.1234567}) == 0.123457
    assert exp3690._point("not-a-number") is None
    assert exp3690._critical_flag_kinds({}) == []
    assert exp3690._is_verify_clean({}) is True

    monkeypatch.setattr(
        exp3690.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="could not load adversarial verifier"):
        exp3690._run_adversarial_verify(tmp_path / "missing.json")

    class _Loader:
        def create_module(self, _spec: ModuleSpec) -> None:
            return None

        def exec_module(self, module: object) -> None:
            module.verify_artifact = lambda _path: []  # type: ignore[attr-defined]

    monkeypatch.setattr(
        exp3690.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: ModuleSpec("fake_verify", _Loader()),
    )
    with pytest.raises(RuntimeError, match="non-object report"):
        exp3690._run_adversarial_verify(tmp_path / "missing.json")
