"""Tests for Exp 3702 v338 archive and v339 activation.

Spec: REQ-REPORT-3702, SCENARIO-REPORT-3702.
"""

from __future__ import annotations

import json
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from carnot.reporting import archive_v338_activate_v339_3702 as exp3702


TERMINAL_VERDICT = (
    "complete: "
    "archived_v338_refreeze_candidate_ambiguous_code_native_provisional_"
    "selection_closing_kv260_reachable_v339_active_paper_ready_true_"
    "frozen_headline_unchanged"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _seed_repo(root: Path, *, active_milestone: str = "2026.06.339") -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{active_milestone}"\n'
        "tasks:\n"
        "  - id: exp3702-archive-v338-activate-v339\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "# completed\n\n"
        "milestones:\n"
        "- id: 2026.06.337\n"
        "  finding: previous archive\n"
        "- id: 2026.06.338\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3692-refreeze-package-clean-reemit\n"
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
        root / "results" / "experiment_3692_refreeze_package_clean_reemit.json",
        {
            "honest_verdict": (
                "complete: refreeze_package_reemitted_clean_for_operator_"
                "frozen_headline_unchanged"
            ),
            "acceptance_gate": {"passed": True},
            "adversarial_verify_clean": True,
            "candidate_reproduction_asserts_in_ci": True,
            "existing_0_9131_reproduction_still_green": True,
            "north_star_unmodified_assert": True,
            "frozen_headline_unchanged_assert": True,
            "random_seed": 3692,
            "duration_s": 24.34,
        },
    )
    _write_json(
        root / "results" / "experiment_3693_external_comparator_dependency_vs_deentangled.json",
        {
            "honest_verdict": (
                "complete: dependency_aware_candidate_ties_or_loses_"
                "external_baseline_refreeze_narrowed"
            ),
            "adversarial_verify_clean": True,
            "candidate_beats_external_comparator": False,
            "dependency_aware_auroc": 0.924869,
            "external_comparator_auroc": 0.928737,
            "dependency_vs_external_delta_ci": {
                "point": -0.003868,
                "ci95": [-0.006204, -0.001577],
            },
            "random_seed": 3693,
            "duration_s": 31.65,
        },
    )
    _write_json(
        root / "results" / "experiment_3694_selection_gap_proper_rediagnosis.json",
        {
            "honest_verdict": "complete: blocked_no_multi_candidate_corpus",
            "adversarial_verify_clean": True,
            "acceptance_gate": {"passed": False},
            "block_reason": "cached per-candidate energy corpus unavailable",
            "n_examples": 0,
            "selection_gap_closed": False,
            "random_seed": 3694,
            "duration_s": 0.13,
        },
    )
    _write_json(
        root / "results" / "experiment_3695_code_native_verifier.json",
        {
            "honest_verdict": "complete: code_native_signal_recovered_beats_chance_floor",
            "adversarial_verify_clean": True,
            "code_signal_recovered": True,
            "code_native_auroc": 1.0,
            "code_native_auroc_ci": [1.0, 1.0],
            "n_examples_code": 60,
            "random_seed": 3695,
            "duration_s": 1.8,
        },
    )
    _write_json(
        root / "results" / "experiment_3696_reship_detector_math_plus_code.json",
        {
            "honest_verdict": (
                "complete: detector_reshipped_math_plus_code_operating_point_e2e_green"
            ),
            "adversarial_verify_clean": True,
            "module_code_path_updated": True,
            "math_operating_point_unchanged": True,
            "e2e_test_passed": True,
            "code_operating_point_auroc": 1.0,
            "random_seed": 3696,
            "duration_s": 4.98,
        },
    )
    _write_json(
        root / "results" / "experiment_3697_fr11_continuous_self_learning_v12.json",
        {
            "honest_verdict": (
                "complete: fr11_v12_drift_reset_and_cross_session_persistence_"
                "no_collapse_quality_maintained"
            ),
            "adversarial_verify": "clean",
            "drift_detected_deploy_arm": True,
            "reset_triggered_on_transient_drift": True,
            "structure_persisted_and_restored": True,
            "collapse_detected_deploy_arm": False,
            "quality_maintained": True,
            "random_seed": 3697,
            "duration_s": 0.68,
        },
    )
    _write_json(
        root / "results" / "experiment_3698_kv260_continuity_v25.json",
        {
            "honest_verdict": "complete: kv260_continuity_confirmed_reachable",
            "kv260_ssh_reachable": True,
            "consecutive_unreachable_milestones": 0,
            "continuity_history": {
                "current_unreachable_streak_if_blocked": 8,
                "previous_unreachable_milestones": [
                    ".331",
                    ".332",
                    ".333",
                    ".334",
                    ".335",
                    ".336",
                    ".337",
                ],
            },
            "random_seed": 3698,
            "duration_s": 5.35,
        },
    )
    _write_json(
        root / "results" / "experiment_3701_capstone_and_g_gate_v338.json",
        {
            "honest_verdict": (
                "complete: capstone_v338_refreeze_reemitted_clean_for_operator_"
                "external_ties_or_loses_selection_not_measured_detector_code_"
                "code_native_recovered_reshipped_paper_ready_true_"
                "frozen_headline_unchanged"
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
            "refreeze_package_status": "reemitted_clean_for_operator",
            "candidate_beats_external_comparator": "ties_or_loses",
            "selection_gap_verdict": "not_measured",
            "code_detector_status": "code_native_recovered_reshipped",
            "fr11_v12_result": (
                "drift_reset_and_cross_session_persistence_no_collapse_quality_"
                "maintained"
            ),
            "random_seed": 3701,
            "duration_s": 0.36,
        },
    )


def test_req_report_3702_run_archives_v338_and_writes_clean_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3702: archive records ambiguity and provisional code."""

    _seed_repo(tmp_path)
    before_conductor = (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    )
    before_ops = {
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
    }

    out_path = exp3702.run(tmp_path)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    exp3702.validate_artifact(artifact)
    assert artifact["honest_verdict"] == TERMINAL_VERDICT
    assert set(exp3702.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3702.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3702.INFERENCE_SUBSTRATE
    assert artifact["v338_outcome_recorded_as"] == exp3702.V338_OUTCOME
    assert artifact["refreeze_candidate_ambiguous_recorded"] is True
    assert artifact["code_native_provisional_recorded"] is True
    assert artifact["selection_diagnosis_closing_recorded"] is True
    assert artifact["kv260_reachable_again_recorded"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["p01_status_preserved"] == "honest-negative"
    assert artifact["n_tasks_archived"] == 12
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"]["max_severity"] < 2
    assert artifact["random_seed"] == 3702
    assert artifact["duration_s"] >= 0.0001
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["v339_active_confirmed"] is True
    assert artifact["dependency_aware_candidate_auroc_recorded"] == 0.924869
    assert artifact["external_baseline_auroc_recorded"] == 0.928737
    assert artifact["frozen_headline_auroc_preserved"] == 0.9131
    assert artifact["code_native_provisional_evidence"]["code_native_auroc"] == 1.0
    assert artifact["code_native_provisional_evidence"]["shipped_detector_wired"] is True
    assert artifact["selection_diagnosis_evidence"]["blocked_second_time"] is True
    assert artifact["kv260_reachable_evidence"]["previous_unreachable_milestones"] == 7
    assert artifact["scripts_research_conductor_modified"] is False
    encoded = json.dumps(artifact)
    assert "model_specs" not in encoded
    assert "target_model" not in encoded

    complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    assert complete.count("- id: 2026.06.338") == 1
    assert "See conductor log" not in complete
    assert "REFREEZE CANDIDATE AMBIGUOUS" in complete
    assert "external baseline 0.9287 beat dependency-aware 0.9249" in complete
    assert "code-native AUROC 1.0 is PROVISIONAL" in complete
    assert "selection diagnosis blocked a second time" in complete
    assert "KV260 became SSH-reachable again" in complete
    assert "paper_ready stayed TRUE" in complete
    assert "frozen FoVer 0.9131 stayed frozen" in complete
    assert complete.count("deliverable: results/experiment_") == 12
    assert "result: AMBIGUOUS; disambiguate in .339" in complete
    assert "result: PROVISIONAL; leak-audit + held-out replicate in .339" in complete
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


def test_req_report_3702_research_complete_rewrite_is_idempotent(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3702: missing or existing v338 archive entries stay stable."""

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").write_text(
        "# completed\n\nmilestones:\n- id: 2026.06.337\n  finding: previous\n",
        encoding="utf-8",
    )

    first_path = exp3702.run(tmp_path)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    first_artifact = json.loads(first_path.read_text(encoding="utf-8"))

    second_path = exp3702.run(tmp_path)
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second_artifact = json.loads(second_path.read_text(encoding="utf-8"))

    assert first_complete == second_complete
    assert first_complete.count("- id: 2026.06.338") == 1
    assert first_artifact == second_artifact


def test_req_report_3702_validate_artifact_rejects_dishonest_fields(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3702: schema validation blocks silent regression."""

    _seed_repo(tmp_path)
    artifact = exp3702.run(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    exp3702.validate_artifact(payload)

    invalid_payloads = [
        (lambda p: p.pop("selection_diagnosis_closing_recorded"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (
            lambda p: p["field_principles"].pop("adversarial_verify_clean"),
            "missing field principles",
        ),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(inference_substrate="live_inference"), "inference_substrate"),
        (lambda p: p.update(v339_active_confirmed=False), "v339"),
        (lambda p: p.update(v338_outcome_recorded_as="all_clean"), "v338 outcome"),
        (lambda p: p.update(refreeze_candidate_ambiguous_recorded=False), "ambiguous"),
        (lambda p: p.update(code_native_provisional_recorded=False), "code-native"),
        (lambda p: p.update(selection_diagnosis_closing_recorded=False), "selection"),
        (lambda p: p.update(kv260_reachable_again_recorded=False), "KV260"),
        (lambda p: p.update(paper_ready_preserved=False), "paper_ready"),
        (lambda p: p.update(p01_status_preserved="positive"), "P0.1"),
        (lambda p: p.update(n_tasks_archived=11), "12"),
        (lambda p: p.update(adversarial_verify_clean=False), "adversarial_verify_clean"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(target_model=None), "target_model"),
    ]
    for mutate, message in invalid_payloads:
        broken = json.loads(json.dumps(payload))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            exp3702.validate_artifact(broken)


def test_req_report_3702_requires_v339_to_be_active(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3702: the archive cannot claim a wrong active milestone."""

    _seed_repo(tmp_path, active_milestone="2026.06.338")

    with pytest.raises(ValueError, match="v339"):
        exp3702.run(tmp_path)


def test_req_report_3702_defensive_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-3702: malformed inputs do not produce a terminal archive."""

    _seed_repo(tmp_path)
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="v339"):
        exp3702.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    (
        tmp_path
        / "results"
        / "experiment_3693_external_comparator_dependency_vs_deentangled.json"
    ).write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        exp3702.build_artifact(tmp_path)

    assert exp3702._point({"point": 0.1234567}) == 0.123457
    assert exp3702._point("not-a-number") is None
    assert exp3702._is_verify_clean({}) is True

    monkeypatch.setattr(
        exp3702.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="could not load adversarial verifier"):
        exp3702._run_adversarial_verify(tmp_path / "missing.json")

    class _Loader:
        def create_module(self, _spec: ModuleSpec) -> None:
            return None

        def exec_module(self, module: object) -> None:
            module.verify_artifact = lambda _path: []  # type: ignore[attr-defined]

    monkeypatch.setattr(
        exp3702.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: ModuleSpec("fake_verify", _Loader()),
    )
    with pytest.raises(RuntimeError, match="non-object report"):
        exp3702._run_adversarial_verify(tmp_path / "missing.json")


def test_scenario_report_3702_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3702: conductor entrypoint delegates to the module."""

    script = Path("scripts/experiment_3702_archive_v338_activate_v339.py")
    assert script.exists()
    assert "archive_v338_activate_v339_3702" in script.read_text(encoding="utf-8")
