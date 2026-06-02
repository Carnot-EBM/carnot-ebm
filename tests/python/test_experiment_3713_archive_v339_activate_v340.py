"""Tests for Exp 3713 v339 archive and v340 activation.

Spec: REQ-REPORT-3713, SCENARIO-REPORT-3713.
"""

from __future__ import annotations

import json
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from carnot.reporting import archive_v339_activate_v340_3713 as exp3713


TERMINAL_VERDICT = (
    "complete: "
    "archived_v339_convergence_refreeze_closed_negative_code_leak_narrowed_"
    "selection_closed_kv260_terminal_v340_active_paper_ready_true_"
    "frozen_headline_unchanged"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _seed_repo(root: Path, *, active_milestone: str = "2026.06.340") -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{active_milestone}"\n'
        "tasks:\n"
        "  - id: exp3713-archive-v339-activate-v340\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "# completed\n\n"
        "milestones:\n"
        "- id: 2026.06.338\n"
        "  finding: previous archive\n"
        "- id: 2026.06.339\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3704-refreeze-disambiguate-dependency-vs-external-vs-fusion\n"
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
        root
        / "results"
        / "experiment_3704_refreeze_disambiguate_dependency_vs_external_vs_fusion.json",
        {
            "honest_verdict": (
                "complete: refreeze_disambiguated_no_candidate_beats_frozen_"
                "headline_stays_0_9131"
            ),
            "acceptance_gate": {"passed": False},
            "adversarial_verify_clean": False,
            "flagged_adversarial": True,
            "dependency_aware_auroc": 0.924869,
            "external_comparator_auroc": 0.928737,
            "fusion_auroc": 0.928462,
            "strongest_candidate": "external",
            "strongest_candidate_auroc": 0.928737,
            "carnot_current_auroc": 0.91303,
            "frozen_headline_auroc": 0.9131,
            "winner_vs_runnerup_delta_ci": {
                "point": 0.000275,
                "ci95": [-0.000044, 0.000625],
                "winner": "external",
                "comparison": "fusion",
            },
            "corrigendum_pending": [
                {
                    "kind": "TAUTOLOGY",
                    "severity": "critical",
                    "detail": (
                        "external_comparator_auroc=0.928737 and "
                        "strongest_candidate_auroc=0.928737"
                    ),
                }
            ],
            "frozen_headline_unchanged_assert": True,
            "publication_gate_paper_ready_after": True,
            "random_seed": 3704,
            "duration_s": 67.31,
        },
    )
    _write_json(
        root / "results" / "experiment_3705_code_native_leak_audit_heldout.json",
        {
            "honest_verdict": (
                "complete: code_native_one_point_zero_was_a_leak_code_claim_"
                "narrowed_earned"
            ),
            "acceptance_gate": {"passed": True},
            "adversarial_verify_clean": True,
            "code_signal_survives_heldout": False,
            "leak_detected": True,
            "in_corpus_code_auroc": 1.0,
            "heldout_code_auroc": 0.993243,
            "heldout_code_auroc_ci": [0.982808, 1.0],
            "random_seed": 3705,
            "duration_s": 0.49,
        },
    )
    _write_json(
        root / "results" / "experiment_3706_reconcile_shipped_detector_heldout.json",
        {
            "honest_verdict": (
                "complete: shipped_detector_narrowed_to_math_only_abstain_on_"
                "code_e2e_green"
            ),
            "acceptance_gate": {"passed": True},
            "adversarial_verify_clean": True,
            "reconciliation_action": "narrowed_to_math_only_abstain",
            "overclaim_removed": True,
            "code_surface_abstains": True,
            "math_operating_point_unchanged": True,
            "e2e_test_passed": True,
            "shipped_code_operating_point_auroc": None,
            "random_seed": 3706,
            "duration_s": 5.27,
        },
    )
    _write_json(
        root / "results" / "experiment_3707_selection_diagnosis_formal_closure.json",
        {
            "honest_verdict": (
                "complete: selection_diagnosis_formally_closed_retirement_"
                "recommended_to_operator"
            ),
            "adversarial_verify_clean": True,
            "question_closed": True,
            "operator_retirement_recommendation": "retire under operator authority",
            "random_seed": 3707,
            "duration_s": 0.0001,
        },
    )
    _write_json(
        root / "results" / "experiment_3708_fr11_continuous_self_learning_v13.json",
        {
            "honest_verdict": (
                "complete: fr11_v13_multi_session_consolidation_transfers_no_"
                "collapse_quality_maintained"
            ),
            "quality_maintained": True,
            "fresh_session_transfer_auroc_gain": 0.021697,
            "random_seed": 3708,
            "duration_s": 0.73,
        },
    )
    _write_json(
        root / "results" / "experiment_3709_kv260_drive_to_terminal_latency_transcript.json",
        {
            "honest_verdict": (
                "complete: kv260_board_latency_transcript_captured_poc_anchor_"
                "terminal_candidate"
            ),
            "kv260_ssh_reachable": True,
            "kv260_overlay_loaded": True,
            "terminal_condition_met": True,
            "board_latency_median_ms": 0.025465,
            "speedup_claim_avoided_assert": True,
            "random_seed": 3709,
            "duration_s": 6.84,
        },
    )
    _write_json(
        root / "results" / "experiment_3712_capstone_and_g_gate_v339.json",
        {
            "honest_verdict": (
                "complete: capstone_v339_refreeze_winner_none_code_native_one_"
                "point_zero_was_a_leak_selection_closed_kv260_latency_"
                "transcript_captured_terminal_candidate_paper_ready_true_"
                "frozen_headline_unchanged"
            ),
            "adversarial_verify_clean": True,
            "refreeze_candidate": {
                "candidate": "none",
                "flagged_or_live_critical": True,
                "refreeze_package_reemitted_for_winner": False,
            },
            "strongest_refreeze_candidate": "none",
            "refreeze_package_status": "not_measured",
            "code_native_heldout_verdict": "one_point_zero_was_a_leak",
            "shipped_detector_reconciliation": "narrowed_to_math_only_abstain",
            "selection_diagnosis_closed": True,
            "kv260_terminal_status": "latency_transcript_captured_terminal_candidate",
            "fr11_v13_result": "multi_session_consolidation_transferred_no_collapse",
            "paper_ready": True,
            "p01_status": "honest-negative",
            "g1": True,
            "g2": True,
            "g3": True,
            "g4": True,
            "frozen_fover_headline_auroc": 0.9131,
            "frozen_headline_unchanged": True,
            "random_seed": 3712,
            "duration_s": 0.3,
        },
    )


def test_req_report_3713_run_archives_v339_and_writes_clean_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3713: archive records convergence and activates .340."""

    _seed_repo(tmp_path)
    before_conductor = (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    )
    before_docs = {
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
    }

    out_path = exp3713.run(tmp_path)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    exp3713.validate_artifact(artifact)
    assert artifact["honest_verdict"] == TERMINAL_VERDICT
    assert set(exp3713.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3713.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3713.INFERENCE_SUBSTRATE
    assert artifact["v339_outcome_recorded_as"] == exp3713.V339_OUTCOME
    assert artifact["refreeze_closed_negative_recorded"] is True
    assert artifact["exp3704_benign_flag_recorded"] is True
    assert artifact["code_leak_recorded"] is True
    assert artifact["selection_diagnosis_closed_recorded"] is True
    assert artifact["kv260_terminal_recorded"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["p01_status_preserved"] == "honest-negative"
    assert artifact["n_tasks_archived"] == 11
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"]["max_severity"] < 2
    assert artifact["random_seed"] == 3713
    assert artifact["duration_s"] >= 0.0001
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["v340_active_confirmed"] is True
    assert artifact["dependency_aware_auroc_recorded"] == 0.924869
    assert artifact["external_comparator_auroc_recorded"] == 0.928737
    assert artifact["fusion_auroc_recorded"] == 0.928462
    assert artifact["frozen_headline_auroc_preserved"] == 0.9131
    assert artifact["code_leak_evidence"]["in_corpus_code_auroc"] == 1.0
    assert artifact["code_leak_evidence"]["heldout_code_auroc"] == 0.993243
    assert artifact["shipped_detector_evidence"]["code_surface_abstains"] is True
    assert artifact["selection_diagnosis_evidence"]["question_closed"] is True
    assert artifact["kv260_terminal_evidence"]["terminal_condition_met"] is True
    assert artifact["fr11_v13_recorded"] is True
    assert artifact["scripts_research_conductor_modified"] is False
    encoded = json.dumps(artifact)
    assert "model_specs" not in encoded
    assert "target_model" not in encoded
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded

    complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    assert complete.count("- id: 2026.06.339") == 1
    assert "See conductor log" not in complete
    assert "CONVERGENCE MILESTONE" in complete
    assert "re-freeze CLOSED-NEGATIVE" in complete
    assert "benign TAUTOLOGY" in complete
    assert "code AUROC 1.0 was a LEAK" in complete
    assert "math-only-with-abstain" in complete
    assert "selection diagnosis FORMALLY CLOSED" in complete
    assert "KV260 captured a terminal latency transcript" in complete
    assert "paper_ready stayed TRUE" in complete
    assert "frozen FoVer 0.9131 stayed frozen" in complete
    assert complete.count("deliverable: results/experiment_") == 11
    assert "result: CLOSED-NEGATIVE; headline stays frozen" in complete
    assert "result: LEAK; detector narrowed in exp3706" in complete
    assert "result: TERMINAL latency transcript candidate" in complete
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before_docs[
        "status"
    ]
    assert (tmp_path / "ops" / "changelog.md").read_text(
        encoding="utf-8"
    ) == before_docs["changelog"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(
        encoding="utf-8"
    ) == before_docs["trace"]


def test_req_report_3713_research_complete_rewrite_is_idempotent(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3713: missing or existing v339 archive entries stay stable."""

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").write_text(
        "# completed\n\nmilestones:\n- id: 2026.06.338\n  finding: previous\n",
        encoding="utf-8",
    )

    first_path = exp3713.run(tmp_path)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    first_artifact = json.loads(first_path.read_text(encoding="utf-8"))

    second_path = exp3713.run(tmp_path)
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second_artifact = json.loads(second_path.read_text(encoding="utf-8"))

    assert first_complete == second_complete
    assert first_complete.count("- id: 2026.06.339") == 1
    assert first_artifact == second_artifact


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("code_leak_recorded"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (
            lambda p: p["field_principles"].pop("adversarial_verify_clean"),
            "missing field principles",
        ),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(inference_substrate="live_inference"), "inference_substrate"),
        (lambda p: p.update(v340_active_confirmed=False), "v340"),
        (lambda p: p.update(v339_outcome_recorded_as="all_positive"), "v339 outcome"),
        (lambda p: p.update(refreeze_closed_negative_recorded=False), "closed-negative"),
        (lambda p: p.update(exp3704_benign_flag_recorded=False), "benign"),
        (lambda p: p.update(code_leak_recorded=False), "code leak"),
        (lambda p: p.update(selection_diagnosis_closed_recorded=False), "selection"),
        (lambda p: p.update(kv260_terminal_recorded=False), "KV260"),
        (lambda p: p.update(paper_ready_preserved=False), "paper_ready"),
        (lambda p: p.update(p01_status_preserved="positive"), "P0.1"),
        (lambda p: p.update(n_tasks_archived=10), "11"),
        (lambda p: p.update(adversarial_verify_clean=False), "adversarial_verify_clean"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(target_model=None), "target_model"),
    ],
)
def test_req_report_3713_validate_artifact_rejects_dishonest_fields(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    """REQ-REPORT-3713: schema validation blocks silent regression."""

    _seed_repo(tmp_path)
    artifact_path = exp3713.run(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    exp3713.validate_artifact(payload)

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        exp3713.validate_artifact(broken)


def test_req_report_3713_requires_v340_to_be_active(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3713: the archive cannot claim a wrong active milestone."""

    _seed_repo(tmp_path, active_milestone="2026.06.339")

    with pytest.raises(ValueError, match="v340"):
        exp3713.run(tmp_path)


def test_req_report_3713_defensive_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-3713: malformed inputs do not produce a terminal archive."""

    _seed_repo(tmp_path)
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="v340"):
        exp3713.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    (
        tmp_path
        / "results"
        / "experiment_3704_refreeze_disambiguate_dependency_vs_external_vs_fusion.json"
    ).write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        exp3713.build_artifact(tmp_path)

    assert exp3713._point({"point": 0.1234567}) == 0.123457
    assert exp3713._point("not-a-number") is None
    assert exp3713._ci95("not-a-ci") is None
    assert exp3713._ci95({"ci95": [0.1]}) is None
    assert exp3713._ci95({"ci95": ["bad", 0.2]}) is None
    assert exp3713._has_flag({}, "TAUTOLOGY") is False
    assert exp3713._is_verify_clean({}) is True

    monkeypatch.setattr(
        exp3713.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="could not load adversarial verifier"):
        exp3713._run_adversarial_verify(tmp_path / "missing.json")

    class _Loader:
        def create_module(self, _spec: ModuleSpec) -> None:
            return None

        def exec_module(self, module: object) -> None:
            module.verify_artifact = lambda _path: []  # type: ignore[attr-defined]

    monkeypatch.setattr(
        exp3713.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: ModuleSpec("fake_verify", _Loader()),
    )
    with pytest.raises(RuntimeError, match="non-object report"):
        exp3713._run_adversarial_verify(tmp_path / "missing.json")


def test_scenario_report_3713_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3713: conductor entrypoint delegates to the module."""

    script = Path("scripts/experiment_3713_archive_v339_activate_v340.py")
    assert script.exists()
    assert "archive_v339_activate_v340_3713" in script.read_text(encoding="utf-8")
