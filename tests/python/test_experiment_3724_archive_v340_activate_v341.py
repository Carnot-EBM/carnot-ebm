"""Tests for Exp 3724 v340 archive and v341 Thesis-A activation.

Spec: REQ-REPORT-3724, SCENARIO-REPORT-3724.
"""

from __future__ import annotations

import json
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from carnot.reporting import archive_v340_activate_v341_3724 as exp3724


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
TERMINAL_VERDICT = (
    "complete: "
    "archived_v340_convergence_hardened_thesis_a_energy_generator_seeded_"
    "v341_active_paper_ready_true_frozen_headline_unchanged"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _seed_repo(root: Path, *, active_milestone: str = "2026.06.341") -> None:
    (root / "docs" / "research-notes").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{active_milestone}"\n'
        'milestone_title: "PHASE-3 THESIS A BRING-UP - energy-as-GENERATOR (EBT)"\n'
        "tasks:\n"
        "  - id: exp3724-archive-v340-activate-v341\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "# completed\n\n"
        "milestones:\n"
        "- id: 2026.06.339\n"
        "  finding: previous archive\n"
        "- id: 2026.06.340\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3715-refreeze-disambiguation-clean-corrigendum\n"
        "    result: OK (conductor)\n",
        encoding="utf-8",
    )
    (root / "docs" / "research-notes" / "phase3-alternative-thesis-menu.md").write_text(
        "## Thesis A - Energy as the GENERATOR\n"
        "**SELECTED 2026-06-02** by operator seed.\n"
        "The comparison is matched-COMPUTE and is not energy-selection reranking.\n",
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
        root / "results" / "experiment_3715_refreeze_disambiguation_clean_corrigendum.json",
        {
            "honest_verdict": (
                "complete: refreeze_disambiguation_corrigendum_clean_no_candidate_"
                "beats_frozen_headline_stays_0_9131"
            ),
            "adversarial_verify_clean": True,
            "no_candidate_beats_frozen": True,
            "frozen_headline_unchanged_assert": True,
            "frozen_headline_auroc": 0.9131,
            "random_seed": 3715,
            "duration_s": 0.098,
        },
    )
    _write_json(
        root / "results" / "experiment_3716_ship_paper_v6_narrowing_lint.json",
        {
            "honest_verdict": (
                "complete: paper_v6_narrowing_lint_shipped_g3_mechanically_"
                "enforced_current_paper_clean"
            ),
            "g3_now_mechanically_enforced": True,
            "current_paper_lint_clean": True,
            "adversarial_verify_clean": True,
            "random_seed": 3716,
            "duration_s": 1.6,
        },
    )
    _write_json(
        root / "results" / "experiment_3717_g4_full_provenance_audit.json",
        {
            "honest_verdict": (
                "complete: g4_fully_traced_every_headline_number_to_clean_"
                "primary_artifact"
            ),
            "all_numbers_trace_to_clean_artifacts": True,
            "g4_provenance_audit_result": "fully_traced",
            "n_numbers_audited": 7,
            "north_star_unmodified_assert": True,
            "adversarial_verify_clean": True,
            "random_seed": 3717,
            "duration_s": 0.002,
        },
    )
    _write_json(
        root / "results" / "experiment_3718_risk_coverage_abstention_characterization.json",
        {
            "honest_verdict": (
                "complete: energy_is_a_better_selective_prediction_signal_than_"
                "entropy_deployable_abstention_gate"
            ),
            "energy_beats_baseline_abstention": True,
            "energy_aurc": 0.000789,
            "baseline_aurc": 0.075498,
            "adversarial_verify_clean": True,
            "adversarial_verify_report": {
                "flags": [{"severity": "warn", "kind": "IMPLAUSIBLE_TIGHT_CI"}],
                "max_severity": 1,
            },
            "random_seed": 3718,
            "duration_s": 9.4,
        },
    )
    _write_json(
        root / "results" / "experiment_3719_headline_replication_fresh_corpus.json",
        {
            "honest_verdict": (
                "complete: headline_discrimination_is_fover_specific_"
                "generalization_narrowed_honest"
            ),
            "fresh_corpus_generalization": "fover_specific",
            "fresh_corpus_auroc": 0.798604,
            "frozen_fover_auroc": 0.9131,
            "adversarial_verify_clean": True,
            "random_seed": 3719,
            "duration_s": 0.82,
        },
    )
    _write_json(
        root / "results" / "experiment_3720_fr11_continuous_self_learning_v14.json",
        {
            "honest_verdict": (
                "complete: fr11_v14_template_falls_back_gracefully_under_shift_"
                "no_collapse"
            ),
            "template_robust_or_graceful_fallback": True,
            "collapse_detected_deploy_arm": False,
            "template_library_bounded": True,
            "random_seed": 3720,
            "duration_s": 0.81,
        },
    )
    _write_json(
        root / "results" / "experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json",
        {
            "honest_verdict": (
                "complete: kv260_terminal_confirmed_mandate_lift_recommended_"
                "polarfire_gatemate_audited"
            ),
            "kv260_terminal_condition_confirmed": True,
            "kv260_terminal_transcript_present": True,
            "kv260_mandate_lift_recommendation": (
                "recommend_operator_lift_per_milestone_kv260_mandate"
            ),
            "speedup_claim_avoided_assert": True,
            "random_seed": 3721,
            "duration_s": 6.65,
        },
    )
    _write_json(
        root / "results" / "experiment_3722_convergence_synthesis_operator_next_thesis.json",
        {
            "honest_verdict": (
                "complete: convergence_synthesized_next_theses_presented_"
                "operator_decision_requested"
            ),
            "all_self_generable_threads_settled": True,
            "operator_decision_request": "Which thesis should drive .341+?",
            "candidate_next_theses": [{"thesis": "human_seeded_energy_as_generator_ebt"}],
            "paper_ready_status": True,
            "adversarial_verify_clean": True,
            "random_seed": 3722,
            "duration_s": 0.0001,
        },
    )
    _write_json(
        root / "results" / "experiment_3723_capstone_and_g_gate_v340.json",
        {
            "honest_verdict": (
                "complete: capstone_v340_convergence_gates_hardened_g3_"
                "mechanical_g4_audited_abstention_energy_better_than_entropy_"
                "fresh_corpus_fover_specific_kv260_terminal_operator_thesis_"
                "requested_paper_ready_true_frozen_headline_unchanged"
            ),
            "adversarial_verify_clean": True,
            "paper_ready": True,
            "g1": True,
            "g2": True,
            "g3": True,
            "g4": True,
            "frozen_headline_unchanged": True,
            "frozen_fover_headline_auroc": 0.9131,
            "p01_status": "honest-negative",
            "selection_diagnosis_closed": True,
            "exp3704_corrigendum_clean": True,
            "g3_mechanically_enforced": True,
            "g4_provenance_audit_result": "fully_traced",
            "energy_abstention_verdict": "energy_better_than_entropy",
            "fresh_corpus_generalization": "fover_specific",
            "fr11_v14_result": "falls_back_gracefully_under_shift_no_collapse",
            "kv260_terminal_confirmed": True,
            "operator_next_thesis_recorded": True,
            "random_seed": 3723,
            "duration_s": 0.61,
        },
    )


def test_req_report_3724_spec_anchor_exists() -> None:
    """REQ-REPORT-3724: OpenSpec declares the archive/activation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-REPORT-3724" in spec
    assert "SCENARIO-REPORT-3724" in spec
    assert exp3724.OUTPUT_REL_PATH.as_posix() in spec


def test_req_report_3724_run_archives_v340_and_writes_clean_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3724: archive records .340 and activates Thesis A."""

    _seed_repo(tmp_path)
    before_conductor = (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    )
    before_north = (tmp_path / "ops" / "north-star.md").read_text(encoding="utf-8")
    before_docs = {
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
    }

    out_path = exp3724.run(tmp_path)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    exp3724.validate_artifact(artifact)
    assert artifact["honest_verdict"] == TERMINAL_VERDICT
    assert set(exp3724.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3724.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3724.INFERENCE_SUBSTRATE
    assert artifact["v340_outcome_recorded"] == exp3724.V340_OUTCOME
    assert artifact["thesis_a_seeded_recorded"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["p01_status_preserved"] == exp3724.P01_STATUS
    assert artifact["n_tasks_archived"] == 11
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"]["max_severity"] < 2
    assert artifact["random_seed"] == 3724
    assert artifact["duration_s"] >= 0.0001
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["v341_active_confirmed"] is True
    assert artifact["frozen_headline_auroc_preserved"] == 0.9131
    assert artifact["g_gates_preserved"] == {"g1": True, "g2": True, "g3": True, "g4": True}
    assert artifact["v340_evidence"]["g3_mechanical"] is True
    assert artifact["v340_evidence"]["g4_fully_traced"] is True
    assert artifact["v340_evidence"]["fresh_corpus_generalization"] == "fover_specific"
    assert artifact["v340_evidence"]["risk_coverage_abstention_gate"] == (
        "energy_better_than_entropy"
    )
    assert artifact["thesis_a_evidence"]["mechanism"] == "energy_as_generator_not_selector"
    assert artifact["thesis_a_evidence"]["human_seed_required"] is True
    assert artifact["scripts_research_conductor_modified"] is False
    encoded = json.dumps(artifact)
    assert "model_specs" not in encoded
    assert "target_model" not in encoded
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded

    complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    assert complete.count("- id: 2026.06.340") == 1
    assert "See conductor log" not in complete
    assert "CONVERGENCE-HARDENING MILESTONE" in complete
    assert "G3 mechanically enforced" in complete
    assert "G4 fully traced" in complete
    assert "risk-coverage abstention" in complete
    assert "fresh-corpus result was FoVer-specific" in complete
    assert "operator seeded Thesis A" in complete
    assert "energy-as-generator, not selector" in complete
    assert "P0.1 stayed honest-negative-bounded" in complete
    assert "paper_ready stayed TRUE" in complete
    assert "frozen FoVer 0.9131 stayed frozen" in complete
    assert complete.count("deliverable: results/experiment_") == 11
    assert "result: G3 mechanically enforced; paper clean" in complete
    assert "result: G4 fully traced to clean primary artifacts" in complete
    assert "result: THESIS-A operator seed recorded for .341" in complete
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor
    assert (tmp_path / "ops" / "north-star.md").read_text(encoding="utf-8") == before_north
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before_docs[
        "status"
    ]
    assert (tmp_path / "ops" / "changelog.md").read_text(
        encoding="utf-8"
    ) == before_docs["changelog"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(
        encoding="utf-8"
    ) == before_docs["trace"]


def test_req_report_3724_research_complete_rewrite_is_idempotent(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3724: missing or existing v340 archive entries stay stable."""

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").write_text(
        "# completed\n\nmilestones:\n- id: 2026.06.339\n  finding: previous\n",
        encoding="utf-8",
    )

    first_path = exp3724.run(tmp_path)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    first_artifact = json.loads(first_path.read_text(encoding="utf-8"))

    second_path = exp3724.run(tmp_path)
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second_artifact = json.loads(second_path.read_text(encoding="utf-8"))

    assert first_complete == second_complete
    assert first_complete.count("- id: 2026.06.340") == 1
    assert first_artifact == second_artifact


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("v340_outcome_recorded"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (
            lambda p: p["field_principles"].pop("thesis_a_seeded_recorded"),
            "missing field principles",
        ),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(inference_substrate="live_inference"), "inference_substrate"),
        (lambda p: p.update(v341_active_confirmed=False), "v341"),
        (lambda p: p.update(v340_outcome_recorded="all_positive"), ".340 outcome"),
        (lambda p: p.update(thesis_a_seeded_recorded=False), "Thesis A"),
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
def test_req_report_3724_validate_artifact_rejects_dishonest_fields(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3724: schema validation blocks silent regression."""

    _seed_repo(tmp_path)
    artifact_path = exp3724.run(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    exp3724.validate_artifact(payload)

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        exp3724.validate_artifact(broken)


def test_req_report_3724_requires_v341_to_be_active(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3724: the archive cannot claim a wrong active milestone."""

    _seed_repo(tmp_path, active_milestone="2026.06.340")

    with pytest.raises(ValueError, match="v341"):
        exp3724.run(tmp_path)


def test_req_report_3724_defensive_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-3724: malformed inputs do not produce a terminal archive."""

    _seed_repo(tmp_path)
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="v341"):
        exp3724.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    (
        tmp_path
        / "results"
        / "experiment_3723_capstone_and_g_gate_v340.json"
    ).write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        exp3724.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    (tmp_path / "docs" / "research-notes" / "phase3-alternative-thesis-menu.md").write_text(
        "Thesis B selected\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Thesis A"):
        exp3724.build_artifact(tmp_path)

    with pytest.raises(ValueError, match="required text input missing"):
        exp3724._read_text_required(tmp_path / "missing.txt")
    assert exp3724._point({"point": 0.1234567}) == 0.123457
    assert exp3724._point("not-a-number") is None
    assert exp3724._max_report_severity(
        {"adversarial_verify_report": {"flags": [{"severity": "warn"}]}}
    ) == 1
    assert exp3724._max_report_severity(
        {"adversarial_verify_report": {"flags": []}}
    ) == -1
    assert exp3724._max_report_severity(
        {"adversarial_verify_report": {"flags": "bad"}}
    ) == -1
    assert exp3724._is_verify_clean({"flags": [{"severity": "warn"}]}) is True
    assert exp3724._is_verify_clean({"flags": [{"severity": "critical"}]}) is False
    assert exp3724._is_verify_clean({"flags": "bad"}) is True
    assert exp3724._compact_verify_report({"flags": [{"severity": "warn"}, "bad"]}) == {
        "flag_count": 1,
        "max_severity": 1,
        "flags": [{"severity": "warn"}],
    }

    monkeypatch.setattr(
        exp3724.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="could not load adversarial verifier"):
        exp3724._run_adversarial_verify(tmp_path / "missing.json")

    class _Loader:
        def create_module(self, _spec: ModuleSpec) -> None:
            return None

        def exec_module(self, module: object) -> None:
            module.verify_artifact = lambda _path: []  # type: ignore[attr-defined]

    monkeypatch.setattr(
        exp3724.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: ModuleSpec("fake_verify", _Loader()),
    )
    with pytest.raises(RuntimeError, match="non-object report"):
        exp3724._run_adversarial_verify(tmp_path / "missing.json")


def test_scenario_report_3724_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3724: conductor entrypoint delegates to the module."""

    script = Path("scripts/experiment_3724_archive_v340_activate_v341.py")
    assert script.exists()
    assert "archive_v340_activate_v341_3724" in script.read_text(encoding="utf-8")
