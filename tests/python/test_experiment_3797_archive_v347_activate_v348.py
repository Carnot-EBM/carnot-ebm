"""Tests for Exp 3797 v347 archive and v348 activation.

Spec refs: REQ-REPORT-3797, SCENARIO-REPORT-3797,
SCENARIO-REPORT-3797-P1-HANDOFF-GUARD.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v347_activate_v348_3797 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _publication_gate_ready() -> dict[str, object]:
    return {
        "paper_ready": True,
        "gates": {
            "G1": {"pass": True, "detail": "headline measured"},
            "G2": {"pass": True, "detail": "independent reproducer"},
            "G3": {"pass": True, "detail": "narrowing clean"},
            "G4": {"pass": True, "detail": "numbers trace"},
        },
        "unmet_gates": [],
    }


def _seed_repo(root: Path, *, active_milestone: str = "2026.06.348") -> None:
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{active_milestone}"\n'
        'milestone_title: "POST-CONVERGENCE -- ADVANCE THE HEADLINE (G4 RESTORATION) + '
        'HARDEN/REPAIR THE BANKED PRODUCT"\n'
        "tasks:\n"
        "  - id: exp3797-archive-v347-activate-v348\n"
        "    title: Archive .347 and activate .348 headline advancement\n"
        "  - id: exp3798-g4-product-headline-restoration\n"
        "    title: ADVANCE the product headline via G4 restoration\n"
        "  - id: exp3800-gaming-resistance-mitigation-v2\n"
        "    title: HARDEN/REPAIR product context_compaction evasion\n"
        "  - id: exp3802-anomaly-escalation-classifier-v2-tuning\n"
        "    title: REPAIR anomaly-escalation classifier false-escalation\n"
        "  - id: exp3803-fr11-v20-tier3-fast-path-gate\n"
        "    title: CONTINUE Tier-3 self-learning fast-path gate\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap - Milestone 2026.06.348\n\n"
        ".347 landed all tasks. exp3787 blocked on no-free-GPU for the second "
        "consecutive milestone and handoff_to_operator=true; .348 does NOT re-queue "
        "P1. EDLM preflight GO stays operator-gated. Product headline is partially "
        "restorable: exp2090 passes G4, exp1999 fails G4. .348 ADVANCEs the headline, "
        "HARDEN/REPAIRs the banked product, tunes the anomaly-escalation classifier, "
        "continues Tier-3 self-learning, re-grinds nothing bounded, and self-seeds no "
        "paradigm.\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.346\n"
        "  finding: previous archive\n"
        "- id: 2026.06.347\n"
        "  title: stale conductor archive\n"
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md\n"
        "  completed: '2026-06-04'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3787-p1-discrete-search-adjudication-v3-retry\n"
        "    title: P1 discrete-search v3 retry\n"
        "    deliverable: results/experiment_3787_p1_discrete_search_adjudication_v3_retry.json\n"
        "    result: OK (conductor)\n",
        encoding="utf-8",
    )
    (root / "ops" / "north-star.md").write_text(
        "# North Star\n\nFrozen FoVer headline AUROC: 0.9131.\n",
        encoding="utf-8",
    )
    (root / "ops" / "changelog.md").write_text(
        "Milestone 2026.06.348 Planning. .347 landed all tasks; exp3787 blocked "
        "on no-free-GPU for the second consecutive milestone and set "
        "handoff_to_operator=true. exp3792 found exp2090 G4 pass and exp1999 G4 "
        "fail. exp3793 EDLM preflight GO. paper_ready=TRUE. .348 does not re-queue P1.\n",
        encoding="utf-8",
    )
    (root / "ops" / "status.md").write_text("status before\n", encoding="utf-8")
    (root / "_bmad" / "traceability.md").write_text("trace before\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text(
        "# conductor unchanged\n",
        encoding="utf-8",
    )
    _write_json(
        root / "results" / "experiment_3796_capstone_v347.json",
        {
            "honest_verdict": (
                "complete: capstone_v347_p1_blocked_no_free_gpu_"
                "energy_as_generator_still_bounded_verifier_product_hardened_"
                "fr11_v19_tier3_anomaly_validated_edlm_preflighted_"
                "paper_ready_true_frozen_headline_unchanged"
            ),
            "paper_ready_preserved": True,
            "publication_gate_state": {
                "paper_ready": True,
                "g1": True,
                "g2": True,
                "g3": True,
                "g4": True,
                "unmet_gates": [],
            },
            "p1_adjudication": "blocked_no_free_gpu",
            "p1_handoff_to_operator": True,
            "p1_mechanism_status": "open_for_operator: blocked_no_free_gpu",
            "p1_positive_control_passed": False,
            "energy_as_generator_still_bounded": True,
            "energy_as_selector_status": "honest-negative-bounded",
            "energy_as_generator_status": "honest-negative-bounded",
            "product_headline_restorable": "not_yet_eligible",
            "edlm_seed_preflighted": {
                "preflighted": True,
                "readiness_verdict": "go",
                "loop_does_not_commit": True,
            },
            "fr11_v19_tier3_self_learning": {
                "validated": True,
                "predictive_auroc": 0.9715,
                "headline_ensemble_unchanged": True,
                "memory_contribution_preserved": True,
            },
            "verifier_product_hardened": {
                "hardened": True,
                "abstention_cli_batch_surface": True,
                "gaming_resistance_curve": True,
                "product_headline_provenance_confirmed": True,
            },
            "anomaly_escalation_validated": True,
            "anomaly_escalation_validation": {
                "validated": True,
                "supports_wiring_in": False,
                "false_escalation_rate": 0.833333,
                "frame_violating_recall": 1.0,
            },
            "not_landed_or_blocked_recorded_honestly": [
                {
                    "experiment_id": 3787,
                    "path": str(
                        root
                        / "results"
                        / "experiment_3787_p1_discrete_search_adjudication_v3_retry.json"
                    ),
                    "reason": "blocked_no_free_gpu",
                    "status": "blocked",
                }
            ],
            "cited_upstream_artifacts": [
                {"experiment_id": experiment_id, "status": status}
                for experiment_id, status in [
                    (3786, "landed"),
                    (3787, "blocked"),
                    (3788, "landed"),
                    (3789, "landed"),
                    (3790, "landed"),
                    (3791, "landed"),
                    (3792, "landed"),
                    (3793, "landed"),
                    (3794, "landed"),
                    (3795, "landed"),
                ]
            ],
            "flagged_artifacts_excluded": [],
            "frozen_headline_unchanged": True,
            "frozen_fover_auroc": 0.9131,
            "random_seed": 3796,
            "reproducibility_checksum": "6" * 64,
            "duration_s": 0.5,
        },
    )


def test_req_report_3797_spec_anchor_exists() -> None:
    """REQ-REPORT-3797: OpenSpec declares the archive/activation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3797" in spec
    assert "SCENARIO-REPORT-3797" in spec
    assert "SCENARIO-REPORT-3797-P1-HANDOFF-GUARD" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_3797_run_archives_v347_handoff_and_activates_v348(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3797: archive records P1 handoff and .348 focus."""

    _seed_repo(tmp_path)
    before_docs = {
        "roadmap": (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8"),
        "conductor": (tmp_path / "scripts" / "research_conductor.py").read_text(
            encoding="utf-8"
        ),
        "north": (tmp_path / "ops" / "north-star.md").read_text(encoding="utf-8"),
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
    }

    out_path = mod.run(
        tmp_path,
        publication_gate_report=_publication_gate_ready(),
        started_s=10.0,
        now_s=10.5,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    complete = yaml.safe_load(complete_text)
    row = [item for item in complete["milestones"] if str(item.get("id")) == "2026.06.347"]
    task_results = {task["id"]: task["result"] for task in row[0]["tasks"]}

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.TERMINAL_VERDICT
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["v347_outcome_recorded"] == mod.V347_OUTCOME_RECORDED
    assert artifact["v348_focus_recorded"] == mod.V348_FOCUS_RECORDED
    assert artifact["p1_handed_to_operator_recorded"] is True
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["both_energy_routes_still_bounded"] is True
    assert artifact["n_tasks_archived"] == 11
    assert artifact["n_tasks_terminal"] == 11
    assert artifact["blocked_handoff_task_ids"] == [
        "exp3787-p1-discrete-search-adjudication-v3-retry"
    ]
    assert artifact["v348_active_confirmed"] is True
    assert artifact["active_roadmap_path"] == "research-roadmap.yaml"
    assert artifact["paper_ready_evidence"] == {
        "publication_gate_source": "provided_report",
        "paper_ready": True,
        "g1": True,
        "g2": True,
        "g3": True,
        "g4": True,
        "unmet_gates": [],
        "capstone_paper_ready": True,
        "frozen_headline_auroc": 0.9131,
        "frozen_headline_unchanged": True,
    }
    assert artifact["v347_capstone_evidence"]["exp3787_blocked_no_free_gpu"] is True
    assert artifact["v347_capstone_evidence"]["p1_handoff_to_operator"] is True
    assert artifact["v347_capstone_evidence"]["edlm_preflight_go"] is True
    assert artifact["v347_capstone_evidence"]["product_headline_restorable"] == "not_yet_eligible"
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"]["max_severity"] < 2
    assert artifact["random_seed"] == 3797
    assert artifact["duration_s"] == 0.5
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    encoded = json.dumps(artifact, sort_keys=True)
    assert "model_specs" not in encoded
    assert "target_model" not in encoded
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "live-model" not in encoded

    assert complete_text.count("- id: 2026.06.347") == 1
    assert yaml.safe_load(complete_text)
    assert "See conductor log" not in complete_text
    assert "OK (conductor)" not in complete_text
    assert "LANDED TERMINAL MILESTONE" in row[0]["finding"]
    assert "blocked on no-free-GPU" in row[0]["finding"]
    assert "handed to the operator" in row[0]["finding"]
    assert "not a research negative" in row[0]["finding"]
    assert len(row[0]["tasks"]) == 11
    assert task_results["exp3786-archive-v346-activate-v347"].startswith("COMPLETE:")
    assert task_results["exp3787-p1-discrete-search-adjudication-v3-retry"].startswith(
        "BLOCKED_RESOURCE_HANDOFF:"
    )
    assert "no-free-GPU" in task_results["exp3787-p1-discrete-search-adjudication-v3-retry"]
    assert "operator" in task_results["exp3787-p1-discrete-search-adjudication-v3-retry"]
    assert task_results["exp3796-capstone-v347"].startswith("COMPLETE:")
    assert 'result: "BLOCKED_RESOURCE_HANDOFF: exp3787 blocked on no-free-GPU' in complete_text

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_docs[
        "roadmap"
    ]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_docs["conductor"]
    assert (tmp_path / "ops" / "north-star.md").read_text(encoding="utf-8") == before_docs[
        "north"
    ]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before_docs[
        "status"
    ]
    assert (tmp_path / "ops" / "changelog.md").read_text(
        encoding="utf-8"
    ) == before_docs["changelog"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(
        encoding="utf-8"
    ) == before_docs["trace"]


def test_req_report_3797_research_complete_rewrite_is_idempotent(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3797: the .347 archive row is replaced once and stays stable."""

    _seed_repo(tmp_path)

    first_path = mod.run(
        tmp_path,
        publication_gate_report=_publication_gate_ready(),
        started_s=2.0,
        now_s=2.125,
    )
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    first_artifact = json.loads(first_path.read_text(encoding="utf-8"))

    second_path = mod.run(
        tmp_path,
        publication_gate_report=_publication_gate_ready(),
        started_s=2.0,
        now_s=2.125,
    )
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second_artifact = json.loads(second_path.read_text(encoding="utf-8"))

    assert first_complete == second_complete
    assert first_complete.count("- id: 2026.06.347") == 1
    assert first_artifact == second_artifact


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("v347_outcome_recorded"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(inference_substrate="aggregation plus model"), "inference"),
        (lambda p: p.update(v347_outcome_recorded="partial"), ".347 outcome"),
        (lambda p: p.update(v348_focus_recorded="unclear"), ".348 focus"),
        (lambda p: p.update(p1_handed_to_operator_recorded=False), "P1 handoff"),
        (lambda p: p.update(research_complete_yaml_parses=False), "safe-load"),
        (lambda p: p.update(paper_ready_preserved=False), "paper_ready"),
        (lambda p: p.update(both_energy_routes_still_bounded=False), "energy routes"),
        (lambda p: p.update(n_tasks_archived=10), "11"),
        (lambda p: p.update(n_tasks_terminal=10), "terminal"),
        (lambda p: p.update(blocked_handoff_task_ids=[]), "exp3787"),
        (lambda p: p.update(adversarial_verify_clean=False), "adversarial"),
        (lambda p: p.update(random_seed=1), "random_seed"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(target_model="none"), "target_model"),
        (lambda p: p.update(copied_marker="GGUF"), "compute-bound markers"),
    ],
)
def test_req_report_3797_validate_artifact_rejects_dishonest_fields(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3797: validation blocks silent archive regressions."""

    _seed_repo(tmp_path)
    payload = json.loads(
        mod.run(
            tmp_path,
            publication_gate_report=_publication_gate_ready(),
            started_s=4.0,
            now_s=4.5,
        ).read_text(encoding="utf-8")
    )
    mod.validate_artifact(payload)

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_3797_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3797: malformed inputs do not produce a terminal archive."""

    _seed_repo(tmp_path, active_milestone="2026.06.347")
    with pytest.raises(ValueError, match=".348 active"):
        mod.run(
            tmp_path,
            publication_gate_report=_publication_gate_ready(),
            started_s=1.0,
            now_s=1.1,
        )

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3796_capstone_v347.json",
        {"paper_ready_preserved": False, "not_landed_or_blocked_recorded_honestly": []},
    )
    with pytest.raises(ValueError, match=".347 capstone"):
        mod.build_artifact(
            tmp_path,
            research_complete_yaml_parses=True,
            publication_gate_report=_publication_gate_ready(),
        )

    _seed_repo(tmp_path)
    with pytest.raises(ValueError, match="publication gate"):
        mod.build_artifact(
            tmp_path,
            research_complete_yaml_parses=True,
            publication_gate_report={"paper_ready": False, "gates": {}, "unmet_gates": ["G2"]},
        )

    _seed_repo(tmp_path)
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match=".348 active"):
        mod.build_artifact(
            tmp_path,
            research_complete_yaml_parses=True,
            publication_gate_report=_publication_gate_ready(),
        )

    appended = mod.rewrite_research_complete("milestones:\n- id: 2026.06.346\n")
    assert appended.count("- id: 2026.06.347") == 1
    assert yaml.safe_load(appended)
    assert mod.rewrite_research_complete("").startswith("milestones:\n- id: 2026.06.347")
    assert "milestones:\n- id: 2026.06.347" in mod.rewrite_research_complete(
        "# completed only\n"
    )


def test_scenario_report_3797_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3797: the requested script entrypoint exists."""

    script = Path("scripts/experiment_3797_archive_v347_activate_v348.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v347_activate_v348_3797" in text
