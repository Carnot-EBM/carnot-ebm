"""Tests for Exp 3808 v348 archive and v349 activation.

Spec refs: REQ-REPORT-3808, SCENARIO-REPORT-3808,
SCENARIO-REPORT-3808-PRODUCT-HEADLINE-GUARD.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v348_activate_v349_3808 as mod


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


def _seed_repo(root: Path, *, active_milestone: str = "2026.06.349") -> None:
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{active_milestone}"\n'
        'milestone_title: "POST-CONVERGENCE -- LEAN MAINTENANCE + '
        'OPERATOR-FORK STAGING"\n'
        "tasks:\n"
        "  - id: exp3808-archive-v348-activate-v349\n"
        "    title: Archive .348 and activate .349 lean maintenance\n"
        "  - id: exp3809-anomaly-escalation-advisory-hook\n"
        "    title: WIRE the advisory hook\n"
        "  - id: exp3810-abstention-http-rest-surface-v2\n"
        "    title: REPAIR the HTTP/REST surface\n"
        "  - id: exp3811-cross-surface-abstention-parity\n"
        "    title: Confirm cross-surface parity\n"
        "  - id: exp3812-product-headline-status-consolidation\n"
        "    title: RECORD product headline status honestly\n"
        "  - id: exp3813-fr11-v21-tier3-cross-split\n"
        "    title: CONTINUE Tier-3 self-learning\n"
        "  - id: exp3814-publication-gate-regression\n"
        "    title: CONFIRM publication-gate invariants\n"
        "  - id: exp3815-edlm-operator-seed-staging\n"
        "    title: STAGE EDLM seed for operator\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap - Milestone 2026.06.349\n\n"
        ".348 landed all tasks. Product headline STAYS DEMOTED because exp3798 "
        "delta=0.0pp and exp2090 flags CRITICAL on live re-check, so both "
        "candidate positives fail provenance. .349 is LEAN MAINTENANCE: WIRE "
        "the advisory hook, REPAIR HTTP/REST plus parity, RECORD product-headline "
        "status, CONTINUE Tier-3 self-learning, CONFIRM publication-gate "
        "invariants, and STAGE EDLM seed. It re-grinds nothing bounded and "
        "self-seeds no paradigm.\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.347\n"
        "  finding: previous archive\n"
        "- id: 2026.06.348\n"
        "  title: stale conductor archive\n"
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md\n"
        "  completed: '2026-06-04'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3801-abstention-http-rest-surface\n"
        "    title: HTTP REST surface\n"
        "    deliverable: results/experiment_3801_abstention_http_rest_surface.json\n"
        "    result: OK (conductor)\n",
        encoding="utf-8",
    )
    (root / "ops" / "north-star.md").write_text(
        "# North Star\n\nFrozen FoVer headline AUROC: 0.9131.\n",
        encoding="utf-8",
    )
    (root / "ops" / "changelog.md").write_text("operator reconciler owns this\n", encoding="utf-8")
    (root / "ops" / "status.md").write_text("status before\n", encoding="utf-8")
    (root / "_bmad" / "traceability.md").write_text("trace before\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text("# conductor unchanged\n", encoding="utf-8")
    _write_json(
        root / "results" / "experiment_3807_capstone_v348.json",
        {
            "honest_verdict": (
                "complete: capstone_v348_product_headline_demoted_verifier_product_"
                "hardened_http_rest_blocked_anomaly_classifier_repaired_fr11_v20_"
                "tier3_fast_path_paper_ready_true_frozen_headline_unchanged_"
                "both_energy_routes_bounded"
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
            "product_headline_advanced": {
                "headline_stays_demoted": True,
                "product_headline_restorable": "not_yet_eligible",
                "rerun": {
                    "g4_provenance_complete": True,
                    "positive_control_passed": True,
                    "repair_delta_pp": 0.0,
                },
            },
            "product_headline_restorable": "not_yet_eligible",
            "not_landed_or_blocked_recorded_honestly": [
                {
                    "experiment_id": 3801,
                    "path": str(root / "results" / "experiment_3801_abstention_http_rest_surface.json"),
                    "reason": "blocked_http_abstention_e2e_failed",
                    "status": "blocked",
                }
            ],
            "verifier_product_hardened": {
                "context_compaction_closed": True,
                "clean_auroc_preserved": True,
                "http_rest_blocked": True,
            },
            "anomaly_classifier_repaired": {
                "repaired": True,
                "false_escalation_rate_before": 0.833333,
                "false_escalation_rate_after": 0.0,
                "frame_violating_recall": 1.0,
                "supports_wiring_in": True,
                "conductor_unmodified": True,
            },
            "fr11_v20_tier3_fast_path": {
                "validated": True,
                "skip_rate_at_no_regression": 0.56,
                "effective_auroc_in_frozen_ci": True,
                "headline_ensemble_unchanged": True,
            },
            "energy_as_generator_still_bounded": True,
            "energy_as_selector_status": "honest-negative-bounded",
            "energy_as_generator_status": "honest-negative-bounded",
            "frozen_headline_unchanged": True,
            "frozen_fover_auroc": 0.9131,
            "next_thesis_remains_operator_surface": True,
            "regrinds_nothing_already_bounded": True,
            "no_new_existential_claim": True,
            "cited_upstream_artifacts": [
                {"experiment_id": experiment_id, "status": status}
                for experiment_id, status in [
                    (3797, "landed"),
                    (3798, "landed"),
                    (3799, "landed"),
                    (3800, "landed"),
                    (3801, "blocked"),
                    (3802, "landed"),
                    (3803, "landed"),
                    (3805, "landed"),
                    (3806, "landed"),
                ]
            ],
            "flagged_artifacts_excluded": [{"experiment_id": 3798}],
            "random_seed": 3807,
            "reproducibility_checksum": "7" * 64,
            "duration_s": 0.5,
        },
    )


def test_req_report_3808_spec_anchor_exists() -> None:
    """REQ-REPORT-3808: OpenSpec declares the archive/activation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3808" in spec
    assert "SCENARIO-REPORT-3808" in spec
    assert "SCENARIO-REPORT-3808-PRODUCT-HEADLINE-GUARD" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_3808_run_archives_v348_demotion_and_activates_v349(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3808: archive records demotion and .349 focus."""

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
        started_s=20.0,
        now_s=20.5,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    complete = yaml.safe_load(complete_text)
    row = [item for item in complete["milestones"] if str(item.get("id")) == "2026.06.348"]
    task_results = {task["id"]: task["result"] for task in row[0]["tasks"]}

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.TERMINAL_VERDICT
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["v348_outcome_recorded"] == mod.V348_OUTCOME_RECORDED
    assert artifact["v349_focus_recorded"] == mod.V349_FOCUS_RECORDED
    assert artifact["product_headline_demoted_recorded"] is True
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["both_energy_routes_still_bounded"] is True
    assert artifact["edlm_remains_operator_seed_surface"] is True
    assert artifact["n_tasks_archived"] == 10
    assert artifact["n_tasks_terminal"] == 10
    assert artifact["v349_active_confirmed"] is True
    assert artifact["active_roadmap_path"] == "research-roadmap.yaml"
    assert artifact["paper_ready_evidence"]["paper_ready"] is True
    assert artifact["paper_ready_evidence"]["frozen_headline_auroc"] == 0.9131
    assert artifact["v348_capstone_evidence"]["product_headline_demoted"] is True
    assert artifact["v348_capstone_evidence"]["both_product_positives_fail_provenance"] is True
    assert artifact["v348_capstone_evidence"]["http_rest_repair_target"] is True
    assert artifact["v348_capstone_evidence"]["anomaly_classifier_wirable"] is True
    assert artifact["v348_capstone_evidence"]["tier3_fast_path_landed"] is True
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"]["max_severity"] < 2
    assert artifact["random_seed"] == 3808
    assert artifact["duration_s"] == 0.5
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    encoded = json.dumps(artifact, sort_keys=True)
    assert "model_specs" not in artifact
    assert "target_model" not in artifact
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "live-model" not in encoded

    assert complete_text.count("- id: 2026.06.348") == 1
    assert yaml.safe_load(complete_text)
    assert "See conductor log" not in complete_text
    assert "OK (conductor)" not in complete_text
    assert "LANDED TERMINAL MILESTONE" in row[0]["finding"]
    assert "product headline stays demoted" in row[0]["finding"]
    assert "both candidate product positives fail provenance" in row[0]["finding"]
    assert "not a research negative" in row[0]["finding"]
    assert len(row[0]["tasks"]) == 10
    assert task_results["exp3798-g4-product-headline-restoration"].startswith("COMPLETE_DEMOTED:")
    assert "delta=0.0pp" in task_results["exp3798-g4-product-headline-restoration"]
    assert task_results["exp3801-abstention-http-rest-surface"].startswith(
        "BLOCKED_REPAIR_TARGET:"
    )
    assert "blocked_http_abstention_e2e_failed" in task_results["exp3801-abstention-http-rest-surface"]
    assert task_results["exp3807-capstone-v348"].startswith("COMPLETE:")
    assert 'result: "COMPLETE_DEMOTED: exp3798 G4 re-run produced delta=0.0pp' in complete_text
    assert 'finding: "LANDED TERMINAL MILESTONE:' in complete_text

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


def test_req_report_3808_research_complete_rewrite_is_idempotent(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3808: the .348 archive row is replaced once and stays stable."""

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
    assert first_complete.count("- id: 2026.06.348") == 1
    assert first_artifact == second_artifact


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("v348_outcome_recorded"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(inference_substrate="aggregation plus model"), "inference"),
        (lambda p: p.update(v348_outcome_recorded="partial"), ".348 outcome"),
        (lambda p: p.update(v349_focus_recorded="unclear"), ".349 focus"),
        (lambda p: p.update(product_headline_demoted_recorded=False), "product headline"),
        (lambda p: p.update(research_complete_yaml_parses=False), "safe-load"),
        (lambda p: p.update(paper_ready_preserved=False), "paper_ready"),
        (lambda p: p.update(both_energy_routes_still_bounded=False), "energy routes"),
        (lambda p: p.update(edlm_remains_operator_seed_surface=False), "EDLM"),
        (lambda p: p.update(n_tasks_archived=9), "10"),
        (lambda p: p.update(n_tasks_terminal=9), "terminal"),
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
def test_req_report_3808_validate_artifact_rejects_dishonest_fields(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3808: validation blocks silent archive regressions."""

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


def test_req_report_3808_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3808: malformed inputs do not produce a terminal archive."""

    _seed_repo(tmp_path, active_milestone="2026.06.348")
    with pytest.raises(ValueError, match=".349 active"):
        mod.run(
            tmp_path,
            publication_gate_report=_publication_gate_ready(),
            started_s=1.0,
            now_s=1.1,
        )

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3807_capstone_v348.json",
        {"paper_ready_preserved": False, "not_landed_or_blocked_recorded_honestly": []},
    )
    with pytest.raises(ValueError, match=".348 capstone"):
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
    with pytest.raises(ValueError, match=".349 active"):
        mod.build_artifact(
            tmp_path,
            research_complete_yaml_parses=True,
            publication_gate_report=_publication_gate_ready(),
        )

    appended = mod.rewrite_research_complete("milestones:\n- id: 2026.06.347\n")
    assert appended.count("- id: 2026.06.348") == 1
    assert yaml.safe_load(appended)
    assert mod.rewrite_research_complete("").startswith("milestones:\n- id: 2026.06.348")
    assert "milestones:\n- id: 2026.06.348" in mod.rewrite_research_complete(
        "# completed only\n"
    )


def test_scenario_report_3808_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3808: the requested script entrypoint exists."""

    script = Path("scripts/experiment_3808_archive_v348_activate_v349.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v348_activate_v349_3808" in text
