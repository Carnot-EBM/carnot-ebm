"""Tests for Exp 3786 v346 archive and v347 activation.

Spec refs: REQ-REPORT-3786, SCENARIO-REPORT-3786.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v346_activate_v347_3786 as mod


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


def _seed_repo(root: Path, *, active_milestone: str = "2026.06.347") -> None:
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{active_milestone}"\n'
        'milestone_title: "POST-CONVERGENCE -- retry P1 v3, harden the banked '
        'verifier product, Tier-3 self-learning, anomaly-escalation validation, '
        'EDLM preflight, and re-grind nothing bounded"\n'
        "tasks:\n"
        "  - id: exp3786-archive-v346-activate-v347\n"
        "    title: Archive .346 and activate .347 post-convergence\n"
        "  - id: exp3787-p1-discrete-search-adjudication-v3-retry\n"
        "    title: Retry P1 discrete-search v3\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap - Milestone 2026.06.347\n\n"
        ".346 landed 10/11 with exp3777 blocked on no-free-GPU. .347 is a lean "
        "POST-CONVERGENCE milestone: retry P1 v3, harden the banked verifier "
        "product, continue Tier-3 self-learning, validate anomaly-escalation, "
        "scaffold the EDLM preflight, and re-grind nothing bounded.\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.345\n"
        "  finding: previous archive\n"
        "- id: 2026.06.346\n"
        "  title: stale conductor archive\n"
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md\n"
        "  completed: '2026-06-04'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3777-p1-discrete-search-adjudication-v3\n"
        "    title: P1 discrete-search v3\n"
        "    deliverable: results/experiment_3777_p1_discrete_search_adjudication_v3.json\n"
        "    result: OK (conductor)\n",
        encoding="utf-8",
    )
    (root / "ops" / "north-star.md").write_text(
        "# North Star\n\nFrozen FoVer headline AUROC: 0.9131.\n",
        encoding="utf-8",
    )
    (root / "ops" / "changelog.md").write_text(
        "Planned milestone 2026.06.346. Tasks included exp3777 P1 discrete-search v3.\n"
        "Capstone .346 complete: blocked_missing_upstream_artifact; paper_ready TRUE; "
        "both energy routes bounded; .347 planned as post-convergence.\n",
        encoding="utf-8",
    )
    (root / "ops" / "status.md").write_text("status before\n", encoding="utf-8")
    (root / "_bmad" / "traceability.md").write_text("trace before\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text(
        "# conductor unchanged\n",
        encoding="utf-8",
    )
    _write_json(
        root / "results" / "experiment_3785_capstone_v346.json",
        {
            "honest_verdict": (
                "complete: capstone_v346_p1_blocked_missing_upstream_artifact_"
                "energy_as_generator_still_bounded_verifier_product_banked_"
                "anomaly_escalation_prototyped_edlm_scaffolded_fr11_v18_"
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
            "p1_adjudication": "blocked_missing_upstream_artifact",
            "p1_mechanism_status": "open_for_operator: blocked_missing_upstream_artifact",
            "p1_positive_control_passed": False,
            "energy_as_generator_still_bounded": True,
            "energy_as_selector_status": "honest-negative-bounded",
            "energy_as_generator_status": "honest-negative-bounded",
            "verifier_product_banked": True,
            "anomaly_escalation_prototyped": True,
            "edlm_seed_scaffolded": True,
            "fr11_v18_self_learning": True,
            "g4_correction_prepped": True,
            "frozen_headline_unchanged": True,
            "frozen_fover_auroc": 0.9131,
            "headline_aggregation_experiment_ids": [
                3776,
                3778,
                3779,
                3780,
                3781,
                3782,
                3783,
                3784,
            ],
            "not_landed_or_blocked_recorded_honestly": [
                {
                    "experiment_id": 3777,
                    "path": str(
                        root
                        / "results"
                        / "experiment_3777_p1_discrete_search_adjudication_v3.json"
                    ),
                    "reason": "artifact_missing",
                    "status": "not-landed",
                }
            ],
            "flagged_artifacts_excluded": [],
            "random_seed": 3785,
            "reproducibility_checksum": "5" * 64,
            "duration_s": 0.1,
        },
    )


def test_req_report_3786_spec_anchor_exists() -> None:
    """REQ-REPORT-3786: OpenSpec declares the archive/activation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3786" in spec
    assert "SCENARIO-REPORT-3786" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_3786_run_archives_v346_landed_10_of_11(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3786: archive records .346 as 10/11 with exp3777 blocked."""

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
    row = [item for item in complete["milestones"] if str(item.get("id")) == "2026.06.346"]
    task_results = {task["id"]: task["result"] for task in row[0]["tasks"]}

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.TERMINAL_VERDICT
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["v346_outcome_recorded"] == mod.V346_OUTCOME_RECORDED
    assert artifact["v347_focus_recorded"] == mod.V347_FOCUS_RECORDED
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["both_energy_routes_still_bounded"] is True
    assert artifact["n_tasks_archived"] == 11
    assert artifact["n_tasks_landed"] == 10
    assert artifact["blocked_task_ids"] == ["exp3777-p1-discrete-search-adjudication-v3"]
    assert artifact["v347_active_confirmed"] is True
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
    assert artifact["v346_capstone_evidence"]["exp3777_blocked_no_free_gpu"] is True
    assert artifact["v346_capstone_evidence"]["p1_adjudication"] == (
        "blocked_missing_upstream_artifact"
    )
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"]["max_severity"] < 2
    assert artifact["random_seed"] == 3786
    assert artifact["duration_s"] == 0.5
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    encoded = json.dumps(artifact, sort_keys=True)
    assert "model_specs" not in encoded
    assert "target_model" not in encoded
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "live-model" not in encoded

    assert complete_text.count("- id: 2026.06.346") == 1
    assert yaml.safe_load(complete_text)
    assert "See conductor log" not in complete_text
    assert "OK (conductor)" not in complete_text
    assert "LANDED 10/11 MILESTONE" in row[0]["finding"]
    assert "blocked on no-free-GPU" in row[0]["finding"]
    assert "not a research negative" in row[0]["finding"]
    assert len(row[0]["tasks"]) == 11
    assert task_results["exp3776-archive-v345-activate-v346"].startswith("COMPLETE:")
    assert task_results["exp3777-p1-discrete-search-adjudication-v3"].startswith(
        "BLOCKED_RESOURCE:"
    )
    assert "no-free-GPU" in task_results["exp3777-p1-discrete-search-adjudication-v3"]
    assert task_results["exp3785-capstone-v346"].startswith("COMPLETE:")
    assert 'result: "BLOCKED_RESOURCE: exp3777 blocked on no-free-GPU' in complete_text

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


def test_req_report_3786_research_complete_rewrite_is_idempotent(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3786: the .346 archive row is replaced once and stays stable."""

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
    assert first_complete.count("- id: 2026.06.346") == 1
    assert first_artifact == second_artifact


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("v346_outcome_recorded"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(inference_substrate="aggregation plus model"), "inference"),
        (lambda p: p.update(v346_outcome_recorded="partial"), ".346 outcome"),
        (lambda p: p.update(v347_focus_recorded="unclear"), ".347 focus"),
        (lambda p: p.update(research_complete_yaml_parses=False), "safe-load"),
        (lambda p: p.update(paper_ready_preserved=False), "paper_ready"),
        (lambda p: p.update(both_energy_routes_still_bounded=False), "energy routes"),
        (lambda p: p.update(n_tasks_archived=10), "11"),
        (lambda p: p.update(n_tasks_landed=9), "10"),
        (lambda p: p.update(blocked_task_ids=[]), "exp3777"),
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
def test_req_report_3786_validate_artifact_rejects_dishonest_fields(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3786: validation blocks silent archive regressions."""

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


def test_req_report_3786_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3786: malformed inputs do not produce a terminal archive."""

    _seed_repo(tmp_path, active_milestone="2026.06.346")
    with pytest.raises(ValueError, match=".347 active"):
        mod.run(
            tmp_path,
            publication_gate_report=_publication_gate_ready(),
            started_s=1.0,
            now_s=1.1,
        )

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3785_capstone_v346.json",
        {"paper_ready_preserved": False, "not_landed_or_blocked_recorded_honestly": []},
    )
    with pytest.raises(ValueError, match=".346 capstone"):
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
    with pytest.raises(ValueError, match=".347 active"):
        mod.build_artifact(
            tmp_path,
            research_complete_yaml_parses=True,
            publication_gate_report=_publication_gate_ready(),
        )

    appended = mod.rewrite_research_complete("milestones:\n- id: 2026.06.345\n")
    assert appended.count("- id: 2026.06.346") == 1
    assert yaml.safe_load(appended)
    assert mod.rewrite_research_complete("").startswith("milestones:\n- id: 2026.06.346")
    assert "milestones:\n- id: 2026.06.346" in mod.rewrite_research_complete(
        "# completed only\n"
    )


def test_scenario_report_3786_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3786: the requested script entrypoint exists."""

    script = Path("scripts/experiment_3786_archive_v346_activate_v347.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v346_activate_v347_3786" in text
