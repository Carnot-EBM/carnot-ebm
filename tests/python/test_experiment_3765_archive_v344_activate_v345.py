"""Tests for Exp 3765 v344 archive and v345 activation.

Spec refs: REQ-REPORT-3765, SCENARIO-REPORT-3765.
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest
import yaml


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_PATH = Path("python/carnot/reporting/archive_v344_activate_v345_3765.py")
MODULE_SPEC = importlib.util.spec_from_file_location("exp3765", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
exp3765 = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(exp3765)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_repo(root: Path, *, active_milestone: str = "2026.06.345") -> None:
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{active_milestone}"\n'
        'milestone_title: "PRODUCT-BANKING RECOVERY -- .344 did not land"\n'
        "tasks:\n"
        "  - id: exp3765-archive-v344-activate-v345\n"
        "    title: Archive .344 and activate .345 recovery\n"
        "  - id: exp3766-thesis-a-definitive-reconcile\n"
        "    title: Reconcile the un-landed .344 agenda\n"
        "  - id: exp3767-g2-mechanical-reproducer\n"
        "    title: Mechanize gates\n"
        "  - id: exp3771-certified-abstention-operating-point\n"
        "    title: Bank verifier and certified abstention\n"
        "  - id: exp3772-fr11-self-learning-v17-verifier-precision-tracker\n"
        "    title: Self-learning recovery\n"
        "  - id: exp3773-verifier-prm-positioning\n"
        "    title: PRM-positioning recovery\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap - Milestone 2026.06.345\n\n"
        ".344 produced ZERO experiments after an unquoted embedded colon in "
        "research-complete.yaml raised yaml.ScannerError, failed test_public_docs_*, "
        "and SKIP-cascaded the whole milestone. .345 re-executes the un-landed "
        ".344 agenda: reconcile, mechanize-gates, bank-verifier, "
        "certified-abstention, self-learning, PRM-positioning.\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "# completed\n\n"
        "milestones:\n"
        "- id: 2026.06.342\n"
        "  finding: previous archive\n"
        "- id: 2026.06.344\n"
        "  title: stale generic archive\n"
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md\n"
        "  completed: '2026-06-03'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3754-archive-v343-activate-v344\n"
        "    title: Archive .343 and activate .344\n"
        "    deliverable: results/experiment_3754_archive_v343_activate_v344.json\n"
        "    result: OK (conductor)\n",
        encoding="utf-8",
    )
    (root / "ops" / "north-star.md").write_text(
        "# North Star\n\nFrozen FoVer headline AUROC: 0.9131.\n",
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
        root / "results" / "operational_retro_2026_06_344.json",
        {
            "milestone": "2026.06.344",
            "experiments_completed": 0,
            "summary": "This milestone produced no experiment commits.",
        },
    )
    _write_json(
        root / "results" / "experiment_3764_capstone_v344.json",
        {
            "honest_verdict": (
                "complete: capstone_v344_thesis_a_not_closed_"
                "both_energy_routes_not_fully_cited_gates_not_mechanized_"
                "verifier_not_banked_abstention_point_skipped_fr11_not_pivoted_"
                "next_thesis_to_operator_paper_ready_true_frozen_headline_unchanged"
            ),
            "both_energy_routes_bounded": False,
            "missing_upstream_artifacts": [
                {
                    "experiment_id": experiment_id,
                    "path": f"results/experiment_{experiment_id}_missing.json",
                    "reason": "artifact_missing",
                }
                for experiment_id in range(3754, 3762)
            ],
            "cited_upstream_artifacts": [
                {
                    "experiment_id": experiment_id,
                    "path": f"results/experiment_{experiment_id}_partial.json",
                    "sha256": f"{experiment_id:064x}"[-64:],
                }
                for experiment_id in (3762, 3763)
            ],
            "paper_ready_preserved": True,
            "frozen_headline_unchanged": True,
            "frozen_fover_auroc": 0.9131,
            "publication_gate": {
                "paper_ready": True,
                "g1": True,
                "g2": True,
                "g3": True,
                "g4": True,
                "unmet_gates": [],
            },
        },
    )


def test_req_report_3765_spec_anchor_exists() -> None:
    """REQ-REPORT-3765: OpenSpec declares the archive/activation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-REPORT-3765" in spec
    assert "SCENARIO-REPORT-3765" in spec
    assert exp3765.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_3765_run_archives_skip_cascade_honestly(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3765: archive records the zero-experiment SKIP cascade."""

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

    out_path = exp3765.run(tmp_path, started_s=10.0, now_s=10.25)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    complete = yaml.safe_load(complete_text)
    row = [item for item in complete["milestones"] if str(item.get("id")) == "2026.06.344"]
    task_results = {task["id"]: task["result"] for task in row[0]["tasks"]}

    exp3765.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp3765.TERMINAL_VERDICT
    assert set(exp3765.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3765.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3765.INFERENCE_SUBSTRATE
    assert artifact["v344_outcome_recorded"] == exp3765.V344_OUTCOME_RECORDED
    assert artifact["v344_skip_cause_recorded"] == exp3765.V344_SKIP_CAUSE_RECORDED
    assert artifact["v345_focus_recorded"] == exp3765.V345_FOCUS_RECORDED
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["n_tasks_archived"] == 11
    assert artifact["v345_active_confirmed"] is True
    assert artifact["active_roadmap_path"] == "research-roadmap.yaml"
    assert artifact["experiments_completed"] == 0
    assert artifact["partial_artifacts_recorded"] == ["exp3762", "exp3763", "exp3764"]
    assert artifact["unlanded_v344_agenda_carried_to_v345"] == [
        "exp3754",
        "exp3755",
        "exp3756",
        "exp3757",
        "exp3758",
        "exp3759",
        "exp3760",
        "exp3761",
    ]
    assert artifact["paper_ready_evidence"] == {
        "g1": True,
        "g2": True,
        "g3": True,
        "g4": True,
        "paper_ready": True,
        "frozen_headline_auroc": 0.9131,
        "frozen_headline_unchanged": True,
    }
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"]["max_severity"] < 2
    assert artifact["random_seed"] == 3765
    assert artifact["duration_s"] == 0.25
    assert artifact["reproducibility_checksum"] == exp3765.payload_checksum(artifact)
    encoded = json.dumps(artifact, sort_keys=True)
    assert "model_specs" not in encoded
    assert "target_model" not in encoded
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "live-model" not in encoded

    assert complete_text.count("- id: 2026.06.344") == 1
    assert yaml.safe_load(complete_text)
    assert "See conductor log" not in complete_text
    assert "OK (conductor)" not in complete_text
    assert "ZERO completed experiments" in row[0]["finding"]
    assert "SKIP cascade" in row[0]["finding"]
    assert "carried into .345" in row[0]["finding"]
    assert len(row[0]["tasks"]) == 11
    assert task_results["exp3754-archive-v343-activate-v344"].startswith(
        "SKIPPED_BY_PRETEST_GATE"
    )
    assert task_results["exp3761-fr11-self-learning-v17-verifier-precision-tracker"].startswith(
        "SKIPPED_BY_PRETEST_GATE"
    )
    assert task_results["exp3762-kv260-opportunistic-continuity-audit"].startswith(
        "PARTIAL_ARTIFACT"
    )
    assert task_results["exp3763-next-phase3-thesis-decision-menu"].startswith(
        "PARTIAL_ARTIFACT"
    )
    assert task_results["exp3764-capstone-v344"].startswith("PARTIAL_ARTIFACT")
    assert 'result: "PARTIAL_ARTIFACT: exp3764 honestly reports missing upstreams"' in (
        complete_text
    )

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


def test_req_report_3765_research_complete_rewrite_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3765: the .344 archive row is replaced once and stays stable."""

    _seed_repo(tmp_path)

    first_path = exp3765.run(tmp_path, started_s=2.0, now_s=2.125)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    first_artifact = json.loads(first_path.read_text(encoding="utf-8"))

    second_path = exp3765.run(tmp_path, started_s=2.0, now_s=2.125)
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second_artifact = json.loads(second_path.read_text(encoding="utf-8"))

    assert first_complete == second_complete
    assert first_complete.count("- id: 2026.06.344") == 1
    assert first_artifact == second_artifact


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("v344_outcome_recorded"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (
            lambda p: p["field_principles"].pop("v345_focus_recorded"),
            "missing field principles",
        ),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(inference_substrate="aggregation plus model"), "inference"),
        (lambda p: p.update(v344_outcome_recorded="negative_result"), ".344 outcome"),
        (lambda p: p.update(v344_skip_cause_recorded="unknown"), "skip cause"),
        (lambda p: p.update(v345_focus_recorded="unclear"), ".345 focus"),
        (lambda p: p.update(research_complete_yaml_parses=False), "safe-load"),
        (lambda p: p.update(paper_ready_preserved=False), "paper_ready"),
        (lambda p: p.update(n_tasks_archived=10), "11"),
        (lambda p: p.update(adversarial_verify_clean=False), "adversarial"),
        (lambda p: p.update(random_seed=1), "random_seed"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(target_model="none"), "target_model"),
    ],
)
def test_req_report_3765_validate_artifact_rejects_dishonest_fields(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3765: validation blocks silent archive regressions."""

    _seed_repo(tmp_path)
    payload = json.loads(
        exp3765.run(tmp_path, started_s=4.0, now_s=4.5).read_text(encoding="utf-8")
    )
    exp3765.validate_artifact(payload)

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        exp3765.validate_artifact(broken)


def test_req_report_3765_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3765: malformed inputs do not produce a terminal archive."""

    _seed_repo(tmp_path, active_milestone="2026.06.344")
    with pytest.raises(ValueError, match=".345 active"):
        exp3765.run(tmp_path, started_s=1.0, now_s=1.1)

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "operational_retro_2026_06_344.json",
        {"experiments_completed": 1},
    )
    with pytest.raises(ValueError, match="zero completed"):
        exp3765.build_artifact(tmp_path, research_complete_yaml_parses=True)

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3764_capstone_v344.json",
        {"paper_ready_preserved": False, "missing_upstream_artifacts": []},
    )
    with pytest.raises(ValueError, match="partial capstone"):
        exp3765.build_artifact(tmp_path, research_complete_yaml_parses=True)

    _seed_repo(tmp_path)
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n", encoding="utf-8")
    assert exp3765.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")

    appended = exp3765.rewrite_research_complete("milestones:\n- id: 2026.06.343\n")
    assert appended.count("- id: 2026.06.344") == 1
    assert yaml.safe_load(appended)
    assert exp3765.rewrite_research_complete("").startswith("milestones:\n- id: 2026.06.344")
    assert "milestones:\n- id: 2026.06.344" in exp3765.rewrite_research_complete(
        "# completed only\n"
    )
    assert exp3765.yaml_parses("bad: [") is False
    assert exp3765.yaml_parses(appended) is True
    assert exp3765.safe_point({"point": 0.9131234}) == 0.9131
    assert exp3765.safe_point("not-number") is None
    assert exp3765.duration_from(None, None) == 0.0001
    assert exp3765.compact_verify_report({"flags": [{"severity": "warn"}]}) == {
        "flag_count": 1,
        "max_severity": 1,
        "flags": [{"severity": "warn"}],
    }
    assert exp3765.sha256_path(tmp_path / "missing.json") == (
        "769b8995b8bf4407c89e906d67601a46266d34922a63ab1754440eecb0657aab"
    )


def test_scenario_report_3765_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3765: the requested script entrypoint exists."""

    script = Path("scripts/experiment_3765_archive_v344_activate_v345.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v344_activate_v345_3765" in text
