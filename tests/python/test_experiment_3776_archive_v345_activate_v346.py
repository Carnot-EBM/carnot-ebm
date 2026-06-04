"""Tests for Exp 3776 v345 archive and v346 activation.

Spec refs: REQ-REPORT-3776, SCENARIO-REPORT-3776.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import yaml

from carnot.reporting import archive_v345_activate_v346_3776 as mod


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


def _seed_repo(root: Path, *, active_milestone: str = "2026.06.346") -> None:
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{active_milestone}"\n'
        'milestone_title: "POST-BOUNDED CONVERGENCE -- P1 discrete-search v3, '
        'bank verifier product, Anomaly-Escalation, EDLM, self-learning, re-grinds nothing bounded"\n'
        "tasks:\n"
        "  - id: exp3776-archive-v345-activate-v346\n"
        "    title: Archive .345 and activate .346 convergence\n"
        "  - id: exp3777-thesis-a-p1-discrete-search-v3\n"
        "    title: P1 discrete-search v3\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap - Milestone 2026.06.346\n\n"
        ".345 fully landed -- 11/11 tasks and paper_ready=true. .346 is a "
        "CONVERGENCE milestone: settle P1 discrete-search v3, bank the verifier "
        "product, build Anomaly-Escalation, scaffold EDLM, continue self-learning, "
        "and re-grind nothing bounded.\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.344\n"
        "  finding: previous archive\n"
        "- id: 2026.06.345\n"
        "  title: stale generic archive\n"
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md\n"
        "  completed: '2026-06-04'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3775-capstone-v345\n"
        "    title: Capstone .345\n"
        "    deliverable: results/experiment_3775_capstone_v345.json\n"
        "    result: OK (conductor)\n",
        encoding="utf-8",
    )
    (root / "ops" / "north-star.md").write_text(
        "# North Star\n\nFrozen FoVer headline AUROC: 0.9131.\n",
        encoding="utf-8",
    )
    (root / "ops" / "changelog.md").write_text(
        "Capstone .345 completed: 11/11 tasks; paper_ready TRUE; both energy routes bounded.\n"
        "Planned milestone 2026.06.346: P1 discrete-search v3, verifier product, "
        "Anomaly-Escalation, EDLM, self-learning.\n",
        encoding="utf-8",
    )
    (root / "ops" / "status.md").write_text("status before\n", encoding="utf-8")
    (root / "_bmad" / "traceability.md").write_text("trace before\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text(
        "# conductor unchanged\n",
        encoding="utf-8",
    )
    _write_json(
        root / "results" / "experiment_3775_capstone_v345.json",
        {
            "honest_verdict": (
                "complete: capstone_v345_skip_cascade_recovered_thesis_a_closed_"
                "both_energy_routes_bounded_gates_mechanized_verifier_banked_"
                "abstention_point_shipped_fr11_v17_prm_positioned_paper_ready_true_"
                "frozen_headline_unchanged"
            ),
            "paper_ready_preserved": True,
            "publication_gate_state": {
                "paper_ready": True,
                "g1": True,
                "g2": True,
                "g3": True,
                "g4": True,
            },
            "both_energy_routes_bounded": True,
            "energy_as_selector_status": "honest-negative-bounded",
            "energy_as_generator_status": "honest-negative-bounded",
            "certified_abstention_point_status": "shipped",
            "verifier_banked_for_ship": True,
            "frozen_headline_unchanged": True,
            "frozen_fover_auroc": 0.9131,
            "headline_aggregation_experiment_ids": list(range(3765, 3775)),
            "not_landed_artifacts_recorded_honestly": [],
            "flagged_artifacts_excluded": [],
            "random_seed": 3775,
            "reproducibility_checksum": "5" * 64,
            "duration_s": 0.1,
        },
    )


def test_req_report_3776_spec_anchor_exists() -> None:
    """REQ-REPORT-3776: OpenSpec declares the archive/activation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3776" in spec
    assert "SCENARIO-REPORT-3776" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_3776_run_archives_v345_fully_landed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3776: archive records .345 as fully landed and .346 active."""

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
    row = [item for item in complete["milestones"] if str(item.get("id")) == "2026.06.345"]
    task_results = {task["id"]: task["result"] for task in row[0]["tasks"]}

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.TERMINAL_VERDICT
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["v345_outcome_recorded"] == mod.V345_OUTCOME_RECORDED
    assert artifact["v346_focus_recorded"] == mod.V346_FOCUS_RECORDED
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["both_energy_routes_still_bounded"] is True
    assert artifact["n_tasks_archived"] == 11
    assert artifact["v346_active_confirmed"] is True
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
    assert artifact["v345_capstone_evidence"]["n_upstream_tasks_landed"] == 10
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"]["max_severity"] < 2
    assert artifact["random_seed"] == 3776
    assert artifact["duration_s"] == 0.5
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    encoded = json.dumps(artifact, sort_keys=True)
    assert "model_specs" not in encoded
    assert "target_model" not in encoded
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "live-model" not in encoded

    assert complete_text.count("- id: 2026.06.345") == 1
    assert yaml.safe_load(complete_text)
    assert "See conductor log" not in complete_text
    assert "OK (conductor)" not in complete_text
    assert "FULLY-LANDED MILESTONE" in row[0]["finding"]
    assert "11/11 tasks completed" in row[0]["finding"]
    assert "paper_ready TRUE" in row[0]["finding"]
    assert "verifier product banked" in row[0]["finding"]
    assert "both energy routes stayed bounded" in row[0]["finding"]
    assert len(row[0]["tasks"]) == 11
    assert task_results["exp3765-archive-v344-activate-v345"].startswith("COMPLETE:")
    assert task_results["exp3775-capstone-v345"].startswith("COMPLETE:")
    assert 'result: "COMPLETE: capstone .345 aggregated all 11/11 tasks' in complete_text

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


def test_req_report_3776_research_complete_rewrite_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3776: the .345 archive row is replaced once and stays stable."""

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
    assert first_complete.count("- id: 2026.06.345") == 1
    assert first_artifact == second_artifact


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("v345_outcome_recorded"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (
            lambda p: p["field_principles"].pop("v346_focus_recorded"),
            "missing field principles",
        ),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(inference_substrate="aggregation plus model"), "inference"),
        (lambda p: p.update(v345_outcome_recorded="partial"), ".345 outcome"),
        (lambda p: p.update(v346_focus_recorded="unclear"), ".346 focus"),
        (lambda p: p.update(research_complete_yaml_parses=False), "safe-load"),
        (lambda p: p.update(paper_ready_preserved=False), "paper_ready"),
        (lambda p: p.update(both_energy_routes_still_bounded=False), "energy routes"),
        (lambda p: p.update(n_tasks_archived=10), "11"),
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
def test_req_report_3776_validate_artifact_rejects_dishonest_fields(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3776: validation blocks silent archive regressions."""

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


def test_req_report_3776_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3776: malformed inputs do not produce a terminal archive."""

    _seed_repo(tmp_path, active_milestone="2026.06.345")
    with pytest.raises(ValueError, match=".346 active"):
        mod.run(tmp_path, publication_gate_report=_publication_gate_ready(), started_s=1.0, now_s=1.1)

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3775_capstone_v345.json",
        {"paper_ready_preserved": False, "headline_aggregation_experiment_ids": []},
    )
    with pytest.raises(ValueError, match="fully landed"):
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
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")

    appended = mod.rewrite_research_complete("milestones:\n- id: 2026.06.344\n")
    assert appended.count("- id: 2026.06.345") == 1
    assert yaml.safe_load(appended)
    assert mod.rewrite_research_complete("").startswith("milestones:\n- id: 2026.06.345")
    assert "milestones:\n- id: 2026.06.345" in mod.rewrite_research_complete(
        "# completed only\n"
    )
    assert mod.yaml_parses("bad: [") is False
    assert mod.yaml_parses(appended) is True
    assert mod.safe_point({"point": 0.9131234}) == 0.9131
    assert mod.safe_point("not-number") is None
    assert mod.duration_from(None, None) == 0.0001
    assert mod.compact_verify_report({"flags": [{"severity": "warn"}]}) == {
        "flag_count": 1,
        "max_severity": 1,
        "flags": [{"severity": "warn"}],
    }
    assert mod.sha256_path(tmp_path / "missing.json") == (
        "769b8995b8bf4407c89e906d67601a46266d34922a63ab1754440eecb0657aab"
    )


def test_req_report_3776_publication_gate_subprocess_and_report_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3776: publication-gate subprocess output is normalized."""

    calls: list[dict[str, object]] = []

    def fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(
            {
                "cmd": cmd,
                "cwd": cwd,
                "check": check,
                "capture_output": capture_output,
                "text": text,
            }
        )
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(_publication_gate_ready()),
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    report = mod.evaluate_publication_gate(tmp_path)

    assert calls == [
        {
            "cmd": [mod.sys.executable, "scripts/publication_gate.py", "--json"],
            "cwd": tmp_path,
            "check": True,
            "capture_output": True,
            "text": True,
        }
    ]
    assert report["__source__"] == "scripts/publication_gate.py --json"
    assert mod.extract_paper_ready_evidence(
        {"paper_ready_preserved": True, "frozen_fover_auroc": 0.9131, "frozen_headline_unchanged": True},
        {"paper_ready": True, "g1": True, "g2": True, "g3": True, "g4": True},
    ) == {
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
    assert mod.report_is_clean(None) is True


def test_scenario_report_3776_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3776: the requested script entrypoint exists."""

    script = Path("scripts/experiment_3776_archive_v345_activate_v346.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v345_activate_v346_3776" in text
