"""Tests for Exp 3857 .355 wipeout archive and .356 activation.

Spec refs: REQ-REPORT-3857, SCENARIO-REPORT-3857,
SCENARIO-REPORT-3857-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v355_activate_v356_3857 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _publication_gate_ready() -> dict[str, object]:
    return {
        "paper_ready": True,
        "gates": {
            "G1": {
                "pass": True,
                "detail": "headline measured",
                "source": "experiment_2850_fover_dual_condition_integrity_v4.json",
            },
            "G2": {"pass": True, "detail": "independent reproducer"},
            "G3": {"pass": True, "detail": "narrowing clean"},
            "G4": {
                "pass": True,
                "detail": "numbers trace",
                "source": "experiment_2850_fover_dual_condition_integrity_v4.json",
            },
        },
        "unmet_gates": [],
    }


def _write_headline_artifact(root: Path, auroc: float = 0.9131336) -> None:
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "experiment_2850_fover_dual_condition_integrity_v4.json").write_text(
        json.dumps(
            {
                "condition_a_production_auroc_mean": auroc,
                "n_seeds": 5,
                "random_seed": 42,
                "reproducibility_checksum": "a" * 64,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _seed_repo(root: Path, *, corrupt_complete: bool = False) -> None:
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (root / "openspec" / "change-proposals" / "research-roadmap-v356.md").write_text(
        "# Research Roadmap v356\n\n"
        ".356 re-issues the verifier-MOAT durability question after the .355 "
        "poison-test wipeout. Every task is codex plus requires_codex. "
        "paper_ready stays TRUE and FoVer 0.9131 stays frozen. KV260 terminal, "
        "GateMate and PolarFire pending.\n",
        encoding="utf-8",
    )
    (root / "research-roadmap.yaml").write_text(
        'milestone: "2026.06.356"\n'
        "tasks:\n"
        "  - id: exp3857-archive-v355-activate-v356\n"
        "    agent_type: codex\n"
        "    requires_codex: true\n"
        "  - id: exp3858-build-balanced-step-error-corpus-v2\n"
        "    agent_type: codex\n"
        "    requires_codex: true\n",
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.354\n"
        "  finding: previous archive with poison fixed\n"
        "  tasks:\n"
        "  - id: exp3833-ldt-gap\n"
        "    result: 'complete: ldt_gap_LATTICE_VIABLE_ensemble_sound_abstraction_inform0.591_soundmargin0.010'\n"
        "- id: 2026.06.355\n"
        "  title: stale conductor archive\n"
        "  doc: openspec/change-proposals/research-roadmap-v355.md\n"
        "  completed: '2026-06-05'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3845-archive-v354-activate-v355\n"
        "    result: OK (conductor)\n"
        "  - id: exp3856-capstone-v355\n"
        "    result: OK (conductor)\n"
    )
    if corrupt_complete:
        complete_text += "  - id: poison\n    result: complete: unquoted colon\n"
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")
    (root / "ops" / "changelog.md").write_text("changelog before\n", encoding="utf-8")
    (root / "ops" / "status.md").write_text("status before\n", encoding="utf-8")
    (root / "_bmad" / "traceability.md").write_text("trace before\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text("# conductor before\n", encoding="utf-8")
    (root / "ops" / "north-star.md").write_text(
        "Frozen FoVer headline AUROC: 0.9131.\n",
        encoding="utf-8",
    )
    _write_headline_artifact(root)


def test_req_report_3857_spec_anchor_exists() -> None:
    """REQ-REPORT-3857: OpenSpec declares the wipeout archive contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3857" in spec
    assert "SCENARIO-REPORT-3857" in spec
    assert "SCENARIO-REPORT-3857-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.TERMINAL_VERDICT in spec


def test_scenario_report_3857_run_appends_wipeout_and_activation(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3857: append-only correction records the real .355 state."""

    _seed_repo(tmp_path)
    before = {
        "complete": (tmp_path / "research-complete.yaml").read_text(encoding="utf-8"),
        "roadmap": (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
        "conductor": (tmp_path / "scripts" / "research_conductor.py").read_text(
            encoding="utf-8"
        ),
    }

    out_path = mod.run(
        tmp_path,
        publication_gate_report=_publication_gate_ready(),
        pretest_subset_green=True,
        started_s=5.0,
        now_s=5.5,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    complete = yaml.safe_load(complete_text)
    corrective = complete["milestones"][-1]
    task_results = {task["id"]: task["result"] for task in corrective["tasks"]}

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.TERMINAL_VERDICT
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["archived_milestone"] == "2026.06.355"
    assert artifact["activated_milestone"] == "2026.06.356"
    assert artifact["v356_active_confirmed"] is True
    assert len(artifact["v355_wipeout_root_causes"]) == 2
    assert "poison-test YAML corruption" in artifact["v355_wipeout_root_causes"][0]
    assert "gemini-CLI crash" in artifact["v355_wipeout_root_causes"][1]
    assert artifact["poison_test_fixed"] is True
    assert artifact["pretest_subset_green"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_gate_unmet_gates"] == []
    assert artifact["frozen_fover_auroc_unchanged"] is True
    assert artifact["frozen_fover_auroc"] == 0.9131
    assert artifact["preconditions_checked"] == {
        "v356_design_doc_exists": True,
        "research_complete_yaml_parsed_before": True,
        "research_complete_yaml_parsed_after": True,
        "active_milestone": "2026.06.356",
        "active_roadmap_path": "research-roadmap.yaml",
    }
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["random_seed"] == 3857
    assert artifact["duration_s"] == 0.5
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    assert complete_text.startswith(before["complete"].rstrip())
    assert complete_text.count("correction_type: v355_total_wipeout_archive") == 1
    assert complete_text.count("- id: 2026.06.355") == 2
    assert "See conductor log for per-experiment results." in complete_text
    assert "OK (conductor)" in complete_text
    assert "TOTAL WIPEOUT" in corrective["finding"]
    assert "zero usable result artifacts" in corrective["finding"]
    assert "lines 35485 and 35569" in corrective["finding"]
    assert corrective["activation_recorded"] == "exp3857-archive-v355-activate-v356"
    assert task_results["exp3845-archive-v354-activate-v355"].startswith("FAIL:")
    assert "chunk-NBZI34" in task_results["exp3845-archive-v354-activate-v355"]
    assert task_results["exp3846-build-balanced-step-error-corpus"].startswith(
        "SKIPPED_BY_PRETEST_GATE:"
    )
    assert task_results["exp3847-moat-scissor-at-scale-v2"].startswith("GATE_BLOCK:")
    assert task_results["exp3856-capstone-v355"].startswith("SKIPPED_BY_PRETEST_GATE:")
    assert task_results["exp3857-archive-v355-activate-v356"].startswith("COMPLETE:")
    assert "result: complete:" not in complete_text
    assert "result: 'FAIL: gemini-CLI chunk-NBZI34 crash / 429 Too Many Requests" in complete_text
    assert "result: 'COMPLETE: exp3857 archived .355 wipeout and activated .356" in complete_text
    assert yaml.safe_load(complete_text)

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before["roadmap"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before["changelog"]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before["status"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before["trace"]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before["conductor"]


def test_req_report_3857_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3857: rerunning does not append duplicate corrective records."""

    _seed_repo(tmp_path)

    first = mod.run(
        tmp_path,
        publication_gate_report=_publication_gate_ready(),
        pretest_subset_green=True,
        started_s=2.0,
        now_s=2.25,
    ).read_text(encoding="utf-8")
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second = mod.run(
        tmp_path,
        publication_gate_report=_publication_gate_ready(),
        pretest_subset_green=True,
        started_s=2.0,
        now_s=2.25,
    ).read_text(encoding="utf-8")
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")

    assert first == second
    assert first_complete == second_complete
    assert second_complete.count("correction_type: v355_total_wipeout_archive") == 1


def test_scenario_report_3857_blocked_yaml_writes_artifact_without_append(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3857-BLOCKED-YAML: corrupt YAML exits before appending."""

    _seed_repo(tmp_path, corrupt_complete=True)
    before = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")

    out_path = mod.run(
        tmp_path,
        publication_gate_report=_publication_gate_ready(),
        pretest_subset_green=True,
        started_s=7.0,
        now_s=7.1,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_corrupt")
    assert artifact["preconditions_checked"]["v356_design_doc_exists"] is True
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert artifact["poison_test_fixed"] is False
    assert artifact["pretest_subset_green"] is False
    assert artifact["paper_ready"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("archived_milestone"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(archived_milestone="2026.06.354"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.355"), "activated milestone"),
        (lambda p: p.update(v355_wipeout_root_causes=["poison only"]), "root causes"),
        (lambda p: p.update(poison_test_fixed=False), "poison test"),
        (lambda p: p.update(pretest_subset_green=False), "pretest subset"),
        (lambda p: p.update(paper_ready=False), "paper_ready"),
        (lambda p: p.update(frozen_fover_auroc_unchanged=False), "frozen"),
        (lambda p: p.update(inference_substrate="aggregation_wrong"), "inference"),
        (lambda p: p.update(random_seed=1), "random_seed"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="GGUF"), "compute-bound markers"),
    ],
)
def test_req_report_3857_validate_artifact_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3857: validation rejects fields that would hide the wipeout."""

    _seed_repo(tmp_path)
    payload = json.loads(
        mod.run(
            tmp_path,
            publication_gate_report=_publication_gate_ready(),
            pretest_subset_green=True,
            started_s=9.0,
            now_s=9.5,
        ).read_text(encoding="utf-8")
    )

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_3857_helpers_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3857: helper failures block instead of fabricating success."""

    _seed_repo(tmp_path)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-v356.md").unlink()
    artifact = json.loads(
        mod.run(
            tmp_path,
            publication_gate_report=_publication_gate_ready(),
            pretest_subset_green=True,
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_v356_design_doc_missing")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            publication_gate_report=_publication_gate_ready(),
            pretest_subset_green=False,
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_pretest_subset_failed")
    assert artifact["poison_test_fixed"] is False

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            publication_gate_report={"paper_ready": False, "gates": {}, "unmet_gates": ["G2"]},
            pretest_subset_green=True,
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_publication_gate_unmet")
    assert artifact["publication_gate_unmet_gates"] == ["G2"]

    _seed_repo(tmp_path)
    _write_headline_artifact(tmp_path, auroc=0.9126)
    artifact = json.loads(
        mod.run(
            tmp_path,
            publication_gate_report=_publication_gate_ready(),
            pretest_subset_green=True,
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_frozen_fover_headline_changed")

    def fake_gate(root: Path) -> dict[str, object]:
        assert root == tmp_path
        return _publication_gate_ready()

    _seed_repo(tmp_path)
    monkeypatch.setattr(mod, "evaluate_publication_gate", fake_gate)
    artifact = json.loads(
        mod.run(tmp_path, pretest_subset_green=True, started_s=1.0, now_s=1.1).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["paper_ready"] is True

    empty_append = mod.append_research_complete_record("")
    assert empty_append.startswith("milestones:\n- id: 2026.06.355")
    no_header_append = mod.append_research_complete_record("# completed only\n")
    assert no_header_append.startswith("# completed only\nmilestones:\n- id: 2026.06.355")

    _seed_repo(tmp_path)
    before = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    monkeypatch.setattr(
        mod,
        "append_research_complete_record",
        lambda _text: "milestones:\n- id: bad\n  result: complete: broken\n",
    )
    artifact = json.loads(
        mod.run(
            tmp_path,
            publication_gate_report=_publication_gate_ready(),
            pretest_subset_green=True,
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_research_complete_append_invalid")
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before


def test_req_report_3857_pretest_subprocess_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3857: smart-subset helper maps subprocess status to a bare bool."""

    calls: list[list[str]] = []

    def succeed(cmd: list[str], **kwargs: object) -> object:
        calls.append(cmd)
        assert kwargs["cwd"] == tmp_path
        return object()

    monkeypatch.setattr(mod.subprocess, "run", succeed)
    assert mod.evaluate_pretest_subset(tmp_path) is True
    assert calls == [
        [
            str(mod.PYTHON_BIN),
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "tests/python/test_pipeline_extract.py",
            "tests/python/test_docs.py",
            "-q",
        ]
    ]

    def fail(cmd: list[str], **kwargs: object) -> object:
        raise mod.subprocess.CalledProcessError(1, cmd, output="", stderr="failed")

    monkeypatch.setattr(mod.subprocess, "run", fail)
    assert mod.evaluate_pretest_subset(tmp_path) is False


def test_scenario_report_3857_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3857: the requested script entrypoint exists."""

    script = Path("scripts/experiments/experiment_3857_archive_v355_activate_v356.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v355_activate_v356_3857" in text
