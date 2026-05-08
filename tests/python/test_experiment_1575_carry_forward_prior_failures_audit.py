"""Tests for the Exp 1575 carry-forward prior-failure audit.

Spec: REQ-REPORT-064, SCENARIO-REPORT-064
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "experiment_1575_carry_forward_prior_failures_autofill_audit.py"
)


def _load_module():
    """Load the standalone script without requiring scripts/ to be a package."""
    spec = importlib.util.spec_from_file_location("experiment_1575_audit", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_1575_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_source_artifact(project_root: Path, exp_id: str, verdict: str) -> None:
    number, slug = exp_id.removeprefix("exp").split("-", 1)
    path = project_root / "results" / f"experiment_{number}_{slug.replace('-', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": "complete", "honest_verdict": verdict}) + "\n",
        encoding="utf-8",
    )


def _prior(exp_id: str, verdict: str, retire: bool) -> dict[str, object]:
    return {
        "experiment_id": exp_id,
        "verdict": verdict,
        "addressed_by": f"SCENARIO-REPORT-064 fixture covers {exp_id}",
        "retire_if_same_verdict": retire,
    }


def _write_roadmap(
    project_root: Path,
    exp1576_priors: list[dict[str, object]],
    exp1577_priors: list[dict[str, object]],
    *,
    include_exp1577: bool = True,
) -> Path:
    tasks = [
        {
            "id": "exp1576-paper-v6-section-3-sampler-draft-resumed",
            "milestone": "2026.05.121",
            "deliverable": "results/experiment_1576_fixture.json",
            "title": "Paper-v6 Section 3 Sampler Draft Resumed from .120 exp1569",
            "prompt": "fixture",
            "prior_failures": exp1576_priors,
        },
    ]
    if include_exp1577:
        tasks.append(
            {
                "id": "exp1577-extropic-z1-readiness-packet-thrml-alignment-resumed",
                "milestone": "2026.05.121",
                "deliverable": "results/experiment_1577_fixture.json",
                "title": "Extropic Z1 Readiness Packet THRML Alignment Resumed from .120 exp1573",
                "prompt": "fixture",
                "prior_failures": exp1577_priors,
            }
        )
    roadmap = project_root / "research-roadmap.yaml"
    roadmap.write_text(
        yaml.safe_dump(
            {
                "milestone": "2026.05.121",
                "milestone_title": "Fixture",
                "milestone_doc": "openspec/capabilities/research-reporting/spec.md",
                "tasks": tasks,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return roadmap


def test_req_report_064_run_experiment_writes_ready_artifact_with_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-REPORT-064: fallback active roadmap plus exact verdict checks can pass."""
    mod = _load_module()
    _write_source_artifact(tmp_path, "exp1569-paper-v6-section-3-sampler-draft", "blocked_gate_check_failed")
    _write_source_artifact(tmp_path, "exp1269-paper-v6-critical-fixes-v2", "paper_v6_critical_fixes_v2_complete")
    _write_source_artifact(
        tmp_path,
        "exp1573-extropic-z1-readiness-packet-thrml-alignment-update",
        "blocked_gate_check_failed",
    )
    _write_source_artifact(
        tmp_path,
        "exp1558-thrml-post-rng-scale-decision-extropic-update",
        "blocked_gate_check_failed",
    )
    _write_roadmap(
        tmp_path,
        [
            _prior("exp1569-paper-v6-section-3-sampler-draft", "blocked_gate_check_failed", True),
            _prior("exp1269-paper-v6-critical-fixes-v2", "paper_v6_critical_fixes_v2_complete", False),
        ],
        [
            _prior(
                "exp1573-extropic-z1-readiness-packet-thrml-alignment-update",
                "blocked_gate_check_failed",
                True,
            ),
            _prior(
                "exp1558-thrml-post-rng-scale-decision-extropic-update",
                "blocked_gate_check_failed",
                False,
            ),
        ],
    )
    calls: list[list[str]] = []

    def fake_run_command(args: list[str], cwd: Path) -> dict[str, object]:
        calls.append(args)
        stdout = "ok\n"
        if args[1].endswith("conductor_priors_autofill.py"):
            stdout = "2 tasks scanned, 0 stubs generated, 2 already populated\n"
        return {"command": args, "returncode": 0, "stdout": stdout, "stderr": ""}

    monkeypatch.setattr(mod, "_run_command", fake_run_command)
    output_path = tmp_path / "results" / "experiment_1575_carry_forward_prior_failures_autofill_audit.json"

    artifact = mod.run_experiment(project_root=tmp_path, output_path=output_path)

    assert artifact["status"] == "complete"
    assert artifact["autofill_dry_run_completed"] is True
    assert artifact["validate_prior_failures_passed"] is True
    assert artifact["audit_roadmap_gates_passed"] is True
    assert artifact["exp1576_prior_failures_valid"] is True
    assert artifact["exp1577_prior_failures_valid"] is True
    assert artifact["carryforward_prior_failures_ready"] is True
    assert artifact["honest_verdict"] == "carryforward_prior_failures_ready"
    assert artifact["autofill_summary"] == {
        "tasks_scanned": 2,
        "stubs_generated": 0,
        "already_populated": 2,
    }
    assert artifact["roadmap_path_used"].endswith("research-roadmap.yaml")
    assert "requested next roadmap missing" in artifact["roadmap_path_note"]
    assert output_path.exists()
    assert len(calls) == 3
    assert all(any(arg.endswith("research-roadmap.yaml") for arg in call) for call in calls)


def test_scenario_report_064_inspection_reports_mismatch_and_missing_source(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-064: exact task and field details explain blocked gaps."""
    mod = _load_module()
    _write_source_artifact(tmp_path, "exp1569-paper-v6-section-3-sampler-draft", "blocked_gate_check_failed")
    _write_source_artifact(
        tmp_path,
        "exp1573-extropic-z1-readiness-packet-thrml-alignment-update",
        "blocked_gate_check_failed",
    )
    roadmap = _write_roadmap(
        tmp_path,
        [
            {
                "experiment_id": "exp1569-paper-v6-section-3-sampler-draft",
                "verdict": "wrong_verdict",
                "addressed_by": "SCENARIO-REPORT-064 mismatch fixture",
            }
        ],
        [
            _prior(
                "exp1573-extropic-z1-readiness-packet-thrml-alignment-update",
                "blocked_gate_check_failed",
                True,
            ),
            _prior(
                "exp1558-thrml-post-rng-scale-decision-extropic-update",
                "blocked_gate_check_failed",
                False,
            ),
        ],
    )

    inspection = mod.inspect_target_prior_failures(tmp_path, roadmap)

    assert inspection["exp1576_prior_failures_valid"] is False
    assert inspection["exp1577_prior_failures_valid"] is False
    details = inspection["prior_failure_gap_details"]
    assert {
        "task_id": "exp1576-paper-v6-section-3-sampler-draft-resumed",
        "field": "prior_failures[0].retire_if_same_verdict",
        "detail": "prior_failures[0] missing/empty fields: ['retire_if_same_verdict']",
    } in details
    assert {
        "task_id": "exp1576-paper-v6-section-3-sampler-draft-resumed",
        "field": "prior_failures[0].verdict",
        "detail": (
            "expected blocked_gate_check_failed from "
            f"{tmp_path / 'results' / 'experiment_1569_paper_v6_section_3_sampler_draft.json'}, "
            "got wrong_verdict"
        ),
    } in details
    assert {
        "task_id": "exp1577-extropic-z1-readiness-packet-thrml-alignment-resumed",
        "field": "prior_failures[1].experiment_id",
        "detail": "no source artifact found for exp1558-thrml-post-rng-scale-decision-extropic-update",
    } in details


def test_scenario_report_064_missing_target_task_is_not_ready(tmp_path: Path) -> None:
    """SCENARIO-REPORT-064: absent carry-forward tasks are explicit blockers."""
    mod = _load_module()
    _write_source_artifact(tmp_path, "exp1569-paper-v6-section-3-sampler-draft", "blocked_gate_check_failed")
    roadmap = _write_roadmap(
        tmp_path,
        [_prior("exp1569-paper-v6-section-3-sampler-draft", "blocked_gate_check_failed", True)],
        [],
        include_exp1577=False,
    )

    inspection = mod.inspect_target_prior_failures(tmp_path, roadmap)

    assert inspection["exp1576_prior_failures_valid"] is True
    assert inspection["exp1577_prior_failures_valid"] is False
    assert {
        "task_id": "exp1577-extropic-z1-readiness-packet-thrml-alignment-resumed",
        "field": "task",
        "detail": "task not found in selected roadmap",
    } in inspection["prior_failure_gap_details"]


def test_req_report_064_command_and_parser_helpers() -> None:
    """REQ-REPORT-064: command outputs and dry-run counts are mechanically recorded."""
    mod = _load_module()

    result = mod._run_command([sys.executable, "-c", "print('ok')"], cwd=REPO_ROOT)

    assert result["returncode"] == 0
    assert result["stdout"] == "ok\n"
    assert mod._parse_autofill_counts(
        "14 tasks scanned, 3 stubs generated, 11 already populated\n"
    ) == {
        "tasks_scanned": 14,
        "stubs_generated": 3,
        "already_populated": 11,
    }
    assert mod._parse_autofill_counts("unexpected") == {
        "tasks_scanned": None,
        "stubs_generated": None,
        "already_populated": None,
    }
    assert mod._in_progress_artifact(REPO_ROOT, "20260508")["status"] == "in_progress"


def test_req_report_064_defensive_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-064: malformed local inputs produce explicit audit gaps."""
    mod = _load_module()
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- not-a-mapping\n", encoding="utf-8")

    try:
        mod._load_yaml_mapping(bad_yaml)
    except ValueError as exc:
        assert "Top-level YAML value must be a mapping" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-mapping YAML to fail")

    assert mod._artifact_path_for_prior(tmp_path, "operator_directive") is None
    assert mod._discipline_gaps("exp-test", None) == [
        {
            "task_id": "exp-test",
            "field": "prior_failures",
            "detail": "prior_failures must be a non-empty list",
        }
    ]
    validation = mod._validate_target_task(
        tmp_path,
        {
            "id": "exp-edge",
            "prior_failures": [
                "not-a-dict",
                {
                    "experiment_id": "operator_directive",
                    "verdict": "not_applicable",
                    "addressed_by": "fixture",
                    "retire_if_same_verdict": False,
                },
            ],
        },
        "exp9999-required-prior",
    )

    assert validation["valid"] is False
    assert {
        "task_id": "exp-edge",
        "field": "prior_failures[0]",
        "detail": "prior_failures entry is not a dict",
    } in validation["details"]
    assert {
        "task_id": "exp-edge",
        "field": "prior_failures.experiment_id",
        "detail": "missing required prior exp9999-required-prior",
    } in validation["details"]
