"""Tests for the Exp 1140 roadmap gate/prior-failures audit.

Spec: REQ-INFRA-075, SCENARIO-INFRA-084, SCENARIO-INFRA-085,
      SCENARIO-INFRA-086
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module(name: str, path: Path):
    """Load a script module from disk without requiring scripts/ to be a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


audit_mod = _load_module("audit_roadmap_gates", SCRIPTS_DIR / "audit_roadmap_gates.py")
experiment_mod = _load_module(
    "experiment_1140_roadmap_gate_prior_failures_audit",
    SCRIPTS_DIR / "experiment_1140_roadmap_gate_prior_failures_audit.py",
)


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _roadmap(tasks: list[dict]) -> dict:
    return {
        "milestone": "2026.04.99",
        "milestone_title": "Synthetic milestone",
        "milestone_doc": "openspec/change-proposals/synthetic.md",
        "tasks": tasks,
    }


def _task(task_id: str, title: str, **overrides) -> dict:
    base = {
        "id": task_id,
        "milestone": "2026.04.99",
        "deliverable": f"results/{task_id}.json",
        "title": title,
        "prompt": "REQUIRED ARTIFACT FIELDS:\n- audit_ok: bool\n",
    }
    base.update(overrides)
    return base


def _complete(tasks: list[dict]) -> dict:
    return {
        "milestones": [
            {
                "id": "2026.04.98",
                "title": "Synthetic complete milestone",
                "tasks": tasks,
            }
        ]
    }


def test_clean_roadmap_passes_all_audit_gates(tmp_path: Path) -> None:
    """REQ-INFRA-075: clean gates, prior-failure coverage, and Codex model pass."""
    roadmap_path = _write_yaml(
        tmp_path / "research-roadmap.yaml",
        _roadmap(
            [
                _task(
                    "exp1-upstream",
                    "Fresh Certifier Producer",
                    prompt="REQUIRED ARTIFACT FIELDS:\n- audit_ok: bool\n",
                ),
                _task(
                    "exp2-downstream",
                    "Fresh Consumer",
                    agent_type="codex",
                    model="gpt-5.5",
                    gated_on=[
                        {
                            "upstream": "exp1-upstream",
                            "artifact_field": "audit_ok",
                            "op": "==",
                            "value": True,
                        }
                    ],
                ),
            ]
        ),
    )
    complete_path = _write_yaml(
        tmp_path / "research-complete.yaml",
        _complete([{"id": "exp0-old", "title": "Unrelated Finished Task"}]),
    )

    result = audit_mod.audit_roadmap(roadmap_path, complete_path=complete_path)
    artifact = result.to_artifact()

    assert artifact["n_tasks_audited"] == 2
    assert artifact["n_gate_upstream_checks"] == 1
    assert artifact["n_gate_upstream_failures"] == 0
    assert artifact["n_prior_failures_checks"] == 2
    assert artifact["n_prior_failures_missing"] == 0
    assert artifact["n_model_agent_coherence_failures"] == 0
    assert artifact["n_gate_field_cross_ref_failures"] == 0
    assert artifact["roadmap_gate_audit_passed"] is True
    assert artifact["honest_verdict"] == "all_checks_pass"
    assert artifact["failure_details"] == []


def test_missing_upstream_and_missing_artifact_field_are_reported(tmp_path: Path) -> None:
    """SCENARIO-INFRA-084: gate upstream and required-field gaps are failures."""
    roadmap_path = _write_yaml(
        tmp_path / "research-roadmap.yaml",
        _roadmap(
            [
                _task(
                    "exp1-upstream",
                    "Fresh Producer",
                    prompt="REQUIRED ARTIFACT FIELDS:\n- different_field: bool\n",
                ),
                _task(
                    "exp2-downstream",
                    "Fresh Consumer",
                    gated_on=[
                        {
                            "upstream": "exp999-missing",
                            "artifact_field": "audit_ok",
                            "op": "==",
                            "value": True,
                        },
                        {
                            "upstream": "exp1-upstream",
                            "artifact_field": "audit_ok",
                            "op": "==",
                            "value": True,
                        },
                    ],
                ),
            ]
        ),
    )
    complete_path = _write_yaml(tmp_path / "research-complete.yaml", _complete([]))

    artifact = audit_mod.audit_roadmap(roadmap_path, complete_path=complete_path).to_artifact()

    assert artifact["n_gate_upstream_checks"] == 2
    assert artifact["n_gate_upstream_failures"] == 1
    assert artifact["n_gate_field_cross_ref_failures"] == 1
    assert artifact["roadmap_gate_audit_passed"] is False
    assert artifact["honest_verdict"] == "gate_field_gaps_found"
    assert any("GATE_UPSTREAM_EXISTS" in line for line in artifact["failure_details"])
    assert any("GATE_FIELD_CROSS_REF" in line for line in artifact["failure_details"])


def test_prior_failure_keyword_overlap_requires_declaration(tmp_path: Path) -> None:
    """SCENARIO-INFRA-085: >=2 scope-keyword overlap needs prior_failures."""
    complete_path = _write_yaml(
        tmp_path / "research-complete.yaml",
        _complete(
            [
                {
                    "id": "exp1136-wopr-slitherlink-cartridge",
                    "title": "WOPR Slitherlink Puzzle Cartridge",
                }
            ]
        ),
    )
    roadmap_path = _write_yaml(
        tmp_path / "research-roadmap.yaml",
        _roadmap(
            [
                _task(
                    "exp1141-wopr-slitherlink-rescue",
                    "WOPR Slitherlink Puzzle Cartridge Rescue",
                ),
                _task(
                    "exp1142-wopr-slitherlink-documented",
                    "WOPR Slitherlink Followup",
                    prior_failures=[
                        {
                            "experiment_id": "exp1136-wopr-slitherlink-cartridge",
                            "verdict": "blocked_gate_check_failed",
                            "addressed_by": "Declared for this retry.",
                        }
                    ],
                ),
            ]
        ),
    )

    artifact = audit_mod.audit_roadmap(roadmap_path, complete_path=complete_path).to_artifact()

    assert audit_mod.scope_keywords("The WOPR Slitherlink v1 of") == {
        "slitherlink",
        "wopr",
    }
    assert artifact["n_prior_failures_checks"] == 2
    assert artifact["n_prior_failures_missing"] == 1
    assert artifact["honest_verdict"] == "prior_failures_gaps_found"
    assert any("PRIOR_FAILURES_COVERAGE" in line for line in artifact["failure_details"])
    assert any("exp1141-wopr-slitherlink-rescue" in line for line in artifact["failure_details"])


def test_model_agent_coherence_rejects_bad_codex_model_and_gemini(tmp_path: Path) -> None:
    """SCENARIO-INFRA-086: Codex requires gpt-5.5 and Gemini is not allowed."""
    roadmap_path = _write_yaml(
        tmp_path / "research-roadmap.yaml",
        _roadmap(
            [
                _task("exp1-good-codex", "Fresh Codex Task", agent_type="codex", model="gpt-5.5"),
                _task("exp2-bad-codex", "Fresh Bad Codex Task", agent_type="codex", model="opus"),
                _task(
                    "exp3-gemini",
                    "Fresh Gemini Task",
                    agent_type="gemini",
                    model="gemini-3.1-pro-preview",
                ),
            ]
        ),
    )
    complete_path = _write_yaml(tmp_path / "research-complete.yaml", _complete([]))

    artifact = audit_mod.audit_roadmap(roadmap_path, complete_path=complete_path).to_artifact()

    assert artifact["n_model_agent_coherence_failures"] == 2
    assert artifact["honest_verdict"] == "model_agent_incoherence_found"
    assert any("exp2-bad-codex" in line for line in artifact["failure_details"])
    assert any("agent_type=gemini" in line for line in artifact["failure_details"])


def test_experiment_wrapper_writes_required_json_artifact(tmp_path: Path) -> None:
    """REQ-INFRA-075: Exp 1140 runner writes the required artifact schema."""
    project_root = tmp_path
    (project_root / "scripts").mkdir()
    (project_root / "results").mkdir()
    roadmap_path = _write_yaml(
        project_root / "research-roadmap.yaml",
        _roadmap([_task("exp1-upstream", "Fresh Certifier Producer")]),
    )
    complete_path = _write_yaml(project_root / "research-complete.yaml", _complete([]))
    output_path = (
        project_root / "results" / "experiment_1140_roadmap_gate_prior_failures_audit.json"
    )

    artifact = experiment_mod.run_experiment(
        project_root=project_root,
        requested_roadmap=project_root / "research-roadmap-next.yaml",
        complete_path=complete_path,
        output_path=output_path,
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["n_tasks_audited"] == 1
    assert written["audit_script_written"] is True
    assert written["roadmap_gate_audit_passed"] is True
    assert written["honest_verdict"] == "all_checks_pass"
    assert written["roadmap_path_requested"].endswith("research-roadmap-next.yaml")
    assert written["roadmap_path_used"] == str(roadmap_path)
    assert "active research-roadmap.yaml" in written["roadmap_path_note"]


def test_path_selection_yaml_and_prompt_edge_cases(tmp_path: Path) -> None:
    """REQ-INFRA-075: defensive path/YAML helpers have explicit outcomes."""
    existing = tmp_path / "research-roadmap-next.yaml"
    existing.write_text("tasks: []\n", encoding="utf-8")
    selected, note = audit_mod.select_roadmap_path(existing, active_path=tmp_path / "missing.yaml")
    assert selected == existing
    assert note == "requested roadmap path exists"

    missing = tmp_path / "missing-roadmap.yaml"
    selected, note = audit_mod.select_roadmap_path(missing, active_path=tmp_path / "missing.yaml")
    assert selected == missing
    assert note == "requested roadmap path is missing"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    try:
        audit_mod._load_yaml_mapping(tmp_path / "does-not-exist.yaml")
    except FileNotFoundError as exc:
        assert "does-not-exist" in str(exc)
    else:  # pragma: no cover - the assertion above is the expected path
        raise AssertionError("missing YAML did not raise")
    try:
        audit_mod._load_yaml_mapping(bad_yaml)
    except ValueError as exc:
        assert "Top-level YAML value" in str(exc)
    else:  # pragma: no cover - the assertion above is the expected path
        raise AssertionError("non-mapping YAML did not raise")

    assert audit_mod._completed_task_titles(tmp_path / "missing-complete.yaml") == []
    complete_path = _write_yaml(
        tmp_path / "research-complete.yaml",
        {
            "milestones": [
                "not-a-milestone",
                {"tasks": ["not-a-task", {"id": "exp1", "title": "Roadmap Hygiene"}]},
            ]
        },
    )
    assert audit_mod._completed_task_titles(complete_path) == [
        ("exp1", "Roadmap Hygiene", {"hygiene", "roadmap"})
    ]

    assert audit_mod._required_artifact_fields_block("No required fields here") == ""
    assert (
        audit_mod._required_artifact_fields_block(
            "Intro\nREQUIRED ARTIFACT FIELDS:\n- alpha: bool\n\nRun: command\n"
        )
        == "REQUIRED ARTIFACT FIELDS:\n- alpha: bool"
    )
    assert (
        audit_mod._required_artifact_fields_block(
            "REQUIRED ARTIFACT FIELDS:\n- alpha: bool\nNEXT SECTION:\n- beta: bool\n"
        )
        == "REQUIRED ARTIFACT FIELDS:\n- alpha: bool"
    )


def test_malformed_gate_shapes_are_ignored(tmp_path: Path) -> None:
    """REQ-INFRA-075: malformed gated_on shapes do not crash the audit."""
    roadmap_path = _write_yaml(
        tmp_path / "research-roadmap.yaml",
        _roadmap(
            [
                _task("exp1-dict-gates", "Fresh Dict Gates", gated_on={"not": "a-list"}),
                _task("exp2-string-gate", "Fresh String Gate", gated_on=["not-a-dict"]),
            ]
        ),
    )
    artifact = audit_mod.audit_roadmap(
        roadmap_path,
        complete_path=tmp_path / "missing-complete.yaml",
    ).to_artifact()

    assert artifact["n_tasks_audited"] == 2
    assert artifact["n_gate_upstream_checks"] == 0
    assert artifact["roadmap_gate_audit_passed"] is True


def test_audit_cli_main_reports_success_and_missing_file(tmp_path: Path, capsys) -> None:
    """REQ-INFRA-075: CLI prints JSON and returns status by audit outcome."""
    roadmap_path = _write_yaml(
        tmp_path / "research-roadmap.yaml",
        _roadmap([_task("exp1-upstream", "Fresh Certifier Producer")]),
    )
    complete_path = _write_yaml(tmp_path / "research-complete.yaml", _complete([]))

    assert audit_mod.main([str(roadmap_path), "--complete", str(complete_path)]) == 0
    success = json.loads(capsys.readouterr().out)
    assert success["roadmap_gate_audit_passed"] is True
    assert success["roadmap_path_used"] == str(roadmap_path)

    assert (
        audit_mod.main([str(tmp_path / "does-not-exist.yaml"), "--complete", str(complete_path)])
        == 2
    )
    missing = json.loads(capsys.readouterr().out)
    assert "YAML file not found" in missing["error"]


def test_experiment_main_prints_artifact(monkeypatch, capsys) -> None:
    """REQ-INFRA-075: experiment CLI entry point delegates to run_experiment."""
    monkeypatch.setattr(
        experiment_mod, "run_experiment", lambda: {"honest_verdict": "all_checks_pass"}
    )

    assert experiment_mod.main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"honest_verdict": "all_checks_pass"}
