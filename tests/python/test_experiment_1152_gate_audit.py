"""Tests for the Exp 1152 roadmap pre-activation audit artifact.

Spec: REQ-INFRA-075, SCENARIO-INFRA-087
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


_load_module("audit_roadmap_gates", SCRIPTS_DIR / "audit_roadmap_gates.py")
experiment_mod = _load_module(
    "experiment_1152_gate_audit_pre_activation_v2",
    SCRIPTS_DIR / "experiment_1152_gate_audit_pre_activation_v2.py",
)


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _roadmap(tasks: list[dict]) -> dict:
    return {
        "milestone": "2026.04.90",
        "milestone_title": "Synthetic milestone",
        "tasks": tasks,
    }


def _task(task_id: str, title: str, **overrides) -> dict:
    base = {
        "id": task_id,
        "milestone": "2026.04.90",
        "deliverable": f"results/{task_id}.json",
        "title": title,
        "prompt": "REQUIRED ARTIFACT FIELDS:\n- audit_ok: bool\n",
    }
    base.update(overrides)
    return base


def _complete(tasks: list[dict]) -> dict:
    return {"milestones": [{"id": "2026.04.89", "tasks": tasks}]}


def _arxiv_task(prior_ids: list[str]) -> dict:
    return _task(
        "exp1153-arxiv-final-submission-v4",
        "arXiv Final Submission v4",
        prior_failures=[
            {
                "experiment_id": prior_id,
                "verdict": "blocked_or_stale",
                "addressed_by": "SCENARIO-INFRA-087 coverage fixture",
            }
            for prior_id in prior_ids
        ],
    )


def test_exp1152_runner_writes_required_schema_and_fix_guidance(tmp_path: Path) -> None:
    """REQ-INFRA-075: Exp 1152 writes audit counts with per-failure fix guidance."""
    project_root = tmp_path
    (project_root / "results").mkdir()
    roadmap_path = _write_yaml(
        project_root / "research-roadmap-next.yaml",
        _roadmap(
            [
                _task("exp1152-gate-audit-pre-activation-v2", "Roadmap Gate Audit"),
                _arxiv_task(list(experiment_mod.REQUIRED_ARXIV_PRIOR_IDS)),
            ]
        ),
    )
    complete_path = _write_yaml(
        project_root / "research-complete.yaml",
        _complete([{"id": "exp1140-roadmap-gate-audit", "title": "Roadmap Gate Audit"}]),
    )
    output_path = project_root / "results" / "experiment_1152_gate_audit_pre_activation_v2.json"

    artifact = experiment_mod.run_experiment(
        project_root=project_root,
        requested_roadmap=roadmap_path,
        complete_path=complete_path,
        output_path=output_path,
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["n_tasks_audited"] == 2
    assert written["n_prior_failures_missing"] == 1
    assert written["n_gate_upstream_failures"] == 0
    assert written["n_model_agent_coherence_failures"] == 0
    assert written["n_gate_field_cross_ref_failures"] == 0
    assert written["arxiv_task_prior_failures_complete"] is True
    assert written["roadmap_gate_audit_passed"] is False
    assert written["honest_verdict"] == "prior_failures_gaps_found"
    assert written["roadmap_path_used"] == str(roadmap_path)
    assert any("fix_needed:" in detail for detail in written["failure_details"])


def test_exp1152_arxiv_prior_failures_must_include_all_three_ids(tmp_path: Path) -> None:
    """SCENARIO-INFRA-087: exp1153 must declare exp1139, exp1127, and exp1116."""
    project_root = tmp_path
    (project_root / "results").mkdir()
    roadmap_path = _write_yaml(
        project_root / "research-roadmap-next.yaml",
        _roadmap(
            [
                _arxiv_task(
                    [
                        "exp1139-arxiv-final-submission-v3",
                        "exp1127-arxiv-pdf-compilation-final-submission",
                    ]
                )
            ]
        ),
    )
    complete_path = _write_yaml(project_root / "research-complete.yaml", _complete([]))

    artifact = experiment_mod.run_experiment(
        project_root=project_root,
        requested_roadmap=roadmap_path,
        complete_path=complete_path,
        output_path=project_root / "results" / "artifact.json",
    )

    assert artifact["arxiv_task_prior_failures_complete"] is False
    assert artifact["n_prior_failures_missing"] == 1
    assert artifact["honest_verdict"] == "prior_failures_gaps_found"
    assert any(
        "exp1116-arxiv-pdf-compilation-submission" in detail
        for detail in artifact["failure_details"]
    )


def test_exp1152_helper_branches_and_main_entrypoint(tmp_path: Path, monkeypatch, capsys) -> None:
    """REQ-INFRA-075: helper branches provide deterministic Exp 1152 outputs."""
    assert "same roadmap" in experiment_mod._failure_detail_with_fix(
        "GATE_UPSTREAM_EXISTS exp1: missing"
    )
    assert "REQUIRED ARTIFACT FIELDS" in experiment_mod._failure_detail_with_fix(
        "GATE_FIELD_CROSS_REF exp1: missing"
    )
    assert "model=gpt-5.5" in experiment_mod._failure_detail_with_fix(
        "MODEL_AGENT_COHERENCE exp1: bad"
    )
    assert "inspect this audit failure" in experiment_mod._failure_detail_with_fix("UNKNOWN")
    assert experiment_mod._declared_prior_failure_ids(None) == set()
    assert experiment_mod._declared_prior_failure_ids(
        {
            "prior_failures": [
                "exp1",
                {"id": "exp2"},
                {"experiment_id": "exp3"},
                {"experiment_id": ""},
            ]
        }
    ) == {"exp1", "exp2", "exp3"}

    roadmap_path = _write_yaml(
        tmp_path / "research-roadmap.yaml",
        _roadmap([_task("exp1", "Fresh Task")]),
    )
    assert experiment_mod._find_task(roadmap_path, "missing") is None
    arxiv_complete, missing = experiment_mod._arxiv_prior_failure_status(roadmap_path)
    assert arxiv_complete is False
    assert missing == list(experiment_mod.REQUIRED_ARXIV_PRIOR_IDS)
    assert (
        experiment_mod._honest_verdict(
            {
                "n_prior_failures_missing": 0,
                "n_gate_upstream_failures": 1,
                "n_gate_field_cross_ref_failures": 0,
                "n_model_agent_coherence_failures": 0,
            }
        )
        == "gate_field_gaps_found"
    )
    assert (
        experiment_mod._honest_verdict(
            {
                "n_prior_failures_missing": 0,
                "n_gate_upstream_failures": 0,
                "n_gate_field_cross_ref_failures": 0,
                "n_model_agent_coherence_failures": 0,
            }
        )
        == "all_checks_pass"
    )

    monkeypatch.setattr(
        experiment_mod, "run_experiment", lambda: {"honest_verdict": "all_checks_pass"}
    )
    assert experiment_mod.main() == 0
    assert json.loads(capsys.readouterr().out) == {"honest_verdict": "all_checks_pass"}
