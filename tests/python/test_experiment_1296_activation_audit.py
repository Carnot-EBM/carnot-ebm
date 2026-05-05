"""Tests for the Exp 1296 prior-failures activation audit.

Spec: REQ-INFRA-1296, SCENARIO-INFRA-1296, SCENARIO-INFRA-1296-BLOCKED
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_1296_prior_failures_activation_audit.py"


def _load_module():
    """Load the standalone experiment runner without requiring scripts/ packaging."""
    spec = importlib.util.spec_from_file_location(
        "experiment_1296_prior_failures_activation_audit", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_1296_prior_failures_activation_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _roadmap(tasks: list[dict]) -> dict:
    return {
        "milestone": "2026.04.101",
        "milestone_title": "Synthetic .101",
        "milestone_doc": "openspec/change-proposals/synthetic.md",
        "tasks": tasks,
    }


def _task(task_id: str, title: str, **overrides) -> dict:
    base = {
        "id": task_id,
        "milestone": "2026.04.101",
        "deliverable": f"results/{task_id}.json",
        "title": title,
        "prompt": "REQUIRED ARTIFACT FIELDS:\n- audit_ok\n",
    }
    base.update(overrides)
    return base


def _complete(tasks: list[dict]) -> dict:
    return {"milestones": [{"id": "2026.04.100", "tasks": tasks}]}


def _seed_source_artifacts(project_root: Path, *, grammar: bool = True, memory: bool = True) -> None:
    results = project_root / "results"
    results.mkdir()
    _write_json(
        results / "experiment_1283_certificate_grammar_backend_bakeoff.json",
        {"status": "complete", "grammar_backend_available": grammar},
    )
    _write_json(
        results / "experiment_1288_interwhen_dvi_verifier_feedback_replay.json",
        {"status": "complete", "memory_update_written": memory},
    )


def test_exp1296_passes_on_clean_active_roadmap_fallback(tmp_path: Path) -> None:
    """REQ-INFRA-1296 / SCENARIO-INFRA-1296: clean active .101 fallback passes."""
    mod = _load_module()
    project_root = tmp_path
    _seed_source_artifacts(project_root)
    active_roadmap = _write_yaml(
        project_root / "research-roadmap.yaml",
        _roadmap(
            [
                _task(
                    "exp1296-prior-failures-activation-audit",
                    "Fresh Activation Audit",
                    prompt=(
                        "REQUIRED ARTIFACT FIELDS:\n"
                        "- prior_failures_coverage_ok\n"
                        "- exp1283_grammar_backend_available\n"
                    ),
                ),
                _task(
                    "exp1297-cache-preflight",
                    "Fresh Cache Preflight",
                    gated_on=[
                        {
                            "upstream": "exp1296-prior-failures-activation-audit",
                            "artifact_field": "prior_failures_coverage_ok",
                            "op": "==",
                            "value": True,
                        }
                    ],
                ),
            ]
        ),
    )
    _write_yaml(
        project_root / "research-complete.yaml",
        _complete([{"id": "exp0-old", "title": "Completely Unrelated Finished Work"}]),
    )

    artifact = mod.run_experiment(project_root=project_root, run_date="20260505")
    written = json.loads(
        (project_root / "results" / "experiment_1296_prior_failures_activation_audit.json")
        .read_text(encoding="utf-8")
    )

    assert artifact == written
    assert written["status"] == "complete"
    assert written["run_date"] == "20260505"
    assert written["prior_failures_coverage_ok"] is True
    assert written["roadmap_gate_audit_passed"] is True
    assert written["exp1283_grammar_backend_available"] is True
    assert written["exp1288_memory_update_written"] is True
    assert written["n_prior_failures_missing"] == 0
    assert written["activation_blockers"] == []
    assert written["honest_verdict"] == "activation_audit_passed"
    assert written["roadmap_path_requested"].endswith("research-roadmap-next.yaml")
    assert written["roadmap_path_used"] == str(active_roadmap)
    assert "active research-roadmap.yaml" in written["roadmap_path_note"]


def test_exp1296_blocks_with_exact_prior_and_gate_field_details(tmp_path: Path) -> None:
    """SCENARIO-INFRA-1296-BLOCKED: blockers name task ids and fields."""
    mod = _load_module()
    project_root = tmp_path
    _seed_source_artifacts(project_root)
    _write_yaml(
        project_root / "research-roadmap-next.yaml",
        _roadmap(
            [
                _task(
                    "exp1296-old-scope",
                    "WOPR Slitherlink Puzzle Cartridge Rescue",
                    prompt="REQUIRED ARTIFACT FIELDS:\n- present_field\n",
                ),
                _task(
                    "exp1297-downstream",
                    "Fresh Downstream",
                    gated_on=[
                        {
                            "upstream": "exp1296-old-scope",
                            "artifact_field": "missing_field",
                            "op": "==",
                            "value": True,
                        }
                    ],
                ),
            ]
        ),
    )
    _write_yaml(
        project_root / "research-complete.yaml",
        _complete(
            [
                {
                    "id": "exp1136-wopr-slitherlink-cartridge",
                    "title": "WOPR Slitherlink Puzzle Cartridge",
                }
            ]
        ),
    )

    artifact = mod.run_experiment(project_root=project_root, run_date="20260505")

    assert artifact["status"] == "complete"
    assert artifact["prior_failures_coverage_ok"] is False
    assert artifact["roadmap_gate_audit_passed"] is False
    assert artifact["n_prior_failures_missing"] == 1
    assert artifact["n_gate_upstream_failures"] == 0
    assert artifact["n_gate_field_cross_ref_failures"] == 1
    assert artifact["honest_verdict"] == "activation_audit_blocked"
    assert any(
        blocker["task_id"] == "exp1296-old-scope" and blocker["field"] == "prior_failures"
        for blocker in artifact["activation_blockers"]
    )
    assert any(
        blocker["task_id"] == "exp1297-downstream" and blocker["field"] == "missing_field"
        for blocker in artifact["activation_blockers"]
    )


def test_exp1296_helper_branches_are_deterministic(tmp_path: Path) -> None:
    """REQ-INFRA-1296: parser helpers and missing source booleans are explicit."""
    mod = _load_module()

    schema_blocker = mod._blocker_from_schema_error("Schema error at tasks -> 0: broken")
    assert schema_blocker["source"] == "validate_prior_failures"
    assert schema_blocker["field"] == "tasks -> 0"
    assert mod._blocker_from_schema_error("File not found: roadmap")["field"] == "roadmap_path"

    upstream_blocker = mod._blocker_from_gate_detail(
        "GATE_UPSTREAM_EXISTS exp1: gated_on upstream exp0 is not in roadmap"
    )
    assert upstream_blocker["task_id"] == "exp1"
    assert upstream_blocker["field"] == "gated_on[].upstream"
    model_blocker = mod._blocker_from_gate_detail("MODEL_AGENT_COHERENCE exp2: bad model")
    assert model_blocker["field"] == "agent_type/model"
    assert mod._blocker_from_gate_detail("UNKNOWN exp3: details")["field"] == "unknown"

    assert mod._artifact_flag(tmp_path / "missing.json", "flag") is False
    assert mod._artifact_flag(_write_json(tmp_path / "source.json", {"flag": False}), "flag") is False


def test_exp1296_main_prints_terminal_artifact(monkeypatch, capsys) -> None:
    """REQ-INFRA-1296: CLI entry point prints the terminal artifact JSON."""
    mod = _load_module()
    monkeypatch.setattr(
        mod,
        "run_experiment",
        lambda: {"honest_verdict": "activation_audit_passed"},
    )

    assert mod.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "honest_verdict": "activation_audit_passed"
    }
