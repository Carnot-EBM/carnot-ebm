"""REQ-REPORT-8705: reuse repaired immutable primitives, never historical readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import arc_censoring_decision_8691 as _original
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design

original = _original
Json = dict[str, Any]
ROOT = original.ROOT
NAME = "experiment_8705_v746_arc_decision_qualification"
TASK, MILESTONE = "exp8705-arc-decision-qualification", "2026.11.746"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
TESTS = [original.TESTS[0], "tests/python/test_arc_decision_qualification_8705.py"]
OWNED = original.OWNED + [
    "python/carnot/reporting/arc_decision_qualification_8705.py",
    "python/carnot/reporting/arc_decision_qualification_runner_8705.py",
    CLI,
]
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
TASKS_PIN = "43132b7a0dad4165413d7bcce82bdc9aa5602d17fc1a5144a960edb9d01c3c97"
reference, read, checksum = original.reference, original.read, original.checksum
atomic_json, sha256_file = original.atomic_json, original.sha256_file


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts let the parent distinguish bounded reading from a stalled child."""
    print(f"[exp8705] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(root: Path) -> Json:
    """An independent digest binds full prompts and lineage as well as visible columns."""
    text = (root / DESIGN).read_text()
    shown, tasks = parse_design(text, milestone=MILESTONE)
    actual = yaml.safe_load((root / "research-roadmap.yaml").read_bytes())
    comparison = compare_contract(text, actual, actual, milestone=MILESTONE, first_id=8695)
    envelope = json.loads(
        text.split("<!-- V746_TASK_CONTRACT_START -->", 1)[1]
        .split("```json", 1)[1]
        .split("```", 1)[0]
    )
    ready = (
        comparison["passed"]
        and actual["tasks"] == tasks
        and canonical_hash(tasks) == "sha256:" + TASKS_PIN
        and envelope["canonical_tasks_sha256"] == TASKS_PIN
        and f"Canonical task SHA256: `{TASKS_PIN}`" in text
        and "<!-- V746_TASK_CONTRACT_END -->" in text
    )
    if not ready:
        raise ValueError("v746_full_object_authority")
    return dict(
        ready=True,
        canonical_tasks_sha256=TASKS_PIN,
        visible_table=shown,
        tasks=tasks,
        task=tasks[10],
        comparison=comparison,
    )


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """The original report primitive keeps every decision assertion; identity is invocation-local."""
    with patch.multiple(original, NAME=NAME, TASK=TASK):
        value = original.build(work, receipts, output)
    production = all(
        any(r["name"] == name and r["passed"] for r in receipts)
        for name in ("original_disjoint_union", "preserved_raw_child_sqlite", "coverage_report")
    )
    ready = int(value["arc_decision_ready_score"] and production)
    value.update(
        experiment_id=8705,
        milestone=MILESTONE,
        random_seed=8705,
        honest_verdict=f"complete_{value['verdict_class']}_arc_decision_qualification"
        if value["verdict_class"] != "blocked"
        else "complete_blocked_original_inputs",
        arc_decision_ready_score=ready,
        branch_gate_values=dict(arc_decision_ready_score=ready),
        authority=work["authority"],
        registered_methods=dict(
            games=4, pairs=8, episodes=16, horizon=120, new_episodes=0, supervisor_refinement=True
        ),
        diagnostic_closed=bool(not work["failures"] and value["completed_count"] == 16),
        qualification_scope="production" if production else "private_control_only",
        original_owned_test_file=original.TESTS[0],
    )
    value["acceptance_gates"].update(production_receipts=production, arc_decision_ready_score=ready)
    value["historical_quarantines"].update(
        exp8681_flagged_adversarial=True, exp8691_flagged_adversarial=True
    )
    value["field_principles"] = {
        k: "Exact original operands, independent V746 task seal and fresh owned receipts bind this field; private controls grant no readiness, censored evidence grants no policy benefit or solve credit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return dict(value)
