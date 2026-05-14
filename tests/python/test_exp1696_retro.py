"""Tests for the Milestone .170 retrospective artifact.

Spec coverage: REQ-RETRO-1696 (milestone retrospective schema validation)
"""

from __future__ import annotations

import json
from pathlib import Path

RETRO_PATH = Path(__file__).parents[2] / "results" / "experiment_1696_retro.json"

REQUIRED_FIELDS = [
    "schema",
    "milestone",
    "tasks_summary",
    "gates_passed_count",
    "gates_failed_count",
    "actual_agent_backend_distribution",
    "honest_verdict",
]

TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


def _load() -> dict:
    return json.loads(RETRO_PATH.read_text())


def test_artifact_exists():
    # REQ-RETRO-1696: the retro artifact must exist
    assert RETRO_PATH.exists(), f"Missing: {RETRO_PATH}"


def test_required_fields_present():
    # REQ-RETRO-1696: all REQUIRED ARTIFACT FIELDS must be present
    data = _load()
    for field in REQUIRED_FIELDS:
        assert field in data, f"Missing required field: {field}"


def test_milestone_label():
    # REQ-RETRO-1696: milestone must reference .170
    data = _load()
    assert "170" in data["milestone"], "milestone field must reference .170"


def test_gates_counts_consistent():
    # REQ-RETRO-1696: passed + failed must equal task_count
    data = _load()
    passed = data["gates_passed_count"]
    failed = data["gates_failed_count"]
    tasks = data["tasks_summary"]
    assert passed + failed == len(tasks), (
        f"gates_passed_count({passed}) + gates_failed_count({failed}) "
        f"!= len(tasks_summary)({len(tasks)})"
    )


def test_tasks_summary_nonempty():
    # REQ-RETRO-1696: at least one task must be summarised
    data = _load()
    assert len(data["tasks_summary"]) > 0


def test_honest_verdict_has_terminal_prefix():
    # REQ-RETRO-1696 / Verdict Terminal-Prefix Discipline (CLAUDE.md)
    data = _load()
    verdict = data["honest_verdict"]
    assert any(verdict.startswith(p) for p in TERMINAL_PREFIXES), (
        f"honest_verdict must start with a terminal prefix; got: {verdict!r}"
    )


def test_agent_distribution_nonempty():
    # REQ-RETRO-1696: actual_agent_backend_distribution must list at least one backend
    data = _load()
    dist = data["actual_agent_backend_distribution"]
    assert isinstance(dist, dict) and len(dist) > 0


def test_each_task_has_gate_passed_field():
    # REQ-RETRO-1696: every task_summary entry must declare gate_passed
    data = _load()
    for task in data["tasks_summary"]:
        assert "gate_passed" in task, f"Task {task.get('id')} missing gate_passed"


def test_gate_passed_ids_match_count():
    # REQ-RETRO-1696: ids listed in gates_passed_ids match gates_passed_count
    data = _load()
    if "gates_passed_ids" in data:
        assert len(data["gates_passed_ids"]) == data["gates_passed_count"]
    if "gates_failed_ids" in data:
        assert len(data["gates_failed_ids"]) == data["gates_failed_count"]
