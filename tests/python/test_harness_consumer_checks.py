"""Deterministic checks for the defect classes that reached production repeatedly.

Spec: REQ-HARNESS-CONSUMER-1, SCENARIO-HARNESS-CONSUMER-1 through 4
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "harness_consumer_checks.py"


def _module():
    spec = importlib.util.spec_from_file_location("_hcc", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hcc"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_scenario_harness_consumer_1_a_dict_key_is_a_write_not_a_read() -> None:
    """REQ-HARNESS-CONSUMER-1: counting writers as readers is why a first version was wrong.

    It reported sixteen production readers for a field with no live consumer; the sixteen were
    the planner WRITING it, a dead supervisor, and this checker's own docstring.
    """

    mod = _module()
    writes_only = ast.parse('task = {"estimated_wall_time_min": 70}\n')
    assert mod.field_reads(writes_only, "estimated_wall_time_min") == []
    reads = ast.parse(
        'x = task.get("estimated_wall_time_min")\ny = task["estimated_wall_time_min"]\n'
    )
    assert len(mod.field_reads(reads, "estimated_wall_time_min")) == 2


def test_scenario_harness_consumer_1_only_a_runtime_reader_can_act(tmp_path: Path) -> None:
    """REQ-HARNESS-CONSUMER-1: a field read only by analysis code changes nothing."""

    mod = _module()
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "python" / "carnot").mkdir(parents=True)
    (tmp_path / "python" / "carnot" / "analysis_only.py").write_text(
        'v = task.get("some_budget_field")\n'
    )
    r = mod.unread_field("some_budget_field", root=tmp_path)
    assert r["has_runtime_reader"] is False
    assert r["readers"]["analysis"], r
    (tmp_path / "scripts" / "research_conductor.py").write_text(
        'v = task.get("some_budget_field")\n'
    )
    assert mod.unread_field("some_budget_field", root=tmp_path)["has_runtime_reader"] is True


def test_scenario_harness_consumer_2_spec_claimed_guard_with_no_caller(tmp_path: Path) -> None:
    """REQ-HARNESS-CONSUMER-2: the audit_orphan_test_imports shape, spec'd and uncalled."""

    mod = _module()
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "openspec").mkdir(parents=True)
    (tmp_path / "scripts" / "some_thing_lint.py").write_text("# a guard\n")
    (tmp_path / "scripts" / "other_thing_lint.py").write_text("# a guard\n")
    (tmp_path / "scripts" / "caller.py").write_text("import other_thing_lint\n")
    (tmp_path / "openspec" / "spec.md").write_text(
        "| REQ-X | Implemented (`scripts/some_thing_lint.py`) |\n"
        "| REQ-Y | Implemented (`scripts/other_thing_lint.py`) |\n"
    )
    r = mod.uncalled_guards(root=tmp_path)
    assert "scripts/some_thing_lint.py" in r["uncalled"], r
    assert "scripts/other_thing_lint.py" not in r["uncalled"], r


def test_scenario_harness_consumer_2_an_unclaimed_one_off_audit_is_not_flagged(
    tmp_path: Path,
) -> None:
    """REQ-HARNESS-CONSUMER-2: flagging every uncalled audit fires on 27%, which cries wolf."""

    mod = _module()
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "openspec").mkdir(parents=True)
    (tmp_path / "scripts" / "audit_1717.py").write_text("# a one-off historical audit\n")
    (tmp_path / "openspec" / "spec.md").write_text("nothing claimed here\n")
    assert mod.uncalled_guards(root=tmp_path)["uncalled"] == []


def test_scenario_harness_consumer_3_only_a_missing_PARENT_is_invented(tmp_path: Path) -> None:
    """REQ-HARNESS-CONSUMER-3: a missing file in an existing directory is a forward reference.

    Measured over fourteen milestones, keying on the parent separates 19 real hits from 98
    ordinary references to the artifact a task is about to write.
    """

    mod = _module()
    (tmp_path / "results").mkdir()
    text = (
        "prompt: |\n"
        "  - {project_root}/results/experiment_9999_not_yet_written.json\n"
        "  - {project_root}/python/carnot/agents/invented_module.py\n"
    )
    bad = mod.invented_prompt_paths(text, root=tmp_path)
    assert bad == ["python/carnot/agents/invented_module.py"], bad


def test_scenario_harness_consumer_1_cli_exits_nonzero_with_no_runtime_reader(
    tmp_path: Path,
) -> None:
    """REQ-HARNESS-CONSUMER-1: the exit code must carry the finding, not just the printout."""

    import subprocess

    out = subprocess.run(
        [
            str(REPO / ".venv" / "bin" / "python"),
            str(SCRIPT),
            "unread-field",
            "estimated_wall_time_min",
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 1, out.stdout
    assert "NO RUNTIME READER" in out.stdout, out.stdout


def test_scenario_harness_consumer_3_the_dashboard_actually_calls_it() -> None:
    """REQ-HARNESS-CONSUMER-3: the check must be WIRED, or it becomes what it detects.

    An uncalled guard is one of the nineteen defects this work exists to prevent. A check that
    ships without a caller reproduces it, so the call site is asserted from the AST rather than
    trusted.
    """

    dash = (REPO / "scripts" / "outer_loop_dashboard.py").read_text()
    tree = ast.parse(dash)
    render = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "render"
    )
    called = {
        c.func.id
        for c in ast.walk(render)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    }
    assert "invented_path_line" in called, sorted(called)
