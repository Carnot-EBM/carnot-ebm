"""The gate-cascade join fires on exists-and-fails, never on absent (REQ-CONDUCTOR-CASCADE-1).

SCENARIO-CONDUCTOR-CASCADE-1-FIRES: the exp6942 shape (field present, failing) reports.
SCENARIO-CONDUCTOR-CASCADE-1-SILENT: absent upstream / in-flight bootstrap stays silent.
SCENARIO-CONDUCTOR-CASCADE-1-FINAL-MISSING: a final upstream lacking the field reports.

Origin: milestone 608 lost 10 of 12 tasks to one ready_score=0 nobody joined
against the roadmap; the discrimination below was hand-tested on that incident
before being built (known-issues 2026-09-03 21:35Z).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "gate_cascade_check",
    Path(__file__).resolve().parents[2] / "scripts" / "gate_cascade_check.py",
)
gcc = importlib.util.module_from_spec(_SPEC)
sys.modules["gate_cascade_check"] = gcc
_SPEC.loader.exec_module(gcc)


def _roadmap(tmp_path: Path, tasks: list[dict]) -> Path:
    import yaml

    path = tmp_path / "roadmap.yaml"
    path.write_text(yaml.safe_dump({"milestone": "test", "tasks": tasks}))
    return path


def _gate(upstream: str, field: str, value=1) -> dict:
    return {"upstream": upstream, "artifact_field": field, "op": "==", "value": value}


def test_fires_on_present_and_failing_field(tmp_path: Path) -> None:
    """SCENARIO-CONDUCTOR-CASCADE-1-FIRES: the exp6942 shape, three dependents."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_942_preflight.json").write_text(
        json.dumps({"honest_verdict": "blocked_preflight", "ready_score": 0})
    )
    tasks = [
        {"id": f"exp94{i}-downstream", "gated_on": [_gate("exp942-preflight", "ready_score")]}
        for i in (3, 4, 5)
    ]
    findings, notices = gcc.pending_cascades(_roadmap(tmp_path, tasks), results)

    assert findings is not None and len(findings) == 1
    assert "exp942-preflight.ready_score" in findings[0]
    assert "3 task(s)" in findings[0]
    assert any("population:" in n for n in notices)


def test_silent_on_absent_upstream_and_inflight_bootstrap(tmp_path: Path) -> None:
    """SCENARIO-CONDUCTOR-CASCADE-1-SILENT: the 609 shape and the bootstrap shape."""

    results = tmp_path / "results"
    results.mkdir()
    # In-flight upstream: artifact exists, field absent, NOT final.
    (results / "experiment_950_running.json").write_text(
        json.dumps({"status": "bootstrap", "note": "still running"})
    )
    tasks = [
        {"id": "expA", "gated_on": [_gate("exp999-never-ran", "ready_score")]},
        {"id": "expB", "gated_on": [_gate("exp950-running", "ready_score")]},
    ]
    findings, notices = gcc.pending_cascades(_roadmap(tmp_path, tasks), results)

    assert findings == []
    assert any("1 upstream(s) absent" in n for n in notices)


def test_fires_when_final_upstream_lacks_the_field(tmp_path: Path) -> None:
    """SCENARIO-CONDUCTOR-CASCADE-1-FINAL-MISSING: the exp6756 retry-burn shape."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_960_done.json").write_text(
        json.dumps({"honest_verdict": "complete_done_without_the_field"})
    )
    tasks = [{"id": "expC", "gated_on": [_gate("exp960-done", "never_written_score")]}]
    findings, _ = gcc.pending_cascades(_roadmap(tmp_path, tasks), results)

    assert findings is not None and len(findings) == 1
    assert "FINAL upstream" in findings[0]


def test_passing_gate_is_silent(tmp_path: Path) -> None:
    """A satisfied gate is not a cascade, whatever the artifact's verdict says."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_970_ok.json").write_text(
        json.dumps({"honest_verdict": "complete_ok", "ready_score": 1})
    )
    tasks = [{"id": "expD", "gated_on": [_gate("exp970-ok", "ready_score")]}]
    findings, _ = gcc.pending_cascades(_roadmap(tmp_path, tasks), results)

    assert findings == []


def test_unreadable_roadmap_fails_closed(tmp_path: Path) -> None:
    """A missing roadmap returns None (cannot answer), never an empty clean list."""

    findings, notices = gcc.pending_cascades(tmp_path / "missing.yaml", tmp_path)
    assert findings is None
    assert any("unreadable" in n for n in notices)


def test_already_blocked_dependent_is_annotated(tmp_path: Path) -> None:
    """A dependent that already burned its attempts reads differently from a pending one."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_942_preflight.json").write_text(
        json.dumps({"honest_verdict": "blocked_preflight", "ready_score": 0})
    )
    (results / "experiment_943_downstream.json").write_text(
        json.dumps({"honest_verdict": "blocked_gate_check_failed"})
    )
    tasks = [
        {"id": "exp943-downstream", "gated_on": [_gate("exp942-preflight", "ready_score")]},
        {"id": "exp944-downstream", "gated_on": [_gate("exp942-preflight", "ready_score")]},
    ]
    findings, _ = gcc.pending_cascades(_roadmap(tmp_path, tasks), results)

    assert len(findings) == 1
    assert "exp943-downstream(already blocked)" in findings[0]
    assert "exp944-downstream(already blocked)" not in findings[0]


def _chain_tasks() -> list[dict[str, object]]:
    """Rebuild the milestone .623 shape: two direct dependents, five in the chain.

    exp7099 doomed exp7100; exp7100 doomed a provenance task and exp7102; exp7102
    doomed two more. The direct count is 1 and the real loss was 5, which is what
    the dashboard line understated.
    """

    def gate(upstream: str) -> dict[str, object]:
        return {"gated_on": [{"upstream": upstream, "artifact_field": "f", "op": "==", "value": 1}]}

    return [
        {"id": "exp7099"},
        {"id": "exp7100", **gate("exp7099")},
        {"id": "exp7101-provenance", **gate("exp7100")},
        {"id": "exp7102", **gate("exp7100")},
        {"id": "exp7103-ab", **gate("exp7102")},
        {"id": "exp7104-receipt", **gate("exp7102")},
    ]


def test_transitive_dependents_counts_the_whole_chain_not_the_first_hop() -> None:
    """A blocked task is itself an upstream, so depth-1 understates the loss."""

    edges = gcc.dependency_edges(_chain_tasks())
    assert edges["exp7099"] == ["exp7100"]
    reachable = gcc.transitive_dependents(edges, ["exp7099"])
    assert reachable == {
        "exp7100",
        "exp7101-provenance",
        "exp7102",
        "exp7103-ab",
        "exp7104-receipt",
    }
    assert len(reachable) == 5


def test_transitive_dependents_terminates_on_a_cycle() -> None:
    """A hand-edited roadmap can name a cycle; the dashboard must not hang."""

    edges = gcc.dependency_edges(
        [
            {"id": "a", "gated_on": [{"upstream": "b"}]},
            {"id": "b", "gated_on": [{"upstream": "a"}]},
        ]
    )
    assert gcc.transitive_dependents(edges, ["a"]) == {"b"}


def test_pending_cascade_line_reports_the_deeper_tasks(tmp_path: Path) -> None:
    """REQ: the rendered dashboard string carries the chain size, not just hop one."""

    import yaml

    roadmap = tmp_path / "research-roadmap.yaml"
    roadmap.write_text(yaml.safe_dump({"tasks": _chain_tasks()}))
    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_7099_root.json").write_text(
        json.dumps({"experiment_id": "exp7099", "f": 0, "honest_verdict": "complete_null"})
    )
    findings, _notices = gcc.pending_cascades(
        roadmap_path=roadmap, results_dir=results, log_path=tmp_path / "empty-log.md"
    )
    assert findings, "the root gate fails, so a cascade must be reported"
    line = findings[0]
    assert "1 task(s) gate on it, 4 more downstream" in line, line
