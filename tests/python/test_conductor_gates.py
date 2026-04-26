"""Tests for the pre-gate check + max_turns hint in the research conductor.

The pre-gate check evaluates `gated_on:` declarations from
research-roadmap.yaml against the upstream experiment's artifact JSON,
returning a structured GateCheckResult. When any gate fails, the
conductor writes a blocked artifact directly and skips the 5-9 min
Sonnet research-step call — saving wall time and API quota on iterations
that would have written a blocked artifact anyway.

The max_turns hint reads an optional `max_turns:` field from the task
YAML and clamps it to a safe range so a malformed entry can't trigger
runaway agent costs.
Spec: REQ-INFRA-072
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from conductor_gates import (  # noqa: E402
    GateCheckResult,
    GateResult,
    evaluate_gates,
    select_max_turns,
    write_blocked_artifact,
)

# ---------------------------------------------------------------------------
# evaluate_gates — operator coverage
# ---------------------------------------------------------------------------


def _seed_artifact(results_dir: Path, exp_num: int, slug: str, data: dict) -> None:
    """Helper: write a minimal artifact JSON for an upstream task."""
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / f"experiment_{exp_num}_{slug}.json"
    target.write_text(json.dumps(data))


def test_no_gates_declared_passes_vacuously(tmp_path):
    """A task with no `gated_on` field is treated as ungated."""
    task = {"id": "exp900-demo", "title": "Exp 900: Demo"}
    result = evaluate_gates(task, results_dir=tmp_path)
    assert result.passed is True
    assert "no gates" in result.summary
    assert result.gates_evaluated == []


def test_single_gate_passes_with_gt_operator(tmp_path):
    """A `>` gate passes when the upstream artifact's field exceeds the threshold."""
    _seed_artifact(tmp_path, 819, "field_fix", {"discrimination_rate": 1.0})
    task = {
        "id": "exp900-demo",
        "title": "Exp 900",
        "gated_on": [
            {
                "upstream": "exp819-field-fix",
                "artifact_field": "discrimination_rate",
                "op": ">",
                "value": 0.5,
            },
        ],
    }
    result = evaluate_gates(task, results_dir=tmp_path)
    assert result.passed is True
    assert len(result.gates_evaluated) == 1
    assert result.gates_evaluated[0].passed is True


def test_single_gate_fails_with_eq_zero(tmp_path):
    """A `> 0` gate fails when the upstream's field is exactly 0 (real .63 case)."""
    _seed_artifact(tmp_path, 821, "constraint_v2", {"delta_overall": 0.0})
    task = {
        "id": "exp823-relay",
        "title": "Exp 823",
        "gated_on": [
            {
                "upstream": "exp821-constraint-v2",
                "artifact_field": "delta_overall",
                "op": ">",
                "value": 0.0,
            },
        ],
    }
    result = evaluate_gates(task, results_dir=tmp_path)
    assert result.passed is False
    assert result.gates_evaluated[0].passed is False
    assert "0" in result.gates_evaluated[0].reason


def test_in_operator_against_verdict_list(tmp_path):
    """An `in` gate passes when the upstream's verdict is in the allowed set."""
    _seed_artifact(tmp_path, 819, "field_fix", {"honest_verdict": "injection_field_fixed"})
    task = {
        "id": "exp900-demo",
        "title": "Exp 900",
        "gated_on": [
            {
                "upstream": "exp819-field-fix",
                "artifact_field": "honest_verdict",
                "op": "in",
                "value": ["injection_field_fixed", "discrimination_above_baseline"],
            },
        ],
    }
    result = evaluate_gates(task, results_dir=tmp_path)
    assert result.passed is True


def test_in_operator_rejects_when_expected_not_a_list(tmp_path):
    """`in` requires a list/tuple/set on the right; a string fails defensively."""
    _seed_artifact(tmp_path, 819, "field_fix", {"honest_verdict": "ok"})
    task = {
        "id": "exp900-demo",
        "title": "Exp 900",
        "gated_on": [
            {
                "upstream": "exp819-field-fix",
                "artifact_field": "honest_verdict",
                "op": "in",
                "value": "ok",
            },  # string, not a list — malformed
        ],
    }
    result = evaluate_gates(task, results_dir=tmp_path)
    assert result.passed is False
    assert "list" in result.gates_evaluated[0].reason or "tuple" in result.gates_evaluated[0].reason


def test_contains_operator(tmp_path):
    """`contains` checks whether the value is a substring of the field."""
    _seed_artifact(tmp_path, 819, "field_fix", {"honest_verdict": "retro_028_closed"})
    task = {
        "id": "exp900-demo",
        "title": "Exp 900",
        "gated_on": [
            {
                "upstream": "exp819-field-fix",
                "artifact_field": "honest_verdict",
                "op": "contains",
                "value": "closed",
            },
        ],
    }
    result = evaluate_gates(task, results_dir=tmp_path)
    assert result.passed is True


def test_numeric_op_with_none_actual_fails(tmp_path):
    """Comparing None numerically must fail rather than raise TypeError."""
    _seed_artifact(tmp_path, 819, "field_fix", {"discrimination_rate": None})
    task = {
        "id": "exp900-demo",
        "title": "Exp 900",
        "gated_on": [
            {
                "upstream": "exp819-field-fix",
                "artifact_field": "discrimination_rate",
                "op": ">",
                "value": 0.5,
            },
        ],
    }
    result = evaluate_gates(task, results_dir=tmp_path)
    assert result.passed is False
    assert "None" in result.gates_evaluated[0].reason


def test_unknown_op_fails_safely(tmp_path):
    """An unrecognised operator fails with a descriptive reason — never raises."""
    _seed_artifact(tmp_path, 819, "field_fix", {"x": 1})
    task = {
        "id": "exp900-demo",
        "title": "Exp 900",
        "gated_on": [
            {"upstream": "exp819-field-fix", "artifact_field": "x", "op": "<<<", "value": 1},
        ],
    }
    result = evaluate_gates(task, results_dir=tmp_path)
    assert result.passed is False
    assert "unknown op" in result.gates_evaluated[0].reason


# ---------------------------------------------------------------------------
# evaluate_gates — multi-gate semantics + missing artifact
# ---------------------------------------------------------------------------


def test_all_gates_must_pass(tmp_path):
    """If any gate fails the task is blocked — gate logic is conjunctive (AND)."""
    _seed_artifact(tmp_path, 819, "fix_a", {"verdict": "ok"})
    _seed_artifact(tmp_path, 820, "fix_b", {"verdict": "broken"})
    task = {
        "id": "exp900-demo",
        "title": "Exp 900",
        "gated_on": [
            {"upstream": "exp819-fix-a", "artifact_field": "verdict", "op": "==", "value": "ok"},
            {"upstream": "exp820-fix-b", "artifact_field": "verdict", "op": "==", "value": "ok"},
        ],
    }
    result = evaluate_gates(task, results_dir=tmp_path)
    assert result.passed is False
    # First gate passed; second failed
    assert result.gates_evaluated[0].passed is True
    assert result.gates_evaluated[1].passed is False
    # Summary names which gate failed first
    assert "exp820-fix-b" in result.summary


def test_missing_upstream_artifact_fails_safely(tmp_path):
    """Upstream artifact missing → gate fails with a clear reason."""
    task = {
        "id": "exp900-demo",
        "title": "Exp 900",
        "gated_on": [
            {"upstream": "exp819-not-yet-run", "artifact_field": "x", "op": "==", "value": 1},
        ],
    }
    result = evaluate_gates(task, results_dir=tmp_path)
    assert result.passed is False
    assert "not found" in result.gates_evaluated[0].reason


def test_corrupt_upstream_artifact_fails_safely(tmp_path):
    """Upstream artifact with invalid JSON → gate fails, no exception."""
    (tmp_path / "experiment_819_corrupt.json").write_text("{not json")
    task = {
        "id": "exp900-demo",
        "title": "Exp 900",
        "gated_on": [
            {"upstream": "exp819-corrupt", "artifact_field": "x", "op": "==", "value": 1},
        ],
    }
    result = evaluate_gates(task, results_dir=tmp_path)
    assert result.passed is False
    assert "unreadable" in result.gates_evaluated[0].reason


# ---------------------------------------------------------------------------
# write_blocked_artifact — produces a valid downstream-parseable artifact
# ---------------------------------------------------------------------------


def test_write_blocked_artifact_satisfies_required_fields(tmp_path):
    """The blocked artifact contains every REQUIRED_RESULT_FIELDS key.

    Mirrors REQUIRED_RESULT_FIELDS in scripts/experiment_template.py:153.
    Downstream tooling — including the in-process doc reconciler — relies
    on these fields being present.
    """
    task = {"id": "exp900-some-task", "title": "Exp 900: Some Task"}
    gate_check = GateCheckResult(
        passed=False,
        gates_evaluated=[
            GateResult(
                upstream="exp819-fix",
                artifact_field="x",
                op=">",
                expected=0.5,
                actual=0.0,
                passed=False,
                reason="0.0 not > 0.5",
            )
        ],
        summary="1 of 1 gate(s) failed",
    )
    path = write_blocked_artifact(task, gate_check, results_dir=tmp_path)
    assert path is not None
    data = json.loads(path.read_text())
    for field in (
        "experiment",
        "schema",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "title",
    ):
        assert field in data, f"missing required field {field!r}"
    assert data["status"] == "blocked"
    assert data["honest_verdict"] == "blocked_gate_check_failed"
    assert data["blocked_at_layer"] == "conductor_pre_gate"
    # Gate evaluation captured for the reader's diagnostic benefit
    assert len(data["gates_evaluated"]) == 1


def test_write_blocked_artifact_filename_matches_task_id(tmp_path):
    """Filename is derived deterministically from the task id."""
    task = {"id": "exp823-fr11-tier1-live-relay-v2", "title": "Exp 823"}
    gate_check = GateCheckResult(passed=False, gates_evaluated=[], summary="")
    path = write_blocked_artifact(task, gate_check, results_dir=tmp_path)
    assert path is not None
    assert path.name == "experiment_823_fr11_tier1_live_relay_v2.json"


def test_write_blocked_artifact_returns_none_for_unparseable_id(tmp_path):
    """If the task id doesn't match expNNN-..., return None — caller falls back."""
    task = {"id": "not-a-task-id", "title": "?"}
    gate_check = GateCheckResult(passed=False, gates_evaluated=[], summary="")
    assert write_blocked_artifact(task, gate_check, results_dir=tmp_path) is None


# ---------------------------------------------------------------------------
# select_max_turns
# ---------------------------------------------------------------------------


def test_select_max_turns_default_when_absent():
    """Default 50 mirrors the historical hard-coded value."""
    assert select_max_turns({}) == 50


def test_select_max_turns_picks_yaml_value():
    """A small int from YAML is honoured when in bounds."""
    assert select_max_turns({"max_turns": 20}) == 20


def test_select_max_turns_clamps_negative():
    """Negative or zero → fall back to default rather than letting the agent fail."""
    assert select_max_turns({"max_turns": 0}) == 50
    assert select_max_turns({"max_turns": -10}) == 50


def test_select_max_turns_clamps_excessive():
    """Values above 100 fall back — bounds protect against runaway costs."""
    assert select_max_turns({"max_turns": 500}) == 50


def test_select_max_turns_rejects_non_int():
    """A string or float in YAML is rejected rather than coerced silently."""
    assert select_max_turns({"max_turns": "twenty"}) == 50
    assert select_max_turns({"max_turns": 20.5}) == 50


def test_select_max_turns_custom_default():
    """Caller-supplied default flows through."""
    assert select_max_turns({}, default=30) == 30


# ---------------------------------------------------------------------------
# Integration smoke — pre-gate behaves correctly on the real .63 shape
# ---------------------------------------------------------------------------


def test_real_shape_82x_cascade(tmp_path):
    """Reproduces the .63 cascade where Exp 821 zero-delta blocks Exp 823.

    Without a pre-gate check, the conductor wastes ~7 min in Sonnet on
    Exp 823 only to discover the gate inside the experiment script. With
    a pre-gate check declared in the YAML, the same finding is reached
    in ~50ms.
    """
    # Exp 819 closed cleanly
    _seed_artifact(
        tmp_path,
        819,
        "field_fix",
        {
            "honest_verdict": "injection_field_fixed",
            "discrimination_rate": 1.0,
        },
    )
    # Exp 821 ran but exposed the Layer-2 problem (zero delta)
    _seed_artifact(
        tmp_path,
        821,
        "constraint_v2",
        {
            "honest_verdict": "constraint_addition_no_delta_live",
            "delta_overall": 0.0,
        },
    )

    # Exp 823 declares two gates: 819 must show discrimination_rate > 0.5,
    # AND 821 must show delta_overall > 0.0
    task_823 = {
        "id": "exp823-fr11-tier1-relay-v2",
        "title": "Exp 823: FR-11 Tier 1 Relay v2",
        "gated_on": [
            {
                "upstream": "exp819-field-fix",
                "artifact_field": "discrimination_rate",
                "op": ">",
                "value": 0.5,
            },
            {
                "upstream": "exp821-constraint-v2",
                "artifact_field": "delta_overall",
                "op": ">",
                "value": 0.0,
            },
        ],
    }
    result = evaluate_gates(task_823, results_dir=tmp_path)
    assert result.passed is False
    # First gate passed (819 cleared discrimination), second failed (821 zero delta)
    assert result.gates_evaluated[0].passed is True
    assert result.gates_evaluated[1].passed is False
    assert "delta_overall" in result.gates_evaluated[1].artifact_field
