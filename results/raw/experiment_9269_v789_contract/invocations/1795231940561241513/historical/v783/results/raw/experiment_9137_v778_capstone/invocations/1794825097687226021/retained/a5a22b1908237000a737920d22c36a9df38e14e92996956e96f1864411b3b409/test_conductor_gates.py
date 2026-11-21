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
    _coerce_gate_value,
    _eval_op,
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


def test_exists_operator_passes_when_field_present(tmp_path):
    """`exists` (a schema-valid op that the runtime previously rejected as 'unknown op',
    blocking the .365 ARC capstone exp3951) passes when the upstream produced the field."""
    _seed_artifact(tmp_path, 3946, "r11l-first-solve", {"honest_verdict": "complete: r11l_solved"})
    task = {
        "id": "exp3951-capstone",
        "title": "Capstone gated on the r11l solve having landed",
        "gated_on": [
            {
                "upstream": "exp3946-r11l-first-solve",
                "artifact_field": "honest_verdict",
                "op": "exists",
                "value": True,
            },
        ],
    }
    result = evaluate_gates(task, results_dir=tmp_path)
    assert result.passed is True


def test_exists_operator_fails_when_field_missing(tmp_path):
    """`exists` rejects when the upstream artifact is missing the gated field (or it is null)."""
    _seed_artifact(tmp_path, 3946, "r11l-first-solve", {"some_other_field": 1})
    task = {
        "id": "exp3951-capstone",
        "title": "Capstone gated on a field the upstream did not produce",
        "gated_on": [
            {
                "upstream": "exp3946-r11l-first-solve",
                "artifact_field": "honest_verdict",
                "op": "exists",
                "value": True,
            },
        ],
    }
    result = evaluate_gates(task, results_dir=tmp_path)
    assert result.passed is False
    assert "absent" in result.gates_evaluated[0].reason


def test_exists_value_false_means_should_not_exist(tmp_path):
    """`exists` with value False is an inline not-exists: passes only when the field is absent."""
    _seed_artifact(tmp_path, 3946, "r11l-first-solve", {"honest_verdict": "complete: x"})
    task = {
        "id": "exp-demo",
        "title": "demo",
        "gated_on": [
            {
                "upstream": "exp3946-r11l-first-solve",
                "artifact_field": "honest_verdict",
                "op": "exists",
                "value": False,
            }
        ],
    }
    assert evaluate_gates(task, results_dir=tmp_path).passed is False


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


def test_scenario_ops_recurring_gate_6425_blocked_artifact_diagnostic_contract(tmp_path):
    """SCENARIO-OPS-RECURRING-GATE-6425-DIAGNOSTIC-CONTRACT.

    A reader must not need to parse a prose summary to find the failed gate.
    The blocked artifact exposes the first failed upstream, field, operator,
    expected value, observed value, observed type, and evidence path.
    """
    _seed_artifact(tmp_path, 819, "fix", {"ready_score": 0.0})
    task = {
        "id": "exp900-some-task",
        "title": "Exp 900: Some Task",
        "gated_on": [
            {
                "upstream": "exp819-fix",
                "artifact_field": "ready_score",
                "op": "==",
                "value": 1.0,
            }
        ],
    }

    gate_check = evaluate_gates(task, results_dir=tmp_path)
    path = write_blocked_artifact(task, gate_check, results_dir=tmp_path)
    assert path is not None
    data = json.loads(path.read_text())

    assert data["blocked_reason"] == "actual=0.0 == expected=1.0"
    assert data["failed_upstream"] == "exp819-fix"
    assert data["failed_field"] == "ready_score"
    assert data["failed_operator"] == "=="
    assert data["failed_expected"] == 1.0
    assert data["failed_observed"] == 0.0
    assert data["failed_observed_type"] == "float"
    assert data["failed_evidence_path"].endswith("experiment_819_fix.json")
    assert data["blocked_diagnostic_contract"]["version"] == "blocked_gate_diagnostic_v1"


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


def test_scenario_ops_recurring_gate_6425_nan_numeric_gate_fails_closed():
    """SCENARIO-OPS-RECURRING-GATE-6425-MUTATIONS-FAIL-CLOSED."""
    passed, reason = _eval_op(float("nan"), ">", 0.0)

    assert passed is False
    assert "NaN" in reason
    assert "numeric comparison rejected" in reason


# ---------------------------------------------------------------------------
# select_max_turns
# ---------------------------------------------------------------------------


def test_select_max_turns_default_when_absent():
    """Default raised 50→100 on 2026-04-28 after .80 cascade evidence:
    Exps 1028 / 1029 both hit the 50-turn cap on tasks that were
    genuinely making progress. 100 is the upper bound enforced by
    select_max_turns. See scripts/conductor_gates.py docstring."""
    assert select_max_turns({}) == 100


def test_select_max_turns_picks_yaml_value():
    """A small int from YAML is honoured when in bounds."""
    assert select_max_turns({"max_turns": 20}) == 20


def test_select_max_turns_clamps_negative():
    """Negative or zero → fall back to default (100) rather than letting the agent fail."""
    assert select_max_turns({"max_turns": 0}) == 100
    assert select_max_turns({"max_turns": -10}) == 100


def test_select_max_turns_clamps_excessive():
    """Values above 100 fall back to default — bounds protect against runaway costs."""
    assert select_max_turns({"max_turns": 500}) == 100


def test_select_max_turns_rejects_non_int():
    """A string or float in YAML is rejected rather than coerced silently."""
    assert select_max_turns({"max_turns": "twenty"}) == 100
    assert select_max_turns({"max_turns": 20.5}) == 100


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


# ---------------------------------------------------------------------------
# _coerce_gate_value + bool-aware == / != — the .80 wedge fix.
# Regression guard: exp1030 GATE_BLOCKed three times in milestone .80
# because the YAML expected `True` (Python bool) while the upstream artifact
# at one point carried the string "True" (or vice versa). These tests pin
# the bool-coercion behavior so the wedge cannot recur silently.
# ---------------------------------------------------------------------------


def test_coerce_gate_value_passes_bools_unchanged():
    assert _coerce_gate_value(True) is True
    assert _coerce_gate_value(False) is False


def test_coerce_gate_value_truthy_strings_become_true():
    for s in ("True", "true", "TRUE", "1", "yes", "y", "on"):
        assert _coerce_gate_value(s) is True, f"failed for {s!r}"


def test_coerce_gate_value_falsy_strings_become_false():
    for s in ("False", "false", "FALSE", "0", "no", "n", "off"):
        assert _coerce_gate_value(s) is False, f"failed for {s!r}"


def test_coerce_gate_value_non_bool_strings_passthrough():
    # Verdict strings, hex ids, free-form text must NOT be silently coerced.
    for s in ("preflight_complete", "no_improvement", "exp_823", "running"):
        assert _coerce_gate_value(s) == s


def test_coerce_gate_value_numbers():
    assert _coerce_gate_value(0) is False
    assert _coerce_gate_value(1) is True
    assert _coerce_gate_value(0.0) is False
    assert _coerce_gate_value(0.7) is True


def test_eval_op_eq_string_true_against_bool_true_passes():
    """The .80 wedge: actual='True' (string) vs expected=True (bool)."""
    passed, _ = _eval_op("True", "==", True)
    assert passed is True


def test_eval_op_eq_bool_true_against_string_true_passes():
    """Symmetric direction: actual=True vs expected='True'."""
    passed, _ = _eval_op(True, "==", "True")
    assert passed is True


def test_eval_op_eq_string_false_against_bool_false_passes():
    passed, _ = _eval_op("False", "==", False)
    assert passed is True


def test_eval_op_eq_string_one_against_bool_true_passes():
    passed, _ = _eval_op("1", "==", True)
    assert passed is True


def test_eval_op_neq_handles_bool_coercion():
    """!= must also coerce so the symmetric case stays consistent."""
    passed, _ = _eval_op("True", "!=", True)
    assert passed is False  # string True does NOT differ from bool True


def test_eval_op_eq_strings_unaffected():
    """No-bool-on-either-side: behavior is unchanged for string equality."""
    passed, _ = _eval_op("preflight_complete", "==", "preflight_complete")
    assert passed is True
    passed, _ = _eval_op("running", "==", "success")
    assert passed is False


def test_evaluate_gates_string_true_against_bool_true_passes(tmp_path):
    """End-to-end: gate succeeds even when artifact stores 'True' as string."""
    _seed_artifact(tmp_path, 1028, "preflight_v30", {"pre_test_fixed": "True"})
    task = {
        "id": "exp1030-triple-integration",
        "gated_on": [
            {
                "upstream": "exp1028-preflight-v30",
                "artifact_field": "pre_test_fixed",
                "op": "==",
                "value": True,
            }
        ],
    }
    result = evaluate_gates(task, results_dir=tmp_path)
    assert result.passed is True, result.summary


# ---------------------------------------------------------------------------
# REQ-OPS-GATE-NULL-DIAGNOSTIC-6266: say WHY a gated field read None.
#
# Milestone .539 cascade-blocked four tasks on exp6228. Every block recorded the same message:
# `actual=None == expected=1`. True, and useless -- it does not distinguish "the upstream forgot
# the field" from "the upstream wrote it as null" from "the upstream produced nothing", and the
# three have different fixes. Nobody could act on it, so the blocks were never examined and the
# recurring-blocker ledger counted them as four more anonymous `blocked_gate_check_failed`.
#
# The real cause was the worst of the three: exp6228's artifact carried every gated field,
# spelled correctly, with every value null -- `honest_verdict` included. A skeleton, not a result.
# A check that asks only "is the key present?" sees a complete artifact.
# ---------------------------------------------------------------------------


def test_null_valued_gate_field_is_reported_as_null_not_as_absent(tmp_path):
    """The .539 case. The field IS there; it is empty. Say so, and name the other empties."""
    _seed_artifact(
        tmp_path,
        6228,
        "runtime_endurance",
        {
            "gemma_4_31b_runtime_ready_score": None,
            "qwen_runtime_ready_score": None,
            "honest_verdict": None,
        },
    )
    task = {
        "id": "exp6229-downstream",
        "gated_on": [
            {
                "upstream": "exp6228-supervised-three-family-runtime-endurance",
                "artifact_field": "gemma_4_31b_runtime_ready_score",
                "op": "==",
                "value": 1,
            }
        ],
    }
    r = evaluate_gates(task, results_dir=tmp_path)

    assert not r.passed
    assert "wrote 'gemma_4_31b_runtime_ready_score' as null" in r.summary
    # The tell that the whole artifact is a template, which is what makes re-running the
    # DOWNSTREAM task pointless. Without this the reader retries the wrong task.
    assert "template and never filled it" in r.summary
    assert "Re-run the UPSTREAM task" in r.summary
    assert "honest_verdict" in r.summary


def test_absent_gate_field_says_the_upstream_never_promised_it_and_names_near_misses(tmp_path):
    """The other half. An absent field is a broken contract, not an empty result.

    The near-miss list exists because the usual cause is spelling drift between the gate and the
    upstream's REQUIRED ARTIFACT FIELDS -- `scorer_ready` gated against an artifact that wrote
    `ebcn_scorer_ready`. Naming the close key turns a cascade into a one-word fix.
    """
    _seed_artifact(tmp_path, 6228, "runtime_endurance", {"qwen_runtime_ready_score": 1.0})
    task = {
        "id": "exp6229-downstream",
        "gated_on": [
            {
                "upstream": "exp6228-supervised-three-family-runtime-endurance",
                "artifact_field": "qwen_runtime_ready",
                "op": "==",
                "value": 1,
            }
        ],
    }
    r = evaluate_gates(task, results_dir=tmp_path)

    assert not r.passed
    assert "has NO field 'qwen_runtime_ready'" in r.summary
    assert "REQUIRED ARTIFACT FIELDS" in r.summary
    assert "qwen_runtime_ready_score" in r.summary, "must name the near-miss key"
    assert "as null" not in r.summary, "absent and null must not be described the same way"


def test_diagnosis_never_changes_whether_a_gate_passes(tmp_path):
    """Diagnosis is prose. A guard that alters a verdict while explaining it is a new bug class."""
    _seed_artifact(tmp_path, 6228, "runtime_endurance", {"ready_score": 1.0})
    task = {
        "id": "exp6229-downstream",
        "gated_on": [
            {
                "upstream": "exp6228-supervised-three-family-runtime-endurance",
                "artifact_field": "ready_score",
                "op": "==",
                "value": 1.0,
            }
        ],
    }
    r = evaluate_gates(task, results_dir=tmp_path)

    assert r.passed, "a satisfied gate must still pass"
    assert "as null" not in r.summary and "has NO field" not in r.summary


def test_quarantined_upstream_is_named_ahead_of_the_null_field(tmp_path):
    """The .539 answer was already on disk and nothing surfaced it.

    exp6228 wrote `status: preconditions_recorded`, `duration_s: 0.0` and
    `flagged_adversarial: true` with a CRITICAL NONTERMINAL_DECLARED_ARTIFACT corrigendum -- it
    recorded its preconditions and never ran the experiment, and adversarial_verify.py caught
    exactly that. Four gates then read the same file and said only `actual=None`, never mentioning
    the stamp sitting beside the field they were reading.

    Quarantine outranks the null/absent split because it changes the instruction. "The field is
    null, fill it in" is wrong advice for an artifact the fabrication gate has already rejected.
    """
    _seed_artifact(
        tmp_path,
        6228,
        "runtime_endurance",
        {
            "gemma_4_31b_runtime_ready_score": None,
            "status": "preconditions_recorded",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "NONTERMINAL_DECLARED_ARTIFACT"}],
        },
    )
    task = {
        "id": "exp6229-downstream",
        "gated_on": [
            {
                "upstream": "exp6228-supervised-three-family-runtime-endurance",
                "artifact_field": "gemma_4_31b_runtime_ready_score",
                "op": "==",
                "value": 1,
            }
        ],
    }
    r = evaluate_gates(task, results_dir=tmp_path)

    assert not r.passed
    assert "UPSTREAM IS QUARANTINED" in r.summary
    assert "NONTERMINAL_DECLARED_ARTIFACT" in r.summary
    assert "preconditions_recorded" in r.summary
    # Must NOT tell the reader to fill in the field -- that is the wrong fix here.
    assert "template and never filled it" not in r.summary
    assert "the upstream task has to " in r.summary
