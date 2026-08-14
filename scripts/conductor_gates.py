"""Pre-gate check for the research conductor.

Before spawning the 50-turn Sonnet research-step call, evaluate any
gates declared on the task. If a gate fails (a prerequisite experiment's
artifact doesn't satisfy the declared condition), write a blocked
artifact directly and skip the Sonnet call entirely.

The Sonnet call is the most expensive part of an iteration (5-9 min on
typical experiments). When the experiment is going to write a blocked
artifact anyway because its prerequisite failed, those 5-9 min are pure
waste. The pre-gate check moves the gate evaluation up to the conductor,
where it costs ~50ms instead of a full LLM round-trip.

The gate format is declarative and lives in research-roadmap.yaml:

    - id: exp823-fr11-tier1-live-relay-v2
      title: Exp 823: FR-11 Tier 1 Live Relay v2 ...
      gated_on:
        - upstream: exp821-constraint-addition-live-v2
          artifact_field: delta_overall
          op: ">"
          value: 0.0
        - upstream: exp819-injection-field-fix
          artifact_field: honest_verdict
          op: "in"
          value: ["injection_field_fixed", "discrimination_above_baseline"]

All gates in the list must pass for the task to proceed. Absence of a
`gated_on` field means the task is ungated — the pre-gate check is a
no-op and the conductor proceeds to the Sonnet call as before. This is
the backwards-compatible default; any existing roadmap YAML keeps
working unchanged.

Supported `op` values (intentionally small):
  ==, !=, >, >=, <, <=, in, not_in, contains, not_contains
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reuse the artifact-finding logic from the in-process doc reconciler.
# Both modules need to map a task id like "exp819-something" to its
# results/experiment_819_*.json file, and there's no reason to duplicate.


def _find_artifact_by_task_id(task_id: str, results_dir: Path) -> Path | None:
    """Locate an experiment's artifact JSON by its YAML task id.

    Mirrors the implementation in scripts/in_process_doc_reconcile.py.
    Kept as a private helper here to avoid an inter-module import cycle
    if the conductor imports both modules — the conductor adds scripts/
    to sys.path before either import, but a duplicate is safer than a
    cycle.
    """
    match = re.match(r"exp(\d+)", task_id.lower())
    if not match:
        return None
    exp_num = match.group(1)
    candidates = sorted(results_dir.glob(f"experiment_{exp_num}_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


@dataclass
class GateResult:
    """Outcome of evaluating a single gate predicate."""

    upstream: str
    artifact_field: str
    op: str
    expected: Any
    actual: Any
    passed: bool
    reason: str = ""
    artifact_path: str | None = None
    artifact_sha256: str | None = None


@dataclass
class GateCheckResult:
    """Outcome of evaluating all gates on a task."""

    passed: bool
    gates_evaluated: list[GateResult] = field(default_factory=list)
    summary: str = ""


_TRUE_TOKENS = frozenset({"true", "1", "yes", "y", "on"})
_FALSE_TOKENS = frozenset({"false", "0", "no", "n", "off"})


def _coerce_gate_value(v: Any) -> Any:
    """Normalize a value that should be treated as a Boolean.

    Background: milestone .80 wedged because exp1030 had a gate of the
    form ``upstream.pre_test_fixed == True`` (Python bool, parsed from
    YAML) but the *upstream* artifact at one point contained the string
    "True" (or vice versa, depending on how Sonnet serialized the
    artifact). Python's ``==`` says ``"True" != True``, so the gate
    blocked even though the upstream had "really" succeeded.

    This helper maps the common bool-ish shapes onto Python bool:
      - bool stays bool
      - common truthy/falsy strings (case-insensitive) -> bool
      - 0/1 ints/floats -> bool
      - everything else -> unchanged (caller decides)

    The caller (``_eval_op``) only invokes this when at least one side
    of an ``==``/``!=`` comparison is itself a bool, so a string-equality
    gate like ``honest_verdict == "preflight_complete"`` is *not*
    rewritten — the function returns the original string in that case.
    Narrow scope is deliberate: we are fixing the bool-coercion wedge,
    not changing how every gate compares values.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        low = v.strip().lower()
        if low in _TRUE_TOKENS:
            return True
        if low in _FALSE_TOKENS:
            return False
        return v  # not a bool-ish string; leave it alone
    if isinstance(v, (int, float)):
        return bool(v)
    return v


def _gate_value_type(value: Any) -> str:
    """Name the observed JSON type in blocked gate diagnostics."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _artifact_sha256(path: Path) -> str | None:
    """Hash the exact upstream artifact read by a gate."""
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _eval_op(actual: Any, op: str, expected: Any) -> tuple[bool, str]:
    """Apply a single comparison operator. Returns (passed, reason).

    The `reason` string is human-readable and goes straight into the
    blocked artifact when the gate fails. Designed so a reader can
    diagnose the block from the artifact alone, without re-running.

    Bool coercion: when either side of an ``==``/``!=`` is a Python
    bool, both sides are passed through ``_coerce_gate_value`` so that
    string "True"/"true"/"1" matches Python ``True`` and string
    "False"/"false"/"0" matches Python ``False``. This closes the
    .80 wedge where the gate evaluator reported
    ``actual='True' == expected=True -> False``. See the
    ``conductor-fastpath-bootstrap-skip`` proposal sibling fix for
    full context.
    """
    if op == "==":
        a, e = (
            (_coerce_gate_value(actual), _coerce_gate_value(expected))
            if isinstance(actual, bool) or isinstance(expected, bool)
            else (actual, expected)
        )
        return a == e, f"actual={actual!r} == expected={expected!r}"
    if op == "!=":
        a, e = (
            (_coerce_gate_value(actual), _coerce_gate_value(expected))
            if isinstance(actual, bool) or isinstance(expected, bool)
            else (actual, expected)
        )
        return a != e, f"actual={actual!r} != expected={expected!r}"
    if op in (">", ">=", "<", "<=") and (actual is None or expected is None):
        return (
            False,
            f"numeric comparison rejected because one side is None (actual={actual!r}, expected={expected!r})",
        )
    if op in (">", ">=", "<", "<="):
        if (isinstance(actual, float) and math.isnan(actual)) or (
            isinstance(expected, float) and math.isnan(expected)
        ):
            return (
                False,
                "numeric comparison rejected because NaN is not comparable "
                f"(actual={actual!r}, expected={expected!r})",
            )
        try:
            if op == ">":
                return actual > expected, f"actual={actual} > expected={expected}"
            if op == ">=":
                return actual >= expected, f"actual={actual} >= expected={expected}"
            if op == "<":
                return actual < expected, f"actual={actual} < expected={expected}"
            return actual <= expected, f"actual={actual} <= expected={expected}"
        except TypeError:
            return (
                False,
                "numeric comparison rejected because values are not comparable "
                f"(actual_type={type(actual).__name__}, expected_type={type(expected).__name__})",
            )
    if op == "in":
        if not isinstance(expected, (list, tuple, set)):
            return (
                False,
                f"'in' op requires a list/tuple/set on the right (got {type(expected).__name__})",
            )
        return actual in expected, f"actual={actual!r} in expected={list(expected)!r}"
    if op == "not_in":
        if not isinstance(expected, (list, tuple, set)):
            return (
                False,
                f"'not_in' op requires a list/tuple/set on the right (got {type(expected).__name__})",
            )
        return actual not in expected, f"actual={actual!r} not in expected={list(expected)!r}"
    if op == "contains":
        if actual is None:
            return False, f"actual is None, cannot contain {expected!r}"
        return expected in actual, f"actual={actual!r} contains expected={expected!r}"
    if op == "not_contains":
        if actual is None:
            return True, f"actual is None, vacuously does not contain {expected!r}"
        return expected not in actual, f"actual={actual!r} does not contain expected={expected!r}"
    if op == "exists":
        # The field is considered to EXIST when it is present with a real (non-null)
        # value. `evaluate_gates` fetches via data.get(field), so a missing key and an
        # explicit null both arrive as actual=None — both mean "the upstream hasn't
        # produced this field yet", which is exactly what `exists` should reject.
        # The required `value` (schema mandates one) is honored: value true/None ->
        # pass when present; value false -> pass when absent (an inline not_exists).
        present = actual is not None
        want = _coerce_gate_value(expected)
        if isinstance(want, bool):
            return present == want, (
                f"field is {'present' if present else 'absent'} "
                f"(actual={actual!r}); expected exists={want}"
            )
        return present, f"field is {'present' if present else 'absent'} (actual={actual!r})"
    if op == "not_exists":
        return actual is None, (
            f"field is {'absent' if actual is None else 'present'} (actual={actual!r})"
        )
    return False, f"unknown op {op!r}"


def _diagnose_missing_field(data: dict, field: str, base_reason: str) -> str:
    """Say WHY a gated field read None. `dict.get` cannot tell these three cases apart.

    WHY THIS EXISTS. Milestone .539 cascade-blocked four tasks on
    `exp6228-supervised-three-family-runtime-endurance`. Each block recorded the same message:
    `actual=None == expected=1`. That is true and useless. It does not say whether the upstream
    forgot the field, wrote it as null, or produced nothing at all -- and the three have different
    fixes. Nobody could act on it, so the block sat unexamined and the ledger counted it as one
    more anonymous `blocked_gate_check_failed`.

    What had actually happened was the worst of the three: exp6228's artifact carried every field
    the gates named, spelled correctly, with every value `null` -- including `honest_verdict`. A
    skeleton, not a result. A check that asks only "is the key present?" sees a complete artifact.

    THE THREE CASES, and why each gets its own sentence:

      absent      the upstream did not honour its own REQUIRED ARTIFACT FIELDS. Fix the upstream
                  task's prompt, or the gate is naming a field nobody promised. Near-miss keys are
                  listed because the usual cause is a spelling drift (`scorer_ready` written as
                  `ebcn_scorer_ready`).
      null-only   the field is there and empty. If several required-looking fields are null
                  together, the upstream emitted a template and never filled it, so re-running the
                  DOWNSTREAM task can never help -- the upstream is what has to run again.
      real value  not this function's business; the caller only calls it when actual is None.

    Diagnosis only. It never changes whether the gate passes.
    """
    # Check the fabrication gate FIRST. It usually already knows, and it outranks everything
    # below: if the upstream is quarantined, its field being null or absent is a symptom, and
    # "fill in the field" is the wrong instruction.
    #
    # This is the .539 case exactly. exp6228 recorded `status: preconditions_recorded`,
    # `duration_s: 0.0`, and `flagged_adversarial: true` with a CRITICAL
    # `NONTERMINAL_DECLARED_ARTIFACT` corrigendum. The task never ran the experiment -- it wrote
    # its preconditions and stopped. adversarial_verify.py caught that and stamped it. Four gates
    # then read the same artifact and reported `actual=None`, never mentioning the stamp sitting
    # beside the field they were reading. The answer was already on disk and nothing surfaced it.
    if data.get("flagged_adversarial"):
        pend = data.get("corrigendum_pending") or []
        kinds = sorted({p.get("kind", "?") for p in pend if isinstance(p, dict)})
        status = data.get("status")
        return (
            f"{base_reason} -- UPSTREAM IS QUARANTINED: it carries flagged_adversarial=true"
            + (f" ({', '.join(kinds)})" if kinds else "")
            + (f", status={status!r}" if status else "")
            + ". The fabrication gate already rejected this artifact, so the null field is a "
            "symptom, not the problem. Do not fill the field in; the upstream task has to "
            "actually run. Per CLAUDE.md, never aggregate a flagged artifact's numbers."
        )

    if field in data:
        nulls = [
            k
            for k, v in data.items()
            if v is None and (k.endswith(("_score", "_ready", "_verdict")) or k == field)
        ]
        extra = (
            f" The upstream ALSO left {len(nulls) - 1} other required-looking field(s) null "
            f"({sorted(k for k in nulls if k != field)[:4]}), so it emitted a template and never "
            "filled it. Re-run the UPSTREAM task; re-running this one cannot help."
            if len(nulls) > 1
            else " Re-run the UPSTREAM task; re-running this one cannot help."
        )
        return f"{base_reason} -- upstream wrote {field!r} as null.{extra}"

    stem = field.rsplit("_", 2)[0] if "_" in field else field
    near = sorted(k for k in data if stem and stem in k and k != field)[:4]
    hint = f" Closest keys it DID write: {near}." if near else ""
    return (
        f"{base_reason} -- upstream artifact has NO field {field!r}; it did not honour its own "
        f"REQUIRED ARTIFACT FIELDS, or this gate names a field nobody promised.{hint}"
    )


def evaluate_gates(
    task: dict,
    results_dir: Path | None = None,
) -> GateCheckResult:
    """Evaluate all gates declared on a task. Returns a GateCheckResult.

    A task with no `gated_on` field passes vacuously. A task whose every
    gate evaluates to True passes. A task with at least one failing gate
    fails, and the failure reason captures *which* gate and *why*.

    The function never raises; any internal error (missing file, malformed
    YAML, JSON decode error) is surfaced through GateResult.passed=False
    with a descriptive reason. The conductor treats both "real failure"
    and "internal error" the same way — write a blocked artifact and
    skip the Sonnet call. Defensive: better to wastefully block one
    runnable experiment than to silently bypass a real gate.
    """
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results"

    gates = task.get("gated_on") or []
    if not gates:
        return GateCheckResult(passed=True, summary="no gates declared")

    results: list[GateResult] = []
    all_passed = True
    for gate_spec in gates:
        upstream = gate_spec.get("upstream", "")
        artifact_field = gate_spec.get("artifact_field", "")
        op = gate_spec.get("op", "==")
        expected = gate_spec.get("value")

        artifact_path = _find_artifact_by_task_id(upstream, results_dir)
        if artifact_path is None:
            results.append(
                GateResult(
                    upstream=upstream,
                    artifact_field=artifact_field,
                    op=op,
                    expected=expected,
                    actual=None,
                    passed=False,
                    reason=f"upstream artifact not found for task id {upstream!r}",
                    artifact_path=None,
                    artifact_sha256=None,
                )
            )
            all_passed = False
            continue

        artifact_sha = _artifact_sha256(artifact_path)
        try:
            data = json.loads(artifact_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            results.append(
                GateResult(
                    upstream=upstream,
                    artifact_field=artifact_field,
                    op=op,
                    expected=expected,
                    actual=None,
                    passed=False,
                    reason=f"upstream artifact unreadable: {exc}",
                    artifact_path=artifact_path.as_posix(),
                    artifact_sha256=artifact_sha,
                )
            )
            all_passed = False
            continue

        actual = data.get(artifact_field)
        passed, op_reason = _eval_op(actual, op, expected)
        if actual is None:
            op_reason = _diagnose_missing_field(data, artifact_field, op_reason)
        results.append(
            GateResult(
                upstream=upstream,
                artifact_field=artifact_field,
                op=op,
                expected=expected,
                actual=actual,
                passed=passed,
                reason=op_reason,
                artifact_path=artifact_path.as_posix(),
                artifact_sha256=artifact_sha,
            )
        )
        if not passed:
            all_passed = False

    if all_passed:
        summary = f"{len(results)} gate(s) satisfied"
    else:
        failing = [g for g in results if not g.passed]
        first = failing[0]
        summary = (
            f"{len(failing)} of {len(results)} gate(s) failed; "
            f"first failure: {first.upstream}.{first.artifact_field} ({first.reason})"
        )

    return GateCheckResult(passed=all_passed, gates_evaluated=results, summary=summary)


def _first_failed_gate(gate_check: GateCheckResult) -> GateResult | None:
    """Return the failed gate a blocked artifact should expose."""
    return next((g for g in gate_check.gates_evaluated if not g.passed), None)


def _blocked_diagnostic_contract(first_failed: GateResult | None) -> dict[str, Any]:
    """Build the stable failed-gate contract for future blocked artifacts."""
    if first_failed is None:
        return {
            "version": "blocked_gate_diagnostic_v1",
            "blocked_reason": None,
            "failed_upstream": None,
            "failed_field": None,
            "failed_operator": None,
            "failed_expected": None,
            "failed_expected_type": "null",
            "failed_observed": None,
            "failed_observed_type": "null",
            "failed_evidence_path": None,
            "failed_evidence_sha256": None,
        }
    return {
        "version": "blocked_gate_diagnostic_v1",
        "blocked_reason": first_failed.reason,
        "failed_upstream": first_failed.upstream,
        "failed_field": first_failed.artifact_field,
        "failed_operator": first_failed.op,
        "failed_expected": first_failed.expected,
        "failed_expected_type": _gate_value_type(first_failed.expected),
        "failed_observed": first_failed.actual,
        "failed_observed_type": _gate_value_type(first_failed.actual),
        "failed_evidence_path": first_failed.artifact_path,
        "failed_evidence_sha256": first_failed.artifact_sha256,
    }


def write_blocked_artifact(
    task: dict,
    gate_check: GateCheckResult,
    results_dir: Path | None = None,
) -> Path | None:
    """Write a minimal blocked artifact when the pre-gate check fails.

    The artifact contains every field required by REQUIRED_RESULT_FIELDS
    in scripts/experiment_template.py so the in-process doc reconciler
    and downstream tooling can parse it like any other experiment
    artifact. The honest_verdict is "blocked_gate_check_failed", which
    the reconciler maps to "⚠️ Blocked" via the standard mapping table.

    Returns the path of the written file, or None if the task id can't
    be parsed into an experiment number (in which case the conductor
    should fall through to the Sonnet path — defensive).
    """
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results"

    task_id = task.get("id", "")
    match = re.match(r"exp(\d+)-(.+)$", task_id.lower())
    if not match:
        return None
    exp_num = match.group(1)
    slug = match.group(2).replace("-", "_")
    target = results_dir / f"experiment_{exp_num}_{slug}.json"

    now = datetime.now(UTC)
    iso_now = now.isoformat()
    first_failed = _first_failed_gate(gate_check)
    diagnostic_contract = _blocked_diagnostic_contract(first_failed)

    artifact = {
        # REQUIRED_RESULT_FIELDS — keep aligned with scripts/experiment_template.py:153
        "experiment": int(exp_num),
        "schema": "blocked_gate_check_v1",
        "run_date": now.strftime("%Y-%m-%d"),
        "started_at": iso_now,
        "finished_at": iso_now,
        "duration_s": 0.0,
        "status": "blocked",
        "title": task.get("title", f"Exp {exp_num}: (untitled)"),
        # Domain-specific fields
        "honest_verdict": "blocked_gate_check_failed",
        "blocked_reason": diagnostic_contract["blocked_reason"] or gate_check.summary,
        "failed_upstream": diagnostic_contract["failed_upstream"],
        "failed_field": diagnostic_contract["failed_field"],
        "failed_operator": diagnostic_contract["failed_operator"],
        "failed_expected": diagnostic_contract["failed_expected"],
        "failed_expected_type": diagnostic_contract["failed_expected_type"],
        "failed_observed": diagnostic_contract["failed_observed"],
        "failed_observed_type": diagnostic_contract["failed_observed_type"],
        "failed_evidence_path": diagnostic_contract["failed_evidence_path"],
        "failed_evidence_sha256": diagnostic_contract["failed_evidence_sha256"],
        "blocked_diagnostic_contract": diagnostic_contract,
        "gate_check_summary": gate_check.summary,
        "gates_evaluated": [
            {
                "upstream": g.upstream,
                "artifact_field": g.artifact_field,
                "op": g.op,
                "expected": g.expected,
                "expected_type": _gate_value_type(g.expected),
                "actual": g.actual,
                "actual_type": _gate_value_type(g.actual),
                "passed": g.passed,
                "reason": g.reason,
                "artifact_path": g.artifact_path,
                "artifact_sha256": g.artifact_sha256,
            }
            for g in gate_check.gates_evaluated
        ],
        # Marker so downstream tooling can recognise pre-gate-blocked
        # artifacts vs Sonnet-written blocked artifacts.
        "blocked_at_layer": "conductor_pre_gate",
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2) + "\n")
    return target


def select_max_turns(task: dict, default: int = 100) -> int:
    """Pick the per-task max_turns value with a sensible default.

    Per-experiment override lives in research-roadmap.yaml as a top-level
    `max_turns:` field on the task. Simple experiments (CPU-only retros,
    documentation passes, configuration changes) can opt into a smaller
    budget, freeing API quota and shaving wall time on the Sonnet call.

    Default raised from 50 to 100 on 2026-04-28 after the .80 cascade:
    Exps 1028 (Preflight v30), 1029 (FoVer Corpus v2) both hit the 50-turn
    cap on tasks that were genuinely making progress. The capacity-bound
    failure mode produced no artifact and cascade-blocked downstream
    experiments. 100 is the upper bound enforced below.
    """
    val = task.get("max_turns", default)
    if not isinstance(val, int):
        # Reject malformed YAML rather than silently picking a wrong number.
        # The conductor will see an explicit log entry and fall back to default.
        return default
    if val < 1 or val > 100:
        # Bounds sanity. 100 is well above any seen budget; 1 means the
        # agent gets one turn which is essentially useless. Out-of-bounds
        # values fall back to the default.
        return default
    # Transition-task turn floor (2026-05-31 operator directive). Archive /
    # capstone / plan-milestone tasks run on claude and were repeatedly hitting
    # their planner-emitted 20-turn cap ("Claude Code error: Reached max turns
    # (20)") — failing, then re-running, ~2 wasted claude calls/milestone (17
    # Claude-side errors across .315-.322 vs 1 gemini). These ops tasks read
    # upstream artifacts and write retros/changelog + validate YAML, so they
    # genuinely need more headroom than the planner gives them. Floor them at 40
    # (still within the <=100 bound) regardless of the emitted value.
    _tid = str(task.get("id", "")).lower()
    if val < 40 and any(_p in _tid for _p in ("archive-v", "capstone-v", "plan-milestone-")):
        return 40
    return val
