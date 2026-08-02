"""Prove, mechanically, that the goal-dedup change is INERT with its flag unset.

WHY THIS EXISTS RATHER THAN A SENTENCE IN THE ARTIFACT. `artifact_freshness_lint.py`
refuses a commit when an analyser artifact pins the sha256 of a source file that has
since changed, and 7 committed artifacts pin `arc_executable_world_model.py`. The
sanctioned remedy is a `provenance.freshness_acknowledgements` entry -- but an
acknowledgement whose `evidence` field is "I read the diff and it looked fine" is the
kind of claim this project has been burned by. The lint's own closing line is the
standard: a rebuild that silently changes a published figure is a correction owed.
These artifacts carry MEASURED wall-clock timings that a rebuild would destroy, so
inertness has to be established directly instead.

THE THREE CHECKS, each of which can fail independently.

1. PRE-EXISTING FUNCTIONS ARE GUARDED. Diff the changed module against `HEAD`, and for
   every pre-existing function whose body changed, assert that each newly-added
   EXECUTABLE statement is lexically inside an `if _goal_dedup_on():` block. A new
   statement outside such a guard would run on the shipped path.

2. NEW FUNCTIONS ARE UNREACHABLE WITH THE FLAG OFF, TRANSITIVELY. The change also adds
   top-level helpers. Compute a least fixed point over the call graph: a helper is
   unreachable when every call site is lexically inside a guard body, protected by
   `_goal_dedup_on() and ...` short-circuit, or inside a helper already proved
   unreachable. Both refinements are load-bearing rather than conveniences -- the real
   call site is `if _goal_dedup_on() and _engine_half_goal_usable(eng):`, where the
   second operand is never evaluated with the flag off, and one helper calls another
   with no guard of its own.

3. THE FLAG IS OFF BY DEFAULT AND RESOLVES EXACTLY LIKE ITS SIBLINGS. Assert
   `_goal_dedup_on()` is False when unset, and that every probe value resolves to
   `value.strip() == "1"` -- the convention every other flag in the module uses. Note
   this asserts agreement with the convention, not "everything but the literal `1` is
   False": `"1 "` IS enabling, deliberately.

Check 1 is the load-bearing one; 2 and 3 close the ways it could be true and still not
matter. Prints a report and exits non-zero if any check fails -- so if a later edit to
this module stops being inert, regenerating the acknowledgement will FAIL rather than
quietly restate a claim that has become false.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TARGET = "python/carnot/agentic/arc_executable_world_model.py"
GUARD = "_goal_dedup_on"


def _head_source() -> str:
    return subprocess.run(  # noqa: S603
        ["git", "show", f"HEAD:{TARGET}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _top_level_funcs(tree: ast.AST) -> dict[str, ast.AST]:
    """Every function the module defines, keyed by a qualified-enough name.

    Methods are keyed `Class.method` so a method body change is not confused with a
    module-level function of the same name.
    """
    out: dict[str, ast.AST] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{sub.name}"] = sub
    return out


def _guarded_line_ranges(fn: ast.AST) -> list[tuple[int, int]]:
    """Line ranges of every `if _goal_dedup_on()...:` body inside `fn`."""
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        names = {
            n.func.id
            for n in ast.walk(node.test)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        if GUARD not in names:
            continue
        body = node.body
        if body:
            ranges.append((body[0].lineno, max(s.end_lineno or s.lineno for s in body)))
    return ranges


def _short_circuit_protected(fn: ast.AST) -> list[tuple[int, int]]:
    """Line ranges protected by `_goal_dedup_on() and <...>` short-circuit evaluation.

    `if _goal_dedup_on() and _engine_half_goal_usable(eng):` does NOT call the second
    operand when the first is False -- Python's `and` stops at the first falsey value. A
    call there is therefore just as unreachable with the flag unset as one inside the
    guard's body, but a naive "is this line inside the if-BODY" test would report it as
    running on the shipped path. Modelling this explicitly matters: without it the proof
    fails on a construct that is genuinely inert, and the temptation would be to relax
    the check rather than teach it the language's semantics.

    Only operands AFTER the guard call are protected -- in `X() and _goal_dedup_on()`
    the call to `X` happens first and is not protected by anything.
    """
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.And):
            continue
        seen_guard = False
        for operand in node.values:
            if seen_guard:
                ranges.append((operand.lineno, operand.end_lineno or operand.lineno))
            calls = {
                n.func.id
                for n in ast.walk(operand)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            if GUARD in calls:
                seen_guard = True
    return ranges


def _executable_statements(fn: ast.AST) -> list[ast.stmt]:
    """Statements that DO something: docstrings and bare constants excluded."""
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.stmt) and not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            out.append(node)
    return out


def main() -> int:
    now_src = (REPO / TARGET).read_text()
    head_src = _head_source()
    now_tree, head_tree = ast.parse(now_src), ast.parse(head_src)
    now_fns, head_fns = _top_level_funcs(now_tree), _top_level_funcs(head_tree)

    report: dict[str, Any] = {
        "target": TARGET,
        "sha256_was": hashlib.sha256(head_src.encode()).hexdigest(),
        "sha256_now": hashlib.sha256(now_src.encode()).hexdigest(),
        "guard": f"{GUARD}()",
    }
    failures: list[str] = []

    # --- Check 1: changed pre-existing functions -----------------------------------
    changed = []
    for name, fn in now_fns.items():
        if name not in head_fns:
            continue
        if ast.dump(fn) == ast.dump(head_fns[name]):
            continue
        changed.append(name)
        guarded = _guarded_line_ranges(fn)
        head_stmt_dumps = {ast.dump(s) for s in _executable_statements(head_fns[name])}
        unguarded_new = []
        for stmt in _executable_statements(fn):
            if ast.dump(stmt) in head_stmt_dumps:
                continue  # statement existed before, unchanged
            if any(lo <= stmt.lineno <= hi for lo, hi in guarded):
                continue  # new, but inside the flag guard
            # A statement whose own body is entirely guarded (e.g. the guard `if` itself)
            # is not itself a new executable effect.
            if isinstance(stmt, ast.If) and any(
                lo <= (stmt.body[0].lineno if stmt.body else stmt.lineno) <= hi
                for lo, hi in guarded
            ):
                continue
            unguarded_new.append(
                {"line": stmt.lineno, "src": ast.get_source_segment(now_src, stmt) or ""}
            )
        if unguarded_new:
            failures.append(f"{name}: {len(unguarded_new)} unguarded new statement(s)")
        report.setdefault("changed_functions", []).append(
            {
                "function": name,
                "guarded_ranges": guarded,
                "unguarded_new_statements": unguarded_new,
            }
        )
    report["n_changed_preexisting_functions"] = len(changed)

    # --- Check 2: new helpers are unreachable with the flag off ---------------------
    # TRANSITIVE, by least-fixed-point. A new helper is unreachable if EVERY call site is
    # either (a) lexically inside an `if _goal_dedup_on():` body, (b) short-circuit-protected
    # by `_goal_dedup_on() and ...`, or (c) inside another helper already known unreachable.
    # Case (c) is what makes this a closure rather than a one-hop check: `_engine_half_goal_usable`
    # calls `_goal_predicate_is_constant_false` with no guard of its own, and that is fine
    # precisely because its own caller never runs. Starting from "nothing is known unreachable"
    # and only ever ADDING names means the result cannot be inflated by assuming what it proves.
    new_funcs = sorted(set(now_fns) - set(head_fns))
    report["new_functions"] = new_funcs

    def _sites_for(bare: str) -> list[dict[str, Any]]:
        sites = []
        for holder, fn in now_fns.items():
            guarded = _guarded_line_ranges(fn) + _short_circuit_protected(fn)
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == bare
                ):
                    sites.append(
                        {
                            "in": holder,
                            "line": node.lineno,
                            "lexically_guarded": any(lo <= node.lineno <= hi for lo, hi in guarded),
                        }
                    )
        return sites

    callsites = {name: _sites_for(name.split(".")[-1]) for name in new_funcs}
    # `_goal_dedup_on` IS the guard; it is meant to be called on the shipped path (it is what
    # returns False there), so it seeds the unreachable set trivially rather than being proved.
    unreachable: set[str] = {GUARD}
    changed_fixpoint = True
    while changed_fixpoint:
        changed_fixpoint = False
        for name in new_funcs:
            if name in unreachable:
                continue
            sites = callsites[name]
            if sites and all(s["lexically_guarded"] or s["in"] in unreachable for s in sites):
                unreachable.add(name)
                changed_fixpoint = True
    for name in new_funcs:
        if name in unreachable:
            continue
        bad = [s for s in callsites[name] if not s["lexically_guarded"]]
        for s in bad:
            failures.append(f"{name} reachable with flag off, called at {s['in']}:{s['line']}")
    report["new_function_callsites"] = callsites
    report["unreachable_with_flag_off"] = sorted(unreachable)

    # --- Check 3: the flag defaults off and fails closed ---------------------------
    sys.path.insert(0, str(REPO / "python"))
    import os

    from carnot.agentic import arc_executable_world_model as awm

    # `"1 "` and `" 1"` are ENABLING values, not malformed ones: every sibling flag in this
    # module resolves with `raw.strip() == "1"`, so whitespace padding is deliberate tolerance
    # and matching it is the point. Only a value that is not a whitespace-padded "1" must be
    # rejected. An earlier revision of this script asserted the opposite and "failed" on `"1 "`
    # -- recorded rather than quietly amended, because the tempting fix at that moment was to
    # change the module to match the checker, which would have made this flag the odd one out.
    probe_values = ("", "0", "true", "yes", "2", " ", "01", "1 ", " 1", "11", "1", "TRUE")
    flag_probe: dict[str, bool] = {}
    saved = os.environ.pop("CARNOT_ARC_GOAL_DEDUP", None)
    try:
        flag_probe["unset"] = awm._goal_dedup_on()
        for value in probe_values:
            os.environ["CARNOT_ARC_GOAL_DEDUP"] = value
            flag_probe[repr(value)] = awm._goal_dedup_on()
    finally:
        os.environ.pop("CARNOT_ARC_GOAL_DEDUP", None)
        if saved is not None:
            os.environ["CARNOT_ARC_GOAL_DEDUP"] = saved
    report["flag_probe"] = flag_probe
    report["flag_convention"] = "enabled iff os.environ[...].strip() == '1'"
    if flag_probe["unset"] is not False:
        failures.append("flag is not off by default")
    for value in probe_values:
        expected = value.strip() == "1"
        if flag_probe[repr(value)] is not expected:
            failures.append(
                f"flag value {value!r} resolved {flag_probe[repr(value)]}, expected {expected}"
            )

    report["failures"] = failures
    report["inert_with_flag_unset"] = not failures
    (HERE / "inertness_proof.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"changed pre-existing functions : {changed}")
    print(f"new functions                  : {new_funcs}")
    print(f"flag unset -> {flag_probe['unset']}, flag '1' -> {flag_probe[chr(39) + '1' + chr(39)]}")
    print(f"INERT WITH FLAG UNSET          : {not failures}")
    for f in failures:
        print(f"  FAILURE: {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
