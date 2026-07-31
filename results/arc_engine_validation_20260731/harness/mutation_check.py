#!/usr/bin/env python3
"""PHASE 2, STEP 3 -- prove every new test can FAIL, by breaking the code it guards.

WHY. A passing test proves nothing on its own: a test that asserts something the code cannot
violate is a vacuous pass, and this repo has already been bitten by exactly that (see
`tests/python/test_arc_generator_stderr_capture.py`'s note about assertions "guarded against
the vacuous pass that occurred while writing it"). The only mechanical evidence that a test is
load-bearing is that a deliberate defect in the code makes it RED.

HOW. Each mutation is a literal string substitution into
`python/carnot/agentic/arc_engine_static_validation.py` that reverses one specific design
decision. The named tests are then run against the mutated module. A mutation is PROVED when
every one of its named tests fails; a mutation whose tests still pass identifies a test that is
not actually guarding its check, and is reported as NOT PROVED rather than quietly dropped.

The module is restored from an in-memory copy in a `finally`, and the restoration is verified
byte-for-byte at the end. Nothing here writes to any evidence directory.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path("/home/ianblenke/github.com/ianblenke/carnot")
MODULE = REPO / "python/carnot/agentic/arc_engine_static_validation.py"
TESTS = REPO / "tests/python/test_arc_engine_static_validation.py"
PY = "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python"
OUT = pathlib.Path(__file__).resolve().parent.parent / "mutation_check.json"

# (name, why this mutation is the right reversal, old, new, tests that MUST go red)
MUTATIONS: list[tuple[str, str, str, str, list[str]]] = [
    (
        "if_without_else_terminates",
        "The core of the ft09/tu93 detection: an `if` with no `else` can be skipped, so it "
        "cannot terminate a path. Claiming it does blinds the checker to exactly the shape both "
        "games failed on.",
        "        if not stmt.orelse:\n            return False",
        "        if not stmt.orelse:\n            return True",
        [
            "test_ft09_frozen_engine_is_flagged_missing_return",
            "test_tu93_shape_is_flagged_missing_return",
        ],
    ),
    (
        "loops_always_terminate",
        "A `for`/`while` may execute zero times. Treating every loop as terminating is the "
        "single most tempting simplification and it silently accepts the tu93 shape.",
        "        if isinstance(stmt, ast.While) and _is_literal_true(stmt.test) and not _has_break(stmt):\n            return True\n        return False",
        "        return True",
        ["test_loop_that_may_run_zero_times_IS_flagged"],
    ),
    (
        "while_true_not_special_cased",
        "The reverse error: `while True:` with no `break` never exits normally, so treating it "
        "like an ordinary loop REJECTS working code -- the false-positive direction.",
        "        if isinstance(stmt, ast.While) and _is_literal_true(stmt.test) and not _has_break(stmt):\n            return True\n",
        "",
        ["test_no_false_positive_on_terminating_shapes"],
    ),
    (
        "break_binds_across_nested_loops",
        "`_has_break` must not count a `break` that belongs to an inner loop, or an outer "
        "`while True` looks exitable and a correct engine is rejected.",
        "        if isinstance(child, (ast.While, ast.For, ast.AsyncFor)):\n            continue  # a break in there belongs to the inner loop\n",
        "",
        ["test_break_in_a_nested_loop_does_not_count_for_the_outer_one"],
    ),
    (
        "first_function_definition_wins",
        "Python binds the LAST definition. Grading the first grades a function that never runs.",
        "        if isinstance(node, ast.FunctionDef) and node.name == name:\n            found = node",
        "        if isinstance(node, ast.FunctionDef) and node.name == name and found is None:\n            found = node",
        ["test_last_definition_wins"],
    ),
    (
        "try_without_finally_always_terminates",
        "A try/except whose body and handlers all fall through plainly falls through; claiming "
        "otherwise hides a real defect inside a try block.",
        "        parts = [stmt.body + stmt.orelse] + [h.body for h in stmt.handlers]\n        return not any(_falls_through(p) for p in parts)",
        "        return True",
        ["test_try_except_falling_through_IS_flagged"],
    ),
    (
        "goal_arm_removed_from_dry_run",
        "lp85's observed failure is raised by `is_level_complete`, not `engine`. Dropping the "
        "goal arm makes the validator silently miss one of the four failures it exists for.",
        "    out.extend(_goal_defects(ns, transitions, limit=limit))\n",
        "",
        [
            "test_lp85_unbound_local_is_caught_by_the_goal_dry_run",
            "test_goal_returning_an_array_is_flagged_not_boolean",
        ],
    ),
    (
        "truncation_ignores_whether_symbols_arrived",
        "Hitting the cap AFTER writing both functions loses nothing. Flagging it anyway would "
        "throw away usable engines and retry forever.",
        "    missing = [fn for fn in required if f\"def {fn}\" not in code]\n    if not missing:\n        return None",
        "    missing = [fn for fn in required if f\"def {fn}\" not in code]",
        ["test_truncation_not_flagged_when_capped_but_complete"],
    ),
    (
        "truncation_does_not_short_circuit",
        "A truncated file has no end, so 'it never returns' is a false statement about it. "
        "Reporting both would send a half-written file back as a repair.",
        "    if trunc is not None:\n        return [trunc]",
        "    if trunc is not None:\n        pass",
        ["test_truncation_short_circuits_the_other_checks"],
    ),
    (
        "repair_prompt_includes_unrepairable",
        "A truncation in the repair block spends the very budget the retry needs.",
        "    actionable = [d for d in defects if d.repairable]",
        "    actionable = list(defects)",
        ["test_repair_prompt_carries_the_exception_text_and_omits_truncation"],
    ),
    (
        "dry_run_grades_predictions",
        "Reporting a merely-wrong prediction turns this defect scanner into a second, weaker "
        "trust gate -- the exact thing the module docstring forbids.",
        "        arr = np.asarray(pred)\n        if arr.shape != grid.shape:",
        "        arr = np.asarray(pred)\n        nxt = np.asarray(getattr(t, 'next_grid', grid))\n"
        "        if arr.shape == nxt.shape and not np.array_equal(arr, nxt):\n"
        "            out.append(EngineDefect(kind='engine_wrong_prediction', detail='wrong'))\n"
        "        if arr.shape != grid.shape:",
        ["test_dry_run_does_NOT_report_a_merely_wrong_prediction"],
    ),
    (
        "bare_return_not_flagged",
        "`return` and `return None` hand the caller None just as surely as falling off the end.",
        "            node.value is None\n            or (isinstance(node.value, ast.Constant) and node.value.value is None)",
        "            isinstance(node.value, ast.Constant) and node.value.value is Ellipsis",
        ["test_explicit_return_none_is_flagged", "test_bare_return_is_flagged"],
    ),
]


def run_tests(names: list[str]) -> tuple[bool, str]:
    """Run the named tests. Returns (all_failed, tail-of-output)."""
    expr = " or ".join(names)
    proc = subprocess.run(
        [PY, "-m", "pytest", str(TESTS), "--no-cov", "-q", "-p", "no:xdist", "-k", expr],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=900,
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-4:]
    return proc.returncode != 0, "\n".join(tail)


def main() -> int:
    t0 = time.time()
    original = MODULE.read_text()
    rows = []
    try:
        for name, why, old, new, tests in MUTATIONS:
            if old not in original:
                rows.append(
                    {
                        "mutation": name,
                        "status": "ANCHOR_NOT_FOUND",
                        "proved": False,
                        "why": why,
                        "tests": tests,
                    }
                )
                continue
            if original.count(old) != 1:
                rows.append(
                    {
                        "mutation": name,
                        "status": f"ANCHOR_AMBIGUOUS_x{original.count(old)}",
                        "proved": False,
                        "why": why,
                        "tests": tests,
                    }
                )
                continue
            MODULE.write_text(original.replace(old, new, 1))
            went_red, tail = run_tests(tests)
            rows.append(
                {
                    "mutation": name,
                    "why": why,
                    "tests": tests,
                    "status": "PROVED" if went_red else "NOT_PROVED_TESTS_STILL_GREEN",
                    "proved": went_red,
                    "pytest_tail": tail,
                }
            )
            print(f"  {'PROVED   ' if went_red else 'NOT PROVED'} {name}")
            MODULE.write_text(original)
    finally:
        MODULE.write_text(original)

    restored_ok = MODULE.read_text() == original
    # A clean baseline is part of the evidence: if the whole file were red for an unrelated
    # reason, every mutation would trivially "prove".
    base = subprocess.run(
        [PY, "-m", "pytest", str(TESTS), "--no-cov", "-q"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=900,
    )
    out = {
        "generated_by": "results/arc_engine_validation_20260731/harness/mutation_check.py",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(time.time() - t0, 2),
        "module_restored_byte_identical": restored_ok,
        "baseline_all_green": base.returncode == 0,
        "baseline_tail": (base.stdout + base.stderr).strip().splitlines()[-2:],
        "n_mutations": len(rows),
        "n_proved": sum(1 for r in rows if r["proved"]),
        "mutations": rows,
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        f"\n{out['n_proved']}/{out['n_mutations']} mutations proved caught; "
        f"baseline green={out['baseline_all_green']}; restored={restored_ok}"
    )
    print(f"wrote {OUT}")
    return 0 if (out["n_proved"] == out["n_mutations"] and restored_ok and base.returncode == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
