#!/usr/bin/env python3
"""Mutation proof for the depth/degeneracy split.

Each mutation REVERSES A DESIGN DECISION rather than mangling a character, and each must be
killed by the test suite. A mutation that survives is reported as surviving -- the point of the
exercise is to find the assertions that are decorative.

Restores every file byte-identically at the end and verifies the baseline is green both before
and after, so a crash mid-run cannot leave a mutated tree behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
PY = "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python"
GATE = os.path.join(REPO, "python/carnot/agentic/arc_llm_reinduction.py")
AGENT = os.path.join(REPO, "python/carnot/agentic/arc_competition_agent.py")

TESTS = [
    "tests/python/test_arc_goal_gate_depth_vs_degenerate_2026_07_31.py",
    "tests/python/test_arc_goal_gate_budget_vs_degenerate_2026_07_30.py",
    "tests/python/test_experiment_4664_l2_goal_predicate_induction_live.py",
    "tests/python/test_arc_win_state_positive_example_2026_07_29.py",
    "tests/python/test_goal_repair_degenerate_predicate.py",
]

MUTATIONS = [
    (
        "M1_never_count_a_depth_drop",
        GATE,
        "            depth_truncated_nodes += 1\n            continue",
        "            continue",
        "The counter is the whole mechanism. Without it every depth-capped search reverts to "
        "`degenerate_goal_predicate` -- the tn36 mislabel, restored.",
    ),
    (
        "M2_depth_cap_is_off_by_one",
        GATE,
        "        if depth >= int(max_depth):",
        "        if depth > int(max_depth):",
        "Expands one layer past the cap. Changes the SEARCH, not just the label, so tn36's "
        "1480/41 arithmetic no longer reproduces -- which is why the real-cell test asserts the "
        "raw counts and not only the kind.",
    ),
    (
        "CONTROL_inert_edit_must_SURVIVE",
        GATE,
        '        kind = "goal_unreached_within_depth"\n        termination = "depth_capped"',
        '        kind = "goal_unreached_within_depth"\n        termination = "depth_capped"\n'
        "        # MUTANT: relabelling became relaxing\n"
        "        _mutant_admit = True",
        "NOT A MUTATION -- a deliberate inert edit (an unused local) that MUST survive. Without "
        "it a 7/7 kill rate would be unfalsifiable: it proves the harness is capable of "
        "reporting SURVIVED at all, so the other six kills mean something.",
    ),
    (
        "M3b_depth_truncation_flips_satisfiable",
        GATE,
        '    return {\n        "satisfiable": False,\n'
        '        "reachable_grids_evaluated": int(evaluated),\n'
        '        "engine_calls": int(engine_calls),\n        "engine_errors": int(engine_errors),',
        '    return {\n        "satisfiable": bool(depth_truncated),\n'
        '        "reachable_grids_evaluated": int(evaluated),\n'
        '        "engine_calls": int(engine_calls),\n        "engine_errors": int(engine_errors),',
        "THE DANGEROUS ONE: turning a label fix into a gate relaxation. Admitting a goal the "
        "planner cannot reach is exactly the widening this commit promises it is not doing.",
    ),
    (
        "M4_restore_the_false_exhaustiveness_claim",
        GATE,
        'f"discarded unexpanded at max_depth={int(max_depth)}, so the reachable set was NOT "',
        'f"discarded unexpanded at max_depth={int(max_depth)}, so the reachable set was "',
        "The `detail` string WAS the damage -- the audit had to disbelieve it by hand. Dropping "
        "the negation restores a false claim while every kind/termination assertion stays green.",
    ),
    (
        "M5_plain_path_flattens_the_depth_kind",
        AGENT,
        '                            "goal_unreached_within_depth",\n',
        "",
        "The agent's allow-list is a whitelist. Silently dropping the new kind reinstates the "
        "mislabel in the artifact while the gate itself reports correctly -- the worst shape, "
        "because the two records then disagree.",
    ),
    (
        "M6_route_depth_away_from_goal_repair",
        GATE,
        '                == "goal_unreached_within_budget"',
        '                in ("goal_unreached_within_budget", "goal_unreached_within_depth")',
        "Treating depth-truncation like budget-exhaustion. It LOOKS principled (both are "
        "'undecided') but is a real behaviour change: GOAL-REPAIR stops firing on a case where "
        "the veto is earned and a reachable proxy is the productive fallback. Not shipped "
        "without a measurement.",
    ),
]


def run_tests() -> tuple[bool, str]:
    proc = subprocess.run(
        [PY, "-m", "pytest", *TESTS, "-q", "--no-cov", "-x"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-400:]


def main() -> int:
    originals = {p: pathlib.Path(p).read_bytes() for p in (GATE, AGENT)}
    digests = {p: hashlib.sha256(b).hexdigest() for p, b in originals.items()}

    ok, tail = run_tests()
    if not ok:
        print("BASELINE NOT GREEN -- aborting\n" + tail)
        return 1
    print("baseline: green")

    results = []
    try:
        for name, path, old, new, why in MUTATIONS:
            src = originals[path].decode()
            if src.count(old) != 1:
                results.append(
                    {"mutation": name, "status": "NOT_APPLIED", "occurrences": src.count(old)}
                )
                print(f"{name}: NOT APPLIED (found {src.count(old)} occurrences)")
                continue
            with open(path, "w") as fh:
                fh.write(src.replace(old, new, 1))
            passed, tail = run_tests()
            for p, b in originals.items():
                with open(p, "wb") as fh:
                    fh.write(b)
            status = "SURVIVED" if passed else "KILLED"
            results.append({"mutation": name, "status": status, "why": why, "tail": tail})
            print(f"{name}: {status}")
    finally:
        for p, b in originals.items():
            with open(p, "wb") as fh:
                fh.write(b)

    for p in (GATE, AGENT):
        assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() == digests[p], (
            f"{p} was not restored byte-identically"
        )
    ok, tail = run_tests()
    print("post-restore baseline:", "green" if ok else "RED\n" + tail)

    killed = sum(1 for r in results if r["status"] == "KILLED")
    applied = sum(1 for r in results if r["status"] != "NOT_APPLIED")
    real = [r for r in results if not r["mutation"].startswith("CONTROL_")]
    control_ok = all(
        r["status"] == "SURVIVED" for r in results if r["mutation"].startswith("CONTROL_")
    )
    summary = {
        "killed": killed,
        "applied": applied,
        "real_mutations_killed": (f"{sum(1 for r in real if r['status'] == 'KILLED')}/{len(real)}"),
        "control_behaved_correctly": bool(control_ok),
        "survived": [r["mutation"] for r in results if r["status"] == "SURVIVED"],
        "not_applied": [r["mutation"] for r in results if r["status"] == "NOT_APPLIED"],
        "baseline_green_before_and_after": bool(ok),
        "results": results,
    }
    # Write to the CANONICAL committed path, one level up. An earlier revision wrote
    # `harness/p5_mutation_check.json`, so a reader who re-ran this harness produced a file
    # that was NOT the record the spec cites -- the two agreed on every verdict but a
    # reproduction that lands somewhere else is not a reproduction anyone can check.
    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mutation_check.json"
    )
    pathlib.Path(out).write_text(json.dumps(summary, indent=2))
    print(f"\n{killed}/{applied} killed -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
