#!/usr/bin/env python3
"""A capstone must not validate its frozen MILESTONE against the LIVE roadmap.

WHY THIS EXISTS (2026-08-29). A capstone freezes a milestone identifier at authoring time --
`MILESTONE = "2026.08.580"`. If it then reads `research-roadmap.yaml` and demands the live file
still carry that milestone, it is green only while its own milestone is active and broken
forever afterwards. The roadmap advances every milestone, so this rots by construction.

That is not a local annoyance. `tests/python/test_experiment_6659_v580_capstone.py` called
`build_artifact` at MODULE scope, so the raise landed during COLLECTION, and pytest answers a
collection error by abandoning the whole run:

    ERROR tests/python/test_experiment_6659_v580_capstone.py - ValueError: expect...
    !!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!
    57917 tests collected, 1 error

One stale capstone took down all 57,917 tests, and every conductor task that shells out to
`pytest tests/python` failed with it -- exp6682's `verification_failure` among them.

THE CORRECT PATTERN ALREADY EXISTED, 23 TIMES. Capstones back to v469 compare
`artifact["milestone"] != MILESTONE` -- the milestone recorded in the artifact they are
building, which is self-consistent and cannot rot. Only two modules, both authored within three
days of each other, compared against the live file instead. This lint exists so that a
twenty-sixth capstone cannot rediscover the mistake: the repo had the answer and a new author
had no way to know.

WHAT IS ALLOWED. Reading the live roadmap is fine, and so is comparing to MILESTONE -- it is
the COMBINATION inside one module that rots. Recovering the archived roadmap from git history
(`_roadmap_payload_for_milestone`, the fix applied to both offenders) is explicitly fine: it
uses the live file while it still matches and falls back to history afterwards, so it never
rots.

Exit 0 clean, 1 on violation. Scoped to capstone modules; pass paths to check only those.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: A module is a capstone-shaped candidate when it freezes a milestone constant.
MILESTONE_CONST = re.compile(r'^MILESTONE\s*=\s*["\']20\d\d\.\d\d\.\d+["\']', re.M)

# NO GIT-RECOVERY EXEMPTION, deliberately (2026-08-29). One was written, and a mutation proof
# showed it was DECORATIVE: deleting it left the suite green, the repo clean, and both real
# offenders still caught. The reason is structural rather than lucky -- a
# `_roadmap_payload_for_milestone`-style helper raises at function level after exhausting git
# history, not inside an `if ... != MILESTONE`, so the raise-path rule never fires on it and no
# exemption is needed. Deleted rather than kept with a test built around it: a rule whose
# removal changes nothing is not protecting anything, and carrying it would imply the check
# depends on a list of blessed helper names that a future author must remember to join.


def _roadmap_alias_names(tree: ast.AST) -> set[str]:
    """Module-level constants bound to the live roadmap path.

    Both offenders read `ROADMAP_RELATIVE_PATH`, not the literal string, so a check that only
    looked for "research-roadmap.yaml" inside the function found nothing and was decorative --
    clean on the repo AND blind to the incident it was written for. Resolve the alias first.
    """

    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(c, ast.Constant)
            and isinstance(c.value, str)
            and "research-roadmap.yaml" in c.value
            for c in ast.walk(node.value)
        ):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _reads_roadmap(node: ast.AST, aliases: set[str]) -> bool:
    """Does this function read the live roadmap, by literal or by module constant?"""

    for n in ast.walk(node):
        if (
            isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and "research-roadmap.yaml" in n.value
        ):
            return True
        if isinstance(n, ast.Name) and n.id in aliases:
            return True
    return False


def _refuses_on_milestone(node: ast.AST) -> int | None:
    """Line where this body RAISES because the milestone does not match, if it does.

    Comparing is not the defect; REFUSING is. A capstone that records
    `"milestone_matches": roadmap.get("milestone") == MILESTONE` as a field is reporting
    honestly and never rots -- exp5917 and seven others do exactly that, and an earlier version
    of this lint flagged all eight. The rot is an `if ... != MILESTONE: raise` (or an assert),
    which turns a moved roadmap into a hard failure forever after.

    That distinction cost two false-positive rounds to find: first on variable name, then on
    comparison alone. Both would have shipped a check that cries wolf.
    """

    def _mentions_milestone(n: ast.AST) -> bool:
        return any(isinstance(c, ast.Name) and c.id == "MILESTONE" for c in ast.walk(n))

    for n in ast.walk(node):
        if isinstance(n, ast.Assert) and _mentions_milestone(n.test):
            return n.lineno
        if isinstance(n, ast.If) and _mentions_milestone(n.test):
            if any(isinstance(b, ast.Raise) for b in ast.walk(n)):
                return n.lineno
    return None


def violations(paths: list[Path]) -> list[tuple[Path, str]]:
    """Return (path, reason) for each module that will rot.

    SCOPED BY FUNCTION, NOT BY FILE, and that distinction is the whole lint. A first version
    matched `payload[...] != MILESTONE` anywhere in the file and flagged SIX correct capstones
    (5244, 5522, 5535, 5549, 5563, 5577), because their validators name the artifact under
    validation `payload` too -- comparing the ARTIFACT's milestone to the constant is the
    correct 23-instance pattern, and the variable name cannot tell the two apart. A lint with
    six false positives in a population of twenty-five is a check that cries wolf, which
    CLAUDE.md rightly calls worse than the gap it closes.

    The rot is specifically: ONE function both reads the live roadmap AND demands its milestone
    equal the frozen constant. That is checkable, and it is what this walks.
    """

    found: list[tuple[Path, str]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        if not MILESTONE_CONST.search(text):
            continue  # not a capstone-shaped module
        aliases = _roadmap_alias_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not _reads_roadmap(node, aliases):
                continue
            line = _refuses_on_milestone(node)
            if line is not None:
                found.append(
                    (
                        path,
                        f"{node.name}() at line {line}: this function reads the LIVE roadmap "
                        f"AND RAISES when its milestone is not the frozen MILESTONE constant, so "
                        f"it is green only during its own milestone. Compare the milestone "
                        f"recorded in the ARTIFACT being built (what 23 earlier capstones do), "
                        f"or recover the archived roadmap from git via a "
                        f"`_roadmap_payload_for_milestone`-style helper.",
                    )
                )
                break
    return found


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("-")]
    paths = (
        [Path(a) for a in args]
        if args
        else sorted((REPO / "python" / "carnot").glob("experiment_*capstone*.py"))
    )
    bad = violations([p for p in paths if p.suffix == ".py"])
    if not bad:
        print(f"capstone-milestone-rot-lint: OK ({len(paths)} module(s) checked)")
        return 0
    print("capstone-milestone-rot-lint: REFUSING.")
    print(
        "  A capstone that demands the LIVE roadmap still carry its frozen milestone is green\n"
        "  only while that milestone is active. One such module aborted the entire 57,917-test\n"
        "  suite at collection on 2026-08-29.\n"
    )
    for path, reason in bad:
        print(f"  {path}\n    {reason}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
