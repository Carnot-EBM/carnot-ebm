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

# RECOVERY IS RECOGNISED BY SHAPE, NOT BY A NAME OR A LITERAL (2026-09-05). The 2026-08-29
# rule only looked INSIDE the `if ... != MILESTONE` for a raise, so a helper that raises at
# function level after the `if` never matched. A QA-layer audit named the missed input:
# `if payload.get("milestone") == MILESTONE: return payload` followed by a sibling `raise`
# rots exactly like the inline form. The rule now catches that shape -- UNLESS a fallback
# path can still return between the guard and the raise (a git walk, an archive on disk,
# any second source). A first draft exempted any function containing the literal "git";
# an adversarial review showed that both misses a rotter that merely mentions "git" and
# refuses a recoverer that reads an archive file with no git at all. The only name-keyed
# exemption left is a replay helper that pins the input to the closing commit BEFORE the
# guard, which leaves no fallback to detect. See commit history for both incidents.
_REPLAY_HELPERS = frozenset({"_replay_bytes", "receipt_bytes"})


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
    # The SIBLING-RAISE shape (2026-09-05): `if ... == MILESTONE: return payload` as a guard,
    # then a `raise` further down the same statement list. It refuses on a moved roadmap just
    # like the inline form, but the raise sits BESIDE the `if`, not inside it, so the walk
    # above never saw it. A QA-layer audit named this exact input; the lint exited 0 on it.
    # A `return` anywhere BETWEEN the guard and that raise is a fallback path that can still
    # succeed (an archived copy from git or from disk), so the function recovers, not rots.
    for statements in _statement_lists(node):
        for index, statement in enumerate(statements):
            if not _is_milestone_guard(statement):
                continue
            for later_index in range(index + 1, len(statements)):
                later = statements[later_index]
                if not isinstance(later, ast.Raise):
                    continue
                between = statements[index + 1 : later_index]
                if any(isinstance(b, ast.Return) for s in between for b in ast.walk(s)):
                    break  # recovery: something between the guard and the raise can return
                return later.lineno
    return None


def _is_milestone_guard(statement: ast.stmt) -> bool:
    """An `if` whose test mentions MILESTONE.

    Whether its body returns is deliberately NOT required: a bare `raise` later in the same
    statement list, with no return between, means the function cannot get past that point
    without raising, whatever the `if` body did. A first draft required the return and a
    mutation proof showed the clause decorative. What the MILESTONE mention protects is the
    ordinary schema guard (`if not tasks: return ...` then `raise`), which is not the rot.
    """

    return isinstance(statement, ast.If) and any(
        isinstance(c, ast.Name) and c.id == "MILESTONE" for c in ast.walk(statement.test)
    )


def _has_guard_then_sibling_raise(node: ast.AST) -> bool:
    """Does the function hold a milestone guard with a raise later in the same list, at all?

    Used by tests to prove the recovery reading is exercised: a helper that HAS this shape
    and is still clean must have a fallback return between the two, or a replay helper.
    """

    for statements in _statement_lists(node):
        for index, statement in enumerate(statements):
            if _is_milestone_guard(statement) and any(
                isinstance(later, ast.Raise) for later in statements[index + 1 :]
            ):
                return True
    return False


def _statement_lists(node: ast.AST) -> list[list[ast.stmt]]:
    """Every statement list inside a function: its body and each nested block."""

    lists: list[list[ast.stmt]] = []
    for n in ast.walk(node):
        for attr in ("body", "orelse", "finalbody"):
            block = getattr(n, attr, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                lists.append(block)
    return lists


def _recovers_from_history(node: ast.AST) -> bool:
    """Does this function pin its roadmap bytes through a replay helper?

    `_replay_bytes` / `receipt_bytes` read the input at the commit that closed the milestone,
    BEFORE the guard, so there is no fallback path after it to detect; the name is the only
    evidence. Every other recovery (git walk, archive on disk) is recognised by shape in
    `_refuses_on_milestone`: a return between the guard and the raise. Residual, stated: a
    rotter that returns from an unrelated branch after its guard reads as a recoverer.
    """

    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            func = n.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in _REPLAY_HELPERS:
                return True
    return False


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
        except (OSError, SyntaxError) as exc:
            # FAIL CLOSED (2026-09-05). This used to `continue`, so a module that could not be
            # read or parsed was reported clean. Unreadable is not the same as clean: a guard
            # that says "OK" about a file it never looked at is trusted and silent.
            found.append((path, f"could not be read or parsed ({type(exc).__name__}: {exc})"))
            continue
        if not MILESTONE_CONST.search(text):
            continue  # not a capstone-shaped module
        aliases = _roadmap_alias_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not _reads_roadmap(node, aliases):
                continue
            if _recovers_from_history(node):
                continue  # a replay-pinned helper cannot rot; see the note at the top
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
