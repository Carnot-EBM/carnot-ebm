#!/usr/bin/env python3
"""Refuse a commit that widens the fabrication gate's no-LLM allowlist without evidence.

REQ-SUBSTRATE-ALIAS-1 (openspec/capabilities/research-harnesses/spec.md).

WHY THIS EXISTS. `adversarial_verify.py` skips its DURATION_TOO_SHORT check for
any artifact whose declared `inference_substrate` appears in
`NO_LLM_SUBSTRATE_ALIASES`. Nothing governed who may add a name to that tuple.
Measured 2026-08-23: 38 aliases, 19 added in two days, and every sampled
addition landed in the SAME commit as the artifact it exempted (for example
commit 59c8f8602d, which registered the alias at 22:24Z and wrote
experiment_6520 at 22:25Z).

That means an experiment that trips the fabrication gate could clear itself by
naming its own substrate in the gate's allowlist, with no test and no review.
The layer that decides whether every other result counts was being edited by
the results it judges.

WHAT IT ASKS FOR. A new alias must arrive with one of two things:

  1. a test under tests/python/ that names the alias string, or
  2. a dated line in ops/substrate_alias_acks.md saying why that substrate
     genuinely invokes no LLM.

Either is cheap. Neither is automatic, which is the point -- someone has to
state the reason in a place a reader can find it.

WHAT IT DOES NOT ASK FOR. Removing an alias, renaming a variable, or editing
any other part of the file is fine. The requirement is on WIDENING the gate.

FAIL DIRECTION: closed. If the staged diff cannot be read, this refuses rather
than passing -- a guard that returns "clean" when it could not look is the
worst state a guard can be in (CLAUDE.md, QA-Layer Authenticity Discipline).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GATE_FILE = "scripts/adversarial_verify.py"
ACK_FILE = "ops/substrate_alias_acks.md"
TEST_DIR = "tests/python"

# The alias shape the gate consults. Deliberately the STRING LITERAL form: a
# constant name (LOCAL_COMPACT_..._SUBSTRATE) can be renamed freely, but the
# literal is what an artifact's inference_substrate field is compared against.
# EITHER quote style: the first ship matched double quotes only, and ruff
# format normalises this repo to double quotes -- so a hand-edit written as
# 'local_python_no_llm' walked straight through (QA-layer SILENT_NON_FIRING
# finding, 2026-08-23). The backreference anchors on the quote PAIR, so an
# apostrophe in prose cannot open a match that a later apostrophe closes
# unless they wrap exactly an alias-shaped token.
ALIAS_RE = re.compile(r'(["\'])([a-z0-9_]*_no_llm)\1')


def find_alias_literals(text: str) -> list[str]:
    """Alias literals in `text`, either quote style, in order of appearance."""
    return [m.group(2) for m in ALIAS_RE.finditer(text)]


class GitUnavailable(RuntimeError):
    """Raised when the staged diff cannot be read. Callers must refuse, not pass."""


def new_aliases(diff_text: str, head_text: str) -> list[str]:
    """Alias literals ADDED by this diff that HEAD did not already contain.

    Subtracting HEAD's aliases matters because one alias normally appears twice
    in a real change -- once as a module constant, once inside the tuple -- and
    because reformatting can re-add a line that was never new.
    """
    already = set(find_alias_literals(head_text))
    added: list[str] = []
    for line in diff_text.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for alias in find_alias_literals(line):
            if alias not in already and alias not in added:
                added.append(alias)
    return added


def find_evidence(alias: str, test_texts: dict[str, str], ack_text: str) -> str | None:
    """Return a human-readable evidence location, or None when there is none."""
    if alias in ack_text:
        return ACK_FILE
    for name, body in test_texts.items():
        if alias in body:
            return name
    return None


def _run_git(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # git missing entirely
        raise GitUnavailable(f"could not run git: {exc}") from exc
    if proc.returncode != 0:
        raise GitUnavailable(
            f"git {' '.join(args)} failed rc={proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout


def collect_evidence_texts() -> tuple[dict[str, str], str]:
    """Read every test file and the ack file. Missing ack file is not an error."""
    tests: dict[str, str] = {}
    test_root = PROJECT_ROOT / TEST_DIR
    if test_root.is_dir():
        for path in sorted(test_root.rglob("*.py")):
            try:
                tests[str(path.relative_to(PROJECT_ROOT))] = path.read_text(encoding="utf-8")
            except OSError:
                continue
    ack_path = PROJECT_ROOT / ACK_FILE
    ack_text = ""
    if ack_path.is_file():
        try:
            ack_text = ack_path.read_text(encoding="utf-8")
        except OSError:
            ack_text = ""
    return tests, ack_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # pre-commit passes the staged filenames; we ignore them and ask git
    # directly, because a rename or a partial stage would otherwise hide the
    # addition.
    parser.add_argument("files", nargs="*", help="ignored; the staged diff is authoritative")
    args = parser.parse_args(argv)
    del args

    try:
        diff = _run_git(["diff", "--cached", "-U0", "--", GATE_FILE])
        if not diff.strip():
            return 0
        head = _run_git(["show", f"HEAD:{GATE_FILE}"])
    except GitUnavailable as exc:
        print(f"substrate-alias-evidence-lint: REFUSING -- {exc}", file=sys.stderr)
        print("  Cannot verify whether this commit widens the fabrication gate.", file=sys.stderr)
        return 1

    aliases = new_aliases(diff, head)
    if not aliases:
        return 0

    tests, ack_text = collect_evidence_texts()
    unsupported = [a for a in aliases if find_evidence(a, tests, ack_text) is None]

    for alias in aliases:
        where = find_evidence(alias, tests, ack_text)
        if where:
            print(f"substrate-alias-evidence-lint: {alias} -- evidence in {where}")

    if not unsupported:
        return 0

    print("", file=sys.stderr)
    print("substrate-alias-evidence-lint: REFUSING (REQ-SUBSTRATE-ALIAS-1)", file=sys.stderr)
    print("", file=sys.stderr)
    print("This commit widens the fabrication gate's no-LLM allowlist. Each new", file=sys.stderr)
    print("alias exempts artifacts from the DURATION_TOO_SHORT check, so it needs", file=sys.stderr)
    print("evidence a reader can find. Missing evidence for:", file=sys.stderr)
    for alias in unsupported:
        print(f"    {alias}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Supply ONE of:", file=sys.stderr)
    print(f"  - a test under {TEST_DIR}/ that names the alias string, or", file=sys.stderr)
    print(f"  - a dated line in {ACK_FILE} saying why this substrate runs no LLM.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
