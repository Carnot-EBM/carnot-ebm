#!/usr/bin/env python3
"""Refuse any commit whose staged Python carries a mutation marker.

WHY THIS EXISTS (REQ-OPS-MUTATION-PROOF-2, 2026-08-25). On 2026-08-25 two
hand-run mutation proofs ran against one working tree and the tree carried

    python/carnot/agentic/arc_executable_world_model.py:6466: pass  # MUTATED M6

on the LIVE ARC scored path. `pass  # MUTATED M6` is valid Python. It parses,
it imports, it passes ruff, mypy and every other hook in this repo, and the
conductor commits `git add -A` with hooks skipped on its own schedule. A
checkpoint firing inside a mutation window publishes it silently.

WHY THIS AND NOT ONLY THE SESSION WRAPPER. The sibling mechanism -- the
`--mutation-begin` / `--mutation-end` session in test_suite_mutation_check.py --
is OPT-IN. An agent has to remember to wrap. A wrapper nobody calls is this
project's own named bug class: `--check-targets` shipped with no caller and ran
only when a human remembered to type it, which CLAUDE.md records as
trust-without-verification one level up from the thing it was written to defeat.

So this hook does not care whether a session was used. It watches for the ACTUAL
harm -- a mutated line reaching a commit -- and refuses regardless of how the
line got there.

RESIDUAL, stated rather than implied: a commit that skips hooks bypasses this
entirely, and that is exactly how the observed incident would have landed. The
two mechanisms are complements, not alternatives. The session wrapper catches a
marker the committer never tried to commit; this hook catches one nobody
noticed. Neither catches a deliberate hook-skipping commit.

FAIL DIRECTION: closed everywhere. An unqueryable git, an unreadable file, or a
file that will not decode all REFUSE. A guard that answers "clean" when it could
not look is the failure this repo keeps re-learning.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: Where THIS script lives. Deliberately separate from REPO, which tests point
#: at a throwaway tree: the marker's definition is always beside this file, and
#: resolving it through REPO made every test raise FileNotFoundError.
_SCRIPTS = Path(__file__).resolve().parent


def _marker() -> str:
    """The marker token, imported from the module that defines it.

    One list, one home. A second copy here is how a lint silently stops
    matching the convention it was written for.
    """
    spec = importlib.util.spec_from_file_location(
        "test_suite_mutation_check", _SCRIPTS / "test_suite_mutation_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("test_suite_mutation_check", module)
    spec.loader.exec_module(module)
    return module.MUTATION_MARKER


#: Word-START boundary only. `\bMUTATED` matches `MUTATED`, `MUTATED_M6` and
#: `MUTATEDX`, and does NOT match `PERMUTATED`. Requiring a word END too would
#: let `# MUTATED_M6` through, and the fail direction here is toward catching.
def _pattern(marker: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(marker)}")


#: Files that must contain the literal token to DEFINE or TEST it. Four entries,
#: each named because it cannot do its job without the word. Everything else in
#: the repo is scanned -- this is an allow-list, not a directory filter.
ALLOWLIST = {
    "scripts/mutation_marker_lint.py",
    "scripts/test_suite_mutation_check.py",
    "tests/python/test_mutation_marker_lint.py",
    "tests/python/test_test_suite_mutation_check.py",
}


class LintError(RuntimeError):
    """Something could not be determined. Always becomes a refusal."""


def _staged_blob(rel: str) -> str | None:
    """The STAGED bytes of a path, or None when it is not in the index.

    Reads the index rather than the working tree because the question is what
    the COMMIT will contain. Under pre-commit the two agree (it stashes unstaged
    work), but this tool is also run by hand, and then they do not.
    """
    result = subprocess.run(
        ["git", "show", f":{rel}"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace")
        if "does not exist" in stderr or "exists on disk, but not in" in stderr:
            return None
        raise LintError(f"git could not read staged {rel}: {stderr.strip()[:200]}")
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LintError(f"staged {rel} is not valid UTF-8: {exc}") from exc


def _read(rel: str, path: Path, source: str) -> str:
    """The text to scan for one candidate file.

    `source="index"` is the production question -- what will the COMMIT
    contain -- and costs one `git show` per file, which is fine for a staged
    set and far too slow for a whole-repo sweep. `source="worktree"` exists for
    that sweep. Anything unreadable RAISES; it is never treated as empty.
    """
    if source == "index":
        blob = _staged_blob(rel)
        if blob is not None:
            return blob
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise LintError(f"{rel} could not be read: {exc}") from exc


def scan(paths: list[Path], marker: str | None = None, source: str = "index") -> list[str]:
    """`rel:line: text` for every mutation marker in the given Python files."""
    pattern = _pattern(marker or _marker())
    hits: list[str] = []
    for path in paths:
        try:
            rel = str(path.resolve().relative_to(REPO))
        except ValueError:
            rel = str(path)
        if path.suffix != ".py" or rel in ALLOWLIST:
            continue
        for number, line in enumerate(_read(rel, path, source).splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{rel}:{number}: {line.strip()[:120]}")
    return hits


def staged_python() -> list[Path]:
    """Every Python file staged for commit."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise LintError(f"git could not list staged files: {result.stderr.strip()[:200]}")
    return [REPO / n for n in result.stdout.split("\0") if n.endswith(".py")]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="paths to check (default: staged Python)")
    args = parser.parse_args(argv)

    try:
        paths = [Path(f) for f in args.files] if args.files else staged_python()
        hits = scan(paths)
    except LintError as exc:
        # Refuse rather than report clean. This hook's whole value is that it
        # answers a question about the commit; when it cannot, silence is a lie.
        print("mutation-marker-lint: REFUSING THE COMMIT -- the check could not run.")
        print(f"  {exc}")
        return 1

    if not hits:
        return 0

    print("mutation-marker-lint: REFUSING THE COMMIT.")
    print(
        f"  {len(hits)} mutation marker(s) in staged Python. A mutated line is valid Python:\n"
        "  it parses, imports, and clears every other hook, so nothing else would stop it\n"
        "  reaching the record. Restore the file before committing.\n"
    )
    for hit in hits:
        print(f"    {hit}")
    print(
        "\n  If this is a mutation proof in flight, finish it:\n"
        "    python3 scripts/test_suite_mutation_check.py --mutation-end --run-id <id>"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
