"""Refuse, loudly, any attempt by the test suite to WRITE an operator-curated document.

WHY THIS EXISTS (read this before changing anything here)
---------------------------------------------------------
CLAUDE.md's "Public Documentation Discipline" says a specific set of files -- `README.md`,
the landing page, the blog, the getting-started/CLI/MCP guides, `docs/CNAME` and friends --
are OPERATOR-CURATED. The autonomous loop is forbidden from editing them at all. They are the
project's front door: a stranger judges Carnot's credibility by them, and they are the one
class of file where a silent machine edit is unambiguously wrong rather than merely untidy.

The rule was honour-discipline at the *commit* layer only. `scripts/operator_curated_docs_lint.py`
refuses a `[conductor]` commit that touches one of these paths -- but that lint only sees files
that reach `git add`. It cannot see a file that is rewritten, read back, and rewritten again on
every single test run, which is what was actually happening:

    scripts/experiment_1750.py, line 40-41:

        model_card_path = Path("README.md")     # <-- CWD-relative, NOT repo-relative
        model_card_path.write_text(model_card)  # <-- 6-line HuggingFace model card

`tests/python/test_experiment_1750.py` calls `run_experiment()` twice with no `chdir`, and
pytest's working directory is the repo root. So both tests silently replaced the operator's
hand-written `README.md` with a HuggingFace model card, on every suite run, and the tests
passed while doing it. The
`docs/research-notes/test-suite-rewrites-the-record-survey-2026-07-29.md` survey caught the
symptom (README.md in the rewritten set) but its per-test attribution ran under `-n 4`, so it
credited a *group* of concurrently-running test nodes -- naming
`test_experiment_209_cleanup.py::test_run_cleanup_rewrites_public_docs_with_provenance_labels`
first because of its suggestive name. That test is in fact correctly sandboxed: every one of
its four cases passes an explicit `--root tmp_path`. The real writer was `experiment_1750`.

That misattribution is the whole argument for this guard. A survey run under xdist can tell you
*that* the record moved but not reliably *who* moved it, and a wrong name sends the repair at
innocent code. A guard that fires at the moment of the write names the writer exactly, with a
stack trace, and cannot be fooled by concurrency.

WHY AN AUDIT HOOK, AND NOT A FIXTURE
------------------------------------
A fixture can only check state before and after a test; it cannot see the write itself, so it
inherits the same attribution ambiguity under `-n 4` and the same "the record was wrong for a
window" problem. A CPython audit hook (PEP 578) fires synchronously *inside* the `open()` call,
so:

  * the offending test fails with the write still on the stack -- the traceback IS the attribution
  * nothing is ever written, so there is no window in which the record on disk is wrong, and
    no revert step that could race another worker or clobber a human's concurrent edit
    (auto-reverting has already destroyed in-flight work twice in this project -- see
    `scripts/test_suite_mutation_check.py`'s `backup()` notes)

The cost is that an audit hook cannot be uninstalled once added, and it runs on every `open()`
in the process. Both are handled below: installation is idempotent, and the hot path exits on a
string comparison against a small basename set before doing any filesystem work.

Measured overhead (2026-07-29, this machine, 4,000 open+read calls): **0.97 us per open** --
about 32% on a loop that does nothing BUT open files, and nothing worth noticing in a suite
that also executes tests. A million opens would cost one second.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
  * It does not block READS. Plenty of tests legitimately read `README.md` to assert on its
    contents; that is the point of `tests/python/test_docs.py`.
  * It does not block writes OUTSIDE the repository. A test that copies `README.md` into
    `tmp_path` and runs a rewriter against the copy is doing exactly the right thing, and must
    keep working -- that is `test_experiment_209_cleanup.py`. Resolution is what distinguishes
    the two cases, so it must be done correctly (see `_canonical_repo_root`).
  * It does not police the non-curated research record (`results/**`, `openspec/**`). That is a
    much larger problem with a different shape, handled by
    `scripts/test_suite_mutation_check.py`'s commit-time interlock.
  * **It does not stop writes made by SUBPROCESSES.** This is the largest hole and the one
    most likely to be misread as covered, so it is stated plainly.

    A PEP 578 audit hook lives in the interpreter that installed it. It is NOT inherited by
    child processes. So this passes straight through the guard::

        subprocess.run(["python", "scripts/publish_huggingface.py"])   # can destroy README.md
        subprocess.run("echo x > README.md", shell=True)               # likewise

    Verified empirically, not assumed: with the guard installed, both a `sh -c` redirect and a
    child `python -c` open() succeeded and really did modify README.md.

    This matters at scale here -- roughly 184 test files spawn subprocesses and about 159
    reference `scripts/`, so a large fraction of the suite's write surface is unguarded. That
    is precisely why the three known CWD-relative `Path("README.md")` writers were ALSO fixed
    at source (`scripts/experiment_1750.py`, `scripts/publish_huggingface.py`,
    `python/carnot/pipeline/hf_publisher.py`): an in-process guard alone would have left the
    subprocess and the run-it-by-hand paths exposed. Fixing the writer protects every caller;
    fixing the caller protects one.

    Closing the gap generally would mean getting the hook into child interpreters too -- e.g. a
    `sitecustomize.py` on `PYTHONPATH`, or an env flag the child honours on startup. Both are
    process-global and would affect unrelated subprocesses (including long-running research
    jobs), so neither was switched on as a side effect of this change. It is recorded as
    follow-up work rather than done quietly.

    Until then the honest summary is: **in-process writes are blocked; subprocess writes are
    not.** A green run of the guard is not proof that nothing wrote to an operator-curated doc
    -- `scripts/test_suite_mutation_check.py --check` and `git status` remain the backstop that
    catches what this cannot see.
"""

from __future__ import annotations

import fnmatch
import os
import sys
import traceback
from pathlib import Path

__all__ = [
    "OPERATOR_CURATED_PATHS",
    "OperatorCuratedDocWriteError",
    "install",
    "is_installed",
    "recorded_violations",
    "clear_violations",
    "repo_root",
]


# Kept byte-identical to `scripts/operator_curated_docs_lint.py:OPERATOR_CURATED_PATHS`.
#
# It is duplicated rather than imported ON PURPOSE. Importing it would mean putting the repo
# root on `sys.path` from inside `python/carnot/`, and that specific hack is itself a known
# defect in this tree -- it poisons an entire xdist worker so that even correctly repo-relative
# scripts resolve against the operator's checkout. Duplicating a 16-entry tuple is the smaller
# evil, and `tests/python/test_operator_curated_doc_guard.py` asserts the two lists stay equal,
# so the duplication cannot silently drift.
OPERATOR_CURATED_PATHS: tuple[str, ...] = (
    "README.md",
    "NOTICE",
    "LICENSE",
    "docs/index.html",
    "docs/roadmap.md",
    "docs/research-log.md",
    "docs/blog/*.html",
    "docs/blog/**/*.html",
    "docs/getting-started.md",
    "docs/cli-usage.md",
    "docs/mcp-server.md",
    "docs/tutorial.md",
    "docs/concepts.md",
    "docs/api-reference.md",
    "docs/CNAME",
    "docs/arxiv-paper/main.tex",
)


class OperatorCuratedDocWriteError(RuntimeError):
    """Raised at the instant a test tries to write an operator-curated document."""


def _canonical_repo_root() -> Path:
    """Return the repository root as its CANONICAL path, with symlinks resolved.

    This resolution is the whole correctness argument for the guard, so it is worth being
    explicit about why `.resolve()` is not optional here.

    This checkout is reachable under two names: the real directory
    `/home/ianblenke/github.com/ianblenke/carnot`, and a symlink alias
    `/home/ianblenke/github.com/Carnot-EBM/carnot-ebm` (the venv lives under the alias, so
    plenty of processes run with the alias as their prefix). Two different strings, one
    inode. If the guard compared unresolved strings, a write arriving via the alias would not
    match the watch list and would sail straight through -- the guard would appear to work and
    silently not.

    `Path.resolve()` collapses both names to the same canonical path, so a write is caught
    regardless of which door it came in by. The same call is applied to the candidate path in
    `_violation_for`, so the comparison is canonical on both sides.

    `parents[3]` walks python/carnot/testing/<this file> -> carnot -> python -> repo root.
    """
    return Path(__file__).resolve().parents[3]


_REPO_ROOT: Path = _canonical_repo_root()

# Absolute canonical paths for the literal (non-glob) entries, for an O(1) set membership test.
_WATCHED_ABSOLUTE: frozenset[str] = frozenset(
    str(_REPO_ROOT / p) for p in OPERATOR_CURATED_PATHS if "*" not in p
)
# The glob entries, kept as repo-relative patterns and matched with fnmatch.
_WATCHED_GLOBS: tuple[str, ...] = tuple(p for p in OPERATOR_CURATED_PATHS if "*" in p)

# Hot-path prefilter. The audit hook runs on EVERY open() in the process, so before touching
# the filesystem we reject on a cheap basename comparison. Only paths whose final component is
# one we care about -- or that live somewhere under a "blog" directory, covering the two glob
# entries -- pay the cost of resolution.
_WATCHED_BASENAMES: frozenset[str] = frozenset(p.rpartition("/")[2] for p in OPERATOR_CURATED_PATHS)

_installed = False
_violations: list[dict[str, str]] = []


def repo_root() -> Path:
    """The canonical repository root the guard is protecting."""
    return _REPO_ROOT


def is_installed() -> bool:
    """True once `install()` has added the audit hook to this process."""
    return _installed


def recorded_violations() -> list[dict[str, str]]:
    """Every violation seen so far, as ``{"path", "event", "stack"}`` dicts.

    This exists so the guard survives a test that swallows exceptions. Raising from the audit
    hook makes `open()` fail, but nothing stops a test body from wrapping the call in a bare
    `except Exception: pass` and reporting green anyway -- and a guard that a careless test can
    silence is not a guard. `conftest.py` reads this ledger in `pytest_runtest_teardown` and
    fails the test on a non-empty result, so the only way to get a passing test is to not
    perform the write.
    """
    return list(_violations)


def clear_violations() -> None:
    """Reset the ledger. Called per-test by conftest so a failure is attributed to one test."""
    _violations.clear()


def _is_write_intent(mode: object, flags: object) -> bool:
    """True when an `open` audit event describes a write, not a read.

    The `open` audit event carries `(path, mode, flags)`, and which of `mode`/`flags` is
    populated depends on the caller: `builtins.open()` supplies the mode STRING, while
    `os.open()` supplies an integer flag mask and leaves mode as None. Both forms have to be
    understood or the guard misses half the ways Python can write a file.
    """
    if isinstance(mode, str):
        return any(c in mode for c in "wxa+")
    if isinstance(flags, int):
        write_bits = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
        return bool(flags & write_bits)
    # Neither form was usable. Fail SAFE (treat as a write): a false alarm on this path is a
    # loud, fixable test failure, whereas a missed write silently corrupts the front door.
    return True


def _violation_for(path: object, *, allow_relative: bool) -> str | None:
    """Return the canonical path string if `path` is a protected doc, else None.

    `allow_relative` decides whether a RELATIVE path may be resolved against the process's
    current working directory, and getting it wrong breaks the guard in one direction or the
    other. Both directions were observed:

    * `allow_relative=True` is REQUIRED for the `open` event, because the incident this guard
      exists for is precisely a relative write -- `Path("README.md").write_text(...)` executed
      with the repo root as the working directory. Refusing to resolve relative paths here
      would make the guard blind to the exact bug it was written to catch.

    * `allow_relative=False` is REQUIRED for the `os.*` mutation events, because several of
      them are FD-RELATIVE and the audit event does not say so. `shutil.rmtree` walks a tree
      with `_rmtree_safe_fd` and calls `os.unlink(entry.name, dir_fd=topfd)`; the `os.remove`
      audit event then carries the BARE NAME `"README.md"` with no indication that it is
      relative to a directory file descriptor pointing at some `tmp_path` sandbox. Resolving
      that against the working directory produces `<repo>/README.md` and fires a false alarm.

      This is not hypothetical: it fired on eight tests the first time this guard ran --
      including `test_experiment_209_cleanup.py`, which is *correctly* sandboxed and whose only
      crime was that pytest cleaned up a `tmp_path` containing a `README.md` copy. A guard that
      refuses honest work gets disabled by the first person it inconveniences, which leaves the
      record unprotected -- the exact outcome it exists to prevent.

    The residual gap is a WRITE-intent `os.open(name, ..., dir_fd=fd)` whose bare name happens
    to be a watched one; that would still be resolved against the working directory and could
    false-fire. It is accepted deliberately: `dir_fd` is used almost exclusively by tree-walking
    code, whose opens are read-only (and so never reach this function), whereas the CWD-relative
    write is a bug that has actually happened in this repository.
    """
    if isinstance(path, bytes):
        try:
            path = path.decode("utf-8", "surrogateescape")
        except Exception:  # noqa: BLE001 - an undecodable path cannot be one of ours
            return None
    if not isinstance(path, str) or not path:
        return None

    # Cheap prefilter -- see _WATCHED_BASENAMES. Everything below this line is cold.
    basename = path.rpartition("/")[2]
    if basename not in _WATCHED_BASENAMES and "blog" not in path:
        return None

    if not allow_relative and not os.path.isabs(path):
        return None

    try:
        # strict=False so a not-yet-existing file (the common case for a write) still resolves.
        resolved = Path(path).resolve(strict=False)
    except (OSError, ValueError, RuntimeError):
        return None

    resolved_str = str(resolved)
    if resolved_str in _WATCHED_ABSOLUTE:
        return resolved_str

    if _WATCHED_GLOBS:
        try:
            relative = resolved.relative_to(_REPO_ROOT).as_posix()
        except ValueError:
            return None  # outside the repo -- a tmp_path copy, which is legitimate
        for pattern in _WATCHED_GLOBS:
            if fnmatch.fnmatch(relative, pattern):
                return resolved_str
    return None


def _record_and_raise(resolved: str, event: str) -> None:
    relative = os.path.relpath(resolved, str(_REPO_ROOT))
    _violations.append(
        {
            "path": relative,
            "event": event,
            # The stack is captured here, at the write, because that is the attribution the
            # xdist-merged survey could not produce.
            "stack": "".join(traceback.format_stack(limit=12)),
        }
    )
    raise OperatorCuratedDocWriteError(
        f"Test attempted to {event} the operator-curated document {relative!r}.\n"
        f"\n"
        f"CLAUDE.md 'Public Documentation Discipline' forbids the autonomous loop from editing\n"
        f"this file at all; a test that rewrites it corrupts the project's front door on every\n"
        f"suite run, and passes while doing it.\n"
        f"\n"
        f"Almost always the cause is a CWD-relative path in the script under test, e.g.\n"
        f"    Path('README.md').write_text(...)\n"
        f"which resolves against the repo root because that is pytest's working directory.\n"
        f"\n"
        f"Fix it by redirecting the write, NOT by deleting the test:\n"
        f"  * `monkeypatch.chdir(tmp_path)` if the script writes CWD-relative (keeps the real\n"
        f"    script fully executed -- no coverage is lost), or\n"
        f"  * pass an explicit root/output directory if the script accepts one, or\n"
        f"  * `monkeypatch.setenv('CARNOT_REPO_ROOT', str(tmp_path))` for scripts that honour it.\n"
    )


def _audit_hook(event: str, args: tuple) -> None:
    """The PEP 578 hook. Must stay cheap: it runs on every open() in the process."""
    if event == "open":
        if len(args) < 3:
            return
        if not _is_write_intent(args[1], args[2]):
            return
        # allow_relative=True: the incident mechanism is a CWD-relative write. See _violation_for.
        hit = _violation_for(args[0], allow_relative=True)
        if hit is not None:
            _record_and_raise(hit, "write")
        return

    # Atomic-write and move patterns never call open() on the destination, so they need their
    # own events or the guard has an obvious hole: write a temp file, then rename it over the
    # protected path.
    #
    # allow_relative=False on all of these: they can be fd-relative, and CWD-resolving a bare
    # name from `shutil.rmtree`'s fd-relative walk is what made this guard false-fire on eight
    # correctly-sandboxed tests. See _violation_for for the full argument.
    if event in ("os.rename", "os.replace", "shutil.move", "shutil.copyfile", "shutil.copy2"):
        if len(args) < 2:
            return
        hit = _violation_for(args[1], allow_relative=False)
        if hit is not None:
            _record_and_raise(hit, event.rpartition(".")[2] + " onto")
        return

    if event in ("os.remove", "os.unlink", "os.truncate"):
        if not args:
            return
        hit = _violation_for(args[0], allow_relative=False)
        if hit is not None:
            _record_and_raise(hit, event.rpartition(".")[2])


def install() -> bool:
    """Install the audit hook. Idempotent; returns True if this call installed it.

    Idempotence matters because `pytest_configure` runs once in the xdist controller AND once
    per worker, and an audit hook can never be removed once added -- installing it four times
    would quadruple the per-open cost for the whole session with no added protection.
    """
    global _installed
    if _installed:
        return False
    sys.addaudithook(_audit_hook)
    _installed = True
    return True
