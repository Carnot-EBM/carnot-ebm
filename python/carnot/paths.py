"""The ONE place that decides where this repository lives, and where output goes.

WHY THIS MODULE EXISTS (read this before adding another way to find the repo root)
---------------------------------------------------------------------------------
Before this module, 232 files across ``scripts/``, ``python/carnot/`` and ``tests/``
each answered the question "where is the repo root?" on their own. The two dominant
answers were both wrong in a way that only shows up on someone else's machine:

1. ``PROJECT_ROOT = "/home/ianblenke/github.com/ianblenke/carnot"`` -- a hardcoded
   absolute path to ONE developer's checkout. A fresh clone anywhere else still
   writes its results into that developer's tree. That is a reproducibility defect,
   not a style nit: an independent reproducer cannot run the experiment without
   silently contaminating (or being contaminated by) the original author's files.

2. ``Path(os.getcwd())`` or a bare relative path like ``Path("README.md")`` -- which
   depends on where the process happened to be started. Under pytest the working
   directory is the repo root, so a script that "just writes a relative path" writes
   into tracked, operator-curated files. That is how ``README.md`` came to be
   rewritten by a passing test.

Both failure modes are silent. Nothing errors; the write just lands somewhere it
should not. Centralising the decision here means there is exactly one behaviour to
audit, one behaviour to test, and one place to fix if it is ever wrong again.

THE SYMLINK TRAP (the specific thing that makes ``os.getcwd()`` unsafe here)
---------------------------------------------------------------------------
This checkout is reachable under two different absolute paths::

    /home/ianblenke/github.com/ianblenke/carnot            <- the real directory
    /home/ianblenke/github.com/Carnot-EBM/carnot-ebm       <- a chain of symlinks to it

They are the same bytes on disk, but they are *different strings*.

Be precise about HOW the alias leaks in, because the obvious guess is wrong:
``os.getcwd()`` is a syscall that returns the kernel's canonical path and therefore
NEVER contains a symlink -- a process started under the alias still reports
``/home/ianblenke/github.com/ianblenke/carnot`` from ``os.getcwd()`` (measured, not
assumed). ``Path.cwd()`` is built on it and behaves the same.

The two vectors that really do carry the alias spelling are:

1. ``$PWD`` -- the *shell's* idea of the working directory. The shell tracks the path
   you typed, symlinks and all, so a script that trusts ``$PWD`` (or anything derived
   from it, including some ``os.path.abspath`` uses on a relative argument) sees the
   alias.
2. ``__file__`` -- if the interpreter was pointed at a script *through* the alias
   (``python /home/ianblenke/github.com/Carnot-EBM/carnot-ebm/scripts/x.py``), then
   ``__file__`` is the alias spelling, and walking up from it unresolved lands on the
   alias's ancestors.

Vector 2 is the one this module is directly exposed to, since root detection starts
from a caller's ``__file__``. It is handled by resolving the start point BEFORE
walking up (see ``repo_root``). ``Path.resolve()`` follows symlink *chains*, which
matters here because the alias points at another symlink rather than at the target
directly.

This module always returns the CANONICAL (fully symlink-resolved) path, so both entry
points agree.

HOW THE ROOT IS FOUND
---------------------
In priority order:

1. ``$CARNOT_REPO_ROOT`` if set. This is the override that lets a test redirect every
   write into a ``tmp_path`` sandbox, and lets a git worktree keep its output to
   itself. The name is not new -- it is the convention already used by ~52 scripts
   and ~56 test files, so this module adopts it rather than inventing a rival.

2. Otherwise, walk UP from the calling file looking for a ``.git`` marker. This is
   the "walk up to the git root" rule. We look for the marker directly instead of
   shelling out to ``git rev-parse`` because a subprocess costs ~10ms per call, can
   fail in a sandbox with no git binary, and is a surprising thing for a path lookup
   to do.

   ``.git`` is accepted as either a directory (normal clone) or a FILE (git worktrees
   and submodules write a ``gitdir:`` pointer file instead of a directory). Checking
   only for a directory would break every worktree, which is one of the two cases the
   env override above exists to serve.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not import anything outside the standard library -- no JAX, no numpy, no
carnot model code. Path resolution happens during module import in most callers, and
making "where do I write my output" depend on a heavy scientific stack would be both
slow and a new source of import cycles.

It does not decide the FILENAME of an artifact, only the directory it belongs in.
Naming stays with the experiment that owns the result.

It does not create ``results/`` for you unless you ask (``ensure=True``). A read-only
caller should not have the side effect of creating directories.

CALL TIME vs IMPORT TIME (a real limitation -- read this before relying on sandboxing)
--------------------------------------------------------------------------------------
``$CARNOT_REPO_ROOT`` is read on every call, not cached. That is what lets a test set
it and have subsequent writes land in a sandbox.

But a module that does this at module scope::

    DEFAULT_OUTPUT = repo_root() / "results" / "x.json"     # frozen at import

has already resolved the path by the time any test runs. Setting the override
afterwards (``monkeypatch.setenv``) CANNOT redirect it -- the constant still holds the
value computed when the module was first imported, and because ``sys.modules`` caches
the module, re-importing will not recompute it either. The sandbox silently does not
apply, which is the same class of silent no-op this module exists to eliminate.

Consequences, in order of preference:

* Prefer resolving inside the function that writes (``if path is None: path =
  results_path(...)``). This is what the migrated ``python/carnot/pipeline`` entry
  points do, and it is the only form that is sandboxable.
* If a module-scope constant is genuinely more readable, that is acceptable for code
  no test needs to redirect -- but it must be recorded as import-time in the migration
  follow-up list, so that a future attempt to sandbox it fails visibly (someone reads
  the list) rather than silently (the write escapes the sandbox).
* To sandbox an import-time constant anyway, the override must be set BEFORE the
  module is first imported -- e.g. in the environment of a subprocess, or via a
  session-scoped fixture that runs before the import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "REPO_ROOT_ENV",
    "repo_root",
    "results_dir",
    "output_dir",
    "repo_path",
    "results_path",
    "is_canonical_repo_root",
]

#: Environment variable that overrides root detection entirely.
#:
#: Set this to redirect every repo-relative write -- that is the supported way for a
#: test to sandbox output into ``tmp_path``, and for a worktree to avoid writing into
#: the primary checkout. Chosen because it is the name this repository already used
#: in ~108 places before centralisation; adopting the incumbent avoids a second
#: convention competing with the first.
REPO_ROOT_ENV = "CARNOT_REPO_ROOT"

#: Files/directories that mark a directory as the root of a git checkout.
#:
#: ``.git`` covers the normal clone (a directory) and worktrees/submodules (a file
#: containing a ``gitdir:`` pointer). We accept either kind of filesystem entry --
#: see the module docstring for why insisting on a directory silently breaks
#: worktrees.
_GIT_MARKER = ".git"

#: Secondary markers, used ONLY when no ``.git`` is found anywhere up the tree.
#:
#: This is the source-tarball / vendored-copy case: real code, no VCS metadata. We
#: require the marker file AND the directory layout this repo actually has, so we do
#: not mistake some unrelated parent directory that happens to contain a
#: ``pyproject.toml`` for the Carnot root.
_FALLBACK_MARKER_FILES = ("pyproject.toml",)
_FALLBACK_MARKER_DIRS = ("python", "scripts")


def _caller_file(depth: int = 2) -> str | None:
    """Return the ``__file__`` of the caller ``depth`` frames up, or None.

    WHY frame inspection rather than making every caller pass ``__file__``:
    ``repo_root()`` is intended to be a drop-in replacement for a hardcoded literal
    at ~200 call sites. Requiring an argument at each one is more churn and one more
    thing each site can get subtly wrong (passing the wrong module's ``__file__``
    after a copy-paste is an easy and invisible mistake).

    Callers that want no magic at all can always pass ``start=__file__`` explicitly,
    and anything running where a stack frame has no ``__file__`` -- a REPL, ``exec``
    of a string, a frozen binary -- simply falls through to the None branch, which
    the caller handles by using this module's own location instead.
    """
    try:
        frame = sys._getframe(depth)
    except ValueError:
        # Stack is shallower than requested (called from module scope of the
        # top-level script under some embedders). Not an error -- just unknown.
        return None
    filename = frame.f_globals.get("__file__")
    return str(filename) if filename else None


def _walk_up_for_marker(start: Path) -> Path | None:
    """Walk from ``start`` toward / looking for a git checkout root.

    Returns the FIRST directory containing a ``.git`` entry. "First" matters: if this
    repo is ever nested inside another checkout, we want the inner (own) root, not the
    outer one, so we must stop at the nearest marker rather than the furthest.
    """
    # ``start`` may be a file (a caller's __file__) or a directory. Normalise to a
    # directory before walking so the first candidate is meaningful either way.
    current = start if start.is_dir() else start.parent
    # ``[current, *current.parents]`` is the full chain up to the filesystem root and
    # is finite, so this cannot loop forever even on a pathological symlink layout
    # (resolve() has already collapsed symlinks by the time we get here).
    for candidate in (current, *current.parents):
        if (candidate / _GIT_MARKER).exists():
            return candidate
    return None


def _walk_up_for_fallback(start: Path) -> Path | None:
    """Walk up looking for the source-layout markers, for VCS-less copies.

    Only consulted when no ``.git`` exists anywhere above ``start``. Requires ALL of
    the expected layout to be present so an unrelated ancestor with a stray
    ``pyproject.toml`` cannot be mistaken for the Carnot root.
    """
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        has_file = any((candidate / name).is_file() for name in _FALLBACK_MARKER_FILES)
        has_dirs = all((candidate / name).is_dir() for name in _FALLBACK_MARKER_DIRS)
        if has_file and has_dirs:
            return candidate
    return None


def repo_root(start: str | Path | None = None) -> Path:
    """Return the canonical, symlink-resolved absolute path of the repository root.

    Args:
        start: A file or directory to search upward from. Defaults to the calling
            module's ``__file__``, which is what makes this correct for a script
            executed out of a worktree or a second clone: the answer follows the code,
            not the process's working directory.

    Returns:
        An absolute :class:`~pathlib.Path` with all symlinks resolved. Given this
        checkout, that is ``/home/ianblenke/github.com/ianblenke/carnot`` whether the
        caller arrived via the real path or via the ``Carnot-EBM/carnot-ebm`` alias.

    Raises:
        RuntimeError: If no repository root can be located. Failing loudly is
            deliberate. The alternative -- quietly returning the current working
            directory -- is precisely the behaviour that let tests write into tracked
            files, so an unlocatable root must be an error a human sees, not a default
            that keeps going.

    Note:
        ``$CARNOT_REPO_ROOT`` wins over everything, and is NOT required to look like a
        git checkout -- a test pointing it at an empty ``tmp_path`` is a valid and
        expected use, so validating it as a repo would defeat its main purpose.
    """
    override = os.environ.get(REPO_ROOT_ENV)
    if override:
        # ``strict=False`` (the default): the target need not exist yet. A test may
        # set this before creating the sandbox, and a missing directory here should
        # surface at the actual write, with that write's context, rather than as a
        # confusing failure inside path resolution.
        return Path(override).expanduser().resolve()

    if start is None:
        # depth=2: frame 0 is _caller_file, frame 1 is repo_root, frame 2 is the
        # caller we actually want.
        start = _caller_file(depth=2)

    # Fall back to this module's own location. Correct whenever carnot is imported
    # from a real checkout, which is the case for both editable installs and direct
    # ``python/`` sys.path use.
    origin = Path(start) if start else Path(__file__)

    # resolve() BEFORE walking. This is the load-bearing line for the symlink trap:
    # walking up from the unresolved alias path would find the alias's ancestors, so
    # the root would come back as the alias string and provenance would disagree with
    # a run started from the real path.
    origin = origin.expanduser().resolve()

    found = _walk_up_for_marker(origin) or _walk_up_for_fallback(origin)
    if found is None:
        raise RuntimeError(
            f"Could not locate the Carnot repository root above {origin!r}: no "
            f"{_GIT_MARKER!r} marker and no source layout "
            f"({_FALLBACK_MARKER_FILES} + {_FALLBACK_MARKER_DIRS}) was found in any "
            f"parent directory. Set ${REPO_ROOT_ENV} to point at the repository root "
            f"explicitly."
        )
    # Already canonical (we resolved the origin and only walked up real directories),
    # but resolve() again so the postcondition "the return value is fully resolved"
    # holds unconditionally rather than by argument.
    return found.resolve()


def repo_path(*parts: str | Path, start: str | Path | None = None) -> Path:
    """Join ``parts`` onto the repository root.

    Convenience for the common ``repo_root() / "results" / name`` shape. Keeping it
    here means a caller never has to decide between ``/`` and ``os.path.join`` when
    building a repo-relative path, and never accidentally builds one off ``os.getcwd``.
    """
    if start is None:
        start = _caller_file(depth=2)
    return repo_root(start=start).joinpath(*parts)


def results_dir(*, ensure: bool = False, start: str | Path | None = None) -> Path:
    """Return the ``results/`` directory -- the standard home for experiment artifacts.

    Args:
        ensure: Create the directory (and parents) if missing. Defaults to False so a
            merely-inspecting caller has no side effects; pass True from code that is
            about to write.

    ``results/`` is treated as the default artifact destination throughout this
    project, so centralising it here removes one more thing each experiment has to
    spell correctly.
    """
    if start is None:
        start = _caller_file(depth=2)
    path = repo_root(start=start) / "results"
    if ensure:
        path.mkdir(parents=True, exist_ok=True)
    return path


def results_path(
    name: str, *, ensure_parent: bool = False, start: str | Path | None = None
) -> Path:
    """Return the full path for a named artifact inside ``results/``.

    Args:
        name: The artifact filename, e.g. ``"experiment_1234_results.json"``. May
            include subdirectories (``"arc_e3/run.json"``); parents are created only
            when ``ensure_parent`` is True.
        ensure_parent: Create the containing directory before returning.

    This is the call an experiment should use to decide where its artifact goes.
    """
    if start is None:
        start = _caller_file(depth=2)
    path = results_dir(start=start) / name
    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(*, ensure: bool = False, start: str | Path | None = None) -> Path:
    """Return the ``output/`` directory (scratch/derived output, distinct from results).

    ``results/`` holds the experimental record and is treated as evidence; ``output/``
    holds regenerable derived files. They are separate directories because conflating
    them makes it impossible to say which files may safely be deleted or rebuilt.
    """
    if start is None:
        start = _caller_file(depth=2)
    path = repo_root(start=start) / "output"
    if ensure:
        path.mkdir(parents=True, exist_ok=True)
    return path


def is_canonical_repo_root(path: str | Path) -> bool:
    """True if ``path`` names the same directory as the detected repository root.

    Compares fully-resolved paths, so the real path and the symlink alias compare
    EQUAL -- which is the whole point. Use this instead of string equality when
    checking "is this write landing in the repo?", because a string comparison
    against one spelling silently misses the other.
    """
    try:
        return Path(path).expanduser().resolve() == repo_root(start=__file__)
    except (OSError, RuntimeError, ValueError):
        # An unresolvable path is definitionally not the repo root; callers asking a
        # yes/no question should not have to handle an exception for a bad input.
        #
        # ValueError is NOT redundant with OSError: a path containing an embedded NUL
        # raises ValueError from os.lstat before any syscall happens, so catching only
        # OSError leaves this helper crashing on exactly the malformed input a
        # yes/no guard is most likely to be handed. Found by the regression test of
        # the same name rather than by inspection.
        return False
