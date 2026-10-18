"""Carry the tracked-``results/`` write protection across the subprocess boundary.

WHY THIS EXISTS (read before changing anything here)
----------------------------------------------------
`tracked_results_guard` already protects `results/**` from test writes, and it works.
It has two mechanisms, and BOTH live inside one interpreter:

  * `install()` adds a PEP 578 audit hook, and
  * `install_legacy_results_write_compat()` monkeypatches `builtins.open` and friends.

Neither is inherited by a child process. A PEP 578 hook belongs to the interpreter that
added it. A monkeypatched `builtins.open` belongs to that interpreter's memory. So a test
that shells out is unprotected, and the project's tests shell out a lot.

That is not a theoretical hole. Measured 2026-08-24 on `experiment_3361`, whose test file
happens to exercise both paths on the same writer::

    pytest ...::test_experiment_3361_module    # write_artifact() in-process -> tree CLEAN
    pytest ...::test_experiment_3361_script    # same writer via subprocess  -> tree DIRTY

The in-process case is redirected and harmless. The subprocess case rewrites the committed
artifact and the test still reports green. `test_experiment_1736_kanele_synth.py` is the
same shape with no in-process half, which is why it is hit by every run.

The damage is not a crash, it is a SILENT DOWNGRADE of the research record. Those writers
rebuild their artifact from a hardcoded template, so every key the conductor added later --
`flagged_adversarial`, `corrigendum_note`, `corrigendum_pending` -- is dropped. A
quarantined artifact silently reads as clean until the next backfill re-stamps it.

WHAT THIS DOES
--------------
`install()` wraps `subprocess.Popen` so every child Python interpreter starts with a
`sitecustomize.py` that reinstalls the same redirect. A write aimed at `<repo>/results/...`
lands under the session's temp artifact root instead. Reads are untouched, so a child that
writes and then reads the path back still sees its own bytes.

REDIRECT, NOT REFUSE, and that is deliberate. It matches what the in-process layer already
does, so it introduces no new policy and no new failure mode: the tests above stay green
and simply stop touching the record. Refusing would be stricter and would fail those tests
outright; that is a decision for whoever owns REQ-REPORT-6157, not a side effect of closing
this gap.

WHAT IT DELIBERATELY DOES NOT CATCH
-----------------------------------
Stated plainly, because a guard believed to be total is worse than one known to be partial:

  * **Non-Python children.** `sh -c 'echo x > results/y.json'`, a compiled binary, or a
    vendor tool such as Vivado writing its own output. `PYTHONPATH` means nothing to them.
  * **`os.system`, `os.exec*`, `os.posix_spawn`.** Only `subprocess.Popen` is wrapped, so
    only the `subprocess` family (`run`, `call`, `check_output`, ...) is covered.
  * **`python -S` or `python -E`.** Both skip `sitecustomize`; `-E` also ignores the
    injected `PYTHONPATH`.
  * **A child that rewrites `PYTHONPATH` for its own grandchildren.** Ordinary inheritance
    does reach grandchildren, because the variable propagates; a child that overwrites it
    does not.
  * **Writes outside `<repo>/results/`.** `openspec/**`, `ops/**` and `output/**` are just
    as tracked and are not covered here.

So a clean run under this guard is not proof that nothing wrote to the record.
`scripts/test_suite_mutation_check.py` and `git status` remain the backstop.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

__all__ = [
    "CHILD_REDIRECT_ROOT_ENV",
    "CHILD_REPO_ROOT_ENV",
    "install",
    "is_installed",
    "shim_dir",
    "child_env",
]

# Read by the generated shim in the child. Named separately from
# CARNOT_EXPERIMENT_ARTIFACT_ROOT so that arming the child guard is its own decision, and
# so a stray inherited value cannot switch it on outside a test session.
CHILD_REPO_ROOT_ENV = "CARNOT_CHILD_GUARD_REPO_ROOT"
CHILD_REDIRECT_ROOT_ENV = "CARNOT_CHILD_GUARD_REDIRECT_ROOT"

_REPO_ROOT = Path(__file__).resolve().parents[3]

_installed = False
_shim_dir: Path | None = None
_ORIGINAL_POPEN_INIT = subprocess.Popen.__init__


def _env_positional_index() -> int:
    """Index of `env` within Popen.__init__'s positional args after `self` and `args`.

    Derived from the live signature rather than hardcoded, so a CPython reordering cannot
    silently make the guard rewrite the wrong argument. Falls back to the historical slot.
    """
    try:
        import inspect

        names = list(inspect.signature(_ORIGINAL_POPEN_INIT).parameters)
        return names.index("env") - 2  # drop `self` and `args`
    except Exception:  # noqa: BLE001 - a diagnostic must never fail at import
        return 10


_ENV_POSITIONAL_INDEX = _env_positional_index()


# The child shim. Stdlib only, on purpose: the children this must protect are launched as
# bare `python scripts/...`, which on this box is /usr/bin/python -- no venv, no `carnot`.
# Importing anything from the project here would raise in exactly the case that matters.
_SHIM_SOURCE = '''\
"""Auto-generated by carnot.testing.child_results_guard. Do not edit or commit.

Redirects writes aimed at the repository's tracked `results/` tree into a temp root, so a
test that shells out cannot rewrite committed evidence. Stdlib only: the children this
protects often run under a bare system python with no project packages installed.
"""

import builtins
import io
import os
import sys

_REPO = os.environ.get("CARNOT_CHILD_GUARD_REPO_ROOT") or ""
_ROOT = os.environ.get("CARNOT_CHILD_GUARD_REDIRECT_ROOT") or ""


def _install():
    if not _REPO or not _ROOT:
        return
    results = os.path.join(_REPO, "results")

    def _target(path):
        """Return a redirected path when `path` points into <repo>/results, else None."""
        try:
            text = os.fspath(path)
        except TypeError:
            return None
        if isinstance(text, bytes):
            text = text.decode("utf-8", "surrogateescape")
        if not text or "results" not in text:
            return None
        try:
            resolved = os.path.realpath(os.path.abspath(text))
        except (OSError, ValueError):
            return None
        if resolved != results and not resolved.startswith(results + os.sep):
            return None
        rel = os.path.relpath(resolved, results)
        dest = os.path.join(_ROOT, rel)
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        except OSError:
            return None
        return dest

    def _writing_mode(mode):
        return isinstance(mode, str) and any(c in mode for c in "wxa+")

    def _writing_flags(flags):
        bits = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
        return isinstance(flags, int) and bool(flags & bits)

    real_open, real_io_open, real_os_open = builtins.open, io.open, os.open
    real_rename, real_replace = os.rename, os.replace

    def _wrap_open(real):
        def opened(file, mode="r", *args, **kwargs):
            if _writing_mode(mode):
                dest = _target(file)
                if dest is not None:
                    file = dest
            return real(file, mode, *args, **kwargs)

        return opened

    def _os_open(path, flags, mode=0o777, **kwargs):
        if _writing_flags(flags):
            dest = _target(path)
            if dest is not None:
                path = dest
        return real_os_open(path, flags, mode, **kwargs)

    def _wrap_move(real):
        def moved(src, dst, *args, **kwargs):
            dest = _target(dst)
            return real(src, dest if dest is not None else dst, *args, **kwargs)

        return moved

    builtins.open = _wrap_open(real_open)
    io.open = _wrap_open(real_io_open)
    os.open = _os_open
    os.rename = _wrap_move(real_rename)
    os.replace = _wrap_move(real_replace)


try:
    _install()
except Exception:
    # Never break a child over a diagnostic. A guard that can crash the run it guards is
    # removed by the first person it inconveniences, which leaves the record unprotected.
    pass

# Chain to any real sitecustomize further along sys.path, so shadowing ours does not
# silently drop someone else's startup hook. Neither interpreter on this box ships one
# (checked 2026-08-24), so this is insurance rather than a live requirement.
#
# The scan is a plain isfile() loop, NOT importlib.find_spec. This runs at the start of
# EVERY child process, and find_spec's full finder protocol measured ~25 ms per spawn here
# -- more than the cost of starting the interpreter. The loop costs microseconds and the
# only case it misses is a sitecustomize shipped inside a zip or a namespace package.
try:
    _here = os.path.dirname(os.path.abspath(__file__))
    for _entry in sys.path:
        _dir = os.path.abspath(_entry or ".")
        if _dir == _here:
            continue
        _candidate = os.path.join(_dir, "sitecustomize.py")
        if os.path.isfile(_candidate):
            import importlib.util

            _spec = importlib.util.spec_from_file_location("sitecustomize", _candidate)
            if _spec is not None and _spec.loader is not None:
                _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
            break
except Exception:
    pass
'''


def is_installed() -> bool:
    """True once `install()` has wrapped `subprocess.Popen` in this process."""
    return _installed


def shim_dir() -> Path:
    """Create (once) the directory holding the generated child `sitecustomize.py`."""
    global _shim_dir
    if _shim_dir is None:
        created = Path(tempfile.mkdtemp(prefix="carnot-child-results-guard-"))
        (created / "sitecustomize.py").write_text(_SHIM_SOURCE, encoding="utf-8")
        _shim_dir = created
    return _shim_dir


def child_env(env: dict | None = None, *, redirect_root: str | None = None) -> dict:
    """Return `env` with the shim on PYTHONPATH and the two shim variables set.

    Returns the mapping unchanged when there is no redirect root to point at, so a
    non-test process that somehow reaches this code spawns children exactly as before.
    """
    base = dict(os.environ if env is None else env)
    root = redirect_root or base.get(CHILD_REDIRECT_ROOT_ENV) or ""
    if not root:
        return base
    shim = str(shim_dir())
    existing = base.get("PYTHONPATH") or ""
    parts = [p for p in existing.split(os.pathsep) if p and p != shim]
    base["PYTHONPATH"] = os.pathsep.join([shim, *parts])
    # setdefault, not assignment: a caller that already named a repo root means it. Plain
    # assignment would silently overwrite it on the way through the Popen wrapper, which
    # also makes the guard impossible to exercise against a fixture tree.
    base.setdefault(CHILD_REPO_ROOT_ENV, str(_REPO_ROOT))
    base[CHILD_REDIRECT_ROOT_ENV] = root
    return base


def install(redirect_root: str | None = None) -> bool:
    """Wrap `subprocess.Popen` so children inherit the redirect. Idempotent.

    Idempotence matters because pytest imports conftest once per xdist worker, and wrapping
    an already-wrapped `Popen` would nest the wrapper on every call.
    """
    global _installed
    if _installed:
        return False
    root = redirect_root or os.environ.get(CHILD_REDIRECT_ROOT_ENV) or ""
    if not root:
        return False

    os.environ[CHILD_REPO_ROOT_ENV] = str(_REPO_ROOT)
    os.environ[CHILD_REDIRECT_ROOT_ENV] = root

    def _patched_init(self, args, *rest, **kwargs):
        # The caller's explicit env wins on every key except ours. `env` is the 11th
        # positional parameter after `args`, so a caller MAY pass it positionally; writing
        # kwargs["env"] in that case would raise "multiple values for argument 'env'" and
        # break the spawn. Rewrite whichever slot the caller actually used.
        try:
            if len(rest) > _ENV_POSITIONAL_INDEX:
                supplied = rest[_ENV_POSITIONAL_INDEX]
                merged = child_env(
                    supplied if supplied is not None else os.environ, redirect_root=root
                )
                rest = (*rest[:_ENV_POSITIONAL_INDEX], merged, *rest[_ENV_POSITIONAL_INDEX + 1 :])
            else:
                supplied = kwargs.get("env")
                kwargs["env"] = child_env(
                    supplied if supplied is not None else os.environ, redirect_root=root
                )
        except Exception:  # noqa: BLE001 - never block a spawn over a diagnostic
            pass
        return _ORIGINAL_POPEN_INIT(self, args, *rest, **kwargs)

    subprocess.Popen.__init__ = _patched_init  # type: ignore[method-assign]
    _installed = True
    return True
