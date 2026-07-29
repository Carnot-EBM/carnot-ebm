"""Pytest plugin that records, per test, which repo files were opened for WRITE.

WHY THIS EXISTS (verbose, per the project's documentation standard):

The research record in `results/*.json` is the input to the fabrication gate
(`scripts/adversarial_verify.py`), to every capstone aggregation, and to the
paper. A test that re-executes a real experiment script rewrites that record as
a side effect. Running the suite therefore silently mutates the evidence.

Enumerating *which* test rewrites *which* artifact by running each of ~3500 test
files individually is not feasible (import cost alone dominates). Instead we run
the suite ONCE with a CPython audit hook installed (`sys.addaudithook`). The
audit subsystem fires an `open` event for every file open in the process,
including opens performed deep inside a `runpy.run_path` of an experiment
script. We filter those events to writes of files inside the repository, and
attribute each to whichever pytest node was executing at the time.

This gives an exact test -> file mapping from a single run, with no guessing
about which tests "look like" they write.

Two independent signals are produced:
  * `writes`  -- what was OPENED FOR WRITE (this plugin). A superset: a script
                 may rewrite a file with byte-identical content.
  * `git status` after the run (captured by the caller). The ground truth for
                 "the record actually changed".
Intersecting the two gives attribution for real mutations.

USAGE
-----
Run pytest against a THROWAWAY worktree, never the canonical repo -- the whole
point is that the run rewrites files, and this plugin only observes, it does not
prevent::

    git worktree add --detach /tmp/wt HEAD
    cp -a environment_files /tmp/wt/            # gitignored, but the ARC tests
                                                # abort early without it
    cd /tmp/wt
    SURVEY_REPO=/tmp/wt SURVEY_OUT=/tmp/survey \
    PYTHONPATH=<repo>/scripts:/tmp/wt/python:/tmp/wt/scripts/experiments \
      python -m pytest tests/python/test_experiment_1*.py \
        -p pytest_write_audit_plugin --no-cov -q

    # then merge the per-worker shards
    python -c "import json,glob;
      m={};
      [m.setdefault(k,{}).update(v) for f in glob.glob('/tmp/survey.*.json')
         for k,v in json.load(open(f)).items()];
      print(json.dumps(m, indent=1))"

THE EDITABLE-INSTALL TRAP -- do not skip the PYTHONPATH. The venv installs
``carnot`` in editable mode pointing at the CANONICAL checkout, so a worktree run
would import the real repo's package and any module deriving a path from
``carnot.__file__`` would write into the real repo. Putting the worktree's
``python/`` on PYTHONPATH puts it ahead of the ``.pth`` entry in ``sys.path``.

COST -- an audit hook is called for EVERY audited event in the process, so an
import-heavy suite slows down substantially. That is affordable on a few hundred
test files and painful on several thousand; for the large case, batch with plain
``git status`` first and only run this plugin on the batches that moved a file.

Cross-references:
- ``scripts/test_suite_mutation_check.py`` -- the always-on detector this
  informs (broad, shallow, no audit hook, cheap enough to wrap every run)
- ``docs/research-notes/test-suite-rewrites-the-record-survey-2026-07-29.md``
  -- the survey this plugin was written for, including what it found
- commit ``b3e31d341`` -- the hazard
"""

from __future__ import annotations

import atexit
import json
import os
import sys

_REPO = os.path.realpath(os.environ["SURVEY_REPO"])
# The suite runs under pytest-xdist (`-n 4` in addopts), so several worker
# processes each load this plugin. They must not clobber one another's output,
# hence the per-worker/per-pid suffix. The caller merges the shards.
_WORKER = f"{os.environ.get('PYTEST_XDIST_WORKER', 'main')}-pid{os.getpid()}"
_OUT = f"{os.environ['SURVEY_OUT']}.{_WORKER}.json"

# nodeid of the test currently executing. Writes that happen during collection /
# import (which is itself a real hazard class -- an import-time side effect) are
# attributed to the sentinel below.
_current = {"nodeid": "<collection-or-import>"}

# nodeid -> {path: count}
_records: dict[str, dict[str, int]] = {}

_WRITE_MODE_CHARS = ("w", "a", "x", "+")


def _record(path) -> None:
    """Attribute one write-open of `path` to the currently-running test.

    Must never raise and must never itself perform an audited operation, or the
    hook would recurse. Only pure-Python string work happens here.
    """
    if not isinstance(path, str):
        return
    # Cheap prefix test first; realpath() is comparatively expensive and this
    # hook runs on every open in the process.
    if not path.startswith("/"):
        path = os.path.join(_cwd, path)
    if not path.startswith(_REPO):
        return
    rel = path[len(_REPO) :].lstrip("/")
    if not rel or rel.startswith(".git/"):
        return
    bucket = _records.setdefault(_current["nodeid"], {})
    bucket[rel] = bucket.get(rel, 0) + 1


_cwd = os.getcwd()


def _hook(event: str, args) -> None:
    try:
        if event == "open":
            path, mode, _flags = args
            if mode and any(c in mode for c in _WRITE_MODE_CHARS):
                _record(path)
        elif event in ("os.rename", "os.replace", "shutil.copyfile", "shutil.move"):
            # Atomic writes are "write a temp file, rename it over the target".
            # The DESTINATION is the file that got rewritten; the source is scratch.
            _record(args[1])
        elif event in ("os.remove", "os.unlink", "os.truncate", "os.mkdir"):
            _record(args[0])
    except Exception:
        # A crash in an audit hook would take down the whole interpreter for
        # every subsequent open. Swallow everything.
        pass


def _dump() -> None:
    try:
        with open(_OUT, "w") as fh:
            json.dump(_records, fh, indent=1, sort_keys=True)
    except Exception:
        pass


sys.addaudithook(_hook)
atexit.register(_dump)


# ---------------------------------------------------------------- pytest hooks


def pytest_runtest_logstart(nodeid, location):  # noqa: ARG001
    _current["nodeid"] = nodeid


def pytest_runtest_logfinish(nodeid, location):  # noqa: ARG001
    _current["nodeid"] = "<between-tests>"


def pytest_collectstart(collector):
    _current["nodeid"] = f"<collect> {getattr(collector, 'nodeid', '?')}"


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    _current["nodeid"] = "<sessionfinish>"
    _dump()
