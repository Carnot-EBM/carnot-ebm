"""The survey plugin must record writes it sees and stay silent about reads.

REQ-ARC-WMTE-6041 / SCENARIOs: write-opens-are-recorded, reads-are-ignored,
outside-repo-writes-are-ignored, the-hook-never-raises

WHY THIS EXISTS (2026-07-29). ``scripts/pytest_write_audit_plugin.py`` is the instrument that
produced the test -> artifact mapping in
``docs/research-notes/test-suite-rewrites-the-record-survey-2026-07-29.md``. An instrument whose
own correctness is unpinned turns a survey into an anecdote: if the hook silently missed a write
mode, the survey's "this test moves nothing" rows would be false negatives, and a repair would be
scoped against them.

The most load-bearing test here is ``test_the_hook_never_raises``. A CPython audit hook cannot be
removed once installed and runs on EVERY audited event in the process, so an exception escaping it
would break every subsequent file open in the interpreter -- turning a diagnostic into an outage
in whatever it was wrapping.
"""

from __future__ import annotations

import atexit
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PLUGIN_SRC = REPO / "scripts" / "pytest_write_audit_plugin.py"


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    """Load the plugin WITHOUT letting it install its audit hook in this interpreter.

    The module calls ``sys.addaudithook`` at import time, and a CPython audit hook can never be
    removed once installed. Importing it normally -- once per test -- would leave a growing pile
    of permanent hooks firing on every audited event for the REST of the suite, i.e. this test
    file would slow down every test that runs after it. So ``sys.addaudithook`` is stubbed to a
    no-op for the duration of the exec, and the hook function is then driven directly. Same
    coverage, no contamination.

    The module also reads its config from the environment at import time, so the env is pointed
    at a throwaway tree first.
    """
    monkeypatch.setenv("SURVEY_REPO", str(tmp_path))
    monkeypatch.setenv("SURVEY_OUT", str(tmp_path / "out"))
    monkeypatch.setattr(sys, "addaudithook", lambda _fn: None)
    # Same reasoning for atexit: the module registers a dump callback at import time, and 17
    # registrations pointing at deleted tmp_paths would fire at interpreter shutdown.
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: fn)

    spec = importlib.util.spec_from_file_location("_plugin_under_test", PLUGIN_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod._records.clear()
    mod._current["nodeid"] = "test::node"
    return mod, tmp_path


def _recorded(mod, node="test::node"):
    return set(mod._records.get(node, {}))


def test_a_write_open_is_recorded_against_the_running_test(plugin):
    """The core signal: an artifact opened for write, attributed to whoever was running."""
    mod, root = plugin
    mod._hook("open", (str(root / "results" / "experiment_1_x.json"), "w", 0))
    assert _recorded(mod) == {"results/experiment_1_x.json"}


@pytest.mark.parametrize("mode", ["w", "a", "x", "r+", "wb", "w+b"])
def test_every_writing_mode_counts(plugin, mode):
    """Missing a mode would silently under-report -- ``r+`` in particular truncates nothing but
    still rewrites in place, which is exactly how a JSON artifact gets edited."""
    mod, root = plugin
    mod._hook("open", (str(root / "results" / f"a_{mode}.json"), mode, 0))
    assert f"results/a_{mode}.json" in _recorded(mod)


def test_a_read_is_not_recorded(plugin):
    """Reads are the overwhelming majority of opens. Recording them would drown the signal."""
    mod, root = plugin
    mod._hook("open", (str(root / "results" / "read_only.json"), "r", 0))
    assert _recorded(mod) == set()


def test_a_write_outside_the_repo_is_not_recorded(plugin):
    """Only the tracked research record matters; /tmp scratch writes are noise."""
    mod, _ = plugin
    mod._hook("open", ("/tmp/somewhere/else.json", "w", 0))
    assert _recorded(mod) == set()


def test_git_internals_are_not_recorded(plugin):
    """``.git/`` churns constantly during a run and says nothing about the research record."""
    mod, root = plugin
    mod._hook("open", (str(root / ".git" / "index.lock"), "w", 0))
    assert _recorded(mod) == set()


def test_a_rename_records_the_destination_not_the_source(plugin):
    """Atomic writes are ``write tmp; rename over target``. The TARGET is what got rewritten."""
    mod, root = plugin
    mod._hook(
        "os.rename", (str(root / "results" / "tmp.json"), str(root / "results" / "real.json"))
    )
    assert _recorded(mod) == {"results/real.json"}


def test_a_delete_is_recorded(plugin):
    """Deleting an artifact is the most destructive rewrite of all."""
    mod, root = plugin
    mod._hook("os.remove", (str(root / "results" / "gone.json"),))
    assert _recorded(mod) == {"results/gone.json"}


def test_attribution_follows_the_running_node(plugin):
    """Two tests in one worker must not have their writes merged."""
    mod, root = plugin
    mod.pytest_runtest_logstart("tests/a.py::test_one", None)
    mod._hook("open", (str(root / "results" / "from_one.json"), "w", 0))
    mod.pytest_runtest_logstart("tests/b.py::test_two", None)
    mod._hook("open", (str(root / "results" / "from_two.json"), "w", 0))

    assert _recorded(mod, "tests/a.py::test_one") == {"results/from_one.json"}
    assert _recorded(mod, "tests/b.py::test_two") == {"results/from_two.json"}


def test_import_time_writes_are_attributed_not_dropped(plugin):
    """A write during collection/import is a real hazard class, not something to discard.

    It is the harder one to reason about, because it happens before any test body runs -- so it
    gets an explicit sentinel nodeid rather than being silently merged into whatever test
    happened to be next.
    """
    mod, root = plugin
    mod._current["nodeid"] = "<collection-or-import>"
    mod._hook("open", (str(root / "results" / "at_import.json"), "w", 0))
    assert _recorded(mod, "<collection-or-import>") == {"results/at_import.json"}


def test_the_hook_never_raises(plugin):
    """An audit hook cannot be uninstalled and runs on every open in the process.

    If it raised, every subsequent file operation in the interpreter would fail -- the
    instrument would take down the thing it is measuring. Feed it malformed events for each
    branch and assert it swallows all of them.
    """
    mod, _ = plugin
    for event, args in [
        ("open", ()),  # too few args
        ("open", (None, None, None)),  # path is None (an fd-based open)
        ("open", (3, "w", 0)),  # path is an int fd
        ("os.rename", ("only-one-arg",)),
        ("os.remove", ()),
        ("os.mkdir", (object(),)),  # unhashable-ish junk
        ("some.unknown.event", (1, 2, 3)),
    ]:
        mod._hook(event, args)  # must not raise

    # Surviving the loop is half the assertion; the other half is that malformed events did not
    # smuggle junk into the record, which would corrupt the survey rather than crash it.
    assert mod._records == {}


def test_the_shard_name_is_unique_per_worker(plugin):
    """xdist runs 4 workers by default; a shared output path would have them clobber each other."""
    mod, _ = plugin
    assert str(os.getpid()) in mod._OUT
    assert mod._OUT.endswith(".json")


def test_the_dump_writes_readable_json(plugin):
    """The survey is only as good as its output actually landing on disk."""
    import json

    mod, root = plugin
    mod._hook("open", (str(root / "results" / "x.json"), "w", 0))
    mod._dump()
    written = json.loads(Path(mod._OUT).read_text())
    assert written["test::node"]["results/x.json"] >= 1
