"""Spec: REQ-WORKTREE-IMPORT-1, SCENARIO-WORKTREE-IMPORT-1-B

The foreign-checkout guard covers every test directory, not only `tests/python/`.

QA-LAYER FINDING 2026-09-04, confirmed 2026-09-05. The guard refuses a run whose tests come from
one checkout while `import carnot` resolves to another -- the trap that makes a worktree mutation
proof read GREEN while measuring the main checkout.

It was wired into `tests/python/conftest.py`. pytest loads a conftest only for its own directory
and that directory's descendants, `tests/python/` is not an ancestor of `tests/archive/`, and
this repository has no root conftest. So the named missed input, `tests/archive/
test_weight_steering.py` collected from a worktree, ran with the guard never loaded. Measured
before the fix: that file collected 23 tests with no check performed.

The concept is "a pytest run must not silently test a foreign checkout". The wiring covered one
directory. `tests/conftest.py` moves the wiring up to the concept.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE_TEST = REPO / "tests" / "archive" / "test_weight_steering.py"


def test_a_tree_wide_conftest_exists_and_wires_the_guard() -> None:
    text = (REPO / "tests" / "conftest.py").read_text()
    assert "worktree_import_guard" in text
    assert "_check_worktree_import(" in text


def test_the_named_missed_input_still_exists_to_be_protected() -> None:
    """A guard aimed at a path that has moved protects nothing."""
    assert ARCHIVE_TEST.is_file()


def test_a_foreign_checkout_is_refused_when_collecting_outside_tests_python(
    tmp_path: Path,
) -> None:
    """The incident end to end: collect the archive test with carnot pointed elsewhere.

    Runs pytest in a subprocess because the failure must happen at CONFTEST LOAD, which cannot
    be observed from inside an already-loaded session. `--basetemp` is passed because sibling
    pytest runs otherwise delete each other's tmp base.
    """
    # A foreign checkout complete enough for the conftest to REACH the guard. A bare
    # __init__.py is not: the conftest imports `carnot.testing.worktree_import_guard`, so a
    # stub dies on ImportError and the run fails for the wrong reason -- which is what the
    # first version of this test actually asserted, and it would have passed with the guard
    # removed. The real module is copied in, so the only difference is the PATH.
    foreign = tmp_path / "otherco" / "python"
    (foreign / "carnot" / "testing").mkdir(parents=True)
    (foreign / "carnot" / "__init__.py").write_text("__version__ = '0'\n")
    (foreign / "carnot" / "testing" / "__init__.py").write_text("")
    shutil.copy2(
        REPO / "python" / "carnot" / "testing" / "worktree_import_guard.py",
        foreign / "carnot" / "testing" / "worktree_import_guard.py",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(ARCHIVE_TEST),
            "--collect-only",
            "-q",
            "--no-cov",
            "-p",
            "no:randomly",
            "--basetemp",
            str(tmp_path / "pt"),
        ],
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(foreign), "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0, "collection succeeded against a foreign carnot"
    assert "foreign checkout" in (proc.stdout + proc.stderr)
