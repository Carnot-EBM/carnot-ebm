"""Tree-wide pytest hooks for every test directory, not only `tests/python/`.

WHY THIS FILE EXISTS (QA-layer SILENT_NON_FIRING, 2026-09-04, confirmed 2026-09-05).

`carnot.testing.worktree_import_guard` refuses a run whose tests come from one checkout while
`import carnot` resolves to another -- the trap that makes a worktree mutation proof read GREEN
while measuring the main checkout. It was wired into `tests/python/conftest.py`.

pytest loads a `conftest.py` only for the directory it sits in and that directory's descendants.
`tests/python/conftest.py` is not an ancestor of `tests/archive/`, and this repository has no
root `conftest.py`. So the named missed input --
`tests/archive/test_weight_steering.py` collected from a worktree while `carnot.__file__`
resolves to the main checkout -- ran with the guard never loaded at all. Measured before the
fix: collecting that file reported "23 tests collected" and the guard did not run.

The guard's concept is "a pytest run must not silently test a foreign checkout". Its wiring
covered one directory. This file moves the wiring up to the concept: every directory under
`tests/` is now checked. `tests/python/conftest.py` keeps its own call, which is harmless -- the
check is a pure comparison and running it twice cannot disagree with itself.
"""

from __future__ import annotations

from pathlib import Path

import carnot as _carnot
from carnot.testing.worktree_import_guard import check as _check_worktree_import

# Runs at import, so a foreign-checkout run dies during collection rather than after an hour of
# meaningless passes. parents[1] is the repo root: this file is <root>/tests/conftest.py.
_check_worktree_import(Path(__file__).resolve().parents[1], Path(_carnot.__file__).resolve().parent)
