"""Per-invocation tmp bases: no pytest run may delete another run's live tmp base.

WHY (2026-08-28). `tmp_path_retention_count = 1` plus the shared default tmp root let
any concurrent pytest prune a running job's base. Two ~8h full-suite measurement runs
were destroyed, and the mid-run deletion turned failures into setup errors, which
SUPPRESSED the failure count those runs existed to measure. The fix gives every
invocation a private base; these tests pin each rule of it.
"""

from __future__ import annotations

import getpass
import os
import time
from pathlib import Path
from types import SimpleNamespace

from carnot.testing.pytest_basetemp_isolation import (
    MAX_AGE_S,
    basetemp_parent,
    install_isolated_basetemp,
    new_isolated_basetemp,
    prune_stale_bases,
)


def test_two_invocations_never_share_a_base() -> None:
    # SCENARIO-HARNESS-5930-ISOLATION: sharing a base is the destruction vector; uniqueness is the whole fix.
    a = new_isolated_basetemp(pid=111, now=1_000_000.0)
    b = new_isolated_basetemp(pid=111, now=1_000_000.0)
    assert a != b, "same pid and same second must still yield distinct bases"
    assert a.parent == b.parent == basetemp_parent()


def test_explicit_basetemp_is_respected_untouched() -> None:
    # SCENARIO-HARNESS-5930-EXPLICIT-RESPECTED: xdist workers arrive with a controller-assigned base; overriding it would
    # scatter one invocation across several roots. An explicit choice always wins.
    config = SimpleNamespace(option=SimpleNamespace(basetemp="/somewhere/explicit"))
    assert install_isolated_basetemp(config) is None
    assert config.option.basetemp == "/somewhere/explicit"


def test_missing_basetemp_gets_a_private_one() -> None:
    # REQ-HARNESS-5930: a missing --basetemp gets a private, per-invocation base.
    config = SimpleNamespace(option=SimpleNamespace(basetemp=None))
    assigned = install_isolated_basetemp(config)
    assert assigned is not None
    assert Path(config.option.basetemp) == assigned
    assert assigned.parent == basetemp_parent()


def test_prune_removes_only_stale_siblings(tmp_path: Path) -> None:
    # SCENARIO-HARNESS-5930-LIVE-RUN-NEVER-PRUNED.
    parent = tmp_path / "bases"
    parent.mkdir()
    now = time.time()
    stale = parent / "old-run"
    fresh = parent / "fresh-run"
    keep = parent / "current-run"
    for d in (stale, fresh, keep):
        d.mkdir()
        (d / "evidence.txt").write_text("x")
    os.utime(stale, (now - MAX_AGE_S - 60, now - MAX_AGE_S - 60))
    os.utime(fresh, (now - 60, now - 60))
    os.utime(keep, (now - MAX_AGE_S - 60, now - MAX_AGE_S - 60))

    removed = prune_stale_bases(parent, keep=keep, now=now)

    assert removed == [stale]
    assert not stale.exists()
    assert fresh.exists(), "a young base may belong to a LIVE run and must never be pruned"
    assert keep.exists(), "the base just created is never pruned, whatever its mtime"


def test_prune_survives_a_missing_parent(tmp_path: Path) -> None:
    # REQ-HARNESS-5930: cleanup is best-effort and must never break a session.
    assert prune_stale_bases(tmp_path / "never-created") == []


def test_this_very_session_is_isolated(tmp_path_factory, pytestconfig) -> None:
    """REQ-HARNESS-5930: the wiring proof, asserted in-session so it can never be skipped.

    The destruction vector is the SHARED default rotation root, `pytest-of-<user>`:
    every invocation using it prunes the others' numbered bases down to the retention
    count. So whatever way this suite was launched, its session base must NOT live
    there -- either the invoker chose an explicit private base (respected untouched),
    or the conftest assigned one under the per-user isolated parent.

    Deliberately worker-proof: pytest-xdist hands each worker `<session_base>/popen-gwN`
    even when the controller had NO basetemp, so asserting "basetemp is set" would pass
    vacuously in workers. Asserting on the normalized SESSION base does not.
    """
    base = Path(tmp_path_factory.getbasetemp()).resolve()
    # In an xdist worker the base is <session_base>/popen-<workerid>; normalize to the
    # session base so controller and workers assert the same thing.
    session_base = base.parent if base.name.startswith("popen-") else base
    shared_default_root = f"pytest-of-{getpass.getuser()}"
    assert shared_default_root not in {p.name for p in [session_base, *session_base.parents]}, (
        f"session tmp base {session_base} sits under the SHARED rotation root "
        f"{shared_default_root!r}; the conftest isolation did not run, and any concurrent "
        "pytest can prune this session's tmp base mid-run"
    )
