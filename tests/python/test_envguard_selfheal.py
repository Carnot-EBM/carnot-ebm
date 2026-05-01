"""Regression tests for the EnvPropagationGuard self-heal graceful-fallback fix.

Background — the .82 cascade-failure pattern (researcher summary)
------------------------------------------------------------------
For three consecutive milestones (.75, .82, and the .83 mandatory-first slot)
the conductor ran its pre-test gate, hit a single failing test, and the
``test_summary`` string surfaced as ``"Pre-tests failing, self-heal failed:
EnvPropagationGuard failed to load CARNOT_ variables"``.  That string is
a TRUNCATED form of the RuntimeError raised by
``ExperimentTemplate.assert_live_env_if_gpu()`` (full message:
``"EnvPropagationGuard failed to load CARNOT_FORCE_LIVE=1. ..."``).

The mechanism was:

1. ``research_conductor.py:run_tests()`` strips ``CARNOT_FORCE_LIVE`` from
   the pretest env so live-mode-only tests do not gate the smart subset on
   a ROCm/AMD dev machine (the .80 fix in commit f89cd1e9).
2. The smart subset includes test files that import experiment scripts
   for collection.  Many experiment scripts call ``tmpl.setup()`` at
   module-top-level when invoked as ``__main__``.
3. Pytest collection imports the script.  The single-run lock acquisition
   in ``setup()`` was already gated on ``__name__ == "__main__"`` (the
   2026-04-29 ``_caller_main_module`` fix), but
   ``assert_live_env_if_gpu()`` was NOT — it ran unconditionally.
4. With ``CARNOT_FORCE_LIVE`` stripped and ``requires_gpu=True``, the
   assert raised RuntimeError, the test failed, and the conductor logged
   the truncated message as a SKIP.  Three milestones in a row were
   blocked by the same unrelated import-time crash.

The fix
-------
``ExperimentTemplate.setup()`` now ALSO skips
``assert_live_env_if_gpu()`` when the caller is not ``__main__``.  This is
symmetric with the lock-acquisition skip: if we are being imported (not
invoked as a script), neither lock nor live-env assert is meaningful.
The assert still runs — and still fails fast — when an actual experiment
script is launched as a process.

What these tests cover
----------------------
- ``envguard_graceful_when_no_carnot_vars``: an ``ExperimentTemplate``
  with ``requires_gpu=True`` whose ``setup()`` is invoked from a non-
  ``__main__`` caller (i.e. test/import context) does NOT raise even
  when ``CARNOT_FORCE_LIVE`` is absent from the environment.  This is
  the regression test for the .82 SKIP cascade.
- ``self_heal_continues_after_envguard_miss``: confirms that when the
  guard is bypassed at import time, ``setup()`` still completes the rest
  of its work (output / checkpoint dirs created, checkpoint loaded as
  ``None``).  Bypassing the assert must not silently abort the rest of
  setup — otherwise the pre-test gate would succeed but the experiment
  itself would be in a broken half-initialised state.
- ``envguard_still_raises_when_invoked_as_main``: confirms the symmetric
  property — the guard still raises when the caller IS ``__main__``,
  protecting the production fail-fast contract.  We patch
  ``_caller_main_module`` to simulate a ``__main__`` call without
  actually exec'ing a script.
- ``envguard_no_op_for_cpu_experiment``: confirms that CPU-only
  experiments (``requires_gpu=False``) never trigger the guard — even
  when invoked as ``__main__``.  The guard is GPU-specific by design;
  this test pins that behaviour.

Spec: REQ-INFRA-070, REQ-INFRA-072
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from scripts.experiment_template import ExperimentTemplate


def _strip_force_live(env: dict[str, str]) -> dict[str, str]:
    """Return a copy of env with CARNOT_FORCE_LIVE removed.

    Mirrors ``research_conductor.run_tests()``'s pretest_env construction so
    these tests reproduce the conductor's actual gating environment.
    """
    return {k: v for k, v in env.items() if k != "CARNOT_FORCE_LIVE"}


def test_envguard_graceful_when_no_carnot_vars(tmp_path: Path) -> None:
    """Calling setup() at import time on a GPU experiment must NOT raise
    when CARNOT_FORCE_LIVE is absent.  This is the regression test for the
    .82 SKIP cascade where pre-test imports of GPU experiment scripts
    blew up with RuntimeError.

    The assertion: setup() returns cleanly even with requires_gpu=True
    and no live-env var, AS LONG AS the caller is not ``__main__``.
    """
    deliverable = tmp_path / "results" / "experiment_99999_dummy.json"
    tmpl = ExperimentTemplate(
        exp_id=99999,
        title="envguard self-heal regression",
        deliverable=str(deliverable),
        requires_gpu=True,
        repo_root=tmp_path,
    )

    env_no_live = _strip_force_live(dict(os.environ))
    # Force the caller-detection to report "not __main__" so the lock skip
    # path is exercised — this matches what happens during pytest collection
    # of a file that imports an experiment script.
    with (
        mock.patch.dict(os.environ, env_no_live, clear=True),
        mock.patch.object(
            ExperimentTemplate,
            "_caller_main_module",
            staticmethod(lambda: "tests.test_envguard_selfheal"),
        ),
    ):
        tmpl.setup()  # must not raise

    # Confirm the side-effects of setup() ran (this is the
    # "self-heal continues" half — captured separately below for clarity).
    assert deliverable.parent.exists()


def test_self_heal_continues_after_envguard_miss(tmp_path: Path) -> None:
    """When the guard is skipped at import time, the rest of setup() must
    still run to completion.  Bypassing the assert must not silently
    abort downstream work — otherwise we trade a loud crash for a quiet
    half-initialised experiment that fails opaquely later.

    The assertions: results dir created, checkpoint dir created,
    checkpoint loaded as None (no prior checkpoint exists).
    """
    deliverable = tmp_path / "results" / "experiment_99998_continue.json"
    tmpl = ExperimentTemplate(
        exp_id=99998,
        title="envguard self-heal continuation",
        deliverable=str(deliverable),
        requires_gpu=True,
        repo_root=tmp_path,
    )

    env_no_live = _strip_force_live(dict(os.environ))
    with (
        mock.patch.dict(os.environ, env_no_live, clear=True),
        mock.patch.object(
            ExperimentTemplate,
            "_caller_main_module",
            staticmethod(lambda: "tests.test_envguard_selfheal"),
        ),
    ):
        tmpl.setup()

    # The output directory was created
    assert deliverable.parent.exists(), "setup() must create the results dir"
    # The checkpoint directory was created under the repo root
    assert (tmp_path / "results" / "checkpoints" / "experiment_99998").exists(), (
        "setup() must create the per-experiment checkpoint dir"
    )
    # No checkpoint existed, so checkpoint is None
    assert tmpl.checkpoint is None
    # The single-run lock context manager was skipped (import-time call)
    assert tmpl._single_run_lock_cm is None


def test_envguard_still_raises_when_invoked_as_main(tmp_path: Path) -> None:
    """Symmetric guarantee: the assert still raises when the caller IS
    ``__main__``.  This is the production fail-fast contract for
    ``python scripts/experiment_X.py`` invocations on a host where the
    operator forgot to set CARNOT_FORCE_LIVE — the experiment must NOT
    silently degrade to non-live mode.

    We patch ``_caller_main_module`` to return ``"__main__"`` so we can
    test the production code path without exec'ing a real script.
    """
    deliverable = tmp_path / "results" / "experiment_99997_main.json"
    tmpl = ExperimentTemplate(
        exp_id=99997,
        title="envguard fail-fast under __main__",
        deliverable=str(deliverable),
        requires_gpu=True,
        repo_root=tmp_path,
    )

    env_no_live = _strip_force_live(dict(os.environ))
    with (
        mock.patch.dict(os.environ, env_no_live, clear=True),
        mock.patch.object(
            ExperimentTemplate,
            "_caller_main_module",
            staticmethod(lambda: "__main__"),
        ),
        # Stub out the single-run lock so this test does not actually
        # acquire a flock — we are only validating the guard, not the
        # locking semantics.
        mock.patch(
            "carnot.conductor.acquire",
            new=lambda _name: mock.MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None),
        ),
    ):
        with pytest.raises(
            RuntimeError, match="EnvPropagationGuard failed to load CARNOT_FORCE_LIVE"
        ):
            tmpl.setup()


def test_envguard_no_op_for_cpu_experiment(tmp_path: Path) -> None:
    """CPU-only experiments (``requires_gpu=False``) must never trigger the
    guard — even when called as ``__main__`` and even when
    CARNOT_FORCE_LIVE is absent.  The guard is GPU-specific by design;
    this test pins that behaviour so a future refactor cannot
    accidentally tighten the guard to all experiments.
    """
    deliverable = tmp_path / "results" / "experiment_99996_cpu.json"
    tmpl = ExperimentTemplate(
        exp_id=99996,
        title="envguard no-op for CPU",
        deliverable=str(deliverable),
        requires_gpu=False,
        repo_root=tmp_path,
    )

    env_no_live = _strip_force_live(dict(os.environ))
    with (
        mock.patch.dict(os.environ, env_no_live, clear=True),
        mock.patch.object(
            ExperimentTemplate,
            "_caller_main_module",
            staticmethod(lambda: "__main__"),
        ),
        mock.patch(
            "carnot.conductor.acquire",
            new=lambda _name: mock.MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None),
        ),
    ):
        tmpl.setup()  # must not raise — CPU experiment

    assert deliverable.parent.exists()
