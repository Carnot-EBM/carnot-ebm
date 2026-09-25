"""Run REQ-REPORT-7666 focused cases under coverage without optional JAX."""

from __future__ import annotations

import importlib.util
import builtins


REAL_IMPORT = builtins.__import__


def _no_jax_import(name, *args, **kwargs):
    if name == "jax" or name.startswith("jax."):
        error = ModuleNotFoundError("optional JAX disabled for CPU-only coverage")
        error.name = "jax"
        raise error
    return REAL_IMPORT(name, *args, **kwargs)


try:
    builtins.__import__ = _no_jax_import
    import pytest

    spec = importlib.util.spec_from_file_location(
        "exp7666_tests", "tests/python/test_experiment_7666_v668_arc_goal_confirmation.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for seed in range(8):
        for case, level, state, layers, expected in (
            ("wrong_goal", 0, "NOT_FINISHED", 1, "contradiction"),
            ("true_goal", 1, "NOT_FINISHED", 1, "confirmed"),
            ("final_action_level_change", 2, "WIN", 1, "confirmed"),
            ("delayed_animation", 0, "NOT_FINISHED", 3, "unknown"),
            ("unknown_terminal", 0, "MYSTERY", 1, "unknown"),
            ("loss", 0, "GAME_OVER", 1, "contradiction"),
        ):
            module.test_independent_goal_fixture(seed, case, level, state, layers, expected)
    module.test_stale_frame_and_timeout_remain_unknown()
    module.test_hidden_alias_and_goal_before_dedup()
    module.test_missing_endpoint_and_bad_level_remain_unknown()
    for test in (
        module.test_scored_factory_reaches_guard_and_recovers,
        module.test_disabled_flag_preserves_plan_action_and_random_state,
    ):
        monkeypatch = pytest.MonkeyPatch()
        try:
            test(monkeypatch)
        finally:
            monkeypatch.undo()
    print("53 focused test cases passed under coverage", flush=True)
finally:
    builtins.__import__ = REAL_IMPORT
