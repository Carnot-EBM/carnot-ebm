"""Tests for SpeedupProfile — KAEM large-variable crossover profiling.

Every test traces to a spec requirement or scenario from
openspec/capabilities/verifiable-reasoning/spec.md.

Spec: REQ-KAEM-005, REQ-KAEM-006,
      SCENARIO-KAEM-010, SCENARIO-KAEM-011
"""

from __future__ import annotations

import pytest

# Import the module under test.  The module lives in scripts/ which is not a
# package, so we use importlib after inserting the scripts directory to sys.path.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from experiment_459_kaem_large_vars import SpeedupProfile  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_profile(
    n_vars_list: list[int],
    kaem_times: list[float],
    mcmc_times: list[float],
) -> SpeedupProfile:
    """Helper: build a SpeedupProfile from parallel lists."""
    return SpeedupProfile(
        n_vars_list=n_vars_list,
        kaem_times=kaem_times,
        mcmc_times=mcmc_times,
    )


# ---------------------------------------------------------------------------
# speedup_at()
# ---------------------------------------------------------------------------


def test_speedup_at_returns_ratio_mcmc_over_kaem() -> None:
    """speedup_at(n) returns mcmc_time / kaem_time for that n_vars.

    REQ-KAEM-005: per-size speedup ratios must be queryable.
    """
    profile = _make_profile([50, 100], [10.0, 20.0], [40.0, 30.0])
    assert profile.speedup_at(50) == pytest.approx(4.0)
    assert profile.speedup_at(100) == pytest.approx(1.5)


def test_speedup_at_raises_for_unknown_n_vars() -> None:
    """speedup_at raises KeyError when n_vars not in the profile list.

    REQ-KAEM-005: only profiled sizes are queryable.
    """
    profile = _make_profile([50], [10.0], [20.0])
    with pytest.raises(KeyError):
        profile.speedup_at(999)


def test_speedup_at_kaem_slower_returns_sub_one() -> None:
    """speedup_at returns < 1.0 when KAEM is slower than MCMC.

    REQ-KAEM-006: speedup < 1.0 means KAEM has not crossed over.
    """
    profile = _make_profile([50], [20.0], [10.0])
    assert profile.speedup_at(50) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# crossover_n_vars()
# ---------------------------------------------------------------------------


def test_crossover_at_200_when_kaem_wins_there_not_at_100() -> None:
    """crossover_n_vars returns 200 when that is the first n_vars where KAEM wins.

    SCENARIO-KAEM-010: crossover found at n_vars=200.
    """
    # kaem faster at 200 (mcmc=20 > kaem=8), but not at 50 or 100
    profile = _make_profile(
        n_vars_list=[50, 100, 200, 500, 1000],
        kaem_times=[10.0, 12.0, 8.0, 6.0, 5.0],
        mcmc_times=[8.0, 10.0, 20.0, 50.0, 100.0],
    )
    assert profile.crossover_n_vars() == 200


def test_crossover_none_when_kaem_never_faster() -> None:
    """crossover_n_vars returns None when KAEM is always slower.

    SCENARIO-KAEM-011: no crossover found in profiled range.
    """
    profile = _make_profile(
        n_vars_list=[50, 100, 200],
        kaem_times=[10.0, 15.0, 25.0],
        mcmc_times=[8.0, 10.0, 20.0],
    )
    assert profile.crossover_n_vars() is None


def test_crossover_at_first_point_when_always_faster() -> None:
    """crossover_n_vars returns the first n_vars when KAEM wins everywhere.

    REQ-KAEM-006: first point where speedup > 1.0 is the crossover.
    """
    profile = _make_profile(
        n_vars_list=[50, 100, 200],
        kaem_times=[5.0, 8.0, 10.0],
        mcmc_times=[10.0, 20.0, 30.0],
    )
    assert profile.crossover_n_vars() == 50


def test_crossover_at_last_point() -> None:
    """crossover_n_vars returns the last n_vars when KAEM only wins at the end.

    REQ-KAEM-006: boundary case — crossover at list tail.
    """
    profile = _make_profile(
        n_vars_list=[50, 100, 200],
        kaem_times=[20.0, 15.0, 5.0],
        mcmc_times=[10.0, 12.0, 8.0],
    )
    assert profile.crossover_n_vars() == 200


# ---------------------------------------------------------------------------
# max_speedup()
# ---------------------------------------------------------------------------


def test_max_speedup_returns_correct_n_vars_and_value() -> None:
    """max_speedup returns (n_vars, speedup) for the entry with highest speedup.

    REQ-KAEM-005: max_speedup must report the best observed performance point.
    """
    profile = _make_profile(
        n_vars_list=[50, 100, 200],
        kaem_times=[10.0, 5.0, 20.0],
        mcmc_times=[20.0, 50.0, 30.0],
    )
    n_vars, speedup = profile.max_speedup()
    assert n_vars == 100
    assert speedup == pytest.approx(10.0)


def test_max_speedup_single_entry() -> None:
    """max_speedup works correctly when only one n_vars is profiled.

    REQ-KAEM-005: degenerate case — single-entry profile.
    """
    profile = _make_profile([100], [4.0], [12.0])
    n_vars, speedup = profile.max_speedup()
    assert n_vars == 100
    assert speedup == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_mismatched_list_lengths_raise() -> None:
    """SpeedupProfile raises ValueError when list lengths differ.

    REQ-KAEM-005: data integrity — all three lists must be the same length.
    """
    with pytest.raises(ValueError, match="same length"):
        SpeedupProfile(
            n_vars_list=[50, 100],
            kaem_times=[1.0],
            mcmc_times=[2.0, 3.0],
        )


def test_empty_profile_raises() -> None:
    """SpeedupProfile raises ValueError when lists are empty.

    REQ-KAEM-005: at least one measurement required.
    """
    with pytest.raises(ValueError, match="non-empty"):
        SpeedupProfile(n_vars_list=[], kaem_times=[], mcmc_times=[])
