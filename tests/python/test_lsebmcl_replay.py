"""Tests for LSEBMCLReplayBuffer and ReplaySession.

All tests trace to REQ-SELF-021 (LSEBMCL replay buffer prevents catastrophic
forgetting with forgetting_rate < 0.05 across 3 sessions).
"""

import pytest

from carnot.pipeline.lsebmcl_replay import LSEBMCLReplayBuffer, ReplaySession


# Shared toy energy function used across tests.
def _energy(p: str) -> float:
    return float(len(p)) / 100.0


# ── ReplaySession dataclass ────────────────────────────────────────────────────


def test_replay_session_fields():
    """REQ-SELF-021-1: ReplaySession stores all required fields correctly."""
    s = ReplaySession(
        session_id=1,
        n_templates=3,
        template_patterns=["a", "b", "c"],
        ebm_energy_mean=0.5,
    )
    assert s.session_id == 1
    assert s.n_templates == 3
    assert s.template_patterns == ["a", "b", "c"]
    assert s.ebm_energy_mean == pytest.approx(0.5)


# ── LSEBMCLReplayBuffer.add_session ───────────────────────────────────────────


def test_add_session_returns_replay_session():
    """REQ-SELF-021-1: add_session returns a ReplaySession with correct session_id."""
    buf = LSEBMCLReplayBuffer(energy_fn=_energy)
    patterns = ["hello world", "foo bar"]
    session = buf.add_session(1, patterns)
    assert isinstance(session, ReplaySession)
    assert session.session_id == 1


def test_add_session_stores_session():
    """REQ-SELF-021-1: add_session appends to internal sessions list."""
    buf = LSEBMCLReplayBuffer(energy_fn=_energy)
    buf.add_session(1, ["a", "b"])
    assert len(buf.sessions) == 1


def test_add_session_truncates_to_max_replay():
    """REQ-SELF-021-1: template_patterns is capped at max_replay_per_session."""
    buf = LSEBMCLReplayBuffer(energy_fn=_energy, max_replay_per_session=2)
    session = buf.add_session(1, ["a", "b", "c", "d"])
    assert len(session.template_patterns) == 2


def test_add_session_keeps_all_when_under_max():
    """REQ-SELF-021-1: all patterns kept when count <= max_replay_per_session."""
    buf = LSEBMCLReplayBuffer(energy_fn=_energy, max_replay_per_session=10)
    session = buf.add_session(1, ["x", "y"])
    assert len(session.template_patterns) == 2


def test_add_session_computes_mean_energy():
    """REQ-SELF-021-1: mean EBM energy is averaged over all patterns, not just truncated set."""
    buf = LSEBMCLReplayBuffer(energy_fn=lambda p: 1.0, max_replay_per_session=1)
    session = buf.add_session(1, ["a", "bb", "ccc"])
    assert session.ebm_energy_mean == pytest.approx(1.0)


def test_add_session_empty_patterns_mean_energy():
    """REQ-SELF-021-1: empty pattern list yields mean energy 0.0 without division error."""
    buf = LSEBMCLReplayBuffer(energy_fn=_energy)
    session = buf.add_session(1, [])
    assert session.ebm_energy_mean == pytest.approx(0.0)
    assert session.n_templates == 0


def test_add_session_n_templates_reflects_full_list():
    """REQ-SELF-021-1: n_templates equals full input length, not truncated."""
    buf = LSEBMCLReplayBuffer(energy_fn=_energy, max_replay_per_session=2)
    session = buf.add_session(1, ["a", "b", "c"])
    assert session.n_templates == 3


# ── LSEBMCLReplayBuffer.generate_replay ───────────────────────────────────────


def test_generate_replay_returns_prior_session_patterns():
    """REQ-SELF-021-2 / SCENARIO-SELF-027: replay includes all patterns from prior sessions."""
    buf = LSEBMCLReplayBuffer(energy_fn=_energy, max_replay_per_session=5)
    buf.add_session(0, ["COMPUTE: 47 + 28 = 76", "total is 80", "result is 15"])
    replay = buf.generate_replay(1)
    assert "COMPUTE: 47 + 28 = 76" in replay
    assert "total is 80" in replay
    assert "result is 15" in replay


def test_generate_replay_excludes_current_session():
    """REQ-SELF-021-2: current session_id is not included in replay."""
    buf = LSEBMCLReplayBuffer(energy_fn=_energy, max_replay_per_session=5)
    buf.add_session(1, ["old"])
    buf.add_session(2, ["current"])
    replay = buf.generate_replay(2)
    assert "current" not in replay
    assert "old" in replay


def test_generate_replay_empty_when_no_prior():
    """REQ-SELF-021-2: empty replay when no sessions precede current_session_id."""
    buf = LSEBMCLReplayBuffer(energy_fn=_energy)
    buf.add_session(1, ["a"])
    replay = buf.generate_replay(1)
    assert replay == []


def test_generate_replay_multiple_prior_sessions():
    """REQ-SELF-021-2: replay aggregates patterns from all prior sessions."""
    buf = LSEBMCLReplayBuffer(energy_fn=_energy, max_replay_per_session=5)
    buf.add_session(1, ["s1p1"])
    buf.add_session(2, ["s2p1"])
    buf.add_session(3, ["s3p1"])
    replay = buf.generate_replay(3)
    assert "s1p1" in replay
    assert "s2p1" in replay
    assert "s3p1" not in replay


# ── LSEBMCLReplayBuffer.compute_forgetting_rate ───────────────────────────────


def test_compute_forgetting_rate_zero_when_single_session():
    """REQ-SELF-021-3: returns 0.0 when fewer than 2 sessions provided."""
    buf = LSEBMCLReplayBuffer(energy_fn=_energy)
    rate = buf.compute_forgetting_rate([["a", "b"]])
    assert rate == pytest.approx(0.0)


def test_compute_forgetting_rate_zero_when_empty():
    """REQ-SELF-021-3: returns 0.0 for empty input."""
    buf = LSEBMCLReplayBuffer(energy_fn=_energy)
    rate = buf.compute_forgetting_rate([])
    assert rate == pytest.approx(0.0)


def test_compute_forgetting_rate_below_threshold_three_sessions():
    """REQ-SELF-021-4 / SCENARIO-SELF-028: forgetting_rate < 0.05 across 3 sessions."""
    s1 = ["COMPUTE: 47 + 28 = 76", "total is 80", "result is 15"]
    s2 = ["COMPUTE: 100 / 5 = 18", "therefore 25 apples", "sum is 90"]
    s3 = ["COMPUTE: 3 * 12 = 37", "balance is 50", "so 7 items"]

    buf = LSEBMCLReplayBuffer(energy_fn=_energy, max_replay_per_session=5)
    # 0-indexed so generate_replay(i) for i in [1,2] covers sessions 0 and 0..1.
    for sid, patterns in enumerate([s1, s2, s3], 0):
        buf.add_session(sid, patterns)

    rate = buf.compute_forgetting_rate([s1, s2, s3])
    assert rate < 0.05


def test_compute_forgetting_rate_full_forgetting():
    """REQ-SELF-021-3: rate == 1.0 when replay buffer has no prior patterns at all."""
    buf = LSEBMCLReplayBuffer(energy_fn=_energy, max_replay_per_session=5)
    # Do NOT add any sessions — so generate_replay returns nothing.
    rate = buf.compute_forgetting_rate([["a", "b"], ["c", "d"]])
    assert rate == pytest.approx(1.0)


def test_compute_forgetting_rate_partial_coverage():
    """REQ-SELF-021-3: partial overlap yields rate between 0 and 1."""
    buf = LSEBMCLReplayBuffer(energy_fn=_energy, max_replay_per_session=1)
    s1 = ["keep_me", "lose_me"]
    s2 = ["new"]
    buf.add_session(1, s1)
    buf.add_session(2, s2)
    rate = buf.compute_forgetting_rate([s1, s2])
    # Only 1 of 2 s1 patterns fits in max_replay=1, so 1 forgotten / 2 total = 0.5
    assert 0.0 < rate <= 1.0


# ── Integration: full 3-session simulation ────────────────────────────────────


def test_full_three_session_simulation():
    """REQ-SELF-021-4 / SCENARIO-SELF-027 / SCENARIO-SELF-028: end-to-end check."""
    s1 = ["COMPUTE: 47 + 28 = 76", "total is 80", "result is 15"]
    s2 = ["COMPUTE: 100 / 5 = 18", "therefore 25 apples", "sum is 90"]
    s3 = ["COMPUTE: 3 * 12 = 37", "balance is 50", "so 7 items"]

    buf = LSEBMCLReplayBuffer(energy_fn=lambda p: len(p) / 100.0, max_replay_per_session=5)
    # Sessions are 0-indexed: 0, 1, 2.
    for sid, pats in enumerate([s1, s2, s3], 0):
        buf.add_session(sid, pats)

    # When training on session 1 (id=1), replay contains session 0 = s1.
    replay_at_1 = buf.generate_replay(1)
    for p in s1:
        assert p in replay_at_1

    # When training on session 2 (id=2), replay contains sessions 0,1 = s1+s2.
    replay_at_2 = buf.generate_replay(2)
    for p in s1 + s2:
        assert p in replay_at_2

    forgetting_rate = buf.compute_forgetting_rate([s1, s2, s3])
    assert forgetting_rate < 0.05
    assert forgetting_rate == pytest.approx(0.0)
