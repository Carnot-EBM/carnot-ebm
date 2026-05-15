import pytest
from carnot.phase4.alpha_t_max_caliber import alpha_t_prime_trajectory

def test_alpha_t_prime_trajectory_k1():
    # REQ-PHASE4-TRAJECTORY: alpha_t_prime_trajectory emits mld_steps values
    mld_steps = 100
    traj = alpha_t_prime_trajectory(k_verifiers=1, random_fraction=0.0, mld_steps=mld_steps, seed=100)
    assert len(traj) == mld_steps
    # At k=1, it should follow the invariant exp(-step/10) curve
    assert traj[0] == 0.04 * 1.0 * 0.0001
    assert abs(traj[99] - 0.04 * (2.718281828459045 ** (-9.9)) * 0.0001) < 1e-12

def test_alpha_t_prime_trajectory_k6():
    # REQ-PHASE4-TRAJECTORY: verify shape and distinct values for k=6
    mld_steps = 10
    traj = alpha_t_prime_trajectory(k_verifiers=6, random_fraction=0.5, mld_steps=mld_steps, seed=42)
    assert len(traj) == mld_steps
    # They should not all be identical due to rng
    assert len(set(traj)) > 1
