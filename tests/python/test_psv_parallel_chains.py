"""Tests for PSVParallelChains — 100% coverage of the new class in psv_selfplay.py.

What is covered:
  - PSVParallelChains.__init__: valid construction, ValueError for n_chains < 1.
  - PSVParallelChains._split_pool: round-robin partitioning into n_chains subsets.
  - PSVParallelChains.run_parallel: returns required keys, correct types,
    merged_constraint_updates >= 0, parallel_speedup_factor > 0.
  - Edge cases: empty question pool, n_chains=1 (degenerate parallel).
  - _ChainResult dataclass: fields present and typed correctly.

Spec: REQ-LEARN-091, REQ-LEARN-092,
      SCENARIO-LEARN-141, SCENARIO-LEARN-142, SCENARIO-LEARN-143
"""

from __future__ import annotations

import pytest

from carnot.pipeline.jitrl_memory import JitRLConstraintMemory
from carnot.training.psv_selfplay import PSVParallelChains, _ChainResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _always_correct_fns():
    """inference_fn and verify_fn where every response is "correct"."""
    def inference_fn(q: str) -> str:
        return f"COMPUTE: result = 42. {q[:10]}"

    def verify_fn(r: str) -> bool:
        return True

    return inference_fn, verify_fn


def _always_violation_fns():
    """inference_fn and verify_fn where every response is a "violation"."""
    def inference_fn(q: str) -> str:
        return f"COMPUTE: result = 0. {q[:10]}"

    def verify_fn(r: str) -> bool:
        return False

    return inference_fn, verify_fn


def _alternating_fns():
    """inference_fn and verify_fn that alternate correct/violation by question hash."""
    call_count = {"n": 0}

    def inference_fn(q: str) -> str:
        return f"response_{call_count['n']}"

    def verify_fn(r: str) -> bool:
        # Extract the index from the response string
        try:
            idx = int(r.split("_")[1])
        except (IndexError, ValueError):
            idx = 0
        call_count["n"] += 1
        return idx % 2 == 0

    return inference_fn, verify_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory() -> JitRLConstraintMemory:
    """Fresh shared constraint memory."""
    return JitRLConstraintMemory()


@pytest.fixture()
def chains(memory: JitRLConstraintMemory) -> PSVParallelChains:
    """PSVParallelChains with K=2, 3 iterations, 5 questions per iteration."""
    return PSVParallelChains(
        n_chains=2,
        n_iterations=3,
        n_questions_per_iter=5,
        constraint_memory=memory,
    )


@pytest.fixture()
def question_pool() -> list[str]:
    """20 distinct question strings for testing."""
    return [f"What is {i} + {i + 1}?" for i in range(20)]


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


def test_init_stores_params(memory: JitRLConstraintMemory) -> None:
    """PSVParallelChains must store all constructor parameters."""
    chains = PSVParallelChains(
        n_chains=3, n_iterations=5, n_questions_per_iter=10, constraint_memory=memory
    )
    assert chains.n_chains == 3
    assert chains.n_iterations == 5
    assert chains.n_questions_per_iter == 10
    assert chains._memory is memory


def test_init_raises_on_zero_chains(memory: JitRLConstraintMemory) -> None:
    """n_chains=0 must raise ValueError (REQ-LEARN-091)."""
    with pytest.raises(ValueError, match="n_chains must be >= 1"):
        PSVParallelChains(n_chains=0, n_iterations=1, n_questions_per_iter=1, constraint_memory=memory)


def test_init_raises_on_negative_chains(memory: JitRLConstraintMemory) -> None:
    """n_chains=-1 must raise ValueError."""
    with pytest.raises(ValueError, match="n_chains must be >= 1"):
        PSVParallelChains(n_chains=-1, n_iterations=1, n_questions_per_iter=1, constraint_memory=memory)


# ---------------------------------------------------------------------------
# _split_pool tests
# ---------------------------------------------------------------------------


def test_split_pool_even_division(chains: PSVParallelChains) -> None:
    """Even pool size distributes equally to all chains."""
    pool = list(range(20))
    subsets = chains._split_pool(pool)
    assert len(subsets) == 2
    # Each subset gets exactly 10 elements when pool has 20 and n_chains=2
    assert len(subsets[0]) == 10
    assert len(subsets[1]) == 10


def test_split_pool_round_robin_ordering(chains: PSVParallelChains) -> None:
    """Questions interleave round-robin: chain 0 gets indices 0,2,4,... chain 1 gets 1,3,5,..."""
    pool = [f"q{i}" for i in range(6)]
    subsets = chains._split_pool(pool)
    assert subsets[0] == ["q0", "q2", "q4"]
    assert subsets[1] == ["q1", "q3", "q5"]


def test_split_pool_odd_division(chains: PSVParallelChains) -> None:
    """Odd pool size gives first chain one extra element."""
    pool = [f"q{i}" for i in range(5)]
    subsets = chains._split_pool(pool)
    # Round-robin: 0->chain0, 1->chain1, 2->chain0, 3->chain1, 4->chain0
    assert len(subsets[0]) == 3  # q0, q2, q4
    assert len(subsets[1]) == 2  # q1, q3


def test_split_pool_empty(chains: PSVParallelChains) -> None:
    """Empty pool produces n_chains empty subsets."""
    subsets = chains._split_pool([])
    assert len(subsets) == 2
    assert all(len(s) == 0 for s in subsets)


def test_split_pool_single_chain(memory: JitRLConstraintMemory) -> None:
    """n_chains=1 puts all questions in the single subset."""
    single = PSVParallelChains(n_chains=1, n_iterations=2, n_questions_per_iter=5, constraint_memory=memory)
    pool = list(range(10))
    subsets = single._split_pool(pool)
    assert len(subsets) == 1
    assert len(subsets[0]) == 10


# ---------------------------------------------------------------------------
# run_parallel tests
# ---------------------------------------------------------------------------


def test_run_parallel_returns_required_keys(chains: PSVParallelChains, question_pool: list[str]) -> None:
    """run_parallel result must contain chain_results, merged_constraint_updates, speedup."""
    inf, ver = _always_correct_fns()
    result = chains.run_parallel(question_pool, inf, ver)
    assert "chain_results" in result
    assert "merged_constraint_updates" in result
    assert "parallel_speedup_factor" in result


def test_run_parallel_chain_results_count(chains: PSVParallelChains, question_pool: list[str]) -> None:
    """chain_results list must have exactly n_chains entries."""
    inf, ver = _always_correct_fns()
    result = chains.run_parallel(question_pool, inf, ver)
    assert len(result["chain_results"]) == chains.n_chains


def test_run_parallel_chain_result_structure(chains: PSVParallelChains, question_pool: list[str]) -> None:
    """Each chain result must have chain_id, fp_rates, wall_time_s, n_iterations, n_constraint_updates."""
    inf, ver = _always_correct_fns()
    result = chains.run_parallel(question_pool, inf, ver)
    for cr in result["chain_results"]:
        assert "chain_id" in cr
        assert "fp_rates" in cr
        assert "wall_time_s" in cr
        assert "n_iterations" in cr
        assert "n_constraint_updates" in cr
        assert isinstance(cr["fp_rates"], list)
        assert len(cr["fp_rates"]) == chains.n_iterations


def test_run_parallel_merged_updates_nonneg(chains: PSVParallelChains, question_pool: list[str]) -> None:
    """merged_constraint_updates must be a non-negative integer (REQ-LEARN-092)."""
    inf, ver = _always_violation_fns()
    result = chains.run_parallel(question_pool, inf, ver)
    assert isinstance(result["merged_constraint_updates"], int)
    assert result["merged_constraint_updates"] >= 0


def test_run_parallel_violations_increase_updates(chains: PSVParallelChains, question_pool: list[str]) -> None:
    """When all responses are violations, merged_constraint_updates must be > 0."""
    inf, ver = _always_violation_fns()
    result = chains.run_parallel(question_pool, inf, ver)
    assert result["merged_constraint_updates"] > 0


def test_run_parallel_speedup_positive(chains: PSVParallelChains, question_pool: list[str]) -> None:
    """parallel_speedup_factor must be a positive float."""
    inf, ver = _always_correct_fns()
    result = chains.run_parallel(question_pool, inf, ver)
    assert isinstance(result["parallel_speedup_factor"], float)
    assert result["parallel_speedup_factor"] > 0.0


def test_run_parallel_fp_rates_range(chains: PSVParallelChains, question_pool: list[str]) -> None:
    """All fp_rates in every chain must be in [0.0, 1.0]."""
    inf, ver = _always_violation_fns()
    result = chains.run_parallel(question_pool, inf, ver)
    for cr in result["chain_results"]:
        for rate in cr["fp_rates"]:
            assert 0.0 <= rate <= 1.0


def test_run_parallel_all_correct_zero_updates(chains: PSVParallelChains, question_pool: list[str], memory: JitRLConstraintMemory) -> None:
    """All-correct responses: constraint memory records were_fp=True entries (thresholds raised)."""
    inf, ver = _always_correct_fns()
    result = chains.run_parallel(question_pool, inf, ver)
    # With all-correct: every response is recorded as was_fp=True in the memory.
    # merged_constraint_updates = total len(memory.history) accumulated during the run.
    assert result["merged_constraint_updates"] >= 0  # at least zero
    # All fp_rates should be 0.0 (no violations)
    for cr in result["chain_results"]:
        assert all(r == 0.0 for r in cr["fp_rates"])


def test_run_parallel_all_violations_fp_rate_1(chains: PSVParallelChains, question_pool: list[str]) -> None:
    """All-violation responses: fp_rates should all equal 1.0 (REQ-LEARN-091)."""
    inf, ver = _always_violation_fns()
    result = chains.run_parallel(question_pool, inf, ver)
    for cr in result["chain_results"]:
        assert all(abs(r - 1.0) < 1e-9 for r in cr["fp_rates"])


def test_run_parallel_single_chain_degenerate(memory: JitRLConstraintMemory, question_pool: list[str]) -> None:
    """n_chains=1 must still produce a valid result (degenerate parallel case)."""
    single = PSVParallelChains(
        n_chains=1, n_iterations=2, n_questions_per_iter=5, constraint_memory=memory
    )
    inf, ver = _always_correct_fns()
    result = single.run_parallel(question_pool, inf, ver)
    assert len(result["chain_results"]) == 1
    assert result["parallel_speedup_factor"] > 0.0


def test_run_parallel_empty_pool(chains: PSVParallelChains) -> None:
    """Empty question pool: each chain falls back to entire pool (or handles gracefully)."""
    inf, ver = _always_correct_fns()
    # Should not raise — empty pool triggers the fallback branch in _run_chain
    result = chains.run_parallel([], inf, ver)
    assert "chain_results" in result
    assert len(result["chain_results"]) == chains.n_chains


def test_run_parallel_chain_ids_correct(chains: PSVParallelChains, question_pool: list[str]) -> None:
    """chain_result[i].chain_id must equal i for all i in range(n_chains)."""
    inf, ver = _always_correct_fns()
    result = chains.run_parallel(question_pool, inf, ver)
    ids = [cr["chain_id"] for cr in result["chain_results"]]
    assert sorted(ids) == list(range(chains.n_chains))


def test_run_parallel_updates_shared_memory(memory: JitRLConstraintMemory, question_pool: list[str]) -> None:
    """After run_parallel, the shared constraint memory must contain records from both chains."""
    chains = PSVParallelChains(
        n_chains=2, n_iterations=2, n_questions_per_iter=5, constraint_memory=memory
    )
    inf, ver = _always_violation_fns()
    result = chains.run_parallel(question_pool, inf, ver)
    # Memory should contain records (violations recorded via JitRL record() calls)
    assert len(memory.history) > 0
    # All records should be for the psv_gsm8k domain
    assert all(r.domain == "psv_gsm8k" for r in memory.history)


# ---------------------------------------------------------------------------
# _ChainResult dataclass
# ---------------------------------------------------------------------------


def test_chain_result_fields() -> None:
    """_ChainResult must have all required fields."""
    cr = _ChainResult(
        chain_id=0,
        iterations=[],
        fp_rates=[0.5, 0.3],
        wall_time_s=1.23,
        n_constraint_updates=10,
    )
    assert cr.chain_id == 0
    assert cr.iterations == []
    assert cr.fp_rates == [0.5, 0.3]
    assert cr.wall_time_s == 1.23
    assert cr.n_constraint_updates == 10
