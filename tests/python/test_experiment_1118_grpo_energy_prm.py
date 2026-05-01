"""Tests for ``scripts/experiment_1118_grpo_energy_prm_v1.py``.

Spec: REQ-VERIFY-083 (live_gpu provenance), REQ-LEARN-011 (continuous
self-learning), REQ-INFER-SOTA-001 (SOTA-tier model gate).

These tests cover the pure-function helpers and the verdict-derivation
table that are the load-bearing logic of exp1118. We deliberately do
NOT exercise the live SOTA-model path: that requires a 2 × RTX 3090 +
~21 GB of GGUF weights on disk and is the wrong place for unit tests.
The test suite verifies:

    * GRPO group-relative advantages match the closed-form identity
      (``a_i = r_i - mean(r)``) and sum to zero, including the
      degenerate constant-score group case.
    * Inference-time logit-bias multipliers obey ``exp(w * a_i)`` and
      remain numerically stable for advantages on either side of zero.
    * Best-of-N selection is deterministic, returns the top index, and
      handles empty input gracefully (the eval pass relies on this
      when llama.cpp produces zero completions inside the wall budget).
    * GSM8K-style answer extraction matches the last numeric literal,
      including signs, decimals, and the empty-response edge case.
    * The honest-verdict mapping returns the canonical labels listed
      in the script docstring for every defined input shape — this is
      the artifact field the conductor's failure-ledger consumes.

Why these specific tests: every other piece of exp1118 is either a
thin wrapper over llama.cpp (which we cannot mock without recreating
the SDK) or a delegation to the existing ExperimentTemplate (covered
by its own test suite). The pure-function tests below are sufficient
to detect any regression in GRPO arithmetic, advantage scaling, or
verdict labels — which is what the conductor and retrospective
pipelines actually depend on.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_1118_grpo_energy_prm_v1.py"


def _load_module():
    """Hand-load the experiment script as a module ``exp1118``.

    We use ``importlib`` directly rather than relying on ``PYTHONPATH``
    because the conductor only injects ``scripts/`` into the path when
    running the experiment end-to-end; pytest does not.
    """
    spec = importlib.util.spec_from_file_location("exp1118", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp1118"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exp1118():
    return _load_module()


# ---------------------------------------------------------------------------
# GRPO advantage arithmetic
# ---------------------------------------------------------------------------


def test_grpo_advantages_sum_to_zero(exp1118):
    """The advantage identity ``sum(r_i - mean(r)) == 0`` is exact.

    Why this matters: GRPO's policy-gradient unbiasedness depends on
    advantages summing to zero within each group. A bug that biased
    them would shift the gradient every step.
    """
    advs = exp1118.grpo_group_advantages([0.1, 0.5, 0.9, 0.3])
    assert math.isclose(sum(advs), 0.0, abs_tol=1e-12)


def test_grpo_advantages_match_closed_form(exp1118):
    """Each advantage equals ``r_i - mean(r)`` to numerical precision."""
    scores = [0.2, 0.8, 0.4]
    expected_mean = sum(scores) / len(scores)
    advs = exp1118.grpo_group_advantages(scores)
    for s, a in zip(scores, advs, strict=True):
        assert math.isclose(a, s - expected_mean, abs_tol=1e-12)


def test_grpo_advantages_constant_group_returns_zero(exp1118):
    """A group where every completion has the same score has zero variance.

    Down-stream, ``derive_honest_verdict`` reads ``advantage_stdev <=
    1e-9`` as the "PRM degenerate" signal. The all-zero advantage list
    is what produces that stdev, so this is the load-bearing check.
    """
    advs = exp1118.grpo_group_advantages([0.5, 0.5, 0.5, 0.5])
    assert advs == [0.0, 0.0, 0.0, 0.0]


def test_grpo_advantages_empty_list(exp1118):
    """Empty input returns an empty list — no division-by-zero path."""
    assert exp1118.grpo_group_advantages([]) == []


# ---------------------------------------------------------------------------
# Logit-bias multipliers
# ---------------------------------------------------------------------------


def test_grpo_logit_bias_zero_advantage_yields_unit_bias(exp1118):
    """``exp(w * 0) == 1`` for any ``w`` — neutral completion stays neutral."""
    biases = exp1118.grpo_logit_bias([0.0, 0.0, 0.0], advantage_weight=0.1)
    assert all(math.isclose(b, 1.0, abs_tol=1e-12) for b in biases)


def test_grpo_logit_bias_signs_preserved(exp1118):
    """Positive advantage → bias > 1; negative advantage → bias < 1.

    This is what ``advantage_weight=0.1`` is supposed to encode: up-
    weight high-reward completions, down-weight low-reward ones, both
    monotonically and symmetrically around 1.0.
    """
    biases = exp1118.grpo_logit_bias(
        [0.4, -0.4, 0.0],
        advantage_weight=0.1,
    )
    assert biases[0] > 1.0
    assert biases[1] < 1.0
    assert math.isclose(biases[2], 1.0, abs_tol=1e-12)
    # symmetric around 1.0 in the multiplicative sense
    assert math.isclose(biases[0] * biases[1], 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Best-of-N selection
# ---------------------------------------------------------------------------


def test_best_of_n_returns_top(exp1118):
    """Picks the highest-scoring completion."""
    idx, text, score = exp1118.best_of_n_select(
        ["a", "b", "c"],
        [0.1, 0.9, 0.5],
    )
    assert idx == 1
    assert text == "b"
    assert math.isclose(score, 0.9, abs_tol=1e-12)


def test_best_of_n_tie_break_is_deterministic(exp1118):
    """Ties break toward the earlier index — required for reproducibility."""
    idx, text, _ = exp1118.best_of_n_select(
        ["first", "second", "third"],
        [0.5, 0.5, 0.5],
    )
    assert idx == 0
    assert text == "first"


def test_best_of_n_empty_returns_sentinel(exp1118):
    """Empty inputs yield ``(-1, "", 0.0)`` — no exception path."""
    assert exp1118.best_of_n_select([], []) == (-1, "", 0.0)


def test_best_of_n_length_mismatch_raises(exp1118):
    """Mismatched lengths surface a ValueError rather than corrupting state."""
    with pytest.raises(ValueError):
        exp1118.best_of_n_select(["a", "b"], [0.1])


# ---------------------------------------------------------------------------
# GSM8K answer extraction
# ---------------------------------------------------------------------------


def test_final_answer_correct_matches_last_int(exp1118):
    assert exp1118.final_answer_correct("step 1 ... = 12. Answer: 42", 42.0) is True


def test_final_answer_correct_decimal(exp1118):
    assert exp1118.final_answer_correct("Total = 3.14", 3.14) is True


def test_final_answer_correct_negative(exp1118):
    assert exp1118.final_answer_correct("balance is -5", -5.0) is True


def test_final_answer_correct_empty_response_is_false(exp1118):
    assert exp1118.final_answer_correct("", 1.0) is False


def test_final_answer_correct_no_number_is_false(exp1118):
    assert exp1118.final_answer_correct("I don't know", 7.0) is False


# ---------------------------------------------------------------------------
# ThinkPRM v2 score (proxy)
# ---------------------------------------------------------------------------


def test_thinkprm_v2_score_is_in_unit_interval(exp1118):
    """Reward must always lie in [0, 1] so GRPO advantages stay bounded."""
    score = exp1118.thinkprm_v2_score("Step 1: 2+3=5. Final answer: 5", "What is 2+3?")
    assert 0.0 <= score <= 1.0


def test_thinkprm_v2_score_empty_response_is_low(exp1118):
    """Empty responses get the lowest score (no length, no final number)."""
    s_empty = exp1118.thinkprm_v2_score("", "Q?")
    s_filled = exp1118.thinkprm_v2_score(
        "Step 1: 6 * 7 = 42. The answer is 42.",
        "What is 6 * 7?",
    )
    assert s_filled > s_empty


# ---------------------------------------------------------------------------
# load_thinkprm_v2_auroc
# ---------------------------------------------------------------------------


def test_load_thinkprm_v2_auroc_returns_zero_on_missing_file(exp1118, tmp_path):
    """Missing artifact must NOT raise — the experiment can still run."""
    missing = tmp_path / "nope.json"
    assert exp1118.load_thinkprm_v2_auroc(missing) == 0.0


def test_load_thinkprm_v2_auroc_reads_real_artifact(exp1118):
    """The real exp1111 artifact contains AUROC = 0.9946."""
    auroc = exp1118.load_thinkprm_v2_auroc(exp1118.THINKPRM_V2_ARTIFACT)
    assert 0.99 < auroc < 1.0


# ---------------------------------------------------------------------------
# Honest-verdict mapping
# ---------------------------------------------------------------------------


def test_verdict_blocked_when_cuda_unavailable(exp1118):
    assert (
        exp1118.derive_honest_verdict(
            cuda_count=0,
            sota_path="/path",
            n_eval=10,
            advantage_stdev=0.1,
            improvement=0.5,
        )
        == "blocked_gpu"
    )


def test_verdict_blocked_when_sota_path_missing(exp1118):
    assert (
        exp1118.derive_honest_verdict(
            cuda_count=2,
            sota_path=None,
            n_eval=10,
            advantage_stdev=0.1,
            improvement=0.5,
        )
        == "blocked_gpu"
    )


def test_verdict_partial_when_eval_empty(exp1118):
    assert (
        exp1118.derive_honest_verdict(
            cuda_count=2,
            sota_path="/path",
            n_eval=0,
            advantage_stdev=0.1,
            improvement=0.0,
        )
        == "partial"
    )


def test_verdict_neutral_when_advantage_degenerate(exp1118):
    assert (
        exp1118.derive_honest_verdict(
            cuda_count=2,
            sota_path="/path",
            n_eval=5,
            advantage_stdev=0.0,
            improvement=0.0,
        )
        == "neutral"
    )


def test_verdict_positive_improvement(exp1118):
    assert (
        exp1118.derive_honest_verdict(
            cuda_count=2,
            sota_path="/path",
            n_eval=5,
            advantage_stdev=0.1,
            improvement=0.05,
        )
        == "positive_improvement"
    )


def test_verdict_honest_negative(exp1118):
    assert (
        exp1118.derive_honest_verdict(
            cuda_count=2,
            sota_path="/path",
            n_eval=5,
            advantage_stdev=0.1,
            improvement=0.0,
        )
        == "honest_negative"
    )
