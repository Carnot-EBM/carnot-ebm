"""Tests for ``scripts/experiment_1129_grpo_energy_prm_v2.py``.

Spec: REQ-VERIFY-083 (live_gpu provenance), REQ-LEARN-011 (continuous
self-learning), REQ-INFER-SOTA-001 (SOTA-tier model gate).

These tests cover the v2-specific pure-function helpers introduced
beyond exp1118: cosine similarity, the DRA-GRPO diversity penalty, the
CPPO proxy-reuse buffer, the v2 verdict-mapping table, and the
training-loop integration of the two new mechanisms.  We do NOT
exercise the live SOTA-model path -- that requires a 2 x RTX 3090 +
~21 GB of GGUF weights and is the wrong place for unit tests.

What is verified:

    * cosine_similarity_text returns 1.0 on identical token bags,
      0.0 on disjoint or empty inputs, and is symmetric on real text.
    * diversity_penalty_counts identifies near-duplicate clusters at
      the documented 0.90 threshold.
    * diversity_adjusted_advantages reduces to plain GRPO when no
      pair exceeds the threshold AND applies the penalty when one
      does, with the right per-completion count.
    * ProxyReuseBuffer obeys FIFO eviction at max_size, returns up
      to k entries ranked by question similarity, and degrades
      gracefully when empty.
    * derive_honest_verdict emits the v2-spec labels for every
      defined input shape (the conductor's failure ledger consumes
      these directly).
    * grpo_v2_training_pass integrates diversity penalty + proxy
      reuse end-to-end against a deterministic injected scorer,
      proving the two mechanisms compose without breaking the
      advantage-mean-zero direction GRPO needs.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_1129_grpo_energy_prm_v2.py"


def _load_module():
    """Hand-load the experiment script as a module ``exp1129``.

    Same pattern as the exp1118 test file: ``scripts/`` is not on the
    pytest import path, so we use ``importlib`` directly.  Setting
    ``sys.modules['exp1129']`` makes the load idempotent across
    multiple test functions in the same session.
    """
    spec = importlib.util.spec_from_file_location("exp1129", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp1129"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exp1129():
    return _load_module()


# ---------------------------------------------------------------------------
# cosine_similarity_text
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_strings_is_one(exp1129):
    """Identical token bags -> exact 1.0.

    The diversity penalty fires when sim > 0.90, so the upper bound
    must be exactly 1.0 for verbatim duplicates to definitely fire.
    """
    assert math.isclose(
        exp1129.cosine_similarity_text("hello world", "hello world"),
        1.0,
        abs_tol=1e-12,
    )


def test_cosine_similarity_empty_inputs_are_zero(exp1129):
    """Empty inputs cannot share tokens -> 0.0, never NaN.

    The training loop relies on a real number here so subsequent
    threshold comparisons do not raise.
    """
    assert exp1129.cosine_similarity_text("", "") == 0.0
    assert exp1129.cosine_similarity_text("hello", "") == 0.0
    assert exp1129.cosine_similarity_text("", "hello") == 0.0


def test_cosine_similarity_disjoint_tokens_are_zero(exp1129):
    """No shared tokens -> 0.0 even when both inputs are non-empty."""
    assert exp1129.cosine_similarity_text("alpha beta", "gamma delta") == 0.0


def test_cosine_similarity_is_symmetric(exp1129):
    """``sim(a,b) == sim(b,a)`` -- cosine is symmetric in the L2 inner product.

    Asymmetry would mean the duplicate-counting logic could double-
    count one direction of a pair and miss the other.
    """
    a = "step 1 plus step 2 equals 3"
    b = "the answer is step 1 plus step 2"
    assert math.isclose(
        exp1129.cosine_similarity_text(a, b),
        exp1129.cosine_similarity_text(b, a),
        abs_tol=1e-12,
    )


def test_cosine_similarity_in_unit_interval(exp1129):
    """Output is always in [0, 1] -- never negative, never > 1.

    The DRA-GRPO threshold check assumes this; values outside the
    interval would imply a bug in the dot-product / norm formula.
    """
    s = exp1129.cosine_similarity_text(
        "the cat sat on the mat",
        "a cat is sitting on a mat",
    )
    assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# diversity_penalty_counts
# ---------------------------------------------------------------------------


def test_diversity_penalty_counts_no_duplicates(exp1129):
    """When all completions are dissimilar, every count is zero."""
    completions = ["alpha one", "beta two", "gamma three"]
    counts = exp1129.diversity_penalty_counts(completions, threshold=0.90)
    assert counts == [0, 0, 0]


def test_diversity_penalty_counts_identifies_near_duplicates(exp1129):
    """Verbatim duplicates score 1.0 cosine -> count = (n-1) per duplicate."""
    completions = ["the answer is 42", "the answer is 42", "something else entirely"]
    counts = exp1129.diversity_penalty_counts(completions, threshold=0.90)
    # 0 and 1 are duplicates of each other (1 partner each); 2 is unique.
    assert counts[0] == 1
    assert counts[1] == 1
    assert counts[2] == 0


def test_diversity_penalty_counts_three_way_duplicate(exp1129):
    """Three-way duplicate cluster -> each member counts the other two."""
    completions = [
        "the answer is 42",
        "the answer is 42",
        "the answer is 42",
        "different reasoning altogether",
    ]
    counts = exp1129.diversity_penalty_counts(completions, threshold=0.90)
    assert counts[0] == 2
    assert counts[1] == 2
    assert counts[2] == 2
    assert counts[3] == 0


# ---------------------------------------------------------------------------
# diversity_adjusted_advantages
# ---------------------------------------------------------------------------


def test_diversity_adjusted_advantages_no_dups_matches_grpo(exp1129):
    """When no pair exceeds the threshold the result equals plain GRPO.

    This is the falsifiable check that v2 is a strict generalization
    of v1: turn off duplicates and the training math is identical.
    """
    scores = [0.1, 0.5, 0.9, 0.3]
    completions = ["alpha", "beta", "gamma", "delta"]
    adjusted, counts, applied = exp1129.diversity_adjusted_advantages(
        scores, completions, threshold=0.90, penalty=0.05
    )
    expected = exp1129.grpo_group_advantages(scores)
    for a, e in zip(adjusted, expected, strict=True):
        assert math.isclose(a, e, abs_tol=1e-12)
    assert counts == [0, 0, 0, 0]
    assert applied is False


def test_diversity_adjusted_advantages_applies_penalty(exp1129):
    """Near-duplicate cluster has advantages reduced by ``count * penalty``.

    With ``penalty = 0.05`` and three duplicate completions sharing
    score 1.0, each duplicate's adjusted advantage drops by
    ``2 * 0.05 = 0.10`` relative to the un-penalised case.  Anything
    less means the penalty is applied incorrectly; anything more
    means it is double-counted.
    """
    scores = [1.0, 1.0, 1.0, 0.0]
    completions = [
        "the answer is 42",
        "the answer is 42",
        "the answer is 42",
        "totally unrelated reasoning",
    ]
    adjusted, counts, applied = exp1129.diversity_adjusted_advantages(
        scores, completions, threshold=0.90, penalty=0.05
    )
    base = exp1129.grpo_group_advantages(scores)
    # Each duplicate should be base - 2 * penalty; the unique completion is unchanged.
    assert math.isclose(adjusted[0], base[0] - 2 * 0.05, abs_tol=1e-12)
    assert math.isclose(adjusted[1], base[1] - 2 * 0.05, abs_tol=1e-12)
    assert math.isclose(adjusted[2], base[2] - 2 * 0.05, abs_tol=1e-12)
    assert math.isclose(adjusted[3], base[3], abs_tol=1e-12)
    assert counts == [2, 2, 2, 0]
    assert applied is True


def test_diversity_adjusted_advantages_empty(exp1129):
    """Empty inputs yield empty outputs and ``applied=False`` -- no crash path."""
    adjusted, counts, applied = exp1129.diversity_adjusted_advantages([], [])
    assert adjusted == []
    assert counts == []
    assert applied is False


def test_diversity_adjusted_advantages_length_mismatch_raises(exp1129):
    """Mismatched list lengths are a programmer error and must raise.

    Silently truncating would corrupt the advantage estimate; a clean
    ``ValueError`` is what the training loop's pre-conditions expect.
    """
    with pytest.raises(ValueError):
        exp1129.diversity_adjusted_advantages([0.1, 0.2], ["only_one"])


# ---------------------------------------------------------------------------
# ProxyReuseBuffer
# ---------------------------------------------------------------------------


def test_proxy_buffer_starts_empty(exp1129):
    """A fresh buffer has size 0 and returns no proxies."""
    b = exp1129.ProxyReuseBuffer(max_size=5)
    assert len(b) == 0
    assert b.select_proxies("anything", k=3) == []


def test_proxy_buffer_select_returns_most_similar(exp1129):
    """select_proxies ranks by question-question cosine similarity.

    The buffer holds three entries; the new question is verbatim
    identical to entry #1; selecting k=1 must return entry #1.
    """
    b = exp1129.ProxyReuseBuffer(max_size=10)
    b.add("what is 2 plus 2", "answer is 4", 0.9)
    b.add("compute 5 times 6", "answer is 30", 0.8)
    b.add("how many apples are there", "there are 7 apples", 0.7)
    proxies = b.select_proxies("compute 5 times 6", k=1)
    assert len(proxies) == 1
    assert proxies[0]["completion"] == "answer is 30"


def test_proxy_buffer_select_k_zero_or_empty_buffer(exp1129):
    """k=0 returns empty list, regardless of buffer state."""
    b = exp1129.ProxyReuseBuffer(max_size=5)
    b.add("q1", "c1", 0.5)
    assert b.select_proxies("q1", k=0) == []


def test_proxy_buffer_fifo_eviction_at_max_size(exp1129):
    """Beyond max_size, oldest entry is evicted (FIFO)."""
    b = exp1129.ProxyReuseBuffer(max_size=2)
    b.add("first question", "first completion", 0.1)
    b.add("second question", "second completion", 0.2)
    b.add("third question", "third completion", 0.3)
    assert len(b) == 2
    # The first entry must be gone; verify by selecting all entries
    # using a query that matches every entry equally weakly.  We can
    # at least check the completion texts present.
    proxies = b.select_proxies("third question", k=2)
    assert any(p["completion"] == "third completion" for p in proxies)
    completions_all = {p["completion"] for p in b.select_proxies("anything", k=10)}
    assert "first completion" not in completions_all
    assert "second completion" in completions_all
    assert "third completion" in completions_all


def test_proxy_buffer_select_caps_at_buffer_size(exp1129):
    """Asking for more proxies than the buffer holds returns all of them."""
    b = exp1129.ProxyReuseBuffer(max_size=10)
    b.add("q1 alpha", "c1", 0.1)
    b.add("q2 beta", "c2", 0.2)
    proxies = b.select_proxies("q1 alpha", k=5)
    assert len(proxies) == 2


# ---------------------------------------------------------------------------
# derive_honest_verdict
# ---------------------------------------------------------------------------


def test_verdict_blocked_no_dualgpu_when_cuda_unavailable(exp1129):
    assert (
        exp1129.derive_honest_verdict(
            cuda_count=0,
            sota_path="/path",
            n_eval=10,
            advantage_stdev=0.1,
            improvement=0.5,
        )
        == "blocked_no_dualgpu"
    )


def test_verdict_blocked_no_dualgpu_when_sota_missing(exp1129):
    assert (
        exp1129.derive_honest_verdict(
            cuda_count=2,
            sota_path=None,
            n_eval=10,
            advantage_stdev=0.1,
            improvement=0.5,
        )
        == "blocked_no_dualgpu"
    )


def test_verdict_no_improvement_when_eval_empty(exp1129):
    """v2 collapses 'partial' into 'no_improvement' per the task spec labels."""
    assert (
        exp1129.derive_honest_verdict(
            cuda_count=2,
            sota_path="/path",
            n_eval=0,
            advantage_stdev=0.1,
            improvement=0.0,
        )
        == "no_improvement"
    )


def test_verdict_no_improvement_when_advantage_degenerate(exp1129):
    assert (
        exp1129.derive_honest_verdict(
            cuda_count=2,
            sota_path="/path",
            n_eval=5,
            advantage_stdev=0.0,
            improvement=0.0,
        )
        == "no_improvement"
    )


def test_verdict_positive_improvement(exp1129):
    assert (
        exp1129.derive_honest_verdict(
            cuda_count=2,
            sota_path="/path",
            n_eval=5,
            advantage_stdev=0.1,
            improvement=0.05,
        )
        == "positive_improvement"
    )


def test_verdict_negative_regression(exp1129):
    """v2 adds a ``negative_regression`` label that v1 did not have."""
    assert (
        exp1129.derive_honest_verdict(
            cuda_count=2,
            sota_path="/path",
            n_eval=5,
            advantage_stdev=0.1,
            improvement=-0.05,
        )
        == "negative_regression"
    )


# ---------------------------------------------------------------------------
# load_gsm8k_v2_slices offsets — pure constants check (no live HF call)
# ---------------------------------------------------------------------------


def test_v2_offsets_are_disjoint_from_v1(exp1129):
    """v2 train [500, 600) and eval [700, 750) cannot overlap v1's [250, 325).

    The retro for exp1118's ``training_wall_budget_hit`` flagged this
    explicitly: the holdout must not bleed into training across
    experiment versions.
    """
    train_lo = exp1129.GSM8K_TRAIN_OFFSET
    train_hi = train_lo + exp1129.N_TRAIN_QUESTIONS_TARGET
    eval_lo = exp1129.GSM8K_EVAL_OFFSET
    eval_hi = eval_lo + exp1129.N_EVAL_QUESTIONS
    # v1 used 250..324 (50 train + 25 eval).
    v1_lo, v1_hi = 250, 325
    assert train_hi <= v1_lo or train_lo >= v1_hi
    assert eval_hi <= v1_lo or eval_lo >= v1_hi
    # train and eval slices must be disjoint with each other too.
    assert train_hi <= eval_lo or eval_hi <= train_lo


# ---------------------------------------------------------------------------
# grpo_v2_training_pass integration
# ---------------------------------------------------------------------------


class _DeterministicLLM:
    """A test double that returns a deterministic completion per group call.

    The real ``Llama`` callable returns ``{'choices': [{'text': ...}]}``;
    we mimic that shape so ``_generate_one`` does not need to be patched.
    Returning a fixed-but-varying string per call lets the diversity
    penalty either fire (when we feed identical prompts and get cycle-
    repeated outputs) or not (when the cycle is wider than the group).
    """

    def __init__(self, outputs: list[str]):
        self._outputs = outputs
        self._i = 0

    def __call__(self, prompt: str, **kwargs) -> dict:
        text = self._outputs[self._i % len(self._outputs)]
        self._i += 1
        return {"choices": [{"text": text}]}


def test_grpo_v2_training_pass_with_unique_completions(exp1129):
    """Unique completions: diversity_penalty_applied stays False, advantages OK."""
    llm = _DeterministicLLM(
        [
            "alpha alpha alpha = 1",
            "beta beta beta = 2",
            "gamma gamma gamma = 3",
            "delta delta delta = 4",
            "epsilon epsilon epsilon = 5",
            "zeta zeta zeta = 6",
            "eta eta eta = 7",
            "theta theta theta = 8",
        ]
    )

    # Deterministic scorer: longer text -> higher score.  Bounded in [0, 1].
    def score_fn(text: str, _q: str) -> float:
        return min(1.0, len(text) / 30.0)

    questions = [{"question_id": "q0", "question": "what is the answer", "answer": 0.0}]
    meta = exp1129.grpo_v2_training_pass(
        llm,
        questions,
        group_size=8,
        wall_budget_s=60.0,
        proxy_reuse_k=0,  # disable reuse for the first-question test
        score_fn=score_fn,
    )
    assert meta["n_training_questions_processed"] == 1
    # All 8 completions are unique -> no penalty fires.
    assert meta["diversity_penalty_applied"] is False
    # advantage_stdev > 0 because scores vary by length.
    assert meta["advantage_stdev"] > 0.0


def test_grpo_v2_training_pass_fires_diversity_penalty(exp1129):
    """Cycle of 2 outputs across 8 completions -> 4-way duplicate clusters fire penalty."""
    llm = _DeterministicLLM(
        [
            "the answer is 42",
            "the result is 17 different words long total",
        ]
    )

    def score_fn(text: str, _q: str) -> float:
        # Constant score so duplicates would otherwise have zero advantage --
        # the penalty is what creates non-trivial advantages here.
        return 1.0 if "42" in text else 0.5

    questions = [{"question_id": "q0", "question": "anything", "answer": 0.0}]
    meta = exp1129.grpo_v2_training_pass(
        llm,
        questions,
        group_size=8,
        wall_budget_s=60.0,
        proxy_reuse_k=0,
        score_fn=score_fn,
    )
    assert meta["diversity_penalty_applied"] is True
    pq = meta["per_question"][0]
    assert any(c > 0 for c in pq["duplicate_counts"])


def test_grpo_v2_training_pass_proxy_reuse_kicks_in_on_second_question(exp1129):
    """First question fills buffer; second question reuses up to k=3 proxies.

    The ``n_fresh_completions_total`` should be ``8 + (8 - 3) = 13``
    rather than ``16`` -- proof CPPO actually saved inference calls.
    """
    llm = _DeterministicLLM(
        [
            "alpha = 1",
            "beta = 2",
            "gamma = 3",
            "delta = 4",
            "epsilon = 5",
            "zeta = 6",
            "eta = 7",
            "theta = 8",
            # second question's fresh completions:
            "iota = 9",
            "kappa = 10",
            "lambda = 11",
            "mu = 12",
            "nu = 13",
        ]
    )

    def score_fn(text: str, _q: str) -> float:
        return min(1.0, len(text) / 30.0)

    questions = [
        {"question_id": "q0", "question": "what is the answer", "answer": 0.0},
        {"question_id": "q1", "question": "what is the answer please", "answer": 0.0},
    ]
    meta = exp1129.grpo_v2_training_pass(
        llm,
        questions,
        group_size=8,
        wall_budget_s=60.0,
        proxy_reuse_k=3,
        score_fn=score_fn,
    )
    assert meta["n_training_questions_processed"] == 2
    assert meta["proxy_reuse_applied"] is True
    # First question: 8 fresh, 0 proxies. Second: 5 fresh, 3 proxies -> 13 total fresh.
    assert meta["n_fresh_completions_total"] == 13
    assert meta["n_proxy_reuses"] == 3


# ---------------------------------------------------------------------------
# load_thinkprm_v2_auroc — sanity check (artifact must exist on disk)
# ---------------------------------------------------------------------------


def test_load_thinkprm_v2_auroc_returns_zero_on_missing_file(exp1129, tmp_path):
    """Missing artifact must NOT raise -- the experiment can still run blocked."""
    missing = tmp_path / "nope.json"
    assert exp1129.load_thinkprm_v2_auroc(missing) == 0.0


def test_load_thinkprm_v2_auroc_reads_real_artifact(exp1129):
    """The real exp1111 artifact contains AUROC = 0.9946."""
    auroc = exp1129.load_thinkprm_v2_auroc(exp1129.THINKPRM_V2_ARTIFACT)
    assert 0.99 < auroc < 1.0


# ---------------------------------------------------------------------------
# final_answer_correct (re-implemented in v2; test only that it works)
# ---------------------------------------------------------------------------


def test_final_answer_correct_decimal(exp1129):
    assert exp1129.final_answer_correct("Total = 3.14", 3.14) is True


def test_final_answer_correct_negative(exp1129):
    assert exp1129.final_answer_correct("balance is -5", -5.0) is True


def test_final_answer_correct_no_number(exp1129):
    assert exp1129.final_answer_correct("I don't know", 7.0) is False
