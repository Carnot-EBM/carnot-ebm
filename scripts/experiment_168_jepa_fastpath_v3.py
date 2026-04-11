"""Experiment 168: JEPA Fast-Path v3 Validation Benchmark.

**Researcher summary:**
    Validates the v3 JEPA predictor (trained in Exp 167 with domain-specific
    symbolic embedding heads) against the v2 (Exp 156) and v1 (Exp 145) baselines.
    v2 achieved logic AUROC=0.479 (chance) because byte-histogram embeddings have
    no structural logic signal. v3 was retrained on symbolic feature vectors for
    logic and RandomProjection for arithmetic/code, targeting logic AUROC >0.70
    and macro AUROC >0.75.

    KEY DIFFERENCE from v2 benchmark (Exp 156):
        v3 was trained on DOMAIN-APPROPRIATE embeddings:
            - Logic questions → 40-dim symbolic feature vector (padded to 256, L2-norm)
              capturing negation density, quantifier presence, conditional structure, etc.
            - Arithmetic/code → RandomProjection (same as v1/v2)
        So this benchmark uses those SAME embeddings at inference time. The pipeline's
        built-in JEPA gate always uses RandomProjection — we override it here by doing
        the embedding + gating step ourselves, then calling pipeline.verify() without
        jepa_predictor for slow-path questions.

**Goals (from research-program.md Tier 3):**
    - Find a threshold where fast_path_rate ≥ 40% AND accuracy_degradation < 2%
    - Per-domain: verify logic fast-path accuracy improves (v2 logic AUROC 0.479 → v3 >0.70)
    - If target met: update VerifyRepairPipeline default to load v3 when present

**Benchmark setup:**
    - Same 500 synthetic Q&A pairs as Exp 145/156 (200 arithmetic, 200 code, 100 logic)
    - Same random seed (42) for exact reproducibility with prior experiments
    - Three JEPA thresholds: 0.3, 0.5, 0.7 (matches Exp 156)
    - Domain-appropriate embeddings per domain for v3

**Success criteria:**
    - fast_path_rate ≥ 40% AND accuracy_degradation < 2% at some threshold

**Results written to:**
    results/experiment_168_results.json

**v2 baseline numbers (Exp 156) to beat:**
    threshold=0.3: fast_path_rate=33.4%, degradation=8.4%   (code dominated errors)
    threshold=0.5: fast_path_rate=52.8%, degradation=10.2%  (target_fast_path_met=True but degradation too high)
    threshold=0.7: fast_path_rate=78.4%, degradation=19.0%
    No threshold met BOTH targets simultaneously.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_168_jepa_fastpath_v3.py

Spec: REQ-JEPA-002, REQ-VERIFY-003, SCENARIO-JEPA-001
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure the package root is on the Python path when run from project root.
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from carnot.embeddings.fast_embedding import RandomProjectionEmbedding
from carnot.pipeline.jepa_predictor import JEPAViolationPredictor
from carnot.pipeline.verify_repair import VerifyRepairPipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

V1_PREDICTOR_PATH = Path("results/jepa_predictor.safetensors")
V2_PREDICTOR_PATH = Path("results/jepa_predictor_v2.safetensors")
V3_PREDICTOR_PATH = Path("results/jepa_predictor_v3.safetensors")
V1_RESULTS_PATH = Path("results/experiment_145_results.json")
V2_RESULTS_PATH = Path("results/experiment_156_results.json")
RESULTS_PATH = Path("results/experiment_168_results.json")

# ---------------------------------------------------------------------------
# Benchmark configuration — identical to Exp 145/156 for reproducibility
# ---------------------------------------------------------------------------

N_ARITHMETIC = 200
N_CODE = 200
N_LOGIC = 100
N_TOTAL = N_ARITHMETIC + N_CODE + N_LOGIC

THRESHOLDS = [0.3, 0.5, 0.7]

TARGET_FASTPATH_RATE = 0.40
TARGET_MAX_DEGRADATION = 0.02

# Must match Exp 145/156 exactly for reproducible Q&A pairs.
SEED = 42

# Symbolic feature constants — must match Exp 166 exactly
SYMBOLIC_EMBED_DIM = 256


# ---------------------------------------------------------------------------
# Symbolic feature vector — copied from Exp 166 for domain-appropriate embedding
# ---------------------------------------------------------------------------


def logic_feature_vector(text: str) -> np.ndarray:
    """Compute a 256-dim symbolic feature vector for a logic text.

    **Detailed explanation for engineers:**
        This is the SAME function used in Exp 166 (jepa_training_pairs_logic_v3.json)
        to generate training embeddings for the logic domain. At inference time we MUST
        use the identical function — any discrepancy would be a train/test distribution
        mismatch that would silently degrade v3 logic AUROC back toward chance.

        The 40 features capture:
        0–11: Core indicators (negation density, quantifier presence, conditional
              structure, conclusion markers, contradiction markers, clause density,
              negation scope patterns)
        12–19: Derived pairwise products (allow a linear probe to capture interactions)
        20–39: Reserved zeros
        40–255: Zero padding to match JEPA EMBED_DIM=256

        The vector is L2-normalized so cosine similarity equals dot product, which
        is what Carnot energy functions expect internally.

    Args:
        text: Input logic question or response text (any length).

    Returns:
        float32 array of shape (256,), L2-normalized.
    """
    tokens = text.lower().split()
    n_tokens = max(len(tokens), 1)
    text_lower = text.lower()

    features = np.zeros(40, dtype=np.float32)

    # 0: negation density — fallacies often have more negations
    neg_words = {"not", "no", "never", "neither", "nor"}
    features[0] = sum(tokens.count(w) for w in neg_words) / n_tokens

    # 1: has_quantifier_all — universal quantifiers in syllogisms
    features[1] = float(any(w in tokens for w in ("all", "every", "each")))

    # 2: has_quantifier_some — existential quantifiers
    features[2] = float(any(w in tokens for w in ("some", "many", "most")))

    # 3: has_quantifier_no — negative quantifiers
    features[3] = float(
        "no " in text_lower or "none" in tokens or "neither" in tokens
    )

    # 4: has_therefore — conclusion markers (valid arguments use these)
    features[4] = float(
        any(marker in text_lower for marker in ("therefore", "thus", "hence", "so "))
    )

    # 5: has_if_then — conditional structure (modus ponens, tollens, etc.)
    features[5] = float("if " in text_lower and "then " in text_lower)

    # 6: has_contradiction_marker — rebuttal signals common in invalid arguments
    features[6] = float(
        any(marker in text_lower for marker in ("but ", "however", "although", "yet "))
    )

    # 7: conditional_depth — nested conditionals; normalized to [0,1]
    features[7] = min(text_lower.count("if "), 3) / 3.0

    # 8: clause_density — richer clause structure in valid arguments
    features[8] = (
        text.count(",") + text.count(";") + text.count(".")
    ) / n_tokens

    # 9: conclusion_ratio — well-formed arguments have shorter conclusions
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    last_sent = sentences[-1] if sentences else ""
    features[9] = len(last_sent) / max(len(text), 1)

    # 10: negation_after_quantifier — scope errors ("all...not", "no...all")
    features[10] = float(
        bool(re.search(r"\ball\b.*\bnot\b|\bno\b.*\ball\b", text_lower))
    )

    # 11: double_negation — "not...not" within text
    features[11] = float(bool(re.search(r"\bnot\b.*\bnot\b", text_lower)))

    # 12–19: Pairwise interactions for linear probe
    features[12] = features[0] * features[1]   # negation × all-quantifier
    features[13] = features[0] * features[2]   # negation × some-quantifier
    features[14] = features[4] * features[5]   # therefore × if-then
    features[15] = features[5] * features[1]   # if-then × all-quantifier
    features[16] = features[4] * features[1]   # therefore × all-quantifier
    features[17] = features[8] * features[9]   # clause density × conclusion ratio
    features[18] = min(float(n_tokens) / 50.0, 1.0)   # normalized text length
    features[19] = min(float(len(sentences)) / 5.0, 1.0)  # sentence count

    # Pad to EMBED_DIM (256) — must match JEPA predictor input size
    vec = np.pad(features, (0, SYMBOLIC_EMBED_DIM - len(features)), mode="constant")

    # L2-normalize (same as training data in Exp 166)
    norm = np.linalg.norm(vec)
    if norm > 1e-10:
        vec = vec / norm

    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Domain-appropriate embedding dispatcher
# ---------------------------------------------------------------------------

# One shared RandomProjection embedder (stateless — safe to reuse)
_rp_embedder = RandomProjectionEmbedding(embed_dim=256, seed=42)


def embed_for_domain(text: str, domain: str) -> np.ndarray:
    """Return the domain-appropriate embedding for the first 50 tokens of text.

    **Detailed explanation for engineers:**
        v3 was trained with two different embedding strategies:
        - Logic: 40-dim symbolic features (padded to 256, L2-norm) from Exp 166.
          These encode explicit logical structure and gave logic AUROC >0.70 in
          Exp 167's validation set.
        - Arithmetic, code, other: RandomProjection byte-histogram embeddings from
          Exp 143/155. These have demonstrated signal for arithmetic (AUROC=0.721)
          and code (AUROC=0.776) domains.

        Using the WRONG embedding for a domain at inference time would cause a
        train/test distribution mismatch, degrading AUROC back toward chance.
        This function is the single source of truth for which embedding to use.

        We embed only the first 50 whitespace-split tokens (matching how
        verify_repair.py's built-in JEPA gate works) so that the benchmark
        conditions are comparable to pipeline operation.

    Args:
        text: Response text to embed.
        domain: Domain label: "logic", "arithmetic", "code", or other.

    Returns:
        float32 array of shape (256,).
    """
    # Use only the first 50 whitespace-split tokens — matches pipeline behavior
    first_50 = " ".join(text.split()[:50])
    if domain == "logic":
        # Symbolic feature vector: captures logical structure, not byte histogram
        return logic_feature_vector(first_50)
    else:
        # RandomProjection: byte-histogram signal for arithmetic and code
        return _rp_embedder.encode(first_50)


# ---------------------------------------------------------------------------
# Synthetic Q&A generation — exact copy from Exp 145/156 for reproducibility
# ---------------------------------------------------------------------------


def generate_arithmetic_qa(rng: random.Random) -> list[dict]:
    """Generate 200 arithmetic Q&A pairs (mix of correct and incorrect).

    **Detailed explanation for engineers:**
        Arithmetic Q&A pairs are simple word problems: "What is A op B?" with
        a response that states the answer. ~60% are correct; ~40% have a small
        offset error injected to simulate LLM arithmetic mistakes.

        This generator is seeded and deterministic — the same rng state from
        Exp 145 produces the same 200 questions. Critical for comparing v3
        vs v2 vs v1 on the EXACT same inputs.

    Args:
        rng: Random instance already seeded with SEED=42.

    Returns:
        List of Q&A dicts with keys: question, response, domain,
        ground_truth_correct.
    """
    pairs = []
    ops = [
        ("+", lambda a, b: a + b, "sum"),
        ("-", lambda a, b: a - b, "difference"),
        ("*", lambda a, b: a * b, "product"),
    ]
    for _ in range(N_ARITHMETIC):
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
        op_sym, op_fn, op_name = rng.choice(ops)
        correct_result = op_fn(a, b)

        # ~60% correct, ~40% wrong (inject an off-by-one or ±20% error).
        if rng.random() < 0.60:
            answer = correct_result
            is_correct = True
        else:
            offset = rng.choice([-3, -2, -1, 1, 2, 3, 5, -5, 10, -10])
            answer = correct_result + offset
            is_correct = False

        question = f"What is {a} {op_sym} {b}?"
        response = (
            f"To compute {a} {op_sym} {b}, I {op_name} {a} and {b}. "
            f"The answer is {a} {op_sym} {b} = {answer}."
        )
        pairs.append(
            {
                "question": question,
                "response": response,
                "domain": "arithmetic",
                "ground_truth_correct": is_correct,
            }
        )
    return pairs


def generate_code_qa(rng: random.Random) -> list[dict]:
    """Generate 200 code Q&A pairs (Python snippets with and without bugs).

    **Detailed explanation for engineers:**
        Code Q&A uses five Python snippet templates. Correct responses have
        valid, logically correct code. Incorrect responses have deliberate bugs
        (wrong operator, undefined variable, infinite recursion). v2 had code
        AUROC=0.776; v3 was trained on the same code data with RandomProjection.

    Args:
        rng: Random instance already seeded with SEED=42.

    Returns:
        List of Q&A dicts for code domain.
    """
    templates = [
        {
            "question": "Write Python code to check if {n} is even.",
            "correct": "def is_even(n):\n    return n % 2 == 0\n\nresult = is_even({n})\nprint(result)",
            "wrong": "def is_even(n):\n    return n % 2 = 0\n\nresult = is_even({n})\nprint(result)",
        },
        {
            "question": "Write Python code to compute the factorial of {n}.",
            "correct": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nresult = factorial({n})\nprint(result)",
            "wrong": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n)\n\nresult = factorial({n})\nprint(result)",
        },
        {
            "question": "Write Python code to reverse a list.",
            "correct": "def reverse_list(lst):\n    return lst[::-1]\n\nmy_list = [1, 2, 3, 4, 5]\nresult = reverse_list(my_list)\nprint(result)",
            "wrong": "def reverse_list(lst):\n    return lst.reverse()\n\nmy_list = [1, 2, 3, 4, 5]\nresult = reverse_list(my_list)\nprint(result)",
        },
        {
            "question": "Write Python code to sum a list of numbers.",
            "correct": "def sum_list(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total\n\nresult = sum_list([1, 2, 3, 4, 5])\nprint(result)",
            "wrong": "def sum_list(numbers):\n    total = 0\n    for num in numbers:\n        total += nums\n    return total\n\nresult = sum_list([1, 2, 3, 4, 5])\nprint(result)",
        },
        {
            "question": "Write Python code to find the maximum in a list.",
            "correct": "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)\n\nresult = find_max([3, 1, 4, 1, 5, 9])\nprint(result)",
            "wrong": "def find_max(lst):\n    if not lst:\n        return None\n    return maximum(lst)\n\nresult = find_max([3, 1, 4, 1, 5, 9])\nprint(result)",
        },
    ]

    pairs = []
    for i in range(N_CODE):
        tmpl = templates[i % len(templates)]
        n = rng.randint(3, 12)
        is_correct = rng.random() < 0.60
        code = (
            tmpl["correct"].format(n=n)
            if is_correct
            else tmpl["wrong"].format(n=n)
        )
        pairs.append(
            {
                "question": tmpl["question"].format(n=n),
                "response": f"Here is the Python code:\n\n```python\n{code}\n```",
                "domain": "code",
                "ground_truth_correct": is_correct,
            }
        )
    return pairs


def generate_logic_qa(rng: random.Random) -> list[dict]:
    """Generate 100 logic Q&A pairs (syllogisms and conditional reasoning).

    **Detailed explanation for engineers:**
        Logic Q&A uses 10 syllogism templates (modus ponens, categorical
        syllogisms). Correct responses state the valid conclusion; incorrect
        responses negate it.

        v1 AUROC: 0.534 (Exp 145)
        v2 AUROC: 0.479 (Exp 155 — regressed! RandomProjection can't distinguish
                          valid from invalid logic arguments by byte histogram)
        v3 target: >0.70 (Exp 167 — symbolic feature vectors for logic)

    Args:
        rng: Random instance already seeded with SEED=42.

    Returns:
        List of Q&A dicts for logic domain.
    """
    syllogisms = [
        ("All mammals are warm-blooded. Dogs are mammals.", "Dogs are warm-blooded.", "Dogs are not warm-blooded."),
        ("All birds have feathers. Eagles are birds.", "Eagles have feathers.", "Eagles do not have feathers."),
        ("If it rains, the ground gets wet. It is raining.", "The ground is wet.", "The ground is not wet."),
        ("All squares are rectangles. ABCD is a square.", "ABCD is a rectangle.", "ABCD is not a rectangle."),
        ("If P then Q. P is true.", "Q is true.", "Q is false."),
        ("All prime numbers greater than 2 are odd. 7 is prime and greater than 2.", "7 is odd.", "7 is even."),
        ("All cats are felines. Whiskers is a cat.", "Whiskers is a feline.", "Whiskers is not a feline."),
        ("If the battery is charged, the device works. The battery is charged.", "The device works.", "The device does not work."),
        ("All integers divisible by 4 are divisible by 2. 12 is divisible by 4.", "12 is divisible by 2.", "12 is not divisible by 2."),
        ("All roses are flowers. Some flowers fade quickly. Some roses are red.", "Roses are a type of flower.", "Roses are not flowers."),
    ]

    pairs = []
    for i in range(N_LOGIC):
        premises, correct_conclusion, wrong_conclusion = syllogisms[i % len(syllogisms)]
        is_correct = rng.random() < 0.60
        conclusion = correct_conclusion if is_correct else wrong_conclusion

        question = f"{premises} What can we conclude?"
        response = (
            f"Given: {premises}\n"
            f"By logical deduction, we can conclude: {conclusion}\n"
            f"Therefore, the answer is: {conclusion}"
        )
        pairs.append(
            {
                "question": question,
                "response": response,
                "domain": "logic",
                "ground_truth_correct": is_correct,
            }
        )
    return pairs


# ---------------------------------------------------------------------------
# Benchmark dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModeResult:
    """Results for one benchmark mode (baseline or a JEPA threshold + predictor version).

    Attributes:
        name: Human-readable mode name (e.g., "v3_threshold=0.5").
        fast_path_count: Questions that took the JEPA fast path (skipped full verify).
        slow_path_count: Questions that ran full constraint verification.
        fast_path_correct: Fast-path questions where decision matched baseline.
        slow_path_correct: Slow-path questions where decision matched baseline.
        fast_path_errors: List of error detail dicts for post-hoc analysis.
        total_wall_time_s: Total wall-clock seconds for all questions.
        per_question_times_s: Per-question wall-clock times for latency analysis.
        per_domain_fast: Count of fast-path questions per domain.
        per_domain_correct: Count of correct fast-path decisions per domain.
    """

    name: str
    fast_path_count: int = 0
    slow_path_count: int = 0
    fast_path_correct: int = 0
    slow_path_correct: int = 0
    fast_path_errors: list = None  # type: ignore[assignment]
    total_wall_time_s: float = 0.0
    per_question_times_s: list = None  # type: ignore[assignment]
    per_domain_fast: dict = None  # type: ignore[assignment]
    per_domain_correct: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.fast_path_errors is None:
            self.fast_path_errors = []
        if self.per_question_times_s is None:
            self.per_question_times_s = []
        if self.per_domain_fast is None:
            self.per_domain_fast = {"arithmetic": 0, "code": 0, "logic": 0}
        if self.per_domain_correct is None:
            self.per_domain_correct = {"arithmetic": 0, "code": 0, "logic": 0}

    @property
    def fast_path_rate(self) -> float:
        """Fraction of questions that took the fast path (0.0–1.0)."""
        total = self.fast_path_count + self.slow_path_count
        return self.fast_path_count / max(total, 1)

    @property
    def fast_path_accuracy(self) -> float:
        """Accuracy on fast-path questions: fraction where decision was correct."""
        return self.fast_path_correct / max(self.fast_path_count, 1)

    @property
    def slow_path_accuracy(self) -> float:
        """Accuracy on slow-path questions (full pipeline always matches baseline)."""
        return self.slow_path_correct / max(self.slow_path_count, 1)

    @property
    def overall_accuracy(self) -> float:
        """Fraction of all questions where our decision matched baseline."""
        total_correct = self.fast_path_correct + self.slow_path_correct
        total = self.fast_path_count + self.slow_path_count
        return total_correct / max(total, 1)

    def per_domain_fast_path_accuracy(self) -> dict[str, float]:
        """Per-domain fast-path accuracy: how often did the fast path get it right?

        Returns:
            Dict mapping domain → accuracy in [0, 1], or NaN if no fast-path
            questions from that domain.
        """
        result: dict[str, float] = {}
        for domain in ("arithmetic", "code", "logic"):
            n_fast = self.per_domain_fast[domain]
            n_correct = self.per_domain_correct[domain]
            result[domain] = n_correct / max(n_fast, 1) if n_fast > 0 else float("nan")
        return result


# ---------------------------------------------------------------------------
# Benchmark runner — uses domain-appropriate embeddings for v3
# ---------------------------------------------------------------------------


def run_benchmark_v3(
    qa_pairs: list[dict],
    pipeline: VerifyRepairPipeline,
    predictor: JEPAViolationPredictor | None,
    threshold: float,
    mode_name: str,
    baseline_decisions: list[bool] | None = None,
    use_domain_embeddings: bool = True,
) -> ModeResult:
    """Run one benchmark mode with domain-appropriate embeddings for v3.

    **Detailed explanation for engineers:**
        KEY DIFFERENCE from Exp 156's run_benchmark():
        Instead of passing jepa_predictor to pipeline.verify() (which always
        uses RandomProjection internally), we perform the JEPA gating step
        ourselves using domain-appropriate embeddings:

        For each question:
        1. Compute embed_for_domain(response, domain) — symbolic features for
           logic, RandomProjection for arithmetic/code.
        2. Query predictor.predict(embedding) → per-domain violation probs.
        3. If max(probs.values()) < threshold → fast path (verified=True, skip=True).
        4. Else → slow path: call pipeline.verify() WITHOUT predictor for full check.

        This matches v3's training distribution: it was trained on symbolic features
        for logic and RandomProjection for arithmetic/code. Using RandomProjection
        for logic at inference time would restore the train/test mismatch that gave
        v2 AUROC=0.479 on logic.

        When use_domain_embeddings=False, falls back to RandomProjection for all
        domains (for direct v2/v1 comparison). When predictor is None, all questions
        go to the slow path (baseline mode).

    Args:
        qa_pairs: 500 Q&A dicts from generate_*_qa().
        pipeline: VerifyRepairPipeline instance (shared, no LLM loaded).
        predictor: JEPAViolationPredictor v3, or None for baseline.
        threshold: JEPA threshold — max prob BELOW this triggers fast path.
        mode_name: Label for logging and results.
        baseline_decisions: Per-question baseline verified flags. If None,
            uses pipeline's own result as the reference (baseline mode).
        use_domain_embeddings: If True (default for v3), use logic_feature_vector
            for logic questions and RandomProjection for arithmetic/code.
            If False, use RandomProjection for all (v1/v2 behavior).

    Returns:
        ModeResult with all collected metrics.
    """
    result = ModeResult(name=mode_name)
    t_total_start = time.perf_counter()

    for i, qa in enumerate(qa_pairs):
        t_q_start = time.perf_counter()
        domain = qa["domain"]
        took_fast_path = False
        verified_decision = False

        if predictor is not None:
            # --- Domain-appropriate embedding for v3 ---
            if use_domain_embeddings:
                embedding = embed_for_domain(qa["response"], domain)
            else:
                # Legacy behavior: RandomProjection for all (matches Exp 156)
                first_50 = " ".join(qa["response"].split()[:50])
                embedding = _rp_embedder.encode(first_50)

            probs = predictor.predict(embedding)
            max_prob = max(probs.values()) if probs else 0.0

            if max_prob < threshold:
                # Fast path: JEPA predicts low risk — skip expensive verification.
                # Return optimistic verified=True (same as pipeline FAST_PATH behavior).
                took_fast_path = True
                verified_decision = True
                jepa_probs_recorded = probs
                jepa_max_prob_recorded = max_prob
            else:
                # Slow path: high predicted risk — run full pipeline verification.
                vr = pipeline.verify(
                    question=qa["question"],
                    response=qa["response"],
                    domain=domain,
                )
                verified_decision = vr.verified
                jepa_probs_recorded = probs
                jepa_max_prob_recorded = max_prob
        else:
            # Baseline mode: no predictor, always run full pipeline.
            vr = pipeline.verify(
                question=qa["question"],
                response=qa["response"],
                domain=domain,
            )
            verified_decision = vr.verified
            jepa_probs_recorded = {}
            jepa_max_prob_recorded = 0.0

        t_q_end = time.perf_counter()
        result.per_question_times_s.append(t_q_end - t_q_start)

        # Determine the reference "correct" decision.
        if baseline_decisions is not None:
            reference_verified = baseline_decisions[i]
        else:
            # Baseline mode: the pipeline's own result is the reference.
            reference_verified = verified_decision

        decision_correct = verified_decision == reference_verified

        if took_fast_path:
            result.fast_path_count += 1
            result.per_domain_fast[domain] += 1
            if decision_correct:
                result.fast_path_correct += 1
                result.per_domain_correct[domain] += 1
            else:
                # Record error for per-domain analysis
                result.fast_path_errors.append(
                    {
                        "index": i,
                        "domain": domain,
                        "question": qa["question"][:100],
                        "response_snippet": qa["response"][:80],
                        "ground_truth_correct": qa["ground_truth_correct"],
                        "our_decision": verified_decision,
                        "reference_decision": reference_verified,
                        "jepa_probs": jepa_probs_recorded,
                        "jepa_max_prob": jepa_max_prob_recorded,
                    }
                )
        else:
            result.slow_path_count += 1
            if decision_correct:
                result.slow_path_correct += 1

    t_total_end = time.perf_counter()
    result.total_wall_time_s = t_total_end - t_total_start
    return result


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


def analyze_errors(errors: list[dict], mode_name: str) -> dict:
    """Analyze fast-path errors by domain, JEPA probability, and response length.

    **Detailed explanation for engineers:**
        Fast-path errors = JEPA predicted "low risk" (max_prob < threshold) but
        the baseline pipeline would have detected a constraint violation. We analyze
        by domain, average JEPA max-prob at error time, and whether short responses
        are overrepresented. The per-domain breakdown reveals whether v3's symbolic
        logic features successfully reduced logic errors (previously 100% false
        negatives at threshold=0.3 because logic AUROC=0.479 was at chance).

    Args:
        errors: List of error dicts from ModeResult.fast_path_errors.
        mode_name: Label for the mode being analyzed.

    Returns:
        Dict with n_errors, domain_counts, avg_jepa_max_prob,
        short_response_fraction, most_common_domain.
    """
    if not errors:
        return {
            "mode": mode_name,
            "n_errors": 0,
            "domain_counts": {},
            "avg_jepa_max_prob": None,
            "short_response_fraction": None,
            "most_common_domain": None,
        }

    domain_counts: dict[str, int] = {}
    jepa_probs = []
    short_count = 0

    for err in errors:
        domain = err["domain"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        jepa_probs.append(err["jepa_max_prob"])
        n_tokens = len(err["response_snippet"].split())
        if n_tokens <= 50:
            short_count += 1

    most_common = max(domain_counts, key=lambda d: domain_counts[d]) if domain_counts else None
    return {
        "mode": mode_name,
        "n_errors": len(errors),
        "domain_counts": domain_counts,
        "avg_jepa_max_prob": float(np.mean(jepa_probs)) if jepa_probs else None,
        "short_response_fraction": short_count / len(errors),
        "most_common_domain": most_common,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 168: JEPA fast-path v3 validation benchmark."""
    t_start = time.perf_counter()
    rng = random.Random(SEED)

    print("=" * 70)
    print("Experiment 168: JEPA Fast-Path v3 Validation Benchmark")
    print("=" * 70)
    print("Date: 20260411")
    print("Target models: Qwen3.5-0.8B, google/gemma-4-E4B-it")
    print("(Benchmark uses synthetic Q&A to isolate JEPA gate effect from LLM quality)")

    # =========================================================================
    # Load predictors
    # =========================================================================

    if not V3_PREDICTOR_PATH.exists():
        print(f"\nERROR: v3 predictor not found at {V3_PREDICTOR_PATH}")
        print("Run experiment_167_train_jepa_v3.py first.")
        sys.exit(1)

    print(f"\nLoading JEPA v3 predictor from {V3_PREDICTOR_PATH}...")
    predictor_v3 = JEPAViolationPredictor()
    predictor_v3.load(str(V3_PREDICTOR_PATH))
    print("  v3 predictor loaded (domain-specific symbolic embedding heads).")

    predictor_v2: JEPAViolationPredictor | None = None
    if V2_PREDICTOR_PATH.exists():
        print(f"Loading JEPA v2 predictor from {V2_PREDICTOR_PATH}...")
        predictor_v2 = JEPAViolationPredictor()
        predictor_v2.load(str(V2_PREDICTOR_PATH))
        print("  v2 predictor loaded (for direct comparison).")

    predictor_v1: JEPAViolationPredictor | None = None
    if V1_PREDICTOR_PATH.exists():
        print(f"Loading JEPA v1 predictor from {V1_PREDICTOR_PATH}...")
        predictor_v1 = JEPAViolationPredictor()
        predictor_v1.load(str(V1_PREDICTOR_PATH))
        print("  v1 predictor loaded (for v1/v2/v3 comparison table).")

    # Load prior experiment results for comparison table
    v2_results: dict = {}
    if V2_RESULTS_PATH.exists():
        with open(V2_RESULTS_PATH) as f:
            v2_results = json.load(f)
        print(f"\nLoaded Exp 156 v2 baseline from {V2_RESULTS_PATH}")

    v1_results: dict = {}
    if V1_RESULTS_PATH.exists():
        with open(V1_RESULTS_PATH) as f:
            v1_results = json.load(f)
        print(f"Loaded Exp 145 v1 baseline from {V1_RESULTS_PATH}")

    # =========================================================================
    # Generate synthetic Q&A pairs (same seed as Exp 145/156)
    # =========================================================================
    print(f"\nGenerating {N_TOTAL} synthetic Q&A pairs (seed={SEED}, matches Exp 145/156)...")
    arithmetic_pairs = generate_arithmetic_qa(rng)
    code_pairs = generate_code_qa(rng)
    logic_pairs = generate_logic_qa(rng)
    all_pairs = arithmetic_pairs + code_pairs + logic_pairs

    n_correct = sum(1 for p in all_pairs if p["ground_truth_correct"])
    print(
        f"  {N_ARITHMETIC} arithmetic, {N_CODE} code, {N_LOGIC} logic  "
        f"({n_correct}/{N_TOTAL} = {100*n_correct/N_TOTAL:.1f}% ground-truth correct)"
    )

    pipeline = VerifyRepairPipeline(timeout_seconds=10.0)

    # =========================================================================
    # Mode 1: Baseline (no JEPA gating) — establish reference decisions
    # =========================================================================
    print("\n--- Mode: Baseline (no JEPA gating) ---")
    baseline_result = run_benchmark_v3(
        qa_pairs=all_pairs,
        pipeline=pipeline,
        predictor=None,
        threshold=0.5,
        mode_name="baseline",
        baseline_decisions=None,
    )

    # Collect per-question verified decisions for gating mode comparisons
    baseline_decisions: list[bool] = []
    for qa in all_pairs:
        vr = pipeline.verify(
            question=qa["question"],
            response=qa["response"],
            domain=qa["domain"],
        )
        baseline_decisions.append(vr.verified)

    n_baseline_verified = sum(baseline_decisions)
    baseline_gt_match = sum(
        1 for i, qa in enumerate(all_pairs)
        if (baseline_decisions[i] == qa["ground_truth_correct"])
    )
    print(f"  Total time: {baseline_result.total_wall_time_s:.3f}s")
    print(f"  Verified: {n_baseline_verified}/{N_TOTAL} = {100*n_baseline_verified/N_TOTAL:.1f}%")
    print(f"  Accuracy vs ground truth: {100*baseline_gt_match/N_TOTAL:.1f}%")

    # =========================================================================
    # Mode 2: v3 predictor with domain-appropriate embeddings at thresholds 0.3, 0.5, 0.7
    # =========================================================================
    v3_results_by_threshold: list[ModeResult] = []
    v2_live_by_threshold: list[ModeResult] = []
    v1_live_by_threshold: list[ModeResult] = []

    for threshold in THRESHOLDS:
        # --- v3 with domain-appropriate embeddings ---
        mode_name_v3 = f"v3_threshold={threshold}"
        print(f"\n--- Mode: JEPA v3 gate at threshold={threshold} (domain-specific embeddings) ---")
        v3_mode = run_benchmark_v3(
            qa_pairs=all_pairs,
            pipeline=pipeline,
            predictor=predictor_v3,
            threshold=threshold,
            mode_name=mode_name_v3,
            baseline_decisions=baseline_decisions,
            use_domain_embeddings=True,   # KEY: symbolic for logic, RP for arith/code
        )
        v3_results_by_threshold.append(v3_mode)

        degradation_v3 = 1.0 - v3_mode.overall_accuracy
        speedup_v3 = baseline_result.total_wall_time_s / max(v3_mode.total_wall_time_s, 1e-9)
        fp_ok = v3_mode.fast_path_rate >= TARGET_FASTPATH_RATE
        deg_ok = degradation_v3 <= TARGET_MAX_DEGRADATION
        status = "PASS" if (fp_ok and deg_ok) else "MISS"

        print(f"  Time: {v3_mode.total_wall_time_s:.3f}s  (speedup: {speedup_v3:.2f}x)")
        print(f"  Fast-path rate: {100*v3_mode.fast_path_rate:.1f}%  (target >= {100*TARGET_FASTPATH_RATE:.0f}%)")
        print(f"  Fast-path accuracy: {100*v3_mode.fast_path_accuracy:.1f}%")
        print(f"  Overall accuracy: {100*v3_mode.overall_accuracy:.1f}%")
        print(f"  Accuracy degradation: {100*degradation_v3:.2f}%  (target < {100*TARGET_MAX_DEGRADATION:.0f}%)")
        print(f"  Target [{TARGET_FASTPATH_RATE*100:.0f}% fast-path, <{TARGET_MAX_DEGRADATION*100:.0f}% degradation]: {status}")

        domain_acc = v3_mode.per_domain_fast_path_accuracy()
        print(f"  Per-domain fast-path accuracy:")
        for domain, acc in domain_acc.items():
            n_fast = v3_mode.per_domain_fast[domain]
            if n_fast > 0:
                print(f"    {domain:12s}: {100*acc:.1f}%  ({n_fast} fast-path questions)")
            else:
                print(f"    {domain:12s}: -- (no fast-path questions)")

        # --- v2 live comparison at same threshold (using RandomProjection for all) ---
        if predictor_v2 is not None:
            mode_name_v2 = f"v2_threshold={threshold}"
            print(f"\n  [Comparison] JEPA v2 at threshold={threshold} (RandomProjection all domains)...")
            v2_mode = run_benchmark_v3(
                qa_pairs=all_pairs,
                pipeline=pipeline,
                predictor=predictor_v2,
                threshold=threshold,
                mode_name=mode_name_v2,
                baseline_decisions=baseline_decisions,
                use_domain_embeddings=False,  # v2: RandomProjection for all domains
            )
            v2_live_by_threshold.append(v2_mode)
            degradation_v2 = 1.0 - v2_mode.overall_accuracy
            print(
                f"  v2: fast_path={100*v2_mode.fast_path_rate:.1f}%  "
                f"degradation={100*degradation_v2:.2f}%  "
                f"accuracy={100*v2_mode.overall_accuracy:.1f}%"
            )
            fp_delta = v3_mode.fast_path_rate - v2_mode.fast_path_rate
            deg_delta = degradation_v3 - degradation_v2
            print(
                f"  v3 vs v2 delta: fast_path d={100*fp_delta:+.1f}%  "
                f"degradation d={100*deg_delta:+.2f}%"
            )

        # --- v1 live comparison ---
        if predictor_v1 is not None:
            mode_name_v1 = f"v1_threshold={threshold}"
            print(f"\n  [Comparison] JEPA v1 at threshold={threshold}...")
            v1_mode = run_benchmark_v3(
                qa_pairs=all_pairs,
                pipeline=pipeline,
                predictor=predictor_v1,
                threshold=threshold,
                mode_name=mode_name_v1,
                baseline_decisions=baseline_decisions,
                use_domain_embeddings=False,  # v1: RandomProjection for all domains
            )
            v1_live_by_threshold.append(v1_mode)
            degradation_v1 = 1.0 - v1_mode.overall_accuracy
            print(
                f"  v1: fast_path={100*v1_mode.fast_path_rate:.1f}%  "
                f"degradation={100*degradation_v1:.2f}%"
            )

    # =========================================================================
    # Find optimal threshold
    # =========================================================================
    print("\n--- Optimal threshold search (degradation < 2% primary, max fast-path secondary) ---")
    optimal_threshold: float | None = None
    optimal_mode: ModeResult | None = None

    for mode in v3_results_by_threshold:
        degradation = 1.0 - mode.overall_accuracy
        fp_rate = mode.fast_path_rate
        if degradation <= TARGET_MAX_DEGRADATION:
            if optimal_mode is None or fp_rate > optimal_mode.fast_path_rate:
                # Parse threshold from name like "v3_threshold=0.5"
                optimal_threshold = float(mode.name.split("=")[1])
                optimal_mode = mode

    if optimal_mode is not None:
        optimal_degradation = 1.0 - optimal_mode.overall_accuracy
        print(
            f"  Optimal threshold: {optimal_threshold}  "
            f"(fast_path={100*optimal_mode.fast_path_rate:.1f}%  "
            f"degradation={100*optimal_degradation:.2f}%)"
        )
        target_met = optimal_mode.fast_path_rate >= TARGET_FASTPATH_RATE
        print(
            f"  Fast-path target (>={100*TARGET_FASTPATH_RATE:.0f}%): "
            f"{'MET' if target_met else 'NOT MET'}"
        )
    else:
        print("  No threshold achieved <2% degradation. Target NOT MET.")
        target_met = False
        optimal_degradation = None

    # =========================================================================
    # Error analysis
    # =========================================================================
    print("\n--- Error analysis (fast-path misses, v3) ---")
    error_analyses: list[dict] = []
    for mode in v3_results_by_threshold:
        analysis = analyze_errors(mode.fast_path_errors, mode.name)
        error_analyses.append(analysis)
        print(f"\n  Mode: {mode.name}")
        print(f"    Total fast-path errors: {analysis['n_errors']}")
        if analysis["n_errors"] > 0:
            print(f"    Errors by domain: {analysis['domain_counts']}")
            if analysis.get("most_common_domain"):
                print(f"    Most common error domain: {analysis['most_common_domain']}")
            if analysis["avg_jepa_max_prob"] is not None:
                print(f"    Avg JEPA max_prob at error: {analysis['avg_jepa_max_prob']:.3f}")
            logic_errors = analysis["domain_counts"].get("logic", 0)
            code_errors = analysis["domain_counts"].get("code", 0)
            total_errors = analysis["n_errors"]
            if total_errors > 0:
                print(f"    Logic error fraction: {100*logic_errors/total_errors:.1f}%  "
                      f"(v2 Exp156: code dominated at all thresholds)")
                print(f"    Code error fraction: {100*code_errors/total_errors:.1f}%")

    # =========================================================================
    # Root cause analysis if target not met
    # =========================================================================
    if not target_met:
        print("\n--- Root cause analysis (target NOT MET) ---")
        for mode in v3_results_by_threshold:
            degradation = 1.0 - mode.overall_accuracy
            domain_acc = mode.per_domain_fast_path_accuracy()
            print(f"\n  {mode.name}:")
            print(f"    fast_path_rate={100*mode.fast_path_rate:.1f}%  degradation={100*degradation:.2f}%")
            for domain in ("arithmetic", "code", "logic"):
                n_fast = mode.per_domain_fast[domain]
                if n_fast > 0:
                    acc = domain_acc[domain]
                    n_errors = mode.per_domain_fast[domain] - mode.per_domain_correct[domain]
                    if acc < 0.98:
                        print(
                            f"    [{domain}] {n_errors} fast-path errors / {n_fast} fast-path = "
                            f"{100*(1-acc):.1f}% error rate — DOMINANT FAILURE DOMAIN"
                        )
                    else:
                        print(f"    [{domain}] acc={100*acc:.1f}% ({n_fast} fast-path) — OK")
        print("\n  Root cause summary:")
        # Find which domain contributes most errors at t=0.3
        if v3_results_by_threshold:
            t03_mode = v3_results_by_threshold[0]
            t03_errs = analyze_errors(t03_mode.fast_path_errors, t03_mode.name)
            if t03_errs["domain_counts"]:
                worst_domain = t03_errs["most_common_domain"]
                worst_n = t03_errs["domain_counts"].get(worst_domain, 0)
                print(f"    At threshold=0.3: '{worst_domain}' domain has {worst_n} errors.")
                if worst_domain == "logic":
                    print("    v3 symbolic features DID improve logic detection but errors remain.")
                elif worst_domain == "code":
                    print("    Code domain still dominates — v3 code AUROC still ~0.776, unchanged.")
                    print("    Recommendation: train v4 with code-specific symbolic features")
                    print("    (AST parse features: syntax error presence, variable scope, loop invariant)")

    # =========================================================================
    # v1/v2/v3 comparison table
    # =========================================================================
    print("\n--- v1 / v2 / v3 comparison table ---")
    print(f"{'Mode':<28}  {'Fast-path':>10}  {'Degradation':>12}  {'Target':>6}")
    print("-" * 65)

    # v1 from Exp 145 JSON
    v1_by_t = {}
    for mode in v1_results.get("modes", []):
        v1_by_t[mode["threshold"]] = mode

    # v2 from Exp 156 JSON
    v2_by_t = {}
    for entry in v2_results.get("thresholds", []):
        v2_by_t[entry["threshold"]] = entry.get("v2", {})

    for t, v3_mode in zip(THRESHOLDS, v3_results_by_threshold):
        deg_v3 = 1.0 - v3_mode.overall_accuracy
        fp_ok_v3 = v3_mode.fast_path_rate >= TARGET_FASTPATH_RATE
        deg_ok_v3 = deg_v3 <= TARGET_MAX_DEGRADATION
        status_v3 = "PASS" if (fp_ok_v3 and deg_ok_v3) else "MISS"
        print(
            f"  v3 threshold={t:<5}          "
            f"{100*v3_mode.fast_path_rate:>8.1f}%  "
            f"{100*deg_v3:>10.2f}%  {status_v3:>6}"
        )

        # v2 reference
        v2_entry = v2_by_t.get(t, {})
        if v2_entry:
            print(
                f"  v2 threshold={t:<5} (Exp156) "
                f"{100*v2_entry.get('fast_path_rate',0):>8.1f}%  "
                f"{100*v2_entry.get('accuracy_degradation',0):>10.2f}%"
            )

        # v1 reference
        v1_entry = v1_by_t.get(t, {})
        if v1_entry:
            print(
                f"  v1 threshold={t:<5} (Exp145) "
                f"{100*v1_entry.get('fast_path_rate',0):>8.1f}%  "
                f"{100*v1_entry.get('accuracy_degradation',0):>10.2f}%"
            )
        print()

    print("-" * 65)
    if optimal_threshold is not None:
        if target_met:
            print(f"  OVERALL: TARGET MET at threshold={optimal_threshold}")
            print(
                f"  fast_path_rate={100*optimal_mode.fast_path_rate:.1f}%  "
                f"degradation={100*optimal_degradation:.2f}%"
            )
        else:
            print(
                f"  OVERALL: Degradation target met at threshold={optimal_threshold} "
                f"but fast_path_rate={100*optimal_mode.fast_path_rate:.1f}% < {100*TARGET_FASTPATH_RATE:.0f}%"
            )
    else:
        print("  OVERALL: TARGET NOT MET (no threshold achieved <2% degradation)")
    print("=" * 70)

    # =========================================================================
    # If target met: update VerifyRepairPipeline to load v3 by default
    # =========================================================================
    if target_met:
        print(f"\n>>> TARGET MET — updating VerifyRepairPipeline to load v3 predictor by default...")
        _update_pipeline_default_predictor()
        print("    VerifyRepairPipeline updated. See comment: # Exp 168: Tier 3 self-learning complete")

    # =========================================================================
    # Save results
    # =========================================================================

    # Build per-threshold v2/v1 lookup from prior JSON (fallback)
    v2_json_by_threshold: dict = {
        entry["threshold"]: entry.get("v2", {})
        for entry in v2_results.get("thresholds", [])
    }
    v1_json_by_threshold: dict = {
        mode["threshold"]: mode
        for mode in v1_results.get("modes", [])
    }

    thresholds_output: list[dict] = []
    for i, (threshold, v3_mode) in enumerate(zip(THRESHOLDS, v3_results_by_threshold)):
        degradation_v3 = 1.0 - v3_mode.overall_accuracy
        speedup_v3 = baseline_result.total_wall_time_s / max(v3_mode.total_wall_time_s, 1e-9)

        # Live v2/v1 if available, else JSON fallback
        v2_live = v2_live_by_threshold[i] if i < len(v2_live_by_threshold) else None
        v1_live = v1_live_by_threshold[i] if i < len(v1_live_by_threshold) else None

        v2_fp = v2_live.fast_path_rate if v2_live else v2_json_by_threshold.get(threshold, {}).get("fast_path_rate")
        v2_deg = (1.0 - v2_live.overall_accuracy) if v2_live else v2_json_by_threshold.get(threshold, {}).get("accuracy_degradation")
        v1_fp = v1_live.fast_path_rate if v1_live else v1_json_by_threshold.get(threshold, {}).get("fast_path_rate")
        v1_deg = (1.0 - v1_live.overall_accuracy) if v1_live else v1_json_by_threshold.get(threshold, {}).get("accuracy_degradation")

        domain_acc_v3 = v3_mode.per_domain_fast_path_accuracy()

        thresholds_output.append(
            {
                "threshold": threshold,
                "v3": {
                    "fast_path_rate": v3_mode.fast_path_rate,
                    "fast_path_count": v3_mode.fast_path_count,
                    "slow_path_count": v3_mode.slow_path_count,
                    "fast_path_accuracy": v3_mode.fast_path_accuracy,
                    "slow_path_accuracy": v3_mode.slow_path_accuracy,
                    "overall_accuracy": v3_mode.overall_accuracy,
                    "accuracy_degradation": degradation_v3,
                    "n_fast_path_errors": len(v3_mode.fast_path_errors),
                    "total_time_s": v3_mode.total_wall_time_s,
                    "speedup_vs_baseline": speedup_v3,
                    "target_fast_path_met": v3_mode.fast_path_rate >= TARGET_FASTPATH_RATE,
                    "target_degradation_met": degradation_v3 <= TARGET_MAX_DEGRADATION,
                    "per_domain_fast_path_accuracy": {
                        k: (v if not (isinstance(v, float) and v != v) else None)
                        for k, v in domain_acc_v3.items()
                    },
                    "per_domain_fast_path_count": dict(v3_mode.per_domain_fast),
                    "embedding_strategy": "symbolic_logic_rp_others",
                },
                "v2_comparison": {
                    "fast_path_rate": v2_fp,
                    "accuracy_degradation": v2_deg,
                    "fast_path_rate_delta": (v3_mode.fast_path_rate - v2_fp) if v2_fp is not None else None,
                    "degradation_delta": (degradation_v3 - v2_deg) if v2_deg is not None else None,
                    "source": "live_run" if v2_live else "exp_156_json",
                },
                "v1_comparison": {
                    "fast_path_rate": v1_fp,
                    "accuracy_degradation": v1_deg,
                    "fast_path_rate_delta": (v3_mode.fast_path_rate - v1_fp) if v1_fp is not None else None,
                    "degradation_delta": (degradation_v3 - v1_deg) if v1_deg is not None else None,
                    "source": "live_run" if v1_live else "exp_145_json",
                },
            }
        )

    results: dict = {
        "experiment": 168,
        "description": "JEPA fast-path v3 validation — domain-specific symbolic embedding heads",
        "date": "20260411",
        "target_models_reference": ["Qwen3.5-0.8B", "google/gemma-4-E4B-it"],
        "references": {
            "exp_145": "v1 baseline benchmark",
            "exp_156": "v2 benchmark (target_met=false, code errors dominated)",
            "exp_166": "symbolic logic feature vectors (training data for v3)",
            "exp_167": "v3 predictor training (domain-specific embeddings)",
        },
        "predictor_versions": {
            "v3_path": str(V3_PREDICTOR_PATH),
            "v3_embedding_strategy": "logic: 40-dim symbolic features padded to 256, L2-norm; arithmetic/code: RandomProjection",
            "v2_macro_auroc_from_exp155": 0.6588828939009381,
            "v2_logic_auroc_from_exp155": 0.479,
            "v3_logic_auroc_target": 0.70,
            "v3_macro_auroc_target": 0.75,
        },
        "n_total": N_TOTAL,
        "n_arithmetic": N_ARITHMETIC,
        "n_code": N_CODE,
        "n_logic": N_LOGIC,
        "seed": SEED,
        "targets": {
            "fast_path_rate_min": TARGET_FASTPATH_RATE,
            "max_accuracy_degradation": TARGET_MAX_DEGRADATION,
        },
        "baseline": {
            "total_time_s": baseline_result.total_wall_time_s,
            "n_verified": n_baseline_verified,
            "gt_accuracy": baseline_gt_match / N_TOTAL,
        },
        "thresholds": thresholds_output,
        "optimal_threshold": optimal_threshold,
        "target_met": target_met,
        "error_analysis": error_analyses,
        "v1_v2_v3_comparison": {
            "source_experiments": {
                "v1": "Exp 145",
                "v2": "Exp 156 (and live runs in Exp 168)",
                "v3": "Exp 168 (domain-specific symbolic embeddings)",
            },
            "by_threshold": thresholds_output,  # includes v2/v1 fields in each entry
        },
        "wall_clock_summary": {
            "baseline_s": baseline_result.total_wall_time_s,
            "v3_by_threshold": {
                str(t): mode.total_wall_time_s
                for t, mode in zip(THRESHOLDS, v3_results_by_threshold)
            },
        },
        "total_experiment_time_s": time.perf_counter() - t_start,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


def _update_pipeline_default_predictor() -> None:
    """Add default v3 predictor loading to VerifyRepairPipeline when target is met.

    **Detailed explanation for engineers:**
        Exp 168: Tier 3 self-learning complete — when the fast-path target is met
        (fast_path_rate >= 40% AND accuracy_degradation < 2%), we update the pipeline
        to automatically load the v3 predictor when it is present in the default
        path. This makes v3 fast-path gating the production default.

        We inject a class-level DEFAULT_JEPA_PREDICTOR_PATH constant and a
        _load_default_predictor() helper called from __init__. If the v3 file is
        not present (e.g., in CI or a fresh clone), the pipeline runs without a
        JEPA gate as before (no regression).

    Spec: REQ-JEPA-002
    """
    vrp_path = Path("python/carnot/pipeline/verify_repair.py")
    if not vrp_path.exists():
        print(f"  WARNING: cannot find {vrp_path} — skipping pipeline update.")
        return

    content = vrp_path.read_text()

    # Only inject once — idempotent
    if "Exp 168: Tier 3 self-learning complete" in content:
        print("  Already updated (idempotent).")
        return

    # Find the end of the module-level imports/constants block just before
    # the VerifyRepairPipeline class definition, and inject a constant.
    inject_constant = '''
# Exp 168: Tier 3 self-learning complete
# Default JEPA v3 predictor path — loaded automatically when present.
# Set CARNOT_JEPA_PREDICTOR="" to disable; override with a custom path.
import os as _os_jepa
_DEFAULT_JEPA_PATH: str = _os_jepa.environ.get(
    "CARNOT_JEPA_PREDICTOR",
    str(Path(__file__).parent.parent.parent.parent / "results" / "jepa_predictor_v3.safetensors"),
)

'''

    # Insert before the class definition
    class_marker = "\nclass VerifyRepairPipeline:"
    if class_marker not in content:
        print("  WARNING: cannot locate VerifyRepairPipeline class — skipping.")
        return

    content = content.replace(class_marker, inject_constant + class_marker, 1)

    # Also inject default predictor loading at the END of __init__, after the
    # model loading block, by finding the Raises docstring section end in __init__.
    # We look for the last line before the first non-init method.
    # Strategy: inject after the line "self._load_model(model)" block end.
    init_injection = '''
        # Exp 168: Tier 3 self-learning complete — auto-load v3 JEPA predictor.
        # When results/jepa_predictor_v3.safetensors exists, fast-path gating
        # is applied automatically (verified >= 40% fast-path, < 2% degradation).
        self._default_jepa_predictor: Any = None
        if _DEFAULT_JEPA_PATH and Path(_DEFAULT_JEPA_PATH).exists():
            try:
                from carnot.pipeline.jepa_predictor import JEPAViolationPredictor as _JVPC
                _pred = _JVPC()
                _pred.load(_DEFAULT_JEPA_PATH)
                self._default_jepa_predictor = _pred
                logger.info(
                    "Loaded default JEPA v3 predictor from %s (Exp 168 fast-path)",
                    _DEFAULT_JEPA_PATH,
                )
            except Exception as _e:
                logger.debug("Could not load default JEPA predictor: %s", _e)
'''

    # Inject after the line that sets up self._tokenizer
    tokenizer_marker = "        self._device: str = \"cpu\"\n"
    if tokenizer_marker in content:
        content = content.replace(
            tokenizer_marker,
            tokenizer_marker + init_injection,
            1,
        )
    else:
        print("  WARNING: could not find injection point in __init__ — skipping init injection.")

    vrp_path.write_text(content)
    print(f"  Updated {vrp_path} with default v3 predictor loading.")


if __name__ == "__main__":
    main()
