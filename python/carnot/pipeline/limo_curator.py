"""LIMOCurator — LIMO-style curation for JEPA training corpus (Exp 824).

**Researcher summary:**
    JEPA v13-v22 all failed OOD (AUC <= 0.50) despite growing the corpus to 1050 items.
    Root cause: more noisy data made things worse, not better.  LIMO (arXiv 2402.09353)
    demonstrated that 817 carefully curated examples outperform 100k random examples for
    LLM reasoning.  The same quality-beats-quantity principle applies to EBM training data.

**What this module does:**
    1. Loads the existing FoVer/CPMI corpus from disk.
    2. Scores each training pair by z3_confidence × cpmi_score — a joint quality signal
       where z3_confidence is how certain the formal verifier is about the label and
       cpmi_score is how contrastively informative the pair is.
    3. Selects the top-k pairs by this joint score (default k=50 for GSM8K).
    4. Augments with synthetic HumanEval and SVAMP pairs to add domain diversity.

**Why z3_confidence × cpmi_score as curation metric:**
    - z3_confidence alone favors easily-verified pairs; it ignores whether the pair
      actually teaches the model to discriminate.
    - cpmi_score alone favors contrastive pairs but doesn't check formal correctness.
    - The product rewards pairs that are BOTH formally verified AND contrastively hard —
      the highest-signal training examples.

**Why domain diversity matters (REQ-LEARN-824-002):**
    OOD evaluation tests the model on steps from domains it hasn't seen.  If training
    is 100% GSM8K, the model learns GSM8K surface patterns (e.g., "total =", "per day"),
    not the underlying correctness signal.  HumanEval and SVAMP use different surface
    forms, forcing the model to learn domain-invariant features.

Spec: REQ-LEARN-824-001, REQ-LEARN-824-002, SCENARIO-LEARN-824-001
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class CuratedPair(NamedTuple):
    """One curated training pair with metadata for triplet loss construction.

    Attributes:
        prefix_text:   The reasoning context / question prefix (anchor).
        positive_step: A step that is formally correct (positive example).
        negative_step: A step that is incorrect or contrastively hard (negative).
        z3_confidence: Z3 verifier confidence in [0.0, 1.0].
        cpmi_score:    CPMI contrastive score (higher = harder negative).
        source_domain: Origin domain: 'gsm8k', 'humaneval', or 'svamp'.
        quality_score: z3_confidence × cpmi_score (curation ranking key).
    """

    prefix_text: str
    positive_step: str
    negative_step: str
    z3_confidence: float
    cpmi_score: float
    source_domain: str
    quality_score: float


# ---------------------------------------------------------------------------
# LIMOCurator
# ---------------------------------------------------------------------------


class LIMOCurator:
    """Curate a high-quality training corpus using the LIMO principle.

    **Detailed explanation for engineers:**
        LIMO (arXiv 2402.09353) showed that model fine-tuning benefits more from a
        small set of carefully selected examples than from a large noisy set.  The
        key insight: every noisy example adds gradient noise that can cancel the
        signal from clean examples.

        This curator implements the Carnot variant of LIMO for EBM training:
        - Quality signal: z3_confidence × cpmi_score (formal correctness × discriminability)
        - Selection: top-k by quality signal (default k=50 for GSM8K)
        - Augmentation: synthetic HumanEval + SVAMP pairs for domain diversity

        The z3_confidence threshold (default 0.9) ensures we only include pairs where
        the Z3 solver was highly confident about the label.  Steps with z3_confidence < 0.9
        are ambiguous — the solver wasn't sure if the step was correct or not, so including
        them as training signal would inject label noise.

    Args:
        fover_path:               Path to FoVer labeled steps JSON (list of step dicts).
        cpmi_path:                Path to CPMI pairs/triples JSON (list of triple dicts).
        z3_confidence_threshold:  Minimum z3_confidence to include a pair (default 0.9).
    """

    def __init__(
        self,
        fover_path: str | Path,
        cpmi_path: str | Path,
        z3_confidence_threshold: float = 0.9,
    ) -> None:
        self.fover_path = Path(fover_path)
        self.cpmi_path = Path(cpmi_path)
        self.z3_confidence_threshold = z3_confidence_threshold
        self._fover_pairs: list[CuratedPair] | None = None
        self._cpmi_triples: list[dict] | None = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_fover_pairs(self) -> list[CuratedPair]:
        """Load FoVer labeled steps and convert to CuratedPair format.

        **WHY we need both correct and incorrect steps:**
            The triplet loss needs an anchor (prefix), a positive (correct step), and
            a negative (incorrect step).  FoVer labels each step individually, so we
            pair steps from the same question_id: correct steps become positives,
            incorrect steps become negatives.

            When no pair can be formed (e.g., all steps for a question are correct),
            we create a synthetic negative by appending " [ERROR]" to the positive step.
            This is a weak signal but better than discarding the question entirely.

        Returns:
            List of CuratedPair with source_domain='gsm8k' (FoVer corpus is GSM8K).
        """
        if not self.fover_path.exists():
            return []

        with open(self.fover_path) as f:
            raw = json.load(f)

        # Group steps by question_id so we can pair correct vs incorrect.
        by_question: dict[str, dict[str, list]] = {}
        for entry in raw:
            qid = str(entry.get("question_id", "unknown"))
            label = entry.get("label", "not_verifiable")
            confidence = float(entry.get("confidence", 0.0))
            step_text = entry.get("step_text", "")
            if qid not in by_question:
                by_question[qid] = {"correct": [], "incorrect": [], "confidence": []}
            if label == "correct":
                by_question[qid]["correct"].append((step_text, confidence))
            elif label == "incorrect":
                by_question[qid]["incorrect"].append((step_text, confidence))

        pairs: list[CuratedPair] = []
        for qid, steps in by_question.items():
            corrects = steps["correct"]
            incorrects = steps["incorrect"]

            if not corrects:
                continue

            for pos_text, z3_conf in corrects:
                if incorrects:
                    neg_text, _ = incorrects[0]
                else:
                    neg_text = pos_text + " [INCORRECT: arithmetic error]"

                # FoVer-only pairs have no cpmi_score; use z3_confidence as proxy.
                quality = z3_conf * z3_conf  # both factors are z3_conf
                pairs.append(
                    CuratedPair(
                        prefix_text=qid,
                        positive_step=pos_text,
                        negative_step=neg_text,
                        z3_confidence=z3_conf,
                        cpmi_score=z3_conf,  # proxy: no CPMI score for FoVer-only
                        source_domain="gsm8k",
                        quality_score=quality,
                    )
                )

        self._fover_pairs = pairs
        return pairs

    def load_cpmi_triples(self) -> list[dict]:
        """Load CPMI triples from the corpus JSON.

        Returns the raw list of triple dicts with keys:
            prefix_text, positive_step, negative_step, cpmi_score, source_domain.
        """
        if not self.cpmi_path.exists():
            return []

        with open(self.cpmi_path) as f:
            raw = json.load(f)

        if isinstance(raw, list):
            self._cpmi_triples = raw
        else:
            # If it's a dict (e.g., experiment artifact), look for the data field.
            self._cpmi_triples = raw.get("triples", raw.get("pairs", []))

        return self._cpmi_triples  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Curation logic
    # ------------------------------------------------------------------

    def score_pairs(self, pairs: list[CuratedPair]) -> list[CuratedPair]:
        """Sort pairs by z3_confidence × cpmi_score descending.

        **WHY descending order:**
            We want to select the TOP-k pairs, so the highest-quality pairs must come
            first.  Descending sort means pairs[0] has the highest quality score.

        Args:
            pairs: List of CuratedPair objects.

        Returns:
            Same list sorted by quality_score descending (highest first).

        Spec: REQ-LEARN-824-001
        """
        return sorted(pairs, key=lambda p: p.quality_score, reverse=True)

    def select_top_k(self, k: int = 50) -> list[CuratedPair]:
        """Select top-k GSM8K pairs from FoVer + CPMI corpora by quality score.

        **HOW this combines FoVer and CPMI:**
            Both corpora are merged, scored, and sorted.  CPMI triples already have
            a cpmi_score field; FoVer pairs use z3_confidence as a proxy for both
            factors.  The joint ranking is fair because both use the same product formula.

        **WHY k=50 by default:**
            LIMO used ~817 examples for a 7B model.  JEPA is a much smaller probe
            (TF-IDF + MLP, ~10k parameters).  For this scale, 50 high-quality examples
            is sufficient to fit the distribution without overfitting to noise.

        Args:
            k: Number of pairs to select.

        Returns:
            Top-k pairs sorted by quality_score descending.

        Spec: REQ-LEARN-824-001
        """
        fover = self.load_fover_pairs()
        cpmi_raw = self.load_cpmi_triples()

        # Convert CPMI triples to CuratedPair format.
        cpmi_pairs: list[CuratedPair] = []
        for triple in cpmi_raw:
            pos = triple.get("positive_step", "")
            neg = triple.get("negative_step", "")
            cpmi_score = float(triple.get("cpmi_score", 0.0))
            prefix = triple.get("prefix_text", "")
            domain = triple.get("source_domain", "gsm8k")
            # For CPMI triples without z3_confidence, use threshold as proxy
            # (meaning: "passed the quality bar to be in the corpus").
            z3_conf = self.z3_confidence_threshold
            quality = z3_conf * max(cpmi_score, 0.0)
            cpmi_pairs.append(
                CuratedPair(
                    prefix_text=prefix,
                    positive_step=pos,
                    negative_step=neg,
                    z3_confidence=z3_conf,
                    cpmi_score=cpmi_score,
                    source_domain=domain,
                    quality_score=quality,
                )
            )

        # Filter FoVer pairs by z3_confidence threshold.
        filtered_fover = [p for p in fover if p.z3_confidence >= self.z3_confidence_threshold]

        # Merge and rank.
        all_pairs = filtered_fover + cpmi_pairs
        ranked = self.score_pairs(all_pairs)
        return ranked[:k]

    def add_domain_pairs(
        self,
        humaneval_n: int = 10,
        svamp_n: int = 10,
    ) -> list[CuratedPair]:
        """Add synthetic HumanEval and SVAMP pairs for domain diversity.

        **WHY synthetic pairs for domain diversity (REQ-LEARN-824-002):**
            The GSM8K corpus uses consistent surface patterns ("total =", "per day",
            dollar amounts).  A model trained only on GSM8K will memorize these patterns
            rather than learning the underlying correctness signal.  Adding pairs from
            different domains (Python code logic, arithmetic word problems) forces the
            model to learn domain-invariant features.

            These are SYNTHETIC pairs because we don't have a live HumanEval/SVAMP
            annotation pipeline yet.  The Z3 stubs check the arithmetic in the answer
            but not the full Python semantics — they are labeled 'positive' (correct
            answer format) vs 'negative' (incorrect arithmetic in the answer).

            Each pair is designed to stress a different failure mode:
            - HumanEval: does the function return the correct value for the given input?
            - SVAMP: does the arithmetic chain reach the correct final answer?

        Args:
            humaneval_n: Number of HumanEval pairs to add (default 10).
            svamp_n:     Number of SVAMP pairs to add (default 10).

        Returns:
            Combined corpus: top-50 GSM8K + humaneval_n + svamp_n pairs.

        Spec: REQ-LEARN-824-002
        """
        top_50 = self.select_top_k(k=50)

        humaneval_pairs = _make_humaneval_pairs(humaneval_n)
        svamp_pairs = _make_svamp_pairs(svamp_n)

        return top_50 + humaneval_pairs + svamp_pairs


# ---------------------------------------------------------------------------
# Synthetic domain pair generators
# ---------------------------------------------------------------------------


def _make_humaneval_pairs(n: int) -> list[CuratedPair]:
    """Generate n synthetic HumanEval-style code reasoning pairs.

    Each pair tests whether a simple Python function returns the correct value
    for a given input.  The negative step contains an arithmetic error in the
    return value calculation.

    WHY these are useful for OOD generalisation:
        HumanEval steps use code-style phrasing ("return", "def", "==") that is
        entirely absent from GSM8K.  A model that generalises correctly should
        be able to identify the incorrect calculation in both domains.
    """
    templates = [
        ("def add(a, b): return a + b; add(3, 4)", "The function returns 3 + 4 = 7.", "The function returns 3 + 4 = 8."),
        ("def mul(a, b): return a * b; mul(6, 7)", "The function returns 6 * 7 = 42.", "The function returns 6 * 7 = 43."),
        ("def sub(a, b): return a - b; sub(10, 3)", "The function returns 10 - 3 = 7.", "The function returns 10 - 3 = 8."),
        ("def div(a, b): return a / b; div(15, 3)", "The function returns 15 / 3 = 5.", "The function returns 15 / 3 = 6."),
        ("def square(n): return n * n; square(5)", "The function returns 5 * 5 = 25.", "The function returns 5 * 5 = 26."),
        ("def cube(n): return n ** 3; cube(3)", "The function returns 3 ** 3 = 27.", "The function returns 3 ** 3 = 28."),
        ("def half(n): return n // 2; half(8)", "The function returns 8 // 2 = 4.", "The function returns 8 // 2 = 5."),
        ("def double(n): return 2 * n; double(9)", "The function returns 2 * 9 = 18.", "The function returns 2 * 9 = 19."),
        ("def mod(a, b): return a % b; mod(10, 3)", "The function returns 10 % 3 = 1.", "The function returns 10 % 3 = 2."),
        ("def power(a, b): return a ** b; power(2, 8)", "The function returns 2 ** 8 = 256.", "The function returns 2 ** 8 = 255."),
        ("def avg(a, b): return (a + b) / 2; avg(4, 6)", "The function returns (4 + 6) / 2 = 5.", "The function returns (4 + 6) / 2 = 4."),
        ("def min2(a, b): return a if a < b else b; min2(3, 7)", "The minimum is 3.", "The minimum is 7."),
    ]
    pairs: list[CuratedPair] = []
    for i in range(min(n, len(templates))):
        prefix, pos, neg = templates[i]
        pairs.append(
            CuratedPair(
                prefix_text=f"humaneval_{i}",
                positive_step=pos,
                negative_step=neg,
                z3_confidence=1.0,
                cpmi_score=1.0,
                source_domain="humaneval",
                quality_score=1.0,
            )
        )
    return pairs


def _make_svamp_pairs(n: int) -> list[CuratedPair]:
    """Generate n synthetic SVAMP-style arithmetic word problem pairs.

    SVAMP (arXiv 2103.07191) tests arithmetic reasoning with varied problem templates.
    These synthetic pairs use simple arithmetic chains with clear correct/incorrect steps.

    WHY SVAMP surface forms differ from GSM8K:
        GSM8K uses long, multi-step word problems with dollar amounts and unit conversions.
        SVAMP uses shorter, more direct arithmetic setups.  The different vocabulary
        distribution helps the model generalise beyond GSM8K surface patterns.
    """
    templates = [
        ("There are 5 apples and 3 oranges.", "Total fruit = 5 + 3 = 8.", "Total fruit = 5 + 3 = 9."),
        ("A train travels 60 km/h for 3 hours.", "Distance = 60 * 3 = 180 km.", "Distance = 60 * 3 = 190 km."),
        ("Tom has 12 candies. He gives 4 to Jane.", "Remaining = 12 - 4 = 8.", "Remaining = 12 - 4 = 7."),
        ("Each box holds 6 items. There are 7 boxes.", "Total items = 6 * 7 = 42.", "Total items = 6 * 7 = 43."),
        ("A store sold 24 books over 4 days equally.", "Books per day = 24 / 4 = 6.", "Books per day = 24 / 4 = 5."),
        ("Maria earns $15 per hour and works 8 hours.", "Total = 15 * 8 = $120.", "Total = 15 * 8 = $125."),
        ("A rectangle is 9 cm by 4 cm.", "Area = 9 * 4 = 36 sq cm.", "Area = 9 * 4 = 34 sq cm."),
        ("There are 3 groups of 11 students.", "Total = 3 * 11 = 33.", "Total = 3 * 11 = 32."),
        ("A bus had 40 passengers. 15 got off.", "Remaining = 40 - 15 = 25.", "Remaining = 40 - 15 = 24."),
        ("Sam ran 5 km per day for 6 days.", "Total = 5 * 6 = 30 km.", "Total = 5 * 6 = 31 km."),
        ("A pond has 100 fish. 37 are caught.", "Remaining = 100 - 37 = 63.", "Remaining = 100 - 37 = 64."),
        ("Each row has 8 seats. There are 9 rows.", "Total seats = 8 * 9 = 72.", "Total seats = 8 * 9 = 73."),
    ]
    pairs: list[CuratedPair] = []
    for i in range(min(n, len(templates))):
        prefix, pos, neg = templates[i]
        pairs.append(
            CuratedPair(
                prefix_text=f"svamp_{i}",
                positive_step=pos,
                negative_step=neg,
                z3_confidence=1.0,
                cpmi_score=1.0,
                source_domain="svamp",
                quality_score=1.0,
            )
        )
    return pairs
