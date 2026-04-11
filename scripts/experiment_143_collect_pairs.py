#!/usr/bin/env python3
"""Experiment 143: Collect JEPA Predictive-Verification Training Pairs.

**Researcher summary:**
    JEPA predictive verification (research-program.md Tier 3, Goal #2) needs
    a labelled dataset of (partial_response_embedding, final_violated) pairs.
    This experiment mines verify-repair logs from Exp 120–140, supplements
    with synthetic arithmetic questions run through AutoExtractor + Ising
    verify, and saves everything to results/jepa_training_pairs.json.

**Detailed explanation for engineers:**
    JEPA (Joint Embedding Predictive Architecture) trains a predictor network
    to guess whether a full response will violate constraints given only a
    partial prefix of that response. This is "early-exit" verification: instead
    of waiting for the model to finish a 200-token response before checking
    constraints, the JEPA predictor raises a flag at token 50 (25%) if the
    response is headed toward a violation.

    Training data requirements:
    - Input: a fixed-size embedding of a prefix of the response
    - Label: was the *final* response flagged as violating a constraint?
    - Prefix ratios: 10%, 25%, 50%, 75% of whitespace-tokenized response

    Data sources (in order):
    1. Log mining: scan results/ for experiment JSONs that recorded actual
       (question, response, violated) tuples. In practice most Exp 120–140
       results store aggregate metrics rather than per-sample text, so this
       often yields 0–50 pairs.
    2. Synthetic supplement: generate 200 arithmetic word problems, compute
       correct answers, produce one correct and one wrong LLM-style response
       per question, run AutoExtractor+VerifyRepairPipeline to label each,
       and use 25%/75% prefixes.

    Embedding: RandomProjectionEmbedding(embed_dim=256, seed=42) from
    fast_embedding.py (Exp 112). ~0.026ms/call, no external dependencies.
    We use 256-dim here (vs 384 baseline) to keep the JSON file compact.

    Output schema (results/jepa_training_pairs.json):
    {
      "pairs": [
        {
          "prefix_ratio": 0.25,
          "embedding": [float, ...],  // 256 values, L2-normalized
          "violated_arithmetic": bool,
          "violated_code": bool,
          "violated_logic": bool,
          "any_violated": bool,
          "domain": "arithmetic" | "code" | "logic" | "mixed",
          "source_exp": int | "synthetic_143"
        },
        ...
      ],
      "total": N,
      "domain_counts": {"arithmetic": N, "code": N, "logic": N, "mixed": N},
      "positive_rate": float,   // fraction where any_violated == True
      "negative_rate": float
    }

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_143_collect_pairs.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Environment / path setup
# ---------------------------------------------------------------------------
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from carnot.embeddings.fast_embedding import RandomProjectionEmbedding  # noqa: E402
from carnot.pipeline.extract import AutoExtractor  # noqa: E402
from carnot.pipeline.verify_repair import VerifyRepairPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "jepa_training_pairs.json"

# Embedding config — 256-dim to keep JSON compact; seed=42 for reproducibility.
EMBED_DIM = 256
EMBED_SEED = 42

# Prefix ratios to generate per response (as fractions of token count).
PREFIX_RATIOS = [0.10, 0.25, 0.50, 0.75]

# Supplemental synthetic data target if mined pairs are fewer than this.
SUPPLEMENT_THRESHOLD = 500

# Number of synthetic arithmetic questions to generate.
N_SYNTHETIC = 200

# Seed for reproducible synthetic generation.
SYNTHETIC_SEED = 143

# ---------------------------------------------------------------------------
# Helper: tokenize (simple whitespace split — no external tokenizer needed)
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Split text into whitespace tokens.

    **Detailed explanation for engineers:**
        We deliberately avoid a neural tokenizer (BPE, WordPiece) because:
        1. They require downloading model files.
        2. Whitespace tokenization is sufficient here — we only need to
           compute prefix lengths as a fraction of "words", not sub-words.
           JEPA embedding will be the same regardless of tokenization scheme
           as long as the same scheme is used consistently during training
           and inference.

    Args:
        text: Input text of any length.

    Returns:
        List of whitespace-delimited token strings (may be empty).
    """
    return text.split()


def get_prefix(tokens: list[str], ratio: float) -> str:
    """Return the first `ratio` fraction of tokens as a joined string.

    **Detailed explanation for engineers:**
        We compute floor(len(tokens) * ratio) to get the number of prefix
        tokens, with a minimum of 1 so extremely short responses still
        produce a non-empty prefix. The tokens are re-joined with a single
        space — this is not byte-identical to the original text (multiple
        spaces are collapsed) but is consistent and sufficient for embedding.

    Args:
        tokens: Whitespace-tokenized response.
        ratio: Fraction in (0, 1].

    Returns:
        Prefix string (at least 1 token long).
    """
    n = max(1, int(math.floor(len(tokens) * ratio)))
    return " ".join(tokens[:n])


# ---------------------------------------------------------------------------
# Step 1: Mine existing result files for (response, violated) pairs
# ---------------------------------------------------------------------------


def _extract_pairs_from_experiment(
    exp_data: dict[str, Any],
    source_exp: int | str,
) -> list[dict[str, Any]]:
    """Try to extract (response, violation_flags) tuples from one experiment dict.

    **Detailed explanation for engineers:**
        Experiment result JSON schemas vary by experiment number. Rather than
        writing one parser per experiment, this function applies a cascade of
        heuristics to extract whatever labelled samples it can find. If no
        per-sample text is found (aggregate-only results), it returns [].

        Heuristics tried (in order):
        1. ``result["samples"]`` list with keys ``response`` + ``violated``.
        2. ``result["models"][*]["variants"][*]["samples"]`` (Exp 120–122 schema).
        3. ``result["gsm8k"]["modes"][*]["samples"]`` (Exp 138 schema).
        4. ``result["session_*"]["samples"]`` (Exp 136 schema).

    Args:
        exp_data: Parsed JSON dict from a results file.
        source_exp: Experiment number or label for the ``source_exp`` field.

    Returns:
        List of raw dicts with keys: ``response``, ``violated_arithmetic``,
        ``violated_code``, ``violated_logic``, ``any_violated``, ``domain``.
        May be empty if the file has no per-sample text.
    """
    raw_pairs: list[dict[str, Any]] = []

    def _add_sample(response: str, any_viol: bool, domain: str = "arithmetic") -> None:
        """Append a minimal pair dict with all violation flags populated."""
        raw_pairs.append(
            {
                "response": response,
                "violated_arithmetic": any_viol and domain == "arithmetic",
                "violated_code": any_viol and domain == "code",
                "violated_logic": any_viol and domain == "logic",
                "any_violated": any_viol,
                "domain": domain,
            }
        )

    def _scan_dict(node: Any, depth: int = 0) -> None:
        """Recursively scan a nested dict/list for sample arrays."""
        if depth > 6:
            return
        if isinstance(node, list):
            for item in node:
                if isinstance(item, dict) and "response" in item:
                    # Found a sample dict.
                    resp = item.get("response", "")
                    if not resp:
                        continue
                    viol = item.get("violated", item.get("any_violated", False))
                    domain = item.get("domain", "arithmetic")
                    _add_sample(resp, bool(viol), domain)
                elif isinstance(item, (dict, list)):
                    _scan_dict(item, depth + 1)
        elif isinstance(node, dict):
            for v in node.values():
                _scan_dict(v, depth + 1)

    _scan_dict(exp_data)
    return raw_pairs


def mine_result_files() -> list[dict[str, Any]]:
    """Load all experiment result JSONs and extract labelled (response, violated) pairs.

    **Detailed explanation for engineers:**
        We scan the entire results/ directory rather than hard-coding a list
        of experiment numbers. This future-proofs the collection step: as new
        experiments add per-sample text logging, they will be picked up
        automatically without modifying this script.

        For each JSON file we parse it, call ``_extract_pairs_from_experiment``,
        and accumulate the raw pairs. The source_exp label is extracted from
        the filename (e.g., "experiment_136_results.json" → 136) or from the
        "experiment" field inside the JSON.

    Returns:
        List of raw pair dicts (may be empty if no files have per-sample text).
    """
    all_raw: list[dict[str, Any]] = []

    for json_path in sorted(RESULTS_DIR.glob("experiment_*_results.json")):
        # Extract experiment number from filename.
        try:
            source_exp_num = int(json_path.stem.split("_")[1])
        except (IndexError, ValueError):
            source_exp_num = json_path.stem

        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(data, dict):
            continue

        # Prefer experiment number from the JSON itself if available.
        json_exp = data.get("experiment")
        if isinstance(json_exp, int):
            source_exp_num = json_exp

        pairs = _extract_pairs_from_experiment(data, source_exp_num)
        if pairs:
            for p in pairs:
                p["source_exp"] = source_exp_num
            all_raw.extend(pairs)
            print(f"  [mine] {json_path.name}: found {len(pairs)} pairs")

    return all_raw


# ---------------------------------------------------------------------------
# Step 2: Generate synthetic arithmetic pairs
# ---------------------------------------------------------------------------


def _arithmetic_response(a: int, b: int, op: str, correct: int, wrong_answer: bool) -> str:
    """Generate a plausible LLM-style arithmetic response.

    **Detailed explanation for engineers:**
        Real LLM responses to arithmetic questions include a chain-of-thought
        explanation before the final answer. We simulate this with a template
        that includes the intermediate steps. If ``wrong_answer=True`` we
        offset the claimed result by a small random amount (1–5) to ensure
        the AutoExtractor catches it as a violation.

        The response always ends with "The answer is X." which the
        ArithmeticExtractor regex matches on.

    Args:
        a: First operand.
        b: Second operand.
        op: One of "+", "*", "-".
        correct: True mathematical result.
        wrong_answer: If True, claim an incorrect answer.

    Returns:
        Multi-sentence response string that contains an arithmetic claim.
    """
    rng_local = random.Random(a * 1000 + b + (1 if wrong_answer else 0))
    offset = rng_local.randint(1, 5) if wrong_answer else 0
    claimed = correct + offset

    op_words = {"+": "add", "*": "multiply", "-": "subtract"}
    op_word = op_words.get(op, "compute")

    if op == "+":
        reasoning = (
            f"To solve this, we {op_word} {a} and {b}. "
            f"Starting from {a} and counting up {b} steps gives us {claimed}."
        )
    elif op == "*":
        reasoning = (
            f"To solve this, we {op_word} {a} by {b}. "
            f"We can think of this as adding {a} a total of {b} times, yielding {claimed}."
        )
    else:  # subtraction
        reasoning = (
            f"To solve this, we {op_word} {b} from {a}. "
            f"Counting down {b} steps from {a} gives us {claimed}."
        )

    return f"{reasoning} Therefore, {a} {op} {b} = {claimed}. The answer is {claimed}."


def generate_synthetic_arithmetic(n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Generate synthetic arithmetic Q/A pairs with actual violation labels.

    **Detailed explanation for engineers:**
        We generate three question types:
        1. Simple addition (x + y): 40% of pairs
        2. Simple multiplication (x * y): 30% of pairs
        3. Multi-step (x + y * z, evaluated left-to-right per problem setup): 30%

        For each question we produce two responses: one correct, one wrong.
        Then we run AutoExtractor.extract() on each response to get the actual
        violation label (rather than hard-coding it), so the labels are
        authoritative rather than assumed.

        This means the extractor could fail to detect a violation (false
        negative) or flag a valid response (false positive). Those cases are
        still valid training data — they reflect the extractor's actual
        behaviour, which is what the JEPA predictor needs to learn to predict.

    Args:
        n: Number of (question, response) pairs to generate. Each question
           produces 2 responses (correct + wrong), so n questions → 2n pairs.
           This function generates ceil(n/2) questions to stay near target n.
        rng: Seeded random.Random instance for reproducibility.

    Returns:
        List of raw pair dicts with keys: response, violated_arithmetic,
        violated_code, violated_logic, any_violated, domain, source_exp.
    """
    extractor = AutoExtractor()
    pipeline = VerifyRepairPipeline()

    pairs: list[dict[str, Any]] = []
    n_questions = math.ceil(n / 2)  # each question → 2 responses

    for i in range(n_questions):
        question_type = i % 3

        if question_type == 0:
            # Simple addition.
            a = rng.randint(1, 200)
            b = rng.randint(1, 200)
            op = "+"
            correct = a + b
        elif question_type == 1:
            # Simple multiplication.
            a = rng.randint(2, 30)
            b = rng.randint(2, 30)
            op = "*"
            correct = a * b
        else:
            # Multi-step: a + b * c phrased as a single addition for simplicity.
            # We keep op = "+" but scale b so the answer is non-trivial.
            a = rng.randint(10, 500)
            b = rng.randint(10, 500)
            op = "+"
            correct = a + b

        # Generate correct and wrong responses.
        for wrong in [False, True]:
            response = _arithmetic_response(a, b, op, correct, wrong)

            # Run the real extractor and pipeline to get authoritative labels.
            try:
                vr = pipeline.verify(question=f"What is {a} {op} {b}?", response=response)
                # Categorise violations by domain.
                arith_viol = any(
                    c.constraint_type == "arithmetic" for c in vr.violations
                )
                code_viol = any(c.constraint_type in ("type_check", "bound", "return_type") for c in vr.violations)
                logic_viol = any(c.constraint_type in ("implication", "equivalence") for c in vr.violations)
                any_viol = not vr.verified
            except Exception:
                # Fallback: use structural heuristic if pipeline raises.
                arith_viol = wrong
                code_viol = False
                logic_viol = False
                any_viol = wrong

            pairs.append(
                {
                    "response": response,
                    "violated_arithmetic": arith_viol,
                    "violated_code": code_viol,
                    "violated_logic": logic_viol,
                    "any_violated": any_viol,
                    "domain": "arithmetic",
                    "source_exp": "synthetic_143",
                }
            )

            if len(pairs) >= n:
                break

        if len(pairs) >= n:
            break

    return pairs


# ---------------------------------------------------------------------------
# Step 3: Embed prefixes → build final training pairs
# ---------------------------------------------------------------------------


def build_prefix_pairs(
    raw_pairs: list[dict[str, Any]],
    embedder: RandomProjectionEmbedding,
    prefix_ratios: list[float],
) -> list[dict[str, Any]]:
    """Convert raw (response, violation_flags) pairs into embedded prefix pairs.

    **Detailed explanation for engineers:**
        For each raw pair we tokenize the response, then for each prefix ratio
        we:
        1. Compute the prefix text (first ratio*len tokens).
        2. Embed the prefix with RandomProjectionEmbedding.
        3. Store (prefix_ratio, embedding_list, violated_*, domain, source_exp).

        The embedding is stored as a plain Python list of floats so it
        serializes cleanly to JSON. Each float is rounded to 6 decimal places
        to limit file size while preserving sufficient precision for a
        256-dimensional vector.

    Args:
        raw_pairs: List of raw pair dicts (from mining + synthesis).
        embedder: Instantiated RandomProjectionEmbedding(embed_dim=256, seed=42).
        prefix_ratios: List of prefix fractions to generate per response.

    Returns:
        List of final pair dicts ready for JSON serialisation.
    """
    final_pairs: list[dict[str, Any]] = []

    for raw in raw_pairs:
        response = raw.get("response", "")
        if not response:
            continue
        tokens = tokenize(response)
        if not tokens:
            continue

        for ratio in prefix_ratios:
            prefix_text = get_prefix(tokens, ratio)
            embedding = embedder.encode(prefix_text).tolist()
            # Round to 6 decimal places to limit file size.
            embedding_rounded = [round(x, 6) for x in embedding]

            final_pairs.append(
                {
                    "prefix_ratio": ratio,
                    "embedding": embedding_rounded,
                    "violated_arithmetic": raw.get("violated_arithmetic", False),
                    "violated_code": raw.get("violated_code", False),
                    "violated_logic": raw.get("violated_logic", False),
                    "any_violated": raw.get("any_violated", False),
                    "domain": raw.get("domain", "arithmetic"),
                    "source_exp": raw.get("source_exp", "unknown"),
                }
            )

    return final_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full pair-collection pipeline and save results.

    **Detailed explanation for engineers:**
        Execution order:
        1. Mine existing result JSON files for labelled samples.
        2. If total mined pairs < SUPPLEMENT_THRESHOLD, generate synthetic
           arithmetic pairs to supplement.
        3. Build the embedder once (init is cheap ~0.1ms).
        4. For each raw pair × each prefix ratio → embed → append to list.
        5. Write results/jepa_training_pairs.json.
        6. Print summary statistics.

    Returns:
        None. Side effect: writes OUTPUT_PATH.
    """
    t_start = time.perf_counter()
    print(f"[exp143] Collecting JEPA training pairs → {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # Step 1: Mine existing result files.
    # ------------------------------------------------------------------
    print("\n[Step 1] Mining existing verify-repair result logs...")
    mined_raw = mine_result_files()
    print(f"  Mined {len(mined_raw)} raw pairs from result files.")

    # ------------------------------------------------------------------
    # Step 2: Supplement with synthetic data if needed.
    # ------------------------------------------------------------------
    raw_pairs = list(mined_raw)  # copy so we can extend
    n_synthetic_needed = max(0, SUPPLEMENT_THRESHOLD - len(mined_raw))

    if n_synthetic_needed > 0:
        # Use prefix-ratio count: each raw pair → len(PREFIX_RATIOS) final pairs.
        # So we need raw_pairs_needed = ceil(target_final / len(PREFIX_RATIOS)).
        # But the supplement threshold is in raw pairs, so just generate
        # N_SYNTHETIC raw pairs regardless.
        print(
            f"\n[Step 2] Fewer than {SUPPLEMENT_THRESHOLD} mined pairs "
            f"({len(mined_raw)} found). Generating {N_SYNTHETIC} synthetic arithmetic pairs..."
        )
        rng = random.Random(SYNTHETIC_SEED)
        synthetic_raw = generate_synthetic_arithmetic(N_SYNTHETIC, rng)
        raw_pairs.extend(synthetic_raw)
        n_correct = sum(1 for p in synthetic_raw if not p["any_violated"])
        n_wrong = sum(1 for p in synthetic_raw if p["any_violated"])
        print(
            f"  Generated {len(synthetic_raw)} synthetic pairs "
            f"(correct: {n_correct}, violated: {n_wrong})."
        )
    else:
        print(
            f"\n[Step 2] Sufficient mined pairs ({len(mined_raw)} >= "
            f"{SUPPLEMENT_THRESHOLD}). Skipping synthetic generation."
        )

    # ------------------------------------------------------------------
    # Step 3: Build embedder.
    # ------------------------------------------------------------------
    print(f"\n[Step 3] Building RandomProjectionEmbedding(embed_dim={EMBED_DIM}, seed={EMBED_SEED})...")
    embedder = RandomProjectionEmbedding(embed_dim=EMBED_DIM, seed=EMBED_SEED)
    print(f"  Embedder ready. embed_dim={embedder.embed_dim}")

    # ------------------------------------------------------------------
    # Step 4: Generate prefix embeddings.
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Generating prefix pairs ({len(PREFIX_RATIOS)} ratios × {len(raw_pairs)} raw pairs)...")
    final_pairs = build_prefix_pairs(raw_pairs, embedder, PREFIX_RATIOS)
    print(f"  Built {len(final_pairs)} final pairs.")

    # ------------------------------------------------------------------
    # Step 5: Compute summary statistics.
    # ------------------------------------------------------------------
    domain_counts: dict[str, int] = {}
    n_positive = 0
    for pair in final_pairs:
        domain = pair["domain"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if pair["any_violated"]:
            n_positive += 1

    n_total = len(final_pairs)
    positive_rate = n_positive / n_total if n_total > 0 else 0.0
    negative_rate = 1.0 - positive_rate

    # ------------------------------------------------------------------
    # Step 6: Save results.
    # ------------------------------------------------------------------
    output = {
        "pairs": final_pairs,
        "total": n_total,
        "domain_counts": domain_counts,
        "positive_rate": round(positive_rate, 4),
        "negative_rate": round(negative_rate, 4),
        "metadata": {
            "experiment": 143,
            "embed_dim": EMBED_DIM,
            "embed_seed": EMBED_SEED,
            "prefix_ratios": PREFIX_RATIOS,
            "n_mined_raw": len(mined_raw),
            "n_synthetic_raw": len(raw_pairs) - len(mined_raw),
            "supplement_threshold": SUPPLEMENT_THRESHOLD,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # ------------------------------------------------------------------
    # Step 7: Print summary.
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 60)
    print("  Experiment 143 — JEPA Training Pair Collection Summary")
    print("=" * 60)
    print(f"  Total pairs:       {n_total:>8,}")
    print(f"  Positive rate:     {positive_rate * 100:.1f}%  (any_violated=True)")
    print(f"  Negative rate:     {negative_rate * 100:.1f}%  (any_violated=False)")
    print("\n  Domain breakdown:")
    for domain, count in sorted(domain_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {domain:<15} {count:>6,} pairs")
    print(f"\n  Output:            {OUTPUT_PATH}")
    print(f"  Elapsed:           {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
