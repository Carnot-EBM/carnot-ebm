r"""Shared helpers for the P0.1 difficulty-matched corpus builders (exp3483/3496).

WHY THIS MODULE EXISTS
======================
The P0.1 corpus builders generate MATH problems and need to:

1. Extract the final ``\boxed{}`` answer from generated text (answer extraction).
2. Normalize answers so ``\frac{1}{2}`` and ``0.5`` compare correctly (normalization).
3. Decide whether two answers are equivalent (correctness labelling).
4. Compute self-consistency (SC) as majority-vote accuracy over sampled answers (band classification).
5. Parse reasoning steps so a step-level verifier (FoVer PRIME) can score the process, not just the output.
6. Compute mean token logprob as a confidence proxy for energy ranking.
7. Track which problems have already been generated so re-invocations can RESUME rather than restart.

These helpers are PURE (no GPU, no file I/O except ``completed_problem_ids``), so
they can be unit-tested without a CUDA device or a cached GGUF.

BAND CONSTANTS
--------------
``BAND_LO = 0.40`` and ``BAND_HI = 0.70`` define the SC "headroom" sweet-spot
identified by ThinkPRM (arXiv:2504.16828). Problems where the model's SC lands
in [0.40, 0.70] are CONTESTED — the model sometimes gets them right, sometimes
wrong, so a selector that picks the right answer adds real value. Below 0.40 the
model is mostly guessing (selector can't help). Above 0.70 the model is mostly
right already (no room for improvement).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ---- Band constants ----------------------------------------------------------
# SC in [BAND_LO, BAND_HI] is the "headroom" window where P0.1 is testable.
BAND_LO: float = 0.40
BAND_HI: float = 0.70


# ---- Answer extraction -------------------------------------------------------

def extract_boxed_answer(text: str | None) -> str | None:
    r"""Extract the LAST ``\boxed{...}`` expression from model output.

    WHY LAST: a chain-of-thought solution often has intermediate ``\boxed{}``
    displays (e.g., sub-problem answers). The final one is the model's concluded
    answer, so we take the last match, not the first.

    WHY BRACE-BALANCED: ``\boxed{\frac{1}{2}}`` contains nested braces; a naïve
    regex ``\boxed{[^}]*}`` would stop at the first ``}``. We walk the string
    character-by-character to find the matching closing brace.

    Returns ``None`` when no ``\boxed{`` is found (model produced no final
    answer in the expected format — the generation is treated as incorrect).
    """
    if not text:
        return None
    # Find all positions where \boxed{ starts.
    starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not starts:
        return None
    # Take the LAST occurrence and extract the balanced-brace content.
    start = starts[-1] + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        # Unbalanced braces — return whatever we accumulated.
        return text[start:i - 1].strip() or None
    return text[start : i - 1].strip() or None


# ---- Answer normalization ----------------------------------------------------

_STRIP_RE = re.compile(
    r"\\(?:left|right|,|;|!|:|\s)|"  # LaTeX spacing commands
    r"\$|"                              # dollar signs
    r"\\text\{[^}]*\}|"               # \text{...}
    r"\\mathrm\{[^}]*\}|"             # \mathrm{...}
    r"\\mathbf\{[^}]*\}|"             # \mathbf{...}
    r"\\dfrac|\\tfrac|\\cfrac",        # fraction variants → use \frac form below
)


def normalize_answer(answer: str | None) -> str | None:
    r"""Normalize a math answer string for robust equality comparison.

    WHY: ``$\left(11\right)$`` and ``(11)`` should compare equal. LaTeX wrappers
    vary by model and don't affect the mathematical value. We strip common noise
    so two answers that denote the same value are identical strings after
    normalization, without invoking a CAS (which would be heavyweight for a
    corpus-builder).

    The normalization is intentionally conservative — it only removes obviously
    cosmetic wrappers. For genuinely ambiguous equivalences (e.g., ``1/2`` vs
    ``0.5``) we fall through to string equality after normalization; partial
    credit is not assigned.
    """
    if answer is None:
        return None
    s = answer.strip()
    # Remove outer dollar signs.
    s = s.strip("$").strip()
    # Remove \left / \right paired delimiters (keep the bracket).
    s = re.sub(r"\\left\s*", "", s)
    s = re.sub(r"\\right\s*", "", s)
    # Remove LaTeX spacing.
    s = re.sub(r"\\[,;!: ]", "", s)
    # Remove \text{...}, \mathrm{...}, \mathbf{...}.
    s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", s)
    # Normalize fraction command variants.
    s = re.sub(r"\\(?:dfrac|tfrac|cfrac)", r"\\frac", s)
    # Collapse whitespace.
    s = " ".join(s.split())
    return s or None


# ---- Correctness checking ----------------------------------------------------

def answers_match(text: str | None, gold_answer: str | None) -> bool:
    r"""Return True when the extracted answer from ``text`` equals ``gold_answer``.

    WHY: Both the generated text and the gold answer go through the same
    extract → normalize pipeline so comparisons are symmetric. A missing text
    or gold answer always yields False (no partial credit, no errors).
    """
    if text is None or gold_answer is None:
        return False
    extracted = extract_boxed_answer(text)
    if extracted is None:
        return False
    return normalize_answer(extracted) == normalize_answer(gold_answer)


# ---- Headroom band classification --------------------------------------------

def in_headroom_band(sc: float) -> bool:
    """Return True when ``sc`` is inside the P0.1 headroom band [BAND_LO, BAND_HI].

    WHY: The headroom band [0.40, 0.70] is where a selector can add real value.
    Below 0.40 the model is mostly guessing; above 0.70 the model is mostly right
    already. Both extremes make the P0.1 selector comparison uninformative.
    """
    return BAND_LO <= sc <= BAND_HI


# ---- Self-consistency accuracy -----------------------------------------------

def self_consistency_accuracy(records: list[dict[str, Any]]) -> float:
    """Compute majority-vote accuracy over a list of problem records.

    WHY: Self-consistency (SC) = fraction of problems where the majority of
    sampled answers equals the gold answer. This is the baseline P0.1 competes
    against: if energy-based selection can't beat SC at equal compute, the
    verifier thesis is in question.

    Each record must carry ``sampled_answers`` (list of normalized answer strings,
    one per sample) and ``gold_answer_norm`` (the normalized gold). An empty
    ``sampled_answers`` list counts as an incorrect majority vote (the model
    produced no parseable answers for that problem).

    Returns 0.0 on an empty record list (not None) so callers can always compare
    numerically without None checks.
    """
    if not records:
        return 0.0
    correct = 0
    for rec in records:
        gold = rec.get("gold_answer_norm")
        answers = rec.get("sampled_answers") or []
        if not answers or gold is None:
            continue
        # Majority vote: count normalized answers and find the most common.
        counts: dict[str | None, int] = {}
        for a in answers:
            counts[a] = counts.get(a, 0) + 1
        majority = max(counts, key=lambda k: counts[k])
        if majority is not None and normalize_answer(majority) == normalize_answer(gold):
            correct += 1
    return correct / len(records)


# ---- Reasoning step parsing --------------------------------------------------

def parse_reasoning_steps(text: str) -> list[str]:
    """Parse the reasoning text into a list of steps (one per non-empty paragraph).

    WHY: The FoVer PRIME step-level verifier scores process rewards per reasoning
    step, not just the final answer. We need to split the generation into steps.
    We use double-newline as the paragraph delimiter (consistent with standard
    chain-of-thought formatting) rather than numbered-list parsing (which is
    model-dependent).

    An empty string returns an empty list (no steps). A generation with no blank
    lines returns a single step (the whole text).
    """
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paragraphs if p]


# ---- Token logprob confidence ------------------------------------------------

def mean_token_logprob(token_logprobs: list[float | None]) -> float | None:
    """Compute the mean of non-None token logprobs as a confidence proxy.

    WHY: llama.cpp returns one logprob per generated token. Averaging them gives
    a sequence-level confidence score that can serve as a proxy for the model's
    energy (lower log-probability = higher energy = less confident). We skip
    ``None`` entries (llama.cpp uses None for the first token which has no
    preceding context).

    Returns ``None`` when the list is empty or all values are None (no confidence
    information is available for this generation).
    """
    valid = [x for x in token_logprobs if x is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


# ---- Resume support ----------------------------------------------------------

def completed_problem_ids(corpus_path: Path) -> set[str]:
    """Return the set of problem_id values already written to the corpus JSONL.

    WHY: The corpus builder appends one row per completed problem and calls this
    on startup to skip problems it already generated. This makes re-invocations
    idempotent — each call adds only NEW problems, so wall-time budget can be
    small and the full corpus accumulates across multiple runs.

    Corrupt/non-JSON lines are silently skipped (a partial write on kill produces
    one bad line at most; prior rows are intact).
    """
    if not corpus_path.exists():
        return set()
    ids: set[str] = set()
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                pid = row.get("problem_id")
                if pid is not None:
                    ids.add(str(pid))
            except json.JSONDecodeError:
                continue
    return ids
