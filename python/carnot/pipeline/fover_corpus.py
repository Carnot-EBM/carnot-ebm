"""fover_corpus — FOVER Corpus v2: merge, deduplicate, audit diversity, and balance.

**Researcher summary (RETRO-056, RETRO-058):**
    Exp 543 (JEPA retrain) produced AUC=0.444 — below random — because the training
    corpus had only 24 pairs and 88% were carry-violation type.  Shannon entropy of the
    constraint_type distribution was effectively 0 bits (all one class).

    A model trained on a single-class corpus cannot generalise: it learns to fire on
    carry violations and misses every other constraint type.  The minimum for meaningful
    generalisation is at least 2 constraint types (1.0 bits) and ideally >=1.5 bits.

    This module fixes the corpus problem by:
    1. Merging ALL available real pairs from Exps 442, 538, 551, 552.
    2. Deduplicating by (question, model_id) so no single question-model pair appears twice.
    3. Computing Shannon entropy of the constraint_type distribution.
    4. Downsampling overrepresented types when entropy < target, until entropy >= 1.5 bits.

**Constraint types:**
    The FOVER annotation pipeline labels each CoT step with a Z3 verdict: 'correct',
    'incorrect', or 'not_verifiable'.  The dominant "constraint type" for an entry is the
    most frequent label across its steps.  This is a proxy for the kind of error the
    model is making (or not making) — and it controls the diversity of the corpus.

    In practice, the Exp 543 corpus was saturated with carry violations ('incorrect' steps
    on carry arithmetic).  Downsampling those entries gives the corpus room to represent
    'correct' and 'not_verifiable' classes, raising entropy.

**Shannon entropy:**
    H = -sum(p_i * log2(p_i)) over all unique constraint types.
    H = 0 bits when all entries have the same type (useless corpus).
    H = 1.0 bits when two types are equally represented.
    H = 1.585 bits when three types are equally represented.
    H >= 1.5 bits is our gate for JEPA retraining.

Spec: REQ-DATA-003, REQ-DATA-004,
      SCENARIO-DATA-007, SCENARIO-DATA-008, SCENARIO-DATA-009
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# FOVERCorpusEntry dataclass
# ---------------------------------------------------------------------------


@dataclass
class FOVERCorpusEntry:
    """One training entry in the FOVER corpus.

    **Detailed explanation for engineers:**
        Each entry represents a single (question, model_id) inference attempt with its
        FOVER-annotated chain-of-thought steps.

        ``constraint_types`` is a list of the dominant Z3 label per step.  For most entries
        this will be ['not_verifiable'] for prose steps, ['correct'] for valid arithmetic,
        or ['incorrect'] for carry/sign errors.  The distribution of these labels across
        the entire corpus is what we measure for diversity.

        ``cot_steps`` stores the raw step objects (as dicts from JSON) so that downstream
        trainers can use step-level granularity if needed.

    Attributes:
        question:         The original question text.
        response:         The full model response (CoT text).
        model_id:         Model identifier string (e.g. 'Qwen3.5-0.8B').
        is_correct:       Whether the final answer was correct per ground-truth.
        constraint_types: List of Z3 labels for each step in this entry's CoT.
        cot_steps:        Raw step dicts from FOVER annotation.

    Spec: REQ-DATA-003
    """

    question: str
    response: str
    model_id: str
    is_correct: bool
    constraint_types: list[str]
    cot_steps: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# merge_fover_sources
# ---------------------------------------------------------------------------


def merge_fover_sources(sources: list[str]) -> list[FOVERCorpusEntry]:
    """Load FOVER pairs from each JSON file and deduplicate by (question, model_id).

    **Detailed explanation for engineers:**
        Different experiments saved their outputs in different schemas:

        - ``fover_labeled_steps_live.json`` (Exp 442): flat list of step-level dicts
          with keys {question_id, step_text, label, confidence}.  No question text,
          no model_id.  We synthesise a FOVERCorpusEntry per unique question_id, using
          the step labels as constraint_types and empty strings for response/model_id.

        - ``live_pairs_551.json`` / ``live_pairs_552.json`` (Exps 551-552): entry-level
          dicts with keys {question_index, question, model, response, is_correct,
          cot_steps, fover_labels}.  Directly maps to FOVERCorpusEntry.

        - ``exp538_cot_pairs.json`` (Exp 538 indirect): entry-level dicts with keys
          {question, cot_text, correct, model_id, latency_s}.  No FOVER labels —
          we synthesise empty constraint_types.

        Files that do not exist are silently skipped (Exp 551 data may be absent).

        Deduplication key: (question.strip(), model_id.strip()).  The first occurrence wins.

    Args:
        sources: List of file paths (relative or absolute) to load.

    Returns:
        Deduplicated list of FOVERCorpusEntry objects.

    Spec: REQ-DATA-003, SCENARIO-DATA-007
    """
    entries: list[FOVERCorpusEntry] = []
    seen: set[tuple[str, str]] = set()

    def _add(entry: FOVERCorpusEntry) -> None:
        key = (entry.question.strip(), entry.model_id.strip())
        if key not in seen:
            seen.add(key)
            entries.append(entry)

    for source_path in sources:
        path = Path(source_path)
        if not path.exists():
            # Missing sources are expected (e.g. live_pairs_551.json may not exist).
            continue

        raw = json.loads(path.read_text())

        if not isinstance(raw, list) or not raw:
            continue

        first = raw[0]

        # ------------------------------------------------------------------
        # Schema: Exp 442 step-level (question_id, step_text, label, confidence)
        # ------------------------------------------------------------------
        if "step_text" in first and "label" in first and "question_id" in first:
            # Group steps by question_id to form one entry per question.
            by_q: dict[str, list[dict]] = {}
            for step in raw:
                qid = str(step.get("question_id", ""))
                by_q.setdefault(qid, []).append(step)
            for qid, steps in by_q.items():
                constraint_types = [s.get("label", "not_verifiable") for s in steps]
                _add(
                    FOVERCorpusEntry(
                        question=qid,
                        response="",
                        model_id="unknown",
                        is_correct=False,
                        constraint_types=constraint_types,
                        cot_steps=[
                            {"step_text": s.get("step_text", ""), "z3_label": s.get("label")}
                            for s in steps
                        ],
                    )
                )

        # ------------------------------------------------------------------
        # Schema: Exps 551/552 entry-level (question, model, response, is_correct, ...)
        # ------------------------------------------------------------------
        elif "question" in first and "model" in first and "fover_labels" in first:
            for item in raw:
                fover_labels: list[str] = item.get("fover_labels", [])
                cot_steps: list[dict] = item.get("cot_steps", [])
                _add(
                    FOVERCorpusEntry(
                        question=item.get("question", ""),
                        response=item.get("response", ""),
                        model_id=item.get("model", "unknown"),
                        is_correct=bool(item.get("is_correct", False)),
                        constraint_types=fover_labels,
                        cot_steps=cot_steps,
                    )
                )

        # ------------------------------------------------------------------
        # Schema: Exp 538 indirect (question, cot_text, correct, model_id)
        # ------------------------------------------------------------------
        elif "cot_text" in first and "model_id" in first:
            for item in raw:
                _add(
                    FOVERCorpusEntry(
                        question=item.get("question", ""),
                        response=item.get("cot_text", ""),
                        model_id=item.get("model_id", "unknown"),
                        is_correct=bool(item.get("correct", False)),
                        constraint_types=[],
                        cot_steps=[],
                    )
                )

        # ------------------------------------------------------------------
        # Unknown schema — skip rather than corrupt the corpus.
        # ------------------------------------------------------------------

    return entries


# ---------------------------------------------------------------------------
# compute_corpus_diversity
# ---------------------------------------------------------------------------


def compute_corpus_diversity(entries: list[FOVERCorpusEntry]) -> dict:
    """Compute diversity metrics for a list of FOVERCorpusEntry objects.

    **Detailed explanation for engineers:**
        The primary diversity metric is Shannon entropy over the constraint_type
        distribution.  Each entry contributes its ``constraint_types`` list to a
        global counter.  We count individual step labels, not per-entry majority labels,
        so that multi-step entries with varied labels contribute proportionally.

        Shannon entropy H = -sum(p_i * log2(p_i)) where p_i is the fraction of steps
        with constraint type i.  Empty entries (no steps) do not contribute to the counter.

        ``carry_pct`` is the fraction of entries (not steps) where the majority label
        across the entry's steps is 'incorrect'.  This is the RETRO-056 quantity —
        it tracks over-representation of carry violations at the entry level.

    Args:
        entries: List of FOVERCorpusEntry objects.

    Returns:
        Dict with keys: constraint_type_counts (dict), constraint_type_entropy (float),
        carry_pct (float), correct_pct (float), n_labeled (int).

    Spec: REQ-DATA-003, SCENARIO-DATA-008
    """
    step_label_counter: Counter[str] = Counter()
    carry_count = 0
    correct_count = 0

    for entry in entries:
        for label in entry.constraint_types:
            step_label_counter[label] += 1
        # Majority label across this entry's steps determines its "type" for carry_pct.
        if entry.constraint_types:
            majority = Counter(entry.constraint_types).most_common(1)[0][0]
            if majority == "incorrect":
                carry_count += 1
            elif majority == "correct":
                correct_count += 1

    total_steps = sum(step_label_counter.values())
    if total_steps > 0:
        entropy = -sum(
            (count / total_steps) * math.log2(count / total_steps)
            for count in step_label_counter.values()
            if count > 0
        )
    else:
        entropy = 0.0

    n = len(entries)
    carry_pct = carry_count / n if n > 0 else 0.0
    correct_pct = correct_count / n if n > 0 else 0.0

    return {
        "constraint_type_counts": dict(step_label_counter),
        "constraint_type_entropy": entropy,
        "carry_pct": carry_pct,
        "correct_pct": correct_pct,
        "n_labeled": n,
    }


# ---------------------------------------------------------------------------
# balance_corpus
# ---------------------------------------------------------------------------


def balance_corpus(
    entries: list[FOVERCorpusEntry],
    target_entropy: float = 1.5,
) -> list[FOVERCorpusEntry]:
    """Downsample overrepresented entries to raise Shannon entropy to target_entropy.

    **Detailed explanation for engineers:**
        The Exp 543 corpus had 88% carry-violation entries (majority label 'incorrect').
        More generally, any single constraint type can dominate if the data collection
        was biased.  This method addresses over-representation by:

        1. Computing the current step-level entropy.
        2. If entropy >= target_entropy, return the corpus unchanged.
        3. Otherwise, find the DOMINANT step-label type (the most common label in the
           step-level counter across all entries).
        4. Find the last entry whose majority_label matches the dominant type.
        5. Remove it and re-evaluate.
        6. Repeat until entropy >= target_entropy, no more dominant-type entries exist,
           or the corpus would shrink below 10 entries.

        We target the dominant type dynamically (not always 'incorrect') because the
        actual corpus may be dominated by 'not_verifiable' steps (prose-heavy CoT data).
        Pinning to 'incorrect' would remove the wrong entries in that case.

        The downsampling removes from the tail (last occurrence first) so earlier
        entries — which often correspond to earlier experiments with better provenance —
        are preserved.

    Args:
        entries:          Full corpus list (will not be mutated).
        target_entropy:   Minimum Shannon entropy in bits to target.  Default 1.5.

    Returns:
        Balanced subset (copy) of entries with improved diversity.

    Spec: REQ-DATA-003, SCENARIO-DATA-009
    """
    if not entries:
        return []

    diversity = compute_corpus_diversity(entries)
    if diversity["constraint_type_entropy"] >= target_entropy:
        return list(entries)

    balanced = list(entries)

    def _majority_label(e: FOVERCorpusEntry) -> Optional[str]:
        if not e.constraint_types:
            return None
        return Counter(e.constraint_types).most_common(1)[0][0]

    max_iterations = len(balanced)
    for _ in range(max_iterations):
        diversity = compute_corpus_diversity(balanced)
        if diversity["constraint_type_entropy"] >= target_entropy:
            break

        # Find the dominant step-label type at the current iteration.
        counts: Counter[str] = Counter()
        for e in balanced:
            for label in e.constraint_types:
                counts[label] += 1
        if not counts:
            break
        dominant_type = counts.most_common(1)[0][0]

        # Guard: don't remove if too few entries remain.
        if len(balanced) <= 10:
            break

        # Find the last entry whose majority label matches the dominant type.
        remove_idx: Optional[int] = None
        for i in range(len(balanced) - 1, -1, -1):
            if _majority_label(balanced[i]) == dominant_type:
                remove_idx = i
                break

        if remove_idx is None:
            # No entry with dominant type majority; cannot improve further.
            break

        balanced.pop(remove_idx)

    return balanced
