#!/usr/bin/env python3
"""Exp 542 — FOVER Corpus Expansion via multi-source merge.

**Researcher summary:**
    JEPA v7 (Exp 535) was trained on 57 real FOVER pairs from fover_labeled_steps_live.json.
    Exp 538 produced 25 additional live CoT responses (50 total across two models).
    This experiment annotates the Exp 538 responses with FOVERAnnotator (CPU-only, Z3-based),
    merges them with the 57 prior pairs, deduplicates by step_text SHA-256 hash, and writes
    the expanded corpus to results/fover_labeled_steps_expanded.json.

    Target: 100+ labeled pairs for JEPA v8 retrain (Exp 543).
    No GPU required — FOVER annotation is pure Z3 on CPU.

Spec: REQ-LEARN-055, SCENARIO-LEARN-086, SCENARIO-LEARN-087
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root bootstrap — allows running from any CWD.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402


# ---------------------------------------------------------------------------
# Public helper: merge_fover_corpora
# ---------------------------------------------------------------------------


def merge_fover_corpora(
    prior_pairs: list[dict],
    new_pairs: list[dict],
) -> list[dict]:
    """Merge two FOVER training-pair lists, deduplicating by SHA-256 of step_text.

    **Why SHA-256 dedup instead of exact equality:**
        step_text strings can be long multi-line CoT fragments.  Hashing them to
        32 bytes makes the seen-set O(1) lookup constant regardless of step length,
        and avoids embedding the full text as a dict key.

        Two steps are considered duplicates when their step_text produces the same
        SHA-256 digest.  This is collision-resistant for strings of practical length.

    **Why prior_pairs first:**
        The merge inserts prior_pairs first so that existing training data is never
        displaced by a new annotation of the same text.  New pairs fill the remaining
        slots.  Order within each source list is preserved.

    Args:
        prior_pairs: Existing training pairs (e.g. from fover_labeled_steps_live.json).
        new_pairs:   Newly annotated pairs from a fresh FOVERAnnotator run.

    Returns:
        Deduplicated merged list.  Length <= len(prior_pairs) + len(new_pairs).

    Spec: REQ-LEARN-055, SCENARIO-LEARN-086
    """
    seen: set[str] = set()
    merged: list[dict] = []

    for pair in prior_pairs + new_pairs:
        text = pair.get("step_text", "")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            merged.append(pair)

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def pick_honest_verdict(n_prior: int, n_new: int, n_total: int) -> str:
    """Return the honest_verdict string based on corpus size outcomes.

    **Why a separate function:**
        Isolates the verdict logic so it can be tested independently of
        file I/O and the watchdog context manager.  Same logic as in main().

    Args:
        n_prior: Number of prior training pairs loaded.
        n_new:   Number of newly annotated pairs from exp538.
        n_total: Total unique pairs after merge + dedup.

    Returns:
        'corpus_expanded'   — target achieved (>= 100 total pairs).
        'partial_expansion' — some new pairs added but below 100.
        'synthetic_fallback'— no new pairs added (n_total <= n_prior).

    Spec: REQ-LEARN-055
    """
    if n_total >= 100:
        return "corpus_expanded"
    if n_total > n_prior:
        return "partial_expansion"
    return "synthetic_fallback"


def main() -> None:
    """Expand the FOVER training corpus from two sources and write the merged output.

    Execution steps:
    1. apply_env_autofix() — fixes common JAX/ROCm env issues before anything loads.
    2. ExperimentTimeoutWatchdog(542, 30) — hard 30-minute wall-clock kill.
    3. ExperimentTemplate.setup() — creates output dirs, loads any checkpoint.
    4. Load prior 57-pair corpus from results/fover_labeled_steps_live.json.
    5. Load Exp 538 CoT pairs from results/exp538_cot_pairs.json (if present).
    6. Run FOVERAnnotator on Exp 538 responses.
    7. Convert annotated steps to training pairs via to_training_pairs().
    8. Merge + deduplicate by step_text SHA-256.
    9. Write merged corpus to results/fover_labeled_steps_expanded.json.
    10. Build and write artifact; assert deliverable written.
    """
    apply_env_autofix()

    with ExperimentTimeoutWatchdog(542, timeout_minutes=30):
        tmpl = ExperimentTemplate(
            542,
            "FOVER Corpus Expansion",
            "results/experiment_542_fover_expansion.json",
            requires_gpu=False,
        )
        tmpl.setup()

        repo_root = _REPO_ROOT

        # ------------------------------------------------------------------
        # Step 4: Load prior corpus
        # ------------------------------------------------------------------
        prior_path = repo_root / "results" / "fover_labeled_steps_live.json"
        prior_pairs: list[dict] = []
        if prior_path.exists():
            prior_pairs = json.loads(prior_path.read_text())
        n_prior = len(prior_pairs)

        # ------------------------------------------------------------------
        # Step 5: Load Exp 538 CoT pairs (tolerates absence per SCENARIO-LEARN-087)
        # ------------------------------------------------------------------
        exp538_path = repo_root / "results" / "exp538_cot_pairs.json"
        exp538_items: list[dict] = []
        if exp538_path.exists():
            exp538_items = json.loads(exp538_path.read_text())

        # ------------------------------------------------------------------
        # Step 6: Annotate new CoT responses with FOVERAnnotator
        # ------------------------------------------------------------------
        # Import here to avoid pulling Z3 into the module namespace at load time,
        # keeping the module importable even in Z3-free test environments that mock
        # the annotator.
        from carnot.pipeline.fover_annotator import FOVERAnnotator  # noqa: PLC0415

        annotator = FOVERAnnotator(z3_timeout_seconds=5)

        # exp538_cot_pairs items have keys: question, cot_text, correct, model_id, latency_s
        # FOVERAnnotator.annotate_corpus() expects list[dict] with 'response' and optional 'question_id'.
        response_dicts = [
            {"response": item.get("cot_text", ""), "question_id": f"exp538_{i}"}
            for i, item in enumerate(exp538_items)
        ]
        annotated = annotator.annotate_corpus(response_dicts)
        new_pairs = annotator.to_training_pairs(annotated, response_dicts)
        n_new = len(new_pairs)

        # ------------------------------------------------------------------
        # Step 8: Merge and deduplicate
        # ------------------------------------------------------------------
        merged = merge_fover_corpora(prior_pairs, new_pairs)
        n_total = len(merged)

        # ------------------------------------------------------------------
        # Step 9: Write expanded corpus
        # ------------------------------------------------------------------
        expanded_path = repo_root / "results" / "fover_labeled_steps_expanded.json"
        expanded_path.write_text(json.dumps(merged, indent=2))

        # ------------------------------------------------------------------
        # Step 10: Build artifact and write
        # ------------------------------------------------------------------
        honest_verdict = pick_honest_verdict(n_prior, n_new, n_total)

        artifact = tmpl.build_result(
            {
                "schema": "carnot.fover_expansion.v1",
                "n_prior_pairs": n_prior,
                "n_new_pairs": n_new,
                "n_total_pairs": n_total,
                "expanded_corpus_path": str(expanded_path.relative_to(repo_root)),
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        expanded_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = repo_root / "results" / "experiment_542_fover_expansion.json"
        output_path.write_text(json.dumps(artifact, indent=2))

        tmpl.assert_deliverable_written()

        print(
            f"Exp 542 complete: prior={n_prior} + new={n_new} → total={n_total} "
            f"pairs ({honest_verdict})"
        )


if __name__ == "__main__":
    main()
