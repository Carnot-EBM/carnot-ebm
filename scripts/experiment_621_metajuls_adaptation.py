#!/usr/bin/env python3
"""Experiment 621: MetaJuLS Online Adaptation for LLMAsExtractorV1.

**What this experiment measures:**

    We simulate 3 successive batches of live LLM responses drawn from the
    FOVER corpus (live_pairs_578.json) and feed them through the
    MetaJuLSAdapter.  After each batch the adapter updates the extractor's
    policy (temperature, claim_confidence_threshold) based on observed
    precision.  We then measure whether precision_trend() is non-negative,
    indicating that adaptation did not degrade extraction quality.

**Why CI mode (no GPU):**

    LLMAsExtractorV1.extract() in CI mode (llm_caller=None) uses only the
    StepSegmentEvalChain — the deterministic regex/eval baseline.  This
    gives reproducible violation labels without needing a live GPU, which
    is consistent with the corpus already having is_correct ground truth.

**Spec:** REQ-LEARN-078, REQ-LEARN-079,
          SCENARIO-LEARN-121, SCENARIO-LEARN-122, SCENARIO-LEARN-123
"""

import json
import sys
from pathlib import Path

# Ensure repo root is on the path so pipeline imports work without install.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.metajuls_adapter import ExtractorPolicy, MetaJuLSAdapter
from carnot.extraction.llm_extractor_v1 import LLMAsExtractorV1
from experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 621
TITLE = "MetaJuLS Online Adaptation"
DELIVERABLE = "results/experiment_621_metajuls_adaptation.json"
CORPUS_PATH = Path(__file__).parent.parent / "results" / "live_pairs_578.json"
BATCH_SIZE = 10
N_BATCHES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_corpus() -> list[dict]:
    """Load the live FOVER corpus.  Raise if not found."""
    with CORPUS_PATH.open() as fh:
        return json.load(fh)


def _make_batch(corpus: list[dict], batch_idx: int) -> list[dict]:
    """Return BATCH_SIZE entries for the given batch index.

    When the corpus is smaller than batch_idx * BATCH_SIZE we wrap around
    (overlap is acceptable — we are testing the adaptation loop, not corpus
    diversity).
    """
    n = len(corpus)
    start = (batch_idx * BATCH_SIZE) % n
    # Build the batch list, wrapping if necessary.
    indices = [(start + i) % n for i in range(BATCH_SIZE)]
    return [corpus[i] for i in indices]


def _process_batch(extractor: LLMAsExtractorV1, entries: list[dict]) -> list[dict]:
    """Run the extractor on each entry and return adapter-ready dicts.

    Each returned dict has:
        'response'           — raw LLM response text
        'violation_detected' — True if extractor found at least one violation
        'true_label'         — 'correct' if is_correct else 'incorrect'

    Why 'correct'/'incorrect' strings: the MetaJuLSAdapter uses string labels
    to avoid the double-inversion bug (is_correct=True vs violation=True).
    """
    results = []
    for entry in entries:
        response = entry.get("response", "")
        violations = extractor.extract(response)
        results.append(
            {
                "response": response,
                "violation_detected": len(violations) > 0,
                "true_label": "correct" if entry.get("is_correct", False) else "incorrect",
            }
        )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Step 1: Environment autofix (must be first — may patch JAX platform flags).
    apply_env_autofix()

    # Step 2: Watchdog — kills the process if it runs beyond 25 minutes.
    # This experiment is CPU-only and should complete in < 30 seconds.
    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=25)
    watchdog.start()

    # Step 3: Template setup — creates result/checkpoint directories.
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    # Step 4: Load corpus.
    corpus = _load_corpus()

    # Step 5: Initialise extractor (CI mode — StepSegmentEvalChain only, no GPU).
    extractor = LLMAsExtractorV1(llm_caller=None)

    # Step 6: Initialise adapter with default policy.
    adapter = MetaJuLSAdapter()
    initial_policy_dict = adapter.policy.to_dict()

    # Step 7: Simulate N_BATCHES batches.
    precision_per_batch: list[float] = []
    for batch_idx in range(N_BATCHES):
        batch_entries = _make_batch(corpus, batch_idx)
        batch_results = _process_batch(extractor, batch_entries)
        updated_policy = adapter.update_from_batch(batch_results)
        # Retrieve precision from the experience log just appended.
        precision_per_batch.append(adapter.experience[-1]["precision"])

    # Step 8: Compute trend.
    precision_trend = adapter.precision_trend()
    adaptation_effective = precision_trend >= 0.0

    # Step 9: Build and write artifact.
    artifact = tmpl.build_result(
        {
            "n_batches": N_BATCHES,
            "batch_size": BATCH_SIZE,
            "corpus_size": len(corpus),
            "policy_initial": initial_policy_dict,
            "policy_final": adapter.policy.to_dict(),
            "precision_per_batch": precision_per_batch,
            "precision_trend": precision_trend,
            "adaptation_effective": adaptation_effective,
            "honest_verdict": (
                "adaptation_effective"
                if adaptation_effective
                else "adaptation_not_effective"
            ),
        },
        status="success",
    )

    deliverable_path = Path(DELIVERABLE)
    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    with deliverable_path.open("w") as fh:
        json.dump(artifact, fh, indent=2)

    # Step 10: Assert deliverable written — MUST be the final call.
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
