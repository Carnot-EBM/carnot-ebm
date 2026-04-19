#!/usr/bin/env python3
"""Experiment 496: NUP Probe v2 — Bayesian Semantic Entropy (RETRO-047).

**Research question:**
    Can Bayesian credible intervals over Shannon entropy improve NUPProbe AUC
    from 0.600 (Exp 484) to > 0.700, qualifying it as Tier 0c in the pipeline?

**Root cause of Exp 484 failure (RETRO-047):**
    Character-entropy fallback (no logprobs available) has much lower discriminative
    power because: (a) character entropy correlates only weakly with token-level LLM
    uncertainty, and (b) many steps land near the 1.5 nat threshold, causing the
    point-estimate rule to fire false positives on 'uncertain' cases.

**v2 fix — Bayesian credible intervals:**
    BayesianEntropyEstimator computes [lower_ci, upper_ci] via Beta-conjugate posterior
    on token probabilities.  NUPProbeV2.predict_violation() fires only when
    lower_ci > threshold (confidently high entropy), not when entropy is merely
    near-threshold.  This reduces FP rate on ambiguous steps.
    (arXiv 2603.22812, AAAI 2026 oral: 12.6% AUROC improvement from adaptive sampling.)

**Data sources:**
    1. results/fover_labeled_steps_live.json — 57 real labeled CoT steps (always present)
    2. results/exp488_cot_pairs.json         — 100 CoT pairs from Exp 488 (if present)
    3. results/exp489_cot_pairs.json         — 200 CoT pairs from Exp 489 (if present)

Spec: REQ-VERIFY-098, REQ-VERIFY-099, REQ-VERIFY-100,
      SCENARIO-VERIFY-131, SCENARIO-VERIFY-132, SCENARIO-VERIFY-133
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — scripts/ is not a package; add repo root to sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Apply env autofix FIRST (REQ-INFRA-021)
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.nup_probe_v2 import NUPProbeV2  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

_DELIVERABLE = "results/experiment_496_nup_probe_v2.json"

# AUC from Exp 484 (NUPProbe v1) — reference baseline
_AUC_V1 = 0.600


def load_labeled_pairs() -> tuple[list[dict], int, int]:
    """Load and merge labeled CoT pairs from all available sources.

    Returns (pairs, n_with_logprobs, n_char_entropy_fallback).

    **Why merge sources:**
        More pairs → lower variance in AUC estimate.  fover_labeled_steps_live.json
        is always present.  Exp 488/489 pairs add volume if available, increasing
        statistical power of the AUC estimate.

    **Why count logprob availability:**
        Transparent reporting tells researchers how much of the AUC signal came from
        the precise token-logprob path vs. the character-entropy proxy.
    """
    pairs: list[dict] = []

    # Always load the primary labeled set
    fover_path = _REPO_ROOT / "results" / "fover_labeled_steps_live.json"
    if fover_path.exists():
        raw = json.loads(fover_path.read_text())
        # Normalise field names: fover uses 'step_text' and 'label'
        for item in raw:
            pairs.append({
                "step_text": item.get("step_text", item.get("cot_text", "")),
                "label": item.get("label", "incorrect"),
                "logprobs": item.get("logprobs"),
            })
        _log.info("Loaded %d pairs from fover_labeled_steps_live.json", len(raw))
    else:
        _log.warning("fover_labeled_steps_live.json not found — proceeding with empty set")

    # Optional: Exp 488 CoT pairs
    exp488_path = _REPO_ROOT / "results" / "exp488_cot_pairs.json"
    if exp488_path.exists():
        raw488 = json.loads(exp488_path.read_text())
        for item in (raw488 if isinstance(raw488, list) else raw488.get("pairs", [])):
            pairs.append({
                "step_text": item.get("step_text", item.get("cot_text", "")),
                "label": item.get("label", "incorrect"),
                "logprobs": item.get("logprobs"),
            })
        _log.info("Loaded %d pairs from exp488_cot_pairs.json", len(raw488) if isinstance(raw488, list) else len(raw488.get("pairs", [])))

    # Optional: Exp 489 CoT pairs
    exp489_path = _REPO_ROOT / "results" / "exp489_cot_pairs.json"
    if exp489_path.exists():
        raw489 = json.loads(exp489_path.read_text())
        for item in (raw489 if isinstance(raw489, list) else raw489.get("pairs", [])):
            pairs.append({
                "step_text": item.get("step_text", item.get("cot_text", "")),
                "label": item.get("label", "incorrect"),
                "logprobs": item.get("logprobs"),
            })
        _log.info("Loaded %d pairs from exp489_cot_pairs.json", len(raw489) if isinstance(raw489, list) else len(raw489.get("pairs", [])))

    n_with_logprobs = sum(1 for p in pairs if p.get("logprobs") is not None)
    n_char_entropy_fallback = len(pairs) - n_with_logprobs
    return pairs, n_with_logprobs, n_char_entropy_fallback


def train_test_split(pairs: list[dict], test_fraction: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Split pairs into train/test sets using a deterministic 80/20 split.

    **Why held-out test set:**
        AUC computed on training data is optimistically biased.  A 20% held-out
        split provides an unbiased estimate of generalisation performance.
        We use a deterministic (not random) split for reproducibility.
    """
    n_test = max(2, int(len(pairs) * test_fraction))
    test_pairs = pairs[-n_test:]
    train_pairs = pairs[:-n_test]
    return train_pairs, test_pairs


def main() -> None:
    """Run NUP Probe v2 Bayesian entropy experiment."""
    tmpl = ExperimentTemplate(
        496,
        "NUP Probe v2 — Bayesian Semantic Entropy",
        _DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    guard = DeliverableGuard(_DELIVERABLE)

    with ExperimentTimeoutWatchdog(496, timeout_minutes=30):
        # Load all available labeled pairs
        all_pairs, n_with_logprobs, n_char_entropy_fallback = load_labeled_pairs()
        _log.info(
            "Total pairs: %d | with logprobs: %d | char-entropy fallback: %d",
            len(all_pairs), n_with_logprobs, n_char_entropy_fallback,
        )

        if len(all_pairs) < 4:
            _log.error("Insufficient labeled pairs (%d) — need at least 4", len(all_pairs))
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.nup_probe.v2",
                    "n_pairs": len(all_pairs),
                    "n_with_logprobs": n_with_logprobs,
                    "n_char_entropy_fallback": n_char_entropy_fallback,
                    "auc_v1": _AUC_V1,
                    "auc_v2": 0.5,
                    "auc_improvement": 0.5 - _AUC_V1,
                    "is_viable_tier_0c": False,
                    "retro_047_closed": False,
                    "honest_verdict": "insufficient_data",
                },
                status="blocked",
            )
            Path(_DELIVERABLE).write_text(json.dumps(artifact, indent=2))
            guard.assert_written()
            return

        # 80/20 train/test split (evaluate on held-out 20%)
        _train_pairs, test_pairs = train_test_split(all_pairs, test_fraction=0.2)
        _log.info("Evaluating on %d held-out test pairs", len(test_pairs))

        # Instantiate NUPProbeV2 and evaluate
        probe = NUPProbeV2(hallucination_threshold=1.5, confidence_level=0.95)
        result = probe.evaluate(test_pairs)

        auc_v2 = result.auc
        auc_improvement = auc_v2 - _AUC_V1
        is_viable = result.is_viable_tier_0c
        honest_verdict = "tier_0c_viable" if is_viable else "improvement_below_threshold"

        _log.info(
            "NUPProbeV2 AUC=%.4f (v1 baseline=%.3f, improvement=%.4f) | viable=%s",
            auc_v2, _AUC_V1, auc_improvement, is_viable,
        )

        artifact = tmpl.build_result(
            {
                "schema": "carnot.nup_probe.v2",
                "n_pairs": len(test_pairs),
                "n_with_logprobs": sum(1 for p in test_pairs if p.get("logprobs") is not None),
                "n_char_entropy_fallback": sum(1 for p in test_pairs if p.get("logprobs") is None),
                "auc_v1": _AUC_V1,
                "auc_v2": auc_v2,
                "auc_improvement": auc_improvement,
                "is_viable_tier_0c": is_viable,
                "retro_047_closed": is_viable,
                "honest_verdict": honest_verdict,
                "probe_latency_ms": result.probe_latency_ms,
                "hallucination_threshold": probe.hallucination_threshold,
                "confidence_level": 0.95,
            },
            status="success",
        )

        Path(_DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        _log.info("Deliverable written to %s", _DELIVERABLE)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
