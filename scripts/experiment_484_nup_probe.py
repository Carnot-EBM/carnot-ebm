#!/usr/bin/env python3
"""Experiment 484: NUP Probe Tier 0c — evaluate continuation-entropy as violation pre-filter.

**Research question:**
    Can the Shannon entropy of a CoT step's continuation distribution (from the Neural
    Uncertainty Principle, arXiv 2603.19562) predict constraint violations with AUC > 0.700
    on live labeled CoT data?  If yes, NUPProbe qualifies as Tier 0c in the Carnot
    verification cascade — a zero-latency pre-filter requiring no LLM call.

**Why this matters:**
    The verification cascade is ordered by cost.  Adding a zero-latency pre-filter that
    filters 30–70% of steps before Ising or ThinkProbe runs reduces total verification cost
    in proportion to its skip rate.  The AUC threshold of 0.700 ensures the signal quality
    justifies the cascade position.

**Data sources:**
    1. results/fover_labeled_steps_live.json — 57 real labeled CoT steps from live runs
    2. results/exp476_cot_pairs.json         — additional pairs from Exp 476 (if present)

**Logprob availability:**
    Current live data does not include per-step token log-probabilities.  NUPProbe falls
    back to character-level entropy as a structural proxy.  This experiment reports
    n_char_entropy_fallback to be transparent about this limitation.

Spec: REQ-VERIFY-096, REQ-VERIFY-097,
      SCENARIO-VERIFY-129, SCENARIO-VERIFY-130
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — scripts/ is not a package; add repo root to sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Apply env autofix FIRST (REQ-INFRA-021: belt-and-suspenders GPU env fix)
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.nup_probe import NUPProbe, NUPProbeResult  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

_DELIVERABLE = "results/experiment_484_nup_probe.json"


def load_labeled_pairs() -> tuple[list[dict], int, int]:
    """Load and merge labeled CoT pairs from all available sources.

    Returns a tuple of (pairs, n_with_logprobs, n_char_entropy_fallback).

    **Why merge two sources:**
        fover_labeled_steps_live.json provides ground-truth labeled steps from live
        model runs.  exp476_cot_pairs.json (if present) adds additional pairs from
        Exp 476.  Merging increases the held-out evaluation set size, reducing
        variance in the AUC estimate.

    **Why count logprob availability:**
        Transparent reporting of n_char_entropy_fallback tells future researchers
        how much of the AUC signal came from the precise token-logprob path vs. the
        less-precise character-entropy proxy.
    """
    pairs: list[dict] = []
    n_with_logprobs = 0

    live_path = _REPO_ROOT / "results" / "fover_labeled_steps_live.json"
    if live_path.exists():
        with open(live_path) as f:
            live_data = json.load(f)
        for item in live_data:
            if "step_text" not in item and "cot_text" not in item:
                # Normalise: fover data uses step_text
                item = dict(item)
            if "logprobs" in item and item["logprobs"]:
                n_with_logprobs += 1
            pairs.append(item)
        _log.info("Loaded %d pairs from fover_labeled_steps_live.json", len(live_data))
    else:
        _log.warning("fover_labeled_steps_live.json not found — no live data")

    exp476_path = _REPO_ROOT / "results" / "exp476_cot_pairs.json"
    if exp476_path.exists():
        with open(exp476_path) as f:
            exp476_data = json.load(f)
        for item in exp476_data:
            if "logprobs" in item and item["logprobs"]:
                n_with_logprobs += 1
            pairs.append(item)
        _log.info("Loaded %d pairs from exp476_cot_pairs.json", len(exp476_data))

    n_char_entropy_fallback = len(pairs) - n_with_logprobs
    return pairs, n_with_logprobs, n_char_entropy_fallback


def held_out_split(pairs: list[dict], held_out_frac: float = 0.2) -> list[dict]:
    """Return the held-out 20% of pairs for AUC evaluation.

    We use the last 20% (not random) for reproducibility — the same split every run
    regardless of Python's random seed state.  With 57 pairs this is 11–12 pairs.
    With more pairs from exp476, the split grows proportionally.
    """
    n_held = max(2, int(len(pairs) * held_out_frac))
    return pairs[-n_held:]


def main() -> None:
    # Experiment template: manages dirs, checkpoint, timing, result schema
    tmpl = ExperimentTemplate(
        484,
        "NUP Probe Tier 0c",
        _DELIVERABLE,
        requires_gpu=False,  # CPU-only: pure arithmetic on frozen activations
    )
    tmpl.setup()

    guard = DeliverableGuard(_DELIVERABLE)

    with ExperimentTimeoutWatchdog(484, timeout_minutes=30):
        # Load data
        all_pairs, n_with_logprobs, n_char_entropy_fallback = load_labeled_pairs()
        if len(all_pairs) < 4:
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.nup_probe.v1",
                    "n_pairs": len(all_pairs),
                    "n_with_logprobs": n_with_logprobs,
                    "n_char_entropy_fallback": n_char_entropy_fallback,
                    "auc": 0.5,
                    "threshold": 1.5,
                    "probe_latency_ms": 0.0,
                    "is_viable_tier_0c": False,
                    "honest_verdict": "insufficient_data",
                },
                status="blocked",
            )
            output_path = _REPO_ROOT / _DELIVERABLE
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(artifact, f, indent=2)
            _log.info("Blocked: insufficient data (%d pairs)", len(all_pairs))
            tmpl.assert_deliverable_written()
            return

        # Build held-out set for AUC evaluation
        held_out = held_out_split(all_pairs)
        _log.info(
            "Evaluating on %d held-out pairs (20%% of %d total)",
            len(held_out),
            len(all_pairs),
        )

        # Instantiate probe
        probe = NUPProbe(entropy_threshold=1.5)

        # Measure per-call latency on the full set
        t0 = time.perf_counter()
        for pair in all_pairs:
            text = pair.get("step_text") or pair.get("cot_text", "")
            lp = pair.get("logprobs") or None
            probe.score(text, lp)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        probe_latency_ms = elapsed_ms / max(1, len(all_pairs))

        # Evaluate AUC on held-out set
        auc = probe.evaluate_auc(held_out)
        _log.info("NUPProbe AUC on held-out set: %.4f", auc)

        # Build NUPProbeResult
        result_obj = NUPProbeResult(
            n_pairs=len(held_out),
            auc=auc,
            threshold_used=1.5,
            probe_latency_ms=probe_latency_ms,
        )

        honest_verdict = (
            "tier_0c_viable" if result_obj.is_viable_tier_0c else "improvement_below_threshold"
        )

        artifact = tmpl.build_result(
            {
                "schema": "carnot.nup_probe.v1",
                "n_pairs": result_obj.n_pairs,
                "n_with_logprobs": n_with_logprobs,
                "n_char_entropy_fallback": n_char_entropy_fallback,
                "auc": result_obj.auc,
                "threshold": result_obj.threshold_used,
                "probe_latency_ms": result_obj.probe_latency_ms,
                "is_viable_tier_0c": result_obj.is_viable_tier_0c,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        output_path = _REPO_ROOT / _DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2)

        _log.info(
            "Experiment 484 complete: AUC=%.4f, is_viable_tier_0c=%s, verdict=%s",
            auc,
            result_obj.is_viable_tier_0c,
            honest_verdict,
        )

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
