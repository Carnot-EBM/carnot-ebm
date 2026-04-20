#!/usr/bin/env python3
"""Experiment 605: Live Extractor Diagnostic v4 — CoACEV4 vs DSVDAdapter gate for Exp 609.

Measures the live TP/FP of BOTH CoACEExtractorV4 (Exp 603) AND DSVDAdapter (Exp 604)
on 25 known-incorrect responses and 10 known-correct responses from live_pairs_578.json.

Why this experiment: Exp 603 (v4_recall=0.04) and Exp 604 (DSVD val_auc=0.158) both failed
individually.  Before scheduling Exp 609 (next live verify-repair attempt for RETRO-033),
we must confirm whether EITHER extractor achieves recall >= 0.20 on the test set.  If
neither crosses that threshold, scheduling Exp 609 would produce another zero-improvement run
and waste GPU resources.

Gate rule:
    gate_open = max(coace_v4_recall, dsvd_recall) >= 0.20

Protocol:
    1. apply_env_autofix() FIRST.
    2. assert_live_or_ci_skip().
    3. ExperimentTimeoutWatchdog(605, timeout_minutes=20).
    4. Load test sets from live_pairs_578.json.
    5. Run CoACEExtractorV4 on each entry.
    6. Fit DSVDLinearProbe on training split, wrap in DSVDAdapter, score each entry.
    7. Compute recall / FP rate for both extractors.
    8. Emit gate decision artifact.

Spec: REQ-BENCH-058, SCENARIO-BENCH-050, SCENARIO-BENCH-051, SCENARIO-BENCH-052
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# apply_env_autofix MUST be called before any JAX or CUDA import.
from carnot.pipeline.env_autofix import apply_env_autofix

apply_env_autofix()

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.pipeline.live_assertion import assert_live_or_ci_skip  # noqa: E402

assert_live_or_ci_skip()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_test_sets(
    live_pairs_path: Path,
    n_incorrect: int = 25,
    n_correct: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Load first n_incorrect and n_correct entries from the live pairs corpus.

    Why first-N rather than random: deterministic ordering means re-runs on the
    same corpus produce the same test set, making results directly comparable
    across Exp 603, 604, and 605.

    Returns:
        (incorrect_entries, correct_entries) — both as lists of pair dicts.
    """
    with open(live_pairs_path, encoding="utf-8") as fh:
        all_pairs = json.load(fh)

    incorrect: list[dict] = []
    correct: list[dict] = []
    for pair in all_pairs:
        if not pair.get("is_correct", True) and len(incorrect) < n_incorrect:
            incorrect.append(pair)
        elif pair.get("is_correct", False) and len(correct) < n_correct:
            correct.append(pair)
        if len(incorrect) >= n_incorrect and len(correct) >= n_correct:
            break

    return incorrect, correct


def _load_training_pairs(
    live_pairs_path: Path,
    exclude_incorrect: list[dict],
    exclude_correct: list[dict],
) -> tuple[list[str], list[float]]:
    """Build training texts and labels from all corpus pairs NOT in the test sets.

    Why exclude test pairs: avoids data leakage — the probe must not have seen
    any test-set response during training.

    Label convention: 1.0 = incorrect (violation), 0.0 = correct (no violation).
    This matches DSVDLinearProbe.fit() expectation.

    Returns:
        (step_texts, labels) — parallel lists for DSVDLinearProbe.fit().
    """
    exclude_keys: set[tuple] = set()
    for pair in exclude_incorrect + exclude_correct:
        key = (pair.get("question_index"), pair.get("model"), pair.get("response", "")[:40])
        exclude_keys.add(key)

    with open(live_pairs_path, encoding="utf-8") as fh:
        all_pairs = json.load(fh)

    texts: list[str] = []
    labels: list[float] = []
    for pair in all_pairs:
        key = (pair.get("question_index"), pair.get("model"), pair.get("response", "")[:40])
        if key in exclude_keys:
            continue
        response = pair.get("response", "")
        is_correct = pair.get("is_correct", True)
        texts.append(response)
        labels.append(0.0 if is_correct else 1.0)

    return texts, labels


def _run_coace_v4(entries: list[dict]) -> list[bool]:
    """Run CoACEExtractorV4.extract() on each entry and return violation flags.

    Why run the full V4 extractor rather than V3: we want to benchmark the V4
    GenPRM-augmented path since that is the extractor being evaluated for production.
    A violation is detected when n_violations > 0 in the returned CoACEResult.

    Returns:
        List of bools — True if a violation was detected for that entry.
    """
    from carnot.extraction.coace_extractor_v4 import CoACEExtractorV4  # noqa: PLC0415

    extractor = CoACEExtractorV4(llm_caller=None)  # CI stub mode — no LLM needed
    flags: list[bool] = []
    for entry in entries:
        response = entry.get("response", "")
        try:
            result = extractor.extract(response)
            flags.append(result.n_violations > 0)
        except Exception as exc:  # noqa: BLE001 — extractor errors are non-fatal
            _log.warning("CoACEV4 extract() failed for entry: %s", exc)
            flags.append(False)
    return flags


def _build_dsvd_adapter(train_texts: list[str], train_labels: list[float]):
    """Create, optionally fit, and return a DSVDAdapter.

    Why fit on training data: the DSVDLinearProbe starts with zero weights
    (all predictions = 0.5), which means nothing is ever flagged as a violation.
    Fitting on real correct/incorrect pairs gives the probe a chance to learn
    the text-feature distribution that separates the two classes.

    When no training data is available (empty lists), returns an unfitted adapter.
    Predictions will cluster at 0.5 — all below the 0.5 threshold — yielding
    recall=0 and fp_rate=0.  This is an honest result if no training data exists.
    """
    from carnot.pipeline.dsvd_adapter import DSVDAdapter, DSVDLinearProbe  # noqa: PLC0415

    probe = DSVDLinearProbe(hidden_dim=64)
    if train_texts:
        probe.fit(train_texts, train_labels)
    return DSVDAdapter(probe=probe, violation_threshold=0.5)


def _run_dsvd(adapter, entries: list[dict]) -> list[bool]:
    """Score each entry with DSVDAdapter.verify_step() and return violation flags.

    Each entry's full response text is treated as a single CoT step.
    A violation is detected when violation_probability > adapter.violation_threshold.

    Returns:
        List of bools — True if the probe flagged the entry.
    """
    flags: list[bool] = []
    for entry in entries:
        response = entry.get("response", "")
        try:
            result = adapter.verify_step(response)
            flags.append(result.violation_probability > adapter.violation_threshold)
        except Exception as exc:  # noqa: BLE001 — probe errors are non-fatal
            _log.warning("DSVDAdapter.verify_step() failed: %s", exc)
            flags.append(False)
    return flags


def _build_artifact(
    tmpl,
    coace_v4_tp_flags: list[bool],
    coace_v4_fp_flags: list[bool],
    dsvd_tp_flags: list[bool],
    dsvd_fp_flags: list[bool],
    n_incorrect: int,
    n_correct: int,
) -> dict:
    """Compute all metrics and return the result artifact dict.

    Why a dedicated builder: isolates the metric math so tests can call it
    without running the full experiment pipeline.
    """
    coace_v4_tp = sum(coace_v4_tp_flags)
    coace_v4_fp = sum(coace_v4_fp_flags)
    coace_v4_recall = coace_v4_tp / n_incorrect if n_incorrect > 0 else 0.0
    coace_v4_fp_rate = coace_v4_fp / n_correct if n_correct > 0 else 0.0

    dsvd_tp = sum(dsvd_tp_flags)
    dsvd_fp = sum(dsvd_fp_flags)
    dsvd_recall = dsvd_tp / n_incorrect if n_incorrect > 0 else 0.0
    dsvd_fp_rate = dsvd_fp / n_correct if n_correct > 0 else 0.0

    winning_extractor = "coace_v4" if coace_v4_recall >= dsvd_recall else "dsvd"
    best_recall = max(coace_v4_recall, dsvd_recall)
    gate_open = best_recall >= 0.20

    if gate_open:
        gate_note = "Proceed to Exp 609 if gate_open=True"
        honest_verdict = "gate_open_proceed_to_vr"
    else:
        gate_note = "DO NOT schedule Exp 609 — recall below 20%. Escalate to RETRO-070."
        honest_verdict = "gate_closed_recall_below_threshold"

    payload = {
        "result_schema": "carnot.extractor_diagnostic_v4.v1",
        "n_incorrect": n_incorrect,
        "n_correct": n_correct,
        "coace_v4_recall": round(coace_v4_recall, 4),
        "coace_v4_fp_rate": round(coace_v4_fp_rate, 4),
        "dsvd_recall": round(dsvd_recall, 4),
        "dsvd_fp_rate": round(dsvd_fp_rate, 4),
        "winning_extractor": winning_extractor,
        "best_recall": round(best_recall, 4),
        "gate_open": gate_open,
        "gate_note": gate_note,
        "honest_verdict": honest_verdict,
    }

    return tmpl.build_result(payload, status="success")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the combined extractor diagnostic and emit the gate decision artifact."""
    import sys
    _SCRIPT_DIR = Path(__file__).resolve().parent
    if str(_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPT_DIR))

    # Import here so the sys.path is set before loading ExperimentTemplate
    _SCRIPTS_ROOT = _REPO_ROOT / "scripts"
    if str(_SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_ROOT))

    from experiment_template import ExperimentTemplate  # noqa: PLC0415

    _DELIVERABLE = "results/experiment_605_extractor_diagnostic_v4.json"

    with ExperimentTimeoutWatchdog(
        605,
        timeout_minutes=20,
        result_path=str(_REPO_ROOT / _DELIVERABLE),
    ):
        tmpl = ExperimentTemplate(
            605,
            "Live Extractor Diagnostic v4",
            _DELIVERABLE,
            requires_gpu=False,
        )
        tmpl.setup()

        live_pairs_path = _REPO_ROOT / "results" / "live_pairs_578.json"
        _log.info("Loading test sets from %s", live_pairs_path)

        incorrect_entries, correct_entries = _load_test_sets(live_pairs_path, 25, 10)
        _log.info(
            "Test set: %d incorrect, %d correct",
            len(incorrect_entries),
            len(correct_entries),
        )

        # Training data for DSVD probe — exclude test entries to prevent leakage.
        train_texts, train_labels = _load_training_pairs(
            live_pairs_path, incorrect_entries, correct_entries
        )
        _log.info("DSVD training split: %d entries", len(train_texts))

        # CoACEExtractorV4 pass
        _log.info("Running CoACEExtractorV4 on %d incorrect entries ...", len(incorrect_entries))
        coace_tp_flags = _run_coace_v4(incorrect_entries)

        _log.info("Running CoACEExtractorV4 on %d correct entries (FP check) ...", len(correct_entries))
        coace_fp_flags = _run_coace_v4(correct_entries)

        # DSVDAdapter pass
        _log.info("Building DSVDAdapter (fitting on %d training examples) ...", len(train_texts))
        adapter = _build_dsvd_adapter(train_texts, train_labels)

        _log.info("Running DSVDAdapter on %d incorrect entries ...", len(incorrect_entries))
        dsvd_tp_flags = _run_dsvd(adapter, incorrect_entries)

        _log.info("Running DSVDAdapter on %d correct entries (FP check) ...", len(correct_entries))
        dsvd_fp_flags = _run_dsvd(adapter, correct_entries)

        # Build and write artifact
        artifact = _build_artifact(
            tmpl,
            coace_tp_flags,
            coace_fp_flags,
            dsvd_tp_flags,
            dsvd_fp_flags,
            n_incorrect=len(incorrect_entries),
            n_correct=len(correct_entries),
        )

        output_path = _REPO_ROOT / _DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=2)

        _log.info("Artifact written to %s", output_path)
        _log.info(
            "Gate: coace_v4_recall=%.4f  dsvd_recall=%.4f  gate_open=%s",
            artifact["coace_v4_recall"],
            artifact["dsvd_recall"],
            artifact["gate_open"],
        )

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
