#!/usr/bin/env python3
"""Experiment 606: Interleaved Formal Logic Verifier.

Measures the recall and false-positive rate of InterleavedLogicVerifier on
25 known-incorrect and 10 known-correct live responses from live_pairs_578.json.

Why this experiment: arXiv 2601.22642 shows 10.4-14.2% accuracy gains from
inserting lightweight Z3 checks at CoT step boundaries rather than post-hoc.
Exp 605 showed CoACEV3 recall=0.04 — too low to use.  ILV is CPU-only (Z3 on
constant-folded expressions), so it requires no GPU and has <1 ms per step.
This experiment gates the usefulness of ILV for the Tier 3.5 pipeline slot.

Gate rule:
    honest_verdict = 'ilv_viable' if ilv_recall >= 0.20
                  else 'ilv_improved' if ilv_recall > 0.04
                  else 'ilv_no_improvement'

Spec: REQ-VERIFY-135, SCENARIO-VERIFY-168, SCENARIO-VERIFY-169, SCENARIO-VERIFY-170
"""

from __future__ import annotations

import json
import logging
import os
import sys
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

    Why first-N rather than random: deterministic ordering makes results directly
    comparable across Exp 603, 604, 605, and 606.

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


def _run_ilv(entries: list[dict]) -> list[bool]:
    """Run InterleavedLogicVerifier.verify_response() on each entry.

    A violation is detected when any step in the response has violation_detected=True.

    Returns:
        List of bools — True if a violation was detected for that entry.
    """
    from carnot.pipeline.interleaved_verifier import InterleavedLogicVerifier  # noqa: PLC0415

    verifier = InterleavedLogicVerifier(z3_timeout_ms=50)
    flags: list[bool] = []
    for entry in entries:
        response = entry.get("response", "")
        try:
            results = verifier.verify_response(response)
            flags.append(any(r.violation_detected for r in results))
        except Exception as exc:  # noqa: BLE001 — verifier errors are non-fatal
            _log.warning("ILV verify_response() failed: %s", exc)
            flags.append(False)
    return flags


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the interleaved logic verifier diagnostic and emit the result artifact."""
    _SCRIPT_DIR = Path(__file__).resolve().parent
    if str(_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPT_DIR))

    _SCRIPTS_ROOT = _REPO_ROOT / "scripts"
    if str(_SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_ROOT))

    from experiment_template import ExperimentTemplate  # noqa: PLC0415

    _DELIVERABLE = "results/experiment_606_interleaved_logic.json"

    with ExperimentTimeoutWatchdog(
        606,
        timeout_minutes=30,
        result_path=str(_REPO_ROOT / _DELIVERABLE),
    ):
        tmpl = ExperimentTemplate(
            606,
            "Interleaved Formal Logic Verifier",
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

        n_incorrect = len(incorrect_entries)
        n_correct = len(correct_entries)

        # Run ILV on incorrect entries (recall measurement)
        _log.info("Running InterleavedLogicVerifier on %d incorrect entries ...", n_incorrect)
        tp_flags = _run_ilv(incorrect_entries)

        # Run ILV on correct entries (false-positive measurement)
        _log.info("Running InterleavedLogicVerifier on %d correct entries (FP check) ...", n_correct)
        fp_flags = _run_ilv(correct_entries)

        ilv_recall = sum(tp_flags) / n_incorrect if n_incorrect > 0 else 0.0
        ilv_fp_rate = sum(fp_flags) / n_correct if n_correct > 0 else 0.0

        coace_v3_recall_baseline = 0.04
        improvement_over_coace_v3 = ilv_recall > coace_v3_recall_baseline

        if ilv_recall >= 0.20:
            honest_verdict = "ilv_viable"
        elif ilv_recall > coace_v3_recall_baseline:
            honest_verdict = "ilv_improved"
        else:
            honest_verdict = "ilv_no_improvement"

        _log.info(
            "ILV: recall=%.4f  fp_rate=%.4f  verdict=%s",
            ilv_recall, ilv_fp_rate, honest_verdict,
        )

        payload = {
            "result_schema": "carnot.interleaved_logic.v1",
            "n_incorrect": n_incorrect,
            "n_correct": n_correct,
            "ilv_recall": round(ilv_recall, 4),
            "ilv_fp_rate": round(ilv_fp_rate, 4),
            "coace_v3_recall_baseline": coace_v3_recall_baseline,
            "improvement_over_coace_v3": improvement_over_coace_v3,
            "z3_timeout_ms": 50,
            "honest_verdict": honest_verdict,
        }

        artifact = tmpl.build_result(payload, status="success")

        output_path = _REPO_ROOT / _DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=2)

        _log.info("Artifact written to %s", output_path)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
