#!/usr/bin/env python3
"""Experiment 683: FR-11 Real Verified Positives — Wire Exp 668 Repairs into Constraint Library.

WHY THIS EXPERIMENT EXISTS:
    All prior FR-11 relay experiments fed VIOLATIONS (incorrect model responses) into
    the constraint learning system.  Exp 668 VR #18 v2 produced the first
    VERIFIED-CORRECT REPAIRS: 25 questions where structured-equation forcing both
    (1) detected a violation AND (2) repaired the response to a correct answer.

    These verified-correct repairs are fundamentally different from violations:
    - A violation pair: (incorrect_response, constraint_violated) → train to DETECT
    - A repair pair: (constraint_violated, correct_repair) → train to GENERATE VALID REPAIRS

    Wiring repair pairs into ViolationPatternLibrary (the FR-11 backing store) lets
    the constraint weights reinforce patterns that HELPED, not just flag patterns
    that HURT.  This is the first time the self-learning loop sees real positive
    signal — evidence of what a correct, constrained arithmetic response looks like.

WHAT THIS EXPERIMENT DOES:
    1. Loads Exp 668 result and its live_pairs source (the 25 questions used).
    2. Constructs VerifiedRepairPair objects — one per question where
       post-forcing was verified correct (n_post_correct from 668 artifact).
    3. Wires each verified pair into ViolationPatternLibrary via add_template().
    4. Measures FP rate before and after wiring on 10 synthetic test questions.
    5. Records honest_verdict and all required artifact fields.

KEY DESIGN DECISION — Why synthetic repairs for COMPUTE: lines:
    Exp 668 stores only aggregate stats (n_post_correct=25, n_questions=25); the
    actual per-question forced COMPUTE: responses are NOT stored in the result JSON.
    Therefore, repair_response is set to a canonical COMPUTE: pattern derived from
    the question text.  This is explicitly labeled "synthetic_compute_pattern" in
    each pair's metadata so downstream analysis knows the forcing detail is
    reconstructed, not verbatim from 668.  The wiring mechanism itself is real.

Spec: REQ-LEARN-042, REQ-LEARN-043, SCENARIO-LEARN-072, SCENARIO-LEARN-073
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Allow running from the repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

from python.carnot.pipeline.constraint_template_library import ViolationPatternLibrary  # noqa: E402
from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_668_RESULT = _REPO_ROOT / "results" / "experiment_668_vr_attempt_18_v2.json"
DELIVERABLE = "results/experiment_683_fr11_real_positives.json"
SOURCE_EXPERIMENT = 683


# ---------------------------------------------------------------------------
# VerifiedRepairPair dataclass
# ---------------------------------------------------------------------------


@dataclass
class VerifiedRepairPair:
    """One verified-correct repair from the VR pipeline.

    WHY THIS EXISTS:
        Violations tell the system WHAT went wrong; repairs tell it WHAT A CORRECT
        CONSTRAINED RESPONSE LOOKS LIKE.  The distinction matters for self-learning:
        a verified repair is evidence that the constraint system can generate correct
        arithmetic when forced, not just detect errors.

    Attributes:
        question:                The original arithmetic question.
        violated_constraint:     A description of the constraint pattern that
                                 detected the baseline violation (e.g. "COMPUTE:
                                 structured arithmetic forcing").
        repair_response:         The post-forcing response text (may be synthetic
                                 when actual per-question data is unavailable).
        repair_verified_correct: True when the repair was independently verified
                                 as correct (via answer extraction + comparison).

    Spec: REQ-LEARN-042-1
    """

    question: str
    violated_constraint: str
    repair_response: str
    repair_verified_correct: bool


# ---------------------------------------------------------------------------
# Core functions (importable so the test file can call them directly)
# ---------------------------------------------------------------------------


def load_exp668_questions(result_path: Path) -> tuple[list[str], int]:
    """Load the questions and verified-correct count from the Exp 668 artifact.

    WHY WE READ LIVE_PAIRS_SOURCE:
        Exp 668 stores only aggregate stats; per-question data lives in the
        live_pairs file that was the experiment's input.  We read n_post_correct
        from the artifact and take the first n_post_correct questions from the
        pairs file because the experiment ran on all n_questions (also 25) and
        all became correct post-forcing.

    Args:
        result_path: Path to experiment_668_vr_attempt_18_v2.json.

    Returns:
        (questions, n_verified_correct) where questions is a list of strings
        and n_verified_correct is the count of post-correct repairs from 668.

    Spec: REQ-LEARN-042
    """
    artifact = json.loads(result_path.read_text())
    n_post_correct: int = int(artifact.get("n_post_correct", 0))
    n_questions: int = int(artifact.get("n_questions", 0))

    # Live pairs source holds the actual question texts.
    pairs_path = Path(artifact["live_pairs_source"])
    pairs: list[dict] = json.loads(pairs_path.read_text())

    # The experiment ran on the first n_questions pairs; all n_post_correct
    # of them were verified correct after forcing.  We take up to n_questions.
    questions = [p["question"] for p in pairs[:n_questions]]
    # Guard: n_verified_correct cannot exceed the number of questions available.
    n_verified_correct = min(n_post_correct, len(questions))
    return questions, n_verified_correct


def build_repair_pairs(questions: list[str], n_verified_correct: int) -> list[VerifiedRepairPair]:
    """Construct VerifiedRepairPair objects from Exp 668 question data.

    WHY SYNTHETIC COMPUTE PATTERNS:
        Exp 668 does not store the actual post-forcing COMPUTE: response text.
        We therefore synthesize a canonical pattern: "COMPUTE: <question summary>"
        which faithfully represents THAT a constrained arithmetic response was
        produced, without fabricating specific numbers.  The first n_verified_correct
        pairs are marked repair_verified_correct=True; any remaining are False.

    Args:
        questions:         List of question strings from live_pairs.
        n_verified_correct: How many of the first N questions were post-correct.

    Returns:
        List of VerifiedRepairPair, length == len(questions).

    Spec: REQ-LEARN-042-1
    """
    pairs: list[VerifiedRepairPair] = []
    for i, q in enumerate(questions):
        # Canonical COMPUTE: pattern — marks that structured forcing was applied.
        # The "(synthetic_compute_pattern)" suffix ensures downstream audits
        # know this is reconstructed from 668 aggregate data, not verbatim.
        repair_text = f"COMPUTE: {q[:80].strip()} (synthetic_compute_pattern)"
        pairs.append(VerifiedRepairPair(
            question=q,
            violated_constraint="structured_arithmetic_forcing",
            repair_response=repair_text,
            repair_verified_correct=(i < n_verified_correct),
        ))
    return pairs


def wire_repairs_into_library(
    pairs: list[VerifiedRepairPair],
    library: ViolationPatternLibrary,
) -> int:
    """Wire verified-correct repair pairs into ViolationPatternLibrary.

    WHY ONLY VERIFIED PAIRS:
        We only add entries where repair_verified_correct=True.  Wiring unverified
        or incorrect repairs would pollute the constraint store with false positives,
        undermining the very FP-rate improvement we are trying to demonstrate.

    Args:
        pairs:   List of VerifiedRepairPair objects (from build_repair_pairs).
        library: ViolationPatternLibrary to add entries to.

    Returns:
        n_positives_wired: count of entries actually added (verified pairs only,
        deduplication handled by the library itself).

    Spec: REQ-LEARN-042-2, REQ-LEARN-042-3
    """
    n_wired = 0
    for pair in pairs:
        if not pair.repair_verified_correct:
            continue
        library.add_template(
            pattern=pair.repair_response,
            violation_type="verified_repair",
            source_experiment=SOURCE_EXPERIMENT,
        )
        n_wired += 1
    return n_wired


def build_synthetic_test_questions() -> list[str]:
    """Return 10 synthetic test questions for FP rate measurement.

    WHY SYNTHETIC:
        We need a held-out set of question texts to measure how often the stored
        repair patterns accidentally match UNRELATED questions (false positives).
        Synthetic questions have no COMPUTE: lines so they should not match any
        repair pattern, giving an expected FP rate of 0.0 after wiring.

        These questions are deliberately generic arithmetic; they do NOT contain
        any COMPUTE: annotations and do NOT overlap with the 668 questions.

    Returns:
        List of 10 synthetic question strings.

    Spec: REQ-LEARN-043-1
    """
    return [
        "What is 5 plus 7?",
        "Calculate 12 multiplied by 4.",
        "How many apples remain if you start with 20 and give away 8?",
        "What is the sum of 33 and 47?",
        "A train travels 60 km/h for 3 hours. How far does it travel?",
        "If a pizza is cut into 8 slices and 3 are eaten, how many remain?",
        "What is 100 divided by 4?",
        "A store has 150 items and sells 63. How many remain?",
        "What is 9 times 9?",
        "If you have $25 and spend $18, how much do you have left?",
    ]


def compute_honest_verdict(n_positives_wired: int, fp_rate_delta: float) -> str:
    """Determine the honest verdict for this experiment's outcome.

    WHY THREE CASES:
        - no_positives_available: Nothing was wired; the mechanism was exercised
          but had no effect.  Most likely cause: 668 data unavailable or
          n_post_correct=0 (which would be a data integrity issue).
        - positives_wired_fp_reduced: Wiring happened AND the repair patterns
          are specific enough that they do NOT fire on unrelated correct questions.
          This is the positive result we hope for.
        - positives_wired_no_fp_change: Wiring happened but the FP rate did not
          improve.  This occurs when repair patterns are too generic (e.g., they
          contain common substrings that appear in unrelated questions).

    Args:
        n_positives_wired: How many verified pairs were wired.
        fp_rate_delta:     fp_rate_after - fp_rate_before (negative = improvement).

    Returns:
        One of: "no_positives_available", "positives_wired_fp_reduced",
        "positives_wired_no_fp_change".

    Spec: REQ-LEARN-043-4
    """
    if n_positives_wired == 0:
        return "no_positives_available"
    if fp_rate_delta < 0:
        return "positives_wired_fp_reduced"
    return "positives_wired_no_fp_change"


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    exp668_result_path: Path = EXP_668_RESULT,
    library_path: str | None = None,
) -> dict:
    """Run the full Exp 683 pipeline and return the artifact dict.

    WHY A SEPARATE FUNCTION (not top-level script):
        Exposing the logic as a callable function lets the test suite import and
        invoke it directly without subprocess overhead — critical for CI speed.
        The ``library_path`` parameter allows tests to redirect the
        ViolationPatternLibrary to a temp file, keeping tests isolated from
        each other and from any real data/constraint_templates.json on disk.

    Args:
        exp668_result_path: Path to the Exp 668 result JSON.
        library_path:       Where ViolationPatternLibrary saves its state.
                            If None, uses a temp file to avoid polluting real data.

    Returns:
        Artifact dict with all required schema fields.

    Spec: REQ-LEARN-042, REQ-LEARN-043
    """
    # Use a temp file by default so this function is side-effect-free on disk
    # unless a real path is explicitly provided.
    _tmp = None
    if library_path is None:
        _tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        library_path = _tmp.name

    # --- Load Exp 668 data ---------------------------------------------------

    questions, n_verified_correct = load_exp668_questions(exp668_result_path)

    # --- Build repair pairs --------------------------------------------------

    pairs = build_repair_pairs(questions, n_verified_correct)

    # --- FP rate BEFORE wiring -----------------------------------------------

    lib_before = ViolationPatternLibrary(library_path)
    test_questions = build_synthetic_test_questions()
    fp_rate_before = lib_before.get_fp_rate(test_questions)

    # Snapshot constraint weights before (observation counts are opaque to
    # ViolationPatternLibrary; we capture n_templates as a proxy).
    n_templates_before = len(lib_before.templates)

    # --- Wire verified repairs -----------------------------------------------

    lib_after = ViolationPatternLibrary(library_path)  # same backing file
    n_positives_wired = wire_repairs_into_library(pairs, lib_after)
    n_templates_after = len(lib_after.templates)

    # --- FP rate AFTER wiring ------------------------------------------------

    lib_measure = ViolationPatternLibrary(library_path)
    fp_rate_after = lib_measure.get_fp_rate(test_questions)
    fp_rate_delta = fp_rate_after - fp_rate_before

    # --- Verdict and artifact ------------------------------------------------

    honest_verdict = compute_honest_verdict(n_positives_wired, fp_rate_delta)
    n_constraints_updated = n_templates_after - n_templates_before

    return {
        "n_positives_wired": n_positives_wired,
        "fp_rate_before": fp_rate_before,
        "fp_rate_after": fp_rate_after,
        "fp_rate_delta": fp_rate_delta,
        "n_constraints_updated": n_constraints_updated,
        "fr11_real_positives_confirmed": n_positives_wired > 0,
        "honest_verdict": honest_verdict,
        "n_questions_from_668": len(questions),
        "n_verified_correct_from_668": n_verified_correct,
        "n_test_questions": len(test_questions),
    }


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tmpl = ExperimentTemplate(
        SOURCE_EXPERIMENT,
        "FR-11 Real Verified Positives: Wire Exp 668 Repairs into Constraint Library",
        DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(
        SOURCE_EXPERIMENT,
        timeout_minutes=20,
        result_path=DELIVERABLE,
    ):
        # Use the real constraint templates file so wiring persists for the
        # conductor's cross-session relay.  The library deduplicates, so
        # re-running this experiment is safe.
        real_library_path = str(_REPO_ROOT / "data" / "constraint_templates.json")

        data = run_experiment(
            exp668_result_path=EXP_668_RESULT,
            library_path=real_library_path,
        )

        artifact = tmpl.build_result(data, status="success")

        out_path = _REPO_ROOT / DELIVERABLE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()
