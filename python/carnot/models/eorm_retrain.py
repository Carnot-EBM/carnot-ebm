"""EORM real-data retrain: load live CoT pairs, merge with synthetic, retrain, compare AUC-ROC.

**Researcher summary:**
    Exp 346 trained EORM entirely on synthetic (question, correct_response, incorrect_response)
    pairs because the live GPU experiments (Exp 340, 341, 355) returned simulated results with no
    real model responses. This module provides the data pipeline to retrain EORM whenever real
    pairs become available, while gracefully falling back to the synthetic corpus when they are not.

**Why real data matters:**
    Synthetic training pairs are generated from template text — they do not capture the
    characteristic mistakes that real LLMs make (unit confusion, off-by-one arithmetic, missing
    edge cases). Retraining on (question, real_response, correctness_label) triples teaches EORM
    to recognize the specific failure modes of the target model, improving AUC-ROC on real
    benchmarks rather than just on held-out synthetic data.

**Data pipeline:**
    1. ``load_real_cot_pairs`` reads each experiment result JSON and extracts any entries that
       have a ``response`` (or ``generated_code``) field and a ``correct`` (or ``passed_tests``)
       boolean. Missing files, missing keys, empty lists — all handled gracefully.

    2. ``merge_cot_corpora`` combines real and synthetic pairs, capping each source to avoid
       over-representation. Real pairs always come first so the model sees them in every epoch.

    3. ``EORMRetrainResult`` carries the before/after AUC-ROC and the honest verdict about
       what kind of data was available.

    4. ``build_retrain_artifact`` formats the result for the experiment JSON output, including
       an ``honest_verdict`` field that distinguishes genuine real-data improvement from
       synthetic-only runs so downstream consumers never confuse the two.

Spec: REQ-LEARN-025, SCENARIO-LEARN-043, SCENARIO-LEARN-044
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from carnot.embeddings.jepa_retrain import ViolationPair, _make_synthetic_pairs


# ---------------------------------------------------------------------------
# load_real_cot_pairs
# ---------------------------------------------------------------------------


def load_real_cot_pairs(result_files: list[str]) -> list[ViolationPair]:
    """Load (question, response, correctness) triples from experiment JSON files.

    **Researcher summary:**
        Iterates over a list of experiment result file paths, extracts any entries
        that have both a text response and a correctness label, and returns them as
        ``ViolationPair`` objects. All error conditions (missing file, missing JSON
        key, empty list, wrong schema) are handled by skipping silently — the caller
        should check the returned list length to decide whether real data is available.

    **Supported schemas:**
        The function recognises two result layouts produced by Carnot experiments:

        Layout A — Exp 340/355 (GSM8K-style, ``responses`` top-level key)::

            {
              "responses": [
                {"question_id": "q001", "model_id": "gemma4", "response": "...", "correct": true},
                ...
              ]
            }

        Layout B — Exp 341 (HumanEval-style, ``per_problem_results`` top-level key)::

            {
              "per_problem_results": [
                {"problem_id": "HumanEval/0", "generated_code": "...", "passed_tests": true},
                ...
              ]
            }

        For Layout A:
        - ``response`` → ``full_response`` (also ``partial_response`` — full text used as proxy)
        - ``correct`` → ``has_violation = not correct``
        - ``question_id`` and ``model_id`` are preserved if present

        For Layout B:
        - ``generated_code`` → ``full_response``; ``problem_id`` → ``question_id``
        - ``passed_tests`` → ``has_violation = not passed_tests``
        - ``model_id`` defaults to ``"humaneval_unknown"``

    **Why use the full response as both partial and full?**
        The EORM model uses the *full* (question, response) pair for energy scoring —
        it does not split at a prefix fraction. ViolationPair was designed for the JEPA
        prefix-prediction task. We reuse it here for EORM because the full response is
        also a valid "partial" from EORM's perspective: the model scores the complete text.

    Args:
        result_files: List of absolute or relative file paths to experiment JSON files.
            Paths that do not exist are skipped without error.

    Returns:
        Flat list of ``ViolationPair`` objects extracted from all valid files.
        May be empty if no valid entries are found.

    Spec: REQ-LEARN-025-1, SCENARIO-LEARN-043
    """
    pairs: list[ViolationPair] = []

    for fpath in result_files:
        try:
            path = Path(fpath)
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        # Layout A: top-level "responses" list (Exp 340, 355 GSM8K-style)
        responses = data.get("responses")
        if isinstance(responses, list):
            for entry in responses:
                if not isinstance(entry, dict):
                    continue
                response_text: str = str(entry.get("response") or "")
                if not response_text:
                    continue
                correct: bool = bool(entry.get("correct", False))
                pairs.append(
                    ViolationPair(
                        partial_response=response_text,  # full text is the "prefix" for EORM
                        full_response=response_text,
                        has_violation=not correct,
                        model_id=str(entry.get("model_id") or "unknown"),
                        question_id=str(entry.get("question_id") or "unknown"),
                    )
                )

        # Layout B: top-level "per_problem_results" list (Exp 341 HumanEval-style)
        per_problem = data.get("per_problem_results")
        if isinstance(per_problem, list):
            for entry in per_problem:
                if not isinstance(entry, dict):
                    continue
                code_text: str = str(entry.get("generated_code") or "")
                if not code_text:
                    continue
                passed: bool = bool(entry.get("passed_tests", False))
                pairs.append(
                    ViolationPair(
                        partial_response=code_text,
                        full_response=code_text,
                        has_violation=not passed,
                        model_id="humaneval_unknown",
                        question_id=str(entry.get("problem_id") or "unknown"),
                    )
                )

    return pairs


# ---------------------------------------------------------------------------
# merge_cot_corpora
# ---------------------------------------------------------------------------


def merge_cot_corpora(
    real_pairs: list[ViolationPair],
    synthetic_pairs: list[ViolationPair],
    max_real: int = 300,
    max_synthetic: int = 100,
) -> list[ViolationPair]:
    """Merge real and synthetic ViolationPair corpora, preferring real pairs.

    **Researcher summary:**
        Combines real (live GPU) pairs with synthetic fallback pairs, capping each
        source. Real pairs always come first so that early-stopping or small-epoch
        training always sees real data first. Synthetic pairs are used only to fill
        out the corpus when real pairs are scarce.

    **Why cap both sources?**
        Too many synthetic pairs would dominate the gradient signal and negate the
        benefit of real data. Too many real pairs might overfit if the real corpus
        is small and biased. The caps ``max_real=300`` and ``max_synthetic=100``
        preserve a 3:1 real-to-synthetic ratio by default, which was chosen to give
        synthetic examples a regularizing role without swamping real signal.

    Args:
        real_pairs: Pairs loaded from live experiment result files.
        synthetic_pairs: Pairs from the Exp 346 synthetic corpus or ``_make_synthetic_pairs``.
        max_real: Maximum number of real pairs to include. Default 300.
        max_synthetic: Maximum number of synthetic pairs to include. Default 100.

    Returns:
        Merged list: first up to ``max_real`` real pairs, then up to ``max_synthetic``
        synthetic pairs. The total length is at most ``max_real + max_synthetic``.

    Spec: REQ-LEARN-025-2, SCENARIO-LEARN-044
    """
    selected_real = real_pairs[:max_real]
    selected_synthetic = synthetic_pairs[:max_synthetic]
    return list(selected_real) + list(selected_synthetic)


# ---------------------------------------------------------------------------
# EORMRetrainResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class EORMRetrainResult:
    """Summary of one EORM real-data retrain run.

    **For engineers:**
        Carries the key metrics from an EORM retrain experiment so they can be
        serialised into the experiment JSON artifact and inspected by downstream
        analysis.

        ``retrain_mode`` is the honest declaration of what data was used:
        - ``"real_data"``: At least 50 real (question, response, correctness) pairs
          were available and used for training.
        - ``"synthetic_only"``: Fewer than 50 real pairs were found; the run trained
          entirely on synthetic data. AUC improvement in this mode does not indicate
          real-world progress.

        ``auc_improvement`` is signed: positive = retrain helped, negative = retrain
        hurt. Both outcomes are reported because the honest negative result is still
        scientifically useful.

    Attributes:
        n_real_pairs: Number of real pairs used in training.
        n_synthetic_pairs: Number of synthetic pairs used in training.
        before_auc: AUC-ROC on the 20% test split BEFORE retraining.
        after_auc: AUC-ROC on the 20% test split AFTER retraining.
        auc_improvement: ``after_auc - before_auc`` (signed float).
        retrain_mode: ``"real_data"`` or ``"synthetic_only"``.
        model_path: Path to the saved retrained model safetensors file.

    Spec: REQ-LEARN-025-3
    """

    n_real_pairs: int
    n_synthetic_pairs: int
    before_auc: float
    after_auc: float
    auc_improvement: float
    retrain_mode: str
    model_path: str


# ---------------------------------------------------------------------------
# build_retrain_artifact
# ---------------------------------------------------------------------------


def build_retrain_artifact(result: EORMRetrainResult) -> dict:
    """Build a summary artifact dict for the EORM real-data retrain experiment.

    **For engineers:**
        This function converts an ``EORMRetrainResult`` into the standard Carnot
        experiment artifact format. The ``honest_verdict`` field is critical:

        - ``"real_data_improvement"``: Real data was available AND AUC improved.
          This is the only outcome that constitutes genuine evidence of real-world
          EORM improvement.

        - ``"real_data_no_improvement"``: Real data was available but AUC did not
          improve. Logged honestly for future analysis (e.g., more data needed,
          learning rate tuning required).

        - ``"synthetic_only"``: No sufficient real data was found. The retrain ran
          on synthetic pairs only. AUC numbers are reported but labeled clearly so
          they are never mistaken for live GPU results.

        All ``auc_*`` values are rounded to 6 decimal places to keep the artifact
        JSON compact without losing meaningful precision.

    Args:
        result: An ``EORMRetrainResult`` populated by the experiment script.

    Returns:
        Dict suitable for merging into an ExperimentTemplate artifact.
        Keys: ``schema``, ``retrain_mode``, ``n_real_pairs``, ``n_synthetic_pairs``,
        ``before_auc``, ``after_auc``, ``auc_improvement``, ``honest_verdict``.

    Spec: REQ-LEARN-025-4
    """
    # Determine honest verdict
    if result.retrain_mode == "real_data":
        if result.auc_improvement > 0:
            honest_verdict = "real_data_improvement"
        else:
            honest_verdict = "real_data_no_improvement"
    else:
        honest_verdict = "synthetic_only"

    return {
        "schema": "carnot.eorm_retrain.v1",
        "retrain_mode": result.retrain_mode,
        "n_real_pairs": int(result.n_real_pairs),
        "n_synthetic_pairs": int(result.n_synthetic_pairs),
        "before_auc": round(float(result.before_auc), 6),
        "after_auc": round(float(result.after_auc), 6),
        "auc_improvement": round(float(result.auc_improvement), 6),
        "honest_verdict": honest_verdict,
        "model_path": str(result.model_path),
    }


# ---------------------------------------------------------------------------
# make_synthetic_eorm_pairs (re-exported for experiment use)
# ---------------------------------------------------------------------------


def make_synthetic_eorm_pairs(n: int = 100, seed: int = 359) -> list[ViolationPair]:
    """Generate deterministic synthetic ViolationPairs for EORM retrain fallback.

    **For engineers:**
        Wraps ``_make_synthetic_pairs`` from jepa_retrain with a different default
        seed (359, matching the experiment number) to distinguish the EORM synthetic
        corpus from the JEPA synthetic corpus (seed 42). This makes it easy to tell
        in debugging which corpus a pair came from.

        Used by Exp 359 when real data is insufficient (fewer than 50 real pairs).

    Args:
        n: Number of pairs. Default 100 (covers max_synthetic cap with headroom).
        seed: Random seed for determinism. Default 359.

    Returns:
        List of exactly ``n`` ViolationPair objects.

    Spec: REQ-LEARN-025-1
    """
    return _make_synthetic_pairs(n=n, seed=seed)
