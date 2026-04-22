"""Machine-checkable invariants for experiment result artifacts.

**Why this module exists:**

The honest-verdict enum (REQ-SAFE-009, REQ-VERIFY-* etc.) constrains the *shape*
of a result's ``honest_verdict`` field — it picks from a fixed vocabulary — but
it does not constrain the *truth* of that verdict.  In milestones 2026.04.51-.52
we saw three cases where the enum let through a verdict whose string was
technically valid but whose data contradicted the claim:

1. **Exp 652 / 669 / old-678:** emitted ``distillation_*`` verdicts in runs
   with ``duration_s`` = 17-30 seconds.  A 20B GGUF cannot infer 200-2000
   prompts in under 30 seconds; the teacher did not actually run.  The first
   real teacher inference (Exp 690) took 6256 seconds on 200 prompts.

2. **Exp 691:** emitted ``generalization_verified_publishable`` with
   ``mean_auroc=0.9585`` across three datasets — but the confusion matrices
   at threshold 0.5 were ``tp=0, fp=0, tn=N, fn=N`` on every dataset.  The
   classifier detected zero injections in practice.  AUROC-without-TP was
   insufficient gate logic.

3. **Exp 679:** emitted ``vr_200q_positive`` with ``baseline_accuracy=0.0``
   and ``post_accuracy=1.0``.  A 0/200 baseline on Qwen3.5-0.8B is physically
   implausible — the model normally scores 25-45% on GSM8K without any
   special prompting.  The "baseline" code path was silently broken.

4. **Exp 691 vs 690:** training-distribution AUROC = 0.7995, held-out
   cross-dataset AUROC = 0.9585.  Real ML generalization always degrades
   OOD; a model that scores 15 AUROC points HIGHER on truly-held-out data
   is measuring something other than the target capability.

**What this module does:**

Provides four invariant checks covering these four patterns.  Each invariant
takes a result artifact dict (loaded from a ``results/experiment_*.json``)
and returns an :class:`InvariantResult` indicating pass/fail, a reason string
on fail, and a suggested substitute verdict the caller can use to re-write
the artifact honestly.

Usage — post-experiment hook (recommended):

    from carnot.invariants import run_invariants
    artifact = tmpl.build_result(...)
    violations = run_invariants(artifact)
    if violations:
        # rewrite verdict to reflect the violation honestly
        artifact['honest_verdict'] = violations[0].suggested_verdict
        artifact['invariant_violations'] = [v.as_dict() for v in violations]

Usage — CLI at milestone retro:

    .venv/bin/python scripts/check_invariants.py results/experiment_*.json

Adding a new invariant: write a function with the :class:`InvariantCheck`
signature, register it in :data:`_INVARIANTS`.  Keep the predicate narrow
so it only triggers on the verdict strings it's meant to gate — false
positives on benign runs burn operator attention and erode trust in the
system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class InvariantResult:
    """Outcome of one invariant check against one artifact.

    Attributes:
        passed: True if the invariant holds OR is not applicable.  False
            only when the invariant SHOULD hold but does not.
        invariant_name: Stable identifier (used in result JSON).
        reason: One-sentence explanation when ``passed`` is False.
        suggested_verdict: A string the caller can use to rewrite the
            artifact's ``honest_verdict`` field to reflect the violation
            honestly.  None when ``passed`` is True.
        evidence: Key numeric fields cited in the failure reason, so the
            violation artifact is self-contained and auditable.
    """

    passed: bool
    invariant_name: str
    reason: str | None = None
    suggested_verdict: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "invariant_name": self.invariant_name,
            "reason": self.reason,
            "suggested_verdict": self.suggested_verdict,
            "evidence": self.evidence,
        }


InvariantCheck = Callable[[dict[str, Any]], InvariantResult]


# ---------------------------------------------------------------------------
# Invariant 1: distillation claims require real teacher inference duration
# ---------------------------------------------------------------------------


# Per-prompt floor (seconds) for a real teacher-model inference on a 20B+
# GGUF.  Observed empirically: Exp 690 averaged 31.28 s/prompt on CPU for
# gpt-oss-safeguard-20b-Q4_K_M.  GPU can push this to 1-5 s/prompt.  The
# 0.5 s floor is a LOWER bound: anything faster than 0.5 s/prompt on a 20B
# model is physically implausible without external caching.
_TEACHER_INFERENCE_FLOOR_PER_PROMPT_S = 0.5


def check_distillation_has_real_teacher_time(artifact: dict[str, Any]) -> InvariantResult:
    """Reject ``distillation_*`` verdicts whose total teacher time is too low.

    Invariant: if ``honest_verdict`` contains the substring ``distillation_``
    AND does not indicate a blocked / invariant-violated run, then
    ``teacher_inference_duration_s`` must be present and at least
    ``max(corpus_size * 0.5, 100)`` seconds.

    Would have caught Exps 652, 669, and the original 678 (all emitted
    ``distillation_*`` with durations under 30 seconds for 200-2000 prompts).
    """
    name = "distillation_has_real_teacher_time"
    verdict = str(artifact.get("honest_verdict", ""))

    # Only applies to verdicts that claim distillation happened.
    if "distillation_" not in verdict:
        return InvariantResult(passed=True, invariant_name=name)
    if any(marker in verdict for marker in (
        "blocked_on", "invariant_violated", "corpus_not_built",
    )):
        return InvariantResult(passed=True, invariant_name=name)

    corpus_size = _resolve_corpus_size(artifact)
    floor_s = max(corpus_size * _TEACHER_INFERENCE_FLOOR_PER_PROMPT_S, 100.0)
    teacher_s = artifact.get("teacher_inference_duration_s")

    if teacher_s is None:
        return InvariantResult(
            passed=False, invariant_name=name,
            reason=(
                "Verdict claims distillation but 'teacher_inference_duration_s' "
                "is missing from the artifact.  Every real distillation run must "
                "record how long the teacher model actually ran."
            ),
            suggested_verdict=_with_invariant_violation(
                verdict, "distillation_invariant_violated_no_teacher_time_field",
            ),
            evidence={"corpus_size": corpus_size, "floor_s": floor_s},
        )
    if teacher_s < floor_s:
        return InvariantResult(
            passed=False, invariant_name=name,
            reason=(
                f"Verdict claims distillation but teacher_inference_duration_s="
                f"{teacher_s:.1f}s is below the physical-plausibility floor of "
                f"max(corpus_size*0.5, 100) = {floor_s:.1f}s for corpus_size="
                f"{corpus_size}.  Teacher model inference cannot run this fast; "
                f"labels almost certainly came from a non-teacher source."
            ),
            suggested_verdict=_with_invariant_violation(
                verdict, "distillation_invariant_violated_teacher_too_fast",
            ),
            evidence={
                "teacher_inference_duration_s": teacher_s,
                "corpus_size": corpus_size,
                "floor_s": floor_s,
            },
        )
    return InvariantResult(passed=True, invariant_name=name)


# ---------------------------------------------------------------------------
# Invariant 2: "verified_publishable" requires non-zero detections
# ---------------------------------------------------------------------------


def check_publishable_has_nonzero_tp(artifact: dict[str, Any]) -> InvariantResult:
    """Reject ``*verified_publishable`` verdicts with zero true positives.

    Invariant: if ``honest_verdict`` contains ``verified_publishable``, then
    the artifact's confusion matrices (field ``per_dataset_cm``) must have
    at least one true positive across all datasets — i.e. the classifier
    actually fires on at least one positive example at its recommended
    threshold.

    Would have caught Exp 691 (AUROC 0.96 but TP=0 on all three datasets).
    """
    name = "publishable_has_nonzero_tp"
    verdict = str(artifact.get("honest_verdict", ""))
    if "verified_publishable" not in verdict:
        return InvariantResult(passed=True, invariant_name=name)
    if any(marker in verdict for marker in (
        "blocked_on", "invariant_violated",
    )):
        return InvariantResult(passed=True, invariant_name=name)

    per_ds_cm = artifact.get("per_dataset_cm")
    if per_ds_cm is None:
        return InvariantResult(
            passed=False, invariant_name=name,
            reason=(
                "Verdict claims publishable but per_dataset_cm (confusion "
                "matrices) is missing.  Publishing a classifier requires "
                "evidence it actually fires at its threshold."
            ),
            suggested_verdict=_with_invariant_violation(
                verdict, "publishable_invariant_violated_no_confusion_matrix",
            ),
        )

    total_tp = 0
    per_ds_tp: dict[str, int] = {}
    for ds_name, cm in per_ds_cm.items():
        tp = int(cm.get("tp", 0)) if isinstance(cm, dict) else 0
        per_ds_tp[ds_name] = tp
        total_tp += tp
    if total_tp == 0:
        return InvariantResult(
            passed=False, invariant_name=name,
            reason=(
                f"Verdict claims publishable but classifier has zero true "
                f"positives across all datasets: {per_ds_tp}.  At the decision "
                f"threshold used, the model flags nothing as the positive "
                f"class.  AUROC may be high but the classifier is unusable "
                f"without threshold calibration."
            ),
            suggested_verdict=_with_invariant_violation(
                verdict, "publishable_invariant_violated_zero_true_positives",
            ),
            evidence={"per_dataset_tp": per_ds_tp, "total_tp": total_tp},
        )
    return InvariantResult(passed=True, invariant_name=name)


# ---------------------------------------------------------------------------
# Invariant 3: OOD AUROC should not dramatically beat in-distribution AUROC
# ---------------------------------------------------------------------------


# Tolerance: we allow OOD AUROC to exceed training-distribution AUROC by up
# to this much (due to corpus-selection noise).  Larger gaps indicate the
# OOD datasets share an artifact the model latched onto.
_OOD_EXCESS_TOLERANCE = 0.05


def check_ood_not_dramatically_better_than_indist(
    artifact: dict[str, Any],
) -> InvariantResult:
    """Reject runs where OOD AUROC exceeds in-distribution AUROC by > 0.05.

    Invariant: real ML generalization always degrades on truly held-out data.
    If ``mean_auroc`` (or ``mean_cross_dataset_auroc``) on held-out datasets
    exceeds ``training_distribution_auroc`` by more than 0.05, the OOD
    datasets almost certainly share a spurious signal the model learned.

    Would have caught Exp 691 (OOD 0.96 vs in-dist 0.80 = +0.16 excess).
    """
    name = "ood_not_dramatically_better_than_indist"
    indist = artifact.get("training_distribution_auroc")
    ood = (
        artifact.get("mean_cross_dataset_auroc")
        or artifact.get("mean_auroc")
    )
    if indist is None or ood is None:
        # Invariant does not apply when either field is missing.
        return InvariantResult(passed=True, invariant_name=name)

    excess = ood - indist
    if excess > _OOD_EXCESS_TOLERANCE:
        verdict = str(artifact.get("honest_verdict", ""))
        return InvariantResult(
            passed=False, invariant_name=name,
            reason=(
                f"Cross-dataset AUROC ({ood:.4f}) exceeds training-distribution "
                f"AUROC ({indist:.4f}) by {excess:+.4f}, which is "
                f"physically implausible in real ML — generalization should "
                f"degrade (or at most stay flat) on truly-held-out data.  "
                f"The OOD datasets likely share a spurious artifact the "
                f"model latched onto."
            ),
            suggested_verdict=_with_invariant_violation(
                verdict, "generalization_invariant_violated_ood_exceeds_indist",
            ),
            evidence={
                "training_distribution_auroc": indist,
                "cross_dataset_auroc": ood,
                "excess": excess,
                "tolerance": _OOD_EXCESS_TOLERANCE,
            },
        )
    return InvariantResult(passed=True, invariant_name=name)


# ---------------------------------------------------------------------------
# Invariant 4: VR "positive" verdicts require a plausible baseline
# ---------------------------------------------------------------------------


# Minimum baseline accuracy expected from any modern LLM on GSM8K-class
# tasks without special prompting.  Qwen3.5-0.8B scores 25-45% on GSM8K;
# Gemma-4-E4B-it scores similarly.  A baseline below 0.05 indicates the
# baseline code path is broken (wrong prompt template, grader regex
# mismatch, or silent short-circuit), not that the model can't do math.
_VR_BASELINE_FLOOR = 0.05


def check_vr_positive_has_plausible_baseline(
    artifact: dict[str, Any],
) -> InvariantResult:
    """Reject ``vr_*_positive`` verdicts with implausibly-low baselines.

    Invariant: if ``honest_verdict`` contains ``vr_`` and ``_positive``,
    then ``baseline_accuracy`` must be >= 0.05.  A 0/N baseline on a
    modern LLM is a broken baseline path, not a reasoning improvement.

    Would have caught Exp 679 (baseline=0.0 on 200 GSM8K questions).
    """
    name = "vr_positive_has_plausible_baseline"
    verdict = str(artifact.get("honest_verdict", ""))
    # Trigger on any verdict that asserts a positive VR outcome.
    triggers = ("vr_positive", "_positive")
    is_vr_verdict = "vr" in verdict.lower()
    asserts_positive = any(t in verdict for t in triggers)
    if not (is_vr_verdict and asserts_positive):
        return InvariantResult(passed=True, invariant_name=name)
    if any(marker in verdict for marker in (
        "blocked_on", "invariant_violated",
    )):
        return InvariantResult(passed=True, invariant_name=name)

    baseline = artifact.get("baseline_accuracy")
    if baseline is None:
        # Invariant doesn't fire without data — but the conductor should
        # probably require this field separately.
        return InvariantResult(passed=True, invariant_name=name)

    if baseline < _VR_BASELINE_FLOOR:
        n_questions = artifact.get("n_questions") or artifact.get("n_test")
        return InvariantResult(
            passed=False, invariant_name=name,
            reason=(
                f"VR-positive verdict with baseline_accuracy={baseline:.3f} "
                f"is implausible — modern LLMs score at least "
                f"{_VR_BASELINE_FLOOR*100:.0f}% on GSM8K-class tasks without "
                f"special prompting.  A near-zero baseline indicates the "
                f"baseline code path is broken (wrong prompt template, "
                f"grader regex mismatch, or silent short-circuit), not "
                f"that the repair improved reasoning."
            ),
            suggested_verdict=_with_invariant_violation(
                verdict, "vr_invariant_violated_baseline_implausibly_low",
            ),
            evidence={
                "baseline_accuracy": baseline,
                "floor": _VR_BASELINE_FLOOR,
                "n_questions": n_questions,
            },
        )
    return InvariantResult(passed=True, invariant_name=name)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


_INVARIANTS: list[InvariantCheck] = [
    check_distillation_has_real_teacher_time,
    check_publishable_has_nonzero_tp,
    check_ood_not_dramatically_better_than_indist,
    check_vr_positive_has_plausible_baseline,
]


def run_invariants(artifact: dict[str, Any]) -> list[InvariantResult]:
    """Run all registered invariants against an artifact.

    Returns only the FAILED invariants (empty list when everything passes).
    Callers that want the full pass/fail grid should use :data:`_INVARIANTS`
    and iterate themselves.
    """
    violations: list[InvariantResult] = []
    for inv in _INVARIANTS:
        result = inv(artifact)
        if not result.passed:
            violations.append(result)
    return violations


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_corpus_size(artifact: dict[str, Any]) -> int:
    """Resolve a corpus-size integer from whatever field the experiment used.

    Different experiments use different field names for "how many examples
    were in the corpus".  Prefer the most specific (n_train + n_test), fall
    back to less specific (n_pairs, n_questions), return 0 if nothing known.
    """
    n_train = artifact.get("n_train")
    n_test = artifact.get("n_test")
    if isinstance(n_train, int) and isinstance(n_test, int):
        return n_train + n_test
    for fallback_field in ("n_pairs", "n_questions", "corpus_size", "n_labeled"):
        val = artifact.get(fallback_field)
        if isinstance(val, int) and val > 0:
            return val
    return 0


def _with_invariant_violation(original_verdict: str, substitute: str) -> str:
    """Compute a suggested replacement verdict for a failed invariant.

    The substitute is guaranteed to contain the marker ``invariant_violated``
    so later runs of the same invariants treat it as already-flagged and
    do not re-flag it.
    """
    del original_verdict  # kept for future use (e.g. preserve decision_class prefix)
    assert "invariant_violated" in substitute, (
        "Substitute verdicts must contain 'invariant_violated' so re-runs "
        "of the invariant system do not double-flag the same artifact."
    )
    return substitute
