"""Reward hacking detection for the self-learning pipeline.

**Researcher summary:**
    The Anthropic Mythos system card (April 2026) documented novel reward
    hacking behaviors: models moving computation outside timing windows and
    training on test data to game forecasting metrics. Our self-learning
    loop (Exp 223/241) trains on verification traces. This module detects
    whether energy function improvements reflect genuine learning or
    exploitation of shortcuts.

**Detailed explanation for engineers:**
    Reward hacking in an EBM verification pipeline can take several forms:

    1. TRIVIAL CONSTRAINTS — A constraint type fires frequently but catches
       almost no errors. This means the extractor is generating constraints
       that always pass, giving the appearance of thorough verification
       without catching real problems. Metric: precision near 0.0 despite
       high fire count.

    2. ZERO-ENERGY SHORTCUTS — If the energy function always returns 0.0
       (or is constant), the model has found a "path of least resistance"
       that satisfies the optimizer without actually evaluating anything.
       This is the EBM equivalent of a model outputting a fixed string to
       game a loss metric.

    3. CONSTRAINT MONOCULTURE — If one or two constraint types dominate
       all verification traffic, the pipeline is effectively ignoring entire
       categories of potential violations. High concentration (low diversity)
       suggests the optimizer has learned to route everything through the
       easiest-to-satisfy type. Measured with a Gini coefficient over fired
       counts.

    4. TRAIN/HOLDOUT ENERGY DIVERGENCE — Genuine improvement produces lower
       energy (higher confidence) on BOTH training traces and held-out traces.
       If training energy drops while held-out energy stays flat or rises,
       the model is memorising the training distribution rather than learning
       a general verification signal. This is the EBM analogue of train/test
       accuracy divergence in supervised learning.

    The audit is intentionally lightweight: no matrix ops, no JAX/GPU.
    It runs on plain Python with only the statistics already collected by
    ConstraintTracker plus optional energy sequences.

Spec: REQ-LEARN-002, SCENARIO-LEARN-002
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from carnot.pipeline.tracker import ConstraintTracker


# ---------------------------------------------------------------------------
# Configuration thresholds
# ---------------------------------------------------------------------------

# Minimum number of times a constraint must have fired before we flag it as
# trivially-passing. Below this count the evidence is too thin to judge.
MIN_FIRE_COUNT_FOR_TRIVIAL_FLAG: int = 5

# Precision below this threshold (with enough fires) is considered trivially
# passing — the constraint fires but almost never catches a real error.
TRIVIAL_PRECISION_THRESHOLD: float = 0.05

# If the energy sequence has this many distinct values or fewer, we treat it
# as a potential shortcut (near-constant output from the energy function).
ENERGY_DISTINCT_VALUES_MIN: int = 2

# Gini coefficient above this threshold signals low constraint diversity.
# 0.0 = perfectly uniform (all types fire equally); 1.0 = one type fires all.
# Note: for n=2 types the theoretical maximum Gini is 0.5, and for n=3 it is
# ~0.667. The threshold is set to 0.45 so it can flag a dominant-type pattern
# in both 2-type and 3-type trackers with realistic imbalances (e.g. 100:1).
GINI_DIVERSITY_THRESHOLD: float = 0.45

# Minimum absolute gap between mean training energy and mean held-out energy
# before we flag train/holdout divergence. Small floating-point differences
# are not meaningful.
DIVERGENCE_MIN_GAP: float = 0.05

# Minimum number of energy samples required to compute meaningful statistics.
MIN_ENERGY_SAMPLES: int = 3


# ---------------------------------------------------------------------------
# Finding data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrivialConstraintFinding:
    """A constraint type that fires frequently but almost never catches errors.

    **Detailed explanation for engineers:**
        A high-fire, near-zero-precision constraint adds overhead (it runs
        every verification) but provides no signal. In a self-learning loop
        this is dangerous because the tracker precision metric may reward
        low-precision types if the optimiser is not correctly penalising them.
        If many such types appear together, the pipeline may look busy while
        doing nothing useful.
    """

    constraint_type: str
    fired: int
    precision: float
    threshold: float = TRIVIAL_PRECISION_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "trivial_constraint",
            "constraint_type": self.constraint_type,
            "fired": self.fired,
            "precision": self.precision,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class ZeroEnergyFinding:
    """Energy sequence is near-constant — likely a shortcut path.

    **Detailed explanation for engineers:**
        If the energy function produces the same value (or a very small set
        of values) across many different inputs, the optimizer has probably
        collapsed onto a degenerate solution: output a constant regardless of
        the input. This bypasses genuine verification. Detection: if the
        number of distinct energy values in a sequence is very small relative
        to the sequence length, it is flagged as a shortcut.
    """

    sequence_length: int
    distinct_values: int
    min_distinct_required: int = ENERGY_DISTINCT_VALUES_MIN

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "zero_energy_shortcut",
            "sequence_length": self.sequence_length,
            "distinct_values": self.distinct_values,
            "min_distinct_required": self.min_distinct_required,
        }


@dataclass(frozen=True)
class LowDiversityFinding:
    """Constraint type distribution is highly concentrated (monoculture).

    **Detailed explanation for engineers:**
        Gini coefficient measures how unevenly fire counts are distributed
        across constraint types. A Gini of 0.0 means all types fire equally;
        a Gini of 1.0 means one type fires everything. High concentration
        (Gini > threshold) suggests the pipeline has collapsed onto a single
        "easy" constraint path and is ignoring the rest of the verification
        space. In self-learning this can emerge if one type consistently gets
        rewarded (even if it is only catching easy-to-fake violations).
    """

    gini: float
    n_types: int
    dominant_type: str
    dominant_fraction: float
    threshold: float = GINI_DIVERSITY_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "low_diversity",
            "gini": self.gini,
            "n_types": self.n_types,
            "dominant_type": self.dominant_type,
            "dominant_fraction": self.dominant_fraction,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class TrainHoldoutDivergenceFinding:
    """Training energy improves but held-out energy does not — possible gaming.

    **Detailed explanation for engineers:**
        In a well-trained EBM the energy function assigns low energy to
        genuine solutions and high energy to violations. If the training
        energy drops substantially while the held-out energy stays flat or
        rises, it means the model has memorised training traces rather than
        learning a general constraint. This mirrors train-accuracy vs.
        test-accuracy divergence in supervised ML. The gap metric here is:

            gap = mean(held_out_energies) - mean(train_energies)

        A large positive gap means training improved but held-out did not —
        flag as potential gaming. A gap near zero or negative means both
        trajectories improved together — genuine learning.
    """

    mean_train_energy: float
    mean_holdout_energy: float
    gap: float
    min_gap_threshold: float = DIVERGENCE_MIN_GAP

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "train_holdout_divergence",
            "mean_train_energy": self.mean_train_energy,
            "mean_holdout_energy": self.mean_holdout_energy,
            "gap": self.gap,
            "min_gap_threshold": self.min_gap_threshold,
        }


# ---------------------------------------------------------------------------
# Audit result
# ---------------------------------------------------------------------------


@dataclass
class RewardHackingReport:
    """Aggregated audit result from one run of the reward hacking detector.

    **Detailed explanation for engineers:**
        After calling audit_tracker() and/or audit_energy_trajectory(), the
        result is a RewardHackingReport. Check .clean to see if any issues
        were found. Iterate .findings for structured detail on each issue.
        Use .to_dict() to serialise for logging or downstream analysis.

        The report is intentionally separate from the findings list so that
        callers can check .clean quickly without examining every finding.
    """

    findings: list[
        TrivialConstraintFinding
        | ZeroEnergyFinding
        | LowDiversityFinding
        | TrainHoldoutDivergenceFinding
    ] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True if no reward hacking signals were detected."""
        return len(self.findings) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "n_findings": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _gini_coefficient(values: list[float]) -> float:
    """Compute the Gini coefficient of a list of non-negative values.

    **Detailed explanation for engineers:**
        The Gini coefficient summarises inequality in a distribution.
        For our use case the "wealth" is fire count per constraint type.
        Formula (stable numerics, handles zeros):

            G = (2 * sum(i * x_i) / (n * sum(x_i))) - (n + 1) / n

        where x_i is the sorted (ascending) i-th value (1-indexed).

        Returns 0.0 for a uniform distribution (all values equal) and
        approaches 1.0 as concentration increases. Returns 0.0 for
        degenerate inputs (all zeros or single element).

    Args:
        values: List of non-negative floats.

    Returns:
        Gini coefficient in [0.0, 1.0).
    """
    n = len(values)
    if n <= 1:
        return 0.0
    total = sum(values)
    if total == 0.0:
        return 0.0
    sorted_vals = sorted(values)
    numerator = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2.0 * numerator) / (n * total) - (n + 1) / n


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of a list of floats.

    Returns 0.0 for an empty list to avoid ZeroDivisionError.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_tracker(
    tracker: ConstraintTracker,
    *,
    min_fire_count: int = MIN_FIRE_COUNT_FOR_TRIVIAL_FLAG,
    trivial_precision_threshold: float = TRIVIAL_PRECISION_THRESHOLD,
    gini_threshold: float = GINI_DIVERSITY_THRESHOLD,
) -> RewardHackingReport:
    """Audit a ConstraintTracker for reward hacking signals.

    **Detailed explanation for engineers:**
        Runs three checks against the accumulated tracker statistics:

        1. TRIVIAL CONSTRAINTS — any type with fired >= min_fire_count and
           precision < trivial_precision_threshold is flagged. These are
           constraint types that generate lots of results but never catch
           real errors, suggesting they are either always satisfied or
           have been optimised to trivially pass.

        2. LOW DIVERSITY — if the fire count distribution across all types
           has a Gini coefficient above gini_threshold, the pipeline is
           dominated by one or two types, suggesting monoculture collapse.
           Only evaluated when there are at least 2 constraint types.

        No energy trajectory check is done here; for that see
        audit_energy_trajectory().

    Args:
        tracker: The ConstraintTracker to inspect.
        min_fire_count: Minimum fires before a type is eligible for the
            trivial-constraint flag (default 5). Too-few-fires cases
            lack statistical power.
        trivial_precision_threshold: Precision below this for high-fire
            types is considered trivial (default 0.05).
        gini_threshold: Gini coefficient above this signals low diversity
            (default 0.7).

    Returns:
        RewardHackingReport with zero or more findings.

    Spec: REQ-LEARN-002, SCENARIO-LEARN-002
    """
    report = RewardHackingReport()
    stats = tracker.stats()

    if not stats:
        return report

    # --- Check 1: trivial constraints ---
    for ctype, s in stats.items():
        fired: int = int(s.get("fired") or 0)
        precision: float = float(s.get("precision") or 0.0)
        if fired >= min_fire_count and precision < trivial_precision_threshold:
            report.findings.append(
                TrivialConstraintFinding(
                    constraint_type=ctype,
                    fired=fired,
                    precision=precision,
                    threshold=trivial_precision_threshold,
                )
            )

    # --- Check 2: constraint diversity (Gini) ---
    if len(stats) >= 2:
        fired_counts = [float(s.get("fired") or 0) for s in stats.values()]
        gini = _gini_coefficient(fired_counts)
        if gini > gini_threshold:
            # Find the dominant type (highest fire count).
            total_fired = sum(fired_counts)
            max_fired = 0.0
            dominant_type = ""
            for ctype, s in stats.items():
                f = float(s.get("fired") or 0)
                if f > max_fired:
                    max_fired = f
                    dominant_type = ctype
            dominant_fraction = max_fired / total_fired if total_fired > 0 else 0.0
            report.findings.append(
                LowDiversityFinding(
                    gini=gini,
                    n_types=len(stats),
                    dominant_type=dominant_type,
                    dominant_fraction=dominant_fraction,
                    threshold=gini_threshold,
                )
            )

    return report


def audit_energy_trajectory(
    train_energies: list[float],
    holdout_energies: list[float],
    *,
    min_samples: int = MIN_ENERGY_SAMPLES,
    min_gap: float = DIVERGENCE_MIN_GAP,
    distinct_values_min: int = ENERGY_DISTINCT_VALUES_MIN,
) -> RewardHackingReport:
    """Audit energy trajectories for shortcut and gaming signals.

    **Detailed explanation for engineers:**
        Runs two checks against the energy sequences:

        1. ZERO-ENERGY SHORTCUT — if either the training or held-out energy
           sequence has fewer than distinct_values_min distinct floating-point
           values (after rounding to 6 decimal places to handle FP noise),
           it is flagged as a potential constant-output shortcut. The energy
           function appears to be ignoring its input.

        2. TRAIN/HOLDOUT DIVERGENCE — if mean(holdout) - mean(train) > min_gap,
           the model improved on training data but not on held-out data.
           This is the canonical signature of overfitting / gaming.

        Either sequence below min_samples results in an empty report (not
        enough data to judge).

    Args:
        train_energies: Sequence of scalar energy values from training traces.
            Lower is better (convention: 0 = satisfied, >0 = violation).
        holdout_energies: Sequence of energy values from held-out traces,
            collected in the same order as training (temporally aligned).
        min_samples: Minimum length for both sequences (default 3).
        min_gap: Minimum divergence gap to flag train/holdout divergence
            (default 0.05). Smaller differences are within noise.
        distinct_values_min: Flag zero-energy shortcut when distinct rounded
            values in either sequence is below this (default 2). A sequence
            of length N with only 1 distinct value is almost certainly constant.

    Returns:
        RewardHackingReport with zero or more findings.

    Spec: REQ-LEARN-002, SCENARIO-LEARN-002
    """
    report = RewardHackingReport()

    # Need enough samples in both sequences to draw conclusions.
    if len(train_energies) < min_samples or len(holdout_energies) < min_samples:
        return report

    # --- Check 1: zero-energy shortcut detection ---
    # Round to 6 decimal places to collapse floating-point rounding noise while
    # still treating genuinely distinct values as distinct.
    def _distinct(seq: list[float]) -> int:
        return len({round(v, 6) for v in seq})

    train_distinct = _distinct(train_energies)
    holdout_distinct = _distinct(holdout_energies)

    if train_distinct < distinct_values_min or holdout_distinct < distinct_values_min:
        # Report whichever is worse (fewest distinct values).
        seq_len = len(train_energies) if train_distinct <= holdout_distinct else len(holdout_energies)
        distinct = min(train_distinct, holdout_distinct)
        report.findings.append(
            ZeroEnergyFinding(
                sequence_length=seq_len,
                distinct_values=distinct,
                min_distinct_required=distinct_values_min,
            )
        )

    # --- Check 2: train/holdout divergence ---
    mean_train = _mean(train_energies)
    mean_holdout = _mean(holdout_energies)
    gap = mean_holdout - mean_train

    # A large positive gap means held-out energy stayed high while training
    # energy fell — the hallmark of gaming the training distribution.
    if gap > min_gap:
        report.findings.append(
            TrainHoldoutDivergenceFinding(
                mean_train_energy=mean_train,
                mean_holdout_energy=mean_holdout,
                gap=gap,
                min_gap_threshold=min_gap,
            )
        )

    return report


def audit_full(
    tracker: ConstraintTracker,
    train_energies: list[float],
    holdout_energies: list[float],
    *,
    min_fire_count: int = MIN_FIRE_COUNT_FOR_TRIVIAL_FLAG,
    trivial_precision_threshold: float = TRIVIAL_PRECISION_THRESHOLD,
    gini_threshold: float = GINI_DIVERSITY_THRESHOLD,
    min_samples: int = MIN_ENERGY_SAMPLES,
    min_gap: float = DIVERGENCE_MIN_GAP,
    distinct_values_min: int = ENERGY_DISTINCT_VALUES_MIN,
) -> RewardHackingReport:
    """Run all reward hacking checks and combine findings into one report.

    **Detailed explanation for engineers:**
        Convenience wrapper that calls audit_tracker() and
        audit_energy_trajectory() and merges the findings into a single
        RewardHackingReport. Use this when you have both a tracker and
        energy trajectory data available (the common case in Exp 223/241
        replay evaluations).

        Parameters mirror those of the individual audit functions.

    Args:
        tracker: Populated ConstraintTracker from the pipeline run.
        train_energies: Energy values from training trace evaluation.
        holdout_energies: Energy values from held-out trace evaluation.
        All remaining keyword args: forwarded to the respective audit fns.

    Returns:
        Combined RewardHackingReport containing all findings.

    Spec: REQ-LEARN-002, SCENARIO-LEARN-002
    """
    tracker_report = audit_tracker(
        tracker,
        min_fire_count=min_fire_count,
        trivial_precision_threshold=trivial_precision_threshold,
        gini_threshold=gini_threshold,
    )
    energy_report = audit_energy_trajectory(
        train_energies,
        holdout_energies,
        min_samples=min_samples,
        min_gap=min_gap,
        distinct_values_min=distinct_values_min,
    )
    combined = RewardHackingReport(
        findings=tracker_report.findings + energy_report.findings
    )
    return combined


__all__ = [
    "MIN_FIRE_COUNT_FOR_TRIVIAL_FLAG",
    "TRIVIAL_PRECISION_THRESHOLD",
    "ENERGY_DISTINCT_VALUES_MIN",
    "GINI_DIVERSITY_THRESHOLD",
    "DIVERGENCE_MIN_GAP",
    "MIN_ENERGY_SAMPLES",
    "TrivialConstraintFinding",
    "ZeroEnergyFinding",
    "LowDiversityFinding",
    "TrainHoldoutDivergenceFinding",
    "RewardHackingReport",
    "audit_tracker",
    "audit_energy_trajectory",
    "audit_full",
]
