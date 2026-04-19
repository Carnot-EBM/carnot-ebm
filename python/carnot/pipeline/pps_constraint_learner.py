"""Progressive Parameter Selection EBM Constraint Learner (PPSEBM, arXiv 2512.15658).

**Why domain-partitioned parameter isolation (the core PPSEBM insight):**
    LSEBMCL (Exp 457, arXiv 2501.05495) improved cross-session FP rate from 0.46 to 0.0
    by using an Ising EBM to replay Session 1 violation patterns in Session 2.  However,
    LSEBMCL uses a SINGLE shared parameter space for all constraint domains (arithmetic,
    code, logical).

    The failure mode: when arithmetic sessions and code sessions interleave in production,
    gradient updates from arithmetic training bleed into the parameter regions that encode
    code constraint patterns.  Improving arithmetic carry-error detection degrades code
    constraint detection — catastrophic interference.

    PPSEBM fixes this with Progressive Parameter Selection: each constraint domain gets
    its OWN isolated parameter partition.  When the arithmetic partition is updated, the
    code and logical partitions are frozen.  The update cannot bleed across partition walls.

**Why boundary violations from EBM (not random noise):**
    To stress-test partition isolation, we need samples near the boundary of each domain's
    learned distribution — not random strings.  Random violations would not reflect the
    actual error distribution.  The EBM for each domain encodes learned co-occurrence
    patterns (e.g., carry + sign errors tend to co-occur in arithmetic).  Sampling from
    the EBM generates violations that are statistically realistic for that domain, and
    "boundary" samples (near the edge of the learned distribution) are the hardest cases
    for the partition — most likely to cause cross-domain bleed if isolation fails.

**Why cosine distance as the isolation metric:**
    Cosine similarity measures the DIRECTION of gradient updates, not their magnitude.
    If two partitions' gradient update vectors point in the same direction, the partitions
    are sharing signal — they are not truly isolated.  Cosine DISTANCE = 1 - cosine_similarity,
    so a cosine distance near 1.0 means the update directions are orthogonal (fully independent).
    The PPSEBM paper (arXiv 2512.15658) uses a threshold of 0.8 to declare isolation: at most
    0.2 of shared directional signal between any two domain partitions.

Spec: REQ-SELFLEARN-016, REQ-SELFLEARN-017, REQ-SELFLEARN-018,
      SCENARIO-SELFLEARN-016, SCENARIO-SELFLEARN-017, SCENARIO-SELFLEARN-018
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from carnot.pipeline.lsebm_replayer import LSEBMConstraintReplayer

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# ConstraintDomain
# ---------------------------------------------------------------------------


class ConstraintDomain(enum.Enum):
    """The three constraint domains supported by PPSConstraintLearner.

    **Why exactly three domains:**
        Carnot's pipeline handles three fundamentally different constraint types:
        - ARITHMETIC: numerical computation correctness (carry errors, sign errors,
          unit inconsistencies, comparison direction).  Ground truth is checkable
          via arithmetic rules or Z3.
        - CODE: program correctness constraints (assertion failures, type errors,
          off-by-one, missing cases).  Ground truth is checkable via execution.
        - LOGICAL: propositional/first-order logic constraints (invalid inferences,
          contradictions, scope errors).  Ground truth is checkable via SAT/logic.

        Keeping domains separate prevents a model that is good at arithmetic from
        being confused by code-style violations and vice versa.

    Spec: REQ-SELFLEARN-016
    """

    ARITHMETIC = "arithmetic"
    CODE = "code"
    LOGICAL = "logical"


# ---------------------------------------------------------------------------
# DomainParameterPartition
# ---------------------------------------------------------------------------


@dataclass
class DomainParameterPartition:
    """One isolated parameter partition for a single constraint domain.

    **Why a separate partition per domain (not shared weights):**
        In LSEBMCL, a single weight vector encodes all domain patterns.  When
        arithmetic violations are replayed, the gradient update touches weights
        that also encode code and logical patterns — those weights shift, degrading
        performance on non-arithmetic domains.

        A DomainParameterPartition owns its own weight array.  Only this partition's
        ``update()`` method modifies the weights.  All other partitions are unaffected.

    **Gradient tracking:**
        The partition records the cumulative gradient update applied via ``update()``.
        ``gradient_direction()`` returns the normalised cumulative gradient, used by
        ``PartitionIsolationScore`` to measure cosine distance between domains.

    Args:
        domain: Which constraint domain this partition represents.
        weights: Initial weight vector (1D numpy array).  Copied on construction
                 so the caller cannot mutate the partition's state externally.

    Spec: REQ-SELFLEARN-016, REQ-SELFLEARN-018
    """

    domain: ConstraintDomain
    weights: np.ndarray
    # Cumulative gradient update applied via update() — used for isolation scoring.
    _cumulative_gradient: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.weights = np.array(self.weights, dtype=np.float64)
        self._cumulative_gradient = np.zeros_like(self.weights)

    def update(self, gradient: np.ndarray) -> None:
        """Apply a gradient update to THIS partition's weights only.

        Other partitions' weights are never touched by this call — that is the
        entire point of partition isolation.  The gradient is also accumulated
        in ``_cumulative_gradient`` so ``PartitionIsolationScore`` can measure
        how different the update directions are across domains.

        Args:
            gradient: Gradient vector (same shape as weights).  Represents the
                      direction to move weights to reduce energy for this domain's
                      violation patterns.

        Spec: REQ-SELFLEARN-016
        """
        grad = np.array(gradient, dtype=np.float64)
        self.weights += grad
        self._cumulative_gradient += grad

    def gradient_direction(self) -> np.ndarray:
        """Return the normalised cumulative gradient update direction.

        **Why normalise:**
            PartitionIsolationScore uses cosine distance, which is a DIRECTIONAL
            measure.  Two partitions with large but orthogonal gradients have high
            cosine distance (good isolation).  Normalising removes magnitude and
            focuses on direction.

        Returns:
            Unit vector in the direction of the cumulative gradient.  Returns the
            zero vector (no update applied yet) if the cumulative gradient is zero.

        Spec: REQ-SELFLEARN-018
        """
        norm = float(np.linalg.norm(self._cumulative_gradient))
        if norm < 1e-12:
            return np.zeros_like(self._cumulative_gradient)
        return self._cumulative_gradient / norm


# ---------------------------------------------------------------------------
# PartitionIsolationScore
# ---------------------------------------------------------------------------


class PartitionIsolationScore:
    """Measure how isolated the parameter partitions are from each other.

    **Why measure isolation:**
        After training each domain independently, we want to verify that the
        gradient updates for different domains are pointing in different directions.
        If ARITHMETIC and CODE gradients point in the same direction, it means
        the model is learning a shared representation — partition isolation has
        failed and cross-domain bleed is occurring.

    **Score computation:**
        For each pair of distinct partitions, compute the cosine similarity between
        their ``gradient_direction()`` vectors.  Cosine DISTANCE = 1 - cosine_similarity.
        The score is the MINIMUM cosine distance across all pairs — i.e., the most
        similar pair is the bottleneck.

        A score of 1.0 means all pairs are perfectly orthogonal (ideal isolation).
        A score of 0.0 means at least one pair is pointing in the exact same direction.

    Args:
        partitions: List of DomainParameterPartition objects (one per domain).

    Spec: REQ-SELFLEARN-018
    """

    def __init__(self, partitions: list[DomainParameterPartition]) -> None:
        self.partitions = partitions

    def score(self) -> float:
        """Return minimum cosine distance between any two partition gradient directions.

        Returns 1.0 if fewer than two partitions have non-zero gradients (trivially
        isolated — no cross-domain comparison possible).

        Returns:
            Float in [0.0, 1.0].  Higher is better (more isolated).

        Spec: REQ-SELFLEARN-018
        """
        directions = []
        for p in self.partitions:
            d = p.gradient_direction()
            if np.linalg.norm(d) > 1e-12:
                directions.append(d)

        if len(directions) < 2:
            # Cannot compute pairwise distance with fewer than 2 non-zero gradients.
            return 1.0

        min_distance = 1.0
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                # Cosine similarity: dot product of unit vectors.
                cos_sim = float(np.dot(directions[i], directions[j]))
                # Clamp to [-1, 1] to handle floating point drift.
                cos_sim = max(-1.0, min(1.0, cos_sim))
                cos_dist = 1.0 - cos_sim
                if cos_dist < min_distance:
                    min_distance = cos_dist

        return min_distance

    def is_isolated(self, threshold: float = 0.8) -> bool:
        """Return True iff partition_isolation_score >= threshold.

        The PPSEBM paper (arXiv 2512.15658) uses 0.8 as the default threshold,
        meaning at most 0.2 of shared directional signal is permitted between any
        two domain partitions.

        Args:
            threshold: Minimum acceptable cosine distance (default 0.8).

        Returns:
            True if all domain pairs are sufficiently isolated.

        Spec: REQ-SELFLEARN-018
        """
        return self.score() >= threshold


# ---------------------------------------------------------------------------
# PPSConstraintLearner
# ---------------------------------------------------------------------------


class PPSConstraintLearner:
    """Progressive Parameter Selection constraint learner with per-domain isolation.

    **Lifecycle:**
        1. Construct with a list of ConstraintDomain values and a shared
           LSEBMConstraintReplayer (used as the base for per-domain sub-replayers).
        2. Call ``fit_domain(domain, violations)`` for each domain independently.
        3. Call ``generate_boundary_violations(domain, n)`` to get EBM-sampled violations
           for stress-testing the partition walls.
        4. Call ``session_fp_rate(domain, test_questions)`` to evaluate per-domain FP rate.
        5. Use ``PartitionIsolationScore(self._partitions.values())`` to measure isolation.

    **Why per-domain replayers (not one shared replayer):**
        A single replayer trained on a mix of arithmetic + code violations would encode
        both distributions in one parameter space — exactly the problem PPSEBM solves.
        Each domain gets its own LSEBMConstraintReplayer, trained only on that domain's
        violations.  The EBM for arithmetic knows only arithmetic patterns; the EBM for
        code knows only code patterns.

    **Why the shared replayer argument:**
        The constructor accepts a base replayer for configuration parameters (n_replay,
        ebm_n_iter).  Each domain gets a FRESH instance with the same config, not the
        shared instance itself.  This avoids accidental state sharing between domains.

    Args:
        domains: List of ConstraintDomain values to support.
        replayer: Base LSEBMConstraintReplayer used to inherit n_replay and ebm_n_iter
                  config.  Each domain gets its own fresh instance.

    Spec: REQ-SELFLEARN-016, REQ-SELFLEARN-017, SCENARIO-SELFLEARN-016/017/018
    """

    # Weight vector dimension per partition — small enough for fast CPU operation,
    # large enough to encode meaningful directional differences across domains.
    _PARTITION_DIM = 16

    def __init__(
        self,
        domains: list[ConstraintDomain],
        replayer: LSEBMConstraintReplayer,
    ) -> None:
        self.domains = list(domains)
        self._n_replay = replayer.n_replay
        self._ebm_n_iter = replayer.ebm_n_iter

        # Per-domain isolated parameter partitions — initialised with different
        # random seeds so they start in different regions of parameter space.
        # WHY different seeds: if all partitions start identically, gradient updates
        # in the same direction would not be detectable as non-isolated.
        rng = np.random.default_rng(seed=42)
        self._partitions: dict[ConstraintDomain, DomainParameterPartition] = {}
        for domain in self.domains:
            # Xavier-uniform-style initialisation: weights ~ U(-1/sqrt(d), 1/sqrt(d)).
            scale = 1.0 / np.sqrt(self._PARTITION_DIM)
            init_weights = rng.uniform(-scale, scale, size=self._PARTITION_DIM)
            self._partitions[domain] = DomainParameterPartition(
                domain=domain, weights=init_weights
            )

        # Per-domain EBM replayers — each domain gets a fresh instance.
        self._replayers: dict[ConstraintDomain, LSEBMConstraintReplayer] = {
            d: LSEBMConstraintReplayer(
                n_replay=self._n_replay, ebm_n_iter=self._ebm_n_iter
            )
            for d in self.domains
        }

        # Per-domain violation vocabulary — populated by fit_domain().
        self._domain_violations: dict[ConstraintDomain, list[str]] = {
            d: [] for d in self.domains
        }

    def fit_domain(self, domain: ConstraintDomain, violations: list[str]) -> None:
        """Update ONLY the specified domain's parameter partition.

        All other domain partitions remain numerically unchanged — this is the
        core isolation guarantee of PPSEBM.

        The gradient update is computed from the violation frequency distribution:
        each violation type contributes a unit vector to the gradient, weighted by
        its observed count.  The partition's weight vector is updated in the direction
        of the most commonly observed violations for this domain.

        WHY this gradient encoding: we want the partition to "point toward" the centroid
        of this domain's violation distribution in the _PARTITION_DIM-dimensional space.
        Subsequent scoring via cosine distance then measures whether different domains'
        centroids are in different directions — i.e., whether the domains are represented
        independently.

        Args:
            domain: The domain whose partition to update.
            violations: List of violation type strings for this domain's training session.
                        May contain duplicates — more frequent types have more influence.

        Spec: REQ-SELFLEARN-016, SCENARIO-SELFLEARN-016
        """
        if not violations:
            return

        # Train the per-domain EBM replayer.
        self._replayers[domain].fit(violations)
        self._domain_violations[domain] = list(violations)

        # Build the gradient for this domain's partition.
        # WHY this approach: we compute a vocabulary-based gradient where each unique
        # violation type maps to one dimension (mod _PARTITION_DIM to handle large vocabs).
        # The gradient magnitude for each dimension equals the count of that violation type.
        # This ensures the gradient direction encodes WHAT violations were seen (not just
        # how many total), making cosine distance meaningful for isolation testing.
        counts: dict[str, int] = {}
        for v in violations:
            counts[v] = counts.get(v, 0) + 1

        grad = np.zeros(self._PARTITION_DIM, dtype=np.float64)
        for vtype, count in counts.items():
            # Use a stable MD5-based hash of the violation type string to determine which
            # gradient dimension it updates.  WHY MD5 (not Python's hash()): Python's hash()
            # is PYTHONHASHSEED-randomised per process, giving non-reproducible dimension
            # assignments across runs.  MD5 is deterministic, so "carry" always maps to
            # the same dimension regardless of when or where the code runs.  This is critical
            # for partition isolation: different domains use different vocabulary strings
            # (e.g. "carry" vs "type_error"), and stable hashing ensures they update
            # different gradient dimensions — giving orthogonal gradients and high cosine distance.
            dim_idx = int(hashlib.md5(vtype.encode()).hexdigest(), 16) % self._PARTITION_DIM
            grad[dim_idx] += count

        # Normalise gradient magnitude to learning-rate scale before update.
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1e-12:
            grad = grad / grad_norm

        # Update ONLY the target partition.
        self._partitions[domain].update(grad)

        # All other partitions are intentionally NOT touched — partition isolation.

    def generate_boundary_violations(
        self, domain: ConstraintDomain, n: int
    ) -> list[str]:
        """Generate n EBM-sampled boundary violations for the specified domain.

        WHY EBM sampling (not random): the domain's LSEBMConstraintReplayer was fitted
        on real violation data from that domain.  Sampling from the EBM produces violations
        that are statistically plausible for this domain — near the boundary of the learned
        distribution.  These are harder test cases than random violations and more likely
        to expose cross-domain bleed if partition isolation fails.

        Args:
            domain: The domain to generate violations for.
            n: Number of boundary violation strings to generate.

        Returns:
            List of n violation type strings from the domain's EBM.  Returns an empty
            list if the domain's replayer has not been fitted yet.

        Spec: REQ-SELFLEARN-017, SCENARIO-SELFLEARN-017
        """
        return self._replayers[domain].generate_replay(n)

    def session_fp_rate(
        self, domain: ConstraintDomain, test_questions: list
    ) -> float:
        """Simulate session FP rate for a domain after partition training.

        WHY simulation (not live LLM inference): this is a CPU-only experiment.
        We simulate the FP rate by measuring how well the domain's weight partition
        "covers" the test question set — i.e., how many test questions produce
        violations in the domain's learned vocabulary.

        The FP rate here is defined as the fraction of test questions where the
        domain's partition FAILS to detect a violation that was in its training set.
        If the domain was trained on 'carry' errors and a test question's expected
        violation type is 'carry', a miss is a false negative (but we measure it
        as a proxy FP contribution for comparison with LSEBMCL baseline).

        For Exp 470's synthetic benchmark, test_questions is a list of (question, expected_vtype)
        tuples.  If the element is a plain string, we treat the expected violation as 'unknown'
        and count it as not-detected.

        Args:
            domain: The domain to evaluate.
            test_questions: List of (question_text, expected_violation_type) tuples,
                            or plain strings (treated as unknown violation type).

        Returns:
            Float in [0.0, 1.0]: fraction of questions where the domain partition
            failed to cover the expected violation type.

        Spec: REQ-SELFLEARN-016
        """
        if not test_questions:
            return 0.0

        trained_vocab = set(self._domain_violations[domain])
        failures = 0

        for item in test_questions:
            if isinstance(item, tuple) and len(item) == 2:
                _, expected_vtype = item
                if expected_vtype not in trained_vocab:
                    failures += 1
            else:
                # Plain string or unknown format — count as not-detected.
                failures += 1

        return failures / len(test_questions)

    @property
    def partitions(self) -> list[DomainParameterPartition]:
        """Return all domain partitions as an ordered list (order matches self.domains)."""
        return [self._partitions[d] for d in self.domains]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ConstraintDomain",
    "DomainParameterPartition",
    "PartitionIsolationScore",
    "PPSConstraintLearner",
]
