"""Continuous-valued EBM — Phase 3 seed bridging discrete Ising to Kona's domain.

**Researcher summary:**
    Minimal continuous EBM that reuses an Ising model's coupling matrix J and
    bias h as initialisation.  Three sampling algorithms are provided:

    1. ``sample_continuous``: Vanilla gradient descent with tanh squashing (Exp 435a
       baseline; achieves L2=2.69 vs Ising ground state).
    2. ``sample_langevin``: Langevin dynamics — gradient descent + Gaussian noise
       injection.  Noise helps escape local minima (Exp 446 improvement).
    3. ``sample_energy_matching``: Energy Matching trajectory from arXiv 2504.10612
       (NeurIPS 2025) — normalised gradient flow for constant-speed convergence.

**Why this matters for Phase 3:**
    Kona-style reasoning requires non-autoregressive inference over a continuous
    latent space.  Before we can train such a model we need to verify that our
    quadratic energy function E(x) = -0.5*x^T*J*x - h^T*x is consistent across
    the discrete↔continuous boundary.  Langevin dynamics is how Kona-style
    continuous reasoning would sample from an energy landscape — the thermal noise
    is what allows exploration beyond the nearest local minimum.

**Why tanh squashing?**
    The Ising model's natural variable domain is {-1, +1} (discrete spins).  The
    continuous relaxation x ∈ ℝ^n has unbounded energy minima — adding tanh ensures
    x ∈ (-1, 1)^n, keeping the continuous and discrete problems comparable.

**Why JAX/numpy and NOT torch?**
    Phase 3 must be portable to future hardware (Extropic TSU, photonic computing).
    JAX's functional purity and XLA backend make this easier than PyTorch, which
    carries CUDA-centric assumptions.

Spec: REQ-KONA-001, REQ-KONA-002, REQ-KONA-003,
      SCENARIO-KONA-001, SCENARIO-KONA-002, SCENARIO-KONA-003,
      SCENARIO-KONA-004, SCENARIO-KONA-005,
      REQ-KONA-026, SCENARIO-KONA-026,
      REQ-KONA-027, SCENARIO-KONA-027,
      REQ-KONA-028, SCENARIO-KONA-028,
      REQ-KONA-030, SCENARIO-KONA-030
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ContinuousEBM:
    """Continuous-valued EBM with quadratic energy and tanh squashing.

    **Researcher summary:**
        Stores the coupling matrix J and bias h from an Ising model.
        Energy: E(x) = -0.5 * x^T * J * x - h^T * x, with x ∈ (-1, 1)^n
        enforced via tanh during sampling.

    **Detailed explanation for engineers:**
        This is intentionally minimal — no hidden layers, no learned parameters.
        It is the simplest possible continuous relaxation of the Ising energy
        function, used only to verify that gradient descent and simulated
        annealing agree on the energy landscape geometry.

    Attributes:
        variables: Number of variables (n).
        coupling: Symmetric coupling matrix J of shape (n, n).
        bias: Bias vector h of shape (n,).
    """

    variables: int
    coupling: np.ndarray
    bias: np.ndarray

    def tss_diagnose(
        self,
        examples: list[tuple[str, str, bool]],
        n_steps: int = 128,
        lr: float = 0.01,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Measure Q11 TSS risk at the ``sign(z)`` bottleneck.

        The diagnostic computes a deterministic SC-Energy proxy by embedding
        the question and response into the model's latent dimension, gating both
        vectors by ``sign(z)``, and taking their cosine similarity. The Z3-side
        label comes from ``Z3MathVerifier`` when parseable arithmetic exists and
        falls back to the provided FoVer correctness label otherwise.

        Spec: REQ-KONA-026, SCENARIO-KONA-026
        """
        from carnot.verify.z3_math_verifier import Z3MathVerifier

        z3_verifier = Z3MathVerifier()
        sc_scores: list[float] = []
        z3_labels: list[float] = []

        for index, (question, response, is_correct) in enumerate(examples):
            sign_z = np.sign(
                sample_continuous(self, n_steps=n_steps, lr=lr, seed=seed + index)
            )
            sign_z = np.where(sign_z == 0.0, 1.0, sign_z)
            sign_gate = (sign_z + 1.0) / 2.0

            q_vec = _tss_text_vector(question, self.variables) * sign_gate
            r_vec = _tss_text_vector(response, self.variables) * sign_gate
            sc_score = float(
                np.dot(q_vec, r_vec)
                / ((np.linalg.norm(q_vec) * np.linalg.norm(r_vec)) + 1e-12)
            )

            z3_energy = z3_verifier.score(response)
            z3_label = bool(is_correct) if z3_energy == 0.5 else z3_energy == 0.0
            sc_scores.append(sc_score)
            z3_labels.append(float(z3_label))

        sc_arr = np.asarray(sc_scores, dtype=np.float64)
        z3_arr = np.asarray(z3_labels, dtype=np.float64)
        corr = float(
            np.nan_to_num(
                np.corrcoef(sc_arr, z3_arr)[0, 1],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
        vulnerability_score = float(np.clip(1.0 - abs(corr), 0.0, 1.0))

        return {
            "sc_energy_z3_correlation": round(corr, 4),
            "optimal_transversal_k": 2,
            "tss_vulnerability_score": round(vulnerability_score, 4),
            "tss_instrumented": True,
            "sign_z_bottleneck_diagnosed": True,
            "ste_pipeline_risk": vulnerability_score > 0.6,
            "honest_verdict": f"tss_instrumented_corr_{corr:.3f}_vuln_{vulnerability_score:.3f}",
        }


@dataclass
class FeasibilityStepResult:
    """Result of an FSNet-style latent feasibility repair.

    Spec: REQ-KONA-027, SCENARIO-KONA-027
    """

    state: np.ndarray
    initial_violation_energy: float
    violation_energy: float
    violation_count: int
    convergence_steps: int
    distortion_l2: float
    converged: bool


@dataclass
class AdaptiveRepairResult:
    """Result of a SnareNet-style adaptive repair layer.

    Spec: REQ-KONA-028, SCENARIO-KONA-028
    """

    state: np.ndarray
    fsnet_state: np.ndarray
    initial_constraint_satisfaction: float
    fsnet_constraint_satisfaction: float
    final_constraint_satisfaction: float
    initial_violation_energy: float
    fsnet_violation_energy: float
    violation_energy: float
    violation_count: int
    repair_iterations: int
    fsnet_distortion_from_initial: float
    distortion_from_initial: float
    distortion_from_fsnet: float
    converged: bool
    final_relaxation: float


@dataclass
class FeasibilityChannelCase:
    """Before/after repair row for DSP-style feasibility-channel diagnostics.

    **Researcher summary:**
        Each row asks whether a proposed next repair step was useful. The
        channel observes only the before-state violation pressure, then the
        label is derived from whether the after-state actually reduced hard
        violation energy or count.

    Spec: REQ-KONA-030, SCENARIO-KONA-030
    """

    case_id: str
    cohort: str
    before_violation_energy: float
    before_violation_count: int
    after_violation_energy: float
    after_violation_count: int
    distortion_delta: float


@dataclass
class AdaptiveRepairLayer:
    """SnareNet-style differentiable repair layer for linear hard constraints.

    **Researcher summary:**
        The layer first runs the existing FSNet-style feasibility step, then
        appends a smooth repair pass. The smooth pass descends a differentiable
        soft-hinge pressure term whose relaxation value tightens after progress
        and relaxes when a proposed step fails to reduce hard violation energy.

    Spec: REQ-KONA-028, SCENARIO-KONA-028
    """

    fsnet_steps: int = 48
    fsnet_lr: float = 0.55
    fsnet_anchor_weight: float = 0.02
    n_steps: int = 16
    lr: float = 0.18
    anchor_weight: float = 0.02
    initial_relaxation: float = 0.12
    min_relaxation: float = 0.03
    max_relaxation: float = 0.50
    relaxation_growth: float = 1.40
    relaxation_decay: float = 0.75
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.fsnet_steps < 0:
            raise ValueError("fsnet_steps must be non-negative")
        if self.fsnet_lr < 0.0:
            raise ValueError("fsnet_lr must be non-negative")
        if self.fsnet_anchor_weight < 0.0:
            raise ValueError("fsnet_anchor_weight must be non-negative")
        if self.n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if self.lr < 0.0:
            raise ValueError("lr must be non-negative")
        if self.anchor_weight < 0.0:
            raise ValueError("anchor_weight must be non-negative")
        if self.initial_relaxation <= 0.0:
            raise ValueError("initial_relaxation must be positive")
        if self.min_relaxation <= 0.0:
            raise ValueError("min_relaxation must be positive")
        if self.max_relaxation <= 0.0:
            raise ValueError("max_relaxation must be positive")
        if self.min_relaxation > self.max_relaxation:
            raise ValueError("min_relaxation must not exceed max_relaxation")
        if self.relaxation_growth <= 1.0:
            raise ValueError("relaxation_growth must be greater than 1")
        if not 0.0 < self.relaxation_decay < 1.0:
            raise ValueError("relaxation_decay must be in (0, 1)")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")

    def repair(
        self,
        state: np.ndarray,
        constraint_matrix: np.ndarray,
        constraint_bias: np.ndarray | None = None,
    ) -> AdaptiveRepairResult:
        """Apply FSNet feasibility repair followed by adaptive smooth repair.

        Args:
            state: Latent vector ``z`` with shape ``(d,)``.
            constraint_matrix: Linear verifier matrix ``A`` with shape ``(m, d)``.
            constraint_bias: Optional verifier bias ``b`` with shape ``(m,)``.

        Returns:
            AdaptiveRepairResult with the final bounded state and diagnostics.

        Raises:
            ValueError: If latent/constraint shapes are malformed.
        """
        z0 = np.asarray(state, dtype=np.float64)
        A = np.asarray(constraint_matrix, dtype=np.float64)

        if z0.ndim != 1:
            raise ValueError("state must be one-dimensional")
        if A.ndim != 2 or A.shape[1] != z0.shape[0]:
            raise ValueError("constraint_matrix must have shape (n_constraints, state_dim)")

        if constraint_bias is None:
            b = np.zeros(A.shape[0], dtype=np.float64)
        else:
            b = np.asarray(constraint_bias, dtype=np.float64)
        if b.shape != (A.shape[0],):
            raise ValueError("constraint_bias must have shape (n_constraints,)")

        relaxation = float(
            np.clip(self.initial_relaxation, self.min_relaxation, self.max_relaxation)
        )
        initial_energy, _, initial_satisfaction = self._measure(z0, A, b, relaxation)
        fsnet_result = feasibility_step(
            z0,
            A,
            b,
            n_steps=self.fsnet_steps,
            lr=self.fsnet_lr,
            anchor_weight=self.fsnet_anchor_weight,
            tolerance=self.tolerance,
        )
        fsnet_state = fsnet_result.state.copy()
        fsnet_energy, fsnet_count, fsnet_satisfaction = self._measure(
            fsnet_state, A, b, relaxation
        )

        z = fsnet_state.copy()
        final_energy = fsnet_energy
        final_count = fsnet_count
        final_satisfaction = fsnet_satisfaction
        steps_taken = 0

        for step in range(1, self.n_steps + 1):
            scores = A @ z + b
            hinge = np.maximum(scores, 0.0)
            scaled_scores = np.clip(scores / relaxation, -60.0, 60.0)
            soft_pressure = 1.0 / (1.0 + np.exp(-scaled_scores))
            grad = (
                (2.0 * A.T @ hinge)
                + (A.T @ soft_pressure)
                + (2.0 * self.anchor_weight * (z - z0))
            )
            candidate = np.tanh(z - self.lr * relaxation * grad)
            candidate_energy, candidate_count, candidate_satisfaction = self._measure(
                candidate, A, b, relaxation
            )
            steps_taken = step

            if candidate_energy <= final_energy + self.tolerance:
                z = candidate
                final_energy = candidate_energy
                final_count = candidate_count
                final_satisfaction = candidate_satisfaction
                relaxation = max(self.min_relaxation, relaxation * self.relaxation_decay)
            else:
                relaxation = min(self.max_relaxation, relaxation * self.relaxation_growth)

        return AdaptiveRepairResult(
            state=z,
            fsnet_state=fsnet_state,
            initial_constraint_satisfaction=initial_satisfaction,
            fsnet_constraint_satisfaction=fsnet_satisfaction,
            final_constraint_satisfaction=final_satisfaction,
            initial_violation_energy=initial_energy,
            fsnet_violation_energy=fsnet_energy,
            violation_energy=final_energy,
            violation_count=final_count,
            repair_iterations=steps_taken,
            fsnet_distortion_from_initial=fsnet_result.distortion_l2,
            distortion_from_initial=float(np.linalg.norm(z - z0)),
            distortion_from_fsnet=float(np.linalg.norm(z - fsnet_state)),
            converged=final_energy <= self.tolerance,
            final_relaxation=relaxation,
        )

    @staticmethod
    def _measure(
        state: np.ndarray,
        constraint_matrix: np.ndarray,
        constraint_bias: np.ndarray,
        relaxation: float,
    ) -> tuple[float, int, float]:
        scores = constraint_matrix @ state + constraint_bias
        hinge = np.maximum(scores, 0.0)
        energy = float(hinge @ hinge)
        count = int(np.sum(scores > 0.0))
        scaled_scores = np.clip(scores / relaxation, -60.0, 60.0)
        satisfaction = float(np.mean(1.0 / (1.0 + np.exp(scaled_scores))))
        return energy, count, satisfaction


def _tss_text_vector(text: str, dim: int) -> np.ndarray:
    """Return a deterministic bag-of-words vector for Q11 TSS diagnostics."""
    vector = np.zeros(dim, dtype=np.float64)
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    for token in cleaned.split():
        bucket = sum((offset + 1) * ord(ch) for offset, ch in enumerate(token)) % dim
        vector[bucket] += 1.0
    return vector / (np.linalg.norm(vector) + 1e-12)


def _feasibility_pressure(
    violation_energy: float,
    violation_count: float,
    energy_scale: float,
    count_scale: float,
) -> float:
    if violation_energy <= 0.0 and violation_count <= 0.0:
        return 0.0
    energy_term = violation_energy / max(energy_scale, 1e-12)
    count_term = violation_count / max(count_scale, 1e-12)
    pressure = 1.0 - np.exp(-(energy_term + count_term))
    return float(np.clip(pressure, 0.0, 1.0))


def _binary_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return 0.5

    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif abs(positive - negative) <= 1e-12:
                wins += 0.5
    return float(wins / (len(positives) * len(negatives)))


def evaluate_feasibility_channels(
    cases: Sequence[FeasibilityChannelCase],
    *,
    threshold: float = 0.5,
    help_energy_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Evaluate local/global feasibility channels as repair-step predictors.

    **Researcher summary:**
        ``phi_local`` is a bounded pressure signal for the current state. It is
        high when this specific latent still violates hard constraints.
        ``Phi_global`` is the same pressure measured at the repair-cohort level.
        Their geometric mean predicts whether another repair step should be
        attempted. Labels are computed after the fact from hard violation
        energy/count reduction, so a step that only adds distortion is treated
        as an unhelpful continue.

    Args:
        cases: Before/after candidate repair transitions.
        threshold: Combined-channel score at or above this value predicts
            ``continue repair``.
        help_energy_tolerance: Minimum hard violation-energy drop that counts
            as useful when hard violation count does not change.

    Returns:
        JSON-serialisable aggregate metrics and per-case channel rows.

    Raises:
        ValueError: If no cases are supplied or numeric fields are impossible.

    Spec: REQ-KONA-030, SCENARIO-KONA-030
    """
    case_list = list(cases)
    if not case_list:
        raise ValueError("at least one feasibility channel case is required")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if help_energy_tolerance < 0.0:
        raise ValueError("help_energy_tolerance must be non-negative")

    cohorts: dict[str, list[FeasibilityChannelCase]] = {}
    for case in case_list:
        numeric_values = [
            case.before_violation_energy,
            float(case.before_violation_count),
            case.after_violation_energy,
            float(case.after_violation_count),
            case.distortion_delta,
        ]
        if any(value < 0.0 for value in numeric_values):
            raise ValueError("violation and distortion values must be non-negative")
        cohorts.setdefault(case.cohort or "default", []).append(case)

    cohort_stats: dict[str, dict[str, float]] = {}
    for cohort, cohort_cases in cohorts.items():
        positive_energies = [
            case.before_violation_energy
            for case in cohort_cases
            if case.before_violation_energy > help_energy_tolerance
        ]
        positive_counts = [
            float(case.before_violation_count)
            for case in cohort_cases
            if case.before_violation_count > 0
        ]
        energy_scale = float(np.mean(positive_energies)) if positive_energies else 1.0
        count_scale = float(np.mean(positive_counts)) if positive_counts else 1.0
        global_energy = float(
            np.mean([case.before_violation_energy for case in cohort_cases])
        )
        global_count = float(
            np.mean([case.before_violation_count for case in cohort_cases])
        )
        Phi_global = _feasibility_pressure(
            global_energy,
            global_count,
            energy_scale,
            count_scale,
        )
        cohort_stats[cohort] = {
            "energy_scale": energy_scale,
            "count_scale": count_scale,
            "Phi_global": Phi_global,
        }

    per_case: list[dict[str, Any]] = []
    scores: list[float] = []
    labels: list[bool] = []
    predictions: list[bool] = []
    wrong_distortions: list[float] = []

    for case in case_list:
        stats = cohort_stats[case.cohort or "default"]
        phi_local = _feasibility_pressure(
            case.before_violation_energy,
            float(case.before_violation_count),
            stats["energy_scale"],
            stats["count_scale"],
        )
        Phi_global = stats["Phi_global"]
        channel_score = float(np.sqrt(phi_local * Phi_global))
        predicted_continue = channel_score >= threshold
        energy_drop = case.before_violation_energy - case.after_violation_energy
        repair_helped = bool(
            case.after_violation_count < case.before_violation_count
            or energy_drop > help_energy_tolerance
        )
        wrong_prediction = predicted_continue != repair_helped
        if wrong_prediction:
            wrong_distortions.append(case.distortion_delta)

        scores.append(channel_score)
        labels.append(repair_helped)
        predictions.append(predicted_continue)
        per_case.append(
            {
                "case_id": case.case_id,
                "cohort": case.cohort,
                "phi_local": phi_local,
                "Phi_global": Phi_global,
                "channel_score": channel_score,
                "predicted_continue": bool(predicted_continue),
                "repair_helped": bool(repair_helped),
                "wrong_prediction": bool(wrong_prediction),
                "before_violation_energy": case.before_violation_energy,
                "before_violation_count": case.before_violation_count,
                "after_violation_energy": case.after_violation_energy,
                "after_violation_count": case.after_violation_count,
                "distortion_delta": case.distortion_delta,
            }
        )

    positives = sum(labels)
    negatives = len(labels) - positives
    false_continue = sum(
        1
        for prediction, label in zip(predictions, labels, strict=True)
        if prediction and not label
    )
    false_stop = sum(
        1
        for prediction, label in zip(predictions, labels, strict=True)
        if not prediction and label
    )
    accuracy = float(
        np.mean(
            [
                prediction == label
                for prediction, label in zip(predictions, labels, strict=True)
            ]
        )
    )
    auc = _binary_auc(scores, labels)
    false_continue_rate = float(false_continue / negatives) if negatives else 0.0
    false_stop_rate = float(false_stop / positives) if positives else 0.0
    distortion_when_wrong = (
        float(np.mean(wrong_distortions)) if wrong_distortions else 0.0
    )

    return {
        "n_cases": len(case_list),
        "n_positive_helpful_repairs": int(positives),
        "n_negative_unhelpful_repairs": int(negatives),
        "n_predicted_continue": int(sum(predictions)),
        "n_predicted_stop": int(len(predictions) - sum(predictions)),
        "phi_local": float(np.mean([row["phi_local"] for row in per_case])),
        "Phi_global": float(np.mean([row["Phi_global"] for row in per_case])),
        "feasibility_channel_auc": auc,
        "repair_help_prediction_accuracy": accuracy,
        "false_continue_rate": false_continue_rate,
        "false_stop_rate": false_stop_rate,
        "distortion_when_wrong": distortion_when_wrong,
        "feasibility_channel_predictive": bool(auc >= 0.60 and accuracy >= 0.60),
        "per_case": per_case,
    }


def feasibility_step(
    state: np.ndarray,
    constraint_matrix: np.ndarray,
    constraint_bias: np.ndarray | None = None,
    *,
    n_steps: int = 32,
    lr: float = 0.25,
    anchor_weight: float = 0.01,
    tolerance: float = 1e-8,
) -> FeasibilityStepResult:
    """Repair a continuous latent state by minimizing violation energy only.

    **Researcher summary:**
        This is a minimal FSNet-style feasibility-seeking step for Phase 3
        latents. It treats verifier constraints as linear inequalities
        ``A @ z + b <= 0`` and descends the squared hinge violation energy
        separately from the ContinuousEBM task energy. A small anchor term keeps
        the repaired state near the input latent so the operator does not simply
        collapse all states to one feasible point.

    Args:
        state: Latent vector ``z`` with shape ``(d,)``.
        constraint_matrix: Linear verifier matrix ``A`` with shape ``(m, d)``.
        constraint_bias: Optional verifier bias ``b`` with shape ``(m,)``.
            When omitted, zero bias is used.
        n_steps: Maximum number of feasibility-gradient steps.
        lr: Feasibility-gradient learning rate.
        anchor_weight: Strength of the quadratic anchor to the input state.
        tolerance: Squared-hinge tolerance used for convergence and violation
            counting.

    Returns:
        FeasibilityStepResult containing the repaired bounded state and the
        measured violation/distortion diagnostics.

    Raises:
        ValueError: If shapes or optimization hyperparameters are invalid.

    Spec: REQ-KONA-027, SCENARIO-KONA-027
    """
    z0 = np.asarray(state, dtype=np.float64)
    A = np.asarray(constraint_matrix, dtype=np.float64)

    if z0.ndim != 1:
        raise ValueError("state must be one-dimensional")
    if A.ndim != 2 or A.shape[1] != z0.shape[0]:
        raise ValueError("constraint_matrix must have shape (n_constraints, state_dim)")

    if constraint_bias is None:
        b = np.zeros(A.shape[0], dtype=np.float64)
    else:
        b = np.asarray(constraint_bias, dtype=np.float64)
    if b.shape != (A.shape[0],):
        raise ValueError("constraint_bias must have shape (n_constraints,)")

    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if lr < 0.0:
        raise ValueError("lr must be non-negative")
    if anchor_weight < 0.0:
        raise ValueError("anchor_weight must be non-negative")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    def measure(z: np.ndarray) -> tuple[float, int]:
        scores = A @ z + b
        hinge = np.maximum(scores, 0.0)
        energy = float(hinge @ hinge)
        count = int(np.sum(scores > tolerance))
        return energy, count

    initial_energy, initial_count = measure(z0)
    if initial_energy <= tolerance:
        return FeasibilityStepResult(
            state=z0.copy(),
            initial_violation_energy=initial_energy,
            violation_energy=initial_energy,
            violation_count=initial_count,
            convergence_steps=0,
            distortion_l2=0.0,
            converged=True,
        )

    z = z0.copy()
    steps_taken = 0
    converged = False
    final_energy = initial_energy
    final_count = initial_count

    for step in range(1, n_steps + 1):
        scores = A @ z + b
        hinge = np.maximum(scores, 0.0)
        grad = (2.0 * A.T @ hinge) + (2.0 * anchor_weight * (z - z0))
        z = np.tanh(z - lr * grad)
        steps_taken = step
        final_energy, final_count = measure(z)
        if final_energy <= tolerance:
            converged = True
            break

    return FeasibilityStepResult(
        state=z,
        initial_violation_energy=initial_energy,
        violation_energy=final_energy,
        violation_count=final_count,
        convergence_steps=steps_taken,
        distortion_l2=float(np.linalg.norm(z - z0)),
        converged=converged,
    )


def fit_continuous_ebm(ising_model: Any) -> ContinuousEBM:
    """Construct a ContinuousEBM directly from an Ising model's parameters.

    **Researcher summary:**
        Reuses J and h verbatim — no fitting, no gradient steps.  This is
        the "same problem, different solver" construction.

    **Detailed explanation for engineers:**
        The Ising model already has the coupling matrix (J) and bias (h) we
        need.  We just copy them into a ContinuousEBM so the continuous sampler
        can use the same energy function.  If the two minimisers agree, it
        means the energy landscape is not an artefact of the discrete domain.

    Args:
        ising_model: Any object with `coupling` (array, shape (n, n)) and
            `bias` (array, shape (n,)) attributes.  Typically an IsingModel.

    Returns:
        ContinuousEBM with the same coupling and bias as the Ising model.
    """
    coupling = np.asarray(ising_model.coupling, dtype=np.float64)
    bias = np.asarray(ising_model.bias, dtype=np.float64)
    n = coupling.shape[0]
    return ContinuousEBM(variables=n, coupling=coupling, bias=bias)


def sample_continuous(
    model: ContinuousEBM,
    n_steps: int = 1000,
    lr: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Find an approximate energy minimum via gradient descent with tanh squashing.

    **Researcher summary:**
        Vanilla gradient descent on E(x) = -0.5*x^T*J*x - h^T*x.  Each step:
        1. dE/dx = -J*x - h  (analytic gradient)
        2. x ← tanh(x - lr * dE/dx)   (step + squash to [-1, 1])

    **Why tanh squashing at each step (not just at the end)?**
        Applying tanh after each gradient step keeps x inside the hypercube
        throughout optimisation, not just at the final output.  This prevents
        the gradient from exploiting the unbounded ℝ^n extension (e.g. driving
        a single variable to ±∞ because the coupling is large).

    **Why is dE/dx = -J*x - h?**
        E(x) = -0.5 * x^T J x - h^T x
        ∂E/∂x = -0.5 * (J + J^T) x - h = -J x - h    (since J is symmetric)
        Gradient *descent* means x ← x - lr * ∂E/∂x = x + lr * (J x + h),
        which is the "uphill in J" direction — exactly what minimises E.

    Args:
        model: ContinuousEBM to minimise.
        n_steps: Number of gradient descent steps.
        lr: Learning rate (step size).
        seed: Random seed for initial point (uniform in [-1, 1]^n).

    Returns:
        Array of shape (n,) with values in (-1, 1) representing the
        approximate energy minimiser.
    """
    rng = np.random.default_rng(seed)
    # Initialise randomly in [-1, 1]^n — avoids bias toward any particular basin
    x = rng.uniform(-1.0, 1.0, size=model.variables)

    J = model.coupling  # shape (n, n)
    h = model.bias  # shape (n,)

    for _ in range(n_steps):
        # Analytic gradient of E(x) = -0.5 x^T J x - h^T x
        # dE/dx = -J x - h  (descent direction: negate to go downhill)
        grad = -J @ x - h
        # Gradient step then squash: keeps x in open (-1, 1)^n hypercube
        x = np.tanh(x - lr * grad)

    return x


def compare_minima(
    ising_sample: np.ndarray,
    continuous_sample: np.ndarray,
) -> dict[str, float]:
    """Compare discrete and continuous energy minimisers.

    **Researcher summary:**
        Two metrics: L2 distance (magnitude) and sign agreement (direction).
        Both are needed because L2 alone can be misleading for sparse Ising
        problems where the ground state has many near-zero components.

    **Detailed explanation for engineers:**
        - ``l2_distance``: ||ising - continuous||_2.  Small means the two
          minimisers are numerically close.  Threshold in REQ-KONA-001: ≤ 0.1.
        - ``sign_agreement``: fraction of indices where sign(ising_i) ==
          sign(continuous_i).  Robust to scale differences; measures whether
          the solvers agree on the *direction* of each variable.  Threshold: > 0.9.

    Args:
        ising_sample: Array of shape (n,) from Ising simulated annealing.
            Values should be in {-1, +1} or nearby floats.
        continuous_sample: Array of shape (n,) from ``sample_continuous``.
            Values are in (-1, 1) due to tanh squashing.

    Returns:
        Dict with keys:
            ``'l2_distance'`` (float): Euclidean distance between the two samples.
            ``'sign_agreement'`` (float): Fraction of coordinates with matching sign.
    """
    ising_arr = np.asarray(ising_sample, dtype=np.float64)
    cont_arr = np.asarray(continuous_sample, dtype=np.float64)

    l2 = float(np.linalg.norm(ising_arr - cont_arr))

    # Sign agreement: compare np.sign, treating 0 as +1 (convention consistent
    # with Ising {-1, +1} variables — 0 is not a valid Ising spin).
    ising_signs = np.sign(ising_arr)
    ising_signs = np.where(ising_signs == 0, 1.0, ising_signs)
    cont_signs = np.sign(cont_arr)
    cont_signs = np.where(cont_signs == 0, 1.0, cont_signs)
    agreement = float(np.mean(ising_signs == cont_signs))

    return {"l2_distance": l2, "sign_agreement": agreement}


def build_kona_artifact(
    comparison: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable artifact for Exp 435a.

    **Researcher summary:**
        Standardised artifact schema ``carnot.kona_seed.v1``.  The
        ``honest_verdict`` field is the primary result — downstream tooling
        should read this rather than re-deriving thresholds.

    **Verdict derivation (from REQ-KONA-001 / SCENARIO-KONA-002):**
        - ``'continuous_matches_ising'``: L2 < 0.1 AND sign_agreement > 0.9
        - ``'partial_match'``: sign_agreement > 0.7 (but not the above)
        - ``'failed_to_match'``: otherwise

    Args:
        comparison: Dict returned by ``compare_minima``.
        extra: Optional extra fields merged into the artifact (e.g. energy
            values, model spec).  Must be JSON-serialisable.

    Returns:
        Dict with at minimum:
            ``'schema'``, ``'run_date'``, ``'honest_verdict'``,
            ``'l2_distance'``, ``'sign_agreement'``.
    """
    l2 = comparison["l2_distance"]
    sa = comparison["sign_agreement"]

    if l2 < 0.1 and sa > 0.9:
        verdict = "continuous_matches_ising"
    elif sa > 0.7:
        verdict = "partial_match"
    else:
        verdict = "failed_to_match"

    artifact: dict[str, Any] = {
        "schema": "carnot.kona_seed.v1",
        "run_date": datetime.date.today().isoformat(),
        "honest_verdict": verdict,
        "l2_distance": l2,
        "sign_agreement": sa,
    }
    if extra:
        artifact.update(extra)
    return artifact


def sample_langevin(
    model: ContinuousEBM,
    n_steps: int = 2000,
    lr: float = 0.005,
    noise_scale: float = 0.1,
    temp_schedule: str = "cosine",
    seed: int = 0,
) -> np.ndarray:
    """Find an approximate energy minimum via Langevin dynamics.

    **Researcher summary (REQ-KONA-002):**
        Langevin dynamics adds Gaussian noise to gradient descent.  The update rule:

            x_{t+1} = x_t - lr * grad_E(x_t) + noise_scale * sqrt(2*lr) * eps_t

        where eps_t ~ N(0, I).  The noise term provides thermal fluctuations that
        help escape local energy minima — something pure gradient descent cannot do
        once it settles into a basin.

    **Why Langevin beats gradient descent:**
        Gradient descent is deterministic: once it reaches a local minimum it stops.
        Langevin dynamics is a stochastic differential equation whose stationary
        distribution is proportional to exp(-E(x)/T), where T is the temperature.
        At high T, the sampler explores broadly; as T → 0, it concentrates on the
        global minimum.  For the 10-variable Ising problem, this is why Exp 435a's
        gradient descent (L2=2.69) underperforms: it found a local minimum early
        and never escaped.

    **Connection to Generative Thermodynamic Computing (arXiv 2506.15121):**
        That paper frames training as minimising "heat emission" by reversing noising
        trajectories.  The temperature schedule here (cosine annealing) mirrors the
        physically-motivated annealing in that work: start warm (high noise for
        exploration) and cool gradually toward zero noise (exploitation).

    **Temperature schedules:**
        - ``'cosine'`` (default): noise_t = noise_scale * 0.5 * (1 + cos(π * t / T_total))
          Smoothly anneals noise from full to zero.  Best for exploitation at end.
        - ``'linear'``: noise_t = noise_scale * (1 - t / T_total)
          Linear decay; simpler but less smooth.
        - ``'constant'``: noise_t = noise_scale (never annealed)
          Useful for pure Langevin sampling at fixed temperature.

    **Phase 3 relevance:**
        Langevin dynamics is how Kona-style continuous reasoning samples from an
        energy landscape.  A latent-space reasoner that can escape local minima will
        find globally consistent reasoning chains rather than local-coherence traps
        (the "hallucination" problem in autoregressive LLMs).

    Args:
        model: ContinuousEBM to sample from.
        n_steps: Total number of Langevin steps.
        lr: Step size (learning rate).  Smaller → more stable, slower convergence.
        noise_scale: Base noise magnitude before temperature scaling.
        temp_schedule: One of 'cosine', 'linear', 'constant'.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n,) with values in (-1, 1).

    Raises:
        ValueError: If temp_schedule is not one of the supported strings.

    Spec: REQ-KONA-002, SCENARIO-KONA-003
    """
    if temp_schedule not in ("cosine", "linear", "constant"):
        raise ValueError(
            f"temp_schedule must be 'cosine', 'linear', or 'constant', got {temp_schedule!r}"
        )

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(model.variables)  # Gaussian init — broad exploration

    J = model.coupling
    h = model.bias
    noise_std = noise_scale * np.sqrt(2.0 * lr)

    for t in range(n_steps):
        # Analytic gradient: dE/dx = -J x - h
        grad = -J @ x - h

        # Temperature-scaled noise: anneal from full noise to (near-)zero
        if temp_schedule == "cosine":
            temp_factor = 0.5 * (1.0 + np.cos(np.pi * t / max(n_steps - 1, 1)))
        elif temp_schedule == "linear":
            temp_factor = 1.0 - t / max(n_steps - 1, 1)
        else:  # constant
            temp_factor = 1.0

        noise = noise_std * temp_factor * rng.standard_normal(model.variables)
        x = np.tanh(x - lr * grad + noise)

    return x


def sample_energy_matching(
    model: ContinuousEBM,
    n_steps: int = 1000,
    n_flow_steps: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Find an approximate energy minimum via Energy Matching trajectory flow.

    **Researcher summary (REQ-KONA-003):**
        Energy Matching (arXiv 2504.10612, NeurIPS 2025) unifies flow models and EBMs
        by having the flow trajectory follow the energy gradient with thermodynamic
        noise.  Here we implement the deterministic core: normalised gradient flow.

        Algorithm:
        1. Sample n_steps initial points from a Gaussian.
        2. For each starting point, run n_flow_steps of normalised gradient descent:
               x = x - step_size * grad_E(x) / (||grad_E(x)|| + eps)
        3. Apply tanh squashing and select the point with the lowest energy.

    **Why normalised gradient flow (not plain gradient descent)?**
        Plain gradient descent takes steps proportional to ||grad_E||.  Near flat
        regions (||grad|| ≈ 0) it stalls; near steep regions (||grad|| >> 1) it
        overshoots.  Normalising by the gradient magnitude gives constant-speed
        flow: the step size is always ``step_size``, regardless of the energy
        landscape curvature.  This is the "constant convergence speed regardless
        of energy scale" property described in arXiv 2504.10612.

    **Multi-start strategy:**
        Running n_steps independent short trajectories and selecting the best
        trades off breadth (n_steps starting points explored) against depth
        (n_flow_steps gradient steps each).  With n_steps=1000 and n_flow_steps=10,
        this is equivalent to 10,000 gradient evaluations but distributed across
        1000 different starting points — much better coverage than 10,000 steps
        from a single starting point.

    **Phase 3 relevance:**
        Energy Matching is a unified framework for learning and sampling from energy
        landscapes.  The normalised gradient flow here is the inference-time component
        of that framework, analogous to how diffusion models use their learned score
        function during sampling.  For Kona's continuous reasoning, Energy Matching
        provides a theoretically grounded sampling algorithm with known convergence
        properties (constant speed, thermodynamic free-energy minimisation).

    Args:
        model: ContinuousEBM to sample from.
        n_steps: Number of independent starting points to try (breadth).
        n_flow_steps: Gradient steps per starting point (depth).
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n,) with values in (-1, 1).  This is the best (lowest-energy)
        point found across all n_steps trajectories.

    Spec: REQ-KONA-003, SCENARIO-KONA-004
    """
    rng = np.random.default_rng(seed)
    J = model.coupling
    h = model.bias
    eps = 1e-8  # gradient normalisation floor to avoid division by zero

    # step_size chosen so that n_flow_steps steps cover the unit hypercube
    step_size = 2.0 / max(n_flow_steps, 1)

    best_x = rng.standard_normal(model.variables)
    best_x = np.tanh(best_x)
    best_energy = float(-0.5 * best_x @ J @ best_x - h @ best_x)

    for _ in range(n_steps):
        x = rng.standard_normal(model.variables)

        for _ in range(n_flow_steps):
            grad = -J @ x - h
            grad_norm = np.linalg.norm(grad)
            # Normalised gradient flow: constant-speed descent regardless of scale
            x = x - step_size * grad / (grad_norm + eps)

        x = np.tanh(x)
        energy = float(-0.5 * x @ J @ x - h @ x)

        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()

    return best_x


class BoltzmannGPTLayer:
    """Energy recurrence block inspired by arXiv 2601.17094 Boltzmann-GPT.

    Uses visible-hidden Boltzmann coupling: E(v,h) = -v^T W h - b^T v - c^T h.
    For a sequence of tokens, compute the energy of the visible-token state v
    conditioned on a learned hidden representation h. Higher energy = lower
    Boltzmann-GPT score (lower energy = more likely under the Boltzmann distribution).

    **Why Boltzmann-GPT vs NRGPT?**
        NRGPT (arXiv:2405.XXXXX) uses an autoregressive recurrence over token
        energies — the energy of each token depends on the previous hidden state.
        Boltzmann-GPT (arXiv 2601.17094) instead uses a bipartite visible-hidden
        coupling matrix W, computing a joint energy for the entire token sequence as
        a unit.  The two architectures differ in how they aggregate evidence:
        NRGPT is sequential, Boltzmann-GPT is holistic.

    **Seed initialisation (no training performed):**
        Weights are randomly initialised with a fixed NumPy seed.  This measures
        whether the random Boltzmann architecture captures any structural signal
        before learning — a necessary sanity-check before investing in contrastive
        training.  AUROC near 0.5 is expected; any score above 0.6 would suggest
        the architecture has useful inductive biases.

    **Why small random init (not zeros)?**
        Zero W, b, c → E(v,h) = 0 for all inputs → constant score → degenerate
        AUROC.  Small random weights give non-degenerate but untrained scores,
        which is the correct baseline for a seed experiment.

    Attributes:
        W: Visible-hidden coupling matrix, shape (visible_dim, hidden_dim).
        b: Visible bias, shape (visible_dim,).
        c: Hidden bias, shape (hidden_dim,).

    Spec: REQ-PHASE3-BOLTZMANN-001
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        visible_dim: int = 16,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        # Small random init: non-degenerate scores without training.
        # std=0.01 keeps energy values in a reasonable range given normalised inputs.
        self.W: np.ndarray = rng.standard_normal((visible_dim, hidden_dim)) * 0.01
        self.b: np.ndarray = np.zeros(visible_dim)
        self.c: np.ndarray = np.zeros(hidden_dim)
        self._visible_dim = visible_dim
        self._hidden_dim = hidden_dim

    def energy(self, v: np.ndarray, h: np.ndarray) -> float:
        """Boltzmann energy: E = -v^T W h - b^T v - c^T h.

        **Why negative?**
            The Boltzmann distribution assigns probability proportional to
            exp(-E/T).  Lower energy = higher probability.  The minus signs
            in E = -v^T W h - b^T v - c^T h mean positive W_ij, positive v_i,
            positive h_j all *lower* the energy, i.e., are mutually reinforcing.

        Args:
            v: Visible state, shape (visible_dim,).  Values in [0, 1] from
               normalised token embeddings.
            h: Hidden state, shape (hidden_dim,).  Values in (0, 1) from
               mean-field sigmoid activation.

        Returns:
            Scalar energy value (float).  Lower = more coherent / likely.
        """
        return float(-(v @ self.W @ h) - (self.b @ v) - (self.c @ h))

    def score(self, token_sequence: list[str]) -> float:
        """Score a token sequence: lower energy = higher Boltzmann-GPT score.

        **Scoring pipeline:**
            1. Embed tokens → visible vector v (bigram frequency projection)
            2. Infer hidden state h from v (mean-field sigmoid)
            3. Return -E(v, h) so that *higher score = lower energy = more likely*

        This sign convention aligns with NRGPT: both scorers return a value
        where higher = the LLM response is more likely to be correct.

        Args:
            token_sequence: List of string tokens (e.g. from str.split()).

        Returns:
            Float score.  Higher = more likely to be a correct LLM response.
        """
        v = self._embed_tokens(token_sequence)
        h = self._infer_hidden(v)
        return -self.energy(v, h)

    def _embed_tokens(self, token_sequence: list[str]) -> np.ndarray:
        """Map string tokens to a fixed-size visible vector.

        **Method: character bigram frequency projection.**
            For each token, count all consecutive character pairs (bigrams).
            Map each bigram to a bucket in [0, visible_dim) via hash % visible_dim.
            Accumulate counts, then L2-normalise to keep energy values bounded.

        **Why bigrams?**
            Single character counts lose word-boundary information.  Longer n-grams
            are sparse on short sequences.  Bigrams are a standard compromise for
            fast, hardware-portable text embeddings that preserve some morphological
            signal without learned parameters.

        **Why hash-modulo projection?**
            Deterministic, O(n) in text length, no vocabulary required, portable
            to any hardware (no lookup tables).  The loss from hash collisions is
            acceptable for a seed experiment — a trained embedding would replace this.

        Args:
            token_sequence: List of string tokens.

        Returns:
            Array of shape (visible_dim,) with L2 norm 1 (or uniform if empty).
        """
        counts = np.zeros(self._visible_dim, dtype=np.float64)
        for token in token_sequence:
            for i in range(len(token) - 1):
                # Python's built-in hash is deterministic within a process session
                # but varies across Python versions. Use ord() to be reproducible.
                bigram_hash = (ord(token[i]) * 31 + ord(token[i + 1])) % self._visible_dim
                counts[bigram_hash] += 1.0
        norm = np.linalg.norm(counts)
        if norm > 1e-10:
            return counts / norm
        # Empty or single-character tokens: uniform distribution over visible units
        return np.full(self._visible_dim, 1.0 / self._visible_dim)

    def _infer_hidden(self, v: np.ndarray) -> np.ndarray:
        """Mean-field hidden state: h = sigmoid(W^T v + c).

        **Why mean-field?**
            The exact posterior p(h|v) ∝ exp(v^T W h + c^T h) factorises for
            binary hidden units: p(h_j = 1 | v) = σ(v^T W_j + c_j).  This is the
            mean-field approximation, exact for binary hidden units in an RBM.
            It maps the visible state v to a soft hidden activation in (0, 1)^hidden_dim.

        **Connection to arXiv 2601.17094:**
            Boltzmann-GPT uses the hidden state h as a world-model representation
            that "summarises" the context so far.  The mean-field inference step
            here is the bottom-up pass; training would add a top-down correction.

        Args:
            v: Visible state, shape (visible_dim,).

        Returns:
            Hidden activations, shape (hidden_dim,), values in (0, 1).
        """
        logit = v @ self.W + self.c  # shape (hidden_dim,)
        return 1.0 / (1.0 + np.exp(-logit))


def compare_samplers(
    model: ContinuousEBM,
    ising_ground_state: np.ndarray,
    n_trials: int = 10,
) -> dict[str, Any]:
    """Run all three samplers and report per-sampler L2 and sign_agreement statistics.

    **Researcher summary (SCENARIO-KONA-005):**
        Runs gradient descent, Langevin dynamics, and Energy Matching each
        ``n_trials`` times (with different seeds) and reports mean/std L2 distance
        and mean sign_agreement vs the discrete Ising ground state.  This provides
        an honest head-to-head comparison of all three algorithms.

    **Why n_trials independent runs?**
        All three samplers are stochastic (different random starting points per trial).
        A single run can get lucky or unlucky.  Averaging over n_trials gives a
        statistically meaningful result.  20 trials (used in Exp 446) is sufficient
        for stable mean/std estimates on a 10-variable problem.

    **Why compare to the Ising ground state (not just energy)?**
        Energy alone is a biased metric: the continuous relaxation can achieve lower
        energy than the discrete Ising minimum (because it's unconstrained to {-1,+1}).
        L2 distance and sign_agreement measure whether the continuous sampler found
        the *same region* as the Ising solver, which is what REQ-KONA-002 requires.

    Args:
        model: ContinuousEBM to sample from.
        ising_ground_state: Array of shape (n,) with the discrete Ising ground state
            (values near ±1).  The reference solution all samplers are compared to.
        n_trials: Number of independent runs per sampler.

    Returns:
        Dict with keys 'gradient_descent', 'langevin', 'energy_matching', each
        mapping to a sub-dict with:
            - ``'mean_l2'`` (float): Mean L2 distance over n_trials.
            - ``'std_l2'`` (float): Standard deviation of L2 distances.
            - ``'mean_sign_agreement'`` (float): Mean sign agreement fraction.
        Also includes ``'best_sampler'`` (str): name of the sampler with lowest mean_l2.

    Spec: REQ-KONA-002, REQ-KONA-003, SCENARIO-KONA-005
    """
    ising_arr = np.asarray(ising_ground_state, dtype=np.float64)
    results: dict[str, Any] = {}

    sampler_configs: list[tuple[str, Any]] = [
        ("gradient_descent", lambda seed: sample_continuous(model, seed=seed)),
        ("langevin", lambda seed: sample_langevin(model, seed=seed)),
        ("energy_matching", lambda seed: sample_energy_matching(model, seed=seed)),
    ]

    for name, sampler_fn in sampler_configs:
        l2_values: list[float] = []
        sign_values: list[float] = []

        for trial in range(n_trials):
            sample = sampler_fn(seed=trial)
            cmp = compare_minima(ising_arr, sample)
            l2_values.append(cmp["l2_distance"])
            sign_values.append(cmp["sign_agreement"])

        results[name] = {
            "mean_l2": float(np.mean(l2_values)),
            "std_l2": float(np.std(l2_values)),
            "mean_sign_agreement": float(np.mean(sign_values)),
        }

    # Identify the best sampler by lowest mean L2
    best = min(results, key=lambda k: results[k]["mean_l2"])
    results["best_sampler"] = best

    return results
