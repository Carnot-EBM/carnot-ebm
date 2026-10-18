"""EBM-CoT consistency calibration over synthetic reasoning traces.

This module is a CPU-only, pure-NumPy implementation of the Phase-4
free-energy verifier probe in Exp 2358. It treats a chain-of-thought trace as
the sampled object: low energy means adjacent reasoning steps preserve the same
claim polarity, while high energy means an adjacent step negates a claim that
the previous step asserted.

Spec: REQ-VERIFY-2358, SCENARIO-VERIFY-2358.
"""

from __future__ import annotations

import datetime as _datetime
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TERMINAL_VERDICT_PREFIXES = ("complete:", "success:", "blocked:", "failed:")
RANDOM_SEED = 42
N_TRACES = 50
N_CONSISTENT = 25
N_INCONSISTENT = 25
N_REFINED_TRACES = 5
N_LANGEVIN_STEPS = 50
OUTPUT_FILE = "experiment_2358_ebm_cot.json"

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "ebm_cot_validated",
    "ebm_cot_auroc",
    "energy_reduction_mean",
    "n_traces",
    "random_seed",
}

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix required.",
    "ebm_cot_validated": "True if AUROC >= 0.60 on synthetic CoT corpus.",
    "ebm_cot_auroc": "AUROC on consistent vs inconsistent CoT discrimination.",
    "energy_reduction_mean": (
        "Mean energy reduction after 50 Langevin steps on inconsistent traces."
    ),
    "n_traces": "Must be 50.",
    "random_seed": "Reproducibility. Must be 42.",
}

_CLAUSE_SPLIT_RE = re.compile(r"[.;]|\b(?:therefore|because|hence|then|so)\b|,")
_STEP_PREFIX_RE = re.compile(r"^\s*(?:step\s*)?\d+\s*[:.)-]\s*", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_SPACE_RE = re.compile(r"\s+")
_CONTRACTION_REPLACEMENTS = (
    (re.compile(r"\bisn't\b", re.IGNORECASE), "is not"),
    (re.compile(r"\baren't\b", re.IGNORECASE), "are not"),
    (re.compile(r"\bwasn't\b", re.IGNORECASE), "was not"),
    (re.compile(r"\bweren't\b", re.IGNORECASE), "were not"),
    (re.compile(r"\bdoesn't\b", re.IGNORECASE), "does not"),
    (re.compile(r"\bdon't\b", re.IGNORECASE), "do not"),
    (re.compile(r"\bdidn't\b", re.IGNORECASE), "did not"),
    (re.compile(r"\bcan't\b", re.IGNORECASE), "cannot"),
    (re.compile(r"\bwon't\b", re.IGNORECASE), "will not"),
)
_NEGATION_REPAIRS = (
    (re.compile(r"\bis\s+not\b", re.IGNORECASE), "is"),
    (re.compile(r"\bare\s+not\b", re.IGNORECASE), "are"),
    (re.compile(r"\bwas\s+not\b", re.IGNORECASE), "was"),
    (re.compile(r"\bwere\s+not\b", re.IGNORECASE), "were"),
    (re.compile(r"\bhas\s+no\b", re.IGNORECASE), "has"),
    (re.compile(r"\bhave\s+no\b", re.IGNORECASE), "have"),
    (re.compile(r"\bdoes\s+not\b", re.IGNORECASE), ""),
    (re.compile(r"\bdo\s+not\b", re.IGNORECASE), ""),
    (re.compile(r"\bdid\s+not\b", re.IGNORECASE), ""),
    (re.compile(r"\bcannot\b", re.IGNORECASE), "can"),
    (re.compile(r"\bcan\s+not\b", re.IGNORECASE), "can"),
    (re.compile(r"\bwill\s+not\b", re.IGNORECASE), "will"),
    (re.compile(r"\bmust\s+not\b", re.IGNORECASE), "must"),
    (re.compile(r"\bnever\b", re.IGNORECASE), ""),
    (re.compile(r"\bnot\b", re.IGNORECASE), ""),
    (re.compile(r"\bno\b", re.IGNORECASE), ""),
)
_NEGATION_PATTERNS = tuple(pattern for pattern, _replacement in _NEGATION_REPAIRS)
_FILLER_TOKENS = {
    "a",
    "an",
    "claim",
    "given",
    "observation",
    "step",
    "that",
    "the",
    "thus",
    "we",
    "know",
}


@dataclass
class EbmCotCalibrator:
    """Heuristic EBM-CoT calibrator with seeded discrete Langevin refinement.

    `energy()` counts adjacent polarity contradictions between normalized
    claims. `langevin_refine()` samples one trace element per step, proposes a
    local text perturbation, and accepts the proposal only when it lowers the
    trace energy.

    Spec: REQ-VERIFY-2358.
    """

    seed: int = RANDOM_SEED
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def energy(self, trace: list[str]) -> float:
        """Return adjacent-step inconsistency energy for a CoT trace."""
        if len(trace) < 2:
            return 0.0

        polarized = [_extract_polarized_claims(step) for step in trace]
        total = 0
        for left, right in zip(polarized, polarized[1:]):
            left_pos, left_neg = left
            right_pos, right_neg = right
            contradictions = (left_pos & right_neg) | (left_neg & right_pos)
            total += len(contradictions)
        return float(total)

    def calibrate(self, traces: list[list[str]]) -> list[float]:
        """Return one energy score per trace, preserving input order."""
        return [self.energy(trace) for trace in traces]

    def langevin_refine(self, trace: list[str], n_steps: int = N_LANGEVIN_STEPS) -> list[str]:
        """Run discrete low-energy trace proposals for `n_steps`.

        The update is a text-space analogue of Langevin refinement: randomly
        select one trace element, perturb it, and accept only energy-decreasing
        proposals.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        current = list(trace)
        if not current:
            return current

        current_energy = self.energy(current)
        for _ in range(n_steps):
            index = int(self._rng.integers(0, len(current)))
            proposal = list(current)
            proposal[index] = self._propose_step(proposal, index)
            proposal_energy = self.energy(proposal)
            if proposal_energy < current_energy:
                current = proposal
                current_energy = proposal_energy
        return current

    def _propose_step(self, trace: list[str], index: int) -> str:
        repaired = _repair_step_against_neighbors(trace, index)
        if repaired != trace[index]:
            return repaired
        return _paraphrase_step(trace[index], self._rng)


def build_synthetic_cot_corpus(random_seed: int = RANDOM_SEED) -> tuple[list[list[str]], list[int]]:
    """Build 25 consistent and 25 adjacent-contradiction CoT traces.

    Labels use the AUROC convention requested by the task: consistent traces
    are positive (`1`), inconsistent traces are negative (`0`).
    """
    traces: list[list[str]] = []
    labels: list[int] = []

    for index in range(N_CONSISTENT):
        subject = f"case_{index:02d}"
        downstream = f"downstream_{index:02d}"
        conclusion = f"final_{index:02d}"
        traces.append(
            [
                f"Step 1: Claim: {subject} is stable.",
                f"Step 2: Claim: {subject} is stable, so {downstream} is accepted.",
                f"Step 3: Claim: {downstream} is accepted, so {conclusion} is accepted.",
            ]
        )
        labels.append(1)

    for index in range(N_INCONSISTENT):
        subject = f"case_{index:02d}"
        downstream = f"downstream_{index:02d}"
        conclusion = f"final_{index:02d}"
        traces.append(
            [
                f"Step 1: Claim: {subject} is stable.",
                f"Step 2: Claim: {subject} is not stable, so {downstream} is rejected.",
                f"Step 3: Claim: {downstream} is rejected, so {conclusion} is rejected.",
            ]
        )
        labels.append(0)

    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(traces))
    return [traces[int(i)] for i in order], [labels[int(i)] for i in order]


def consistency_auroc(labels: list[int], energies: list[float]) -> float:
    """Compute AUROC with consistent traces as the positive class.

    Low energy means positive, so AUROC is computed over score `-energy`.
    Ties receive half credit.
    """
    if len(labels) != len(energies):
        raise ValueError("labels and energies must have the same length")

    positive_scores = [-energy for label, energy in zip(labels, energies) if label == 1]
    negative_scores = [-energy for label, energy in zip(labels, energies) if label == 0]
    if not positive_scores or not negative_scores:
        raise ValueError("AUROC requires at least one positive and one negative label")

    wins = 0.0
    total = 0
    for positive in positive_scores:
        for negative in negative_scores:
            total += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return float(wins / total)


def build_experiment_2358_artifact(random_seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Run the synthetic EBM-CoT calibration probe and return its artifact."""
    traces, labels = build_synthetic_cot_corpus(random_seed=random_seed)
    calibrator = EbmCotCalibrator(seed=random_seed)
    energies = calibrator.calibrate(traces)
    auroc = consistency_auroc(labels, energies)

    inconsistent = [trace for trace, label in zip(traces, labels) if label == 0]
    refinement_rows = []
    reductions = []
    for trace in inconsistent[:N_REFINED_TRACES]:
        before = calibrator.energy(trace)
        refined = calibrator.langevin_refine(trace, n_steps=N_LANGEVIN_STEPS)
        after = calibrator.energy(refined)
        reduction = before - after
        reductions.append(reduction)
        refinement_rows.append(
            {
                "before_energy": before,
                "after_energy": after,
                "energy_reduction": reduction,
                "refined_trace": refined,
            }
        )

    energy_reduction_mean = float(np.mean(reductions)) if reductions else 0.0
    consistent_energies = [energy for energy, label in zip(energies, labels) if label == 1]
    inconsistent_energies = [energy for energy, label in zip(energies, labels) if label == 0]
    validated = auroc >= 0.60

    artifact: dict[str, Any] = {
        "schema": "carnot.experiment_2358_ebm_cot.v1",
        "experiment": "2358_ebm_cot",
        "status": "success" if validated else "failed",
        "run_date": _datetime.date.today().strftime("%Y%m%d"),
        "spec_refs": ["REQ-VERIFY-2358", "SCENARIO-VERIFY-2358"],
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": _honest_verdict(validated, auroc, energy_reduction_mean),
        "ebm_cot_validated": validated,
        "ebm_cot_auroc": auroc,
        "energy_reduction_mean": energy_reduction_mean,
        "n_traces": len(traces),
        "n_consistent": len(consistent_energies),
        "n_inconsistent": len(inconsistent_energies),
        "n_refined_traces": len(refinement_rows),
        "langevin_steps": N_LANGEVIN_STEPS,
        "random_seed": random_seed,
        "energy_summary": {
            "consistent_mean": float(np.mean(consistent_energies)),
            "inconsistent_mean": float(np.mean(inconsistent_energies)),
            "consistent_min": float(np.min(consistent_energies)),
            "consistent_max": float(np.max(consistent_energies)),
            "inconsistent_min": float(np.min(inconsistent_energies)),
            "inconsistent_max": float(np.max(inconsistent_energies)),
        },
        "refinement_results": refinement_rows,
    }
    validate_experiment_2358_artifact(artifact)
    return _json_ready(artifact)


def validate_experiment_2358_artifact(artifact: dict[str, Any]) -> None:
    """Validate the required Exp 2358 artifact fields and gates."""
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")

    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith(TERMINAL_VERDICT_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if artifact["n_traces"] != N_TRACES:
        raise ValueError(f"n_traces must be {N_TRACES}")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError(f"random_seed must be {RANDOM_SEED}")

    auroc = float(artifact["ebm_cot_auroc"])
    if not 0.0 <= auroc <= 1.0:
        raise ValueError("ebm_cot_auroc must be in [0, 1]")
    if bool(artifact["ebm_cot_validated"]) != (auroc >= 0.60):
        raise ValueError("ebm_cot_validated must equal AUROC >= 0.60")
    if float(artifact["energy_reduction_mean"]) < 0.0:
        raise ValueError("energy_reduction_mean must be non-negative")


def write_experiment_2358_artifact(
    output_path: str | Path = Path("results") / OUTPUT_FILE,
) -> dict[str, Any]:
    """Write `results/experiment_2358_ebm_cot.json` and return the artifact."""
    artifact = build_experiment_2358_artifact(random_seed=RANDOM_SEED)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _extract_polarized_claims(text: str) -> tuple[set[str], set[str]]:
    expanded = _expand_contractions(text)
    positives: set[str] = set()
    negatives: set[str] = set()
    for raw_clause in _CLAUSE_SPLIT_RE.split(expanded):
        clause = _normalize_clause(raw_clause)
        if not clause:
            continue
        is_negative = _contains_negation(clause)
        canonical = _canonicalize_claim(_remove_negation(clause) if is_negative else clause)
        if not canonical:
            continue
        if is_negative:
            negatives.add(canonical)
        else:
            positives.add(canonical)
    return positives, negatives


def _expand_contractions(text: str) -> str:
    expanded = text
    for pattern, replacement in _CONTRACTION_REPLACEMENTS:
        expanded = pattern.sub(replacement, expanded)
    return expanded


def _contains_negation(clause: str) -> bool:
    return any(pattern.search(clause) for pattern in _NEGATION_PATTERNS)


def _remove_negation(clause: str) -> str:
    repaired = clause
    for pattern, replacement in _NEGATION_REPAIRS:
        repaired = pattern.sub(replacement, repaired)
    return _SPACE_RE.sub(" ", repaired).strip()


def _normalize_clause(clause: str) -> str:
    cleaned = _STEP_PREFIX_RE.sub("", clause.strip().lower())
    cleaned = re.sub(r"^\s*(?:claim|observation|given)\s*:\s*", "", cleaned)
    return _SPACE_RE.sub(" ", cleaned).strip()


def _canonicalize_claim(clause: str) -> str:
    tokens = []
    for token in _TOKEN_RE.findall(clause.lower()):
        if token in _FILLER_TOKENS:
            continue
        tokens.append(_stem_token(token))
    return " ".join(tokens)


def _stem_token(token: str) -> str:
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _repair_step_against_neighbors(trace: list[str], index: int) -> str:
    step_pos, step_neg = _extract_polarized_claims(trace[index])
    for neighbor_index in (index - 1, index + 1):
        if neighbor_index < 0 or neighbor_index >= len(trace):
            continue
        neighbor_pos, _neighbor_neg = _extract_polarized_claims(trace[neighbor_index])
        if step_neg & neighbor_pos:
            return _remove_first_negation(trace[index])
    return trace[index]


def _remove_first_negation(text: str) -> str:
    expanded = _expand_contractions(text)
    for pattern, replacement in _NEGATION_REPAIRS:
        if pattern.search(expanded):
            repaired = pattern.sub(replacement, expanded, count=1)
            return _SPACE_RE.sub(" ", repaired).strip()
    return text


def _paraphrase_step(text: str, rng: np.random.Generator) -> str:
    replacements = (
        (re.compile(r"\bClaim\b"), "Observation"),
        (re.compile(r"\btherefore\b", re.IGNORECASE), "thus"),
        (re.compile(r"\bso\b", re.IGNORECASE), "therefore"),
    )
    order = rng.permutation(len(replacements))
    for replacement_index in order:
        pattern, replacement = replacements[int(replacement_index)]
        if pattern.search(text):
            return pattern.sub(replacement, text, count=1)
    return text


def _honest_verdict(validated: bool, auroc: float, energy_reduction_mean: float) -> str:
    if validated:
        return (
            "complete: EBM-CoT synthetic calibration validated "
            f"(AUROC={auroc:.3f}, mean energy reduction={energy_reduction_mean:.3f})"
        )
    return (
        "failed: EBM-CoT synthetic calibration did not meet AUROC gate "
        f"(AUROC={auroc:.3f}, mean energy reduction={energy_reduction_mean:.3f})"
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    write_experiment_2358_artifact()
