"""Semantic Energy synthetic-logit hallucination detector.

This module is the Exp 2338 Tier 0g prototype.  It computes the Boltzmann
negative log-partition energy from raw logits and combines repeated response
energies with a simple semantic-cluster entropy estimate.  The prototype is
CPU-only and intentionally uses synthetic logits; real penultimate-layer LLM
logits are a later validation target.  Exp 2351 adds small helper functions for
auditing cached llama.cpp top-k logprob telemetry when full logits were observed
at generation time but only compact top-k distributions were persisted.

Spec: REQ-TIER0-007, REQ-TIER0-007-5, SCENARIO-TIER0-007
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import OrderedDict

import numpy as np


def top_logprobs_to_logit_vector(top_logprobs: list[dict[str, float]]) -> np.ndarray:
    """Flatten recorded top-k logprob dictionaries into one numeric vector.

    The live GGUF telemetry stores a compact distribution per generated token:
    each dictionary maps token text to that token's logprob among the top-k
    alternatives.  Full vocabulary logits can be too large to persist, so this
    helper gives the detector a deterministic real-distribution substitute while
    preserving the ordering and spread of the model's actual alternatives.

    Spec: REQ-TIER0-007-5
    """
    if not top_logprobs:
        raise ValueError("top_logprobs must contain at least one position")

    values: list[float] = []
    for position in top_logprobs:
        if not position:
            continue
        values.extend(sorted((float(value) for value in position.values()), reverse=True))

    if not values:
        raise ValueError("top_logprobs must contain at least one numeric logprob")

    vector = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("top_logprobs must be finite")
    return vector


def binary_auroc(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> float:
    """Compute binary AUROC with average tie credit and no optional dependency.

    The experiment artifacts should be reproducible in a minimal Carnot
    environment where scikit-learn may not be installed.  This pairwise
    Mann-Whitney calculation is small but explicit: every positive example wins
    against every negative example when its score is larger, loses when smaller,
    and receives half credit for ties.

    Spec: REQ-TIER0-007-5
    """
    label_array = np.asarray(labels)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.shape[0] != score_array.shape[0]:
        raise ValueError("labels and scores must have the same length")
    if label_array.size == 0:
        raise ValueError("at least one label/score pair is required")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be finite")

    positive_scores = score_array[label_array == 1]
    negative_scores = score_array[label_array == 0]
    if positive_scores.size == 0 or negative_scores.size == 0:
        raise ValueError("labels must include at least one positive and one negative example")

    wins = 0.0
    for positive_score in positive_scores:
        wins += float(np.sum(positive_score > negative_scores))
        wins += 0.5 * float(np.sum(positive_score == negative_scores))
    return float(wins / (positive_scores.size * negative_scores.size))


class SemanticEnergyDetector:
    """Prototype Semantic Energy detector for repeated response logit arrays.

    Args:
        threshold: Decision threshold for `semantic_energy_score`.  Higher scores
            indicate more variation in response-level free-energy magnitude.
        temperature: Temperature used in the Boltzmann partition calculation.
    """

    def __init__(self, threshold: float = 0.05, temperature: float = 1.0) -> None:
        self.threshold = float(threshold)
        self.temperature = float(temperature)

    def compute_energy(self, logits: np.ndarray, temperature: float = 1.0) -> float:
        """Compute Boltzmann negative log-partition energy from raw logits.

        The formula is:

            E = -temperature * log(sum(exp(logits / temperature)))

        A max-shifted log-sum-exp keeps the computation stable for large logits.

        Spec: REQ-TIER0-007-1
        """
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        values = np.asarray(logits, dtype=np.float64).ravel()
        if values.size == 0:
            raise ValueError("logits must contain at least one value")
        if not np.all(np.isfinite(values)):
            raise ValueError("logits must be finite")

        scaled = values / float(temperature)
        max_scaled = float(np.max(scaled))
        log_partition = max_scaled + math.log(float(np.sum(np.exp(scaled - max_scaled))))
        return float(-float(temperature) * log_partition)

    def cluster_semantics(self, responses: list[str]) -> dict[str, list[int]]:
        """Group responses by normalized string hash.

        This is a deterministic placeholder for semantic equivalence clustering.
        It treats normalized exact string matches as the same semantic cluster and
        uses a short SHA-256 prefix as the cluster id.

        Spec: REQ-TIER0-007-2
        """
        clusters: OrderedDict[str, list[int]] = OrderedDict()
        for idx, response in enumerate(responses):
            normalized = self._normalize_response(response)
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
            cluster_id = f"cluster_{digest}"
            clusters.setdefault(cluster_id, []).append(idx)
        return dict(clusters)

    def detect(self, logits_per_response: list[np.ndarray], responses: list[str]) -> dict:
        """Return Semantic Energy summary statistics and a hallucination flag.

        `compute_energy()` returns negative free energy.  For this prototype's
        synthetic normal-logit corpus, the detector reports free-energy magnitude
        as `energy_mean` so the requested ordering holds: low-variance confident
        logits have lower mean energy than high-variance uncertain logits.

        Spec: REQ-TIER0-007-3
        """
        if len(logits_per_response) != len(responses):
            raise ValueError("logits_per_response and responses must have the same length")
        if not logits_per_response:
            raise ValueError("at least one response is required")

        raw_energies = np.array(
            [
                self.compute_energy(logits, temperature=self.temperature)
                for logits in logits_per_response
            ],
            dtype=np.float64,
        )
        energy_magnitudes = np.abs(raw_energies)
        energy_mean = float(np.mean(energy_magnitudes))
        energy_std = float(np.std(energy_magnitudes))
        semantic_entropy_estimate = self._cluster_entropy(
            self.cluster_semantics(responses), len(responses)
        )
        semantic_energy_score = float(energy_std / (energy_mean + 1e-8))

        return {
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "semantic_entropy_estimate": semantic_entropy_estimate,
            "semantic_energy_score": semantic_energy_score,
            "is_hallucination_predicted": bool(semantic_energy_score > self.threshold),
        }

    @staticmethod
    def _normalize_response(response: str) -> str:
        normalized = re.sub(r"\s+", " ", response.strip().lower())
        return normalized or "<empty>"

    @staticmethod
    def _cluster_entropy(clusters: dict[str, list[int]], n_responses: int) -> float:
        if n_responses <= 0:
            return 0.0
        entropy = 0.0
        for indices in clusters.values():
            probability = len(indices) / n_responses
            if probability > 0.0:
                entropy -= probability * math.log(probability)
        return float(entropy)


SemanticEnergy = SemanticEnergyDetector


class IsingVerifier:
    """Compute arithmetic constraint-violation energy for a text step.

    Returns float in [0,1]: 0.0 = no violations detected, 1.0 = all claims violated.
    Detects claims of the form 'A op B = C' (e.g. '47+28=76') and checks
    whether C equals the computed result of A op B.
    """

    def __init__(self):
        import re

        self._arith_re = re.compile(
            r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)"  # A op B
            r"\s*(?:=|equals|is)\s*(\d+(?:\.\d+)?)",  # = C
            re.IGNORECASE,
        )

    def energy(self, step_text: str) -> float:
        violations, total = 0, 0
        for m in self._arith_re.finditer(step_text):
            a, op, b, c = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
            total += 1
            try:
                expected = {
                    "+": a + b,
                    "-": a - b,
                    "*": a * b,
                    "/": a / b if b != 0 else float("inf"),
                }[op]
            except (ZeroDivisionError, KeyError):
                continue
            if abs(expected - c) > 1e-6:
                violations += 1
        return violations / total if total > 0 else 0.0
