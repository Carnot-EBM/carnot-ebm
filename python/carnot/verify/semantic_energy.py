"""Semantic Energy synthetic-logit hallucination detector.

This module is the Exp 2338 Tier 0g prototype.  It computes the Boltzmann
negative log-partition energy from raw logits and combines repeated response
energies with a simple semantic-cluster entropy estimate.  The prototype is
CPU-only and intentionally uses synthetic logits; real penultimate-layer LLM
logits are a later validation target.

Spec: REQ-TIER0-007, SCENARIO-TIER0-007
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import OrderedDict

import numpy as np


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
