"""EBT sidecar scoring with KAN energy formulation.

This module integrates KAN-based energy scoring with the EBT sidecar pipeline.
Spec refs: REQ-KAN-3374, SCENARIO-KAN-3374.
"""

from __future__ import annotations

import numpy as np
from carnot.models.kan import KAN
from carnot.inference.ebt_arm_sidecar_adapter import (
    SidecarReplayScorer,
    SidecarReplayScore,
    JsonDict,
    ReplayWeights,
    _constraint_violation_energy,
    _arm_sequence_energy,
    _verifier_feedback_energy,
    _stable_float,
    canonical_fingerprint,
    REPLAY_INFERENCE_SUBSTRATE
)

class KANSidecarScorer(SidecarReplayScorer):
    """Sidecar scorer that uses a minimal KAN model to compute an energy term."""

    def __init__(
        self,
        kan_model: KAN,
        weights: ReplayWeights | None = None,
        schema: JsonDict | None = None
    ) -> None:
        super().__init__(weights=weights, schema=schema)
        self.kan_model = kan_model

    def extract_features(self, record: JsonDict) -> np.ndarray:
        """Extract a fixed-size feature vector for the KAN model."""
        candidate = record["candidate"]
        confidence = record["confidence"]

        f1 = _constraint_violation_energy(record["constraints"])
        f2 = _arm_sequence_energy(candidate.get("token_logprobs", []))
        f3 = _verifier_feedback_energy(record["verifier_feedback"])
        f4 = 1.0 - float(confidence["confidence"])
        f5 = 1.0 if confidence["abstain"] else 0.0

        features = np.zeros(self.kan_model.n_params, dtype=np.float64)
        if self.kan_model.n_params >= 5:
            features[0] = f1
            features[1] = f2
            features[2] = f3
            features[3] = f4
            features[4] = f5
        return features

    def score(self, record: JsonDict) -> SidecarReplayScore:
        base_score = super().score(record)
        
        features = self.extract_features(record)
        kan_energy_val = float(self.kan_model.logits(features.reshape(1, -1))[0])
        
        kan_term = self._term(
            "kan_energy",
            kan_energy_val,
            1.0,
            "kan_model_inference"
        )
        
        new_terms = list(base_score.energy_terms) + [kan_term]
        new_total = _stable_float(base_score.total_energy + kan_term["weighted_value"])
        
        return SidecarReplayScore(
            record_id=base_score.record_id,
            candidate_id=base_score.candidate_id,
            total_energy=new_total,
            energy_terms=new_terms,
            confidence=base_score.confidence,
            abstain=base_score.abstain,
            input_fingerprint=base_score.input_fingerprint,
            inference_substrate=base_score.inference_substrate,
        )
