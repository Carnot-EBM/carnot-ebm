"""EORM Verification Layer.

Provides a verification layer that reranks candidate trajectories
using the EORM CoT energy model.
"""

from __future__ import annotations

from typing import Any

from carnot.models.eorm import CoTEnergyInput, EORMModel


class EORMVerifier:
    """Verifies and reranks Chain-of-Thought reasoning steps using EORM."""

    def __init__(self, model: EORMModel) -> None:
        """Initialize the verifier with an EORM model.

        Args:
            model: The EORM energy model to use for scoring.
        """
        self.model = model

    def verify_and_rerank(self, question: str, candidates: list[str]) -> dict[str, Any]:
        """Rerank candidate trajectories based on their EORM energy.

        Args:
            question: The reasoning question.
            candidates: A list of candidate response trajectories.

        Returns:
            A dictionary containing:
                - "best_candidate": The candidate with the lowest energy.
                - "best_energy": The energy of the best candidate.
                - "ranked_candidates": A list of all candidates sorted by energy (lowest first).
                - "energies": A list of energies corresponding to the ranked_candidates.
        """
        if not candidates:
            raise ValueError("Must provide at least one candidate trajectory.")

        # EORMModel.rank sorts candidate responses by energy (lowest first)
        # It returns a list of indices representing the sorted order.
        ranked_indices = self.model.rank(candidates, question=question)

        ranked_candidates = [candidates[i] for i in ranked_indices]
        
        energies = []
        for c in ranked_candidates:
            cot_input = CoTEnergyInput(question_text=question, response_text=c)
            energy = self.model.energy(cot_input)
            energies.append(energy)

        return {
            "best_candidate": ranked_candidates[0],
            "best_energy": float(energies[0]),
            "ranked_candidates": ranked_candidates,
            "energies": [float(e) for e in energies],
        }
