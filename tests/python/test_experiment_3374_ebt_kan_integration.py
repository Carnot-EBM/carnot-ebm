"""Tests for KAN energy formulation sidecar scoring.

Spec refs: REQ-KAN-3374, SCENARIO-KAN-3374.
"""

from carnot.models.kan import KAN
from carnot.inference.ebt_kan_sidecar import KANSidecarScorer
from carnot.inference.ebt_arm_sidecar_adapter import example_sidecar_records

def test_kan_sidecar_scorer() -> None:
    """SCENARIO-KAN-3374: Integrate KAN energy formulation."""
    kan_model = KAN(n_params=256, seed=42)
    scorer = KANSidecarScorer(kan_model=kan_model)
    
    records = example_sidecar_records()
    assert len(records) > 0
    
    for record in records:
        score = scorer.score(record)
        kan_terms = [t for t in score.energy_terms if t["name"] == "kan_energy"]
        assert len(kan_terms) == 1
        assert kan_terms[0]["source"] == "kan_model_inference"
        # Verify it was added to total energy
        assert score.total_energy is not None
