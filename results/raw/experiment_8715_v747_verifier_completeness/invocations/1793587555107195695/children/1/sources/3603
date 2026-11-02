"""PWAKArAtAttention implementation.

Spec references: REQ-KAN-1686, SCENARIO-KAN-1686.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from carnot.models.karat_attention import RationalKArAtLayer
from carnot.models.pwa_kan import PWAKANUnit
from carnot.models.rkan import to_fraction


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_RESULT_PATH = _repo_root() / "results/experiment_1686_pwa_karat.json"


class PWAKArAtAttention:
    """PWA wrapper for KArAt attention."""

    def __init__(self, karat_layer: RationalKArAtLayer, samples_per_segment: int = 33) -> None:
        self.karat_layer = karat_layer
        
        domain = karat_layer.attention_spline.domain
        lower = float(to_fraction(domain[0]))
        upper = float(to_fraction(domain[1]))
        
        num_points = len(karat_layer.attention_spline.control_points)
        num_segments = num_points - 1
        
        breakpoints = []
        for i in range(num_points):
            x = lower + (upper - lower) * (i / num_segments)
            breakpoints.append(x)
            
        def evaluator(x: float) -> float:
            val = karat_layer.attention_spline.evaluate(Fraction(float(x)))
            return float(to_fraction(val))

        self.pwa_unit = PWAKANUnit.from_callable(
            name="pwa_karat_attention",
            evaluator=evaluator,
            breakpoints=tuple(breakpoints),
            samples_per_segment=samples_per_segment,
        )

    def evaluate(self, x: float) -> float:
        """Evaluate the PWA unit."""
        return self.pwa_unit.evaluate(x)


def build_experiment_1686_artifact() -> dict[str, Any]:
    """Build the stable artifact for experiment 1686."""
    layer = RationalKArAtLayer(seq_len=2, dim=2, spline_points=[-1, 0, 1])
    pwa_attention = PWAKArAtAttention(layer)
    
    return {
        "schema": "carnot.pwa_karat_attention.v1",
        "status": "complete",
        "experiment": 1686,
        "experiment_id": 1686,
        "run_date": "20260510",
        "title": "PWA KArAt Attention Approximation",
        "spec": ["REQ-KAN-1686", "SCENARIO-KAN-1686"],
        "module": "python/carnot/models/pwa_karat.py",
        "artifact_path": "results/experiment_1686_pwa_karat.json",
        "pwa_unit": pwa_attention.pwa_unit.as_serializable(),
        "honest_verdict": "complete: pwa_karat_attention_implemented"
    }


def write_experiment_1686_artifact(
    output_path: str | Path = DEFAULT_RESULT_PATH,
) -> dict[str, Any]:
    """Write the artifact to the specified path."""
    artifact = build_experiment_1686_artifact()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


__all__ = [
    "DEFAULT_RESULT_PATH",
    "PWAKArAtAttention",
    "build_experiment_1686_artifact",
    "write_experiment_1686_artifact",
]
