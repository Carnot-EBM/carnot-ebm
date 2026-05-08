"""Inference-time sampling adapters.

Spec: REQ-SAMPLE-058, SCENARIO-SAMPLE-086.
"""

from __future__ import annotations

from .gibbs import (
    build_exp1564_deliverable_payload,
    constructive_kl_to_thrml,
    reference_thrml_sample,
    sample,
    sample_from_payload,
    zero_coupling_hamming_summary,
)

__all__ = [
    "build_exp1564_deliverable_payload",
    "constructive_kl_to_thrml",
    "reference_thrml_sample",
    "sample",
    "sample_from_payload",
    "zero_coupling_hamming_summary",
]
