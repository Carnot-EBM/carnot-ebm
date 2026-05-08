"""Inference-time sampling adapters.

Spec: REQ-SAMPLE-058, SCENARIO-SAMPLE-086, REQ-SAMPLE-059, SCENARIO-SAMPLE-087.
"""

from __future__ import annotations

from . import brs_residual as brs_residual
from . import init_policy_benchmark as init_policy_benchmark
from .gibbs import (
    build_exp1564_deliverable_payload,
    constructive_kl_to_thrml,
    reference_thrml_sample,
    sample,
    sample_from_payload,
    zero_coupling_hamming_summary,
)

__all__ = [
    "brs_residual",
    "build_exp1564_deliverable_payload",
    "constructive_kl_to_thrml",
    "init_policy_benchmark",
    "reference_thrml_sample",
    "sample",
    "sample_from_payload",
    "zero_coupling_hamming_summary",
]
