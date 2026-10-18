"""Inference-time sampling adapters.

Spec: REQ-SAMPLE-058, SCENARIO-SAMPLE-086, REQ-SAMPLE-059, SCENARIO-SAMPLE-087,
REQ-SAMPLE-061, SCENARIO-SAMPLE-089, REQ-SAMPLE-062, SCENARIO-SAMPLE-090.
"""

from __future__ import annotations

from . import brs_residual as brs_residual
from . import init_policy_benchmark as init_policy_benchmark
from . import rho_of_c_measurement as rho_of_c_measurement
from . import soft_gibbs_coverage_bound as soft_gibbs_coverage_bound
from . import temporal_exchange as temporal_exchange
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
    "rho_of_c_measurement",
    "sample",
    "sample_from_payload",
    "soft_gibbs_coverage_bound",
    "temporal_exchange",
    "zero_coupling_hamming_summary",
]
