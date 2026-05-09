"""Exp 1628 EBRM Latent Trace Scoring.

Spec: REQ-VERIFY-1628, SCENARIO-VERIFY-1628.
"""

from typing import Union

import jax
import jax.numpy as jnp

from carnot.verify.nabla_reasoner import differentiable_ebcn_energy


@jax.jit
def score_latent_trace(trace_logits: jnp.ndarray) -> Union[float, jnp.ndarray]:
    """
    Computes an EBRM-style score for a latent trace.
    Higher score indicates a better trace (lower structural energy).
    """
    energy = differentiable_ebcn_energy(trace_logits)
    # Using exp(-energy) as the EBRM-style probability/score
    return jnp.exp(-energy)
