"""PWA abstraction and formal formulation for NLA verifier.

Spec: REQ-VERIFY-1733, SCENARIO-VERIFY-1733
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class NLAFormalBound:
    z3_script: str
    theoretical_bound: float

def build_nla_pwa_abstraction_and_bound(
    encoder_weight: np.ndarray,
    encoder_bias: np.ndarray,
    decoder_weight: np.ndarray,
    decoder_bias: np.ndarray,
    x0: np.ndarray,
    radius: float,
    target_mse_bound: float
) -> NLAFormalBound:
    """
    Abstracts NLA (MinimalSAE) core activation (ReLU) as PWA.
    Formulates minimum confidence bound (via max MSE) as Z3 script.
    Calculates strict theoretical upper bound on MSE via interval arithmetic.
    """
    d_sae, d_model = encoder_weight.shape
    
    lines = [
        "; NLA PWA Formal Verification Script",
        "; Spec: REQ-VERIFY-1733, SCENARIO-VERIFY-1733",
        "(set-logic QF_NRA)"
    ]
    
    for i in range(d_model):
        lines.append(f"(declare-fun x_{i} () Real)")
        lines.append(f"(assert (>= x_{i} {x0[i] - radius}))")
        lines.append(f"(assert (<= x_{i} {x0[i] + radius}))")
        lines.append(f"(declare-fun y_{i} () Real)")
        
    for j in range(d_sae):
        lines.append(f"(declare-fun h_{j} () Real)")
        lines.append(f"(declare-fun h_relu_{j} () Real)")
        
    for j in range(d_sae):
        terms = [f"(* {float(encoder_weight[j, i])} x_{i})" for i in range(d_model)]
        sum_expr = f"(+ {' '.join(terms)} {float(encoder_bias[j])})"
        lines.append(f"(assert (= h_{j} {sum_expr}))")
        lines.append(f"(assert (= h_relu_{j} (ite (> h_{j} 0.0) h_{j} 0.0)))")
        
    for i in range(d_model):
        terms = [f"(* {float(decoder_weight[i, j])} h_relu_{j})" for j in range(d_sae)]
        sum_expr = f"(+ {' '.join(terms)} {float(decoder_bias[i])})"
        lines.append(f"(assert (= y_{i} {sum_expr}))")
        
    mse_terms = [f"(* (- y_{i} x_{i}) (- y_{i} x_{i}))" for i in range(d_model)]
    mse_sum = f"(+ {' '.join(mse_terms)})"
    lines.append(f"(assert (> (/ {mse_sum} {float(d_model)}) {float(target_mse_bound)}))")
    
    lines.append("(check-sat)")
    lines.append("(exit)")
    
    z3_script = "\n".join(lines)
    
    x_lower = x0 - radius
    x_upper = x0 + radius
    
    W_enc_pos = np.maximum(encoder_weight, 0)
    W_enc_neg = np.minimum(encoder_weight, 0)
    h_lower = W_enc_pos @ x_lower + W_enc_neg @ x_upper + encoder_bias
    h_upper = W_enc_pos @ x_upper + W_enc_neg @ x_lower + encoder_bias
    
    h_relu_lower = np.maximum(h_lower, 0)
    h_relu_upper = np.maximum(h_upper, 0)
    
    W_dec_pos = np.maximum(decoder_weight, 0)
    W_dec_neg = np.minimum(decoder_weight, 0)
    y_lower = W_dec_pos @ h_relu_lower + W_dec_neg @ h_relu_upper + decoder_bias
    y_upper = W_dec_pos @ h_relu_upper + W_dec_neg @ h_relu_lower + decoder_bias
    
    diff_lower = y_lower - x_upper
    diff_upper = y_upper - x_lower
    
    max_sq_diff = np.maximum(diff_lower**2, diff_upper**2)
    theoretical_max_mse = float(np.mean(max_sq_diff))
    
    return NLAFormalBound(z3_script=z3_script, theoretical_bound=theoretical_max_mse)
