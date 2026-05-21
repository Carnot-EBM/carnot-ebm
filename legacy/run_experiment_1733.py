import json
import os
import numpy as np
from carnot.verify.nla_probe import MinimalSAE
from carnot.verify.nla_pwa_formal import build_nla_pwa_abstraction_and_bound

def run():
    d_model = 16
    sae = MinimalSAE(d_model=d_model, expansion_factor=2)
    
    encoder_weight = sae.encoder.weight.detach().numpy()
    encoder_bias = sae.encoder.bias.detach().numpy()
    decoder_weight = sae.decoder.weight.detach().numpy()
    decoder_bias = sae.decoder.bias.detach().numpy()
    
    x0 = np.zeros(d_model)
    radius = 0.05
    target_mse = 0.2
    
    bound = build_nla_pwa_abstraction_and_bound(
        encoder_weight, encoder_bias, decoder_weight, decoder_bias, x0, radius, target_mse
    )
    
    res = {
        "status": "complete",
        "pwa_abstraction_generated": True,
        "theoretical_bound": bound.theoretical_bound,
        "z3_script_length": len(bound.z3_script),
        "honest_verdict": "complete: Generated PWA abstraction and Z3 MILP formulation."
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_1733.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    run()
