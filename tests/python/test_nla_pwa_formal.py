import numpy as np
from carnot.verify.nla_pwa_formal import build_nla_pwa_abstraction_and_bound

def test_nla_pwa_formal():
    """
    Test NLA PWA Formal Abstraction.
    Traces to: REQ-VERIFY-1733, SCENARIO-VERIFY-1733
    """
    d_model = 4
    d_sae = 8
    
    encoder_weight = np.random.randn(d_sae, d_model)
    encoder_bias = np.random.randn(d_sae)
    decoder_weight = np.random.randn(d_model, d_sae)
    decoder_bias = np.random.randn(d_model)
    
    x0 = np.zeros(d_model)
    radius = 0.1
    target_mse_bound = 0.5
    
    res = build_nla_pwa_abstraction_and_bound(
        encoder_weight, encoder_bias, decoder_weight, decoder_bias, x0, radius, target_mse_bound
    )
    
    assert res.theoretical_bound >= 0
    assert "QF_NRA" in res.z3_script
    assert "check-sat" in res.z3_script
