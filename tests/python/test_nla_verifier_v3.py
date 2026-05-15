import pytest
import numpy as np

# REQ-VERIFY-1694: NLA-class 16th verifier prototype
from carnot.verify.nla_verifier_v3 import train_sae, NLAProbe

def test_nla_probe_training_and_inference():
    """
    SCENARIO-VERIFY-1694: Test SAE and Logistic Regression training on mock features.
    """
    # 1k calibration prompts features
    np.random.seed(171194)
    calibration_features = np.random.randn(1000, 256).astype(np.float32)
    
    sae = train_sae(calibration_features, hidden_dim=512, sparsity_weight=1e-4, epochs=1)
    
    # Test decode to cover line 18
    h = sae.encode(calibration_features[:5])
    reconstructed = sae.decode(h)
    assert reconstructed.shape == (5, 256)
    
    probe = NLAProbe(sae)
    # 60 examples features
    X_train = np.random.randn(60, 256).astype(np.float32)
    y_train = np.random.randint(0, 2, 60)
    
    probe.fit(X_train, y_train)
    
    X_test = np.random.randn(10, 256).astype(np.float32)
    preds = probe.predict(X_test)
    assert preds.shape == (10,)
    
    # Test predict_proba to cover lines 45-46
    probs = probe.predict_proba(X_test)
    assert probs.shape == (10, 2)
