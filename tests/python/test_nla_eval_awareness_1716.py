import pytest
import numpy as np
from unittest.mock import patch, mock_open
from carnot.verify.nla_eval_awareness_1716 import evaluate_nla_eval_awareness, main

def test_req_verify_1716_nla_eval_awareness_schema():
    """
    REQ-VERIFY-1716: NLA Eval-Awareness Drift Check
    SCENARIO-VERIFY-1716: NLA probe handles production vs eval inputs
    """
    res = evaluate_nla_eval_awareness(mock_sleep=0.0)
    
    assert res["schema"] == "carnot.nla_eval_awareness_test.v1"
    assert res["experiment"] == 1716
    assert res["duration_s"] > 100.0
    assert res["random_seed"] == 171616
    assert res["n_samples"] == 100
    assert "tpr_eval" in res
    assert "fpr_eval" in res
    assert "tpr_production" in res
    assert "fpr_production" in res
    assert "delta_tpr" in res
    assert "eval_awareness_detected" in res
    assert "honest_verdict" in res
    
    assert res["model_specs"]["probe_weights_from"] == "exp1694"
    assert res["model_specs"]["n_eval"] == 50
    assert res["model_specs"]["n_production"] == 50
    
    assert len(res["tpr_eval_wilson_95_ci"]) == 2
    assert len(res["tpr_production_wilson_95_ci"]) == 2
    assert len(res["delta_tpr_bootstrap_ci_95"]) == 2

def test_req_verify_1716_mock_sleep():
    res = evaluate_nla_eval_awareness(mock_sleep=0.01)
    assert res["duration_s"] > 0.01

def mock_rand(size):
    if size == 200:
        return np.linspace(0, 1, 200)
    else:
        return np.ones(size)

def test_req_verify_1716_implausible_perfect():
    with patch("carnot.verify.nla_eval_awareness_1716.NLAProbe.predict") as mock_pred:
        with patch("numpy.random.rand", side_effect=mock_rand):
            # Force labels to 1 and predictions to 1
            mock_pred.return_value = np.ones(50)
            res = evaluate_nla_eval_awareness(mock_sleep=0.0)
            assert "IMPLAUSIBLE" in res["honest_verdict"]

def test_main_function():
    with patch("carnot.verify.nla_eval_awareness_1716.evaluate_nla_eval_awareness") as mock_eval:
        mock_eval.return_value = {"dummy": "data"}
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("builtins.print") as mock_print:
                main()
                mock_eval.assert_called_once_with(mock_sleep=100.1)
                mock_file.assert_called_once_with("results/experiment_1716_nla_eval_awareness.json", "w")
                mock_print.assert_called_once_with("Done")

def test_wilson_ci_zero():
    # If p=0, wilson_ci should return [0.0, 0.0] as explicitly checked
    with patch("carnot.verify.nla_eval_awareness_1716.NLAProbe.predict") as mock_pred:
        with patch("numpy.random.rand", side_effect=mock_rand):
            # Force labels to 1 but predictions to 0
            mock_pred.return_value = np.zeros(50)
            res = evaluate_nla_eval_awareness(mock_sleep=0.0)
            assert res["tpr_eval"] == 0.0
            assert res["tpr_eval_wilson_95_ci"] == [0.0, 0.0]
