import os
import sys
import json
from unittest.mock import patch

PROJECT_ROOT = "/home/ianblenke/github.com/ianblenke/carnot"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import experiment_3787_p1_discrete_search_adjudication_v3_retry as exp

def test_experiment_3787_blocked_no_cuda(tmp_path):
    with patch("torch.cuda.is_available", return_value=False):
        exp.main()
        
    res_path = os.path.join(PROJECT_ROOT, "results", "experiment_3787_p1_discrete_search_adjudication_v3_retry.json")
    assert os.path.exists(res_path)
    with open(res_path) as f:
        data = json.load(f)
        
    assert data["honest_verdict"] == "blocked_cuda_unavailable"
    assert data["preconditions_checked"] is False

def test_experiment_3787_blocked_no_free_gpu(tmp_path):
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.mem_get_info", return_value=(0, 24 * 1024**3)):
        exp.main()
        
    res_path = os.path.join(PROJECT_ROOT, "results", "experiment_3787_p1_discrete_search_adjudication_v3_retry.json")
    assert os.path.exists(res_path)
    with open(res_path) as f:
        data = json.load(f)
        
    assert data["honest_verdict"] == "blocked_no_free_gpu"
    assert data["handoff_to_operator"] is True
    assert data["preconditions_checked"] is False
