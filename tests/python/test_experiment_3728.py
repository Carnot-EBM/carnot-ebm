import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from scripts import experiment_3728_bounded_checkpointed_train_ebt_and_ar

def test_experiment_3728_blocked_cuda(tmp_path):
    os.chdir(tmp_path)
    os.makedirs("results", exist_ok=True)
    # Mock no cuda
    with patch('torch.cuda.is_available', return_value=False):
        try:
            experiment_3728_bounded_checkpointed_train_ebt_and_ar.main()
        except SystemExit:
            pass
        
        with open("results/experiment_3728_bounded_checkpointed_train_ebt_and_ar.json") as f:
            artifact = json.load(f)
            
        assert artifact["honest_verdict"] == "blocked_cuda"
        assert artifact["preconditions_checked"]["cuda"] is False

def test_experiment_3728_blocked_ebt(tmp_path):
    os.chdir(tmp_path)
    os.makedirs("results", exist_ok=True)
    # Mock cuda ok but no ebt
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1):
        
        # We don't mock import error directly, but in our environment carnot.phase3.ebt_upstream is absent
        try:
            experiment_3728_bounded_checkpointed_train_ebt_and_ar.main()
        except SystemExit:
            pass
            
        with open("results/experiment_3728_bounded_checkpointed_train_ebt_and_ar.json") as f:
            artifact = json.load(f)
            
        # Our environment doesn't have it, so it should be blocked_ebt
        assert artifact["honest_verdict"] == "blocked_ebt"
