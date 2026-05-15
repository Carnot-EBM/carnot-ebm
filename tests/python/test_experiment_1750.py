"""
Tests for REQ-PUBLISH-026: HuggingFace Publish Retry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import patch

from scripts.experiment_1750 import run_experiment

def test_experiment_1750_success():
    with patch("scripts.experiment_1750.HfApi") as MockApi, \
         patch("scripts.experiment_1750.create_repo") as mock_create_repo:
        
        instance = MockApi.return_value
        instance.whoami.return_value = {"id": "mock_id"}
        
        deliverable = run_experiment()
        
        assert deliverable["hf_upload_succeeded"] is True
        assert deliverable["honest_verdict"] == "OK: Model published"
        mock_create_repo.assert_called_once()
        assert instance.upload_file.call_count == 2

def test_experiment_1750_blocked():
    with patch("scripts.experiment_1750.HfApi") as MockApi:
        instance = MockApi.return_value
        instance.whoami.side_effect = Exception("Blocked credentials mock")
        
        deliverable = run_experiment()
        
        assert deliverable["hf_upload_succeeded"] is False
        assert deliverable["honest_verdict"] == "blocked_credentials"
