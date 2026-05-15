"""
Tests for Experiment 1773 CARM Evaluation.
References: REQ-CARM-1773-1, SCENARIO-CARM-1773-1
"""
import json
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_1773_carm_eval import run_experiment

@patch("scripts.experiment_1773_carm_eval.Path")
@patch("scripts.experiment_1773_carm_eval.CARMExtractor")
def test_experiment_1773_carm_eval(mock_extractor_class, mock_path):
    """
    Test dual-model evaluation of CARM prototype.
    References: REQ-CARM-1773-1, SCENARIO-CARM-1773-1
    """
    # Setup mock test suite
    mock_test_suite_path = MagicMock()
    mock_test_suite_path.exists.return_value = True
    test_suite_data = {
        "cases": [
            {"id": "c1", "instruction": "do A", "ground_truth": {"tools_required": ["A"]}},
            {"id": "c2", "instruction": "do B", "ground_truth": {"tools_required": ["B"]}},
            {"id": "c3", "instruction": "do nothing", "ground_truth": {}}
        ]
    }
    mock_test_suite_path.read_text.return_value = json.dumps(test_suite_data)
    
    # We also mock the output path to avoid writing to disk during testing
    mock_out_path = MagicMock()
    
    def side_effect(path_str):
        if "test_suite" in str(path_str):
            return mock_test_suite_path
        return mock_out_path
        
    mock_path.side_effect = side_effect
    
    # Setup mock extractors
    # For model 1 (31B): 2 correct, 1 false accept (case c3)
    # For model 2 (26B): 1 correct, 0 false accept
    mock_extractor_1 = MagicMock()
    mock_extractor_1.model_spec = "unsloth/gemma-4-31B-it-GGUF"
    mock_extractor_1.extract_constraints.side_effect = [
        {"tools_required": ["A"]},  # correct (true positive)
        {"tools_required": ["B"]},  # correct (true positive)
        {"tools_required": ["X"]}   # false accept (expected empty)
    ]
    
    mock_extractor_2 = MagicMock()
    mock_extractor_2.model_spec = "unsloth/gemma-4-26B-A4B-it-GGUF"
    mock_extractor_2.extract_constraints.side_effect = [
        {"tools_required": ["A"]},  # correct (true positive)
        {"tools_required": ["X"]},  # incorrect (false negative/wrong)
        {}                          # correct (true negative)
    ]
    
    mock_extractor_class.side_effect = [mock_extractor_1, mock_extractor_2]
    
    deliverable = run_experiment("dummy_out.json")
    
    assert deliverable["schema"] == "carnot.carm.evaluation.v1"
    assert "recall_rate" in deliverable
    assert "false_accept_rate" in deliverable
    
    # 2 true positive cases (c1, c2), 1 true negative case (c3)
    # Model 1 recall: 2/2 = 1.0. False accept: 1/1 = 1.0.
    # Model 2 recall: 1/2 = 0.5. False accept: 0/1 = 0.0.
    # We just ensure the fields exist and are floats as required by the schema.
    assert isinstance(deliverable["recall_rate"], float)
    assert isinstance(deliverable["false_accept_rate"], float)
    
    # Specifically for our mock, overall rates:
    # Total recall = (1.0 + 0.5) / 2 = 0.75
    # Total FA = (1.0 + 0.0) / 2 = 0.5
    assert deliverable["recall_rate"] == 0.75
    assert deliverable["false_accept_rate"] == 0.5

