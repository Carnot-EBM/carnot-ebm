"""Tests for experiment 3827 verifier error independence scissor.

Spec: REQ-VERIFY-3827
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "scripts" / "experiments"))

from experiment_3827_verifier_error_independence_scissor import Exp3827

@patch("experiment_3827_verifier_error_independence_scissor.torch")
def test_blocked_no_cuda(mock_torch, tmp_path):
    mock_torch.cuda.is_available.return_value = False
    
    exp = Exp3827()
    with patch("experiment_3827_verifier_error_independence_scissor.repo_root", new=tmp_path):
        # The experiment writes its blocked-path artifact to repo_root/results/;
        # create that dir under the patched tmp repo_root (the sibling
        # blocked-model-not-cached test does this — this one omitted it, which
        # made the blocked-path artifact write throw FileNotFoundError and
        # poisoned the conductor pre-test gate, cascade-skipping every later task).
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)
        res = exp.run()

    assert res["status"] == "blocked_no_cuda"


@patch("experiment_3827_verifier_error_independence_scissor.torch")
def test_blocked_model_not_cached(mock_torch, tmp_path):
    mock_torch.cuda.is_available.return_value = True
    
    exp = Exp3827()
    with patch("experiment_3827_verifier_error_independence_scissor.repo_root", new=tmp_path):
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)
        
        with patch("experiment_3827_verifier_error_independence_scissor.Path") as mock_path_cls:
            # Provide our own paths
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = False # fail cache_dir.exists()
            mock_path_cls.return_value = mock_path_instance
        
        # Need to fix up repo_root inside Exp3827 or mock Path
        # We can just override the cache_dir check directly using patch
        with patch("experiment_3827_verifier_error_independence_scissor.os.path.expanduser") as mock_expand:
            mock_expand.return_value = str(tmp_path / "does_not_exist")
            res = exp.run()
            
    assert res["status"] == "blocked_model_not_cached_qwen3.6_35b"


@patch("experiment_3827_verifier_error_independence_scissor.torch")
@patch("experiment_3827_verifier_error_independence_scissor.BatchedInferenceRunner")
@patch("experiment_3827_verifier_error_independence_scissor.roc_auc_score")
@patch("experiment_3827_verifier_error_independence_scissor._read_fover_rows")
@patch("experiment_3827_verifier_error_independence_scissor._load_fr11_memory_index")
@patch("experiment_3827_verifier_error_independence_scissor._select_balanced_subset")
@patch("experiment_3827_verifier_error_independence_scissor._score_text_verifiers")
@patch("experiment_3827_verifier_error_independence_scissor._fr11_memory_score")
def test_run_success(
    mock_memory_score,
    mock_score_text,
    mock_select,
    mock_load_index,
    mock_read_rows,
    mock_roc,
    mock_runner_cls,
    mock_torch,
    tmp_path,
):
    mock_torch.cuda.is_available.return_value = True
    
    # Mock llama_cpp import
    import sys
    llama_mock = MagicMock()
    sys.modules["llama_cpp"] = llama_mock
    
    exp = Exp3827()
    with patch("experiment_3827_verifier_error_independence_scissor.repo_root", new=tmp_path):
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "fover_corpus.jsonl").touch()
        
        with patch("experiment_3827_verifier_error_independence_scissor.os.path.expanduser") as mock_expand:
            cache_dir = tmp_path / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "test.gguf").touch()
            mock_expand.return_value = str(cache_dir)
            
            mock_select.return_value = [
                {"label": "incorrect", "step_text": "t1"},
                {"label": "incorrect", "step_text": "t2"},
                {"label": "correct", "step_text": "t3"},
                {"label": "correct", "step_text": "t4"},
            ]
            
            mock_score_text.return_value = {
                "tier0r_curry_howard": [1.0, 1.0, 0.0, 0.0],
                "tier0u_logical_consistency": [1.0, 1.0, 0.0, 0.0]
            }
            mock_memory_score.side_effect = [1.0, 0.0, 0.0, 0.0]
            
            mock_roc.return_value = 0.95
            
            mock_runner_instance = MagicMock()
            mock_runner_cls.return_value = mock_runner_instance
            
            r1 = MagicMock(); r1.response = "no"
            r2 = MagicMock(); r2.response = "yes"
            r3 = MagicMock(); r3.response = "yes"
            r4 = MagicMock(); r4.response = "yes"
            mock_runner_instance.run_batch.return_value = [r1, r2, r3, r4]
            
            with patch.object(exp, "setup_gpu"):
                exp.run()
            
            out_file = tmp_path / "results" / exp.deliverable
            assert out_file.exists()
            
            data = json.loads(out_file.read_text())
            assert "verifier_moat_survives" in data["status"]
            assert data["residual_catch_rate"] == 1.0
        
    del sys.modules["llama_cpp"]
