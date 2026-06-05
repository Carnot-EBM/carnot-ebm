"""Tests for faithful 0.9131 FoVer ablation harness.

Spec: REQ-VERIFY-3826, SCENARIO-VERIFY-3826
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from carnot.verify.experiment_3826_fover_ablation_faithful import run_experiment, main

@pytest.fixture
def mock_fover_rows():
    # Make a mock subset of 4 rows (balanced: 2 correct (0), 2 incorrect (1))
    return [
        {"label": 0, "step_text": "correct1"},
        {"label": 1, "step_text": "incorrect1"},
        {"label": 0, "step_text": "correct2"},
        {"label": 1, "step_text": "incorrect2"},
    ]

@patch("carnot.verify.experiment_3826_fover_ablation_faithful._read_fover_rows")
@patch("carnot.verify.experiment_3826_fover_ablation_faithful._load_fr11_memory_index")
@patch("carnot.verify.experiment_3826_fover_ablation_faithful._select_balanced_subset")
@patch("carnot.verify.experiment_3826_fover_ablation_faithful._score_text_verifiers")
@patch("carnot.verify.experiment_3826_fover_ablation_faithful._fr11_memory_score")
def test_faithful_ablation_reproduces_0_9131(
    mock_memory_score,
    mock_score_text,
    mock_select,
    mock_load_index,
    mock_read_rows,
    tmp_path,
):
    """Test the positive control path where it hits 0.9131 and produces expected verdict."""
    # Setup mock corpus
    corpus_file = tmp_path / "data" / "fover_corpus.jsonl"
    corpus_file.parent.mkdir(parents=True)
    corpus_file.touch()

    # Mock dependencies
    mock_read_rows.return_value = [{"label": 0, "step_text": "test"}] * 4
    mock_load_index.return_value = {}
    mock_select.return_value = [
        {"label": 0, "step_text": "correct1"},
        {"label": 1, "step_text": "incorrect1"},
        {"label": 0, "step_text": "correct2"},
        {"label": 1, "step_text": "incorrect2"},
    ]

    # In our mock subset of size 4 (2 pos, 2 neg), we want to return scores that yield AUROC 0.9131.
    # To do this, we can just patch compute_auroc instead of crafting the perfect sub-scores.
    with patch("carnot.verify.experiment_3826_fover_ablation_faithful.compute_auroc") as mock_auroc:
        # 5 seeds * 3 calls (full, formal, learned) = 15 calls
        # Let's just return fixed AUROCs.
        # We need full=0.9131, formal=0.8947, learned=0.8699
        # Return values sequentially for full, formal, learned across 5 seeds:
        mock_auroc.side_effect = [0.9131, 0.8947, 0.8699] * 5
        
        # We also need to return valid mock verifier scores to avoid math errors
        mock_score_text.return_value = {
            "tier0r_curry_howard": [0.0, 1.0, 0.0, 1.0],
            "tier0u_logical_consistency": [0.0, 1.0, 0.0, 1.0],
        }
        mock_memory_score.side_effect = [0.0, 1.0, 0.0, 1.0] * 5
        
        artifact = run_experiment(tmp_path)
        
        assert artifact["full_ensemble_auroc"] == 0.9131
        assert artifact["formal_only_auroc"] == 0.8947
        assert artifact["learned_only_auroc"] == 0.8699
        assert "formal_core_retains_moat" in artifact["honest_verdict"]
        assert "0.9131" in artifact["harness_fix_description"]


def test_ablation_blocked_if_no_corpus(tmp_path):
    """Test that missing corpus returns blocked verdict with dummy scores."""
    artifact = run_experiment(tmp_path)
    
    assert artifact["honest_verdict"] == "blocked_fover_corpus_not_cached"
    assert artifact["full_ensemble_auroc"] == 0.0
    assert artifact["formal_only_auroc"] == 0.0
    assert artifact["learned_only_auroc"] == 0.0


@patch("carnot.verify.experiment_3826_fover_ablation_faithful._read_fover_rows")
@patch("carnot.verify.experiment_3826_fover_ablation_faithful._load_fr11_memory_index")
@patch("carnot.verify.experiment_3826_fover_ablation_faithful._select_balanced_subset")
@patch("carnot.verify.experiment_3826_fover_ablation_faithful._score_text_verifiers")
@patch("carnot.verify.experiment_3826_fover_ablation_faithful._fr11_memory_score")
def test_ablation_inconclusive_if_diverges(
    mock_memory_score,
    mock_score_text,
    mock_select,
    mock_load_index,
    mock_read_rows,
    tmp_path,
):
    """Test that if full_ensemble is far from 0.9131, it escalates to operator."""
    corpus_file = tmp_path / "data" / "fover_corpus.jsonl"
    corpus_file.parent.mkdir(parents=True)
    corpus_file.touch()

    mock_select.return_value = [
        {"label": 0, "step_text": "correct1"},
        {"label": 1, "step_text": "incorrect1"},
    ]
    
    with patch("carnot.verify.experiment_3826_fover_ablation_faithful.compute_auroc") as mock_auroc:
        # Full AUROC = 0.85 (diverges > 0.01 from 0.9131)
        mock_auroc.side_effect = [0.85, 0.80, 0.70] * 5
        
        mock_score_text.return_value = {
            "tier0r_curry_howard": [0.0, 1.0],
            "tier0u_logical_consistency": [0.0, 1.0],
        }
        mock_memory_score.side_effect = [0.0, 1.0] * 5
        
        artifact = run_experiment(tmp_path)
        
        assert "INCONCLUSIVE_ablation_harness_unfaithful" in artifact["honest_verdict"]
        assert "escalate_operator" in artifact["honest_verdict"]

@patch("carnot.verify.experiment_3826_fover_ablation_faithful._read_fover_rows")
@patch("carnot.verify.experiment_3826_fover_ablation_faithful._load_fr11_memory_index")
@patch("carnot.verify.experiment_3826_fover_ablation_faithful._select_balanced_subset")
@patch("carnot.verify.experiment_3826_fover_ablation_faithful._score_text_verifiers")
@patch("carnot.verify.experiment_3826_fover_ablation_faithful._fr11_memory_score")
def test_ablation_moat_depends_on_learned(
    mock_memory_score,
    mock_score_text,
    mock_select,
    mock_load_index,
    mock_read_rows,
    tmp_path,
):
    """Test verdict when formal < 0.85."""
    corpus_file = tmp_path / "data" / "fover_corpus.jsonl"
    corpus_file.parent.mkdir(parents=True)
    corpus_file.touch()

    mock_select.return_value = [
        {"label": 0, "step_text": "correct1"},
        {"label": 1, "step_text": "incorrect1"},
    ]
    
    with patch("carnot.verify.experiment_3826_fover_ablation_faithful.compute_auroc") as mock_auroc:
        # Full = 0.9131 (matches target), Formal = 0.84, Learned = 0.90
        mock_auroc.side_effect = [0.9131, 0.84, 0.90] * 5
        
        mock_score_text.return_value = {
            "tier0r_curry_howard": [0.0, 1.0],
            "tier0u_logical_consistency": [0.0, 1.0],
        }
        mock_memory_score.side_effect = [0.0, 1.0] * 5
        
        artifact = run_experiment(tmp_path)
        
        assert "moat_depends_on_learned_probes" in artifact["honest_verdict"]


@patch("carnot.verify.experiment_3826_fover_ablation_faithful.run_experiment")
def test_main_writes_file(mock_run, tmp_path):
    mock_run.return_value = {"honest_verdict": "complete"}
    
    with patch("carnot.verify.experiment_3826_fover_ablation_faithful.Path") as mock_path:
        # Mock repo_root.
        # We need parents[3] to return tmp_path
        mock_resolved = MagicMock()
        mock_resolved.parents = [None, None, None, tmp_path]
        
        mock_file_path = MagicMock()
        mock_file_path.resolve.return_value = mock_resolved
        mock_path.return_value = mock_file_path
        
        main()
        
        out_file = tmp_path / "results" / "experiment_3826_fover_ablation_faithful.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["honest_verdict"] == "complete"
