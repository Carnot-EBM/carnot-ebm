import json
from pathlib import Path
import pytest
from unittest.mock import patch
import sys

# Ensure the root directory and python directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "scripts"))

from experiment_3612_validate_factual_corpus_v2 import main

# REQ-TEST-001: The tests cover the factual corpus validation scenarios.

def test_main_degenerate_confidence(tmp_path):
    """
    SCENARIO-1: Degenerate confidence (AUROC <= 0.50).
    The validation script should detect this, set facts_corpus_validated to False,
    and emit the blocked verdict.
    """
    corpus_file = tmp_path / "corpus.jsonl"
    data = []
    # Make AUROC < 0.50 by making model_confidence higher for hallucinations
    for i in range(200):
        is_hal = i % 2
        conf = 0.8 if is_hal == 1 else 0.2
        data.append({
            "question": f"Q{i}",
            "answer": "A",
            "is_hallucination": is_hal,
            "evidence_passage": f"Evidence {i % 10}",
            "model_confidence": conf
        })
    corpus_file.write_text("\n".join(json.dumps(d) for d in data))
    
    with patch("experiment_3612_validate_factual_corpus_v2.ExperimentTemplate.build_result") as mock_build, \
         patch("experiment_3612_validate_factual_corpus_v2.ExperimentTemplate.assert_deliverable_written"), \
         patch("pathlib.Path.write_text"):
        mock_build.return_value = {"mock": "artifact"}
        main(corpus_path_override=str(corpus_file))
        
        result_arg = mock_build.call_args[0][0]
        assert result_arg["honest_verdict"] == "complete: factual_corpus_degenerate_confidence_out_of_band_or_evidence_leaks_facts_row_blocked"
        assert result_arg["facts_corpus_validated"] is False
        assert result_arg["facts_corpus_has_evidence"] is True
        assert type(result_arg["facts_corpus_validated"]) is bool

def test_main_missing_file():
    """
    SCENARIO-2: Missing file should raise FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        main(corpus_path_override="nonexistent.jsonl")

def test_main_rebuild_fallback(tmp_path):
    """
    SCENARIO-3: n < 50 triggers rebuild fallback.
    """
    corpus_file = tmp_path / "corpus.jsonl"
    data = []
    for i in range(10):
        data.append({
            "question": f"Q{i}",
            "answer": "A",
            "is_hallucination": 0,
            "evidence_passage": f"Evidence",
            "model_confidence": 0.9
        })
    corpus_file.write_text("\n".join(json.dumps(d) for d in data))
    
    with patch("experiment_3612_validate_factual_corpus_v2.ExperimentTemplate.build_result") as mock_build, \
         patch("experiment_3612_validate_factual_corpus_v2.ExperimentTemplate.assert_deliverable_written"), \
         patch("pathlib.Path.write_text"):
        mock_build.return_value = {"mock": "artifact"}
        main(corpus_path_override=str(corpus_file))
        
        result_arg = mock_build.call_args[0][0]
        assert result_arg["honest_verdict"] == "complete: factual_corpus_rebuilt_v3_validated_bare_fields_emitted"

def test_main_success(tmp_path):
    """
    SCENARIO-4: Perfect corpus passes all gates.
    """
    corpus_file = tmp_path / "corpus.jsonl"
    data = []
    # Make AUROC > 0.50 and < 0.95 by adding some noise but generally positive correlation
    import random
    random.seed(42)
    for i in range(200):
        is_hal = i % 2
        conf = random.uniform(0.6, 0.9) if is_hal == 0 else random.uniform(0.1, 0.7)
        data.append({
            "question": f"Q{i}",
            "answer": "A",
            "is_hallucination": is_hal,
            "evidence_passage": f"Evidence {i % 10}",
            "model_confidence": conf
        })
    corpus_file.write_text("\n".join(json.dumps(d) for d in data))
    
    with patch("experiment_3612_validate_factual_corpus_v2.ExperimentTemplate.build_result") as mock_build, \
         patch("experiment_3612_validate_factual_corpus_v2.ExperimentTemplate.assert_deliverable_written"), \
         patch("pathlib.Path.write_text"):
        mock_build.return_value = {"mock": "artifact"}
        main(corpus_path_override=str(corpus_file))
        
        result_arg = mock_build.call_args[0][0]
        assert result_arg["honest_verdict"] == "complete: factual_corpus_v2_validated_held_out_evidence_confidence_headroom_confirmed_bare_fields_emitted"
        assert result_arg["facts_corpus_validated"] is True
        assert result_arg["facts_corpus_has_evidence"] is True
        assert type(result_arg["facts_corpus_validated"]) is bool
