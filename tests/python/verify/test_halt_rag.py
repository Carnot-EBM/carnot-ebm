"""Tests for HALT-RAG NLI Detector."""

import pytest
from carnot.verify.halt_rag_nli import HaltRagNliDetector, label_from_entry

def test_halt_rag_nli_detector_fit_verify():
    entries = []
    labels = []
    # Create 12 mock entries to satisfy cv=6 requirement (6 samples per class)
    for i in range(12):
        entry = {
            "top_logprobs": [{"token": -1.0, "other": -2.0}],
            "token_logprobs": [-1.0],
            "correctness_label": "correct" if i % 2 == 0 else "incorrect"
        }
        entries.append(entry)
        labels.append(0 if i % 2 == 0 else 1)

    detector = HaltRagNliDetector(abstention_threshold=0.65, random_seed=42)

    with pytest.raises(RuntimeError):
        detector.verify(entries[0])

    detector.fit(entries, labels)
    result = detector.verify(entries[0])

    assert "halt_rag_score" in result
    assert "abstained" in result
    assert "confidence" in result
    assert result["nli_signals_used"] == 3
    assert 0.5 <= result["confidence"] <= 1.0
    
    if result["abstained"]:
        assert result["halt_rag_score"] == 0.5

def test_label_from_entry():
    assert label_from_entry({"correctness_label": "correct"}) == 0
    assert label_from_entry({"correctness_label": "incorrect"}) == 1
    assert label_from_entry({"correct": True}) == 0
    assert label_from_entry({"correct": False}) == 1
    with pytest.raises(ValueError):
        label_from_entry({})
