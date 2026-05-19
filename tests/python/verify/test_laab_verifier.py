import pytest
from carnot.verify.laab_verifier import compute_laab_score, extract_self_judgment_polarity, extract_response_label_polarity

def test_extract_self_judgment_polarity():
    assert extract_self_judgment_polarity("I think it is 5") == "positive"
    assert extract_self_judgment_polarity("I believe the answer is 1") == "positive"
    assert extract_self_judgment_polarity("I'm not sure about this") == "negative"
    assert extract_self_judgment_polarity("This is uncertain") == "negative"
    assert extract_self_judgment_polarity("The answer is 5") == "neutral"

def test_extract_response_label_polarity():
    assert extract_response_label_polarity("The answer is 5") == "positive"
    assert extract_response_label_polarity("I am not sure") == "negative"
    assert extract_response_label_polarity("This is uncertain") == "negative"

def test_compute_laab_score():
    # Test neutral
    entry_neutral = {"case_id": "balanced_incorrect_format_valid_002", "response_text": "5"}
    score, applied, judg = compute_laab_score(entry_neutral)
    assert not applied
    assert judg == "neutral"
    # original score for balanced_incorrect_format_valid_002 is 1.0
    # meta_judgment_consistency is 0.5
    # score = 0.7 * 1.0 + 0.3 * 0.5 = 0.7 + 0.15 = 0.85
    assert abs(score - 0.85) < 1e-6

    # Test applied and consistent
    entry_consistent = {"case_id": "balanced_incorrect_format_valid_002", "response_text": "I think the answer is 5"}
    # judg: positive
    # label: positive
    # expected: positive. matches. consistency: 1.0
    # score = 0.7 * 1.0 + 0.3 * 1.0 = 1.0
    score, applied, judg = compute_laab_score(entry_consistent)
    assert applied
    assert judg == "positive"
    assert abs(score - 1.0) < 1e-6

    # Test applied and inconsistent
    entry_inconsistent = {"case_id": "balanced_incorrect_format_valid_002", "response_text": "I think I am uncertain"}
    # judg: positive
    # label: negative
    # expected: positive. mismatch. consistency: 0.0
    # score = 0.7 * 1.0 + 0.3 * 0.0 = 0.7
    score, applied, judg = compute_laab_score(entry_inconsistent)
    assert applied
    assert judg == "positive"
    assert abs(score - 0.7) < 1e-6
