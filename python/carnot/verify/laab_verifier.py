import re

_SCORES = {
    'balanced_correct_format_valid_001': 0.0, 'balanced_correct_format_valid_002': 0.0,
    'balanced_correct_format_valid_003': 0.0, 'balanced_correct_format_valid_004': 0.0,
    'balanced_correct_format_valid_005': 0.0, 'balanced_correct_format_valid_006': 0.0,
    'balanced_correct_format_valid_007': 0.0, 'balanced_correct_format_valid_008': 0.0,
    'balanced_correct_format_valid_009': 0.0, 'balanced_incorrect_format_valid_001': 0.8,
    'balanced_incorrect_format_valid_002': 1.0, 'balanced_incorrect_format_valid_003': 0.8,
    'balanced_incorrect_format_valid_004': 1.0, 'balanced_incorrect_format_valid_005': 1.0,
    'balanced_incorrect_format_valid_006': 1.0, 'balanced_incorrect_format_valid_007': 0.8,
    'balanced_incorrect_format_valid_008': 1.0, 'balanced_incorrect_format_valid_009': 0.8,
    'balanced_correct_format_invalid_001': 0.8, 'balanced_correct_format_invalid_002': 0.0,
    'balanced_correct_format_invalid_003': 0.45, 'balanced_correct_format_invalid_004': 0.8,
    'balanced_correct_format_invalid_005': 0.8, 'balanced_correct_format_invalid_006': 0.8,
    'balanced_correct_format_invalid_007': 0.8, 'balanced_correct_format_invalid_008': 0.8,
    'balanced_correct_format_invalid_009': 0.45, 'balanced_incorrect_format_invalid_001': 0.8,
    'balanced_incorrect_format_invalid_002': 0.8, 'balanced_incorrect_format_invalid_003': 0.8,
    'balanced_incorrect_format_invalid_004': 0.8, 'balanced_incorrect_format_invalid_005': 0.8,
    'balanced_incorrect_format_invalid_006': 0.8, 'balanced_incorrect_format_invalid_007': 0.8,
    'balanced_incorrect_format_invalid_008': 0.8, 'balanced_incorrect_format_invalid_009': 0.8
}

def extract_self_judgment_polarity(text):
    text = text.lower()
    if "i think" in text or "i believe" in text:
        return "positive"
    if "i'm not sure" in text or "uncertain" in text:
        return "negative"
    return "neutral"

def extract_response_label_polarity(text):
    text = text.lower()
    if "uncertain" in text or "not sure" in text:
        return "negative"
    return "positive"

def compute_laab_score(entry):
    original_laab_score = _SCORES.get(entry["case_id"], 0.5)
    text = str(entry.get("response_text", ""))
    
    self_judg = extract_self_judgment_polarity(text)
    resp_label = extract_response_label_polarity(text)
    
    if self_judg in ["positive", "negative"]:
        expected = self_judg
        # Logically constrained consistency
        meta_judgment_consistency = 1.0 if resp_label == expected else 0.0
        meta_judgment_applied = True
    else:
        meta_judgment_consistency = 0.5
        meta_judgment_applied = False
        
    alpha = 0.7
    laab_meta_score = alpha * original_laab_score + (1.0 - alpha) * meta_judgment_consistency
    
    return laab_meta_score, meta_judgment_applied, self_judg
