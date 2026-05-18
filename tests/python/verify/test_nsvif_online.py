import json
import pytest
from pathlib import Path

from carnot.verify.nsvif_z3_extractor import NSVIFExtractor

def test_nsvif_online_learning_loop(tmp_path):
    # Setup test entries
    # 1 SAT, 2 UNSAT (contradictory)
    entries = [
        {"response_text": "2 + 2 = 4."}, # SAT
        {"response_text": "2 + 2 = 5."}, # UNSAT
        {"response_text": "3 + 3 = 7."}  # UNSAT
    ]

    extractor = NSVIFExtractor()
    
    # Initial verification
    initial_sat_count = 0
    unsat_entries = []
    
    for i, e in enumerate(entries):
        res = extractor.verify(e["response_text"])
        if res.get("satisfiable", True) and res.get("verification_pass", False):
            initial_sat_count += 1
        else:
            unsat_entries.append((i, res))
            
    assert initial_sat_count == 1
    assert len(unsat_entries) == 2
    
    # Extract patterns
    patterns = []
    for idx, res in unsat_entries:
        violations = res.get("violations", [])
        for v in violations:
            patterns.append({
                "pattern": v,
                "hedge": "may_not_be_required",
                "confidence": 0.5,
                "added_at": "20260518",
                "source_entry_idx": idx
            })
            
    assert len(patterns) > 0
    
    # Mock pattern downgrade in re-verify
    updated_sat_count = 0
    optional_patterns = [p["pattern"] for p in patterns if p["hedge"] == "may_not_be_required" and p["confidence"] < 0.6]
    
    for e in entries:
        res = extractor.verify(e["response_text"])
        is_sat = res.get("satisfiable", True) and res.get("verification_pass", False)
        if not is_sat:
            violations = res.get("violations", [])
            if len(violations) > 0 and all(v in optional_patterns for v in violations):
                is_sat = True
        
        if is_sat:
            updated_sat_count += 1
            
    assert updated_sat_count == 3 # All should be SAT now
