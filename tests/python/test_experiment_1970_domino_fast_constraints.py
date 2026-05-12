import json
import os
import time
from carnot.inference.grammar import LegacyTriePath, DominoGrammarMasker

def test_domino_fast_constraints():
    """
    Test DOMINO-style speculative grammar masking (REQ-INFER-1970).
    Verifies exact-acceptance equivalence vs the legacy trie path and writes the experiment artifact.
    """
    patterns = ["SELECT * FROM table", "INSERT INTO table VALUES (1)", "UPDATE table SET a=1"]
    vocab = ["SELECT", " *", " FROM", " table", "IN", "SERT", " UPDATE", " table SET a=1"]
    
    legacy_trie = LegacyTriePath(patterns)
    domino_masker = DominoGrammarMasker(patterns)

    exact_equivalence_count = 0
    total_checks = 0

    prefixes = ["", "SEL", "SELECT", "INS", "UPDATE"]
    
    start_time_legacy = time.perf_counter()
    for prefix in prefixes:
        for token in vocab:
            _ = legacy_trie.can_accept(prefix, token)
            total_checks += 1
    latency_legacy = time.perf_counter() - start_time_legacy

    start_time_domino = time.perf_counter()
    for prefix in prefixes:
        mask = domino_masker.integrate_decoding_loop(prefix, vocab)
        for i, token in enumerate(vocab):
            legacy_accepts = legacy_trie.can_accept(prefix, token)
            domino_accepts = mask[i]
            if legacy_accepts == domino_accepts:
                exact_equivalence_count += 1
    latency_domino = time.perf_counter() - start_time_domino
    
    exact_acceptance_equivalence = (exact_equivalence_count == total_checks)
    assert exact_acceptance_equivalence, "DOMINO masker did not match LegacyTrie exact-acceptance."
    
    os.makedirs("results", exist_ok=True)
    artifact_path = "results/experiment_1970_domino_fast_constraints.json"
    result = {
        "status": "completed",
        "exact_acceptance_equivalence": exact_acceptance_equivalence,
        "latency_legacy_seconds": latency_legacy,
        "latency_domino_seconds": latency_domino,
        "patterns_tested": len(patterns),
        "vocab_size": len(vocab),
        "total_checks": total_checks
    }
    
    with open(artifact_path, "w") as f:
        json.dump(result, f, indent=2)
        
    assert os.path.exists(artifact_path)
