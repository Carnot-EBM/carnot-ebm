import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path("python").resolve()))

from carnot.eval.fover_memory_leakage_v3 import _score_text_verifiers, _fr11_memory_score, _load_fr11_memory_index, FR11_MEMORY_BOOST

def score_candidate(row, repo_root):
    text = str(row.get("step_text", ""))
    verifier_scores = _score_text_verifiers([text])
    r_score = verifier_scores["tier0r_curry_howard"][0]
    u_score = verifier_scores["tier0u_logical_consistency"][0]
    score = 0.9 * r_score + 0.1 * u_score
    
    memory_index = _load_fr11_memory_index(repo_root)
    if memory_index["question_ids"] or memory_index["prompt_token_sets"]:
        memory_score = _fr11_memory_score(row, memory_index)
        score += FR11_MEMORY_BOOST * memory_score
    return score

if __name__ == "__main__":
    data = json.load(open("data/fover_test_v4.json"))
    repo_root = Path(".").resolve()
    print("Candidate score:", score_candidate(data[0], repo_root))
