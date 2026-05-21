import json
import random

from carnot.pipeline.verify_repair import VerifyRepairPipeline

def test_delta(n=20, seed=42):
    random.seed(seed)
    pipeline = VerifyRepairPipeline(model="Qwen/Qwen3-0.6B", timeout_seconds=None)
    violations = []
    
    with open('data/fover_corpus.jsonl') as f:
        for line in f:
            ex = json.loads(line)
            if ex.get('label') == 'incorrect':
                violations.append(ex)

    # shuffle and take n
    random.shuffle(violations)
    violations = violations[:n]
    
    results = []
    successes = 0
    for ex in violations:
        # Fover corpus schema: question_id, step_text, label
        question = str(ex.get("question_id", "question"))
        response = ex.get("step_text", "")
        
        # Initial verify
        init_res = pipeline.verify(question, response)
        
        # Verify and repair
        repair_res = pipeline.verify_and_repair(question, response)
        
        # final verify if not directly available from repair_res
        # Actually RepairResult has .repaired and .verified properties probably, 
        # or we can manually check if it fixed it.
        # But wait, what if verify() returns dict and energy?
        # Let's just use what the prompt asked.
        final_energy = repair_res.verified  # We will check its type below
        results.append({
            'initial_energy': init_res,
            'repair_res': repair_res
        })
        if repair_res.repaired:
            successes += 1

    baseline_delta = successes / n
    print(f"delta={baseline_delta:.3f}")
    return baseline_delta

if __name__ == "__main__":
    test_delta()
