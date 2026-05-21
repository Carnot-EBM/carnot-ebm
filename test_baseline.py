import sys
sys.path.insert(0, 'python')
import json
from carnot.pipeline.verify_repair import VerifyRepairPipeline

def run_baseline():
    violations = []
    with open('data/fover_corpus.jsonl') as f:
        for line in f:
            ex = json.loads(line)
            if ex.get('label') == 'incorrect':
                violations.append(ex)
                if len(violations) >= 20:
                    break
    
    pipeline = VerifyRepairPipeline()
    results = []
    
    for ex in violations:
        question = ex.get('question_id', '')
        response = ex.get('step_text', '')
        
        # Verify initial
        initial_result = pipeline.verify(question, response)
        initial_energy = initial_result.energy
        
        # Repair
        repair_result = pipeline.verify_and_repair(question, response)
        
        # Verify final
        final_result = pipeline.verify(question, repair_result.final_response)
        final_energy = final_result.energy
        
        improved = final_energy < initial_energy
        results.append({
            'improved': improved,
            'initial_energy': initial_energy,
            'final_energy': final_energy
        })
        print(f"Q: {question}, Initial Energy: {initial_energy}, Final Energy: {final_energy}, Improved: {improved}")
        
    delta = sum(1 for r in results if r['improved']) / len(results)
    print(f"Baseline Delta: {delta}")

if __name__ == '__main__':
    run_baseline()
