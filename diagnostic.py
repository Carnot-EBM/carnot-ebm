import sys
import json

sys.path.insert(0, 'python')
from carnot.pipeline.verify_repair import VerifyRepairPipeline

def main():
    print("Loading FoVer example...")
    fover_example = None
    with open('data/fover_corpus.jsonl', 'r') as f:
        for line in f:
            fover_example = json.loads(line)
            break

    print(f"FoVer response: {fover_example['step_text']}")
    
    print("Initializing pipeline...")
    pipeline = VerifyRepairPipeline()
    
    print("Verifying FoVer example...")
    fover_result = pipeline.verify("What is the question?", fover_example['step_text'])
    fover_energy = fover_result.energy
    print(f"FoVer energy: {fover_energy}")
    print(f"FoVer violations: {len(fover_result.violations)}")
    print(f"FoVer certificate: {fover_result.certificate}")
    
    print("\nVerifying synthetic Qwen example...")
    synthetic_result = pipeline.verify("What is 2+3?", "**Answer:** 5")
    synthetic_energy = synthetic_result.energy
    print(f"Synthetic energy: {synthetic_energy}")
    print(f"Synthetic violations: {len(synthetic_result.violations)}")
    print(f"Synthetic certificate: {synthetic_result.certificate}")

if __name__ == '__main__':
    main()
