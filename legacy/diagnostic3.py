import sys
sys.path.insert(0, 'python')
from carnot.pipeline.verify_repair import VerifyRepairPipeline
import json

def main():
    with open('data/fover_corpus.jsonl') as f:
        fover_example = json.loads(f.readline())
    
    question = "156"
    correct = fover_example.get('step_text', '')

    pipeline = VerifyRepairPipeline(model=None, use_odar=True, jepa_fast_path_threshold=0.2)
    
    print("Running on FoVer example...")
    result_fover = pipeline.verify(question, correct)
    print(f"FoVer energy: {result_fover.energy}, Mode: {result_fover.mode}, Skipped: {result_fover.skipped}")
    print(f"Cert: {result_fover.certificate}")
    
    q_syn = "What is 2+3?"
    r_syn = "**Answer:** 5"
    print("\nRunning on synthetic...")
    result_syn = pipeline.verify(q_syn, r_syn)
    print(f"Synthetic energy: {result_syn.energy}, Mode: {result_syn.mode}, Skipped: {result_syn.skipped}")
    print(f"Cert: {result_syn.certificate}")

if __name__ == "__main__":
    main()
