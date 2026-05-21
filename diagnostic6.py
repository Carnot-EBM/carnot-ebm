import sys
sys.path.insert(0, 'python')
from carnot.pipeline.verify_repair import VerifyRepairPipeline

def main():
    pipeline = VerifyRepairPipeline(model=None, use_odar=True, jepa_fast_path_threshold=0.2)
    
    q_syn = "What is 2+3?"
    r_syn = "The answer is 6."
    result_syn = pipeline.verify(q_syn, r_syn)
    print(f"Synthetic energy: {result_syn.energy}, Constraints: {len(result_syn.constraints)}")
    for c in result_syn.constraints:
        print(f"Type: {c.constraint_type}, Meta keys: {list(c.metadata.keys())}, Energy: {c.metadata.get('energy', 'N/A')}")

if __name__ == "__main__":
    main()
