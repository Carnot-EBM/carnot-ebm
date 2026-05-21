import sys
sys.path.insert(0, 'python')
from carnot.pipeline.verify_repair import VerifyRepairPipeline

def main():
    pipeline = VerifyRepairPipeline(model=None, use_odar=True, jepa_fast_path_threshold=0.2)
    
    # synthetic that violates semantic consistency
    q_syn = "What is 2+3?"
    r_syn = "The answer is 6."
    print("\nRunning on synthetic wrong answer...")
    result_syn = pipeline.verify(q_syn, r_syn)
    print(f"Synthetic energy: {result_syn.energy}, Constraints: {len(result_syn.constraints)}")

if __name__ == "__main__":
    main()
