import os
import numpy as np
from carnot.pipeline.cocom import COCOMPipeline

def main():
    pipeline = COCOMPipeline(learning_rate=0.1, memory_size=5, parameter_dim=2)
    obj_grad = np.array([0.0, 0.0])
    const_grad = np.array([1.0, 0.0])
    epsilon = 0.5
    
    # Run the epsilon update
    pipeline.update_with_epsilon(obj_grad, const_grad, epsilon)
    
    output_path = "/home/ianblenke/github.com/ianblenke/carnot/results/experiment_1843_epsilon_ocl.json"
    pipeline.write_artifact(output_path, experiment_id="1843", honest_verdict="cocom_epsilon_implemented")
    print(f"Wrote artifact to {output_path}")

if __name__ == "__main__":
    main()
