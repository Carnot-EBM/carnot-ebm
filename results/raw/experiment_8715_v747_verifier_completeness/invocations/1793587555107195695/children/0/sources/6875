import numpy as np
from carnot.pipeline.fr11_epsilon import FR11EpsilonTracker

def run():
    tracker = FR11EpsilonTracker(parameter_dim=8)
    obj_grad = np.array([0.1, 0.2, 0.0, -0.1, 0.5, 0.0, 0.0, -0.2])
    const_grad = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    epsilon = 0.05
    
    tracker.enforce_checks_and_update(obj_grad, const_grad, epsilon)
    tracker.write_experiment_artifact("results/experiment_1848_gemma26_epsilon.json", ["unsloth/gemma-4-26B-A4B-it-GGUF"])
    print("Done")

if __name__ == "__main__":
    run()
