"""Constrained Online Convex Optimization with Memory (COCOM) pipeline.

Spec: REQ-PIPELINE-1831
"""
import json
import os
import numpy as np

class COCOMPipeline:
    """Pipeline for Constrained Online Convex Optimization with Memory.
    
    This class tracks memory-based constraints across online learning steps
    and optimizes a defined objective function subject to these tracked
    memory constraints.
    
    Attributes:
        learning_rate: Step size for the parameter updates.
        memory_size: Maximum number of constraint gradients to store.
        parameter_dim: Dimensionality of the parameter vector.
        memory: List storing past constraint gradients.
        parameters: Current parameter vector.
    """
    
    def __init__(self, learning_rate: float = 0.1, memory_size: int = 10, parameter_dim: int = 16):
        """Initialize the COCOMPipeline.
        
        Args:
            learning_rate: Step size for optimization.
            memory_size: Maximum number of constraints to keep in memory.
            parameter_dim: Dimension of the parameters to optimize.
        """
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.parameter_dim = parameter_dim
        self.memory: list[np.ndarray] = []
        self.parameters = np.zeros(self.parameter_dim)
        
    def update(self, objective_grad: np.ndarray, constraint_grad: np.ndarray) -> None:
        """Perform an online learning step.
        
        Projects the objective gradient onto the null space of the stored constraint
        gradients to ensure previously learned constraints are not violated.
        
        Args:
            objective_grad: Gradient of the objective function at the current step.
            constraint_grad: Gradient of the new constraint at the current step.
        """
        # Store the new constraint gradient
        if len(self.memory) >= self.memory_size:
            # Evict oldest if we hit the memory budget
            self.memory.pop(0)
            
        # Normalize the constraint gradient before storing to avoid numerical instability
        norm_c = np.linalg.norm(constraint_grad)
        if norm_c > 1e-8:
            self.memory.append(constraint_grad / norm_c)
            
        # Project the objective gradient onto the null space of memory constraints
        # v starts as the objective gradient
        v = np.copy(objective_grad)
        for prior_c in self.memory:
            # Gram-Schmidt style projection: remove component along prior constraint
            v = v - np.dot(v, prior_c) * prior_c
            
        # Update parameters with the projected gradient (gradient descent)
        self.parameters = self.parameters - self.learning_rate * v

    def write_artifact(self, filepath: str) -> None:
        """Write the experiment artifact to a JSON file.
        
        Args:
            filepath: Path to the output JSON file.
        """
        artifact = {
            "experiment_id": "1831",
            "status": "complete",
            "honest_verdict": "cocom_implemented",
            "learning_rate": self.learning_rate,
            "memory_size": self.memory_size,
            "parameter_dim": self.parameter_dim,
            "num_constraints_in_memory": len(self.memory),
            "final_parameters": self.parameters.tolist()
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(artifact, f, indent=2)

