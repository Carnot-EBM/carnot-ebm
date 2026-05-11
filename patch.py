import re

with open('python/carnot/pipeline/cocom.py', 'r') as f:
    code = f.read()

# 1. Update __init__
old_init = """    def __init__(self, learning_rate: float = 0.1, memory_size: int = 10, parameter_dim: int = 16):
        \"\"\"Initialize the COCOMPipeline.
        
        Args:
            learning_rate: Step size for optimization.
            memory_size: Maximum number of constraints to keep in memory.
            parameter_dim: Dimension of the parameters to optimize.
        \"\"\"
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.parameter_dim = parameter_dim
        self.memory: list[np.ndarray] = []
        self.parameters = np.zeros(self.parameter_dim)
        self.oracle_weights = None"""

new_init = """    def __init__(self, learning_rate: float = 0.1, memory_size: int = 10, parameter_dim: int = 16, similarity_threshold: float = 0.9):
        \"\"\"Initialize the COCOMPipeline.
        
        Args:
            learning_rate: Step size for optimization.
            memory_size: Maximum number of constraints to keep in memory.
            parameter_dim: Dimension of the parameters to optimize.
            similarity_threshold: Cosine similarity threshold for pruning redundant constraints.
        \"\"\"
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.parameter_dim = parameter_dim
        self.similarity_threshold = similarity_threshold
        self.memory: list[np.ndarray] = []
        self.parameters = np.zeros(self.parameter_dim)
        self.oracle_weights = None"""

code = code.replace(old_init, new_init)

# 2. Update update()
old_update = """        # Store the new constraint gradient
        if len(self.memory) >= self.memory_size:
            # Evict oldest if we hit the memory budget
            self.memory.pop(0)
            
        # Normalize the constraint gradient before storing to avoid numerical instability
        norm_c = np.linalg.norm(constraint_grad)
        if norm_c > 1e-8:
            self.memory.append(constraint_grad / norm_c)"""

new_update = """        # Normalize the constraint gradient before storing to avoid numerical instability
        norm_c = np.linalg.norm(constraint_grad)
        if norm_c > 1e-8:
            new_c = constraint_grad / norm_c
            is_redundant = False
            if self.similarity_threshold is not None:
                for prior_c in self.memory:
                    if float(np.dot(prior_c, new_c)) >= self.similarity_threshold:
                        is_redundant = True
                        break
            
            if not is_redundant:
                # Store the new constraint gradient
                if len(self.memory) >= self.memory_size:
                    # Evict oldest if we hit the memory budget
                    self.memory.pop(0)
                self.memory.append(new_c)"""

code = code.replace(old_update, new_update)

# 3. Update update_with_epsilon()
old_epsilon = """        # Store the new constraint gradient
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
            
        norm_c = np.linalg.norm(constraint_grad)
        if norm_c > 1e-8:
            self.memory.append(constraint_grad / norm_c)"""

new_epsilon = """        norm_c = np.linalg.norm(constraint_grad)
        if norm_c > 1e-8:
            new_c = constraint_grad / norm_c
            is_redundant = False
            if self.similarity_threshold is not None:
                for prior_c in self.memory:
                    if float(np.dot(prior_c, new_c)) >= self.similarity_threshold:
                        is_redundant = True
                        break
            
            if not is_redundant:
                if len(self.memory) >= self.memory_size:
                    self.memory.pop(0)
                self.memory.append(new_c)"""

code = code.replace(old_epsilon, new_epsilon)

with open('python/carnot/pipeline/cocom.py', 'w') as f:
    f.write(code)

