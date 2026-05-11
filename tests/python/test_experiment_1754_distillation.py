import numpy as np
from carnot.pipeline.continual_memory import ContinualMemory

def test_continual_memory_distill():
    # REQ-LEARN-1754: Semantic Distillation
    memory = ContinualMemory()
    
    for i in range(10):
        if i < 5:
            vec = np.array([1.0, 1.0]) + np.random.normal(0, 0.1, 2)
            memory.add_state(vec, {"id": i, "cluster": "A"})
        else:
            vec = np.array([10.0, 10.0]) + np.random.normal(0, 0.1, 2)
            memory.add_state(vec, {"id": i, "cluster": "B"})
            
    assert len(memory.memory_states) == 10
    memory.distill(n_clusters=2)
    assert len(memory.memory_states) == 2
    clusters = {state["metadata"]["cluster"] for state in memory.memory_states}
    assert clusters == {"A", "B"}
    
def test_continual_memory_distill_few_states():
    # SCENARIO-LEARN-1754
    memory = ContinualMemory()
    memory.add_state(np.array([1.0]), {"id": 1})
    memory.distill(n_clusters=2)
    assert len(memory.memory_states) == 1
