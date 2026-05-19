import pytest
import os
from carnot.fr11.tier2_memory import Tier2ThresholdMemory

def test_tier2_memory(tmp_path):
    db_path = os.path.join(tmp_path, "test.db")
    memory = Tier2ThresholdMemory(db_path=db_path)
    
    # 32 examples minimum
    examples = [0.8] * 16 + [0.6] * 16
    labels = [1] * 16 + [0] * 16
    
    memory.update_domain_delta("domain1", examples, labels)
    
    delta = memory.get_domain_delta("domain1")
    assert delta > 0.0 # mean is 0.7, so delta is 0.2
    
    adapted = memory.apply_delta("domain1", 0.8)
    assert adapted < 0.8

    # Ensure constraint raises on < 32 examples
    with pytest.raises(ValueError):
        memory.update_domain_delta("domain2", [0.8], [1])
