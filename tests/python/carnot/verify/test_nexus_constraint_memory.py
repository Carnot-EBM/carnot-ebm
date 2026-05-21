import os
import json
import tempfile
from carnot.verify.nexus_constraint_memory import NexusConstraintMemory

def test_nexus_constraint_memory_record_and_synthesize():
    memory = NexusConstraintMemory()
    
    # Record 3 identical violations to trigger rule synthesis
    memory.record_violation("pattern A", "domain1", 1.0)
    memory.record_violation("pattern A", "domain1", 0.8)
    memory.record_violation("pattern A", "domain1", 0.6)
    
    # Record 2 identical violations (should not synthesize with default min_support=3)
    memory.record_violation("pattern B", "domain1", 1.0)
    memory.record_violation("pattern B", "domain1", 1.0)
    
    rules = memory.synthesize_rules()
    assert len(rules) == 1
    assert "pattern A" in rules[0]
    assert "domain1" in rules[0]
    
    # Check internal rules dicts that memory keeps
    assert len(memory.rules) == 1
    assert memory.rules[0]["pattern"] == "pattern A"
    assert memory.rules[0]["domain"] == "domain1"
    assert memory.rules[0]["count"] == 3
    assert abs(memory.rules[0]["avg_severity"] - (1.0 + 0.8 + 0.6) / 3) < 1e-6

def test_nexus_constraint_memory_add_violation():
    memory = NexusConstraintMemory()
    q = "What is 2+2?"
    a = "It is 5."
    e = 0.95
    
    memory.add_violation(q, a, e)
    
    domain = "ORCA_TTT"
    pattern = f"Repair needed for Q: {q[:30]}... due to {a[:30]}..."
    
    assert domain in memory.violations
    assert pattern in memory.violations[domain]
    assert memory.violations[domain][pattern] == [e]

def test_nexus_constraint_memory_consolidate():
    memory = NexusConstraintMemory()
    # Manually add some redundant rules
    memory.rules = [
        {"pattern": "pattern X", "domain": "domainX", "count": 3, "avg_severity": 1.0},
        {"pattern": "pattern X", "domain": "domainX", "count": 4, "avg_severity": 0.5},
        {"pattern": "pattern Y", "domain": "domainX", "count": 3, "avg_severity": 1.0}
    ]
    
    memory.consolidate()
    assert len(memory.rules) == 2
    
    # The consolidated rule for pattern X should have count 7 and avg severity (3*1.0 + 4*0.5) / 7 = 5.0 / 7
    rule_x = next(r for r in memory.rules if r["pattern"] == "pattern X")
    assert rule_x["count"] == 7
    assert abs(rule_x["avg_severity"] - (5.0 / 7)) < 1e-6

def test_nexus_constraint_memory_save_load():
    memory = NexusConstraintMemory()
    memory.record_violation("pattern A", "domain1", 1.0)
    memory.rules = [{"pattern": "pattern A", "domain": "domain1", "count": 3, "avg_severity": 1.0}]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mem.json")
        memory.save(path)
        
        memory2 = NexusConstraintMemory()
        memory2.load(path)
        
        assert memory2.violations == memory.violations
        assert memory2.rules == memory.rules
