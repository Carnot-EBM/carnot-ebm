import pytest
from typing import Dict
from carnot.eval.fr11_online_verifier_memory_nonforgetting_v5 import VerifierMemoryControllerV5, MemoryControllerUpdateV5, run_experiment_3345

def test_verifier_memory_controller_v5():
    """Test memory controller updates and rollbacks. Satisfies REQ-LEARN-080 and SCENARIO-LEARN-120."""
    controller = VerifierMemoryControllerV5()
    
    # Test rejection (missing id)
    controller.observe({"task": "new", "verified": True}, lambda m: 0.0)
    assert controller.rejected == 1
    assert controller.accepted == 0
    
    # Test rejection (verified == False)
    controller.observe({"id": "2", "task": "new", "verified": False}, lambda m: 0.0)
    assert controller.rejected == 2
    
    # Test false positive guard
    controller.observe({"id": "3", "task": "new", "verified": True, "is_false_positive": True}, lambda m: 0.0)
    assert controller.false_positive_guards == 1
    assert controller.rejected == 3
    
    # Test acceptance
    controller.observe({"id": "4", "task": "new", "verified": True}, lambda m: 0.0)
    assert controller.accepted == 1
    assert "4" in controller.memory
    
    # Test rollback logic (no prev state)
    # Make old task evaluator return > 0.05
    did_rollback = controller.observe({"id": "5", "task": "new", "verified": True}, lambda m: 0.1)
    assert did_rollback is True
    assert controller.rollbacks == 1
    assert controller.rollback_recovered_count == 1
    assert "5" not in controller.memory

    # Test rollback logic (with prev state)
    controller.memory["6"] = MemoryControllerUpdateV5(key="6", action="store", weight=1.0, task_type="new")
    assert "6" in controller.memory
    did_rollback = controller.observe({"id": "6", "task": "new", "verified": True}, lambda m: 0.1)
    assert did_rollback is True
    assert controller.rollbacks == 2
    assert "6" in controller.memory

def test_run_experiment_3345():
    """Verify artifact fields and nonforgetting readiness. Satisfies REQ-LEARN-080."""
    artifact = run_experiment_3345()
    
    required_fields = [
        "honest_verdict", "inference_substrate", "random_seed", 
        "reproducibility_checksum", "duration_s", "files_updated",
        "n_update", "n_new_eval", "n_old_eval", "n_false_positive_guard",
        "new_task_delta", "old_task_delta", "false_positive_delta", 
        "soundness_error_delta", "completeness_error_delta",
        "rollback_count", "rollback_recovered_count", 
        "fr11_nonforgetting_ready", "blocked_reasons"
    ]
    
    for field in required_fields:
        assert field in artifact
        
    assert artifact["fr11_nonforgetting_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
