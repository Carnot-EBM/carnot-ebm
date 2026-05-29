import pytest
from carnot.eval.fr11_online_verifier_memory_nonforgetting_v4 import VerifierMemoryController, run_experiment_3334

def test_verifier_memory_controller():
    controller = VerifierMemoryController()
    
    # Test rejection (missing id)
    controller.observe({"task": "new", "verified": True})
    assert controller.rejected == 1
    assert controller.accepted == 0
    
    # Test rejection (task == old)
    controller.observe({"id": "1", "task": "old", "verified": True})
    assert controller.rejected == 2
    
    # Test rejection (verified == False)
    controller.observe({"id": "2", "task": "new", "verified": False})
    assert controller.rejected == 3
    
    # Test acceptance
    controller.observe({"id": "3", "task": "new", "verified": True})
    assert controller.accepted == 1
    assert "3" in controller.memory
    
    # Test rollback logic
    for i in range(4, 13):
        controller.observe({"id": str(i), "task": "new", "verified": True})
        
    # We added 10 total items, so rollback should trigger once
    assert controller.accepted == 10
    assert controller.rollbacks == 1
    assert "12" not in controller.memory  # The 10th item triggered rollback

def test_run_experiment_3334():
    artifact = run_experiment_3334()
    
    # Verify required schema fields
    required_fields = [
        "honest_verdict", "inference_substrate", "random_seed", 
        "reproducibility_checksum", "duration_s", "n_update_cases", 
        "n_new_eval_cases", "n_old_holdout_cases", "new_task_delta", 
        "old_task_delta", "false_positive_delta", "rollback_count", 
        "fr11_nonforgetting_ready", "blocked_reasons"
    ]
    
    for field in required_fields:
        assert field in artifact
        
    assert artifact["fr11_nonforgetting_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")