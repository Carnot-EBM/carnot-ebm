import os
import json
import pytest
from carnot.retro_166 import generate_retro

def test_generate_retro_creates_file_and_validates_schema(tmp_path):
    output_path = tmp_path / "experiment_2109_retro.json"
    artifact = generate_retro(str(output_path))
    
    # Verify file was created
    assert output_path.exists()
    
    # Read the file
    with open(output_path) as fh:
        data = json.load(fh)
        
    # Validate required artifact fields
    assert data["schema"] == "carnot.milestone_research_retro.v1"
    assert data["milestone"] == "2026.05.166"
    assert "tasks_summary" in data
    assert len(data["tasks_summary"]) == 4
    assert data["gates_passed_count"] == 3
    assert data["gates_failed_count"] == 1
    
    # Validate actual_agent_backend_distribution
    dist = data["actual_agent_backend_distribution"]
    assert "codex" in dist
    assert "gemini" in dist
    assert "claude" in dist
    
    # Validate paper_v6_carryforward_items
    assert isinstance(data["paper_v6_carryforward_items"], list)
    assert len(data["paper_v6_carryforward_items"]) > 0
    
    # Validate adversarial verify
    assert data["adversarial_verify_flag_count"] == 0
    
    # Validate honest_verdict
    assert data["honest_verdict"].startswith("complete:")
    
    # Validate meta_reflection for the agent routing issue
    assert "meta_reflection" in data
    assert "codex" in data["meta_reflection"]
