"""Tests for the experiment 2012 E2E script wrapper.

Spec: REQ-2012-E2E, SCENARIO-2012-E2E
"""

import sys
from pathlib import Path

# Add project root and python to path
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "python"))

def test_experiment_2012_e2e_wrapper_exists():
    """Verify that the experiment 2012 wrapper script exists.
    
    SCENARIO-2012-E2E
    """
    script_path = root_dir / "scripts" / "experiment_2012_e2e.py"
    assert script_path.exists()
