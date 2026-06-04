"""Test the Exp 3821 latent symbol bridge artifact builder.

References: REQ-3821, SCENARIO-3821.
"""

from unittest.mock import patch, MagicMock
from scripts.experiments.experiment_3821_latent_symbol_bridge_unblocked import (
    run_preconditions_check,
    build_artifact,
    main,
)
import os

@patch("scripts.experiments.experiment_3821_latent_symbol_bridge_unblocked.os.path.exists")
def test_run_preconditions_check_torch_avail(mock_exists):
    """Test preconditions with torch available."""
    # Test REQ-3821: Must check preconditions before running
    mock_exists.return_value = True
    
    # We don't want to actually load torch because of the memory leak watchdog
    import sys
    sys.modules["torch"] = MagicMock()
    
    # Mock src.nn.models.trm import
    mock_src = MagicMock()
    sys.modules["src.nn.models.trm"] = mock_src
    
    try:
        res = run_preconditions_check()
        assert "torch_available" in res
        assert "trm_pretrained_checkpoint_available" in res
        assert res["bounded_tiny_train_feasible_under_20min"] == True
    finally:
        del sys.modules["torch"]
        del sys.modules["src.nn.models.trm"]

@patch("scripts.experiments.experiment_3821_latent_symbol_bridge_unblocked.os.path.exists")
def test_run_preconditions_check_trm_import_fails(mock_exists):
    mock_exists.return_value = True
    import sys
    sys.modules["torch"] = MagicMock()
    if "src.nn.models.trm" in sys.modules:
        del sys.modules["src.nn.models.trm"]
        
    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "src.nn.models.trm":
            raise ImportError("mock import error")
        return real_import(name, *args, **kwargs)
    
    with patch("builtins.__import__", side_effect=mock_import):
        res = run_preconditions_check()
        assert not res["bounded_tiny_train_feasible_under_20min"]
    
    del sys.modules["torch"]

@patch("scripts.experiments.experiment_3821_latent_symbol_bridge_unblocked.os.path.exists")
def test_run_preconditions_check_torch_not_avail(mock_exists):
    mock_exists.return_value = False
    import sys
    if "torch" in sys.modules:
        del sys.modules["torch"]
    
    # Force ImportError
    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("mock import error")
        return real_import(name, *args, **kwargs)
    
    with patch("builtins.__import__", side_effect=mock_import):
        res = run_preconditions_check()
        assert not res["torch_available"]


@patch("scripts.experiments.experiment_3821_latent_symbol_bridge_unblocked.run_preconditions_check")
def test_build_artifact_blocked(mock_precond):
    """Test artifact building when blocked."""
    # Test SCENARIO-3821: Fallback if tiny-train infeasible
    mock_precond.return_value = {
        "torch_available": False,
        "trm_pretrained_checkpoint_available": False,
        "bounded_tiny_train_feasible_under_20min": False,
    }
    artifact = build_artifact()
    assert artifact["honest_verdict"] == "blocked_trm_checkpoint_not_available"
    assert artifact["n_trajectories"] == 0

@patch("scripts.experiments.experiment_3821_latent_symbol_bridge_unblocked.run_preconditions_check")
def test_build_artifact_falsified(mock_precond):
    """Test artifact building when successful falsification occurs."""
    mock_precond.return_value = {
        "torch_available": True,
        "trm_pretrained_checkpoint_available": False,
        "bounded_tiny_train_feasible_under_20min": True,
    }
    
    # We will mock the import of TRMModule and torch
    import sys
    sys.modules["torch"] = MagicMock()
    
    class MockTRMModule:
        pass
        
    mock_src = MagicMock()
    mock_src.nn.models.trm.TRMModule = MockTRMModule
    sys.modules["src"] = mock_src
    sys.modules["src.nn"] = mock_src.nn
    sys.modules["src.nn.models"] = mock_src.nn.models
    sys.modules["src.nn.models.trm"] = mock_src.nn.models.trm
    
    try:
        artifact = build_artifact()
        assert "FALSIFIED" in artifact["honest_verdict"]
        assert artifact["n_trajectories"] == 100
        assert artifact["inference_substrate"] == "TRM on CPU/GPU via nano-trm"
    finally:
        if "src" in sys.modules:
            del sys.modules["src"]
        if "src.nn" in sys.modules:
            del sys.modules["src.nn"]
        if "src.nn.models" in sys.modules:
            del sys.modules["src.nn.models"]
        if "src.nn.models.trm" in sys.modules:
            del sys.modules["src.nn.models.trm"]
        if "torch" in sys.modules:
            del sys.modules["torch"]

@patch("scripts.experiments.experiment_3821_latent_symbol_bridge_unblocked.run_preconditions_check")
def test_build_artifact_import_error(mock_precond):
    """Test artifact building when nano-trm import fails."""
    mock_precond.return_value = {
        "torch_available": True,
        "trm_pretrained_checkpoint_available": False,
        "bounded_tiny_train_feasible_under_20min": True,
    }
    
    # ensure 'src' is NOT in sys.modules and mock torch
    import sys
    sys.modules["torch"] = MagicMock()
    if "src" in sys.modules:
        del sys.modules["src"]
    if "src.nn.models.trm" in sys.modules:
        del sys.modules["src.nn.models.trm"]
        
    try:
        artifact = build_artifact()
        assert "blocked_trm_nano_trm_import_failed" in artifact["honest_verdict"]
    finally:
        if "torch" in sys.modules:
            del sys.modules["torch"]

@patch("scripts.experiments.experiment_3821_latent_symbol_bridge_unblocked.build_artifact")
@patch("scripts.experiments.experiment_3821_latent_symbol_bridge_unblocked.Path.mkdir")
@patch("scripts.experiments.experiment_3821_latent_symbol_bridge_unblocked.Path.open")
@patch("scripts.experiments.experiment_3821_latent_symbol_bridge_unblocked.json.dump")
def test_main(mock_dump, mock_open, mock_mkdir, mock_build):
    mock_build.return_value = {"test": 1}
    main()
    mock_build.assert_called_once()
    mock_dump.assert_called_once()
    
