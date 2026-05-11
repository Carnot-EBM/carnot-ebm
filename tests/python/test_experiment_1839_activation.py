import json
from unittest.mock import patch, MagicMock
from scripts.experiment_1839_activation import main

def test_experiment_1839_activation(tmp_path):
    """Test the activation script creates the correct artifact.
    
    Spec: REQ-INFRA-060
    """
    with patch("scripts.experiment_1839_activation.ExperimentTemplate") as mock_tmpl_cls:
        mock_tmpl = MagicMock()
        mock_tmpl_cls.return_value = mock_tmpl
        
        # Setup the mock to return a fake artifact
        mock_artifact = {"experiment": 1839, "status": "success"}
        mock_tmpl.build_result.return_value = mock_artifact
        
        # Setup a real path for output_path
        out_file = tmp_path / "experiment_1839_activation.json"
        mock_tmpl._output_path = out_file
        
        # Bypass the assert_deliverable_written check starting with 'assert'
        setattr(mock_tmpl, "assert_deliverable_written", MagicMock())
        
        main()
        
        mock_tmpl_cls.assert_called_once_with(
            exp_id=1839,
            title="Exp 1839: Archive .142 and Activate .143",
            deliverable="results/experiment_1839_activation.json",
            requires_gpu=False,
        )
        mock_tmpl.setup.assert_called_once()
        mock_tmpl.build_result.assert_called_once()
        getattr(mock_tmpl, "assert_deliverable_written").assert_called_once()
        
        # Verify the file was written
        assert out_file.exists()
        saved_artifact = json.loads(out_file.read_text())
        assert saved_artifact == mock_artifact
