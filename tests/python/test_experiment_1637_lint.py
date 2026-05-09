import os
import json
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.abspath("."))

import scripts.experiment_1637_lint as exp1637

def test_vivado_not_installed():
    """
    Test that checking for vivado and linting correctly handle when vivado is not installed.
    Verifies REQ-KAN-1637 and SCENARIO-KAN-1637.
    """
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert not exp1637.check_vivado()
        assert not exp1637.run_lint()

def test_vivado_installed_lint_passed():
    """
    Test that checking for vivado and linting correctly handle when vivado is installed and lint passes.
    Verifies REQ-KAN-1637 and SCENARIO-KAN-1637.
    """
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        with patch("os.path.exists", return_value=True):
            assert exp1637.check_vivado()
            assert exp1637.run_lint()

def test_main_outputs_json():
    """
    Test that main outputs the correctly structured JSON artifact.
    Verifies REQ-KAN-1637 and SCENARIO-KAN-1637.
    """
    with patch("scripts.experiment_1637_lint.check_vivado", return_value=True):
        with patch("scripts.experiment_1637_lint.run_lint", return_value=True):
            exp1637.main()
            
            assert os.path.exists("results/experiment_1637_vivado_lint.json")
            with open("results/experiment_1637_vivado_lint.json") as f:
                data = json.load(f)
                assert data["vivado_installed"] is True
                assert data["lint_passed"] is True
                assert data["experiment_id"] == "1637"
