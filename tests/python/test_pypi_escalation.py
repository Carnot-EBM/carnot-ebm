import json
import os
import unittest
from unittest.mock import patch, MagicMock

from carnot.pypi_escalation import check_pypi_escalation, run_escalation

class TestPyPIEscalation(unittest.TestCase):
    @patch('subprocess.run')
    @patch('urllib.request.urlopen')
    def test_escalation_urllib_fallback_failure(self, mock_urlopen, mock_subproc):
        # Setup mocks
        mock_subproc.return_value = MagicMock(returncode=1) # gh not available
        
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "workflow_runs": [
                {
                    "id": 25964771166,
                    "head_branch": "v0.1.0b1",
                    "status": "completed",
                    "conclusion": "failure"
                }
            ]
        }).encode('utf-8')
        # We need mock_urlopen to be a context manager returning mock_response
        mock_urlopen.return_value.__enter__.return_value = mock_response

        artifact = check_pypi_escalation(2041)
        
        self.assertEqual(artifact["schema"], "carnot.pypi_escalation.v1")
        self.assertEqual(artifact["experiment"], 2041)
        self.assertEqual(artifact["workflow_run_status"], "completed")
        self.assertEqual(artifact["workflow_run_conclusion"], "failure")
        self.assertEqual(artifact["actionable_next_step"], "investigate_failure")
        self.assertIn("blocked_gh_cli_unavailable_fallback_to_urllib", artifact["preconditions_checked"])
        self.assertFalse(artifact["re_trigger_attempted"])

    @patch('subprocess.run')
    @patch('urllib.request.urlopen')
    def test_escalation_cancelled(self, mock_urlopen, mock_subproc):
        mock_subproc.return_value = MagicMock(returncode=1) # gh not available
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "workflow_runs": [
                {
                    "id": 123,
                    "head_branch": "v0.1.0b1",
                    "status": "cancelled",
                    "conclusion": "cancelled"
                }
            ]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        artifact = check_pypi_escalation(2041)
        self.assertTrue(artifact["re_trigger_attempted"])
        self.assertEqual(artifact["re_trigger_outcome"], "failed_gh_cli_unavailable")
        
    @patch('shutil.which')
    @patch('urllib.request.urlopen')
    def test_escalation_waiting(self, mock_urlopen, mock_which):
        mock_which.return_value = None
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "workflow_runs": [
                {
                    "id": 123,
                    "head_branch": "v0.1.0b1",
                    "status": "waiting",
                    "conclusion": None
                }
            ]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        artifact = check_pypi_escalation(2041)
        self.assertEqual(artifact["workflow_run_status"], "waiting")
        self.assertEqual(artifact["actionable_next_step"], "operator_approve")

    @patch('carnot.pypi_escalation.check_pypi_escalation')
    def test_run_escalation(self, mock_check):
        mock_check.return_value = {"test": "data"}
        test_path = "test_escalation.json"
        try:
            run_escalation(2041, test_path)
            self.assertTrue(os.path.exists(test_path))
            with open(test_path, "r") as f:
                data = json.load(f)
                self.assertEqual(data, {"test": "data"})
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)
