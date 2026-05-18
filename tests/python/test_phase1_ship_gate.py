import os
import pytest
from unittest import mock
from carnot.phase1_ship_gate import check_pypi, check_hf, evaluate_gate

@mock.patch("urllib.request.urlopen")
def test_check_pypi(mock_urlopen):
    mock_resp = mock.MagicMock()
    mock_resp.read.return_value = b'{"info": {"version": "0.1.0"}}'
    mock_urlopen.return_value = mock_resp
    
    reachable, published, version = check_pypi()
    assert reachable is True
    assert published is True
    assert version == "0.1.0"

@mock.patch("urllib.request.urlopen")
def test_check_hf(mock_urlopen):
    mock_resp = mock.MagicMock()
    mock_resp.read.return_value = b'[{"id": "Carnot-EBM/model"}]'
    mock_urlopen.return_value = mock_resp
    
    reachable, mirror_up = check_hf()
    assert reachable is True
    assert mirror_up is True

@mock.patch("carnot.phase1_ship_gate.check_pypi")
@mock.patch("carnot.phase1_ship_gate.check_hf")
@mock.patch("glob.glob")
@mock.patch("os.path.exists")
def test_evaluate_gate(mock_exists, mock_glob, mock_check_hf, mock_check_pypi):
    mock_check_pypi.return_value = (True, True, "1.0.0")
    mock_check_hf.return_value = (True, True)
    
    def glob_side_effect(pattern):
        if "mcp" in pattern: return ["docs/mcp.md"]
        if "cli" in pattern: return ["docs/cli.md"]
        if "workflows" in pattern: return [".github/workflows/ci.yml"]
        return []
    mock_glob.side_effect = glob_side_effect
    
    mock_exists.return_value = True
    
    result = evaluate_gate()
    assert result["phase1_ship_gate_met"] is True
    assert result["pypi_published"] is True
    assert result["hf_mirror_up"] is True
    assert result["mcp_docs_present"] is True
    assert result["cli_docs_present"] is True
    assert result["external_reproducer_exists"] is True
    assert len(result["missing_criteria"]) == 0

@mock.patch("carnot.phase1_ship_gate.check_pypi")
@mock.patch("carnot.phase1_ship_gate.check_hf")
@mock.patch("glob.glob")
@mock.patch("os.path.exists")
def test_evaluate_gate_failures(mock_exists, mock_glob, mock_check_hf, mock_check_pypi):
    mock_check_pypi.return_value = (True, False, None)
    mock_check_hf.return_value = (True, False)
    
    mock_glob.return_value = []
    mock_exists.return_value = False
    
    result = evaluate_gate()
    assert result["phase1_ship_gate_met"] is False
    assert result["pypi_published"] is False
    assert result["hf_mirror_up"] is False
    assert result["mcp_docs_present"] is False
    assert result["cli_docs_present"] is False
    assert result["external_reproducer_exists"] is False
    assert len(result["missing_criteria"]) == 5
