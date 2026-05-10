"""Tests for FourierCSPExtractor."""

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

from carnot.pipeline.fouriercsp_extractor import FourierCSPExtractor, MultilinearPolynomial, _default_generate


def test_extract_offline_mode():
    """Test offline extraction fallback."""
    if "CARNOT_FORCE_LIVE" in os.environ:
        del os.environ["CARNOT_FORCE_LIVE"]
    
    extractor = FourierCSPExtractor()
    result = extractor.extract("some prompt")
    assert result is not None
    assert result.variables == ["x"]
    assert result.polynomial == "x"


def test_extract_live_mode_success(monkeypatch):
    """Test live extraction success."""
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    
    def mock_generate(prompt):
        return json.dumps({
            "variables": ["a", "b"],
            "expression": "a AND b"
        })
    
    extractor = FourierCSPExtractor(generate_fn=mock_generate)
    result = extractor.extract("a and b must be true")
    assert result is not None
    assert result.variables == ["a", "b"]
    assert result.polynomial == "a * b"


def test_extract_live_mode_markdown(monkeypatch):
    """Test live extraction with markdown format."""
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    
    def mock_generate(prompt):
        return "```json\n" + json.dumps({
            "variables": ["c"],
            "expression": "NOT c"
        }) + "\n```"
    
    extractor = FourierCSPExtractor(generate_fn=mock_generate)
    result = extractor.extract("c is false")
    assert result is not None
    assert result.variables == ["c"]
    assert result.polynomial == "1-c"


def test_extract_live_mode_failure(monkeypatch):
    """Test live extraction handles invalid JSON."""
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    
    def mock_generate(prompt):
        return "not valid json"
    
    extractor = FourierCSPExtractor(generate_fn=mock_generate)
    result = extractor.extract("something")
    assert result is None


def test_extract_live_mode_exception(monkeypatch):
    """Test live extraction handles exception."""
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    
    def mock_generate(prompt):
        raise ValueError("LLM Error")
    
    extractor = FourierCSPExtractor(generate_fn=mock_generate)
    result = extractor.extract("something")
    assert result is None


def test_default_generate(monkeypatch):
    """Test the default generate fallback logic."""
    mock_inference = MagicMock()
    mock_model_loader = MagicMock()
    mock_model_loader.load_model.return_value = ("mock_model", "mock_tokenizer")
    mock_model_loader.generate.return_value = '{"variables": ["y"], "expression": "y"}'
    
    sys.modules["carnot.inference"] = mock_inference
    sys.modules["carnot.inference.model_loader"] = mock_model_loader
    
    res = _default_generate("test prompt")
    assert "variables" in res
