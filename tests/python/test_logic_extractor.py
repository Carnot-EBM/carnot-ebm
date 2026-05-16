"""Tests for LogicExtractor.

Traces to: REQ-VERIFY-1974, SCENARIO-VERIFY-1974
"""

import json
from carnot.pipeline.logic_extractor import LogicExtractor, ContinuousConstraint

def test_extract_valid_json():
    """Test extracting constraints from valid JSON."""
    def mock_generate(prompt):
        return json.dumps([{"type": "limit", "target": "x", "value": 42.0}])
    
    extractor = LogicExtractor(generate_fn=mock_generate)
    result = extractor.extract("some prompt")
    assert len(result) == 1
    assert isinstance(result[0], ContinuousConstraint)
    assert result[0].type == "limit"
    assert result[0].target == "x"
    assert result[0].value == 42.0

def test_extract_invalid_json():
    """Test handling invalid output."""
    def mock_generate(prompt):
        return "Not JSON"
    
    extractor = LogicExtractor(generate_fn=mock_generate)
    result = extractor.extract("some prompt")
    assert len(result) == 0

def test_extract_markdown_json():
    """Test extracting constraints from markdown-wrapped JSON."""
    def mock_generate(prompt):
        return "```json\n[{\"type\": \"max\", \"target\": \"y\", \"value\": 10.0}]\n```"
    
    extractor = LogicExtractor(generate_fn=mock_generate)
    result = extractor.extract("some prompt")
    assert len(result) == 1
    assert result[0].type == "max"
    assert result[0].target == "y"
    assert result[0].value == 10.0

def test_extract_invalid_value():
    """Test handling invalid float values."""
    def mock_generate(prompt):
        return json.dumps([{"type": "max", "target": "y", "value": "not-a-float"}])
    
    extractor = LogicExtractor(generate_fn=mock_generate)
    result = extractor.extract("some prompt")
    assert len(result) == 0

def test_default_generate():
    """Test the default generate fn fallback."""
    extractor = LogicExtractor()
    result = extractor.extract("some prompt")
    assert len(result) == 1
    assert result[0].type == "mock"
    assert result[0].target == "mock"
    assert result[0].value == 0.0
