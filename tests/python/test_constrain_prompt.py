import sys
import pytest
from carnot.solvers.constrain_prompt import ConstrainPromptCompiler

class MockMessage:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)

class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

class MockChatCompletions:
    def create(self, model, messages, temperature):
        rules = messages[0]["content"]
        if "A must be True" in rules:
            code = "def validate_constraints(assignment: dict) -> bool:\n    return assignment.get('A') is True\n"
        elif "A and B must have different values" in rules:
            code = "def validate_constraints(assignment: dict) -> bool:\n    return assignment.get('A') != assignment.get('B')\n"
        elif "Invalid code test" in rules:
            code = "def some_other_function(assignment: dict) -> bool:\n    return True\n"
        elif "Markdown test" in rules:
            code = "```python\ndef validate_constraints(assignment: dict) -> bool:\n    return assignment.get('A') is True\n```"
        elif "Markdown plain test" in rules:
            code = "```\ndef validate_constraints(assignment: dict) -> bool:\n    return assignment.get('A') is True\n```"
        else:
            code = "def validate_constraints(assignment: dict) -> bool:\n    return True\n"
        return MockResponse(code)

class MockChat:
    def __init__(self):
        self.completions = MockChatCompletions()

class MockOpenAI:
    def __init__(self, base_url=None, api_key=None):
        self.chat = MockChat()

def mock_openai_module(monkeypatch):
    mock_module = type("openai", (), {"OpenAI": MockOpenAI})
    monkeypatch.setitem(sys.modules, "openai", mock_module)

def test_compile_valid_rule(monkeypatch):
    """
    Test compiling a valid rule.
    Spec: REQ-CONSTRAIN-001, SCENARIO-CONSTRAIN-001
    """
    mock_openai_module(monkeypatch)
    
    compiler = ConstrainPromptCompiler()
    validator = compiler.compile("A must be True")
    
    assert validator({"A": True}) is True
    assert validator({"A": False}) is False

def test_compile_xor_rule(monkeypatch):
    """
    Test compiling an XOR rule.
    Spec: REQ-CONSTRAIN-001, SCENARIO-CONSTRAIN-001
    """
    mock_openai_module(monkeypatch)
    
    compiler = ConstrainPromptCompiler()
    validator = compiler.compile("A and B must have different values")
    
    assert validator({"A": True, "B": False}) is True
    assert validator({"A": True, "B": True}) is False

def test_compile_invalid_code(monkeypatch):
    """
    Test compiling code that does not define the expected function.
    Spec: REQ-CONSTRAIN-001
    """
    mock_openai_module(monkeypatch)
    
    compiler = ConstrainPromptCompiler()
    with pytest.raises(ValueError, match="Generated code does not contain"):
        compiler.compile("Invalid code test")

def test_compile_markdown_code(monkeypatch):
    """
    Test compiling markdown block code.
    Spec: REQ-CONSTRAIN-001
    """
    mock_openai_module(monkeypatch)
    
    compiler = ConstrainPromptCompiler()
    validator = compiler.compile("Markdown test")
    assert validator({"A": True}) is True

def test_compile_markdown_plain_code(monkeypatch):
    """
    Test compiling plain markdown block code.
    Spec: REQ-CONSTRAIN-001
    """
    mock_openai_module(monkeypatch)
    
    compiler = ConstrainPromptCompiler()
    validator = compiler.compile("Markdown plain test")
    assert validator({"A": True}) is True

def test_compile_missing_openai(monkeypatch):
    """
    Test when openai is not installed.
    Spec: REQ-CONSTRAIN-001
    """
    monkeypatch.setitem(sys.modules, "openai", None)
    
    compiler = ConstrainPromptCompiler()
    with pytest.raises(ImportError, match="openai not installed"):
        compiler.compile("A must be True")
