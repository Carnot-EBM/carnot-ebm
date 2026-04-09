# Carnot Examples

Production-ready examples showing real use cases for the Carnot EBM verification framework.

## Examples

| Example | What it does |
|---------|-------------|
| [verify_api_responses.py](verify_api_responses.py) | Verify LLM API responses for arithmetic and logical correctness |
| [code_review_pipeline.py](code_review_pipeline.py) | Verify LLM-generated Python code for type errors, undefined variables, and structural issues |
| [batch_verify.py](batch_verify.py) | Batch-verify a JSON file of question/answer pairs and produce a summary report |
| [custom_extractor.py](custom_extractor.py) | Create a domain-specific constraint extractor (units-of-measure checking) |
| [mcp_integration.py](mcp_integration.py) | Configure and test the Carnot MCP server for use with Claude Code |

## Prerequisites

```bash
# Install carnot in development mode
pip install -e ".[dev]"

# Or install from PyPI (when available)
pip install carnot
```

## Running

All examples are standalone scripts. Run with the project venv:

```bash
# Force CPU-only JAX for reproducibility
JAX_PLATFORMS=cpu .venv/bin/python examples/verify_api_responses.py

JAX_PLATFORMS=cpu .venv/bin/python examples/code_review_pipeline.py

# Batch verify from a JSON file (uses built-in sample data if no file given)
JAX_PLATFORMS=cpu .venv/bin/python examples/batch_verify.py
JAX_PLATFORMS=cpu .venv/bin/python examples/batch_verify.py my_qa_pairs.json

# Custom extractor demo
JAX_PLATFORMS=cpu .venv/bin/python examples/custom_extractor.py

# MCP integration setup guide (prints config, does not start server)
JAX_PLATFORMS=cpu .venv/bin/python examples/mcp_integration.py
```

## JSON format for batch_verify.py

```json
[
  {
    "question": "What is 15 + 27?",
    "response": "15 + 27 = 42"
  },
  {
    "question": "Write a Python function to add two numbers",
    "response": "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```"
  }
]
```
