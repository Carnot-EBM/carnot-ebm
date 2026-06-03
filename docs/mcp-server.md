# Carnot MCP Server

## Overview
The Carnot Model Context Protocol (MCP) Server provides a hardened, production-grade interface to Carnot's Energy-Based Model (EBM) verification and repair pipeline. By exposing these capabilities over MCP (stdio JSON-RPC), LLM agents like Claude Code or any MCP-compatible client can directly call Carnot's code verification, constraint extraction, and property-based testing tools.

## Installation
The MCP server is included with the `carnot` package. Ensure you have installed the project:

```bash
pip install -e .
```

## Configuration
No special configuration is needed. The server runs locally over standard input/output (stdio). It enforces a 30-second execution timeout and a 10,000-character input limit for stability.

## Available Tools

The server exposes the following tools:

1. **`verify_code`**: Run structural tests on a Python function by passing input/output pairs.
2. **`verify_with_properties`**: Run property-based tests using randomly generated inputs to check invariant properties.
3. **`verify_code_with_pbt`**: Run packaged static checks plus Hypothesis-backed verification for generated Python code.
4. **`verify_llm_output`**: Extract and verify constraints (arithmetic, code, logic, natural language) from an LLM response text.
5. **`verify_stream`**: Verify multiple candidate responses in parallel with early stopping based on energy margins.
6. **`verify_and_repair`**: Run verification and return natural-language feedback that the calling LLM can use to self-repair its output.
7. **`score_agent_outputs`**: Score competing agent responses and return a ranked arbitration result based on energy levels.
8. **`score_candidates`**: Score candidate outputs with the calibrated second-pair detector.
9. **`list_domains`**: List available constraint extraction domains (arithmetic, code, logic, nl) for verification.
10. **`health_check`**: Returns server version, status, and tool counts (useful as a liveness probe).

## Usage Examples

### Example 1: Verifying Python Code
You can pass code and tests to the `verify_code` tool:
```json
{
  "code": "def add(a, b): return a + b",
  "func_name": "add",
  "test_cases": [
    {"args": [1, 2], "expected": 3},
    {"args": [0, 0], "expected": 0}
  ]
}
```

### Example 2: Verifying an LLM Output
You can verify an LLM's natural language or reasoning response using `verify_llm_output`:
```json
{
  "question": "What is 47 + 28?",
  "response": "The answer is 75.",
  "domain": "arithmetic"
}
```

### Example 3: Scoring Candidate Outputs
You can score candidate outputs with the shipped second-pair detector:
```json
{
  "candidates": [
    {
      "candidate_id": "candidate-a",
      "domain": "math",
      "text": "We compute 7 + 5 = 13, so the answer is 13.",
      "confidence": 0.2,
      "ensemble_energy": 0.8
    }
  ],
  "domain": "math"
}
```

### Example 4: Verifying with Properties
You can verify a property of a function:
```json
{
  "code": "def add(a, b): return a + b",
  "func_name": "add",
  "properties": [
    {
      "name": "is_commutative",
      "generator": "pair_int",
      "check": "lambda result, a, b: result == a + b"
    }
  ]
}
```

## Integration with Claude Desktop

To integrate the Carnot MCP server with Claude Desktop, add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "carnot-verify": {
      "command": "python",
      "args": ["-m", "carnot.mcp"]
    }
  }
}
```
Ensure that the `python` command points to the virtual environment where `carnot` is installed.
