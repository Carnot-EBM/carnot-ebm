# Getting Started with Carnot

Carnot verifies LLM output by extracting constraints (arithmetic, code, logic) and checking them against energy-based models. If something is wrong, it tells you what failed and can guide the LLM to fix it.

## Install

```bash
pip install -e ".[dev]"
```

This installs the `carnot` CLI and the Python library. Requires Python 3.11+.

> JAX runs on CPU by default. For GPU acceleration, install with `pip install carnot[cuda]`.
> On AMD/ROCm systems, force CPU mode: `JAX_PLATFORMS=cpu`.

## Quick Start (5 lines)

```python
from carnot.pipeline import VerifyRepairPipeline

pipeline = VerifyRepairPipeline()
result = pipeline.verify("What is 15 + 27?", "15 + 27 = 42")
print(result.verified)    # True
print(result.violations)  # []
```

That's it. The pipeline extracts the arithmetic claim `15 + 27 = 42`, checks it, and confirms it's correct. No GPU, no model download, no configuration.

### Catching a wrong answer

```python
result = pipeline.verify("What is 15 + 27?", "15 + 27 = 43")
print(result.verified)    # False
print(result.violations)  # [ConstraintResult: "15 + 27 = 43 (correct: 42)"]
```

### Verifying code

```python
code_response = '''```python
def double(x: int) -> int:
    return x * 2
```'''

result = pipeline.verify("Write a double function", code_response, domain="code")
print(result.verified)     # True
print(result.constraints)  # type annotations, return type checks, etc.
```

### Multi-domain verification

```python
pipeline = VerifyRepairPipeline(domains=["arithmetic", "logic"])
result = pipeline.verify(
    "If it rains, the ground is wet. What is 10 - 3?",
    "If it rains, then the ground is wet. 10 - 3 = 7."
)
# Checks both the logical implication and the arithmetic
```

## CLI

The `carnot` CLI verifies Python functions against test cases:

```bash
# Verify a function with test cases
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4" --test "(7,13):1"

# Add property-based testing (random inputs)
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4" --properties

# Score activations with a pre-trained EBM
carnot score --list-models
carnot score --model per-token-ebm-qwen35-08b-nothink --activations-file data.safetensors
```

### Test case format

Tests use `input:expected` where both sides are Python literals:

```
5:120             # f(5) == 120
(12,8):4          # f(12, 8) == 4
([3,1,2],):[1,2,3]  # f([3,1,2]) == [1,2,3]
```

## MCP Server Setup (Claude Code)

Carnot can run as an MCP server so Claude Code verifies its own outputs during conversation.

### 1. Add to your Claude Code settings

Add this to `~/.claude/settings.json` (or `.claude/settings.json` in your project):

```json
{
  "mcpServers": {
    "carnot-verify": {
      "command": "python",
      "args": ["-m", "carnot.mcp"],
      "env": {
        "JAX_PLATFORMS": "cpu"
      }
    }
  }
}
```

> Replace `python` with the full path to your venv Python if needed (e.g., `/path/to/venv/bin/python`).

### 2. Restart Claude Code

After restarting, these tools become available to the LLM:

| Tool | What it does |
|------|-------------|
| `verify_code` | Run structural tests on Python functions |
| `verify_with_properties` | Property-based testing with random inputs |
| `verify_llm_output` | Verify LLM responses via constraint extraction |
| `verify_and_repair` | Verify and get natural-language repair feedback |
| `list_domains` | List available constraint extraction domains |

### 3. How it works in practice

```
User:  What is 347 + 258?
Claude: 347 + 258 = 605.
        [Claude calls verify_llm_output tool]
        Tool result: {verified: true}
Claude: 347 + 258 = 605. (verified)
```

When verification fails, Claude gets specific feedback about what went wrong and can self-correct.

## Next Steps

- [Concepts](concepts.md) -- understand constraint verification in plain English
- [API Reference](api-reference.md) -- all public classes and methods
- [Examples](https://github.com/Carnot-EBM/carnot-ebm/tree/main/examples) -- runnable integration examples
- [Technical Writeup](technical-writeup.html) -- the research behind Carnot
