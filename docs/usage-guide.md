# Carnot Tools — Usage Guide

How to use Carnot's MCP server and CLI to verify code in your own projects.

## Installation

```bash
# Clone the repository
git clone https://github.com/Carnot-EBM/carnot-ebm
cd carnot

# Install the Python package (includes CLI)
pip install -e ".[dev]"

# Verify installation
carnot verify --help
```

## CLI: `carnot verify`

The CLI verifies Python functions against test cases and optional property-based tests.

### Basic Usage

```bash
# Verify a function with test cases
carnot verify your_project/utils.py --func gcd --test "(12,8):4" --test "(7,13):1"

# Verify with expected return type
carnot verify your_project/sort.py --func merge_sort --test "([3,1,2],):[1,2,3]" --type list

# Verify with property-based testing (random inputs)
carnot verify your_project/math.py --func factorial --test "5:120" --properties
```

### Test Case Format

Tests use the format `input:expected` where both sides are Python literals:

| Format | Example | Meaning |
|--------|---------|---------|
| Single arg | `5:120` | `f(5) == 120` |
| Multiple args | `(12,8):4` | `f(12, 8) == 4` |
| String args | `"hello":5` | `f("hello") == 5` |
| List args | `([3,1,2],):[1,2,3]` | `f([3,1,2]) == [1,2,3]` |
| Boolean result | `(0,):True` | `f(0) == True` |

### Output

```
============================================================
CARNOT VERIFY
  File:     your_project/utils.py
  Function: gcd
  Type:     int
  Tests:    2
============================================================

--- Structural Tests ---
  [PASS] gcd(12, 8) == 4
  [PASS] gcd(7, 13) == 1

--- Energy Breakdown ---
  [OK] return_type: energy=0.0000 (weighted=0.0000)
  [OK] no_exception: energy=0.0000 (weighted=0.0000)
  [OK] test_pass: energy=0.0000 (weighted=0.0000)

============================================================
  Total energy: 0.0000
  Verdict:      PASS
============================================================
```

Exit code 0 = all tests pass. Exit code 1 = failures found.

### CI Integration

```yaml
# GitHub Actions example
- name: Verify critical functions
  run: |
    carnot verify src/crypto.py --func encrypt --test "('hello','key'):'encrypted'" --properties
    carnot verify src/math.py --func sqrt --test "4:2.0" --test "9:3.0" --type float
```

## MCP Server: Claude Code Integration

The MCP server lets Claude Code automatically verify code it generates.

### Setup

Copy the template to get started: `cp .mcp.json.example .mcp.json`. The default config:

```json
{
  "mcpServers": {
    "carnot-verify": {
      "command": ".venv/bin/python3",
      "args": ["tools/verify-mcp/server.py"],
      "env": {
        "PYTHONPATH": ".:python"
      }
    }
  }
}
```

To use in another project, copy `.mcp.json` and adjust paths, or add to your project's MCP config:

```json
{
  "mcpServers": {
    "carnot-verify": {
      "command": "/path/to/carnot/.venv/bin/python3",
      "args": ["/path/to/carnot/tools/verify-mcp/server.py"],
      "env": {
        "PYTHONPATH": "/path/to/carnot:/path/to/carnot/python"
      }
    }
  }
}
```

### Available Tools

#### `verify_code`

Run structural tests on a Python function.

```json
{
  "code": "def add(a, b):\n    return a + b\n",
  "func_name": "add",
  "test_cases": [
    {"args": [2, 3], "expected": 5},
    {"args": [0, 0], "expected": 0}
  ]
}
```

Returns:
```json
{
  "energy": 0.0,
  "n_passed": 2,
  "n_failed": 0,
  "n_total": 2,
  "details": [...]
}
```

#### `verify_with_properties`

Run property-based tests with random inputs.

```json
{
  "code": "def add(a, b):\n    return a + b\n",
  "func_name": "add",
  "properties": [
    {
      "name": "commutative",
      "generator": "pair_int",
      "check": "lambda result, a, b: result == b + a"
    },
    {
      "name": "identity",
      "generator": "int",
      "check": "lambda result, x: result == x"
    }
  ],
  "n_samples": 100
}
```

Built-in generators: `int`, `pos_int`, `string`, `list_int`, `pair_int`

Built-in checks: `returns_int`, `returns_float`, `returns_str`, `returns_list`, `returns_bool`, `non_negative`

Custom checks: any lambda expression string like `"lambda result, a, b: result == a + b"`

#### `score_candidates`

Score multiple candidate responses and select the best.

```json
{
  "question": "What is the capital of France?",
  "candidates": ["Paris", "London", "The capital is Paris"],
  "strategy": "length"
}
```

### Using with Claude Code

When working in a project with Carnot's MCP server configured, Claude Code will automatically have access to the verification tools. You can ask it to:

- "Verify this function works correctly"
- "Run property-based tests on my sort implementation"
- "Check if this code handles edge cases"

Claude Code will call the MCP tools and report results inline.

## Using Pre-trained EBMs from HuggingFace

Pre-trained Gibbs EBM models for hallucination detection are available at `huggingface.co/Carnot-EBM`.

### Loading a Pre-trained EBM

```python
from safetensors.numpy import load_file
import jax.numpy as jnp
from carnot.models.gibbs import GibbsConfig, GibbsModel
import jax.random as jrandom

# Load pre-trained weights
weights = load_file("carnot-ebm/per-token-ebm-qwen35-08b-nothink/model.safetensors")

# Reconstruct model
config = GibbsConfig(input_dim=1024, hidden_dims=[256, 64], activation="silu")
ebm = GibbsModel(config, key=jrandom.PRNGKey(0))

# Load weights into model
ebm.layers = [(weights["layer_0_weight"], weights["layer_0_bias"]),
              (weights["layer_1_weight"], weights["layer_1_bias"])]
ebm.output_weight = weights["output_weight"]
ebm.output_bias = weights["output_bias"]

# Score an activation vector
activation = jnp.zeros(1024)  # from your model's hidden states
energy = float(ebm.energy(activation))
# Low energy = likely correct, high energy = likely hallucination
```

### Available Models

| Model | Description | Accuracy | Best For |
|-------|-------------|----------|----------|
| `per-token-ebm-qwen3-06b` | Trained on Qwen3-0.6B activations | 84.5% | Base model detection |
| `per-token-ebm-qwen35-08b-nothink` | Qwen3.5-0.8B, thinking disabled | 75.5% | Instruction-tuned models |
| `per-token-ebm-qwen35-08b-think` | Qwen3.5-0.8B, thinking enabled | 61.3% | When thinking is required |

### Important: Model Compatibility

EBM models are trained on activations from a specific LLM. You must use the matching LLM:

- `per-token-ebm-qwen3-06b` → requires Qwen3-0.6B hidden states
- `per-token-ebm-qwen35-08b-*` → requires Qwen3.5-0.8B hidden states

Activations from different models occupy different representation spaces and cannot be mixed.

### Thinking Mode Recommendation

For hallucination detection with Qwen3.5-0.8B:

```python
# RECOMMENDED: Disable thinking for better detection
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,  # 75.5% detection vs 61.3% with thinking
)
```

Disabling thinking improves detection accuracy by 14.2% because chain-of-thought makes hidden states more uniform (Principle 10). If you need both good answers AND detection, generate with thinking enabled but run a second forward pass with thinking disabled for scoring.
