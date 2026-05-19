# Carnot CLI Guide

## Installation
The Carnot CLI is installed alongside the main package. Install it via pip:

```bash
pip install -e .
```
This registers the `carnot` console command.

## Commands

### `carnot verify`
Verify a Python function against test cases and optional property-based tests using EBM energy constraints.
- **Usage**: `carnot verify <file> --func <func_name> --test <INPUT:EXPECTED> [--type <type>] [--properties] [--prop-samples <N>] [--prop-seed <N>]`
- **Example**: `carnot verify examples/math_funcs.py --func gcd --test "(12,8):4"`

### `carnot verify-code`
Verify a Python function with packaged static checks and optional Hypothesis-backed Property-Based Testing (PBT).
- **Usage**: `carnot verify-code <file> --func <func_name> [--prompt-file <file>] [--tests-file <file>] [--pbt]`
- **Example**: `carnot verify-code examples/math_funcs.py --func gcd --pbt`

### `carnot score`
Score activation vectors using a pre-trained EBM from HuggingFace.
- **Usage**: `carnot score [--model <model_id>] [--activations-file <file>] [--list-models]`
- **Example**: `carnot score --activations-file acts.safetensors`

### `carnot memory`
Import, export, and diff portable SessionMemory JSON packs.
- **`carnot memory export`**: Export local SessionMemory state.
  - Usage: `carnot memory export --storage-dir <dir> --model-id <id> -o <file> [--source <src>] [--redact-provenance]`
- **`carnot memory import`**: Import a SessionMemory pack.
  - Usage: `carnot memory import <pack> --storage-dir <dir> [--model-id <id>] [--merge | --replace] [--dry-run]`
- **`carnot memory diff`**: Compare two SessionMemory packs.
  - Usage: `carnot memory diff <left> <right>`

## Examples

### 1. Basic Function Verification
```bash
carnot verify math.py --func add --test "(1,2):3" --test "(0,0):0"
```

### 2. Verification with Property-Based Tests
```bash
carnot verify math.py --func add --test "(1,2):3" --properties --prop-samples 50
```

### 3. Packaged Verification with Hypothesis (PBT)
```bash
carnot verify-code my_sort.py --func sort_list --pbt
```

### 4. Scoring Activations
```bash
# List available models
carnot score --list-models

# Score using default model
carnot score --activations-file my_activations.safetensors
```

### 5. Exporting and Importing Session Memory
```bash
# Export memory for a specific model
carnot memory export --storage-dir .carnot_sessions --model-id qwen -o pack.json

# Import memory
carnot memory import pack.json --storage-dir .carnot_sessions_new --merge
```

## Configuration
No extra configuration files are required. The CLI relies on command-line arguments. It operates on Python files and standard data formats (e.g., `safetensors` for scoring, JSON for memory packs).
