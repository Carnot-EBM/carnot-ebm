# Carnot Integrator Guide

This guide explains how to install and integrate Carnot's code verification pipeline into your environment. You can use Carnot as a standalone Command Line Interface (CLI) application or integrate it with Claude Desktop using the Model Context Protocol (MCP).

## Section 1: Quickstart

You can clone the repository, install the dependencies, and run your first verification command in five simple steps.

```bash
git clone https://github.com/Carnot-EBM/carnot-ebm.git
cd carnot
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4"
```

## Section 2: MCP Installation and Claude Desktop Configuration

Carnot provides an MCP server that exposes tools like `verify_code` and `verify_with_properties` directly to Claude. To configure Claude Desktop to use Carnot, you must add the following configuration to your Claude Desktop configuration file (typically located at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS).

This configuration is also provided in the `.mcp.json.example` file in the repository root.

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

Note: Ensure that the command paths are absolute if you are not running Claude Desktop from the repository root. The example above assumes the current working directory is the Carnot repository root.

## Section 3: CLI Examples

The Carnot CLI provides several subcommands for verifying Python code against test cases and properties.

### Carnot Verify

The standard `carnot verify` command evaluates a Python function against a provided set of input and output pairs.

```bash
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4"
```

### Carnot Verify-Code with Property-Based Testing

The `carnot verify-code` command allows for packaged static checks. You can enable Hypothesis-backed property-based testing by passing the `--pbt` flag.

```bash
carnot verify-code examples/math_funcs.py --func gcd --pbt
```

### Carnot Verify with Properties

You can also run random property-based testing sampling alongside your standard structural tests by passing the `--properties` flag to the `carnot verify` command.

```bash
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4" --properties --prop-samples 100
```

## Section 4: End-to-End Example

This section demonstrates an end-to-end workflow: inputting code, receiving verifier output, and interpreting the repair feedback.

**Input Code (`examples/math_funcs.py`):**
```python
def gcd(a, b):
    # Buggy implementation
    if b == 0:
        return a
    return gcd(b, a % b) + 1
```

**Verifier Output (Command):**
```bash
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4"
```

**Verifier Output (Result):**
The verifier will indicate a failure because the output does not match the expected result.
```
--- Structural Tests ---
  [FAIL] gcd(12, 8) == 4 (got 5)
...
  Verdict:      FAIL
  Violations:   structural_tests
```

**Repair Output (Feedback):**
The feedback clearly indicates that for the input `(12, 8)`, the expected output is `4`, but the actual output returned by the function was `5`. The developer can observe this and correct the `+ 1` off-by-one error in the return statement to produce a compliant implementation.

## Section 5: Reproducing Paper-v6 Results

To reproduce the interim empirical results related to the Paper-v6 publication, specifically the Fast-Slow Variant, please consult the `.194` empirical results logs. These logs confirm the replication of the Fast-Slow Variant and are maintained as the current benchmark standard for the project.