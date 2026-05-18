# Carnot MCP Server

Carnot ships an MCP (Model Context Protocol) server that exposes its code-verification and energy-based-model scoring tools directly to LLM clients such as Claude Code, Claude Desktop, and Cursor. The server is a single Python module — `carnot.mcp` — and runs locally on the integrator's machine. No network calls leave the host unless an exposed tool explicitly fetches an external model from HuggingFace.

This page documents how to install the server, register it with a client, and call each of its tools. For a higher-level architectural overview see `docs/concepts.md`; for the underlying Python API see `docs/api-reference.md`.

## Installation

The MCP server ships in the `carnot-ebm` PyPI package. The published import name is `carnot` (the `-ebm` suffix only disambiguates the project on PyPI).

```bash
pip install carnot-ebm
```

Verify the server module loads:

```bash
python -m carnot.mcp --help
```

The first invocation will print the registered tool list. No further setup is required for the structural verification tools; PBT-mode tools require `hypothesis` (already pulled in by the package) and the energy-based-model scoring tools require a one-time HuggingFace download on first call.

## Client Configuration

### Claude Code

Append a `carnot` entry to `~/.claude/settings.json` (or whichever `settings.json` file your Claude Code installation uses) under `mcpServers`:

```json
{
  "mcpServers": {
    "carnot": {
      "command": "python",
      "args": ["-m", "carnot.mcp"]
    }
  }
}
```

If `python` does not resolve to a Python interpreter with `carnot-ebm` installed (for example, you are using a virtual environment), use the absolute path to that interpreter's `python` binary instead.

Restart Claude Code. The Carnot tools will appear in the tool palette and Claude Code can invoke them like any other MCP tool.

### Claude Desktop (macOS / Windows)

Add the same `mcpServers` block to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%/Claude/claude_desktop_config.json` (Windows). Restart Claude Desktop.

### Cursor and Other MCP Clients

Any client that speaks MCP over stdio can launch `python -m carnot.mcp` as the server command. Consult the client's MCP-server-config documentation for the exact registration syntax.

## Exposed Tools

The server exposes nine tools. Each tool has a JSON-typed input schema and returns a structured JSON object. Input validation rejects untrusted-by-default code execution: the server does not eval arbitrary strings; it dispatches to the verifier modules in `carnot.pipeline` which guard the actual execution.

### `verify_code(code, func_name, test_cases)`

Run structural verification against a list of explicit input/expected-output pairs. The simplest and fastest tool — useful when the integrator already has reference outputs.

- `code` (str): the Python source defining the function under test.
- `func_name` (str): the function symbol to call.
- `test_cases` (list of dict): each dict has `"input"` (list-of-args) and `"expected"`.

Returns: `{ "verdict": "PASS" | "FAIL", "failures": [...], "details": {...} }`.

### `verify_code_with_pbt(code, func_name, test_cases, n_pbt_samples=100, pbt_seed=42)`

Combines structural tests with Hypothesis-backed property-based tests. Properties are auto-derived from the function's type annotations and the failure surface observed during PBT is reported in the result.

This is the load-bearing tool for HumanEval-style verification workflows; it underlies the `+3.0 pp` Gemma 4 E4B benchmark result reported in `docs/technical-report.md` (Exp 226).

### `verify_with_properties(code, func_name, properties)`

Run pure property-based verification with caller-supplied property predicates. Use this when the integrator has explicit invariants (idempotence, monotonicity, no-exception, etc.) rather than reference outputs.

### `verify_llm_output(question, response, domain=None)`

Verify a natural-language LLM output against the relevant domain verifier. Supported domains include arithmetic, code, factual-recall, and constraint-graph. Pass `domain=None` to auto-detect.

Returns the structured verifier verdict plus the per-constraint energy contributions.

### `verify_and_repair(question, response, max_repair_attempts=3, domain=None)`

The verify-then-repair pipeline. If the verifier rejects the response, the tool emits structured repair feedback that an LLM can consume to produce a corrected output. Up to `max_repair_attempts` rounds. This is the canonical Phase 1 entry point: a single call hides the verify / repair / re-verify loop.

### `verify_stream(question, partial_response, domain=None)`

The streaming-friendly counterpart to `verify_llm_output`. Designed for inline-during-generation verification: pass partial outputs as they arrive; the tool returns a confidence-graded verdict plus a "stop now" recommendation when the partial output is already known to be unrecoverable.

### `list_domains()`

Return the registered verifier domains and their tool capabilities. Use this for client-side discovery rather than hard-coding the domain list.

### `health_check()`

Return server liveness, registered tool count, and the resolved `carnot` package version. The intended entry point for client-side liveness probes (the MCP `pinging` extension uses this).

### `score_agent_outputs(question, responses)`

Score a list of candidate LLM responses with the trained energy-based model. The lowest-energy response is the model's best guess; the returned object includes a full ranking plus per-response energy values for downstream calibration. The EBM weights are loaded from the HuggingFace mirror `Carnot-EBM/per-token-ebm-qwen35-08b-nothink` on first call; subsequent calls reuse the cached weights.

## Error Handling

The server returns structured error responses rather than raising exceptions across the MCP boundary. An error response has the shape:

```json
{
  "error": {
    "code": "INVALID_INPUT" | "EXECUTION_BLOCKED" | "VERIFIER_INTERNAL_ERROR" | ...,
    "detail": "human-readable explanation",
    "tool": "verify_code_with_pbt"
  }
}
```

Integrators should treat any response with an `error` key as a non-result. The `code` field is stable across releases; the `detail` field is best-effort and may change.

## Security Notes

- **Untrusted code execution**: the verifier modules execute the supplied `code` in a constrained subprocess. The MCP server does not add additional sandboxing; consult `python/carnot/pipeline/code_verification.py` for the exact isolation contract. For production hardening, run the server inside a gVisor / Firejail / container with `CARNOT_USE_SANDBOX=1`.
- **HuggingFace downloads**: only `score_agent_outputs` fetches a remote model, and only on first call. Set `CARNOT_TRUST_REMOTE_CODE=0` (default) to refuse models that ship custom Python; set `=1` only after auditing.
- **Logs**: the server writes a structured log to stderr by default. Set `CARNOT_MCP_LOG_LEVEL=ERROR` to silence routine telemetry.

## Versioning

The MCP tool schemas follow semver:

- **Patch** releases (`0.1.0b1` -> `0.1.0b2`): tool signatures and response shapes are stable.
- **Minor** releases (`0.1` -> `0.2`): tools may gain optional fields; existing fields and signatures remain stable.
- **Major** releases (`0` -> `1`): tool signatures may change; deprecations are announced one minor cycle ahead.

The current series is `0.1.x` (Phase 1 ship gate). Track release notes at `https://github.com/Carnot-EBM/carnot-ebm/releases`.
