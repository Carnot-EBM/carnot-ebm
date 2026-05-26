# Carnot

**Energy-based verification for LLM output.** Catch the mistakes your LLM
confidently makes up.

Large language models sometimes produce answers that sound right but
aren't. Carnot is an open-source framework that checks whether an answer
is internally consistent -- by extracting constraints (arithmetic, code,
logic, schemas) and verifying them against energy-based models -- and
suggests a fix when it isn't.

- **License:** Apache 2.0
- **PyPI:** `pip install carnot-ebm`
- **HuggingFace:** [huggingface.co/Carnot-EBM](https://huggingface.co/Carnot-EBM)
- **Website:** [carnot-ebm.org](https://carnot-ebm.org)

## Quick Start

```bash
pip install carnot-ebm
```

```python
from carnot.pipeline import VerifyRepairPipeline

pipeline = VerifyRepairPipeline()
result = pipeline.verify("What is 15 + 27?", "15 + 27 = 42")
print(result.verified)    # True
```

Catching a wrong answer:

```python
result = pipeline.verify("What is 15 + 27?", "15 + 27 = 43")
print(result.verified)    # False
print(result.violations)  # [ConstraintResult: "15 + 27 = 43 (correct: 42)"]
```

> Python 3.11 or newer is required. JAX runs on CPU by default; GPU support
> is available via `pip install carnot-ebm[cuda]`.

## What's in the Box

- **Python library** -- `carnot.pipeline.VerifyRepairPipeline` plus
  constraint extractors for arithmetic, code, logic, and JSON schemas.
- **CLI** -- `carnot verify`, `carnot score`, `carnot verify-code` for
  shell-level usage.
- **MCP server** -- Claude Code and other MCP-compatible agents can verify
  their own output mid-conversation.
- **Rust core** -- high-performance EBM training and sampling crates under
  `crates/carnot-*/`.
- **Trained verifiers** -- model weights mirrored on HuggingFace at
  [Carnot-EBM/](https://huggingface.co/Carnot-EBM).

## Documentation

| Where to start                                            | What it covers                                                          |
|-----------------------------------------------------------|-------------------------------------------------------------------------|
| [**30-minute tutorial**](docs/tutorial.md)                | Build a hallucination-resistant math helper end-to-end                  |
| [Getting started](docs/getting-started.md)                | Install, Quick Start, CLI, MCP server setup, macOS / `uv` notes         |
| [Concepts](docs/concepts.md)                              | What "energy-based verification" actually means, in plain English       |
| [CLI usage](docs/cli-usage.md)                            | Every `carnot` subcommand with examples                                 |
| [MCP server](docs/mcp-server.md)                          | Wire Carnot into Claude Code or another MCP-compatible agent            |
| [API reference](docs/api-reference.md)                    | Public classes and methods                                              |
| [Examples](examples/)                                     | Runnable integration examples (FastAPI route, batch verifier, ...)      |
| [Technical report](docs/technical-report.md)              | The research behind Carnot                                              |

## Technology

- **Languages:** Python 3.11+ (JAX / Flax / Optax) and Rust (ndarray,
  rayon).
- **Bridge:** PyO3 0.24+; models cross the language boundary as
  safetensors.
- **Hardware:** CPU by default. CUDA via `pip install carnot-ebm[cuda]`.
  ROCm (Strix Point gfx1150 and similar) and NPU paths are under active
  development; see the technical report for current status.

## Project Status

Phase 1 (the verify-and-repair pipeline shipped as a usable software
product) is complete: PyPI package, HuggingFace mirror, CLI, MCP server,
documented tutorial, independent reproducer. Phases 2 (hardware
acceleration) and 3 (open-source foundation model) are ongoing research
tracks; see [`_bmad/prd.md`](_bmad/prd.md) for the long-form vision.

## Contributing

Carnot is Apache-2.0 and welcomes contributions. Start with the
[tutorial](docs/tutorial.md) to get oriented, then have a look at
[`ops/known-issues.md`](ops/known-issues.md) for current priorities and
[`CONTRIBUTING.md`](CONTRIBUTING.md) (when present) for the development
workflow.

## Citation

```bibtex
@software{carnot2026,
  author = {The Carnot Authors (ian@blenke.com)},
  title  = {Carnot: Energy-Based Verification},
  year   = {2026},
  url    = {https://github.com/Carnot-EBM/carnot-ebm}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
