# Carnot CLI Usage

The `carnot` command-line interface exposes Carnot's verification, repair, and EBM scoring pipeline from the shell. It is the recommended entry point for one-off verification runs, scripted batch jobs, and CI integration. For programmatic use from Python, see `docs/api-reference.md`; for in-LLM-conversation tool calls, see `docs/mcp-server.md`.

## Installation

The CLI ships in the `carnot-ebm` PyPI package. The installed binary is named `carnot` (the `-ebm` PyPI suffix is only a project-name disambiguator).

```bash
pip install carnot-ebm
carnot --help
```

If your environment's `pip` does not place the `carnot` binary on `PATH`, the equivalent module form is:

```bash
python -m carnot.cli --help
```

This is the form used inside `.venv` setups and CI runners.

## Top-Level Commands

```
carnot <subcommand> [<args>...]
```

Subcommands:

| Subcommand | Purpose |
|---|---|
| `verify` | Verify a Python function against explicit test cases and optional property-based tests. |
| `verify-code` | Run packaged static + property-based verification on a function. The HumanEval / benchmark entry point. |
| `score` | Score LLM responses using a pre-trained energy-based model from HuggingFace. |
| `score-candidates` | Score candidate outputs with the calibrated second-pair detector used by the MCP surface. |
| `memory` | Export, import, or diff Carnot's persistent verification-memory archives. |

Every subcommand accepts `--help` for the full flag list.

## `carnot verify`

Verify a Python function against caller-supplied input/expected-output pairs, optionally augmented with property-based testing.

```
carnot verify <file> --func <name> --test "INPUT:EXPECTED" [--test "..."]
              [--type <return-type>]
              [--properties]
              [--prop-samples <N>]
              [--prop-seed <SEED>]
```

- `<file>` (positional): path to the Python source file.
- `--func` (required): function symbol to verify.
- `--test "(args):expected"` (repeatable): one structural test. Multiple `--test` flags accumulate.
- `--type` (default `int`): expected return type. Used by the verifier's type-coercion layer.
- `--properties`: also run random property-based testing on top of the structural tests.
- `--prop-samples` (default `100`): number of PBT samples when `--properties` is set.
- `--prop-seed` (default `42`): PBT seed for reproducibility.

Example:

```bash
carnot verify examples/math_funcs.py --func gcd \
  --test "(12,8):4" --test "(100,75):25" \
  --properties --prop-samples 200
```

Exit code is `0` on PASS, non-zero on FAIL. The structured verdict is written to stdout as JSON.

## `carnot verify-code`

The packaged-static + property-based verification entry point used for HumanEval-style benchmarks. This is the command behind the `+3.0pp` Gemma 4 E4B HumanEval result reported in `docs/technical-report.md` (Exp 226).

```
carnot verify-code <file> --func <name>
                   [--prompt-file <path>]
                   [--tests-file <path>]
                   [--pbt]
```

- `<file>` (positional): Python source file.
- `--func` (required): function symbol.
- `--prompt-file` (optional): HumanEval-style prompt or signature context. The verifier uses this to derive property predicates when explicit tests are not available.
- `--tests-file` (optional): official tests / `check()` harness text. When provided, the verifier runs the official harness in addition to its own checks.
- `--pbt`: enable Hypothesis-backed property-based testing.

Example:

```bash
carnot verify-code candidates/humaneval_42.py --func solve \
  --prompt-file prompts/humaneval_42.txt \
  --tests-file harness/humaneval_42_check.py \
  --pbt
```

Exit code `0` on PASS, non-zero on FAIL. Output is a structured JSON verdict including the structural-test result, the PBT result, and (if applicable) the official-harness result.

## `carnot score`

Score one or more LLM responses against a trained energy-based model. Returns a ranking from lowest to highest energy. The lowest-energy response is the EBM's best candidate.

```
carnot score --question <text> --response <text> [--response <text> ...]
             [--model <hf-model-id>]
             [--device <cpu|cuda|rocm>]
```

- `--question` (required): the question or prompt the responses are answering.
- `--response` (required, repeatable): one candidate response. Multiple `--response` flags accumulate.
- `--model` (default `per-token-ebm-qwen35-08b-nothink`): HuggingFace model ID under the `Carnot-EBM` organization.
- `--device` (default auto-detected): inference backend.

Example:

```bash
carnot score \
  --question "What is the capital of Japan?" \
  --response "Tokyo." \
  --response "Kyoto." \
  --response "Osaka."
```

Output is a JSON-formatted ranking with per-response energy values.

The first invocation downloads the EBM weights from HuggingFace and caches them locally. Subsequent calls reuse the cache. Set `CARNOT_TRUST_REMOTE_CODE=0` (default) to refuse models that ship custom Python; set `=1` only after auditing the model card.

## `carnot score-candidates`

Score one or more candidate outputs with the shipped calibrated second-pair detector. This is the CLI equivalent of the MCP `score_candidates` tool.

```
carnot score-candidates --domain <domain>
                        --candidates-json '[{"candidate_id":"a","domain":"math","text":"...","confidence":0.2,"ensemble_energy":0.8}]'
```

The output is JSON containing `scores`, where each row includes `calibrated_error_score`, `ensemble_energy`, `confidence_error`, `domain`, and `operating_point`.

## `carnot memory`

Export, import, or diff Carnot's verification-memory archive. The memory archive stores per-task verification outcomes so subsequent runs can short-circuit redundant work. Useful for CI caching and for sharing pre-warmed memory between hosts.

### `carnot memory export <output-path>`

Write the current memory archive to `<output-path>` as a portable tarball.

### `carnot memory import <input-path>`

Read a memory archive from `<input-path>` and merge it with the current archive.

### `carnot memory diff <a-path> <b-path>`

Compute a structured diff between two memory archives. Used for debugging memory drift across hosts.

## Environment Variables

| Variable | Effect |
|---|---|
| `CARNOT_USE_SANDBOX` | Run code-execution verifiers inside a gVisor-style sandbox (production default). Unset for in-process exec (development default). |
| `CARNOT_TRUST_REMOTE_CODE` | Allow `score` to load HuggingFace models that ship custom Python. Default `0` (refuse). |
| `CARNOT_MCP_LOG_LEVEL` | Adjust stderr log verbosity. Same levels as Python `logging`. |
| `CARNOT_HF_CACHE_DIR` | Override the HuggingFace download cache directory. |

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success (PASS verdict on `verify` / `verify-code`; ranking produced for `score`). |
| `1` | Verification FAIL (the function under test does not pass the verifier). |
| `2` | Invalid arguments. |
| `3` | Pipeline error (verifier crashed; not a function failure). |
| `4` | Missing dependency or precondition (model not cached, network unreachable, etc.). |

`verify` and `verify-code` use exit code `1` exclusively for verification failures — pipeline errors return `3` so CI scripts can distinguish "code under test is broken" from "Carnot itself is broken."

## CI Integration

The CLI is designed for non-interactive use: structured JSON on stdout, status on exit code, all log noise on stderr. A minimal GitHub Actions step:

```yaml
- name: Verify generated code
  run: |
    pip install carnot-ebm
    carnot verify-code generated/solution.py --func solve --pbt
```

For HumanEval-style cohort runs see `scripts/experiment_226_pbt_humaneval_full.py` in the repository, which drives the CLI across 164 problems and aggregates the verdicts.

## Versioning

Same semver convention as the MCP server (see `docs/mcp-server.md`). Subcommand signatures are stable within a minor release. The current series is `0.1.x` (Phase 1 ship gate). Track release notes at `https://github.com/Carnot-EBM/carnot-ebm/releases`.
