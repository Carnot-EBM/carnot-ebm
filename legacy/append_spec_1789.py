import sys

spec_addition = """
### REQ-BENCH-1789: NRGPT Math Benchmark

Carnot MUST provide a benchmark script to validate the Aleph-style orchestrator and NRGPT exploration on a mathematical benchmark.
The implementation MUST use `unsloth/Qwen3.6-35B-A3B-GGUF` in MODEL_SPECS.
Results MUST be written to `results/experiment_1789_math_benchmark.json`.

**Acceptance criteria:**
- Script `scripts/experiment_1789_math_benchmark.py` exists.
- Records machine-checkable proof success rate.
- Artifact is written to `results/experiment_1789_math_benchmark.json`.

### SCENARIO-BENCH-1789: NRGPT Math Benchmark Execution

**Given** the NRGPTExplorer
**When** the benchmark script runs using `unsloth/Qwen3.6-35B-A3B-GGUF`
**Then** it performs the execution evaluation
**And** produces a valid `results/experiment_1789_math_benchmark.json` artifact containing machine-checkable proof success rate.

**Spec traces:** REQ-BENCH-1789
"""

with open("openspec/capabilities/benchmarks/spec.md", "a") as f:
    f.write(spec_addition)
