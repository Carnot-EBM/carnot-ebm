# Carnot

Carnot is an Energy-Based Model framework for **verifying and repairing LLM outputs**. The repository now distinguishes validated live artifacts from simulated or otherwise unverified ones instead of presenting them as equivalent evidence.

**Evidence status:** The provenance audit found 1 validated live artifacts, 1 simulated artifacts, and 1 artifacts missing explicit live provenance.

**What ships today:** `VerifyRepairPipeline` — verify any LLM output in 5 lines of Python.

**What we learned:** Activation-based EBMs detect confidence, not correctness (50% practical). The 14 principles from our systematic negative results save other researchers months of dead ends. Structural constraint verification is what actually works. See the [technical report](docs/technical-report.md) for full results.

## Key Results (160+ experiments, 16 models, 11 milestones)

### Headline results with provenance

The table keeps the strongest historical numbers in the research record while making the evidence status explicit.

| Claim | Result | Provenance | Caveat |
|-------|--------|------------|--------|

### What works on test sets but fails in practice
