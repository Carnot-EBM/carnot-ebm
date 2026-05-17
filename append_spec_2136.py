import os

spec_path = "openspec/capabilities/self-learning/spec.md"
with open(spec_path, "a") as f:
    f.write("""
## REQ-LEARN-2136: Substrate Shifting CSL
**Given** the Continuous Self-Learning (CSL) loop
**When** mode concentration is detected and substrate_shift parameters are provided
**Then** the CSL loop MUST translate the underlying KAN LUT grids by applying the energy grid translation parameters
**And** the experiment artifact MUST report `substrate_shifting_ready=True` and `integrated_with_kan_tiers=True`.

### SCENARIO-LEARN-2136: Substrate Shifting Applied
**Given** grid_translation parameters
**When** run_csl_loop is executed with substrate_shift
**Then** the parameters MUST be translated and the artifact saved.
""")
