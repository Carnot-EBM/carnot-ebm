import sys

with open("openspec/capabilities/training-inference/spec.md", "a") as f:
    f.write("\n### REQ-SAMPLE-038: Denoising Thermodynamic Model (DTM) Simulation\n\n")
    f.write("**Description:** The system SHALL provide a simulation of Denoising Thermodynamic Models (DTM) using `thrml`.\n\n")
    f.write("#### SCENARIO-SAMPLE-038-1: DTM Diffusion-like Sampling (Exp 1806)\n\n")
    f.write("**Given** an environment with `thrml`\n")
    f.write("**When** running the DTM simulation script\n")
    f.write("**Then** it SHALL output distribution convergence and `thrml_import_ready`.\n")
