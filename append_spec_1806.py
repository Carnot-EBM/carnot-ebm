import os

spec_path = "openspec/capabilities/phase3-kona/spec.md"

with open(spec_path, "a") as f:
    f.write("\n### REQ-KONA-040: Non-Autoregressive Reasoning Model\n\n")
    f.write("The repository shall provide an Energy-Based Reasoning Model (EBRM) in `python/carnot/models/kona_ebrm.py` that maps a simple logic puzzle into a continuous latent space and applies an energy function to detect inconsistencies, editing the trace via gradient descent.\n\n")
    f.write("### SCENARIO-KONA-040: Exp 1806 Writes Kona EBRM Artifact\n\n")
    f.write("**Given** a logic puzzle mapped to a continuous latent space\n")
    f.write("**When** Exp 1806 applies gradient descent to refine the entire reasoning trace simultaneously\n")
    f.write("**Then** it writes `results/experiment_1806_kona_ebrm.json` with all REQ-KONA-040 required fields.\n\n")
    f.write("**Spec traces:** REQ-KONA-040\n")

print("Appended to spec.")
