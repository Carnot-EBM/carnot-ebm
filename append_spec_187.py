import os

with open("openspec/capabilities/autoresearch/spec.md", "a") as f:
    f.write("\n### REQ-RETRO-187: Milestone 2026.05.187 Retrospective\n")
    f.write("The system SHALL generate a retrospective JSON artifact for milestone 2026.05.187.\n\n")
    f.write("### SCENARIO-RETRO-187: Validate 187 Retro\n")
    f.write("GIVEN the 187 retrospective is generated\n")
    f.write("WHEN the artifact is parsed\n")
    f.write("THEN it contains the required honest_verdict and schema fields.\n")
