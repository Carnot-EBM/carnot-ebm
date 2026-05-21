import os

with open("openspec/capabilities/pipeline/spec.md", "a") as f:
    f.write("\n\n### REQ-PIPELINE-1826: Fail-Fast Doomed Reruns\n")
    f.write("The pipeline API MUST provide a fail-fast check for doomed reruns at activation time.\n")
    f.write("It MUST write a terminal artifact with `status=\"blocked\"` and `honest_verdict=\"blocked_doomed_rerun\"` when a task is doomed.\n")
    f.write("\n### SCENARIO-PIPELINE-1826: Fail-Fast Artifact Generation\n")
    f.write("**Given** a doomed task definition\n")
    f.write("**When** the pipeline fail-fast check is invoked\n")
    f.write("**Then** it writes a blocked artifact and returns True.\n")
