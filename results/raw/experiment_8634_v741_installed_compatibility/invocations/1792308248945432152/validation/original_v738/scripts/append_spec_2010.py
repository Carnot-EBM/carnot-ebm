import os

def append_to_file(path, text):
    with open(path, "a") as f:
        f.write(text)

append_to_file(
    "openspec/capabilities/research-reporting/spec.md",
    "\n\n### REQ-REPORT-2010: Consolidated Phase 1 Audit\n"
    "The pipeline SHALL run a consolidated audit for milestones .198 to .201, generate a Phase 1 dashboard, and write results/experiment_2010_consolidated_audit.json.\n"
)

try:
    with open("ops/status.md", "r") as f:
        content = f.read()

    if "ops/phase-1-dashboard.md" not in content and "## What's Working" in content:
        with open("ops/status.md", "w") as f:
            f.write(content.replace("## What's Working", "## What's Working\n- [Phase 1 Ship-Track Dashboard](phase-1-dashboard.md)"))
except FileNotFoundError:
    pass
