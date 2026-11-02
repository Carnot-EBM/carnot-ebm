import json
import os

def generate_retro(input_paths, output_path):
    """
    Generates the milestone retrospective for 144 based on REQ-REPORT-1853.
    """
    gates_passed = 0
    gates_failed = 0
    tasks_summary = []
    paper_carryforward = []

    for path in input_paths:
        if not os.path.exists(path):
            tasks_summary.append({
                "file": path,
                "status": "missing",
                "hypothesis_resolution": "inconclusive"
            })
            continue

        with open(path, "r") as f:
            data = json.load(f)
        
        verdict = data.get("honest_verdict", "unknown")
        gate_passed = data.get("acceptance_gate_passed", None)
        
        resolution = "inconclusive"
        if gate_passed is True:
            gates_passed += 1
            resolution = "passed gate"
            paper_carryforward.append(f"Incorporate positive result from {os.path.basename(path)}: {verdict}")
        elif gate_passed is False:
            gates_failed += 1
            resolution = "failed gate"
            paper_carryforward.append(f"Note limitations/failed gate from {os.path.basename(path)}: {verdict}")

        tasks_summary.append({
            "file": os.path.basename(path),
            "verdict": verdict,
            "hypothesis_resolution": resolution
        })

    if not paper_carryforward:
        paper_carryforward.append("No carryforward items identified.")

    result = {
        "schema": "carnot.milestone_research_retro.v1",
        "milestone": "2026.05.144",
        "tasks_summary": tasks_summary,
        "gates_passed_count": gates_passed,
        "gates_failed_count": gates_failed,
        "paper_v6_carryforward_items": paper_carryforward,
        "honest_verdict": f"complete: milestone_144_retro_{gates_passed}_of_{len(input_paths)}_gates_passed"
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
