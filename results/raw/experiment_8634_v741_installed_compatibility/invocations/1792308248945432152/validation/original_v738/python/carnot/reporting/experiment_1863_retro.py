import json

def generate_retro(output_path, vl_proxy_pass_rate, s2kan_pass_rate):
    """
    Generates the milestone retrospective for 145 based on REQ-REPORT-0863.
    """
    result = {
        "schema": "carnot.milestone_research_retro.v1",
        "milestone": "2026.05.145",
        "vl_proxy_pass_rate": vl_proxy_pass_rate,
        "s2kan_pass_rate": s2kan_pass_rate,
        "honest_verdict": "milestone_145_retro_complete"
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
