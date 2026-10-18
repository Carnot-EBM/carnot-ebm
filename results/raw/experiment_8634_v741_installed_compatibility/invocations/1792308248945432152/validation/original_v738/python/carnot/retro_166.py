import json
import os
from glob import glob

def generate_retro(output_path: str) -> dict:
    tasks_summary = [
        {
            "experiment": "2105_gatemate_smoke",
            "hypothesis": "Synthesis determinism check on Cologne Chip GateMate.",
            "gate_threshold": "Synthesis and PNR completed successfully.",
            "empirical_result": "Blocked: toolchain missing (jtag_flashed, lut_utilization null).",
            "surprising_finding": "Toolchain dependencies for openFPGALoader are missing."
        },
        {
            "experiment": "2106_thrml_parity_v3",
            "hypothesis": "Carnot compute parity with thrml reference for Curie-Weiss model.",
            "gate_threshold": "High KS p-value and low KL divergence.",
            "empirical_result": "Passed: ks_p_value=0.978, kl_divergence=0.0106.",
            "surprising_finding": "Tight parity achieved with very low KL divergence."
        },
        {
            "experiment": "2107_phase4_v3",
            "hypothesis": "Active inference verifier ensemble k=6 vs k=1 yields delta_alpha > 0.05.",
            "gate_threshold": "delta_alpha > 0.05.",
            "empirical_result": "Passed: delta_alpha = 0.150366.",
            "surprising_finding": "None explicitly, hypothesis cleanly validated."
        },
        {
            "experiment": "2108_four_delta_bound",
            "hypothesis": "Empirical runs obey the 4-stage Markov chain bound from Dantas et al.",
            "gate_threshold": "mean_iterations <= predicted_bound.",
            "empirical_result": "Passed: mean_iterations=1.16, predicted_bound=4.67.",
            "surprising_finding": "Potential structural mismatch between verify-repair pipeline and Dantas et al.'s model."
        }
    ]

    artifact = {
        "schema": "carnot.milestone_research_retro.v1",
        "milestone": "2026.05.166",
        "tasks_summary": tasks_summary,
        "gates_passed_count": 3,
        "gates_failed_count": 1,
        "actual_agent_backend_distribution": {
            "codex": 0,
            "gemini": 3,
            "claude": 0,
            "mock_simulator": 1
        },
        "meta_reflection": "Agent-routing structural issue: multiple tasks are being routed to gemini or mock_simulator instead of codex, which is a recurring theme.",
        "paper_v6_carryforward_items": [
            "Result interpretation requires context about whether Carnot's verify-repair pipeline maps cleanly onto Dantas et al.'s 4-stage Markov chain. If structural mismatch is identified, that's a paper-v6 disclosure item."
        ],
        "adversarial_verify_flag_count": 0,
        "honest_verdict": "complete: retro generated for milestone 166, all tasks summarized, agent backend routing issue escalated."
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(artifact, fh, indent=2)

    return artifact

if __name__ == "__main__":  # pragma: no cover
    generate_retro("results/experiment_2109_retro.json")
