#!/usr/bin/env python3
"""Run archive v316 / activate v317 — Depth-Over-Breadth II milestone.

Writes results/operational_retro_2026_05_316.json and
results/experiment_3436_archive_v316_activate_v317.json.
"""
import json
import sys
import time
from datetime import timezone, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.archive_v316_activate_v317_3436 import write_artifact


def write_retro() -> Path:
    """Write the operational retrospective JSON for milestone .316."""
    retro = {
        "schema": "carnot.operational_retro.v64",
        "milestone": "2026.05.316",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "retro_date": "2026-05-30",
        "retro_type": "operational_full",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "honest_verdict": "complete: operational_retro_2026_05_316_written",
        "experiments_completed": 7,
        "experiments_blocked_or_error": 0,
        "experiments_missing": 3,
        "flagged_adversarial_count": 1,
        "compute_bound_experiments_count": 1,
        "total_wall_time_minutes": 11,
        "slowest_experiments": [
            {
                "experiment_id": "exp3426",
                "duration_minutes": 10.7,
                "note": (
                    "Live GGUF inference — Qwen3.6-35B-A3B, n=200 GSM8K x k=5 "
                    "samples (642 s). Clean run but harness broken: "
                    "multi-sample answer extraction returned null for all 200x5 "
                    "candidates. P0.1 hypothesis untested."
                ),
            }
        ],
        "flagged_artifacts": [
            {
                "experiment_id": "exp3435",
                "reason": (
                    "Capstone v316 flagged_adversarial CRITICAL: TAUTOLOGY. "
                    "Quarantined. Gate status forwarded from exp3434 (gate synthesis)."
                ),
                "status": "quarantined",
            }
        ],
        "completed_experiments": [
            "exp3425-archive-v315-activate-v316",
            "exp3426-energy-descent-vs-ar-vs-self-consistency-premise-v2",
            "exp3430-fover-g2-cleanroom-validation-v1",
            "exp3431-kv260-terminal-latency-transcript-v2",
            "exp3432-gatemate-apply-verdict-fix-and-flash-v1",
            "exp3433-polarfire-reachability-audit-v2",
            "exp3434-g-gate-status-synthesis-v316",
        ],
        "blocked_or_error_experiments": [],
        "missing_experiments": [
            "exp3427-p0.2-verifier-ensemble-joint-null-space",
            "exp3428-kona-global-optimization-solve-rate-gate",
            "exp3429-ensemble-vs-adaptive-prompt-injection",
        ],
        "milestone_assessment": (
            "PARTIAL_DEPTH: .316 was the second Depth-Over-Breadth milestone. "
            "P0.1 v2 (exp3426) ran authentically (642 s live GGUF) but the "
            "multi-sample answer-extraction harness was broken — every k-sample "
            "condition returned null candidate_preds (0.0) while greedy AR scored "
            "0.75. Energy returned a constant value across all candidates. The P0.1 "
            "hypothesis is UNTESTED, not refuted. P0.2 (exp3427), Kona (exp3428), "
            "and ensemble-vs-injection (exp3429) produced NO artifact for the 2nd "
            "consecutive milestone: root cause is gemini-cli crashing with an "
            "internal JS runtime error (.js:345500:14). G2 cleanroom CI gate FAILED "
            "(exp3430). Capstone flagged_adversarial (TAUTOLOGY, exp3435). "
            "Depth-Over-Breadth Forcing Function remains ACTIVE for .317."
        ),
        "depth_tasks_status": {
            "P0.1_energy_vs_self_consistency": {
                "experiment_id": "exp3426",
                "status": "complete_harness_broken",
                "result": (
                    "642 s live GGUF run. Energy substrate fired but produced "
                    "constant latent energy (-569.818848) across all candidates. "
                    "Multi-sample answer extraction: 0/200 null candidate_preds. "
                    "Greedy AR = 0.75; self_consistency = energy_weighted_vote = 0.0. "
                    "delta_energy_vs_self_consistency = 0.0 (due to broken harness). "
                    "Hypothesis untested. Root cause: temperature=0.8 format "
                    "incompatible with GSM8K answer regex tuned for greedy output."
                ),
                "headline_eligible": False,
            },
            "P0.2_verifier_diversity_alpha_t": {
                "experiment_id": "exp3427",
                "status": "missing_gemini_crash",
                "result": None,
                "headline_eligible": False,
            },
            "kona_solve_rate_gate": {
                "experiment_id": "exp3428",
                "status": "missing_gemini_crash",
                "result": None,
                "headline_eligible": False,
            },
            "ensemble_vs_injection": {
                "experiment_id": "exp3429",
                "status": "missing_gemini_crash",
                "result": None,
                "headline_eligible": False,
            },
            "G2_cleanroom": {
                "experiment_id": "exp3430",
                "status": "ci_gate_failed",
                "result": (
                    "Isolated worktree + fresh venv ran but FoVer recompute did "
                    "not land in published CI bounds. G2 still unmet. External "
                    "reproducer path requires a passing internal CI run first."
                ),
                "headline_eligible": False,
            },
        },
        "top_forward_gap": "finish_existential_block_cleanly",
        "top_forward_gap_detail": (
            "For .317: "
            "(1) P0.1 v3 — fix multi-sample answer extraction (update GSM8K regex "
            "for temperature-sampled format or switch to k greedy re-samples from "
            "different seeds); "
            "(2) P0.2 / Kona / injection — re-route from gemini to claude "
            "(agent_type: claude, requires_claude: true) — gemini-cli is crashing; "
            "(3) G2 root-cause — investigate exp3430 CI gate failure before "
            "attempting an external run; "
            "(4) gemini-cli outage — operator should inspect Node.js runtime "
            "crash at .js:345500:14 before re-enabling gemini tasks."
        ),
        "paper_ready": False,
        "unmet_gates": ["G2"],
        "g1": True,
        "g2": False,
        "g3": True,
        "g4": True,
        "summary": (
            "Milestone .316 ran 7 experiments to completion (not counting flagged "
            "capstone), produced 1 adversarially-flagged artifact (exp3435 capstone, "
            "TAUTOLOGY), and 3 missing (exp3427/3428/3429 — gemini-cli crash). "
            "The sole depth task that ran (P0.1 v2, exp3426) was authentic but "
            "yielded an uninterpretable result due to a harness bug. "
            "Depth-Over-Breadth Forcing Function stays active."
        ),
        "bottlenecks_identified": [
            (
                "gemini-cli crashes with JS internal runtime error "
                "(.js:345500:14) — 3 depth tasks missing for 2nd consecutive "
                "milestone. Node.js upgrade or downgrade needed."
            ),
            (
                "P0.1 v2 answer-extraction harness broken: temperature=0.8 "
                "sampling generates text the GSM8K regex cannot parse. "
                "Energy substrate also returned constant energy — not integrating "
                "with the GGUF tokenizer correctly."
            ),
            (
                "G2 cleanroom CI gate failed (exp3430) — root cause unknown. "
                "The external-reproducer path is blocked until internal CI passes."
            ),
            (
                "Capstone (exp3435) flagged TAUTOLOGY — the aggregation script "
                "duplicated identical values across conceptually-distinct fields."
            ),
        ],
        "improvements_suggested": [
            (
                "Route P0.2 / Kona / injection to agent_type: claude for .317 "
                "(set requires_claude: true in roadmap YAML). Do not re-attempt "
                "gemini until the JS runtime crash is resolved."
            ),
            (
                "P0.1 v3: update the GSM8K answer extraction regex to handle "
                "temperature=0.8 format, OR switch from top-p sampling to k "
                "greedy re-samples with different seeds (avoids tokenizer-format "
                "mismatch entirely)."
            ),
            (
                "G2 CI investigation: run scripts/reproduce_fover_headline.py "
                "locally and inspect which of the 5 seeds falls outside the "
                "published CI bounds. May require a carnot version pin."
            ),
            (
                "Capstone script: add a cross-field uniqueness check to catch "
                "TAUTOLOGY before writing. Flag self-identical floating-point "
                "values across conceptually-distinct metric fields."
            ),
        ],
    }

    out_path = REPO_ROOT / "results" / "operational_retro_2026_05_316.json"
    out_path.write_text(json.dumps(retro, indent=2))
    return out_path


def main() -> None:
    start = time.perf_counter()

    retro_path = write_retro()
    print(f"Written retro: {retro_path}")

    artifact_path = write_artifact()
    artifact = json.loads(artifact_path.read_text())
    artifact["duration_s"] = round(time.perf_counter() - start, 6)
    artifact_path.write_text(json.dumps(artifact, indent=2))

    print(f"Written: {artifact_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"archived_milestone: {artifact['archived_milestone']}")
    print(f"activated_milestone: {artifact['activated_milestone']}")
    print(f"archive_v316_activate_v317_ready: {artifact['archive_v316_activate_v317_ready']}")
    print(f"G1={artifact['g1']} G2={artifact['g2']} G3={artifact['g3']} G4={artifact['g4']}")
    print(f"unmet_gates: {artifact['unmet_gates']}")
    print(f"paper_ready: {artifact['paper_ready']}")
    print(f"depth_forcing_function_can_relax: {artifact['depth_forcing_function_can_relax']}")
    print(f"missing_artifacts: {artifact['missing_artifacts']}")


if __name__ == "__main__":
    main()
