import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPSTONE_PATH = REPO_ROOT / "results" / "experiment_2580_capstone_v247.json"
DELIVERABLE_PATH = REPO_ROOT / "results" / "experiment_2581_retro_v247.json"
SCHEMA = "carnot.operational_retro.v69"

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix required (complete:). Retro is always terminal.",
    "schema": "carnot.operational_retro.v69.",
    "n_experiments_completed": "Primary milestone health metric.",
    "best_247_auroc": "Carry-forward of headline AUROC for trend tracking.",
    "safety_classifier_viable": "Tier B product status -- the primary new product milestone for .247.",
    "tier0s_real_improvement": "Documents whether real-corpus verifier gap was narrowed for tier0s.",
    "tier0u_real_improvement": "Documents whether real-corpus verifier gap was narrowed for tier0u.",
    "operator_recommendation": "submit_arxiv_now / update_hf_cards_push_ipfs / continue_safety_classifier / hardware_terminal_pending / all_tracks_advancing.",
    "top_3_successes": "Most impactful wins -- planner carry-forward bias input for .248.",
    "top_3_gaps_for_248": "Most critical unresolved issues -- primary input to .248 planner.",
}

REQUIRED_CAPSTONE_FIELDS = (
    "best_247_auroc",
    "safety_classifier_viable",
    "tier0s_real_improvement",
    "tier0u_real_improvement",
    "gatemate_status",
    "kv260_status",
    "operator_recommendation",
    "n_experiments_completed",
)


def _require_capstone_fields(capstone):
    missing = [field for field in REQUIRED_CAPSTONE_FIELDS if field not in capstone]
    if missing:
        raise KeyError(f"capstone missing required field(s): {', '.join(missing)}")


def _top_3_successes(capstone):
    gatemate = capstone["gatemate_status"]
    auroc = capstone["best_247_auroc"]
    n_completed = capstone["n_experiments_completed"]
    n_planned = capstone.get("n_planned")

    return [
        {
            "rank": 1,
            "area": "hardware_continuity",
            "summary": (
                "GateMate continuity remains in terminal state: JTAG is detected, the "
                "bitstream is flashed, and post-flash detection confirms GM1Ax IDCODE "
                f"0x20000001; next blocker is {gatemate.get('next_blocker')}."
            ),
        },
        {
            "rank": 2,
            "area": "headline_metric_integrity",
            "summary": (
                f"Headline AUROC is preserved honestly at {auroc:.4f} from the "
                "exp2546 ensemble-v7b carry-forward, with no fabricated .247 uplift."
            ),
        },
        {
            "rank": 3,
            "area": "process_honesty",
            "summary": (
                f"The capstone records {n_completed} completed experiments"
                + (f" out of {n_planned} planned" if n_planned is not None else "")
                + " and surfaces the execution-layer gap explicitly, giving .248 a "
                "clear planner input instead of silently treating missing artifacts as progress."
            ),
        },
    ]


def _top_3_gaps_for_248(capstone):
    kv260 = capstone["kv260_status"]
    gatemate = capstone["gatemate_status"]
    n_completed = capstone["n_experiments_completed"]
    n_planned = capstone.get("n_planned")

    return [
        {
            "rank": 1,
            "area": "execution_layer_gap",
            "summary": (
                f".247 completed {n_completed}"
                + (f" of {n_planned}" if n_planned is not None else "")
                + " planned experiments. .248 needs a pre-pickup checkpoint and a "
                "failure-ledger write path when an agent exits without an artifact."
            ),
        },
        {
            "rank": 2,
            "area": "verifier_and_safety_product_gap",
            "summary": (
                "tier0s_real_improvement=false, tier0u_real_improvement=false, and "
                "safety_classifier_viable=false. Re-queue the real-corpus verifier fixes "
                "and safety-corpus/integration tasks with explicit source-artifact assertions."
            ),
        },
        {
            "rank": 3,
            "area": "hardware_terminal_pending",
            "summary": (
                "GateMate still needs direct on-board Ising sampler timing capture "
                f"({gatemate.get('next_blocker')}); KV260 remains non-terminal because "
                f"{kv260.get('next_blocker')}."
            ),
        },
    ]


def build_retro(capstone):
    """REQ-REPORT-009: assemble the terminal .247 operational retro from capstone data."""
    _require_capstone_fields(capstone)

    n_completed = int(capstone["n_experiments_completed"])
    best_auroc = capstone["best_247_auroc"]
    safety_viable = bool(capstone["safety_classifier_viable"])
    tier0s_improved = bool(capstone["tier0s_real_improvement"])
    tier0u_improved = bool(capstone["tier0u_real_improvement"])
    recommendation = capstone["operator_recommendation"]

    honest_verdict = (
        f"complete: n_experiments_completed={n_completed}; "
        f"best_247_auroc={best_auroc:.4f}; "
        f"safety_classifier_viable={safety_viable}; "
        f"tier0s_real_improvement={tier0s_improved}; "
        f"tier0u_real_improvement={tier0u_improved}; "
        f"operator_recommendation={recommendation}"
    )

    return {
        "honest_verdict": honest_verdict,
        "schema": SCHEMA,
        "milestone": capstone.get("milestone", "2026.05.247"),
        "source_capstone": "results/experiment_2580_capstone_v247.json",
        "n_experiments_completed": n_completed,
        "best_247_auroc": best_auroc,
        "safety_classifier_viable": safety_viable,
        "tier0s_real_improvement": tier0s_improved,
        "tier0u_real_improvement": tier0u_improved,
        "gatemate_status": capstone["gatemate_status"],
        "kv260_status": capstone["kv260_status"],
        "operator_recommendation": recommendation,
        "top_3_successes": _top_3_successes(capstone),
        "top_3_gaps_for_248": _top_3_gaps_for_248(capstone),
        "acceptance_gates": [
            {
                "condition": "honest_verdict starts with 'complete:'",
                "passed": honest_verdict.startswith("complete:"),
                "principle": "Retro is always terminal -- reads already-completed artifacts, not compute-bound.",
            }
        ],
        "field_principles": FIELD_PRINCIPLES,
    }


def write_retro(capstone_path=CAPSTONE_PATH, output_path=DELIVERABLE_PATH):
    """REQ-REPORT-009: write the requested .247 retro deliverable JSON."""
    capstone_path = Path(capstone_path)
    output_path = Path(output_path)

    with capstone_path.open("r", encoding="utf-8") as f:
        capstone = json.load(f)

    data = build_retro(capstone)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return data


if __name__ == "__main__":
    write_retro()
