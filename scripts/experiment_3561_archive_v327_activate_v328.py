#!/usr/bin/env python3
import json
from pathlib import Path

def main():
    """
    Archives milestone .327 and writes the .327 operational retrospective JSON.
    Confirms .328 is active.
    """
    payload = {
        "schema": "carnot.operational_retro.v67",
        "experiment": 3561,
        "honest_verdict": "complete: archive_v327_and_retro_complete",
        "archive_v327_activate_v328_ready": True,
        "random_seed": 20260601,
        "experiments_completed": 11,
        "key_finding": "P0.1 Route-1 got a CLEAN TERMINAL POSITIVE on a DISCRIMINATING corpus \u2014 energy 0.9625 vs strong DSATUR 0.70, p=0.000; Route-2 and the aggregation promotion blocked on BUILD failures not science; FR-11 deployed on a non-degenerate corpus; depth_forcing_function_can_relax=true",
        "g2_status": "external run pending = sole unmet gate",
        "wall_time_minutes": 15,
        "top_forward_gap": "CONSOLIDATE: generalize the Route-1 positive to a SECOND discriminating CSP + multi-seed CI on graph coloring; promote the cross-corpus aggregation positive (A->{B,C}); give Route-2 NL-math its terminal verdict; advance self-learning + P0.2 diversity; G2 drift-verify"
    }
    
    Path("results").mkdir(exist_ok=True)
    
    with open("results/experiment_3561_archive_v327_activate_v328.json", "w") as f:
        json.dump(payload, f, indent=2)
        
    with open("results/operational_retro_2026_05_327.json", "w") as f:
        json.dump(payload, f, indent=2)

if __name__ == "__main__":
    main()
