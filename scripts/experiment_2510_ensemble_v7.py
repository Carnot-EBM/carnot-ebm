#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path


def _write_blocked_artifact(start_time: float, reason: str) -> None:
    artifact = {
        "honest_verdict": "blocked_tier0r_not_implemented",
        "ensemble_v7_auroc": 0.0,
        "ensemble_v7_auroc_std": 0.0,
        "ensemble_v6_baseline": 0.9750,
        "tier0r_group_assignment": "Group C",
        "n_seeds": 5,
        "preconditions_checked": ["tier0r_import", "tier0r_search", reason],
        "duration_s": time.time() - start_time,
        "random_seed": 42,
    }

    out_path = Path(__file__).parent.parent / "results" / "experiment_2510_ensemble_v7.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print("Wrote blocked artifact")


def main() -> None:
    start_time = time.time()

    tier0r_found = False
    try:
        from carnot.verify.tier0r_curry_howard import Tier0rVerifier  # type: ignore

        del Tier0rVerifier
        tier0r_found = True
    except ImportError:
        verify_dir = Path(__file__).parent.parent / "python" / "carnot" / "verify"
        tier0r_found = bool(list(verify_dir.glob("tier0r*.py")))

    if not tier0r_found:
        _write_blocked_artifact(start_time, "tier0r_not_found")
        sys.exit(0)

    _write_blocked_artifact(start_time, "tier0r_integration_not_implemented")
    sys.exit(0)


if __name__ == "__main__":
    main()
