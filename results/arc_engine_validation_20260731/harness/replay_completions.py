#!/usr/bin/env python3
"""PHASE 2, STEP 2 -- replay every REAL completion captured in Phase 1 through the validator.

WHY THIS IS THE RIGHT TEST FOR THE TRUNCATION CHECK. `truncation_defect` keys on llama-server's
own `stop_type` field, which only exists on a live completion. A unit test can assert the
function's logic but cannot show that the field ever takes the value the check keys on, on real
traffic. The Phase-1 budget sweep (2026-07-31) captured 36 completions from the live generator
against ft09's real induce prompts, each with its recorded `stop_type`, `predicted_n` and the
raw completion text on disk. Replaying those is the closest thing to a live measurement that
costs no GPU.

WHAT IT REPORTS. Per completion: what the SHIPPED `generate()` did with it
(`usable_engine`/`generate_would_accept`, recorded in Phase 1) next to what the validator says.
The interesting cells are the disagreements:

  * shipped ACCEPTED, validator FLAGS  -- a defect that reached the trust gate as a bad
    prediction instead of as broken code.
  * shipped REJECTED, validator says RETRYABLE -- a completion thrown away as a bad model when
    it was a truncated observation.

Read-only: opens the Phase-1 capture files, writes one JSON into this directory.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
sys.path.insert(0, os.path.join(REPO, "python"))

from carnot.agentic.arc_engine_static_validation import (  # noqa: E402
    missing_return_defects,
    truncation_defect,
    validate_engine_code,
)

HERE = pathlib.Path(__file__).resolve().parent
CAPTURE = pathlib.Path(REPO) / "results/arc_induce_budget_20260731"
OUT = HERE.parent / "completion_replay.json"

# Which required-symbol tuple each sweep's call shape asked for. `generate()` is called with
# ("engine","is_level_complete") on the combined and refactor lanes and ("engine",) on the
# split-induce engine lane -- getting this wrong would make the truncation check look better or
# worse than it is, so it is taken from the sweep's own recorded `required` field where present.
SWEEPS = {
    "sweep": ("engine",),  # split-induce engine-only lane
    "sweep_combined": ("engine", "is_level_complete"),
    "sweep_refactor": ("engine", "is_level_complete"),
    "sweep_sampler": ("engine",),
}


def _extract_python(text: str) -> str:
    """The SHIPPED extraction, copied verbatim from `arc_executable_world_model._extract_python`.

    Copied rather than imported so this replay cannot be perturbed by a later edit to that
    module -- the point is to reproduce what Phase 1's completions were subjected to.
    """
    if "```python" in text:
        text = text.split("```python", 1)[1]
    if "```" in text:
        text = text.split("```", 1)[0]
    return text.strip()


def main() -> int:
    t0 = time.time()
    rows = []
    for sweep_name, default_required in SWEEPS.items():
        sweep_json = CAPTURE / sweep_name / "sweep.json"
        if not sweep_json.exists():
            continue
        data = json.loads(sweep_json.read_text())
        for r in data.get("rows", []):
            fname = r.get("completion_file")
            if not fname:
                continue
            path = CAPTURE / sweep_name / fname
            if not path.exists():
                rows.append({"sweep": sweep_name, "file": fname, "error": "missing_capture"})
                continue
            raw = path.read_text(errors="replace")
            code = _extract_python(raw) or raw.strip()
            required = tuple(r.get("required") or default_required)
            stop_type = r.get("stop_type")
            budget = r.get("budget")

            trunc = truncation_defect(
                stop_type=stop_type, code=code, required=required, budget=budget
            )
            defects = validate_engine_code(
                code, stop_type=stop_type, required=required, budget=budget
            )
            static_only = missing_return_defects(code)
            rows.append(
                {
                    "sweep": sweep_name,
                    "file": fname,
                    "budget": budget,
                    "attempt": r.get("attempt"),
                    "arm": r.get("arm"),
                    "stop_type": stop_type,
                    "predicted_n": r.get("predicted_n"),
                    # what Phase 1 recorded the shipped path doing
                    "shipped_would_accept": r.get("generate_would_accept"),
                    "shipped_usable_engine": r.get("usable_engine"),
                    "shipped_returns_on_all_paths": r.get("engine_returns_on_all_paths"),
                    # what the validator says
                    "validator_kinds": sorted({d.kind for d in defects}),
                    "validator_retryable": any(d.retryable for d in defects),
                    "validator_repairable": any(d.repairable for d in defects),
                    "truncation_flagged": trunc is not None,
                    "static_kinds": sorted({d.kind for d in static_only}),
                }
            )

    live = [r for r in rows if "error" not in r]
    capped = [r for r in live if r["stop_type"] == "limit"]
    trunc_flagged = [r for r in live if r["truncation_flagged"]]
    # The two disagreement cells that matter (see the module docstring).
    accepted_but_flagged = [
        r
        for r in live
        if r.get("shipped_would_accept") and r["validator_kinds"] and not r["truncation_flagged"]
    ]
    rejected_but_retryable = [
        r for r in live if not r.get("shipped_would_accept") and r["validator_retryable"]
    ]

    out = {
        "generated_by": "results/arc_engine_validation_20260731/harness/replay_completions.py",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(time.time() - t0, 3),
        "n_completions_replayed": len(live),
        "n_hit_output_cap": len(capped),
        "n_truncation_flagged": len(trunc_flagged),
        "n_shipped_accepted_but_validator_flags": len(accepted_but_flagged),
        "n_shipped_rejected_but_retryable": len(rejected_but_retryable),
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        f"replayed {len(live)} real completions: {len(capped)} hit the output cap, "
        f"{len(trunc_flagged)} flagged truncated-before-required-symbols"
    )
    print(
        f"  shipped ACCEPTED but validator flags a defect: "
        f"{len(accepted_but_flagged)}  -- {[r['file'] for r in accepted_but_flagged]}"
    )
    print(
        f"  shipped REJECTED but retryable (truncation): "
        f"{len(rejected_but_retryable)}  -- {[r['file'] for r in rejected_but_retryable]}"
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
