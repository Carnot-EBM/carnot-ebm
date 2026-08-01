#!/usr/bin/env python3
"""Two independent full runs must produce identical scores. Writes the witness.

WHY THIS IS NOT CEREMONY. Every number in this artifact depends on a window rebuilt by
`build_progress_window`, which STEPS A REAL ENVIRONMENT. If that env is not deterministic
across invocations, the masked and unmasked columns could differ because the corpus differed,
and the entire pass would be measuring env noise while calling it a mask effect. The unmasked
reproduction gate already rules that out against the A/B's published values; this rules it out
against a second run of THIS harness, which is the part the gate cannot see (the gate would
pass on any run whose unmasked arm happened to match, even if the masked arms wandered).

Compares EVERY arm of EVERY unit -- not just the headline -- because a wandering value in an
arm nobody looked at is exactly the kind of thing that is discovered later, by someone else.
"""

from __future__ import annotations

import json
import pathlib
import sys

FIELDS = (
    "change_fidelity",
    "accuracy",
    "n_changing",
    "n_noop",
    "hud_mask_status",
    "hud_mask_cells",
)


def index(payload: dict) -> dict:
    out = {}
    for c in payload["ab_cells"]:
        if c.get("status") != "ok":
            out[("ab", c.get("cell"))] = {"status": c.get("status")}
            continue
        out[("ab", c["cell"])] = {
            a: {f: v.get(f) for f in FIELDS} for a, v in (c.get("arms") or {}).items()
        }
    for c in payload["bon_candidates"]:
        key = ("bon", f"{c.get('game')}_k{c.get('cand')}")
        if c.get("status") != "ok":
            out[key] = {"status": c.get("status")}
            continue
        out[key] = {a: {f: v.get(f) for f in FIELDS} for a, v in (c.get("arms") or {}).items()}
    return out


def main() -> int:
    a = index(json.loads(pathlib.Path(sys.argv[1]).read_text()))
    b = index(json.loads(pathlib.Path(sys.argv[2]).read_text()))
    only_a = sorted(str(k) for k in set(a) - set(b))
    only_b = sorted(str(k) for k in set(b) - set(a))
    diffs = []
    for k in sorted(set(a) & set(b), key=str):
        if a[k] != b[k]:
            diffs.append({"unit": str(k), "run_A": a[k], "run_B": b[k]})
    witness = {
        "what_this_proves": (
            "two independent full runs of run.py, each rebuilding every window from scratch by "
            "stepping the real environment, produce identical scores under EVERY mask arm. "
            "Without it, a masked-vs-unmasked difference could be env nondeterminism."
        ),
        "n_units_run_A": len(a),
        "n_units_run_B": len(b),
        "units_only_in_A": only_a,
        "units_only_in_B": only_b,
        "n_units_compared": len(set(a) & set(b)),
        "n_units_differing": len(diffs),
        "identical": bool(not diffs and not only_a and not only_b),
        "fields_compared_per_arm": list(FIELDS),
        "differing_units": diffs[:20],
    }
    out = pathlib.Path(sys.argv[3])
    out.write_text(json.dumps(witness, indent=1))
    print(
        f"identical={witness['identical']} compared={witness['n_units_compared']} diff={len(diffs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
