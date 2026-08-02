#!/usr/bin/env python3
"""Correct the within-corpus Fisher p that `_fisher_exact_2x2`'s absolute epsilon inflated.

SURGICAL, NOT A REBUILD. Re-running `build_artifact.py` would regenerate every leaf from the
sweep inputs, and the sweeps are not being re-run -- so a rebuild would churn unrelated fields
and change bytes that other artifacts declare a sha256 of. This script touches exactly the two
leaves the bug affected, at the artifact's own `indent=2, sort_keys=False` serialisation with a
trailing newline, so the diff is minimal and reviewable.

NEVER-PRUNE: the wrong value is preserved beside the right one.

THE BUG, in one line: the tail sum used `v <= obs + 1e-12` (absolute) instead of
`v <= obs * (1 + 1e-9)` (relative), and this table's `obs` is 8.8e-25 -- far below the epsilon --
so every table with probability under 1e-12 was summed into the tail. See the docstring on
`build_artifact._fisher_exact_2x2` for the full account. The error direction is conservative: it
UNDERSTATED the significance of the mechanism finding, so no conclusion in this artifact moves.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "results" / "outer_loop_arc_goal_predicate_shadowing_20260802.json"

DUMP = {"indent": 2, "sort_keys": False, "ensure_ascii": False}

WAS = 2.2989949241439297e-14
NOW = 8.813423028029883e-25


def main() -> int:
    original = ARTIFACT.read_text(encoding="utf-8")
    d = json.loads(original)

    # Fail CLOSED if the file is not the serialisation this script was written against -- a
    # silent reformat is the exact damage this script is shaped to avoid.
    if json.dumps(d, **DUMP) + "\n" != original:
        print(
            "REFUSING: artifact does not round-trip at indent=2/sort_keys=False.", file=sys.stderr
        )
        return 2

    wc = d["mechanism"]["within_corpus_test"]
    if wc.get("fisher_exact_two_sided_p") != WAS:
        print(
            "REFUSING: within-corpus p is not the value this correction targets.", file=sys.stderr
        )
        return 2

    wc["fisher_exact_two_sided_p"] = NOW
    wc["fisher_exact_two_sided_p_SUPERSEDED_VALUE"] = WAS
    wc["fisher_exact_two_sided_p_correction_note"] = (
        "Corrected 2026-08-02. The reported 2.2989949241439297e-14 came from an absolute "
        "tie-breaking epsilon in `build_artifact._fisher_exact_2x2` (`v <= obs + 1e-12`). This "
        "table's observed probability is 8.81e-25, thirteen orders of magnitude below that "
        "epsilon, so the two-sided tail summed every table under 1e-12 instead of every table at "
        "most as probable as the observed one. Recomputed with a relative tolerance the value is "
        "8.813423028029883e-25. The error direction was CONSERVATIVE -- it reported a larger p "
        "and understated the finding -- so the mechanism conclusion is unchanged and slightly "
        "strengthened. The cross-corpus p (0.001195) used the same helper and is UNAFFECTED, "
        "because its observed probability sits far above the epsilon; that is why the defect "
        "stayed invisible until a near-perfectly-separated table appeared."
    )
    wc["independently_recounted_2026_08_02"] = (
        "The 2x2 itself was re-derived from the corpus rather than taken from analysis.json: "
        "parsing all 116 world_model.py files and counting TOP-LEVEL `is_level_complete` "
        "definitions against the duplicated `import numpy as np` signature reproduces the "
        "contingency table exactly -- (signature, two definitions) = 23, (no signature, one "
        "definition) = 93, and both off-diagonal cells empty."
    )

    ARTIFACT.write_text(json.dumps(d, **DUMP) + "\n", encoding="utf-8")
    print(f"OK: Fisher correction applied to {ARTIFACT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
