#!/usr/bin/env python3
"""How much do the treatment's probe set and the outcome's scoring set actually DISAGREE?

THE CAVEAT THIS PUTS A NUMBER ON. The accept check asks "is this predicate constant over the
SHOWN frames"; the primary outcome asks "does it discriminate on the HELD-OUT frames". Those
are disjoint frames, which is what makes the primary a generalization test rather than the
treatment restated. But disjoint is not independent: both halves come from one episode of one
game, and 88.7% of control predicates are constant on BOTH. If the two sets agreed perfectly,
the primary would be the treatment wearing a different name and the whole design would be
circular.

So measure the disagreement instead of asserting it is small or large. Pure arithmetic over
`pre/preflight_outcomes.json`, which already holds the per-frame booleans for both halves; no
LLM, no engine execution, nothing new is run.

Read the output as a bound on how much room the primary has to move INDEPENDENTLY of the
treatment. A high agreement rate is not a defect to hide -- it is the honest size of the
generalization gap, and it belongs next to the result.
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def const(vals: list) -> bool:
    """Constant over these frames, counting a raise/non-bool as carrying no information."""
    seen = {v for v in vals if isinstance(v, bool)}
    return len(seen) <= 1


def main() -> int:
    pre = json.loads((HERE / "pre" / "preflight_outcomes.json").read_text())
    ok = [r for r in pre if r.get("status") == "ok"]
    both, only_shown, only_held, neither = 0, 0, 0, 0
    for r in ok:
        cs = const(list(r["shown_before"]) + list(r["shown_after"]))
        ch = const(list(r["held_before"]) + list(r["held_after"]))
        if cs and ch:
            both += 1
        elif cs and not ch:
            only_shown += 1
        elif ch and not cs:
            only_held += 1
        else:
            neither += 1
    n = len(ok)
    agree = both + neither
    out = {
        "what_this_is": "the size of the generalization gap between the treatment's probe set "
        "(SHOWN frames) and the primary outcome's scoring set (HELD-OUT frames), measured on "
        "the 115 frozen engines rather than asserted.",
        "n": n,
        "constant_on_both": both,
        "constant_on_shown_only": only_shown,
        "constant_on_heldout_only": only_held,
        "constant_on_neither": neither,
        "agreement_rate": round(agree / n, 4) if n else None,
        "disagreement_rate": round((only_shown + only_held) / n, 4) if n else None,
        "reading": (
            "`constant_on_shown_only` is the slice where the gate REJECTS a predicate that "
            "would have scored positively on the primary -- the false-positive cost, arising "
            "purely because the two frame sets differ. `constant_on_heldout_only` is the "
            "reverse: the gate keeps a predicate the primary then scores as a failure. Both "
            "are non-zero, which is what makes the primary something the treatment does not "
            "mechanically determine. But the agreement rate is high, so the primary and the "
            "treatment are strongly correlated and the primary must NOT be presented as an "
            "independent construct. It is a generalization test with a thin gap, and the "
            "thinness is quantified here rather than glossed."
        ),
    }
    (HERE / "pre" / "circularity_gap.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
