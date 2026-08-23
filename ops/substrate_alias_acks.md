# Substrate alias acknowledgements

Operator acknowledgements for names added to `NO_LLM_SUBSTRATE_ALIASES` in
`scripts/adversarial_verify.py` (REQ-SUBSTRATE-ALIAS-1).

## Why this file exists

An alias in that tuple exempts an artifact from the fabrication gate's
`DURATION_TOO_SHORT` check. On 2026-08-23 the tuple held 38 aliases, 19 of them
added in the preceding two days, and every sampled addition landed in the same
commit as the artifact it exempted. Nothing recorded why any of them was
believed to run no LLM.

An alias needs either a test naming it or a line here. This file is the second
option: cheaper than a test when the substrate is obviously deterministic, and
still a written reason a later reader can check.

## How to add a line

One line per alias. Date it, name it exactly as it appears in the tuple, and
say in plain words what the experiment actually ran and why no model was
invoked. "Deterministic" alone is not a reason — say what did the work.

    - 2026-08-23 `example_exact_solver_no_llm` — runs the Z3 solver over a
      pre-built instance file; no model is loaded and no server is contacted.

## Acknowledgements

<!-- Append below. Never rewrite or remove a line: an ack that turned out to be
     wrong gets a dated correction underneath it, not a deletion. -->

_None yet. The 38 aliases present when this rule shipped are grandfathered —
the lint governs additions from 2026-08-23 onward. Auditing the existing 38 is
separate, still-open work; see the QA-layer audit's coverage of
`adversarial_verify.py`._
