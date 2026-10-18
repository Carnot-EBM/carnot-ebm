# ManipulableSignalDependency constraint template

**Status:** Draft change proposal.
**Origin:** [GitHub issue #6](https://github.com/Carnot-EBM/carnot-ebm/issues/6) (2026-04-24).
**Target milestone:** 2026.04.63.
**Priority:** High. Directly addresses a threat class that the current
  cascade does not catch (structural single-signal-dependency bugs, as
  distinct from logical contradiction or arithmetic error).
**Depends on:** existing `LLMConstraintExtractor` reasoning-graph output.

## Summary

A recurring LLM failure pattern isn't a logical contradiction or arithmetic
error — it's structural over-reliance on a single externally-sourced
signal with no independent corroboration. Examples:

- "The API returned status=healthy, therefore the service is up" — when
  that API proxies through the service it's supposed to check.
- "The search result says X, therefore X is correct" — when the search
  layer is the attack surface.
- "The sensor reports T=20°C, therefore it's 20°C" — when the sensor is
  failed or adversarially tampered.
- RAG answers citing a retrieved document whose contents are
  attacker-influenced.
- Tool outputs cited as ground truth when the tool is itself LLM-driven.

A contradiction verifier doesn't flag these — the chain is *internally*
consistent. It's the graph structure that's weak: too many load-bearing
conclusions rest on a single manipulable node.

See issue #6 for detection-sketch details.

## Proposed experiments

### Exp A — Manipulable-signal pattern catalogue

**Deliverable:** `examples/constraint_packs/manipulable_signal_v1.json` +
`results/experiment_<N>_manipulable_signal_corpus.json`.

**What it does:** Build a labelled corpus of ~200 reasoning chains, half
exhibiting single-signal-dependency and half with redundant corroboration.
Hand-label the load-bearing node for each positive case.

### Exp B — `ManipulableSignalDependency` constraint-template primitive

**Deliverable:** `python/carnot/constraints/manipulable_signal.py` +
`results/experiment_<N>_manipulable_signal_template.json`.

**Detection sketch (from issue #6):**

Over an extracted reasoning graph (similar to `LLMConstraintExtractor`
output):

1. Identify "anchor" claims — conclusions whose support chain terminates at
   a single non-axiomatic source (API response, search result, sensor
   reading, retrieved document, tool output).
2. Compute an "influence centrality" score — fraction of the final verdict
   that depends on each anchor.
3. Flag when any single anchor's centrality exceeds a threshold (default
   0.5) AND the anchor is of a type known to be manipulable (typed
   classification: API-proxied, retrieval-sourced, sensor-sourced,
   tool-output).
4. Energy score: monotonic in centrality, high when the anchor is both
   load-bearing and manipulable.

**Acceptance gates:**

1. On the Exp A corpus, AUROC ≥ 0.85 separating single-signal-dependency
   cases from redundantly-corroborated ones.
2. False-positive rate on a 100-sample benign-axiomatic corpus ≤ 5%.
3. Honest-verdict enum: `manipulable_signal_detector_ships`,
   `detector_auroc_below_gate`, `detector_false_positives_above_budget`.

### Exp C — Cascade wire-in as optional tier

Same pattern as the probability-calibration verifier (issue #1): opt-in
flag on `VerifyRepairPipeline`. Slots as a side-car between extraction and
EORM, because the pattern is graph-structural and can be evaluated before
energy scoring.

## Risks

- **Anchor-typing is domain-specific.** "API-proxied" means nothing in a
  medical-triage domain. Mitigation: ship `manipulable_signal_v1.json` with
  a domain tag; users can extend the typed-classification list for their
  domain.
- **False positives on axiomatic anchors.** A chain that reasons from
  genuinely-trustworthy sources (a formal proof, a well-known mathematical
  identity) shouldn't trigger. Mitigation: the typed-classification step
  whitelists `axiomatic` / `formal_proof` / `mathematical_identity` as
  non-manipulable.
- **Gaming by injection of fake corroboration.** An attacker could add
  synthetic "corroborating" statements to defeat centrality detection.
  Mitigation: the dogfood safeguard (see
  `conductor-self-protection-safeguard.md`) catches obvious injection at
  the input boundary; this proposal complements, doesn't replace, that
  defence.
