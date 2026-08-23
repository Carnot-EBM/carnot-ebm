# Claim-Audit Precision on a Stratified Corpus Sample (2026-08-23)

## Why this was measured

`scripts/experiment_claim_audit.py` is wired into milestone close, and its
flagged verdicts become OPEN rows in `ops/audit-findings-ledger.md` that a
human must dispose of. That makes its PRECISION load-bearing: a noisy audit
trains people to rubber-stamp WONTFIX, which manufactures a paper trail of
considered decisions that were never considered. Prior calibration was 5 of
9 findings judged supported on one small recent sample — an anecdote, not a
rate.

## How the sample was drawn

Deliberate stratification, not most-recent-N (recent artifacts share one
milestone's conventions and understate variety):

- Frame: all `results/experiment_*.json` — 5,110 eligible after excluding
  549 `flagged_adversarial` (the audit skips them by design), 14 files over
  8 MB (row corpora; noted cap), 8 unreadable.
- Era axis: experiment-number quartiles (IDs are assigned monotonically
  over time and are immune to this corpus's mtime rewrites). Boundaries:
  exp 1609 / 3388 / 4911 — spanning April through August.
- Verdict axis: terminal_positive / partial_negative / blocked.
- 3 artifacts per (era x verdict class) cell = 36, chosen to maximize
  substrate variety inside each cell (17 distinct `inference_substrate`
  values in the sample, including live_llm_inference, aggregation,
  verifier-ensemble, hardware_smoke, ARC offline, embedding extraction,
  and `undeclared`). Deterministic seed 20260822.

## How the audit ran

The DEPLOYED configuration: `codex` / `gpt-5.6-sol` (the conductor's
`AGENT_TYPE_AUDIT` / `AGENT_MODEL_AUDIT` from the systemd drop-in), report
redirected to the job scratch dir so `ops/experiment_claim_audit_report.md`
(the milestone-close receipt and the ledger's ingest source) stayed
untouched. 36 of 36 artifacts reviewed in ~14 minutes; no budget
truncation; zero reviewer-call failures; zero integrity-guard downgrades.

## Result

| verdict | count |
|---|---|
| NO_CLAIM | 17 |
| CLAIM_SUPPORTED | 14 |
| CLAIM_OVERSTATED | 4 |
| CLAIM_REFUTED_BY_OWN_DATA | 1 |

Each flagged finding was judged against the artifact as written (read via
`scripts/summarize_artifact.py` plus the raw JSON), never from the audit's
own summary:

| artifact | audit verdict | judgment |
|---|---|---|
| exp6149 certified_strategy_schema_fixture | REFUTED_BY_OWN_DATA | **TRUE POSITIVE.** Verdict token `complete_partial: test_commands_clean` while the artifact's own `structured_gate_receipt.gates.test_commands_clean = false`, `all_gates_passed = false`, ready score 0.0. `summarize_artifact` independently shows a live CRITICAL (nonterminal declared artifact, unstamped). |
| exp2769 ensemble_v13_tier0z | OVERSTATED | **TRUE POSITIVE.** `tier0z_auroc = 0.5065` (chance) yet `ensemble_v13_viable: true` via `diversity_criterion_met: true` — random noise is maximally uncorrelated, so the viability boolean is true by construction under a diversity-only criterion. |
| exp3304 fr11_redteam_repair_memory_replay_v2 | OVERSTATED | **TRUE POSITIVE.** `duration_s = 3e-06`; `adaptation_score = 1.0` is stored-update count over stored-update count (11/11, cannot fail); retention/forgetting copied from prior exp3291; `tests_run: []`. Honest substrate disclosure, but the verdict token carries the scores as measured capability. |
| exp847 constraint_retrieval_l2_fix | OVERSTATED | **FALSE POSITIVE.** The consumed verdict is `retrieval_partial` — already partial. The audit built its "headline claim" from the TITLE ("L2-Normalization Fix") and `root_cause` prose, and objected to `delta_simulated` — a field whose own name discloses the simulation. |
| exp3993 fourth_game_verifier_pruned | OVERSTATED | **BORDERLINE.** The verdict is an honest null ("no solve") whose causal suffix ("pruner_rejected_unseen_dynamics") is genuinely unevidenced — the artifact names no fourth game and records no rejection event (7 scalar fields total). NARROW_CLAIM is defensible advice; as a ledger row demanding disposition on a stale null it is noise. |

**Precision of flagged verdicts: 3/5 strict (0.60); 4/5 (0.80) if the
borderline is counted as a hit.** Denominator = flagged verdicts surviving
the integrity guard; there were no downgrades, so no findings were lost to
hallucinated evidence.

## What did NOT go wrong (the feared classes)

The pre-registered FP classes largely failed to materialize:

- Aggregation-judged-as-measurement (class C): all 4 sampled capstones and
  the archive task drew NO_CLAIM. Zero hits.
- Blocked-verdict-as-claim: all 12 blocked artifacts drew NO_CLAIM. Zero
  hits.
- Principle-wrapped fields (class D): both artifacts with wrapped
  `honest_verdict` (exp5324, exp5306) were read correctly and drew
  CLAIM_SUPPORTED.
- Hedged-claims-read-as-unhedged (class A): the sample's heavily
  self-disclaiming artifacts (`speedup_claim_made: false` etc.) all passed.
- Reviewer hallucination: zero integrity-guard downgrades in 36 reviews.

The one real FP class observed: **claim reconstructed from title/prose
above a partial or null verdict token** (exp847 fully, exp3993 partially).
The audit prompt tells the reviewer that downstream consumers read the
verdict token, not the hedges — the observed failure is the mirror image:
the reviewer read the TITLE, not the token, when the token was
partial/null.

## Stumbled false negatives

None found among the artifacts read for judging. The 14 CLAIM_SUPPORTED
verdicts were not systematically re-derived (out of scope); nothing read
in passing looked wrongly passed. exp5964's failed headroom controls sat
under a `blocked:` verdict and correctly drew NO_CLAIM.

## Recommendation

Precision 0.60-0.80 with this shape is GOOD ENOUGH to keep the ledger
running, for three reasons: the volume is low (5 flagged in 36, and the
production path audits ~8 recent artifacts per close, yielding ~1 flagged
row per close); all three true positives were real catches worth a human
minute (one is a verdict contradicted by its own gate field); and the false
positives cluster in ONE class rather than being diffuse.

The class to fix first, when the operator wants the numbers acted on:
**partial/null verdict tokens should bound the claim the reviewer is
allowed to reconstruct.** Concretely, the prompt (or a mechanical pre-pass
note) should state the reconciler's verdict class for the artifact's token
— "this verdict is consumed as PARTIAL/NULL" — so the reviewer stops
promoting titles and root-cause prose into headline claims over an
already-hedged token. That is a prompt/pre-pass change, deliberately NOT
made here: this was a measurement, and the audit's logic is unchanged by
it.

Secondary observation for the substrate-alias work: exp3304 runs at
3 microseconds under `inference_substrate: artifact_only_controller_memory_
replay`, an alias outside the standard table — the duration-floor gap the
`SUBSTRATE_HAS_NO_DURATION_FLOOR` warning class already tracks.

## Provenance

- Sample list + per-artifact strata: job scratch `ca_sample.json`
  (36 entries; regenerable from the seed and the frame script printed in
  the session log).
- Full review text: job scratch `ca_precision_report.md`.
- No repo state was written by the measurement run; `ops/` report and
  ledger untouched (verified via git status before and after).
