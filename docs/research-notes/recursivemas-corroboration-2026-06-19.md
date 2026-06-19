# SOTA ingestion (corroboration): RecursiveMAS (arXiv:2604.25917), 2026-06-19

**Source (VERIFIED):** "Recursive Multi-Agent Systems: Scaling Agent Collaboration through Latent-space
Recursion," Yang, Zou, Pan et al. (UIUC / Stanford / NVIDIA / MIT), arXiv:2604.25917,
https://recursivemas.github.io/. Operator-handed; logged as CORROBORATION (weaker fit than the
FinAcumen / VibeThinker ingestions — it is a multi-agent LLM *training + latent-recursion* technique
for text reasoning, not a drop-in for our execution-grounded offline ARC agent).

## What it is

Multiple LLM agents (Planner→Critic→Solver / Mixture / Distillation / Deliberation) collaborate in
LATENT space across recursive rounds, decoding text only on the final round, via a tiny trained
`RecursiveLink` module (~13M params, 0.31%; frozen base LLMs). Results: +8.3% avg accuracy, +18.1%
AIME-2025, 2.4× speedup, −75.6% tokens at recursion r=3.

## What it CORROBORATES (we already embody this — 3rd independent datapoint)

- **An explicit Critic/verifier in the loop improves accuracy.** Their Sequential pattern has the
  Critic GROUND intermediate work before the Solver finalizes = our propose→verify→refactor with the
  Carnot verifier. Adds to FinAcumen (selective gate) + our verifier-moat thesis.
- **Iterative refinement recovers errors** (their case study: round-1 off-by-one → rounds 2–3 recover)
  = our E3 induce→verify→**refactor**-on-mismatch loop. Test-time depth helps.

## FLAGGED (parked for the HIERARCHICAL-PLANNING track, NOT the live agent)

The **Planner→Critic→Solver DECOMPOSITION** pattern (a Planner decomposes a hard task into subgoals
before the Solver acts) speaks to the hierarchical-planning wall we keep hitting (the vc33/L4 ceiling;
[[reference_lecun_world_models]] names hierarchical planning as THE open problem). **DEV-SIDE idea
(corpus-building, frontier model + time — NOT the offline 16GB live agent):** use the frontier model
as a PLANNER to decompose a deep-tail game into subgoals, have the verifier ground each subgoal-solve,
and bank the decomposition as a richer corpus entry the live transfer-router can reuse. Parked until
we attack hierarchical planning; not a current ARC-sprint task.

## Why it does NOT transfer to the LIVE agent (honest caveats)

- **Latent collaboration bypasses text decoding — incompatible with our EXECUTION-grounded verifier**,
  which must run the actual `world_model.py` code every round to ground it. The moat IS the
  text/code grounding; we cannot collaborate purely in latent space.
- Needs trained `RecursiveLink` modules (we don't train the generator); multi-LLM-agent collaboration
  is infeasible on one offline 16GB P100 within the 8h/10-steps live budget.

**No experiment staged** (weaker fit; corroboration + one parked dev-side idea). Cross-refs:
`docs/research-notes/finacumen-experience-memory-ingestion-2026-06-19.md`,
[[reference_lecun_world_models]] (hierarchical-planning open problem), [[project_arc_agi3_north_star]].
