# Carnot North Star — Headline Claim, Stable Gate, Focus

**Created:** 2026-05-29 (Opus 4.8 fresh-eyes review, operator-authorized).
**Purpose:** Give the autonomous loop ONE thing to monotonically improve and
ONE fixed finish line. The project runs ~30 milestones/day, produces 3,100+
artifacts and 450+ milestones, yet `paper_ready` has been `False` throughout
and the version numbers (cross-corpus matrix v36, repair panel v11, SOTA
receipt v9, KAN distill v4, clean verifier rerun v15) climb in lockstep —
the signature of *iteration without convergence*. This document is the
convergence anchor.

**This document is operator-curated.** The autonomous loop may read it as the
authoritative north star but must not edit it (Public Documentation
Discipline). Numbers here trace to primary artifacts, not prose.

---

## 0. THE NORTH STAR (operator directive 2026-06-08): solve ARC-AGI-3, accurately and efficiently

> **"Solving ARC-AGI-3 is our new north star, as accurately and efficiently as
> possible."** — operator, 2026-06-08.

This is the project's **destination**. Everything below (§1 the FoVer headline,
§2 the publication gate, §3 hardware, §5 the verifier-moat + agentic harness)
is now **in service of this goal**, not a parallel track. The reframe is not a
discard — the FoVer headline and the stable gate remain the things we *publish*
and the discipline we *converge with*; they are now **supporting results** that
de-risk the path to ARC-AGI-3, not the end of the road.

**The two metrics (both first-class — "accurately AND efficiently"):**

| Axis | Metric | Why it is the north star, not just a number |
|---|---|---|
| **ACCURACY** | ARC-AGI-3 solve-rate / official ARC score on the live benchmark | The hard, un-gamed test of *general* directed reasoning — the project's core motivation (escape hallucination, autonomous directed self-learning). |
| **EFFICIENCY** | actions-to-solve **and** compute/latency/cost per solve | ARC-AGI-3 is interactive (act→observe→act); a solver that brute-forces actions or burns an LLM-judge per step does not scale. Efficiency is where Carnot's *verifier* earns its place (router + action-pruner; the Exp1165 pilot showed ~4× fewer actions). |

**The honest division of labor (do NOT overclaim the verifier here).**
ARC-AGI-3 is an *induction* benchmark — inferring a latent rule from few
interactions. The **generator** does the induction: a learned amortized
recursive refiner (TRM-class, which reaches ~0.82 on Sudoku-Extreme where AR
and energy-descent get ~0%) and/or an open local LLM. The **energy verifier**
does NOT induce; it makes the search *accurate and efficient* — routing the
cheap encodable specialist vs escalating to the LLM (Meta-EBM Cascade Router),
pruning hopeless actions (verifier-as-free-energy), and verifying
state/trajectory at scale. This is the HYBRID architecture of §5: generator =
commodity/third-party, energy = verification layer. The north star is the
*system's* solve-rate; the verifier is Carnot's specific, load-bearing,
existential contribution to it.

**Access is OPEN — the blocker is the BUILD, not access (corrected 2026-06-08).**
Earlier framing called real-benchmark access the binding blocker (exp1166
"leaderboard_unavailable_email_drafted"). That is now disproven: the official SDK
(`pip install arc-agi`, v0.9.8) auto-issues an **anonymous key** and exposes **25
live ARC-AGI-3 environments** today (verified — `results/arc_agi3_access_probe.json`).
A registered `ARC_API_KEY` (free, three.arcprize.org) is only needed for rate limits +
official scorecards; submitting to the official leaderboard remains operator-only
(external publication). So the real blocker is that **we have not built the harness
against the live env** — everything to date is synthetic (Exp1165 toy env, exp3919
synthetic scaffold, exp3929 synthetic action-efficiency). The metrics map directly:
ACCURACY = `EnvironmentScore.score`/`levels_completed`; EFFICIENCY = actions taken vs
`EnvironmentInfo.baseline_actions[level]` (the per-level reference count). Per the
sequence discipline below, the OFFLINE verifier proof still goes first; but a real-env
random/greedy baseline is now a cheap, unblocked grounding step available immediately.

**The sequence (the §5 discipline still governs — offline proof FIRST).**
The harness's value DERIVES from the verifier; a beautiful harness around an
unproven verifier is still glue. So the path to the north star is:

1. **Offline verifier proof** (fast, no harness, already staged): the
   moat-scissor accuracy rerun (does the external verifier catch what the model
   self-verification misses, in-distribution?) + the energy-verifier-vs-LLM-judge
   efficiency head-to-head (target: "parity at 10–100× cheaper"). This is the
   FoVer-domain proof; the published headline (§1) is its receipt.
2. **Verifier domain expansion** (the registry program, `ops/verifier_registry.yaml`):
   grow beyond math toward the perception/grid/rule-induction domains ARC-AGI-3
   needs, with formal oracles where available. The verifier is domain-bound
   today (math strong, facts earned-negative, code weak — memory
   `verifier-domain-bound-math-only`); ARC-AGI-3 demands new domains.
3. **ARC-AGI-3 real harness** (gated on access): the agentic integration surface
   of §5 — router + action-pruner + state-verify — measured on the two metrics
   above. Building it verifier-first makes harness-building and verifier-proving
   the SAME work.

**The rule (extends §1's rule).** A milestone advances the north star if it
raises ARC-AGI-3 accuracy, lowers ARC-AGI-3 cost/actions, OR de-risks the path
(an offline verifier proof, a new verifier domain, real-benchmark access). A
milestone that re-measures an already-settled artifact without moving any of
those is churn.

**What does NOT change.** The §1 FoVer headline (0.9131) and the §2 G1–G4 gate
stay fixed and remain the publication target — reframed as the *supporting
evidence that the verification layer works*, which is the precondition for it
to be useful inside the ARC-AGI-3 harness. Sovereignty/decentralization rules,
hardware-as-energy-evaluator (§3, §5), and all CLAUDE.md disciplines are
unchanged.

---

## 1. THE HEADLINE CLAIM (the one surviving positive)

Reconstructed 2026-05-29 by auditing every positive claim against the
Paper-v6 Narrowing Discipline retraction list. Two claims survive all the
walk-backs (FoVer 0.9857→0.9131, hardware speedup retracted, replacement-grade
refuted, Spera-generalization retracted, thermalization retracted):

### Methods headline (most rigorous)
> **Carnot's verifier ensemble reaches AUROC 0.9131 on the FoVer
> step-error corpus (n=1,000, 5 seeds, dual-condition, CI95 [0.9027, 0.9235]),
> and an isolated memory-ablation shows the FR-11 self-learning component
> contributes +0.0185 AUROC (CI95 [0.0125, 0.0245]).**

Source: exp2837 (`results/experiment_2837_fover_memory_leakage_v3.json`),
re-run as exp2850. Adequate n, 5-seed, adversarial-verify clean, not
contradicted by any later experiment, not on the retraction list.

**Precision corrections (surfaced 2026-05-29 while scoping the G2 runbook —
the earlier draft overstated this):**
- It is a **4-verifier** score (`fr11_session_memory`, `tier0r_curry_howard`,
  `tier0s_arithmetic_gap`, `tier0u_logical_consistency`), NOT "k=15". The
  broader ensemble is larger; only these four score FoVer.
- It is **verifier-scoring against the labeled corpus on CPU**
  (`live_model_invoked: False`, ~16s), NOT "live RTX 3090 inference". This is
  a *strength* for G2: the headline is cheaply, externally reproducible with
  no GPU or 35B model. See `ops/reproduction-runbook-fover-headline.md`.

### Product headline — DEMOTED 2026-05-29: prose numbers do NOT trace to artifacts (G4 catch)

The originally-drafted product headline ("HumanEval 8%→80%", "0%→36% pass@1")
was reconstructed from the technical-report *trajectory prose*. G4 verification
against primary artifacts (2026-05-29) **refuted it**:

- `results/experiment_227_results.json` (the cited source): n=30, model
  Qwen3.5-0.8B (CPU-smoke tier, not 35B), baseline pass@1 0.233 →
  verify_repair pass@1 **0.233**, **improvement delta = 0.0, n_repaired = 0**.
- `results/experiment_226_results.json` (Gemma4-E4B-it, 164q): baseline 11.6%,
  verify_only *degrades* to 5.5%.
- Recent full-ensemble HumanEval evals (exp2830/2838) were CUDA/GGUF-blocked
  (AUROC None, 0 labeled candidates) — no numbers.

**No artifact supports "8%→80%" or "0%→36%". Baselines are 11–66%, never 0%.**
The technical-report trajectory paragraph overstates the code-repair result.

The **genuine surviving** positive code results (more modest, differently
framed, still need full n + seed + checksum confirmation before headlining):
- exp1999 (code_verification_humaneval): baseline 0.66 → repair **0.84** (+18pp)
- exp2090 (CRANE constrained decoding, n=50): rigid 0.70 → CRANE **0.85** (+15pp)

Until those are fully provenance-confirmed, **the FoVer methods headline
(0.9131) is the SOLE defensible headline.** Do not cite the 8%→80% / 0%→36%
numbers anywhere.

**OPERATOR ACTION (technical report is operator-curated — flagged, not
auto-edited):** correct the technical-report trajectory paragraph to replace
"8%→80%"/"0%→36%"/"+3.0pp" with the real exp1999/exp2090 numbers, or remove
the code-repair claim until a clean live-GPU full-HumanEval repair run lands.

### The rule
Every milestone either advances the headline claim (tightens the CI, raises
the AUROC, replicates on a new seed/corpus, or confirms the product numbers
from primary artifacts) or it is **noise**. A milestone that produces a new
version of an existing artifact without moving the headline is churn and
should be questioned at planning time.

---

## 2. THE STABLE PUBLICATION GATE (replaces publication_blocker_count)

`publication_blocker_count` is RETIRED as a steering metric. Evidence it
cannot steer: it went 105 → 10 between capstone v303 and v304 via
`blocker_delta_from_v303: -95` — a recount, not 95 resolutions. A metric that
can move 90% by redefinition is not a finish line.

**Replacement — a FIXED, EXTERNAL, 4-condition gate. Do NOT redefine these to
show progress; redefining the gate is the failure mode this replaces.**

| Gate | Condition | Current status |
|---|---|---|
| **G1 — Headline measured** | The headline claim's metric, on a NAMED FROZEN eval set, replicated ≥5 seeds, CI reported, live-GPU, adversarial-verify clean | ✅ MET (FoVer 0.9131, exp2837) |
| **G2 — Independently reproduced** | ≥1 reproducer who is not the operator re-runs the headline experiment and lands within the CI | ❌ UNMET (external) |
| **G3 — Prose is narrowing-clean** | Paper draft states ONLY surviving claims; passes a narrowing lint; zero retracted phrasings | ❌ UNMET (no narrowing lint exists yet) |
| **G4 — Numbers trace to primary artifacts** | Every headline number resolves to a `results/experiment_*.json` carrying `random_seed` + `reproducibility_checksum` (not prose) | ⚠️ PARTIAL — FoVer ✅; **product numbers FAILED G4 2026-05-29** (cited exp227 shows delta=0.0; "8%→80%"/"0%→36%" trace to no artifact — demoted, see §1) |

`paper_ready := G1 ∧ G2 ∧ G3 ∧ G4`. The capstone should report **which of
G1–G4 are unmet**, not a count. Three of four gates are now well-defined and
two are within reach (G3: ship the narrowing lint; G4: confirm product numbers
from artifacts). G2 is the genuine external dependency.

**Follow-up (operator-gated, code change):** wire the conductor's capstone
logic to emit `g1..g4` booleans instead of `publication_blocker_count`. Not
done here — it changes conductor behavior and is the operator's call.

---

## 3. HARDWARE FOCUS — one sovereignty story, the rest is honest future-work

The portfolio was already narrowed to 3 active FPGA boards (Exp 1460 scope
reduction); Tenstorrent / Extropic Z1 / photonic / D-Wave are already
deferred. But three live boards still consume disproportionate operational
attention (the CUDA outage, the GateMate flash chain, board-reachability
gates). Fresh-eyes recommendation:

| Board | Decision | Rationale |
|---|---|---|
| **KV260** | **THE sovereignty story — drive to terminal, then freeze** | Only board near terminal state: reachable via SSH, `carnot_ising_v4` flashed, exp2898 latency anchored, exp2938/2939 measurements landed. One more milestone (board-latency transcript) reaches terminal. |
| GateMate | Opportunistic only | First flash landed (operator evidence); host-IO smoke pending. Do NOT block milestones on it. |
| PolarFire | Opportunistic only | Scaling validated to 1000 clauses; no terminal-state mandate. |
| Tenstorrent / Extropic / photonic / NPU | Honest future-work | Document as roadmap, not active tracks. exp1584 Wormhole preflight stays a blocked-on-access artifact. |

**The narrowed honest hardware claim** (per Paper-v6 Narrowing #3/#9): KV260
is a **POC functional simulator anchoring future high-N deployment** +
**local edge deployability** with Vivado/Xilinx dependencies disclosed. NOT
"hardware speedup" (it is ~0.98× = slower than CPU at d=128) and NOT "hardware
sovereignty" (the toolchain is commercial).

**The rule:** Hardware-Task Continuity Discipline currently mandates ≥1 task
per board per milestone. Recommend relaxing to: KV260 until terminal, then the
mandate lifts and all three become opportunistic. This stops the loop from
spending a task slot per milestone on boards that are not the focus.

---

## 4. RULE GARBAGE-COLLECTION — classification (see CLAUDE.md rule index)

CLAUDE.md is ~2,815 lines / ~29 distinct MANDATORY rules. Audit classification
(2026-05-29):

- **~19 ACTIVE** — load-bearing, require judgment, keep in the active path.
- **~8 MECHANICALLY-ENFORCED** — a pre-commit lint or conductor guard now
  enforces them; the prose is reference-only. Enforcement does NOT depend on
  the prose, so these are safe to mark HISTORICAL without disabling anything.
- **~2 HISTORICAL/SUPERSEDED** — Codex-Default (superseded by Gemini-Default;
  already marked historical), Paper-v6 Narrowing (has explicit retirement
  trigger when corrigenda land).

**Safe-to-archive candidates (enforcement is automatic; prose is redundant):**
- Canonical Repository URL Discipline → `canonical_url_lint.py`
- Calendar-Month Prefix Rollover → `_expected_next_milestone()`
- Overdue-Priority Forcing Function → `overdue_priority_lint.py`

A navigational **Rule Index** was added to the top of CLAUDE.md (2026-05-29,
additive only — no prose removed, per never-prune-docs) so agents can find the
~19 load-bearing rules without reading all 29. Actually MOVING the mechanically-
enforced rules into a HISTORICAL section is an operator governance decision
and is NOT done automatically — the index is the low-risk first step.

---

## 5. STRATEGIC REFRAME (2026-06-06): energy VERIFIES, refinement GENERATES

**What is now settled (overwhelmingly, multi-domain).** Energy-as-GENERATOR is
closed-negative: a contrastively-trained energy SCORES near-perfectly but
GENERATES nothing by descent — refuted on Sudoku (v1–v4 ablation: 0% even with a
perfect latent + perfect carving), on GSM8K (exp3882 EBT kill-gate: energy-descent
0.000 vs AR 0.94, positive control passed), and reinforced by exp3883 (K-curve
plateau) and an external corroboration (NVIDIA's open QEC neural decoder beats the
MWPM energy-minimization decoder; surface-code decoding ↔ random-bond Ising). The
GENERATOR is a learned amortized **recursive refiner** (TRM-class): with NO energy
it reaches ~0.82 on Sudoku-Extreme (SOTA-adjacent) where AR and energy-descent get
~0%. See docs/research-notes/energy-as-generator-sudoku-thesis.md, memories
`energy-as-generator-sudoku`, `nvidia-ising-qec-amortized`.

**What Carnot IS, post-reframe (HYBRID, pragmatic — memory
`hybrid-pragmatic-architecture`).** An energy-based **VERIFICATION layer** grounding
a learned generator — NOT an energy-based generative foundation model. Generator =
commodity (open local LLM Qwen/Gemma) or a small third-party refiner (TRM) per
domain; the energy ensemble is the **verifier / scorer / oracle / abstention gate**,
never the generator. LLM-free general reasoning is a north-star research direction,
NOT a near-term gate. Sovereignty preserved (open/local). This is exactly the role
the publication-ready FoVer headline (§1) already occupies — the reframe prunes the
refuted generator ambition, it does not touch the headline.

**What this CONCENTRATES (the existential point).** With the generator commodity/
third-party, the VERIFIER is Carnot's entire value-add — and its value is UNPROVEN:
the moat-scissor (does the external verifier beat the model self-verifying?) is
INCONCLUSIVE (exp3885); energy-rerank HURT the generator (v4); the verifier is
domain-bound (math strong, facts earned-negative, code weak — memory
`verifier-domain-bound-math-only`). So all of Carnot's risk now sits in ONE place.

**The win condition (operator 2026-06-06).** The verifier earns its place if it is
**equally effective as the LM at lower cost/latency** (efficiency-parity) — it does
NOT need an accuracy edge, though accuracy gains are still pursued where worthwhile
(Pareto: dominate the LLM baseline — cheaper at equal accuracy AND/OR more accurate
at equal cost). Efficiency-parity is also RSI-SCALE load-bearing: verifying a
machine-scale "virtual lab" can't afford a generative LLM-judge per artifact, so a
cheap, hardware-acceleratable, externally-grounded forward-pass verifier is the
property that makes scaled verification tractable (memory
`quote-anthropic-corroboration`, docs/research-notes/anthropic-rsi-and-w2s-citations.md).

**Hardware, repurposed.** The DBAE→Ising→energy-MINIMIZATION-generates path is dead.
Ising/FPGA/thermodynamic devices instead **EVALUATE** energy (cheap forward
verification at scale) — the verifier is the hardware-acceleratable primitive. The
NVIDIA-Ising calibration pattern maps onto Carnot's analog-sampler drift-correction.
Disambiguate "Ising MACHINE/sampler" from NVIDIA Ising in all docs.

**THE LOAD-BEARING NEXT WORK (supersedes generator experiments).** Prove the
verifier earns its place — two axes, both currently owed: (A) ACCURACY — an
infra-fixed moat-scissor rerun (does the external verifier catch what the model's
own self-verification misses, in-distribution?); (B) EFFICIENCY — an energy-verifier
vs LLM-as-judge head-to-head reporting BOTH accuracy parity (within CI) AND the
compute/latency ratio. Target result: "parity at 10–100× cheaper." Re-scope Phase-3
to the hybrid (refiner generator + energy verifier); retire energy-generates prose.
Staged as a MANDATORY-NEXT-MILESTONE PRIORITY in ops/known-issues.md.

**THE AGENTIC PROOF VENUE — the ARC-AGI-3 harness (Phase-4, sequenced SECOND).**
ARC-AGI-3 (the target) is interactive (act->observe->act), so it REQUIRES a harness:
environment adapter + perception/encoding + router + generators + scoring. The
domain-match + encodability routing (which generator per state) lives INSIDE that
harness. Designed VERIFIER-FIRST, the harness is where the energy verifier does
three load-bearing jobs at once — (1) ROUTER/arbiter (run the cheap encodable
specialist, verify, escalate to the LLM on failure — the Meta-EBM Cascade Router),
(2) ACTION-PRUNER (verifier-as-free-energy action selection; the Exp1165 pilot
showed ~4x fewer actions on synthetic ARC-AGI-3-style puzzles), (3) scaled
state/trajectory verification. So the harness is not plumbing — it is the AGENTIC
PROOF of the verifier (action-efficiency + cost as metrics) AND the integration
surface that turns the hybrid into a working agent; building it verifier-first makes
harness-building and verifier-proving the SAME work. DISCIPLINE (so it doesn't
become breadth): the harness's value DERIVES from the verifier (a beautiful harness
around an unproven verifier is still glue), so SEQUENCE the OFFLINE verifier proof
FIRST (moat-scissor + efficiency head-to-head on FoVer — no harness needed, fast,
already staged), THEN the ARC-AGI-3 harness as the agentic extension. It is a real
build we have NOT started (the Exp1165 pilot used a toy env, not the real
benchmark). See memory `arc-agi3-harness-verifier-proof`,
docs/research-notes/active-inference-phase4-bridge.md.

---

## Why this document exists

The project has world-class **rigor** (trustworthy negatives: v4 refuted,
FoVer repinned, 11 claims retracted) but was under-served on **convergence**.
The same machinery that makes its negatives trustworthy let it run
indefinitely without foregrounding a positive. This north star fixes the
target (§1), fixes the finish line (§2), focuses the surface (§3), and lowers
the governance tax (§4). Review and adjust the headline claim and gate
conditions as the operator sees fit — but keep them FIXED between adjustments,
because a moving target is what prevents convergence.

## Cross-references
- research-references.md "HEADLINE NEGATIVE RESULT" (v4 refutation)
- CLAUDE.md "Paper-v6 Narrowing Discipline" (the 11 retractions)
- CLAUDE.md Rule Index (added 2026-05-29)
- exp2837 / exp2948 (FoVer 0.9131 headline source)
- research-hardware-wishlist.md Exp 1460 portfolio narrowing
- _bmad/prd.md (the long-term "autonomous directed self-learning" vision)
