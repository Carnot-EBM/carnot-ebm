# Research Roadmap — Milestone 2026.06.346

**POST-BOUNDED CONVERGENCE: settle the LAST open energy-existential question
(the P1 discrete-search adjudication, v3 — the positive control was starved in
v1/v2, the harness is now fixed), bank the verifier product, build the P3
Anomaly-Escalation process upgrade, scaffold the operator's next-thesis seed
(EDLM), and continue self-learning on the live verifier — without re-grinding
anything bounded.**

Planner: Claude Opus 4.8 — 2026-06-03. Outer-loop pre-staged roadmap per the
Pre-Staged Roadmap Convention.

---

## 1. What the previous milestones proved — the converged state

The project is **converged on its one surviving positive** and has bounded both
energy-foundation routes. As of `.345 (which fully landed — 11/11 tasks,
`paper_ready=true` confirmed live via `scripts/publication_gate.py`):

### The banked positive (frozen)
> **Carnot's 4-verifier ensemble reaches AUROC 0.9131 on the FoVer step-error
> corpus (n=1,000, 5 seeds, dual-condition, CI95 [0.9027, 0.9235]); the FR-11
> self-learning component contributes +0.0185 (CI95 [0.0125, 0.0245]).**

Source `exp2837`/`exp2850`. **G1–G4 all met** (G2 independently reproduced on a
clean GitHub Actions runner, 2026-05-31, run 26725185125; G2 re-hardened locally
as `exp3767`, AUROC 0.913134 within CI). The headline is **frozen**. A certified
abstention operating point shipped in `.345 (`exp3771`: threshold 0.733,
coverage 0.998 at selective-risk ≤5%, split-conformal, δ=0.05, n=2,619). The
verifier is **math/structured-reasoning-bound** — earned-negative on facts
(RAGTruth) and weak on code; this is settled and must NOT be re-tested without a
genuinely-new architecture. `[[project_verifier_domain_bound]]`

### Both energy-foundation routes are bounded
- **energy-as-SELECTOR** (P0.1) — bounded: reranking does not beat
  self-consistency where SC is strong (math/CSP), adds no value where SC is weak.
  `[[project_energy_selection_thesis_bounded]]`
- **energy-as-GENERATOR** (Thesis A / tiny EBT) — bounded at scale: with a
  learned emb→token decoder + 3-digit + 16k steps, the EBT scores 0.000 vs AR
  0.84 at matched compute (discriminative-not-generative). Field-consistent with
  EBT-Policy (arXiv:2510.27545). `[[project_thesis_a_ebt_seeded]]`

### The verifier-moat thread is CLOSED (do not re-open)
The 2026-06-03 Deep Think round settled the moat (P2) for ~zero GPU: on standard
MATH, self-consistency AUROC is near-ceiling (~0.95) and the SC-failure residual
(the moat's only home) was ~5 items — all of which turned out to be
answer-**normalization** false-negatives, not genuine confident-correlated
errors. **The verifier's defensible claim is a MATH reasoning/step-error checker
(the banked FoVer product), NOT a general independence-moat over final answers.**
A scissor-plot GPU sweep would be re-grinding a closed thread — it is NOT in this
milestone. `[[reference_deep_think_post_bounded_2026_06]]`

### The one genuinely-open energy-existential question — P1 mechanism
The Deep Think P1 round asked: *is the energy-as-generator bound an ARTIFACT of
the decode method (Langevin descends off-manifold into the void between valid
token embeddings) or FUNDAMENTAL (a global symmetric energy E(y₁…y_T) cannot
enforce AR's step-by-step causal factorization — the "Causal Inductive Bias
Gap")?* The decisive cheap test: **discard Langevin + learned decoder; decode by
pure DISCRETE search over valid-token embeddings only** (beam / discrete-MCMC).
`>0%` → artifact-bounded (decode was the culprit; reopen). Still `0%` with a
**valid positive control** → landscape itself misshaped → supports
causal-inductive-bias = FUNDAMENTAL → writes the closing theorem.

**This test was attempted twice today and is INCONCLUSIVE — through no fault of
the energy side:**
- `thesis_a_p1_discrete_search` (v1): the AR positive control collapsed to 0.0
  (reused the too-small matched-compute model at 3-digit). DEGENERATE.
- `thesis_a_p1_discrete_search_v2` (v2): AR still 0.0 because the harness used
  `n_train=min(20000,…)` — 20k examples starve the from-scratch AR on 3-digit
  MSD-first addition. The `ar_best<0.3` guard correctly returned INCONCLUSIVE
  rather than falsely claiming FUNDAMENTAL.

Root cause is **diagnosed and the harness `n_train` is fixed to 40000** (the
regime where AR reaches 0.84) and a post-train checkpoint was added. The
corrected **v3** (delete the stale checkpoint, rerun ~100 min on a free GPU)
settles artifact-vs-fundamental. This is the depth anchor of `.346 and satisfies
the Failed-Experiment Rerun Discipline cleanly (named prior failures + diagnosed
root cause + a positive-control gate `ar_best≥0.3`).

> Note: energy-as-generator remains BOUNDED from part-b **regardless** of the v3
> outcome. P1 only sharpens the *mechanism* (artifact vs fundamental); it does
> NOT reopen the strategic conclusion or seed a new paradigm.

### The strategic frame (P3) — the loop cannot self-seed
A nascent paradigm starts in a "valley of disappointment"; the loop's
adversarial verifier reads "higher error → prune" and auto-reconciles it as a
dead-end (the **Verification Trap**). So: **human = epistemic director** (seeds
frames, triages anomalies); **loop = apex postdoc** (exhaustive bounding,
falsification, frictionless execution). The one cheap endorsed upgrade is
**Anomaly-Escalation** — stop auto-reconciling EVERY negative; distinguish a
clean bounded negative (auto-reconcile) from a frame-VIOLATING anomaly (halt
pruning, escalate to human). `.346 prototypes it as a standalone classifier +
change-proposal (never auto-relaxing verification — that conflicts with the
anti-fabrication discipline; valley-funding stays human-gated).

---

## 2. The three biggest gaps between current state and the PRD vision

1. **The energy-foundation mechanism is not yet theorem-closed.** Both routes
   are empirically bounded, but P1 (artifact-vs-fundamental) is the missing
   mechanistic adjudication that either writes the closing theorem (Causal
   Inductive Bias Gap is fundamental) or reopens a narrow decode-fix. → `exp3777`.
2. **The banked verifier product is an artifact, not yet a deployable surface.**
   The certified abstention operating point exists as a number (`exp3771`) but
   is not wired into the verify API as an opt-in feature an integrator can call.
   → `exp3779`.
3. **The next Phase-3 thesis has no lowered-cost seeding surface.** The operator
   must seed it (the loop cannot), and the top menu route (EDLM — energy as a
   residual-corrector over discrete diffusion) has no feasibility scoping. →
   `exp3781`.

---

## 3. Milestone shape (11 tasks, 5 phases)

| # | Task | Phase | Agent | Why it is NOT churn |
|---|------|-------|-------|----------------------|
| exp3776 | Archive `.345 / activate `.346 | Activate | codex | routine transition |
| exp3777 | **P1 discrete-search adjudication v3** (valid positive control) | Settle-the-science | claude/opus, GPU | the last open energy-existential mechanism; rerun discipline satisfied |
| exp3778 | FR-11 v18 — Tier-2 constraint-memory consolidation on the live verifier | Product+self-learning | codex | NEW tier (v17 was Tier-1); the self-learning mandate |
| exp3779 | Wire the certified abstention point into the verify API (opt-in feature + E2E) | Product+self-learning | codex | NEW deployable product surface, not a re-measure |
| exp3780 | **Anomaly-Escalation** classifier prototype + change-proposal | Process (P3) | codex | NEW process capability the DT round endorsed |
| exp3781 | EDLM next-thesis **feasibility scoping** (operator seeding scaffold) | Scaffold | claude | lowers operator seeding cost; NOT a loop commitment |
| exp3782 | Technical-report G4 correction PREP (operator proposal; no curated edit) | Record | codex | the standing north-star §1 OPERATOR ACTION, prepared |
| exp3783 | External research refresh (file the new moat/PRM/entanglement papers) | Record | codex | genuinely-new 2026 papers found in planning |
| exp3784 | KV260 opportunistic continuity audit | Record | codex | opportunistic hardware mandate (north-star §3) |
| exp3785 | Capstone `.346 | Record | codex | routine aggregation |

### Dependency graph
All tasks read upstream via **graceful disk-presence fallback** (the `.340
proven-safe pattern — never crash on a missing field, never label un-run as a
negative). **No hard `gated_on`** this milestone (keeps the run robust; the
capstone reads each artifact's presence and records honestly). `exp3777` is the
one GPU task; if the conductor's GPU-zombie reaper or a competing GPU job
interferes, it emits `blocked_*` honestly and the operator runs the corrected
harness directly (the v1/v2 pattern).

### Routing reality (unchanged since `.343–`.345)
Gemini CRASHES real GPU workloads and has wiped milestones via quota 429s
(`[[incident_333_gemini_quota_crash_wipeout]]`), so the cheap-default is CODEX
(`requires_codex`). The two judgment-heavy tasks are CLAUDE: `exp3777` (the P1
adjudication — open-ended mechanistic judgment + GPU/bootstrap risk → `opus`,
`max_turns: 100`) and `exp3781` (EDLM feasibility synthesis — cross-paper
judgment). Every Run command pins `.venv/bin/python`
(`[[incident_ebt_training_venv_python_cuda]]`: bare `python` is
`/usr/bin/python3` with no torch).

### Invariants re-asserted at capstone
- `paper_ready` stays **TRUE** (G1–G4).
- FoVer **0.9131 frozen** — `.346 reproduces/uses it, never moves it.
- Both energy routes stay **bounded** — P1 v3 sharpens the mechanism only; no
  re-grind, no new existential claim.
- The next-Phase-3-thesis decision stays an **operator-seeding surface** — the
  loop scaffolds (EDLM feasibility) but does NOT commit.
- The agent never edits operator-curated docs, never triggers CI, never
  publishes, never pushes.

### Anti-poison-test discipline (the `.344 root cause, carried forward)
Any shipped test MUST assert against the script's REAL behavior, and any new
`research-complete.yaml` value containing a colon MUST be quoted. A failing
shipped test — or a stray unquoted colon — poisons the conductor pre-test gate
and SKIPs the whole milestone. This remains the single most load-bearing
discipline for `.346 not repeating `.344.

---

## 4. Hardware requirements

- `exp3777`: one free CUDA GPU (RTX 3090), ~100 min, from-scratch tiny-EBT +
  matched-AR training (`n_train=40000`). Precondition-gated; blocks honestly if
  no free GPU.
- All other tasks: CPU only (verifier-scoring against cached candidates, or
  aggregation), plus `exp3784` SSH to the KV260.

## 5. New references filed this sweep (see research-references.md `.346 additions)
- arXiv:2604.07650 — auditing behavioral entanglement + de-entangled verifier
  ensemble reweighting (the error-independence methodology; corroborates the
  closed moat thread).
- arXiv:2506.07962 — Correlated Errors in LLMs (more accurate models → more
  correlated errors; the subsumption mechanism, operationalized).
- arXiv:2601.17223 — Verifiable Process Reward Models (deterministic rule-based
  step verifiers; corroborates Carnot's objective-energy positioning).
- arXiv:2604.15149 — LLMs Gaming Verifiers (ICLR 2026 workshop; the
  null-space-mimicry / verifier-robustness frontier).
- arXiv:2502.11157 — Dyve, fast/slow dynamic process verification (the
  fast/slow escalation precedent for the Anomaly-Escalation prototype).
