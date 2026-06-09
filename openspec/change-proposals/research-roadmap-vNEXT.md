# Research Roadmap — Milestone 2026.06.368

**Planned:** 2026-06-09 (outer-loop planning agent, Claude Opus 4.8)
**Milestone doc for:** `research-roadmap-next.yaml` (`milestone: 2026.06.368`)
**Prior milestone:** 2026.06.367
**North star:** `ops/north-star.md` §0 — solve ARC-AGI-3, accurately AND efficiently.

---

## 0. One-line thesis

Build and prove the **execution / program-synthesis verifier (GAP-4)** — the single
direction BOTH retired ARC verifier threads independently converged on — and use it to
answer the project's existential question on BOTH owed axes: ACCURACY (executed-rule
consistency selects gold past the generator's own vote, reaching the proven 16pp headroom
the retired content-energy lineage could not) and EFFICIENCY (the energy verifier at parity
with an LLM-judge for 10–100× less cost). Plus: grow ARC-AGI-3 accuracy via execution-guided
re-induction (the L2-fails fix + a 4th game), continue self-learning (ArcMemo solve-transfer),
hardware continuity.

---

## 1. What the previous milestones proved (the convergence)

Two distinct ARC verifier programs ran to ground on 2026-06-09. They failed for the SAME
reason and point at the SAME next step.

### Thread A — the ARC-AGI-3 interactive agent (.365/.366/.367)

| Result | Status | Artifact |
|---|---|---|
| **THREE games solved** (r11l, lp85, sc25), each at level 1 | ✅ method generalizes (2 = coincidence, 3 = method) | exp3946 / exp3954 / exp3966 |
| **Incremental level progress** (L1→L2) | ❌ r11l + lp85 both stop at L2 (`first_fail2`) — the induced L1 mechanic does not transfer to the harder config | exp3964 / exp3965 |
| **M3 "verifier earns its place" efficiency** | ❌ FABRICATED .366 (exp3959, flagged), HONESTLY BLOCKED .367 (exp3967, `blocked_verifier_not_in_loop`) — **unanswered for a 2nd milestone** | exp3959 / exp3967 |
| **World-model induction across the 6 non-spatial games** | ❌ 0/6 trustworthy (consistency energy ≤0.15); vc33's 0.005 is a one-off | exp3968 |
| **Hidden-state recovery** (Pinductor belief-likelihood) | ❌ no energy drop (2nd negative) → latent-augmentation hypothesis RETIRED | exp3969 |
| **Cross-game self-learning** (ArcMemo NL concept memory) | ✅ transfer win — reused 2 concepts at lower induction cost | exp3970 |
| **Offline quota-gate** | ✅ CLEARED — hybrid 3 levels vs 0 baselines → operator MAY run an online scored game | exp3971 |

### Thread B — the GAP-3 TRM-candidate-rerank verifier (offline ARC-AGI-1 pool)

The trained-content-energy ARC selector lineage RETIRED in full (Stages 0/1/2v1/2v2 all
NEGATIVE, adversarially confirmed 5/5; on `ops/exclusion_manifest.yaml`). The honest bound:

> The ~16pp oracle headroom (oracle pass@2 **0.6129** vs frequency-vote **0.4516** on the
> 31-task headroom pool) is REAL but UNREACHED by scalar (q_halt), latent (z_H probe), or
> trained-content-energy (v1+v2) selectors. They master what they train on but score AUROC
> 0.43–0.50 on the **dominant real-error class — same-shape, plausible-but-wrong rule
> applications** (59.1% of errors, 81.4% of wrong-pair mass), where vote scores 0.92–0.98.

The distilled missing-verifier spec (`ops/verifier_gaps.md` **GAP-4**, priority HIGH):

> **missing discriminator:** *does the candidate output follow from applying the task's
> induced rule to the test input* — "is this the right transformation," not "is this grid
> damaged." **candidate design:** execution / program-synthesis verification — induce the
> rule as a program from the demo pairs, execute it on the test input, compare to the
> candidate. *Synthesizing the missing negative class IS program synthesis — which is why no
> cheaper energy can fake it.*

### The convergence (why this milestone exists)

Both threads independently conclude the next verifier must be **execution-based**:
- Thread B (GAP-4): induce the rule as a program, execute it, compare → reach the 16pp headroom.
- Thread A (M2/M3): the induced world-model already IS a program; the consistency-energy
  already verifies it with no oracle (it caught codex's overfit programs). The in-house
  M2-v3/v4 codex+consistency-energy stack is the named in-house precedent for GAP-4.

External literature (2025–2026) corroborates this as the dominant, OOD-robust ARC technique:
- **arXiv:2507.15877** (Ouellette) — execution-guided neural program synthesis OUTPERFORMS all
  references at composing NOVEL solutions; test-time fine-tuning only elicits in-distribution
  knowledge and does NOT generalize → execution-guidance is the right tool for a harder level
  or a new game, NOT re-fitting.
- **arXiv:2603.20334** (ABPR) — candidate programs as "executable declarative hypotheses of the
  latent rule"; proof-trees expose WHY a rule fails → targeted semantic refinement; GPT-5.5
  98.33% Pass@2 on ARC-AGI-2 public eval.
- **arXiv:2605.05138** (EWM, ARC-AGI-3 SOTA RHAE 58.12%) — coding-agent maintains an executable
  world model, verifies it against transitions, plans before acting.
- **arXiv:2603.10282** (Yilun Du) — verifier-as-EBM scores/steers a frozen policy with no
  parameter updates (the update-free in-the-loop verifier formulation).

---

## 2. The three biggest gaps (current state vs PRD / north-star)

1. **The verifier's existential proof is UNPROVEN — on BOTH axes.** North-star §5: with the
   generator commodity, the verifier is Carnot's entire value-add and its value is owed on two
   axes — ACCURACY (does the external verifier beat the generator's self-verification?) and
   EFFICIENCY (parity at 10–100× cheaper than an LLM-judge?). The ACCURACY moat is inconclusive;
   the EFFICIENCY proof failed/blocked twice. This milestone proves both with the GAP-4
   execution verifier as the common primitive.
2. **ARC-AGI-3 accuracy cannot grow.** Every solved game stops at level 2 because the agent
   re-fits the L1 mechanic instead of RE-INDUCING the harder L2 rule — the exact failure mode
   arXiv:2507.15877 predicts for non-execution-guided adaptation. Accuracy is frozen at 3×L1.
3. **The execution/program-synthesis verifier (GAP-4) is unbuilt.** It is Carnot's #1
   core-product backlog item (`ops/verifier_gaps.md`, HIGH) and the convergent conclusion of
   both retired threads — yet no experiment has built it.

---

## 3. Architecture — the GAP-4 execution verifier (generator induces, verifier executes)

```
                       demo pairs (input_i -> output_i)
                                  │
              ┌───────────────────┴────────────────────┐
              │  GENERATOR (induction; commodity/local) │   ← NOT energy-descent
              │  local grid-DSL  +  SOTA local GGUF      │     (closed-negative)
              │  proposer (gemma-4)  [+ codex optional]  │
              └───────────────────┬────────────────────┘
                                  │  rule as an EXECUTABLE PROGRAM  P
                                  ▼
                          P( test_input )  =  predicted_output*
                                  │
   candidate grids ──────────────┤
   (TRM pool / ARC-AGI-3          ▼
    action outcomes)   ┌──────────────────────────────────────┐
                       │  VERIFIER (Carnot's value-add)        │
                       │  executed-rule-consistency:           │
                       │  E(cand) = disagreement(cand, P(test))│  ← execution, NOT trained
                       │  + held-out consistency_energy(P)     │     energy (retired lineage)
                       └──────────────────┬───────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                            ▼                           ▼
    ACCURACY (Phase 1)          EFFICIENCY (Phase 2)        ARC-AGI-3 accuracy (Phase 3)
    select gold past vote        verifier vs LLM-judge:      EG re-induction per level;
    toward oracle (16pp)         parity at 10-100x cheaper   4th game first-solve
```

**On-thesis (north-star §5):** the program synthesizer is the GENERATOR (commodity/local);
Carnot's contribution is the **executed-consistency VERIFIER** that scores candidates. This is
NOT energy-as-generator (closed-negative) and NOT a trained content-energy (retired lineage) —
it is execution-based verification, the endorsed remaining candidate.

**Decentralization (Rule 1):** the headline synthesizer is the LOCAL grid-DSL + a SOTA local
GGUF proposer (`unsloth/gemma-4-26B-A4B-it-GGUF`); codex is an optional stronger comparator arm
only. The accuracy/efficiency headline must be reproducible local-only.

---

## 4. Phases & experiments (12 tasks)

**Phase 0 — Activation**
- **exp3974** archive .367 → activate .368; green-gate (ARC substrate tests + agentic-module
  imports + YAML parse); record the .367 truth.

**Phase 1 — GAP-4 verifier, the ACCURACY moat**
- **exp3975** BUILD the GAP-4 executed-rule-consistency verifier for ARC-1 static grids; positive
  control + program-synthesis COVERAGE on held-out ARC-1 training tasks. *opus.*
- **exp3976** EVALUATE on the TRM-rerank headroom pool: executed-consistency selector +
  vote-primary-hybrid-gated-by-consistency vs vote (0.45) → oracle (0.61). *opus; gated on
  exp3975 positive control.*
- **exp3977** RE-DERIVATION / independence audit of any exp3976 positive (CPU re-score, leak +
  coverage forensics). *codex; gated on exp3976 beating vote.*

**Phase 2 — verifier EFFICIENCY (north-star §5 owed head-to-head)**
- **exp3978** energy-consistency-VERIFIER vs SOTA-local-LLM-as-JUDGE for the induced-world-model
  acceptance decision: accuracy parity AND cost ratio (target "parity at 10–100× cheaper");
  anti-fabrication token/second audit. *opus; replaces the unwireable env-step action-pruner M3.*

**Phase 3 — ARC-AGI-3 accuracy (execution-guided)**
- **exp3979** world-model induction generalization via EXECUTION-GUIDED synthesis (the 0/6 fix).
- **exp3980** INCREMENTAL levels via execution-guided RE-INDUCTION (the L2-fails fix); +1 level.
- **exp3981** FOURTH game first-solve (tn36 / su15 / dc22). *opus.*

**Phase 4 — self-learning + hardware + infra + capstone**
- **exp3982** ArcMemo SOLVE-transfer (self-learning mandate): does concept memory make the SOLVE
  cheaper, not just induction?
- **exp3983** Hardware continuity (KV260 drive-to-terminal; GateMate/PolarFire opportunistic).
- **exp3984** Operational-retro commit-detector fix (4-milestone false-zero-count infra bug).
- **exp3985** Capstone .368 (UNGATED): did the verifier earn its place on EITHER axis?

---

## 5. Dependency graph

```
exp3974 (activate)
   ├─► exp3975 (GAP-4 build) ─► exp3976 (GAP-4 eval, gated) ─► exp3977 (re-derive, gated on +)
   ├─► exp3978 (verifier-vs-judge efficiency)          [uses exp3968 induced models]
   ├─► exp3979 (world-model gen via EG-synthesis)      [retries exp3968]
   ├─► exp3980 (incremental levels via EG re-induction) [continues exp3964/3965]
   ├─► exp3981 (4th game solve) ─► exp3982 (ArcMemo solve-transfer)
   ├─► exp3983 (hardware), exp3984 (retro fix)
   └─► exp3985 (capstone, UNGATED — aggregates whatever landed)
```

Two structured gates only: exp3976←exp3975 (positive control), exp3977←exp3976 (beats vote).
Everything else runs independently; the capstone is UNGATED (the .365 `op:exists` and .366
no-artifact lessons).

## 6. Hardware requirements

- **GPU:** none required for the headline GAP-4 / efficiency tasks (CPU re-scoring of the saved
  8041-candidate pool). The SOTA local GGUF proposer (gemma-4-26B-A4B-it) runs on the RTX 3090
  rig when invoked; precondition the cache first.
- **Boards (continuity only):** KV260 (`ssh kria`), GateMate (`openFPGALoader --detect`),
  PolarFire (`ssh polarfire`). KV260 = sovereignty story → drive to terminal then freeze
  (north-star §3); the other two opportunistic. NEVER use a host SD-card device as a KV260
  precondition (SSH-reachability only).

## 7. Disciplines honored

Incremental-Progress Scoping (exp3980 +1 level, exp3981 first-solve — never "all levels") ·
Missing-Verifier Gap Logging (exp3975/3976 build against GAP-4; emit `missing_verifier_gaps`) ·
Failed-Experiment Rerun (prior_failures on exp3976/3978/3979; operator_override on routine
continuations) · Adversarial Artifact Verification + anti-fabrication self-audits on both
existential proofs (exp3976/3978) · Gemini banned → codex for mechanical + opus for the 4
anti-fabrication-critical / real-env tasks · SOTA local GGUF + local-first (decentralization
Rule 1) · Hardware-Task Continuity (exp3983) · Pre-Launch Preconditions (step-0 blocks) ·
Principle-annotated bare-scalar artifact fields · Verdict Terminal-Prefix.
