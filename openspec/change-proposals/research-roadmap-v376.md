# Research Roadmap v376 — FIX the powering mechanism so the operator's TOP-PRIORITY off-ARC verifier transfer + the G3 sovereign-base finally become measurable; re-run the cascade-lost efficiency / 9th-game / cross-game-transfer tasks

**Milestone:** 2026.06.376
**Planned:** 2026-06-11 (planning agent, Claude Opus 4.8)
**Prior:** 2026.06.375 (openspec/change-proposals/research-roadmap-v375.md)
**North star:** solve ARC-AGI-3 accurately AND efficiently (ops/north-star.md §0).
The verifier is the project's existential value-add (§5); these experiments
prove it transfers (off-ARC), scales sovereignly (local base), and makes the
agent efficient (action-pruner).

---

## 1. What .375 measured (the honest read, from the artifacts not the prose)

.375's job was to make G1 (off-ARC transfer on an un-saturated corpus) and G3
(sovereign base) decision-grade via the operator's resume-not-restart directive,
and to pivot the banked-G2 effort into the verifier-as-action-pruner EFFICIENCY
axis. **It largely failed — not on the science, on the MECHANISM.** Capstone
exp4065 verdict: `offarc_accumulating_n0_g3_accumulating_efficiency_null_games8`.

| Track | .375 task | Outcome | Why |
|---|---|---|---|
| **G1 off-ARC power** (operator TOP PRIORITY) | exp4056 BUILD / exp4057 COLLECT | **accumulated N = 0** | exp4056 `blocked_smoke_failed`: smoke errored, `smoke_oracle_headroom_present:false`, `launched_pid:0` → the background run **never launched**. FLAGGED DURATION_TOO_SHORT. |
| **G3 decentralization MoE** | exp4058 BUILD / exp4059 COLLECT | **gate-blocked, 0 new** | exp4058 codex **1202s idle-timeout** → SKIP; it shipped a failing pre-test → cascade. exp4059 GATE_BLOCK (upstream "retired"). |
| **EFFICIENCY** (verifier action-pruner) | exp4061 | **SKIPPED** | poison-test cascade from exp4058. |
| **ACCURACY** (9th game) | exp4060 | **SKIPPED** → still 8 games | poison-test cascade from exp4058. |
| **SELF-LEARNING** (ArcMemo v8) | exp4062 | `no_cross_game_transfer` | no usable 9th-game trace (exp4060 was skipped) — non-result. |
| **HARDWARE** | exp4064 | OK | GateMate/PolarFire continuity recorded; KV260 terminal. |

**The diagnosis (load-bearing for this milestone).** This is the **third**
consecutive milestone (.373, .374, .375) the off-ARC powering run produced no
useful N, and the **second** the poison-test cascade destroyed downstream tasks.
The root cause is NOT corpus saturation alone and NOT throughput alone — it is
the **split-BUILD-LAUNCH-backgrounded + COLLECT-poll mechanism itself**:

1. The detached background run does **not survive** the build-agent exit + the
   conductor iteration boundary (`launched_pid:0`; no `*.log` files remain).
2. The BUILD's pre-launch smoke **gate-blocks** the whole measurement when the
   smoke errors (a saturated/erroring smoke should *route to a harder corpus*,
   not block).
3. A long BUILD prompt **hangs codex** (1202s idle-timeout) and the half-written
   task ships a **failing pre-test** that cascade-SKIPs every downstream task.

**The data is intact.** `results/experiment_4045_offarc_transfer_power.checkpoint.json`
is 238 KB of already-generated candidate programs (22 tasks);
`results/experiment_4048_decentralization_moe_base_raw.checkpoint.json` is 14
MoE-scored tasks. Nothing was lost. The measurements are *one robust runner
away*. .376 supplies that runner.

---

## 2. The three biggest gaps (current state → north-star vision)

1. **The verifier has NEVER been measured off-ARC** (operator TOP PRIORITY,
   stuck 3 milestones). `ops/verifier_gaps.md` is 100% ARC-grid; "the GAP-4
   demo-fit primitive is domain-general" is ARGUED, not MEASURED. This is the
   cleanest possible "the verifier earns its place in MORE than one domain"
   datum (north-star §0 step 2). **Gap: a working powering run on an
   un-saturated code corpus.**

2. **The sovereign base is unmeasured** (G3). Decentralization rule 1 needs a
   LOCAL open-weight model that can induce ARC rules; gemma-4-12B cannot
   (0.2581, exp4012). Whether the MoE Qwen3.6-35B-A3B raises the ceiling
   (latent → distillation viable) or not (absent → the Invisible Leash holds)
   decides the sovereign path. **Gap: the same broken powering mechanism.**

3. **The north-star EFFICIENCY axis is unmeasured.** ARC-AGI-3 is "accurate AND
   efficient"; the verifier-as-action-pruner is the agentic proof that the
   verifier makes the harness efficient (Exp1165 pilot: ~4× fewer actions).
   exp4061 was cascade-SKIPPED. **Gap: a clean WITH-vs-WITHOUT-pruning ablation
   on solved games.**

**The cross-cutting infrastructure gap** behind all three: the powering-run
mechanism. .376's central move is to **replace the fragile split-background-poll
mechanism with a single SYNCHRONOUS bounded-batch resume-accumulate task** that
cannot fail to launch, cannot ship a poison test, and cannot hang codex.

---

## 3. The mechanism fix (the heart of .376)

Every large-N local-GGUF powering run in .376 is a **single self-contained task**
(no BUILD/COLLECT split, no `setsid` detachment, no PID handoff, no poll, no
pre-launch smoke-gate). The shape:

```
1. RESUME the STABLE (corpus+model+k)-keyed checkpoint  → accumulated_n so far
2. HEADROOM-ROUTE (not gate): probe oracle headroom on ~8 tasks; if the corpus is
   saturated for THIS model, SWITCH to a harder corpus (EvalPlus → LiveCodeBench
   v6) — route, never block.
3. RUN a BOUNDED batch synchronously: as many NEW tasks as fit a ~3000 s (50 min)
   self-imposed budget, well under the 80-min wall cap. Re-score the already-
   generated 238 KB / 14-task candidate pools (cheap; no regeneration) AND extend
   to new tasks. CHECKPOINT after EACH task. PRINT a progress line after EACH task
   so codex's 1201 s idle-timeout NEVER fires (this is the specific fix for the
   exp4058 1202 s kill).
4. COMPUTE the bootstrap CI on the ACCUMULATED sample and WRITE the terminal
   artifact reporting accumulated-N. Always finishes within budget → always emits
   a terminal verdict.
```

Why this fixes the 3 root causes: (1) synchronous → no background process to be
reaped; (2) headroom-route → a saturated/erroring smoke escalates the corpus
instead of blocking; (3) per-task progress + self-bounded budget → codex never
idle-times-out and the artifact is always written. N accumulates one bounded
batch per milestone (22→~55→… for off-ARC; 14→~22→30 for MoE) — the operator's
resume-not-restart intent, made robust. The stable checkpoint (the load-bearing
part of the operator's directive) is preserved verbatim; only the fragile
execution shell is replaced.

---

## 4. Architecture — where each .376 experiment sits

```
                    NORTH STAR: ARC-AGI-3, accurate + efficient
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │ VERIFIER value-add         │ GENERATOR (commodity/local) │ AGENT
        │                            │                             │
   off-ARC transfer            sovereign base               action-pruner
   exp4068 (G1, TOP)           exp4069 (G3)                 exp4071 (EFFICIENCY)
   demo-fit primitive →        MoE Qwen3.6-35B-A3B →        GAP-4 verifier prunes
   EvalPlus/LiveCodeBench      can a LOCAL model            actions online during
   hidden-test transfer,       induce ARC rules?            explore-first; actions
   CI excl 0 with headroom     latent/absent vs 0.2581      + wall-clock @ equal
        │                            │                      solve-rate
        └──────────── all reuse the SAME GAP-4 demo-fit / content-hash primitive ──┘
                                     │
   ACCURACY: exp4070 (9th game, explore-first, +1 monotonic)
   SELF-LEARNING: exp4072 (ArcMemo v9 richer cross-game library)
   INFRA: exp4066 (archive+green-gate) · exp4073 (registry/gaps hygiene)
   HARDWARE: exp4074 (GateMate/PolarFire→terminal; KV260 done)
   INGEST: exp4067 (SOTA slot) · CAPSTONE: exp4075
```

The off-ARC code transfer (exp4068), the sovereign base (exp4069), and the
action-pruner (exp4071) ALL exercise the same execution-consistency primitive
(`python/carnot/agentic/arc_gap4_execution_verifier.py` + the
`python/carnot/verify/sandbox.py` restricted executor) — the point is that one
verifier serves verification, generation-selection, and action-pruning.

---

## 5. Phases & dependency graph

**Phase 0 — transition + ingestion (infra; reserved slots).**
- exp4066 archive .375 → activate .376; keep the hardened green-gate +
  poison-test quarantine; record the .375 close-state honestly (G1 N=0, G3
  gate-blocked, efficiency/9th cascade-SKIPPED). [claude/opus]
- exp4067 SOTA-ingestion slot (mandatory; corpus is fresh → light pass). [codex]

**Phase 1 — the two stuck powering runs, RE-ARCHITECTED (the headline).**
- exp4068 off-ARC transfer power — SINGLE synchronous resume-accumulate +
  headroom-route (EvalPlus → LiveCodeBench v6). operator TOP PRIORITY. [codex]
- exp4069 sovereign-base MoE — SINGLE synchronous resume-accumulate toward
  N≥30. [codex]

**Phase 2 — ARC north-star (accuracy + efficiency + self-learning).**
- exp4070 9th-game first-solve (explore-first, +1 monotonic). [codex]
- exp4071 verifier-as-action-pruner efficiency (the G2 pivot). [codex]
- exp4072 ArcMemo v9 richer cross-game transfer (self-learning mandate). [codex]

**Phase 3 — hygiene / hardware / capstone.**
- exp4073 verifier-registry + gaps hygiene (reserved infra slot 2). [codex]
- exp4074 hardware continuity (GateMate + PolarFire). [codex]
- exp4075 capstone .376. [codex]

**Dependencies: NONE are hard gates.** Deliberately. .375 lost 4 tasks to a
gate/poison cascade; .376 uses ZERO `gated_on` chains. Every task is
self-contained and reads upstream artifacts *defensively* (exp4072 reads
exp4070's trace if present, else measures transfer on the attempt; exp4073/4075
aggregate whatever exists). No task can cascade-block another.

---

## 6. Hardware requirements

- exp4068/4069: 1× local GGUF via llama.cpp (gemma-4-12B-it-GGUF for off-ARC;
  Qwen3.6-35B-A3B-GGUF MoE ~3B-active for sovereign-base). CPU/RTX-3090 either.
  Both RESUME existing on-disk checkpoints — most generation cost is already paid.
- exp4070/4071: ARC-AGI-3 SDK (anonymous key auto-issued; 25 live envs; offline
  driver). No GPU.
- exp4074: GateMate (`openFPGALoader -c dirtyJtag --detect`) + PolarFire
  (`ssh polarfire`). KV260 is TERMINAL (opportunistic confirm only). SSH/USB
  preconditions ONLY (KV260 SSH-Not-SD-Card Discipline).

---

## 7. Routing & discipline

- **Codex-Default v2 (gemini BANNED):** every experiment task is `codex` /
  `gpt-5.5`. Only exp4066 (archive — multi-step infra coordination) is
  `claude` / `opus` with `requires_claude_verified: true`.
- **The codex idle-timeout fix is in the DESIGN:** the synchronous runners print
  a per-task progress line so the 1201 s idle-timeout never fires (the specific
  cause of the exp4058 1202 s kill). The runner self-bounds to ~50 min so the
  terminal artifact is always written before the 80-min wall.
- **No `gated_on` anywhere** (anti-cascade; see §5).
- **operator_override on every task:** the standing 2026-05-29 auto-override
  classes (milestone-transition / hardware-continuity / versioned-lineage) +
  the explicit 2026-06-11 resume-not-restart + off-ARC TOP-PRIORITY directives +
  the 2026-06-06 agentic-proof directive. A single override clears both the
  exclusion-manifest and doomed-rerun guards (CLAUDE.md).
- **Resume-not-restart:** no premature retire on a throughput-truncated window;
  `retire_if_same_verdict` fires ONLY at accumulated-N ≥ target AND gate-still-fails.
- **Adversarial rigor:** every powering artifact carries `model_specs`,
  `random_seed`, `reproducibility_checksum`; a positive control (oracle headroom)
  guards every null (FALSE_NEGATIVE_RISK); bare gated/headline fields.
- **SOTA GGUF mandate:** gemma-4-12B (lightweight SOTA, approved) + Qwen3.6-35B-A3B
  (flagship MoE). Reserved infra slots (≥2): exp4066, exp4073. Self-learning:
  exp4072. SOTA-ingestion slot: exp4067. Hardware continuity per non-terminal
  board: exp4074.

---

## 8. Acceptance — what makes .376 a win

.376 is a win if the powering mechanism is FIXED and at least the operator's
TOP-PRIORITY question advances from "non-measurement" to "measured-with-an-N":

1. **G1 (TOP PRIORITY):** the off-ARC runner RESUMES the 238 KB pool, picks a
   corpus with real 12B oracle headroom (EvalPlus or LiveCodeBench v6), and
   reports an accumulated-N with a bootstrap CI — `accumulated_n` strictly > the
   .375 value of 0, ideally well toward N≥160. Whether the demo-fit CI excludes
   zero is the science; that it PRODUCES a measurement is the milestone win.
2. **G3:** the MoE runner reports accumulated coverage at accumulated-N (14 →
   toward 30) vs the 0.2581 ceiling, with a positive control — latent / absent /
   accumulating, no false retire.
3. **EFFICIENCY:** a clean verifier-pruner WITH-vs-WITHOUT ablation on solved
   games, solve-rate parity held (positive control), honest action/wall-clock delta.
4. **ACCURACY:** total_games_solved advances 8 → 9 (or an honest no-solve).
5. **No cascade:** every task writes a terminal artifact; zero poison-test SKIPs.

An honest negative, an "uninformative/saturated→escalated" outcome, or a "still
accumulating, here is accumulated-N" outcome are all real results — the failure
mode this milestone fixes is producing NO measurement at all.

---

## 9. Cross-references

- ops/north-star.md §0 (ARC-AGI-3), §5 (energy verifies / refinement generates)
- ops/known-issues.md 2026-06-11 resume-not-restart + off-ARC TOP-PRIORITY entries
- research-roadmap-v375.md (the mechanism that failed) + exp4065 capstone
- results/experiment_4045_offarc_transfer_power.checkpoint.json (238 KB pool to resume)
- results/experiment_4048_decentralization_moe_base_raw.checkpoint.json (14-task MoE pool)
- results/experiment_4012_gap4_local_best_of_n.json (the 0.2581 baseline + 30-task pool)
- python/carnot/agentic/arc_gap4_execution_verifier.py + python/carnot/verify/sandbox.py
- docs/research-notes/sota-ingestion-2026-06-11-unsaturated-execverif-and-verifier-pruner.md
- EvalPlus (arXiv:2305.01210) · LiveCodeBench v6 (arXiv:2403.07974) · ACES
  (arXiv:2604.03922) · symbolic-equivalence partition (arXiv:2604.06485) ·
  online verification (arXiv:2602.01070) · update-free verifier steering
  (arXiv:2603.10282) · The Invisible Leash (arXiv:2507.14843)
