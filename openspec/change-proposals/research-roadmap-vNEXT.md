# Research Roadmap v359 — Execute the Three Forward Bets (Hardened)

**Milestone:** 2026.06.359
**Planned:** 2026-06-05 (outer-loop, Claude Opus 4.8)
**Supersedes draft:** identical copy committed as `research-roadmap-next.yaml`

**Title:** Execute the three forward bets — EBT energy-as-GENERATOR kill-gate
(import-fixed), the verifier moat on an IN-DISTRIBUTION corpus, and a
de-fabricated facts graph-grounding verifier — hardened against the
poison-test SKIP cascade.

**Status of headline:** `paper_ready = TRUE` (G1∧G2∧G3∧G4); FoVer headline
AUROC **0.9131** frozen and G2-reproduced on a clean CI runner. This milestone
adds **lenses** (generation, durability, factual breadth), not a new headline.

---

## 1. What the previous milestone actually proved (and didn't)

`.358 was designed correctly — the three operator-seeded forward bets are the
right questions — but it was an **EXECUTION wipeout**, not a research result.
Of 11 tasks, only 3 produced artifacts:

| Task | Fate | Root cause |
|---|---|---|
| exp3870 archive/activate | OK | — |
| exp3871 EBT part-b kill-gate | **blocked_scaled_harness_import** | `import scripts.thesis_a_part_b_scaled` → `ModuleNotFoundError`. The harness imports fine when `scripts/` is on `sys.path`. A **one-line bug**, not a research finding. |
| exp3872 EBT System-2 | GATE_BLOCK | hard `gated_on` exp3871.positive_control_passed==true (False because 3871 blocked) |
| exp3873 in-dist corpus | **SKIP** | poison pretest ("1 failed, 80 passed"), self-heal failed |
| exp3874 moat scissor | GATE_BLOCK | "upstream retired (exp3873)" — gated_on retire-on-skip cascade |
| exp3875 graph-grounding facts | **SKIP** | poison pretest |
| exp3876 facts complementarity | GATE_BLOCK | upstream retired (exp3875) |
| exp3877 FR-11 v24 | **SKIP** | poison pretest |
| exp3878 / exp3879 hardware | **SKIP** | poison pretest |
| exp3880 capstone | **SKIP** | poison pretest |

**Two dominant operational failure modes** (both recurring in user-memory
across `.350–`.358):

1. **Poison-test SKIP cascade.** A single failing test in the conductor's
   smart-subset pretest ("1 failed, 80 passed") fails the self-heal gate and
   SKIPs every subsequent task. The conductor already runs a task's own
   uncommitted tests (the exp3521/exp3544 fix), but `.358 still cascaded — the
   poison was transient (an agent-shipped or changed-source test) and cleared
   by session end (the core subset `test_pipeline_extract.py` + `test_docs.py`
   is **green now**: 81 passed).
2. **`gated_on` retire-on-skip.** When an upstream task SKIPs, a hard
   `gated_on` downstream is "pre-emptively skipped — upstream retired,"
   converting one poison into a whole-chain cascade. The `.340/`.346
   **proven-safe pattern is "no hard gated_on" + in-script disk-read-with-
   fallback**.

**What survived from `.356/`.357 (the real research state):**

- **EBT (Q1):** the scaled checkpoint cleanly measured **AR=0.84,
  EBT-greedy-argmin=0.0** (`results/thesis_a_part_b_scaled_seed1.json`) —
  headroom exists. The decisive probe (global discrete beam search over
  cumulative EBT energy → ARTIFACT vs FUNDAMENTAL) **has never run.**
- **Moat (Q2):** exp3869 ran but was **INCONCLUSIVE** — PRMBench is
  out-of-distribution for the math-bound ensemble (carnot AUROC=0.55, error
  overlap Jaccard=0.0). Needs an **in-distribution** error-rich corpus.
- **Facts (Q3):** exp3862 reported a graph-grounding **signal** (facts AUROC
  0.643 vs math-baseline 0.411, delta 0.232) but was **fabrication-flagged**
  (1.02s, rule-based stub — the model was never invoked). Needs a real, ≥60s,
  per-item-scored run.
- **FR-11:** v23 (exp3864) landed clean — learned ensemble AUROC 0.9059 (in
  frozen CI), +0.0185 memory-ablation contribution preserved, state persisted.
- **Hardware:** PolarFire terminal hash-verified dispatch is clean (exp3867,
  soft-CPU, no fabric claim). GateMate n=16 tile flashed (fmax 139.12 MHz) but
  **TAUTOLOGY-flagged** (duration_s bit-identical to run_duration_s).

---

## 2. The three biggest gaps between current state and PRD vision

1. **Phase-3 foundation-model bet is untested for GENERATION.** The PRD endgame
   (Phase 3) is an energy-based foundation model with non-autoregressive
   reasoning. Carnot has only ever used energy for *selection/verification*
   (P0.1 settled: energy-selection does **not** beat AR/SC). The
   energy-as-**generator** question (EBT) is the live Phase-3 kill-gate, blocked
   by infrastructure, not science. Closing it (ARTIFACT or FUNDAMENTAL) is the
   highest-leverage move available. **Peer: NRGPT (arXiv:2512.16762)** decodes
   language by energy-landscape descent but never reports a compute-matched
   comparison vs AR — exactly Carnot's open question.

2. **The verifier moat is unproven at scale on an in-band corpus.** The
   project's credibility thesis (PRD Tier A "LLM Output Verification") rests on
   Carnot catching errors a strong frontier reasoner's own self-verification
   misses (DT-P2 o1-subsumption durability). The literature now strongly
   supports the *premise* — **Self-Verification Dilemma (arXiv:2602.03485)**
   shows 85–95% of a reasoner's self-rechecks never catch an error — but
   Carnot's own measurement has only ever run OOD (PRMBench). The
   in-distribution scissor is well-scoped and just needs a clean execution.

3. **The verifier is math-domain-bound; facts is the only broadening route
   with signal.** PRD Tier C "Factual Grounding Gate" needs the verifier to
   work beyond math. The verifier is earned-negative on facts, and the ONE
   new-architecture route with traction is graph-grounding (MemGraphRAG, and
   now the near-exact peer **HalluGraph arXiv:2512.01659** with an explicit
   Entity-Grounding + Relation-Preservation decomposition). The exp3862 signal
   must be de-fabricated before it is bankable.

---

## 3. Milestone architecture

`.359 is a **hardened re-issue** of `.358's forward bets. The science is
unchanged; the engineering is fixed.

```
 PHASE 0 (hygiene)   exp3881  archive .358 → activate .359
                              • research-complete.yaml parses (colon-poison guard)
                              • core pretest GREEN
                              • EBT scaled harness IMPORTS via sys.path (the .358 blocker)
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                               ▼
 PHASE 1 (Q1 EBT)          PHASE 2 (Q2 moat)            PHASE 3 (Q3 facts)
 exp3882 part-b kill-gate  exp3884 in-dist corpus       exp3886 graph verifier
   IMPORT-FIXED [GPU]        (ensemble AUROC≥0.65)         DE-FABRICATED [GPU]
   ARTIFACT/FUNDAMENTAL    exp3885 moat scissor          exp3887 facts
 exp3883 System-2 K-curve    IN-DISTRIBUTION [GPU]         complementarity
   [GPU, disk-read 3882]     (disk-read 3884)              (disk-read 3886)
        └──────────────────────────────┼──────────────────────────────┘
                                        ▼
 PHASE 4 (mandates + hardware + capstone)
   exp3888 FR-11 v24 self-learning (load v23 state)   [research-program MANDATE]
   exp3889 GateMate corrigendum (de-flag TAUTOLOGY + readback)   [hardware]
   exp3890 PolarFire + KV260 consolidated continuity            [hardware]
   exp3891 capstone .359  (aggregate non-flagged; paper_ready stays TRUE)
```

**11 tasks, 5 phases.**

### Hardening changes vs `.358 (the load-bearing differences)

1. **EBT import bug fixed.** exp3882 imports the scaled harness via
   `sys.path.insert(0, str(PROJECT_ROOT / "scripts"))` then
   `import thesis_a_part_b_scaled` — NOT `import scripts.thesis_a_part_b_scaled`.
   exp3881 asserts this as a Phase-0 precondition so the kill-gate cannot
   re-block on the same fault.
2. **No hard `gated_on` on the critical path.** Every downstream task reads its
   upstream artifact off disk in-script and emits `blocked_upstream_*` if
   absent (the `.340 proven-safe disk-fallback pattern). A skipped upstream
   costs ONE task, never a chain.
3. **Bare field emission.** Every REQUIRED ARTIFACT FIELD is emitted as a
   **bare scalar** — the `principle:` annotations are guidance for the agent,
   NOT a `{value, principle}` wrapper. (exp3871 wrapped every value in a dict,
   breaking `summarize_artifact.py` and the adversarial verifier.)
4. **All tasks codex + requires_codex + gpt-5.5; GPU tasks add requires_gpu;
   every Run command uses `{project_root}/.venv/bin/python`** (bare `python`
   has no torch → silent CPU drop, the fault that blocked the EBT kill-gate
   once already). gemini crashes on GPU workloads and 429-wiped `.333/`.355;
   codex is the reliable conductor backend (standing operator gemini↔codex
   flip authority, 2026-06-05).

### Phase descriptions

**Phase 0 — hygiene + de-risk (exp3881).** Archive `.358 into
research-complete.yaml (every appended value with a colon-space QUOTED — the
`.355 poison guard), activate `.359, and assert three green-gates before any
research runs: (a) research-complete.yaml loads under `yaml.safe_load`; (b) the
core pretest subset is green; (c) the EBT scaled harness imports via the
path-insert method. Records the backend routing diagnostic.

**Phase 1 — EBT energy-as-GENERATOR (exp3882, exp3883).** exp3882 is the
Phase-3 kill-gate: on a confirmed-headroom checkpoint (held-out AR in
[0.4, 0.95], asserted FIRST — the FALSE_NEGATIVE_RISK guard), does a GLOBAL
discrete beam search over cumulative EBT energy recover AR-level accuracy
(greedy was the bottleneck → **ARTIFACT**) or also fail (energy landscape
misshaped → **FUNDAMENTAL**)? At matched inference FLOPs. exp3883 isolates the
EBT System-2 sub-claim on the same checkpoint: does held-out accuracy rise
monotonically with the energy-descent budget K∈{1,2,4,8,16}? Cite NRGPT
(2512.16762) as the peer whose compute-matched gap this closes.

**Phase 2 — verifier moat IN-DISTRIBUTION (exp3884, exp3885).** exp3884 builds
an in-distribution error-rich corpus (pool FoVer-family incorrect steps +
≤40% synthetic same-style perturbations) and **proves** the k=15 ensemble
discriminates on it (AUROC≥0.65) before it is usable. exp3885 runs the moat
scissor against it: of the gold-incorrect steps a strong reasoner
(Qwen3.6-35B) self-verification MISSES, what fraction does Carnot independently
catch (residual_catch + bootstrap CI95 + error-overlap Jaccard + reasoner
positive control)? Finally adjudicates DT-P2 on an in-band corpus. Cite
Self-Verification Dilemma (2602.03485) as premise, ThinkPRM (2504.16828) as
the generative-PRM comparator.

**Phase 3 — facts via graph-grounding (exp3886, exp3887).** exp3886
de-fabricates exp3862: a REAL graph-grounding invocation (entity/relation
extraction + KB consistency, the HalluGraph/MemGraphRAG mechanism, duration
≥60s, per-item scores persisted) on a RAGTruth-style factual corpus. exp3887
asks the product question: does graph-grounding catch hallucinations the
math-bound ensemble MISSES (low error-mask correlation), making {math+graph}
a strictly broader fact-aware verifier?

**Phase 4 — mandates + hardware + capstone (exp3888–exp3891).** exp3888 is the
FR-11 continuous self-learning MANDATE (v24, loads the persisted v23 state,
Tier-1 online independence-reweighting on a fresh corpus, must hold the frozen
CI band + memory contribution). exp3889 de-flags the GateMate TAUTOLOGY
(distinct timers + JTAG readback). exp3890 consolidates PolarFire + KV260
continuity (SSH-reachability only; honest no-fabric-claim record). exp3891
aggregates non-flagged verdicts; `paper_ready` stays TRUE.

---

## 4. Dependency graph (logical, not hard-gated)

```
exp3881 ──► everything (Phase-0 green-gate; if it blocks, the milestone is unsafe)
exp3882 ──► exp3883            (disk-read checkpoint; graceful blocked_upstream)
exp3884 ──► exp3885            (disk-read corpus; graceful blocked_upstream)
exp3886 ──► exp3887            (disk-read per-item scores; graceful blocked_upstream)
exp3888  (independent)
exp3889, exp3890  (independent hardware)
{all} ──► exp3891 capstone     (reads via summarize_artifact; skips flagged)
```

No `gated_on` field is used. Downstream tasks self-gate by reading the upstream
artifact from disk and emitting `blocked_upstream_*` if it is missing or
INCONCLUSIVE — so a single skipped/blocked upstream never cascades.

---

## 5. Hardware requirements

| Task | Hardware | Precondition |
|---|---|---|
| exp3882, exp3883 | 2×RTX 3090 (CUDA) | `torch.cuda.is_available()` via `.venv/bin/python` |
| exp3885 | CUDA + Qwen3.6-35B-A3B-GGUF cached | `.gguf` path + llama_cpp (NOT AutoTokenizer on the GGUF repo id) |
| exp3886 | CUDA + SOTA GGUF cached | `.gguf` path + llama_cpp |
| exp3889 | GateMate A1-EVB-2M | `nextpnr-himbaechel` + `openFPGALoader -c dirtyJtag --detect` |
| exp3890 | PolarFire + KV260 | `ssh polarfire` / `ssh kria` reachability (KV260: SSH, never host SD card) |
| exp3884, exp3887, exp3888, exp3881, exp3891 | CPU only | — |

**Models (per CLAUDE.md SOTA mandate):** `unsloth/Qwen3.6-35B-A3B-GGUF`
(flagship MoE, exp3885/3886 headline), fallback
`unsloth/gemma-4-26B-A4B-it-GGUF`. Load via the `.gguf` path + `llama_cpp`
(the GGUF repos ship no HF tokenizer files — the tokenizer is embedded).

---

## 6. Invariants (do not violate)

- `paper_ready` stays **TRUE**; report `unmet_gates`, never a count.
- FoVer headline **0.9131** is frozen — never silently substituted; both energy
  routes (selection P0.1, generation Thesis-A) remain honestly bounded unless a
  new measurement says otherwise.
- Never aggregate a `flagged_adversarial: true` artifact's numbers into a
  capstone or forward-facing claim (the fabrication gate).
- External publication is operator-only; no task submits anything.
- `docs/index.html` and operator-curated docs are untouched by autonomous work.

---

## 7. Continuous self-learning (research-program.md mandate)

exp3888 (FR-11 v24) is the mandated self-learning task: Tier-1 online
constraint reweighting (CPU counter updates, <1µs/update — the Tier-1 hardware
path) that upweights verifiers catching the residual errors others miss
(error-independence reweighting, arXiv:2604.07650), loading the persisted v23
state and continuing on a fresh corpus to test that the learned weighting holds
the invariant across iterations (no long-horizon overfit).

---

## 8. New references this milestone (filed in research-references.md)

- **arXiv:2512.16762 NRGPT** — energy-based GPT alternative (ICLR 2026); the
  closest energy-as-generator peer; does NOT do the compute-matched comparison
  exp3882 measures.
- **arXiv:2602.03485 Self-Verification Dilemma** — 85–95% of a reasoner's
  self-rechecks never catch an error; the moat premise.
- **arXiv:2504.16828 ThinkPRM** — generative long-CoT process verifier;
  moat comparator.
- **arXiv:2512.01659 HalluGraph** — KG-alignment hallucination detection
  (entity-grounding + relation-preservation); facts-verifier architecture.
- **arXiv:2604.17109 / 2504.04543** — parallel p-bit Ising machines (inertia
  term; 2048-p-bit Xilinx FPGA); the FPGA/Ising sampler track.
