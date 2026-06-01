# Research Roadmap — Milestone 2026.06.334

**Status:** PRE-STAGED by outer-loop Claude Opus 4.8 (2026-06-01).
**Supersedes:** 2026.06.333 (which produced ZERO science — see below).

---

## 0. What the previous milestone (.333) actually produced: NOTHING

This is the load-bearing fact for .334 and must not be soft-pedalled.

**Every one of the 14 tasks in milestone 2026.06.333 FAILED with the same
error:** `Gemini CLI error: .js:345500:14`. No experiment script was written,
no artifact landed, no `honest_verdict` was emitted. `results/` contains no
`experiment_362x`/`experiment_363x` files. The conductor log
(`ops/conductor-log.md`, 06:09–07:39 UTC 2026-06-01) shows `FAIL` on all 14,
including the four tasks declared `agent_type: claude` — the per-task agent
selection did not route around the failure.

### Root cause (diagnosed during this planning pass)

1. **Gemini quota is exhausted.** A live probe during planning returned:
   `reason: 'QUOTA_EXHAUSTED', code: 429, "Your quota will reset after ~3h"`.
2. **gemini-cli 0.44.0 crashes instead of erroring gracefully on a 429.** It
   throws `An unexpected critical error occurred:[object Object]` at
   `.js:345500:14` rather than returning a clean retryable error, so the
   conductor's self-heal/retry logic could not recover.
3. **The conductor environment forces gemini regardless of per-task
   `agent_type`:** `AGENT_TYPE=gemini` + `GEMINI_FORCE_EXPERIMENTS=1` are set.
   This is why even the `requires_claude: true` tasks crashed via gemini — the
   force-coercion routed them to the dead backend.

**This is an infrastructure outage, not a scientific result.** The .333
*design* was sound (it baked in every lesson from .329–.332: real evidence
corpus, firing code verifiers, valid positive control, bare gated fields,
quarantined poison test). It simply never executed.

### The standing question is now 5 milestones old — always for infra reasons

| Milestone | Cross-domain attempt outcome | Infra root cause (never the science) |
|---|---|---|
| .329 | contaminated null | degenerate corpus (confidence AUROC=1.0), verifiers inert |
| .330 | cascade | dict-vs-bare gated-field break |
| .331 | cascade | empty `{}` artifact |
| .332 | cascade | poison test (exp3612 asserted a degenerate corpus validates) |
| .333 | **total wipeout** | **gemini quota exhaustion + gemini-cli crash-on-429** |

Per the **Failed-Experiment Rerun Discipline**, re-proposing this science is
legitimate: each prior failure names a *distinct* infrastructure cause, and
.334 names what is different (a working backend + a quota-resilience
diagnostic). It is not a doomed scientific rerun — the science has never run.

---

## 1. The .334 thesis: run the .333 science on a backend that works

.334 = the .333 science, **routed to codex (gpt-5.5)** — the verified-working
reserve backend — plus a gemini-quota-crash resilience diagnostic, plus a
prominent operator action to flip the conductor's backend coercion.

### Backend routing decision (every experiment task)

- **`agent_type: codex` + `model: gpt-5.5` + `requires_codex: true`** on every
  experiment task.
- **Why codex:** gemini quota is exhausted and gemini-cli crashes on 429;
  codex-cli 0.135.0 is verified working during planning. This is exactly the
  `requires_codex` positive criterion #3 in CLAUDE.md ("GEMINI QUOTA IS
  EXHAUSTED").
- **Why `requires_codex: true` specifically:** it is the one flag the
  `GEMINI_FORCE_EXPERIMENTS=1` coercion explicitly honors as an exemption
  (`coerces codex → gemini unless task has requires_codex: true`). Without it,
  the dead-gemini coercion would crash .334 the same way it crashed .333.

### OPERATOR ACTION (surfaced, not auto-applied — cannot modify the conductor)

The most robust fix is an operator/conductor-env change, which the autonomous
loop is forbidden from making:

> **Set `CODEX_FORCE_EXPERIMENTS=1` (and unset/override `GEMINI_FORCE_EXPERIMENTS=1`)
> on the conductor until gemini quota recovers.** This flips the runtime
> coercion so all experiment tasks route to working codex without depending on
> the `requires_codex` flag on each task. Alternatively, wait for the ~3h
> gemini quota reset before relaunching the conductor.

Memory pin `feedback_pin_gemini_3_1_pro_preview_until_flash_available` governs
the *model*; it does not require staying on a *crashed CLI with no quota*.
Codex is the documented reserve (`feedback_inner_loop_switched_to_gemini`).

---

## 2. Invariants preserved (do NOT regress)

- **`paper_ready=true` (G1∧G2∧G3∧G4).** G2 closed 2026-05-31 (FoVer headline
  independently reproduced on CI). The capstone re-checks `publication_gate.py`
  and must not regress it.
- **P0.1 stays honest-negative.** Depth-Over-Breadth retired 2026-05-31; do NOT
  re-test Route-1/Route-2. The verifier-vs-SC *headroom* study (exp3645) is the
  *complementary* "where does a verifier help" direction, not a P0.1 re-test.
- **Every `gated_on` field is a BARE scalar** (`feedback_gated_fields_must_be_bare`).
- **No poison tests.** Any shipped pytest parametrizes over the script's HONEST
  verdicts for synthetic fixtures built from realistic strings; it NEVER
  hard-asserts a single success verdict against a real on-disk corpus, and
  NEVER uses placeholder tokens (`Q\d+`/`R\d+`/`H\d+`) the script rejects. The
  .325/.326/.332 poison-test cascade must not recur. The .332 poison test stays
  quarantined under `tests/python/quarantine/`.
- **Fabrication gate.** Any `flagged_adversarial` artifact is excluded from the
  headline/synthesis. Treat any factual-row AUROC of 1.0 as a leak unless
  proven `grounding_leak_free`.

---

## 3. Architecture (unchanged — this milestone is execution, not redesign)

```
                 model-confidence / self-consistency  (the baseline to beat)
                                  |
   +--------------+--------------+----------------+--------------------+
   |   MATH        |   CODE                        |   FACTS            |
   | FoVer 0.9131  | execution-applicable verifiers| real NLI grounding |
   | (frozen,      | (controlled_invariance,       | verifier on        |
   |  exp2837)     |  ast_structure, runtime_adapter|  held-out evidence |
   |               |  ...) on exp1999-derived corpus|  (no 'H'-token leak)|
   +--------------+----------------+---------------+--------------------+
                                   |
            corrected cross-domain re-measurement (CENTERPIECE)
            vs strong confidence baseline, VALID positive control,
            graceful per-row degradation (a blocked row != a null)
                                   |
   +--------------+----------------+----------+--------------------+
   additivity     Weaver peer +         verifier-beats-SC      trained-EBM-judge
   (2nd pair of   inter-verifier        on headroom corpus     OOD counterpoint
    eyes,McNemar) correlation matrix    + hybrid (budget)      (candidate FIX
                                                                for "math-only")
                                   |
                  synthesis (correct the .329-.333 record) -> capstone + G1-G4
```

---

## 4. Phases & tasks (14 tasks, conductor execution order)

**Phase 0 — Transition + infra resilience (exp3638, exp3639)**
- exp3638: archive .333 honestly (record the gemini-quota total wipeout, the
  still-open cross-domain question, the .332 archive-leftover) + activate .334.
- exp3639: gemini-cli quota-crash **resilience diagnostic** — record the live
  gemini quota state + the `.js:345500:14` crash-on-429 signature + the
  conductor's `GEMINI_FORCE_EXPERIMENTS=1` coercion, and write the operator
  recommendation. Documentation only — must NOT touch the conductor or env.

**Phase 1 — Build the fair apparatus (exp3640, exp3641)**
- exp3640: build factual corpus v3 from a real evidence-bearing labeled dataset
  (HaluEval QA `knowledge` / FELM / RAGTruth / Mu-SHROOM); confidence AUROC in
  (0.5,0.95); held-out evidence independent of label; BARE gated fields.
- exp3641: build the labeled CODE corpus from exp1999 + wire the four
  execution-applicable verifiers to FIRE + math->code PRM-transfer stress-test
  (arXiv:2506.00027 vs ThinkPRM arXiv:2504.16828).

**Phase 2 — The fair measurement (exp3642 CENTERPIECE, exp3643)**
- exp3642: corrected cross-domain re-measurement v4 — math (0.9131 frozen) |
  code (firing verifiers) | facts (real leak-free NLI grounding verifier) vs
  strong confidence baseline, VALID positive control, graceful per-row
  degradation. Emits BARE `positive_control_valid` + `at_least_one_nonmath_row_ran`.
- exp3643: additivity / "second pair of eyes" — conditional catch-rate +
  McNemar (gated on exp3642 `at_least_one_nonmath_row_ran == true`, BARE).

**Phase 3 — New breadth (exp3644, exp3645, exp3646)**
- exp3644: position Carnot vs the SOTA weak-verifier peer Weaver
  (arXiv:2506.18203) + measure the inter-verifier correlation matrix Weaver
  assumes away.
- exp3645: where does a verifier beat self-consistency? Build a corpus where
  oracle > SC (real headroom) + verifier-vs-SC + hybrid under a compute budget
  (arXiv:2510.14913). Complements P0.1's no-headroom null.
- exp3646: trained-EBM-judge OOD counterpoint (arXiv:2505.14999) — does a small
  judge TRAINED on reasoning-validity transfer cross-domain where Carnot's
  FIXED ensemble does not? May name the fix for "math-only".

**Phase 4 — Self-learning + hardware continuity (exp3647–exp3650)**
- exp3647: FR-11 continuous self-learning v8 — online correlation-aware verifier
  weighting without collapse (closes Weaver's no-online-adaptation gap).
- exp3648: KV260 SSH-reachability continuity (unreachable .331–.333 — re-check,
  flag the multi-milestone outage as operator action).
- exp3649: PolarFire opportunistic reachability + continuity audit.
- exp3650: GateMate continuity audit (documentation-only — openFPGALoader
  missing per known-issues).

**Phase 5 — Synthesis + capstone (exp3651, combined)**
- exp3651: capstone v334 + cross-domain synthesis + G1-G4 gate. Builds the
  corrected generalization table, corrects the .329-.333 record (the null was
  asserted from blocked/skipped/wiped rows 5x; .333 produced zero artifacts),
  states the scope now that facts + code rows actually ran, and re-checks
  `publication_gate.py` — `paper_ready` must stay true. (Synthesis + capstone
  are folded into one task to keep the milestone at 14; the capstone already
  aggregated every upstream artifact.)

---

## 5. Dependency graph

```
exp3638 (archive/activate) -> exp3639 (gemini resilience diagnostic)
exp3640 (facts corpus v3) --+
exp3641 (code corpus+fire) -+--> exp3642 (CENTERPIECE) --> exp3643 (additivity, BARE-gated)
exp3644 (Weaver+correlation) -----------------------------> exp3647 (FR-11 v8 uses correlation matrix)
exp3645 (headroom vs SC)
exp3646 (trained-judge OOD)
exp3648/3649/3650 (hardware, independent)
exp3640..3647 --> exp3651 (capstone v334 = synthesis + G1-G4 gate, combined)
```

Only one hard structured gate (exp3643 on exp3642's bare `at_least_one_nonmath_row_ran`).
Everything else degrades gracefully per row so a single blocked corpus can
never SKIP the whole milestone (the .331/.332 cascade lesson).

---

## 6. Hardware requirements

- exp3642 / exp3646 declare `requires_gpu: true` (NLI checkpoint + small judge
  train/eval on the RTX 3090 rig). All others CPU-only verifier scoring or
  aggregation.
- Hardware continuity: KV260 / PolarFire / GateMate per Hardware-Task
  Continuity Discipline (SSH reachability for KV260/PolarFire — never host SD
  card; documentation-only audit for GateMate).

---

## 7. Models

- Math row: frozen FoVer headline (exp2837) — no re-run.
- Facts/code corpora: real labeled datasets + small NLI checkpoint
  (DeBERTa/MiniLM-NLI) or a disclosed text-statistical proxy (pcib_probe.py
  pattern). Where a SOTA LLM is invoked, use `cached_sota_pair()` — at least
  one of `unsloth/Qwen3.6-35B-A3B-GGUF` / `unsloth/gemma-4-31B-it-GGUF` /
  `unsloth/gemma-4-26B-A4B-it-GGUF` via the `.gguf` path (embedded tokenizer;
  NEVER `AutoTokenizer.from_pretrained` on a GGUF repo id).

---

## 8. Self-learning coverage

exp3647 (FR-11 v8, online correlation-aware weighting without collapse)
satisfies the per-milestone continuous-self-learning mandate
(research-program.md "Continuous Self-Learning", Tier 1/2).
