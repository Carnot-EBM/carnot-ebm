# Research Roadmap — Milestone 2026.06.339

**Status:** Pre-staged by outer-loop Claude (Opus 4.8), 2026-06-02.
**Predecessor:** 2026.06.338 (re-freeze package re-emitted clean; but the external published baseline
beat the dependency-aware candidate; the code-native verifier returned an implausible AUROC=1.0 and the
detector was already shipped on it; the selection diagnosis blocked a 2nd time; KV260 reachable again).
**Milestone doc for:** `research-roadmap-next.yaml` (`milestone: 2026.06.339`)

---

## 1. What the previous milestone (.338) proved

`.338` finished CLEAN (no DURATION_TOO_SHORT false-flags — the .337 hygiene bug is fixed) but turned two
"wins" into things that must be VALIDATED before they can touch the headline, and exposed an ambiguous
re-freeze. Read via `scripts/summarize_artifact.py`:

| Result | Finding | Artifact |
|---|---|---|
| **Re-freeze package re-emitted CLEAN** | The operator package for the dependency-aware candidate is now adversarial-clean (the exp3681 vestigial-marker flag is cleared). | exp3692 |
| **External baseline BEAT the dependency-aware candidate** | The PUBLISHED de-entangled/CIG reweighting baseline (arXiv:2604.07650) scored **0.9287** — HIGHER than Carnot's dependency-aware candidate **0.9249** (both > frozen 0.9131). Verdict `ties_or_loses`: the dependency-aware candidate is **not uniquely best**. The re-freeze candidate is **ambiguous**. | exp3693 |
| **Code-native verifier returned AUROC = 1.0** | `code_signal_recovered`, but **1.0 is IMPLAUSIBLE_PERFECT** (the exp1851/exp3397 fabrication pattern + the "AUROC≥0.99 = leak unless proven leak-free" rule). **Provisional.** | exp3695 |
| **Detector re-shipped on the 1.0** | The shipped second-pair-of-eyes detector was re-wired with a code operating point built on exp3695's 1.0; E2E green. So the shipped surface now **depends on an unverified perfect number**. | exp3696 |
| **Selection diagnosis BLOCKED a 2nd time** | `blocked_no_multi_candidate_corpus`; `per_candidate_auroc` not reproduced. Second failed diagnosis (exp3682 degenerate, exp3694 blocked). | exp3694 |
| **FR-11 v12** | Drift-reset + single-boundary cross-session persistence succeeded, no collapse. | exp3697 |
| **Hardware** | **KV260 SSH-reachable again** after **8 unreachable milestones**; PolarFire reachable; GateMate `openFPGALoader` still missing. | exp3698/99/3700 |
| **Publication gate** | **paper_ready = TRUE.** G1 (FoVer 0.9131, exp2850), G2 (CI run 26725185125), G3, G4. Frozen headline 0.9131 unchanged. | exp3701 |
| **Backend** | exp3691: gemini probe OK but **real-workload crash re-confirmed** (`.js:345500:14`). Keep codex. | exp3691 |

**Strategic position:** `.338` did not *advance* the headline — it produced two candidate-positives that
are not yet trustworthy. Per `ops/north-star.md` §1, a result is headline-eligible only when it survives
rigor. `.339`'s job is therefore **validation, not new breadth**: pick the single strongest re-freeze
candidate at G1-rigor, and prove (or refute) the code-native AUROC=1.0 on held-out data before it can
headline or stay shipped. This is the convergence-respecting move (north-star anti-churn): a milestone
that lets an IMPLAUSIBLE_PERFECT number reach the shipped surface unverified would be the exp1851/exp3397
trap.

---

## 2. The three biggest gaps (PRD vision vs current state)

1. **The re-freeze candidate is ambiguous — the operator may have a clean package for the wrong number.**
   exp3692 re-emitted a clean package for the dependency-aware candidate (0.9249), but exp3693 showed a
   PUBLISHED external baseline scores higher (0.9287). A credible re-freeze must run Carnot's
   label-conditional dependency-aware weighting vs the external de-entangled/CIG baseline vs a **FUSION**
   of both (multi-verifier composition, arXiv:2502.20379) under the FROZEN dual-condition protocol, pick
   the single strongest > frozen 0.9131, and re-emit the clean package for THAT winner. The frozen 0.9131
   stays frozen; north-star §1 stays operator-curated; the agent prepares, never substitutes.

2. **A provisional AUROC=1.0 has already reached the shipped product.** exp3695's code-native AUROC=1.0 is
   the classic leak/separable-by-construction signature, and exp3696 shipped a code operating point on it.
   This must be leak-audited (arXiv:2603.21454 session-isolated contamination; arXiv:2502.00678 kernel
   divergence) and re-measured on a DIFFERENT held-out corpus (arXiv:2605.11006 execution-verified
   benchmark / fresh GGUF-generated candidates) with a TraceCoder-style execution-trace signal
   (arXiv:2602.06875). Then the shipped surface must be reconciled: recalibrate to the held-out number if
   the signal survives, or NARROW back to math-only-with-abstain if the 1.0 was a leak. A detector that
   ships an inflated code claim is not Phase-1 software-operational.

3. **A settled question keeps getting re-proposed.** The discrimination-vs-selection diagnosis has an
   earned-negative (exp3672) and two failed diagnoses (exp3682 degenerate, exp3694 blocked). Energy
   selection is settled-bounded (the `project_energy_selection_thesis_bounded` memory + arXiv:2512.23067
   Reward Model Selection Crisis + arXiv:2605.30619 margin-vs-connectivity). Per north-star anti-churn,
   `.339` FORMALLY CLOSES it (recommends operator retirement) rather than grinding a third attempt.

---

## 3. Milestone architecture (4 phases, 11 tasks)

```
Phase 0 — Transition + routing safety (exp3702, exp3703)
    archive .338 / activate .339  -->  backend-state diagnostic v5 (5th gemini probe; gates a .340 flip)

Phase 1 — DISAMBIGUATE & FINISH THE RE-FREEZE
    exp3704 dependency-aware vs external vs FUSION at full G1-rigor; pick the single strongest > frozen
            0.9131 and re-emit the CLEAN operator package for the WINNER (prepare-only)

Phase 2 — VALIDATE THE PROVISIONAL CODE-NATIVE WIN (the AUROC=1.0 must survive rigor)
    exp3705 leak-audit + HELD-OUT replication of the code-native verifier (TraceCoder 2602.06875,
            contamination 2603.21454 / 2502.00678, held-out corpus 2605.11006)
              |  code_signal_survives_heldout / leak_detected
              v
    exp3706 reconcile the SHIPPED detector's code operating point with the held-out audit
            (recalibrate-to-held-out if survived, else narrow to math-only-abstain) — runs UNCONDITIONALLY

Phase 3 — CLOSE, SELF-LEARN, HARDWARE, SYNTHESIZE
    exp3707 FORMALLY CLOSE the selection diagnosis (recommend retirement; doc/synthesis-only)
    exp3708 FR-11 v13 — multi-session Tier-2 CONSOLIDATION into a bounded reusable template library
    exp3709 KV260 DRIVE-TO-TERMINAL — board-latency transcript (reachable again; north-star §3 terminal)
    exp3710 PolarFire continuity    exp3711 GateMate audit
    exp3712 capstone + G1-G4 v339
```

### Dependency graph (cascade-proof)

- **No intra-milestone `gated_on` skip this milestone.** exp3706 reads exp3705's verdict but runs
  UNCONDITIONALLY — the shipped surface must be made honest whether the code signal survived or was a
  leak (if it was a leak, the detector is currently overclaiming and MUST be narrowed). Gating it would
  leave the overclaim in place on the leak branch.
- exp3704 reads the already-complete exp3680/exp3692/exp3693 (`.337`/`.338`, on disk) as PRECONDITIONS,
  not gates — no skip risk.
- Every other task is independent. The capstone (exp3712) is UNGATED and treats any skipped/blocked task
  as `not_measured` (never reads a missing field as None).

### Honest-negative discipline carried forward

- **The frozen FoVer headline (0.9131) stays frozen.** The strongest candidate (whatever it is) is a
  re-freeze CANDIDATE with an operator-ready package — NEVER a silent substitution. north-star §1 is
  operator-curated; the agent never edits `north-star.md` or triggers the CI reproducer.
- **The code-native AUROC=1.0 is provisional, not a headline.** It headlines/stays-shipped ONLY if it
  survives held-out at AUROC > 0.5 (CI excludes 0.5) AND < 0.99 (a held-out 1.0 is still a leak flag).
- **facts-generalization RETIRED** (exp3670). **trained-judge-OOD RETIRED** (exp3659). **P0.1 stays
  honest-negative.** **Energy-selection-beats-SC is settled-bounded** (exp3672) — `.339` formally closes
  the diagnosis (exp3707), it does NOT re-grind it.

---

## 4. SOTA models / inference substrate

Every scoring task is `verifier_ensemble_against_cached_candidates` (scores the cached FoVer /
balanced-code corpora on CPU; no LLM load — a *strength* for reproducibility per north-star §1). The
exception is exp3705's held-out code corpus: if the held-out set must be GENERATED fresh, MODEL_SPECS
must name at least one mandated SOTA GGUF (`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`,
or `unsloth/gemma-4-26B-A4B-it-GGUF`) loaded via the `.gguf` path (`cached_sota_pair()` + the llama.cpp
loader — NEVER `AutoTokenizer` on a `-GGUF` repo id, per the CLAUDE.md GGUF tokenizer rule), in which
case `inference_substrate=live_llm_inference` (60s floor applies legitimately). Aggregation/transition/
closure tasks are `aggregation_from_upstream_artifacts`. **No aggregation/scoring task carries a
GGUF/CUDA marker in `model_specs`** (the .337 DURATION_TOO_SHORT false-flag fix).

## 5. Hardware requirements

- **KV260:** SSH-reachability precondition only (`ssh kria 'true'`); host SD-card checks permanently
  retired. **Now reachable again** — `.339` drives it to its north-star §3 terminal state by capturing a
  non-fabricated board-level Ising-sampler **latency transcript** (POC functional anchor, **NO speedup /
  thermalization claim** per Paper-v6 Narrowing #2/#3).
- **PolarFire:** opportunistic reachability/continuity (reachable in `.338`).
- **GateMate:** documentation/audit-only (`openFPGALoader` missing; flash/smoke host-IO hang is a known
  blocker — do NOT run it).

## 6. Backend routing

`.339` keeps **codex + `requires_codex`** on every task (anti-wipeout): gemini probed stable several
times but `.333` wiped a whole milestone and exp3691 (`.338`) re-confirmed a real-workload crash even
when the one-shot probe passed. `.339` carries the headline-VALIDATION work. exp3703 runs a 5th
confirmation probe that gates a `.340` flip (eligibility requires probe OK AND no observed real-workload
crash). The operator may override to gemini-default at activation if quota preservation outweighs the
wipeout risk.

## 7. Self-learning (mandatory per research-program.md)

exp3708 is the continuous-self-learning experiment: **FR-11 v13 multi-session Tier-2 consolidation** —
accumulate learned dependency structures across N≥3 session boundaries into a bounded reusable template
library (forward diff from v12's single-boundary persistence; a step toward Tier-4 adaptive structure)
and validate that a consolidated template transfers to a fresh session better than cold-start, still
collapse-guarded (α_t grounding).

## 8. Invariants checklist (validated before activation)

- `milestone: 2026.06.339` (calendar-month prefix from today's UTC date 2026-06-02; trailing index +1). ✓
- 11 tasks; all `agent_type: codex` + `requires_codex` (anti-wipeout). ✓
- Every `gated_on` value is a **BARE scalar** — there are NO `gated_on` fields this milestone (exp3706
  runs unconditionally), so the bare-field trap is structurally avoided. ✓
- Every scope-matched legit continuation carries an `operator_override` ≥10 chars; exclusion-manifest
  lint = **6 WARNINGs, all operator_override present → activation proceeds, no HARD violations**. ✓
- No poison tests (`.325/.326/.332` cascade): every task's prompt mandates parametrizing pytests over
  honest verdicts on synthetic fixtures; never hard-assert one success string against a real corpus; no
  Q/R/H-number placeholders. ✓
- De-tautology + leak-guard on every scoring artifact (distinct field per distinct metric; **AUROC ≥0.99
  on n≥1000 is a leak red-flag — the exp3695 1.0 trigger drives exp3705**; seeds may be content-derived
  but never the experiment number alone). ✓
- Inference-substrate hygiene: every aggregation/scoring task sets `inference_substrate` correctly and
  carries NO compute-bound marker, then runs `adversarial_verify` and confirms clean (the .337 fix). ✓
- Every required artifact field carries a `principle:` annotation; bare booleans stored bare. ✓
- Operator-only external publication / public-doc discipline: exp3704 prepares the re-freeze package but
  never edits `north-star.md` / the CI workflow / triggers the run; exp3707 never edits
  `exclusion_manifest.yaml`. ✓
