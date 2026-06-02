# Research Roadmap — Milestone 2026.06.338

**Status:** Pre-staged by outer-loop Claude (Opus 4.8), 2026-06-01.
**Predecessor:** 2026.06.337 (dependency-aware weighting cleared G1-rigor CLEAN; two product
diagnoses came back degenerate/earned; re-freeze package tripped a linter false-positive).
**Milestone doc for:** `research-roadmap-next.yaml` (`milestone: 2026.06.338`)

---

## 1. What the previous milestone (.337) proved

`.337` confirmed the headline lead at full G1-rigor but exposed that the FINISH line and two
product gaps still need real work. Read via `scripts/summarize_artifact.py`:

| Result | Finding | Artifact |
|---|---|---|
| **Dependency-aware weighting clears G1-rigor — CLEAN candidate** | Full dual-condition integrity (mirror exp2850): production dependency-aware AUROC **0.9253** vs Carnot-current **0.9131** (= frozen headline), delta **+0.0122**, 5 seeds, gate passed, adversarial-clean + leak-free. `dependency_aware_g1_rigor_confirmed=true`. | exp3680 |
| **Re-freeze package prepared but FLAGGED** | Verdict `refreeze_package_ready_for_operator` but **DURATION_TOO_SHORT CRITICAL** — a linter false-positive (25 s verifier-scoring artifact embedding vestigial GGUF/CUDA `model_specs` markers). Capstone recorded `package_blocked`. | exp3681 |
| **Selection-gap diagnosis DEGENERATE — not a real verdict** | Flagged **TAUTOLOGY**: per-question-normalized (0.3443) == raw ensemble (0.3443) == self-certainty (0.3443) — the "fixes" were no-ops; `per_candidate_auroc` collapsed to **0.5555**; ranking calib **0.15**. The discrimination-vs-selection question is **still open**. | exp3682 |
| **Code: reweighting math verifiers does NOT fix code — earned** | Code AUROC stays **0.5** under dependency-aware weighting + recalibration. Detector is honestly math-only; the fix is a code-NATIVE signal, not a reweighting. | exp3683 |
| **Product value survives the stronger baseline** | Fused detector adds value OVER self-certainty (not just plain confidence). | exp3684 |
| **FR-11 v11 drift-aware online weighting** | Recovers after drift, no collapse, +0.088 over v10. | exp3685 |
| **Publication gate** | **paper_ready = TRUE.** G1 (FoVer 0.9131, exp2850), G2 (CI run 26725185125), G3 (narrowing-clean), G4 (traces). Frozen headline 0.9131 unchanged. | exp3689 |
| **Backend** | gemini stable on a **3rd** consecutive probe (exp3679) → `.338` is flip-eligible. But `.333` was a total gemini-crash wipeout. | exp3679 |
| **Hardware** | KV260 SSH-unreachable for **8 consecutive milestones** (operator-action); PolarFire reachable; GateMate `openFPGALoader` missing. | exp3686/87/88 |

**Strategic position:** the headline lead is *measured and confirmed* — what remains is to
**finish the re-freeze cleanly and credibly** (a clean operator package + a published external
comparator the candidate beats), then **fix the two product gaps that were not actually resolved**
(the selection diagnosis was a buggy no-op; the code detector needs a new signal, not a reweight).
Per `ops/north-star.md` §1, every milestone advances the headline, ships product value, or earns a
trustworthy negative. `.338` does all three.

---

## 2. The three biggest gaps (PRD vision vs current state)

1. **The headline candidate is confirmed but the re-freeze is not operator-actionable.** exp3680
   confirmed dependency-aware 0.9253 > frozen 0.9131 at G1-rigor, but the operator package (exp3681)
   is flagged_adversarial (a vestigial-marker false-positive) and has no external comparator. A
   credible re-freeze needs (a) a CLEAN package the operator can act on, and (b) a beat-the-published-
   baseline result — Carnot's *label-conditional* dependency weighting vs the published
   de-entangled reweighting / CIG (arXiv:2604.07650, +4.5% over majority). The frozen 0.9131 stays
   frozen; north-star §1 stays operator-curated; the agent prepares, never substitutes.

2. **The discrimination-vs-selection diagnosis was never actually run.** exp3682 was degenerate
   (no-op "fixes", `per_candidate_auroc` collapsed to 0.55). The open question — can an energy
   verifier's selection recover above self-consistency via per-question calibration / pessimistic
   (LCB) BoN (arXiv:2604.04648) / self-certainty composition (arXiv:2502.18581), or is the
   decoupling fundamental (arXiv:2512.23067) — needs a correct, non-degenerate run. This is the
   candidate-ranker product (Tier A `score_candidates`).

3. **The shipped detector is still code-blind, and reweighting can't fix it.** exp3683 earned that
   reweighting the MATH verifiers leaves code AUROC at 0.5. The fix is a code-NATIVE verifier
   (attribution-graph / structural / execution signal, arXiv:2602.07080 CodeCircuit;
   arXiv:2510.09312 shows error signatures are domain-specific). A detector that returns noise on
   code is half-shipped (Phase-1 software-operational gate).

---

## 3. Milestone architecture (4 phases, 12 tasks)

```
Phase 0 — Transition + routing safety (exp3690, exp3691)
    archive .337 / activate .338  -->  backend-state diagnostic v4 (4th gemini probe; gates a .339 flip)

Phase 1 — FINISH THE HEADLINE RE-FREEZE
    exp3692 RE-EMIT the operator re-freeze package CLEAN (clear the exp3681 vestigial-marker flag)
    exp3693 EXTERNAL comparator: CIG / de-entangled reweighting (2604.07650) — Carnot beats it?

Phase 2 — FIX THE TWO PRODUCT GAPS
    exp3694 REDO the selection-gap diagnosis PROPERLY (pessimistic LCB 2604.04648 + real per-question
            calibration + self-certainty 2502.18581; verify non-degeneracy)
    exp3695 code-NATIVE verifier for the code-blind detector (2602.07080 attribution/structural/exec)
              |  code_signal_recovered (BARE bool)
              v
    exp3696 re-ship the detector with a math+code operating point (gated on exp3695)

Phase 3 — SELF-LEARNING + HARDWARE + SYNTHESIS
    exp3697 FR-11 v12 drift-aware online weighting + RDumb++-style reset policy (2601.15544)
    exp3698 KV260 continuity (8th unreachable milestone — operator-action)
    exp3699 PolarFire continuity    exp3700 GateMate audit
    exp3701 capstone + G1-G4 v338
```

### Dependency graph (cascade-proof)

- **exp3696 `gated_on` exp3695.`code_signal_recovered == true`** (BARE bool). The only intra-milestone
  gate. If the code signal does not recover, exp3696 is skipped and the detector stays honestly
  math-only — no cascade.
- exp3692 reads the **already-complete** exp3680 (`.337`, on disk, confirmed=true) as a PRECONDITION,
  not a gate — no skip risk.
- Every other task is independent. The capstone (exp3701) is UNGATED and treats any skipped task as
  `not_measured` (never reads a missing field as None).

### Honest-negative discipline carried forward

- **The frozen FoVer headline (0.9131) stays frozen.** A dependency-aware win is a re-freeze
  CANDIDATE with an operator-ready package — NEVER a silent substitution. north-star §1 is
  operator-curated.
- **facts-generalization RETIRED** (exp3670 same-verdict on real RAGTruth) — do NOT re-propose.
- **trained-judge-OOD RETIRED** (exp3659). **P0.1 stays honest-negative.**
- **The selection earned-negative (exp3672) stands** as the baseline; exp3694 is the proper
  diagnosis of *why* and *whether it is fixable*, with `retire_if_same_verdict: true` if no
  non-degenerate fix beats SC.

---

## 4. SOTA models / inference substrate

Every scoring task is `verifier_ensemble_against_cached_candidates` (scores the cached FoVer /
balanced-code corpora on CPU; no LLM load — this is a *strength* for reproducibility, per
north-star §1). Where a task needs fresh candidate generation or logits (exp3695 code corpus,
exp3694 self-certainty signal), MODEL_SPECS must name at least one mandated SOTA GGUF
(`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or `unsloth/gemma-4-26B-A4B-it-GGUF`)
loaded via the `.gguf` path (`cached_sota_pair()` + the llama.cpp loader — NEVER `AutoTokenizer`
on a `-GGUF` repo id, per CLAUDE.md GGUF tokenizer rule). Re-use cached corpora where they exist;
generate only what is missing, behind a PRECONDITIONS check.

## 5. Hardware requirements

- **KV260:** SSH-reachability precondition only (`ssh kria 'true'`); host SD-card checks permanently
  retired. Record the 8-milestone outage as an operator-action item.
- **PolarFire:** opportunistic reachability/continuity (reachable in `.337`).
- **GateMate:** documentation/audit-only (`openFPGALoader` missing; flash/smoke host-IO hang is a
  known blocker — do NOT run it).

## 6. Backend routing

`.338` keeps **codex + `requires_codex`** on every task (anti-wipeout): gemini probed stable a 3rd
time but `.333` wiped a whole milestone and `.338` carries the headline-FINISH crown jewels.
exp3691 runs a 4th confirmation probe that gates a `.339` flip. The operator may override to
gemini-default at activation if quota preservation outweighs the wipeout risk.

## 7. Invariants checklist (validated before activation)

- `milestone: 2026.06.338` (calendar-month prefix from today's UTC date; trailing index +1).
- Every `gated_on` value is a **BARE scalar** (`feedback_gated_fields_must_be_bare`).
- Every `prior_failures` entry carries all four sub-fields; every scope-matched legit continuation
  carries an `operator_override` ≥10 chars.
- No poison tests (`.325/.326/.332` cascade): pytests parametrize over honest verdicts on synthetic
  fixtures; never hard-assert one success string against a real corpus; no Q/R/H-number placeholders.
- De-tautology + leak-guard on every scoring artifact (distinct field per distinct metric; AUROC
  ≥0.99 on n≥1000 is a leak red-flag; seeds may be content-derived but never the experiment number
  alone — the exp3506 trap).
- Every required artifact field carries a `principle:` annotation; gated/bare booleans stored bare.
