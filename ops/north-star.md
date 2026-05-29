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

## 1. THE HEADLINE CLAIM (the one surviving positive)

Reconstructed 2026-05-29 by auditing every positive claim against the
Paper-v6 Narrowing Discipline retraction list. Two claims survive all the
walk-backs (FoVer 0.9857→0.9131, hardware speedup retracted, replacement-grade
refuted, Spera-generalization retracted, thermalization retracted):

### Methods headline (most rigorous)
> **Carnot's k=15 verifier ensemble reaches AUROC 0.9131 on the FoVer
> step-error corpus (n=1,000, 5 seeds, dual-condition, live RTX 3090,
> CI95 [0.9027, 0.9235]), and an isolated memory-ablation shows the FR-11
> self-learning component contributes +0.0185 AUROC (CI95 [0.0125, 0.0245]).**

Source: exp2837 (dual-condition rescue), confirmed in capstone exp2948.
Status: live-GPU, adequate n, 5-seed, adversarial-verify clean, not
contradicted by any later experiment, not on the retraction list.

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
