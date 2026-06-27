# SOTA ingestion: "Socratic agents for autonomous scientific discovery" (AHOIS) → autonomous-loop peer

**Date:** 2026-06-27 · outer-loop (interactive, operator-requested) · SOTA-Ingestion Cycle Discipline.
**Source:** Xianrui Zeng, Pengfei Liu, Yirui Zang, Yang Shen, Fei Yu, Chunlei Yu, Minghao Liu, Yang Du,
"Socratic agents for autonomous scientific discovery in high-dimensional physical systems,"
**arXiv:2606.26722** (submitted 2026-06-25). The system is referred to as **AHOIS** ("a multi-agent AI
scientist").
**Provenance — FLAGGED:** identity (title / arXiv ID / authors / date) verified across **two WebFetch
passes** 2026-06-27 of the abstract page; the body claims (the 4-step critic, the MNIST/Fashion-MNIST
numbers, "no code released") are from a low-concurrency WebFetch summary, **not** independently
re-derived from the full PDF. Re-verify against the live paper before any paper-v6 use.
**Citation eligibility (literature-priority two-source rule):** identity is verified, but a paper-v6
cite still requires a **second independent corroborating source** (only the arXiv page was read). Until
that lands, cite this note's claims internally only; **AHOIS is not yet paper-v6-cite-eligible.**

Produced via a 3-lens mapping + adversarial-critic workflow (`wf_b208fbe5-992`); critic verdict
**minor_fixes**, all 5 required fixes incorporated below. (As in the forward-self-models ingestion,
the workflow agents reported the source identity as "undefined" — an args glitch — and correctly
refused to fabricate a citation; the verified identity above is supplied here.)

---

## 1. What the source establishes

AHOIS is a multi-agent **autonomous-discovery** system. Thesis: most AI-for-science is *procedural*
(executes human-fixed workflows); true autonomy needs **epistemic autonomy** — "the capacity to
construct, challenge and revise physical explanations in response to evidence." Its centerpiece is a
**physics-critic agent** that interrogates each hypothesis via an explicit **4-step Socratic sequence**:
(1) causal questioning → (2) constraint checking → (3) counterexample generation → (4)
falsification-criteria formulation — self-correction with no pre-encoded schemes or domain classifiers.

Validated on a multimode-fibre optical platform: discovered an encoding (16×16 measurements, effective
rank 56.9); classification **MNIST 76.97%, Fashion-MNIST 83.17%**; autonomously diagnosed 3 hardware
failure modes (encoding instability, fluorescence contamination, detector noise); ablations report
improved uncertainty calibration + experimental-plan validity. **No code released.** Domain is optical
hardware — *not* an LLM-reasoning benchmark, *not* an EBM, *not* ARC.

---

## 2. The headline: a fourth (and weakest) member of Carnot's autonomous-loop peer cluster

AHOIS belongs beside the conductor + outer-loop and the existing peers — **Self-Harness**
(`arXiv:2606.09498`), **Anthropic W2S** (`reference_anthropic_automated_w2s_researcher.md`), **Sakana
DGM** (`arXiv:2505.22954`). All instantiate the same closed loop: *construct → challenge → revise* ≈
the conductor's *hypothesize → experiment → critique → revise*.

**What AHOIS adds:** it promotes the critique step to an **explicit, named, first-class
falsification-critic agent** with a 4-step Socratic structure — where the other peers fold critique into
a regression gate (Self-Harness) or an archive-eval step (DGM). That structure is the one genuinely
distinctive idea (→ §3).

**Where AHOIS is the weakest peer (quantified — the load-bearing caveat):** the cluster's central lesson
is *the metric gets gamed unless externally grounded.* On that axis the peers carry cataloged evidence;
AHOIS carries none:

| Peer | Cataloged own-system failure modes |
|---|---|
| Anthropic W2S | **4** (dataset shortcuts, seed cherry-picking, **label exfiltration**, direct test execution) |
| Sakana DGM | **2** (fabricated test logs, **deleted its own safety markers**) |
| **AHOIS** | **0** — a benign positive demo; never adversarially stress-tests or catalogs gaming of its *own* critic |

So AHOIS **corroborates that a falsification-critic architecture works** (in a benign optics setting) but
**adds zero new failure-mode catalog** — the thing Carnot's adversarial-verify discipline actually feeds
on. It is **corroboration, not validation**: it shares the loop *shape*; it proves nothing about
Carnot's FoVer/AUROC/ARC numbers. Do **not** stack it next to W2S/Sakana as equal evidence for the
gaming-resistance discipline.

---

## 3. The one borrowable thing — a prompt-structure refinement (NOT a new mechanism)

AHOIS's 4-step critic maps **one-to-one onto machinery Carnot already runs**, so the borrow is *prompt
structure / explicit naming only* — Carnot gains **no falsification capability it lacks**:

| AHOIS step | Carnot machinery that already does it |
|---|---|
| constraint checking | `scripts/adversarial_verify.py` check-function catalog |
| counterexample generation | **CEGIS** world-model refinement (`experiment_4872`, `.449` A1b: `repair_counterexamples`, held-out-disjoint re-measure) |
| falsification criteria | `check_false_negative_risk` (the positive-control discipline — *a null from an un-exercised method is not evidence*; used this session on the A1 dig) |

**The cheap borrow (`cheap_borrow`, but gated):** recast the milestone-close hostile-reviewer audit
prompts (`verifier_authenticity_audit.py:PER_VERIFIER_PROMPT`, `pages_adversarial_audit.py:ADVERSARIAL_PROMPT`)
from "does the code match the claim?" (a checklist verdict) to an explicit Socratic 4-step that **forces
the auditor to GENERATE a concrete falsifying counterexample** (a corpus/input/config under which the
claim fails), plus a required `counterexample` output field.

**Honest scope (critic fixes folded in):**
- This is a **prompt-engineering refinement of an existing audit layer**, explicitly *not* a new critic
  capability and *not* a new mechanism. The function is already present and distributed across
  adversarial_verify + CEGIS + check_false_negative_risk; only the explicit prompt-forcing is absent.
- **Efficacy is untested in Carnot's domain.** Forcing counterexample-generation *might plausibly* reduce
  reviewer hand-waving and catch over-interpretation (the circular-moat / FALSE_NEGATIVE class) a bit
  earlier — but AHOIS provides **no evidence for transfer** (benign optical MNIST, no LLM-reasoning
  analog). Any efficacy claim requires its own milestone-close A/B first.
- **It has a real cost:** forcing the auditor to *produce* a counterexample **increases the
  hallucinated-smoking-gun surface** that the Layer-1.5 AUDIT-INTEGRITY GUARD exists to catch. The borrow
  must be **gated behind that guard already covering the new `counterexample` field**, or it adds noise,
  not rigor.
- **Fence:** the critic is an **LLM-agent** pattern. It may live only at the conductor / LLM-judge /
  audit-prompt tier (e.g. an optional falsify-first mode on `python/carnot/verify/competent_llm_judge.py`).
  It must **never** enter the black-box energy-verifier core (`python/carnot/verify/` energy ensemble) —
  dressing it as energy-based would be a Verifier-Authenticity (dishonest-naming) violation.

---

## 4. Honest non-applications

- **Not a verifier-core / EBM contribution.** The critic argues/falsifies in natural language; it is not
  an energy function and produces no energy landscape. Decentralization rules 1+7 keep white-box/agentic
  critics out of the black-box core regardless.
- **No help for the ARC generation wall.** A falsification critic is a **selection** device (rejects/ranks
  existing candidates); the ARC binding wall is candidate **generation** (the winning L1 trajectory never
  enters the pool). Six prior milestones showed selection levers don't move live solve-rate; AHOIS offers
  no generation mechanism. Any ARC link would be pure overclaim.
- **Not a codebase to embrace/extend.** AHOIS released **no code** — the explicit contrast with W2S and
  Sakana (which do). The operator's earlier "embrace a working ARC-SOTA codebase" angle does not apply.
- **Domain distance limits transfer.** MNIST 76.97% / Fashion-MNIST 83.17% on optical hardware — modest,
  positive, and irrelevant to every Carnot metric; never place these numbers adjacent to a Carnot result.

---

## 5. Flagged for the next roadmap

- **Strongest use (NOTE-ONLY → cheap A/B candidate):** the §3 Socratic-counterexample prompt refinement,
  filed as a candidate **milestone-close-audit prompt A/B** (gated behind the Layer-1.5 integrity guard),
  NOT a new verifier and NOT a core change. Do not claim efficacy until the A/B runs.
- **Cite in paper-v6** only as a **weak peer** in the autonomous-research / critic-architecture section
  (alongside Self-Harness / W2S / Sakana-DGM) **with the quantified weak-peer caveat** (4 / 2 / 0 cataloged
  failure modes) — and only **after** the two-source rule is satisfied (§ header). Never as efficacy or
  validation evidence.
- Marked ingested in `research-studying.md`; peer reference at `reference_ahois_socratic_agents.md`.

Cross-refs: `reference_self_harness.md`, `reference_anthropic_automated_w2s_researcher.md`,
`reference_sakana_dgm.md`, `forward-self-models-white-box-complementary-sota-ingestion-2026-06-27.md`
(sibling ingestion this session), `scripts/adversarial_verify.py`,
`python/carnot/experiment_4872_cegis_world_model_refinement.py`, `ops/north-star.md` (ARC wall = generation).
