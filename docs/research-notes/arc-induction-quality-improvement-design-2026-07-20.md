# ARC world-model induction-quality improvement — scoping design (2026-07-20)

**Status:** SCOPING / DESIGN ONLY. No code written, no live-path file modified. This note proposes ONE
concrete, falsifiable next experiment to attack the induction-quality wall diagnosed tonight
(`heldout_accuracy ≈ 0.0` on 29/37 induced world models). It grounds the design in the actual Carnot
code (with `file:line` for every mechanism claim), in the ARC-AGI-3 leaderboard-winner architecture
audit, and in the CEGIS / iterative-code-repair / world-model-induction literature, and includes an
up-front adversarial self-critique.

**Reads as required input:** CLAUDE.md "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent" (foundational
framing), "ARC Live-Path Reachability Discipline", "Failed-Experiment Rerun Discipline", "Missing-Verifier
Gap Logging", "Phase Prototype + Empirical Validation + Adversarial Check", "Literature Priority
Discipline", "Decentralization-Respecting Design Constraints".

**Immediate prerequisites (read in full):**
`docs/research-notes/arc-world-model-induction-quality-diagnosis-2026-07-20.md` (the diagnosis this design
targets); `docs/research-notes/arc-top-project-search-architecture-audit-2026-07-20.md` (the winner-vs-Carnot
architecture audit that reframes the gap); `docs/research-notes/arc-action-effect-representation-redesign-2026-07-19.md`
(the rigor/structure template).

---

## 1. The problem, stated precisely (with tonight's real numbers)

The induction-quality diagnosis (2026-07-20) measured 37 *successful* world-model inductions (31 ThinkingCap-27B
+ 6 Qwen-9B, `induce_ok` = valid `engine()`+`is_level_complete()` code emitted) across 17 distinct games,
each seeded on a real level-up-straddling window. The load-bearing numbers:

- **`heldout_accuracy` (exact full-grid transition match) min/median/mean/max = 0.0 / 0.0 / 0.124 / 1.0.**
  **29 of 37 (78 %) score exactly 0.0** — the induced code *never once* reproduces a full observed
  transition. 33/37 are below the live default `exact` gate of 0.5.
- The softer `cell_recall` (changed-cell recall on state-changing transitions) has median 0.369 — the
  models get *some* changed cells right (locations/direction) but are never byte-perfect.
- **`reached_levelup` = 0 / 80** across both arms; every plan built inside these models reached zero real
  level-ups.
- ThinkingCap's edge (31/40 vs Qwen's 6/40 `induce_ok`) is *code-emission*, not correctness: the 34 Qwen
  failures are `last_stop_type="limit"` + `overran=True` (hit the `n_predict=16384` cap before closing
  valid code). Finishing the program is upstream of, and orthogonal to, whether the program is correct.

Reading the actual induced code (`results/arc_e3/<game>/world_model.py`) found six recurring failure
patterns (diagnosis §4): (1) pure identity `return grid`; (2) single hardcoded click-recolor; (3)
confabulated plausible-but-wrong mechanic (self-commented "simplified rule"); (4) **window memorization** —
hardcoding literal observed pixel coordinates (`ls20`: "Place a block at (61, 13)") rather than inferring a
rule, which is the mechanism behind *high cell_recall + zero heldout* (memorized coords reproduce the very
transitions they were fit to, generalize to nothing); (5) degenerate goal predicates (`return False` /
`return True` / whole-grid-uniform); and (6) the ONE clean success class — near-perfect induction only when
the mechanic matches a template the LLM already knows (`sp80` is a textbook 2048-merge model, `heldout`
hit 1.0).

**The synthesis (diagnosis §5):** the induced models are near-universally wrong about the game's real
action→effect DYNAMICS. The trust gate correctly rejects most of them (33/37 fail the exact `≥0.5`
gate) — **the gate is not the problem; the induction quality is.** The binding wall is *generating a correct
world model*, not selecting/searching among candidates (corroborated by REQ-ARC-FCP-5757: candidates are
generated/ranked correctly 93 %+, and by the search-architecture audit: Carnot already has *more* search
machinery than any Milestone-1 winner).

---

## 2. The architectural question: is "induce an explicit, verifiable Python world model" the right shape?

This is load-bearing and the task rightly forbids dodging it. The 1st-place winner (Duck Harness, Tufa Labs)
does **not** induce a formal `engine()` function up front — it gives a capable 27B LLM object-centric
perception + up to 12 REPL inspection turns and lets it react turn-by-turn, committing one greedy action per
turn with **no verified symbolic model** (search-arch audit §1a: `tool_agent.py:1495,1976`). So does the
Duck evidence refute Carnot's explicit-induction architecture?

**My answer: explicit induction is still the right bet FOR CARNOT — but the conviction is CONDITIONAL and
its central mechanism has never actually been tested (see §3). Here is the genuine, evidence-grounded
reasoning, not a default to the built thing.**

**Why the Duck comparison does not refute explicit induction:**

1. **Duck abandons verification; that is precisely what Carnot's founding thesis rejects.** Duck's LLM is
   the final authority on every action — no component checks its world model against reality. Carnot's
   entire reason to exist is "escape LLM hallucinations + verify outputs against objective ground truth"
   (`project_core_motivation`, CLAUDE.md Project Vision, the E3 module docstring: "LLMs are most reliable
   when used not as final authorities, but as PROPOSAL mechanisms inside systems that can check their
   outputs"). Adopting Duck's reactive loop wholesale means deleting the verifier — the moat. The Duck
   result says "a big model's un-verified judgment beats a flat greedy policy," which is orthogonal to
   whether verifier-grounded induction can work.

2. **Duck needs a 27–31B model in the action loop; Carnot's Kaggle-16GB constraint structurally forbids
   that.** The scored eval is iGPU/~16GB-class; Carnot's live submission generator is frozen at
   **Qwen3.5-9B-MTP** (`project_arc_live_generator`). Every winner puts a *bigger* generator directly in
   the action loop (Duck 27B, Reki/forge 31B); Carnot uses a *smaller* generator only for induction and
   lets classical search decide (search-arch audit §2 table). The decentralization thesis
   (CLAUDE.md Decentralization-Respecting Design Constraints rule 1: local-first, open-weight, small-enough
   to run on hardware a user already owns) is *why* Carnot induces-then-verifies with a small model rather
   than betting on a large reactive one. Explicit induction is the shape that lets a *small* open model +
   a verifier be competitive with a *large* model's implicit planning.

3. **The audit's own finding is that the winners have a better GENERATOR, not a better induction method.**
   The search-arch audit (§3) concludes the winners "route around" weak-model induction by using a bigger
   model — none of them has a superior induction technique Carnot is failing to copy. So "induction is the
   bottleneck" and "the winners don't induce" are the *same* fact viewed twice: the winners avoid the
   induction problem by throwing a 27–31B model at it. Carnot's bet is that verifier-grounded
   *counterexample-guided refinement* can lift a small model's induction to competitiveness — the classic
   "small model + verifier + iterative repair beats big model single-shot" hypothesis (WorldCoder / REx,
   §4).

**Where I do NOT default to (a):** that bet — verifier-grounded refinement lifts a small frozen model's
induction quality — **has never been measured** (§3 proves tonight's null is a *single-shot* null; the
refinement loop was bypassed). And the literature is genuinely split on whether it holds for *small frozen*
models (§4: WorldCoder/E3 show it works with GPT-4/GPT-5.5-class models; "Falsification, Not Exposure"
arXiv:2606.31511 shows self-repair feedback does *not* significantly improve small frozen code models). So
the honest position is: **the explicit-induction architecture deserves exactly one clean test of its
central mechanism before we either commit to it or reconsider.** If that test is *also* null, the
architectural reconsideration becomes evidence-backed and concrete (§5 negative branch): either relax the
constraint and run a larger *offline* induction model on the conductor's dedicated GPU-0 3090 (permitted per
the 2026-06-27 GPU-allocation directive for offline induction), or pivot toward a Duck-style reactive loop
with the verifier demoted from *planner* to *filter* (which "Falsification, Not Exposure" suggests is the
only role feedback reliably plays for small frozen models anyway).

---

## 3. What is already built vs. genuinely new — the refinement loop exists and is UNMEASURED offline

The task flagged a critical checkable fact: is there already an iterative refinement mechanism
(induce → test against held-out data → feed the specific wrong predictions back → ask for a code fix), and
does the live path (or tonight's harness) actually call it? **Answered from source, both halves:**

### 3a. The CEGIS refinement loop EXISTS and is a genuine counterexample-guided repair loop

`execute_bounded_llm_reinduction` (`python/carnot/agentic/arc_llm_reinduction.py:654`, `MAX_REFINEMENT_ROUNDS
= 3` at `:30`) is a real CEGIS loop:

- **Round 0** calls `proposer.induce(...)` (`:726`).
- **When the induced engine's `heldout_accuracy < min_heldout_accuracy`** (`accepted` check at `:775`,
  `if not accepted:` at `:796`), it scores the engine with `WorldModelVerifier(list(transitions)).score(...)`
  and attaches **real per-transition mismatch evidence** — `real_mismatches` carrying BEFORE / your
  PREDICTED / true OBSERVED deltas (`:804-814`) — then `continue`s to the next round.
- **Rounds 1–2** call `proposer.refactor(game, _counterexample_result(last_counterexample))` (`:735`),
  which builds `refactor_prompt` (`arc_executable_world_model.py:1285`): *"reproduces only {n_correct}/{n}
  … Below are failing cases (BEFORE / your PREDICTED / the true OBSERVED next grid). Fix engine() so it
  reproduces these too, and REFACTOR toward simpler, more general rules (replace special cases with shared
  rules)."* That "replace special cases with shared rules" instruction is a **direct countermeasure to the
  window-memorization failure mode** (diagnosis §4 pattern 4).
- It also has a **goal-repair** half (`_repair_degenerate_goal`, `:606`) that substitutes a non-degenerate
  reachable proxy when the induced `is_level_complete` is degenerate — attacking failure pattern 5.

This is textbook counterexample-guided inductive synthesis (§4): induce a program, verify against
observations, feed the counterexamples back, repair. **It is not new infrastructure — it is built, wired,
and default-on live.**

### 3b. The LIVE path calls it; tonight's OFFLINE harness does NOT

- **Live:** `arc_competition_agent.py:3885` (level-up reinduction, `min_heldout_accuracy=1.0`) and
  `:4005` (stall / first-contact, default-on since REQ-ARC-FCP-5699-35, opt-out `CARNOT_ARC_STALL_REFACTOR_LOOP=0`).
  Both route induction through the full CEGIS loop.
- **Tonight's harness (`run_seeded_progress`, `arc_actions_to_progress.py:641`) does a SINGLE-SHOT induce
  and stops:** `induce_ok, _detail = proposer.induce(game, list(window), int(cell))` at `:702`, then
  `load_engine` + one `plan_in_model` call (`:704-710`). There is **no `execute_bounded_llm_reinduction`
  call, no `refactor()`, no verify→feed-back→re-induce.** The dead giveaway: **`n_refinement_rounds=0` is
  hardcoded in the returned result** (`:764`). The harness's own docstring says it "call[s]
  `proposer.induce(...)` DIRECTLY … guaranteed injection" — a deliberate single-shot isolation of the
  *prompt-content* arms (playbook/think), which happens to bypass refinement entirely.

**Conclusion (a real finding either way, per the task):** the refinement loop is **built and live but was
never exercised in tonight's diagnosis.** Every `heldout_accuracy ≈ 0.0` number tonight is a **single-shot
induction** number. Whether the existing verifier-grounded refinement lifts those numbers is **unmeasured
offline**, and only coarsely probed live: the one prior live measurement (REQ-ARC-FCP-5699-32, noted at
`arc_competition_agent.py:3986`) found the `min_heldout_accuracy=1.0` gate "rarely met (0/6 rounds across a
full real run on g50t)" — but that is a **binary gate-met count at threshold 1.0**, not the **continuous
per-round `heldout_accuracy` trajectory** (did refactor move heldout from 0.0 → 0.3 → 0.5 even while never
reaching 1.0?). That continuous quantity — the actual measure of whether refinement improves induction — has
never been isolated. Crucially, **the per-round heldout is ALREADY recorded** in
`outcome.rounds[*]["heldout_accuracy"]` (`arc_llm_reinduction.py:790`); the harness simply never invokes the
loop that would populate it. This makes the proposed measurement extremely cheap.

**Narrower memorization-specific levers already exist too, also unmeasured for effect:** the DEV-only
`CARNOT_ARC_INDUCE_TRANSITIONS_K` override (`arc_executable_world_model.py:1121`, REQ-ARC-FCP-5699-23, shows
the LLM more per-action examples) and the playbook-exemplar prior (`:1142`) that explicitly warns "a rule
that memorizes exact coordinates rarely generalizes." Their effect on `heldout_accuracy` / the
memorization rate has never been isolated.

---

## 4. Real precedent from the literature (CEGIS / iterative repair / world-model code induction)

The "induce a program world-model from a few observed transitions, verify it, and repair it against
counterexamples" shape is a well-studied problem with directly-cited precedent.

- **CEGIS origin — Solar-Lezama, "Program Synthesis by Sketching" (2008)** ([thesis](https://people.csail.mit.edu/asolar/papers/thesis.pdf)):
  counterexample-guided inductive synthesis — a synthesizer proposes a candidate, a verifier returns a
  counterexample, the counterexample is added to the constraint set, repeat. Carnot's `induce → verify →
  refactor(mismatches)` loop is a direct LLM instantiation of exactly this.

- **WorldCoder — "a Model-Based LLM Agent: Building World Models by Writing Code and Interacting with the
  Environment," Tang, Ellis et al., NeurIPS 2024** ([arXiv:2402.12275](https://arxiv.org/abs/2402.12275)).
  The closest precedent to Carnot's E3: an agent builds a **Python program world model** and refines it from
  interaction. Its central claim is the one Carnot is implicitly betting on: *"Refinement can be very
  successful when the target program has many corner-cases, each of which can be inferred from a few
  examples … which is exactly the case for world models, which might need to handle a wide range of objects
  and their interactions, but typically don't demand intricate algorithms."* WorldCoder reports refinement
  is more sample- and compute-efficient than deep RL / ReAct, and that a new model can be built by
  *refining an old one* rather than from scratch (relevant to cross-level transfer). **But WorldCoder used a
  GPT-4-class model.**

- **REx — "Code Repair with LLMs gives an Exploration-Exploitation Tradeoff," Tang, Ellis et al.,
  NeurIPS 2024** ([arXiv:2405.17503](https://arxiv.org/abs/2405.17503)). The refinement-*scheduling*
  companion to WorldCoder: which candidate to refine next is an arm-acquiring bandit solved with Thompson
  sampling over a *tree* of refinements; it solves more problems in fewer LLM calls than greedy single-line
  refinement. **Directly relevant to a follow-up:** Carnot's loop refines a *single* linear chain (round 0
  → refactor → refactor); REx says a *tree* with explore/exploit scheduling beats a linear chain — a cheap
  future upgrade IF the base refinement mechanism is shown to carry signal at all.

- **Chen et al., "Teaching Large Language Models to Self-Debug," ICLR 2024**
  ([arXiv:2304.05128](https://arxiv.org/abs/2304.05128)) and **Madaan et al., "Self-Refine: Iterative
  Refinement with Self-Feedback," NeurIPS 2023** ([arXiv:2303.17651](https://arxiv.org/abs/2303.17651)):
  the general execution-feedback / self-feedback iterative-repair family Carnot's loop belongs to.

- **The E3 paper itself — "Executable World Models for ARC-AGI-3 in the Era of Coding Agents"**
  ([arXiv:2605.05138](https://arxiv.org/html/2605.05138v2), cited in the module docstring): GPT-5.5 fully
  solves **15/25** games via exactly this induce→verify→refactor→plan loop. **This is the strongest positive
  precedent — but it, too, used a frontier model (GPT-5.5), not a 9B.**

- **THE ADVERSARIAL COUNTERWEIGHT — "Falsification, Not Exposure: An Internally Preregistered
  Placebo-Controlled Decomposition of Self-Repair Feedback in Frozen Small Code Models"**
  ([arXiv:2606.31511](https://arxiv.org/pdf/2606.31511)). Preregistered, placebo-controlled finding: for
  **small FROZEN code models**, self-repair feedback does **NOT** significantly improve pass rates. Decomposing
  the feedback into (i) exposure to failing code, (ii) execution evidence, (iii) counterexample
  falsification, **only the falsification component carries measurable signal — and it functions by
  filtering/rejecting outputs, not by improving the next generation.** ("Frozen models cannot improve
  through feedback exposure alone; any apparent benefit comes from the falsification process filtering
  outputs, not from actual model adaptation.") Corroborated by "Measuring and mitigating debugging
  effectiveness decay in code language models" (Nature Sci. Rep. 2025,
  [s41598-025-27846-5](https://www.nature.com/articles/s41598-025-27846-5)), which finds debugging
  effectiveness *decays* over successive iterations.

**The precise literature tension this experiment resolves:** WorldCoder + E3 say verifier-grounded
counterexample refinement of a code world-model works *well* — with a **frontier** model. "Falsification,
Not Exposure" says self-repair feedback does *not* improve **small frozen** models (which is Carnot's
Kaggle-16GB regime). Carnot's frozen 9B (and the 27B ThinkingCap it tested tonight) sit exactly on the
disputed boundary. **No one has measured which side of that boundary Carnot's induction refinement falls
on.** That is the experiment.

---

## 5. The ONE proposed next experiment

**Title (working):** Does verifier-grounded CEGIS refinement lift world-model induction quality? — isolating
the per-round `heldout_accuracy` trajectory of the *existing* `execute_bounded_llm_reinduction` loop against
tonight's single-shot baseline, on matched games/budgets, for a 9B and a 27B frozen model.

**One-paragraph statement.** Re-run the SAME games / windows / budgets as tonight's REQ-ARC-WMTE-5726
single-shot diagnosis (the 17 games ThinkingCap induced on), but route induction through the EXISTING
`execute_bounded_llm_reinduction` CEGIS loop (`min_heldout_accuracy=1.0` so the dynamics-refactor rounds
actually fire, `candidate_provider` returning just the loaded engine, `load_engine=e3.load_engine`,
`plan_in_model=e3.plan_in_model`) instead of single-shot `proposer.induce()`. For each game/trial, capture
the **per-round `heldout_accuracy` trajectory** already recorded in `outcome.rounds[*]` (`arc_llm_reinduction.py:790`):
round 0 = tonight's single-shot baseline, rounds 1–2 = counterexample-guided refactor. Primary output:
`Δheldout = heldout(best refined round) − heldout(round 0)`, per game, pooled. Run BOTH ThinkingCap-27B and
Qwen-9B to locate the model-size threshold the §4 literature disputes. Secondarily, run the same structural
**window-memorization detector** on the induced source before vs. after refinement (count engines whose
`engine()` body hardcodes literal coordinate constants matching observed-window cells), because the
refactor prompt's "replace special cases with shared rules" instruction is the direct countermeasure to that
failure mode. **Also report the `refactor`-round code-emission rate** so a null is attributable (budget
overrun vs. genuine no-improvement).

**Why this specifically will NOT repeat tonight's null pattern.** Tonight's 7 peripheral A/Bs
(REQ-ARC-FCP-5590/5728/5729/5730/5732/5740/5756) each MODIFIED a scoring/perception component and measured
`level_gain_delta` — a near-zero-headroom outcome that returns "no delta" whenever the corpus has no
selectable headroom (the FALSE_NEGATIVE_RISK failure mode `adversarial_verify.py` warns about). This
experiment (a) targets **induction directly** — the exact mechanism the diagnosis named as the binding
wall — not a peripheral scorer; (b) measures a **continuous induction-quality metric with real headroom**
(baseline median `heldout` is 0.0; any improvement is measurable and *cannot* come back "no headroom");
and (c) exercises an **existing, wired, default-on live mechanism that tonight's harness simply bypassed** —
so it is a cheap high-leverage measurement of a real live behavior, not a new-scorer build. It is also NOT a
rerun of any prior experiment: the *mechanism* changes (refinement loop, not single-shot) and the *metric*
changes (per-round heldout trajectory, not a level-gain delta).

**Falsifiable acceptance gate.** Pre-register the game list (≥12 of tonight's 17), trials/seeds per game
(≥3, matching tonight's stochastic design), and both models. "Verifier-grounded CEGIS refinement genuinely
improves induction quality" iff ALL hold:

- **Pooled mean `Δheldout` > 0.15** across ≥12 games (per-game averaged over trials), bootstrap 95 % CI
  excluding 0;
- **positive on ≥50 % of games** (not driven by one already-solved template-match game like `sp80`/`ft09`);
- **paired sign-test** (game as unit, best-refined vs round-0) **p < 0.05**;
- **secondary:** window-memorization rate drops by **≥0.2 absolute**;
- **degradation guard:** on games already at round-0 `heldout=1.0` (`sp80`, `ft09`), refined `heldout` must
  NOT drop below round-0 (refinement must not corrupt a correct model);
- **attribution guard:** the `refactor`-round code-emission rate is reported; the primary result is only
  interpreted when emission is healthy (>0.6), else the verdict is the emission-confound branch below.

**Three honest outcome branches (pre-registered):**

1. **POSITIVE (gate met):** verifier-grounded refinement lifts a small frozen model's induction — the
   explicit-induction architecture is vindicated for Carnot's model class, and the concrete next build is
   the REx tree-scheduling upgrade (§4, [arXiv:2405.17503](https://arxiv.org/abs/2405.17503)) and wiring the
   refinement loop into the *offline dev harness* so future induction work measures refined quality by
   default. (Note: still a **partial** pipeline win — a heldout gain need not yield a level-up while the
   separate execute-stage bug persists; see §6.4.)
2. **HONEST-NEGATIVE (`Δheldout ≤ 0.05` pooled AND memorization unchanged AND emission-rate healthy):**
   verifier-grounded CEGIS refinement does **not** lift a small frozen model's ARC dynamics induction —
   corroborating [arXiv:2606.31511](https://arxiv.org/pdf/2606.31511) on this task. This is a load-bearing
   negative, not a wasted run: log a NEW `ops/verifier_gaps.md` entry (GAP-ARC-INDUCTION-REFINEMENT-NULL),
   down-weight the small-model-refinement lever, and **elevate the §2 architectural reconsideration to an
   evidence-backed decision** — run a larger *offline* induction model on the conductor's dedicated GPU-0
   3090 (permitted for offline induction per the 2026-06-27 directive) and/or demote the verifier from
   planner to filter in a Duck-style reactive loop. Do NOT retire the live loop itself (it may still help on
   the games where it fires) — record the measured bound.
3. **EMISSION-CONFOUND (`refactor`-emission rate ≤ 0.6):** the refactor rounds overran the budget and
   emitted no code, so `Δheldout≈0` is a *mechanical* artifact, not evidence against refinement. Verdict:
   "refinement efficacy untestable at this budget; fix code-emission first (a codeonly refactor variant / a
   larger `n_predict`) before re-judging." This branch prevents a budget artifact from falsely condemning
   the mechanism.

**Prior-failure block (Failed-Experiment Rerun Discipline compliance).**
- Names the priors: tonight's single-shot induction diagnosis (REQ-ARC-WMTE-5726) + the 7 peripheral-tweak
  nulls (REQ-ARC-FCP-5590/5728/5729/5730/5732/5740/5756) + the induce-completion measurements
  (REQ-ARC-WMTE-5720/5724, induce 0/12).
- Diagnosed root cause of those nulls: (a) tonight's `heldout≈0.0` is a *single-shot* number — the
  refinement loop was bypassed (§3b); (b) the 7 A/Bs measured a level-gain/completion *delta* on a
  near-zero-headroom corpus, so a null could not distinguish "component useless" from "no headroom."
- What is different: this exercises the EXISTING-but-offline-unmeasured refinement mechanism and measures a
  *continuous induction-quality* metric with real headroom, on the exact games tonight's single-shot run
  covered — a mechanism change + a metric change, not a re-tune of a peripheral scorer.
- `retire_if_same_verdict`: if this comes back HONEST-NEGATIVE (branch 2), do NOT re-propose small-model
  induction-refinement variants; the next induction attempt must change the *model class* (bigger offline
  model) or the *architecture* (reactive-with-filter), per branch 2.

**Scope / effort.** ~1 day including GPU wall-time. **Substrate: `live_llm_inference`** (real ThinkingCap-27B
and Qwen-9B GGUF generation across up to 3 rounds per game — genuine compute, 60s floor). PRECONDITIONS
block required: both GGUFs cached, and `llama_cpp.llama_supports_gpu_offload()` verified True (CLAUDE.md's
2026-07-06 CUDA-build rule — a CPU-only wheel would silently run induction on CPU). No new perception/search
infra: the harness change is small — `run_seeded_progress` swaps its single-shot `proposer.induce()`
(`arc_actions_to_progress.py:702`) for an `execute_bounded_llm_reinduction(...)` call with a trivial
`candidate_provider` and records `outcome.rounds`; the per-round `heldout_accuracy` is already computed
(`arc_llm_reinduction.py:790`). The window-memorization detector is a ~30-line regex/AST scan of the induced
`world_model.py` source. This is a diagnostic over existing machinery, not a live-path modification.

**Live-path target (Live-Path Reachability Discipline).** The mechanism under test is ALREADY the live
mechanism (`arc_competition_agent.py:3885/4005`); this experiment measures its offline dev-twin behavior. A
POSITIVE result directly improves the live path (the refinement loop is what runs live); the actionable
build is the REx scheduling upgrade to that same loop + defaulting the offline harness to measure refined
quality. No orphan module.

---

## 6. Adversarial self-critique — the most likely ways THIS also fails (up front)

Per the Phase-Prototype + Adversarial-Check discipline, a hostile-reviewer pass on my own proposal, in
order of probability:

1. **Small-frozen-model refinement may genuinely not help — the single most likely null.**
   [arXiv:2606.31511](https://arxiv.org/pdf/2606.31511) preregistered exactly this: for small frozen code
   models, self-repair feedback does not improve pass rate; only the falsification/filter component carries
   signal. If Carnot's frozen 9B/27B are in that regime, `Δheldout` will be ≈0 and the gate fails. **Why
   the experiment is still worth running:** this converts an *assumption Carnot's whole live architecture
   rests on* (that its refinement loop helps) into a *measurement*, and — uniquely — testing 9B AND 27B
   *locates the model-size threshold* rather than assuming it. A null is a real, load-bearing finding
   (branch 2), pre-registered with a concrete architectural consequence, not a wasted run. This is the
   honest heart of the proposal: it is a *fair test of Carnot's core bet*, and it might lose.

2. **The `refactor` rounds overrun the token budget and emit no code (the Qwen 34/40 overrun, worsened).**
   Tonight 85 % of Qwen's failures were `limit`+`overran` on the *codeonly* induce; `refactor` is
   deliberately NON-codeonly (it asks the model to reason about BEFORE/PREDICTED/OBSERVED, `arc_executable_world_model.py:1920-1925`),
   so it is *more* prone to overrun. If refactor emits nothing, `Δheldout=0` for a mechanical reason.
   **Mitigation:** ThinkingCap-27B is the primary arm (31/40 emission tonight); the pre-registered
   emission-confound branch (§5 branch 3) explicitly quarantines this outcome as "fix emission first," not
   "refinement useless." I instrument the emission rate precisely because this confound is likely.

3. **`min_heldout_accuracy=1.0` makes refinement chase an impossible target and give up early.**
   REQ-ARC-FCP-5699-32 already found 0/6 rounds met the 1.0 gate on g50t. **Mitigation:** the primary
   metric is the *continuous* `Δheldout` (did refactor move heldout 0→0.3 at all?), NOT the binary
   gate-met rate — measured independent of ever reaching 1.0. I keep the 1.0 threshold precisely *because*
   a lower one (e.g. 0.5) would let the loop `accept`-and-stop before refining, defeating the measurement.

4. **A heldout gain need not yield a level-up — the sp80 execute-stage bug is separate.** `sp80` t1 had
   `heldout=cell_recall=goal=1.0` yet `reached_levelup=False` with `plan_len=1` (diagnosis §6, open question;
   the plan's win-predicate fired one action in, which the real env did not treat as a level-up — under
   parallel investigation by the sp80-execute-diagnosis thread). So even a perfect induced model can fail
   downstream. **This is exactly why I scope the gate to induction-quality metrics (`heldout`/memorization),
   NOT level-ups** — per the task's guidance. A POSITIVE result here is a genuine but *partial* win: it
   removes the induction wall; the pipeline still needs the separate execute-stage fix to convert improved
   models into level-ups. I state this rather than over-claiming a level-up.

5. **The 8-mismatch cap starves the refactor of evidence.** `WorldModelVerifier.score` returns at most
   `max_mismatch=8` failing transitions (`arc_executable_world_model.py:712`); on a wildly-wrong 64×64
   engine, 8 examples may be an unrepresentative slice, so a null could be "under-fed," not
   "refinement-is-wrong." **Mitigation:** report the mismatch count fed each round; as a cheap *secondary*
   lever (not confounding the primary test), note that raising `max_mismatch` is a follow-up if branch 2
   fires with healthy emission but suspiciously sparse counterexamples.

**Net honest read.** Unlike a scorer-tweak A/B, this proposal has no delta-to-fake and no zero-headroom trap
— the baseline is a measured 0.0 median, so any real refinement signal *will* show, and any null is
*informative* (it adjudicates the WorldCoder/E3-vs-"Falsification-Not-Exposure" tension for Carnot's exact
model class). Its biggest genuine risks are (2) an emission confound and (1) a true small-model null — both
pre-registered as distinct, honest branches with concrete consequences, so the experiment produces an
actionable decision in *every* outcome, including the one where Carnot's core architectural bet loses.

---

## 7. Cross-references

- **The diagnosis this targets:** `docs/research-notes/arc-world-model-induction-quality-diagnosis-2026-07-20.md`
  (37 inductions, 29/37 `heldout=0.0`, six failure patterns);
  `results/experiment_5726_thinkingcap_16k_dualgpu_reason_ab.json` + `results/exp5726_thinkingcap_16k_dualgpu_shard.jsonl`.
- **The architecture reframe:** `docs/research-notes/arc-top-project-search-architecture-audit-2026-07-20.md`
  (winners are greedy, no lookahead; Carnot has more search; bottleneck = induction/generation, not
  selection/planning). Rigor template: `docs/research-notes/arc-action-effect-representation-redesign-2026-07-19.md`.
- **The refinement mechanism (already built):** `python/carnot/agentic/arc_llm_reinduction.py`
  (`execute_bounded_llm_reinduction:654`, `MAX_REFINEMENT_ROUNDS:30`, real-mismatch attach `:804-814`,
  `proposer.refactor` `:735`, per-round heldout `:790`, goal-repair `:606`);
  `arc_executable_world_model.py` (`refactor_prompt:1285`, `WorldModelVerifier.score:711`,
  `CARNOT_ARC_INDUCE_TRANSITIONS_K:1121`, playbook prior `:1142`).
- **Live call sites (uses the loop):** `arc_competition_agent.py:3885` (level-up, `min_heldout_accuracy=1.0`),
  `:4005` (stall, default-on), `:3986` (the prior "0/6 met 1.0 on g50t" measurement).
- **Offline harness (bypasses the loop):** `arc_actions_to_progress.py:641` (`run_seeded_progress`),
  single-shot induce `:702`, `n_refinement_rounds=0` hardcoded `:764`.
- **Literature:** [arXiv:2402.12275](https://arxiv.org/abs/2402.12275) (WorldCoder — code world-models +
  refinement); [arXiv:2405.17503](https://arxiv.org/abs/2405.17503) (REx — refinement scheduling as
  explore-exploit bandit); [arXiv:2605.05138](https://arxiv.org/html/2605.05138v2) (E3 — GPT-5.5 solves
  15/25 via induce→verify→refactor); [arXiv:2606.31511](https://arxiv.org/pdf/2606.31511) (Falsification,
  Not Exposure — small frozen models don't improve from self-repair feedback, THE adversarial counterweight);
  [arXiv:2304.05128](https://arxiv.org/abs/2304.05128) (Teaching LLMs to Self-Debug);
  [arXiv:2303.17651](https://arxiv.org/abs/2303.17651) (Self-Refine); Solar-Lezama 2008 (CEGIS origin,
  [thesis](https://people.csail.mit.edu/asolar/papers/thesis.pdf)); Nature Sci. Rep.
  [s41598-025-27846-5](https://www.nature.com/articles/s41598-025-27846-5) (debugging-effectiveness decay).
- **Gap to log on a HONEST-NEGATIVE outcome:** new `ops/verifier_gaps.md` entry
  GAP-ARC-INDUCTION-REFINEMENT-NULL; related existing entries GAP-WM-TRUST-GATE (`:2686`, the trust gate,
  correctly NOT the problem here), GAP-ARCH-FRAME-CHANGE-PREDICTOR (`:2623`).
- **Corroborating memory:** `project_arc_actions_to_progress_metric` (bottleneck = dynamics-induction;
  heldout uniformly 0.0), `project_arc_live_agent_learning_gaps`, `feedback_arc_value_is_process_not_weights`,
  `project_arc_live_generator` (the frozen Qwen3.5-9B live stack), `reference_zendoworld_hypothesis_uncertainty`
  (generation-not-selection corroboration).
