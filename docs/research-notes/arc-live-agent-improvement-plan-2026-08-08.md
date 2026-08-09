# ARC Live-Agent Improvement Plan — 2026-08-08

**Origin.** Operator request: "Given the state of our live agent and our previous experiments to
improve our score, where do we head next? Build me a plan that I can follow to try the best
remaining possibilities." This plan synthesizes three parallel investigations run 2026-08-08:

1. An analysis of the five locally-cloned leaderboard/reference projects under `external/`
   (duck-harness, arc-m1-2nd-reki, arc-m1-3rd-forge, ARC-GEN, paperbanana).
2. A full ledger of our own experiment record: closed axes, built-but-unmeasured levers,
   in-flight conductor A/Bs, and open gaps.
3. A ranked sweep of the whitepaper/idea backlog against the measured wall.

Everything below cites the artifact or source line that supports it. Nothing here re-proposes a
closed axis (see "Guardrails" at the end).

---

## The three findings that shape this plan

### Finding 1 — the score formula rewards DEPTH first, efficiency second

Verified against `external/duck-harness/tufa-arc-agi-framework/src/taaf/game.py:381`. Per level:
`min(115, (baseline/actions)^2 * 100)`, weighted by level index. The final score is capped at
`max_weights / total_weights * 100`, where `max_weights` sums only levels actually COMPLETED.
In the Duck's own bundled 500-run set, 114 of 252 scoring passes (45%) land exactly on a
completion cap — extra efficiency on those runs was worth zero.

Consequence: at our ~0.08-class hidden score, we likely complete zero or one level per game. The
cap is therefore near zero, and pure efficiency work cannot raise it much. **Levers that help
complete one more level dominate levers that shave actions on levels we already complete.** The
efficiency levers already measured (trajectory transfer, budget-aware search) are still worth
shipping — they are paid for and they compound — but new investment goes to depth.

### Finding 2 — part of the measured wall is a measurement artifact, and the fix is built but OFF

The binding constraint on record (0 of 296 clean engine-units reach held-out
`change_accuracy >= 0.5`, hostile-reproduced; corrections C2/C4 attached) is a statement about
the local single-shot GGUF inducer. But `python/carnot/agentic/arc_executable_world_model.py:923`
says in its own comment: "`cell_recall`'s change mask is dominated by counter cells rather than
by game state. Part of the measured median-0.0 trust score is therefore a MEASUREMENT ARTIFACT,
not purely a capability wall." Three repairs exist and ship default OFF:

- `SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED` — compare-time HUD masking.
- `SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED` — symmetric change-fidelity gate (current metrics
  are recall-only: structurally blind to cells the engine INVENTED; 21 of 33 audited rows are
  runaway writers with 3-10x spurious cells).
- The hidden-state branch coverage fix (REQ-ARC-WMTE-6013) — the hidden-state branch covers
  every 0.08-wall game and never calls the change gate at all.

The 0/296 closure has never been re-derived under honest metrics. Re-deriving it is the cheapest
possible attack on the wall: a re-scoring pass over already-banked engines, no GPU, no new data.
A pre-registered null HARDENS the closure; a positive reopens the axis on measurement grounds,
not prompt-phrasing grounds.

### Finding 3 — the /think result was under-read, and the true effect is a lower bound

`results/experiment_6199_gemma_think_mode_ab.json` (14,478s live A/B, 10 games) headlined
`no_think_higher_winrecognition_1_to_2`. But on **held-out exact accuracy — the metric the live
admission gate actually uses** — think went **4 wins, 0 losses, 6 ties**, and moved sp80 and
vc33 from 0.000 to exactly 1.000, i.e. across the live admission bar (mean 0.10 -> 0.37). At 4
discordant games the minimum reachable p is 0.125: the result hit the significance floor
available to it. It was maximally concordant, not weak.

And the think arm was handicapped twice, both biasing AGAINST it:

- The chat-completions route silently drops `repeat_penalty`/`repeat_last_n`
  (`arc_executable_world_model.py:6451-6455`) — repeat_penalty alone was measured to raise
  usable engines from 13/36 to 22/36.
- Two of the three induce prompt shapes append a reasoning-suppressing pre-opened code fence
  UNCONDITIONALLY (goal-only at `:6892`, split-induce fallback at `:6976`); only the combined
  prompt gates it on `induce_think_on()`. Think is default ON in the scored path today
  (`ARC_LIVE_GENERATOR_THINK_SCORED_DEFAULT = "1"`, flipped by commit `21e44408ab`), so on the
  scored path one of three prompt shapes reasons and two do not.

External corroboration: GPT-5.6 on ARC-3 scores ~26x higher at max reasoning effort vs low
(7.8% vs 0.3%) — the strongest external signal that reasoning effort is THE lever on ARC-3
specifically, and it matches our own generation-not-selection diagnosis.

---

## The plan

Phases are ordered by cost and by dependency. Each item names its falsifiable gate. Every A/B
follows the standing harness discipline: treatment-activation pre-flight, A/A control, per-game
clustering, pre-registered metric before the shot.

### Phase 0 — unblocks and repairs (hours, no GPU)

**0a. Make exp6216 admissible; rerun the portfolio experiment.**
exp6216 (budget-aware search) posted the strongest lever numbers on record (deadline misses
6 -> 0, efficiency gain 7.0, zero harm, mutation-proven) and is inadmissible ONLY because its
artifact carries a `DURATION_TOO_SHORT` adversarial flag caused by a non-canonical
`inference_substrate` string plus a missing `random_seed`. The run never loaded a model; the
honest canonical substrate is `verifier_ensemble_against_cached_candidates` (floor 1s, which its
1.32s clears). Declare the substrate honestly, add the seed, apply the same repair to exp6214,
then rerun exp6218 (the lever-PORTFOLIO experiment, which skipped with "fewer than two eligible
levers"). This is minutes of work standing between us and our first measured lever-interaction
result. *Gate: exp6218 runs with >= 2 eligible levers and reports a portfolio verdict.*
**DONE (2026-08-08, commit `9aa0728676`):** both substrates corrected, seeds added, exp6218
re-run twice (a pre-existing, unrelated `honest_verdict` terminal-prefix bug in exp6218 was found
and fixed along the way). A separate structural gap was found and filed, not fixed: a corrected
artifact's fabrication-gate quarantine cannot currently be cleared by the lever-portfolio gate,
because `corrigendum_pending` is permanent-once-populated per `determination_preservation_lint.py`
while `terminal_artifacts.py`'s `_artifact_flagged()` treats any non-empty `corrigendum_pending` as
still flagged. See `ops/known-issues.md`'s 2026-08-08 Phase 0a entry for the full reasoning.

**0b. Close the think-routing gaps (the cheapest scored-path improvement available).**
Gate the two unconditional reasoning-suppressing fences (`arc_executable_world_model.py:6892`,
`:6976`) on `induce_think_on()`, matching the combined prompt's existing pattern. Fix the
chat-completions route dropping `repeat_penalty`/`repeat_last_n`. Fix the
`use_chat_template=True` + codeonly + `stop=["```"]` truncation interaction. Refresh the three
stale "default OFF" comments (`:6485`, `:6948`, `arc_competition_agent.py:7852`). Think's
measured edge is specifically in INDUCTION quality — exactly what the two ungated shapes govern.
*Gate: all three induce prompt shapes reason when think is on; unit test pins the gating.*
**DONE (2026-08-08, commit `e587931bd0`):** both fences gated on `induce_think_on()`, repeat_penalty/
repeat_last_n threaded through `_chat_complete_request`, the `use_chat_template`+codeonly
truncation interaction documented as real but dormant (not reachable on the live path today, so
left as a comment rather than a redesign), all three stale comments corrected. 6 new tests (15/15
in the file passing). A broader regression sweep found 41 pre-existing failures across 7 files
(confirmed via `git stash` clean-HEAD comparison — none newly introduced; the true blast radius of
an already-filed bug class was larger than previously measured), documented in `ops/known-issues.md`.

**0c. Fix the two broken default-OFF levers so future A/Bs don't retire them for wrong reasons.**
`CARNOT_ARC_GRADED_GOAL_BIAS`: the "win-state exemplar" is provably the current level's OPENING
board, making the bias an inverted gradient pulling search back toward the start
(`arc_competition_agent.py:5052-5058, 5200-5323`). `CARNOT_ARC_ACTIVE_PROBE`: plan/pi bookkeeping
asymmetry — post-probe reinduction either skips a new plan's first step or raises IndexError.
*Gate: regression tests reproducing each defect, then green.*
**DONE (2026-08-08, commit `507f072d74`; REQ-ARC-WMTE-6231 / REQ-ARC-WMTE-6232):** the win-state
exemplar now reads the last-admitted transition's pre-action grid instead of the post-transition
frame; `_induce_and_plan` resets `self.plan`/`self.pi` at entry, plus a bounds guard on the induce
branch as defense in depth. 4 new regression tests confirmed to fail on pre-fix source (via `git
stash` comparison) and pass on the fix; the broader `arc_competition_agent` test surface (90 tests
across 7 files) stayed green. Both fixes are no-ops today since both env vars default off.

**0d. Re-derive the completion-cap arithmetic for OUR hidden runs.**
Pull per-game level-completion counts from our Kaggle scorecards. Compute how much of our score
is completion-capped versus efficiency-limited. This one number decides the standing budget
split between depth work and efficiency work. *Gate: a one-page note with the per-game cap
table; decision recorded.*
**DONE (2026-08-08):** full note at
`docs/research-notes/arc-completion-cap-vs-efficiency-2026-08-08.md`. Using the real
`arc_agi.scorecard` formula (a level scores 0 if not completed, otherwise
`min((baseline/agent)^2*100, 115)`, weighted-averaged per game by level index) against our own
25-game level-count structure: the full efficiency range (worst plausible to the formula's own
maximum bonus) moves a game's score by under 1.7x, while going from "reach level 1" to "reach
level 1 and level 2" roughly triples it at the same first-win rate. **Decision: the standing
budget favors depth (reach one more level) over efficiency (reach the same level in fewer
actions) by roughly an order of magnitude in expected score impact.** Honest caveat recorded in
the note: this model, even at a 100% first-win rate with zero deepening, cannot fully reproduce
the observed 0.08 hidden score on our own game-count structure — matching it requires most games
to reach level 2 and some level 3, deeper than our documented "live multi-level rate ~0" belief.
That gap does not change the decision (a calibration that closed it would require MORE depth, not
less), but it is flagged as unresolved rather than papered over.

### Phase 1 — re-measure the wall; ship the paid-for wins (one CPU pass + one short GPU pass)

**1a. Re-derive the 0/296 closure under honest metrics (Finding 2).**
Flip HUD masking + the symmetric change gate + the hidden-state branch fix ON in a MEASUREMENT
pass (not the scored default), re-score the already-banked engine corpus, and report how much of
the wall survives. Pre-register before running: this is not gate-threshold tuning (closed —
thresholds 1.0 down to 0.6 admit the identical 9 rounds); it is repairing metrics the source
itself documents as contaminated (recall-only + HUD-dominated), then re-deriving the baseline
the closure rests on. *Gate: the corrected wall number, published either way. A null hardens the
closure; a positive quantifies how much induction capability was being mis-scored.*
**DONE (2026-08-08, `results/arc_wall_rederivation_20260808.json`, REQ-ARC-WMTE-6233; no code
change, no new compute):** the taxonomy's own recovery script for its 296-unit corpus was never
committed and cannot be re-derived byte-for-byte, so this reused an already-executed,
comparable-scope measurement instead: `experiment_6011_world_model_change_gate_four_arm.json`'s
four-arm matrix already re-scored 75 real on-disk engines (`results/arc_e3_origin_fixtures`, 25
games x 3 seeds) with `mask=1|gate=1` — HUD masking ON plus the symmetric `change_gate_decision`
call, the identical function REQ-ARC-WMTE-6013 wires into the hidden-state branch. Of 69 eligible
(n_changing>=3) engines: **0 pass under the corrected reading, identical to 0 under the naive
(unmasked, gate-disabled) reading.** HUD masking measurably narrows near-miss cases (best
change_fidelity 0.062 -> 0.145, matching the independent 2026-08-01 finding that masking is real
but small) without flipping a single verdict. **The closure hardens** — corrected metrics do not
reopen the axis on this corpus. Honest caveat: this corpus is one of the taxonomy's own
provenance-unproven families (frozen, but induction-time purity unconfirmed); per the taxonomy's
own reasoning, contamination can only inflate a score, so a zero here is not weakened by that gap.

**1b. Promote trajectory transfer (exp6215).**
The one lever that passed BOTH quality and safety in the portfolio audit (fired 4/4 games,
verifier accepted 4/0, 4 LLM induction calls avoided, zero harmful regressions,
promotion-ready score 1.0). Still default OFF. Run the live-path A/B (actions-to-first-solve,
no-solve-regression floor), then flip `SUBMITTED_OBJECT_RELATIVE_TRAJECTORY_TRANSFER_ENABLED`.
*Gate: no regression on solve rate; actions-to-first-solve improves or holds.*

**1c. Re-measure budget-aware search cleanly; promote if it holds.**
After 0a's repair, one unflagged re-run under the same pre-registration. If deadline-miss
elimination (6 -> 0) reproduces, flip `BUDGET_AWARE_SEARCH_ENABLED` (`arc_solver_kit.py:5177`).
*Gate: same as its own prereg; unflagged artifact.*

### Phase 2 — the think shot (one to two GPU nights; the biggest single bet)

**2a. De-confounded /think A/B, pre-registered on ADMISSION RATE.**
After 0b removes both confounds, re-run think vs no-think on >= 16 games. Primary metric:
fraction of induction attempts whose engine clears the live admission gate (held-out exact
accuracy 1.0), clustered per game, exact two-sided sign test. Power arithmetic pre-committed:
below 6 discordant games p < 0.05 is unreachable — same arithmetic as the 2026-08-03 prereg.
Run the generator through the NATIVE llama-server path: exp6212 records
`recovered_by_server_canary: true` — the native-server canary survives the still-unidentified
llama-server reaper that kills the `llama_cpp` Python-binding path, and that reaper is currently
spinning the conductor's own think A/B (exp6217, three GATE_BLOCK commits in a row). Wall-clock
cost is real (induce mean 110s -> 715s under think) but the operator directive is explicit that
local wall-clock is a dev artifact, not a live-budget violation.
*Gate: >= 6 discordant games favoring think at p < 0.05 -> think stays on and 2b unlocks;
fewer -> record honestly, think's default reverts to the exp6199 evidence base alone.*

**2b. (Conditional on 2a or 1a moving the distribution) Best-of-N engine sampling.**
Selection is measured NOT to be our bottleneck, and the held-out gate is an oracle-distinct
selector already in place — so parallel diversity (distinct from the CLOSED sequential-refinement
axis) becomes cheap to exploit the moment the single-shot distribution shifts. The repeat_penalty
result (13/36 -> 22/36 usable from sampler diversity alone) supports the mechanism. High GPU cost
(N x induce, worse under think) — hence conditional. *Gate: pre-register N; admission rate at N
must beat N=1 or the lever retires.*

### Phase 3 — representation upgrades to the inducer (days; operator-authorized lever #1 territory)

This attacks the generation wall at the INPUT-representation level. It is not prompt phrasing
(closed); it is the unfinished remainder of the object-centric-input lever the 2026-08-07
operator directive authorizes, informed by what the reference stacks actually do.

**3a. Wire the already-ported Duck segmentation into the live induce path.**
Verified gap: `object_hash` (translation-invariant sha1 of color + bbox-normalized cells) and
`blob_topology` (containment tree via complement flood-fill + 4-adjacency) were ported into
`python/carnot/agentic/arc_color_blob_salience.py:548,571` — credited in the file's own
docstring to a top-3 competitor's open-sourced code — and the live scored agent NEVER imports
them (`arc_competition_agent.py:37` imports only `ColorBlobSaliencePrior` and
`connected_color_blobs`; every `blob_topology` caller is an offline experiment). These are the
Duck's PRIMARY board representation. Feed the induce prompt: per-object hash identity across
frames, containment, adjacency, plus the per-transition changed-cell COUNT as explicit evidence
(both notebook stacks converged on "trust numeric transitions over visual guesses"), plus
semantic action names (UP/DOWN/LEFT/RIGHT/SPACE/MOUSE instead of ACTION1..6 — grep found no
semantic naming anywhere in our prompts; an opaque integer triggers none of a general model's
priors). NOTE: exp6214 (object-DELTA form) came back p=0.625 with three harmful regressions —
this is a DIFFERENT construction (identity/topology table, not deltas), and the prereg must name
exp6214 as the differentiated prior attempt. *Gate: held-out change fidelity (HUD-masked,
symmetric, per 1a) on a leave-one-game-out split; >= 4 of 5 held-out games improve, no live
admission-rate regression.*

**3b. (Optional arm inside 3a's A/B) Withhold the raw numeric grid.**
The Duck deliberately refuses to expose the raw grid, forcing object-level reasoning, keeping a
small-crop ASCII escape hatch. Near-zero cost as a third arm in 3a's A/B.

### Phase 4 — depth levers (bigger builds; start only after Phases 0-2 read out)

Ranked by evidence strength. Do not start all four; pick by what Phases 1-2 revealed.

**4a. CNN dynamics prior: changed-REGION target + trust() unlock.**
The strongest orphaned asset in the repo (`docs/research-notes/arc-pretrain-prior-transfers-2026-06-21.md`):
a prior trained on 20 public games transfers to 5 UNSEEN games — warm changed-cell recall 0.5485
vs the LLM inducer's 0.0033, warm-start wins 5/5. It is on the live path (`models/arc_dynamics_prior.pt`)
but `trust()` gates it OUT on unseen games (warm 0/5 by exact-match) "even though the dynamics
genuinely transferred" per its own docstring. Two named unbuilt levers: predict the changed
REGION (object/delta encoding) instead of per-pixel exact match, and within-region recursion for
iterative mechanics. Retrain over the existing 14.6k-transition corpus; the RAM leak that blocked
GPU pretraining is fixed. *Gate: leave-one-game-out held-out change fidelity (per 1a metrics) up
on >= 4 of 5 games; live admission rate not regressed.*

**4b. Extend structured mechanic-class inducers beyond navigation.**
`InducedNavWorldModel` (behind `CARNOT_ARC_STRUCTURED_NAV=1`) cleared a five-game scored gate:
+1 level, zero regressions — a rare positive. Only the navigation class exists. Add the next
mechanic class(es); the template, detector, and pre-LLM cascade slot all exist. Attacks Mode A
(12 of 33 rows: the engine predicts nothing ever changes). *Gate: class-fit + plan-found, total
banked levels up, zero regressions on non-family controls — the same gate shape the nav work
pre-registered.*

**4c. Mode B attack: change-magnitude prior + off-path drift falsifier.**
The 2026-07-29 admission-bottleneck note splits failures into two modes wanting OPPOSITE
corrections and names a sparsity/change-magnitude constraint on predicted writes as "cheap to
test" — never tried. The drift falsifier (arXiv:2511.06136 family) is the unbuilt survivor of
the object-probe ingestion. Both are constraints, not planners. CPU-side over existing fixtures.
*Gate: spurious-to-true changed-cell ratio and noop_hallucination_rate down vs recorded
baselines (up to 10.3x spurious on sc25@0, hallucination 1.0 on cn04@0).*

**4d. Bounded re-induction A/B (`CARNOT_ARC_BOUNDED_REINDUCTION`).**
The one genuinely built-and-never-measured lever (shipped 2026-08-08, commit `e73efa7c85`): the
scored path latches induction OFF after one attempt per level, then spends up to ~1,900 actions
accumulating evidence the LLM can never see. The always-on witness fields
(`induction_tier_latched_off`, `transitions_since_last_induction_attempt`) quantify the waste
for free on the next scored run. Run this A/B AFTER 2a settles think, because re-induction's
value depends on the per-attempt success rate think sets. *Gate: banked levels or admission
rate up under the flag; no action-budget regression on games that never stall.*

**4e. (Reserve bet) Real Python REPL sandbox on the 31B generator.**
The Duck's core architecture — arbitrary-Python sandbox with mid-program `action()` round-trips —
versus our five-function dispatch table (`arc_tool_loop_lookahead.py`), which its own docstring
calls a deliberate scope-reduction FOR A 9B MODEL. The existing null A/B ran on Qwen3.5-9B,
budget-short by its own verdict (FALSE_NEGATIVE_RISK: uninformative). Three factors have changed:
31B generator, gVisor sandbox in-house (`carnot.autoresearch.sandbox`), and the Duck's portable
576-line reference implementation. A rerun REQUIRES a `prior_failures:` block naming all three.
Caution: the Duck's own results are 0 wins in 500 runs (median 0.07) — port the architecture as
a hypothesis, not as a proven winner. *Gate: pre-register levels-completed on a >= 10-game
offline set vs the current cascade; retire on parity or worse.*

### Standing infrastructure item (parallel, not gating)

**The unidentified llama-server reaper.** Blocks the conductor's exp6217 today, will recur on any
live generator run. Everything obvious is already ruled out BY MEASUREMENT (OOM, janitor, CUDA
guard, slot arithmetic, name-matching, process groups, SIG_IGN). Workaround exists (native
llama-server canary path). The next diagnostic step nobody has taken: audit-level syscall
tracing on the sender (auditd rule on kill(2) targeting the server PID, or eBPF/bpftrace on
signal delivery) during a deliberate reproduction. Fold into whichever Phase-2 GPU night runs
first.

---

## What we explicitly do NOT do (guardrails)

- No prompt-phrasing variations on the single-shot inducer (closed 2026-08-03; hostile-reproduced).
- No refinement/CEGIS re-runs on the current corpus (closed twice; exp6091 complete-data
  confirmation: 0 of 11 discordant, p=1.0).
- No re-use of the human-replay corpus as a level-bank lever (exhausted; imitation made solve
  rate WORSE), no MATM retrieval, no CoEx landmarks re-run without a real differentiator, no IGE
  granularity tuning, no energy-as-RFT-teacher (9-agent audit NO), no TRM training
  (operator-killed), no PoE-World blending (selection beats blending on deterministic
  transitions), no gate-threshold tuning in isolation (identical 9 rounds at every threshold).
- No source reading of hidden games, ever (operator directive; also inviable — frames-only
  remote gateway).
- Every lever ships default OFF until its pre-registered A/B passes; every A/B carries a
  treatment-fire pre-flight and an A/A control; artifacts declare canonical
  `inference_substrate` values and carry `random_seed` (the exp6216 lesson).

## Sequencing summary

Week 1: Phase 0 (all four items) + Phase 1 (1a metric re-derivation; 1b/1c promotions).
Week 2: Phase 2a think shot on the de-confounded stack; 0d's cap table decides how much
Phase-3-vs-Phase-4 budget follows. Then Phase 3a/3b representation A/B.
After that: at most two of Phase 4's levers, chosen by what 1a and 2a revealed — 4a/4b if the
wall partially dissolved under honest metrics (depth is buyable), 4c/4d if it held (constrain
the writer, spend attempts better).

The record's own honest framing, which this plan accepts: across 44 catalogued levers the
evidence supports COMPOUNDING MEASURED GAINS, not a step change — with two exceptions that
genuinely could be step changes and are therefore front-loaded here: the honest-metric
re-derivation of the wall (1a) and the de-confounded think shot (2a).
