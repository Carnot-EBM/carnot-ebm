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
**DONE (2026-08-08, REQ-ARC-WMTE-6234):** the conductor ran the live-path A/B same-day, ahead of
this session reaching it (`[conductor] Within-game object-relative trajectory-transfer causal
A/B`, commit `104405d43b`; `results/experiment_6215_arc_object_relative_trajectory_transfer_ab.json`,
`status: complete_ready`). Result reused rather than re-run: 4/4 fired, verifier accepted 4/0, 0
harmful regressions, `treatment_minus_control_actions: 0` and `_score: 0.0` on all 4 games
(actions-to-first-solve held exactly), `promotion_ready_score: 1.0`, mutation-proven, live
entrypoint confirmed. Gate met — flipped `SUBMITTED_OBJECT_RELATIVE_TRAJECTORY_TRANSFER_ENABLED`
to `True`. One pinning test updated (`test_arc_trajectory_transfer_cascade.py`'s explicit-off
test now sets the env override explicitly rather than relying on the old bare default; a new test
pins the new bare-default value). 31+94+37 tests across directly-touching and collateral files
pass.

**1c. Re-measure budget-aware search cleanly; promote if it holds.**
After 0a's repair, one unflagged re-run under the same pre-registration. If deadline-miss
elimination (6 -> 0) reproduces, flip `BUDGET_AWARE_SEARCH_ENABLED` (`arc_solver_kit.py:5177`).
*Gate: same as its own prereg; unflagged artifact.*
**DONE (2026-08-09, REQ-ARC-WMTE-6235):** re-ran fresh (not metadata-patched) now that the
source script emits the canonical substrate directly. Reproduced identically: deadline misses
6 -> 0 across 6 games, 0 harmful regressions, mutation-proven. Flipped
`BUDGET_AWARE_SEARCH_ENABLED` to `True`. The fresh re-run's `build_artifact()` call silently
dropped the artifact's prior corrigendum record (no memory of past review across a full
rebuild) -- `determination_preservation_lint.py` caught it before any commit; fixed by restoring
the five determination fields from git history plus a new re-verification note, and recomputing
the checksum. Tests updated in two files (bare-default pin flip, one promotion-score expectation
recalculated to `5/6` since one of its own checks is now permanently unsatisfiable
post-promotion by design); a third file's tests needed no changes once the corrigendum was
restored. 181-test collateral sweep passes.

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

**DONE (2026-08-09, REQ-ARC-WMTE-6242, negative — gate NOT MET).**
`scripts/run_exp6199_expanded_roster.py` ran all 16 games (4 attempts, 2 llama-server reaper
recurrences + one external-timeout cutoff, all recovered cleanly via checkpoint, zero data loss)
-> `results/experiment_6221_gemma_think_mode_ab_expanded_roster.json`.
`scripts/analyze_exp6221_admission_rate.py` computed the pre-registered metric: of 13 comparable
games (3 window-only/error-excluded), 1 favors think (sp80), 0 favor no_think, 12 tie — 11 of
those 12 ties are BOTH ARMS AT THE ACCURACY FLOOR (0.0), so this is a low-power, largely
floor-driven null, not clean evidence think never helps. Sign test p=1.0 — gate not met, 2b does
not unlock. Bonus finding: cross-referencing the pre-confound-fix exp6199 artifact shows vc33's
think arm scored `heldout_accuracy=1.0` (admitted) BEFORE the fix and `0.0` (floor) AFTER — direct
confirmation the confound this session's own Phase 2a fix removed was inflating think's apparent
advantage. sp80 is the one result that replicates across both runs. See REQ-ARC-WMTE-6242 for
full detail.

**2c. (New, operator-directed, beyond this plan's original scope) Conditional think-arm
fallback.** Not "always run both" (2b, still locked) or "prefer think generally" (the sign test
found no such edge) — a narrower lever the per-game data specifically supports: if the CURRENT
arm produces no engine at all (`heldout_accuracy is None`, sp80's no_think shape: `"local model
code unusable after 3 tries"`), retry once with the other arm before giving up. Excludes the 11
of 13 floor-tie games (both arms already at 0.0 — retrying buys nothing there). Direction reads
`e3.induce_think_on()` at call time rather than hardcoding a side, so it stays correct against
whichever arm is currently primary (today: think, per the 2026-08-08 operator flip).
**DONE (2026-08-09, REQ-ARC-WMTE-6243) — built and tested, default OFF pending its own live-path
A/B, same standing convention as every other default flip in this plan.**
`E3AgentPolicy._execute_bounded_llm_reinduction_with_arm_fallback` wraps both
`_induce_and_plan` call sites; env-var toggle is restored in a `finally` block (the exact hazard
class REQ-6242's confound-fix bonus finding was about). 8 new unit tests + 59-test collateral
sweep pass unchanged (byte-identical no-op when the flag is off). Also resolved, by reading the
code rather than building infra: gemma-4's think/no_think split is a PROMPT choice against the
already-running server (`/no_think` is inert on gemma-4; the real switch is reasoning-engaged
chat template vs reasoning-suppressed codeonly), not a model-load/server-restart difference — so
the sequential single-retry design pays no teardown cost on this model already, and the L4x4
dual-preload idea that prompted asking about it is better scoped as a SEPARATE, not-yet-built
lever (concurrent arm racing to cut wall-clock, not to avoid setup cost).

**CORRECTION (2026-08-09, same day, later).** The "cd82/ls20/sk48 mirror shape" claim above is
retracted. Operator asked whether the think crashes were context exhaustion; reading the server's
own stderr log showed zero tokens generated and two back-to-back SIGINTs mid-generation (this
project's pre-existing "reaper" issue, not context exhaustion, not VRAM — KV cache is already
q8_0, no OOM/CUDA-error anywhere in the log). Reran all three cleanly after stopping
`carnot-conductor.service` + `carnot-orphan-cleanup.timer` as an isolation test (3/3 succeeded on
the first attempt, vs 4/4 crashes while those units were running — suggestive, small-n, see
known-issues.md). All three now measure a genuine `heldout_accuracy=0.0` floor tie with no_think.
sp80 is the ONLY real total-induction-failure case in this corpus; the lever's design is
unaffected, but its evidence base narrows from "two complementary failure shapes" to one clean
example. The admission-rate sign test result itself is unchanged (still 1/0/12, p=1.0, gate not
met) — `0.0` was never going to cross the `1.0` bar regardless of which side crashed. Full detail
in REQ-ARC-WMTE-6242/6243's own corrections in spec.md.

**2b. (Conditional on 2a or 1a moving the distribution) Best-of-N engine sampling.**
NOT UNLOCKED — 2a's gate was not met (see above).
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
**PARTIALLY DONE (2026-08-09, REQ-ARC-WMTE-6241) — built and tested, measurement deferred.** This
item's own stated premise was stale: `objects_block`/`_object_perception_on` already exist and
have been default ON since 2026-08-07 (REQ-ARC-WMTE-5830) — the live agent DOES already import the
segmentation. The genuinely remaining gap, confirmed by reading the current code rather than the
plan's own (outdated) grep: no explicit changed-cell COUNT, no computed object-identity linkage
across frames (only a text hint), no semantic action names. Built all three behind one new flag
(`SUBMITTED_INDUCE_PROMPT_ENRICHMENT_ENABLED`, default OFF) as two purely additive functions —
neither `_transitions_block` nor `objects_block` was modified. 12 new tests (byte-identical when
off, correct content when on, action 7 deliberately left unnamed). GPU 1 was occupied by Phase 2a
at implementation time, so the actual leave-one-game-out A/B this item's gate requires is deferred
to a follow-up task, not skipped or fabricated.

**DONE, NEGATIVE (2026-08-09, later, REQ-ARC-WMTE-6246) — gate NOT MET; flag stays default OFF.**
5-game A/B (m0r0, ft09, tr87, cn04, ar25), HUD-masked held-out change fidelity per 1a's own
methodology (`hud_mask_enabled=True` forced explicitly — a real bug the smoke test caught before
the full run, since the ambient default silently scores unmasked). Only 4 of 5 comparable (m0r0's
`on` arm hit a genuine 1575s timeout, not a crash, not fabricated as a tie). Of the 4: tr87
improved substantially (+0.33), cn04 marginally (+0.01), ft09 tied (unmeasurable — HUD mask
refused as swallowing), **ar25 regressed substantially (-0.67)** — a real, larger-magnitude
regression than tr87's gain. Gate fails for two independent reasons: the sample is short of the
gate's own >=5 floor, AND the available signal is mixed-to-negative regardless (a hypothetical
clean 5th data point could not alone satisfy ">=4 of 5 improve" without also explaining away
ar25's regression). Ran isolated from the conductor's shared engine store
(`CARNOT_ARC_E3_DIR` pointed at a private scratch dir) to avoid the known
`project_arc_engine_store_regression` overwrite hazard. `SUBMITTED_INDUCE_PROMPT_ENRICHMENT_
ENABLED` stays `False`. Full breakdown in REQ-ARC-WMTE-6246.

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

**REDIRECTED (2026-08-09) — premise checked and found stale before building; see REQ-ARC-WMTE-6244.**
This item's stated premise ("trust() gates it out") does not hold: `gated_engine_from_transitions`'s
cell-recall gate already passes in production (sp80, `heldout_cell_recall=0.98-1.0`), and a
much deeper existing investigation thread (REQ-ARC-FCP-5699-14 through -20) already root-caused
and exhaustively tested the REAL downstream bottleneck — a trusted, gate-passing dynamics model
still yields no plan, traced to zero-gradient goal-energy for never-completed levels, unfixed by
5x budget or a novelty-energy fallback. Retraining an already-98%-accurate model is unlikely to
move a bottleneck that persists at near-perfect accuracy. Redirected to Phase 4b's Mode A
question instead (see REQ-ARC-WMTE-6244 below) — a CPU-only characterization the 2026-07-29
admission-bottleneck note itself named as the next step. Finding: FOUR distinct root causes
across ar25/ka59/re86/cd82 (a code-generation bug, memorization of induce-time coordinates, a
wrong click-vs-keyboard action-modality assumption, and an incomplete/give-up implementation) —
none of them a prediction-quality problem, which further confirms retraining the CNN was the
wrong lever. Bonus check: cd82 superficially passes the shipped nav template's confidence gate
but scores 0 held-out exact-match, reproduced across two independent samples — a genuine
false-positive gap in `is_confident_nav`, not a hidden win. No code shipped; diagnosis only.

**4b. Extend structured mechanic-class inducers beyond navigation.**
`InducedNavWorldModel` (behind `CARNOT_ARC_STRUCTURED_NAV=1`) cleared a five-game scored gate:
+1 level, zero regressions — a rare positive. Only the navigation class exists. Add the next
mechanic class(es); the template, detector, and pre-LLM cascade slot all exist. Attacks Mode A
(12 of 33 rows: the engine predicts nothing ever changes). *Gate: class-fit + plan-found, total
banked levels up, zero regressions on non-family controls — the same gate shape the nav work
pre-registered.*

**PARTIALLY DONE (2026-08-09, REQ-ARC-WMTE-6244) — diagnosis, not a new mechanic class.**
Mode A turned out to be FOUR distinct root causes (ar25 crashes on a codegen bug, ka59 memorizes
induce-time coordinates, re86 assumes the wrong action modality, cd82 gives up on keyboard
actions), not one shared gap a single new mechanic-class template would close. Tried the cheapest
adjacent fix — hardening the EXISTING nav template's confidence gate to reject cd82's
false-positive fit (REQ-ARC-WMTE-6245) — and found it does not work at this project's typical
transition-sample sizes (neither exact-match nor changed-cell recall discriminates the real nav
game from the false positives); reverted before committing, negative result recorded. Building an
actual new mechanic-class detector for ka59 (push-block) or re86 (toggle/move) remains open and
unstarted — this diagnosis narrows WHAT to build for each game rather than picking one blindly.

**4c. Mode B attack: change-magnitude prior + off-path drift falsifier.**
The 2026-07-29 admission-bottleneck note splits failures into two modes wanting OPPOSITE
corrections and names a sparsity/change-magnitude constraint on predicted writes as "cheap to
test" — never tried. The drift falsifier (arXiv:2511.06136 family) is the unbuilt survivor of
the object-probe ingestion. Both are constraints, not planners. CPU-side over existing fixtures.
*Gate: spurious-to-true changed-cell ratio and noop_hallucination_rate down vs recorded
baselines (up to 10.3x spurious on sc25@0, hallucination 1.0 on cn04@0).*
**DONE — negative result for the change-magnitude prior half (2026-08-09, REQ-ARC-WMTE-6240;
the drift falsifier half was not attempted).** Re-derived the same 21 Mode B cells from the same
frozen fixtures the 2026-07-29 note measured, capped each engine's per-transition write count at
the max seen in its own shown portion. The raw counts look like a win (13/21 cells improve their
spurious-to-correct ratio) but an honest per-cell classification shows why that reading is wrong:
11/21 cells collapse to a PURE NO-OP (correct AND spurious both hit 0 -- Mode B turned into Mode
A, not fixed), 8/21 are inert, and only 2/21 show the intended trim-excess-keep-genuine effect.
**Retires the naive hard-cap construction specifically** (a single threshold with no partial
credit is too brittle once later-episode transitions are genuinely larger than the shown
prefix's); does not retire the general idea that Mode B is constrainable — a graded penalty
rather than a hard cutoff remains untested and open. No code was shipped; analysis only.

**4d. Bounded re-induction A/B (`CARNOT_ARC_BOUNDED_REINDUCTION`).**
The one genuinely built-and-never-measured lever (shipped 2026-08-08, commit `e73efa7c85`): the
scored path latches induction OFF after one attempt per level, then spends up to ~1,900 actions
accumulating evidence the LLM can never see. The always-on witness fields
(`induction_tier_latched_off`, `transitions_since_last_induction_attempt`) quantify the waste
for free on the next scored run. Run this A/B AFTER 2a settles think, because re-induction's
value depends on the per-attempt success rate think sets. *Gate: banked levels or admission
rate up under the flag; no action-budget regression on games that never stall.*

**ATTEMPTED, INFRASTRUCTURE-BLOCKED on first 4 launches (2026-08-09, REQ-ARC-WMTE-6247) — zero
valid cells, not a lever result.** Four launches against ka59+re86 (budget 1500, real live-path
episodes via `arc_scored_path_lever_harness.py`'s `run_cell`). Fixed 3 real bugs along the way (a
`REPO` path bug crashing the final write; missing invalid-cell detection letting a reaper-corrupted
cell be silently accepted as done; a self-inflicted server storm from a redundant health-check
stacked on top of `_ensure_server()`'s own identical internal check). After fixing all three,
ka59/off still failed identically on all 3 retry attempts (same action count, same failure reason
each time) — the surviving attempt shows 4 internal LLM retries, 0 successful completions, over
~27 minutes. A 4th, uncharacterized storm-like failure then appeared moving to the next cell.
Stopped there rather than chasing a 4th distinct infrastructure issue on a single lever
measurement — the remaining problem implicates the shared, pre-existing, unsolved llama-server
reaper, out of scope for this task.

**RETRIED CLEAN under isolation (2026-08-09, same day, REQ-ARC-WMTE-6247 UPDATE section) — real
gate result obtained.** Same script, config, seed, budget; re-run with `carnot-conductor.service`
+ `carnot-orphan-cleanup.timer` both stopped for the duration (operator-directed isolation test —
second such instance this session, see `ops/known-issues.md`'s reaper section). All 4 cells
completed on their FIRST attempt, all valid, real nonzero LLM responses throughout. Gate result:
the lever FIRED on both games (`renewed_stall_reinduction: 2` each) but `reached_any_level`
stayed false on both arms of both games — a clean negative, not an infrastructure non-result.
**`CARNOT_ARC_BOUNDED_REINDUCTION` stays default OFF** — the measurement is now trustworthy, but
n=2 games is too thin to justify a production flip either way even on a clean result. Framing:
promising-but-preliminary validation that the measurement PATH works (lever fires correctly, the
harness now produces reliable readings), not a gate-passed or gate-failed verdict on the lever's
value — that needs a wider roster. Full detail, including the second reaper-correlation instance,
in REQ-ARC-WMTE-6247's 2026-08-09 UPDATE section and `ops/known-issues.md`.

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

**RETIRED (2026-08-09) — premise checked before building, found already refuted by existing
evidence.** Before building, checked whether this bet's premise still holds. It does not, on two
independent grounds:

1. **The greedy-direct architecture Duck actually uses (large model, no search, direct real-env
   commits) is ALREADY built and measured — 5 times, all null.** `REQ-ARC-WMTE-5829`
   (`carnot.agentic.arc_greedy_direct_agent`) is a faithful port of Duck's execution model (per
   its own spec: greedy commit after up to `max_turns` inspection turns, no rollback, no search)
   already running gemma-4-31B. Five iterations (`results/outer_loop_arc_winner_greedy_direct_
   ab_{20260723,v2objects,v3reflect,v4goalverify,v5sweep8}_20260723.json`) all report the identical
   verdict: `complete_greedy_direct_gemma31_honest_null_no_discovery_matches_our_stack_zero_
   baseline`. So the plan's stated premise ("three factors have changed: 31B generator...") is
   stale — the 31B-generator factor is not new or untested; it already nulled under this exact
   execution model.
2. **The one genuinely-novel remaining axis (arbitrary Python vs a 5-function dispatch table)
   would only matter if multi-step lookahead/search were the binding constraint — and that
   hypothesis is dead, with hard evidence, twice.** `docs/research-notes/arc-lever-triangulation-
   2026-07-23.md` (5-agent synthesis over `REQ-ARC-FCP-5757` +
   `results/outer_loop_arc_candidate_coverage_attribution_20260723.json`) found bucket-b
   (in-candidate-set actions that only pay off via downstream lookahead) is **0.00% across BOTH
   independent runs — 0 of 247 winning-path progress actions, 12 games total.** Duck's own
   architecture doesn't even attempt real lookahead either (per the source audit: "greedy forward
   execution with real, irreversible steps — not planning against a model"), so an arbitrary-code
   sandbox would not unlock a capability Duck itself lacks.

**Where the real bottleneck actually is (per the same triangulation, convergent with this
session's own goal-energy zero-gradient finding above in 4b):** world-model-induction-grade
SEQUENCE ROUTING — composing a correct 13-33-action ordering from individually-available,
individually-frame-changing candidates. A 31B generator swapped in offline (constraint relaxed)
still moves 0 live levels (`experiment_5722`, `delta_0.0`) with held-out induction accuracy only
0.378 (`experiment_5764`) — far below what a 14-step plan needs (`0.378^14 ~ 0`). The triangulation
states this plainly: **"at or near a genuine capability frontier, not an obvious engineering fix."**
Ranking/selection is separately confirmed dead (7-9 A/Bs, all null,
`docs/research-notes/arc-lever-triangulation-2026-07-23.md`'s evidence table).

**Consequence for this plan.** With 4e retired on evidence, EVERY item in this plan (0a-4e) is now
DONE, NEGATIVE, REDIRECTED, or RETIRED. The remaining candidate levers are the triangulation's own
ranked list, both large and outside this plan's original cheap-lever scope: (1) an offline-distilled
cross-game goal-progress signal trained big on the 3090s then distilled to live-inference cost
(the triangulation's own "best-supported untested, constraint-compatible lever" — but explicitly
not guaranteed, partly circular with induction, and prior learned-signal levers already nulled), or
(2) a fresh SOTA literature scan specifically on hidden-game discovery methods (cheap, informative,
recommended to run BEFORE committing to (1) per the triangulation's own ordering). Neither is a
same-session build; both are strategic-scale decisions worth explicit operator input rather than
autonomous selection.

**FOLLOW-UP (2026-08-09, operator-directed): option 2 ran, surfaced Pinductor, and the operator
authorized preparing it.** The SOTA scan (`research-studying.md` 2026-08-09 entry) surfaced
Pinductor (arXiv:2605.13740) — population-based, UCB1-tree, disagreement-guided refinement of
LLM-induced executable world models — and flagged that it lands on the refinement axis held under
the standing "operator decision required" note. The operator gave that decision: "Let's plan out
and prepare Pinductor to be run." Prepared as REQ-ARC-WMTE-6248: plan note
(`docs/research-notes/pinductor-rex-refinement-plan-2026-08-09.md`, with the full prior-failures
block), module (`python/carnot/agentic/arc_rex_refinement.py`), CPU-only tests (18 passing), and
the A/B driver (`scripts/experiments/experiment_6248_pinductor_rex_ab.py`) — staged, not yet
launched. Gate pre-registered: REx beats linear on HELD change_fidelity in >= 4 of 6 games with
pooled delta > 0, retire-on-fail.

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
