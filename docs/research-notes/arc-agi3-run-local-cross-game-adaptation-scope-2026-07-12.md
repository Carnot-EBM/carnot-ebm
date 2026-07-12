# ARC-AGI-3 run-local cross-game adaptation — scope (2026-07-12)

**Provenance:** operator question, after the REQ-CAPSTONE-4582 live-wiring commit
(`ecb2b7bf9`, per-game genre inference from own transitions): "will our live agent ...
learn while playing hidden games to extend the registry dynamically as it plays this
game and additional games during submissions? I am mostly concerned about the
efficiency/bias tradeoff and the ability for our live agent to adapt to games it has
never seen before." Answer given inline: no, not currently, and that's deliberate.
This note scopes what a real version of that feature would look like, gated the same
way every other live-path addition in this project is gated.

**Explicitly NOT built yet.** This is a scope, not a commit. Nothing here should be
read as shipped or as changing current live-agent behavior.

## 1. What exists today (baseline, for contrast)

- `E3AgentPolicy._maybe_route_from_transitions` (commit `ecb2b7bf9`): classifies the
  CURRENT game's mechanic from its own first ~8 transitions, biases search for the
  REST of that same game only. Fresh `E3AgentPolicy` per game (confirmed: `make_carnot_agent`'s
  `CarnotAgent.__init__` constructs a new `E3AgentPolicy(gid, ...)` keyed on `self.game_id`
  each time it's instantiated). No persistence, no cross-game signal.
- `arc_solve_learning.recommend_approach`'s `feature_router` policy (mechanic-class →
  approach mapping) is trained OFFLINE via `learn_feature_router_policy()` from the
  historical corpus (`ops/arc_solve_registry.yaml` + `ops/arc_router_ledger.json` +
  `results/arc_loop_solve_*.json`) and shipped FROZEN in the submission package. It
  does not update during a run.
- `ops/arc_solve_registry.yaml` is a git-tracked file, mutated only by the outer loop
  (dev sessions) between submissions. The live path never writes to it.

## 2. What "dynamic extension during submissions" would actually require

### 2.0 Preconditions (BOTH CLEARED 2026-07-12 — see findings + operator ruling below)

**Does one Kaggle submission process play the FULL hidden game roster sequentially in
one long-lived Python process, or does the harness spawn a fresh process per game?**

**Neither of the two hypothesized branches was right.** Read the actual driver
(`/home/ianblenke/arc-sota-refs/ARC-AGI-3-Agents/agents/swarm.py`, `Swarm.main()`):

```python
for i in range(len(self.GAMES)):
    g = self.GAMES[i % len(self.GAMES)]
    a = self.agent_class(card_id=..., game_id=g, ..., arc_env=self._arc.make(g, ...))
    self.agents.append(a)
for a in self.agents:
    self.threads.append(Thread(target=a.main, daemon=True))
for t in self.threads:
    t.start()          # ALL games start together
for t in self.threads:
    t.join()            # wait for ALL to finish
```

`Agent.main()` (`agent.py:69`) is a plain per-game `while` loop (`choose_action` until
`is_done`/`MAX_ACTIONS`) — one game, run to completion, no surprises there. But `Swarm`
creates every `CarnotAgent` instance UP FRONT and runs **all of them concurrently on
separate threads within ONE process**, not sequentially. This is a materially different
shape than either original hypothesis and changes the design:

- **Confirmed: one process for the whole roster** — so an in-process module-level
  object IS reachable across all game instances without filesystem persistence. Good,
  simplifies storage.
- **Wrong assumption corrected: there is no "game 1 finishes, THEN game 2 starts"
  ordering.** All games race in parallel threads. By the time any given game reaches
  its own early-transition classification window (~8 actions in), most other games in
  the roster may not have completed at all — completion order tracks wall-clock
  difficulty/latency, not roster position. "Later games benefit from earlier games'
  completed lessons" is the wrong mental model; the real one is "whichever games
  happen to finish first contribute noisy, racing signal to whichever games are still
  early when that signal lands." Weaker and noisier than a clean sequential relay.
- **New hard requirement this adds to §2.1: `RunLocalMechanicLedger` must be
  thread-safe** (a lock around read/update, or an equivalent thread-safe structure) —
  multiple `E3AgentPolicy` instances in different threads will read and write it
  concurrently. A non-thread-safe ledger would be a genuine correctness bug (lost
  updates, torn reads), not just a design nicety.
- This also means the validation harness in §2.4 must simulate CONCURRENT play (real
  threads or an equivalent interleaving), not a sequential loop, or the offline A/B
  measurement would not represent the real deployment shape at all — an artificially
  easy/clean sequential simulation could show a benefit that never materializes under
  real racing.

This correction was found by reading the actual vendored framework source
(`/home/ianblenke/arc-sota-refs/ARC-AGI-3-Agents/`) rather than trusting the original
inference from `scripts/kaggle/submission_kernel/main.py` alone, which only shows the
CLASS being registered once and says nothing about the driver's own execution shape —
exactly the kind of assumption an adversarial pass on this scope needed to catch before
any implementation time went in.

**Confirmed this also applies to the REAL scored path, not just the offline reference
copy.** Re-read `scripts/kaggle/submission_kernel/main.py` end to end: on a competition
rerun (`KAGGLE_IS_COMPETITION_RERUN` set) it copies the COMPETITION-PROVIDED
`ARC-AGI-3-Agents` framework to a writable path (line 158), registers `CarnotAgent`
into that copy's `agents/__init__.py` alongside the stock `Swarm` import (line 166),
and runs that framework's own `main.py --agent carnotagent` against the internal
gateway (lines 191-194) with a 12-hour timeout. The comment at line 187 states
directly: "play all gateway games ... main.py fetches the game list from the gateway,
runs the swarm, and the gateway records the scorecard that is scored." One scorecard,
one `Swarm.main()` call, one process, for the WHOLE submission's game roster — this is
the actual scored mechanism, not an inference from the open-source mirror.

**Precondition 2 — RESOLVED (2026-07-12) as "not addressed in public docs," not as a
yes or no.** Searched the ARC Prize / Kaggle public documentation directly rather than
guessing: the main competition page (`arcprize.org/competitions/2026/arc-agi-3`), the
full `docs.arcprize.org` methodology set (`index`, `full-play-test`, `methodology`,
`agents-quickstart`, `scorecards`), and attempted the Kaggle competition rules tab
(login-gated, returned only the page title, unreachable via WebFetch). **None of the
six reachable pages contain any explicit statement permitting or forbidding an agent
from retaining or sharing information across different games within one evaluation
run.** The only tangentially relevant language found: the benchmark is "designed to
measure an AI Agent's ability to generalize in novel, unseen environments" (quickstart
page) and scorecards "aggregate the results from your agent's game performance"
(scorecards page, consistent with the one-scorecard-per-run structure confirmed in
code above) — neither addresses cross-game memory directly.

**This silence is the finding, not a dead end — and it cuts differently than the
original framing assumed.** The competition's OWN reference harness already shares one
process, one Python interpreter, and one scorecard across every game in a run (per the
code evidence above) — there is no structural isolation between games for the
competition's own official harness to have engineered around, unlike the explicit,
separately-documented prohibition on hidden-game source-reading (which the project's
own "ARC Live-Path Reachability Discipline" already encodes because THAT restriction
IS clearly evidenced — by the benchmark's stated design intent and this project's own
prior settled decision, not by an explicit competition rule either, worth noting).
Given genuine silence in the public docs plus a harness architecture that doesn't
prevent it, **this is an operator judgment call, not an engineering one — proceed to
implementation only with explicit operator sign-off**, and consider that sign-off
might reasonably take either form: (a) approve based on the harness-architecture
evidence above, (b) ask ARC Prize / Kaggle support directly for an explicit ruling
before shipping anything, or (c) decline and keep the live agent per-game-only
indefinitely. All three are legitimate; none of them is mine to pick.

**OPERATOR RULING (2026-07-12) — precondition 2 CLEARED, option (a).** Operator
directive, quoted in full because it's the load-bearing rationale for everything
built after this point: *"knowing the rules of a game or a class of games and how to
play them is not the same as memorization, it is learning and adapting which is the
spirit of the challenge. starting from a completely naive position would also
preclude the use of LLMs entirely. as such, we assert that as long as we are not
playing back a prerecorded sequence of events into a game that we pretrained offline,
we are fine."* The operative distinction: **FORBIDDEN** = replaying a literal,
memorized action sequence into a game the agent (or the outer loop) was pretrained
against offline — this is exactly the `GameAdapter`/registry-solution pattern already
excluded from the live path by the "ARC Live-Path Reachability Discipline." **PERMITTED**
= learning and adapting behavior at the level of a game's RULES or a CLASS of games'
shared structure — genre/mechanic-level inference, which is what both the existing
per-game wiring (commit `ecb2b7bf9`) and this run-local cross-game ledger do. The
operator's second point is independently correct and worth preserving as reasoning,
not just as a conclusion: the live agent already uses an LLM generator carrying
enormous pretrained prior knowledge (physics intuition, spatial reasoning, common game
grammars) from its training corpus — "fully naive per game" was never actually true of
this architecture, so genre-level adaptation within or across games in a run is
continuous with what the agent already does, not a new category of concern. This
clears the path to implementation; §2.1-2.3 below now proceed on that basis.

### 2.1 Component: `RunLocalMechanicLedger`

An in-run, NEVER-persisted-to-the-checked-in-registry statistics tracker.

- Storage: an in-process, MODULE-LEVEL singleton (confirmed sufficient by §2.0 — one
  process for the whole roster). No filesystem writes needed.
- **Thread-safe by construction (hard requirement, not optional):** §2.0 confirmed
  `Swarm` runs every game's `Agent.main()` concurrently on its own `Thread`, all
  sharing this process. The ledger MUST guard reads/updates with a lock (or use an
  equivalent thread-safe structure); a non-thread-safe implementation is a
  correctness bug (lost updates / torn reads under real concurrent play), not a
  future nice-to-have. Keep the critical section tiny (append a small tuple, read a
  small aggregate) — this must never become a contention point that slows down
  per-action decisions across games.
- Per completed game, records: `mechanic_class` (from `feature_router`), `approach`
  used, and an outcome metric computed the SAME way the competition scores runs
  (RHAE-style efficiency, not just "won/lost" — matching what's actually optimized,
  per the same principle as REQ-CAPSTONE-4582's own `median_actions_to_first_levelup`
  field).
- Hard rule: this ledger NEVER touches `ops/arc_solve_registry.yaml`. It is
  structurally incapable of writing to the checked-in file from the live path — same
  boundary `GameAdapter`s already respect (development_proxy, offline-only).

### 2.2 Component: confidence-gated policy nudge

- Before any in-run statistic changes behavior, require a minimum sample size per
  mechanic class within the CURRENT run (e.g. >=3 completed games of that class) —
  weaker than the project's own "N>=30 for a percentage-point claim" bar by
  necessity (a submission roster won't have 30 games of one class), so this must be
  documented as a WEAK signal, not oversold as statistically significant.
- Given the 7-class taxonomy and a modest per-submission game count, expect this gate
  to rarely clear in practice — that's intentional conservatism, not a bug. Document
  the expectation up front.
- When it does clear, the effect should be BOUNDED and REVERSIBLE: nudge the existing
  `_FEATURE_ROUTER_MIN_CONFIDENCE` threshold slightly for that specific mechanic
  class this run, rather than opening a new behavior-changing pathway. Worst case
  behavior converges to the already-shipped default.

### 2.3 Component: kill switch

- An env/config flag (`CARNOT_ARC_RUN_LOCAL_ADAPTATION`, default **off**), matching
  the existing `SUBMITTED_*_ENABLED` flag pattern already in
  `arc_competition_agent.py`. Ships off until 2.4 clears.

### 2.4 Empirical validation (REQUIRED before any default-on flip)

Build an offline harness that simulates a "submission run" the way §2.0 confirmed it
actually happens: several PUBLIC games played CONCURRENTLY (real `Thread`s, or an
equivalent interleaving that genuinely exercises the ledger's lock under contention) in
ONE process (via the offline arcade, deliberately not consulting each game's registry
entry mid-run — same development_proxy discipline as everything else offline), with the
ledger active. A sequential-loop simulation would NOT represent the real deployment
shape (per §2.0's correction) and any result from one would not be trustworthy evidence
either way. Measure, matched-compute, same methodology as `exp4582`:

- Do games that finish their early-transition classification window LATER in
  wall-clock time (not roster position — there is no roster position under real
  concurrency) perform better, on average, than an IDENTICAL concurrent run with the
  mechanism disabled? Median actions-to-first-levelup / levels-per-action, matched
  compute.
- Positive control: confirm the ledger actually accumulates under real concurrent
  writes without lost updates (a correctness check on the lock itself, separate from
  the efficacy question) and that the confidence gate fires at least once across the
  simulated run.
- Bootstrap CI on the delta; a claim requires the CI to exclude the no-mechanism
  baseline — same statistical bar `exp4582` already used.
- If null (plausible, given `exp4582`'s adjacent null, and now MORE plausible given
  how racy/noisy the real concurrent signal is per §2.0): retire per the
  Failed-Experiment Rerun Discipline — record the honest result, do not silently
  half-ship it enabled.

### 2.5 Adversarial check (before scaling / before any default-on flip)

- Could the ledger leak game-identity information through the back door? E.g. if
  `mechanic_class` happens to correlate tightly with a narrow subset of games due to
  how the 7-class taxonomy buckets behavior, "later games in the sequence do better"
  could be an artifact of a repeated/similar PUBLIC-game test roster rather than
  genuine transferable behavioral inference — a real hidden roster won't repeat
  games. The offline validation harness must use a DIVERSE game sequence, not a
  narrow one, to guard against this.
- Standard QA-layer checks per the sibling discipline in CLAUDE.md: field-shape
  assumptions (principle-wrapped fields), off-by-one confidence-threshold errors,
  silent no-ops masquerading as active behavior.
- Independent read of whether ANY in-run cross-game memory is within the spirit of
  "discover what winning looks like" per hidden game (2.0's second precondition) —
  this is the check most likely to kill the feature outright, and should run FIRST,
  cheaply, before the more expensive validation work in 2.4.

## 3. Explicitly out of scope for a first version

- No persistence back to `ops/arc_solve_registry.yaml` from the live path, ever.
- No cross-SUBMISSION memory (run N's ledger does not carry into run N+1) — starting
  intra-run-only keeps blast radius smallest; a cross-submission version is a
  separate, even more carefully gated follow-on if intra-run ever clears validation.
- No change to the 7-class mechanic taxonomy itself.
- No change to WHICH solver runs — same restriction as the existing wiring: only
  adjusts existing budget/confidence knobs inside the current `E3AgentPolicy`
  cascade, never swaps to a different top-level solving strategy (that's what
  `exp4582` already found doesn't help).

## 4. Rollout order

1. §2.0 preconditions (process-lifetime fact + competition-rules/spirit check;
   operator sign-off on the rules question specifically).
2. `RunLocalMechanicLedger` + confidence-gated nudge, shipped OFF by default.
3. Offline multi-game-sequence A/B harness (§2.4); run it.
4. Adversarial check (§2.5) on whatever the A/B finds, win or null.
5. Only flip default-on if the A/B clears a real, CI-excludes-baseline bar.
   Otherwise retire per Failed-Experiment Rerun Discipline and record the honest
   null — do not leave a half-validated mechanism silently enabled or silently
   disabled-but-undocumented.
6. Either outcome: log to `ops/verifier_gaps.md`-adjacent tracking or a dated
   corrigendum note so a future session doesn't re-propose the same idea without
   citing this scope + its result.

## Cross-references

- `results/experiment_4582_feature_router_transfer.json` — the offline null this
  scope explicitly builds on top of and must not silently contradict
- commit `ecb2b7bf9` — the per-game (intra-episode) wiring this extends
- `/home/ianblenke/arc-sota-refs/ARC-AGI-3-Agents/agents/swarm.py` (`Swarm.main()`) +
  `agents/agent.py` (`Agent.main()`) — the vendored framework source that resolved
  §2.0's first precondition and corrected the original sequential-execution
  assumption to concurrent multi-threaded execution in one process
- `scripts/kaggle/submission_kernel/main.py` (lines 148-194) — confirmed the SAME
  `Swarm`-driven mechanism governs the real scored submission, not just the offline
  reference copy (competition-provided framework, one gateway, one scorecard, one
  `main.py --agent carnotagent` run per submission)
- docs.arcprize.org: `index`, `full-play-test`, `methodology`, `agents-quickstart`,
  `scorecards` (all fetched 2026-07-12) + `arcprize.org/competitions/2026/arc-agi-3` —
  searched for an explicit cross-game-memory rule, found none; the Kaggle rules tab
  itself is login-gated and was not reachable
- `tests/python/test_experiment_4582_live_feature_router_wiring.py` — existing
  regression coverage for the per-game mechanism
- CLAUDE.md "ARC Live-Path Reachability Discipline" — the source-reading /
  live-vs-development_proxy boundary this scope's §2.0 second precondition mirrors
- CLAUDE.md "Phase Prototype + Empirical Validation + Adversarial Check Discipline"
  — the three-part structure (§2.1-2.3 prototype, §2.4 validation, §2.5 adversarial
  check) this scope follows
- CLAUDE.md "Failed-Experiment Rerun Discipline" — governs what happens if §2.4
  comes back null, same as `exp4582` was handled
