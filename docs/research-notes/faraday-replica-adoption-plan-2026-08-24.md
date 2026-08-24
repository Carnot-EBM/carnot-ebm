# Adopting Faraday / Replica into Carnot — a plan

Date: 2026-08-24. Status: PLAN ONLY. Nothing was built. No GPU work ran.

This note answers one question: which ideas from Inherent Labs' Faraday work
can this project use, given that this project does not train models?

The short answer is in [Part 3](#part-3--the-three-bucket-split). The bucket
of ideas we can use without training is **thinner than it looks**, and the
single most useful thing the paper did for us was not an idea at all. It was
prompting a measurement that found an existing, already-documented,
already-built fix sitting switched off.

## Part 0 — sources and confidence

Primary sources, both read in full:

| Source | What it gave |
|---|---|
| arXiv 2608.13331, "Training AI Scientists to Replicate Research" | full text, 47 pages, extracted from the PDF |
| inherentlabs.ai/research/training-to-replicate | company framing, no new technical facts |

Authors: Falck, Sabri, Surina, Foster, Sims, Devlin, Rogers, Collins,
Aleksiev, Kirsch, Hughes. Submitted 13 August 2026.

**Every number below is self-reported and unreplicated.** No independent
party has checked any of it. Treat the numbers as claims, not facts.

**Nothing is released.** The paper carries the standard arXiv non-exclusive
licence and no release statement for weights, code, or the Replica task
suite. The only contact route is an email address. Plan for nothing to
arrive.

Two name traps, confirmed and avoided: `bartowski/Fara1.5-27B-GGUF` is a
different model with a near-identical name, and `Faradaylab/*` is an
unrelated organisation. Neither is pinned anywhere in this note.

## Part 1 — what the paper actually does

Faraday is a **27B outer agent that directs a frontier coding agent as a
tool**. The paper calls this CAT, "coding agent as a tool". The outer model
is post-trained Qwen3.6-27B. The inner tool is GPT-5.4-mini during most of
training and GPT-5.5 at evaluation. The paper estimates the inner model at
5T parameters.

Replica is the task space. Each task asks an agent to replicate one results
figure from a paper. The agent gets the paper with that figure redacted, 60
minutes, and one seventh of an H200. It also gets a container and **internet
access**. 242 training tasks and 68 test tasks come from 100 papers.

Task generation is automatic. Gemini 2.5 Pro runs three vision stages: find
every results plot, draw its bounding box inside a verifier repair loop, then
irreversibly redact it. Humans then inspect and filter every task by hand.

The reward is a rubric-based judge:

| Stage | Model | Detail |
|---|---|---|
| Rubric generator | Claude Opus 4.7 | one rubric per task, from a hand-written meta-prompt |
| Judge | Codex GPT-5.5 | reads the rubric, the container, git history, and the full rollout trace |
| Aggregation | — | 3 judge samples per rollout during training |

The rubric scores five dimensions: visual match, support for the paper's
claim, whether the experiment implements what the paper describes, use of
the compute budget, and scientific integrity. Each scores 0 to 1. The five
average to one number.

Two anti-gaming choices are worth copying and cost nothing:

1. **The gold plot is hidden from the rubric generator.** The rubric is
   written without the answer key. It therefore describes the paper's claim
   rather than the figure's axis ranges.
2. **The rubric is hidden from the model during training.** The agent cannot
   target the criteria directly.

Post-training is GRPO with LoRA, rank 128, alpha 128, learning rate 6e-6,
128K context. Each step draws 10 tasks with 8 rollouts each. Turn-level
weights spread credit over a rollout's turns, normalised so that
`sum(u_k * n_k) = sum(n_k)`, where `n_k` is the token count of turn k.

Headline claims: Faraday beats Claude Opus 4.8 and GPT-5.5 on 73% of
in-distribution tasks and 60% of held-out tasks, judged by its own rubric
judge.

## Part 2 — three corrections to the brief

I was asked to test these ideas rather than accept them. Three parts of the
framing I received do not survive contact with the paper or the data.

### Correction 1 — the judge does NOT agree with humans well

The brief presents "judge validated against humans" as a strength. The
measured agreement is weak, and the paper's own figure shows it.

| Pair | Kendall tau |
|---|---|
| Human vs human | **0.30** |
| Rubric judge vs human | **0.19** |
| Baseline judge vs human | 0.15 |
| Rubric judge vs itself | **0.66** |
| Baseline judge vs itself | 0.46 |

Read those together. The rubric judge agrees with **itself** more than twice
as well as two humans agree with each other. It agrees with **humans** worse
than humans agree with each other. The judge is precise and not accurate.

That is not a criticism of the paper. Precision is exactly what GRPO needed,
and the paper says so: eight baseline judge samples are needed to reach the
noise level of three rubric judge samples. Lower reward noise was the goal
and they got it.

But it changes what the idea is worth to us. If we copied their validation
bar, we would certify a verifier at tau = 0.19 against human judgement. That
is a low bar. The improvement they report over the baseline judge is 0.19
versus 0.15, from 117 rankings by 20 people. The paper reports no confidence
interval on that difference. Do not treat 0.19 as a validated verifier.

### Correction 2 — the curriculum was never run

The brief calls the curriculum "possibly the sharpest idea". It is the least
evidenced part of the paper.

What actually shipped as a curriculum is a **horizon curriculum**, and only
that: 30 minutes per task early in training, raised to 60 minutes later.
That is the whole of it.

"Feature removal, budget variation, imagined papers" appears in the
Discussion as a direction, not in the Methods as a procedure. The "imagined
papers" are 20 **evaluation** tasks, generated by Claude Opus 4.8 from 10
source papers, in two variants: same claim on a different dataset, and a
different claim in the same setting. Faraday wins 19 of 20.

The authors then disown the result themselves, verbatim:

> "we must caution that our rubric judge was never validated on imagined
> tasks, and so future work is warranted to validate this claim."

So the innovation claim rests on a judge scoring tasks it was never checked
against. That is precisely the failure mode our Circularity discipline
exists to catch. We should not copy it.

### Correction 3 — the seed sweep does not show "game-specific variance"

This is the important one, and it changes the recommendation.

The brief says run-to-run variance is game-specific: tu93 differed by a full
level in 2 of 3 seed pairs, ar25 tied in both. I re-derived this from
`rows.json` and the counts are right. The **interpretation** is not.

Here is every row:

| arm | game | seed | levels | actions to 1st level-up | total actions |
|---|---|---|---|---|---|
| S | tu93 | 20260724 | 1 | 345 | 389 |
| S_replicate | tu93 | 20260724 | 1 | 345 | 389 |
| S | tu93 | 20260725 | 1 | 345 | 389 |
| S_replicate | tu93 | 20260725 | **2** | **43** | 364 |
| S | tu93 | 20260726 | 1 | 345 | 389 |
| S_replicate | tu93 | 20260726 | **2** | **43** | 365 |
| S | ar25 | 20260724 | 1 | 40 | 388 |
| S_replicate | ar25 | 20260724 | 1 | 40 | 388 |
| S | ar25 | 20260725 | 1 | 40 | 384 |
| S_replicate | ar25 | 20260725 | 1 | 40 | 390 |

I verified the two arms carry byte-identical config. `gated_flags`,
`frame_retention_components`, `explore_budget`, `budget`,
`retains_node_frames` and `early_stop_grace` match exactly in all five
pairs. So this is a true A/A comparison.

Three things follow, and none of them is "variance".

**The S arm has ZERO variance across seeds.** Three different seeds on tu93
produce 389 actions, 345 actions-to-first-level-up, and 1 level. Identical,
three times. A noisy process does not do that.

**The outcome is bimodal, not spread.** The winners do not score "a bit
better". They reach their first level-up at action 43 instead of 345, an 8x
separation. Everything downstream follows from that one early divergence.
`levels` is a coarse 1-bit readout of a latent binary event: did an early
plan land.

**We already log the latent variable.** `actions_to_first_levelup` separates
the two modes by 8x while `levels` moves by 1. `induction_planned` separates
them too: 1 for every loser, 2 and 3 for the winners.

So a per-game rubric is the wrong instrument for this problem. A rubric adds
resolution to a scoring function. We do not have a resolution problem. We
have an **uncontrolled nondeterminism** problem, and a higher-resolution
metric we are not reading.

## Part 2.5 — what the measurement actually found

Chasing correction 3 to its cause found something better than any idea in
the paper.

The generator's sampler is not seeded. The harness already knows this. From
`python/carnot/agentic/arc_executable_world_model.py`, in the docstring of
`LocalGGUFProposer.sampling_seed`:

> "two runs of identical code on the identical game with the identical
> harness `seed` produce different LLM output. The harness `seed` argument
> seeds `random`/`numpy` inside the driver; it never reached the server's
> sampler."

Every generation goes out at temperature `0.2 + 0.1*attempt`, which is
nonzero, with no `seed` field. `llama-server` reads an absent seed as -1 and
picks a fresh random one per request.

The fix is already built. `sampling_seed()` reads
`CARNOT_ARC_GENERATOR_SEED` and returns `base * 1000 + attempt`. It varies
with `attempt` on purpose, so the retry ladder keeps its diversity while the
whole run stays reproducible. It defaults to OFF, deliberately, because
changing how the scored agent samples is a behaviour change and not a
measurement change.

**The seed sweep did not set it.** I checked `run.sh`: zero occurrences.

Now the convergence. The docstring reports an independently measured
nondeterminism rate of **2 of 5 cells** under identical code, from
`results/arc_engine_retention_20260729` against
`results/arc_heldout_31b_vs_9b_20260728`. This sweep, run a month later on
different games, measured **2 of 5 pairs** diverging by a full level.

Two independent measurements. Same rate. The docstring's conclusion holds:

> "That floor is at least as large as any treatment effect yet measured on
> this path, so an A/B here is uninterpretable without an A/A control no
> matter how many cells it runs."

This is the binding constraint on every A/B this project runs on the live
path, including every AVO arm. It is not a Faraday idea. Faraday's rubric
work is what prompted me to look.

## Part 3 — the three-bucket split

### Bucket (a) — usable now, without training

| Idea | Why it survives | Honest value |
|---|---|---|
| Hide the answer key from the scorer generator | A design rule, not a training step. Applies to any verifier we write. | Real but small. Free to adopt. |
| Hide the scorer from the agent | Same. Prevents the agent targeting the criteria. | Real. We already do this by accident, not by rule. |
| Multi-sample aggregation and noise-share accounting | This is a statistical protocol. "What fraction of within-group variance is judge noise, as a function of m samples" is computable with no gradient anywhere. | **The most useful item in this bucket.** It is the question our A/B work needs. |
| Multi-dimensional process scoring in place of one terminal number | A scorer is a scorer whether or not anything trains on it. | Real, but see correction 3. We mostly have the fields already. |
| Validating a scorer against an external standard | Nothing about validation needs training. | Real gap for us. Their bar is weak. See P4. |

### Bucket (b) — usable only with training

| Idea | Why it needs training |
|---|---|
| GRPO | It is an RL algorithm. There is no non-gradient form. |
| Turn-level credit assignment | The weights `u_k` multiply a per-token gradient. Without a gradient they multiply nothing. |
| Horizon curriculum, 30 to 60 minutes | Only meaningful as a schedule over training steps. |
| LoRA rank 128, lr 6e-6, 128K context | Training hyperparameters. |

One nuance on credit assignment. The **idea** transfers even though the
**mechanism** does not. Our `TrajectorySupervisor` already keeps an
`arm_outcomes` ledger with `fired` and `helped` counts per arm
(REQ-ARC-WMTE-6640). That is credit assignment over arms rather than over
turns. We built it from AVO, not from Faraday, and it needs no gradient. So
this row is already covered by work in flight.

### Bucket (c) — not applicable

| Idea | Why not |
|---|---|
| CAT: a small model directing a frontier coding agent | **Structural.** Faraday's inner tool is a 5T model reached over the internet. Our scored ARC path runs offline with no internet and no frontier tool. The paper's single largest result has no scored-path analogue. |
| The Replica task space | Paper replication is not our domain, and it is unreleased. |
| The Gemini redaction pipeline | ARC games contain no figure to redact. There is no analogue. |
| Paper-author feedback validation | ARC games have no authors. |

A note on CAT that is worth keeping. CAT does not describe our scored agent,
but it **does** describe our conductor: an outer planner directing codex and
claude as tools, with internet. The difference is that Faraday trains the
outer agent and we prompt it. So CAT is descriptive of what we already have,
not prescriptive of something new. It is not an action item.

### Is bucket (a) thin?

**Yes.** Plainly: most of this paper does not transfer to a project that
does not train.

Of the five items in bucket (a), two are one-line design rules, one is a
statistical protocol available from any measurement textbook, one duplicates
fields we already log, and one has a demonstrated bar (tau = 0.19) too weak
to certify anything. The paper's real contribution is the RL recipe, and the
RL recipe is bucket (b) in its entirety.

That conclusion is worth more than a plan that quietly assumes we will start
training. We will not. See `feedback_trm_training_retired` and
`reference_overtraining_grokking_path`.

## Part 4 — the four candidate ideas, assessed

### Idea 1 — per-task rubrics as noise reduction

**Assessment: misdiagnosed. Do not build a rubric. Fix the sampler first.**

The reasoning is correction 3 plus Part 2.5. Our A/B results are unreadable
because the generator samples with a random seed, not because our metric has
too few dimensions. A rubric layered on top of an unseeded generator
measures the same noise at higher resolution.

There is a real per-game insight underneath, and it is cheaper than a
rubric: read `actions_to_first_levelup` and `induction_planned` instead of
`levels`. Those fields already exist in every row. On tu93 they separate the
two modes by 8x where `levels` separates them by 1.

### Idea 2 — withheld-ground-truth task design as an oracle-distinct target

**Assessment: the strongest transfer in the paper. An ARC analogue exists.**

Their structure is: ground truth exists, is withheld from the agent, and
grading is objective. That is exactly the shape our GAP program keeps
failing to find, because on math and CSP corpora the verifier is the oracle
and self-consistency is already near-optimal.

The ARC analogue is **goal-predicate prediction before any level-up**. The
win condition exists. It is recorded per game in
`ops/arc_solve_registry.yaml`. It is withheld from the live agent, which
must induce it at runtime. Grading is objective: did the induced predicate
later produce a level-up.

This is oracle-distinct in the required sense, with one caveat that must be
stated in any artifact. The env's level counter **is** available to the
agent at runtime as feedback. So the verifier is distinct from the oracle
only for **prediction before the fact**, not for post-hoc scoring. The clean
formulation is therefore narrow and specific:

> Score a candidate goal predicate at induction time, before the agent has
> reached any level on that game. Grade it against whether the plan built on
> it produced a level-up.

The Phase D retirement explicitly leaves this open. From
`ops/verifier_gaps.md`: the retirement "does not apply to future ARC-domain
oracle-distinct verifier work". So this is not a retired direction.

### Idea 3 — the curriculum

**Assessment: do not adopt. See correction 2.**

The curriculum was not run. The one part that was run is a horizon
curriculum, which is bucket (b). The imagined-papers result is scored by a
judge the authors say was never validated on those tasks.

An ARC analogue would mean synthesising games. We cannot make a synthetic
game faithful to the hidden distribution, and we would grade it with a
scorer never validated on it. That reproduces the exact flaw the authors
flag in their own work, and our Circularity discipline would flag it.

One piece of the surrounding idea is sound and cheap, and it is the budget
axis. Their tasks say "produce the most faithful scaled-down version within
budget". Our analogue is the 400-action cap. A budget-varying evaluation
would tell us whether our agent is budget-limited or capability-limited,
which we have not measured. I cost it in P3 and then recommend deferring it,
because it is expensive and the AVO program has priority.

### Idea 4 — judge validated against humans

**Assessment: the gap is real. A human study is the wrong way to close it
for us.**

The brief is right that our Verifier Authenticity Discipline checks whether
a verifier is what its docstring claims and does not check whether it agrees
with anything external. That is a genuine gap.

But human taste is the right standard only when objective ground truth is
absent. That is true for paper replication. It is **not** true for ARC,
where the env supplies an objective outcome: the level-up. Validating an ARC
verifier against human opinion would substitute a weaker standard for a
stronger one we already have.

So close the gap against **outcome**, not against humans. For each verifier
on the live path, record whether its score predicted the objective result on
held-out runs. That is a calibration receipt, it is mechanical, and it costs
no GPU. See P4.

## Part 5 — proposals, with costs

Cost basis, measured from the sweep: 10 runs at budget 400 took 69,867 s
wall, serially. Per-run wall time ranged 4,099 to 9,892 s. **Call it 2
GPU-hours per run.**

### P1 — A/A determinism control, then decide (RECOMMENDED, do first)

**What.** Set `CARNOT_ARC_GENERATOR_SEED` and re-run the A/A pair on the
game that diverged. The docstring already names the success criterion: an
A/A arm should come back byte-identical.

**Steps.**
1. Confirm from the run logs whether the 3-vs-4 `llm.responses` difference
   at the same seed is sampling divergence or control flow. Zero GPU.
2. Re-run tu93, 2 seeds, both arms, with the sampler seed set. 4 runs.
3. If A/A returns byte-identical, the constraint is closed and every future
   A/B on this path becomes readable at small N. If it does not, we have
   found a second nondeterminism source, which is a more valuable finding
   than the first.

**Cost.** Step 1: zero GPU. Steps 2-3: 4 runs, **about 8 GPU-hours**.

**Slot impact.** Step 1 needs no slot. Steps 2-3 fit one infrastructure
slot. I would not spend an ARC floor slot on it.

**Why first.** This is a **prerequisite for the AVO programme, not a
competitor to it.** CLAUDE.md makes AVO adoption the immediate goal. Every
AVO arm will be evaluated by an A/B on this path. A treatment effect of one
level is currently inside a noise floor measured twice at 2-of-5. Until P1
lands, an AVO arm that helps and an AVO arm that does nothing produce the
same evidence.

**Honest caveat.** Seeding the sampler does not guarantee determinism. GPU
reduction order, continuous batching, and server restarts can all break it.
That is why P1 is a measurement with an A/A control, not an assumption.

### P2 — oracle-distinct goal-predicate prediction (RECOMMENDED, after P1)

**What.** Build the measurement in Idea 2. Score each induced goal predicate
at induction time, before any level-up on that game. Grade against whether
the resulting plan produced a level-up.

**Steps.**
1. Retrospective pass over existing artifacts for runs that recorded an
   induced predicate and a subsequent outcome. Zero GPU. This also tells us
   whether we have enough logged data to skip step 2.
2. If the retrospective sample is too small, a live confirm: 3 games, 2
   seeds. 6 runs.

**Cost.** Step 1: zero GPU. Step 2: **about 12 GPU-hours**.

**Slot impact.** One ARC slot. It qualifies under the
Generalization-Testing Floor as reusable-primitive work, and it serves the
Missing-Verifier Gap ledger directly.

**Sequencing.** This depends on P1. Without a determinism control, a null
result here cannot be distinguished from noise, and we would log a false
negative into the gap ledger.

### P3 — budget-varying evaluation (COSTED, then DEFERRED)

**What.** Run at 100, 200, 400 and 800 actions. Answer whether the agent is
budget-limited or capability-limited.

**Cost.** Roughly 1,500 actions per cell against 400 today, so about 3.75x
one run. At 2 games and 2 seeds that is **about 30 GPU-hours**.

**Recommendation: defer.** It is the most expensive proposal here, it does
not unblock anything, and the AVO programme has priority. Revisit after P1
and P2.

### P4 — verifier calibration receipt (RECOMMENDED, cheap)

**What.** Extend the Verifier Authenticity Discipline with a second
question. Today it asks "is this verifier what it claims to be". Add "did
its score predict the objective outcome on held-out runs".

Emit a per-verifier receipt: N held-out cases, agreement with outcome, and
an explicit abstention count. Report the number. Do not gate on it yet; a
new check that fires on honest work trains people to bypass it.

**Cost.** **Zero GPU.** Retrospective over existing artifacts plus a
receipt writer.

**Slot impact.** One infrastructure slot.

**Why this and not a human study.** See Idea 4. Their bar is tau = 0.19
against human rankings that cost £150 per task. We have an objective
outcome. Use it.

### Totals

| Proposal | GPU-hours | Slot | Verdict |
|---|---|---|---|
| P1 determinism control | ~8 | 1 infra | do first |
| P2 goal-predicate prediction | ~12 | 1 ARC | do after P1 |
| P3 budget sweep | ~30 | 1 ARC | defer |
| P4 calibration receipt | 0 | 1 infra | do, cheap |
| **Recommended set (P1, P2, P4)** | **~20** | **2 infra + 1 ARC** | |

At 2 GPU-hours per run and one available GPU, the recommended set is about
one day of GPU time, spread across two milestones.

## Part 6 — what I recommend NOT doing

| Do not | Reason |
|---|---|
| Build a per-game rubric scorer | Correction 3. The problem is an unseeded sampler, not metric resolution. A rubric on top of that measures noise at higher resolution. |
| Adopt the curriculum or synthesise "imagined" ARC games | Correction 2. It was never run, and the one evaluation of it uses a judge the authors say was never validated on those tasks. |
| Run a human-agreement study for our verifiers | Idea 4. ARC has an objective outcome. Human taste is a weaker standard, and their demonstrated agreement is tau = 0.19. |
| Adopt PaperBench or ReplicationBench | See Part 7. Both are real, but both would spend ARC floor slots on paper replication, which is not our domain, not the verifier moat, and not reachable from either scored entrypoint. |
| Plan around obtaining Faraday or Replica | No release statement exists anywhere in the paper or on the company page. |
| Copy the CAT architecture into the live agent | Bucket (c). The scored path is offline with no internet and no frontier tool. |
| Pin `bartowski/Fara1.5-27B-GGUF` or anything under `Faradaylab/*` | Neither is Faraday. Confirmed name collisions. |

### On the two public benchmarks

I checked both, because the brief asked whether they beat waiting on Replica.

**PaperBench** (OpenAI, `openai/preparedness`) is genuinely released: 20
ICML 2024 Spotlight and Oral papers, hierarchical rubrics with author input,
a released judge, and the harness. It is the strongest artifact in this
space.

**ReplicationBench** (arXiv 2510.24591) covers astrophysics, is built with
paper authors, and reports frontier models under 20%. I could not confirm
its release terms from the abstract alone.

Both beat waiting on Replica, which will not arrive. **Neither is worth
adopting**, for the same reason: they measure paper replication. Our
deliverable is a live agent that discovers hidden games. Nothing in either
benchmark is reachable from `E3AgentPolicy` or `arc_loop_solve`, so under
the ARC Live-Path Reachability Discipline the work would not count.

## Part 7 — compliance check

**Exclusion manifest.** Checked. `exp6582` is retired for the one-family
flagship source-shard scope. `exp6581` is un-retired and free to run. The
Phase D external-text-scorer retirement covers LoRA-EBM, uPRM and EBRM
constructions on off-ARC corpora, and states explicitly that it "does not
apply to future ARC-domain oracle-distinct verifier work". P2 is ARC-domain
and oracle-distinct, so it is not a retired direction. No proposal here
matches any retired scope.

**ARC Generalization-Testing Floor.** P2 claims one ARC slot and qualifies
as reusable-primitive work. P1 and P4 claim infrastructure slots and do not
compete with the floor.

**AVO-Method Adoption.** P1 does not compete with the AVO programme. It is a
prerequisite for evaluating it. State this when scheduling.

**Live-Path Reachability.** P2 measures a predicate produced by the live
induction path and is reachable from `E3AgentPolicy`. P1 and P4 are
measurement and receipt work and make no live-path change. Nothing proposed
here adds an orphan solver module.

**Circularity.** P2's artifact must declare `verifier_is_oracle: false` and
must carry the caveat in Idea 2: the env level counter is available at
runtime, so oracle-distinctness holds for prediction before the fact only.

## Part 8 — what I could not determine

1. **Whether Faraday or Replica will ever be released.** No statement
   exists. I found none in the paper, the company page, or the licence.
2. **Whether tau = 0.19 beats tau = 0.15 significantly.** The paper reports
   no confidence interval on that difference. 117 rankings, 20 participants.
3. **Whether seeding the sampler makes our runs deterministic.** The seed
   mechanism exists and is off. Nobody has run the A/A control with it on.
   This is P1 and it is the highest-value unknown in this note.
4. **Whether the 3-vs-4 `llm.responses` difference at the same seed is
   sampling divergence or control flow.** Distinguishing them needs the run
   logs, which I did not read. P1 step 1.
5. **ReplicationBench's release terms and licence.** I read the abstract
   only and did not check the repository.
6. **Whether enough goal-predicate outcomes are already logged for P2's
   retrospective.** P2 step 1 answers this before any GPU is spent.

## Cross-references

- arXiv 2608.13331 — the source paper
- `docs/research-notes/avo-adaptation-for-local-generator-2026-08-21.md` —
  the sibling adoption note. The supervisor arm ledger described there is
  the non-gradient cousin of Faraday's turn-level credit assignment.
- `python/carnot/agentic/arc_executable_world_model.py`,
  `LocalGGUFProposer.sampling_seed` — the built, documented, default-off fix
  at the centre of Part 2.5
- `python/carnot/agentic/arc_trajectory_supervisor.py` — `arm_outcomes`,
  REQ-ARC-WMTE-6640
- `ops/verifier_gaps.md` — where P2's result belongs, filled or open
- `ops/exclusion_manifest.yaml` — checked, Part 7
- CLAUDE.md "AVO-Method Adoption for the Live Agent",
  "ARC-AGI-3 Generalization-Testing Floor",
  "ARC Live-Path Reachability Discipline",
  "Circularity / Oracle-Distinctness Discipline",
  "Verifier Authenticity Discipline" — the disciplines P4 extends
