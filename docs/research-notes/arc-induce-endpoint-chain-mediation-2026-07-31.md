# Can any endpoint closer than banked levels carry the induce intervention?

**Date:** 2026-07-31
**Artifact:** `results/outer_loop_arc_endpoint_chain_mediation_20260731.json`
**Verdict:** `complete_no_valid_endpoint_closer_than_banked_levels_corpus_cannot_localize_break`
**Compute:** none. Pure re-analysis of banked evidence. No GPU, no model load, no live or
scored ARC game, no submission.

This note exists because the determination below was, until it was written, held only in a
session-scoped scratchpad and a chat message. `determination_preservation_lint.py` audits
committed determinations and so could not see it.

---

## 1. The question and the short answer

The Phase-3 refusal (commit `c39f4bde6`) declined a 12-cell banked-levels grid for the
`repeat_penalty 1.1` induce intervention, on the arithmetic that banked levels are downstream
of the induced engine, so `P(discordant endpoint) <= P(discordant induce)` = 5/36 on the
strict-quality channel, giving `P(reach 6 of 12) = 0.0002` and requiring 68 cells. The
follow-on question was whether some endpoint **closer to the intervention** could be decided
more cheaply.

**Answer: no valid endpoint is established closer than banked levels, and the present corpus
cannot even localize where the chain breaks.**

The reason is a design limit, not a measurement. The live corpus is 12 games of which **2**
bank any level. With those margins, a two-sided Fisher test can reach `p<0.05` **only** if the
predictor is positive in 2, 3, 9 or 10 of the 12 games:

| predictor positive in *m* games | best attainable two-sided *p* | can reach 0.05? |
|---|---|---|
| 1 | 0.1667 | no |
| **2** | **0.0152** | **yes** |
| **3** | **0.0455** | **yes** |
| 4 | 0.0909 | no |
| 5 | 0.1515 | no |
| 6 | 0.4545 | no |
| 7 | 0.1515 | no |
| 8 | 0.0909 | no |
| **9** | **0.0455** | **yes** |
| **10** | **0.0152** | **yes** |
| 11 | 0.1667 | no |

So only a predictor that is positive in almost exactly the banking games can be "significant"
— and the only such predictors are the reverse-causal post-bank stages. **"Only reverse-causal
stages predict" is guaranteed by the design, not discovered by it.**

---

## 2. The chain, with floors on both halves

**(a) response** — 36 seed-paired live inductions (`results/arc_induce_confirm_20260731/confirm_scored.json`),
6 games, matched game/seed/temperature/server process. Exact two-sided sign test on discordant
pairs, plus the minimum reachable *p* at that discordance.

**(b) prediction** — the 12 independent `ret0` games of
`results/arc_engine_retention_20260729/cells/`. Fisher exact on (stage) × (banked > 0),
**plus the minimum reachable p at those margins** — the column the first pass omitted.

```
stage                          | (a) ctrl treat disc      p    floor | (b) n   result       p    floor  decidable?
-------------------------------+-------------------------------------+---------------------------------------------
S1  generator returns          |  36/36 36/36   0  CONSTANT          | 12  CONSTANT
S2  engine parses              |  27/36 30/36  13 0.5811   0.0002    | 12  CONSTANT
S3  returns on all paths       |  23/36 35/36  14 0.0018   0.0001    |     UNMEASURED LIVE
S4  no raise at dry run        |  34/36 36/36   2 0.5000   0.5000    |     UNMEASURED LIVE
S5  engine defect-free         |  14/36 29/36  23 0.0026   0.0000    |     UNMEASURED LIVE
S6a engine USABLE (non-inert)  |  13/36 22/36  17 0.0490   0.0000    |     UNMEASURED LIVE
S6b   ...scored-frac proxy     |         (not the same predicate)    | 12  rho +0.229 0.5379 0.0909  NO - FLOOR
S7  held-out accuracy > 0      |   7/36 15/36  12 0.0386   0.0005    | 12  [2,5,0,5] 0.4697 0.1515  NO - FLOOR
S7c held-out accuracy, best    |         (continuous)                | 12  rho +0.545 0.0909 0.0909  NO - FLOOR
S8  clears DYNAMICS gate       |   5/36  8/36   5 0.3750   0.0625    | 12  [2,2,0,8] 0.0909 0.0909  NO - FLOOR
S9  CORRECT (strict quality)   |   6/36  7/36   5 1.0000   0.0625    | 11  [0,4,1,6] 1.0000 0.3636  NO - FLOOR
S9b CORRECT non-vacuous        |   4/36  4/36   2 1.0000   0.5000    | 11  [1,5,0,5] 1.0000 0.4545  NO - FLOOR
S10 goal predicate satisfiable |     UNMEASURED BY PHASE 1           | 12  [2,0,0,10] 0.0152 0.0152  YES
S13 plan found                 |     UNMEASURED BY PHASE 1           | 12  [2,0,0,10] 0.0152 0.0152  YES
```

`floor` = smallest two-sided p attainable with those margins at that n. **A row whose floor is
>= 0.05 is UNDECIDABLE, not null.**

Three corrections to the first pass are embedded above:

- **S5, S6a, S3, S4 have no live measurement.** Retention rounds carry no `defect_kinds` and no
  `engine_changes_anything`. The first pass silently substituted `engine_scored_any` ("an engine
  ran on held-out") and reported the result as though it were the stage. S6b is the closest live
  proxy and is labelled as a proxy.
- **S9's live measurement comes from the `change_gate` block**, which computes Phase 1's own
  strict bar — not from the dynamics gate, which the first pass mistakenly reused (producing a
  row that contradicted its own Phase-1 S9 with no reconciliation).
- **The claim "no stage passes both (a) and (b)" is withdrawn.** It cannot be supported when
  most (b) tests were incapable of rejecting.

### Missing data: vc33

`vc33` banks 2 levels — the top performer — and carries **no `change_gate` block**, so it is
silently excluded from every correctness (b) test (n=11, not 12). Imputed both ways:

| vc33 imputed | table | p | floor | |
|---|---|---|---|---|
| strict_correct = 0 | [0,4,2,6] | 0.5152 | 0.0909 | undecidable |
| strict_correct = 1 | [1,4,1,6] | 1.0000 | 0.1515 | undecidable |

The verdict is unrejectable under both. This is the same failure mode the operator named when
`hv_progress` read 0.0 on vc33 — recurring here as *missing data* rather than as a zero.

### Global caveat

**Every (b) association in this analysis is determined by exactly 2 games** (tu93 = 1,
vc33 = 2). Outlier-dependence limits the stages declared NON-predictive exactly as much as it
limits the reverse-causal ones.

---

## 3. The surviving mediator candidate: S7

S7 (held-out accuracy > 0) is the one stage that **responds** (12 discordant, p = 0.0386,
floor 0.0005 — genuinely decidable) and is **not already saturated** in the live pipeline. Both
banking games sit inside it: `[2,5,0,5]`, the maximum association a group of 7 permits.

The first pass dismissed S7 by analogy to `hv_progress`. **That analogy is inverted.**
`hv_progress` read **0.0 on vc33**, the best cell; S7 reads **1 on both banking games**. S7 is
reinstated as the surviving candidate. Its (b) evidence sits at the n=12 ceiling — it was never
refuted.

---

## 4. Required n

Model replicated exactly from the refusal (`c39f4bde6`): per-cell `P(one-way discordant)` =
majority-direction discordant count / 36; significance needs >= 6 one-way discordant pairs;
"cells" = smallest n with `P(Binom(n,p) >= 6) >= 0.50`. **Scale check: this reproduces the
refusal's published 68 (channel A) and 16 (channel B) exactly.**

```
stage                          p_oneway  P(6 of 12)  n@stage  n@levels(determ)  f=.05  f=.10  f=.20
S5  engine defect-free           0.5278      0.6855       11                34     26     21     16
S6  engine USABLE (non-inert)    0.3611      0.2378       16                34     26     21     16
S7  held-out accuracy > 0        0.2778      0.0858       21                34     26     21     16
S8  clears DYNAMICS gate         0.1111      0.0010       51                51     35     27     18
S9  CORRECT (strict quality)     0.0833      0.0002       68                68     43     31     20
```

**S7 at banked levels costs 34 cells — half the refusal's 68.** If S7 mediates, a materially
cheaper design exists. The catch: whether it mediates is undecidable on this corpus, and more
cells at n=12 cannot decide it. Holding the observed 1-in-6 banking rate, a half-corpus
predictor becomes decidable only at **n >= 36 games**.

### The determinism caveat

The banked-levels column inherits the refusal's assumption that levels are a deterministic
function of the stage. The project's own evidence contradicts it **in both directions**:

- **vc33 banks 2 levels with zero pre-bank inductions** — the endpoint moved while the stage sat
  at its null value.
- **Force-admit produced real plans (lengths 61 and 6) and changed the action trace while levels
  stayed 0** — the stage moved while the endpoint sat still.

A nonzero leak `f` makes the endpoint **easier** to move, so the deterministic figure is an
upper bound on n. But power bought at `f = 0.20` is power bought from rollout noise — that
design measures the harness, not the intervention.

### The ceiling, independent of any mediator

| n | one-way discordance needed for 50% | ceiling P(game banks) = 0.167 | |
|---|---|---|---|
| 12 | 0.460 | | UNREACHABLE |
| 24 | 0.233 | | UNREACHABLE |
| 36 | 0.157 | | reachable |
| 68 | 0.083 | | reachable |

At n=12 the requirement exceeds the ceiling ~2.7×. Ten of twelve games bank nothing and cannot
express a difference in either direction.

---

## 5. Is the endpoint movable at all? (the positive control that was never built)

A positive control is not "the counter is sometimes nonzero" — it is "some manipulation MOVES
it". Three independent manipulations are on disk:

| manipulation | scope | times it moved banked levels |
|---|---|---|
| engine retention ON vs OFF (`ret0` vs `ret1`) | 12 matched games | **0 of 12** |
| LLM induction tier ON vs OFF | 6 games | **0** (3/3 banked levels are vc33's, identical both arms) |
| force-admit the trust gate | 27 admit cells, 5 games | **0** |

**Three manipulations, zero movement.** The endpoint has no demonstrated dynamic range in this
corpus, so every per-stage "does not predict" verdict is untestable rather than tested.

The `ret0`/`ret1` identity was used in the first pass only as a reason to discard half the data.
It is in fact the cleanest available evidence that the endpoint is floored.

---

## 6. The two branches of the candidate framing

The framing under test was: *"the trust gate must admit on a property the intervention improves,
or the intervention must improve correctness."*

### Branch 1 — force-admit the gate. **Not refuted; measured only where it could not be observed.**

The first pass claimed "action traces byte-identical in 10 of 10 matched pairs". That claim is
withdrawn as evidence:

- **All 10 matched pairs have `plan_length = 0` in the admit arm.** Byte-identical traces are
  *mechanically entailed*, not independent evidence. Trace-identity was measured only where
  force-admit did nothing.
- All 10 sit on ft09/lp85/sc25/tn36/tu93 — the same five games whose engine arm provably changes
  nothing.
- **vc33, the only banking game and the only cell with dynamic range, has ZERO admit cells.**

The **stronger honest form**: where force-admit produced and executed real plans (lengths 61 and
6, on tn36) the action trace *did* change and `levels_gained` was still 0. Levels gained across
all 27 admit cells: `[0]`.

Correct statement: force-admit is **neutral on five games that bank nothing under any condition**
and was **never applied to the one game with range.** Branch 1 is not refuted in general.

*(The first pass also stated "`plan_length: 0` in every admit cell." That was wrong — four tn36
admit cells produced plans of length 61 and 6. The corrected figure is inline above.)*

### Branch 2 — the intervention must improve correctness. **Direction unestablished.**

| classification | n | table | p | floor | |
|---|---|---|---|---|---|
| paired corpus (cross-population) | 6 | [0,3,2,1] | 0.4000 | 0.4000 | at floor — undecidable |
| live `change_gate` (single population) | 11 | [0,4,1,6] | 1.0000 | 0.3636 | undecidable |

The two classifications **disagree** about which games produce strict-quality engines: the
paired corpus says ft09/lp85/tn36; the live gate says ls20/su15/tn36/tr87. Only tn36 is common.
The first pass mixed populations (classifier from one corpus, endpoint from the other) and
concluded the direction was "certainly not positive". **That phrase is withdrawn** — at these
floors nothing about direction is established.

---

## 7. Where the largest attrition is — and why that is not an attribution

Funnel over all 24 retention cells:

```
induction rounds attempted        64
engine scored on held-out         46   survival 0.719
cleared the DYNAMICS gate          9   survival 0.196
goal predicate satisfiable         4   survival 0.444   <- 5 of 9 exactly-correct engines die here
a plan was found                   4   survival 1.000
```

Corroborated by `results/outer_loop_arc_llm_on_vs_off_activation_20260730.json`: 8 of 10
dynamics-gate survivors were then killed by the goal gate;
`goal_predicate_satisfiable` false in 6 of 7 induction events on its primary `on` arm.

**But that artifact explicitly declines to attribute the bottleneck**, verbatim:
`"This experiment does NOT decide it. Both readings are consistent."` Reading A — *14 of 29
scored rounds predicted held-out transitions at accuracy exactly 0.0* — is equally consistent
with pre-bank goal predicates being degenerate **because the engines feeding them are wrong**,
i.e. still generator-limited.

This note carries that non-attribution forward. The honest statement is: **the goal gate is
where surviving models die; whether that is a goal-induction defect or a downstream symptom of
wrong dynamics is undecided by the available data.** The first pass asserted "on this evidence
it is the goal predicate," which is more than the source supports.

### The exemplar worth keeping

`ret0__tn36__s1` round 1 produced the best engine anywhere in the grid: `heldout_accuracy 1.0`,
`accepted_by_heldout_verifier: True`, `trust_energy -3.219856` (the grid minimum),
`change_accuracy 1.0`, `cell_recall 1.0`, `nondegenerate: True`. It is
`skipped: degenerate_goal_predicate`. It never planned. It banked **0 levels over 346 actions**.

---

## 8. Engine-corpus filtering (corrected figure)

`results/arc_engine_validation_20260731/corpus_scan.json` — **430 of 439 stored engines (97.9%)
are clean**, against **14 of 36 = 38.9% clean** in raw Phase-1 control samples (defect-carrying
61.1%).

The first pass wrote "97.9% clean against 36.1% defect-carrying". **36.1% is the control arm's
USABLE rate (13/36) — a different quantity**; the sentence compared a clean-rate against a
usable-rate. The corrected number strengthens the point: the retention step already filters the
defect the intervention fixes.

---

## 9. Correct engines per GPU-hour

The intervention halves cost (1.54× cheaper: 35.9 → 55.2 attempts/hour). Does that convert?

| channel | control /h | treatment /h | ratio | Poisson p | Poisson CI |
|---|---|---|---|---|---|
| **usable** | 12.96 | 33.72 | **2.60×** | **0.0055** | **[1.25, 5.62]** |
| strict quality | 5.98 | 10.73 | 1.79× | 0.3952 | [0.52, 6.46] |
| strict non-vacuous | 3.99 | 6.13 | 1.54× | 0.7201 | [0.29, 8.25] |

The Poisson rate-ratio is **unpaired** and assumes independent events, but the attempts are
seed-paired and clustered in 6 games (usable-count var/mean 1.18 control, 1.38 treatment —
overdispersed, so that p is anti-conservative). Both omitted checks were run:

| channel | paired sign test | game-cluster bootstrap (6 clusters) |
|---|---|---|
| usable | **p = 0.0025** (+21 / −5) | **CI [2.13, 6.75]**, 0.0000 of resamples <= 1 |
| strict quality | p = 0.1797 (+7 / −2) | — |
| strict non-vacuous | p = 0.3750 (+4 / −1) | — |

**Usable engines per GPU-hour is a real gain — it survives all three tests.**
**Correct engines per GPU-hour is not** — it fails all three, and the non-vacuous *count is
identical, 4 vs 4*; the entire ratio there is the denominator, not the numerator. Deciding it
needs ~7.4× more GPU-hours (266 attempts/arm), ~22× for non-vacuous (793/arm).

Underneath this sits a vacuity problem: Phase 1's strict bar is passed vacuously on ft09 and
lp85, whose held-out sets contain zero changing transitions, so the bar reduces to "did not
hallucinate on no-ops". The live path flags this itself — `noop_ok_is_vacuous` fires on **10 of
21** induction events carrying the field.

---

## 10. What this licenses, and what it does not

**Licenses:**
- refusing to spend GPU-hours on a 12-cell banked-levels grid for this intervention;
- reporting usable-engines-per-GPU-hour as a real, thrice-confirmed cost win;
- treating the goal gate as the largest observed attrition point.

**Does NOT license:**
- claiming any stage was shown *not* to predict banked levels — most were undecidable;
- claiming the trust gate is refuted as a lever — force-admit never ran on vc33;
- claiming correctness predicts levels negatively — direction is unestablished;
- attributing the bottleneck to goal induction rather than to weak dynamics.

**Banked levels remain 3/3 and the submission gate is UNMET.** Nothing was submitted; no scored
or online game was played.

---

## 11. Reproducing this

Every number above is recomputed from committed artifacts by
`results/outer_loop_arc_endpoint_chain_mediation_20260731.json`'s generator. Upstream sources
(read-only, never modified):

- `results/arc_induce_confirm_20260731/confirm_scored.json` — 36 paired inductions
- `results/arc_engine_retention_20260729/cells/*.json` — 24 cells, chain stages + banked levels
- `results/outer_loop_arc_llm_on_vs_off_activation_20260730.json` — whole-tier ablation
- `results/outer_loop_arc_gate_forceadmit_20260730.json` — the direct branch-1 test
- `results/arc_engine_validation_20260731/corpus_scan.json` — 439 stored engines
- `python/carnot/agentic/arc_competition_agent.py` — `min_heldout_accuracy=1.0`, the live gate

Related notes: `arc-induce-repeat-penalty-confirm-2026-07-31.md`,
`arc-induce-repeat-penalty-wired-2026-07-31.md`, `arc-gate-rejection-audit-2026-07-30.md`,
`arc-engine-static-validation-2026-07-31.md`.
