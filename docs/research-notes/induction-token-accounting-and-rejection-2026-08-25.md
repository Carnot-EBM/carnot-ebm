# Where induction tokens go, and what actually gets rejected (2026-08-25)

Two independent investigations of the same question, run in parallel and kept
separate on purpose. Neither agent saw the other's report. Preserved here because
both lived only in a scratch directory that does not survive the session, and both
CORRECT claims the outer loop had already stated out loud.

## Why both are kept

The measurement agent answered the question. The adversarial agent attacked the
framing of the question. They agree on the main result and disagree on one thing,
and the disagreement is recorded rather than resolved.

Agreed:

- The per-call decode is about 63,000 to 65,000 tokens, not the ~40,000 the outer
  loop first reported. The first figure came from a grep that also matched
  `prompt eval time` lines, so 36 prompt measurements were pooled with 36
  completion measurements and halved the median.
- About 97.6 percent of generated tokens never reach an archived program.
- The reasoning channel explains it. Think mode is on by default, the chat
  endpoint splits the thought channel into `reasoning_content`, and the extractor
  reads only `content`.
- Think mode being on is a deliberate operator decision from 2026-08-08, taken on
  induction-quality evidence. It is not a bug to fix.
- "9 of 10 world models rejected" is a CELL-level rate read as a per-model rate.
  Per attempt it is about half.

Not agreed, and left open: this corpus shows about 11 percent of failures on the
dynamics side; a larger prior corpus shows 78 percent. The measurement agent flags
this as a conflict and says its third recommendation must not be acted on until the
two are reconciled.

## Status of the fix that came out of this

Commit `288ea485f9` (REQ-ARC-WMTE-6710) landed the instrumentation both reports
named as the cheapest next step. One correction is on the record: that commit
message quotes "9 of 18 skip records", a prevalence figure that does not reproduce
and that the field cannot measure at all. See the audit findings ledger.

---

# Where the induction tokens go, and what actually gets rejected

Measured 2026-08-24 from evidence already on disk. No GPU job, no llama-server launch, no ARC
run. The only model load was `llama_cpp.Llama(vocab_only=True)` for tokenization — vocab only,
no weights, no GPU (the pattern CLAUDE.md's GGUF tokenizer rule prescribes).

**Clone exclusion, as required.** `$CLAUDE_JOB_DIR/tmp` holds `abpin/` and `headwt/`, byte-identical
clones of this repository. No scan in this report was recursive over `tmp`. Every glob was rooted
explicitly at `tmp/{seedsweep,supon,retention_run}/e3/*/attempts/wm_*.py` or at the three named
`server_logs/` directories. `abpin` and `headwt` were never touched. Independently confirmed in
adversarial review: both clones contain **zero** `wm_*.py` files, so even a fully recursive glob
over `tmp` returns exactly 40. (An earlier draft justified this by agreement with the brief's own
count of 40, which is circular reasoning; the direct check is what supports it.)

Nothing under `results/**` was written. Outputs live in `tmp/wmreject/`.

---

## Corrigendum — corrections made after adversarial review

An independent hostile reviewer checked every citation, recomputed every number, and found real
errors. All headline arithmetic reproduced exactly and the Q1 code path verified line-by-line, but
**the prior-art handling did not**, in the same way three times. Recorded here rather than silently
patched, per CLAUDE.md's error-lifecycle rule.

| # | What was wrong | Correction |
|---|---|---|
| 1 | Claimed these 40 sources are "the corpus that artifact said did not exist". | **False.** `results/outer_loop_arc_induced_engine_taxonomy_20260802.json` already recovered **901** engine sources. I quoted one scope-limited sentence from a sibling artifact against its own author's explicit correction. Novelty claim withdrawn; see prior-art section. |
| 2 | Presented the 0/40 static identity result as "these are not stubs". | The same artifact measured static identity detection at a **67.9% false-negative rate** (163 of 240 true identity engines missed). Static claim withdrawn — and **replaced with a dynamic probe** (1,230 real engine calls) that measures it properly: 1/41 identity, 0/41 raising. The conclusion happens to survive; the original evidence for it did not. |
| 3 | Claimed the rejection distribution "corroborates" the anatomy artifact. | It **reverses** it: 11.1% dynamics-side here vs 77.9% (n=136) / 87.5% (n=88 valid subset) there. Now reported as a conflict with a named, untested hypothesis. |
| 4 | Never executed any of the 40 programs; not disclosed as a limitation. | **Gap closed, not just disclosed.** All 41 programs executed on real 64×64 game frames — see the dynamic-probe section. A harness bug in the first probe run (2-arg call against a 3-arg contract) produced a false "37/41 always raise"; caught and fixed before it reached the report. |
| 5 | Said `planned` has "exactly one writer semantics". | **Wrong.** 13 write sites, ≥7 non-LLM tiers; site 7941 plans with no world model. The 14/27 reading survives only on two preconditions, now stated and verified. |
| 6 | Said 4 of 13 rows violate `planned + Σskipped ≤ attempts`. | **5.** Missed ar25/20260726/S_llmon. |
| 7 | Presented seedsweep as complete. | Also truncated mid-request (43,758 tokens discarded, uncounted). |
| 8 | "37 of 40 linkable" attributed only 2 unlinked to retention_run. | The third is a supon program archived post-snapshot. |
| 9 | "42 of 44 terminated on EOS" stated as measurement. | `stop_type` absent from logs; it is an inference. |
| 10 | Wall-clock "26–46 minutes" and a derivation that yields one number. | Measured: 19.2 / 36.0 / 46.1 min (min/median/max). |
| 11 | Two miscitations: exp6091 comment line range; "void" attributed to the wrong artifact. | Both fixed. |
| 12 | "45 launched" completion requests. | **46** (37+7+2). Caught before review. |

The load-bearing consequence: **recommendation 3 should not be acted on** until the 77.9%-dynamics
prior corpus is reconciled. The execution gap is now closed, so the predictor negative stands on
dynamic evidence rather than static — but at n=8 vs 8 it is still weak.

---

## Corpus and exact n

| Source | n |
|---|---|
| Archived world-model programs | **40** (seedsweep 33, supon 5, retention_run 2) |
| Games | tu93 27, ar25 13 |
| Cells (rows) with a `rows.json` | **13** (seedsweep 12, supon 1) |
| Induction attempts across those 13 rows | **27** |
| LLM completion responses across those 13 rows | **41** |
| Completion requests in the server logs | **46 launched, 44 completed** (seedsweep 37/36, supon 7/6, retention_run 2/2) |
| Archived programs linkable to a row | **37** of 40 |

Three programs are unlinked, not two: `retention_run`'s 2 (it has no `rows.json` at all) plus
`supon/e3/tu93/attempts/wm_20260825T024732_717500__37803dbeb87c293a.py`, archived after supon's
mid-run `rows.json` snapshot. All three are in the program-property statistics and excluded from
every rate that needs a row.

**Two of the three runs were cut off mid-request, and neither cut-off request's tokens are in any
total below.** supon was **still running** when its logs were parsed (llama-server PID 218269 live
on port 8997, one request mid-decode at n_decoded=33573) — its numbers are a snapshot. seedsweep was
**also** truncated: its last launched task (2386467) was SIGTERM'd mid-decode at **43,758 tokens**
by `stop_before_cd82.sh` once 12 rows had landed (`stop_watcher.log`). Those ~43.8k generated tokens
appear in neither the 2,677,498 denominator nor the 2,891,259 decode total. The direction is
conservative — the true discarded volume is higher than reported.

---

## Q1. Where the generated tokens go

### Answer: the model's reasoning channel, and it is discarded by construction.

Think mode is ON by default and has been since 2026-08-08. The induce call therefore routes to
`/v1/chat/completions`, where llama.cpp splits the thought channel into `reasoning_content`. The
extractor reads **only** the `content` channel. The reasoning is never a candidate for extraction.

The proving code path, in order:

- `arc_executable_world_model.py:3917-3918` — `ARC_LIVE_GENERATOR_THINK_DEFAULT = "1"`,
  `ARC_LIVE_GENERATOR_THINK_SCORED_DEFAULT = "1"`.
- `arc_executable_world_model.py:3932` — `induce_think_on()` — returns
  `ARC_LIVE_GENERATOR_THINK_SCORED_DEFAULT != "0"` when `CARNOT_ARC_INDUCE_THINK` is unset.
  **No `run.sh` in any of the three runs sets it** (grepped all three; zero hits).
- `arc_executable_world_model.py:7589` — `_think_on = induce_think_on() and codeonly_eligible`.
- `arc_executable_world_model.py:7609` — `elif _think_on: pass  # no directive, no pre-opened
  fence`. This is where a `/no_think` prefix would have been applied and is not.
- `arc_executable_world_model.py:7717` — `if self.use_chat_template or _think_on:` — think mode
  **forces** the chat endpoint regardless of the instance setting.
- `arc_executable_world_model.py:6572-6573` — `final = str(msg.get("content") or "")`,
  `reasoning = str(msg.get("reasoning_content") or "")`.
- `arc_executable_world_model.py:6598` — **`extraction = final`**. The reasoning channel is
  recorded (`self.last_reasoning_content`) and never extracted from.

Corroborated by the server's own startup line, identical in all three logs:
`init: init: chat template, thinking = 1`.

### On the brief's `/no_think` hypothesis

The brief asks whether the induction path passes `/no_think`, citing `[[project_arc_live_generator]]`.
It does not, and that is **deliberate, not a bug**. `no_think_prefix` defaults to `""`
(`arc_executable_world_model.py:6026`) and the `_think_on` branch bypasses it entirely. The operator
flipped think ON on 2026-08-08 on exp6199's induction-quality evidence. The memory's `/no_think`
note describes the older frozen stack; it is stale with respect to the induce path. Framing this as
an oversight to fix would be wrong — reversing it reverses a deliberate operator decision that had
evidence behind it.

### Measured token accounting

Deliverable size was measured with the generator's own tokenizer
(`Qwen3.8-27B-Q4_K_M.gguf`, `vocab_only=True`), not estimated from bytes.

Across the 13 rows that have a `rows.json`:

| Quantity | Value |
|---|---|
| Generated (sum of `llm.tokens_predicted`) | **2,677,498** |
| LLM responses | 41 |
| Mean generated per response | **65,305** |
| Archived programs linked | 37 |
| Deliverable tokens (sum of the 37 programs) | **64,088** |
| **Deliverable / generated** | **2.39 %** |
| **Discarded** | **2,613,410 tokens (97.61 %)** |

Per-program, all 40: min 350, p25 1,311, **median 1,473.5**, p75 1,959, max 7,203 tokens.
Per-request decode, all 44 completed: min 38,629, p25 59,384, **median 63,563**, p75 72,848,
max 84,144. Total decoded across the 44: 2,891,259.

**So the brief's "roughly 90%" is an understatement. The measured figure is 97.6%.**

### The four rival explanations, each tested

1. **Reasoning tokens — CONFIRMED.** Code path above. This is the explanation.
2. **Multiple internal attempts per call — TRUE, and it multiplies the cost rather than explaining
   the per-call volume.** `tries` defaults to 3 (`arc_executable_world_model.py:6042`) and
   `generate()` runs `while True: attempt += 1 ... if attempt >= _budget` (`arc_executable_world_model.py:7652`) with
   `temperature = 0.2 + 0.1 * attempt`. Measured: **41 responses for 27 induction attempts = 1.52
   LLM calls per attempt**. Per-attempt retention archives each parseable candidate, which is why
   there are 37 archived programs for 27 attempts. This does not explain where 63k tokens inside a
   *single* call go — it means the true cost per accepted model is ~1.5× the per-call figure.
3. **Repetition / looping degeneracy — LARGELY REFUTED for this corpus.** Only **2 of 44** requests
   hit the cap. The cap is `_pool_clamped_n_predict` = `observed_n_ctx − _INDUCE_WORST_CASE_PROMPT_TOKENS`
   = 106496 − 22352 = **84,144** (`arc_executable_world_model.py:4445`), and exactly two requests
   decoded 84,144 — seedsweep tasks 417563 and 995163, on different slots with different prompts,
   so the identical value is the cap and not coincidence. The other **42 of 44 did not hit the cap**;
   `stop_type` is absent from these logs (0 occurrences in all three), so "terminated on EOS" is an
   INFERENCE from "no stop-string configured and not at the cap", not a logged fact.
   `truncated = 1` appears **0 times**; `context shift` appears **0 times**. The near-greedy sampler
   the source warns about (`arc_executable_world_model.py:7669`, quoting Qwen3's own model card on
   endless thinking repetition) is real and is the likely mechanism for those 2, but it is a 4.5%
   tail, not the main story.
4. **Large prompt echoed back — REFUTED, and the hypothesis is also mis-framed.** True prompt
   length (release `n_tokens` − decoded) is min 7,126, median 9,092, max 10,915. Prompts are 4–12×
   *smaller* than the output. Separately, a prompt does not consume decode tokens at all, so prompt
   size could not explain decode volume even if it were large.
5. **Archive truncating or post-processing — REFUTED.** `_extract_python(text)` pulls the fenced
   block out of `final`; nothing truncates. All 40 archived files parse (below), so the archive is
   receiving complete programs.

### What I could not measure here

The exact reasoning-vs-final byte split **per call**. `last_reasoning_content` and
`last_final_content` are captured in memory (`arc_executable_world_model.py:6579-6580`) and
**neither is persisted** to `rows.json` or the attempt record. So I can prove the reasoning channel
exists, is generated, and is structurally excluded from extraction — and I can bound the discarded
fraction at 97.61% — but I cannot from *these* runs split that 97.61% into "reasoning" versus
"prose the model wrote into `content` around the code block". Both are discarded; only their ratio
is unknown. This is a two-integer instrumentation gap and it is the cheapest fix in this report.

---

## Q2. What distinguishes a rejected world model from an accepted one

### First: the "9 of 10 rejected" figure is a cell-level statistic being read as a model-level one

This correction matters more than anything else in Q2, so it comes first.

The "roughly 9 of 10" in the brief traces to `supon/run.sh`'s own prose. Read literally as cells,
it is right: **11 of 12** seedsweep cells have a non-empty `induction_skipped` (91.7%). Only
`ar25 / 20260725 / S_replicate_llmon` has none.

Read as "9 of 10 induced models are discarded", it is **not** supported. Per attempt, across 13 rows:

| Outcome | count | of 27 attempts |
|---|---|---|
| `induction_planned` (a plan was installed) | **14** | **51.9 %** |
| carries a `skipped` label | **18** | 66.7 % |
| 14 + 18 | 32 | **exceeds 27** |

The sum exceeding the attempt count is not an arithmetic slip — it is a real field-semantics defect,
and it is exactly the "field names lie" trap.

**Mechanism, with line numbers.** `skipped` is initialised to the string
`"no_reachable_plan_after_refinement"` at `arc_llm_reinduction.py:1649` **before the refinement loop
runs** — it is the default label, not a measured verdict. `_induce_and_plan` is a *cascade* of tiers.
The stall tier writes `attempt.update({... "planned": bool(stall_outcome.planned), "skipped":
stall_outcome.skipped ...})` at `arc_competition_agent.py:7865-7868` and then, when not planned,
**falls through** to lower tiers. The plain path at `arc_competition_agent.py:8319` can then set
`attempt["planned"] = True` while the stale `skipped` string is left in place. **Five** of thirteen
rows show the overlap directly (`planned + Σskipped > attempts`): tu93/20260725/S_replicate (3+1>3),
tu93/20260726/S_replicate (2+2>3), ar25/20260724/S_replicate (1+2>2), ar25/20260726/S_llmon (1+1>1),
supon tu93 (1+3>3). (An earlier draft said four; it missed ar25/20260726/S_llmon, whose single
attempt carries both a plan and a skip label.)

Consequence: **`induction_skipped` over-reports rejection**, and `no_reachable_plan_after_refinement`
— the single largest bucket — is the bucket most contaminated, because it is the default string. Any
"9 of 10 rejected" claim built on this field inherits the error.

**`planned` is NOT a clean signal either, and my first draft's claim that it has "exactly one writer
semantics" was wrong.** `arc_competition_agent.py` has **13** sites writing `attempt["planned"] = True`,
and at least 7 belong to non-LLM tiers: `active_reward_machine_disagreement_probe` (6431),
`structured_nav_induced` (7507), `ttt_prior_warmstarted` (7567),
`object_relative_trajectory_transfer` (7622), `cross_level_engine_carry` (7644),
`active_probe_pre_llm_disambiguation` (7941), `active_probe_disambiguation` (8059). Site 7941
installs a **one-action probe plan with no world model at all** and still sets `planned = True`. So
`planned` does not by itself mean "an LLM-induced world model was used".

The 14/27 reading survives on **two preconditions that must be checked, not assumed**, both of which
hold on this corpus:

- **`induction_engine_sources == {}` on all 13 rows.** All 7 tiers listed above tag themselves with
  `engine_source`; an empty counter means none of them fired. The plans came from sites that set no
  tag (`arc_competition_agent.py:7717 / 7867 / 8319 / 8346 / 8382`), which are the LLM paths.
- **`induction_attempts_llm_reached == induction_attempts` on all 13 rows (27 of 27).** This is the
  purpose-built signal (`scripts/arc_scored_path_lever_harness.py:836`, keyed on `model_specs !=
  "offline_dsl_induction_no_llm"`), and its own comment warns that inferring LLM reach from
  `skipped`/`engine_source` instead once "reported llm=0 on a cell that had in fact issued 11 real
  completions". My first draft did not use it. It should have been the first thing checked.

Honest reading of this corpus, with those preconditions verified: **14 of 27 induction attempts
(51.9%) reached the LLM and ended with a plan installed by an LLM-path call site.** The binding
constraint is real but roughly half, not nine tenths — **on these 2 games / 3 seeds / 13 cells**.
Note this sits against `n_planned: 0 of 136` in the 2026-07-27 corpus; see the prior-art section.

### Rejection-reason distribution

Sum over the 13 rows, n = 27 attempts. Read as labels present, not as disjoint verdicts, per the
contamination above.

| reason | count | % of 27 attempts |
|---|---|---|
| `no_reachable_plan_after_refinement` | **9** | 33.3 % |
| `degenerate_goal_predicate` | **4** | 14.8 % |
| `goal_unreached_within_budget` | **3** | 11.1 % |
| `world_model_accuracy_below_threshold` | **1** | 3.7 % |
| `hidden_state_trust_below_threshold` | **1** | 3.7 % |
| total labels | 18 | 66.7 % |

Two structural notes:

- **12 of 18 labels (66.7%) are goal- or plan-side, not dynamics-side.** Only 2 of 18 are the
  dynamics-accuracy gates (`world_model_accuracy_below_threshold`, `hidden_state_trust_below_threshold`).
  The dominant failure is not "the model predicts badly" — it is "nothing could be planned with it."
- **`CARNOT_ARC_PLAIN_PATH_GOAL_SATISFIABILITY_CHECK` is not set in any of the three `run.sh`
  files**, so the plain-path goal check (`arc_competition_agent.py:8272`) never fired. The
  `degenerate_goal_predicate` and `goal_unreached_within_budget` labels here therefore came from
  the refinement path (`arc_llm_reinduction.py:2030, 2065`), not the plain path.

`induction_exceptions`, `induction_tracebacks`, `induction_proposer_notes` and
`induction_engine_sources` are **empty on all 13 rows**. Nothing raised, no proposer failed, and no
plan carried an `engine_source` tag — consistent with every installed plan coming from a call site
that does not set that field (e.g. `arc_competition_agent.py:8319`), not from the structured-nav or
ttt-prior tiers.

### Predictor analysis: no STATIC property predicts rejection — and static is the wrong instrument

This is a negative, but a **weaker one than I first wrote**, for a reason I should have caught
before running it: the first version of this section never executed a single program. Every
property in the table below is a static AST read, and the prior taxonomy artifact measured that
exact method at a **67.9% false-negative rate** on the identity question. The static table is
retained for the record; **the dynamic probe that follows it is the one to read.**

**Every crude defect hypothesis fires at 0 or 40 — no variance at all** (n = 40, all static):

| property | result |
|---|---|
| parses under `ast.parse` | **40 / 40** |
| defines both `engine` and `is_level_complete` | **40 / 40** |
| imports numpy | **40 / 40** |
| engine is identity (every `return` returns arg 0 unmodified) | **0 / 40** |
| engine has no `return` on any path | **0 / 40** |
| `is_level_complete` has no `return` | **0 / 40** |
| `is_level_complete` can only return literal `False` | **0 / 40** |
| `is_level_complete` can only return literal `True` | **0 / 40** |

These are substantial programs by size — median 167 lines, 1,473.5 tokens, `engine` 79.5 lines with
14.5 `if` branches, `is_level_complete` 17.5 lines — and they define the right interface and parse.
**I originally wrote "these are not stubs"; that is not supported and is withdrawn.** Size and
branch count do not establish that an engine mutates anything: the taxonomy artifact's dynamic
probe found 11.6% identity and 7.2% structurally broken on the current generator, and those defects
are invisible to every row of the table above. Whether these 40 are stubs is **unmeasured here**.

**Continuous properties do not separate either.** Cleanest attributable split — cells where *no*
attempt planned (group A) versus cells where *every* attempt planned (group B), 8 programs each,
Mann-Whitney U two-sided:

| property | median A (no plan) | median B (all plan) | U | p |
|---|---|---|---|---|
| tokens | 1,489.5 | 1,787.0 | 30.0 | 0.878 |
| lines | 166.0 | 198.5 | 23.0 | 0.382 |
| engine lines | 87.5 | 90.0 | 25.0 | 0.505 |
| goal lines | 12.0 | 25.0 | 17.5 | 0.141 |
| engine branches | 15.0 | 19.0 | 28.0 | 0.713 |

Nothing reaches significance; the smallest p is 0.141 on goal length, in the direction "accepted
goals are longer", which is a hypothesis worth one cheap follow-up and nothing more at n = 8 per
group. **No static property of the program predicts whether it will be used.**

### DYNAMIC probe: the engines were executed, and they work at the interface level

Added after adversarial review flagged non-execution as the report's largest gap. Every archived
program was imported in isolation and its `engine` called on **6 real 64×64 frames of its own game**,
pulled from the offline arcade (`environment_files/`, zero network, zero quota), for actions 1–5.
**41 programs × 30 (frame, action) pairs = 1,230 real engine calls.** Script: `dynprobe.py`,
frames: `getframes.py` → `frames.json`, per-program results: `dynprobe.json`.

The corpus grew from 40 to **41** during this session — supon is still running and archived one more.
All counts in this subsection are n=41 (tu93 28, ar25 13); every other section of this report is the
n=40 snapshot taken earlier.

**A harness bug caught and fixed before reporting, worth recording.** The first run reported "37 of
41 raise on every call", which would have been a dramatic and false finding. The cause was mine: the
engine contract is **three positional arguments**, `engine(grid, action, data)`
(`arc_executable_world_model.py:2636`), and many programs declare `data` without a default. I had
called it with two. The corrected probe passes `None` for `data`, matching the real caller.

| dynamic property (n = 41, 1,230 calls) | result |
|---|---|
| imports cleanly | **41 / 41** |
| raises on at least one call | **0 / 41** |
| returns wrong shape or a non-array | **0 / 41** |
| **identity — never changed the grid on any of 30 pairs** | **1 / 41 (2.4 %)** |
| changed the grid on **all** 30 pairs | 29 / 41 |
| changed the grid on fewer than half the pairs | 4 / 41 |

Fraction of (frame, action) pairs where the engine changed the grid: min 0.000, p25 0.800,
**median 1.000**, max 1.000.

**What this establishes.** The generator is not emitting stubs. The engines import, never raise,
always return a correctly-shaped grid, and 40 of 41 genuinely mutate it — the median engine changes
the grid on every action tried. The single identity engine is `5b1fd5fa5987500e` (tu93, supon).
This is a real measurement of the property the static table could only guess at, and it **partly
vindicates the static result**: static said 0/41 identity, dynamic says 1/41, so the static pass
missed exactly one here rather than the ~5 the 67.9% miss rate would predict. Small n; do not read
that agreement as a rehabilitation of the static method.

**What it does NOT establish, and one number that must not be over-read.**
`is_level_complete` returned True on **0 of 41** programs across all frames — and that is **not**
evidence of degenerate goals. The 6 frames are the first frames after reset, when the level is
genuinely not complete, so a *correct* predicate should return False on every one of them. The
probe simply contains no win states. It is a null result about the probe, not about the goals.
Likewise, "the engine mutates" is a far weaker property than "the engine mutates *correctly*" —
this probe never compares against the true next frame, so it says nothing about accuracy, and
nothing about root-liveness (GAP-6260), which needs successor novelty at the planning root.

Attribution limit, stated plainly: linkage is **cell-level**, not attempt-level. `induction_archive`
gives a per-row list of sha256 prefixes but no per-attempt outcome, so in a cell with 1 plan out of
2 attempts I cannot say which program was the accepted one. Groups A and B are the only cleanly
attributable cells, which is what caps this analysis at 8 versus 8.

### How this sits against the prior art — including where it CONFLICTS

`results/outer_loop_arc_induce_gate_anatomy_20260802.json`
(`complete_gate_is_binding_and_reached_but_rejections_are_not_near_misses`) measured margins on 52
of 136 attempts: hidden-state branch median `heldout_change_consistency` 0.0000 against a 0.5
threshold, plain-branch `cell_recall` maxing at 0.0476. This corpus adds nothing that contradicts
that and does not revisit it.

**Its sibling, `results/outer_loop_arc_induced_engine_taxonomy_20260802.json`, is the more important
one, and it constrains two of this report's results.** Read it before acting on anything here.

1. **These 40 programs are not a novel corpus.** The taxonomy artifact's
   `engines_recoverable_note` reads: *"Emphatically yes, and 4.4x wider than the previous
   enumeration: 901 distinct engine sources are on disk or in git RIGHT NOW, all of them
   re-scoreable offline on CPU in about two minutes of wall time for the whole corpus."* 901
   versus 40. What these 40 do add is **run-attributable provenance** — each is tied to a known
   cell, seed, arm, and generator config — which the 901 largely are not. That is the honest
   framing of their value, and it is much narrower than novelty.

2. **The 0/40 static identity result is unreliable, by a rate that artifact measured.** Its
   `measurements.static_vs_dynamic`, over 864 units:
   `static=False|dynamic=True: 163`, `static=True|dynamic=True: 77`, `static=True|dynamic=False: 7`,
   `static=False|dynamic=False: 617`. A static AST pass **misses 163 of 240 true identity engines —
   a 67.9% false-negative rate**, because it cannot see mutation delegated to a helper or tell a
   live branch from a dead one. Its dynamic probe found identity at **11.6%** on the clean
   current-generator split and **7.2% structurally broken**. At 11.6%, roughly 4–5 of these 40
   would be expected to be identity engines that my static pass scored as 0. **My "0/40, these are
   not stubs" line should be read as "0/40 by a method with a measured 67.9% miss rate", which is
   close to no evidence at all on that question.**

3. **The rejection-reason distribution is REVERSED against the larger prior corpus, and I cannot
   explain it.** Anatomy `PART_A.skip_census`, n=136: `world_model_accuracy_below_threshold` 63 +
   `hidden_state_trust_below_threshold` 43 = **106 of 136 (77.9%) dynamics-side**. On its own
   defensible `valid_rows_only_subset` (n=88): 43 + 34 = **77 of 88 (87.5%) dynamics-side**. Mine
   is **2 of 18 (11.1%) dynamics-side**. That is a reversal, not a corroboration, at 5× my sample
   size, and it is the fact recommendation 3 rests on.

   A plausible reconciliation — **hypothesis, not a finding**: two changes landed between that run
   (2026-07-27 cells) and these (2026-08-24). Think mode was flipped on 2026-08-08, and the induce
   budget was corrected from the stale 4096 to 131072 on 2026-08-21 (REQ-ARC-WMTE-6620). Either
   could plausibly have raised dynamics quality enough to clear the accuracy gates and expose a
   plan-side blocker underneath. That story fits, and it is also exactly the kind of story that
   fits any reversal after any change. It is untested here.

   The other stark contrast the two corpora show: **`n_planned: 0` of 136 there, versus 14 of 27
   here.** If that difference is real rather than a corpus artifact, it is a larger result than
   anything else in this report, and it deserves its own measurement rather than a paragraph.

---

## What would fix it, ranked

Each item is tagged **generation-side** or **acceptance-side**, and marked
**[measured]** where this report's data supports it or **[inference]** where it does not.

### 1. Persist the two channel lengths. Generation-side (instrumentation). [measured gap]

Two integers on the attempt record: `len(last_reasoning_content)` and `len(last_final_content)`,
already computed at `arc_executable_world_model.py:6579-6580` and thrown away. Without them, the
97.61% discard cannot be split into reasoning versus prose, and every lever below is chosen blind.
This costs nothing, changes no behaviour, and is the precondition for evaluating item 2. The
project has been here before — the `arc_executable_world_model.py:6114-6115` comment records exp6091
spending a 19-cell run unable to tell the two channels apart.

Do this first. It is the only item I would run without further argument.

### 2. Cap the thinking budget rather than disabling thinking. Generation-side. [inference]

Measured `total time` across the 44 completed requests: min **19.2 min**, median **36.0 min**, max
**46.1 min** per LLM call — at 1.52 calls per attempt, roughly 55 min per induction attempt at the
median. The payload already
supports `thinking_budget_tokens` (`arc_executable_world_model.py:6560`). A cap at, say, 8k would
cut wall-clock roughly 8× and let the same GPU-hours buy ~8× more induction attempts, or a real
seed sweep at n large enough to answer questions this report had to leave at n=8.

**Marked inference, and the caveat is load-bearing:** think mode was turned on deliberately on
2026-08-08 on exp6199's induction-quality evidence, and this report contains **no** measurement
that a shorter budget preserves program quality. It could easily be worse. Treat this as an A/B to
run — arms `budget=unlimited` (control) versus `8k` versus `16k`, scored on `induction_planned`
per attempt — not as a change to ship. Do item 1 first so the arms are readable.

### 3. Run the root-liveness check (GAP-6260) — but reconcile the conflict FIRST. Acceptance-side. [CONTESTED]

**Do not act on this before resolving the conflict named in the prior-art section.** In this corpus
12 of 18 rejection labels are goal- or plan-side and only 2 of 18 are dynamics-accuracy. In the
5×-larger 2026-07-27 corpus the split is the reverse — 106 of 136 (77.9%) dynamics-side, or 77 of 88
(87.5%) on its valid-rows subset. One of those two corpora is unrepresentative and I do not know
which. Acting on mine would be acting on n=18 labels from 2 games against n=88–136 from 14.

What is worth doing regardless, because it is cheap and its evidence does not depend on which
distribution is right, is the concrete already-specified candidate: **GAP-6260** in `ops/verifier_gaps.md`: root-liveness — expand the
planning root under all actions, hash the successors, count the novel ones, and reject or re-induce
on zero. It is a handful of engine calls, runs before the search rather than after it, and directly
targets `no_reachable_plan_after_refinement`, the largest bucket.

Corroborating live evidence in this corpus: `seedsweep/runner.log` carries **7** `WINDOW VOCABULARY
VIOLATION` lines — action 7 appears in the induction window and the in-model planner cannot emit it,
so the induced engine models transitions the planner will never request. The log's own text says
such an engine "may be inert at the planning root even at cell_recall 1.0." That is a plan-side
defect with zero relationship to program TEXT, which is a plausible reason no static property
separated the two groups — though with the static method's measured 67.9% miss rate, "no static
property separated them" is weak support for any explanation.

The cheapest way to settle both this and the predictor question at once: **execute the engines.**
The taxonomy artifact reports its whole 901-source corpus is re-scoreable "offline on CPU in about
two minutes". Running root-liveness and an identity probe over these 40 — plus, ideally, the 901 —
costs minutes and would replace two of this report's weakest claims with measurements.

### 4. Fix the `skipped`/`planned` overlap. Acceptance-side (correctness of the record). [measured]

`skipped` is initialised to a verdict string at `arc_llm_reinduction.py:1649` and survives into rows
where a later tier planned successfully. Two options: initialise to `""` and set the label only where
it is actually determined, or have the cascade clear `attempt["skipped"]` whenever it sets
`planned = True`. Either way, add a regression test asserting `planned + Σskipped ≤ attempts` on a
cascade that plans at a lower tier — that invariant is violated by 5 of 13 rows today and
nothing catches it.

This is a guard whose pattern is narrower than its concept, in the field the project uses to decide
whether the generator is working. Under CLAUDE.md's error-lifecycle step 6, it is check-able cheaply,
so it should be a check.

### 5. Do not touch the accuracy threshold. Acceptance-side. [measured, prior art]

Named to close it off rather than to recommend it. `outer_loop_arc_induce_gate_anatomy_20260802.json`
already measured that rejections are not near-misses and that the `cell_recall` metric which would
separate them ran as an arm and still installed 0 plans (in-arm max 0.0476 against a 0.5 threshold).
The word **void** for that arm — admission was arithmetically impossible, so 0 plans is not a
negative result — comes from `results/outer_loop_arc_cellrecall_admit_ceiling_20260802.json`, not
from the anatomy artifact.
This corpus adds no reason to revisit it, and it agrees on direction.

---

## What I could not determine, and why

1. **The reasoning-versus-prose split inside the discarded 97.61%.** Neither channel length is
   persisted (item 1). I can prove the reasoning channel is generated and structurally excluded from
   extraction; I cannot give its share.
2. **Per-attempt attribution of programs to outcomes.** `induction_archive` is a per-row sha list
   with no per-attempt outcome, so a cell with 1 plan of 2 attempts is unattributable. This caps the
   predictor analysis at 8 versus 8 and is why no p-value here is worth much.
3. **Per-attempt `verify_accuracy` / `verify_cell_recall`.** Computed on the attempt dict
   (`arc_competition_agent.py:8207`) but **not projected** by the harness —
   `scripts/arc_scored_path_lever_harness.py:799` lifts only the counters. Grepping `seedsweep/rows.json`
   for `verify_accuracy` returns **0** hits. So I cannot say how far below 0.5 the one
   `world_model_accuracy_below_threshold` rejection fell, and cannot reproduce the anatomy artifact's
   margin analysis on this corpus.
4. **Whether the 2 cap-hits are repetition loops.** `stop_type` is absent from these llama.cpp logs
   (0 occurrences in all three); I inferred the cap from `_pool_clamped_n_predict` = 84,144 and the
   two exact matches. No repetition detector runs, and the raw completions are not retained, so the
   mechanism of those 2 is unconfirmed.
5. **Whether the engines are ACCURATE, as opposed to merely non-degenerate.** The dynamic probe
   (added after review) shows 40 of 41 engines mutate the grid and none raise, but it never compares
   a predicted frame against the true next frame, so it measures liveness, not correctness. It also
   does not measure root-liveness (GAP-6260) — successor novelty at the planning root — which is the
   property most likely to explain the largest rejection bucket. Both are one short script away
   using the same offline arcade this probe already drives.
6. **Which of the two conflicting rejection-mode distributions is representative** (11.1% vs 77.9%
   dynamics-side). Resolving it needs the 2026-07-27 corpus re-scored under the current generator
   config, or the current config run over more games.
7. **Whether any of this generalises past tu93 and ar25.** Two games, 3 seeds, 2 arms, 13 cells,
   27 attempts. Two of the three runs were truncated mid-request. Every rate in this report should
   be read with those n's attached.

---

## Reproduction

Scripts and intermediate data under `/home/ianblenke/.claude/jobs/ad0c053d/tmp/wmreject/`:

- `parse_logs.py` → `requests.json` — per-request prompt/decode/wall/stop from the three server logs
- `report.py` — distribution stats over `requests.json`
- `prog_tokens.json` — per-program token counts, generator tokenizer, `vocab_only=True`
- `prog_props.json` — per-program AST properties (the STATIC pass)
- `linkage.json` — sha256 prefix → cell outcome, from `induction_archive`
- `getframes.py` → `frames.json` — 6 real 64×64 frames per game from the offline arcade
- `dynprobe.py` → `dynprobe.json` — the DYNAMIC probe, 1,230 real engine calls

**Nothing tracked was written.** Verified with `git status` after every step, including after the
arcade run: `results/` is clean and no scorecard or recording landed in the repo. `results/**` was
read only — `outer_loop_arc_induce_gate_anatomy_20260802.json`,
`outer_loop_arc_induced_engine_taxonomy_20260802.json`, and
`outer_loop_arc_cellrecall_admit_ceiling_20260802.json`. Other agents' in-flight edits to
`arc_executable_world_model.py`, `arc_llm_reinduction.py` and `arc_scored_path_lever_harness.py`
were present in the working tree and were left untouched.

All line numbers in this report were read against that working tree on 2026-08-24. Tasks #77–79 are
actively editing two of those files, so line numbers may drift; the identifiers are stable.
