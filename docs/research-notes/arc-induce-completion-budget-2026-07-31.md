# The induce budget is not the lever; a repetition penalty is

Raising the induce completion budget 4x produced zero usable engines on ft09 and zero engines that
change the grid at all. Holding the budget fixed and switching on one repetition penalty produced
a non-degenerate engine on every attempt, with ft09's real mechanic in it, in under half the
budget. The lever is the sampler, and the scored path currently runs with every repetition control
off — not by a tuned decision, but because `generate()` never names the parameter.

Nothing here is shipped. The recommendation is a measurement, not a default change.

**Date:** 2026-07-31
**Artifact:** `results/outer_loop_arc_induce_budget_phase1_20260731.json`
**Evidence:** `results/arc_induce_budget_20260731/` (every prompt, every raw completion)
**Substrate:** live gemma-4-31B-it Q4_K_M on two CUDA llama-servers, build read from
`/proc/<pid>/exe`, per-PID VRAM residency recorded per lane.

## The question

ft09's induction was diagnosed as budget-starved: the generation "spiralled into a wall of `#`
comments and hit the 4096-token output cap," so raise the cap. This note measures whether that
works, states the `n_ctx`/`max_tokens` arithmetic the raise is coupled to, and says what fits on
a 24 GiB card.

## What was already on record, in both directions

This lever has been tested before, and the two prior results disagree — which is why it was worth
re-measuring rather than either assuming or dismissing.

**Against.** `arc_executable_world_model.induce()` carries an inline finding from
`proto_l2_fix_finder` (2026-06-25), written to justify the split-induce fallback:

> the focused goal call is valid in ~3.5s where the combined call fails
> (**a budget bump does NOT help; the model just rambles more**)

**For.** REQ-ARC-FCP-5699-34 (2026-07-16) replayed the *refactor* call on a 27B model and measured
that `max_tokens=8192` plus the structural reminder turns `[HIT n_predict=4096 OUTPUT LIMIT]` into
a natural `stop_type='eos'` with both required functions present. REQ-ARC-FCP-5699-35 then
declined to graduate 8192 into the live default, and said exactly why:

> the 8192 requirement REQ-ARC-FCP-5699-34 found was specific to a 3x larger, non-live candidate,
> not this one

— where "this one" was the 9B live generator. **The live generator became that 3x-larger class on
2026-07-28**, when it was re-pinned to gemma-4-31B-it. The stated reason for keeping 4096 expired
with the model switch, and the default never moved. That is the changed-since-the-prior-failure
this rerun rests on, and it is why the sweep covers the induce and refactor calls separately: the
2026-06-25 note is about the combined induce call, REQ-34 is about refactor, and conflating them
is what let both results sit in the tree unreconciled.

## Method

The ft09 trajectory up to its induction point does not depend on whether a generator exists — the
2026-07-30 grid measured LLM-ON and LLM-OFF action traces as byte-identical on this game
(normalized sha256 `06f1e397129a03d9` on both arms and on the reconstructed control). So the
prompts were captured by re-running the OFF arm with a proposer that records its `induce`
arguments instead of answering them: 6.4 s of pure-Python arcade stepping instead of ~1100 s of
GPU decode, yielding the byte-exact strings the server would have received (code-only directive
and opened fence included, since all three induce calls are `codeonly_eligible`).

**That inheritance was then checked rather than assumed.** The capture's action trace was rebuilt
under the grid harness's own label encoding and compared element-by-element against the banked
live `on__ft09__s1` cell: **60 of 60 identical**, sha256 `fba0baa5d5473eac…` on both. So these are
the prompts the live LLM-ON cell actually built, not a reconstruction that resembles them — which
matters, because a trajectory that diverged even once would put the induce call at a different
transition set and make every number below about a different question.

Each prompt was then fired at 4096 / 8192 / 16384 `n_predict`, three attempts each at the live
path's own temperatures (0.2 + 0.1·attempt) and its own per-attempt seeds
(`CARNOT_ARC_GENERATOR_SEED=3003` → `3003000 + attempt`), against a server pinned exactly as the
live cells pin it: `n_ctx=32768`, `ffn_cpu_layers=0`, MTP off, `CUDA_VISIBLE_DEVICES=""` on the
parent, non-default ports 8933/8934, CUDA build proven from `/proc/<pid>/exe` and per-PID VRAM.

**The acceptance check is not the one `generate()` uses.** `generate()` accepts any completion
containing the required `def`s that parses — and ft09's banked engine passes that bar while
returning `None` on every click. Each completion is therefore also run through an AST reachability
analysis asking whether `engine()` returns on *every* path. That analyser was unit-checked against
the banked ft09 engine (correctly `False`) and a hand-written returning engine (`True`) before any
GPU time was spent on it.

## The arithmetic, and what fits on a 24 GiB card

Two inequalities have to hold at once, and raising `max_tokens` pushes them in opposite
directions.

**(1) Pool admission.** `llama-server` runs 4 `kv_unified` slots sharing ONE pool of `-c` cells:

```
n_ctx  >=  K_concurrent * (prompt_tokens + max_tokens)
```

**(2) VRAM fit** (the measured gemma-4-31B-it Q4_K_M, mtp off, 4 slots, `L` CPU-offloaded FFN
layers), plus the shipped 1500 MiB guard margin:

```
MiB = 18940.7 + 0.050293*n_ctx + 206.83*4 - 195.3*L
```

Both cards in this run reported per-PID residency of **21416 MiB** at `n_ctx=32768, L=0` — exactly
what the envelope predicts. That is an independent confirmation of the fit at a point it was
already fitted on, taken incidentally rather than sought.

**ft09's real induce prompt is 4343 tokens**, measured through the generator's own tokenizer
(`llama_cpp.Llama(model_path=..., vocab_only=True)` on the `.gguf` path — never `AutoTokenizer` on
a GGUF repo id). That is far below the 15767-token worst case `_default_induce_n_ctx()` is sized
for, and the gap is the whole reason the local pin works.

### What the shipped default derivation asks for, and whether a 24 GiB card can give it

`_default_induce_n_ctx()` returns `ceil4096(4 * (15767 + max_tokens))`. Against an idle 3090
(24123 MiB free observed):

| max_tokens | → n_ctx | VRAM (L=0) | + guard | fits 24123? | CPU-FFN layers needed |
|---|---|---|---|---|---|
| **4096 (shipped)** | 81920 | 23888 | 25388 | **no** | 7 |
| 6144 | 90112 | 24300 | 25800 | no | 9 |
| 8192 | 98304 | 24712 | 26212 | no | 11 |
| 12288 | 114688 | 25536 | 27036 | no | 15 |
| 16384 | 131072 | 26360 | 27860 | no | 20 |

**The shipped default already does not fit a 24 GiB card.** That is not a consequence of raising
the budget — it is the state of the tree, and it is why every local measurement pins
`CARNOT_ARC_INDUCE_N_CTX=32768` by hand. The largest 4096-multiple pool a 24 GiB card holds at
`L=0` is **53248**; `L=8` reaches 86016 and `L=16` reaches 118784, at ~195 MiB of VRAM traded per
layer moved to system RAM (and a decode-speed cost this note did not measure).

### What the pinned 32768 pool admits

| K concurrent | per-slot cells | max_tokens ceiling for ft09's 4343-token prompt |
|---|---|---|
| **1** (every cell in this session) | 32768 | **28425** |
| 2 | 16384 | 12041 |
| 4 (the eval-framework shape) | 8192 | **3849** |

Two things follow. First, at K=1 the 4096 stop was never a shared-pool truncation — the pool had
24329 cells to spare — so it was the *intended* budget limit, which is the case a bigger
`max_tokens` can address. The sweep is therefore a fair test of the lever. Second, at K=4 the same
pinned pool leaves ft09 **3849** cells, i.e. *below* the shipped 4096 default: under the eval
framework's one-thread-per-game shape, raising `max_tokens` buys nothing at this pool size until
`n_ctx` rises with it, and today's 4096 is itself already over the line.

## Results

**No budget produced a usable engine on either induce call.** The `usable` column below is the
AST bar — accepted by `generate()` AND `engine()` returns on every path. It is reported first
because it is the bar the Phase-1 question was posed against, and then immediately superseded: the
next section shows an identity engine clearing it, so the real headline is the behavioural one.


| call | budget | n | accepted by `generate()` | **usable** | hit the cap | mean `predicted_n` | mean code lines |
|---|---|---|---|---|---|---|---|
| combined induce | 4096 | 3 | 0/3 | **0/3** | 2/3 | 3813 | 20.3 |
| combined induce | 8192 | 3 | 0/3 | **0/3** | 2/3 | 6544 | 20.3 |
| combined induce | 16384 | 3 | 0/3 | **0/3** | 2/3 | 12005 | 20.3 |
| engine-only induce | 4096 | 3 | 3/3 | **0/3** | 3/3 | 4096 | 10.3 |
| engine-only induce | 8192 | 3 | 3/3 | **0/3** | 3/3 | 8192 | 6.0 |
| engine-only induce | 16384 | 3 | 3/3 | **0/3** | 3/3 | 16384 | 6.0 |
| refactor | 4096 | 3 | 2/3 | **2/3** | 1/3 | 2843 | 74.3 |
| refactor | 8192 | 3 | 1/3 | **1/3** | 2/3 | 5603 | 14.7 |
| refactor | 16384 | 3 | 1/3 | **1/3** | 2/3 | 11065 | 12.0 |


The 4096 rows reproduce the live ft09 failure faithfully, which is what makes the rest
readable: the combined call fails structurally (0/3 accepted, so the live path falls through to
the split fallback), and the engine-only call is accepted 3/3 while returning `None` on every
click 3/3 — exactly the banked engine's shape.

## What the failure actually is

The generator is not running out of room. It reaches a genuine impasse — on ft09 it is the
progress-bar mechanic, `# Let's just find the first pair of 2 cells from the right that are not
11.` — and then emits the same lines over and over until the budget is gone. At 16384 the
combined call's completion is 5345 non-blank lines drawn from **24 distinct ones**.

The decisive number is the **distinct-line count at matched seed and temperature**, because it
says what the extra tokens were spent on:


| call | seed/attempt | distinct non-blank lines (4096 -> 8192 -> 16384) | non-blank lines emitted | code lines |
|---|---|---|---|---|
| engine-only induce | a=0 | 99 -> 67 -> 67 | 190 -> 495 -> 1041 | 18 -> 5 -> 5 |
| engine-only induce | a=1 | 46 -> 46 -> 46 | 1089 -> 2455 -> 5185 | 5 -> 5 -> 5 |
| engine-only induce | a=2 | 118 -> 118 -> 118 | 481 -> 1505 -> 3553 | 8 -> 8 -> 8 |
| combined induce | a=0 | 29 -> 24 -> 24 | 1233 -> 2614 -> 5345 | 5 -> 5 -> 5 |
| combined induce | a=1 | 114 -> 114 -> 114 | 117 -> 117 -> 117 | 23 -> 23 -> 23 |
| combined induce | a=2 | 131 -> 133 -> 133 | 192 -> 349 -> 664 | 33 -> 33 -> 33 |


At two of three engine seeds and two of three combined seeds, doubling the budget leaves the set
of distinct lines **exactly unchanged** while the emitted length doubles or triples. The extra
tokens bought no new content at all — the model cycled what it had already written. That is the
2026-06-25 "the model just rambles more" finding, reproduced on the current generator, on both
call shapes, with a dose-response.

Two internal controls say the instrument is sound rather than the effect being noise:

- `combined` attempt 1 stops naturally on the closing fence at `predicted_n = 3247` and is
  **byte-identical at 4096, 8192 and 16384** — sha256 `bedf7b322f319e79…` on all three, checked
  rather than inferred from matching summary statistics. When the model does not hit the cap the
  budget is correctly irrelevant, and the opt-in sampler seed makes that visible instead of
  assumed.
- Every capped call reports `predicted_n` exactly equal to its budget with
  `stop_type == "limit"`, and never the short-of-budget signature of a shared-pool truncation. The
  pool arithmetic above predicted that, and the server confirms it.

**A repetition penalty is not in play, and nobody chose that.** Read from the running server's own
`/props`: `repeat_penalty 1.0`, `dry_multiplier 0.0`, `frequency_penalty 0.0`,
`presence_penalty 0.0` — every repetition control off. `LocalGGUFProposer.generate()` sends only
`{prompt, n_predict, temperature, cache_prompt, seed, stop}`, so those defaults are what every
induce call in the scored path runs under. That is not a tuned choice; it is the server's default
reaching a code path that never names the parameter.

So the sweep includes a sampler control arm — and it is the one thing in this whole measurement
that works. It is reported below and NOT shipped: changing how the scored agent samples is a
behaviour change, not a measurement change, and this session's remit was the budget.

### Sampler control — the same prompt, seed, temperature and budget, one flag apart

Engine-only induce prompt, budget 4096, 3 seeded attempts per arm, one server, one card.
`off` is the SHIPPED configuration, re-run inside this script rather than reused from the
budget lane, so the comparison never crosses a server process.

| sampler arm | hit the 4096 cap | stopped naturally | accepted | returns on all paths | **changes the grid** | mean cell recall |
|---|---|---|---|---|---|---|
| `off` | 3/3 | 0/3 | 3/3 | 0/3 | **0/3** | 0.0000 |
| `dry_0.8` | 1/3 | 2/3 | 2/3 | 1/3 | **1/3** | 0.1053 |
| `repeat_penalty_1.1` | 0/3 | 3/3 | 3/3 | 3/3 | **3/3** | 0.9474 |

`repeat_penalty_1.1` is the only configuration in the entire sweep — across three call
shapes, three budgets and three sampler arms — that produces a non-degenerate engine, and
it produces one on every attempt. The shipped arm burns the whole 4096-token budget and
returns nothing; `repeat_penalty` stops naturally at 1133-1912 tokens with the real ft09
mechanic in it — a 6x6 block at `(y-2, x-2)` toggling between colours 8 and 9:

```python
current_color = new_grid[target_row, target_col]
if current_color == 9:    new_color = 8
elif current_color == 8:  new_color = 9
else:                     return new_grid  # Not a togglable block
new_grid[target_row:target_row + 6, target_col:target_col + 6] = new_color
```

**The scorer behind that `changes the grid` column was cross-validated against the
shipped gate, to the cell.** A behavioural scorer written for one measurement is worth
nothing if it disagrees with the gate the live agent runs. Scoring the `repeat_penalty
1.1` engine over the same 25 transitions gives 216 correct changed cells of 228 — cell
recall 0.9474. The live `onb` cell's recorded gate fields are
`verify_correct_changed_cells: 216`, `verify_cell_recall: 0.9474`,
`verify_change_fidelity: 0.947368`. Identical. Which also means this arm reaches, on its
FIRST naturally-stopped attempt in 1912 tokens, the cell recall the live LLM-ON run only
reached on one replicate after three refinement rounds.

**Three caveats, all of which matter before anyone acts on this.**

1. **The cell recall is IN-SAMPLE.** All 6 of ft09's 25 grid-changing transitions appear
   in the induce prompt; the 16 held-out transitions are every one a no-op, so there are
   ZERO held-out changed cells and this measurement cannot speak to generalisation. What
   it does show is that the model can express the mechanic it was shown, which the shipped
   arm never got far enough to attempt.
2. **n=3, one game, one prompt.** A sampler change touches every generation the scored
   agent makes, on games this was not measured on.
3. **It is NOT shipped here, deliberately.** Changing how the scored agent samples is a
   behaviour change, not a measurement change, and this session's remit was to measure the
   budget. The recommendation is for the operator: run this arm across the other five
   games before touching the default.


## The bigger finding, which the budget question was hiding

Every generated engine was RUN against ft09's 25 real captured transitions, because the
return-on-all-paths check is gameable and something gamed it. The refactor lane produced
completions that `generate()` accepts, that parse, that return on every path — and that are
the identity function:

```python
def engine(grid, action, data):
    x = data.get('x', 0); y = data.get('y', 0)
    rows = len(grid); cols = len(grid[0]) if rows > 0 else 0
    if action == 6:
        return grid
    return grid
```

**Of 27 engines generated under the SHIPPED sampler, 0 ever produce an
output different from their input.** Not one. That is every induce call shape, the refactor
call, and every budget. (The sampler-control arms below are the exception, and the only one.)

| lane | budget | accepted | returns on all paths | **changes the grid** | mean heldout-exact | mean cell recall |
|---|---|---|---|---|---|---|
| combined induce | 4096 | 0/3 | 0/3 | **0/3** | 0.173 | 0.000 |
| combined induce | 8192 | 0/3 | 0/3 | **0/3** | 0.173 | 0.000 |
| combined induce | 16384 | 0/3 | 0/3 | **0/3** | 0.173 | 0.000 |
| engine-only induce | 4096 | 3/3 | 0/3 | **0/3** | 0.000 | 0.000 |
| engine-only induce | 8192 | 3/3 | 0/3 | **0/3** | 0.000 | 0.000 |
| engine-only induce | 16384 | 3/3 | 0/3 | **0/3** | 0.000 | 0.000 |
| refactor | 4096 | 2/3 | 2/3 | **0/3** | 0.760 | 0.000 |
| refactor | 8192 | 1/3 | 1/3 | **0/3** | 0.253 | 0.000 |
| refactor | 16384 | 1/3 | 1/3 | **0/3** | 0.253 | 0.000 |

The refactor engines score **19 of 25 heldout-exact** — which looks like the best result in
the sweep until you notice that 19 of ft09's 25 transitions are no-ops, and an engine that
predicts "nothing ever changes" gets every one of them right. Their cell recall is 0.000.
That is the vacuous pass the change-aware gate exists to catch, reproduced here from the
generator side, and it is a live argument for the gate that is computed today but left
`gate_enabled: false`.

It also reframes the Phase-1 question. "Does more budget produce a complete `engine()`" was
the wrong question to be able to answer yes to: the engines are not incomplete, they are
inert. No completion budget fixes that.


## The seed does not reach across server instances

`CARNOT_ARC_GENERATOR_SEED` was added on 2026-07-29 because `llama-server` reads an absent seed as
-1 and two identical runs then diverge. It works — but only inside one server process. The sampler
control's `off` arm is the SHIPPED configuration re-run on a second server, same prompt, same seed,
same temperature, same budget, same model file, same `n_ctx`/offload/MTP flags, same GPU model.
Its completions are **not** byte-identical to the budget lane's:

| attempt | budget lane (port 8933, GPU 1) | sampler `off` arm (port 8934, GPU 0) | identical |
|---|---|---|---|
| a=0 | `0cbe655c87403150…` | `e409387090261e48…` | no |
| a=1 | `8dbdd3ca322562eb…` | `d32557b84ea87906…` | no |

Within one process the seed holds exactly — `combined` attempt 1 is byte-identical across a 4x
budget range on the same server. Across processes it does not. The plausible mechanism is
`cache_prompt: true`: the two servers had different request histories, so the prompt KV is reused
differently, the batch shapes differ, and float accumulation order changes under a temperature of
0.2. **That mechanism is not confirmed here** — the observation is, the explanation is a
hypothesis.

It matters beyond this note. It is a live explanation for how ft09's round-2 refactor hit the
output cap 3/3 in the episode while the byte-identical prompt at the same budget completes
naturally here, and it means "seeded, therefore reproducible" is only true for a matched server
history. Any future A/B on this path that compares arms served by different server processes is
carrying this as an uncontrolled term.

## A provenance correction to the ft09 diagnosis

The banked ft09 engine (`world_model.py`, 1144 lines, 11 lines of code, a 1061-line contiguous run
of bare `#`) was attributed to round 1's induce, whose recorded `message` is
`"local gguf (GPU server) wrote world_model.py"` — the *combined* path's message, since the split
path appends `(split induce: engine + focused goal)`.

It was not written by that call. The file starts with exactly `_combine_world_model()`'s injected
`"import numpy as np\n\n"` prefix, and tokenizing its two halves separately gives **4093 tokens for
the engine part** against a 4096 budget, plus 81 for the goal part. That is the *split* fallback,
with its engine-only call stopped dead at the cap. The call that wrote it is
`arc_competition_agent.py:5889`, which re-runs `induce()` after the refinement loop returns
unplanned and discards the message (`ok, _ = ...`), so its outcome never reaches
`refinement_rounds`. The round-1 record and the file on disk are two different calls, and reading
the file's properties as round 1's is a mis-attribution — the same class of mistake as tn36's
wrong rejection label, in the same subsystem.

## Scope — what this does and does not license

- It measures **one game's** induce and refactor calls in isolation, at n=3 per budget. It is not
  a live episode, so **it does not move the funnel count** (the LLM tier fires 6/6 and reaches the
  policy 1/6). Only a re-run live cell can move that, and this note does not claim one.
- The **inert-engine** result is the strong one and does not depend on n: under the shipped
  sampler it is 0 of every engine generated, on every call shape, at every budget. The **budget**
  result is the weaker one — n=3
  per budget is the live path's own retry count, so "how many of 3 tries yield a usable engine" is
  the live-relevant statistic, but it is thin evidence for a *difference* between budgets. The
  mechanism evidence (what the extra tokens were spent on) carries more weight here
  than the pass counts.
- The **sampler** result is a lead, not a licence. It is n=3, one game, one prompt, one call
  shape, with an in-sample cell recall and no held-out changing transition to test. It is enough
  to justify the next measurement and not enough to justify a default change, and nothing here
  changes a default.
- Nothing here solved a level or played any game, scored or offline. Banked ARC levels remain 3/3
  and the submission gate is NOT met.

## What to do next

1. **Run the sampler arm across the other five games** (`vc33`, `tn36`, `tu93`, `sc25`, `lp85`),
   same shape: shipped `off` vs `repeat_penalty 1.1`, budget fixed at 4096, one server per game,
   three seeded attempts. That is the measurement that would justify touching the scored path's
   sampler, and it is cheap — the winning arm finishes in a third of the wall time of the losing
   one, because it stops instead of filling the budget.
2. **Then, and only then, a live cell.** The funnel number (tier reaches the policy 1 of 6) can
   only move in a real episode. A sampler that produces a non-degenerate engine offline still has
   to survive the dynamics gate, the goal gate and the planner.
3. **Do not raise the budget.** It is retired in `ops/exclusion_manifest.yaml` as
   `arc-induce-completion-budget-raise`, with the reopen condition stated there: a budget raise
   returns only alongside something that stops the repetition.
