# ARC Prize 2026 / ARC-AGI-3 Kaggle Submission — Requirements (2026-06-17)

Investigation (read-only; no submission made) of what a public-leaderboard entry
requires, triggered by the operator after our first live API scorecard (13 levels,
account-level record — NOT a public leaderboard entry). Sources: Kaggle competition
`arc-prize-2026-arc-agi-3` (files via kaggle CLI), arcprize.org/competitions/2026/arc-agi-3,
the ARC-AGI-3-Agents framework README + template agent.

## The format (confirmed)

- **You submit an AGENT, not a scorecard or a file of answers.** The competition data
  IS the `ARC-AGI-3-Agents` framework. An agent subclasses `Agent` (from arcengine) and
  implements:
  - `choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction`
  - `is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool`
  Actions are `GameAction` enum values: keyboard (set `reasoning`) or click via
  `set_data(x, y)` with 0–63 coords. (`random_agent.py` is the template.)
- **Evaluated OFFLINE — "No internet access during evaluation."** The agent runs without
  live ARC API connectivity. Our OFFLINE harness (`environment_files/` deterministic sims)
  is the correct shape; our Mode-1 LIVE replays (which hit the API) are NOT the competition
  path.
- **Open-source required for milestone prizes** — "All code and methods must be open
  sourced to be eligible." We are Apache-2.0 → eligible.
- **Milestones / prizes:** #1 **2026-06-30** ($25K/$10K/$2.5K, open-sourced); #2 2026-09-30
  (same); Top Score Awards $40K/$15K/$10K/$5K/$5K (1st–5th); Grand Prize **$700K** (100%).

## RESOLVED (2026-06-17): eval is a HIDDEN/private set — the agent never sees the eval games

Confirmed across sources (ARC-AGI-3 Technical Report public/private split; Preview
30-day-learnings; "agents must learn, not memorize"): **competition scoring runs the
agent on a SEPARATE PRIVATE set of games it has NEVER seen**, "intentionally
out-of-distribution relative to the mechanics found in the public environments"
(public/semi-private/fully-private; the fully-private set is the official leaderboard).
The preview used 3 public + 3 private as a holdout; the main 2026 competition uses a
much larger private holdout (reports cite ~110 games split public/private leaderboard).
**Consequence: our 11 banked replays score ~0 on the leaderboard (those public games are
not in the eval set). The recognize-and-replay v1 agent validated the harness
integration but is WORTHLESS for scoring. The GENERIC step-wise solver (#2) is the
ENTIRE competitive value.** Scoring also rewards EFFICIENCY. Final eval cap (preview): 8h
wall-clock, 10 steps/sec.

**CORRECTION 2026-07-26 — the efficiency formula stated here was a PROJECT PARAPHRASE and it was
wrong.** This paragraph originally read "per-level score = `min(human_actions / agent_actions, 1.0)`,
squared". It was already caught and retracted inside the eval harness on 2026-06-20
(`scripts/arc_leaderboard_eval.py:340-348`) but this document was never corrected, so the wrong
formula stayed quotable for five weeks and was re-derived from here more than once. Read off the
INSTALLED `arc_agi.scorecard` — the package the competition gateway itself runs — the real
definitions are:

- **Per level (the STORED per-level score):**
  `min((baseline_actions / actions_taken)**2 * 100, 115.0)` (`scorecard.py:166-173`). There is no
  clamp at 1.0 before squaring, so a superhuman solve does store a value above 100 — but see the
  next bullet: that headroom is clamped away at game level, so it cannot reward superhuman play in
  the score anyone is graded on.
- **A level that was not completed scores `0.0`**, regardless of how many actions were charged to it
  (`scorecard.py:178-183`).
- **Actions are charged PER LEVEL by differencing successive level-up checkpoints**,
  `level_actions = actions_at_level - prev_actions` (`scorecard.py:474-491`). Everything after the
  LAST level-up therefore lands in the first not-completed level's bucket, which scores 0 — **the
  post-solve tail costs exactly zero score.**
- **Per game:** an INDEX-WEIGHTED mean over the game's full level list, **then clamped to the
  all-scoring-levels ceiling** (`scorecard.py:189-206`):
  `score = min(sum(level_score_i * i) / sum(i), max_weights / total_weights * 100)`, where `i` is the
  1-based level index used as the weight (`level_index=level_idx + 1`, `scorecard.py:486`) and
  `max_weights` sums the weights of only the levels that scored above zero. Unsolved levels
  contribute 0 to the numerator but their weight stays in the denominator, so **DEPTH is the
  dominant lever** — 1/2/4/8 of 8 levels solved at exactly human speed scores
  2.78 / 8.33 / 27.78 / 100 (all four reproduced against the installed scorer, 2026-07-26).
- **A GAME score can never exceed 100, so the 115 per-level cap is unreachable in the graded
  number.** Because `max_weights <= total_weights` always, the clamp above is `<= 100`. Worked
  example, run against the installed `EnvironmentScoreCalculator`: a 20-action human baseline solved
  in 15 actions stores a per-level score of **115.0** and yields a game score of exactly **100.0**.
  A 20,000-configuration random search over level counts, completion patterns, baselines and action
  counts found a maximum game score of exactly 100.0. **What the clamp means in practice: a game can
  never score above what human-speed play on the levels it solved would have scored.** Verified
  against the installed scorer: 1 of 8 levels solved scores 2.7778 at human speed and 2.7778 at 10x
  human speed — identical. Superhuman efficiency is therefore worth exactly zero as a bonus; its only
  value is COMPENSATORY, offsetting a level solved SLOWER than human (2 of 8 with L1 at 1.4x human
  actions scores 6.9728 with human-speed L2, 7.8061 with 10x-superhuman L2, against the 8.3333
  human-speed-everywhere ceiling). **Depth, then not-being-slow. Never speed for its own sake.**
  (Added 2026-07-26: an earlier draft of THIS correction
  block claimed "115, not 100" for that example — the same misreading of the same file, one bullet
  away from the paraphrase it was retracting. The clamp had been omitted.)
- **RESET is charged as an action by the gateway** (`scorecard.py:701-704`) but as zero by our offline
  harness (`arc_leaderboard_eval.py:308-313`), so offline efficiency is optimistic by the number of
  resets inside each level.

Two consequences that were being read backwards from the old paraphrase: raising the action budget
cannot cost score through a post-solve tail (only by delaying a level-up, which a raised cap cannot
do), and cutting that tail cannot gain score either — it is a wall-clock and memory lever. Measured
2026-07-26, `results/outer_loop_arc_early_stop_grace_sweep_20260726.json`.

At the SHIPPED budget (`MAX_ACTIONS = 400`) that wall-clock lever is close to worthless, and the
measurement says so precisely rather than by assertion. A grace window is safe in-sample only above
the largest inter-level-up gap (340.2 frames) and fires at all only below the largest post-solve tail
(372.3 frames), so the whole usable range at b400 is roughly (340, 372). The one value measured inside
it, grace 350, fires on **1 of 75 cells**, costs no level, moves no score, and saves **0.072% of
corpus actions**. Nothing about the mechanism is broken; there is simply almost no post-solve tail to
cut inside a 400-action cap. Note also that no fixed grace value tested was safe at more than one
budget: 3 of the 4 values tested at two or more budgets passed at one and regressed levels at another,
so a fixed window does not generalise — an adaptive one (scaled to the run's own observed gaps) is the
only form that could, and it is unmeasured. `SUBMITTED_EARLY_STOP_GRACE` remains `None`; the decision
is the operator's.

### (historical) The open question this resolved:
- **If eval == the public 25 games (same layouts):** an agent that runs our solver +
  replays our 13 banked solutions when it recognizes a game would score our 13 levels
  directly (the offline sims are deterministic).
- **If eval == a HIDDEN/held-out set (likely, to prevent memorization):** banked
  trajectories are worthless; only our GENERIC OFFLINE SOLVER's generalization counts.
  That solver is exactly `graph_explore_solve_v2` (the salience+graph-explore family that
  is the published 3rd-place approach, arXiv:2512.24156) + the E3 executable-world-model
  inducer — i.e. the work we already built. **Resolve before investing:** read the
  competition rules tab / the eval description in the downloaded ARC-AGI-3-Agents data,
  or a small probe submission.

## How "no internet" works with bundled model weights (TRM / GGUF)

"No internet at eval" blocks LIVE NETWORK CALLS (hosted APIs: GPT/Claude/Gemini/codex) —
it does NOT block reading local files. Trained weights ride along WITH the submission:
1. Upload the weights ONCE (with internet, ahead of time) as a **Kaggle Dataset / Kaggle
   Models** artifact.
2. Attach that artifact to the submission notebook/agent.
3. At eval the organizers mount it read-only into the offline sandbox at
   `/kaggle/input/<dataset>/...`; the agent loads it from disk
   (`torch.load('/kaggle/input/.../trm.ckpt')` or `llama_cpp.Llama(model_path=...)`) —
   a pure disk read, no network. The game environments are likewise local in the sandbox.
So a LOCAL model (trained TRM weights OR an open GGUF) is the correct engine precisely
because it is a bundled FILE, not a hosted service. A hosted API is the ONE thing the
offline rule forbids. The eval is a Kaggle notebook the organizers run (swap data, Save,
within ~12h on the provided GPU); a 5M-param TRM runs trivially, a Q4 12B GGUF fits.

**LICENSING CATCH (action item):** prize-eligible solutions must be released **CC0 or
MIT-0** — MORE permissive than our Apache-2.0. For prize eligibility the submission +
bundled weights need CC0/MIT-0, not just Apache-2.0. (Internal use stays Apache-2.0.)

**LICENSE SCOPE — VERIFIED (2026-06-17, ARC Prize 2026 rules):** "All code and methods
AUTHORED BY THE SUBMITTER must be made open source under a permissive public domain
license (e.g., CC0 or MIT-0). Any 3rd party code NOT authored by the submitter must be
available under at least an open source license which allows public sharing (e.g.,
Apache-2.0, GPLv3)." So the scope is *submitter-authored code that is part of the
solution* — NOT the whole dependency tree. Third-party deps (arcengine, numpy,
llama.cpp, ARC-AGI-3-Agents) stay Apache-2.0/etc. BUT Carnot is OUR code, so every
Carnot module the submission imports is "submitter-authored code in the solution" and
MUST be MIT-0/CC0. Practically the submission path is `python/carnot/agentic/*` (agent,
explorer, solver-kit, world-model, adapters) — the verify/ ensemble + models/ + pipeline/
are NOT imported by the agent, so the crown-jewel verifier core can stay Apache-2.0 under
a CARVE-OUT. Decision (operator, sole author): (A) MIT-0 carve-out of just the submission
modules [keeps patent grant + attribution on the bulk; needs a dependency-boundary audit
so no Apache-2.0 submitter module leaks into the solution], or (B) relicense the WHOLE
project to MIT-0 [foolproof compliance + simplest, but drops Apache-2.0's patent
retaliation clause + attribution everywhere]. MIT-0 preferred over CC0 (real license w/
warranty disclaimer).

**RESOLVED (operator 2026-06-19): took option (B) — the WHOLE project is now MIT-0.** Verified:
`LICENSE` = "MIT No Attribution"; `pyproject.toml` `license = {text = "MIT-0"}`,
`license-files = ["LICENSE", "NOTICE"]` (the NOTICE file carries any vendored/3rd-party carve-outs).
So the prize-eligibility LICENSING blocker is CLEARED — all submitter-authored Carnot code in the
solution is MIT-0, and the 3rd-party deps (llama.cpp = MIT, arcengine/ARC-AGI-3-Agents, the
Qwen3.5-9B GGUF = Apache-2.0) meet the "at least open-source" bar. A first submission can be
**prize-eligible**. The remaining gates are purely technical: the runtime binary (in flight), dataset
packaging, and the offline dry-run — NOT licensing.

**(superseded) Earlier "TO VERIFY":** some ARC tracks (seen in the ARC-AGI-2 description) let
submissions call OUT to third-party compute (Modal/Lambda/RunPod) under a ~$10k runtime
cap — which would re-open external/larger compute. UNCONFIRMED for ARC-AGI-3 (its stated
rule is "no internet, rules out hosted APIs"). Safe default = fully-offline bundled
weights; confirm before relying on a compute-call-out path.

## How our assets map (low integration cost)

Our solver already operates on exactly the `frame -> GameAction` interface:
- `python/carnot/agentic/arc_graph_explore.py` (`graph_explore_solve_v2`, salience +
  HUD-mask) — generic systematic explorer; the competition-shaped, generalizing asset.
- `python/carnot/agentic/arc_solver_kit.py` (verifier-routed best-first) + the learned
  verifier — efficiency (action-count) which the competition rewards (RHAE-style).
- `python/carnot/agentic/arc_executable_world_model.py` (E3) — the deep-tail solver.
- `ops/arc_solve_registry.yaml` — our 13 banked solutions (useful only if eval==public-25).

A `CarnotAgent(Agent)` wrapper whose `choose_action` runs (recognize→replay-if-known
else verifier-routed-explore) is a small, mechanical build on top of these.

## Honest status vs the live scorecard

- The live API scorecard (13 levels, `0f6273ce…`) is an **account-level record**, not a
  public leaderboard entry. It validated env-match (our offline solutions replay live),
  which is a real result, but it does not appear on the Kaggle leaderboard.
- A public entry = build `CarnotAgent`, validate offline via the ARC-AGI-3-Agents harness,
  then an **operator-gated Kaggle submission** (External Publication, operator-only).

## Recommended next step

1. Build `CarnotAgent(Agent)` wrapping `graph_explore_solve_v2` (+ banked-replay fast path).
2. Validate it offline through the actual ARC-AGI-3-Agents harness on the 25 games.
3. Resolve the public-vs-hidden eval question from the competition rules.
4. Operator-gated Kaggle submission before the 2026-06-30 milestone (open-source eligible).

The durable competitive asset is the GENERIC solver (graph-explore + E3), not the 13
banked replays — the competition rewards solving novel games offline, which is precisely
the north-star capability we are already building.

## SELECTED GENERATOR + PACKAGING MANIFEST (2026-06-19)

**Generator: `unsloth/Qwen3.5-9B-MTP-GGUF` (Q4_K_M, 5.9 GB).** Chosen by benchmarking the 16 GB-fit
shortlist on the Layer-B grounding harness ([[project_arc_live_generator]], full tables in
`arc-16gb-model-alternatives-2026-06-18.md`): 62.5% grounding (4-seed) vs DeepSeek-V4-Flash-MTP 25% and a
verbose gemma-4-12B; ~13 tok/s with MTP (fastest tested); Apache-2.0; controllable thinking. Wired into the
live agent at `python/carnot/agentic/arc_competition_agent.py:_proposer()` (E2E-verified: grounds ka59).

**Deploy config** (`LocalGGUFProposer`, validated): `mtp=True` (`--spec-type draft-mtp`, 1.4–2× speedup),
`kv_quant="q8_0"` (`--cache-type-k/v q8_0`, near-lossless, doubles context), `no_think_prefix="/no_think\n"`,
`max_tokens=2560` (n_predict ≥ 2048 — the 1100 cap truncated real solves). Env overrides:
`CARNOT_ARC_GGUF_PATH` (the bundled model path), `CARNOT_ARC_MTP=0` (disable MTP on a tight-VRAM box).

**What to bundle as Kaggle Datasets (upload once, with internet):**
1. **The GGUF** — `Qwen3.5-9B-Q4_K_M.gguf` from `unsloth/Qwen3.5-9B-MTP-GGUF` (5.9 GB). At submission set
   `CARNOT_ARC_GGUF_PATH=/kaggle/input/<dataset>/Qwen3.5-9B-Q4_K_M.gguf`.
2. **A CUDA llama.cpp BINARY (llama-server) — NOT a Python wheel.** This is load-bearing: native MTP
   (`--spec-type draft-mtp`) is implemented in **`libllama-common`** (verified: 30 symbols there, 0 in core
   `libllama`), which the **`llama-server` binary** links and which our `LocalGGUFProposer` already launches.
   The stock **`llama-cpp-python` wheel wraps only core `libllama`** and uses its OWN Python-level
   speculative decoding — it **cannot do the native self-MTP we validated.** So bundle the **CUDA-compiled
   `llama-server` + its shared libs** (`libllama`, `libggml`, `libggml-cuda`, **`libllama-common`**) as a
   Kaggle Dataset, and point the submission at it via **`CARNOT_LLAMA_SERVER=/kaggle/input/<dataset>/llama-server`**
   (env, resolved at import). The binary is **model-agnostic** — it loads Qwen3.5-9B-MTP at runtime via
   `-m` + the MTP/KV flags; nothing is baked in.
   - **Build recipe (ahead of time, internet on):** `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;75;89;89-virtual"`
     (60 = P100, 75 = T4 / Quadro-RTX-6000-Turing, 89 = L4 / RTX-6000-Ada; `89-virtual` = compute_89 PTX as a
     JIT fallback for any newer/unlisted GPU — without PTX a SASS-only binary hard-fails on an arch not in the
     list). V100=70 omitted (Kaggle does not offer it). On a recent llama.cpp (≥ the build that merged
     `draft-mtp`; release b9714 / repo 9606 has it), `--target llama-server`. **Build it inside a Kaggle
     notebook (internet on)** to guarantee the CUDA toolkit + GPU-arch match, then save the build dir as a
     Kaggle Dataset output. (A local CUDA build works only if its arch list covers the Kaggle GPU.) The ARC-AGI-3
     pool is P100 / T4×2 standard + RTX 6000 (internet-disabled); 12h notebook runtime cap; offline eval.
   - **Wheel fallback (no MTP):** if bundling the binary proves fiddly, use the `llama-cpp-python` CUDA wheel
     and run **MTP-OFF** (`CARNOT_ARC_MTP=0`) — base Qwen3.5-9B still grounds, just ~6.6–8.2 tok/s vs ~13.
3. Carnot's own code (the agent/solver/world-model) — already in-repo, MIT-0.

**Two gating verifications before the submission is trusted (run a one-shot Kaggle notebook, internet OFF):**
- **Real per-GPU VRAM** — `nvidia-smi`: T4 16 GB vs **L4 24 GB**. At 24 GB the 16 GB constraint relaxes (a
  27B Q4 would fit) — but Qwen3.5-9B-MTP is still the validated pick on accuracy/speed/efficiency.
- **The CUDA engine loads + generates offline** — the one load-bearing assumption. Smoke: load the bundled
  GGUF with MTP + q8 KV via the bundled binary/wheel and generate 1 token.

Decentralization: mirror the bundled weights per Rule 3 ([[feedback_ipfs_over_gitea_for_mirror_channel]]).

### Local CUDA validation (2026-06-19) — most of the engine question is settled

Ran the existing CUDA `llama-server` (build/, CUDA 13) on a 3090 with Qwen3.5-9B-MTP + `--spec-type
draft-mtp` + q8_0 KV (the exact submission flags). Result:
- **CUDA + MTP + q8 KV works**: `draft-mtp` activated, **80.5% draft acceptance**, generated correct code,
  **159 tok/s** (3090; a T4 is slower but still far above the iGPU's ~13).
- **VRAM footprint = 11.5 GB** (model 5.9 + MTP self-draft ~5.2 + q8 KV @ ctx 4096) → **fits a 16 GB T4
  with ~4.9 GB free for additional KV.** So even at the pessimistic 16 GB / T4 assumption, Qwen3.5-9B-MTP
  + MTP + q8 KV deploys with headroom.

**What this does NOT settle (still needs the Kaggle notebook):** the binary must be built on **Kaggle's
CUDA (≈12.x)** — the local build links CUDA 13, so it would NOT run on Kaggle as-is (this confirms the
build-ON-Kaggle approach is mandatory, not optional). And the real per-GPU VRAM (T4 vs L4) is still a
Kaggle probe. Net: the engine + model + MTP + KV + fit are validated; only the CUDA-version-matched build
and the live VRAM probe remain — exactly what `scripts/kaggle/build_verify_llamacpp_mtp.py` does.
