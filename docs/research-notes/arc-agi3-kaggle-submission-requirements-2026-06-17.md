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
ENTIRE competitive value.** Scoring also rewards EFFICIENCY:
per-level score = `min(human_actions / agent_actions, 1.0)`, squared — so fewer actions
(our verifier-routed search) is directly worth more. Final eval cap (preview): 8h
wall-clock, 10 steps/sec.

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
2. **A CUDA llama.cpp build** — the engine `LocalGGUFProposer` launches is the `llama-server` binary
   (`_resolve_llama_server`), so bundle a **CUDA-compiled llama-server + its shared libs**
   (libllama, libggml-cuda) matching Kaggle's CUDA, OR refactor `LocalGGUFProposer` to the
   `llama-cpp-python` CUDA wheel (`pip install --no-index --find-links=/kaggle/input/...`). Either way the
   build MUST include native MTP (`common_speculative_impl_draft_mtp`) — the repo build 9606 has it.
3. Carnot's own code (the agent/solver/world-model) — already in-repo, MIT-0.

**Two gating verifications before the submission is trusted (run a one-shot Kaggle notebook, internet OFF):**
- **Real per-GPU VRAM** — `nvidia-smi`: T4 16 GB vs **L4 24 GB**. At 24 GB the 16 GB constraint relaxes (a
  27B Q4 would fit) — but Qwen3.5-9B-MTP is still the validated pick on accuracy/speed/efficiency.
- **The CUDA engine loads + generates offline** — the one load-bearing assumption. Smoke: load the bundled
  GGUF with MTP + q8 KV via the bundled binary/wheel and generate 1 token.

Decentralization: mirror the bundled weights per Rule 3 ([[feedback_ipfs_over_gitea_for_mirror_channel]]).
