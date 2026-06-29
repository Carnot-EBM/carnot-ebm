# ARC-AGI-3 SOTA reference clones (local, for embrace-then-extend-with-energy)

Date: 2026-06-23 · Outer-loop (interactive) · operator directive: "clone the top leaderboard projects
locally so you can crawl them as files rather than web-scraping ... embrace SOTA and then extend them with
energy models." Even the SOTA leaders score ~1% of human on the full ARC-AGI-3 hidden eval (the bar is low
in absolute terms — see StochasticGoose below) — they are still the right baselines to study and extend.

## Where they live (DELIBERATELY OUTSIDE the carnot repo)

Clones are at **`/home/ianblenke/arc-sota-refs/`** — OUTSIDE `/home/ianblenke/github.com/ianblenke/carnot/`.
This is intentional: the conductor runs `git add -A`, and a cloned repo (with its own `.git`) committed
into our tree as a mode-160000 gitlink without `.gitmodules` previously broke ALL GitHub Actions checkouts
(see `feedback_no_embedded_repo_gitlinks` / `.gitignore` line ~179). Cloning outside the tree eliminates
that risk entirely. Read them by absolute path; do NOT copy code into the repo (licensing + bloat). Per the
audit-untrusted-code discipline these are STUDY clones (read, do not execute un-sandboxed).

Refresh: `git -C /home/ianblenke/arc-sota-refs/<name> pull` (or re-clone `--depth 1`).

## Cloned (confirmed open-source, ARC-AGI-3-relevant)

| local dir | upstream | what it is | why it matters to Carnot |
|---|---|---|---|
| `arc-agi-3-just-explore/` | github.com/dolphin-in-a-coma/arc-agi-3-just-explore (arXiv:2512.24156, "Graph-Based Exploration", Family-A, 3rd place / open-source) | A fork of the official harness adding `graph_explorer.py` (a `GraphExplorer`: directed state-graph, tracks visited frames, prioritizes UNTESTED action-groups, distance-to-frontier, advances priority groups) + `agents/heuristic_agent.py` | This is the EXTERNAL counterpart of our own `graph_explore_solve_v2` (`solve_via_explore`, the live self-discovery path). The cleanest place to graft an ENERGY signal: its exploration PRIORITY is a hand heuristic (distance-to-frontier + group order) — replace/augment it with an energy/progress signal (GAP-ARCH-ENERGY-PROGRESS-SHAPING + the curiosity-bonus from the live-agent-gaps analysis). |
| `ARC-AGI-3-Agents/` | github.com/arcprize/ARC-AGI-3-Agents (official) | The official agent harness everyone builds on: the `Agent` interface, `swarm`/`recorder`/`tracing`, and templates (`random_agent`, `reasoning_agent`, `llm_agents`, `langgraph_thinking`, `smolagents`, `multimodal`). | The contract our `make_carnot_agent` / `E3AgentPolicy` implements. Reference for the harness + the LLM-agent template patterns (langgraph/smolagents orchestration) we could adopt. |
| `ARC-AGI/` | github.com/arcprize/arc-agi (toolkit) | The ARC-AGI toolkit / data + evaluation tooling. | Reference for the env/eval interface. |

## NOT cloned — repo not findable via public web (operator: do you have the Kaggle/repo links?)

- **StochasticGoose** (Tufa Labs / Dries Smit) — **1st place in the ARC-AGI-3 PREVIEW agent competition (12.58%)**,
  but **dropped to 0.25% on the full official benchmark at launch** (≈ frontier-LLM level). Architecture
  (from the writeup, no repo linked): a **CNN that predicts which of the 5 actions will change the frame**
  (legal-action / state-change predictor) + a spatially-aware decoder for coordinate ACTION6 + RL +
  frame-store/dedup; **no LLM** (token cost). This is DIRECTLY `GAP-ARCH-FRAME-CHANGE-PREDICTOR` — the most
  relevant single baseline to obtain. (writeup: medium.com/@dries.epos 1st-place post.)
- **PersistentAEM** (DQN + replay) and **SubQ** (operator: "7B model, 10M-context sub-quadratic KV-cache,
  low loss, top of leaderboard") — could NOT be located via public web. They may be Kaggle community-
  leaderboard entries (code public on Kaggle, behind auth) or live-leaderboard names not yet documented.
  **Open premise to verify (in flight, workflow wf_4a2a3d20): is SubQ an ARC-AGI-3 agent, or a sub-quadratic
  long-context LM on a DIFFERENT benchmark?** That distinction decides whether the 10M-context angle is
  load-bearing for ARC-AGI-3 (an interactive grid-game problem whose binding constraint we diagnosed as
  PERCEPTION, not context length — see `project_arc_live_agent_learning_gaps`).

## Embrace-then-extend-with-energy — how each maps

- **graph-explore (have it)** → our `solve_via_explore` already is this family. EXTEND: energy/progress
  shaping on the exploration frontier (the cheapest real loop from the gaps analysis: reuse the LiveTTT
  per-cell prediction error as a dense curiosity bonus + one-step value-backup over the GameGraph edges).
- **frame-change CNN (StochasticGoose, want it)** → `GAP-ARCH-FRAME-CHANGE-PREDICTOR`; the action-efficiency
  score lever. EXTEND: the CNN's effect-prediction is itself an energy-like signal; pair it with the energy
  verifier for affordance pruning.
- **official harness** → adopt the langgraph/smolagents LLM-orchestration template patterns if we promote
  the local LLM from one-shot world-model writer to the RE-loop driver (the outer-loop-process insight).

## Status / next

- Captured as the reference for the SubQ/energy architecture synthesis (workflow wf_4a2a3d20, in flight)
  and the conductor's SOTA-ingestion loop. Cross-refs: `project_arc_live_agent_learning_gaps`,
  `reference_arc_agi3_sota_and_plan`, `ops/verifier_gaps.md` (GAP-ARCH-FRAME-CHANGE-PREDICTOR,
  GAP-ARCH-ENERGY-PROGRESS-SHAPING).
- OPERATOR ASK: links to StochasticGoose / PersistentAEM / SubQ repos (or their Kaggle community-leaderboard
  pages) so they can be cloned here too.

## 2026-06-29 — "How do the leaders succeed with frontier models while we fail?" (operator question; ANSWERED: premise inverted)

Operator asked how the cloned top-leaderboard projects find success with frontier models while Carnot fails.
A 9-agent comparative workflow (`arc-leaderboard-frontier-vs-carnot-comparison`) read every cloned solver in
full. **The premise is inverted twice over** (adversarial-verified: premise SUPPORTED):

1. **The leaders do NOT use frontier models.** Verified by keyword scan (zero `openai|anthropic|gpt|claude|
   gemini|chat.completion|from_pretrained` hits in the winning solvers):
   - **nihilisticneuralnet "0.46"** — no LLM. Mechanism: `importlib.exec_module` the hidden game's OWN
     `environment_files/<gid>.py` source, clone the simulator, `set_level()` teleport, brute-force a
     5-algorithm classical search (Dijkstra/Beam/IDA*/BFS/A*/MCTS) over the perfect simulator, read the
     ground-truth win signal (`levels_completed` / private `_current_level_index`) directly. A torch CNN
     (ForgeNet) is only a source-absent fallback.
   - **vyankteshdwivedi "0.39 LB"** — no LLM. Same white-box exploit: read the game source, regex-extract
     the win-field name, pickle-clone + forward-simulate, `set_level`. ChangeNet CNN only as a fallback.
   - **StochasticGoose (preview winner 12.58%)** — no LLM. From-scratch online frame-change CNN.
   - **just-explore (3rd, arXiv:2512.24156)** — no model inference at all; pure heuristic state-graph BFS
     with reverse-BFS distance-to-frontier routing; paper explicitly "substantially outperforms frontier
     LLM-based agents" (which scored ~0% on preview).
2. **Where frontier models ARE the core solver, there is NO success.** The only frontier-as-core code in the
   clones is the official baseline TEMPLATES (`llm_agents`/`reasoning_agent`/`langgraph`/`smolagents`/
   `multimodal`, per-step o3/o4-mini choosers) and the OpenClaw scaffold (`arc3_agents`, routes to
   claude/gpt-5/gemini) — NEITHER has any banked score (OpenClaw never completed a scored run: 401 + empty
   recordings). Independent evidence (arXiv:2603.24621): frontier-LLM agents score **<0.4%** on the hidden
   set; StochasticGoose collapsed **12.58% → 0.25%** preview→hidden.

**Score comparability (the crux).** Carnot's **0.08** is RHAE-efficiency-weighted on the **HIDDEN scored
set via blind live discovery** (never reads game source, never `set_level`-teleports, never reads
ground-truth win flags). The leaders' headline numbers are NOT the same basis: "0.46" appears only in a
filename (no metric/set/output in the notebook); "0.39"/"0.30" are PUBLIC-leaderboard source-reading
numbers; "12/25" / "median 30/52" are LEVELS-COMPLETED on the PUBLIC PREVIEW. Three stacked confounds:
public-preview vs hidden SET; raw-levels vs RHAE METRIC; white-box source-reading vs blind-discovery
MECHANISM. **The only valid hidden-set comparators are StochasticGoose 0.25% and frontier <0.4% — against
which Carnot's 0.08 is comparable-to-AHEAD.** Carnot is not failing where others succeed with frontier
models; Carnot and the frontier-LLM approach hit the SAME hidden-win-state wall (our codex/gpt-5.5 probe on
re86/ft09 reproduced it), and the leaders only look better via an easier set + a source-reading mechanism
Carnot is deliberately forbidden from using (it is exactly the `outer_loop_re` anti-pattern in CLAUDE.md's
ARC Live-Path Reachability discipline).

**Legitimately borrowable (live-discovery, NOT the source-reading cheat):** (1) StochasticGoose's
from-scratch online frame-change CNN for the ACTION6 click explosion — but Carnot already has this
(`arc_frame_change_predictor` exp4490/4547, `arc_online_action_effect_scorer` exp4710/4726). (2)
just-explore's persistent navigable state-graph + reverse-BFS distance-to-frontier routing + connected-
component object segmentation + 5-tier visual-salience action priors — the most actionable un-fully-adopted
lever. **Do NOT borrow** the top-two's `exec_module`-source + `set_level`-teleport + read-ground-truth-flag
exploit — it violates the live-hidden-game-discovery deliverable and collapses on truly-hidden games.
