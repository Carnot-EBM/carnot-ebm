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
