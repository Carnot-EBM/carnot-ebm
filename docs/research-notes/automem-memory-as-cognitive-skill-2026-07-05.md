# AutoMem: memory as a trainable skill (arXiv:2607.01224) — 2026-07-05

**Provenance:** operator-requested review of a specific paper. Read the actual PDF (Stanford,
Wu/Zhu/Zhang/Wang/Yeung-Levy) rather than trusting a fetched summary, per this project's own
established discipline that small-model PDF summaries have repeatedly proven unreliable this week.
Checked this project's own record for prior engagement with the paper's concepts before writing —
found none; this is genuinely new ground.

## Core finding

Optimizing memory management alone — never touching the model's task-action weights — produced
2x-4x progression gains for a 32B open-weight model (Qwen2.5-32B-Instruct) across three
procedurally-generated, long-horizon games (Crafter, MiniHack, NetHack): 25.0%→51.4% on Crafter,
7.5%→30.0% on MiniHack, 0.42%→1.85% on NetHack. That is enough to bring the 32B open model to rough
parity with Claude Opus 4.5 and within a few points of Gemini 3.1 Pro Thinking on these tasks. The
headline claim — "memory management is an independently learnable skill" — is earned empirically,
not just asserted: they hold task-action behavior fixed throughout and only ever optimize the memory
path, so the entire measured gain is attributable to memory alone.

## The method, in two loops

1. **Structure optimization (outer-loop 1).** A meta-LLM (Claude Opus) reads *complete* episode
   traces (up to 10^5 steps) and revises the agent's memory scaffold — file schema, prompts, action
   vocabulary — gated on measured improvement over a fixed list of seeds (a revision is kept only if
   average progression actually improves). The paper's stated reason this must be trajectory-level
   review rather than a per-step reward: memory mistakes have *delayed* consequences (a bad decision
   at step 50 might not surface until step 800), so scalar reward signals discard exactly the
   structure that reveals where memory went wrong.

2. **Proficiency training (outer-loop 2).** A *separate*, LoRA-finetuned "memory specialist" is
   trained on the agent's own good memory decisions (the meta-LLM acts as a filter selecting which
   of the model's own past decisions to reinforce, not a teacher generating new ones), while the
   task-action model stays completely frozen. This split is the mechanism, not incidental: it keeps
   the training signal focused on memory behavior specifically (not diluted by action-format
   examples) and structurally cannot regress existing task competence, since gameplay weights are
   never touched by the finetune.

## The most concrete, transferable lesson

Reviewing NetHack traces, the meta-LLM found an unbounded append-only memory file (a dungeon-map log)
silently accumulating duplicate coordinate entries, burying useful information under repeated writes
of the same tile. The fix was a coordinate-keyed upsert operation (`<|UPSERT_MAP|>`) that overwrites
an existing entry rather than appending alongside it — shrinking per-step memory growth by 95% (from
138 to 6 characters per step, Figure 5). After specialist training, the write-to-search ratio across
all three games dropped 54-72% — the model learned to *check memory before modifying it* rather than
writing blindly, and this discipline, originally only encoded in the scaffold's prompting, ended up
internalized in the trained weights.

## Where this does — and does not — connect to this project's existing work

**Checked directly rather than assumed:** this project's own "Continuous Self-Learning" /
typed-memory track (`python/carnot/pipeline/verifier_memory.py`, spec `REQ-LEARN-5214`) is a
*different* concept from what AutoMem addresses. Carnot's verifier memory holds "controller
artifacts, not model-weight updates" — it decides whether a verifier candidate/heuristic gets
promoted into durable use based on held-out evidence gates (`PROMOTED`/`HELD`/`ROLLED_BACK`).
AutoMem's memory is episodic agent memory during long-horizon gameplay — what to record about the
current episode, what to recall to act. These are genuinely separate layers (verifier-decision memory
vs. within-episode agent memory), and this note does not claim they are the same thing or that
AutoMem's specific mechanics port directly onto the verifier-memory track.

**Where the connection is real:** AutoMem's architecture is the more relevant reference point for
this project's *live ARC-AGI-3 agent's* own within-game memory — what it records about a hidden
game's discovered mechanics as it plays, and how that carries across actions or across levels within
one game (the same layer the "ARC live-agent learning gaps" memory and the `arc_live_ttt` online
world-model induction pattern already operate on). Two ideas worth checking against the live agent's
current design, not assumed to transfer wholesale:

- **Separate the memory-update path from the action-decision path.** AutoMem's clean split (a frozen
  task model, a separately-trained memory specialist) is a specific, testable architectural choice —
  does anything in the live agent's current memory handling mix these two concerns in a way that
  would benefit from separating them?
- **Audit for the unbounded-append anti-pattern.** The single concrete bug AutoMem found and fixed
  (an append-only log silently accumulating duplicates) is a generic, cheap thing to check for in any
  memory structure the live agent already maintains across a long game or across levels.

**Honest modality caveat.** AutoMem's memory substrate is natural-language text files
(`dungeon_map.txt`, `strategy.txt`) read/written via LOG/PLAN actions in the model's own action
space — a different representation from ARC-AGI-3's structured/grid frame data and the live agent's
Python-object-based state tracking. The specific file-schema mechanics do not port directly; what
transfers is the architecture (separate trainable memory skill, trajectory-level meta-review,
upsert-over-append), not the implementation.

## What this note is NOT proposing

- Not proposing to adopt AutoMem's file-system-memory mechanism wholesale for the ARC-AGI-3 live
  agent. The architecture is worth checking against; the specific text-file substrate is not a good
  match for ARC's structured state.
- Not claiming any overlap with or improvement to this project's verifier-memory (Continuous
  Self-Learning) track — that is a genuinely different concept, checked directly, not assumed.
- Not a build task. This is a literature note; any concrete action (an audit of the live agent's
  memory handling for the two ideas above) would be a separate, explicitly scoped follow-on.

## Cross-references

- arXiv:2607.01224 — AutoMem: Automated Learning of Memory as a Cognitive Skill (Wu, Zhu, Zhang,
  Wang, Yeung-Levy, Stanford)
- `python/carnot/pipeline/verifier_memory.py` (spec `REQ-LEARN-5214`) — this project's own
  verifier-decision memory, confirmed as a distinct concept from AutoMem's episodic agent memory
- `project_arc_live_agent_learning_gaps.md` (memory) — the live agent's own perceive/hypothesize/
  test/refine loop, the layer AutoMem's ideas would actually apply to if pursued
- `python/carnot/agentic/arc_competition_agent.py` (`E3AgentPolicy`, `arc_live_ttt`) — the live
  agent's current online world-model induction pattern, the closest existing analog to check against
