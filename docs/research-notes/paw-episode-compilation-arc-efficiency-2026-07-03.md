# PAW-inspired per-episode compilation for ARC-AGI-3 action-efficiency (2026-07-03)

**Provenance:** outer-loop literature discussion, triggered by the operator reading arXiv:2607.02512
("Program-as-Weights: A Programming Paradigm for Fuzzy Functions," Zhang/Hotsko/Kim/Nie/Shieber/Deng,
Waterloo/Cornell/Harvard) and connecting it to the ARC-AGI-3 live-agent efficiency axis. Read the
primary paper before implementation (literature-priority discipline) — this note summarizes the
connection and scopes a falsifiable first pilot, it is not a substitute for the paper.

## The paper's mechanism, precisely

PAW compiles a natural-language "fuzzy function" spec (a task too fuzzy for hand-written rules, e.g.
malformed-JSON repair, intent-based ranking) **once**, using an expensive model, into a small hybrid
artifact: a discrete "pseudo-program" (a clean paraphrase + examples) plus a trained LoRA adapter. A
tiny **frozen, local interpreter model** then runs that compiled artifact cheaply forever after. Real
numbers: a 0.6B Qwen3 interpreter running compiled PAW programs *beats* direct prompting of Qwen3-32B
(73.78% vs 68.70% exact match) at ~1/50th the inference memory, 30 tok/s on a MacBook M3 CPU. The
compiler itself is pretrained on FuzzyBench, a 10M-example dataset covering 29 task categories.

## Why this connects to Carnot — two distinct angles, only one of which is live

**Angle A — GAP-4891's demo-underdetermination gap.** GAP-4's rule-execution verifier already does
PAW's core move (compile once from demo pairs to `def transform(grid)`, run cheap thereafter), but
compiles to *pure symbolic code*. GAP-4891 found this brittle exactly where the paper predicts it
would be: demos that don't uniquely pin down the rule (demo-underdetermination) either get guessed
wrong or force an abstention. PAW's hybrid artifact (discrete structure + continuous LoRA fit to the
same demo pairs) is a genuinely different answer to that specific failure mode — the continuous half
could encode a fuzzier, more generalizing version of the rule a rigid function can't represent. This
is a real, gated-behind-infrastructure direction (see "What this needs before it's buildable" below),
not scoped further in this note.

**Angle B — ARC-AGI-3 live-episode action-efficiency (this note's actual scope).** RHAE squares the
efficiency term, so cost-per-action is scored, not just accuracy. The live generator (Qwen3.5-9B-MTP)
is already resource-budgeted for the Kaggle ~16GB constraint — PAW's *memory* win (replacing a bigger
model with a smaller one) doesn't apply here, that headroom is already spent. What DOES transfer:
after the live agent has spent some actions inducing a hidden game's dynamics (frames/transitions it
legitimately collected via its own exploration — no source-reading, stays inside the
"self-discovery-only" hidden-eval discipline), could a per-game "compile" step turn what it's learned
into a small artifact that handles the *rest* of that episode's action-selection more cheaply than
re-invoking the full 9B generator every step?

## Three real open questions (not yet answered, none resolved by reading the paper harder)

1. **No FuzzyBench-for-ARC exists.** PAW's compiler is itself pretrained on a 10M-example offline
   corpus; only the *per-function* compile is cheap at deployment. The only analogous pretraining
   corpus available is the 25 public games, and every trained-on-public/tested-on-hidden component
   this project has built has shown real-but-bounded transfer degradation (GAP-4's induction rate
   0.93→0.57 on ARC-2 transfer is the load-bearing precedent). A PAW-style compiler would plausibly
   face the same drop. This risk is not specific to this proposal — it is the standing tax on any
   trained ARC component — so it does not block a first pilot, but it caps how much to trust a
   positive in-pilot result before a genuinely held-out check.

2. **Infrastructure gap.** The scored generator runs via llama.cpp against a GGUF checkpoint —
   inference-only, no gradient computation. Fine-tuning a LoRA mid-episode needs a PyTorch/
   HF-transformers path against the *base* (non-GGUF) weights, not currently wired into
   `python/carnot/agentic/arc_competition_agent.py`. This is a real build, not a config flag.

3. **Does the compile cost amortize?** Episodes here run tens to low-hundreds of actions. If the
   compile step burns a meaningful chunk of that budget, it needs enough *remaining* episode length
   to pay itself back in cheaper subsequent steps. This is the cheapest of the three questions to
   answer and the most decisive — it can falsify the whole proposal before touching any new
   infrastructure (open questions 1 and 2 are moot if the answer here is "no").

## Proposed first pilot — pure analysis, zero new infrastructure, falsifiable

Answer question 3 alone before anything else. Two measurements, both against data/hardware Carnot
already has:

1. **Remaining-episode-length distribution.** From existing logged ARC episodes (public-game runs
   already on disk), measure: for a plausible "induction is roughly done" checkpoint (e.g., after the
   agent's first successful level-up, or after N actions with no new transition discovered), how many
   actions typically remain in that episode/game? This is a pure log-analysis pass, no model calls.
2. **Realistic compile wall-clock.** Benchmark how long a lightweight LoRA fine-tune actually takes on
   the target hardware for a model this size (does not need to be ARC-specific data — a generic
   small-LoRA timing benchmark on the same GPU/model class answers this).

**Falsifiable gate:** if the median remaining-episode-length after a plausible induction checkpoint is
too short to amortize even an optimistic compile-then-cheap-inference cost model (compile time +
N_remaining × cheap-inference-time vs. N_remaining × current-inference-time), the proposal is
falsified at near-zero cost and should NOT proceed to building the LoRA training infrastructure in
open question 2. If the gate passes, the next step is a small pilot on public games only (never
touching hidden-game source), comparing action-efficiency with vs. without a mid-episode compile step,
before any consideration of the live/scored submission stack.

## What this note is NOT proposing

- Not a change to the current live submission stack. The frozen Qwen3.5-9B-MTP + MTP + q8 KV setup
  stays as-is per the existing "ARC live generator" decision unless a pilot earns a change.
- Not a claim that this solves the L1-first-contact / trajectory-enumeration generation wall. This is
  strictly an efficiency proposal for episodes the agent is *already* progressing through, not a new
  candidate-generation mechanism (that axis is separately, formally retired per the 3x-nulled
  generation-axis-exploration-signal exclusion-manifest entry — this proposal does not re-open it).
- Not source-reading. Any "compile" step must operate purely on the live agent's own runtime
  observations, matching "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent" and the
  "Source-reading public-dev-only" discipline.

## Cross-references

- arXiv:2607.02512 — the paper
- `ops/verifier_gaps.md` GAP-4, GAP-4891 — the demo-underdetermination precedent (Angle A)
- `ops/exclusion_manifest.yaml:generation_axis_exploration_signal_retired_exp5154_v473` — the
  retired scope this proposal does NOT re-open
- `feedback_arc3_online_gated_on_offline_beating_baselines.md` (memory) — the discipline this pilot's
  gate structure follows (offline-first, gated progression)
- `project_arc_live_generator.md` (memory) — the current frozen generator decision this note does not
  propose changing without pilot evidence
