# HuggingFace Spaces: EBM Sudoku Solver Demo

**Status:** Draft change proposal. Near-term shipping window: 3-5 days
of focused work. Pinned in `ops/known-issues.md` once filed.

**Origin:** 2026-04-29 user observation that the Kona project (Carnot's
Phase 3 inspiration target — see CLAUDE.md "Project Vision") started
with an EBM Sudoku solver as its showcase. Sudoku is uniquely well-
suited as Carnot's first interactive public demo:
- Universally interpretable (everyone knows the rules)
- Pure EBM showcase (no LLM-in-the-loop dependency)
- Visual emergence is striking (energy descent is observable)
- Bridges Phase 1 → Phase 3 narrative (same energy framework)
- Hardware-acceleration story is concrete (Phase 2 path)

**Target milestone:** post-.80 ad-hoc release; not blocked on the
research conductor's milestone planning. Ships independently.

**Priority:** **High.** First interactive public demo of the Carnot
EBM framework. Strategic role: the artifact most likely to make
visitors ask "wait, what *is* Carnot?" — exactly the question we want
partners (HuggingFace, Apple ML, EleutherAI) asking when v0.1 paper
lands.

**Depends on:** existing `python/carnot/verify/sudoku.py` constraint
encoding + a Carnot sampler from `python/carnot/samplers/`. No new
math or core code required.

## Summary

Ship a HuggingFace Spaces app at **`Carnot-EBM/sudoku-energy-solver`**
that:

1. Accepts a partial Sudoku grid (manual entry or preset puzzle).
2. Solves it via Carnot's existing constraint-energy + sampler stack.
3. Visualises the energy-descent process live (cell-by-cell heatmap +
   energy-vs-iteration plot).
4. Reports timing and step counts; offers side-by-side comparison
   against a naive backtracking baseline.

The demo has two operational regimes:

- **Public-tier (CPU only):** runs on Spaces' free tier. Solves
  standard 9×9 Sudoku in seconds.
- **Optional GPU/FPGA-tier (post-Phase-2):** when KV260 Hybrid
  Coprocessor demo lands, link the Spaces page to the FPGA-backed
  endpoint with $\geq 100\times$ throughput.

## What this proposal IS

- A *packaging and UI* deliverable. The math + sampler exist; we
  wrap them in `gradio`.
- A *near-term marketing artifact* that doesn't compete with the
  position paper for shipping resources.
- A *Phase 1 → Phase 3 bridge demo*: same energy framework that
  verifies LLM outputs, applied to a discrete constraint
  satisfaction problem.

## What this proposal IS NOT

- **Not new research.** The Sudoku-as-EBM encoding has been studied
  for years; Carnot's contribution is the deployment, not the math.
- **Not a Phase 1 verify-repair demo.** That's a separate proposal.
  Sudoku is more visually compelling as a *first* demo.
- **Not blocked on the position paper.** Ships before v0.1 paper to
  build awareness ahead of the preprint landing.
- **Not a Sudoku-solving benchmark.** SOTA Sudoku solvers (e.g.,
  Norvig's) outperform EBM-based approaches on raw speed. The point
  is *demonstrating energy-based reasoning*, not winning a speed
  contest.

## Proposed experiments / shipping increments

### Increment 1 — Core Spaces app (1-2 days)

**Deliverable:** `spaces/sudoku-energy-solver/` directory with:
- `app.py` — Gradio UI + Carnot sampler invocation
- `requirements.txt` — `carnot`, `gradio`, `numpy`, `jax[cpu]`,
  `matplotlib`, `pillow`
- `README.md` — explains the energy formulation, links to
  technical-report.html and the new blog post
- `.huggingface.yml` — Spaces metadata (CPU tier, hardware: cpu-basic)

**What it does:**

1. UI presents an empty 9×9 grid + library of preset puzzles
   (Easy / Medium / Hard / Diabolical from public datasets).
2. User clicks "Solve" → backend instantiates
   `build_sudoku_energy(clues)` from `python/carnot/verify/sudoku.py`,
   runs Langevin dynamics or PT-PCD sampler from
   `python/carnot/samplers/`, returns iteration trajectory.
3. UI streams the trajectory: animated grid showing cells flipping +
   energy-vs-iteration plot updating live + cell-energy heatmap
   overlay.
4. On convergence: show solved grid + total iterations + wall-clock
   time + total energy evaluations.
5. Side-by-side comparison option: same puzzle solved by a Norvig-
   style backtracking solver, shown alongside the Carnot trajectory.

**Acceptance:**
- Solves all 4 difficulty tiers on free Spaces CPU tier in <30s
  each.
- Visualisation runs at >5 FPS during animation.
- Comparison-mode shows both solvers in <60s combined.

### Increment 2 — Educational explainer (1 day)

**Deliverable:** `spaces/sudoku-energy-solver/EXPLAINER.md` +
inline tooltips in the Gradio UI.

**What it does:**

1. Plain-English walkthrough of the energy formulation:
   - "27 uniqueness constraints (9 rows + 9 columns + 9 boxes)"
   - "Each clue is a quadratic anchor"
   - "Energy descent = solution discovery"
2. Toggleable advanced view: render the constraint graph as a graph
   visualisation (nodes = cells, edges = shared constraints).
3. Annotated screenshot/GIF for the README.

**Acceptance:** non-technical visitor can read the explainer and
understand why energy descent solves Sudoku, without needing
ML/physics background.

### Increment 3 — Sampler-comparison mode (1 day)

**Deliverable:** advanced UI tab letting power users select among
Carnot's sampler implementations.

**What it does:**

1. Dropdown for sampler choice:
   - Langevin dynamics (continuous-relaxed)
   - Parallel-Ising sampler
   - PT-PCD with replica exchange
   - Norvig backtracking (baseline)
2. Each runs the same puzzle; UI shows side-by-side trajectories.
3. Result table: iterations, wall-clock, energy evaluations, solved
   yes/no.
4. Educational note: "Carnot's KV260 FPGA path (Phase 2) targets
   $10^9$ samples/sec on this exact constraint shape — projected
   $1000\times$ speedup over CPU."

**Acceptance:** all 4 sampler options work; comparison is reproducible
on the same seed; educational note links to the Phase 2 transpiler
documentation.

### Increment 4 — README polish + cross-linking (0.5 days)

**Deliverable:** Spaces README polished + main Carnot README linked +
blog post mentioning the demo.

**What it does:**

1. Spaces README: short pitch, screenshots, links to Carnot main
   site, technical report, Verifier Accuracy Paradox blog post.
2. Main Carnot README adds a "Try it" section with Spaces link
   alongside the existing GitHub badge.
3. Tweet template / Bluesky template / Mastodon template prepared
   for launch announcement.

**Acceptance:** all cross-links resolve; demo is discoverable from
the project's main entry points.

## Why Sudoku specifically

This proposal could have been "build a Spaces demo of *some* Carnot
capability." Sudoku is the right choice because:

| Property | Why it matters |
|----------|---------------|
| Universally interpretable | Zero domain expertise required to evaluate the demo |
| Pure EBM showcase | No LLM dependency, no SOTA-model gating, no API keys |
| Visual emergence | Energy descent is observable in a way LLM token streams aren't |
| Phase 1 → Phase 3 bridge | Same energy framework; demonstrates the paradigm |
| Kona historical resonance | The Kona project (Phase 3 inspiration target per CLAUDE.md) started with EBM Sudoku — direct lineage |
| Hardware story | Constraint-satisfaction-on-discrete-state is exactly the Phase 2 transpiler's target workload |
| Self-correcting | Sudoku has objectively correct answers; can demo repair of wrong-but-violating grids |

## Decentralization implications

**Rule 1 (local-first):** unaffected. Spaces app uses CPU-only Carnot
stack; no closed-weight LLMs anywhere in the demo.

**Rule 3 (distribution mirroring):** *enhanced*. The Spaces demo is
a public, reproducible artifact mirrorable to:
- HuggingFace Spaces (canonical hosting)
- Self-hosted Gradio at carnot-ebm.org/demo (mirror)
- Docker image on GHCR for offline deployment

**Rule 5 (hardware portability):** the demo's CPU-tier baseline
demonstrates Carnot's portability claim. Phase 2 follow-up adds the
KV260 FPGA tier as a hardware-acceleration showcase.

**Rule 7 (no vendor abstractions):** unaffected. Sampler backends are
Carnot's own (`python/carnot/samplers/*`), Gradio is open-source.

## Risks

1. **Spaces free-tier CPU may be too slow for hard puzzles.**
   Mitigation: cap iterations at a budget that completes in <30s
   on free tier; for diabolical puzzles, fall back to Norvig
   backtracking with explanatory note ("This puzzle's energy
   landscape is multi-modal; Carnot's MCMC sampler needs more time
   than free-tier allows. Try the FPGA path: [link]").

2. **Sampler reliability across difficulty tiers.** Some samplers may
   fail on hard puzzles within the iteration budget. Mitigation:
   pre-validate each sampler on the demo's puzzle set; only ship
   samplers that solve all tiers.

3. **Visualisation performance in browser.** Gradio's streaming may
   bottleneck animation. Mitigation: render trajectory frames at
   reasonable cadence (5-10 FPS), not every iteration.

4. **Educational framing mistake.** Risk that visitors interpret the
   demo as "Carnot is competing with Norvig's solver" — wrong frame.
   Mitigation: explainer copy explicitly positions this as a
   *paradigm demonstration*, not a benchmark contest.

5. **Spaces availability/uptime.** HuggingFace Spaces occasionally
   has outages. Mitigation: link to a Docker image users can run
   locally if Spaces is down.

## Acceptance criteria

1. Public Spaces app live at
   `huggingface.co/spaces/Carnot-EBM/sudoku-energy-solver`.
2. Solves Easy/Medium/Hard puzzles on free CPU tier in <30s each.
3. Animated visualisation streams at ≥5 FPS during solving.
4. Side-by-side comparison mode works for at least 3 sampler choices
   + Norvig baseline.
5. Inline explainer + EXPLAINER.md make the energy formulation
   accessible to non-technical visitors.
6. Cross-linked from main Carnot README, technical-report.html, and
   the Verifier Accuracy Paradox blog post.
7. Launch tweet/post template prepared for distribution-channel
   announcement.

## Why this is in change-proposals (not just a code task)

The Sudoku Spaces demo touches:
- A new Spaces deployment surface (decentralization rule 3).
- The Kona-lineage narrative arc (CLAUDE.md "Project Vision").
- Cross-cutting documentation updates (README, blog, technical
  report, Spaces README).
- A Phase 2 follow-up hook (KV260 FPGA path).

Filing as a proposal makes those couplings auditable; the planner
Sonnet sees the Phase 2 follow-up reference when designing future
milestones.

## Estimated effort

3-5 days of focused work, total:

- **Increment 1** (core app): 1-2 days
- **Increment 2** (explainer): 1 day
- **Increment 3** (sampler comparison): 1 day
- **Increment 4** (cross-linking): 0.5 days

Can be parallelised with the position-paper drafting (different
work surfaces, different cognitive modes).

## Strategic placement in the release roadmap

This deliverable repositions the near-term shipping plan:

| When | Ship | Replaces / Enhances |
|------|------|---------------------|
| Today | Verifier Accuracy Paradox blog post | (shipped) |
| 1-2 days | README + landing page revamp | (still planned) |
| 3-5 days | **EBM Sudoku Spaces demo** ← this proposal | replaces "Phase 1 verify-repair Spaces demo" |
| 1 week | AUROC postmortem blog post | (still planned) |
| 2-3 weeks | `carnot-eval-metrics` PyPI | (still planned) |
| 4-6 weeks | Position paper preprint | (still planned) |

The Sudoku demo is now the *visual centerpiece* of the pre-paper
release campaign. When the position paper lands, the Spaces demo is
already in circulation, making the abstract math concrete for any
visitor.

## Forward links

- Phase 2 KV260 follow-up: when the Hybrid Coprocessor demo lands,
  add a "GPU/FPGA mode" tab to the Spaces app showing the same
  puzzles solved at $\geq 100\times$ throughput.
- Phase 3 follow-up: when the multi-verifier rotation architecture
  demo (v0.4) lands, add a second Spaces demo showing rotation
  defending against adversarial grid-corruption attacks.
- Educational follow-up: short blog post "From Sudoku to Foundation
  Models: How Carnot's Energy Framework Scales" — bridges the demo
  to the position paper's broader claims.
