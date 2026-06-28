# Fresh SOTA sweep for the ARC generation wall — options scan (2026-06-28)

**Origin:** operator 2026-06-28 ("our score has not improved … are there no other research papers that
might give us some options to research?") — after the directed-generation program collapsed on design
(both branches already run: exp4883 inducer-strength, exp4893 history-conditioning). This is a FRESH
literature sweep (4 parallel WebSearch/WebFetch scouts) over paradigm families NOT previously ingested,
each filtered against (a) the information-theoretic first-contact wall and (b) Carnot's offline-legal /
weak-local-9B / tiny-per-game-data constraints. Real arXiv IDs only; SOTA-Ingestion Cycle Discipline.

## TL;DR — honest verdict

**No method found escapes the wall under our constraints.** Every intrinsic / skill / world-model /
curriculum objective is computed over *observable experience* (frames, transitions, model uncertainty),
so it can only re-rank or recompose what the explorer already sampled — exactly the nulled
action-history-conditioning class (exp4893). The wall (the winning prefix's order-state is absent from
frame AND action-history) is upstream of all of them. The one flagged "exception" (active-inference
EFE pragmatic goal-prior) reduces to the already-nulled goal-energy-directs-generation experiment
(exp4640). **The single most valuable new input is EXTERNAL VALIDATION that the wall is universal.**

## The decisive new finding — external validation (NOT previously ingested)

**Official ARC-AGI-3 technical report, arXiv:2603.24621.** The preview WINNER —
StochasticGoose / Tufa Labs (Dries Smit), a 4-layer CNN action-predictor + sparse-reward off-policy RL
+ state-hash dedup + per-level retraining — scored **12.58% on the preview set, then 0.25% on the full
unseen set, indistinguishable from frontier LLMs** (Gemini-3.1-Pro 0.37%, GPT-5.4 0.26%, Opus-4.6 0.25%).
Blind Squirrel (state-graph + ResNet18 value) 6.71% → similar collapse. **Only 2/8 agents completed >1
game.** Winners used "smart-random" exploration, NOT structure-learning; poor cross-game generalization
is called out explicitly. Code: github.com/DriesSmit/ARC3-solution.
**Read:** this independently confirms our information-theoretic-wall conclusion — *nobody* has cracked
generalizable first-contact; the best published agent collapses to ~frontier-baseline on unseen games.
It also contextualizes Carnot's standing 0.08 as genuinely competitive, not a failure.

## The four families scouted — all re-skin the wall

| Family | Strongest (arXiv) | Honest verdict |
|---|---|---|
| Unsupervised skill discovery | SkiLD 2410.18416, CSD 2302.05103, METRA 2310.08887, LSD 2202.00914 | Re-skin. Intrinsic reward (MI/distance/controllability/interaction) is over *already-sampled* states; needs 10^6–10^7 steps + factored state we lack. SkiLD's factor-interaction *objective* is a noteworthy verifier-feature idea, not a generation escape. |
| Intrinsic exploration beyond NGU/RND | Plan2Explore 2005.05960, Disagreement, Empowerment 2510.05996 (+critique 2511.04177), BYOL-Explore 2206.08332, EFE 2504.14898 | Epistemic objectives all re-skin RND (no prediction error on an unobserved order-state). Empowerment is counterproductive (avoids bottleneck/goal states, 2511.04177). EFE *epistemic* term re-hits the wall; EFE *pragmatic* goal-prior is the only structurally-different idea — but see caveat below. |
| Learned neural latent world models | Dreamer-XP/Plan2Explore, TD-MPC2 2310.16828, EfficientZero-V2 2403.00564, AXIOM 2505.24784 | None escape. **AutumnBench 2510.19788 is the empirical proof:** on tiny-interactive-data planning tasks ALL agents fail (humans 93.5%). A learned WM lets you PLAN once the winner is in the buffer; it cannot PUT it there from a few hundred transitions + one terminal reward. |
| Open-ended curriculum / library transfer | OMNI/OMNI-EPIC 2306.01711/2405.15568, DreamCoder, Voyager 2305.16291, Compositional Framework 2606.15386 (theory only) | None escape. A reusable skill/program is only invocable once the latent order-state that triggers L1 is discovered — the needle the library can't contain. Cross-game program transfer ≈ the nulled cross-game value transfer (exp4342). Voyager only transferred because the Minecraft API was constant — ARC hidden games change exactly that. |

## The one structurally-different idea — and its honest caveat

**Active-inference EFE with a verifier-supplied PRAGMATIC goal-prior** (EFE = epistemic info-gain +
pragmatic preference-prior p̂(x)). The escape is NOT exploration; it's biasing planning toward
configurations the verifier scores as goal-like that the agent has never visited. On-brand for Carnot's
Phase-4 active-inference track and connected to exp4020's graded goal-energy.
**CAVEAT (the discipline catch): this likely reduces to the already-nulled exp4640** ("wire exp4020's
graded is_goal as a LIVE goal-energy term to DIRECT generation" → `goal_energy_no_live_lift`, 0/100
solved). The core signal (goal-energy directs generation) is identical, and it nulled for the same
reason as everything else: you cannot up-weight / plan-toward a winning prefix the proposal mechanism
never generates. EFE's principled epistemic+pragmatic balance + horizon planning is a *marginal* framing
difference, not a new signal. **Verdict: not worth a probe unless a genuinely-generative (not
re-ranking) framing is found that exp4640 lacked — and that requires a generative model we don't have.**

## The honest options (ranked)

1. **Accept the wall — it is universal (2603.24621).** Ship the standing 0.08 (competitive) + the FoVer
   paper. The energy verifier's proven value is in *verification*-bottlenecked domains, not this
   generation-bottlenecked one. (Highest-confidence recommendation.)
2. **Study the winner's actual code** (github.com/DriesSmit/ARC3-solution) — even though it collapsed on
   unseen, its preview approach (CNN action-predictor + per-level retraining) might lift OUR *offline
   reproducible-level* count or action-efficiency (a real-but-secondary RHAE win), and grounds any future
   claim in the actual competitive frontier. Low cost, no wall-break promised.
3. **A paradigm change outside our constraints** — strong-inducer-as-live-agent (offline-illegal) or a
   much larger offline-legal local model. Bounded: even neural-online-RL winners (StochasticGoose)
   collapsed, so this is not a guaranteed win and is multi-week/engineering, not a research lever.

## Cross-references
- The wall + collapsed program: `docs/research-notes/arc-generation-program-2026-06-28.md`,
  `project_arc_l1_first_contact_wall`, `project_arc_generation_not_selection`, exp4914 / exp4903 /
  exp4893 / exp4883.
- Already-tried (excluded from this sweep): `docs/research-notes/arc-agi3-levers-tried-x-verdict-2026-06-25.md`.
- exp4640 (goal-energy directs generation, nulled) — the EFE-pragmatic caveat.
- New external: arXiv:2603.24621 (ARC-AGI-3 tech report), github.com/DriesSmit/ARC3-solution (winner code).
