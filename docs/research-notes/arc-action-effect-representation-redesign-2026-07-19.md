# ARC action-effect representation redesign — scoping design (2026-07-19)

**Status:** SCOPING / DESIGN ONLY. No code written, no live-path file modified. This note proposes
ONE concrete, falsifiable next experiment to replace the falsified `SmallFrameChangeCNN` action-effect
representation. It grounds the design in real ARC-AGI-3 leaderboard precedent + affordance-learning
literature + existing Carnot repo infrastructure, and includes an up-front adversarial self-critique.

**Reads as required input:** CLAUDE.md "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent",
"ARC Live-Path Reachability Discipline", "Failed-Experiment Rerun Discipline", "Missing-Verifier Gap
Logging", "Phase Prototype + Empirical Validation + Adversarial Check", "Literature Priority Discipline".

---

## 1. The problem, stated precisely (root cause of the falsification)

`ops/verifier_gaps.md :: GAP-ARCH-FRAME-CHANGE-PREDICTOR` is `attempted_and_falsified` (2026-07-19)
across four adversarially-clean experiments. Two structurally different representations — a hand-crafted
linear feature set (`REQ-ARC-WMTE-5727`, `results/experiment_5727_perception_action_effect_adequacy.json`)
and a learned conv net (`REQ-ARC-FCP-5730`, `results/experiment_5730_cnn_baserate_audit.json`) — both
collapse to the **same** degenerate structure: a per-action-**TYPE** base rate.

The load-bearing numbers:

- exp5727 (linear): LOO AUROC 0.844, but the `action_id`-only control also scores 0.883 →
  `frame_adds_over_action_id = −0.039`. The frame representation subtracts.
- exp5730 (CNN): held-out AUROC 0.539 (5-seed mean) vs an `action_id`-only control of 0.549 →
  `frame_adds_over_action_id = −0.010`. Again subtracts.
- The one apparent positive — click-location discrimination WITHIN action-6 (base rate held constant),
  0.918 AUROC on seed 4547 — did **not** survive a 5-seed re-run (0.444–0.918, mean 0.570) and lost to
  its own untrained/random-init structural control (mean 0.580). Seed luck, not signal.

**Why this happens mechanically — two compounding causes, both visible in the data:**

1. **Redundancy with a signal the agent already has for free.** `PersistentAEM`
   (`python/carnot/agentic/arc_solver_kit.py:1703`) already stores exactly the per-action-id +
   per-16px-click-bucket empirical change rate and scores candidates from it at `memory_weight=1.0`
   (vs the CNN's `cnn_weight=0.05`). Any representation whose only signal is the per-action-type base
   rate is *definitionally* redundant with `PersistentAEM` — which is mechanically why exp5729 saw the
   CNN reorder search **zero** times even when consulted ~27,000 times on lp85.

2. **A corpus artifact that removes the negative class.** The human-replay corpus is survivorship-biased:
   held-out `changed_fraction = 0.928`, and the per-action-id train change rates are 0.97/0.97/0.98/0.98
   for the four directional keys and 0.92 for clicks (exp5730 `heldout_summary.per_action_id_base_rate`).
   Humans essentially never demonstrate no-op actions, so "will this action change the frame" is ~93%
   positive and near-constant across states. There is almost no within-type variation to learn from —
   **except** for action-6 clicks, the one action-type with a real negative class (2973 clicks, 420
   no-ops = 14%). The interaction signal we actually want only *exists* in the click channel.

**The revised missing discriminator (from the gap entry):** a representation that captures WHICH
specific action, at WHICH specific state/location, produces a change — a genuine **action × frame
interaction** term — that is NOT reducible to the per-action-type marginal `PersistentAEM` already owns.

---

## 2. Real precedent (what strong ARC-AGI-3 agents and the affordance literature actually do)

The original 2026-06-20 leaderboard dive
(`docs/research-notes/arc-leaderboard-competitive-intel-2026-06-20.md`) described the leader's method
only as "a CNN that predicts which actions cause a frame change." That naive framing is exactly the
raw-grid architecture that failed today. A fresh, targeted pass finds the top of the board has since
moved *past* raw-grid CNNs to an **object-centric** representation.

### 2a. The current leading milestone-1 winner represents the board as OBJECTS, not the raw grid

Tufa Labs' **Duck / TAAF harness** (the 1.21% milestone-1 winner) does not feed the model a raw numeric
grid. Per the daily technique watch (`docs/research-notes/arc-agi3-leaderboard-technique-watch.md`,
2026-07-17 entry on the 暗黑AGI #9 submission, which runs the published TAAF/Duck bundle verbatim):

> "the Python REPL receives 4-connected object segmentation with translation-invariant hashes,
> containment and adjacency—not the raw numeric grid—and explicitly rejects changing border strips as
> likely HUD/timer state."

Tufa's own write-up confirms the perception stack is image + ASCII grid + a **segmentation tool**
([Tufa Labs — Duck Harness](https://tufalabs.ai/research/duck-harness/); source:
[Kaggle notebook `boristown/taaf-duck-harness-kaggle-share-reresubmission`](https://www.kaggle.com/code/boristown/taaf-duck-harness-kaggle-share-reresubmission),
[source bundle `jeroencottaar/taaf-kaggle-source-share`](https://www.kaggle.com/datasets/jeroencottaar/taaf-kaggle-source-share)).
The independent 3rd-place preview agent uses the same primitive class: **segment frames into components,
prioritize actions by visual salience over an object graph** (Rudakov, Shock, Cowley, "Graph-Based
Exploration for ARC-AGI-3 Interactive Reasoning Tasks", [arXiv:2512.24156](https://arxiv.org/abs/2512.24156)).

The takeaway: the field's action unit is **"interact with OBJECT X"** (a segmented, translation-invariant
object) — not **"click pixel (x, y)"** and not **"apply global conv to the raw frame."** Both falsified
Carnot representations are on the wrong side of that line. Carnot's own 2026-07-13 perception-grounding
audit (`docs/research-notes/arc-perception-grounding-audit-2026-07-13.md`) had already flagged the
classical connected-component / color-blob segmentation lever (citing arXiv:2512.24156) as the highest-
leverage next step for representation quality.

### 2b. The affordance literature names our exact failure mode AND its fix

The "a model that only learns a global action-type prior" pathology is a known, named problem in
affordance learning, with a standard remedy: **condition the effect prediction on OBJECT IDENTITY and
learn ONLINE from the agent's own interactions**, rather than fitting a marginal.

- **"Learning Affordances from Interactive Exploration using an Object-level Map"**
  ([arXiv:2501.06047](https://arxiv.org/html/2501.06047v1)). Learns affordances online from the agent's
  own pick/push interactions, keyed on tracked object identity; per its own framing it "explicitly avoids
  marginal priors by grounding affordance predictions in specific object instances tracked across time,
  rather than treating objects interchangeably." This is precisely the antidote to a per-action-type
  marginal, and it is *online / within-episode* — which matters for the hidden-game framing (see §5).
- **COMET — "Causal Object-Centric Models for Planning with Monte Carlo Tree Search"**
  ([arXiv:2606.14418](https://arxiv.org/abs/2606.14418)). Binds an action to an individual object via an
  "action-slot fusion mechanism … in slot transition prediction" — i.e. an explicit **action × object**
  interaction term in the transition model, the structural thing our global-pooled conv lacks.
- **PLATO — "Predicting Latent Affordances Through Object-Centric Play"**
  ([arXiv:2203.05630](https://arxiv.org/pdf/2203.05630)) and selective-contrastive affordance grounding
  ([arXiv:2508.07877](https://arxiv.org/pdf/2508.07877)): object-centric, contrastive affordance
  prediction — learn "what happens to THIS object" contrastively rather than a pointwise classifier.

The statistical analogy for the base-rate control we must keep is the propensity-score / IPTW idea:
measure an effect by conditioning on the treatment assignment (here: the action-TYPE) and looking at
variation *within* it, so the marginal cancels (canonical reference: Williamson et al., "Variance
reduction in randomised trials by inverse probability weighting using the propensity score",
[Stat. Med. 2014, PMC4285308](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4285308/)). exp5727/exp5730's
`action_id`-only control is exactly this idea; the redesign keeps it and adds a stronger one (§4).

### 2c. Carnot already has every object-centric primitive this needs (no new perception infra)

An inventory of `python/carnot/agentic/` shows the object-centric layer is already built and unit-tested;
the falsified CNN simply never used it:

- `object_hash(blob)` and `blob_topology(frame)` (`arc_color_blob_salience.py:548,570`) —
  translation-invariant object identity (sha1 of color + bbox-normalized cells) + a containment tree and
  4-adjacency graph. This is the exact "translation-invariant hash + containment + adjacency"
  representation the Duck harness uses.
- `ObjectHistorySaliencePrior` (`arc_object_history_salience.py:65`) — an *already-live* per-`object_hash`
  online action-effect memory: `observe_transition` tallies whether clicking that object's hash changed
  the frame, and `score` adds `change_rate` to blob salience. This is the object-level analogue of
  `PersistentAEM` and the natural home for the online version of this experiment.
- `object_centric_slots(frame, neighborhood_radius=2)` (`arc_value_learner.py:604`) — the only existing
  spatial-locality extractor: a 3×3 local density patch around each object/keypoint plus
  object-neighborhood-gap slots.
- `compute_grid_delta(prev, nxt)` (`arc_agi3_world_model.py:46`) — the ground-truth changed-cell +
  color-transition primitive that produces the labels.
- `InertClickSigPruner` / `click_signature` (`arc_inert_click_pruner.py`) — an existing structural
  object signature `(color, size, is_rect, twin_count)` and inert/leveled tally.
- `rich_action_candidates` (`arc_graph_explore.py:117`) is the live candidate generator with a
  `frame_change_scorer` slot — the live-path seam where a validated scorer would plug in.

---

## 3. The ONE proposed experiment

**Title (working):** Object-centric within-action click-affordance discriminator vs the action-type +
object-property base rates (offline LOO proxy + online prefix-causal proxy).

**One-paragraph statement.** Replace the raw-frame representation with an **object-centric** one: featurize
each action-6 click candidate by its TARGET OBJECT — its translation-invariant `object_hash` context
(color, area, `is_rect`, normalized shape key), its containment depth and adjacency degree from
`blob_topology`, and a K×K local-neighborhood patch around the object centroid (reusing
`object_centric_slots`) — and train it with a **within-frame contrastive ranking** objective that ranks
the object that actually changed the frame above the inert objects present in the SAME frame under the
SAME action-type. Restricting to action-6 clicks (the only channel with a real negative class) and ranking
*within a frame* means the per-action-type marginal and any global scalar cancel by construction, so the
model can only win by using object structure — closing the exact escape hatch that made exp5727/exp5730
mirages. Evaluate it two ways: an **offline cross-game LOO** proxy (exp5727 methodology: does it transfer
to a held-out GAME?) and an **online prefix-causal** proxy (exp5730-style, but each frame scored using only
transitions observed BEFORE it: does object identity beat `PersistentAEM`'s action-type+click-bucket memory
within a game?).

**Why this specific design should NOT collapse to the base-rate mirage (the most important part).**
The two prior representations collapsed because their objective was a *pointwise* changed/no-op classifier
over the whole corpus, which a per-action-type marginal minimizes trivially, and their representation was a
*global* frame summary with no object localization. This design removes both escape routes structurally:

1. **The objective cancels the marginal.** Contrastive ranking is computed WITHIN one frame across the
   objects present in it, all sharing the same action-type (click). A model that outputs a constant per
   action-type (the base rate) gets ranking-loss 0 gradient — it cannot reduce the loss at all, because
   every candidate in the group has the identical action-type. The only way to win is to separate the
   changing object from the inert ones using object features. This is the propensity/IPTW idea (§2b) made
   into a loss.
2. **The representation can localize.** Object features (`object_hash`, shape, containment/adjacency, a K×K
   patch) are per-candidate-object, not a global pool — so a within-frame difference between two objects is
   representable, which the global-pooled conv provably could not resolve (exp5730's directional head was
   at/below chance and the click head was seed luck).
3. **Restriction to the channel that has signal.** Directional actions 1–5 are ~93–98% change (no negative
   class — a corpus artifact, not a learnable target), so the experiment does not even attempt them; it
   spends its statistical power on action-6 clicks where the 14% no-op class makes an interaction term
   *identifiable*.

**Falsifiable acceptance gate (matching exp5730 rigor exactly).**

Primary metric: `object_features_add_over_baselines` = (5-seed-mean object-model within-action-6 AUROC)
− max(control AUROCs), where the controls are:

- **C1 — action-type + click-bucket base rate** (the exp5730 `action_id`-only / `PersistentAEM` control):
  the "free" signal the agent already has.
- **C2 — object-property-only base rate** (a logistic on just target-object color, area, `is_rect`): guards
  against the SAME redundancy trap one level up — if "which object changes" is fully explained by object
  color/size, the learned/patch representation is re-deriving `ColorBlobSaliencePrior` for nothing.

Gate to count as **real signal** (all must hold):

- `object_features_add_over_baselines > 0.05` AUROC (exp5730's `frame_beats_action_margin_threshold`),
- on the **worst of 5 seeds** (kills the seed-luck mirage exp5730 caught — a lone high seed does not count),
- beating a **untrained/random-init structural control** of the same architecture (exp5730 discipline),
- with the positive control passing (in-sample AUROC > 0.5) and a reported `n_within_frame_pairs` above a
  pre-registered minimum (guards the coverage-collapse mode, §6).

Report BOTH the offline cross-game LOO number and the online prefix-causal number against C1/C2.

**RETIRE condition (Failed-Experiment Rerun Discipline, pre-registered):** if object features beat NEITHER
the cross-game LOO gate NOR the online prefix-causal gate by >0.05 on the worst of 5 seeds, retire the
object-centric offline action-effect-predictor lineage to `ops/exclusion_manifest.yaml`; the live path
keeps only the existing online `ObjectHistorySaliencePrior` memory, and GAP-ARCH-FRAME-CHANGE-PREDICTOR is
closed with the honest bound "no offline-learnable, non-base-rate action-effect representation exists on
this corpus; action-effect is online-within-game only." A cross-game null WITH an online-within-game
positive is a partial win (see §5) and does NOT trigger full retirement — it redirects the live path from
click-buckets to object identity.

**Prior-failure block (rerun-discipline compliance).**
- Names the prior failures: exp5727 (linear, `frame_adds_over_action_id −0.039`), exp5730 (CNN,
  `−0.010`; within-action-6 click discrimination seed-luck null).
- Diagnosed root cause: pointwise-classifier objective (minimizable by the per-action-type marginal) over a
  global/raw-frame representation with no object localization, on a survivorship-biased corpus.
- What is different: (a) object-centric per-candidate representation (segmentation + translation-invariant
  hash + containment/adjacency + local patch) instead of raw-frame/global-pooled conv; (b) within-frame
  contrastive ranking objective that structurally cancels the action-type marginal instead of a pointwise
  loss it minimizes; (c) restriction to the action-6 channel that actually has a negative class; (d) a
  second, stronger control (object-property-only) and an online prefix-causal evaluation the prior runs
  lacked. This is a genuine representation + objective change, NOT a retrain/re-tune of `SmallFrameChangeCNN`
  (which the gap entry explicitly forbids).

**Scope / effort:** ~half a day. No new perception infra (every primitive in §2c already exists), no LLM,
no GPU — a CPU logistic/small-MLP over object features + a small patch, `inference_substrate:
verifier_ensemble_against_cached_candidates` (1s floor). The exp5730 harness (corpus load, 5-seed loop,
LOO, base-rate control, untrained control, seed-robustness fields) is directly reusable as the skeleton;
the only new code is the object-featurizer (compose existing calls) and the within-frame contrastive
grouping/loss.

**Live-path target (Live-Path Reachability Discipline).** The validated scorer plugs into
`rich_action_candidates`'s `frame_change_scorer` slot / `LiveActionEffectScorer`, replacing the falsified
CNN term; the online version is a generalization of `ObjectHistorySaliencePrior` (key its effect memory on
`object_hash` + context features rather than raw hash alone). The offline LOO is the development proxy for
the live within-game quantity, per the ARC framing rule.

---

## 4. Why the two controls are non-negotiable

exp5727/exp5730's entire value was the `action_id`-only control (C1); without it the 0.844 / 0.918 numbers
would have shipped as false positives. The redesign adds C2 (object-property-only) because the object-centric
move introduces a NEW, subtler base-rate escape hatch: object salience (`color-rarity × area`) is already
computed by `ColorBlobSaliencePrior`, and "salient colored objects tend to be interactive" may explain most
of the changing-vs-inert split on its own. If the learned patch/shape representation does not beat a plain
logistic on (color, area, is_rect), then the honest finding is "hand object-salience suffices; no learned
representation needed" — still useful for the live path (cheaper, no training), but it closes the
learned-representation direction rather than opening it. Both controls must be beaten by >0.05 on the worst
seed for the result to be a genuine learned interaction signal.

---

## 5. Reconciling with the hidden-game framing (why online, not just offline)

Per CLAUDE.md "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent", the deliverable is a live agent that
DISCOVERS an unseen game at runtime, not a pre-trained cross-game predictor. This directly shapes the
experiment: the offline cross-game LOO test asks "does a UNIVERSAL object→effect prior exist?" — and the
benchmark is explicitly designed so that it may NOT (games share no semantics by construction). The online
prefix-causal test asks the live-relevant question instead: "given the agent's OWN transitions observed so
far in THIS game, does an object-centric effect memory rank the next click better than `PersistentAEM`'s
action-type + coarse-click-bucket memory?" This is the exp5730 methodology run causally over each game's
trajectory prefix, and it is the version whose positive result would actually improve the live agent. The
literature precedent for this exact online, per-object-identity framing is arXiv:2501.06047 (§2b). Running
both tests in one experiment costs almost nothing (same corpus, same featurizer) and yields a clean
decision matrix: cross-game positive → a transferable object-effect prior (best case); cross-game null +
online positive → use object identity in the online memory (likely case, still a live win); both null →
full retirement (honest close).

---

## 6. Adversarial self-critique — the most likely ways THIS also fails (stated up front)

Per the Phase Prototype + Adversarial Check discipline, before proposing I ran a hostile-reviewer pass on
my own design. The three most likely failure modes, in order of probability:

1. **Cross-game affordance is game-specific by design → the offline LOO is null for the SAME reason the
   frame features were.** In game A a red rectangle is a button; in game B it is a wall. If "which object
   is interactive" is a per-game convention with no universal prior, a cross-game object-effect model sits
   at chance no matter how good the representation — the object-centric move buys transfer of *identity*
   (a red 3×3 block is recognizably the same object across frames) but NOT of *affordance* (whether that
   object does anything). This is the single most likely outcome and I have pre-registered it as a
   partial-win, not a failure (§3 retire condition, §5): the fallback is the online within-game test, which
   is the live-relevant one anyway. If BOTH are null, the design genuinely fails and the lineage retires.
2. **The object-property-only control (C2) already captures the signal.** If a logistic on (color, area,
   is_rect) reaches the same AUROC as the learned patch/shape model, the "learned representation" adds
   nothing — object salience is sufficient. Probability: moderate. This is not a mirage (the object features
   are real and beat C1), but it demotes the finding from "learned energy" to "hand object-features suffice,"
   and the correct action is to ship the cheap object-salience term to the live path and NOT build a learned
   model. I would rather discover this cheaply here than after building training infra.
3. **Within-frame contrastive coverage collapse.** If most frames contain only 1–2 objects, or humans
   almost always clicked the single changing object, there are too few within-frame changing-vs-inert pairs
   to train or evaluate the ranking objective — the GAP-4-style coverage collapse where the mechanism is
   real but n is too small to move a number. Mitigation: report `n_within_frame_pairs`, require a
   pre-registered minimum, and backfill negatives from the agent's own observed no-op clicks (which
   `ObjectHistorySaliencePrior` already collects) rather than only human-clicked objects. If even with the
   backfill the pair count is too low, the honest verdict is "insufficient negative-class data offline; this
   must be measured live" — which again points back to the online path.

Lesser risks: (a) the K×K patch reintroduces raw-pixel dependence that fails to transfer (mitigate: keep the
patch small and color-index-based, and ablate it — the object-hash/shape/topology features should carry most
of the signal); (b) label noise from `compute_grid_delta` counting HUD/timer strips as changes (mitigate:
reuse the Duck harness's border-strip rejection, which the repo already has via
`ColorBlobSaliencePrior.is_status_bar_like`); (c) `object_hash` collisions on tiny objects (low, sha1).

**Net honest read:** the design's *representation* is well-grounded (it is literally what the leaderboard
leader uses and what the repo already implements) and its *objective* structurally closes the base-rate
hole two independent representations fell into. Its biggest risk is not a mirage — the controls prevent that
— but a genuine *null on cross-game transfer* that the benchmark's own design makes plausible. That is why
the experiment is built to also measure the online within-game gain, so that the likely cross-game null
still yields an actionable live-path decision rather than another dead end.

---

## 7. Cross-references

- Falsification chain: `ops/verifier_gaps.md :: GAP-ARCH-FRAME-CHANGE-PREDICTOR`;
  `results/experiment_5727_perception_action_effect_adequacy.json`,
  `results/experiment_5730_cnn_baserate_audit.json`, `..._5728_cnn_weight_sweep.json`,
  `..._5729_gtv_gate_fix_ab.json`, `..._5590_frame_change_cnn_dict_candidate_fix_ab.json`.
- Live-path code: `python/carnot/agentic/arc_frame_change_predictor.py` (the falsified scorer),
  `arc_solver_kit.py:1703` (`PersistentAEM`, the base-rate control), `arc_color_blob_salience.py`
  (`object_hash`/`blob_topology`), `arc_object_history_salience.py` (`ObjectHistorySaliencePrior`, the
  online object-effect memory), `arc_value_learner.py:604` (`object_centric_slots`, the patch extractor),
  `arc_graph_explore.py:117` (`rich_action_candidates`, the live seam).
- Precedent: [Tufa Duck harness](https://tufalabs.ai/research/duck-harness/) +
  [Kaggle TAAF notebook](https://www.kaggle.com/code/boristown/taaf-duck-harness-kaggle-share-reresubmission) +
  [source bundle](https://www.kaggle.com/datasets/jeroencottaar/taaf-kaggle-source-share);
  [arXiv:2512.24156](https://arxiv.org/abs/2512.24156) (graph exploration, object segmentation);
  [arXiv:2501.06047](https://arxiv.org/html/2501.06047v1) (online per-object-identity affordance learning);
  [arXiv:2606.14418](https://arxiv.org/abs/2606.14418) (COMET, action×object interaction);
  [arXiv:2203.05630](https://arxiv.org/pdf/2203.05630) (PLATO, object-centric affordance);
  [PMC4285308](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4285308/) (IPTW/propensity, the base-rate
  control's statistical framing).
- Prior Carnot notes: `docs/research-notes/arc-perception-grounding-audit-2026-07-13.md` (already flagged
  object-segmentation as the higher-leverage lever),
  `docs/research-notes/arc-leaderboard-competitive-intel-2026-06-20.md` (the original, naive CNN framing
  this note supersedes), `docs/research-notes/arc-agi3-leaderboard-technique-watch.md` (2026-07-17 Duck/TAAF
  object-representation detail).
