# ARC imitation/replay SOTA ingestion .415 - 2026-06-20

```json
{
  "arc_mapping": {
    "GAP-ARCH-EXPERT-INJECTION-REPLAY": {
      "next_experiment": "Seed human replay transitions at 5x priority in the DQN/value replay queue, then anneal priority only after self-play produces equal progress evidence.",
      "principle": "Keep scarce expert demonstrations active in replay so sparse-reward training starts from useful behavior instead of no-op exploration.",
      "source_ids": [
        "1704.03732",
        "1511.05952",
        "2302.02948"
      ]
    },
    "GAP-ARCH-FRAME-CHANGE-PREDICTOR": {
      "next_experiment": "Pretrain click heatmap and ACTION1-5 heads on the 14,672-example human corpus before mixing self-generated transitions.",
      "principle": "Turn human frame/action/click demonstrations into a frame-only predictor for which candidate actions change the screen.",
      "source_ids": [
        "2206.11795",
        "2407.15007",
        "2405.17476"
      ]
    },
    "GAP-ARCH-VALUE-ENERGY-HEADS": {
      "next_experiment": "Train value and contrastive energy heads from level_progress, steps-to-go, and human-vs-corrupted state/action pairs.",
      "principle": "Use human progress trajectories as offline value/energy labels while avoiding out-of-dataset action optimism.",
      "source_ids": [
        "2110.06169",
        "1905.11108",
        "2405.17476"
      ]
    }
  },
  "field_principles": {
    "arc_mapping": "maps literature to the actual queued ARC gaps, so follow-on work is actionable.",
    "field_principles": "principle annotations for every top-level artifact field.",
    "honest_verdict": "MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).",
    "human_corpus": "public replay counts must stay bare facts, not inferred hidden-eval solve claims.",
    "inference_substrate": "explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor.",
    "leaderboard_dqn_mapping": "records the expert-injection mechanism separately from Carnot's own training status.",
    "methods": "each source must map to one concrete ARC training decision and one caveat.",
    "preconditions_checked": "records WHICH resources were verified; pre-empts silent-missing-resource fabrication.",
    "random_seed": "bare integer seed for reproducible artifact generation.",
    "research_note_path": "repo-relative markdown path for deterministic parsing.",
    "source_ids": "only arXiv IDs whose abs pages were HTTP-200 checked may anchor the SOTA mapping.",
    "strongest_for_v416": "names the single strongest next hand-off without implying it has already been trained."
  },
  "honest_verdict": "complete: arc_imitation_sota_415_mapped_for_v416",
  "human_corpus": {
    "caveat": "public games only; value transfers only through held-out variants or hidden-game generalization, never public replay memorization",
    "example_count": 14672,
    "frame_change_rate": 0.9707606324972737,
    "frame_changing_actions": 14243,
    "level_progress_positive_count": 132,
    "public_games": 25,
    "replay_count": 342,
    "source": "ARC-AGI-3 public-demo human replay corpus",
    "usage": "bootstrap frame-change/clickability, behavior-prior, and value/energy heads from frame-derived features"
  },
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "leaderboard_dqn_mapping": {
    "carnot_mapping": "seed public human replay transitions into replay/value batches with extra priority, then decay only after cached self-play transitions prove equal or higher progress",
    "dqn_stack_components": [
      "prioritized_experience_replay",
      "expert_imitation_demo_seed",
      "persistent_action_effect_memory",
      "attention_cnn_value_net"
    ],
    "expert_priority_multiplier": 5,
    "pattern": "prioritized replay plus expert-injection",
    "source_note": "docs/research-notes/arc-leaderboard-competitive-intel-2026-06-20.md"
  },
  "methods": [
    {
      "arxiv_id": "1704.03732",
      "mapped_application": "GAP-ARCH-EXPERT-INJECTION-REPLAY",
      "name": "Deep Q-learning from Demonstrations",
      "pitfall": "DQfD is not a solve recipe; it improves sparse-reward exploration only if the demonstration distribution generalizes beyond public games.",
      "stack_mapping": "Use DQfD's mixture of TD learning, supervised expert-action loss, and prioritized replay as the direct template for seeding ARC human replays into the value/replay stack."
    },
    {
      "arxiv_id": "1511.05952",
      "mapped_application": "GAP-ARCH-EXPERT-INJECTION-REPLAY",
      "name": "Prioritized Experience Replay",
      "pitfall": "Priority can overfit public-game demos unless held-out variant transfer is the acceptance gate.",
      "stack_mapping": "Prioritize rare high-progress and expert transitions rather than sampling human and self-play transitions uniformly."
    },
    {
      "arxiv_id": "2206.11795",
      "mapped_application": "GAP-ARCH-FRAME-CHANGE-PREDICTOR",
      "name": "Video PreTraining",
      "pitfall": "ARC replays already have actions, but only 25 public games; the model must remain frame-only and transfer-tested.",
      "stack_mapping": "Treat frame/action replay as behavior pretraining: learn a visual action prior and clickability model before RL-style fine-tuning."
    },
    {
      "arxiv_id": "2110.06169",
      "mapped_application": "GAP-ARCH-VALUE-ENERGY-HEADS",
      "name": "Offline Reinforcement Learning with Implicit Q-Learning",
      "pitfall": "IQL is only appropriate after reward/progress labels are clean; frame_delta alone is not the same as task progress.",
      "stack_mapping": "Fit value/energy heads from logged actions without querying unseen actions, then extract an advantage-weighted behavior policy."
    },
    {
      "arxiv_id": "1905.11108",
      "mapped_application": "GAP-ARCH-VALUE-ENERGY-HEADS",
      "name": "SQIL imitation via sparse rewards",
      "pitfall": "Matching public expert states can still be the wrong objective on novel games, so progress labels and variants remain required.",
      "stack_mapping": "Use simple demonstration-match rewards as a first imitation-energy baseline before adding more brittle inverse-RL machinery."
    },
    {
      "arxiv_id": "2302.02948",
      "mapped_application": "GAP-ARCH-EXPERT-INJECTION-REPLAY",
      "name": "Efficient Online RL with Offline Data",
      "pitfall": "Carnot's current task is offline/cached; any online step must remain competition-legal and separately gated.",
      "stack_mapping": "Keep human replay data in the replay buffer during online/self-play updates instead of treating it as one-off pretraining."
    },
    {
      "arxiv_id": "2407.15007",
      "mapped_application": "GAP-ARCH-FRAME-CHANGE-PREDICTOR",
      "name": "Is Behavior Cloning All You Need?",
      "pitfall": "The theorem does not remove distribution-shift risk; hidden-game evaluation still requires variant transfer.",
      "stack_mapping": "Justifies a supervised behavior-cloning first pass for the click prior when payoffs are bounded and labels are clean."
    },
    {
      "arxiv_id": "2405.17476",
      "mapped_application": "GAP-ARCH-VALUE-ENERGY-HEADS",
      "name": "How to Leverage Diverse Demonstrations in Offline Imitation Learning",
      "pitfall": "Resultant-state overlap can be misleading when public-game states do not cover hidden mechanics.",
      "stack_mapping": "Select non-expert or self-play actions by resultant-state progress toward human-like states, then weight behavior cloning accordingly."
    }
  ],
  "preconditions_checked": {
    "agents_md_read": true,
    "arc_human_replay_notes_read": [
      "docs/research-notes/arc-human-baseline-and-replay-signal.md",
      "docs/research-notes/arc-human-replay-application-spec.md",
      "docs/research-notes/arc-frame-change-predictor-spec.md",
      "docs/research-notes/arc-world-model-trust-energy-spec.md",
      "docs/research-notes/arc-leaderboard-competitive-intel-2026-06-20.md"
    ],
    "arxiv_http_200_verified_ids": [
      "1704.03732",
      "1511.05952",
      "2206.11795",
      "2110.06169",
      "1905.11108",
      "2302.02948",
      "2407.15007",
      "2405.17476"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
    "leaderboard_submission": false,
    "live_llm_inference": false,
    "live_solve_claim": false,
    "ops_docs_modified": false,
    "research_references_filtered": true,
    "research_studying_filtered": true,
    "sweep_clusters_help_succeeded": true,
    "sweep_clusters_urls": [
      "http://export.arxiv.org/api/query?search_query=(abs:\"verifier+ensemble\"+OR+abs:\"verifier+ensembles\"+OR+abs:\"null+space\"+OR+abs:\"specification+gaming\"+OR+abs:\"process+reward+model\"+OR+abs:\"deliberative+alignment\"+OR+abs:\"reward+hacking\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending",
      "http://export.arxiv.org/api/query?search_query=(abs:\"active+inference\"+OR+abs:\"free+energy\"+OR+abs:\"free+energy+principle\"+OR+abs:\"predictive+coding\"+OR+abs:\"world+model\")+AND+(abs:\"LLM\"+OR+abs:\"language+model\"+OR+abs:\"reasoning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ],
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/1704.03732",
      "https://arxiv.org/abs/1511.05952",
      "https://arxiv.org/abs/2206.11795",
      "https://arxiv.org/abs/2110.06169",
      "https://arxiv.org/abs/1905.11108",
      "https://arxiv.org/abs/2302.02948",
      "https://arxiv.org/abs/2407.15007",
      "https://arxiv.org/abs/2405.17476"
    ]
  },
  "random_seed": 4498,
  "research_note_path": "docs/research-notes/arc-imitation-sota-415.md",
  "source_ids": [
    "1704.03732",
    "1511.05952",
    "2206.11795",
    "2110.06169",
    "1905.11108",
    "2302.02948",
    "2407.15007",
    "2405.17476"
  ],
  "strongest_for_v416": "flagged_for_v416: DQfD/PER-style human-replay expert-injection for the ARC frame-change predictor and value/energy heads, anchored by arXiv:1704.03732, arXiv:1511.05952, and arXiv:2206.11795"
}
```

Reliable channel only: `research-studying.md`, `research-references.md`,
`scripts/sweep_clusters.py`, ARC human-replay notes, arXiv abs-page HTTP-200
checks, and low-concurrency WebSearch/WebFetch of the top eight
imitation/behavior-cloning/offline-RL sources. `.venv/bin/python
scripts/sweep_clusters.py --help` succeeded. `scripts/sweep_clusters.py 0
--max-results 8` and `scripts/sweep_clusters.py 3 --max-results 8` emitted
focused cluster URLs used as the repo-standard fresh-pass substrate. No
`/deep-research` call was made. No training, live LLM inference, leaderboard
submission, or live solve was launched. No ops/status/traceability files were
modified.

Sources checked: arXiv:1704.03732, arXiv:1511.05952, arXiv:2206.11795, arXiv:2110.06169, arXiv:1905.11108, arXiv:2302.02948, arXiv:2407.15007, arXiv:2405.17476.

## Local Replay Signal

The active human-replay direction starts from the 14,672-example ARC public
human corpus across 25 public games. The useful labels are frame_delta,
level_progress, action_id, and click location. Humans changed the frame on
14,243 actions and produced 132 level-progress positives, which makes the data
well aligned with frame-change/clickability and sparse progress heads. The
honest caveat is that the corpus is public-game-only; it is training signal for
generalization, not a hidden-eval solve.

## Literature Mapping

- DQfD, arXiv:1704.03732, is the strongest expert-injection template: combine
  TD updates, supervised demonstrator-action loss, and prioritized replay.
- Prioritized Experience Replay, arXiv:1511.05952, supplies the queue policy:
  replay important transitions more often instead of uniform sampling.
- VPT, arXiv:2206.11795, supplies the visual behavior-pretraining analogy for
  frame/action demonstrations and later fine-tuning.
- IQL, arXiv:2110.06169, supplies the offline RL guardrail for value heads:
  avoid evaluating out-of-dataset actions while improving over behavior data.
- SQIL, arXiv:1905.11108, supplies a simple sparse-reward imitation baseline
  before more complex inverse-RL machinery.
- RLPD, arXiv:2302.02948, supports keeping offline human data in the replay
  buffer during later off-policy updates.
- Behavior Cloning Horizon, arXiv:2407.15007, makes the supervised behavior
  cloning first pass defensible when payoffs are bounded and labels are clean.
- Diverse Demonstrations IL, arXiv:2405.17476, maps self-play transitions by
  resultant-state progress toward expert-state manifolds.

## SOTA->Experiment Mapping

- GAP-ARCH-FRAME-CHANGE-PREDICTOR: pretrain the click heatmap and ACTION1-5
  heads from human frame/action/click labels before mixing cached self-play.
- GAP-ARCH-VALUE-ENERGY-HEADS: use `level_progress`, steps-to-go, and
  human-vs-corrupted state/action pairs to bootstrap value and contrastive
  energy heads.
- GAP-ARCH-EXPERT-INJECTION-REPLAY: seed the leaderboard-DQN-style replay queue
  with human demonstrations at 5x priority, then anneal only after self-play
  produces equal progress evidence.

For `.416`, the strongest hand-off is the expert-injection replay package:
DQfD/PER-style human replay injection for the frame-change predictor and
value/energy heads, with VPT-style frame-action behavior pretraining as the
supervised front end. Count this as a planning artifact only:
`inference_substrate=aggregation_from_upstream_artifacts`.

flagged_for_v416: DQfD/PER-style human-replay expert-injection for the ARC frame-change predictor and value/energy heads, anchored by arXiv:1704.03732, arXiv:1511.05952, and arXiv:2206.11795
