# ARC affordance/action-effect SOTA ingestion .416 - 2026-06-20

```json
{
  "energy_augmented_mapping": {
    "caveat": "energy only helps if computed over structural features; frame-marginal energy must stay a null until transfer is proven.",
    "energy_policy": "Use structural objective energy as a progress potential so the explorer prefers changes that look goal-consistent, not merely non-zero.",
    "ranking_formula": "P(frame_change) * (-delta_E)",
    "source_ids": [
      "2407.10341",
      "2601.07060",
      "2602.00460",
      "2602.03201"
    ],
    "target": "energy-augmented ranking for action efficiency"
  },
  "field_principles": {
    "energy_augmented_mapping": "maps progress/potential evidence onto P(frame_change) * (-delta_E) ranking.",
    "field_principles": "principle annotations for every top-level artifact field.",
    "frame_change_mapping": "maps affordance/action-effect evidence onto frame-change predictor pruning.",
    "honest_verdict": "MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).",
    "inference_substrate": "explicit substrate so adversarial_verify applies the right duration floor.",
    "methods": "each source maps to a concrete ARC action-efficiency decision and caveat.",
    "preconditions_checked": "records WHICH resources were verified; pre-empts silent-missing-resource fabrication.",
    "random_seed": "bare integer seed for reproducible artifact generation.",
    "research_note_path": "repo-relative markdown path for deterministic parsing.",
    "source_ids": "only arXiv IDs whose abs pages were HTTP-200 checked may anchor the mapping.",
    "strongest_for_v417": "names the single strongest next hand-off for .417."
  },
  "frame_change_mapping": {
    "acceptance_gate": "reduce median actions-to-first-levelup without reducing held-out solve-rate",
    "candidate_policy": "Predict which action/click cells are feasible and likely to change the frame, then prune low-affordance no-op candidates before BFS expansion.",
    "source_ids": [
      "2006.15085",
      "2008.09241",
      "2501.06047",
      "2404.15648"
    ],
    "target": "affordance-pruned frame-change predictor",
    "training_signal": "Use human replay and cached explorer transitions as action-effect labels: frame_delta, click cell, action_id, and object/contact region."
  },
  "honest_verdict": "complete: arc_affordance_sota_416_mapped_for_v417",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods": [
    {
      "arxiv_id": "2006.15085",
      "mapped_application": "GAP-ARCH-AFFORDANCE-PRUNING",
      "name": "What can I do here? A Theory of Affordances in Reinforcement Learning",
      "pitfall": "availability is not progress; the mask must be paired with energy or level-progress checks before solve claims.",
      "stack_mapping": "Treat affordances as a learned feasible-action mask that reduces the branching factor before transition/value scoring."
    },
    {
      "arxiv_id": "2008.09241",
      "mapped_application": "GAP-ARCH-FRAME-CHANGE-PREDICTOR",
      "name": "Learning Affordance Landscapes for Interaction Exploration in 3D Environments",
      "pitfall": "the source setting is RGB-D 3D exploration; ARC use must stay frame-only and cannot read hidden environment state.",
      "stack_mapping": "Port the image-region-to-action-success idea into an ARC click heatmap plus ACTION1-5 frame-change head."
    },
    {
      "arxiv_id": "2404.15648",
      "mapped_application": "GAP-ARCH-ACTION-EFFECT-REPRESENTATION",
      "name": "Cross-Embodied Affordance Transfer through Learning Affordance Equivalences",
      "pitfall": "robot trajectory transfer does not directly imply discrete ARC action transfer; it only motivates the representation.",
      "stack_mapping": "Use object/action/effect triples as the representation target for cached ARC action-effect examples."
    },
    {
      "arxiv_id": "2407.10341",
      "mapped_application": "GAP-ARCH-ENERGY-PROGRESS-SHAPING",
      "name": "Affordance-Guided Reinforcement Learning via Visual Prompting",
      "pitfall": "a live VLM reward is not an ARC competition substrate; this artifact uses only the shaping pattern.",
      "stack_mapping": "Take the dense affordance-shaped reward pattern, but replace VLM keypoint rewards with local structural energy progress."
    },
    {
      "arxiv_id": "2501.06047",
      "mapped_application": "GAP-ARCH-FRAME-CHANGE-PREDICTOR",
      "name": "Learning Affordances from Interactive Exploration using an Object-level Map",
      "pitfall": "object mapping must be deterministic from frames; no game internals or private state can enter the labels.",
      "stack_mapping": "Track object instances across views so repeated ARC transitions produce denser action-effect labels instead of isolated pixels."
    },
    {
      "arxiv_id": "2601.07060",
      "mapped_application": "GAP-ARCH-ENERGY-PROGRESS-SHAPING",
      "name": "PALM: Progress-Aware Policy Learning via Affordance Reasoning",
      "pitfall": "PALM is a VLA manipulation stack; Carnot should borrow progress cues, not the model substrate.",
      "stack_mapping": "Use progress-aware affordance reasoning to separate repeated frame-changing actions from actions that advance a subtask."
    },
    {
      "arxiv_id": "2602.00460",
      "mapped_application": "GAP-ARCH-FRONTIER-EXPLORATION",
      "name": "Search Inspired Exploration in Reinforcement Learning",
      "pitfall": "online RL frontier growth is not a banked ARC solve; use the idea as a pruning and ordering policy over cached candidates.",
      "stack_mapping": "Select frontier state-action pairs by cost-to-come/cost-to-go and learning progress instead of flat repeated BFS expansion."
    },
    {
      "arxiv_id": "2602.03201",
      "mapped_application": "GAP-ARCH-ENERGY-PROGRESS-SHAPING",
      "name": "SLOPE: Optimistic Potential Landscape Shaping for Model-based Reinforcement Learning",
      "pitfall": "optimistic learned rewards can overstate progress; structural energy and solve-rate guards must stay authoritative.",
      "stack_mapping": "Map optimistic potential landscapes onto Carnot's objective energy term so sparse level-progress can guide search earlier."
    }
  ],
  "preconditions_checked": {
    "agents_md_read": true,
    "arxiv_http_200_verified_ids": [
      "2006.15085",
      "2008.09241",
      "2404.15648",
      "2407.10341",
      "2501.06047",
      "2601.07060",
      "2602.00460",
      "2602.03201"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
    "frame_change_notes_read": [
      "docs/research-notes/arc-frame-change-predictor-spec.md",
      "docs/research-notes/arc-energy-augmented-strategy.md",
      "docs/research-notes/arc-417-shaping-action-efficiency.md",
      "docs/research-notes/arc-imitation-sota-415.md"
    ],
    "leaderboard_submission": false,
    "live_llm_inference": false,
    "live_solve_claim": false,
    "ops_docs_modified": false,
    "research_references_filtered": true,
    "research_studying_filtered": true,
    "sweep_clusters_help_succeeded": true,
    "sweep_clusters_urls": [
      "http://export.arxiv.org/api/query?search_query=(abs:\"active+inference\"+OR+abs:\"free+energy\"+OR+abs:\"free+energy+principle\"+OR+abs:\"predictive+coding\"+OR+abs:\"world+model\")+AND+(abs:\"LLM\"+OR+abs:\"language+model\"+OR+abs:\"reasoning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending",
      "http://export.arxiv.org/api/query?search_query=(abs:\"verifier+ensemble\"+OR+abs:\"verifier+ensembles\"+OR+abs:\"null+space\"+OR+abs:\"specification+gaming\"+OR+abs:\"process+reward+model\"+OR+abs:\"deliberative+alignment\"+OR+abs:\"reward+hacking\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ],
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2006.15085",
      "https://arxiv.org/abs/2008.09241",
      "https://arxiv.org/abs/2404.15648",
      "https://arxiv.org/abs/2407.10341",
      "https://arxiv.org/abs/2501.06047",
      "https://arxiv.org/abs/2601.07060",
      "https://arxiv.org/abs/2602.00460",
      "https://arxiv.org/abs/2602.03201"
    ]
  },
  "random_seed": 4508,
  "research_note_path": "docs/research-notes/arc-affordance-sota-416.md",
  "source_ids": [
    "2006.15085",
    "2008.09241",
    "2404.15648",
    "2407.10341",
    "2501.06047",
    "2601.07060",
    "2602.00460",
    "2602.03201"
  ],
  "strongest_for_v417": "flagged_for_v417: affordance-pruned frame-change predictor with SLOPE-style optimistic energy progress shaping, anchored by arXiv:2008.09241, arXiv:2006.15085, and arXiv:2602.03201"
}
```

Reliable channel only: `research-studying.md`, `research-references.md`,
`scripts/sweep_clusters.py`, ARC frame-change/energy strategy notes, arXiv
abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of the top
eight affordance learning, action-effect model, and sparse-reward exploration
sources. `.venv/bin/python scripts/sweep_clusters.py --help` succeeded.
`scripts/sweep_clusters.py 3 --max-results 8` and
`scripts/sweep_clusters.py 0 --max-results 8` emitted the focused fresh-pass
cluster URLs. No `/deep-research` call was made. No training, live LLM
inference, leaderboard submission, or live solve was launched. No
ops/status/traceability files were modified.

Sources checked: arXiv:2006.15085, arXiv:2008.09241, arXiv:2404.15648, arXiv:2407.10341, arXiv:2501.06047, arXiv:2601.07060, arXiv:2602.00460, arXiv:2602.03201.

## Focused Finding

The direct fit for `.417` is an affordance-pruned frame-change predictor:
learn which action/click cells are feasible and likely to change the frame,
then prune low-affordance candidates before the explorer spends actions on
them. Affordance theory (arXiv:2006.15085) supports the branch-factor reduction;
interaction-exploration affordance landscapes (arXiv:2008.09241) give the
closest CNN-style image-region-to-action-success template; object-map
interactive exploration (arXiv:2501.06047) and action-effect-object
representations (arXiv:2404.15648) sharpen the label representation.

## Energy-Augmented Ranking

The Carnot-specific graft is energy-augmented ranking, not a pure copy of an
affordance classifier. Use affordance probability to remove likely no-ops, then
rank survivors by `P(frame_change) * (-delta_E)`. KAGI (arXiv:2407.10341) and
PALM (arXiv:2601.07060) support progress-aware affordance shaping; SIERL
(arXiv:2602.00460) supports frontier selection by learning progress and
cost-to-go; SLOPE (arXiv:2602.03201) is the strongest energy analogue because
it replaces flat sparse rewards with an optimistic potential landscape.

## SOTA->Experiment Mapping

- Frame-change predictor: train an action-effect model over frame/action/click
  labels and use it to prune no-op candidates before BFS expansion.
- Energy-augmented ranking: score remaining candidates by
  `P(frame_change) * (-delta_E)` where `delta_E` comes only from structural
  objective-energy features.
- Frontier policy: choose state-action frontiers that are reachable, not fully
  exhausted, and promising under the energy-progress term.

For `.417`, the strongest hand-off is the affordance-pruned frame-change
predictor with SLOPE-style optimistic energy progress shaping. Treat this as a
planning artifact only: `inference_substrate=aggregation_from_upstream_artifacts`.

flagged_for_v417: affordance-pruned frame-change predictor with SLOPE-style optimistic energy progress shaping, anchored by arXiv:2008.09241, arXiv:2006.15085, and arXiv:2602.03201
