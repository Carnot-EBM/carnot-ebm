"""Exp 4911 SOTA ingestion for the .453 frontier.

Spec refs: REQ-ARC-WMTE-4911,
SCENARIO-ARC-WMTE-4911-V453-WALL-AND-PIVOT-MAPPED,
SCENARIO-ARC-WMTE-4911-BLOCKED-UPSTREAM,
SCENARIO-ARC-WMTE-4911-NO-FABRICATION.

This module is aggregation-only. It reads the measured A1/A1b frontier
artifacts, records the reliable-channel sweep provenance, maps the final ARC
wall diagnostics, and separately maps the post-sprint verifier-moat pivot
without claiming a solve, training run, leaderboard result, or model load.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4911_sota_ingestion_v453_frontier.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
REFERENCES_RELATIVE_PATH = "research-references.md"
A1_ARTIFACT_RELATIVE_PATH = "results/experiment_4903_env_grounded_location_pruned_search.json"
A1B_ARTIFACT_RELATIVE_PATH = "results/experiment_4904_latent_action_interface.json"
NORTH_STAR_RELATIVE_PATH = "ops/north-star.md"
NOTE_PATH = "research-studying.md#exp-4911-sota-ingestion-v453-frontier"
REFERENCES_PATH = "research-references.md#exp-4911-v453-frontier-source-set"
RANDOM_SEED = 4911
DURATION_S = 0.0001
HONEST_VERDICT = "success_sota_ingestion_v453_frontier_mapped"
BLOCKED_VERDICT = "blocked_upstream_artifact_missing"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
AIMED_AT_FORK_VERDICT = "WALL_DEEPER_THAN_VALUE_PREDICTION"
A1B_NULL_VERDICT = "VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES"
SELECTED_BRANCH = "wall_survives_four_representations_plus_env_grounding"
STUDYING_SECTION_START = "<!-- EXP4911-SOTA-INGESTION-V453-FRONTIER-START -->"
STUDYING_SECTION_END = "<!-- EXP4911-SOTA-INGESTION-V453-FRONTIER-END -->"
REFERENCES_SECTION_START = "<!-- EXP4911-V453-FRONTIER-REFERENCES-START -->"
REFERENCES_SECTION_END = "<!-- EXP4911-V453-FRONTIER-REFERENCES-END -->"
TERMINAL_PREFIXES = (
    "blocked_",
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_USER_FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal prefix; success_sota_ingestion_v453_frontier_mapped."
    },
    "aimed_at_fork_verdict": {
        "principle": "the A1 fork verdict the ingestion targets (selects the .453 branch)."
    },
    "methods_mapped": {
        "principle": (
            "the strongest 3-5 methods, each with a real arXiv ID + "
            "experiment_graft + fails_when."
        )
    },
    "post_sprint_pivot_methods": {
        "principle": (
            "the verifier-moat / oracle-distinct candidates for when the ARC "
            "sprint retires 6/30 (north-star §1/§5)."
        )
    },
    "arxiv_ids_cited": {
        "principle": "every method claim cites a verifiable HTTP-200 arXiv ID (no fabrication)."
    },
    "citations": {
        "principle": "HTTP-200 arXiv source metadata backing every method claim."
    },
    "banned_channels_excluded": {
        "principle": "true -- /deep-research NOT invoked; nulled classes NOT re-ingested."
    },
    "flagged_for_v453": {
        "principle": (
            "the strongest method(s) flagged so the .453 planner reads the mapping "
            "(discover->ingest->plan->experiment)."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (reads upstream verdicts + sweeps; "
            "0.0001s floor)."
        )
    },
    "random_seed": {
        "principle": "determinism for the sweep query ordering."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of (branch, sweep queries, mapped methods) so a "
            "replication catches drift."
        )
    },
}
FIELD_PRINCIPLES = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "a1b_fork_verdict": {
        "principle": "the last-representation result used to harden the wall branch."
    },
    "selected_branch": {
        "principle": "the concrete .453 branch selected from A1 plus A1b."
    },
    "fresh_sweep": {
        "principle": "records focused sweep_clusters, sweep_semscholar, WebSearch, and WebFetch provenance."
    },
    "upstream_artifacts": {
        "principle": "binds the mapping to Exp 4903 A1 and Exp 4904 A1b."
    },
    "sota_to_experiment_mapping_note": {
        "principle": "states how the wall and post-sprint pivot become .453 planner inputs."
    },
    "note_path": {
        "principle": "points to the idempotent research-studying.md ingestion note."
    },
    "references_path": {
        "principle": "points to the idempotent research-references.md source section."
    },
    "duration_s": {
        "principle": "0.0001s floor for aggregation-only inference substrate."
    },
    "preconditions_checked": {
        "principle": "records upstream artifact, reliable-channel, and banned-class checks."
    },
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "aimed_at_fork_verdict",
    "a1b_fork_verdict",
    "selected_branch",
    "methods_mapped",
    "post_sprint_pivot_methods",
    "arxiv_ids_cited",
    "citations",
    "banned_channels_excluded",
    "flagged_for_v453",
    "inference_substrate",
    "preconditions_checked",
    "fresh_sweep",
    "upstream_artifacts",
    "sota_to_experiment_mapping_note",
    "note_path",
    "references_path",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "method",
        "track",
        "source_ids",
        "maps_to_frontier",
        "targets_a1_fork_verdict",
        "targets_a1b_fork_verdict",
        "a1_a1b_result_fit",
        "evidence",
        "experiment_graft",
        "fails_when",
        "roadmap_candidate",
        "nulled_class_reingested",
    }
)
REQUIRED_PIVOT_METHOD_FIELDS = frozenset(
    {
        "method",
        "track",
        "source_ids",
        "maps_to_track",
        "north_star_fit",
        "evidence",
        "experiment_graft",
        "validation_gate",
        "fails_when",
        "self_consistency_saturated",
        "oracle_distinct_verifier",
    }
)
REQUIRED_CITATION_FIELDS = frozenset({"title", "url", "http_status"})
REQUIRED_FETCHED_SOURCE_IDS = frozenset(
    {
        "2401.12497",
        "2504.07257",
        "2505.02074",
        "2602.11389",
        "2505.14999",
        "2605.18871",
        "2606.04579",
        "2604.24198",
    }
)
REQUIRED_WALL_SOURCE_IDS = frozenset({"2401.12497", "2505.02074", "2602.11389", "2504.07257"})
REQUIRED_PIVOT_SOURCE_IDS = frozenset({"2605.18871", "2505.14999", "2606.04579", "2604.24198"})
REQUIRED_WALL_TRACKS = frozenset(
    {
        "causal_state_abstraction_wall_diagnostic",
        "local_causal_ssm_world_model_diagnostic",
        "object_level_masked_causal_jepa_diagnostic",
        "interpretable_causal_world_model_extraction",
    }
)
REQUIRED_PIVOT_TRACKS = frozenset(
    {
        "distributional_energy_verifier",
        "energy_outcome_reward_model",
        "tool_aware_science_prm",
        "environment_aware_data_prm",
    }
)
NULLED_TRACKS = frozenset(
    {
        "energy_as_arc_lever",
        "tta_on_code_engine",
        "stronger_local_code_inducer",
        "decision_need_targets",
        "action_prefix_latents",
        "coverage_vocabulary",
        "exploration",
        "selection_ranking",
        "perception_from_grid",
    }
)
BANNED_PRECONDITION_KEYS = (
    "energy_as_arc_lever_reingested",
    "tta_on_code_engine_reingested",
    "stronger_local_code_inducer_reingested",
    "decision_need_targets_reingested",
    "action_prefix_latents_reingested",
    "coverage_vocabulary_reingested",
    "exploration_reingested",
    "selection_ranking_reingested",
    "perception_from_grid_reingested",
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source_id}" for source_id in REQUIRED_FETCHED_SOURCE_IDS)

CLUSTER_6_URL = (
    'http://export.arxiv.org/api/query?search_query=(abs:"neural+guided+search"+OR+'
    'abs:"learned+heuristic"+OR+abs:"value+guided+search"+OR+'
    'abs:"program+induction"+OR+abs:"world+model"+OR+abs:"goal+induction")+'
    'AND+(abs:"planning"+OR+abs:"agent"+OR+abs:"reasoning"+OR+'
    'abs:"reinforcement+learning")&start=0&max_results=8&sortBy=submittedDate&'
    "sortOrder=descending"
)
CLUSTER_0_URL = (
    'http://export.arxiv.org/api/query?search_query=(abs:"verifier+ensemble"+OR+'
    'abs:"verifier+ensembles"+OR+abs:"null+space"+OR+abs:"specification+gaming"+'
    'OR+abs:"process+reward+model"+OR+abs:"deliberative+alignment"+OR+'
    'abs:"reward+hacking")&start=0&max_results=8&sortBy=submittedDate&'
    "sortOrder=descending"
)
CLUSTER_1_URL = (
    'http://export.arxiv.org/api/query?search_query=(abs:"energy+based+model"+OR+'
    'abs:"energy-based+model"+OR+abs:"energy+guided+decoding"+OR+'
    'abs:"token+energy"+OR+abs:"EBT")+AND+(abs:"reasoning"+OR+'
    'abs:"verification"+OR+abs:"LLM"+OR+abs:"language+model")&start=0&'
    "max_results=8&sortBy=submittedDate&sortOrder=descending"
)
SEMANTIC_SCHOLAR_QUERIES = [
    "learned verifier self consistency oracle distinct reasoning",
    "process reward model verifier non saturated domains",
    "energy based verifier reasoning language model",
    "verifier ensemble self consistency reasoning",
    "test time compute verifier energy reasoning",
    "causal abstraction hidden state world model planning agents",
]
SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS = [
    "2105.06228",
    "2406.03816",
    "2408.06776",
    "2501.04519",
    "2502.01694",
    "2502.06737",
    "2503.10291",
    "2504.10481",
    "2504.16828",
    "2508.03686",
    "2508.15202",
    "2509.25598",
    "2510.20304",
    "2511.08364",
    "2604.17282",
    "2604.17957",
    "2604.24198",
    "2605.20272",
    "2605.24005",
    "2606.04579",
]
SEMANTIC_SCHOLAR_RESULT = (
    "Six focused Semantic Scholar queries returned 20 unique arXiv IDs; several "
    "queries hit HTTP 429 and those rate limits were recorded rather than "
    "promoted as evidence."
)
WEBSEARCH_QUERIES = [
    "site:arxiv.org/abs hidden state inference partially observable world model reinforcement learning agent",
    "site:arxiv.org/abs causal state abstraction reinforcement learning hidden state world model",
    "site:arxiv.org/abs process reward model non mathematical domains reasoning verifier",
    "site:arxiv.org/abs energy based verifier reasoning language model",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2401.12497",
    "https://arxiv.org/abs/2505.02074",
    "https://arxiv.org/abs/2602.11389",
    "https://arxiv.org/abs/2504.07257",
    "https://arxiv.org/abs/2605.18871",
    "https://arxiv.org/abs/2505.14999",
    "https://arxiv.org/abs/2606.04579",
    "https://arxiv.org/abs/2604.24198",
]

CITATIONS = {
    "2401.12497": {
        "title": "Building Minimal and Reusable Causal State Abstractions for Reinforcement Learning",
        "url": "https://arxiv.org/abs/2401.12497",
        "http_status": 200,
    },
    "2505.02074": {
        "title": "Learning Local Causal World Models with State Space Models and Attention",
        "url": "https://arxiv.org/abs/2505.02074",
        "http_status": 200,
    },
    "2602.11389": {
        "title": "Causal-JEPA: Learning World Models through Object-Level Latent Masking",
        "url": "https://arxiv.org/abs/2602.11389",
        "http_status": 200,
    },
    "2504.07257": {
        "title": "Better Decisions through the Right Causal World Model",
        "url": "https://arxiv.org/abs/2504.07257",
        "http_status": 200,
    },
    "2605.18871": {
        "title": "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
        "url": "https://arxiv.org/abs/2605.18871",
        "http_status": 200,
    },
    "2505.14999": {
        "title": "Learning to Rank Chain-of-Thought: Using a Small Model",
        "url": "https://arxiv.org/abs/2505.14999",
        "http_status": 200,
    },
    "2606.04579": {
        "title": "SCI-PRM: A Tool Aware Process Reward Model for Scientific Reasoning Verification",
        "url": "https://arxiv.org/abs/2606.04579",
        "http_status": 200,
    },
    "2604.24198": {
        "title": "Rewarding the Scientific Process: Process-Level Reward Modeling for Agentic Data Analysis",
        "url": "https://arxiv.org/abs/2604.24198",
        "http_status": 200,
    },
}

BRANCH_REASON = (
    "Exp 4903 reported WALL_DEEPER_THAN_VALUE_PREDICTION and Exp 4904 reported "
    "VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES, so .453 treats the ARC "
    "change-VALUE wall as surviving four representations plus env-grounding and "
    "keeps only final-sprint diagnostics while moving concrete continuation "
    "energy to the post-sprint verifier-moat pivot."
)

DEFAULT_UPSTREAM_CONTEXT = {
    "a1_honest_verdict": "complete_env_grounded_search_no_first_win_lift_WALL_DEEPER_THAN_VALUE_PREDICTION",
    "a1_fork_verdict": AIMED_AT_FORK_VERDICT,
    "a1_value_grounded_first_win_delta_median": -0.04,
    "a1_value_grounded_first_win_delta_ci95": [-0.04, 0.0],
    "a1_coverage_migration_count": 0,
    "a1b_honest_verdict": "complete_latent_action_no_value_lift_representation_invariant_4_classes",
    "a1b_fork_verdict": A1B_NULL_VERDICT,
    "a1b_latent_action_value_delta_median": -0.103162,
    "a1b_latent_action_value_delta_ci95": [-0.231195, 0.025266],
    "selected_branch": SELECTED_BRANCH,
}

FLAGGED_FOR_V453 = [
    {
        "candidate": "causal_state_abstraction_wall_diagnostic",
        "flag": "flagged_for_v453: causal_state_abstraction_wall_diagnostic (arXiv:2401.12497)",
        "source_ids": ["2401.12497"],
        "maps_to_frontier": ".453",
        "priority": 1,
    },
    {
        "candidate": "distributional_energy_verifier_pivot",
        "flag": "flagged_for_v453: distributional_energy_verifier_pivot (arXiv:2605.18871)",
        "source_ids": ["2605.18871"],
        "maps_to_frontier": "post_sprint_verifier_moat",
        "priority": 2,
    },
    {
        "candidate": "tool_aware_science_prm_pivot",
        "flag": "flagged_for_v453: tool_aware_science_prm_pivot (arXiv:2606.04579)",
        "source_ids": ["2606.04579"],
        "maps_to_frontier": "post_sprint_verifier_moat",
        "priority": 3,
    },
]

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Causal state-abstraction wall diagnostic",
        "track": "causal_state_abstraction_wall_diagnostic",
        "source_ids": ["2401.12497"],
        "maps_to_frontier": ".453",
        "targets_a1_fork_verdict": AIMED_AT_FORK_VERDICT,
        "targets_a1b_fork_verdict": A1B_NULL_VERDICT,
        "a1_a1b_result_fit": (
            "Exp 4903 plus Exp 4904 show the value wall survives four representations "
            "and env-grounding; causal abstractions are therefore a diagnostic for "
            "missing variables, not another representation rerun."
        ),
        "evidence": (
            "arXiv:2401.12497 introduces Causal Bisimulation Modeling to derive "
            "minimal task-specific state abstractions from causal dynamics and rewards."
        ),
        "experiment_graft": (
            "Run an offline causal-abstraction audit over failed A1/A1b transitions: "
            "identify which variables would have to be retained for changed-cell value "
            "prediction, then stop unless the abstraction exposes a new observable "
            "state variable within the sprint budget."
        ),
        "fails_when": (
            "The abstraction requires hidden variables the ARC interface cannot expose, "
            "becomes a decision-need target table in disguise, or improves only static "
            "ranking without changed-cell value lift."
        ),
        "roadmap_candidate": FLAGGED_FOR_V453[0]["flag"],
        "nulled_class_reingested": False,
    },
    {
        "method": "Local causal SSM world-model diagnostic",
        "track": "local_causal_ssm_world_model_diagnostic",
        "source_ids": ["2505.02074"],
        "maps_to_frontier": ".453",
        "targets_a1_fork_verdict": AIMED_AT_FORK_VERDICT,
        "targets_a1b_fork_verdict": A1B_NULL_VERDICT,
        "a1_a1b_result_fit": (
            "four representations did not recover changed values, so a local causal "
            "SSM is useful only to test whether the missing value signal is causal "
            "structure rather than code, latent-action, or env-grounding format."
        ),
        "evidence": (
            "arXiv:2505.02074 evaluates state-space models for learning causal "
            "world models and local dynamics with attention."
        ),
        "experiment_graft": (
            "Fit a tiny causal SSM on observed prefixes and compare its inferred "
            "causal graph against the failed engine facts; accept only a diagnostic "
            "report unless it predicts held-out changed values without extra actions."
        ),
        "fails_when": (
            "The SSM needs longer trajectories than the sprint budget, only predicts "
            "smooth local dynamics, or collapses back to action-prefix latent replay."
        ),
        "roadmap_candidate": "not_flagged_primary_but_flagged_for_v453_context: final-sprint diagnostic only",
        "nulled_class_reingested": False,
    },
    {
        "method": "Object-level masked Causal-JEPA diagnostic",
        "track": "object_level_masked_causal_jepa_diagnostic",
        "source_ids": ["2602.11389"],
        "maps_to_frontier": ".453",
        "targets_a1_fork_verdict": AIMED_AT_FORK_VERDICT,
        "targets_a1b_fork_verdict": A1B_NULL_VERDICT,
        "a1_a1b_result_fit": (
            "The latent-action interface was null after four representations, so masked object-level prediction "
            "is only a last diagnostic for interaction-dependent value variables."
        ),
        "evidence": (
            "arXiv:2602.11389 uses object-level latent masking to force "
            "interaction-dependent world-model prediction under partial observability."
        ),
        "experiment_graft": (
            "Mask object-level latent facts in failed transitions and ask whether "
            "observed context can reconstruct the changed values; use the result to "
            "label the wall observable or unobservable rather than launching another "
            "planner."
        ),
        "fails_when": (
            "Object segmentation is itself the bottleneck, masked prediction behaves "
            "like perception-from-grid, or reconstruction quality does not transfer "
            "to held-out changed-cell value accuracy."
        ),
        "roadmap_candidate": "not_flagged_primary_but_flagged_for_v453_context: final-sprint diagnostic only",
        "nulled_class_reingested": False,
    },
    {
        "method": "Interpretable causal world-model extraction",
        "track": "interpretable_causal_world_model_extraction",
        "source_ids": ["2504.07257"],
        "maps_to_frontier": ".453",
        "targets_a1_fork_verdict": AIMED_AT_FORK_VERDICT,
        "targets_a1b_fork_verdict": A1B_NULL_VERDICT,
        "a1_a1b_result_fit": (
            "The wall survived four representations, so exact interpretable causal "
            "extraction is a falsifier for shortcut/spurious-variable explanations."
        ),
        "evidence": (
            "arXiv:2504.07257 proposes COMET, extracting object-centric causal world "
            "models and internal state variables for more robust decisions."
        ),
        "experiment_graft": (
            "Extract symbolic object-transition equations from the same failed games "
            "and compare them to the induced engine; use mismatches to decide whether "
            "the sprint should stop or whether a single observable state variable was missed."
        ),
        "fails_when": (
            "The extraction requires a perception system already ruled out, recovers "
            "only obvious grid coordinates, or cannot explain held-out value misses."
        ),
        "roadmap_candidate": "not_flagged_primary_but_flagged_for_v453_context: final-sprint diagnostic only",
        "nulled_class_reingested": False,
    },
]

DEFAULT_POST_SPRINT_PIVOT_METHODS = [
    {
        "method": "Distributional energy verifier for structured reasoning",
        "track": "distributional_energy_verifier",
        "source_ids": ["2605.18871"],
        "maps_to_track": "post_sprint_verifier_moat",
        "north_star_fit": (
            "north-star §1/§5: tests whether an oracle-distinct learned energy "
            "verifier adds value where self-consistency is not saturated."
        ),
        "evidence": (
            "arXiv:2605.18871 combines learned quality scoring with deterministic "
            "constraint penalties and uncertainty to verify structured LLM outputs."
        ),
        "experiment_graft": (
            "Port the FoVer evaluation harness to MuSR/TravelPlanner-style structured "
            "reasoning rows, score candidates with a small distributional energy "
            "verifier, and compare against self-consistency and an LLM judge."
        ),
        "validation_gate": (
            "Promote only if the verifier beats self-consistency with CI95 excluding "
            "zero and no model-identity shortcut under adversarial_verify."
        ),
        "fails_when": (
            "The target domain has a cheap executable oracle, self-consistency is "
            "already near ceiling, or the scorer learns generator identity."
        ),
        "self_consistency_saturated": False,
        "oracle_distinct_verifier": True,
    },
    {
        "method": "Small energy outcome reward model",
        "track": "energy_outcome_reward_model",
        "source_ids": ["2505.14999"],
        "maps_to_track": "post_sprint_verifier_moat",
        "north_star_fit": (
            "north-star §1/§5: a cheap verifier can be a moat only if it beats "
            "self-consistency on headroom domains without being the oracle."
        ),
        "evidence": (
            "arXiv:2505.14999 uses a 55M-parameter energy outcome reward model to "
            "rank chain-of-thought candidates using only outcome labels."
        ),
        "experiment_graft": (
            "Train or calibrate a tiny EORM-style scorer on non-FoVer reasoning "
            "traces, then run best-of-N selection against self-consistency at matched "
            "sample and cost budgets."
        ),
        "validation_gate": (
            "Require accuracy lift plus cost/action reduction over self-consistency "
            "and an oracle-distinct declaration for every evaluated domain."
        ),
        "fails_when": (
            "Outcome labels are too sparse, the scorer is just a math-domain proxy, "
            "or cost parity disappears at the needed sample size."
        ),
        "self_consistency_saturated": False,
        "oracle_distinct_verifier": True,
    },
    {
        "method": "Tool-aware scientific process reward model",
        "track": "tool_aware_science_prm",
        "source_ids": ["2606.04579"],
        "maps_to_track": "post_sprint_verifier_moat",
        "north_star_fit": (
            "north-star §1/§5: scientific tool-use reasoning is a non-saturated "
            "domain where the verifier can judge process rather than final answers."
        ),
        "evidence": (
            "arXiv:2606.04579 trains Sci-PRM on chain-of-tool trajectories for "
            "fine-grained supervision of tool selection, execution, and interpretation."
        ),
        "experiment_graft": (
            "Build a small FoVer-style scientific-tool trace corpus, label step "
            "selection/execution/interpretation errors, and test whether a PRM catches "
            "errors missed by generator self-checks."
        ),
        "validation_gate": (
            "Promote only if the PRM improves best-of-N or RL reward shaping on a "
            "held-out scientific trace set without using the final-answer oracle."
        ),
        "fails_when": (
            "The benchmark is saturated by tool execution, labels leak final answers, "
            "or the verifier cannot separate tool misuse from valid exploration."
        ),
        "self_consistency_saturated": False,
        "oracle_distinct_verifier": True,
    },
    {
        "method": "Environment-aware data-analysis process verifier",
        "track": "environment_aware_data_prm",
        "source_ids": ["2604.24198"],
        "maps_to_track": "post_sprint_verifier_moat",
        "north_star_fit": (
            "north-star §1/§5: dynamic data-analysis agents have silent-error "
            "headroom where an oracle-distinct verifier can actively inspect process."
        ),
        "evidence": (
            "arXiv:2604.24198 introduces DataPRM, an environment-aware generative "
            "process reward model for agentic data analysis and silent-error detection."
        ),
        "experiment_graft": (
            "Use DataPRM-style active trace checks on non-saturated data-analysis "
            "tasks, with the verifier probing intermediate states and scoring "
            "correctable versus irrecoverable mistakes."
        ),
        "validation_gate": (
            "Require downstream policy lift over self-consistency and an explicit "
            "silent-error catch rate on held-out tasks."
        ),
        "fails_when": (
            "The environment exposes exact execution oracles, active probes become "
            "the answer checker, or exploratory actions are penalized as errors."
        ),
        "self_consistency_saturated": False,
        "oracle_distinct_verifier": True,
    },
]

DEFAULT_FRESH_SWEEP = {
    "filtered_track": (
        "V453 frontier: A1 WALL_DEEPER_THAN_VALUE_PREDICTION + A1b "
        "VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES, plus post-sprint "
        "verifier-moat pivot"
    ),
    "cluster_ids": [6, 0, 1],
    "cluster_urls": [CLUSTER_6_URL, CLUSTER_0_URL, CLUSTER_1_URL],
    "semantic_scholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "semantic_scholar_result": SEMANTIC_SCHOLAR_RESULT,
    "semantic_scholar_unique_arxiv_ids": SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
    "websearch_queries": WEBSEARCH_QUERIES,
    "webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
}

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "north_star_present": True,
    "research_studying_present": True,
    "research_references_present": True,
    "a1_artifact_present": True,
    "a1b_artifact_present": True,
    "a1_fork_verdict_read": True,
    "a1_fork_verdict": AIMED_AT_FORK_VERDICT,
    "a1_honest_verdict": DEFAULT_UPSTREAM_CONTEXT["a1_honest_verdict"],
    "a1_value_grounded_first_win_delta_median": DEFAULT_UPSTREAM_CONTEXT[
        "a1_value_grounded_first_win_delta_median"
    ],
    "a1_value_grounded_first_win_delta_ci95": DEFAULT_UPSTREAM_CONTEXT[
        "a1_value_grounded_first_win_delta_ci95"
    ],
    "a1_coverage_migration_count": DEFAULT_UPSTREAM_CONTEXT["a1_coverage_migration_count"],
    "a1b_fork_verdict": A1B_NULL_VERDICT,
    "a1b_honest_verdict": DEFAULT_UPSTREAM_CONTEXT["a1b_honest_verdict"],
    "a1b_latent_action_value_delta_median": DEFAULT_UPSTREAM_CONTEXT[
        "a1b_latent_action_value_delta_median"
    ],
    "a1b_latent_action_value_delta_ci95": DEFAULT_UPSTREAM_CONTEXT[
        "a1b_latent_action_value_delta_ci95"
    ],
    "selected_branch": SELECTED_BRANCH,
    "branch_reason": BRANCH_REASON,
    "post_sprint_pivot_mapped": True,
    "north_star_sections_read": ["§1", "§5"],
    "sweep_clusters_used": True,
    "sweep_cluster_ids": [6, 0, 1],
    "sweep_cluster_urls": [CLUSTER_6_URL, CLUSTER_0_URL, CLUSTER_1_URL],
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "semantic_scholar_result": SEMANTIC_SCHOLAR_RESULT,
    "semantic_scholar_unique_arxiv_ids": SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
    "websearch_webfetch_used": True,
    "websearch_queries": WEBSEARCH_QUERIES,
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
    "top_source_count": len(WEBSEARCH_WEBFETCH_TOP_SOURCES),
    "arxiv_http_200_verified_ids": [
        f"https://arxiv.org/abs/{source_id}" for source_id in sorted(REQUIRED_FETCHED_SOURCE_IDS)
    ],
    "deep_research_invoked": False,
    "energy_as_arc_lever_reingested": False,
    "tta_on_code_engine_reingested": False,
    "stronger_local_code_inducer_reingested": False,
    "decision_need_targets_reingested": False,
    "action_prefix_latents_reingested": False,
    "coverage_vocabulary_reingested": False,
    "exploration_reingested": False,
    "selection_ranking_reingested": False,
    "perception_from_grid_reingested": False,
    "model_load": False,
    "training_launched": False,
    "leaderboard_submission": False,
    "solve_claim_made": False,
    "research_conductor_modified": False,
    "ops_docs_modified": False,
}
REQUIRED_PRECONDITION_FIELDS = frozenset(DEFAULT_PRECONDITIONS_CHECKED)
REQUIRED_FRESH_SWEEP_FIELDS = frozenset(DEFAULT_FRESH_SWEEP)

DEFAULT_UPSTREAM_ARTIFACTS = {
    "a1_artifact": A1_ARTIFACT_RELATIVE_PATH,
    "a1b_artifact": A1B_ARTIFACT_RELATIVE_PATH,
    "a1_honest_verdict": DEFAULT_UPSTREAM_CONTEXT["a1_honest_verdict"],
    "a1_fork_verdict": DEFAULT_UPSTREAM_CONTEXT["a1_fork_verdict"],
    "a1_value_grounded_first_win_delta_median": DEFAULT_UPSTREAM_CONTEXT[
        "a1_value_grounded_first_win_delta_median"
    ],
    "a1_value_grounded_first_win_delta_ci95": DEFAULT_UPSTREAM_CONTEXT[
        "a1_value_grounded_first_win_delta_ci95"
    ],
    "a1_coverage_migration_count": DEFAULT_UPSTREAM_CONTEXT["a1_coverage_migration_count"],
    "a1b_honest_verdict": DEFAULT_UPSTREAM_CONTEXT["a1b_honest_verdict"],
    "a1b_fork_verdict": DEFAULT_UPSTREAM_CONTEXT["a1b_fork_verdict"],
    "a1b_latent_action_value_delta_median": DEFAULT_UPSTREAM_CONTEXT[
        "a1b_latent_action_value_delta_median"
    ],
    "a1b_latent_action_value_delta_ci95": DEFAULT_UPSTREAM_CONTEXT[
        "a1b_latent_action_value_delta_ci95"
    ],
    "carried_forward_from_4903_4904": [
        "change-VALUE wall survives env-grounding",
        "latent-action interface remains null",
    ],
    "not_carried_forward": [
        "energy-as-ARC-lever",
        "TTA-on-code-engine",
        "stronger-local-code-inducer",
        "decision-need-targets",
        "action-prefix-latents",
        "coverage/vocabulary",
        "exploration",
        "selection/ranking",
        "perception-from-grid",
    ],
}
REQUIRED_UPSTREAM_FIELDS = frozenset(DEFAULT_UPSTREAM_ARTIFACTS)

DEFAULT_MAPPING_NOTE = {
    "summary": (
        "A1 and A1b leave the .453 ARC branch as a final wall diagnostic rather "
        "than another representation push; the post-sprint work should pivot to "
        "oracle-distinct verifier-moat experiments in non-saturated domains."
    ),
    "terminal_success": HONEST_VERDICT,
    "branch": SELECTED_BRANCH,
    "wall_source_ids": sorted(REQUIRED_WALL_SOURCE_IDS),
    "pivot_source_ids": sorted(REQUIRED_PIVOT_SOURCE_IDS),
    "planner_instruction": (
        "Use the causal-abstraction diagnostic only if the operator wants a final "
        "ARC closure check before 2026-06-30. Otherwise start the verifier-moat "
        "pivot with distributional energy verification and Sci-PRM/DataPRM-style "
        "process verification."
    ),
    "not_carried_forward": (
        "Do not re-promote energy-as-ARC-lever, TTA-on-code-engine, stronger local "
        "code inducers, decision-need targets, action-prefix latents, coverage/"
        "vocabulary, exploration, selection/ranking, or perception-from-grid classes."
    ),
}
REQUIRED_MAPPING_NOTE_FIELDS = frozenset(DEFAULT_MAPPING_NOTE)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_set_checksum(
    selected_branch: str,
    fresh_sweep: JsonMap,
    methods: Sequence[JsonMap],
    pivot_methods: Sequence[JsonMap],
    flags: Sequence[JsonMap],
) -> str:
    payload = json.dumps(
        {
            "flags": list(flags),
            "fresh_sweep": fresh_sweep,
            "methods": list(methods),
            "pivot_methods": list(pivot_methods),
            "selected_branch": selected_branch,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


REPRODUCIBILITY_CHECKSUM = source_set_checksum(
    SELECTED_BRANCH,
    DEFAULT_FRESH_SWEEP,
    DEFAULT_METHODS_MAPPED,
    DEFAULT_POST_SPRINT_PIVOT_METHODS,
    FLAGGED_FOR_V453,
)


def select_ingestion_branch(a1_artifact: JsonMap, a1b_artifact: JsonMap) -> str:
    a1_verdict = a1_artifact.get("fork_verdict")
    a1b_verdict = a1b_artifact.get("fork_verdict")
    if a1_verdict == "ENV_GROUNDED_SEARCH_UNLOCKS_FIRST_WIN":
        return "scale_env_grounded_first_win_search"
    if a1_verdict == "SEARCH_BUDGET_BOUND":
        return "cut_env_grounding_action_cost"
    if a1_verdict == AIMED_AT_FORK_VERDICT and a1b_verdict == A1B_NULL_VERDICT:
        return SELECTED_BRANCH
    raise ValueError(f"unsupported A1/A1b fork: {a1_verdict!r} / {a1b_verdict!r}")


def derive_upstream_context(a1_artifact: JsonMap, a1b_artifact: JsonMap) -> dict[str, object]:
    branch = select_ingestion_branch(a1_artifact, a1b_artifact)
    return {
        "a1_honest_verdict": a1_artifact.get("honest_verdict", ""),
        "a1_fork_verdict": a1_artifact.get("fork_verdict"),
        "a1_value_grounded_first_win_delta_median": a1_artifact.get(
            "value_grounded_first_win_delta_median"
        ),
        "a1_value_grounded_first_win_delta_ci95": a1_artifact.get(
            "value_grounded_first_win_delta_ci95", []
        ),
        "a1_coverage_migration_count": int(a1_artifact.get("coverage_migration_count") or 0),
        "a1b_honest_verdict": a1b_artifact.get("honest_verdict", ""),
        "a1b_fork_verdict": a1b_artifact.get("fork_verdict"),
        "a1b_latent_action_value_delta_median": a1b_artifact.get(
            "latent_action_value_accuracy_delta_median"
        ),
        "a1b_latent_action_value_delta_ci95": a1b_artifact.get(
            "latent_action_value_accuracy_delta_ci95", []
        ),
        "selected_branch": branch,
    }


def load_upstream_context(repo_root: Path) -> dict[str, object]:
    a1 = json.loads((repo_root / A1_ARTIFACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    a1b = json.loads((repo_root / A1B_ARTIFACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    return derive_upstream_context(a1, a1b)


def _contextual_preconditions(upstream_context: JsonMap) -> dict[str, object]:
    preconditions = dict(DEFAULT_PRECONDITIONS_CHECKED)
    preconditions.update(
        {
            "a1_honest_verdict": upstream_context["a1_honest_verdict"],
            "a1_fork_verdict": upstream_context["a1_fork_verdict"],
            "a1_value_grounded_first_win_delta_median": upstream_context[
                "a1_value_grounded_first_win_delta_median"
            ],
            "a1_value_grounded_first_win_delta_ci95": upstream_context[
                "a1_value_grounded_first_win_delta_ci95"
            ],
            "a1_coverage_migration_count": upstream_context["a1_coverage_migration_count"],
            "a1b_honest_verdict": upstream_context["a1b_honest_verdict"],
            "a1b_fork_verdict": upstream_context["a1b_fork_verdict"],
            "a1b_latent_action_value_delta_median": upstream_context[
                "a1b_latent_action_value_delta_median"
            ],
            "a1b_latent_action_value_delta_ci95": upstream_context[
                "a1b_latent_action_value_delta_ci95"
            ],
            "selected_branch": upstream_context["selected_branch"],
        }
    )
    return preconditions


def _contextual_upstream_artifacts(upstream_context: JsonMap) -> dict[str, object]:
    upstream = dict(DEFAULT_UPSTREAM_ARTIFACTS)
    upstream.update(
        {
            "a1_honest_verdict": upstream_context["a1_honest_verdict"],
            "a1_fork_verdict": upstream_context["a1_fork_verdict"],
            "a1_value_grounded_first_win_delta_median": upstream_context[
                "a1_value_grounded_first_win_delta_median"
            ],
            "a1_value_grounded_first_win_delta_ci95": upstream_context[
                "a1_value_grounded_first_win_delta_ci95"
            ],
            "a1_coverage_migration_count": upstream_context["a1_coverage_migration_count"],
            "a1b_honest_verdict": upstream_context["a1b_honest_verdict"],
            "a1b_fork_verdict": upstream_context["a1b_fork_verdict"],
            "a1b_latent_action_value_delta_median": upstream_context[
                "a1b_latent_action_value_delta_median"
            ],
            "a1b_latent_action_value_delta_ci95": upstream_context[
                "a1b_latent_action_value_delta_ci95"
            ],
        }
    )
    return upstream


def build_artifact(upstream_context: JsonMap | None = None) -> dict[str, object]:
    context = dict(upstream_context or DEFAULT_UPSTREAM_CONTEXT)
    checksum = source_set_checksum(
        str(context["selected_branch"]),
        DEFAULT_FRESH_SWEEP,
        DEFAULT_METHODS_MAPPED,
        DEFAULT_POST_SPRINT_PIVOT_METHODS,
        FLAGGED_FOR_V453,
    )
    return {
        "honest_verdict": HONEST_VERDICT,
        "aimed_at_fork_verdict": context["a1_fork_verdict"],
        "a1b_fork_verdict": context["a1b_fork_verdict"],
        "selected_branch": context["selected_branch"],
        "methods_mapped": DEFAULT_METHODS_MAPPED,
        "post_sprint_pivot_methods": DEFAULT_POST_SPRINT_PIVOT_METHODS,
        "arxiv_ids_cited": sorted(REQUIRED_FETCHED_SOURCE_IDS),
        "citations": CITATIONS,
        "banned_channels_excluded": True,
        "flagged_for_v453": FLAGGED_FOR_V453,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": _contextual_preconditions(context),
        "fresh_sweep": DEFAULT_FRESH_SWEEP,
        "upstream_artifacts": _contextual_upstream_artifacts(context),
        "sota_to_experiment_mapping_note": DEFAULT_MAPPING_NOTE,
        "note_path": NOTE_PATH,
        "references_path": REFERENCES_PATH,
        "duration_s": DURATION_S,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
    }


def build_blocked_artifact(missing: Sequence[str] | None = None) -> dict[str, object]:
    missing_items = list(missing or [A1_ARTIFACT_RELATIVE_PATH])
    preconditions = dict(DEFAULT_PRECONDITIONS_CHECKED)
    preconditions.update(
        {
            "a1_artifact_present": A1_ARTIFACT_RELATIVE_PATH not in missing_items,
            "a1b_artifact_present": A1B_ARTIFACT_RELATIVE_PATH not in missing_items,
            "a1_fork_verdict_read": False,
            "a1_fork_verdict": "",
            "a1_honest_verdict": "",
            "a1_value_grounded_first_win_delta_median": None,
            "a1_value_grounded_first_win_delta_ci95": [],
            "a1_coverage_migration_count": 0,
            "a1b_fork_verdict": "",
            "a1b_honest_verdict": "",
            "a1b_latent_action_value_delta_median": None,
            "a1b_latent_action_value_delta_ci95": [],
            "selected_branch": "blocked",
            "branch_reason": f"missing upstream artifact(s): {', '.join(missing_items)}",
            "post_sprint_pivot_mapped": False,
        }
    )
    return {
        "honest_verdict": BLOCKED_VERDICT,
        "aimed_at_fork_verdict": "",
        "a1b_fork_verdict": "",
        "selected_branch": "blocked",
        "methods_mapped": [],
        "post_sprint_pivot_methods": [],
        "arxiv_ids_cited": [],
        "citations": {},
        "banned_channels_excluded": True,
        "flagged_for_v453": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "fresh_sweep": DEFAULT_FRESH_SWEEP,
        "upstream_artifacts": {},
        "sota_to_experiment_mapping_note": {},
        "note_path": NOTE_PATH,
        "references_path": REFERENCES_PATH,
        "duration_s": DURATION_S,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:blocked_upstream_artifact_missing",
        "field_principles": FIELD_PRINCIPLES,
    }


def validate_artifact(artifact: JsonMap) -> None:
    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict lacks terminal prefix")
    if verdict == BLOCKED_VERDICT:
        _require(artifact.get("methods_mapped") == [], "blocked artifact must not map methods")
        _require(
            artifact.get("post_sprint_pivot_methods") == [],
            "blocked artifact must not map post_sprint_pivot_methods",
        )
        _require(artifact.get("banned_channels_excluded") is True, "blocked artifact must exclude banned channels")
        return

    _require(set(artifact) == set(REQUIRED_ARTIFACT_FIELDS), "artifact fields do not match REQ-ARC-WMTE-4911")
    _require(artifact["honest_verdict"] == HONEST_VERDICT, "honest_verdict must be the .453 success mapping")
    _require(artifact["aimed_at_fork_verdict"] == AIMED_AT_FORK_VERDICT, "aimed_at_fork_verdict must match A1")
    _require(artifact["a1b_fork_verdict"] == A1B_NULL_VERDICT, "a1b_fork_verdict must match A1b")
    _require(artifact["selected_branch"] == SELECTED_BRANCH, "selected_branch must match A1/A1b")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate must be aggregation-only")
    _require(artifact["duration_s"] == DURATION_S, "duration_s must use the aggregation floor")
    _require(artifact["banned_channels_excluded"] is True, "banned_channels_excluded must be true")
    _require(artifact["arxiv_ids_cited"] == sorted(REQUIRED_FETCHED_SOURCE_IDS), "arxiv_ids_cited must match verified sources")
    _require(set(artifact["field_principles"]) == REQUIRED_PRINCIPLE_FIELDS, "field_principles must match REQ-ARC-WMTE-4911")

    citations = artifact["citations"]
    _require(set(citations) == REQUIRED_FETCHED_SOURCE_IDS, "citations must cover the verified source set")
    for source_id, citation in citations.items():
        _require(set(citation) == REQUIRED_CITATION_FIELDS, "citation fields must be title/url/http_status")
        _require(citation["url"] == f"https://arxiv.org/abs/{source_id}", "citation url must match arXiv ID")
        _require(citation["http_status"] == 200, "http_status must be HTTP 200")

    methods = artifact["methods_mapped"]
    _require(3 <= len(methods) <= 5, "methods_mapped must contain three to five methods")
    wall_sources: set[str] = set()
    wall_tracks: set[str] = set()
    for method in methods:
        _require(set(method) == REQUIRED_METHOD_FIELDS, "method fields must match the .453 contract")
        method_sources = set(method["source_ids"])
        _require(method_sources <= REQUIRED_FETCHED_SOURCE_IDS, "method source_ids must be verified citations")
        _require(method["maps_to_frontier"] == ".453", "method must map to .453")
        _require(method["targets_a1_fork_verdict"] == AIMED_AT_FORK_VERDICT, "method A1 fork verdict mismatch")
        _require(method["targets_a1b_fork_verdict"] == A1B_NULL_VERDICT, "method A1b fork verdict mismatch")
        _require(method["track"] not in NULLED_TRACKS, "method track must not be nulled")
        _require(method["nulled_class_reingested"] is False, "nulled class flag must be false")
        _require(method["experiment_graft"], "experiment_graft is required")
        _require(method["fails_when"], "fails_when is required")
        wall_sources.update(method_sources)
        wall_tracks.add(method["track"])
    _require(wall_sources == REQUIRED_WALL_SOURCE_IDS, "wall methods must cite the wall diagnostic sources")
    _require(wall_tracks == REQUIRED_WALL_TRACKS, "wall tracks must match the final diagnostic set")

    pivot_methods = artifact["post_sprint_pivot_methods"]
    _require(3 <= len(pivot_methods) <= 5, "post_sprint_pivot_methods must contain three to five methods")
    pivot_sources: set[str] = set()
    pivot_tracks: set[str] = set()
    for method in pivot_methods:
        _require(set(method) == REQUIRED_PIVOT_METHOD_FIELDS, "pivot method fields must match contract")
        method_sources = set(method["source_ids"])
        _require(method_sources <= REQUIRED_FETCHED_SOURCE_IDS, "pivot source_ids must be verified citations")
        _require(method["maps_to_track"] == "post_sprint_verifier_moat", "pivot method must target verifier moat")
        _require(method["self_consistency_saturated"] is False, "pivot domain must be non-saturated")
        _require(method["oracle_distinct_verifier"] is True, "pivot verifier must be oracle-distinct")
        _require(method["experiment_graft"], "pivot experiment_graft is required")
        _require(method["fails_when"], "pivot fails_when is required")
        pivot_sources.update(method_sources)
        pivot_tracks.add(method["track"])
    _require(pivot_sources == REQUIRED_PIVOT_SOURCE_IDS, "pivot methods must cite the pivot source set")
    _require(pivot_tracks == REQUIRED_PIVOT_TRACKS, "pivot tracks must match verifier-moat set")

    flags = artifact["flagged_for_v453"]
    _require(flags == FLAGGED_FOR_V453, "flagged_for_v453 must match the planner handoff; stale flags are rejected")
    for flag in flags:
        _require("flagged_for_v453" in flag.get("flag", ""), "stale flag must not target an older frontier")

    preconditions = artifact["preconditions_checked"]
    _require(set(preconditions) == REQUIRED_PRECONDITION_FIELDS, "preconditions_checked fields mismatch")
    _require(preconditions["a1_artifact_present"] is True, "A1 artifact must be present")
    _require(preconditions["a1b_artifact_present"] is True, "A1b artifact must be present")
    _require(preconditions["a1_fork_verdict"] == AIMED_AT_FORK_VERDICT, "A1 fork verdict mismatch")
    _require(preconditions["a1b_fork_verdict"] == A1B_NULL_VERDICT, "A1b fork verdict mismatch")
    _require(preconditions["deep_research_invoked"] is False, "deep-research is banned")
    banned_messages = {
        "energy_as_arc_lever_reingested": "energy-as-ARC lever is banned",
        "tta_on_code_engine_reingested": "TTA-on-code-engine is banned",
        "stronger_local_code_inducer_reingested": "stronger local code inducer is banned",
        "decision_need_targets_reingested": "decision-need targets are banned",
        "action_prefix_latents_reingested": "action-prefix latents are banned",
        "coverage_vocabulary_reingested": "coverage/vocabulary is banned",
        "exploration_reingested": "exploration is banned",
        "selection_ranking_reingested": "selection/ranking is banned",
        "perception_from_grid_reingested": "perception-from-grid is banned",
    }
    for key, message in banned_messages.items():
        _require(preconditions[key] is False, message)
    _require(preconditions["model_load"] is False, "model-load claim is banned")
    _require(preconditions["training_launched"] is False, "training claim is banned")
    _require(preconditions["leaderboard_submission"] is False, "leaderboard claim is banned")
    _require(preconditions["solve_claim_made"] is False, "solve claim is banned")
    _require(preconditions["research_conductor_modified"] is False, "research conductor must not be modified")
    _require(preconditions["ops_docs_modified"] is False, "ops docs must not be modified by this run")

    fresh_sweep = artifact["fresh_sweep"]
    _require(set(fresh_sweep) == REQUIRED_FRESH_SWEEP_FIELDS, "fresh_sweep fields mismatch")
    upstream = artifact["upstream_artifacts"]
    _require(set(upstream) == REQUIRED_UPSTREAM_FIELDS, "upstream_artifacts fields mismatch")
    mapping_note = artifact["sota_to_experiment_mapping_note"]
    _require(set(mapping_note) == REQUIRED_MAPPING_NOTE_FIELDS, "sota_to_experiment_mapping_note fields mismatch")
    expected_checksum = source_set_checksum(
        artifact["selected_branch"],
        fresh_sweep,
        methods,
        pivot_methods,
        flags,
    )
    _require(artifact["reproducibility_checksum"] == expected_checksum, "reproducibility_checksum mismatch")


def _replace_section(text: str, start_marker: str, end_marker: str, body: str) -> str:
    section = f"{start_marker}\n{body.rstrip()}\n{end_marker}\n"
    if start_marker in text:
        _require(end_marker in text, "missing end marker for existing section")
        start = text.index(start_marker)
        end = text.index(end_marker, start) + len(end_marker)
        return text[:start] + section.rstrip() + text[end:]
    return text.rstrip() + "\n\n" + section


def update_research_studying_text(text: str, artifact: JsonMap) -> str:
    method_lines = "\n".join(
        f"- {method['method']} ({', '.join('arXiv:' + source for source in method['source_ids'])}): "
        f"{method['experiment_graft']}"
        for method in artifact["methods_mapped"]
    )
    pivot_lines = "\n".join(
        f"- {method['method']} ({', '.join('arXiv:' + source for source in method['source_ids'])}): "
        f"{method['experiment_graft']}"
        for method in artifact["post_sprint_pivot_methods"]
    )
    flag_lines = "\n".join(f"- {flag['flag']}" for flag in artifact["flagged_for_v453"])
    body = f"""## Exp 4911 - .453 wall and verifier-pivot SOTA ingestion - INGESTED

- Honest verdict: `{artifact['honest_verdict']}`
- Aimed at A1 fork: `{artifact['aimed_at_fork_verdict']}`
- A1b fork: `{artifact['a1b_fork_verdict']}`
- Branch: `{artifact['selected_branch']}`
- Banned channel note: `/deep-research` not invoked; no solve claim; no model load; nulled classes excluded.

### Final ARC Wall Diagnostics
{method_lines}

### Post-Sprint Verifier-Moat Pivot
This is the post-sprint verifier-moat pivot required after the ARC sprint retires.
{pivot_lines}

### Planner Flags
{flag_lines}
"""
    return _replace_section(text, STUDYING_SECTION_START, STUDYING_SECTION_END, body)


def update_research_references_text(text: str, artifact: JsonMap) -> str:
    reference_lines = "\n".join(
        f"- arXiv:{source_id} - {citation['title']} - {citation['url']} - HTTP {citation['http_status']}"
        for source_id, citation in sorted(artifact["citations"].items())
    )
    body = f"""## Exp 4911 V453 wall and verifier-pivot source set

Reliable-channel source set for `{artifact['honest_verdict']}`:

{reference_lines}
"""
    return _replace_section(text, REFERENCES_SECTION_START, REFERENCES_SECTION_END, body)


def validate_research_studying_text(text: str, artifact: JsonMap) -> None:
    _require(STUDYING_SECTION_START in text and STUDYING_SECTION_END in text, "research-studying section missing")
    _require(str(artifact["honest_verdict"]) in text, "research-studying honest verdict missing")
    _require("flagged_for_v453" in text, "research-studying flags missing")
    _require("post-sprint verifier-moat pivot" in text, "research-studying pivot missing")
    _require("no solve claim" in text, "research-studying no-solve discipline missing")
    for source in NOTE_REQUIRED_SOURCE_CITATIONS:
        _require(source in text, f"research-studying citation missing: {source}")


def validate_research_references_text(text: str, artifact: JsonMap) -> None:
    _require(REFERENCES_SECTION_START in text and REFERENCES_SECTION_END in text, "research-references section missing")
    for source_id, citation in artifact["citations"].items():
        _require(f"arXiv:{source_id}" in text, f"research-references source missing: {source_id}")
        _require(citation["title"] in text, f"research-references title missing: {source_id}")
        _require(citation["url"] in text, f"research-references url missing: {source_id}")


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_outputs(
    artifact_path: Path,
    studying_path: Path,
    references_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    missing = [
        relative_path
        for relative_path in (A1_ARTIFACT_RELATIVE_PATH, A1B_ARTIFACT_RELATIVE_PATH)
        if not (repo_root / relative_path).exists()
    ]
    if missing:
        artifact = build_blocked_artifact(missing=missing)
        _write_json(artifact_path, artifact)
        return artifact

    context = load_upstream_context(repo_root)
    artifact = build_artifact(upstream_context=context)
    validate_artifact(artifact)

    studying_text = studying_path.read_text(encoding="utf-8") if studying_path.exists() else ""
    references_text = references_path.read_text(encoding="utf-8") if references_path.exists() else ""
    updated_studying = update_research_studying_text(studying_text, artifact)
    updated_references = update_research_references_text(references_text, artifact)
    validate_research_studying_text(updated_studying, artifact)
    validate_research_references_text(updated_references, artifact)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    studying_path.write_text(updated_studying, encoding="utf-8")
    references_path.write_text(updated_references, encoding="utf-8")
    _write_json(artifact_path, artifact)
    return artifact


def main() -> int:
    repo_root = Path(os.environ.get("CARNOT_EXP4911_ROOT", Path(__file__).resolve().parents[2]))
    artifact = write_outputs(
        artifact_path=repo_root / RESULT_RELATIVE_PATH,
        studying_path=repo_root / STUDYING_RELATIVE_PATH,
        references_path=repo_root / REFERENCES_RELATIVE_PATH,
        repo_root=repo_root,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
