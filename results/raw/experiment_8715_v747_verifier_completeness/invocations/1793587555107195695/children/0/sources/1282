"""Exp 4890 SOTA ingestion for the .451 value-gap frontier.

Spec refs: REQ-ARC-WMTE-4890,
SCENARIO-ARC-WMTE-4890-V451-FRONTIER-MAPPED,
SCENARIO-ARC-WMTE-4890-BLOCKED-A1,
SCENARIO-ARC-WMTE-4890-NO-FABRICATION.

This module is intentionally aggregation-only. It reads the measured A1/A1b
artifacts, records the reliable-channel sweep provenance, and maps current
SOTA methods into experiment candidates without claiming a solve, training run,
leaderboard result, or model-load result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4890_sota_ingestion_v451_frontier.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
REFERENCES_RELATIVE_PATH = "research-references.md"
A1_ARTIFACT_RELATIVE_PATH = "results/experiment_4882_ttt_dynamics_value_gap.json"
A1B_ARTIFACT_RELATIVE_PATH = "results/experiment_4883_inducer_ceiling_ab.json"
HANDOFF_ARTIFACT_RELATIVE_PATH = "results/experiment_4879_sota_ingestion_v450_frontier.json"
NOTE_PATH = "research-studying.md#exp-4890-sota-ingestion-v451-frontier"
REFERENCES_PATH = "research-references.md#exp-4890-v451-frontier-source-set"
RANDOM_SEED = 4890
DURATION_S = 0.0001
HONEST_VERDICT = "success_sota_ingestion_v451_frontier_mapped"
BLOCKED_A1_VERDICT = "blocked_a1_artifact_missing"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
AIMED_AT_FORK_VERDICT = "INDUCER_CEILING_HARD"
AIMED_AT_INDUCER_ATTRIBUTION = "METHOD_IS_CEILING"
INGESTION_TRACK = "alternative_world_model_representations"
STUDYING_SECTION_START = "<!-- EXP4890-SOTA-INGESTION-V451-FRONTIER-START -->"
STUDYING_SECTION_END = "<!-- EXP4890-SOTA-INGESTION-V451-FRONTIER-END -->"
REFERENCES_SECTION_START = "<!-- EXP4890-V451-FRONTIER-REFERENCES-START -->"
REFERENCES_SECTION_END = "<!-- EXP4890-V451-FRONTIER-REFERENCES-END -->"
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
        "principle": "terminal prefix; mapping emitted is success_sota_ingestion_v451_frontier_mapped."
    },
    "aimed_at_fork_verdict": {
        "principle": (
            "the A1 fork verdict the ingestion targets (BEATABLE->scale TTA+convert; "
            "PLANNER_GAP->planning/search; CEILING_HARD->stronger local inducer / "
            "alt representation)."
        )
    },
    "methods_mapped": {
        "principle": (
            "the strongest 3-5 methods, each with a real arXiv ID + "
            "experiment_graft + validation_gate + fails_when."
        )
    },
    "arxiv_ids_cited": {
        "principle": (
            "every method claim cites a verifiable HTTP-200 arXiv ID "
            "(no fabrication -- adversarial_verify bar)."
        )
    },
    "flagged_for_v451": {
        "principle": (
            "the strongest method(s) flagged so the .451 planner reads the mapping "
            "(discover->ingest->plan->experiment)."
        )
    },
    "banned_channels_excluded": {
        "principle": (
            "true -- /deep-research NOT invoked; energy/coverage/exploration/selection "
            "classes NOT re-ingested."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (0.0001s floor)."
    },
    "preconditions_checked": {
        "principle": (
            "records reliable-channel checks + the A1 fork-verdict read; banned "
            "channels explicitly excluded."
        )
    },
}
FIELD_PRINCIPLES = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "citations": {
        "principle": "HTTP-200 arXiv source metadata backing every method claim."
    },
    "fresh_sweep": {
        "principle": "records focused sweep_clusters, sweep_semscholar, and WebFetch provenance."
    },
    "upstream_artifacts": {
        "principle": "binds the mapping to Exp 4882 A1, Exp 4883 A1b, and Exp 4879."
    },
    "sota_to_experiment_mapping_note": {
        "principle": "states how each SOTA method becomes a .451 experiment candidate."
    },
    "note_path": {
        "principle": "points to the idempotent research-studying.md ingestion note."
    },
    "references_path": {
        "principle": "points to the idempotent research-references.md source section."
    },
    "random_seed": {
        "principle": "deterministic experiment identifier for reproducible artifact generation."
    },
    "duration_s": {
        "principle": "0.0001s floor for aggregation-only inference substrate."
    },
    "reproducibility_checksum": {
        "principle": "content hash of citations, method map, flags, upstream context, and mapping note."
    },
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "aimed_at_fork_verdict",
    "methods_mapped",
    "arxiv_ids_cited",
    "flagged_for_v451",
    "banned_channels_excluded",
    "inference_substrate",
    "preconditions_checked",
    "citations",
    "fresh_sweep",
    "upstream_artifacts",
    "sota_to_experiment_mapping_note",
    "note_path",
    "references_path",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
    "field_principles",
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "method",
        "track",
        "source_ids",
        "maps_to_frontier",
        "targets_fork_verdict",
        "targets_inducer_attribution",
        "a1b_result_fit",
        "evidence",
        "experiment_graft",
        "validation_gate",
        "fails_when",
        "roadmap_candidate",
        "retired_class_reingested",
    }
)
REQUIRED_CITATION_FIELDS = frozenset({"title", "url", "http_status"})
REQUIRED_PRECONDITION_FIELDS = frozenset(
    {
        "agents_md_read",
        "codex_md_read",
        "research_studying_present",
        "research_references_present",
        "a1_artifact_present",
        "a1b_artifact_present",
        "handoff_artifact_present",
        "a1_fork_verdict_read",
        "a1_fork_verdict",
        "a1_honest_verdict",
        "a1_tta_value_delta_median",
        "a1_coverage_migration_count",
        "a1b_inducer_ceiling_attribution",
        "a1b_honest_verdict",
        "aimed_at_fork_verdict",
        "aimed_at_inducer_attribution",
        "branch_reason",
        "ingestion_track",
        "sweep_clusters_used",
        "sweep_cluster_ids",
        "sweep_cluster_urls",
        "sweep_semscholar_used",
        "sweep_semscholar_queries",
        "semantic_scholar_result",
        "semantic_scholar_unique_arxiv_ids",
        "websearch_webfetch_used",
        "websearch_webfetch_top_sources",
        "top_source_count",
        "arxiv_http_200_verified_ids",
        "deep_research_invoked",
        "retired_energy_classes_reingested",
        "coverage_vocabulary_reingested",
        "exploration_strategy_reingested",
        "selection_ranking_reingested",
        "perception_from_grid_reingested",
        "model_load",
        "training_launched",
        "leaderboard_submission",
        "solve_claim_made",
        "research_conductor_modified",
        "ops_docs_modified",
    }
)
REQUIRED_FRESH_SWEEP_FIELDS = frozenset(
    {
        "filtered_track",
        "cluster_ids",
        "cluster_urls",
        "cluster_top_arxiv_ids",
        "semantic_scholar_queries",
        "semantic_scholar_result",
        "semantic_scholar_unique_arxiv_ids",
        "webfetch_top_sources",
    }
)
REQUIRED_UPSTREAM_FIELDS = frozenset(
    {
        "a1_artifact",
        "a1b_artifact",
        "handoff_artifact",
        "a1_honest_verdict",
        "a1_fork_verdict",
        "a1_tta_value_delta_median",
        "a1_coverage_migration_count",
        "a1b_honest_verdict",
        "a1b_inducer_ceiling_attribution",
        "handoff_honest_verdict",
        "handoff_aimed_at_fork_verdict",
        "carried_forward_from_4879",
        "not_carried_forward_from_4879",
    }
)
REQUIRED_MAPPING_NOTE_FIELDS = frozenset(
    {
        "summary",
        "terminal_success",
        "source_ids",
        "root_cause",
        "planner_instruction",
        "a1_result",
        "a1b_result",
        "not_carried_forward",
    }
)
REQUIRED_SOURCE_IDS = frozenset(
    {
        "2503.18938",
        "2505.08073",
        "2602.23997",
        "2603.19312",
        "2606.25421",
        "2606.26217",
    }
)
REQUIRED_TRACKS = frozenset(
    {
        "agent_authored_decision_need_targets",
        "action_prefix_latent_adapter",
        "latent_action_world_model_adapter",
        "reverse_counterfactual_world_model_targets",
        "verification_calibrated_abstraction_substrate",
    }
)
RETIRED_TRACKS = frozenset(
    {
        "energy_as_arc_lever",
        "coverage_vocabulary",
        "exploration_strategy",
        "selection_ranking",
        "perception_from_grid",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS)

CLUSTER_6_URL = (
    'http://export.arxiv.org/api/query?search_query=(abs:"neural+guided+search"+OR+'
    'abs:"learned+heuristic"+OR+abs:"value+guided+search"+OR+'
    'abs:"program+induction"+OR+abs:"world+model"+OR+abs:"goal+induction")+'
    'AND+(abs:"planning"+OR+abs:"agent"+OR+abs:"reasoning"+OR+'
    'abs:"reinforcement+learning")&start=0&max_results=8&sortBy=submittedDate&'
    "sortOrder=descending"
)
CLUSTER_6_PAGE2_URL = (
    'http://export.arxiv.org/api/query?search_query=(abs:"neural+guided+search"+OR+'
    'abs:"learned+heuristic"+OR+abs:"value+guided+search"+OR+'
    'abs:"program+induction"+OR+abs:"world+model"+OR+abs:"goal+induction")+'
    'AND+(abs:"planning"+OR+abs:"agent"+OR+abs:"reasoning"+OR+'
    'abs:"reinforcement+learning")&start=8&max_results=8&sortBy=submittedDate&'
    "sortOrder=descending"
)
CLUSTER_TOP_ARXIV_IDS = [
    "2606.27364",
    "2606.27014",
    "2606.26969",
    "2606.26964",
    "2606.26713",
    "2606.26321",
    "2606.26217",
    "2606.26057",
    "2606.25880",
    "2606.25527",
    "2606.25421",
    "2606.24842",
    "2606.24742",
    "2606.24597",
    "2606.24476",
    "2606.24256",
]
SEMANTIC_SCHOLAR_QUERIES = [
    "agent authored world model sequential decision making",
    "decision oriented world model planning agents",
    "action prefix latent world model planning",
    "latent action prefix prediction world model agents",
    "world model representation beyond next observation prediction",
]
SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS: list[str] = []
SEMANTIC_SCHOLAR_RESULT = (
    "Five focused Semantic Scholar queries returned HTTP 429; 0 unique arXiv IDs "
    "were recorded, and no rate-limited result was promoted as evidence."
)
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2503.18938",
    "https://arxiv.org/abs/2505.08073",
    "https://arxiv.org/abs/2602.23997",
    "https://arxiv.org/abs/2603.19312",
    "https://arxiv.org/abs/2606.25421",
    "https://arxiv.org/abs/2606.26217",
]

CITATIONS = {
    "2503.18938": {
        "title": "AdaWorld: Learning Adaptable World Models with Latent Actions",
        "url": "https://arxiv.org/abs/2503.18938",
        "http_status": 200,
    },
    "2505.08073": {
        "title": "Explainable Reinforcement Learning Agents Using World Models",
        "url": "https://arxiv.org/abs/2505.08073",
        "http_status": 200,
    },
    "2602.23997": {
        "title": (
            "Foundation World Models for Agents that Learn, Verify, and Adapt "
            "Reliably Beyond Static Environments"
        ),
        "url": "https://arxiv.org/abs/2602.23997",
        "http_status": 200,
    },
    "2603.19312": {
        "title": "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels",
        "url": "https://arxiv.org/abs/2603.19312",
        "http_status": 200,
    },
    "2606.25421": {
        "title": "Beyond Next-Observation Prediction: Agent-Authored World Modeling for Sequential Decision Making",
        "url": "https://arxiv.org/abs/2606.25421",
        "http_status": 200,
    },
    "2606.26217": {
        "title": "Fast LeWorldModel",
        "url": "https://arxiv.org/abs/2606.26217",
        "http_status": 200,
    },
}

BRANCH_REASON = (
    "A1 reported INDUCER_CEILING_HARD and A1b attributed the inducer ceiling to "
    "METHOD_IS_CEILING, so .451 targets alternative world-model representations "
    "beyond executable code rather than more TTA scaling, planner search, or a "
    "stronger local code inducer."
)

DEFAULT_UPSTREAM_CONTEXT = {
    "a1_honest_verdict": "complete_ttt_dynamics_no_value_lift_INDUCER_CEILING_HARD",
    "a1_fork_verdict": AIMED_AT_FORK_VERDICT,
    "a1_tta_value_delta_median": -0.087781,
    "a1_coverage_migration_count": 0,
    "a1b_honest_verdict": "complete_inducer_ceiling_neither_lane_lifts_method_is_ceiling",
    "a1b_inducer_ceiling_attribution": AIMED_AT_INDUCER_ATTRIBUTION,
    "handoff_honest_verdict": "success_sota_ingestion_v450_frontier_mapped",
    "handoff_aimed_at_fork_verdict": "INDUCER_CEILING",
    "ingestion_track": INGESTION_TRACK,
}

FLAGGED_FOR_V451 = [
    {
        "candidate": "agent_authored_decision_need_targets",
        "flag": "flagged_for_v451: agent_authored_decision_need_targets (arXiv:2606.25421)",
        "source_ids": ["2606.25421"],
        "maps_to_frontier": ".451",
        "priority": 1,
    },
    {
        "candidate": "action_prefix_latent_adapter",
        "flag": "flagged_for_v451: action_prefix_latent_adapter (arXiv:2606.26217 + arXiv:2603.19312)",
        "source_ids": ["2606.26217", "2603.19312"],
        "maps_to_frontier": ".451",
        "priority": 2,
    },
    {
        "candidate": "latent_action_world_model_adapter",
        "flag": "flagged_for_v451: latent_action_world_model_adapter (arXiv:2503.18938)",
        "source_ids": ["2503.18938"],
        "maps_to_frontier": ".451",
        "priority": 3,
    },
]

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Agent-authored decision-need world-model targets",
        "track": "agent_authored_decision_need_targets",
        "source_ids": ["2606.25421"],
        "maps_to_frontier": ".451",
        "targets_fork_verdict": AIMED_AT_FORK_VERDICT,
        "targets_inducer_attribution": AIMED_AT_INDUCER_ATTRIBUTION,
        "a1b_result_fit": (
            "A1b says the current executable-code induction method is the ceiling; "
            "decision-need targets replace generic next-observation supervision."
        ),
        "evidence": (
            "arXiv:2606.25421 proposes Agent-Authored World Modeling, where the "
            "policy identifies decision-relevant dynamics before acting."
        ),
        "experiment_graft": (
            "For each A1 held-out engine miss, generate a decision-need target "
            "such as hidden register state, object persistence, or action effect, "
            "then train or prompt a non-code world-model target table before engine loading."
        ),
        "validation_gate": (
            "Promote only if the decision-target representation raises held-out "
            "changed-cell value accuracy on the same A1 games without using the "
            "banked answer as supervision."
        ),
        "fails_when": (
            "The authored targets mirror the current model misconception, require "
            "unobserved facts, or improve next-frame text while held-out dynamics stay flat."
        ),
        "roadmap_candidate": FLAGGED_FOR_V451[0]["flag"],
        "retired_class_reingested": False,
    },
    {
        "method": "Action-prefix latent transition adapter",
        "track": "action_prefix_latent_adapter",
        "source_ids": ["2606.26217", "2603.19312"],
        "maps_to_frontier": ".451",
        "targets_fork_verdict": AIMED_AT_FORK_VERDICT,
        "targets_inducer_attribution": AIMED_AT_INDUCER_ATTRIBUTION,
        "a1b_result_fit": (
            "A1b rules out another same-method executable inducer pass; prefix "
            "latents model multi-step action effects without rolling one-step code."
        ),
        "evidence": (
            "arXiv:2606.26217 replaces repeated one-step latent rollout with "
            "action-prefix prediction; arXiv:2603.19312 supplies the compact "
            "LeWorldModel latent substrate."
        ),
        "experiment_graft": (
            "Encode candidate action prefixes into latent future-state deltas and "
            "score A1 held-out transitions through the latent adapter before "
            "converting only accepted deltas into engine facts."
        ),
        "validation_gate": (
            "Count the graft only when long-horizon held-out transition accuracy "
            "improves and one-step observed-prefix replay does not regress."
        ),
        "fails_when": (
            "Prefix supervision hides wrong mechanics, latent states cannot be "
            "decoded into verifier-checkable facts, or compounding error remains flat."
        ),
        "roadmap_candidate": FLAGGED_FOR_V451[1]["flag"],
        "retired_class_reingested": False,
    },
    {
        "method": "Latent-action adaptable world-model interface",
        "track": "latent_action_world_model_adapter",
        "source_ids": ["2503.18938"],
        "maps_to_frontier": ".451",
        "targets_fork_verdict": AIMED_AT_FORK_VERDICT,
        "targets_inducer_attribution": AIMED_AT_INDUCER_ATTRIBUTION,
        "a1b_result_fit": (
            "A1b method-ceiling means the missing structure may be action semantics, "
            "not code synthesis strength; latent actions give a non-code action layer."
        ),
        "evidence": (
            "arXiv:2503.18938 extracts self-supervised latent actions and conditions "
            "an adaptable world model on them for transfer with limited interactions."
        ),
        "experiment_graft": (
            "Infer latent action tokens from cold-start ARC transitions, align E3 "
            "discrete controls to those tokens, and feed the latent action state into "
            "the held-out transition scorer."
        ),
        "validation_gate": (
            "Promote only if the latent-action adapter predicts off-prefix action "
            "effects better than the current executable engine on the A1 split."
        ),
        "fails_when": (
            "The latent actions do not align with legal controls, collapse across "
            "different mechanics, or require more adaptation interactions than ARC permits."
        ),
        "roadmap_candidate": FLAGGED_FOR_V451[2]["flag"],
        "retired_class_reingested": False,
    },
    {
        "method": "Reverse counterfactual world-model targeter",
        "track": "reverse_counterfactual_world_model_targets",
        "source_ids": ["2505.08073", "2606.25421"],
        "maps_to_frontier": ".451",
        "targets_fork_verdict": AIMED_AT_FORK_VERDICT,
        "targets_inducer_attribution": AIMED_AT_INDUCER_ATTRIBUTION,
        "a1b_result_fit": (
            "A1b method-ceiling suggests the engine needs a targetable state "
            "representation; reverse world models ask what state fact would make "
            "a desired action rational."
        ),
        "evidence": (
            "arXiv:2505.08073 augments model-based RL with a reverse world model "
            "for counterfactual state targets; arXiv:2606.25421 supplies the "
            "decision-need target construction."
        ),
        "experiment_graft": (
            "For each failed A1 transition, ask a reverse model for the missing "
            "state fact that would make the predicted action effect valid, then "
            "turn that fact into a targeted induction or probe row."
        ),
        "validation_gate": (
            "Accept only reverse targets that reduce held-out dynamics errors while "
            "remaining oracle-distinct from level completion."
        ),
        "fails_when": (
            "The counterfactual state is unreachable, leaks the terminal answer, "
            "or explains the policy without improving transition prediction."
        ),
        "roadmap_candidate": FLAGGED_FOR_V451[0]["flag"],
        "retired_class_reingested": False,
    },
    {
        "method": "Verification-calibrated abstraction substrate",
        "track": "verification_calibrated_abstraction_substrate",
        "source_ids": ["2602.23997", "2603.19312"],
        "maps_to_frontier": ".451",
        "targets_fork_verdict": AIMED_AT_FORK_VERDICT,
        "targets_inducer_attribution": AIMED_AT_INDUCER_ATTRIBUTION,
        "a1b_result_fit": (
            "A1b method-ceiling points away from another executable-code attempt; "
            "a calibrated abstraction layer makes representation reliability explicit."
        ),
        "evidence": (
            "arXiv:2602.23997 argues for world models with online abstraction "
            "calibration and verification hooks; arXiv:2603.19312 demonstrates "
            "compact latent world-model structure with physical probes."
        ),
        "experiment_graft": (
            "Insert a persistent latent abstraction state beside the executable "
            "engine and require each abstract fact to carry a verifier-calibrated "
            "confidence before it affects planning."
        ),
        "validation_gate": (
            "Promote only if abstraction confidence predicts held-out engine "
            "mismatches and the calibrated facts improve A1 value accuracy."
        ),
        "fails_when": (
            "The abstraction is too coarse for ARC mechanics, calibration only "
            "tracks seen prefixes, or verifier hooks become a selection/ranking rerun."
        ),
        "roadmap_candidate": FLAGGED_FOR_V451[1]["flag"],
        "retired_class_reingested": False,
    },
]

DEFAULT_FRESH_SWEEP = {
    "filtered_track": (
        "V451 frontier for INDUCER_CEILING_HARD + METHOD_IS_CEILING: "
        "alternative world-model representations beyond executable code"
    ),
    "cluster_ids": [6],
    "cluster_urls": [CLUSTER_6_URL, CLUSTER_6_PAGE2_URL],
    "cluster_top_arxiv_ids": CLUSTER_TOP_ARXIV_IDS,
    "semantic_scholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "semantic_scholar_result": SEMANTIC_SCHOLAR_RESULT,
    "semantic_scholar_unique_arxiv_ids": SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
    "webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
}

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "research_studying_present": True,
    "research_references_present": True,
    "a1_artifact_present": True,
    "a1b_artifact_present": True,
    "handoff_artifact_present": True,
    "a1_fork_verdict_read": True,
    "a1_fork_verdict": AIMED_AT_FORK_VERDICT,
    "a1_honest_verdict": DEFAULT_UPSTREAM_CONTEXT["a1_honest_verdict"],
    "a1_tta_value_delta_median": DEFAULT_UPSTREAM_CONTEXT["a1_tta_value_delta_median"],
    "a1_coverage_migration_count": DEFAULT_UPSTREAM_CONTEXT["a1_coverage_migration_count"],
    "a1b_inducer_ceiling_attribution": AIMED_AT_INDUCER_ATTRIBUTION,
    "a1b_honest_verdict": DEFAULT_UPSTREAM_CONTEXT["a1b_honest_verdict"],
    "aimed_at_fork_verdict": AIMED_AT_FORK_VERDICT,
    "aimed_at_inducer_attribution": AIMED_AT_INDUCER_ATTRIBUTION,
    "branch_reason": BRANCH_REASON,
    "ingestion_track": INGESTION_TRACK,
    "sweep_clusters_used": True,
    "sweep_cluster_ids": [6],
    "sweep_cluster_urls": [CLUSTER_6_URL, CLUSTER_6_PAGE2_URL],
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "semantic_scholar_result": SEMANTIC_SCHOLAR_RESULT,
    "semantic_scholar_unique_arxiv_ids": SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
    "websearch_webfetch_used": True,
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
    "top_source_count": len(WEBSEARCH_WEBFETCH_TOP_SOURCES),
    "arxiv_http_200_verified_ids": [
        f"https://arxiv.org/abs/{source_id}" for source_id in sorted(REQUIRED_SOURCE_IDS)
    ],
    "deep_research_invoked": False,
    "retired_energy_classes_reingested": False,
    "coverage_vocabulary_reingested": False,
    "exploration_strategy_reingested": False,
    "selection_ranking_reingested": False,
    "perception_from_grid_reingested": False,
    "model_load": False,
    "training_launched": False,
    "leaderboard_submission": False,
    "solve_claim_made": False,
    "research_conductor_modified": False,
    "ops_docs_modified": False,
}

DEFAULT_UPSTREAM_ARTIFACTS = {
    "a1_artifact": A1_ARTIFACT_RELATIVE_PATH,
    "a1b_artifact": A1B_ARTIFACT_RELATIVE_PATH,
    "handoff_artifact": HANDOFF_ARTIFACT_RELATIVE_PATH,
    "a1_honest_verdict": DEFAULT_UPSTREAM_CONTEXT["a1_honest_verdict"],
    "a1_fork_verdict": DEFAULT_UPSTREAM_CONTEXT["a1_fork_verdict"],
    "a1_tta_value_delta_median": DEFAULT_UPSTREAM_CONTEXT["a1_tta_value_delta_median"],
    "a1_coverage_migration_count": DEFAULT_UPSTREAM_CONTEXT["a1_coverage_migration_count"],
    "a1b_honest_verdict": DEFAULT_UPSTREAM_CONTEXT["a1b_honest_verdict"],
    "a1b_inducer_ceiling_attribution": DEFAULT_UPSTREAM_CONTEXT["a1b_inducer_ceiling_attribution"],
    "handoff_honest_verdict": DEFAULT_UPSTREAM_CONTEXT["handoff_honest_verdict"],
    "handoff_aimed_at_fork_verdict": DEFAULT_UPSTREAM_CONTEXT["handoff_aimed_at_fork_verdict"],
    "carried_forward_from_4879": [
        "agent_authored_world_model_targets",
        "action_prefix_world_model_adapter",
    ],
    "not_carried_forward_from_4879": [
        "test_time_dynamics_adaptation_scaling",
        "family_b_vs_local_open_code_inducer_ab",
        "cegis_world_model_refinement_loop",
        "energy-as-ARC-lever",
        "coverage/vocabulary",
        "exploration-strategy",
        "selection/ranking",
        "perception-from-grid",
    ],
}

DEFAULT_MAPPING_NOTE = {
    "summary": (
        "A1 reported INDUCER_CEILING_HARD and A1b attributed the ceiling to "
        "METHOD_IS_CEILING; .451 should test alternative world-model "
        "representations beyond executable code."
    ),
    "terminal_success": HONEST_VERDICT,
    "source_ids": sorted(REQUIRED_SOURCE_IDS),
    "root_cause": "method-level executable-code world-model representation ceiling",
    "planner_instruction": (
        "Start with agent-authored decision-need targets, then action-prefix "
        "latent adapters and latent-action world-model interfaces. Promote only "
        "methods that improve A1 held-out transition value accuracy."
    ),
    "a1_result": "A1 fork_verdict=INDUCER_CEILING_HARD with no coverage migration.",
    "a1b_result": "A1b inducer_ceiling_attribution=METHOD_IS_CEILING.",
    "not_carried_forward": (
        "Do not re-promote TTA scaling, stronger local open-code inducers, "
        "energy, coverage/vocabulary, exploration, selection/ranking, or "
        "perception-from-grid classes."
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_set_checksum(
    citations: JsonMap,
    methods: Sequence[JsonMap],
    flags: Sequence[JsonMap],
    upstream_artifacts: JsonMap,
    mapping_note: JsonMap,
) -> str:
    payload = json.dumps(
        {
            "citations": citations,
            "flags": list(flags),
            "mapping_note": mapping_note,
            "methods": list(methods),
            "upstream_artifacts": upstream_artifacts,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def select_ingestion_track(a1_artifact: JsonMap, a1b_artifact: JsonMap) -> str:
    verdict = a1_artifact.get("fork_verdict")
    attribution = a1b_artifact.get("inducer_ceiling_attribution")
    if verdict == "INDUCER_CEILING_BEATABLE":
        return "tta_scaling_first_win_conversion"
    if verdict == "PLANNER_GAP":
        return "neural_guided_planning_search"
    if verdict == "INDUCER_CEILING_HARD" and attribution == "LOCAL_MODEL_IS_CEILING":
        return "stronger_local_open_code_inducers"
    if verdict == "INDUCER_CEILING_HARD" and attribution == "METHOD_IS_CEILING":
        return INGESTION_TRACK
    if verdict == "INDUCER_CEILING_HARD" and attribution == "LOCAL_ALREADY_SUFFICIENT":
        return "local_already_sufficient_scale_conversion"
    raise ValueError(f"unsupported A1 fork / A1b attribution: {verdict!r} / {attribution!r}")


def derive_upstream_context(a1_artifact: JsonMap, a1b_artifact: JsonMap, handoff_artifact: JsonMap) -> dict[str, object]:
    track = select_ingestion_track(a1_artifact, a1b_artifact)
    return {
        "a1_honest_verdict": a1_artifact.get("honest_verdict", ""),
        "a1_fork_verdict": a1_artifact.get("fork_verdict"),
        "a1_tta_value_delta_median": a1_artifact.get("tta_changed_cell_value_accuracy_delta_median"),
        "a1_coverage_migration_count": int(a1_artifact.get("coverage_migration_count") or 0),
        "a1b_honest_verdict": a1b_artifact.get("honest_verdict", ""),
        "a1b_inducer_ceiling_attribution": a1b_artifact.get("inducer_ceiling_attribution"),
        "handoff_honest_verdict": handoff_artifact.get("honest_verdict", ""),
        "handoff_aimed_at_fork_verdict": handoff_artifact.get("aimed_at_fork_verdict", ""),
        "ingestion_track": track,
    }


def load_upstream_context(repo_root: Path) -> dict[str, object]:
    a1 = json.loads((repo_root / A1_ARTIFACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    a1b = json.loads((repo_root / A1B_ARTIFACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    handoff = json.loads((repo_root / HANDOFF_ARTIFACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    return derive_upstream_context(a1, a1b, handoff)


def _preconditions_from_context(upstream_context: JsonMap) -> dict[str, object]:
    return dict(DEFAULT_PRECONDITIONS_CHECKED) | {
        "a1_fork_verdict": upstream_context["a1_fork_verdict"],
        "a1_honest_verdict": upstream_context["a1_honest_verdict"],
        "a1_tta_value_delta_median": upstream_context["a1_tta_value_delta_median"],
        "a1_coverage_migration_count": upstream_context["a1_coverage_migration_count"],
        "a1b_inducer_ceiling_attribution": upstream_context["a1b_inducer_ceiling_attribution"],
        "a1b_honest_verdict": upstream_context["a1b_honest_verdict"],
        "ingestion_track": upstream_context["ingestion_track"],
    }


def _upstream_artifacts_from_context(upstream_context: JsonMap) -> dict[str, object]:
    return dict(DEFAULT_UPSTREAM_ARTIFACTS) | {
        "a1_honest_verdict": upstream_context["a1_honest_verdict"],
        "a1_fork_verdict": upstream_context["a1_fork_verdict"],
        "a1_tta_value_delta_median": upstream_context["a1_tta_value_delta_median"],
        "a1_coverage_migration_count": upstream_context["a1_coverage_migration_count"],
        "a1b_honest_verdict": upstream_context["a1b_honest_verdict"],
        "a1b_inducer_ceiling_attribution": upstream_context["a1b_inducer_ceiling_attribution"],
        "handoff_honest_verdict": upstream_context["handoff_honest_verdict"],
        "handoff_aimed_at_fork_verdict": upstream_context["handoff_aimed_at_fork_verdict"],
    }


def build_artifact(
    *,
    upstream_context: JsonMap = DEFAULT_UPSTREAM_CONTEXT,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    fresh_sweep: JsonMap = DEFAULT_FRESH_SWEEP,
    flagged_for_v451: Sequence[JsonMap] = FLAGGED_FOR_V451,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    upstream_artifacts = _upstream_artifacts_from_context(upstream_context)
    mapping_note = dict(DEFAULT_MAPPING_NOTE)
    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "aimed_at_fork_verdict": AIMED_AT_FORK_VERDICT,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "arxiv_ids_cited": sorted(citations),
        "flagged_for_v451": [dict(flag) for flag in flagged_for_v451],
        "banned_channels_excluded": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": _preconditions_from_context(upstream_context),
        "citations": {source_id: dict(citation) for source_id, citation in citations.items()},
        "fresh_sweep": dict(fresh_sweep),
        "upstream_artifacts": upstream_artifacts,
        "sota_to_experiment_mapping_note": mapping_note,
        "note_path": NOTE_PATH,
        "references_path": REFERENCES_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_S,
        "reproducibility_checksum": source_set_checksum(
            citations,
            methods_mapped,
            flagged_for_v451,
            upstream_artifacts,
            mapping_note,
        ),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


REPRODUCIBILITY_CHECKSUM = source_set_checksum(
    CITATIONS,
    DEFAULT_METHODS_MAPPED,
    FLAGGED_FOR_V451,
    DEFAULT_UPSTREAM_ARTIFACTS,
    DEFAULT_MAPPING_NOTE,
)


def build_blocked_a1_artifact() -> dict[str, object]:
    preconditions = dict(DEFAULT_PRECONDITIONS_CHECKED) | {
        "a1_artifact_present": False,
        "a1_fork_verdict_read": False,
        "a1_fork_verdict": None,
        "a1_honest_verdict": "",
        "a1_tta_value_delta_median": None,
        "a1_coverage_migration_count": 0,
        "a1b_artifact_present": False,
        "a1b_inducer_ceiling_attribution": None,
        "a1b_honest_verdict": "",
        "handoff_artifact_present": False,
        "ingestion_track": "blocked",
        "branch_reason": "A1 artifact missing; no fork verdict was fabricated.",
    }
    artifact = {
        "honest_verdict": BLOCKED_A1_VERDICT,
        "aimed_at_fork_verdict": None,
        "methods_mapped": [],
        "arxiv_ids_cited": [],
        "flagged_for_v451": [],
        "banned_channels_excluded": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "citations": {},
        "fresh_sweep": dict(DEFAULT_FRESH_SWEEP) | {"webfetch_top_sources": []},
        "upstream_artifacts": {},
        "sota_to_experiment_mapping_note": {},
        "note_path": NOTE_PATH,
        "references_path": REFERENCES_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_S,
        "reproducibility_checksum": source_set_checksum({}, [], [], {}, {}),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    _require(artifact["honest_verdict"].startswith("blocked_"), "blocked artifact must use blocked prefix")
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS).difference(artifact)
    extra = set(artifact).difference(REQUIRED_ARTIFACT_FIELDS)
    _require(not missing, f"artifact missing required fields: {sorted(missing)}")
    _require(not extra, f"artifact has unexpected fields: {sorted(extra)}")
    _require(isinstance(artifact["honest_verdict"], str), "honest_verdict must be a string")
    _require(
        artifact["honest_verdict"].startswith(TERMINAL_PREFIXES),
        "honest_verdict must use a terminal prefix",
    )
    _require(artifact["honest_verdict"] == HONEST_VERDICT, f"honest_verdict must equal {HONEST_VERDICT!r}")
    _require(
        artifact["aimed_at_fork_verdict"] == AIMED_AT_FORK_VERDICT,
        "aimed_at_fork_verdict must be INDUCER_CEILING_HARD",
    )
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must be aggregation-only",
    )
    _require(artifact["banned_channels_excluded"] is True, "banned_channels_excluded must be true")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles must match annotations")
    _require(artifact["note_path"] == NOTE_PATH, "note_path must point at the Exp 4890 note")
    _require(artifact["references_path"] == REFERENCES_PATH, "references_path must point at Exp 4890 references")
    _require(artifact["random_seed"] == RANDOM_SEED, "random_seed must be the experiment id")
    _require(artifact["duration_s"] == DURATION_S, "duration_s must preserve the 0.0001s floor")
    _validate_citations(artifact["citations"], artifact["arxiv_ids_cited"])
    _validate_methods(artifact["methods_mapped"], artifact["arxiv_ids_cited"])
    _validate_flags(artifact["flagged_for_v451"], artifact["arxiv_ids_cited"])
    _validate_preconditions(artifact["preconditions_checked"])
    _validate_fresh_sweep(artifact["fresh_sweep"])
    _validate_upstream_artifacts(artifact["upstream_artifacts"])
    _validate_mapping_note(artifact["sota_to_experiment_mapping_note"], artifact["arxiv_ids_cited"])
    _require(
        artifact["reproducibility_checksum"]
        == source_set_checksum(
            artifact["citations"],
            artifact["methods_mapped"],
            artifact["flagged_for_v451"],
            artifact["upstream_artifacts"],
            artifact["sota_to_experiment_mapping_note"],
        ),
        "reproducibility checksum must hash citations, methods, flags, upstream context, and mapping note",
    )


def _validate_citations(citations: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(citations, Mapping), "citations must be a mapping")
    _require(set(citations) == REQUIRED_SOURCE_IDS, "citations must exactly cover verified source IDs")
    _require(arxiv_ids_cited == sorted(REQUIRED_SOURCE_IDS), "arxiv_ids_cited must list every verified arXiv ID")
    for source_id, citation in citations.items():
        _require(isinstance(citation, Mapping), "each citation must be a mapping")
        _require(set(citation) == REQUIRED_CITATION_FIELDS, "each citation must contain title, url, http_status")
        _require(citation["url"] == f"https://arxiv.org/abs/{source_id}", "citation url must match arXiv ID")
        _require(citation["http_status"] == 200, "each citation http_status must be 200")
        _require(bool(citation["title"]), "each citation title must be non-empty")


def _validate_methods(methods: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(methods, Sequence) and not isinstance(methods, str | bytes), "methods_mapped must be a list")
    _require(3 <= len(methods) <= 5, "methods_mapped must contain three to five methods")
    cited = set(arxiv_ids_cited)
    tracks: set[str] = set()
    for method in methods:
        _require(isinstance(method, Mapping), "each method must be a mapping")
        _require(set(method) == REQUIRED_METHOD_FIELDS, "each method must match the required method schema")
        source_ids = method["source_ids"]
        _require(
            isinstance(source_ids, Sequence) and not isinstance(source_ids, str | bytes) and bool(source_ids),
            "each method must cite source_ids",
        )
        _require(set(source_ids).issubset(cited), "method source_ids must be verified citations")
        _require(method["maps_to_frontier"] == ".451", "method must map to the .451 frontier")
        _require(method["targets_fork_verdict"] == AIMED_AT_FORK_VERDICT, "method must target fork verdict")
        _require(
            method["targets_inducer_attribution"] == AIMED_AT_INDUCER_ATTRIBUTION,
            "method must target A1b attribution",
        )
        _require("A1b" in str(method["a1b_result_fit"]), "method must fit the A1b attribution")
        _require("arXiv:" in str(method["evidence"]), "method evidence must cite arXiv IDs")
        _require(bool(method["experiment_graft"]), "each method needs an experiment graft")
        _require(bool(method["validation_gate"]), "each method needs a validation gate")
        _require(bool(method["fails_when"]), "each method needs fails_when")
        _require("flagged_for_v451" in str(method["roadmap_candidate"]), "each method needs a .451 roadmap candidate")
        _require(method["retired_class_reingested"] is False, "retired class must not be reingested")
        track = str(method["track"])
        _require(track not in RETIRED_TRACKS, "retired method track must not be mapped")
        tracks.add(track)
    _require(REQUIRED_TRACKS == tracks, "methods_mapped missing required .451 representation tracks")


def _validate_flags(flags: object, arxiv_ids_cited: object) -> None:
    _require(
        isinstance(flags, Sequence) and not isinstance(flags, str | bytes) and bool(flags),
        "flagged_for_v451 required",
    )
    cited = set(arxiv_ids_cited)
    for flag in flags:
        _require(isinstance(flag, Mapping), "each flagged_for_v451 entry must be a mapping")
        _require("candidate" in flag and "flag" in flag, "each flagged_for_v451 entry needs candidate and flag")
        as_text = json.dumps(flag, sort_keys=True)
        _require("flagged_for_v450" not in as_text, "stale .450 flag found in flagged_for_v451")
        _require("flagged_for_v451" in str(flag["flag"]), "flagged_for_v451 entries must carry the .451 flag")
        _require(set(flag.get("source_ids", [])).issubset(cited), "flagged_for_v451 source_ids must be verified")


def _validate_preconditions(preconditions: object) -> None:
    _require(isinstance(preconditions, Mapping), "preconditions_checked must be a mapping")
    _require(set(preconditions) == REQUIRED_PRECONDITION_FIELDS, "preconditions_checked must match schema")
    _require(preconditions["a1_artifact_present"] is True, "A1 artifact must be present")
    _require(preconditions["a1_fork_verdict_read"] is True, "A1 fork verdict must be read")
    _require(preconditions["a1_fork_verdict"] == AIMED_AT_FORK_VERDICT, "A1 fork verdict mismatch")
    _require(preconditions["a1b_artifact_present"] is True, "A1b artifact must be present")
    _require(
        preconditions["a1b_inducer_ceiling_attribution"] == AIMED_AT_INDUCER_ATTRIBUTION,
        "A1b attribution must be METHOD_IS_CEILING",
    )
    _require(preconditions["aimed_at_fork_verdict"] == AIMED_AT_FORK_VERDICT, "precondition fork target mismatch")
    _require(
        preconditions["aimed_at_inducer_attribution"] == AIMED_AT_INDUCER_ATTRIBUTION,
        "precondition attribution target mismatch",
    )
    _require(preconditions["branch_reason"] == BRANCH_REASON, "branch reason must match A1/A1b branch")
    _require(preconditions["ingestion_track"] == INGESTION_TRACK, "ingestion track must be alt representations")
    _require(preconditions["sweep_clusters_used"] is True, "sweep_clusters must be used")
    _require(preconditions["sweep_cluster_ids"] == [6], "sweep cluster IDs must be [6]")
    _require(preconditions["sweep_semscholar_used"] is True, "sweep_semscholar must be used")
    _require(preconditions["semantic_scholar_result"] == SEMANTIC_SCHOLAR_RESULT, "Semantic Scholar result mismatch")
    _require(preconditions["websearch_webfetch_used"] is True, "WebSearch/WebFetch must be used")
    _require(5 <= int(preconditions["top_source_count"]) <= 8, "top_source_count must record top five to eight sources")
    _require(preconditions["deep_research_invoked"] is False, "deep-research must not be invoked")
    _require(preconditions["retired_energy_classes_reingested"] is False, "retired energy classes must not be reingested")
    _require(preconditions["coverage_vocabulary_reingested"] is False, "coverage/vocabulary must not be reingested")
    _require(preconditions["exploration_strategy_reingested"] is False, "exploration strategy must not be reingested")
    _require(preconditions["selection_ranking_reingested"] is False, "selection/ranking must not be reingested")
    _require(preconditions["perception_from_grid_reingested"] is False, "perception-from-grid must not be reingested")
    _require(preconditions["model_load"] is False, "model load must not occur")
    _require(preconditions["training_launched"] is False, "training must not be launched")
    _require(preconditions["leaderboard_submission"] is False, "leaderboard submission must not occur")
    _require(preconditions["solve_claim_made"] is False, "solve claim must remain false")
    _require(preconditions["research_conductor_modified"] is False, "research_conductor must not be modified")
    _require(preconditions["ops_docs_modified"] is False, "ops docs must not be modified by this workflow")


def _validate_fresh_sweep(fresh_sweep: object) -> None:
    _require(isinstance(fresh_sweep, Mapping), "fresh_sweep must be a mapping")
    _require(set(fresh_sweep) == REQUIRED_FRESH_SWEEP_FIELDS, "fresh_sweep must match schema")
    _require(fresh_sweep["cluster_ids"] == [6], "fresh_sweep must record cluster 6")
    _require(fresh_sweep["semantic_scholar_result"] == SEMANTIC_SCHOLAR_RESULT, "fresh_sweep S2 result mismatch")
    sources = fresh_sweep["webfetch_top_sources"]
    _require(
        isinstance(sources, Sequence) and not isinstance(sources, str | bytes) and 5 <= len(sources) <= 8,
        "fresh_sweep must record top five to eight WebFetch sources",
    )
    _require(list(sources) == WEBSEARCH_WEBFETCH_TOP_SOURCES, "fresh_sweep sources must match verified source set")


def _validate_upstream_artifacts(upstream_artifacts: object) -> None:
    _require(isinstance(upstream_artifacts, Mapping), "upstream_artifacts must be a mapping")
    _require(set(upstream_artifacts) == REQUIRED_UPSTREAM_FIELDS, "upstream_artifacts must match schema")
    _require(upstream_artifacts["a1_artifact"] == A1_ARTIFACT_RELATIVE_PATH, "upstream must cite A1")
    _require(upstream_artifacts["a1b_artifact"] == A1B_ARTIFACT_RELATIVE_PATH, "upstream must cite A1b")
    _require(upstream_artifacts["handoff_artifact"] == HANDOFF_ARTIFACT_RELATIVE_PATH, "upstream must cite Exp 4879")
    _require(upstream_artifacts["a1_fork_verdict"] == AIMED_AT_FORK_VERDICT, "upstream fork target mismatch")
    _require(
        upstream_artifacts["a1b_inducer_ceiling_attribution"] == AIMED_AT_INDUCER_ATTRIBUTION,
        "upstream attribution mismatch",
    )


def _validate_mapping_note(mapping_note: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(mapping_note, Mapping), "SOTA mapping note must be a mapping")
    _require(set(mapping_note) == REQUIRED_MAPPING_NOTE_FIELDS, "mapping note must match schema")
    _require(mapping_note["terminal_success"] == HONEST_VERDICT, "mapping note terminal success must match verdict")
    _require(
        mapping_note["root_cause"] == "method-level executable-code world-model representation ceiling",
        "mapping note root cause must match",
    )
    _require("INDUCER_CEILING_HARD" in str(mapping_note["summary"]), "mapping note must mention A1 verdict")
    _require("METHOD_IS_CEILING" in str(mapping_note["a1b_result"]), "mapping note must mention A1b attribution")
    source_ids = mapping_note["source_ids"]
    _require(
        isinstance(source_ids, Sequence) and not isinstance(source_ids, str | bytes) and bool(source_ids),
        "mapping note must cite source_ids",
    )
    _require(set(source_ids).issubset(set(arxiv_ids_cited)), "mapping note source_ids must be verified")


def build_research_studying_section(artifact: JsonMap | None = None) -> str:
    result = dict(artifact or build_artifact())
    validate_artifact(result)
    citations = result["citations"]
    citation_lines = "\n".join(
        f"- arXiv:{source_id} -- {citations[source_id]['title']}" for source_id in sorted(citations)
    )
    method_lines = "\n".join(
        (
            f"- **{method['method']}** ({', '.join('arXiv:' + source for source in method['source_ids'])}): "
            f"maps to {method['maps_to_frontier']} / {method['targets_fork_verdict']} + "
            f"{method['targets_inducer_attribution']}. A1b fit: {method['a1b_result_fit']} "
            f"Evidence: {method['evidence']} Experiment graft: {method['experiment_graft']} "
            f"Validation gate: {method['validation_gate']} Fails when: {method['fails_when']}"
        )
        for method in result["methods_mapped"]
    )
    flag_lines = "\n".join(flag["flag"] for flag in result["flagged_for_v451"])
    return f"""{STUDYING_SECTION_START}
## 2026-06-27 Exp 4890 - .451 V451 frontier SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4890_sota_ingestion_v451_frontier.json`.

**Preconditions:** `research-studying.md`, `research-references.md`,
`results/experiment_4882_ttt_dynamics_value_gap.json`,
`results/experiment_4883_inducer_ceiling_ab.json`, and
`results/experiment_4879_sota_ingestion_v450_frontier.json` were present. A1's
actual fork verdict is `INDUCER_CEILING_HARD`; A1b's inducer-ceiling attribution
is `METHOD_IS_CEILING`. `scripts/sweep_clusters.py` emitted the
neural-guided-search/world-model cluster 6 URLs. `scripts/sweep_semscholar.py`
was run on five focused queries; HTTP 429 rate limits were recorded rather than
promoted as evidence. Low-concurrency WebSearch/WebFetch plus direct arXiv
HTTP checks verified the top six papers listed below. `/deep-research` was not
invoked. The retired energy-as-ARC-lever, coverage/vocabulary,
exploration-strategy, selection/ranking, and perception-from-grid classes were
not re-ingested. No model load, training, leaderboard submission, or solve claim
was made; this is a no solve claim ingestion note.

**A1/A1b branch:** `INDUCER_CEILING_HARD` with `METHOD_IS_CEILING`, so `.451`
targets alternative world-model representations beyond executable code.

**Verified source set:**
{citation_lines}

**SOTA -> .451 frontier mapping:**
{method_lines}

{flag_lines}

**Bottom line for .451:** start with agent-authored decision-need targets, then
action-prefix latent adapters and latent-action world-model interfaces. Treat
reverse/counterfactual targets and verification-calibrated abstractions as
secondary representation experiments; do not re-run the retired classes.
{STUDYING_SECTION_END}"""


def build_research_references_section(artifact: JsonMap | None = None) -> str:
    result = dict(artifact or build_artifact())
    validate_artifact(result)
    citations = result["citations"]
    source_lines = "\n".join(
        (
            f"- **arXiv:{source_id} -- {citations[source_id]['title']}.** "
            "Exp 4890 use: V451 INDUCER_CEILING_HARD + METHOD_IS_CEILING "
            "source for alternative world-model representations beyond executable code."
        )
        for source_id in sorted(citations)
    )
    return f"""{REFERENCES_SECTION_START}
## 2026-06-27 Exp 4890 V451 frontier source set

Reliable-channel ingestion for `.451`, aimed at A1's actual
`INDUCER_CEILING_HARD` fork and A1b's `METHOD_IS_CEILING` attribution. These
papers are marked INGESTED for the V451 frontier roadmap handoff:

{source_lines}
{REFERENCES_SECTION_END}"""


def update_research_studying_text(text: str, artifact: JsonMap | None = None) -> str:
    section = build_research_studying_section(artifact)
    return _replace_or_insert_marked_section(text, STUDYING_SECTION_START, STUDYING_SECTION_END, section)


def update_research_references_text(text: str, artifact: JsonMap | None = None) -> str:
    section = build_research_references_section(artifact)
    return _replace_or_insert_marked_section(text, REFERENCES_SECTION_START, REFERENCES_SECTION_END, section)


def _replace_or_insert_marked_section(text: str, start_marker: str, end_marker: str, section: str) -> str:
    start = text.find(start_marker)
    if start >= 0:
        end = text.find(end_marker, start)
        _require(end >= 0, "existing marked section missing end marker")
        end += len(end_marker)
        before = text[:start].rstrip()
        tail = text[end:].lstrip()
        removed = before + ("\n\n" + tail if tail else "\n")
        insert_at = _markdown_insert_index(removed)
        if insert_at >= 0:
            return removed[:insert_at].rstrip() + "\n\n" + section + "\n\n" + removed[insert_at:].lstrip()
        return before + "\n\n" + section + ("\n\n" + tail if tail else "\n")
    insert_at = _markdown_insert_index(text)
    return (
        text[:insert_at].rstrip() + "\n\n" + section + "\n\n" + text[insert_at:].lstrip()
        if insert_at >= 0
        else text.rstrip() + "\n\n" + section + "\n"
    )


def _markdown_insert_index(text: str) -> int:
    first_marker = text.find("\n<!-- EXP")
    if first_marker >= 0:
        return first_marker + 1
    first_section = text.find("\n## ")
    return first_section + 1 if first_section >= 0 else -1


def validate_research_studying_text(text: str, artifact: JsonMap | None = None) -> None:
    result = dict(artifact or build_artifact())
    validate_artifact(result)
    start = text.find(STUDYING_SECTION_START)
    end = text.find(STUDYING_SECTION_END, start)
    _require(start >= 0 and end > start, "research-studying Exp 4890 section missing")
    section = text[start:end]
    _require("INGESTED" in section, "research-studying section must mark Exp 4890 ingested")
    _require("SOTA -> .451 frontier mapping" in section, "research-studying section missing mapping note")
    _require("flagged_for_v451" in section, "research-studying section missing .451 flags")
    _require("METHOD_IS_CEILING" in section, "research-studying section must name A1b attribution")
    _require("no solve claim" in section, "research-studying section must preserve no solve claim")
    for required in NOTE_REQUIRED_SOURCE_CITATIONS:
        _require(required in section, f"research-studying section missing {required}")


def validate_research_references_text(text: str, artifact: JsonMap | None = None) -> None:
    result = dict(artifact or build_artifact())
    validate_artifact(result)
    start = text.find(REFERENCES_SECTION_START)
    end = text.find(REFERENCES_SECTION_END, start)
    _require(start >= 0 and end > start, "research-references Exp 4890 section missing")
    section = text[start:end]
    _require("Exp 4890 V451 frontier source set" in section, "references section missing title")
    _require("INGESTED" in section, "references section must mark sources ingested")
    for source_id in REQUIRED_SOURCE_IDS:
        _require(f"arXiv:{source_id}" in section, f"references section missing arXiv:{source_id}")


def write_outputs(
    *,
    artifact_path: Path,
    studying_path: Path,
    references_path: Path,
    repo_root: Path,
    artifact: JsonMap | None = None,
) -> dict[str, object]:
    if artifact is None and not (repo_root / A1_ARTIFACT_RELATIVE_PATH).exists():
        result = build_blocked_a1_artifact()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    result = dict(artifact or build_artifact(upstream_context=load_upstream_context(repo_root)))
    validate_artifact(result)
    studying_text = studying_path.read_text(encoding="utf-8")
    updated_studying = update_research_studying_text(studying_text, result)
    validate_research_studying_text(updated_studying, result)
    references_text = references_path.read_text(encoding="utf-8")
    updated_references = update_research_references_text(references_text, result)
    validate_research_references_text(updated_references, result)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    studying_path.write_text(updated_studying, encoding="utf-8")
    references_path.write_text(updated_references, encoding="utf-8")
    return result


def main() -> int:
    root = Path(os.environ.get("CARNOT_EXP4890_ROOT", Path.cwd()))
    artifact = write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        studying_path=root / STUDYING_RELATIVE_PATH,
        references_path=root / REFERENCES_RELATIVE_PATH,
        repo_root=root,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
