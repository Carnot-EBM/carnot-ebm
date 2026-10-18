"""Exp 4900 SOTA ingestion for the .452 representation frontier.

Spec refs: REQ-ARC-WMTE-4900,
SCENARIO-ARC-WMTE-4900-V452-FRONTIER-MAPPED,
SCENARIO-ARC-WMTE-4900-BLOCKED-A1,
SCENARIO-ARC-WMTE-4900-NO-FABRICATION.

This module is aggregation-only. It reads the measured A1/A1b representation
fork artifacts, records the reliable-channel sweep provenance, and maps the
remaining third-representation SOTA methods into experiment candidates without
claiming a solve, training run, leaderboard result, or model-load result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4900_sota_ingestion_v452_frontier.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
REFERENCES_RELATIVE_PATH = "research-references.md"
A1_ARTIFACT_RELATIVE_PATH = "results/experiment_4892_decision_need_targets_value_gap.json"
A1B_ARTIFACT_RELATIVE_PATH = "results/experiment_4893_action_prefix_latent_adapter.json"
HANDOFF_ARTIFACT_RELATIVE_PATH = "results/experiment_4890_sota_ingestion_v451_frontier.json"
NOTE_PATH = "research-studying.md#exp-4900-sota-ingestion-v452-frontier"
REFERENCES_PATH = "research-references.md#exp-4900-v452-frontier-source-set"
RANDOM_SEED = 4900
DURATION_S = 0.0001
HONEST_VERDICT = "success_sota_ingestion_v452_frontier_mapped"
BLOCKED_A1_VERDICT = "blocked_a1_artifact_missing"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
AIMED_AT_FORK_VERDICT = "VALUE_GAP_REPRESENTATION_INVARIANT"
A1B_HARD_VERDICT = "VALUE_GAP_REPRESENTATION_INVARIANT_HARD"
INGESTION_TRACK = "third_representation_class_or_operator_escalation"
STUDYING_SECTION_START = "<!-- EXP4900-SOTA-INGESTION-V452-FRONTIER-START -->"
STUDYING_SECTION_END = "<!-- EXP4900-SOTA-INGESTION-V452-FRONTIER-END -->"
REFERENCES_SECTION_START = "<!-- EXP4900-V452-FRONTIER-REFERENCES-START -->"
REFERENCES_SECTION_END = "<!-- EXP4900-V452-FRONTIER-REFERENCES-END -->"
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
        "principle": "terminal prefix; mapping emitted is success_sota_ingestion_v452_frontier_mapped."
    },
    "aimed_at_fork_verdict": {
        "principle": (
            "the A1 fork verdict the ingestion targets "
            "(UNLOCKS_VALUE/MATTERS->convert to first-win; PLANNER_GAP->planning/search; "
            "INVARIANT->third representation class or escalate)."
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
    "flagged_for_v452": {
        "principle": (
            "the strongest method(s) flagged so the .452 planner reads the mapping "
            "(discover->ingest->plan->experiment)."
        )
    },
    "banned_channels_excluded": {
        "principle": (
            "true -- /deep-research NOT invoked; "
            "energy/TTA-on-code-engine/stronger-local-code-inducer/coverage/"
            "exploration/selection classes NOT re-ingested."
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
    "a1b_fork_verdict": {
        "principle": "the second-representation result used to harden the .452 branch."
    },
    "citations": {
        "principle": "HTTP-200 arXiv source metadata backing every method claim."
    },
    "fresh_sweep": {
        "principle": "records focused sweep_clusters, sweep_semscholar, WebSearch, and WebFetch provenance."
    },
    "upstream_artifacts": {
        "principle": "binds the mapping to Exp 4892 A1, Exp 4893 A1b, and Exp 4890."
    },
    "sota_to_experiment_mapping_note": {
        "principle": "states how each SOTA method becomes a .452 experiment candidate."
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
    "a1b_fork_verdict",
    "methods_mapped",
    "arxiv_ids_cited",
    "flagged_for_v452",
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
        "targets_a1_fork_verdict",
        "targets_a1b_fork_verdict",
        "a1_result_fit",
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
        "a1_decision_need_value_delta_median",
        "a1_decision_need_value_delta_ci95",
        "a1_coverage_migration_count",
        "a1b_fork_verdict",
        "a1b_honest_verdict",
        "a1b_action_prefix_value_delta_median",
        "a1b_action_prefix_value_delta_ci95",
        "aimed_at_fork_verdict",
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
        "websearch_queries",
        "websearch_webfetch_top_sources",
        "top_source_count",
        "arxiv_http_200_verified_ids",
        "deep_research_invoked",
        "retired_energy_classes_reingested",
        "tta_on_code_engine_reingested",
        "stronger_local_code_inducer_reingested",
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
        "websearch_queries",
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
        "a1_decision_need_value_delta_median",
        "a1_decision_need_value_delta_ci95",
        "a1_coverage_migration_count",
        "a1b_honest_verdict",
        "a1b_fork_verdict",
        "a1b_action_prefix_value_delta_median",
        "a1b_action_prefix_value_delta_ci95",
        "handoff_honest_verdict",
        "handoff_aimed_at_fork_verdict",
        "carried_forward_from_4890",
        "not_carried_forward_from_4890",
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
REQUIRED_MAPPED_SOURCE_IDS = frozenset({"2503.18938", "2505.08073", "2602.23997"})
REQUIRED_FETCHED_SOURCE_IDS = frozenset(
    {
        "2503.18938",
        "2505.08073",
        "2512.10016",
        "2602.16229",
        "2602.23997",
        "2603.19312",
        "2606.25421",
        "2606.26217",
    }
)
REQUIRED_TRACKS = frozenset(
    {
        "latent_action_interface",
        "reverse_counterfactual_targeter",
        "verification_calibrated_abstraction",
    }
)
RETIRED_TRACKS = frozenset(
    {
        "energy_as_arc_lever",
        "tta_on_code_engine",
        "stronger_local_code_inducer",
        "coverage_vocabulary",
        "exploration_strategy",
        "selection_ranking",
        "perception_from_grid",
        "agent_authored_decision_need_targets",
        "action_prefix_latent_adapter",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source_id}" for source_id in REQUIRED_MAPPED_SOURCE_IDS)

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
    "latent action world model adaptable agents",
    "reverse world model counterfactual targets agents",
    "verification calibrated abstraction world models agents",
    "representation invariant value gap world model planning",
]
SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS = [
    "2306.08537",
    "2406.00483",
    "2410.13232",
    "2410.23156",
    "2412.08261",
    "2503.18938",
    "2504.02252",
    "2504.13936",
    "2506.14074",
    "2510.26433",
    "2512.10016",
    "2512.13030",
    "2512.18832",
    "2601.00844",
    "2601.05230",
    "2602.16229",
    "2602.23997",
    "2604.14732",
    "2605.15725",
    "2606.00780",
    "2606.01935",
    "2606.04130",
]
SEMANTIC_SCHOLAR_RESULT = (
    "Four focused Semantic Scholar queries returned 22 unique arXiv IDs; the "
    "reverse-world-model query returned HTTP 429 and no rate-limited result was promoted as evidence."
)
WEBSEARCH_QUERIES = [
    "site:arxiv.org/abs 2503.18938 latent action world model",
    "site:arxiv.org/abs 2505.08073 reverse world model counterfactual targeter",
    "site:arxiv.org/abs 2602.23997 verification calibrated abstraction world model agents",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2503.18938",
    "https://arxiv.org/abs/2505.08073",
    "https://arxiv.org/abs/2602.23997",
    "https://arxiv.org/abs/2512.10016",
    "https://arxiv.org/abs/2602.16229",
    "https://arxiv.org/abs/2606.25421",
    "https://arxiv.org/abs/2606.26217",
    "https://arxiv.org/abs/2603.19312",
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
    "2512.10016": {
        "title": "Latent Action World Models for Control with Unlabeled Trajectories",
        "url": "https://arxiv.org/abs/2512.10016",
        "http_status": 200,
    },
    "2602.16229": {
        "title": "Factored Latent Action World Models",
        "url": "https://arxiv.org/abs/2602.16229",
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
    "Exp 4892 reported VALUE_GAP_REPRESENTATION_INVARIANT and Exp 4893 reported "
    "VALUE_GAP_REPRESENTATION_INVARIANT_HARD, so .452 targets third representation "
    "classes or an explicit operator-escalation note instead of conversion, "
    "planner-only search, TTA-on-code-engine, or stronger local code inducers."
)

DEFAULT_UPSTREAM_CONTEXT = {
    "a1_honest_verdict": "complete_decision_need_no_value_lift_VALUE_GAP_REPRESENTATION_INVARIANT",
    "a1_fork_verdict": AIMED_AT_FORK_VERDICT,
    "a1_decision_need_value_delta_median": -0.101866,
    "a1_decision_need_value_delta_ci95": [-0.227708, 0.025266],
    "a1_coverage_migration_count": 0,
    "a1b_honest_verdict": "complete_action_prefix_latent_no_value_lift_representation_invariant_hard",
    "a1b_fork_verdict": A1B_HARD_VERDICT,
    "a1b_action_prefix_value_delta_median": 0.0,
    "a1b_action_prefix_value_delta_ci95": [-0.134887, 0.025266],
    "handoff_honest_verdict": "success_sota_ingestion_v451_frontier_mapped",
    "handoff_aimed_at_fork_verdict": "INDUCER_CEILING_HARD",
    "ingestion_track": INGESTION_TRACK,
}

FLAGGED_FOR_V452 = [
    {
        "candidate": "latent_action_interface",
        "flag": "flagged_for_v452: latent_action_interface (arXiv:2503.18938)",
        "source_ids": ["2503.18938"],
        "maps_to_frontier": ".452",
        "priority": 1,
    },
    {
        "candidate": "reverse_counterfactual_targeter",
        "flag": "flagged_for_v452: reverse_counterfactual_targeter (arXiv:2505.08073)",
        "source_ids": ["2505.08073"],
        "maps_to_frontier": ".452",
        "priority": 2,
    },
    {
        "candidate": "verification_calibrated_abstraction",
        "flag": "flagged_for_v452: verification_calibrated_abstraction (arXiv:2602.23997)",
        "source_ids": ["2602.23997"],
        "maps_to_frontier": ".452",
        "priority": 3,
    },
]

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Latent-action adaptable representation interface",
        "track": "latent_action_interface",
        "source_ids": ["2503.18938"],
        "maps_to_frontier": ".452",
        "targets_a1_fork_verdict": AIMED_AT_FORK_VERDICT,
        "targets_a1b_fork_verdict": A1B_HARD_VERDICT,
        "a1_result_fit": (
            "Exp 4892 found no value lift from decision-need target tables, so "
            "the next representation must move below hand-authored target rows."
        ),
        "a1b_result_fit": (
            "Exp 4893 found no value lift from action-prefix latents, so the "
            "adapter must expose a distinct latent action interface rather than "
            "another prefix-delta table."
        ),
        "evidence": (
            "arXiv:2503.18938 learns self-supervised latent actions and conditions "
            "an adaptable world model on those action tokens for transfer with "
            "limited interactions."
        ),
        "experiment_graft": (
            "Infer latent action tokens from cold ARC transitions, align E3 legal "
            "controls to those tokens, and feed the latent-action state into the "
            "held-out transition scorer without converting it through the failed "
            "decision-need or action-prefix table formats."
        ),
        "validation_gate": (
            "Promote only if latent actions improve held-out changed-cell value "
            "accuracy over both Exp 4892 and Exp 4893 on the same split while "
            "keeping the positive control non-degenerate and the banked answer out."
        ),
        "fails_when": (
            "The latent actions collapse across mechanics, fail to align with legal "
            "controls, or require more adaptation interactions than the ARC budget permits."
        ),
        "roadmap_candidate": FLAGGED_FOR_V452[0]["flag"],
        "retired_class_reingested": False,
    },
    {
        "method": "Reverse-counterfactual representation targeter",
        "track": "reverse_counterfactual_targeter",
        "source_ids": ["2505.08073"],
        "maps_to_frontier": ".452",
        "targets_a1_fork_verdict": AIMED_AT_FORK_VERDICT,
        "targets_a1b_fork_verdict": A1B_HARD_VERDICT,
        "a1_result_fit": (
            "Exp 4892 shows direct decision-need rows did not find the missing "
            "changed values, so the next target should be induced from failed "
            "counterfactual action effects."
        ),
        "a1b_result_fit": (
            "Exp 4893 shows forward prefix latents remained flat, so this branch "
            "asks the reverse question: which state fact would make the desired "
            "effect rational?"
        ),
        "evidence": (
            "arXiv:2505.08073 augments model-based reinforcement learning with a "
            "Reverse World Model that predicts what state would make a counterfactual "
            "action preferred."
        ),
        "experiment_graft": (
            "For each hard A1/A1b transition miss, ask a reverse model for the "
            "missing pre-state or register fact that would make the desired action "
            "effect valid, then materialize only verifier-checkable facts into a "
            "new representation probe."
        ),
        "validation_gate": (
            "Accept only reverse targets that reduce held-out dynamics errors on "
            "the same split, remain oracle-distinct from level completion, and do "
            "not leak the terminal solution prefix."
        ),
        "fails_when": (
            "The counterfactual state is unreachable, merely explains the policy "
            "without improving transition prediction, or smuggles the banked answer "
            "into the target."
        ),
        "roadmap_candidate": FLAGGED_FOR_V452[1]["flag"],
        "retired_class_reingested": False,
    },
    {
        "method": "Verification-calibrated abstraction substrate",
        "track": "verification_calibrated_abstraction",
        "source_ids": ["2602.23997"],
        "maps_to_frontier": ".452",
        "targets_a1_fork_verdict": AIMED_AT_FORK_VERDICT,
        "targets_a1b_fork_verdict": A1B_HARD_VERDICT,
        "a1_result_fit": (
            "Exp 4892's value gap staying flat means the table representation did "
            "not expose reliable state variables."
        ),
        "a1b_result_fit": (
            "Exp 4893's second representation also stayed flat, so .452 should "
            "make abstraction reliability explicit before any fact affects planning."
        ),
        "evidence": (
            "arXiv:2602.23997 proposes foundation world models with adaptive formal "
            "verification, online abstraction calibration, and verifier-guided "
            "test-time synthesis."
        ),
        "experiment_graft": (
            "Insert a persistent abstraction state beside the executable engine, "
            "attach verifier-calibrated confidence to each abstract fact, and let "
            "only calibrated facts influence held-out transition prediction."
        ),
        "validation_gate": (
            "Promote only if abstraction confidence predicts held-out engine "
            "mismatches and calibrated facts improve changed-cell value accuracy "
            "without becoming a selection/ranking rerun."
        ),
        "fails_when": (
            "The abstraction is too coarse for ARC mechanics, calibration only "
            "tracks seen prefixes, or verifier hooks become another retired "
            "selection/ranking class."
        ),
        "roadmap_candidate": FLAGGED_FOR_V452[2]["flag"],
        "retired_class_reingested": False,
    },
]

DEFAULT_FRESH_SWEEP = {
    "filtered_track": (
        "V452 frontier for VALUE_GAP_REPRESENTATION_INVARIANT + "
        "VALUE_GAP_REPRESENTATION_INVARIANT_HARD: third representation classes or "
        "operator escalation"
    ),
    "cluster_ids": [6],
    "cluster_urls": [CLUSTER_6_URL, CLUSTER_6_PAGE2_URL],
    "cluster_top_arxiv_ids": CLUSTER_TOP_ARXIV_IDS,
    "semantic_scholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "semantic_scholar_result": SEMANTIC_SCHOLAR_RESULT,
    "semantic_scholar_unique_arxiv_ids": SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
    "websearch_queries": WEBSEARCH_QUERIES,
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
    "a1_decision_need_value_delta_median": DEFAULT_UPSTREAM_CONTEXT["a1_decision_need_value_delta_median"],
    "a1_decision_need_value_delta_ci95": DEFAULT_UPSTREAM_CONTEXT["a1_decision_need_value_delta_ci95"],
    "a1_coverage_migration_count": DEFAULT_UPSTREAM_CONTEXT["a1_coverage_migration_count"],
    "a1b_fork_verdict": A1B_HARD_VERDICT,
    "a1b_honest_verdict": DEFAULT_UPSTREAM_CONTEXT["a1b_honest_verdict"],
    "a1b_action_prefix_value_delta_median": DEFAULT_UPSTREAM_CONTEXT["a1b_action_prefix_value_delta_median"],
    "a1b_action_prefix_value_delta_ci95": DEFAULT_UPSTREAM_CONTEXT["a1b_action_prefix_value_delta_ci95"],
    "aimed_at_fork_verdict": AIMED_AT_FORK_VERDICT,
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
    "websearch_queries": WEBSEARCH_QUERIES,
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
    "top_source_count": len(WEBSEARCH_WEBFETCH_TOP_SOURCES),
    "arxiv_http_200_verified_ids": [
        f"https://arxiv.org/abs/{source_id}" for source_id in sorted(REQUIRED_FETCHED_SOURCE_IDS)
    ],
    "deep_research_invoked": False,
    "retired_energy_classes_reingested": False,
    "tta_on_code_engine_reingested": False,
    "stronger_local_code_inducer_reingested": False,
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
    "a1_decision_need_value_delta_median": DEFAULT_UPSTREAM_CONTEXT["a1_decision_need_value_delta_median"],
    "a1_decision_need_value_delta_ci95": DEFAULT_UPSTREAM_CONTEXT["a1_decision_need_value_delta_ci95"],
    "a1_coverage_migration_count": DEFAULT_UPSTREAM_CONTEXT["a1_coverage_migration_count"],
    "a1b_honest_verdict": DEFAULT_UPSTREAM_CONTEXT["a1b_honest_verdict"],
    "a1b_fork_verdict": DEFAULT_UPSTREAM_CONTEXT["a1b_fork_verdict"],
    "a1b_action_prefix_value_delta_median": DEFAULT_UPSTREAM_CONTEXT["a1b_action_prefix_value_delta_median"],
    "a1b_action_prefix_value_delta_ci95": DEFAULT_UPSTREAM_CONTEXT["a1b_action_prefix_value_delta_ci95"],
    "handoff_honest_verdict": DEFAULT_UPSTREAM_CONTEXT["handoff_honest_verdict"],
    "handoff_aimed_at_fork_verdict": DEFAULT_UPSTREAM_CONTEXT["handoff_aimed_at_fork_verdict"],
    "carried_forward_from_4890": [
        "latent_action_world_model_adapter",
        "reverse_counterfactual_world_model_targets",
        "verification_calibrated_abstraction_substrate",
    ],
    "not_carried_forward_from_4890": [
        "agent_authored_decision_need_targets",
        "action_prefix_latent_adapter",
        "energy-as-ARC-lever",
        "TTA-on-code-engine",
        "stronger-local-code-inducer",
        "coverage/vocabulary",
        "exploration-strategy",
        "selection/ranking",
        "perception-from-grid",
    ],
}

DEFAULT_MAPPING_NOTE = {
    "summary": (
        "Exp 4892 and Exp 4893 both failed to close the changed-cell value gap; "
        ".452 should map the remaining third-representation candidates or issue "
        "an operator-escalation note."
    ),
    "terminal_success": HONEST_VERDICT,
    "source_ids": sorted(REQUIRED_MAPPED_SOURCE_IDS),
    "root_cause": "deep representation-invariant changed-cell value gap under two tested non-code representations",
    "planner_instruction": (
        "Start with the latent-action interface, then reverse-counterfactual "
        "targeting, then verification-calibrated abstraction. Promote only methods "
        "that lift held-out changed-cell value accuracy over both Exp 4892 and Exp 4893."
    ),
    "a1_result": "Exp 4892 fork_verdict=VALUE_GAP_REPRESENTATION_INVARIANT.",
    "a1b_result": "Exp 4893 fork_verdict=VALUE_GAP_REPRESENTATION_INVARIANT_HARD.",
    "not_carried_forward": (
        "Do not re-promote decision-need targets, action-prefix latent tables, "
        "energy, TTA-on-code-engine, stronger local code inducers, coverage/"
        "vocabulary, exploration, selection/ranking, or perception-from-grid classes."
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


REPRODUCIBILITY_CHECKSUM = source_set_checksum(
    CITATIONS,
    DEFAULT_METHODS_MAPPED,
    FLAGGED_FOR_V452,
    DEFAULT_UPSTREAM_ARTIFACTS,
    DEFAULT_MAPPING_NOTE,
)


def select_ingestion_track(a1_artifact: JsonMap, a1b_artifact: JsonMap) -> str:
    a1_verdict = a1_artifact.get("fork_verdict")
    a1b_verdict = a1b_artifact.get("fork_verdict")
    if a1_verdict == "REPRESENTATION_UNLOCKS_VALUE" or a1b_verdict == "REPRESENTATION_MATTERS":
        return "first_win_conversion_over_value_accurate_representation"
    if a1_verdict == "PLANNER_GAP":
        return "neural_guided_planning_search"
    if a1_verdict == AIMED_AT_FORK_VERDICT and a1b_verdict == A1B_HARD_VERDICT:
        return INGESTION_TRACK
    raise ValueError(f"unsupported A1/A1b representation fork: {a1_verdict!r} / {a1b_verdict!r}")


def derive_upstream_context(a1_artifact: JsonMap, a1b_artifact: JsonMap, handoff_artifact: JsonMap) -> dict[str, object]:
    track = select_ingestion_track(a1_artifact, a1b_artifact)
    return {
        "a1_honest_verdict": a1_artifact.get("honest_verdict", ""),
        "a1_fork_verdict": a1_artifact.get("fork_verdict"),
        "a1_decision_need_value_delta_median": a1_artifact.get("decision_need_value_accuracy_delta_median"),
        "a1_decision_need_value_delta_ci95": a1_artifact.get("decision_need_value_accuracy_delta_ci95", []),
        "a1_coverage_migration_count": int(a1_artifact.get("coverage_migration_count") or 0),
        "a1b_honest_verdict": a1b_artifact.get("honest_verdict", ""),
        "a1b_fork_verdict": a1b_artifact.get("fork_verdict"),
        "a1b_action_prefix_value_delta_median": a1b_artifact.get("action_prefix_value_accuracy_delta_median"),
        "a1b_action_prefix_value_delta_ci95": a1b_artifact.get("action_prefix_value_accuracy_delta_ci95", []),
        "handoff_honest_verdict": handoff_artifact.get("honest_verdict", ""),
        "handoff_aimed_at_fork_verdict": handoff_artifact.get("aimed_at_fork_verdict", ""),
        "ingestion_track": track,
    }


def load_upstream_context(repo_root: Path) -> dict[str, object]:
    a1 = json.loads((repo_root / A1_ARTIFACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    a1b = json.loads((repo_root / A1B_ARTIFACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    handoff = json.loads((repo_root / HANDOFF_ARTIFACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    return derive_upstream_context(a1, a1b, handoff)


def _contextual_preconditions(upstream_context: JsonMap) -> dict[str, object]:
    preconditions = dict(DEFAULT_PRECONDITIONS_CHECKED)
    preconditions.update(
        {
            "a1_fork_verdict": upstream_context["a1_fork_verdict"],
            "a1_honest_verdict": upstream_context["a1_honest_verdict"],
            "a1_decision_need_value_delta_median": upstream_context["a1_decision_need_value_delta_median"],
            "a1_decision_need_value_delta_ci95": upstream_context["a1_decision_need_value_delta_ci95"],
            "a1_coverage_migration_count": upstream_context["a1_coverage_migration_count"],
            "a1b_fork_verdict": upstream_context["a1b_fork_verdict"],
            "a1b_honest_verdict": upstream_context["a1b_honest_verdict"],
            "a1b_action_prefix_value_delta_median": upstream_context["a1b_action_prefix_value_delta_median"],
            "a1b_action_prefix_value_delta_ci95": upstream_context["a1b_action_prefix_value_delta_ci95"],
            "ingestion_track": upstream_context["ingestion_track"],
        }
    )
    return preconditions


def _contextual_upstream_artifacts(upstream_context: JsonMap) -> dict[str, object]:
    upstream = dict(DEFAULT_UPSTREAM_ARTIFACTS)
    upstream.update(
        {
            "a1_honest_verdict": upstream_context["a1_honest_verdict"],
            "a1_fork_verdict": upstream_context["a1_fork_verdict"],
            "a1_decision_need_value_delta_median": upstream_context["a1_decision_need_value_delta_median"],
            "a1_decision_need_value_delta_ci95": upstream_context["a1_decision_need_value_delta_ci95"],
            "a1_coverage_migration_count": upstream_context["a1_coverage_migration_count"],
            "a1b_honest_verdict": upstream_context["a1b_honest_verdict"],
            "a1b_fork_verdict": upstream_context["a1b_fork_verdict"],
            "a1b_action_prefix_value_delta_median": upstream_context["a1b_action_prefix_value_delta_median"],
            "a1b_action_prefix_value_delta_ci95": upstream_context["a1b_action_prefix_value_delta_ci95"],
            "handoff_honest_verdict": upstream_context["handoff_honest_verdict"],
            "handoff_aimed_at_fork_verdict": upstream_context["handoff_aimed_at_fork_verdict"],
        }
    )
    return upstream


def build_artifact(upstream_context: JsonMap | None = None) -> dict[str, object]:
    context = dict(upstream_context or DEFAULT_UPSTREAM_CONTEXT)
    upstream_artifacts = _contextual_upstream_artifacts(context)
    checksum = source_set_checksum(
        CITATIONS,
        DEFAULT_METHODS_MAPPED,
        FLAGGED_FOR_V452,
        upstream_artifacts,
        DEFAULT_MAPPING_NOTE,
    )
    return {
        "honest_verdict": HONEST_VERDICT,
        "aimed_at_fork_verdict": context["a1_fork_verdict"],
        "a1b_fork_verdict": context["a1b_fork_verdict"],
        "methods_mapped": DEFAULT_METHODS_MAPPED,
        "arxiv_ids_cited": sorted(REQUIRED_FETCHED_SOURCE_IDS),
        "flagged_for_v452": FLAGGED_FOR_V452,
        "banned_channels_excluded": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": _contextual_preconditions(context),
        "citations": CITATIONS,
        "fresh_sweep": DEFAULT_FRESH_SWEEP,
        "upstream_artifacts": upstream_artifacts,
        "sota_to_experiment_mapping_note": DEFAULT_MAPPING_NOTE,
        "note_path": NOTE_PATH,
        "references_path": REFERENCES_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_S,
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
    }


def build_blocked_artifact() -> dict[str, object]:
    preconditions = dict(DEFAULT_PRECONDITIONS_CHECKED)
    preconditions.update(
        {
            "a1_artifact_present": False,
            "a1_fork_verdict_read": False,
            "a1_fork_verdict": "",
            "a1_honest_verdict": "",
            "a1_decision_need_value_delta_median": None,
            "a1_decision_need_value_delta_ci95": [],
            "a1_coverage_migration_count": 0,
            "a1b_artifact_present": False,
            "a1b_fork_verdict": "",
            "a1b_honest_verdict": "",
            "a1b_action_prefix_value_delta_median": None,
            "a1b_action_prefix_value_delta_ci95": [],
            "handoff_artifact_present": False,
            "aimed_at_fork_verdict": "",
            "branch_reason": "Exp 4892 A1 artifact missing; no .452 fork mapping fabricated.",
            "ingestion_track": "blocked",
        }
    )
    return {
        "honest_verdict": BLOCKED_A1_VERDICT,
        "aimed_at_fork_verdict": "",
        "a1b_fork_verdict": "",
        "methods_mapped": [],
        "arxiv_ids_cited": [],
        "flagged_for_v452": [],
        "banned_channels_excluded": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "citations": {},
        "fresh_sweep": DEFAULT_FRESH_SWEEP,
        "upstream_artifacts": {},
        "sota_to_experiment_mapping_note": {},
        "note_path": NOTE_PATH,
        "references_path": REFERENCES_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_S,
        "reproducibility_checksum": "sha256:blocked_a1_artifact_missing",
        "field_principles": FIELD_PRINCIPLES,
    }


def validate_artifact(artifact: JsonMap) -> None:
    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict lacks terminal prefix")
    if verdict == BLOCKED_A1_VERDICT:
        _require(artifact.get("methods_mapped") == [], "blocked artifact must not map methods")
        _require(artifact.get("banned_channels_excluded") is True, "blocked artifact must exclude banned channels")
        _require(artifact.get("preconditions_checked", {}).get("a1_artifact_present") is False, "blocked artifact must name missing A1")
        return

    _require(set(artifact) == set(REQUIRED_ARTIFACT_FIELDS), "artifact fields do not match REQ-ARC-WMTE-4900")
    _require(artifact["honest_verdict"] == HONEST_VERDICT, "honest_verdict must be the .452 success mapping")
    _require(artifact["aimed_at_fork_verdict"] == AIMED_AT_FORK_VERDICT, "aimed_at_fork_verdict must match A1")
    _require(artifact["a1b_fork_verdict"] == A1B_HARD_VERDICT, "a1b_fork_verdict must match A1b")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate must be aggregation-only")
    _require(artifact["duration_s"] == DURATION_S, "duration_s must use the aggregation floor")
    _require(artifact["banned_channels_excluded"] is True, "banned_channels_excluded must be true")
    _require(artifact["arxiv_ids_cited"] == sorted(REQUIRED_FETCHED_SOURCE_IDS), "arxiv_ids_cited must match verified sources")
    _require(set(artifact["field_principles"]) == REQUIRED_PRINCIPLE_FIELDS, "field_principles must match REQ-ARC-WMTE-4900")

    citations = artifact["citations"]
    _require(set(citations) == REQUIRED_FETCHED_SOURCE_IDS, "citations must cover the verified source set")
    for source_id, citation in citations.items():
        _require(set(citation) == REQUIRED_CITATION_FIELDS, "citation fields must be title/url/http_status")
        _require(citation["url"] == f"https://arxiv.org/abs/{source_id}", "citation url must match arXiv ID")
        _require(citation["http_status"] == 200, "http_status must be HTTP 200")

    methods = artifact["methods_mapped"]
    _require(3 <= len(methods) <= 5, "methods_mapped must contain three to five methods")
    mapped_sources: set[str] = set()
    mapped_tracks: set[str] = set()
    for method in methods:
        _require(set(method) == REQUIRED_METHOD_FIELDS, "method fields must match the .452 contract")
        method_sources = set(method["source_ids"])
        _require(method_sources <= REQUIRED_FETCHED_SOURCE_IDS, "method source_ids must be verified citations")
        _require(method["maps_to_frontier"] == ".452", "method must map to .452")
        _require(method["targets_a1_fork_verdict"] == AIMED_AT_FORK_VERDICT, "method A1 fork verdict mismatch")
        _require(method["targets_a1b_fork_verdict"] == A1B_HARD_VERDICT, "method A1b fork verdict mismatch")
        _require(method["track"] not in RETIRED_TRACKS, "method track must not be retired")
        _require(method["retired_class_reingested"] is False, "retired class flag must be false")
        _require(method["experiment_graft"], "experiment_graft is required")
        _require(method["validation_gate"], "validation_gate is required")
        _require(method["fails_when"], "fails_when is required")
        mapped_sources.update(method_sources)
        mapped_tracks.add(method["track"])
    _require(mapped_sources == REQUIRED_MAPPED_SOURCE_IDS, "mapped methods must cite the remaining .450 candidates")
    _require(mapped_tracks == REQUIRED_TRACKS, "mapped tracks must be the third-representation set")

    flags = artifact["flagged_for_v452"]
    _require(flags == FLAGGED_FOR_V452, "flagged_for_v452 must match the planner handoff; stale flags are rejected")
    for flag in flags:
        _require("flagged_for_v452" in flag.get("flag", ""), "stale flag must not target an older frontier")

    preconditions = artifact["preconditions_checked"]
    _require(set(preconditions) == REQUIRED_PRECONDITION_FIELDS, "preconditions_checked fields mismatch")
    _require(preconditions["a1_artifact_present"] is True, "A1 artifact must be present")
    _require(preconditions["a1_fork_verdict_read"] is True, "A1 fork verdict must be read")
    _require(preconditions["a1_fork_verdict"] == AIMED_AT_FORK_VERDICT, "A1 fork verdict mismatch")
    _require(preconditions["a1b_fork_verdict"] == A1B_HARD_VERDICT, "A1b fork verdict mismatch")
    _require(preconditions["deep_research_invoked"] is False, "deep-research is banned")
    _require(preconditions["retired_energy_classes_reingested"] is False, "retired energy classes are banned")
    _require(preconditions["tta_on_code_engine_reingested"] is False, "TTA-on-code-engine is banned")
    _require(preconditions["stronger_local_code_inducer_reingested"] is False, "stronger local code inducer is banned")
    _require(preconditions["coverage_vocabulary_reingested"] is False, "coverage/vocabulary is banned")
    _require(preconditions["exploration_strategy_reingested"] is False, "exploration strategy is banned")
    _require(preconditions["selection_ranking_reingested"] is False, "selection/ranking is banned")
    _require(preconditions["perception_from_grid_reingested"] is False, "perception-from-grid is banned")
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
    expected_checksum = source_set_checksum(citations, methods, flags, upstream, mapping_note)
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
    flag_lines = "\n".join(f"- {flag['flag']}" for flag in artifact["flagged_for_v452"])
    body = f"""## Exp 4900 - .452 representation fork SOTA ingestion - INGESTED

- Honest verdict: `{artifact['honest_verdict']}`
- Aimed at A1 fork: `{artifact['aimed_at_fork_verdict']}`
- A1b fork: `{artifact['a1b_fork_verdict']}`
- Branch: `{artifact['preconditions_checked']['ingestion_track']}`
- Banned channel note: `/deep-research` not invoked; no solve claim; no model load; retired classes excluded.

### Methods
{method_lines}

### Planner Flags
{flag_lines}
"""
    return _replace_section(text, STUDYING_SECTION_START, STUDYING_SECTION_END, body)


def update_research_references_text(text: str, artifact: JsonMap) -> str:
    reference_lines = "\n".join(
        f"- arXiv:{source_id} - {citation['title']} - {citation['url']} - HTTP {citation['http_status']}"
        for source_id, citation in sorted(artifact["citations"].items())
    )
    body = f"""## Exp 4900 V452 representation-fork source set

Reliable-channel source set for `{artifact['honest_verdict']}`:

{reference_lines}
"""
    return _replace_section(text, REFERENCES_SECTION_START, REFERENCES_SECTION_END, body)


def validate_research_studying_text(text: str, artifact: JsonMap) -> None:
    _require(STUDYING_SECTION_START in text and STUDYING_SECTION_END in text, "research-studying section missing")
    _require(str(artifact["honest_verdict"]) in text, "research-studying honest verdict missing")
    _require("flagged_for_v452" in text, "research-studying flags missing")
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
    if not (repo_root / A1_ARTIFACT_RELATIVE_PATH).exists():
        artifact = build_blocked_artifact()
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
    repo_root = Path(os.environ.get("CARNOT_EXP4900_ROOT", Path(__file__).resolve().parents[2]))
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
