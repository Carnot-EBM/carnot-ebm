"""Exp 4879 SOTA ingestion for the .450 generation-wall frontier.

Spec refs: REQ-ARC-WMTE-4879,
SCENARIO-ARC-WMTE-4879-V450-FRONTIER-MAPPED,
SCENARIO-ARC-WMTE-4879-NO-FABRICATION.

This is an aggregation-only mapping artifact. It reads the measured A1 and A1b
artifacts, preserves their caveats, and maps the next `.450` SOTA methods
without claiming a solve, a training run, or a model-load result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4879_sota_ingestion_v450_frontier.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
REFERENCES_RELATIVE_PATH = "research-references.md"
A1_ARTIFACT_RELATIVE_PATH = "results/experiment_4871_generation_wall_fork_probe_gpu_fixed.json"
A1B_ARTIFACT_RELATIVE_PATH = "results/experiment_4872_cegis_world_model_refinement.json"
HANDOFF_ARTIFACT_RELATIVE_PATH = "results/experiment_4868_sota_ingestion_v449_frontier.json"
NOTE_PATH = "research-studying.md#exp-4879-sota-ingestion-v450-frontier"
REFERENCES_PATH = "research-references.md#exp-4879-v450-frontier-source-set"
RANDOM_SEED = 4879
DURATION_S = 0.0001
HONEST_VERDICT = "success_sota_ingestion_v450_frontier_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
AIMED_AT_FORK_VERDICT = "INDUCER_CEILING"
STUDYING_SECTION_START = "<!-- EXP4879-SOTA-INGESTION-V450-FRONTIER-START -->"
STUDYING_SECTION_END = "<!-- EXP4879-SOTA-INGESTION-V450-FRONTIER-END -->"
REFERENCES_SECTION_START = "<!-- EXP4879-V450-FRONTIER-REFERENCES-START -->"
REFERENCES_SECTION_END = "<!-- EXP4879-V450-FRONTIER-REFERENCES-END -->"
TERMINAL_PREFIXES = (
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
        "principle": (
            "terminal prefix; mapping emitted is "
            "success_sota_ingestion_v450_frontier_mapped."
        )
    },
    "methods_mapped": {
        "principle": (
            "the strongest 3-5 methods aimed at A1's ACTUAL fork verdict + "
            "A1b's CEGIS result, each with a real arXiv ID."
        )
    },
    "arxiv_ids_cited": {
        "principle": (
            "every method claim must cite a verifiable arXiv ID "
            "(no fabrication -- adversarial_verify bar)."
        )
    },
    "aimed_at_fork_verdict": {
        "principle": (
            "the A1 fork_verdict the ingestion targets (INDUCER_CEILING -> "
            "inducer/CEGIS scale; GUIDANCE/PLANNER -> planning/search)."
        )
    },
    "flagged_for_v450": {
        "principle": "the strongest method(s) flagged so the .450 planner reads the mapping."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (0.0001s floor)."
    },
}
FIELD_PRINCIPLES = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "preconditions_checked": {
        "principle": "records reliable-channel checks and explicitly excludes banned channels."
    },
    "citations": {
        "principle": "HTTP-200 arXiv source metadata backing every method claim."
    },
    "fresh_sweep": {
        "principle": "records focused sweep_clusters, sweep_semscholar, and WebFetch provenance."
    },
    "upstream_artifacts": {
        "principle": "binds the mapping to Exp 4871 A1, Exp 4872 A1b, and Exp 4868."
    },
    "sota_to_experiment_mapping_note": {
        "principle": "states how each SOTA method becomes a .450 experiment candidate."
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
    "methods_mapped",
    "arxiv_ids_cited",
    "aimed_at_fork_verdict",
    "flagged_for_v450",
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
        "a1b_result_fit",
        "evidence",
        "experiment_graft",
        "validation_gate",
        "sovereignty_note",
        "fails_when",
        "roadmap_candidate",
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
        "a1_source_fork_verdict",
        "a1_computed_fork_verdict",
        "a1_genuinely_diagnostic",
        "a1_positive_control_caveat",
        "aimed_at_fork_verdict",
        "a1b_delta_median",
        "a1b_delta_ci95",
        "a1b_cegis_moved_accuracy",
        "current_cegis_promoted",
        "branch_reason",
        "sweep_clusters_used",
        "sweep_cluster_ids",
        "sweep_cluster_urls",
        "sweep_semscholar_used",
        "sweep_semscholar_queries",
        "sweep_semscholar_result",
        "semantic_scholar_unique_arxiv_ids",
        "websearch_webfetch_used",
        "websearch_webfetch_top_sources",
        "top_source_count",
        "arxiv_http_200_verified_ids",
        "deep_research_invoked",
        "retired_coverage_classes_reingested",
        "exploration_strategy_reingested",
        "energy_classes_reingested",
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
        "a1_source_fork_verdict",
        "a1_computed_fork_verdict",
        "a1_genuinely_diagnostic",
        "a1_positive_control_caveat",
        "a1_median_engine_heldout_accuracy",
        "a1_coverage_migration_count",
        "a1b_honest_verdict",
        "a1b_delta_median",
        "a1b_delta_ci95",
        "a1b_cegis_moved_accuracy",
        "a1b_positive_control_passed",
        "handoff_honest_verdict",
        "handoff_aimed_at_fork_verdict",
        "carried_forward_from_4868",
        "not_carried_forward_from_4868",
    }
)
REQUIRED_MAPPING_NOTE_FIELDS = frozenset(
    {
        "summary",
        "terminal_success",
        "source_ids",
        "root_cause",
        "planner_instruction",
        "a1_caveat",
        "a1b_result",
        "not_carried_forward",
    }
)
REQUIRED_SOURCE_IDS = frozenset(
    {
        "2203.13474",
        "2506.02918",
        "2507.03160",
        "2507.15877",
        "2509.03956",
        "2605.05138",
        "2606.25421",
        "2606.26217",
    }
)
REQUIRED_TRACKS = frozenset(
    {
        "test_time_dynamics_adaptation",
        "family_b_vs_local_open_code_inducer",
        "agent_authored_world_model_targets",
        "action_prefix_world_model_adapter",
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
]
SEMANTIC_SCHOLAR_QUERIES = [
    "test time dynamics adaptation world model language agents",
    "world model test time adaptation embodied agents",
    "execution guided neural program synthesis ARC test time fine tuning",
    "local open code model program synthesis repair verifier feedback",
    "small language models code generation program synthesis benchmarks",
]
SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS = [
    "2108.07732",
    "2305.01210",
    "2403.03997",
    "2502.14604",
    "2503.00686",
    "2505.13938",
    "2506.06137",
    "2506.11121",
    "2507.03160",
    "2509.03956",
    "2511.04847",
    "2606.05684",
]
SEMANTIC_SCHOLAR_RESULT = (
    "Five focused queries: the first query returned 5 arXiv IDs, queries 2-4 "
    "returned HTTP 429, and the small-code-model query completed; 12 unique "
    "arXiv IDs were recorded."
)
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2203.13474",
    "https://arxiv.org/abs/2506.02918",
    "https://arxiv.org/abs/2507.03160",
    "https://arxiv.org/abs/2507.15877",
    "https://arxiv.org/abs/2509.03956",
    "https://arxiv.org/abs/2605.05138",
    "https://arxiv.org/abs/2606.25421",
    "https://arxiv.org/abs/2606.26217",
]

CITATIONS = {
    "2203.13474": {
        "title": "CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis",
        "url": "https://arxiv.org/abs/2203.13474",
        "http_status": 200,
    },
    "2506.02918": {
        "title": "World Modelling Improves Language Model Agents",
        "url": "https://arxiv.org/abs/2506.02918",
        "http_status": 200,
    },
    "2507.03160": {
        "title": "Assessing Small Language Models for Code Generation: An Empirical Study with Benchmarks",
        "url": "https://arxiv.org/abs/2507.03160",
        "http_status": 200,
    },
    "2507.15877": {
        "title": (
            "Out-of-Distribution Generalization in the ARC-AGI Domain: Comparing "
            "Execution-Guided Neural Program Synthesis and Test-Time Fine-Tuning"
        ),
        "url": "https://arxiv.org/abs/2507.15877",
        "http_status": 200,
    },
    "2509.03956": {
        "title": "World Model Implanting for Test-time Adaptation of Embodied Agents",
        "url": "https://arxiv.org/abs/2509.03956",
        "http_status": 200,
    },
    "2605.05138": {
        "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
        "url": "https://arxiv.org/abs/2605.05138",
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
    "A1 reported fork_verdict=null because the positive control failed, but the "
    "computed fork table is INDUCER_CEILING; A1b CEGIS delta was 0.0 with CI95 "
    "[0.0, 0.0], so .450 maps the next inducer candidates instead of promoting "
    "the current CEGIS loop."
)

DEFAULT_UPSTREAM_CONTEXT = {
    "a1_honest_verdict": "complete_generation_wall_fork_probe_retired_positive_control_failed",
    "a1_source_fork_verdict": None,
    "a1_computed_fork_verdict": AIMED_AT_FORK_VERDICT,
    "a1_genuinely_diagnostic": False,
    "a1_positive_control_caveat": True,
    "a1_median_engine_heldout_accuracy": 0.0,
    "a1_coverage_migration_count": 0,
    "a1_n_games_measured": 9,
    "a1_positive_control_migrated": False,
    "a1b_honest_verdict": "complete_cegis_no_heldout_accuracy_lift_residual_positive_control_failed",
    "a1b_delta_median": 0.0,
    "a1b_delta_ci95": [0.0, 0.0],
    "a1b_cegis_moved_accuracy": False,
    "a1b_positive_control_passed": False,
    "handoff_honest_verdict": "success_sota_ingestion_v449_frontier_mapped",
    "handoff_aimed_at_fork_verdict": AIMED_AT_FORK_VERDICT,
    "handoff_flagged_candidates": [
        "test_time_world_model_adaptation_loop",
        "family_b_executable_world_model_inducer_ladder",
        "cegis_world_model_refinement_loop",
    ],
}

FLAGGED_FOR_V450 = [
    {
        "candidate": "test_time_dynamics_adaptation_loop",
        "flag": (
            "flagged_for_v450: test_time_dynamics_adaptation_loop "
            "(arXiv:2506.02918 + arXiv:2509.03956 + arXiv:2507.15877)"
        ),
        "source_ids": ["2506.02918", "2509.03956", "2507.15877"],
        "maps_to_frontier": ".450",
        "priority": 1,
    },
    {
        "candidate": "family_b_vs_local_open_code_inducer_ab",
        "flag": (
            "flagged_for_v450: family_b_vs_local_open_code_inducer_ab "
            "(arXiv:2605.05138 + arXiv:2507.03160 + arXiv:2203.13474)"
        ),
        "source_ids": ["2605.05138", "2507.03160", "2203.13474"],
        "maps_to_frontier": ".450",
        "priority": 2,
    },
    {
        "candidate": "agent_authored_world_model_targets",
        "flag": (
            "flagged_for_v450: agent_authored_world_model_targets "
            "(arXiv:2606.25421 + arXiv:2506.02918)"
        ),
        "source_ids": ["2606.25421", "2506.02918"],
        "maps_to_frontier": ".450",
        "priority": 3,
    },
]

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Test-time world-model and dynamics adaptation loop",
        "track": "test_time_dynamics_adaptation",
        "source_ids": ["2506.02918", "2509.03956", "2507.15877"],
        "maps_to_frontier": ".450",
        "targets_fork_verdict": AIMED_AT_FORK_VERDICT,
        "a1b_result_fit": (
            "A1b CEGIS delta was 0.0, so the next swing adapts or retrieves "
            "dynamics at test time before planning through the engine."
        ),
        "evidence": (
            "arXiv:2506.02918 trains internal state prediction for language "
            "agents; arXiv:2509.03956 composes world models at test time; "
            "arXiv:2507.15877 compares ARC execution-guided synthesis with "
            "test-time fine-tuning."
        ),
        "experiment_graft": (
            "Collect cold-start transitions, fit or retrieve a compact dynamics "
            "adapter, then remeasure held-out transition accuracy before any "
            "planner reranking."
        ),
        "validation_gate": (
            "Promote only if held-out off-prefix transition accuracy improves "
            "on games disjoint from the adapter's observed-prefix fit."
        ),
        "sovereignty_note": (
            "The adapter can be selected or trained locally from game observations, "
            "preserving the air-gapped path."
        ),
        "fails_when": (
            "The adapter memorizes prefix frames, loses hidden state, or improves "
            "observed replay while held-out dynamics remain flat."
        ),
        "roadmap_candidate": FLAGGED_FOR_V450[0]["flag"],
    },
    {
        "method": "Family-B reference versus local open-code inducer A/B",
        "track": "family_b_vs_local_open_code_inducer",
        "source_ids": ["2605.05138", "2507.03160", "2203.13474"],
        "maps_to_frontier": ".450",
        "targets_fork_verdict": AIMED_AT_FORK_VERDICT,
        "a1b_result_fit": (
            "A1b's null CEGIS result means the loop needs a stronger inducer "
            "measurement, not another repair pass from the same engine."
        ),
        "evidence": (
            "arXiv:2605.05138 supplies the executable-world-model coding-agent "
            "reference; arXiv:2507.03160 evaluates small code models; "
            "arXiv:2203.13474 establishes open multi-turn code synthesis."
        ),
        "experiment_graft": (
            "Run one Family-B reference lane and one local open-code lane against "
            "the same engine interface and held-out transition gate."
        ),
        "validation_gate": (
            "The reference lane measures the capability ceiling; the local lane "
            "is promoted only if it beats the current Qwen3.5-9B-MTP inducer "
            "under the A1 held-out game set."
        ),
        "sovereignty_note": (
            "The cloud-strength lane is a ceiling measurement; the desired "
            "deployment lane remains local and open."
        ),
        "fails_when": (
            "The reference lane still overfits observed prefixes, or no local "
            "open inducer can synthesize executable state updates."
        ),
        "roadmap_candidate": FLAGGED_FOR_V450[1]["flag"],
    },
    {
        "method": "Agent-authored world-model target construction",
        "track": "agent_authored_world_model_targets",
        "source_ids": ["2606.25421", "2506.02918"],
        "maps_to_frontier": ".450",
        "targets_fork_verdict": AIMED_AT_FORK_VERDICT,
        "a1b_result_fit": (
            "A1b failed to repair from generic counterexamples; decision-oriented "
            "targets ask the agent what transition facts it needs before acting."
        ),
        "evidence": (
            "arXiv:2606.25421 replaces next-observation prediction with "
            "agent-authored dynamics targets; arXiv:2506.02918 shows state "
            "prediction can support language-agent tool planning."
        ),
        "experiment_graft": (
            "For each failed held-out transition, generate a decision-need target "
            "such as hidden toggle state, object persistence, or action effect, "
            "then train or prompt the inducer against that target."
        ),
        "validation_gate": (
            "Count the method only when targeted transition facts raise held-out "
            "engine accuracy, not just when next-frame text improves."
        ),
        "sovereignty_note": (
            "Target construction is derived from local traces and can feed either "
            "the local inducer or the reference lane."
        ),
        "fails_when": (
            "The generated targets mirror the model's misconception or require "
            "observations the game has not exposed."
        ),
        "roadmap_candidate": FLAGGED_FOR_V450[2]["flag"],
    },
    {
        "method": "Action-prefix latent world-model adapter",
        "track": "action_prefix_world_model_adapter",
        "source_ids": ["2606.26217", "2506.02918", "2507.15877"],
        "maps_to_frontier": ".450",
        "targets_fork_verdict": AIMED_AT_FORK_VERDICT,
        "a1b_result_fit": (
            "A1b's flat delta leaves compounding one-step dynamics error as a "
            "candidate residual; prefix-level prediction attacks that error."
        ),
        "evidence": (
            "arXiv:2606.26217 predicts latents for action prefixes instead of "
            "rolling one step at a time; arXiv:2506.02918 adds state prediction "
            "to agents; arXiv:2507.15877 keeps ARC execution guidance in scope."
        ),
        "experiment_graft": (
            "Add an action-prefix probe over candidate sequences and compare its "
            "held-out transition predictions against the current one-step engine."
        ),
        "validation_gate": (
            "Promote only if long-horizon held-out transition accuracy improves "
            "without degrading one-step observed-prefix replay."
        ),
        "sovereignty_note": (
            "A small prefix adapter can run locally and can be swapped behind the "
            "same executable-engine interface."
        ),
        "fails_when": (
            "Prefix supervision hides wrong mechanics, or the latent state cannot "
            "be decoded into executable game-state checks."
        ),
        "roadmap_candidate": FLAGGED_FOR_V450[0]["flag"],
    },
]

DEFAULT_FRESH_SWEEP = {
    "filtered_track": (
        "V450 frontier for INDUCER_CEILING with A1b CEGIS null: test-time "
        "dynamics adaptation, stronger executable-world-model inducers, and "
        "decision-oriented world-model targets"
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
    "a1_source_fork_verdict": DEFAULT_UPSTREAM_CONTEXT["a1_source_fork_verdict"],
    "a1_computed_fork_verdict": DEFAULT_UPSTREAM_CONTEXT["a1_computed_fork_verdict"],
    "a1_genuinely_diagnostic": DEFAULT_UPSTREAM_CONTEXT["a1_genuinely_diagnostic"],
    "a1_positive_control_caveat": DEFAULT_UPSTREAM_CONTEXT["a1_positive_control_caveat"],
    "aimed_at_fork_verdict": AIMED_AT_FORK_VERDICT,
    "a1b_delta_median": DEFAULT_UPSTREAM_CONTEXT["a1b_delta_median"],
    "a1b_delta_ci95": DEFAULT_UPSTREAM_CONTEXT["a1b_delta_ci95"],
    "a1b_cegis_moved_accuracy": DEFAULT_UPSTREAM_CONTEXT["a1b_cegis_moved_accuracy"],
    "current_cegis_promoted": False,
    "branch_reason": BRANCH_REASON,
    "sweep_clusters_used": True,
    "sweep_cluster_ids": [6],
    "sweep_cluster_urls": [CLUSTER_6_URL, CLUSTER_6_PAGE2_URL],
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "sweep_semscholar_result": SEMANTIC_SCHOLAR_RESULT,
    "semantic_scholar_unique_arxiv_ids": SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
    "websearch_webfetch_used": True,
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
    "top_source_count": len(WEBSEARCH_WEBFETCH_TOP_SOURCES),
    "arxiv_http_200_verified_ids": [
        f"https://arxiv.org/abs/{source_id}" for source_id in sorted(REQUIRED_SOURCE_IDS)
    ],
    "deep_research_invoked": False,
    "retired_coverage_classes_reingested": False,
    "exploration_strategy_reingested": False,
    "energy_classes_reingested": False,
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
    "a1_source_fork_verdict": DEFAULT_UPSTREAM_CONTEXT["a1_source_fork_verdict"],
    "a1_computed_fork_verdict": DEFAULT_UPSTREAM_CONTEXT["a1_computed_fork_verdict"],
    "a1_genuinely_diagnostic": DEFAULT_UPSTREAM_CONTEXT["a1_genuinely_diagnostic"],
    "a1_positive_control_caveat": DEFAULT_UPSTREAM_CONTEXT["a1_positive_control_caveat"],
    "a1_median_engine_heldout_accuracy": DEFAULT_UPSTREAM_CONTEXT["a1_median_engine_heldout_accuracy"],
    "a1_coverage_migration_count": DEFAULT_UPSTREAM_CONTEXT["a1_coverage_migration_count"],
    "a1b_honest_verdict": DEFAULT_UPSTREAM_CONTEXT["a1b_honest_verdict"],
    "a1b_delta_median": DEFAULT_UPSTREAM_CONTEXT["a1b_delta_median"],
    "a1b_delta_ci95": DEFAULT_UPSTREAM_CONTEXT["a1b_delta_ci95"],
    "a1b_cegis_moved_accuracy": DEFAULT_UPSTREAM_CONTEXT["a1b_cegis_moved_accuracy"],
    "a1b_positive_control_passed": DEFAULT_UPSTREAM_CONTEXT["a1b_positive_control_passed"],
    "handoff_honest_verdict": DEFAULT_UPSTREAM_CONTEXT["handoff_honest_verdict"],
    "handoff_aimed_at_fork_verdict": DEFAULT_UPSTREAM_CONTEXT["handoff_aimed_at_fork_verdict"],
    "carried_forward_from_4868": [
        "test_time_world_model_adaptation_loop",
        "family_b_executable_world_model_inducer_ladder",
        "local_open_code_inducer",
    ],
    "not_carried_forward_from_4868": [
        "cegis_world_model_refinement_loop",
        "macro-vocab/click-heatmap coverage",
        "exploration-strategy",
        "energy classes",
    ],
}
DEFAULT_MAPPING_NOTE = {
    "summary": (
        "A1b CEGIS delta was 0.0 after A1's null positive-control-caveated "
        "fork probe; .450 should follow the computed INDUCER_CEILING residual "
        "and test the next inducer candidates."
    ),
    "terminal_success": HONEST_VERDICT,
    "source_ids": sorted(REQUIRED_SOURCE_IDS),
    "root_cause": "world-model inducer quality after nulled CEGIS",
    "planner_instruction": (
        "Run test-time dynamics adaptation first, then a Family-B reference "
        "versus local open-code inducer A/B. Use agent-authored targets and "
        "action-prefix adapters only if they improve held-out engine accuracy."
    ),
    "a1_caveat": (
        "A1 source fork_verdict is null because the positive-control check failed; "
        "the INDUCER_CEILING target is computed from the low-accuracy fork table."
    ),
    "a1b_result": "A1b CEGIS held-out delta median=0.0, CI95=[0.0, 0.0].",
    "not_carried_forward": (
        "The current CEGIS world-model refinement loop is recorded as nulled, "
        "not promoted as a strongest .450 method."
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ci_excludes_zero(ci95: object) -> bool:
    if not isinstance(ci95, Sequence) or isinstance(ci95, str | bytes) or len(ci95) != 2:
        return False
    low = _as_float(ci95[0])
    high = _as_float(ci95[1])
    return high < 0.0 or low > 0.0


def infer_computed_fork_verdict(a1_artifact: JsonMap) -> str:
    """Compute the branch implied by the A1 numeric fork table."""

    reported = a1_artifact.get("fork_verdict")
    if reported in {"INDUCER_CEILING", "GUIDANCE_WALL", "PLANNER_GAP"}:
        return str(reported)
    threshold = _as_float(
        (a1_artifact.get("induce_plan_config") or {}).get("high_accuracy_threshold"),
        default=0.5,
    )
    median_accuracy = _as_float(a1_artifact.get("median_engine_heldout_accuracy"))
    if median_accuracy < threshold:
        return "INDUCER_CEILING"
    if int(a1_artifact.get("coverage_migration_count") or 0) > 0:
        return "GUIDANCE_WALL"
    return "PLANNER_GAP"


def derive_upstream_context(a1_artifact: JsonMap, a1b_artifact: JsonMap, handoff_artifact: JsonMap) -> dict[str, object]:
    """Summarize A1, A1b, and the .448->.449 handoff for this artifact."""

    a1_source_fork = a1_artifact.get("fork_verdict")
    computed_fork = infer_computed_fork_verdict(a1_artifact)
    a1_positive_control_migrated = bool(a1_artifact.get("positive_control_migrated"))
    a1_genuinely_diagnostic = bool(
        a1_source_fork
        and a1_positive_control_migrated
        and a1_artifact.get("live_path_reachable", True)
    )
    delta_median = _as_float(a1b_artifact.get("cegis_heldout_accuracy_delta_median"))
    delta_ci95 = list(a1b_artifact.get("cegis_heldout_accuracy_delta_ci95") or [0.0, 0.0])
    cegis_moved = bool(delta_median != 0.0 and _ci_excludes_zero(delta_ci95))
    flagged = handoff_artifact.get("flagged_for_v449") or []
    flagged_candidates = [
        str(item.get("candidate", item)) if isinstance(item, Mapping) else str(item)
        for item in flagged
    ]
    return {
        "a1_honest_verdict": a1_artifact.get("honest_verdict", ""),
        "a1_source_fork_verdict": a1_source_fork,
        "a1_computed_fork_verdict": computed_fork,
        "a1_genuinely_diagnostic": a1_genuinely_diagnostic,
        "a1_positive_control_caveat": not a1_positive_control_migrated or a1_source_fork is None,
        "a1_median_engine_heldout_accuracy": _as_float(
            a1_artifact.get("median_engine_heldout_accuracy")
        ),
        "a1_coverage_migration_count": int(a1_artifact.get("coverage_migration_count") or 0),
        "a1_n_games_measured": int(a1_artifact.get("n_games_measured") or 0),
        "a1_positive_control_migrated": a1_positive_control_migrated,
        "a1b_honest_verdict": a1b_artifact.get("honest_verdict", ""),
        "a1b_delta_median": delta_median,
        "a1b_delta_ci95": delta_ci95,
        "a1b_cegis_moved_accuracy": cegis_moved,
        "a1b_positive_control_passed": bool(a1b_artifact.get("positive_control_passed")),
        "handoff_honest_verdict": handoff_artifact.get("honest_verdict", ""),
        "handoff_aimed_at_fork_verdict": handoff_artifact.get("aimed_at_fork_verdict", ""),
        "handoff_flagged_candidates": flagged_candidates,
    }


def load_upstream_context(repo_root: Path) -> dict[str, object]:
    a1 = json.loads((repo_root / A1_ARTIFACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    a1b = json.loads((repo_root / A1B_ARTIFACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    handoff = json.loads((repo_root / HANDOFF_ARTIFACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    return derive_upstream_context(a1, a1b, handoff)


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


def _preconditions_from_context(upstream_context: JsonMap) -> dict[str, object]:
    return dict(DEFAULT_PRECONDITIONS_CHECKED) | {
        "a1_source_fork_verdict": upstream_context["a1_source_fork_verdict"],
        "a1_computed_fork_verdict": upstream_context["a1_computed_fork_verdict"],
        "a1_genuinely_diagnostic": upstream_context["a1_genuinely_diagnostic"],
        "a1_positive_control_caveat": upstream_context["a1_positive_control_caveat"],
        "a1b_delta_median": upstream_context["a1b_delta_median"],
        "a1b_delta_ci95": list(upstream_context["a1b_delta_ci95"]),
        "a1b_cegis_moved_accuracy": upstream_context["a1b_cegis_moved_accuracy"],
        "current_cegis_promoted": False,
        "branch_reason": BRANCH_REASON,
    }


def _upstream_artifacts_from_context(upstream_context: JsonMap) -> dict[str, object]:
    return dict(DEFAULT_UPSTREAM_ARTIFACTS) | {
        "a1_honest_verdict": upstream_context["a1_honest_verdict"],
        "a1_source_fork_verdict": upstream_context["a1_source_fork_verdict"],
        "a1_computed_fork_verdict": upstream_context["a1_computed_fork_verdict"],
        "a1_genuinely_diagnostic": upstream_context["a1_genuinely_diagnostic"],
        "a1_positive_control_caveat": upstream_context["a1_positive_control_caveat"],
        "a1_median_engine_heldout_accuracy": upstream_context["a1_median_engine_heldout_accuracy"],
        "a1_coverage_migration_count": upstream_context["a1_coverage_migration_count"],
        "a1b_honest_verdict": upstream_context["a1b_honest_verdict"],
        "a1b_delta_median": upstream_context["a1b_delta_median"],
        "a1b_delta_ci95": list(upstream_context["a1b_delta_ci95"]),
        "a1b_cegis_moved_accuracy": upstream_context["a1b_cegis_moved_accuracy"],
        "a1b_positive_control_passed": upstream_context["a1b_positive_control_passed"],
        "handoff_honest_verdict": upstream_context["handoff_honest_verdict"],
        "handoff_aimed_at_fork_verdict": upstream_context["handoff_aimed_at_fork_verdict"],
    }


def build_artifact(
    *,
    upstream_context: JsonMap = DEFAULT_UPSTREAM_CONTEXT,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    fresh_sweep: JsonMap = DEFAULT_FRESH_SWEEP,
    flagged_for_v450: Sequence[JsonMap] = FLAGGED_FOR_V450,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    upstream_artifacts = _upstream_artifacts_from_context(upstream_context)
    mapping_note = dict(DEFAULT_MAPPING_NOTE)
    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "arxiv_ids_cited": sorted(citations),
        "aimed_at_fork_verdict": AIMED_AT_FORK_VERDICT,
        "flagged_for_v450": [dict(flag) for flag in flagged_for_v450],
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
            flagged_for_v450,
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
    FLAGGED_FOR_V450,
    DEFAULT_UPSTREAM_ARTIFACTS,
    DEFAULT_MAPPING_NOTE,
)


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
        "aimed_at_fork_verdict must be INDUCER_CEILING",
    )
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must be aggregation-only",
    )
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles must match annotations")
    _require(artifact["note_path"] == NOTE_PATH, "note_path must point at the Exp 4879 note")
    _require(artifact["references_path"] == REFERENCES_PATH, "references_path must point at Exp 4879 references")
    _require(artifact["random_seed"] == RANDOM_SEED, "random_seed must be the experiment id")
    _require(artifact["duration_s"] == DURATION_S, "duration_s must preserve the 0.0001s floor")
    _validate_citations(artifact["citations"], artifact["arxiv_ids_cited"])
    _validate_methods(artifact["methods_mapped"], artifact["arxiv_ids_cited"])
    _validate_flags(artifact["flagged_for_v450"], artifact["arxiv_ids_cited"])
    _validate_preconditions(artifact["preconditions_checked"])
    _validate_fresh_sweep(artifact["fresh_sweep"])
    _validate_upstream_artifacts(artifact["upstream_artifacts"])
    _validate_mapping_note(artifact["sota_to_experiment_mapping_note"], artifact["arxiv_ids_cited"])
    _require(
        artifact["reproducibility_checksum"]
        == source_set_checksum(
            artifact["citations"],
            artifact["methods_mapped"],
            artifact["flagged_for_v450"],
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
        _require(method["maps_to_frontier"] == ".450", "method must map to the .450 frontier")
        _require(method["targets_fork_verdict"] == AIMED_AT_FORK_VERDICT, "method must target fork verdict")
        _require("A1b" in str(method["a1b_result_fit"]), "method must fit the A1b CEGIS result")
        _require("arXiv:" in str(method["evidence"]), "method evidence must cite arXiv IDs")
        _require(bool(method["experiment_graft"]), "each method needs an experiment graft")
        _require(bool(method["validation_gate"]), "each method needs a validation gate")
        _require(bool(method["sovereignty_note"]), "each method needs a sovereignty note")
        _require(bool(method["fails_when"]), "each method needs fails_when")
        roadmap = str(method["roadmap_candidate"])
        _require("flagged_for_v450" in roadmap, "each method needs a .450 roadmap candidate")
        _require(
            "cegis_world_model_refinement_loop" not in roadmap,
            "nulled CEGIS loop must not be promoted",
        )
        tracks.add(str(method["track"]))
    _require(REQUIRED_TRACKS == tracks, "methods_mapped missing required .450 inducer tracks")


def _validate_flags(flags: object, arxiv_ids_cited: object) -> None:
    _require(
        isinstance(flags, Sequence) and not isinstance(flags, str | bytes) and bool(flags),
        "flagged_for_v450 required",
    )
    cited = set(arxiv_ids_cited)
    for flag in flags:
        _require(isinstance(flag, Mapping), "each flagged_for_v450 entry must be a mapping")
        _require("candidate" in flag and "flag" in flag, "each flagged_for_v450 entry needs candidate and flag")
        as_text = json.dumps(flag, sort_keys=True)
        _require("flagged_for_v449" not in as_text, "stale .449 flag found in flagged_for_v450")
        _require("flagged_for_v450" in str(flag["flag"]), "flagged_for_v450 entries must carry the .450 flag")
        _require(
            "cegis_world_model_refinement_loop" not in as_text,
            "nulled CEGIS loop must not be promoted",
        )
        _require(set(flag.get("source_ids", [])).issubset(cited), "flagged_for_v450 source_ids must be verified")


def _validate_preconditions(preconditions: object) -> None:
    _require(isinstance(preconditions, Mapping), "preconditions_checked must be a mapping")
    _require(set(preconditions) == REQUIRED_PRECONDITION_FIELDS, "preconditions_checked must match schema")
    _require(preconditions["research_studying_present"] is True, "research-studying precondition must pass")
    _require(preconditions["research_references_present"] is True, "research-references precondition must pass")
    _require(preconditions["a1_artifact_present"] is True, "A1 artifact must be present")
    _require(preconditions["a1b_artifact_present"] is True, "A1b artifact must be present")
    _require(preconditions["handoff_artifact_present"] is True, "Exp 4868 handoff artifact must be present")
    _require(preconditions["a1_fork_verdict_read"] is True, "A1 fork verdict must be read")
    _require(preconditions["a1_source_fork_verdict"] is None, "A1 null source verdict must be recorded")
    _require(preconditions["a1_computed_fork_verdict"] == AIMED_AT_FORK_VERDICT, "computed A1 fork mismatch")
    _require(preconditions["a1_genuinely_diagnostic"] is False, "A1 caveat must remain explicit")
    _require(preconditions["a1_positive_control_caveat"] is True, "A1 positive-control caveat must be recorded")
    _require(preconditions["aimed_at_fork_verdict"] == AIMED_AT_FORK_VERDICT, "precondition fork target mismatch")
    _require(preconditions["a1b_delta_median"] == 0.0, "A1b CEGIS null median must be recorded")
    _require(preconditions["a1b_delta_ci95"] == [0.0, 0.0], "A1b CEGIS null CI must be recorded")
    _require(preconditions["a1b_cegis_moved_accuracy"] is False, "A1b CEGIS null must remain false")
    _require(preconditions["current_cegis_promoted"] is False, "current CEGIS loop must not be promoted")
    _require(preconditions["sweep_clusters_used"] is True, "sweep_clusters must be used")
    _require(preconditions["sweep_cluster_ids"] == [6], "sweep cluster IDs must be [6]")
    _require(preconditions["sweep_semscholar_used"] is True, "sweep_semscholar must be used")
    _require(
        preconditions["semantic_scholar_unique_arxiv_ids"] == SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
        "Semantic Scholar IDs must match reliable-channel output",
    )
    _require(preconditions["websearch_webfetch_used"] is True, "WebSearch/WebFetch must be used")
    _require(5 <= int(preconditions["top_source_count"]) <= 8, "top_source_count must record top five to eight sources")
    _require(preconditions["deep_research_invoked"] is False, "deep-research must not be invoked")
    _require(preconditions["retired_coverage_classes_reingested"] is False, "retired coverage must not be reingested")
    _require(preconditions["exploration_strategy_reingested"] is False, "exploration strategy must not be reingested")
    _require(preconditions["energy_classes_reingested"] is False, "energy classes must not be reingested")
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
    _require(
        fresh_sweep["semantic_scholar_unique_arxiv_ids"] == SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
        "fresh_sweep Semantic Scholar IDs must match reliable-channel output",
    )
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
    _require(upstream_artifacts["handoff_artifact"] == HANDOFF_ARTIFACT_RELATIVE_PATH, "upstream must cite Exp 4868")
    _require(upstream_artifacts["a1_source_fork_verdict"] is None, "upstream A1 source null must be recorded")
    _require(
        upstream_artifacts["a1_computed_fork_verdict"] == AIMED_AT_FORK_VERDICT,
        "upstream fork target must match artifact",
    )
    _require(upstream_artifacts["a1b_cegis_moved_accuracy"] is False, "upstream A1b CEGIS null must be recorded")
    _require(upstream_artifacts["a1b_delta_median"] == 0.0, "upstream A1b delta must be zero")
    not_carried = upstream_artifacts["not_carried_forward_from_4868"]
    _require("cegis_world_model_refinement_loop" in not_carried, "nulled CEGIS loop must be marked not carried")


def _validate_mapping_note(mapping_note: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(mapping_note, Mapping), "SOTA mapping note must be a mapping")
    _require(set(mapping_note) == REQUIRED_MAPPING_NOTE_FIELDS, "mapping note must match schema")
    _require(mapping_note["terminal_success"] == HONEST_VERDICT, "mapping note terminal success must match verdict")
    _require(
        mapping_note["root_cause"] == "world-model inducer quality after nulled CEGIS",
        "mapping note root cause must match",
    )
    _require("A1b CEGIS delta was 0.0" in str(mapping_note["summary"]), "mapping note must mention A1b null")
    _require("positive-control" in str(mapping_note["a1_caveat"]), "mapping note must preserve A1 caveat")
    _require("CEGIS" in str(mapping_note["not_carried_forward"]), "mapping note must record nulled CEGIS")
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
            f"maps to {method['maps_to_frontier']} / {method['targets_fork_verdict']}. "
            f"A1b fit: {method['a1b_result_fit']} Evidence: {method['evidence']} "
            f"Experiment graft: {method['experiment_graft']} Validation gate: {method['validation_gate']} "
            f"Sovereignty: {method['sovereignty_note']} Fails when: {method['fails_when']}"
        )
        for method in result["methods_mapped"]
    )
    flag_lines = "\n".join(flag["flag"] for flag in result["flagged_for_v450"])
    return f"""{STUDYING_SECTION_START}
## 2026-06-27 Exp 4879 - .450 V450 frontier SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4879_sota_ingestion_v450_frontier.json`.

**Preconditions:** `research-studying.md`, `research-references.md`,
`results/experiment_4871_generation_wall_fork_probe_gpu_fixed.json`,
`results/experiment_4872_cegis_world_model_refinement.json`, and
`results/experiment_4868_sota_ingestion_v449_frontier.json` were present. A1's
source `fork_verdict` is null because the positive-control check failed, but the
numeric fork table computes to `INDUCER_CEILING`. A1b CEGIS delta was 0.0 with
CI95 [0.0, 0.0], so this note carries forward the next inducer candidates, not
the current CEGIS refinement loop. `scripts/sweep_clusters.py` emitted the
neural-guided-search/world-model cluster 6 URLs. `scripts/sweep_semscholar.py`
was run on five focused queries; rate limits were recorded rather than promoted
as evidence. Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks
verified the top eight papers listed below. `/deep-research` was not invoked.
The retired macro-vocab/click-heatmap coverage, exploration-strategy, and
energy classes were not re-ingested. No model load, training, leaderboard
submission, or solve claim was made; this is a no solve claim ingestion note.

**A1/A1b branch:** `INDUCER_CEILING` residual with an A1 positive-control caveat
and a nulled A1b CEGIS refinement delta.

**Verified source set:**
{citation_lines}

**SOTA -> .450 frontier mapping:**
{method_lines}

{flag_lines}

**Bottom line for .450:** try test-time dynamics adaptation first, then compare
a Family-B executable-world-model reference inducer with a local open-code
inducer. Use agent-authored targets and action-prefix adapters as targeted
engine-quality improvements; keep the current CEGIS loop recorded as nulled.
{STUDYING_SECTION_END}"""


def build_research_references_section(artifact: JsonMap | None = None) -> str:
    result = dict(artifact or build_artifact())
    validate_artifact(result)
    citations = result["citations"]
    source_lines = "\n".join(
        (
            f"- **arXiv:{source_id} -- {citations[source_id]['title']}.** "
            "Exp 4879 use: V450 INDUCER_CEILING residual source for test-time "
            "dynamics adaptation, stronger executable-world-model induction, "
            "or local open-code inducer selection after nulled A1b CEGIS."
        )
        for source_id in sorted(citations)
    )
    return f"""{REFERENCES_SECTION_START}
## 2026-06-27 Exp 4879 V450 frontier source set

Reliable-channel ingestion for `.450`, aimed at the computed
`INDUCER_CEILING` residual while recording that A1's source fork was null and
A1b's CEGIS delta was zero. These papers are marked INGESTED for the V450
frontier roadmap handoff:

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
    _require(start >= 0 and end > start, "research-studying Exp 4879 section missing")
    section = text[start:end]
    _require("INGESTED" in section, "research-studying section must mark Exp 4879 ingested")
    _require("SOTA -> .450 frontier mapping" in section, "research-studying section missing mapping note")
    _require("flagged_for_v450" in section, "research-studying section missing .450 flags")
    _require("INDUCER_CEILING" in section, "research-studying section must name fork target")
    _require("A1b CEGIS delta was 0.0" in section, "research-studying section must preserve A1b null")
    _require("no solve claim" in section, "research-studying section must preserve no solve claim")
    for required in NOTE_REQUIRED_SOURCE_CITATIONS:
        _require(required in section, f"research-studying section missing {required}")


def validate_research_references_text(text: str, artifact: JsonMap | None = None) -> None:
    result = dict(artifact or build_artifact())
    validate_artifact(result)
    start = text.find(REFERENCES_SECTION_START)
    end = text.find(REFERENCES_SECTION_END, start)
    _require(start >= 0 and end > start, "research-references Exp 4879 section missing")
    section = text[start:end]
    _require("Exp 4879 V450 frontier source set" in section, "references section missing title")
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
    root = Path(os.environ.get("CARNOT_EXP4879_ROOT", Path.cwd()))
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
