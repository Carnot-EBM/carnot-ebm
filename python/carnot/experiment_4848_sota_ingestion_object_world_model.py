"""Exp 4848 SOTA ingestion for object-world-model planning.

Spec refs: REQ-ARC-WMTE-4848, SCENARIO-ARC-WMTE-4848,
SCENARIO-ARC-WMTE-4848-NO-FABRICATION.

This module writes a deterministic literature-ingestion artifact for the .447
roadmap. It maps object-centric world models and relational planners onto the
Exp 4838 A1 perception layer: recovered objects must become a structured state
that a planner can roll forward into a proposable winner on a novel game. It
performs no model load, training, leaderboard submission, or solve claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4848_sota_ingestion_object_world_model.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
UPSTREAM_PERCEPTION_ARTIFACT = "results/experiment_4838_sota_ingestion_perception_representation.json"
NOTE_PATH = "research-studying.md#exp-4848-sota-ingestion-object-world-model"
RANDOM_SEED = 4848
DURATION_S = 0.0001
HONEST_VERDICT = "success_sota_ingestion_object_world_model_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
STUDYING_SECTION_START = "<!-- EXP4848-SOTA-INGESTION-OBJECT-WORLD-MODEL-START -->"
STUDYING_SECTION_END = "<!-- EXP4848-SOTA-INGESTION-OBJECT-WORLD-MODEL-END -->"
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
            "success_sota_ingestion_object_world_model_mapped."
        )
    },
    "methods_mapped": {
        "principle": (
            "the strongest 3-5 object-centric world-model/planning methods "
            "mapped onto consuming the A1 perception layer, each with a real arXiv ID."
        )
    },
    "arxiv_ids_cited": {
        "principle": "every method claim must cite a verifiable arXiv ID."
    },
    "flagged_for_v447": {
        "principle": "the strongest method(s) flagged so the .447 planner reads the mapping."
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
    "a1_perception_layer_input": {
        "principle": "imports Exp 4838 object identity outputs as the planner input contract."
    },
    "object_world_model_mapping_note": {
        "principle": "states how object-relational state becomes a proposable winner for .447."
    },
    "note_path": {
        "principle": "points to the idempotent research-studying.md ingestion note."
    },
    "random_seed": {
        "principle": "deterministic experiment identifier for reproducible artifact generation."
    },
    "duration_s": {
        "principle": "0.0001s floor for aggregation-only inference substrate."
    },
    "reproducibility_checksum": {
        "principle": "content hash of citations, method map, flags, A1 input, and mapping note."
    },
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "methods_mapped",
    "arxiv_ids_cited",
    "flagged_for_v447",
    "inference_substrate",
    "preconditions_checked",
    "citations",
    "fresh_sweep",
    "a1_perception_layer_input",
    "object_world_model_mapping_note",
    "note_path",
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
        "consumes_a1_object_layer",
        "object_relational_state",
        "planning_graft",
        "proposable_winner_output",
        "verification_handoff",
        "takes_over_from_current_stack",
        "fails_when",
        "roadmap_candidate",
    }
)
REQUIRED_CITATION_FIELDS = frozenset({"title", "url", "http_status"})
REQUIRED_PRECONDITION_FIELDS = frozenset(
    {
        "agents_md_read",
        "codex_md_read",
        "claude_md_consulted",
        "research_studying_present",
        "research_references_present",
        "upstream_perception_artifact_present",
        "upstream_flagged_for_v446_read",
        "sweep_clusters_used",
        "sweep_cluster_ids",
        "sweep_cluster_urls",
        "sweep_semscholar_used",
        "sweep_semscholar_queries",
        "semantic_scholar_unique_arxiv_ids",
        "sweep_semscholar_result",
        "websearch_webfetch_used",
        "websearch_webfetch_top_sources",
        "top_source_count",
        "arxiv_http_200_verified_ids",
        "deep_research_invoked",
        "exploration_strategy_reingested",
        "model_load",
        "training_launched",
        "leaderboard_submission",
        "solve_claim_made",
        "ops_docs_modified",
    }
)
REQUIRED_FRESH_SWEEP_FIELDS = frozenset(
    {
        "filtered_track",
        "cluster_ids",
        "semantic_scholar_queries",
        "semantic_scholar_result",
        "semantic_scholar_unique_arxiv_ids",
        "webfetch_top_sources",
    }
)
REQUIRED_A1_INPUT_FIELDS = frozenset(
    {
        "source_artifact",
        "source_honest_verdict",
        "target_roadmap",
        "carried_forward_v446_flags",
        "consumed_state_fields",
        "exploration_strategy_class_closed",
        "planner_question",
    }
)
REQUIRED_MAPPING_NOTE_FIELDS = frozenset(
    {
        "summary",
        "terminal_success",
        "source_ids",
        "root_cause",
        "planner_instruction",
    }
)
REQUIRED_SOURCE_IDS = frozenset(
    {
        "1911.12247",
        "2402.03326",
        "2507.03298",
        "2511.02225",
        "2601.06604",
        "2605.14937",
        "2606.12316",
        "2606.14418",
    }
)
REQUIRED_TRACKS = frozenset(
    {
        "object_relation_transition_graph",
        "slot_structured_rollout_model",
        "object_causal_mcts_planner",
        "goal_conditioned_slot_mpc",
        "interaction_primitive_loop_policy",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS)

CLUSTER_3_URL = (
    'http://export.arxiv.org/api/query?search_query=(abs:"active+inference"+OR+'
    'abs:"free+energy"+OR+abs:"free+energy+principle"+OR+'
    'abs:"predictive+coding"+OR+abs:"world+model")+AND+'
    '(abs:"LLM"+OR+abs:"language+model"+OR+abs:"reasoning")&start=0&'
    "max_results=8&sortBy=submittedDate&sortOrder=descending"
)
CLUSTER_5_URL = (
    'http://export.arxiv.org/api/query?search_query=(abs:"affordance"+OR+'
    'abs:"action+effect"+OR+abs:"clickability"+OR+abs:"frame+prediction"+OR+'
    'abs:"intrinsic+motivation"+OR+abs:"directed+exploration"+OR+'
    'abs:"novelty+search")+AND+(abs:"reinforcement+learning"+OR+abs:"agent"+OR+'
    'abs:"exploration"+OR+abs:"interactive+environment"+OR+abs:"ARC")&start=0&'
    "max_results=8&sortBy=submittedDate&sortOrder=descending"
)
CLUSTER_6_URL = (
    'http://export.arxiv.org/api/query?search_query=(abs:"neural+guided+search"+OR+'
    'abs:"learned+heuristic"+OR+abs:"value+guided+search"+OR+'
    'abs:"program+induction"+OR+abs:"world+model"+OR+abs:"goal+induction")+'
    'AND+(abs:"planning"+OR+abs:"agent"+OR+abs:"reasoning"+OR+'
    'abs:"reinforcement+learning")&start=0&max_results=8&sortBy=submittedDate&'
    "sortOrder=descending"
)
SEMANTIC_SCHOLAR_QUERIES = [
    "object centric world model relational planning MCTS object relational state",
    "slot structured world model object centric planning dynamics",
    "object relational dynamics learning graph planner reinforcement learning",
]
SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS = [
    "2402.03326",
    "2410.08822",
    "2502.07600",
    "2503.06170",
    "2507.03298",
    "2511.02225",
    "2605.14937",
    "2606.14418",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/1911.12247",
    "https://arxiv.org/abs/2402.03326",
    "https://arxiv.org/abs/2507.03298",
    "https://arxiv.org/abs/2511.02225",
    "https://arxiv.org/abs/2601.06604",
    "https://arxiv.org/abs/2605.14937",
    "https://arxiv.org/abs/2606.12316",
    "https://arxiv.org/abs/2606.14418",
]

CITATIONS = {
    "1911.12247": {
        "title": "Contrastive Learning of Structured World Models",
        "url": "https://arxiv.org/abs/1911.12247",
        "http_status": 200,
    },
    "2402.03326": {
        "title": "Slot Structured World Models",
        "url": "https://arxiv.org/abs/2402.03326",
        "http_status": 200,
    },
    "2507.03298": {
        "title": "Dyn-O: Building Structured World Models with Object-Centric Representations",
        "url": "https://arxiv.org/abs/2507.03298",
        "http_status": 200,
    },
    "2511.02225": {
        "title": "Learning Interactive World Model for Object-Centric Reinforcement Learning",
        "url": "https://arxiv.org/abs/2511.02225",
        "http_status": 200,
    },
    "2601.06604": {
        "title": "Object-Centric World Models Meet Monte Carlo Tree Search",
        "url": "https://arxiv.org/abs/2601.06604",
        "http_status": 200,
    },
    "2605.14937": {
        "title": "Slot-MPC: Goal-Conditioned Model Predictive Control with Object-Centric Representations",
        "url": "https://arxiv.org/abs/2605.14937",
        "http_status": 200,
    },
    "2606.12316": {
        "title": "Slots, Transitions, Loops: Learning Composable World Models for ARC",
        "url": "https://arxiv.org/abs/2606.12316",
        "http_status": 200,
    },
    "2606.14418": {
        "title": "Causal Object-Centric Models for Planning with Monte Carlo Tree Search",
        "url": "https://arxiv.org/abs/2606.14418",
        "http_status": 200,
    },
}

FLAGGED_FOR_V447 = [
    {
        "candidate": "comet_object_mcts_planner",
        "flag": (
            "flagged_for_v447: comet_object_mcts_planner "
            "(arXiv:2606.14418 + arXiv:2601.06604 + arXiv:2402.03326)"
        ),
        "source_ids": ["2606.14418", "2601.06604", "2402.03326"],
        "maps_to_frontier": ".447",
    },
    {
        "candidate": "slot_mpc_object_action_optimizer",
        "flag": (
            "flagged_for_v447: slot_mpc_object_action_optimizer "
            "(arXiv:2605.14937 + arXiv:2507.03298)"
        ),
        "source_ids": ["2605.14937", "2507.03298"],
        "maps_to_frontier": ".447",
    },
    {
        "candidate": "loop_owm_interaction_primitive_proposer",
        "flag": (
            "flagged_for_v447: loop_owm_interaction_primitive_proposer "
            "(arXiv:2606.12316 + arXiv:2511.02225 + arXiv:1911.12247)"
        ),
        "source_ids": ["2606.12316", "2511.02225", "1911.12247"],
        "maps_to_frontier": ".447",
    },
]

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Object-relation transition graph proposer",
        "track": "object_relation_transition_graph",
        "source_ids": ["1911.12247", "2402.03326"],
        "maps_to_frontier": ".447",
        "consumes_a1_object_layer": (
            "Consumes the A1 object layer as persistent object IDs, relation edges, "
            "and before/after object-action bindings."
        ),
        "object_relational_state": (
            "State is a set of object slots plus a graph of inferred pairwise "
            "relations and action-conditioned edge changes."
        ),
        "planning_graft": (
            "Learn a compact transition model over object-relation edits and ask "
            "the proposer to enumerate the smallest graph edits that move a near "
            "state toward a terminal-looking structure."
        ),
        "proposable_winner_output": (
            "Produce a proposable winner as a concrete object-relation edit plus "
            "the executable action template that should cause that edit."
        ),
        "verification_handoff": (
            "Replay the action template in the live harness and keep only edits "
            "whose rendered before/after object graph matches the predicted delta."
        ),
        "takes_over_from_current_stack": (
            "Takes over frame-only candidate generation when the pool cannot name "
            "which object relation should change."
        ),
        "fails_when": (
            "The A1 tracker merges objects, graph edges encode visual proximity "
            "instead of mechanics, or relation edits cannot be lowered to actions."
        ),
        "roadmap_candidate": FLAGGED_FOR_V447[2]["flag"],
    },
    {
        "method": "Slot-structured imagined rollout planner",
        "track": "slot_structured_rollout_model",
        "source_ids": ["2402.03326", "2507.03298"],
        "maps_to_frontier": ".447",
        "consumes_a1_object_layer": (
            "Consumes the A1 object layer as slot-aligned object features, "
            "dynamics-aware attributes, and slot persistence tracks."
        ),
        "object_relational_state": (
            "State is an object slot table with dynamics-aware fields separated "
            "from visual nuisance fields, plus learned interaction messages."
        ),
        "planning_graft": (
            "Roll forward short object-slot trajectories, score imagined deltas "
            "for goal-like object changes, and back out the action prefix that "
            "caused the best rollout."
        ),
        "proposable_winner_output": (
            "Produce a proposable winner as a replayable object-slot rollout and "
            "a small action prefix for the live verifier."
        ),
        "verification_handoff": (
            "Promote only rollouts that replay into the predicted object IDs, "
            "positions, and relation deltas without relying on a terminal oracle."
        ),
        "takes_over_from_current_stack": (
            "Takes over static ranking by generating structured futures that were "
            "not present in the old candidate pool."
        ),
        "fails_when": (
            "Slot identity drifts, imagined trajectories chase texture changes, "
            "or rollout error compounds before a concrete action can be verified."
        ),
        "roadmap_candidate": FLAGGED_FOR_V447[1]["flag"],
    },
    {
        "method": "Causal object-centric MCTS planner",
        "track": "object_causal_mcts_planner",
        "source_ids": ["2601.06604", "2606.14418"],
        "maps_to_frontier": ".447",
        "consumes_a1_object_layer": (
            "Consumes the A1 object layer as object tokens, action-slot fusions, "
            "and causal relevance scores for which objects matter to a decision."
        ),
        "object_relational_state": (
            "State is a MuZero-style latent object tree with object-causal "
            "attention over relevant slot interactions."
        ),
        "planning_graft": (
            "Run shallow MCTS over object-latent transitions, using causal "
            "attention to expand actions bound to task-relevant objects first."
        ),
        "proposable_winner_output": (
            "Produce a proposable winner as the best replayable MCTS branch: "
            "object-bound actions plus predicted object-state deltas."
        ),
        "verification_handoff": (
            "Submit only the branch action prefix to the live verifier, then "
            "compare observed object deltas against the MCTS predicted state."
        ),
        "takes_over_from_current_stack": (
            "Takes over unguided first-contact search by deciding which object "
            "interactions to expand before spending live actions."
        ),
        "fails_when": (
            "The latent tree cannot be grounded into executable actions, causal "
            "attention locks onto distractors, or MCTS optimizes an unobservable "
            "latent reward shortcut."
        ),
        "roadmap_candidate": FLAGGED_FOR_V447[0]["flag"],
    },
    {
        "method": "Goal-conditioned slot MPC action optimizer",
        "track": "goal_conditioned_slot_mpc",
        "source_ids": ["2605.14937", "2507.03298"],
        "maps_to_frontier": ".447",
        "consumes_a1_object_layer": (
            "Consumes the A1 object layer as differentiable slot features and "
            "object-level target deltas."
        ),
        "object_relational_state": (
            "State is a differentiable slot dynamics model with action-conditioned "
            "object updates and goal-conditioned target slots."
        ),
        "planning_graft": (
            "Use gradient-based MPC over the object dynamics to optimize a short "
            "action sequence toward a target object configuration."
        ),
        "proposable_winner_output": (
            "Produce a proposable winner as an optimized action sequence that "
            "should realize a specific object-level goal state."
        ),
        "verification_handoff": (
            "Replay the optimized sequence and reject it unless the observed A1 "
            "object tracks reach the planned goal-conditioned slot delta."
        ),
        "takes_over_from_current_stack": (
            "Takes over random coordinate retries by directly optimizing actions "
            "against object-level dynamics."
        ),
        "fails_when": (
            "The action space is not differentiable enough, the goal slot is "
            "wrong, or the optimizer finds an invalid but smooth object shortcut."
        ),
        "roadmap_candidate": FLAGGED_FOR_V447[1]["flag"],
    },
    {
        "method": "Interaction-primitive loop policy for ARC",
        "track": "interaction_primitive_loop_policy",
        "source_ids": ["2511.02225", "2606.12316"],
        "maps_to_frontier": ".447",
        "consumes_a1_object_layer": (
            "Consumes the A1 object layer as composable object interactions, "
            "looped slot transitions, and demonstration-conditioned summaries."
        ),
        "object_relational_state": (
            "State is a structured ARC object graph with interaction primitives "
            "and loop variables over colors, shapes, and spatial relations."
        ),
        "planning_graft": (
            "Select a high-level interaction primitive and looped transition order, "
            "then lower it into object-bound primitive actions."
        ),
        "proposable_winner_output": (
            "Produce a proposable winner as an ARC-specific transformation sketch "
            "plus concrete object-bound actions that enter the candidate pool."
        ),
        "verification_handoff": (
            "Run the lowered actions and retain the sketch only when the observed "
            "object graph follows the predicted looped transition."
        ),
        "takes_over_from_current_stack": (
            "Takes over generic exploration by constructing a new object-level "
            "candidate instead of waiting for one to appear by chance."
        ),
        "fails_when": (
            "The primitive library misses the game mechanic, the loop summary "
            "memorizes family style, or the transformation sketch cannot lower "
            "to live actions."
        ),
        "roadmap_candidate": FLAGGED_FOR_V447[2]["flag"],
    },
]

DEFAULT_FRESH_SWEEP = {
    "filtered_track": (
        "object-centric world-models and relational planners that consume "
        "object-relational state for novel-game proposal generation"
    ),
    "cluster_ids": [3, 5, 6],
    "semantic_scholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "semantic_scholar_result": (
        "One focused query returned eight arXiv IDs; two focused queries returned "
        "HTTP 429, so no S2-only source was promoted."
    ),
    "semantic_scholar_unique_arxiv_ids": SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
    "webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
}
DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "claude_md_consulted": True,
    "research_studying_present": True,
    "research_references_present": True,
    "upstream_perception_artifact_present": True,
    "upstream_flagged_for_v446_read": True,
    "sweep_clusters_used": True,
    "sweep_cluster_ids": [3, 5, 6],
    "sweep_cluster_urls": [CLUSTER_3_URL, CLUSTER_5_URL, CLUSTER_6_URL],
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "semantic_scholar_unique_arxiv_ids": SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
    "sweep_semscholar_result": DEFAULT_FRESH_SWEEP["semantic_scholar_result"],
    "websearch_webfetch_used": True,
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
    "top_source_count": len(WEBSEARCH_WEBFETCH_TOP_SOURCES),
    "arxiv_http_200_verified_ids": [
        f"https://arxiv.org/abs/{source_id}" for source_id in sorted(REQUIRED_SOURCE_IDS)
    ],
    "deep_research_invoked": False,
    "exploration_strategy_reingested": False,
    "model_load": False,
    "training_launched": False,
    "leaderboard_submission": False,
    "solve_claim_made": False,
    "ops_docs_modified": False,
}
DEFAULT_A1_PERCEPTION_LAYER_INPUT = {
    "source_artifact": UPSTREAM_PERCEPTION_ARTIFACT,
    "source_honest_verdict": "success_sota_ingestion_perception_representation_mapped",
    "target_roadmap": ".447",
    "carried_forward_v446_flags": [
        "loop_owm_slot_transition_proposer",
        "object_relational_world_model_mcts",
        "causal_object_jepa_shortcut_guard",
    ],
    "consumed_state_fields": [
        "object_ids",
        "slots",
        "relation_edges",
        "persistence_tracks",
        "object_action_bindings",
        "causal_shortcut_guard_features",
    ],
    "exploration_strategy_class_closed": True,
    "planner_question": (
        "Given recovered object structure, turn the A1 object layer into a "
        "proposable winner on a novel game."
    ),
}
DEFAULT_MAPPING_NOTE = {
    "summary": (
        "The .447 handoff is object-world-model planning: consume the A1 object "
        "layer, roll or search over object-relational state, and emit a proposable "
        "winner before any ranker scores the pool."
    ),
    "terminal_success": HONEST_VERDICT,
    "source_ids": sorted(REQUIRED_SOURCE_IDS),
    "root_cause": "object-world-model planning",
    "planner_instruction": (
        "Prioritize COMET/ObjectZero object-MCTS, Slot-MPC over object slots, "
        "and Loop-OWM/FIOC-WM interaction primitives as the .447 planning inputs."
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_set_checksum(
    citations: JsonMap,
    methods: Sequence[JsonMap],
    flags: Sequence[JsonMap],
    a1_perception_layer_input: JsonMap,
    mapping_note: JsonMap,
) -> str:
    payload = json.dumps(
        {
            "a1_perception_layer_input": a1_perception_layer_input,
            "citations": citations,
            "flags": list(flags),
            "mapping_note": mapping_note,
            "methods": list(methods),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


REPRODUCIBILITY_CHECKSUM = source_set_checksum(
    CITATIONS,
    DEFAULT_METHODS_MAPPED,
    FLAGGED_FOR_V447,
    DEFAULT_A1_PERCEPTION_LAYER_INPUT,
    DEFAULT_MAPPING_NOTE,
)


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    fresh_sweep: JsonMap = DEFAULT_FRESH_SWEEP,
    flagged_for_v447: Sequence[JsonMap] = FLAGGED_FOR_V447,
    a1_perception_layer_input: JsonMap = DEFAULT_A1_PERCEPTION_LAYER_INPUT,
    object_world_model_mapping_note: JsonMap = DEFAULT_MAPPING_NOTE,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "arxiv_ids_cited": sorted(citations),
        "flagged_for_v447": [dict(flag) for flag in flagged_for_v447],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "citations": {source_id: dict(citation) for source_id, citation in citations.items()},
        "fresh_sweep": dict(fresh_sweep),
        "a1_perception_layer_input": dict(a1_perception_layer_input),
        "object_world_model_mapping_note": dict(object_world_model_mapping_note),
        "note_path": NOTE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_S,
        "reproducibility_checksum": source_set_checksum(
            citations,
            methods_mapped,
            flagged_for_v447,
            a1_perception_layer_input,
            object_world_model_mapping_note,
        ),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
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
    _require(
        artifact["honest_verdict"] == HONEST_VERDICT,
        f"honest_verdict must equal {HONEST_VERDICT!r}",
    )
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate must be aggregation-only")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles must match annotations")
    _require(artifact["note_path"] == NOTE_PATH, "note_path must point at the Exp 4848 note")
    _require(artifact["random_seed"] == RANDOM_SEED, "random_seed must be the experiment id")
    _require(artifact["duration_s"] == DURATION_S, "duration_s must preserve the 0.0001s floor")
    _validate_citations(artifact["citations"], artifact["arxiv_ids_cited"])
    _validate_methods(artifact["methods_mapped"], artifact["arxiv_ids_cited"])
    _validate_flags(artifact["flagged_for_v447"], artifact["arxiv_ids_cited"])
    _validate_preconditions(artifact["preconditions_checked"])
    _validate_fresh_sweep(artifact["fresh_sweep"])
    _validate_a1_input(artifact["a1_perception_layer_input"])
    _validate_mapping_note(artifact["object_world_model_mapping_note"], artifact["arxiv_ids_cited"])
    _require(
        artifact["reproducibility_checksum"]
        == source_set_checksum(
            artifact["citations"],
            artifact["methods_mapped"],
            artifact["flagged_for_v447"],
            artifact["a1_perception_layer_input"],
            artifact["object_world_model_mapping_note"],
        ),
        "reproducibility checksum must hash citations, methods, flags, A1 input, and mapping note",
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
        _require(method["maps_to_frontier"] == ".447", "method must map to the .447 frontier")
        _require(
            "A1 object layer" in str(method["consumes_a1_object_layer"]),
            "each method must consume the A1 object layer",
        )
        _require("object" in str(method["object_relational_state"]).lower(), "each method needs object state")
        _require(bool(method["planning_graft"]), "each method needs a planning graft")
        output = str(method["proposable_winner_output"]).lower()
        _require("propos" in output and "winner" in output, "each method needs a proposable winner output")
        _require(bool(method["verification_handoff"]), "each method needs a verification handoff")
        _require(bool(method["takes_over_from_current_stack"]), "each method needs takes_over_from_current_stack")
        _require(bool(method["fails_when"]), "each method needs fails_when")
        _require("flagged_for_v447" in str(method["roadmap_candidate"]), "each method needs a .447 roadmap candidate")
        tracks.add(str(method["track"]))
    _require(REQUIRED_TRACKS == tracks, "methods_mapped missing required object-world-model tracks")


def _validate_flags(flags: object, arxiv_ids_cited: object) -> None:
    _require(
        isinstance(flags, Sequence) and not isinstance(flags, str | bytes) and bool(flags),
        "flagged_for_v447 required",
    )
    cited = set(arxiv_ids_cited)
    for flag in flags:
        _require(isinstance(flag, Mapping), "each flagged_for_v447 entry must be a mapping")
        _require("candidate" in flag and "flag" in flag, "each flagged_for_v447 entry needs candidate and flag")
        _require("flagged_for_v446" not in json.dumps(flag, sort_keys=True), "stale .446 flag found in flagged_for_v447")
        _require("flagged_for_v447" in str(flag["flag"]), "flagged_for_v447 entries must carry the .447 flag")
        _require(set(flag.get("source_ids", [])).issubset(cited), "flagged_for_v447 source_ids must be verified")


def _validate_preconditions(preconditions: object) -> None:
    _require(isinstance(preconditions, Mapping), "preconditions_checked must be a mapping")
    _require(set(preconditions) == REQUIRED_PRECONDITION_FIELDS, "preconditions_checked must match schema")
    _require(preconditions["research_studying_present"] is True, "research-studying precondition must pass")
    _require(preconditions["research_references_present"] is True, "research-references precondition must pass")
    _require(preconditions["upstream_perception_artifact_present"] is True, "upstream Exp 4838 artifact must be present")
    _require(preconditions["upstream_flagged_for_v446_read"] is True, "upstream flagged_for_v446 must be read")
    _require(preconditions["sweep_clusters_used"] is True, "sweep_clusters must be used")
    _require(preconditions["sweep_cluster_ids"] == [3, 5, 6], "sweep cluster IDs must be [3, 5, 6]")
    _require(preconditions["sweep_semscholar_used"] is True, "sweep_semscholar must be used")
    _require(
        preconditions["semantic_scholar_unique_arxiv_ids"] == SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
        "Semantic Scholar IDs must match the reliable-channel sweep output",
    )
    _require(preconditions["websearch_webfetch_used"] is True, "WebSearch/WebFetch must be used")
    _require(5 <= int(preconditions["top_source_count"]) <= 8, "top_source_count must record top five to eight sources")
    _require(preconditions["deep_research_invoked"] is False, "deep-research must not be invoked")
    _require(preconditions["exploration_strategy_reingested"] is False, "exploration strategy must not be reingested")
    _require(preconditions["model_load"] is False, "model load must not occur")
    _require(preconditions["training_launched"] is False, "training must not be launched")
    _require(preconditions["leaderboard_submission"] is False, "leaderboard submission must not occur")
    _require(preconditions["solve_claim_made"] is False, "solve claim must remain false")
    _require(preconditions["ops_docs_modified"] is False, "ops docs must not be modified by this workflow")


def _validate_fresh_sweep(fresh_sweep: object) -> None:
    _require(isinstance(fresh_sweep, Mapping), "fresh_sweep must be a mapping")
    _require(set(fresh_sweep) == REQUIRED_FRESH_SWEEP_FIELDS, "fresh_sweep must match schema")
    _require(fresh_sweep["cluster_ids"] == [3, 5, 6], "fresh_sweep must record clusters 3, 5, and 6")
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


def _validate_a1_input(a1_input: object) -> None:
    _require(isinstance(a1_input, Mapping), "a1_perception_layer_input must be a mapping")
    _require(set(a1_input) == REQUIRED_A1_INPUT_FIELDS, "a1_perception_layer_input must match schema")
    _require(a1_input["source_artifact"] == UPSTREAM_PERCEPTION_ARTIFACT, "A1 input must cite Exp 4838 artifact")
    _require(
        a1_input["source_honest_verdict"] == "success_sota_ingestion_perception_representation_mapped",
        "A1 input must carry the Exp 4838 success verdict",
    )
    _require(a1_input["target_roadmap"] == ".447", "A1 input must target .447")
    _require(a1_input["exploration_strategy_class_closed"] is True, "exploration strategy class must stay closed")
    fields = set(a1_input["consumed_state_fields"])
    for field in ("object_ids", "relation_edges", "object_action_bindings"):
        _require(field in fields, f"A1 input missing consumed state field: {field}")


def _validate_mapping_note(mapping_note: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(mapping_note, Mapping), "object_world_model_mapping_note must be a mapping")
    _require(set(mapping_note) == REQUIRED_MAPPING_NOTE_FIELDS, "mapping note must match schema")
    _require(mapping_note["terminal_success"] == HONEST_VERDICT, "mapping note terminal success must match verdict")
    _require(mapping_note["root_cause"] == "object-world-model planning", "mapping note root cause must match")
    _require("A1 object layer" in str(mapping_note["summary"]), "mapping note must mention the A1 object layer")
    _require("proposable winner" in str(mapping_note["summary"]), "mapping note must mention proposable winner")
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
            f"maps to {method['maps_to_frontier']}; "
            f"{method['consumes_a1_object_layer']} "
            f"Object-relational state: {method['object_relational_state']} "
            f"Planning graft: {method['planning_graft']} "
            f"Proposable winner output: {method['proposable_winner_output']} "
            f"Verification handoff: {method['verification_handoff']} "
            f"Takes over: {method['takes_over_from_current_stack']} "
            f"Fails when: {method['fails_when']}"
        )
        for method in result["methods_mapped"]
    )
    flag_lines = "\n".join(flag["flag"] for flag in result["flagged_for_v447"])
    return f"""{STUDYING_SECTION_START}
## 2026-06-27 Exp 4848 - .447 object-world-model planning SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4848_sota_ingestion_object_world_model.json`.

**Preconditions:** `research-studying.md`, `research-references.md`, and
`results/experiment_4838_sota_ingestion_perception_representation.json` were
present. `scripts/sweep_clusters.py` emitted world-model, affordance/action-effect,
and neural-guided world-model cluster URLs. `scripts/sweep_semscholar.py` was
run on three focused object-world-model planning queries; one query returned
eight arXiv IDs and two returned HTTP 429, so no S2-only source was promoted.
Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks verified the
top eight papers listed below. `/deep-research` was not invoked. The nulled
exploration-strategy class was not re-ingested. No model load, training,
leaderboard submission, or solve claim was made; this is a no solve claim
ingestion note.

**A1 object layer imported from Exp 4838:** the .447 question is not more
perception and not more generic exploration. Given object IDs, slots, relation
edges, persistence tracks, object/action bindings, and causal shortcut guards,
the planner must turn object-relational state into a proposable winner on a
novel game.

**Verified source set:**
{citation_lines}

**SOTA -> object-world-model planning mapping for .447:**
{method_lines}

{flag_lines}

**Bottom line for .447:** prioritize COMET/ObjectZero-style object-MCTS as the
main planner, Slot-MPC as the direct object-action optimizer, and
Loop-OWM/FIOC-WM interaction primitives as the ARC-specific proposal layer.
The handoff is object-relational planning that creates a candidate winner, not
another pass over an unchanged exploration pool.
{STUDYING_SECTION_END}"""


def update_research_studying_text(text: str, artifact: JsonMap | None = None) -> str:
    section = build_research_studying_section(artifact)
    start = text.find(STUDYING_SECTION_START)
    if start >= 0:
        end = text.find(STUDYING_SECTION_END, start)
        _require(end >= 0, "research-studying Exp 4848 section missing end marker")
        end += len(STUDYING_SECTION_END)
        before = text[:start].rstrip()
        tail = text[end:].lstrip()
        removed = before + ("\n\n" + tail if tail else "\n")
        insert_at = _studying_insert_index(removed)
        if insert_at >= 0:
            return removed[:insert_at].rstrip() + "\n\n" + section + "\n\n" + removed[insert_at:].lstrip()
        return before + "\n\n" + section + ("\n\n" + tail if tail else "\n")
    insert_at = _studying_insert_index(text)
    return (
        text[:insert_at].rstrip() + "\n\n" + section + "\n\n" + text[insert_at:].lstrip()
        if insert_at >= 0
        else text.rstrip() + "\n\n" + section + "\n"
    )


def _studying_insert_index(text: str) -> int:
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
    _require(start >= 0 and end >= 0, "research-studying missing Exp 4848 section markers")
    section = text[start : end + len(STUDYING_SECTION_END)]
    for phrase in (
        "SOTA -> object-world-model planning mapping",
        "flagged_for_v447",
        "no solve claim",
        "A1 object layer",
        "proposable winner",
        "ObjectZero",
        "COMET",
        "Slot-MPC",
        "FIOC-WM",
        "Loop-OWM",
        "not more generic exploration",
    ):
        _require(phrase in section, f"research-studying section missing required phrase: {phrase}")
    missing_citations = sorted(citation for citation in NOTE_REQUIRED_SOURCE_CITATIONS if citation not in section)
    _require(not missing_citations, f"research-studying section missing citations: {missing_citations}")
    for method in result["methods_mapped"]:
        _require(method["method"] in section, f"research-studying section missing method: {method['method']}")
    for flag in result["flagged_for_v447"]:
        _require(flag["flag"] in section, "research-studying section missing flagged_for_v447 text")


def write_outputs(
    *,
    artifact_path: Path | None = None,
    studying_path: Path | None = None,
    artifact: JsonMap | None = None,
) -> dict[str, object]:
    result = dict(artifact or build_artifact())
    validate_artifact(result)
    target_artifact = artifact_path or Path(RESULT_RELATIVE_PATH)
    target_studying = studying_path or Path(STUDYING_RELATIVE_PATH)
    target_artifact.parent.mkdir(parents=True, exist_ok=True)
    target_artifact.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    studying_text = target_studying.read_text(encoding="utf-8")
    updated = update_research_studying_text(studying_text, result)
    validate_research_studying_text(updated, result)
    target_studying.write_text(updated, encoding="utf-8")
    return result


def main() -> int:
    root = Path(os.environ.get("CARNOT_EXP4848_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        studying_path=root / STUDYING_RELATIVE_PATH,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
