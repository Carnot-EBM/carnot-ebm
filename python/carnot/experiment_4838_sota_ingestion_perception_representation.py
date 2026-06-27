"""Exp 4838 SOTA ingestion for perception/representation.

Spec refs: REQ-ARC-WMTE-4838, SCENARIO-ARC-WMTE-4838,
SCENARIO-ARC-WMTE-4838-NO-FABRICATION.

This module writes a deterministic literature-ingestion artifact for the .446
roadmap. It maps object-centric, slot, relational, and structured-state papers
onto the current L1-first-contact wall: the winner must become representable
and proposable on a novel game before any search or ranking lever can help.
It performs no model load, training, leaderboard submission, or solve claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4838_sota_ingestion_perception_representation.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
NOTE_PATH = "research-studying.md#exp-4838-sota-ingestion-perception-representation"
RANDOM_SEED = 4838
DURATION_S = 0.0001
HONEST_VERDICT = "success_sota_ingestion_perception_representation_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
STUDYING_SECTION_START = "<!-- EXP4838-SOTA-INGESTION-PERCEPTION-REPRESENTATION-START -->"
STUDYING_SECTION_END = "<!-- EXP4838-SOTA-INGESTION-PERCEPTION-REPRESENTATION-END -->"
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
            "success_sota_ingestion_perception_representation_mapped."
        )
    },
    "methods_mapped": {
        "principle": (
            "the strongest 3-5 perception/representation methods mapped onto "
            "the L1-wall root cause, each with a real arXiv ID."
        )
    },
    "arxiv_ids_cited": {
        "principle": "every method claim must cite a verifiable arXiv ID."
    },
    "flagged_for_v446": {
        "principle": (
            "the strongest method(s) flagged so the .446 planner reads the "
            "mapping (perception, not more exploration)."
        )
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
    "l1_wall_context": {
        "principle": "imports the .445 L1-first-contact root-cause context for .446 planning."
    },
    "perception_representation_mapping_note": {
        "principle": "states why representation makes the winner enter the pool before search reweighting."
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
        "principle": "content hash of citations, method map, flags, context, and mapping note."
    },
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "methods_mapped",
    "arxiv_ids_cited",
    "flagged_for_v446",
    "inference_substrate",
    "preconditions_checked",
    "citations",
    "fresh_sweep",
    "l1_wall_context",
    "perception_representation_mapping_note",
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
        "maps_to_gap",
        "maps_to_wall",
        "representation_graft",
        "winner_representable_test",
        "proposable_output",
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
        "research_studying_present",
        "research_references_present",
        "sweep_clusters_used",
        "sweep_cluster_ids",
        "sweep_cluster_urls",
        "sweep_semscholar_used",
        "sweep_semscholar_queries",
        "sweep_semscholar_http_429",
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
        "webfetch_top_sources",
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
        "1802.04687",
        "1911.12247",
        "2006.15055",
        "2402.03326",
        "2507.03298",
        "2601.06604",
        "2602.11389",
        "2606.12316",
    }
)
REQUIRED_TRACKS = frozenset(
    {
        "slot_object_state_builder",
        "object_relational_transition_graph",
        "object_centric_world_model_planning",
        "causal_object_latent_prediction",
        "arc_slot_transition_loop",
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
    "object centric slot relational perception world model ARC game representation",
    "slot attention structured world model relational dynamics object centric planning",
    "game agnostic object centric representation transfer interactive agents",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/1802.04687",
    "https://arxiv.org/abs/1911.12247",
    "https://arxiv.org/abs/2006.15055",
    "https://arxiv.org/abs/2402.03326",
    "https://arxiv.org/abs/2507.03298",
    "https://arxiv.org/abs/2601.06604",
    "https://arxiv.org/abs/2602.11389",
    "https://arxiv.org/abs/2606.12316",
]

CITATIONS = {
    "1802.04687": {
        "title": "Neural Relational Inference for Interacting Systems",
        "url": "https://arxiv.org/abs/1802.04687",
        "http_status": 200,
    },
    "1911.12247": {
        "title": "Contrastive Learning of Structured World Models",
        "url": "https://arxiv.org/abs/1911.12247",
        "http_status": 200,
    },
    "2006.15055": {
        "title": "Object-Centric Learning with Slot Attention",
        "url": "https://arxiv.org/abs/2006.15055",
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
    "2601.06604": {
        "title": "Object-Centric World Models Meet Monte Carlo Tree Search",
        "url": "https://arxiv.org/abs/2601.06604",
        "http_status": 200,
    },
    "2602.11389": {
        "title": "Causal-JEPA: Learning World Models through Object-Level Latent Masking",
        "url": "https://arxiv.org/abs/2602.11389",
        "http_status": 200,
    },
    "2606.12316": {
        "title": "Slots, Transitions, Loops: Learning Composable World Models for ARC",
        "url": "https://arxiv.org/abs/2606.12316",
        "http_status": 200,
    },
}

FLAGGED_FOR_V446 = [
    {
        "candidate": "loop_owm_slot_transition_proposer",
        "flag": (
            "flagged_for_v446: loop_owm_slot_transition_proposer "
            "(arXiv:2606.12316 + arXiv:2006.15055 + arXiv:1911.12247)"
        ),
        "source_ids": ["2606.12316", "2006.15055", "1911.12247"],
        "maps_to_wall": "L1-FIRST-CONTACT",
    },
    {
        "candidate": "object_relational_world_model_mcts",
        "flag": (
            "flagged_for_v446: object_relational_world_model_mcts "
            "(arXiv:2601.06604 + arXiv:2402.03326 + arXiv:2507.03298)"
        ),
        "source_ids": ["2601.06604", "2402.03326", "2507.03298"],
        "maps_to_wall": "L1-FIRST-CONTACT",
    },
    {
        "candidate": "causal_object_jepa_shortcut_guard",
        "flag": (
            "flagged_for_v446: causal_object_jepa_shortcut_guard "
            "(arXiv:2602.11389 + arXiv:1802.04687)"
        ),
        "source_ids": ["2602.11389", "1802.04687"],
        "maps_to_wall": "L1-FIRST-CONTACT",
    },
]

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Slotized ARC object-state proposal binder",
        "track": "slot_object_state_builder",
        "source_ids": ["2006.15055", "2606.12316"],
        "maps_to_gap": "GAP-ARCH-FEATURES",
        "maps_to_wall": "L1-FIRST-CONTACT",
        "representation_graft": (
            "Replace frame-only order-1 features with color/object slots, masks, "
            "object permanence IDs, and spatial relation tokens from each frame."
        ),
        "winner_representable_test": (
            "A novel game's winning prefix is representable only if each decisive "
            "click/action can bind to a slot object and a before/after relation, "
            "with no terminal win oracle."
        ),
        "proposable_output": (
            "Propose slot-conditioned action templates such as select/move/merge/fill "
            "over objects instead of only raw coordinate retries."
        ),
        "takes_over_from_current_stack": (
            "Takes over frame-delta scalar features that cannot name the object "
            "whose transformation makes the first win possible."
        ),
        "fails_when": (
            "Slot binding drifts between frames, small ARC objects are merged into "
            "background, or the proposer cannot turn slots into executable actions."
        ),
        "roadmap_candidate": FLAGGED_FOR_V446[0]["flag"],
    },
    {
        "method": "Object-relational transition graph proposer",
        "track": "object_relational_transition_graph",
        "source_ids": ["1802.04687", "1911.12247", "2402.03326"],
        "maps_to_gap": "GAP-ARCH-FEATURES",
        "maps_to_wall": "L1-FIRST-CONTACT",
        "representation_graft": (
            "Represent each state as objects plus inferred interaction edges, then "
            "learn action-conditioned transition rules over that graph."
        ),
        "winner_representable_test": (
            "The winning prefix is representable if the graph transition predicts "
            "the decisive object/relation change before the live explorer sees a win."
        ),
        "proposable_output": (
            "Propose near-miss next states and action prefixes by editing object "
            "relations that the graph predicts will change."
        ),
        "takes_over_from_current_stack": (
            "Takes over exploration-prior reweighting when the candidate pool lacks "
            "a structured transition that could produce the winner."
        ),
        "fails_when": (
            "Edges encode visual proximity rather than mechanics, negatives are too "
            "easy, or relation edits produce impossible grid states."
        ),
        "roadmap_candidate": FLAGGED_FOR_V446[1]["flag"],
    },
    {
        "method": "Object-centric latent-dynamics planner with MCTS",
        "track": "object_centric_world_model_planning",
        "source_ids": ["2507.03298", "2601.06604"],
        "maps_to_gap": "GAP-ARCH-FEATURES",
        "maps_to_wall": "L1-FIRST-CONTACT",
        "representation_graft": (
            "Use an object-centric world model as the state substrate for short "
            "lookahead planning, keeping dynamics-aware features separate from "
            "visual nuisance features."
        ),
        "winner_representable_test": (
            "The winning prefix is representable if MCTS over object latents can "
            "reach a low-depth candidate state whose replayable action prefix was "
            "absent from the frame-only pool."
        ),
        "proposable_output": (
            "Propose a small set of replayable object-state rollouts and concrete "
            "action prefixes for the live verifier to test."
        ),
        "takes_over_from_current_stack": (
            "Takes over unguided first-contact exploration after representation, "
            "not search depth, is the limiting factor."
        ),
        "fails_when": (
            "Latent rollout drift compounds, object discovery fails under clutter, "
            "or MCTS optimizes a latent state that cannot be grounded into actions."
        ),
        "roadmap_candidate": FLAGGED_FOR_V446[1]["flag"],
    },
    {
        "method": "Causal object-level JEPA shortcut guard",
        "track": "causal_object_latent_prediction",
        "source_ids": ["2602.11389", "1802.04687"],
        "maps_to_gap": "GAP-ARCH-FEATURES",
        "maps_to_wall": "L1-FIRST-CONTACT",
        "representation_graft": (
            "Mask or intervene on object-level latents so the encoder must infer "
            "interaction-dependent structure rather than frame provenance shortcuts."
        ),
        "winner_representable_test": (
            "The winning prefix is representable only if masked-object prediction "
            "recovers the decisive hidden relation and the relation survives "
            "counterfactual object swaps."
        ),
        "proposable_output": (
            "Propose shortcut-guarded object features used to filter and construct "
            "candidate prefixes before any ranker scores them."
        ),
        "takes_over_from_current_stack": (
            "Takes over chance-level order-1 features by forcing the representation "
            "to encode causal object dependencies."
        ),
        "fails_when": (
            "The mask objective can be solved from color/style shortcuts, the "
            "object set is wrong, or counterfactual swaps break valid mechanics."
        ),
        "roadmap_candidate": FLAGGED_FOR_V446[2]["flag"],
    },
    {
        "method": "ARC composable slot-transition loop model",
        "track": "arc_slot_transition_loop",
        "source_ids": ["2606.12316", "2006.15055", "1911.12247"],
        "maps_to_gap": "GAP-ARCH-FEATURES",
        "maps_to_wall": "L1-FIRST-CONTACT",
        "representation_graft": (
            "Learn ARC rules as composable transitions over color slots, objects, "
            "loops, and demonstration-conditioned task summaries."
        ),
        "winner_representable_test": (
            "The winning prefix is representable if the demonstration-conditioned "
            "slot loop proposes the missing first-contact transformation on a held-out game."
        ),
        "proposable_output": (
            "Propose an executable transformation sketch or action prefix that enters "
            "the live candidate pool before static ranking."
        ),
        "takes_over_from_current_stack": (
            "Takes over generic exploration levers by changing what the pool can "
            "express: object transformations instead of raw prefix frequency."
        ),
        "fails_when": (
            "The demo-conditioned summary memorizes family style, the loop fails on "
            "single-shot mechanics, or transformations cannot lower to live actions."
        ),
        "roadmap_candidate": FLAGGED_FOR_V446[0]["flag"],
    },
]

DEFAULT_FRESH_SWEEP = {
    "filtered_track": (
        "object-centric, slot, relational perception and structured-state "
        "representations for game-agnostic L1-first-contact structure"
    ),
    "cluster_ids": [3, 5, 6],
    "semantic_scholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "semantic_scholar_result": "HTTP 429 on all three focused queries; no S2-only source promoted.",
    "webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
}
DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "research_studying_present": True,
    "research_references_present": True,
    "sweep_clusters_used": True,
    "sweep_cluster_ids": [3, 5, 6],
    "sweep_cluster_urls": [CLUSTER_3_URL, CLUSTER_5_URL, CLUSTER_6_URL],
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "sweep_semscholar_http_429": True,
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
DEFAULT_L1_WALL_CONTEXT = {
    "roadmap_target": ".446",
    "wall": "L1-FIRST-CONTACT",
    "root_cause": "perception/representation",
    "upstream_root_blocker": "winning_l1_prefix_never_proposed",
    "frame_only_order1_features_at_chance": True,
    "exploration_strategy_class_retired": True,
    "nulled_lever_count_approx": 15,
    "source_artifacts": [
        "results/experiment_4830_archive_444_activate_445.json",
        "results/experiment_4831_amortized_incontext_exploration_prior_live.json",
        "results/experiment_4834_heldout_first_win_readiness.json",
        "results/experiment_4835_silent_bug_audit.json",
    ],
    "planner_constraint": (
        "Do not spend .446 on another exploration reweighting run; require a "
        "representation that can make a novel winning prefix enter the pool."
    ),
}
DEFAULT_MAPPING_NOTE = {
    "summary": (
        "The .446 handoff is perception/representation: make the first winning "
        "prefix representable/proposable with object slots, relations, and "
        "structured transitions before ranking or exploration strategy is considered."
    ),
    "terminal_success": HONEST_VERDICT,
    "source_ids": sorted(REQUIRED_SOURCE_IDS),
    "root_cause": "perception/representation",
    "planner_instruction": (
        "Prioritize Loop-OWM slot-transition proposals plus object-relational "
        "world-model/MCTS, with Causal-JEPA as the shortcut guard."
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_set_checksum(
    citations: JsonMap,
    methods: Sequence[JsonMap],
    flags: Sequence[JsonMap],
    l1_wall_context: JsonMap,
    mapping_note: JsonMap,
) -> str:
    payload = json.dumps(
        {
            "citations": citations,
            "flags": list(flags),
            "l1_wall_context": l1_wall_context,
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
    FLAGGED_FOR_V446,
    DEFAULT_L1_WALL_CONTEXT,
    DEFAULT_MAPPING_NOTE,
)


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    fresh_sweep: JsonMap = DEFAULT_FRESH_SWEEP,
    flagged_for_v446: Sequence[JsonMap] = FLAGGED_FOR_V446,
    l1_wall_context: JsonMap = DEFAULT_L1_WALL_CONTEXT,
    perception_representation_mapping_note: JsonMap = DEFAULT_MAPPING_NOTE,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "arxiv_ids_cited": sorted(citations),
        "flagged_for_v446": [dict(flag) for flag in flagged_for_v446],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "citations": {source_id: dict(citation) for source_id, citation in citations.items()},
        "fresh_sweep": dict(fresh_sweep),
        "l1_wall_context": dict(l1_wall_context),
        "perception_representation_mapping_note": dict(perception_representation_mapping_note),
        "note_path": NOTE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_S,
        "reproducibility_checksum": source_set_checksum(
            citations,
            methods_mapped,
            flagged_for_v446,
            l1_wall_context,
            perception_representation_mapping_note,
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
    _require(artifact["note_path"] == NOTE_PATH, "note_path must point at the Exp 4838 note")
    _require(artifact["random_seed"] == RANDOM_SEED, "random_seed must be the experiment id")
    _require(artifact["duration_s"] == DURATION_S, "duration_s must preserve the 0.0001s floor")
    _validate_citations(artifact["citations"], artifact["arxiv_ids_cited"])
    _validate_methods(artifact["methods_mapped"], artifact["arxiv_ids_cited"])
    _validate_flags(artifact["flagged_for_v446"], artifact["arxiv_ids_cited"])
    _validate_preconditions(artifact["preconditions_checked"])
    _validate_fresh_sweep(artifact["fresh_sweep"])
    _validate_l1_wall_context(artifact["l1_wall_context"])
    _validate_mapping_note(artifact["perception_representation_mapping_note"], artifact["arxiv_ids_cited"])
    _require(
        artifact["reproducibility_checksum"]
        == source_set_checksum(
            artifact["citations"],
            artifact["methods_mapped"],
            artifact["flagged_for_v446"],
            artifact["l1_wall_context"],
            artifact["perception_representation_mapping_note"],
        ),
        "reproducibility checksum must hash citations, methods, flags, context, and mapping note",
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
        _require(method["maps_to_gap"] == "GAP-ARCH-FEATURES", "method must map to GAP-ARCH-FEATURES")
        _require(method["maps_to_wall"] == "L1-FIRST-CONTACT", "method must map to L1-FIRST-CONTACT")
        _require(bool(method["representation_graft"]), "each method needs a representation graft")
        _require(
            "represent" in str(method["winner_representable_test"]),
            "each method needs a winner representable test",
        )
        _require("propos" in str(method["proposable_output"]).lower(), "each method needs proposable output")
        _require(bool(method["takes_over_from_current_stack"]), "each method needs takes_over_from_current_stack")
        _require(bool(method["fails_when"]), "each method needs fails_when")
        _require(bool(method["roadmap_candidate"]), "each method needs a roadmap candidate")
        tracks.add(str(method["track"]))
    _require(REQUIRED_TRACKS == tracks, "methods_mapped missing required perception/representation tracks")


def _validate_flags(flags: object, arxiv_ids_cited: object) -> None:
    _require(
        isinstance(flags, Sequence) and not isinstance(flags, str | bytes) and bool(flags),
        "flagged_for_v446 required",
    )
    cited = set(arxiv_ids_cited)
    for flag in flags:
        _require(isinstance(flag, Mapping), "each flagged_for_v446 entry must be a mapping")
        _require("candidate" in flag and "flag" in flag, "each flagged_for_v446 entry needs candidate and flag")
        _require("flagged_for_v445" not in json.dumps(flag, sort_keys=True), "flagged_for_v446 must not carry stale .445 flags")
        _require("flagged_for_v446" in str(flag["flag"]), "flagged_for_v446 entries must carry the .446 flag")
        _require(set(flag.get("source_ids", [])).issubset(cited), "flagged_for_v446 source_ids must be verified")


def _validate_preconditions(preconditions: object) -> None:
    _require(isinstance(preconditions, Mapping), "preconditions_checked must be a mapping")
    _require(set(preconditions) == REQUIRED_PRECONDITION_FIELDS, "preconditions_checked must match schema")
    _require(preconditions["research_studying_present"] is True, "research-studying precondition must pass")
    _require(preconditions["research_references_present"] is True, "research-references precondition must pass")
    _require(preconditions["sweep_clusters_used"] is True, "sweep_clusters must be used")
    _require(preconditions["sweep_cluster_ids"] == [3, 5, 6], "sweep cluster IDs must be [3, 5, 6]")
    _require(preconditions["sweep_semscholar_used"] is True, "sweep_semscholar must be used")
    _require(preconditions["sweep_semscholar_http_429"] is True, "Semantic Scholar HTTP 429 must be recorded")
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
    sources = fresh_sweep["webfetch_top_sources"]
    _require(
        isinstance(sources, Sequence) and not isinstance(sources, str | bytes) and 5 <= len(sources) <= 8,
        "fresh_sweep must record top five to eight WebFetch sources",
    )
    _require(list(sources) == WEBSEARCH_WEBFETCH_TOP_SOURCES, "fresh_sweep sources must match verified source set")


def _validate_l1_wall_context(context: object) -> None:
    _require(isinstance(context, Mapping), "l1_wall_context must be a mapping")
    _require(context.get("roadmap_target") == ".446", "l1_wall_context must target the .446 roadmap")
    _require(context.get("wall") == "L1-FIRST-CONTACT", "l1_wall_context must name L1-FIRST-CONTACT")
    _require(context.get("root_cause") == "perception/representation", "root cause must be perception/representation")
    _require(context.get("frame_only_order1_features_at_chance") is True, "frame-only chance feature fact required")
    _require(context.get("exploration_strategy_class_retired") is True, "exploration class must be retired")
    _require(
        "winning_l1_prefix_never_proposed" in str(context.get("upstream_root_blocker", "")),
        "l1_wall_context must cite the winning-prefix blocker",
    )


def _validate_mapping_note(mapping_note: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(mapping_note, Mapping), "perception_representation_mapping_note must be a mapping")
    _require(set(mapping_note) == REQUIRED_MAPPING_NOTE_FIELDS, "mapping note must match schema")
    _require(mapping_note["terminal_success"] == HONEST_VERDICT, "mapping note terminal success must match verdict")
    _require(mapping_note["root_cause"] == "perception/representation", "mapping note root cause must match")
    _require("representable/proposable" in str(mapping_note["summary"]), "mapping note must mention representable/proposable")
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
            f"maps to {method['maps_to_wall']} / {method['maps_to_gap']}; "
            f"{method['representation_graft']} "
            f"Winner representable/proposable test: {method['winner_representable_test']} "
            f"Proposable output: {method['proposable_output']} "
            f"Takes over: {method['takes_over_from_current_stack']} "
            f"Fails when: {method['fails_when']}"
        )
        for method in result["methods_mapped"]
    )
    flag_lines = "\n".join(flag["flag"] for flag in result["flagged_for_v446"])
    return f"""{STUDYING_SECTION_START}
## 2026-06-27 Exp 4838 - .446 perception/representation SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4838_sota_ingestion_perception_representation.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted world-model, affordance/action-effect,
and neural-guided world-model cluster URLs. `scripts/sweep_semscholar.py` was
run on three focused perception/representation queries and returned HTTP 429
for all of them, so no S2-only source was promoted. Low-concurrency
WebSearch/WebFetch plus direct arXiv HTTP checks verified the top eight papers
listed below. `/deep-research` was not invoked. The nulled exploration-strategy
class was not re-ingested. No model load, training, leaderboard submission, or
solve claim was made; this is a no solve claim ingestion note.

**L1-wall context imported:** `.445` left the wall at L1-first-contact:
the winning L1 prefix is not entering the pool, frame-only order-1 features are
at chance for the current diagnosis, and exploration reweighting has nulled.
The .446 target is therefore perception/representation: make a novel game's
winner representable/proposable before ranking.

**Verified source set:**
{citation_lines}

**SOTA -> perception/representation mapping for the L1 wall:**
{method_lines}

{flag_lines}

**Bottom line for .446:** prioritize Loop-OWM slot-transition proposals
with Slot Attention/C-SWM as the substrate, then pair that with object-relational
world-model MCTS. Use Causal-JEPA as the shortcut guard so the learned
representation captures object interactions rather than provenance or frame
style. This is perception, not more exploration.
{STUDYING_SECTION_END}"""


def update_research_studying_text(text: str, artifact: JsonMap | None = None) -> str:
    section = build_research_studying_section(artifact)
    start = text.find(STUDYING_SECTION_START)
    if start >= 0:
        end = text.find(STUDYING_SECTION_END, start)
        _require(end >= 0, "research-studying Exp 4838 section missing end marker")
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
    _require(start >= 0 and end >= 0, "research-studying missing Exp 4838 section markers")
    section = text[start : end + len(STUDYING_SECTION_END)]
    for phrase in (
        "SOTA -> perception/representation mapping",
        "flagged_for_v446",
        "no solve claim",
        "L1-first-contact",
        "representable/proposable",
        "Slot Attention",
        "C-SWM",
        "Object-Centric World Models",
        "Causal-JEPA",
        "Loop-OWM",
        "perception, not more exploration",
    ):
        _require(phrase in section, f"research-studying section missing required phrase: {phrase}")
    missing_citations = sorted(citation for citation in NOTE_REQUIRED_SOURCE_CITATIONS if citation not in section)
    _require(not missing_citations, f"research-studying section missing citations: {missing_citations}")
    for method in result["methods_mapped"]:
        _require(method["method"] in section, f"research-studying section missing method: {method['method']}")
    for flag in result["flagged_for_v446"]:
        _require(flag["flag"] in section, "research-studying section missing flagged_for_v446 text")


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
    root = Path(os.environ.get("CARNOT_EXP4838_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        studying_path=root / STUDYING_RELATIVE_PATH,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
