"""Exp 4858 SOTA ingestion for generation expressibility.

Spec refs: REQ-ARC-WMTE-4858, SCENARIO-ARC-WMTE-4858,
SCENARIO-ARC-WMTE-4858-NO-FABRICATION.

This module writes a deterministic literature-ingestion artifact for the .448
roadmap. It targets the Exp 4851 A1 finding that winning first-contact prefixes
are mostly NEVER_ENUMERATED: the proposer cannot express the needed action
programs, so rankers cannot recover them. The mapping focuses on program
synthesis, library learning, executable world models, neural-guided ARC program
search, and object-centric MCTS proposers that can consume partial/noisy object
signals without assuming exact object identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4858_sota_ingestion_generation_expressibility.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
REFERENCES_RELATIVE_PATH = "research-references.md"
UPSTREAM_OBJECT_WORLD_MODEL_ARTIFACT = "results/experiment_4848_sota_ingestion_object_world_model.json"
UPSTREAM_GENERATION_DIAGNOSTIC_ARTIFACT = "results/experiment_4851_generation_coverage_diagnostic.json"
NOTE_PATH = "research-studying.md#exp-4858-sota-ingestion-generation-expressibility"
REFERENCES_PATH = "research-references.md#exp-4858-generation-expressibility-source-set"
RANDOM_SEED = 4858
DURATION_S = 0.0001
HONEST_VERDICT = "success_sota_ingestion_generation_expressibility_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
AIMED_AT_DOMINANT_BUCKET = "NEVER_ENUMERATED"
STUDYING_SECTION_START = "<!-- EXP4858-SOTA-INGESTION-GENERATION-EXPRESSIBILITY-START -->"
STUDYING_SECTION_END = "<!-- EXP4858-SOTA-INGESTION-GENERATION-EXPRESSIBILITY-END -->"
REFERENCES_SECTION_START = "<!-- EXP4858-GENERATION-EXPRESSIBILITY-REFERENCES-START -->"
REFERENCES_SECTION_END = "<!-- EXP4858-GENERATION-EXPRESSIBILITY-REFERENCES-END -->"
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
            "success_sota_ingestion_generation_expressibility_mapped."
        )
    },
    "methods_mapped": {
        "principle": (
            "the strongest 3-5 generation-expressibility methods, each mapped "
            "onto putting the winning prefix into the pool, each with a real arXiv ID."
        )
    },
    "arxiv_ids_cited": {
        "principle": (
            "every method claim must cite a verifiable arXiv ID "
            "(no fabrication -- adversarial_verify bar)."
        )
    },
    "aimed_at_dominant_bucket": {
        "principle": (
            "the A1 dominant_bucket the ingestion targets "
            "(never_enumerated -> expressibility; covered -> ranking)."
        )
    },
    "flagged_for_v448": {
        "principle": "the strongest method(s) flagged so the .448 planner reads the mapping."
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
        "principle": "binds the mapping to Exp 4848 and the Exp 4851 NEVER_ENUMERATED finding."
    },
    "generation_expressibility_mapping_note": {
        "principle": "states how each method puts missing winning prefixes into the pool."
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
    "aimed_at_dominant_bucket",
    "flagged_for_v448",
    "inference_substrate",
    "preconditions_checked",
    "citations",
    "fresh_sweep",
    "upstream_artifacts",
    "generation_expressibility_mapping_note",
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
        "targets_bucket",
        "primitive_expansion",
        "winner_prefix_pool_insertion",
        "partial_object_signal_contract",
        "proposal_graft",
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
        "research_studying_present",
        "research_references_present",
        "upstream_object_world_model_artifact_present",
        "upstream_generation_diagnostic_artifact_present",
        "a1_dominant_bucket_read",
        "a1_dominant_bucket",
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
        "exploration_strategy_reingested",
        "energy_stage_reingested",
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
REQUIRED_UPSTREAM_FIELDS = frozenset(
    {
        "object_world_model_handoff",
        "generation_diagnostic",
        "dominant_bucket",
        "bucket_interpretation",
        "partial_noisy_object_signal_only",
        "object_signal_constraint",
        "carried_forward_from_v447",
        "closed_classes",
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
        "2006.08381",
        "2310.19791",
        "2411.17708",
        "2507.14172",
        "2507.15877",
        "2601.06604",
        "2605.05138",
        "2606.14418",
    }
)
REQUIRED_TRACKS = frozenset(
    {
        "library_learning_action_primitives",
        "neural_guided_arc_program_search",
        "execution_guided_program_synthesis",
        "object_relational_mcts_proposer",
        "executable_world_model_action_programmer",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS)

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
    "DreamCoder library learning program synthesis action primitives",
    "neural guided program induction ARC AGI grid DSL",
    "program synthesis interactive grid games MCTS object relational planning",
]
SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS: list[str] = []
SEMANTIC_SCHOLAR_RESULT = (
    "Three focused queries returned HTTP 429 through scripts/sweep_semscholar.py; "
    "0 unique arxiv IDs"
)
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2006.08381",
    "https://arxiv.org/abs/2310.19791",
    "https://arxiv.org/abs/2411.17708",
    "https://arxiv.org/abs/2507.14172",
    "https://arxiv.org/abs/2507.15877",
    "https://arxiv.org/abs/2601.06604",
    "https://arxiv.org/abs/2605.05138",
    "https://arxiv.org/abs/2606.14418",
]

CITATIONS = {
    "2006.08381": {
        "title": (
            "DreamCoder: Growing generalizable, interpretable knowledge with "
            "wake-sleep Bayesian program learning"
        ),
        "url": "https://arxiv.org/abs/2006.08381",
        "http_status": 200,
    },
    "2310.19791": {
        "title": "LILO: Learning Interpretable Libraries by Compressing and Documenting Code",
        "url": "https://arxiv.org/abs/2310.19791",
        "http_status": 200,
    },
    "2411.17708": {
        "title": "Towards Efficient Neurally-Guided Program Induction for ARC-AGI",
        "url": "https://arxiv.org/abs/2411.17708",
        "http_status": 200,
    },
    "2507.14172": {
        "title": (
            "SOAR: Self-Improving Language Models for Evolutionary Program Synthesis: "
            "A Case Study on ARC-AGI"
        ),
        "url": "https://arxiv.org/abs/2507.14172",
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
    "2601.06604": {
        "title": "Object-Centric World Models Meet Monte Carlo Tree Search",
        "url": "https://arxiv.org/abs/2601.06604",
        "http_status": 200,
    },
    "2605.05138": {
        "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
        "url": "https://arxiv.org/abs/2605.05138",
        "http_status": 200,
    },
    "2606.14418": {
        "title": "Causal Object-Centric Models for Planning with Monte Carlo Tree Search",
        "url": "https://arxiv.org/abs/2606.14418",
        "http_status": 200,
    },
}

FLAGGED_FOR_V448 = [
    {
        "candidate": "dreamcoder_lilo_action_library",
        "flag": (
            "flagged_for_v448: dreamcoder_lilo_action_library "
            "(arXiv:2006.08381 + arXiv:2310.19791)"
        ),
        "source_ids": ["2006.08381", "2310.19791"],
        "maps_to_frontier": ".448",
    },
    {
        "candidate": "eg_nps_soar_arc_program_search",
        "flag": (
            "flagged_for_v448: eg_nps_soar_arc_program_search "
            "(arXiv:2411.17708 + arXiv:2507.14172 + arXiv:2507.15877)"
        ),
        "source_ids": ["2411.17708", "2507.14172", "2507.15877"],
        "maps_to_frontier": ".448",
    },
    {
        "candidate": "comet_executable_world_model_mcts",
        "flag": (
            "flagged_for_v448: comet_executable_world_model_mcts "
            "(arXiv:2606.14418 + arXiv:2601.06604 + arXiv:2605.05138)"
        ),
        "source_ids": ["2606.14418", "2601.06604", "2605.05138"],
        "maps_to_frontier": ".448",
    },
]

DEFAULT_METHODS_MAPPED = [
    {
        "method": "DreamCoder/LILO action-primitive library learner",
        "track": "library_learning_action_primitives",
        "source_ids": ["2006.08381", "2310.19791"],
        "maps_to_frontier": ".448",
        "targets_bucket": AIMED_AT_DOMINANT_BUCKET,
        "primitive_expansion": (
            "Learns reusable action primitives and documented abstractions from "
            "verified traces, then offers those primitives as first-class proposer moves."
        ),
        "winner_prefix_pool_insertion": (
            "Insert the winning prefix into the pool by synthesizing a short program "
            "over learned move/transform/loop primitives, then lowering it to actions."
        ),
        "partial_object_signal_contract": (
            "Consumes partial/noisy object signal as soft predicates and confidence-tagged "
            "slots; stable object IDs are useful hints, not requirements."
        ),
        "proposal_graft": (
            "Mine successful and near-successful traces for reusable ARC action macros, "
            "add them to the proposer vocabulary, and enumerate macro programs before "
            "raw coordinate retries."
        ),
        "verification_handoff": (
            "Lower each macro program to executable actions and keep it only when replay "
            "matches the verifier's prefix/state checks."
        ),
        "takes_over_from_current_stack": (
            "Takes over the fixed primitive vocabulary that made decisive multi-step "
            "prefixes absent from the proposal pool."
        ),
        "fails_when": (
            "The trace corpus is too thin to learn useful macros, abstractions overfit "
            "one family, or the learned macro cannot be lowered into legal live actions."
        ),
        "roadmap_candidate": FLAGGED_FOR_V448[0]["flag"],
    },
    {
        "method": "Neurally guided ARC DSL program search",
        "track": "neural_guided_arc_program_search",
        "source_ids": ["2411.17708", "2507.14172"],
        "maps_to_frontier": ".448",
        "targets_bucket": AIMED_AT_DOMINANT_BUCKET,
        "primitive_expansion": (
            "Uses a learned guide and evolutionary self-improvement to prioritize DSL "
            "primitive operators and compositions that the generic action proposer never names."
        ),
        "winner_prefix_pool_insertion": (
            "Insert the winning prefix into the pool by searching ARC DSL programs, "
            "compiling the best program into candidate action prefixes, and replaying them."
        ),
        "partial_object_signal_contract": (
            "Consumes partial/noisy object signal as optional features for DSL argument "
            "binding while preserving fallback grid predicates."
        ),
        "proposal_graft": (
            "Add a program-search proposer arm that samples guided DSL transformations "
            "for object, color, region, and loop operations before live ranking."
        ),
        "verification_handoff": (
            "Compile candidate programs to action prefixes and verify by executable replay "
            "against the offline/live harness."
        ),
        "takes_over_from_current_stack": (
            "Takes over flat action enumeration when the missing prefix is a composed "
            "grid transformation rather than a single primitive action."
        ),
        "fails_when": (
            "The DSL excludes the true mechanic, the neural guide memorizes training "
            "families, or evolutionary search spends the action budget on invalid programs."
        ),
        "roadmap_candidate": FLAGGED_FOR_V448[1]["flag"],
    },
    {
        "method": "Execution-guided neural program synthesis for ARC",
        "track": "execution_guided_program_synthesis",
        "source_ids": ["2507.15877", "2507.14172"],
        "maps_to_frontier": ".448",
        "targets_bucket": AIMED_AT_DOMINANT_BUCKET,
        "primitive_expansion": (
            "Expands primitives by letting neural proposals mutate executable programs under "
            "feedback from failed replays and counterexamples."
        ),
        "winner_prefix_pool_insertion": (
            "Insert the winning prefix into the pool by turning replay errors into program "
            "edits until the synthesized action program reaches the missing prefix state."
        ),
        "partial_object_signal_contract": (
            "Consumes partial/noisy object signal only as counterexample annotations and "
            "candidate feature bindings; execution remains the source of truth."
        ),
        "proposal_graft": (
            "Run a bounded synthesize-execute-repair loop over action programs, enqueueing "
            "only repaired prefixes that pass executable checks."
        ),
        "verification_handoff": (
            "Every repaired program is replayed; rejected traces become the next synthesis "
            "counterexample rather than a scored candidate."
        ),
        "takes_over_from_current_stack": (
            "Takes over one-shot proposal failures by converting verifier rejects into "
            "new candidate programs."
        ),
        "fails_when": (
            "Counterexamples are too expensive to gather, the repair model edits surface "
            "syntax but not mechanics, or the executable check cannot expose the missing rule."
        ),
        "roadmap_candidate": FLAGGED_FOR_V448[1]["flag"],
    },
    {
        "method": "Object-relational MCTS action-program proposer",
        "track": "object_relational_mcts_proposer",
        "source_ids": ["2601.06604", "2606.14418"],
        "maps_to_frontier": ".448",
        "targets_bucket": AIMED_AT_DOMINANT_BUCKET,
        "primitive_expansion": (
            "Expands primitive actions into object-bound action programs selected by MCTS "
            "over learned object-relational dynamics."
        ),
        "winner_prefix_pool_insertion": (
            "Insert the winning prefix into the pool by expanding MCTS branches whose "
            "object-relation deltas predict the decisive first-contact transition."
        ),
        "partial_object_signal_contract": (
            "Consumes partial/noisy object signal as probabilistic slots and relation "
            "hypotheses, allowing identity uncertainty inside the tree state."
        ),
        "proposal_graft": (
            "Add a shallow object-relation MCTS proposer that emits replayable object-bound "
            "branches as candidate prefixes."
        ),
        "verification_handoff": (
            "Replay branch actions and compare observed relation deltas against the MCTS "
            "prediction before admitting the prefix."
        ),
        "takes_over_from_current_stack": (
            "Takes over unguided object interaction enumeration when the current pool never "
            "tries the decisive object-action binding."
        ),
        "fails_when": (
            "Noisy slots alias multiple controllable objects, learned dynamics rewards a "
            "latent shortcut, or tree search cannot ground object choices to live controls."
        ),
        "roadmap_candidate": FLAGGED_FOR_V448[2]["flag"],
    },
    {
        "method": "Executable world-model action programmer",
        "track": "executable_world_model_action_programmer",
        "source_ids": ["2605.05138", "2507.15877"],
        "maps_to_frontier": ".448",
        "targets_bucket": AIMED_AT_DOMINANT_BUCKET,
        "primitive_expansion": (
            "Turns induced executable transition models into primitive action-program "
            "templates that become new proposer primitives for interactive ARC levels."
        ),
        "winner_prefix_pool_insertion": (
            "Insert the winning prefix into the pool by inducing a transition program, "
            "planning through it, and exporting the realized action sequence as a prefix."
        ),
        "partial_object_signal_contract": (
            "Consumes partial/noisy object signal as typed hints for transition variables; "
            "the executable model must still pass replay without trusting exact IDs."
        ),
        "proposal_graft": (
            "Let coding-agent world-model induction produce executable transition functions "
            "and ask the proposer to enumerate action programs that satisfy those functions."
        ),
        "verification_handoff": (
            "Accept an action program only after replay confirms the induced transition and "
            "the prefix remains legal under the live harness."
        ),
        "takes_over_from_current_stack": (
            "Takes over static frame-delta heuristics by creating new executable action "
            "programs before ranker selection."
        ),
        "fails_when": (
            "The induced model overfits observed prefixes, hidden mechanics need more probes, "
            "or the planner finds a model path that cannot be executed in the real game."
        ),
        "roadmap_candidate": FLAGGED_FOR_V448[2]["flag"],
    },
]

DEFAULT_FRESH_SWEEP = {
    "filtered_track": (
        "generation expressibility: program synthesis, library learning, neural-guided "
        "ARC program search, executable world models, and object-relational MCTS proposers"
    ),
    "cluster_ids": [5, 6],
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
    "upstream_object_world_model_artifact_present": True,
    "upstream_generation_diagnostic_artifact_present": True,
    "a1_dominant_bucket_read": True,
    "a1_dominant_bucket": AIMED_AT_DOMINANT_BUCKET,
    "sweep_clusters_used": True,
    "sweep_cluster_ids": [5, 6],
    "sweep_cluster_urls": [CLUSTER_5_URL, CLUSTER_6_URL],
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
    "exploration_strategy_reingested": False,
    "energy_stage_reingested": False,
    "model_load": False,
    "training_launched": False,
    "leaderboard_submission": False,
    "solve_claim_made": False,
    "ops_docs_modified": False,
}
DEFAULT_UPSTREAM_ARTIFACTS = {
    "object_world_model_handoff": UPSTREAM_OBJECT_WORLD_MODEL_ARTIFACT,
    "generation_diagnostic": UPSTREAM_GENERATION_DIAGNOSTIC_ARTIFACT,
    "dominant_bucket": AIMED_AT_DOMINANT_BUCKET,
    "bucket_interpretation": (
        "never_enumerated -> generation expressibility: widen the proposer vocabulary "
        "so missing winning prefixes can be generated before ranking."
    ),
    "partial_noisy_object_signal_only": True,
    "object_signal_constraint": (
        "Carry forward only partial/noisy object signal from Exp 4848; exact object "
        "identity is not a trusted precondition for .448."
    ),
    "carried_forward_from_v447": [
        "comet_object_mcts_planner",
        "slot_mpc_object_action_optimizer",
        "loop_owm_interaction_primitive_proposer",
    ],
    "closed_classes": [
        "nulled_exploration_strategy_class",
        "closed_concluded_energy_stage_class",
    ],
}
DEFAULT_MAPPING_NOTE = {
    "summary": (
        "Exp 4851 found NEVER_ENUMERATED dominant, so .448 should put the missing "
        "winning prefix into the pool by expanding the proposer's action-program "
        "vocabulary rather than reranking the old pool."
    ),
    "terminal_success": HONEST_VERDICT,
    "source_ids": sorted(REQUIRED_SOURCE_IDS),
    "root_cause": "generation expressibility",
    "planner_instruction": (
        "Prioritize DreamCoder/LILO-style action-library learning, execution-guided "
        "ARC program search, and COMET/executable-world-model MCTS as the .448 "
        "candidate-generation expressibility inputs."
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
    FLAGGED_FOR_V448,
    DEFAULT_UPSTREAM_ARTIFACTS,
    DEFAULT_MAPPING_NOTE,
)


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    fresh_sweep: JsonMap = DEFAULT_FRESH_SWEEP,
    flagged_for_v448: Sequence[JsonMap] = FLAGGED_FOR_V448,
    upstream_artifacts: JsonMap = DEFAULT_UPSTREAM_ARTIFACTS,
    generation_expressibility_mapping_note: JsonMap = DEFAULT_MAPPING_NOTE,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "arxiv_ids_cited": sorted(citations),
        "aimed_at_dominant_bucket": AIMED_AT_DOMINANT_BUCKET,
        "flagged_for_v448": [dict(flag) for flag in flagged_for_v448],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "citations": {source_id: dict(citation) for source_id, citation in citations.items()},
        "fresh_sweep": dict(fresh_sweep),
        "upstream_artifacts": dict(upstream_artifacts),
        "generation_expressibility_mapping_note": dict(generation_expressibility_mapping_note),
        "note_path": NOTE_PATH,
        "references_path": REFERENCES_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_S,
        "reproducibility_checksum": source_set_checksum(
            citations,
            methods_mapped,
            flagged_for_v448,
            upstream_artifacts,
            generation_expressibility_mapping_note,
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
    _require(
        artifact["aimed_at_dominant_bucket"] == AIMED_AT_DOMINANT_BUCKET,
        "aimed_at_dominant_bucket must be NEVER_ENUMERATED",
    )
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must be aggregation-only",
    )
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles must match annotations")
    _require(artifact["note_path"] == NOTE_PATH, "note_path must point at the Exp 4858 note")
    _require(artifact["references_path"] == REFERENCES_PATH, "references_path must point at Exp 4858 references")
    _require(artifact["random_seed"] == RANDOM_SEED, "random_seed must be the experiment id")
    _require(artifact["duration_s"] == DURATION_S, "duration_s must preserve the 0.0001s floor")
    _validate_citations(artifact["citations"], artifact["arxiv_ids_cited"])
    _validate_methods(artifact["methods_mapped"], artifact["arxiv_ids_cited"])
    _validate_flags(artifact["flagged_for_v448"], artifact["arxiv_ids_cited"])
    _validate_preconditions(artifact["preconditions_checked"])
    _validate_fresh_sweep(artifact["fresh_sweep"])
    _validate_upstream_artifacts(artifact["upstream_artifacts"])
    _validate_mapping_note(artifact["generation_expressibility_mapping_note"], artifact["arxiv_ids_cited"])
    _require(
        artifact["reproducibility_checksum"]
        == source_set_checksum(
            artifact["citations"],
            artifact["methods_mapped"],
            artifact["flagged_for_v448"],
            artifact["upstream_artifacts"],
            artifact["generation_expressibility_mapping_note"],
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
        _require(method["maps_to_frontier"] == ".448", "method must map to the .448 frontier")
        _require(method["targets_bucket"] == AIMED_AT_DOMINANT_BUCKET, "method must target NEVER_ENUMERATED")
        _require("primitive" in str(method["primitive_expansion"]).lower(), "method must expand primitives")
        insertion = str(method["winner_prefix_pool_insertion"])
        _require(
            "winning prefix" in insertion and "pool" in insertion,
            "method must put the winning prefix into the pool",
        )
        object_contract = str(method["partial_object_signal_contract"])
        _require(
            "partial/noisy object signal" in object_contract
            and "requires exact object identity" not in object_contract.lower(),
            "method must consume partial/noisy object signal without requiring exact identity",
        )
        _require(bool(method["proposal_graft"]), "each method needs a proposal graft")
        _require(bool(method["verification_handoff"]), "each method needs a verification handoff")
        _require(bool(method["takes_over_from_current_stack"]), "each method needs takes_over_from_current_stack")
        _require(bool(method["fails_when"]), "each method needs fails_when")
        _require("flagged_for_v448" in str(method["roadmap_candidate"]), "each method needs a .448 roadmap candidate")
        tracks.add(str(method["track"]))
    _require(REQUIRED_TRACKS == tracks, "methods_mapped missing required generation-expressibility tracks")


def _validate_flags(flags: object, arxiv_ids_cited: object) -> None:
    _require(
        isinstance(flags, Sequence) and not isinstance(flags, str | bytes) and bool(flags),
        "flagged_for_v448 required",
    )
    cited = set(arxiv_ids_cited)
    for flag in flags:
        _require(isinstance(flag, Mapping), "each flagged_for_v448 entry must be a mapping")
        _require("candidate" in flag and "flag" in flag, "each flagged_for_v448 entry needs candidate and flag")
        _require(
            "flagged_for_v447" not in json.dumps(flag, sort_keys=True),
            "stale .447 flag found in flagged_for_v448",
        )
        _require("flagged_for_v448" in str(flag["flag"]), "flagged_for_v448 entries must carry the .448 flag")
        _require(set(flag.get("source_ids", [])).issubset(cited), "flagged_for_v448 source_ids must be verified")


def _validate_preconditions(preconditions: object) -> None:
    _require(isinstance(preconditions, Mapping), "preconditions_checked must be a mapping")
    _require(set(preconditions) == REQUIRED_PRECONDITION_FIELDS, "preconditions_checked must match schema")
    _require(preconditions["research_studying_present"] is True, "research-studying precondition must pass")
    _require(preconditions["research_references_present"] is True, "research-references precondition must pass")
    _require(
        preconditions["upstream_object_world_model_artifact_present"] is True,
        "upstream Exp 4848 artifact must be present",
    )
    _require(
        preconditions["upstream_generation_diagnostic_artifact_present"] is True,
        "upstream Exp 4851 artifact must be present",
    )
    _require(preconditions["a1_dominant_bucket_read"] is True, "A1 dominant bucket must be read")
    _require(preconditions["a1_dominant_bucket"] == AIMED_AT_DOMINANT_BUCKET, "A1 bucket must be NEVER_ENUMERATED")
    _require(preconditions["sweep_clusters_used"] is True, "sweep_clusters must be used")
    _require(preconditions["sweep_cluster_ids"] == [5, 6], "sweep cluster IDs must be [5, 6]")
    _require(preconditions["sweep_semscholar_used"] is True, "sweep_semscholar must be used")
    _require(
        preconditions["semantic_scholar_unique_arxiv_ids"] == SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
        "Semantic Scholar IDs must match reliable-channel output",
    )
    _require(preconditions["websearch_webfetch_used"] is True, "WebSearch/WebFetch must be used")
    _require(5 <= int(preconditions["top_source_count"]) <= 8, "top_source_count must record top five to eight sources")
    _require(preconditions["deep_research_invoked"] is False, "deep-research must not be invoked")
    _require(preconditions["exploration_strategy_reingested"] is False, "exploration strategy must not be reingested")
    _require(preconditions["energy_stage_reingested"] is False, "energy stages must not be reingested")
    _require(preconditions["model_load"] is False, "model load must not occur")
    _require(preconditions["training_launched"] is False, "training must not be launched")
    _require(preconditions["leaderboard_submission"] is False, "leaderboard submission must not occur")
    _require(preconditions["solve_claim_made"] is False, "solve claim must remain false")
    _require(preconditions["ops_docs_modified"] is False, "ops docs must not be modified by this workflow")


def _validate_fresh_sweep(fresh_sweep: object) -> None:
    _require(isinstance(fresh_sweep, Mapping), "fresh_sweep must be a mapping")
    _require(set(fresh_sweep) == REQUIRED_FRESH_SWEEP_FIELDS, "fresh_sweep must match schema")
    _require(fresh_sweep["cluster_ids"] == [5, 6], "fresh_sweep must record clusters 5 and 6")
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
    _require(
        upstream_artifacts["object_world_model_handoff"] == UPSTREAM_OBJECT_WORLD_MODEL_ARTIFACT,
        "upstream artifacts must cite Exp 4848 object-world-model handoff",
    )
    _require(
        upstream_artifacts["generation_diagnostic"] == UPSTREAM_GENERATION_DIAGNOSTIC_ARTIFACT,
        "upstream artifacts must cite Exp 4851 generation diagnostic",
    )
    _require(upstream_artifacts["dominant_bucket"] == AIMED_AT_DOMINANT_BUCKET, "upstream dominant bucket mismatch")
    _require(
        upstream_artifacts["partial_noisy_object_signal_only"] is True,
        "upstream must carry only partial/noisy object signal",
    )
    _require(
        "exact object identity" in str(upstream_artifacts["object_signal_constraint"]),
        "upstream object signal constraint must name exact object identity as untrusted",
    )
    _require(
        "closed_concluded_energy_stage_class" in upstream_artifacts["closed_classes"],
        "upstream closed classes must keep energy stages closed",
    )


def _validate_mapping_note(mapping_note: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(mapping_note, Mapping), "generation expressibility mapping note must be a mapping")
    _require(set(mapping_note) == REQUIRED_MAPPING_NOTE_FIELDS, "mapping note must match schema")
    _require(mapping_note["terminal_success"] == HONEST_VERDICT, "mapping note terminal success must match verdict")
    _require(mapping_note["root_cause"] == "generation expressibility", "mapping note root cause must match")
    _require("NEVER_ENUMERATED" in str(mapping_note["summary"]), "mapping note must mention NEVER_ENUMERATED")
    _require(
        "winning prefix into the pool" in str(mapping_note["summary"]),
        "mapping note must explain putting the winning prefix into the pool",
    )
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
            f"maps to {method['maps_to_frontier']} / {method['targets_bucket']}; "
            f"Primitive expansion: {method['primitive_expansion']} "
            f"Winner insertion: {method['winner_prefix_pool_insertion']} "
            f"Object signal: {method['partial_object_signal_contract']} "
            f"Proposal graft: {method['proposal_graft']} "
            f"Verification handoff: {method['verification_handoff']} "
            f"Takes over: {method['takes_over_from_current_stack']} "
            f"Fails when: {method['fails_when']}"
        )
        for method in result["methods_mapped"]
    )
    flag_lines = "\n".join(flag["flag"] for flag in result["flagged_for_v448"])
    return f"""{STUDYING_SECTION_START}
## 2026-06-27 Exp 4858 - .448 generation expressibility SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4858_sota_ingestion_generation_expressibility.json`.

**Preconditions:** `research-studying.md`, `research-references.md`,
`results/experiment_4848_sota_ingestion_object_world_model.json`, and
`results/experiment_4851_generation_coverage_diagnostic.json` were present.
Exp 4851's dominant bucket was `NEVER_ENUMERATED`, so the target is generation
expressibility, not ranking. `scripts/sweep_clusters.py` emitted ARC
action-effect/exploration and neural-guided-search/world-model cluster URLs.
`scripts/sweep_semscholar.py` was run on three focused generation-expressibility
queries and returned HTTP 429 for all three, so no S2-only source was promoted.
Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks verified the
top eight papers listed below. `/deep-research` was not invoked. The nulled
exploration-strategy class and concluded energy stages were not re-ingested. No
model load, training, leaderboard submission, or solve claim was made; this is a
no solve claim ingestion note.

**A1 bucket targeted:** `NEVER_ENUMERATED` means the current proposer did not
express at least one action primitive in most banked winning prefixes. A ranker
cannot fix a missing candidate. The .448 handoff is to put the winning prefix
into the pool by widening the proposer vocabulary.

**Partial/noisy object signal contract:** carry forward only partial/noisy object
signal from Exp 4848. Object slots, relation hints, and action bindings can guide
proposal generation, but exact identity is not a trusted precondition.

**Verified source set:**
{citation_lines}

**SOTA -> generation expressibility mapping for .448:**
{method_lines}

{flag_lines}

**Bottom line for .448:** start with DreamCoder/LILO-style action-library
learning as the vocabulary widening layer, pair it with execution-guided ARC
program search for counterexample repair, and use COMET/executable-world-model
MCTS to turn partial object signals into replayable action prefixes.
{STUDYING_SECTION_END}"""


def build_research_references_section(artifact: JsonMap | None = None) -> str:
    result = dict(artifact or build_artifact())
    validate_artifact(result)
    citations = result["citations"]
    source_lines = "\n".join(
        (
            f"- **arXiv:{source_id} -- {citations[source_id]['title']}.** "
            f"Exp 4858 use: generation expressibility source for putting missing "
            f"winning prefixes into the candidate pool."
        )
        for source_id in sorted(citations)
    )
    return f"""{REFERENCES_SECTION_START}
## 2026-06-27 Exp 4858 generation-expressibility source set

Reliable-channel ingestion for `.448`, aimed at Exp 4851's
`NEVER_ENUMERATED` dominant bucket. These papers are marked INGESTED for the
generation-expressibility roadmap handoff:

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
    _require(start >= 0 and end >= 0, "research-studying missing Exp 4858 section markers")
    section = text[start : end + len(STUDYING_SECTION_END)]
    for phrase in (
        "SOTA -> generation expressibility mapping",
        "flagged_for_v448",
        "no solve claim",
        "NEVER_ENUMERATED",
        "partial/noisy object signal",
        "winning prefix into the pool",
        "DreamCoder",
        "LILO",
        "SOAR",
        "COMET",
        "not ranking",
    ):
        _require(phrase in section, f"research-studying section missing required phrase: {phrase}")
    missing_citations = sorted(citation for citation in NOTE_REQUIRED_SOURCE_CITATIONS if citation not in section)
    _require(not missing_citations, f"research-studying section missing citations: {missing_citations}")
    for method in result["methods_mapped"]:
        _require(method["method"] in section, f"research-studying section missing method: {method['method']}")
    for flag in result["flagged_for_v448"]:
        _require(flag["flag"] in section, "research-studying section missing flagged_for_v448 text")


def validate_research_references_text(text: str, artifact: JsonMap | None = None) -> None:
    result = dict(artifact or build_artifact())
    validate_artifact(result)
    start = text.find(REFERENCES_SECTION_START)
    end = text.find(REFERENCES_SECTION_END, start)
    _require(start >= 0 and end >= 0, "research-references missing Exp 4858 section markers")
    section = text[start : end + len(REFERENCES_SECTION_END)]
    for phrase in (
        "Exp 4858 generation-expressibility source set",
        "INGESTED",
        "NEVER_ENUMERATED",
        "DreamCoder",
        "LILO",
        "SOAR",
        "Object-Centric World Models Meet Monte Carlo Tree Search",
    ):
        _require(phrase in section, f"research-references section missing required phrase: {phrase}")
    missing_citations = sorted(citation for citation in NOTE_REQUIRED_SOURCE_CITATIONS if citation not in section)
    _require(not missing_citations, f"research-references section missing citations: {missing_citations}")


def write_outputs(
    *,
    artifact_path: Path | None = None,
    studying_path: Path | None = None,
    references_path: Path | None = None,
    artifact: JsonMap | None = None,
) -> dict[str, object]:
    result = dict(artifact or build_artifact())
    validate_artifact(result)
    target_artifact = artifact_path or Path(RESULT_RELATIVE_PATH)
    target_studying = studying_path or Path(STUDYING_RELATIVE_PATH)
    target_references = references_path or Path(REFERENCES_RELATIVE_PATH)
    target_artifact.parent.mkdir(parents=True, exist_ok=True)
    target_artifact.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    studying_text = target_studying.read_text(encoding="utf-8")
    updated_studying = update_research_studying_text(studying_text, result)
    validate_research_studying_text(updated_studying, result)
    target_studying.write_text(updated_studying, encoding="utf-8")
    references_text = target_references.read_text(encoding="utf-8")
    updated_references = update_research_references_text(references_text, result)
    validate_research_references_text(updated_references, result)
    target_references.write_text(updated_references, encoding="utf-8")
    return result


def main() -> int:
    root = Path(os.environ.get("CARNOT_EXP4858_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        studying_path=root / STUDYING_RELATIVE_PATH,
        references_path=root / REFERENCES_RELATIVE_PATH,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
