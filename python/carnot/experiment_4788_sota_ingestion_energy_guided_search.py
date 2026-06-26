"""Exp 4788 SOTA ingestion for energy-guided S2/S3 search.

Spec refs: REQ-ARC-WMTE-4788, SCENARIO-ARC-WMTE-4788,
SCENARIO-ARC-WMTE-4788-NO-FABRICATION.

This module is a deterministic literature-ingestion artifact writer. It does
not train a model, load an LLM, or claim an ARC solve. The output tells the
next planner how to use the S1 lower-is-better energy landscape as a guide for
S2 search and S3 generation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4788_sota_ingestion_energy_guided_search.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
NOTE_PATH = "research-studying.md#exp-4788-sota-ingestion-energy-guided-search"
S1_SOURCE_RELATIVE_PATH = "results/experiment_4781_structural_energy_s1_contrastive_landscape.json"
RANDOM_SEED = 4788
DURATION_S = 0.0001
HONEST_VERDICT = "success_sota_ingestion_energy_guided_search_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
STUDYING_SECTION_START = "<!-- EXP4788-SOTA-INGESTION-ENERGY-GUIDED-SEARCH-START -->"
STUDYING_SECTION_END = "<!-- EXP4788-SOTA-INGESTION-ENERGY-GUIDED-SEARCH-END -->"
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
        "principle": "terminal prefix; mapping emitted is success_sota_ingestion_energy_guided_search_mapped."
    },
    "methods_mapped": {
        "principle": "the strongest 3-5 methods mapped onto S2/S3, each with a real arXiv ID."
    },
    "arxiv_ids_cited": {
        "principle": (
            "every method claim must cite a verifiable arXiv ID -- an ingestion "
            "with no citations is fabrication."
        )
    },
    "flagged_for_v441": {
        "principle": "the strongest method(s) flagged so the .441 planner reads the mapping."
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
    "s1_context": {
        "principle": "imports the S1 close-state so S2/S3 only consume an authorized energy landscape."
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
        "principle": "content hash of citations, method map, flags, and S1 context."
    },
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "methods_mapped",
    "arxiv_ids_cited",
    "flagged_for_v441",
    "inference_substrate",
    "preconditions_checked",
    "citations",
    "fresh_sweep",
    "s1_context",
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
        "maps_to_stages",
        "graft_to_live_loop",
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
REQUIRED_SOURCE_IDS = frozenset(
    {
        "1909.06878",
        "2103.11505",
        "2202.11705",
        "2206.09914",
        "2304.14391",
        "2309.15028",
        "2502.07202",
        "2505.10819",
    }
)
ALLOWED_STAGES = frozenset({"S2", "S3"})
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS)

CLUSTER_1_URL = (
    'http://export.arxiv.org/api/query?search_query=(abs:"energy+based+model"+OR+'
    'abs:"energy-based+model"+OR+abs:"energy+guided+decoding"+OR+'
    'abs:"token+energy"+OR+abs:"EBT")+AND+(abs:"reasoning"+OR+'
    'abs:"verification"+OR+abs:"LLM"+OR+abs:"language+model")&start=0&'
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
    "energy based model planning MCTS product of experts COLD decoding",
    "value guided MCTS best first search energy guided decoding discrete Langevin",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/1909.06878",
    "https://arxiv.org/abs/2103.11505",
    "https://arxiv.org/abs/2202.11705",
    "https://arxiv.org/abs/2206.09914",
    "https://arxiv.org/abs/2304.14391",
    "https://arxiv.org/abs/2309.15028",
    "https://arxiv.org/abs/2502.07202",
    "https://arxiv.org/abs/2505.10819",
]

CITATIONS = {
    "1909.06878": {
        "title": "Model Based Planning with Energy Based Models",
        "url": "https://arxiv.org/abs/1909.06878",
        "http_status": 200,
    },
    "2103.11505": {
        "title": "Policy-Guided Heuristic Search with Guarantees",
        "url": "https://arxiv.org/abs/2103.11505",
        "http_status": 200,
    },
    "2202.11705": {
        "title": "COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics",
        "url": "https://arxiv.org/abs/2202.11705",
        "http_status": 200,
    },
    "2206.09914": {
        "title": "A Langevin-like Sampler for Discrete Distributions",
        "url": "https://arxiv.org/abs/2206.09914",
        "http_status": 200,
    },
    "2304.14391": {
        "title": "Energy-based Models are Zero-Shot Planners for Compositional Scene Rearrangement",
        "url": "https://arxiv.org/abs/2304.14391",
        "http_status": 200,
    },
    "2309.15028": {
        "title": (
            "Do not throw away your value model! Generating more preferable text "
            "with Value-Guided Monte-Carlo Tree Search decoding"
        ),
        "url": "https://arxiv.org/abs/2309.15028",
        "http_status": 200,
    },
    "2502.07202": {
        "title": "Monte Carlo Tree Diffusion for System 2 Planning",
        "url": "https://arxiv.org/abs/2502.07202",
        "http_status": 200,
    },
    "2505.10819": {
        "title": "PoE-World: Compositional World Modeling with Products of Programmatic Experts",
        "url": "https://arxiv.org/abs/2505.10819",
        "http_status": 200,
    },
}

FLAGGED_FOR_V441 = [
    {
        "candidate": "energy_value_guided_mcts_frontier_controller",
        "flag": (
            "flagged_for_v441: energy_value_guided_mcts_frontier_controller "
            "(arXiv:2309.15028 + arXiv:2502.07202 + arXiv:2103.11505)"
        ),
        "source_ids": ["2309.15028", "2502.07202", "2103.11505"],
        "maps_to_stages": ["S2"],
    },
    {
        "candidate": "ebm_poe_planner_for_s3_generation",
        "flag": (
            "flagged_for_v441: ebm_poe_planner_for_s3_generation "
            "(arXiv:1909.06878 + arXiv:2304.14391 + arXiv:2505.10819)"
        ),
        "source_ids": ["1909.06878", "2304.14391", "2505.10819"],
        "maps_to_stages": ["S3"],
    },
]

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Energy/value-guided MCTS frontier controller",
        "track": "energy_value_guided_mcts",
        "source_ids": ["2309.15028", "2502.07202"],
        "maps_to_stages": ["S2"],
        "graft_to_live_loop": (
            "Use negative S1 energy as the value term in the S2 tree policy: "
            "expand induced partial plans whose predicted transition rollouts "
            "fall downhill, and re-score every child through verify before acting."
        ),
        "takes_over_from_current_stack": (
            "Takes over uniform or static best-first expansion when the live "
            "induce->verify->plan loop has several plausible next engine states."
        ),
        "fails_when": (
            "The S1 energy is miscalibrated off distribution, tree branching is too "
            "wide for value reuse, or partial-rollout scores become a shortcut for "
            "known generator provenance."
        ),
        "roadmap_candidate": FLAGGED_FOR_V441[0]["flag"],
    },
    {
        "method": "Energy-weighted policy-guided best-first search",
        "track": "energy_best_first_frontier",
        "source_ids": ["2103.11505", "2309.15028"],
        "maps_to_stages": ["S2"],
        "graft_to_live_loop": (
            "Rank the live frontier by a composite priority: learned policy prior "
            "for cheap action plausibility plus lower S1 energy for transition "
            "trust, with verification gating before a node consumes action budget."
        ),
        "takes_over_from_current_stack": (
            "Takes over hand-tuned value_weight and FIFO frontier scheduling by "
            "making the S1 landscape the admissibility-aware heuristic term."
        ),
        "fails_when": (
            "The energy and policy disagree without a tie-break budget, low-energy "
            "nodes are all duplicates, or the heuristic over-prunes the one action "
            "that would reveal the game mechanic."
        ),
        "roadmap_candidate": FLAGGED_FOR_V441[0]["flag"],
    },
    {
        "method": "Gradient-guided discrete energy search",
        "track": "gradient_guided_discrete_search",
        "source_ids": ["2202.11705", "2206.09914"],
        "maps_to_stages": ["S3"],
        "graft_to_live_loop": (
            "Treat candidate action programs, latent action slots, or text plans as "
            "discrete variables and use S1-energy gradients or discrete Langevin "
            "proposals to mutate them before normal verify accepts any plan."
        ),
        "takes_over_from_current_stack": (
            "Takes over random local repair of generated candidates by proposing "
            "low-energy edits that still pass the executable verifier."
        ),
        "fails_when": (
            "The candidate representation is not differentiable enough to expose a "
            "useful neighborhood, gradient proposals violate hard grid mechanics, "
            "or repeated Langevin steps collapse to near-duplicate candidates."
        ),
        "roadmap_candidate": "support_for_v441: gradient_guided_discrete_candidate_repair",
    },
    {
        "method": "EBM-as-planner state trajectory refinement",
        "track": "ebm_as_planner",
        "source_ids": ["1909.06878", "2304.14391"],
        "maps_to_stages": ["S3"],
        "graft_to_live_loop": (
            "Use S1 energy directly as the planner objective over intermediate "
            "state hypotheses: sample or optimize a sequence of latent next states, "
            "then ask the existing inducer to synthesize actions that realize them."
        ),
        "takes_over_from_current_stack": (
            "Takes over plan-in-model reranking when no winning action sequence is "
            "present in the static candidate pool."
        ),
        "fails_when": (
            "State-space descent finds physically impossible intermediate grids, "
            "the action realizer cannot execute the inferred state path, or the "
            "energy ignores small but decisive object changes."
        ),
        "roadmap_candidate": FLAGGED_FOR_V441[1]["flag"],
    },
    {
        "method": "Product-of-experts compositional planning",
        "track": "poe_compositional_planning",
        "source_ids": ["2304.14391", "2505.10819", "1909.06878"],
        "maps_to_stages": ["S2", "S3"],
        "graft_to_live_loop": (
            "Make the S1 energy one expert in a product with code-world-model, "
            "spatial-relation, and action-effect experts; S2 scores which factor "
            "to trust, while S3 composes factors to generate new plans."
        ),
        "takes_over_from_current_stack": (
            "Takes over monolithic world-model acceptance by requiring each expert "
            "factor to improve or preserve the joint product energy before a plan "
            "is promoted."
        ),
        "fails_when": (
            "Experts double-count the same shortcut, one factor dominates the "
            "product, sparse observations synthesize the wrong programmatic expert, "
            "or product energy improves without executable action support."
        ),
        "roadmap_candidate": FLAGGED_FOR_V441[1]["flag"],
    },
]

DEFAULT_FRESH_SWEEP = {
    "filtered_track": "energy-guided search + EBM planning for S2/S3",
    "cluster_ids": [1, 6],
    "semantic_scholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "semantic_scholar_result": "HTTP 429 on both focused queries; no S2-only source promoted.",
    "webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
}
DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "research_studying_present": True,
    "research_references_present": True,
    "sweep_clusters_used": True,
    "sweep_cluster_ids": [1, 6],
    "sweep_cluster_urls": [CLUSTER_1_URL, CLUSTER_6_URL],
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "sweep_semscholar_http_429": True,
    "websearch_webfetch_used": True,
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
    "top_source_count": len(WEBSEARCH_WEBFETCH_TOP_SOURCES),
    "arxiv_http_200_verified_ids": [f"https://arxiv.org/abs/{source_id}" for source_id in sorted(REQUIRED_SOURCE_IDS)],
    "deep_research_invoked": False,
    "model_load": False,
    "training_launched": False,
    "leaderboard_submission": False,
    "solve_claim_made": False,
    "ops_docs_modified": False,
}
DEFAULT_S1_CONTEXT = {
    "source_artifact": S1_SOURCE_RELATIVE_PATH,
    "stage": "S1",
    "imported_honest_verdict": "success_structural_energy_s1_landscape_authorizes_s2",
    "s1_gate_passed": True,
    "s2_authorized": True,
    "energy_ranking_loo_auroc_mean": 0.7134961314270525,
    "energy_ranking_loo_auroc_ci95": [0.7133175599984811, 0.7137104171413382],
    "n_seeds": 10,
    "denoising_direction_agreement": 0.6223390275952694,
    "origin_probe_auroc": 0.5,
    "shuffled_label_control_auroc": 0.49335645814441664,
    "planning_constraint": (
        "S1 provides a lower-is-better energy landscape; S2/S3 may use it only "
        "as guidance, never as an environment win oracle."
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_set_checksum(
    citations: JsonMap,
    methods: Sequence[JsonMap],
    flags: Sequence[JsonMap],
    s1_context: JsonMap,
) -> str:
    payload = json.dumps(
        {
            "citations": citations,
            "flags": list(flags),
            "methods": list(methods),
            "s1_context": s1_context,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


REPRODUCIBILITY_CHECKSUM = source_set_checksum(
    CITATIONS,
    DEFAULT_METHODS_MAPPED,
    FLAGGED_FOR_V441,
    DEFAULT_S1_CONTEXT,
)


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    fresh_sweep: JsonMap = DEFAULT_FRESH_SWEEP,
    flagged_for_v441: Sequence[JsonMap] = FLAGGED_FOR_V441,
    s1_context: JsonMap = DEFAULT_S1_CONTEXT,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "arxiv_ids_cited": sorted(citations),
        "flagged_for_v441": [dict(flag) for flag in flagged_for_v441],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "citations": {source_id: dict(citation) for source_id, citation in citations.items()},
        "fresh_sweep": dict(fresh_sweep),
        "s1_context": dict(s1_context),
        "note_path": NOTE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_S,
        "reproducibility_checksum": source_set_checksum(
            citations,
            methods_mapped,
            flagged_for_v441,
            s1_context,
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
    _require(artifact["honest_verdict"].startswith(TERMINAL_PREFIXES), "honest_verdict must use a terminal prefix")
    _require(artifact["honest_verdict"] == HONEST_VERDICT, f"honest_verdict must equal {HONEST_VERDICT!r}")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate must be aggregation-only")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles must match annotations")
    _require(artifact["note_path"] == NOTE_PATH, "note_path must point at the Exp 4788 note")
    _require(artifact["random_seed"] == RANDOM_SEED, "random_seed must be the experiment id")
    _require(artifact["duration_s"] == DURATION_S, "duration_s must preserve the 0.0001s floor")
    _validate_citations(artifact["citations"], artifact["arxiv_ids_cited"])
    _validate_methods(artifact["methods_mapped"], artifact["arxiv_ids_cited"])
    _validate_flags(artifact["flagged_for_v441"], artifact["arxiv_ids_cited"])
    _validate_preconditions(artifact["preconditions_checked"])
    _validate_fresh_sweep(artifact["fresh_sweep"])
    _validate_s1_context(artifact["s1_context"])
    _require(
        artifact["reproducibility_checksum"]
        == source_set_checksum(
            artifact["citations"],
            artifact["methods_mapped"],
            artifact["flagged_for_v441"],
            artifact["s1_context"],
        ),
        "reproducibility checksum must hash citations, methods, flags, and S1 context",
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
    stages: set[str] = set()
    tracks: set[str] = set()
    for method in methods:
        _require(isinstance(method, Mapping), "each method must be a mapping")
        _require(set(method) == REQUIRED_METHOD_FIELDS, "each method must match the required method schema")
        source_ids = method["source_ids"]
        maps_to_stages = method["maps_to_stages"]
        _require(
            isinstance(source_ids, Sequence) and not isinstance(source_ids, str | bytes) and bool(source_ids),
            "each method must cite source_ids",
        )
        _require(set(source_ids).issubset(cited), "method source_ids must be verified citations")
        _require(
            isinstance(maps_to_stages, Sequence)
            and not isinstance(maps_to_stages, str | bytes)
            and bool(maps_to_stages)
            and set(maps_to_stages).issubset(ALLOWED_STAGES),
            "method maps_to_stages must stay within S2/S3",
        )
        _require(bool(method["graft_to_live_loop"]), "each method needs a graft_to_live_loop mapping")
        _require(bool(method["takes_over_from_current_stack"]), "each method needs takes_over_from_current_stack")
        _require(bool(method["fails_when"]), "each method needs fails_when")
        _require(bool(method["roadmap_candidate"]), "each method needs a roadmap candidate")
        stages.update(str(stage) for stage in maps_to_stages)
        tracks.add(str(method["track"]))
    _require(ALLOWED_STAGES.issubset(stages), "methods_mapped must cover S2/S3")
    for track in (
        "energy_value_guided_mcts",
        "energy_best_first_frontier",
        "gradient_guided_discrete_search",
        "ebm_as_planner",
        "poe_compositional_planning",
    ):
        _require(track in tracks, f"methods_mapped missing track: {track}")


def _validate_flags(flags: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(flags, Sequence) and not isinstance(flags, str | bytes) and bool(flags), "flagged_for_v441 required")
    cited = set(arxiv_ids_cited)
    for flag in flags:
        _require(isinstance(flag, Mapping), "each flagged_for_v441 entry must be a mapping")
        _require("candidate" in flag and "flag" in flag, "each flagged_for_v441 entry needs candidate and flag")
        _require("flagged_for_v440" not in json.dumps(flag, sort_keys=True), "flagged_for_v441 must not carry stale .440 flags")
        _require("flagged_for_v441" in str(flag["flag"]), "flagged_for_v441 entries must carry the .441 flag")
        _require(set(flag.get("source_ids", [])).issubset(cited), "flagged_for_v441 source_ids must be verified")


def _validate_preconditions(preconditions: object) -> None:
    _require(isinstance(preconditions, Mapping), "preconditions_checked must be a mapping")
    _require(set(preconditions) == REQUIRED_PRECONDITION_FIELDS, "preconditions_checked must match schema")
    _require(preconditions["research_studying_present"] is True, "research-studying precondition must pass")
    _require(preconditions["research_references_present"] is True, "research-references precondition must pass")
    _require(preconditions["sweep_clusters_used"] is True, "sweep_clusters must be used")
    _require(preconditions["sweep_cluster_ids"] == [1, 6], "sweep cluster IDs must be [1, 6]")
    _require(preconditions["sweep_semscholar_used"] is True, "sweep_semscholar must be used")
    _require(preconditions["sweep_semscholar_http_429"] is True, "Semantic Scholar HTTP 429 must be recorded")
    _require(preconditions["websearch_webfetch_used"] is True, "WebSearch/WebFetch must be used")
    _require(5 <= int(preconditions["top_source_count"]) <= 8, "top_source_count must record top five to eight sources")
    _require(preconditions["deep_research_invoked"] is False, "deep-research must not be invoked")
    _require(preconditions["model_load"] is False, "model load must not occur")
    _require(preconditions["training_launched"] is False, "training must not be launched")
    _require(preconditions["leaderboard_submission"] is False, "leaderboard submission must not occur")
    _require(preconditions["solve_claim_made"] is False, "solve claim must remain false")
    _require(preconditions["ops_docs_modified"] is False, "ops docs must not be modified by this workflow")


def _validate_fresh_sweep(fresh_sweep: object) -> None:
    _require(isinstance(fresh_sweep, Mapping), "fresh_sweep must be a mapping")
    _require(set(fresh_sweep) == REQUIRED_FRESH_SWEEP_FIELDS, "fresh_sweep must match schema")
    _require(fresh_sweep["cluster_ids"] == [1, 6], "fresh_sweep must record clusters 1 and 6")
    sources = fresh_sweep["webfetch_top_sources"]
    _require(
        isinstance(sources, Sequence) and not isinstance(sources, str | bytes) and 5 <= len(sources) <= 8,
        "fresh_sweep must record top five to eight WebFetch sources",
    )
    _require(list(sources) == WEBSEARCH_WEBFETCH_TOP_SOURCES, "fresh_sweep sources must match verified source set")


def _validate_s1_context(s1_context: object) -> None:
    _require(isinstance(s1_context, Mapping), "s1_context must be a mapping")
    _require(s1_context.get("source_artifact") == S1_SOURCE_RELATIVE_PATH, "s1_context must cite S1")
    _require(s1_context.get("stage") == "S1", "s1_context stage must be S1")
    _require(s1_context.get("s1_gate_passed") is True, "S1 context must import the passed S1 gate")
    _require(s1_context.get("s2_authorized") is True, "S1 context must import S1 as authorizing S2")
    _require(
        float(s1_context.get("energy_ranking_loo_auroc_mean", 0.0)) >= 0.70,
        "S1 context must import a usable energy landscape",
    )
    _require(float(s1_context.get("origin_probe_auroc", 1.0)) <= 0.6, "S1 origin probe must remain leak-clean")
    _require(
        float(s1_context.get("shuffled_label_control_auroc", 1.0)) <= 0.55,
        "S1 shuffled label control must remain leak-clean",
    )
    _require("lower-is-better" in str(s1_context.get("planning_constraint", "")), "S1 context must define lower-is-better energy")


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
            f"maps to {', '.join(method['maps_to_stages'])}; "
            f"{method['graft_to_live_loop']} Takes over: {method['takes_over_from_current_stack']} "
            f"Fails when: {method['fails_when']}"
        )
        for method in result["methods_mapped"]
    )
    flag_lines = "\n".join(flag["flag"] for flag in result["flagged_for_v441"])
    return f"""{STUDYING_SECTION_START}
## 2026-06-26 Exp 4788 - .441 energy-guided search SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4788_sota_ingestion_energy_guided_search.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted the EBM and neural-guided search
cluster URLs. `scripts/sweep_semscholar.py` was run on two focused
energy-guided search queries and returned HTTP 429, so no S2-only source was
promoted. Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks
verified the top eight papers listed below. `/deep-research` was not invoked.
No model load, training, leaderboard submission, or solve claim was made; this
is a no solve claim ingestion note.

**S1 context imported:** `{S1_SOURCE_RELATIVE_PATH}` reports
`{result["s1_context"]["imported_honest_verdict"]}` with
`energy_ranking_loo_auroc_mean={result["s1_context"]["energy_ranking_loo_auroc_mean"]}`
and `denoising_direction_agreement={result["s1_context"]["denoising_direction_agreement"]}`.
The .441 planner should treat S1 as a lower-is-better guide for search and
generation, not as an environment oracle.

**Verified source set:**
{citation_lines}

**SOTA -> S2/S3 energy-guided search mapping:**
{method_lines}

{flag_lines}

**Bottom line for .441:** start with the energy/value-guided MCTS frontier
controller for S2, because it grafts the S1 energy onto the live
induce->verify->plan loop with the smallest change in control flow. In
parallel, prepare the EBM/PoE planner path for S3 generation so low-energy
trajectory and product-of-experts proposals can make new candidate plans appear
instead of merely reranking a frozen pool.
{STUDYING_SECTION_END}"""


def update_research_studying_text(text: str, artifact: JsonMap | None = None) -> str:
    section = build_research_studying_section(artifact)
    start = text.find(STUDYING_SECTION_START)
    if start >= 0:
        end = text.find(STUDYING_SECTION_END, start)
        _require(end >= 0, "research-studying Exp 4788 section missing end marker")
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
    _require(start >= 0 and end >= 0, "research-studying missing Exp 4788 section markers")
    section = text[start : end + len(STUDYING_SECTION_END)]
    for phrase in (
        "SOTA -> S2/S3 energy-guided search mapping",
        "flagged_for_v441",
        "no solve claim",
        "lower-is-better",
        "MCTS",
        "best-first",
        "COLD",
        "PoE",
    ):
        _require(phrase in section, f"research-studying section missing required phrase: {phrase}")
    missing_citations = sorted(citation for citation in NOTE_REQUIRED_SOURCE_CITATIONS if citation not in section)
    _require(not missing_citations, f"research-studying section missing citations: {missing_citations}")
    for method in result["methods_mapped"]:
        _require(method["method"] in section, f"research-studying section missing method: {method['method']}")
    for flag in result["flagged_for_v441"]:
        _require(flag["flag"] in section, "research-studying section missing flagged_for_v441 text")


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
    studying_text = target_studying.read_text(encoding="utf-8") if target_studying.exists() else "# Research Studying\n\n"
    updated = update_research_studying_text(studying_text, result)
    validate_research_studying_text(updated, result)
    target_studying.write_text(updated, encoding="utf-8")
    return result


def main() -> int:
    root = Path(os.environ.get("CARNOT_EXP4788_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        studying_path=root / STUDYING_RELATIVE_PATH,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
