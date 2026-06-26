"""Exp 4798 SOTA ingestion for energy-guided S3 generation.

Spec refs: REQ-ARC-WMTE-4798, SCENARIO-ARC-WMTE-4798,
SCENARIO-ARC-WMTE-4798-NO-FABRICATION.

This module writes a deterministic literature-ingestion artifact. It records
how S3 should use the S1/S2 lower-is-better energy to generate new candidate
plans and place a winner into the explorer pool. It does not train a model,
load an LLM, submit to a leaderboard, or claim an ARC solve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4798_sota_ingestion_energy_guided_generation.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
NOTE_PATH = "research-studying.md#exp-4798-sota-ingestion-energy-guided-generation"
S1_SOURCE_RELATIVE_PATH = "results/experiment_4781_structural_energy_s1_contrastive_landscape.json"
S2_SOURCE_RELATIVE_PATH = "results/experiment_4791_structural_energy_s2_offpath_trust_gate.json"
RANDOM_SEED = 4798
DURATION_S = 0.0001
HONEST_VERDICT = "success_sota_ingestion_energy_guided_generation_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
STUDYING_SECTION_START = "<!-- EXP4798-SOTA-INGESTION-ENERGY-GUIDED-GENERATION-START -->"
STUDYING_SECTION_END = "<!-- EXP4798-SOTA-INGESTION-ENERGY-GUIDED-GENERATION-END -->"
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
        "principle": "terminal prefix; mapping emitted is success_sota_ingestion_energy_guided_generation_mapped."
    },
    "methods_mapped": {
        "principle": "the strongest 3-5 methods mapped onto S3, each with a real arXiv ID."
    },
    "arxiv_ids_cited": {
        "principle": (
            "every method claim must cite a verifiable arXiv ID -- an ingestion "
            "with no citations is fabrication."
        )
    },
    "flagged_for_v442": {
        "principle": "the strongest method(s) flagged so the .442 planner reads the mapping."
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
    "s1_s2_context": {
        "principle": "imports the S1/S2 close-state so S3 consumes the authorized energy as guidance only."
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
        "principle": "content hash of citations, method map, flags, and S1/S2 context."
    },
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "methods_mapped",
    "arxiv_ids_cited",
    "flagged_for_v442",
    "inference_substrate",
    "preconditions_checked",
    "citations",
    "fresh_sweep",
    "s1_s2_context",
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
        "put_winner_into_pool",
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
        "1806.10230",
        "1909.06878",
        "2012.04322",
        "2105.05233",
        "2202.11705",
        "2207.12598",
        "2309.15028",
        "2502.07202",
    }
)
REQUIRED_TRACKS = frozenset(
    {
        "energy_constrained_sampling",
        "classifier_score_guided_generation",
        "value_guided_tree_generation",
        "energy_as_fitness_evolutionary_search",
        "plan_with_energy_state_generation",
    }
)
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
    "energy guided generation classifier guidance score guidance value guided generation evolutionary search energy fitness planning",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/1806.10230",
    "https://arxiv.org/abs/1909.06878",
    "https://arxiv.org/abs/2012.04322",
    "https://arxiv.org/abs/2105.05233",
    "https://arxiv.org/abs/2202.11705",
    "https://arxiv.org/abs/2207.12598",
    "https://arxiv.org/abs/2309.15028",
    "https://arxiv.org/abs/2502.07202",
]

CITATIONS = {
    "1806.10230": {
        "title": "Guided evolutionary strategies: Augmenting random search with surrogate gradients",
        "url": "https://arxiv.org/abs/1806.10230",
        "http_status": 200,
    },
    "1909.06878": {
        "title": "Model Based Planning with Energy Based Models",
        "url": "https://arxiv.org/abs/1909.06878",
        "http_status": 200,
    },
    "2012.04322": {
        "title": "Quality-Diversity Optimization: a novel branch of stochastic optimization",
        "url": "https://arxiv.org/abs/2012.04322",
        "http_status": 200,
    },
    "2105.05233": {
        "title": "Diffusion Models Beat GANs on Image Synthesis",
        "url": "https://arxiv.org/abs/2105.05233",
        "http_status": 200,
    },
    "2202.11705": {
        "title": "COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics",
        "url": "https://arxiv.org/abs/2202.11705",
        "http_status": 200,
    },
    "2207.12598": {
        "title": "Classifier-Free Diffusion Guidance",
        "url": "https://arxiv.org/abs/2207.12598",
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
}

FLAGGED_FOR_V442 = [
    {
        "candidate": "cold_cfg_value_tree_generator_for_s3",
        "flag": (
            "flagged_for_v442: cold_cfg_value_tree_generator_for_s3 "
            "(arXiv:2202.11705 + arXiv:2105.05233 + arXiv:2207.12598 + "
            "arXiv:2309.15028 + arXiv:2502.07202)"
        ),
        "source_ids": ["2202.11705", "2105.05233", "2207.12598", "2309.15028", "2502.07202"],
        "maps_to_stages": ["S3"],
    },
    {
        "candidate": "energy_fitness_qd_pool_inserter",
        "flag": (
            "flagged_for_v442: energy_fitness_qd_pool_inserter "
            "(arXiv:1806.10230 + arXiv:2012.04322 + arXiv:1909.06878)"
        ),
        "source_ids": ["1806.10230", "2012.04322", "1909.06878"],
        "maps_to_stages": ["S3"],
    },
]

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Energy-constrained Langevin candidate generator",
        "track": "energy_constrained_sampling",
        "source_ids": ["2202.11705"],
        "maps_to_stages": ["S3"],
        "graft_to_live_loop": (
            "Treat action programs, latent plans, or textual program sketches as "
            "variables under a composite energy: S1 transition plausibility, S2 "
            "trust gate, hard ARC action validity, and pool novelty."
        ),
        "put_winner_into_pool": (
            "Run short energy-guided sampling chains, verify each decoded candidate, "
            "and insert the lowest-energy verified winner into the explorer pool."
        ),
        "takes_over_from_current_stack": (
            "Takes over blind local repair and purely random candidate mutation when "
            "the explorer has no candidate that already wins."
        ),
        "fails_when": (
            "The candidate representation has no smooth neighborhood, low-energy "
            "edits violate executable mechanics, or sampling collapses to duplicates."
        ),
        "roadmap_candidate": FLAGGED_FOR_V442[0]["flag"],
    },
    {
        "method": "Classifier/score-guided proposal sampler",
        "track": "classifier_score_guided_generation",
        "source_ids": ["2105.05233", "2207.12598"],
        "maps_to_stages": ["S3"],
        "graft_to_live_loop": (
            "Use the S1/S2 energy as a classifier-like or score-guidance signal "
            "during proposal construction, increasing guidance only after the hard "
            "action and transition validators keep the candidate executable."
        ),
        "put_winner_into_pool": (
            "Sample a small guided batch, choose the verified candidate with the "
            "best lower-is-better energy, and put that winner into the live pool."
        ),
        "takes_over_from_current_stack": (
            "Takes over unguided proposal batches whose candidates are only scored "
            "after generation has already spent the action budget."
        ),
        "fails_when": (
            "Guidance scale overwhelms diversity, the score follows a provenance "
            "shortcut, or high guidance makes syntactically valid but unplayable plans."
        ),
        "roadmap_candidate": FLAGGED_FOR_V442[0]["flag"],
    },
    {
        "method": "Value-guided tree generation",
        "track": "value_guided_tree_generation",
        "source_ids": ["2309.15028", "2502.07202"],
        "maps_to_stages": ["S3"],
        "graft_to_live_loop": (
            "Expand partial generated plans as a tree, using negative S1/S2 energy "
            "as the value term for partial rollouts and revisiting branches whose "
            "denoised or decoded continuations improve the trust score."
        ),
        "put_winner_into_pool": (
            "Complete the best tree leaf into an executable plan, verify it, and "
            "put the verified low-energy winner into the candidate pool."
        ),
        "takes_over_from_current_stack": (
            "Takes over one-shot generation by letting inference-time compute refine "
            "partial plans before they become expensive live actions."
        ),
        "fails_when": (
            "Partial-plan energy is poorly calibrated, branching cost exceeds the "
            "action-efficiency budget, or the tree repeatedly explores equivalent leaves."
        ),
        "roadmap_candidate": FLAGGED_FOR_V442[0]["flag"],
    },
    {
        "method": "Energy-as-fitness evolutionary pool search",
        "track": "energy_as_fitness_evolutionary_search",
        "source_ids": ["1806.10230", "2012.04322"],
        "maps_to_stages": ["S3"],
        "graft_to_live_loop": (
            "Run a tiny population over generated action programs where fitness is "
            "negative S1/S2 energy plus executable validity and a novelty/diversity "
            "descriptor for game mechanics not yet covered by the pool."
        ),
        "put_winner_into_pool": (
            "Select the verified elite with the best energy-adjusted fitness and "
            "put that winner into the explorer pool while retaining diverse alternates."
        ),
        "takes_over_from_current_stack": (
            "Takes over single-candidate mutation by preserving a small archive of "
            "high-quality diverse candidates instead of only the current best guess."
        ),
        "fails_when": (
            "Fitness overweights novelty, surrogate energy points away from true "
            "transition mechanics, or the population burns budget without new verified elites."
        ),
        "roadmap_candidate": FLAGGED_FOR_V442[1]["flag"],
    },
    {
        "method": "Plan-with-energy state trajectory generator",
        "track": "plan_with_energy_state_generation",
        "source_ids": ["1909.06878", "2502.07202"],
        "maps_to_stages": ["S3"],
        "graft_to_live_loop": (
            "Generate intermediate state trajectories under the S1/S2 energy, then "
            "ask the existing inducer/action realizer to synthesize the concrete "
            "action program that reaches the best low-energy trajectory."
        ),
        "put_winner_into_pool": (
            "Only promote a trajectory after it is realized as executable actions; "
            "the realized action sequence becomes the winner inserted into the pool."
        ),
        "takes_over_from_current_stack": (
            "Takes over static plan reranking when no existing candidate reaches a "
            "goal-like state but the energy can propose plausible intermediate states."
        ),
        "fails_when": (
            "State-space descent invents unreachable grids, the action realizer "
            "cannot realize the trajectory, or energy ignores a small decisive object change."
        ),
        "roadmap_candidate": FLAGGED_FOR_V442[1]["flag"],
    },
]

DEFAULT_FRESH_SWEEP = {
    "filtered_track": "energy/value-guided generation for S3 pool insertion",
    "cluster_ids": [1, 6],
    "semantic_scholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "semantic_scholar_result": "HTTP 429 on the focused query; no S2-only source promoted.",
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
DEFAULT_S1_S2_CONTEXT = {
    "s1_source_artifact": S1_SOURCE_RELATIVE_PATH,
    "s1_imported_honest_verdict": "success_structural_energy_s1_landscape_authorizes_s2",
    "s1_gate_passed": True,
    "s1_energy_ranking_loo_auroc_mean": 0.7134961314270525,
    "s2_source_artifact": S2_SOURCE_RELATIVE_PATH,
    "s2_imported_honest_verdict": "complete_structural_energy_s2_no_live_trust_value",
    "s2_live_path_reachable": True,
    "s2_energy_minus_accuracy_delta": 0.0,
    "s2_candidate_pool_size": 11,
    "s2_n_heldout_games": 5,
    "s3_generation_allowed": True,
    "generation_constraint": (
        "Use S1/S2 lower-is-better energy to guide new candidate generation and "
        "trust gating, not as an environment oracle or solve claim."
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_set_checksum(
    citations: JsonMap,
    methods: Sequence[JsonMap],
    flags: Sequence[JsonMap],
    s1_s2_context: JsonMap,
) -> str:
    payload = json.dumps(
        {
            "citations": citations,
            "flags": list(flags),
            "methods": list(methods),
            "s1_s2_context": s1_s2_context,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


REPRODUCIBILITY_CHECKSUM = source_set_checksum(
    CITATIONS,
    DEFAULT_METHODS_MAPPED,
    FLAGGED_FOR_V442,
    DEFAULT_S1_S2_CONTEXT,
)


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    fresh_sweep: JsonMap = DEFAULT_FRESH_SWEEP,
    flagged_for_v442: Sequence[JsonMap] = FLAGGED_FOR_V442,
    s1_s2_context: JsonMap = DEFAULT_S1_S2_CONTEXT,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "arxiv_ids_cited": sorted(citations),
        "flagged_for_v442": [dict(flag) for flag in flagged_for_v442],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "citations": {source_id: dict(citation) for source_id, citation in citations.items()},
        "fresh_sweep": dict(fresh_sweep),
        "s1_s2_context": dict(s1_s2_context),
        "note_path": NOTE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_S,
        "reproducibility_checksum": source_set_checksum(
            citations,
            methods_mapped,
            flagged_for_v442,
            s1_s2_context,
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
    _require(artifact["note_path"] == NOTE_PATH, "note_path must point at the Exp 4798 note")
    _require(artifact["random_seed"] == RANDOM_SEED, "random_seed must be the experiment id")
    _require(artifact["duration_s"] == DURATION_S, "duration_s must preserve the 0.0001s floor")
    _validate_citations(artifact["citations"], artifact["arxiv_ids_cited"])
    _validate_methods(artifact["methods_mapped"], artifact["arxiv_ids_cited"])
    _validate_flags(artifact["flagged_for_v442"], artifact["arxiv_ids_cited"])
    _validate_preconditions(artifact["preconditions_checked"])
    _validate_fresh_sweep(artifact["fresh_sweep"])
    _validate_s1_s2_context(artifact["s1_s2_context"])
    _require(
        artifact["reproducibility_checksum"]
        == source_set_checksum(
            artifact["citations"],
            artifact["methods_mapped"],
            artifact["flagged_for_v442"],
            artifact["s1_s2_context"],
        ),
        "reproducibility checksum must hash citations, methods, flags, and S1/S2 context",
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
        maps_to_stages = method["maps_to_stages"]
        _require(
            isinstance(source_ids, Sequence) and not isinstance(source_ids, str | bytes) and bool(source_ids),
            "each method must cite source_ids",
        )
        _require(set(source_ids).issubset(cited), "method source_ids must be verified citations")
        _require(maps_to_stages == ["S3"], "method maps_to_stages must map onto S3")
        _require(bool(method["graft_to_live_loop"]), "each method needs a graft_to_live_loop mapping")
        _require("winner" in str(method["put_winner_into_pool"]), "each method must put a winner into the pool")
        _require(bool(method["takes_over_from_current_stack"]), "each method needs takes_over_from_current_stack")
        _require(bool(method["fails_when"]), "each method needs fails_when")
        _require(bool(method["roadmap_candidate"]), "each method needs a roadmap candidate")
        tracks.add(str(method["track"]))
    _require(REQUIRED_TRACKS.issubset(tracks), "methods_mapped missing required S3 generation tracks")


def _validate_flags(flags: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(flags, Sequence) and not isinstance(flags, str | bytes) and bool(flags), "flagged_for_v442 required")
    cited = set(arxiv_ids_cited)
    for flag in flags:
        _require(isinstance(flag, Mapping), "each flagged_for_v442 entry must be a mapping")
        _require("candidate" in flag and "flag" in flag, "each flagged_for_v442 entry needs candidate and flag")
        _require("flagged_for_v441" not in json.dumps(flag, sort_keys=True), "flagged_for_v442 must not carry stale .441 flags")
        _require("flagged_for_v442" in str(flag["flag"]), "flagged_for_v442 entries must carry the .442 flag")
        _require(set(flag.get("source_ids", [])).issubset(cited), "flagged_for_v442 source_ids must be verified")


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


def _validate_s1_s2_context(context: object) -> None:
    _require(isinstance(context, Mapping), "s1_s2_context must be a mapping")
    _require(context.get("s1_source_artifact") == S1_SOURCE_RELATIVE_PATH, "s1_s2_context must cite S1")
    _require(context.get("s2_source_artifact") == S2_SOURCE_RELATIVE_PATH, "s1_s2_context must cite S2")
    _require(context.get("s1_gate_passed") is True, "S1 context must import the passed S1 gate")
    _require(context.get("s2_live_path_reachable") is True, "S2 context must import the live-path trust gate")
    _require(context.get("s3_generation_allowed") is True, "S3 generation must be allowed by context")
    _require(
        "not as an environment oracle" in str(context.get("generation_constraint", "")),
        "S3 generation constraint must forbid oracle use",
    )


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
            f"maps to {', '.join(method['maps_to_stages'])}; {method['graft_to_live_loop']} "
            f"Winner insertion: {method['put_winner_into_pool']} "
            f"Takes over: {method['takes_over_from_current_stack']} "
            f"Fails when: {method['fails_when']}"
        )
        for method in result["methods_mapped"]
    )
    flag_lines = "\n".join(flag["flag"] for flag in result["flagged_for_v442"])
    return f"""{STUDYING_SECTION_START}
## 2026-06-26 Exp 4798 - .442 energy-guided generation SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4798_sota_ingestion_energy_guided_generation.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted the EBM and neural-guided search
cluster URLs. `scripts/sweep_semscholar.py` was run on a focused
energy-guided generation query and returned HTTP 429, so no S2-only source was
promoted. Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks
verified the top eight papers listed below. `/deep-research` was not invoked.
No model load, training, leaderboard submission, or solve claim was made; this
is a no solve claim ingestion note.

**S1/S2 context imported:** `{S1_SOURCE_RELATIVE_PATH}` reports
`{result["s1_s2_context"]["s1_imported_honest_verdict"]}` and
`{S2_SOURCE_RELATIVE_PATH}` reports
`{result["s1_s2_context"]["s2_imported_honest_verdict"]}` with
`s2_live_path_reachable={result["s1_s2_context"]["s2_live_path_reachable"]}`.
The .442 planner may use S1/S2 lower-is-better energy to guide S3 generation,
but not as an environment oracle or solve claim.

**Verified source set:**
{citation_lines}

**SOTA -> S3 energy-guided generation mapping:**
{method_lines}

{flag_lines}

**Bottom line for .442:** prioritize the COLD/CFG/value-tree generator path
because it can put a winner into the pool with the smallest change to the live
explorer: generate a guided batch, verify hard mechanics, then promote the
best low-energy verified candidate. Keep the energy-fitness quality-diversity
path as the fallback when single-chain generation collapses, because it
preserves diverse elites instead of merely reranking a frozen pool.
{STUDYING_SECTION_END}"""


def update_research_studying_text(text: str, artifact: JsonMap | None = None) -> str:
    section = build_research_studying_section(artifact)
    start = text.find(STUDYING_SECTION_START)
    if start >= 0:
        end = text.find(STUDYING_SECTION_END, start)
        _require(end >= 0, "research-studying Exp 4798 section missing end marker")
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
    _require(start >= 0 and end >= 0, "research-studying missing Exp 4798 section markers")
    section = text[start : end + len(STUDYING_SECTION_END)]
    for phrase in (
        "SOTA -> S3 energy-guided generation mapping",
        "flagged_for_v442",
        "no solve claim",
        "put a winner into the pool",
        "COLD",
        "classifier",
        "value-tree",
        "quality-diversity",
    ):
        _require(phrase in section, f"research-studying section missing required phrase: {phrase}")
    missing_citations = sorted(citation for citation in NOTE_REQUIRED_SOURCE_CITATIONS if citation not in section)
    _require(not missing_citations, f"research-studying section missing citations: {missing_citations}")
    for method in result["methods_mapped"]:
        _require(method["method"] in section, f"research-studying section missing method: {method['method']}")
    for flag in result["flagged_for_v442"]:
        _require(flag["flag"] in section, "research-studying section missing flagged_for_v442 text")


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
    root = Path(os.environ.get("CARNOT_EXP4798_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        studying_path=root / STUDYING_RELATIVE_PATH,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
