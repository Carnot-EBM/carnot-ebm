"""Exp 4808 SOTA ingestion for energy-guided S3 generation.

Spec refs: REQ-ARC-WMTE-4808, SCENARIO-ARC-WMTE-4808,
SCENARIO-ARC-WMTE-4808-NO-FABRICATION.

This module writes a deterministic literature-ingestion artifact. The artifact
records how S3 should use energy or value feedback during generation itself,
so a newly generated winner can enter the live candidate pool. It does not
train a model, load an LLM, submit to a leaderboard, or claim an ARC solve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4808_sota_ingestion_energy_guided_generation.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
NOTE_PATH = "research-studying.md#exp-4808-sota-ingestion-energy-guided-generation"
RANDOM_SEED = 4808
DURATION_S = 0.0001
HONEST_VERDICT = "success_sota_ingestion_energy_guided_generation_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
STUDYING_SECTION_START = "<!-- EXP4808-SOTA-INGESTION-ENERGY-GUIDED-GENERATION-START -->"
STUDYING_SECTION_END = "<!-- EXP4808-SOTA-INGESTION-ENERGY-GUIDED-GENERATION-END -->"
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
        "principle": "every method claim must cite a verifiable arXiv ID."
    },
    "flagged_for_v443": {
        "principle": "the strongest method(s) flagged so the .443 planner reads the mapping."
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
    "s3_context": {
        "principle": "states that S3 uses energy for generation guidance, not oracle solve claims."
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
        "principle": "content hash of citations, method map, flags, and S3 context."
    },
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "methods_mapped",
    "arxiv_ids_cited",
    "flagged_for_v443",
    "inference_substrate",
    "preconditions_checked",
    "citations",
    "fresh_sweep",
    "s3_context",
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
        "2202.11705",
        "2207.12598",
        "2305.12018",
        "2309.15028",
        "2502.07202",
        "2605.28814",
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
    "energy guided sampling constrained text generation Langevin decoding value guided MCTS diffusion planning",
    "energy as fitness evolutionary search quality diversity generation value guided generation",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/1806.10230",
    "https://arxiv.org/abs/1909.06878",
    "https://arxiv.org/abs/2202.11705",
    "https://arxiv.org/abs/2207.12598",
    "https://arxiv.org/abs/2305.12018",
    "https://arxiv.org/abs/2309.15028",
    "https://arxiv.org/abs/2502.07202",
    "https://arxiv.org/abs/2605.28814",
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
    "2305.12018": {
        "title": "BOLT: Fast Energy-based Controlled Text Generation with Tunable Biases",
        "url": "https://arxiv.org/abs/2305.12018",
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
    "2605.28814": {
        "title": "Self-Improving Language Models with Bidirectional Evolutionary Search",
        "url": "https://arxiv.org/abs/2605.28814",
        "http_status": 200,
    },
}

FLAGGED_FOR_V443 = [
    {
        "candidate": "bolt_cold_cfg_value_tree_generator_for_s3",
        "flag": (
            "flagged_for_v443: bolt_cold_cfg_value_tree_generator_for_s3 "
            "(arXiv:2202.11705 + arXiv:2305.12018 + arXiv:2207.12598 + "
            "arXiv:2309.15028 + arXiv:2502.07202)"
        ),
        "source_ids": ["2202.11705", "2305.12018", "2207.12598", "2309.15028", "2502.07202"],
        "maps_to_stages": ["S3"],
    },
    {
        "candidate": "bes_energy_fitness_pool_inserter",
        "flag": (
            "flagged_for_v443: bes_energy_fitness_pool_inserter "
            "(arXiv:2605.28814 + arXiv:1806.10230 + arXiv:1909.06878)"
        ),
        "source_ids": ["2605.28814", "1806.10230", "1909.06878"],
        "maps_to_stages": ["S3"],
    },
]

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Energy-constrained sampler with fast logit-bias refinement",
        "track": "energy_constrained_sampling",
        "source_ids": ["2202.11705", "2305.12018"],
        "maps_to_stages": ["S3"],
        "graft_to_live_loop": (
            "Represent action programs or latent plans as variables under the S3 "
            "energy, then use short Langevin-style edits or BOLT-style tunable "
            "biases to move candidates before hard mechanics verification."
        ),
        "put_winner_into_pool": (
            "Generate a small guided batch, replay-verify each candidate, and put "
            "the lowest-energy verified winner into the explorer pool."
        ),
        "takes_over_from_current_stack": (
            "Takes over blind local mutation when the pool has no candidate that "
            "already wins but the verifier can score partial plausibility."
        ),
        "fails_when": (
            "The candidate representation has no smooth edit neighborhood, "
            "energy gradients point toward unplayable programs, or bias tuning "
            "collapses diversity before verification."
        ),
        "roadmap_candidate": FLAGGED_FOR_V443[0]["flag"],
    },
    {
        "method": "Classifier/score-guided proposal sampler",
        "track": "classifier_score_guided_generation",
        "source_ids": ["2207.12598"],
        "maps_to_stages": ["S3"],
        "graft_to_live_loop": (
            "Treat S3 energy as a classifier-like guidance signal during proposal "
            "construction, with guidance strength increased only after action "
            "syntax and transition checks keep the candidate executable."
        ),
        "put_winner_into_pool": (
            "Sample a guided batch, choose the verified candidate with the best "
            "lower-is-better score, and put that winner into the live pool."
        ),
        "takes_over_from_current_stack": (
            "Takes over unguided proposal batches whose candidates are scored only "
            "after generation has already spent the action budget."
        ),
        "fails_when": (
            "Guidance scale overwhelms diversity, the score follows a provenance "
            "shortcut, or high guidance yields valid-looking but unplayable plans."
        ),
        "roadmap_candidate": FLAGGED_FOR_V443[0]["flag"],
    },
    {
        "method": "Value-guided tree and diffusion generation",
        "track": "value_guided_tree_generation",
        "source_ids": ["2309.15028", "2502.07202"],
        "maps_to_stages": ["S3"],
        "graft_to_live_loop": (
            "Expand partial generated plans as a tree and use negative energy as "
            "the value term for partial rollouts; revisit branches whose decoded "
            "or denoised continuations improve the trust score."
        ),
        "put_winner_into_pool": (
            "Complete the best tree leaf into executable actions, verify it, and "
            "put the verified low-energy winner into the candidate pool."
        ),
        "takes_over_from_current_stack": (
            "Takes over one-shot generation by spending inference-time compute on "
            "partial candidates before they become expensive live actions."
        ),
        "fails_when": (
            "Partial-plan energy is poorly calibrated, branching cost exceeds the "
            "action-efficiency budget, or the tree repeatedly explores equivalent leaves."
        ),
        "roadmap_candidate": FLAGGED_FOR_V443[0]["flag"],
    },
    {
        "method": "Energy-as-fitness evolutionary pool search",
        "track": "energy_as_fitness_evolutionary_search",
        "source_ids": ["2605.28814", "1806.10230"],
        "maps_to_stages": ["S3"],
        "graft_to_live_loop": (
            "Run a tiny population over action programs where fitness is negative "
            "S3 energy plus executable validity, and use recombination or guided "
            "random search to escape high-probability but losing rollouts."
        ),
        "put_winner_into_pool": (
            "Select the verified elite with the best energy-adjusted fitness and "
            "put that winner into the explorer pool while retaining alternates."
        ),
        "takes_over_from_current_stack": (
            "Takes over single-candidate mutation by preserving a small population "
            "of high-quality candidates instead of only the current best guess."
        ),
        "fails_when": (
            "Fitness overweights novelty, surrogate energy points away from true "
            "transition mechanics, or evolution burns budget without verified elites."
        ),
        "roadmap_candidate": FLAGGED_FOR_V443[1]["flag"],
    },
    {
        "method": "Plan-with-energy state trajectory generator",
        "track": "plan_with_energy_state_generation",
        "source_ids": ["1909.06878", "2502.07202"],
        "maps_to_stages": ["S3"],
        "graft_to_live_loop": (
            "Generate intermediate state trajectories under the energy model, then "
            "ask the action realizer to synthesize concrete actions that reach the "
            "best low-energy trajectory."
        ),
        "put_winner_into_pool": (
            "Promote a trajectory only after it is realized as executable actions; "
            "the realized action sequence becomes the winner inserted into the pool."
        ),
        "takes_over_from_current_stack": (
            "Takes over static plan reranking when no current candidate reaches a "
            "goal-like state but the energy can propose plausible intermediate states."
        ),
        "fails_when": (
            "State-space descent invents unreachable grids, the action realizer "
            "cannot realize the trajectory, or energy ignores a small decisive change."
        ),
        "roadmap_candidate": FLAGGED_FOR_V443[1]["flag"],
    },
]

DEFAULT_FRESH_SWEEP = {
    "filtered_track": "energy/value-guided generation for S3 pool insertion",
    "cluster_ids": [1, 6],
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
DEFAULT_S3_CONTEXT = {
    "roadmap_target": ".443",
    "s3_generation_allowed": True,
    "input_energy_contract": "lower-is-better energy/value feedback from upstream trust and verification artifacts",
    "generation_constraint": (
        "Use energy to guide generation and pool insertion, not as an environment oracle "
        "or solve claim."
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_set_checksum(
    citations: JsonMap,
    methods: Sequence[JsonMap],
    flags: Sequence[JsonMap],
    s3_context: JsonMap,
) -> str:
    payload = json.dumps(
        {
            "citations": citations,
            "flags": list(flags),
            "methods": list(methods),
            "s3_context": s3_context,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


REPRODUCIBILITY_CHECKSUM = source_set_checksum(
    CITATIONS,
    DEFAULT_METHODS_MAPPED,
    FLAGGED_FOR_V443,
    DEFAULT_S3_CONTEXT,
)


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    fresh_sweep: JsonMap = DEFAULT_FRESH_SWEEP,
    flagged_for_v443: Sequence[JsonMap] = FLAGGED_FOR_V443,
    s3_context: JsonMap = DEFAULT_S3_CONTEXT,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "arxiv_ids_cited": sorted(citations),
        "flagged_for_v443": [dict(flag) for flag in flagged_for_v443],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "citations": {source_id: dict(citation) for source_id, citation in citations.items()},
        "fresh_sweep": dict(fresh_sweep),
        "s3_context": dict(s3_context),
        "note_path": NOTE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_S,
        "reproducibility_checksum": source_set_checksum(
            citations,
            methods_mapped,
            flagged_for_v443,
            s3_context,
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
    _require(artifact["note_path"] == NOTE_PATH, "note_path must point at the Exp 4808 note")
    _require(artifact["random_seed"] == RANDOM_SEED, "random_seed must be the experiment id")
    _require(artifact["duration_s"] == DURATION_S, "duration_s must preserve the 0.0001s floor")
    _validate_citations(artifact["citations"], artifact["arxiv_ids_cited"])
    _validate_methods(artifact["methods_mapped"], artifact["arxiv_ids_cited"])
    _validate_flags(artifact["flagged_for_v443"], artifact["arxiv_ids_cited"])
    _validate_preconditions(artifact["preconditions_checked"])
    _validate_fresh_sweep(artifact["fresh_sweep"])
    _validate_s3_context(artifact["s3_context"])
    _require(
        artifact["reproducibility_checksum"]
        == source_set_checksum(
            artifact["citations"],
            artifact["methods_mapped"],
            artifact["flagged_for_v443"],
            artifact["s3_context"],
        ),
        "reproducibility checksum must hash citations, methods, flags, and S3 context",
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
    _require(REQUIRED_TRACKS == tracks, "methods_mapped missing required S3 generation tracks")


def _validate_flags(flags: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(flags, Sequence) and not isinstance(flags, str | bytes) and bool(flags), "flagged_for_v443 required")
    cited = set(arxiv_ids_cited)
    for flag in flags:
        _require(isinstance(flag, Mapping), "each flagged_for_v443 entry must be a mapping")
        _require("candidate" in flag and "flag" in flag, "each flagged_for_v443 entry needs candidate and flag")
        _require("flagged_for_v442" not in json.dumps(flag, sort_keys=True), "flagged_for_v443 must not carry stale .442 flags")
        _require("flagged_for_v443" in str(flag["flag"]), "flagged_for_v443 entries must carry the .443 flag")
        _require(set(flag.get("source_ids", [])).issubset(cited), "flagged_for_v443 source_ids must be verified")


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


def _validate_s3_context(context: object) -> None:
    _require(isinstance(context, Mapping), "s3_context must be a mapping")
    _require(context.get("roadmap_target") == ".443", "s3_context must target the .443 roadmap")
    _require(context.get("s3_generation_allowed") is True, "S3 generation must be allowed by context")
    _require("energy" in str(context.get("input_energy_contract", "")), "s3_context must name energy input")
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
    flag_lines = "\n".join(flag["flag"] for flag in result["flagged_for_v443"])
    return f"""{STUDYING_SECTION_START}
## 2026-06-26 Exp 4808 - .443 energy-guided generation SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4808_sota_ingestion_energy_guided_generation.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted the EBM and neural-guided search
cluster URLs. `scripts/sweep_semscholar.py` was run on three focused
energy-guided generation queries and returned HTTP 429 for all of them, so no
S2-only source was promoted. Low-concurrency WebSearch/WebFetch plus direct
arXiv HTTP checks verified the top eight papers listed below. `/deep-research`
was not invoked. No model load, training, leaderboard submission, or solve
claim was made; this is a no solve claim ingestion note.

**Verified source set:**
{citation_lines}

**SOTA -> S3 energy-guided generation mapping:**
{method_lines}

{flag_lines}

**Bottom line for .443:** prioritize the BOLT/COLD/CFG/value-tree generator
because it can put a winner into the pool with the smallest live-stack change:
generate a guided batch, verify hard mechanics, then promote the best
low-energy verified candidate. Keep BES-style energy-as-fitness evolution as
the fallback when guided chains collapse, because recombination can escape
high-probability losing rollouts while still using the same verifier energy.
{STUDYING_SECTION_END}"""


def update_research_studying_text(text: str, artifact: JsonMap | None = None) -> str:
    section = build_research_studying_section(artifact)
    start = text.find(STUDYING_SECTION_START)
    if start >= 0:
        end = text.find(STUDYING_SECTION_END, start)
        _require(end >= 0, "research-studying Exp 4808 section missing end marker")
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
    _require(start >= 0 and end >= 0, "research-studying missing Exp 4808 section markers")
    section = text[start : end + len(STUDYING_SECTION_END)]
    for phrase in (
        "SOTA -> S3 energy-guided generation mapping",
        "flagged_for_v443",
        "no solve claim",
        "put a winner into the pool",
        "BOLT",
        "COLD",
        "CFG",
        "BES",
    ):
        _require(phrase in section, f"research-studying section missing required phrase: {phrase}")
    missing_citations = sorted(citation for citation in NOTE_REQUIRED_SOURCE_CITATIONS if citation not in section)
    _require(not missing_citations, f"research-studying section missing citations: {missing_citations}")
    for method in result["methods_mapped"]:
        _require(method["method"] in section, f"research-studying section missing method: {method['method']}")
    for flag in result["flagged_for_v443"]:
        _require(flag["flag"] in section, "research-studying section missing flagged_for_v443 text")


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
    root = Path(os.environ.get("CARNOT_EXP4808_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        studying_path=root / STUDYING_RELATIVE_PATH,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
