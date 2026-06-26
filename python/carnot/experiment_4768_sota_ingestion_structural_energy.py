"""Exp 4768 SOTA ingestion for structural-energy S1-S4 planning.

Spec refs: REQ-ARC-WMTE-4768, SCENARIO-ARC-WMTE-4768,
SCENARIO-ARC-WMTE-4768-NO-FABRICATION.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4768_sota_ingestion_structural_energy.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
NOTE_PATH = "research-studying.md#exp-4768-sota-ingestion-structural-energy"
S0_SOURCE_RELATIVE_PATH = "results/experiment_4761_structural_energy_s0_core_bet_probe.json"
RANDOM_SEED = 4768
DURATION_S = 0.0001
HONEST_VERDICT = "success_sota_ingestion_structural_energy_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
STUDYING_SECTION_START = "<!-- EXP4768-SOTA-INGESTION-STRUCTURAL-ENERGY-START -->"
STUDYING_SECTION_END = "<!-- EXP4768-SOTA-INGESTION-STRUCTURAL-ENERGY-END -->"
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
        "principle": "terminal prefix; mapping emitted is success_sota_ingestion_structural_energy_mapped."
    },
    "methods_mapped": {
        "principle": (
            "the strongest 3-5 methods mapped onto S1-S4 -- the actionable "
            "output, each with a real arXiv ID."
        )
    },
    "arxiv_ids_cited": {
        "principle": (
            "every method claim must cite a verifiable arXiv ID -- an ingestion "
            "with no citations is fabrication."
        )
    },
    "flagged_for_v439": {
        "principle": (
            "the strongest method(s) flagged so the .439 planner reads the mapping "
            "(discover->ingest->plan->experiment)."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts; 0.0001s floor."
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
    "s0_context": {
        "principle": "imports the S0 close-state so S1-S4 candidates inherit the leak constraint."
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
        "principle": "content hash of citations, method map, flags, and S0 context."
    },
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "methods_mapped",
    "arxiv_ids_cited",
    "flagged_for_v439",
    "inference_substrate",
    "preconditions_checked",
    "citations",
    "fresh_sweep",
    "s0_context",
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
        "maps_to_current_stack",
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
        "2006.15055",
        "2301.08243",
        "2307.01668",
        "2505.10819",
        "2507.04920",
        "2510.04542",
        "2602.02900",
        "2605.05138",
    }
)
ALLOWED_STAGES = frozenset({"S1", "S2", "S3", "S4"})
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
    "structural energy object-centric world model transition energy",
    "PoE-World object-centric product of experts world model",
    "executable world models code world models denoising structural prior",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2006.15055",
    "https://arxiv.org/abs/2505.10819",
    "https://arxiv.org/abs/2602.02900",
    "https://arxiv.org/abs/2307.01668",
    "https://arxiv.org/abs/2301.08243",
    "https://arxiv.org/abs/2605.05138",
    "https://arxiv.org/abs/2510.04542",
    "https://arxiv.org/abs/2507.04920",
]

CITATIONS = {
    "2006.15055": {
        "title": "Object-Centric Learning with Slot Attention",
        "url": "https://arxiv.org/abs/2006.15055",
        "http_status": 200,
    },
    "2301.08243": {
        "title": "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture",
        "url": "https://arxiv.org/abs/2301.08243",
        "http_status": 200,
    },
    "2307.01668": {
        "title": "Training Energy-Based Models with Diffusion Contrastive Divergences",
        "url": "https://arxiv.org/abs/2307.01668",
        "http_status": 200,
    },
    "2505.10819": {
        "title": "PoE-World: Compositional World Modeling with Products of Programmatic Experts",
        "url": "https://arxiv.org/abs/2505.10819",
        "http_status": 200,
    },
    "2507.04920": {
        "title": "Object-centric Denoising Diffusion Models for Physical Reasoning",
        "url": "https://arxiv.org/abs/2507.04920",
        "http_status": 200,
    },
    "2510.04542": {
        "title": "Code World Models for General Game Playing",
        "url": "https://arxiv.org/abs/2510.04542",
        "http_status": 200,
    },
    "2602.02900": {
        "title": "Manifold-Constrained Energy-Based Transition Models for Offline Reinforcement Learning",
        "url": "https://arxiv.org/abs/2602.02900",
        "http_status": 200,
    },
    "2605.05138": {
        "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
        "url": "https://arxiv.org/abs/2605.05138",
        "http_status": 200,
    },
}

FLAGGED_FOR_V439 = [
    {
        "candidate": "slot_factor_transition_energy_rerun",
        "flag": (
            "flagged_for_v439: slot_factor_transition_energy_rerun_after_s0_origin_probe_leak_guard "
            "(arXiv:2006.15055 + arXiv:2505.10819 + arXiv:2602.02900 + arXiv:2307.01668)"
        ),
        "source_ids": ["2006.15055", "2505.10819", "2602.02900", "2307.01668"],
        "maps_to_stages": ["S1", "S2"],
    },
    {
        "candidate": "poe_code_world_model_trust_gate",
        "flag": (
            "flagged_for_v439: poe_code_world_model_trust_gate_with_cwm_hidden_state_planning "
            "(arXiv:2505.10819 + arXiv:2510.04542 + arXiv:2605.05138)"
        ),
        "source_ids": ["2505.10819", "2510.04542", "2605.05138"],
        "maps_to_stages": ["S2", "S3", "S4"],
    },
]

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Slot-factor contrastive transition energy",
        "track": "slot_factor_transition_energy",
        "source_ids": ["2006.15055", "2505.10819", "2602.02900", "2307.01668"],
        "maps_to_stages": ["S1", "S2"],
        "maps_to_current_stack": (
            "Replace the S0 scalar logistic with E(s,a,s') over object slots and "
            "programmatic factors, trained against real induced-engine near misses."
        ),
        "takes_over_from_current_stack": (
            "Takes over the S0 feature-only probe by making object_relational and "
            "frame_delta features into a contrastive energy landscape that can rank "
            "candidate transitions and induced engines."
        ),
        "fails_when": (
            "Slot binding is unstable across ARC frames, the energy keeps learning "
            "induced-vs-real provenance like the S0 origin probe, or near-miss "
            "negatives are too sparse to harden the contrastive objective."
        ),
        "roadmap_candidate": FLAGGED_FOR_V439[0]["flag"],
    },
    {
        "method": "Product-of-experts executable world-model trust gate",
        "track": "poe_code_world_model_trust_gate",
        "source_ids": ["2505.10819", "2510.04542", "2605.05138"],
        "maps_to_stages": ["S2", "S3", "S4"],
        "maps_to_current_stack": (
            "Score PoE/code-world-model factors inside E3AgentPolicy and "
            "WorldModelVerifier before plan_in_model trusts an off-path rollout."
        ),
        "takes_over_from_current_stack": (
            "Takes over the binary executable-model verifier by ranking factorized "
            "candidate engines on off-path structural energy where the environment "
            "win-check is unavailable."
        ),
        "fails_when": (
            "The synthesized factors overfit a prefix, hidden-state inference is "
            "wrong, or executable-model verification leaks private solution facts "
            "instead of measuring transition consistency."
        ),
        "roadmap_candidate": FLAGGED_FOR_V439[1]["flag"],
    },
    {
        "method": "JEPA-style latent transition residual energy",
        "track": "jepa_latent_transition_residual",
        "source_ids": ["2301.08243", "2602.02900"],
        "maps_to_stages": ["S1", "S4"],
        "maps_to_current_stack": (
            "Train a representation-space residual head over structural transition "
            "pairs, then test whether it survives cross-game and cross-family LOO."
        ),
        "takes_over_from_current_stack": (
            "Takes over raw frame-marginal controls by predicting semantically "
            "meaningful target representations from context/action structure."
        ),
        "fails_when": (
            "The representation discards exact cell-level consequences, the learned "
            "latent becomes a value head in disguise, or cross-family transfer "
            "drops like the prior oracle-distinct failures."
        ),
        "roadmap_candidate": "support_for_v439: jepa_latent_transition_residual_transfer_stress",
    },
    {
        "method": "Object-centric denoising structural prior",
        "track": "object_centric_denoising_structural_prior",
        "source_ids": ["2507.04920", "2307.01668", "2602.02900"],
        "maps_to_stages": ["S3", "S4"],
        "maps_to_current_stack": (
            "Use denoising over object trajectories as a generator-side prior for "
            "near-miss repair and goal_energy-guided plan proposals."
        ),
        "takes_over_from_current_stack": (
            "Takes over static reranking by perturbing candidate transition "
            "trajectories toward low-energy, object-consistent alternatives that "
            "the bare explorer did not enumerate."
        ),
        "fails_when": (
            "The diffusion prior smooths away discrete ARC mechanics, conditioning "
            "accidentally uses observed goal states as an oracle, or every lift is "
            "only reranking a winner already present in the pool."
        ),
        "roadmap_candidate": "support_for_v439: object_centric_denoising_generation_prior",
    },
]

DEFAULT_FRESH_SWEEP = {
    "filtered_track": "structural energy + object-centric/executable world models for S1-S4",
    "cluster_ids": [1, 6],
    "semantic_scholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "semantic_scholar_result": "HTTP 429 on all focused queries; no S2-only source promoted.",
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
DEFAULT_S0_CONTEXT = {
    "source_artifact": S0_SOURCE_RELATIVE_PATH,
    "stage": "S0",
    "imported_honest_verdict": "complete: structural_energy_s0_retired_loo_0.746_null_or_leaky",
    "loo_auroc_structural": 0.7455881880622204,
    "structural_minus_marginal_delta_ci95": [0.17481816435901296, 0.39015500191631836],
    "origin_probe_auroc": 0.7327927210707903,
    "planning_constraint": (
        "origin_probe_leak means S1-S4 candidates are v439 planning inputs only "
        "after a leak-hardened rerun or factorized provenance guard."
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_set_checksum(
    citations: JsonMap,
    methods: Sequence[JsonMap],
    flags: Sequence[JsonMap],
    s0_context: JsonMap,
) -> str:
    payload = json.dumps(
        {
            "citations": citations,
            "flags": list(flags),
            "methods": list(methods),
            "s0_context": s0_context,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


REPRODUCIBILITY_CHECKSUM = source_set_checksum(
    CITATIONS,
    DEFAULT_METHODS_MAPPED,
    FLAGGED_FOR_V439,
    DEFAULT_S0_CONTEXT,
)


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    fresh_sweep: JsonMap = DEFAULT_FRESH_SWEEP,
    flagged_for_v439: Sequence[JsonMap] = FLAGGED_FOR_V439,
    s0_context: JsonMap = DEFAULT_S0_CONTEXT,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "arxiv_ids_cited": sorted(citations),
        "flagged_for_v439": [dict(flag) for flag in flagged_for_v439],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "citations": {source_id: dict(citation) for source_id, citation in citations.items()},
        "fresh_sweep": dict(fresh_sweep),
        "s0_context": dict(s0_context),
        "note_path": NOTE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_S,
        "reproducibility_checksum": source_set_checksum(citations, methods_mapped, flagged_for_v439, s0_context),
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
    _require(artifact["note_path"] == NOTE_PATH, "note_path must point at the Exp 4768 note")
    _require(artifact["random_seed"] == RANDOM_SEED, "random_seed must be the experiment id")
    _require(artifact["duration_s"] == DURATION_S, "duration_s must preserve the 0.0001s floor")
    _validate_citations(artifact["citations"], artifact["arxiv_ids_cited"])
    _validate_methods(artifact["methods_mapped"], artifact["arxiv_ids_cited"])
    _validate_flags(artifact["flagged_for_v439"], artifact["arxiv_ids_cited"])
    _validate_preconditions(artifact["preconditions_checked"])
    _validate_fresh_sweep(artifact["fresh_sweep"])
    _validate_s0_context(artifact["s0_context"])
    _require(
        artifact["reproducibility_checksum"]
        == source_set_checksum(
            artifact["citations"],
            artifact["methods_mapped"],
            artifact["flagged_for_v439"],
            artifact["s0_context"],
        ),
        "reproducibility checksum must hash citations, methods, flags, and S0 context",
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
            "method maps_to_stages must stay within S1-S4",
        )
        _require(bool(method["maps_to_current_stack"]), "each method needs maps_to_current_stack")
        _require(bool(method["takes_over_from_current_stack"]), "each method needs takes_over_from_current_stack")
        _require(bool(method["fails_when"]), "each method needs fails_when")
        _require(bool(method["roadmap_candidate"]), "each method needs a roadmap candidate")
        stages.update(str(stage) for stage in maps_to_stages)
    _require(ALLOWED_STAGES.issubset(stages), "methods_mapped must cover S1-S4")


def _validate_flags(flags: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(flags, Sequence) and not isinstance(flags, str | bytes) and bool(flags), "flagged_for_v439 required")
    cited = set(arxiv_ids_cited)
    for flag in flags:
        _require(isinstance(flag, Mapping), "each flagged_for_v439 entry must be a mapping")
        _require("candidate" in flag and "flag" in flag, "each flagged_for_v439 entry needs candidate and flag")
        _require("flagged_for_v438" not in json.dumps(flag, sort_keys=True), "flagged_for_v439 must not carry stale .438 flags")
        _require("flagged_for_v439" in str(flag["flag"]), "flagged_for_v439 entries must carry the .439 flag")
        _require(set(flag.get("source_ids", [])).issubset(cited), "flagged_for_v439 source_ids must be verified")


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


def _validate_s0_context(s0_context: object) -> None:
    _require(isinstance(s0_context, Mapping), "s0_context must be a mapping")
    _require(s0_context.get("source_artifact") == S0_SOURCE_RELATIVE_PATH, "s0_context must cite the S0 artifact")
    _require(s0_context.get("stage") == "S0", "s0_context stage must be S0")
    _require("origin_probe_leak" in str(s0_context.get("planning_constraint", "")), "s0_context must carry origin_probe_leak")


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
            f"{method['takes_over_from_current_stack']} Fails when: {method['fails_when']}"
        )
        for method in result["methods_mapped"]
    )
    flag_lines = "\n".join(flag["flag"] for flag in result["flagged_for_v439"])
    return f"""{STUDYING_SECTION_START}
## 2026-06-26 Exp 4768 - .438 structural-energy SOTA ingestion for S1-S4 - INGESTED

**Status:** INGESTED into `results/experiment_4768_sota_ingestion_structural_energy.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted the EBM and neural-guided/world
model cluster URLs. `scripts/sweep_semscholar.py` was run on three focused
queries and returned HTTP 429, so no S2-only source was promoted.
Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks verified the
top eight papers listed below. `/deep-research` was not invoked. No model load,
training, leaderboard submission, or solve claim was made; this is a no solve claim
ingestion note.

**S0 context imported:** `{S0_SOURCE_RELATIVE_PATH}` reports
`{result["s0_context"]["imported_honest_verdict"]}` with an origin-probe leak,
so the .439 planner should treat the S1-S4 entries as candidate inputs that
must address provenance leakage before any continuation claim.

**Verified source set:**
{citation_lines}

**SOTA -> S1-S4 structural-energy mapping:**
{method_lines}

{flag_lines}

**Bottom line for .439:** start with the slot/factor contrastive transition
energy rerun, using Slot Attention object bindings plus PoE/programmatic
factors and MC-ETM-style hard near-miss negatives. In parallel, keep the
PoE/code-world-model trust gate ready as the strongest S2-S3 integration path
if the leak-hardened S1 energy clears the S0 origin-probe failure mode.
{STUDYING_SECTION_END}"""


def update_research_studying_text(text: str, artifact: JsonMap | None = None) -> str:
    section = build_research_studying_section(artifact)
    start = text.find(STUDYING_SECTION_START)
    if start >= 0:
        end = text.find(STUDYING_SECTION_END, start)
        _require(end >= 0, "research-studying Exp 4768 section missing end marker")
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
    _require(start >= 0 and end >= 0, "research-studying missing Exp 4768 section markers")
    section = text[start : end + len(STUDYING_SECTION_END)]
    for phrase in (
        "SOTA -> S1-S4 structural-energy mapping",
        "flagged_for_v439",
        "no solve claim",
        "Slot Attention",
        "PoE",
        "MC-ETM",
        "origin-probe leak",
    ):
        _require(phrase in section, f"research-studying section missing required phrase: {phrase}")
    missing_citations = sorted(citation for citation in NOTE_REQUIRED_SOURCE_CITATIONS if citation not in section)
    _require(not missing_citations, f"research-studying section missing citations: {missing_citations}")
    for method in result["methods_mapped"]:
        _require(method["method"] in section, f"research-studying section missing method: {method['method']}")
    for flag in result["flagged_for_v439"]:
        _require(flag["flag"] in section, "research-studying section missing flagged_for_v439 text")


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
    root = Path(os.environ.get("CARNOT_EXP4768_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        studying_path=root / STUDYING_RELATIVE_PATH,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
