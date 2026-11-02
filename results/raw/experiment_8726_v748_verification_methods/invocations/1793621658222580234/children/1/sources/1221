"""Exp 4828 SOTA ingestion for S4 cross-family transfer.

Spec refs: REQ-ARC-WMTE-4828, SCENARIO-ARC-WMTE-4828,
SCENARIO-ARC-WMTE-4828-NO-FABRICATION.

This module writes a deterministic literature-ingestion artifact. The artifact
turns the fresh cross-family transfer sweep into an S4 plan: evaluate learned
energies and verifiers on families they did not train on, prefer worst-family
robustness over pooled averages, and flag the strongest training/evaluation
controls for the .445 planner. It performs no model load, training,
leaderboard submission, or solve claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4828_sota_ingestion_cross_family_transfer.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
NOTE_PATH = "research-studying.md#exp-4828-sota-ingestion-cross-family-transfer"
RANDOM_SEED = 4828
DURATION_S = 0.0001
HONEST_VERDICT = "success_sota_ingestion_cross_family_transfer_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
STUDYING_SECTION_START = "<!-- EXP4828-SOTA-INGESTION-CROSS-FAMILY-TRANSFER-START -->"
STUDYING_SECTION_END = "<!-- EXP4828-SOTA-INGESTION-CROSS-FAMILY-TRANSFER-END -->"
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
            "success_sota_ingestion_cross_family_transfer_mapped."
        )
    },
    "methods_mapped": {
        "principle": "the strongest 3-5 methods mapped onto S4, each with a real arXiv ID."
    },
    "arxiv_ids_cited": {
        "principle": "every method claim must cite a verifiable arXiv ID."
    },
    "flagged_for_v445": {
        "principle": "the strongest method(s) flagged so the .445 planner reads the mapping."
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
    "s4_context": {
        "principle": "states that S4 is a held-out family transfer stress, not a solve claim."
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
        "principle": "content hash of citations, method map, flags, and S4 context."
    },
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "methods_mapped",
    "arxiv_ids_cited",
    "flagged_for_v445",
    "inference_substrate",
    "preconditions_checked",
    "citations",
    "fresh_sweep",
    "s4_context",
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
        "s4_graft",
        "heldout_family_test",
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
        "1911.08731",
        "2003.00688",
        "2007.01434",
        "2012.07421",
        "2311.14743",
        "2403.13787",
        "2602.08489",
        "2605.25629",
    }
)
REQUIRED_TRACKS = frozenset(
    {
        "heldout_transfer_evaluation",
        "representation_anchoring",
        "worst_family_robust_optimization",
        "risk_extrapolation",
        "transferable_reward_stress",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS)

CLUSTER_0_URL = (
    'http://export.arxiv.org/api/query?search_query=(abs:"verifier+ensemble"+OR+'
    'abs:"verifier+ensembles"+OR+abs:"null+space"+OR+abs:"specification+gaming"+'
    'OR+abs:"process+reward+model"+OR+abs:"deliberative+alignment"+OR+'
    'abs:"reward+hacking")&start=0&max_results=8&sortBy=submittedDate&'
    "sortOrder=descending"
)
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
    "cross domain transfer learned verifier reward model distribution shift robustness",
    "domain generalization invariant risk minimization group DRO WILDS reward model verifier",
    "reward model preference shift weak to strong cross dataset transfer robust evaluation",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/1911.08731",
    "https://arxiv.org/abs/2003.00688",
    "https://arxiv.org/abs/2007.01434",
    "https://arxiv.org/abs/2012.07421",
    "https://arxiv.org/abs/2311.14743",
    "https://arxiv.org/abs/2403.13787",
    "https://arxiv.org/abs/2602.08489",
    "https://arxiv.org/abs/2605.25629",
]

CITATIONS = {
    "1911.08731": {
        "title": (
            "Distributionally Robust Neural Networks for Group Shifts: On the "
            "Importance of Regularization for Worst-Case Generalization"
        ),
        "url": "https://arxiv.org/abs/1911.08731",
        "http_status": 200,
    },
    "2003.00688": {
        "title": "Out-of-Distribution Generalization via Risk Extrapolation (REx)",
        "url": "https://arxiv.org/abs/2003.00688",
        "http_status": 200,
    },
    "2007.01434": {
        "title": "In Search of Lost Domain Generalization",
        "url": "https://arxiv.org/abs/2007.01434",
        "http_status": 200,
    },
    "2012.07421": {
        "title": "WILDS: A Benchmark of in-the-Wild Distribution Shifts",
        "url": "https://arxiv.org/abs/2012.07421",
        "http_status": 200,
    },
    "2311.14743": {
        "title": (
            "A Baseline Analysis of Reward Models' Ability To Accurately Analyze "
            "Foundation Models Under Distribution Shift"
        ),
        "url": "https://arxiv.org/abs/2311.14743",
        "http_status": 200,
    },
    "2403.13787": {
        "title": "RewardBench: Evaluating Reward Models for Language Modeling",
        "url": "https://arxiv.org/abs/2403.13787",
        "http_status": 200,
    },
    "2602.08489": {
        "title": "Beyond Correctness: Learning Robust Reasoning via Transfer",
        "url": "https://arxiv.org/abs/2602.08489",
        "http_status": 200,
    },
    "2605.25629": {
        "title": (
            "When In-Distribution Gains Fail: Evaluating Weak-to-Strong Reward "
            "Models under Preference Shift"
        ),
        "url": "https://arxiv.org/abs/2605.25629",
        "http_status": 200,
    },
}

FLAGGED_FOR_V445 = [
    {
        "candidate": "anchor_leave_one_family_transfer_gate",
        "flag": (
            "flagged_for_v445: anchor_leave_one_family_transfer_gate "
            "(arXiv:2605.25629 + arXiv:2311.14743 + arXiv:2403.13787)"
        ),
        "source_ids": ["2605.25629", "2311.14743", "2403.13787"],
        "maps_to_stages": ["S4"],
    },
    {
        "candidate": "worst_family_group_dro_s4_energy",
        "flag": (
            "flagged_for_v445: worst_family_group_dro_s4_energy "
            "(arXiv:1911.08731 + arXiv:2012.07421)"
        ),
        "source_ids": ["1911.08731", "2012.07421"],
        "maps_to_stages": ["S4"],
    },
    {
        "candidate": "rex_transferable_reward_stress",
        "flag": (
            "flagged_for_v445: rex_transferable_reward_stress "
            "(arXiv:2003.00688 + arXiv:2602.08489 + arXiv:2007.01434)"
        ),
        "source_ids": ["2003.00688", "2602.08489", "2007.01434"],
        "maps_to_stages": ["S4"],
    },
]

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Leave-one-family reward/verifier transfer gate",
        "track": "heldout_transfer_evaluation",
        "source_ids": ["2311.14743", "2403.13787", "2007.01434", "2012.07421"],
        "maps_to_stages": ["S4"],
        "s4_graft": (
            "Split the ARC family corpus into source families and one held-out "
            "target family, then report S4 energy accuracy, calibration, and "
            "OOD transfer deltas separately for prompt shift and response shift."
        ),
        "heldout_family_test": (
            "A family is successful only if its held-out score, worst-family "
            "score, and calibration stay positive; pooled source-family averages "
            "cannot authorize S4."
        ),
        "takes_over_from_current_stack": (
            "Takes over the old pooled transfer readout that let a strong source "
            "family hide a brittle held-out family."
        ),
        "fails_when": (
            "The split leaks level identity, the held-out family has too few "
            "examples, or the verifier is calibrated only on source-family outputs."
        ),
        "roadmap_candidate": FLAGGED_FOR_V445[0]["flag"],
    },
    {
        "method": "Representation anchoring for verifier-energy fine-tuning",
        "track": "representation_anchoring",
        "source_ids": ["2605.25629"],
        "maps_to_stages": ["S4"],
        "s4_graft": (
            "Fine-tune the S4 energy with an anchor penalty to keep the verifier "
            "near the pretrained representation while allowing source-family "
            "adaptation where it improves held-out transfer."
        ),
        "heldout_family_test": (
            "Pick the anchor weight by source-family validation only, then score "
            "the held-out family once and require transfer-aware gain rather than "
            "source-family memorization."
        ),
        "takes_over_from_current_stack": (
            "Takes over unconstrained verifier fine-tuning that can chase family "
            "style features and lose the transferable representation."
        ),
        "fails_when": (
            "The base representation lacks cross-family signal, the anchor is so "
            "strong that useful adaptation is blocked, or source labels encode "
            "family shortcuts."
        ),
        "roadmap_candidate": FLAGGED_FOR_V445[0]["flag"],
    },
    {
        "method": "Worst-family group DRO energy training",
        "track": "worst_family_robust_optimization",
        "source_ids": ["1911.08731", "2012.07421"],
        "maps_to_stages": ["S4"],
        "s4_graft": (
            "Treat source game families as groups and optimize the S4 verifier "
            "for worst-family loss with explicit L2 or early-stopping "
            "regularization before the held-out family is touched."
        ),
        "heldout_family_test": (
            "Report worst-source-family and held-out-family energy separation; "
            "the method passes only if the worst group improves without collapsing "
            "the held-out family."
        ),
        "takes_over_from_current_stack": (
            "Takes over average-loss energy training when rare mechanics families "
            "are swamped by easier frequent families."
        ),
        "fails_when": (
            "Family labels are noisy, group sizes are too small for stable worst "
            "losses, or regularization is too weak to generalize beyond training groups."
        ),
        "roadmap_candidate": FLAGGED_FOR_V445[1]["flag"],
    },
    {
        "method": "Risk extrapolation across source families",
        "track": "risk_extrapolation",
        "source_ids": ["2003.00688", "2007.01434"],
        "maps_to_stages": ["S4"],
        "s4_graft": (
            "Add a V-REx-style penalty that reduces variance in verifier-energy "
            "risk across source families, with DomainBed-style model selection "
            "that never tunes on the held-out family."
        ),
        "heldout_family_test": (
            "Score the held-out family after source-only model selection and "
            "compare against ERM, anchored fine-tuning, and group DRO controls."
        ),
        "takes_over_from_current_stack": (
            "Takes over source-family ERM when S4 needs a smoother transfer "
            "surface across mechanics families."
        ),
        "fails_when": (
            "Source-family variation does not cover the held-out shift, the "
            "risk-equality penalty suppresses genuinely useful family-specific "
            "signals, or model selection overfits source domains."
        ),
        "roadmap_candidate": FLAGGED_FOR_V445[2]["flag"],
    },
    {
        "method": "Transferable-reward prefix continuation stress",
        "track": "transferable_reward_stress",
        "source_ids": ["2602.08489"],
        "maps_to_stages": ["S4"],
        "s4_graft": (
            "Stress the S4 energy by asking whether partial plans or reasoning "
            "prefixes generated from one source family help a separate policy "
            "continue in another family, rather than only judging final answers."
        ),
        "heldout_family_test": (
            "A held-out family earns credit only when source-family prefixes "
            "improve continuation quality under the target-family verifier without "
            "manual target labels."
        ),
        "takes_over_from_current_stack": (
            "Takes over final-outcome-only verifier checks that miss brittle "
            "reasoning traces which cannot transfer across models or families."
        ),
        "fails_when": (
            "Families do not share transferable substructure, prefix swaps create "
            "invalid action contexts, or the continuation model learns a style cue."
        ),
        "roadmap_candidate": FLAGGED_FOR_V445[2]["flag"],
    },
]

DEFAULT_FRESH_SWEEP = {
    "filtered_track": (
        "cross-family transfer of learned verifier/energy under held-out "
        "distribution shift"
    ),
    "cluster_ids": [0, 1, 6],
    "semantic_scholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "semantic_scholar_result": "HTTP 429 on all three focused queries; no S2-only source promoted.",
    "webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
}
DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "claude_md_consulted": True,
    "research_studying_present": True,
    "research_references_present": True,
    "sweep_clusters_used": True,
    "sweep_cluster_ids": [0, 1, 6],
    "sweep_cluster_urls": [CLUSTER_0_URL, CLUSTER_1_URL, CLUSTER_6_URL],
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
    "model_load": False,
    "training_launched": False,
    "leaderboard_submission": False,
    "solve_claim_made": False,
    "ops_docs_modified": False,
}
DEFAULT_S4_CONTEXT = {
    "roadmap_target": ".445",
    "s4_cross_family_transfer_required": True,
    "gap_context": (
        "S4 stresses the energy on cross-family transfer because the .393 win "
        "and GAP-4 path did not survive the transfer setting."
    ),
    "evaluation_constraint": (
        "Use leave-one-family-out held-out evaluation and worst-family metrics; "
        "a pooled average or source-family fit cannot authorize S4."
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_set_checksum(
    citations: JsonMap,
    methods: Sequence[JsonMap],
    flags: Sequence[JsonMap],
    s4_context: JsonMap,
) -> str:
    payload = json.dumps(
        {
            "citations": citations,
            "flags": list(flags),
            "methods": list(methods),
            "s4_context": s4_context,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


REPRODUCIBILITY_CHECKSUM = source_set_checksum(
    CITATIONS,
    DEFAULT_METHODS_MAPPED,
    FLAGGED_FOR_V445,
    DEFAULT_S4_CONTEXT,
)


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    fresh_sweep: JsonMap = DEFAULT_FRESH_SWEEP,
    flagged_for_v445: Sequence[JsonMap] = FLAGGED_FOR_V445,
    s4_context: JsonMap = DEFAULT_S4_CONTEXT,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "arxiv_ids_cited": sorted(citations),
        "flagged_for_v445": [dict(flag) for flag in flagged_for_v445],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "citations": {source_id: dict(citation) for source_id, citation in citations.items()},
        "fresh_sweep": dict(fresh_sweep),
        "s4_context": dict(s4_context),
        "note_path": NOTE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_S,
        "reproducibility_checksum": source_set_checksum(
            citations,
            methods_mapped,
            flagged_for_v445,
            s4_context,
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
    _require(artifact["note_path"] == NOTE_PATH, "note_path must point at the Exp 4828 note")
    _require(artifact["random_seed"] == RANDOM_SEED, "random_seed must be the experiment id")
    _require(artifact["duration_s"] == DURATION_S, "duration_s must preserve the 0.0001s floor")
    _validate_citations(artifact["citations"], artifact["arxiv_ids_cited"])
    _validate_methods(artifact["methods_mapped"], artifact["arxiv_ids_cited"])
    _validate_flags(artifact["flagged_for_v445"], artifact["arxiv_ids_cited"])
    _validate_preconditions(artifact["preconditions_checked"])
    _validate_fresh_sweep(artifact["fresh_sweep"])
    _validate_s4_context(artifact["s4_context"])
    _require(
        artifact["reproducibility_checksum"]
        == source_set_checksum(
            artifact["citations"],
            artifact["methods_mapped"],
            artifact["flagged_for_v445"],
            artifact["s4_context"],
        ),
        "reproducibility checksum must hash citations, methods, flags, and S4 context",
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
        _require(maps_to_stages == ["S4"], "method maps_to_stages must map onto S4")
        _require(bool(method["s4_graft"]), "each method needs an S4 graft")
        _require("family" in str(method["heldout_family_test"]), "each method needs held-out family test")
        _require(bool(method["takes_over_from_current_stack"]), "each method needs takes_over_from_current_stack")
        _require(bool(method["fails_when"]), "each method needs fails_when")
        _require(bool(method["roadmap_candidate"]), "each method needs a roadmap candidate")
        tracks.add(str(method["track"]))
    _require(REQUIRED_TRACKS == tracks, "methods_mapped missing required S4 transfer tracks")


def _validate_flags(flags: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(flags, Sequence) and not isinstance(flags, str | bytes) and bool(flags), "flagged_for_v445 required")
    cited = set(arxiv_ids_cited)
    for flag in flags:
        _require(isinstance(flag, Mapping), "each flagged_for_v445 entry must be a mapping")
        _require("candidate" in flag and "flag" in flag, "each flagged_for_v445 entry needs candidate and flag")
        _require("flagged_for_v444" not in json.dumps(flag, sort_keys=True), "flagged_for_v445 must not carry stale .444 flags")
        _require("flagged_for_v445" in str(flag["flag"]), "flagged_for_v445 entries must carry the .445 flag")
        _require(set(flag.get("source_ids", [])).issubset(cited), "flagged_for_v445 source_ids must be verified")


def _validate_preconditions(preconditions: object) -> None:
    _require(isinstance(preconditions, Mapping), "preconditions_checked must be a mapping")
    _require(set(preconditions) == REQUIRED_PRECONDITION_FIELDS, "preconditions_checked must match schema")
    _require(preconditions["research_studying_present"] is True, "research-studying precondition must pass")
    _require(preconditions["research_references_present"] is True, "research-references precondition must pass")
    _require(preconditions["sweep_clusters_used"] is True, "sweep_clusters must be used")
    _require(preconditions["sweep_cluster_ids"] == [0, 1, 6], "sweep cluster IDs must be [0, 1, 6]")
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
    _require(fresh_sweep["cluster_ids"] == [0, 1, 6], "fresh_sweep must record clusters 0, 1, and 6")
    sources = fresh_sweep["webfetch_top_sources"]
    _require(
        isinstance(sources, Sequence) and not isinstance(sources, str | bytes) and 5 <= len(sources) <= 8,
        "fresh_sweep must record top five to eight WebFetch sources",
    )
    _require(list(sources) == WEBSEARCH_WEBFETCH_TOP_SOURCES, "fresh_sweep sources must match verified source set")


def _validate_s4_context(context: object) -> None:
    _require(isinstance(context, Mapping), "s4_context must be a mapping")
    _require(context.get("roadmap_target") == ".445", "s4_context must target the .445 roadmap")
    _require(context.get("s4_cross_family_transfer_required") is True, "S4 cross-family transfer must be required")
    _require("cross-family transfer" in str(context.get("gap_context", "")), "s4_context must name transfer gap")
    _require(
        "pooled average" in str(context.get("evaluation_constraint", "")),
        "S4 evaluation constraint must reject pooled average",
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
            f"maps to {', '.join(method['maps_to_stages'])}; {method['s4_graft']} "
            f"Held-out family test: {method['heldout_family_test']} "
            f"Takes over: {method['takes_over_from_current_stack']} "
            f"Fails when: {method['fails_when']}"
        )
        for method in result["methods_mapped"]
    )
    flag_lines = "\n".join(flag["flag"] for flag in result["flagged_for_v445"])
    return f"""{STUDYING_SECTION_START}
## 2026-06-26 Exp 4828 - .445 cross-family transfer SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4828_sota_ingestion_cross_family_transfer.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted verifier/reward, EBM, and
neural-guided search cluster URLs. `scripts/sweep_semscholar.py` was run on
three focused cross-family transfer queries and returned HTTP 429 for all of
them, so no S2-only source was promoted. Low-concurrency WebSearch/WebFetch
plus direct arXiv HTTP checks verified the top eight papers listed below.
`/deep-research` was not invoked. No model load, training, leaderboard
submission, or solve claim was made; this is a no solve claim ingestion note.

**Verified source set:**
{citation_lines}

**SOTA -> S4 cross-family transfer mapping:**
{method_lines}

{flag_lines}

**Bottom line for .445:** prioritize the Anchor plus leave-one-family transfer
gate because it directly attacks the .393/GAP-4 failure mode: a verifier energy
can look good in-distribution while failing on the family that matters. Pair it
with worst-family Group DRO as the robust-training control, then use REx and
RLTR-style transferable-reward stress as the falsifier when source families
still look too easy.
{STUDYING_SECTION_END}"""


def update_research_studying_text(text: str, artifact: JsonMap | None = None) -> str:
    section = build_research_studying_section(artifact)
    start = text.find(STUDYING_SECTION_START)
    if start >= 0:
        end = text.find(STUDYING_SECTION_END, start)
        _require(end >= 0, "research-studying Exp 4828 section missing end marker")
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
    _require(start >= 0 and end >= 0, "research-studying missing Exp 4828 section markers")
    section = text[start : end + len(STUDYING_SECTION_END)]
    for phrase in (
        "SOTA -> S4 cross-family transfer mapping",
        "flagged_for_v445",
        "no solve claim",
        "held-out family",
        "Group DRO",
        "REx",
        "DomainBed",
        "WILDS",
        "RewardBench",
        "Anchor",
        "RLTR",
    ):
        _require(phrase in section, f"research-studying section missing required phrase: {phrase}")
    missing_citations = sorted(citation for citation in NOTE_REQUIRED_SOURCE_CITATIONS if citation not in section)
    _require(not missing_citations, f"research-studying section missing citations: {missing_citations}")
    for method in result["methods_mapped"]:
        _require(method["method"] in section, f"research-studying section missing method: {method['method']}")
    for flag in result["flagged_for_v445"]:
        _require(flag["flag"] in section, "research-studying section missing flagged_for_v445 text")


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
    root = Path(os.environ.get("CARNOT_EXP4828_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        studying_path=root / STUDYING_RELATIVE_PATH,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
