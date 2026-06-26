"""Exp 4778 SOTA ingestion for S0' structural-energy planning.

Spec refs: REQ-ARC-WMTE-4778, SCENARIO-ARC-WMTE-4778,
SCENARIO-ARC-WMTE-4778-LEAK-ROBUST.

This module does not train a model or claim an ARC solve. It turns a verified
literature slice into a deterministic planning artifact so the next milestone
can pick up the strongest methods without losing the S0/S0' leak controls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4778_sota_ingestion_structural_energy.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
NOTE_PATH = "research-studying.md#exp-4778-sota-ingestion-structural-energy"
S0PRIME_SOURCE_RELATIVE_PATH = "results/experiment_4771_structural_energy_s0prime_origin_matched.json"
RANDOM_SEED = 4778
DURATION_S = 0.0001
HONEST_VERDICT = "success_sota_ingestion_structural_energy_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
STUDYING_SECTION_START = "<!-- EXP4778-SOTA-INGESTION-STRUCTURAL-ENERGY-START -->"
STUDYING_SECTION_END = "<!-- EXP4778-SOTA-INGESTION-STRUCTURAL-ENERGY-END -->"
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
            "the strongest 3-5 methods mapped onto S1-S4 + leak-robust eval, "
            "each with a real arXiv ID."
        )
    },
    "arxiv_ids_cited": {
        "principle": (
            "every method claim must cite a verifiable arXiv ID -- an ingestion "
            "with no citations is fabrication."
        )
    },
    "flagged_for_v440": {
        "principle": "the strongest method(s) flagged so the .440 planner reads the mapping."
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
    "s0prime_context": {
        "principle": "imports the S0' close-state so .440 inherits the leak controls."
    },
    "leak_robust_evaluation_note": {
        "principle": "specific origin/provenance, shortcut, and invariance controls for the energy probe."
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
        "principle": "content hash of citations, method map, flags, S0' context, and leak note."
    },
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "methods_mapped",
    "arxiv_ids_cited",
    "flagged_for_v440",
    "inference_substrate",
    "preconditions_checked",
    "citations",
    "fresh_sweep",
    "s0prime_context",
    "leak_robust_evaluation_note",
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
        "leak_robust_eval_role",
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
REQUIRED_LEAK_NOTE_FIELDS = frozenset({"summary", "source_ids", "required_controls", "roadmap_gate"})
REQUIRED_SOURCE_IDS = frozenset(
    {
        "1907.02893",
        "1911.12247",
        "2006.15055",
        "2301.08243",
        "2505.10819",
        "2505.13910",
        "2510.04542",
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
    "object-centric relational world model transition energy shortcut robustness",
    "contrastive transition energy JEPA executable world model provenance leak",
    "confound shortcut robust representation evaluation invariant counterfactual probes",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2006.15055",
    "https://arxiv.org/abs/1911.12247",
    "https://arxiv.org/abs/2505.10819",
    "https://arxiv.org/abs/2301.08243",
    "https://arxiv.org/abs/2605.05138",
    "https://arxiv.org/abs/2510.04542",
    "https://arxiv.org/abs/2505.13910",
    "https://arxiv.org/abs/1907.02893",
]

CITATIONS = {
    "1907.02893": {
        "title": "Invariant Risk Minimization",
        "url": "https://arxiv.org/abs/1907.02893",
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
    "2301.08243": {
        "title": "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture",
        "url": "https://arxiv.org/abs/2301.08243",
        "http_status": 200,
    },
    "2505.10819": {
        "title": "PoE-World: Compositional World Modeling with Products of Programmatic Experts",
        "url": "https://arxiv.org/abs/2505.10819",
        "http_status": 200,
    },
    "2505.13910": {
        "title": "ShortcutProbe: Probing Prediction Shortcuts for Learning Robust Models",
        "url": "https://arxiv.org/abs/2505.13910",
        "http_status": 200,
    },
    "2510.04542": {
        "title": "Code World Models for General Game Playing",
        "url": "https://arxiv.org/abs/2510.04542",
        "http_status": 200,
    },
    "2605.05138": {
        "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
        "url": "https://arxiv.org/abs/2605.05138",
        "http_status": 200,
    },
}

FLAGGED_FOR_V440 = [
    {
        "candidate": "slot_relational_contrastive_energy_s0prime_guarded",
        "flag": (
            "flagged_for_v440: slot_relational_contrastive_energy_s0prime_guarded "
            "(arXiv:2006.15055 + arXiv:1911.12247 + arXiv:2505.10819 + "
            "arXiv:2505.13910 + arXiv:1907.02893)"
        ),
        "source_ids": ["2006.15055", "1911.12247", "2505.10819", "2505.13910", "1907.02893"],
        "maps_to_stages": ["S1", "S2", "S4"],
    },
    {
        "candidate": "poe_code_world_model_trust_gate_after_s0prime",
        "flag": (
            "flagged_for_v440: poe_code_world_model_trust_gate_after_s0prime "
            "(arXiv:2505.10819 + arXiv:2605.05138 + arXiv:2510.04542)"
        ),
        "source_ids": ["2505.10819", "2605.05138", "2510.04542"],
        "maps_to_stages": ["S2", "S3", "S4"],
    },
]

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Slot-relational contrastive transition energy",
        "track": "slot_relational_contrastive_transition_energy",
        "source_ids": ["2006.15055", "1911.12247", "2505.10819"],
        "maps_to_stages": ["S1", "S2"],
        "maps_to_current_stack": (
            "Use Slot Attention object bindings, C-SWM relational transitions, and "
            "PoE-World factorization to turn S0' object_relational/frame_delta "
            "features into E(s,a,s_hat) over near-miss induced predictions."
        ),
        "takes_over_from_current_stack": (
            "Takes over the S0' scalar structural probe by making the score an "
            "object-factorized transition energy that can rank off-path candidate "
            "next states and induced engines."
        ),
        "leak_robust_eval_role": (
            "Run only on origin-matched induced rows, then require origin/provenance "
            "probe failure and shortcut probes before accepting the energy as "
            "oracle-distinct."
        ),
        "fails_when": (
            "Slot binding drifts across frames, PoE factors memorize source "
            "provenance, or contrastive negatives are too easy to separate without "
            "learning transition mechanics."
        ),
        "roadmap_candidate": FLAGGED_FOR_V440[0]["flag"],
    },
    {
        "method": "Executable PoE/code world-model trust energy",
        "track": "poe_code_world_model_trust_energy",
        "source_ids": ["2505.10819", "2605.05138", "2510.04542"],
        "maps_to_stages": ["S2", "S3", "S4"],
        "maps_to_current_stack": (
            "Use product-of-programmatic-experts and generated code world models as "
            "candidate transition engines, then score their off-path rollouts with "
            "the S0'-guarded structural energy before a planner trusts them."
        ),
        "takes_over_from_current_stack": (
            "Takes over binary executable-model accept/reject checks by ranking "
            "world-model factors and code engines on transition consistency where "
            "the environment win-check is unavailable."
        ),
        "leak_robust_eval_role": (
            "Each executable factor must be evaluated on held-out transitions with "
            "no terminal win oracle, no source-origin token, and a provenance probe "
            "that stays at chance."
        ),
        "fails_when": (
            "The generated program overfits public prefixes, hidden-state inference "
            "is wrong, or the trust energy leaks private solution facts through a "
            "verifier shortcut."
        ),
        "roadmap_candidate": FLAGGED_FOR_V440[1]["flag"],
    },
    {
        "method": "JEPA latent residual energy for transfer survival",
        "track": "jepa_latent_residual_transfer_energy",
        "source_ids": ["2301.08243", "1911.12247"],
        "maps_to_stages": ["S1", "S4"],
        "maps_to_current_stack": (
            "Predict a target latent representation from context/action/object "
            "structure and use the residual as a transfer-stress energy across "
            "games and families."
        ),
        "takes_over_from_current_stack": (
            "Takes over raw frame-marginal controls by asking whether the structural "
            "signal survives in a predictive representation that is not allowed to "
            "encode source provenance."
        ),
        "leak_robust_eval_role": (
            "Pair leave-one-game and leave-one-family folds with shuffled-label "
            "controls so a JEPA residual cannot pass by memorizing nuisance origin."
        ),
        "fails_when": (
            "The latent discards exact grid consequences, learns a value head rather "
            "than transition mechanics, or transfers only within one ARC family."
        ),
        "roadmap_candidate": "support_for_v440: jepa_latent_residual_transfer_stress",
    },
    {
        "method": "Shortcut/invariance leak-robust energy evaluation gate",
        "track": "shortcut_invariance_leak_evaluation_gate",
        "source_ids": ["2505.13910", "1907.02893"],
        "maps_to_stages": ["S1", "S2", "S3", "S4"],
        "maps_to_current_stack": (
            "Add a post-hoc shortcut probe plus environment-invariance checks around "
            "every S1-S4 structural energy claim, treating origin, game family, and "
            "candidate-generator provenance as explicit environments."
        ),
        "takes_over_from_current_stack": (
            "Takes over the ad hoc S0 origin-probe warning by making leak detection "
            "a required gate with provenance probes, shuffled-label controls, and "
            "counterfactual/invariance stress tests."
        ),
        "leak_robust_eval_role": (
            "This is the .440 acceptance harness: S1-S4 methods must pass shortcut "
            "probe, origin/provenance chance probe, and invariance under "
            "counterfactual origin swaps before roadmap promotion."
        ),
        "fails_when": (
            "Shortcut probes have no hard negatives, environments are too correlated "
            "with labels for invariance to identify the causal feature, or the probe "
            "is tuned after seeing the target fold."
        ),
        "roadmap_candidate": "flagged_for_v440: leak_robust_eval_gate_for_all_structural_energy_continuations",
    },
]

DEFAULT_LEAK_ROBUST_EVALUATION_NOTE = {
    "summary": (
        "S0' reopens S1 only if .440 treats origin/provenance leakage as a first-class "
        "failure mode: every energy result needs origin/provenance controls, "
        "shortcut probes, and counterfactual/invariance stress tests."
    ),
    "source_ids": ["2505.13910", "1907.02893"],
    "required_controls": [
        "origin-matched induced-only rows for positive and negative transition candidates",
        "chance-level origin/provenance probe before any oracle-distinct continuation claim",
        "shuffled-label or ShortcutProbe-style latent shortcut control on identical folds",
        "counterfactual/invariance probes over game family, origin, and candidate-generator environments",
    ],
    "roadmap_gate": "flagged_for_v440: leak_robust_eval_gate_for_all_structural_energy_continuations",
}

DEFAULT_FRESH_SWEEP = {
    "filtered_track": "S0' structural energy + S1-S4 world-model build + leak-robust evaluation",
    "cluster_ids": [1, 5, 6],
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
    "sweep_cluster_ids": [1, 5, 6],
    "sweep_cluster_urls": [CLUSTER_1_URL, CLUSTER_5_URL, CLUSTER_6_URL],
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
DEFAULT_S0PRIME_CONTEXT = {
    "source_artifact": S0PRIME_SOURCE_RELATIVE_PATH,
    "stage": "S0'",
    "imported_honest_verdict": "success_structural_energy_s0prime_reopens_s1",
    "s0prime_gate_passed": True,
    "loo_auroc_structural": 0.7386642861889572,
    "loo_auroc_ci95": [0.636412794237013, 0.8332008450933205],
    "origin_probe_auroc": 0.5,
    "shuffled_label_control_auroc": 0.5033091959271814,
    "structural_minus_marginal_delta_ci95": [0.1271826145701076, 0.33248035484558097],
    "flagged_adversarial": True,
    "planning_constraint": (
        "S0' is the headline structural-energy signal, but the adversarial warning "
        "means .440 must promote only methods with explicit leak-robust evaluation."
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_set_checksum(
    citations: JsonMap,
    methods: Sequence[JsonMap],
    flags: Sequence[JsonMap],
    s0prime_context: JsonMap,
    leak_note: JsonMap,
) -> str:
    payload = json.dumps(
        {
            "citations": citations,
            "flags": list(flags),
            "leak_note": leak_note,
            "methods": list(methods),
            "s0prime_context": s0prime_context,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


REPRODUCIBILITY_CHECKSUM = source_set_checksum(
    CITATIONS,
    DEFAULT_METHODS_MAPPED,
    FLAGGED_FOR_V440,
    DEFAULT_S0PRIME_CONTEXT,
    DEFAULT_LEAK_ROBUST_EVALUATION_NOTE,
)


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    fresh_sweep: JsonMap = DEFAULT_FRESH_SWEEP,
    flagged_for_v440: Sequence[JsonMap] = FLAGGED_FOR_V440,
    s0prime_context: JsonMap = DEFAULT_S0PRIME_CONTEXT,
    leak_robust_evaluation_note: JsonMap = DEFAULT_LEAK_ROBUST_EVALUATION_NOTE,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "arxiv_ids_cited": sorted(citations),
        "flagged_for_v440": [dict(flag) for flag in flagged_for_v440],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "citations": {source_id: dict(citation) for source_id, citation in citations.items()},
        "fresh_sweep": dict(fresh_sweep),
        "s0prime_context": dict(s0prime_context),
        "leak_robust_evaluation_note": dict(leak_robust_evaluation_note),
        "note_path": NOTE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_S,
        "reproducibility_checksum": source_set_checksum(
            citations,
            methods_mapped,
            flagged_for_v440,
            s0prime_context,
            leak_robust_evaluation_note,
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
    _require(artifact["note_path"] == NOTE_PATH, "note_path must point at the Exp 4778 note")
    _require(artifact["random_seed"] == RANDOM_SEED, "random_seed must be the experiment id")
    _require(artifact["duration_s"] == DURATION_S, "duration_s must preserve the 0.0001s floor")
    _validate_citations(artifact["citations"], artifact["arxiv_ids_cited"])
    _validate_methods(artifact["methods_mapped"], artifact["arxiv_ids_cited"])
    _validate_flags(artifact["flagged_for_v440"], artifact["arxiv_ids_cited"])
    _validate_preconditions(artifact["preconditions_checked"])
    _validate_fresh_sweep(artifact["fresh_sweep"])
    _validate_s0prime_context(artifact["s0prime_context"])
    _validate_leak_note(artifact["leak_robust_evaluation_note"], artifact["arxiv_ids_cited"])
    _require(
        artifact["reproducibility_checksum"]
        == source_set_checksum(
            artifact["citations"],
            artifact["methods_mapped"],
            artifact["flagged_for_v440"],
            artifact["s0prime_context"],
            artifact["leak_robust_evaluation_note"],
        ),
        "reproducibility checksum must hash citations, methods, flags, S0' context, and leak note",
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
            "method maps_to_stages must stay within S1-S4",
        )
        _require(bool(method["maps_to_current_stack"]), "each method needs maps_to_current_stack")
        _require(bool(method["takes_over_from_current_stack"]), "each method needs takes_over_from_current_stack")
        _require(bool(method["leak_robust_eval_role"]), "each method needs leak_robust_eval_role")
        _require(bool(method["fails_when"]), "each method needs fails_when")
        _require(bool(method["roadmap_candidate"]), "each method needs a roadmap candidate")
        stages.update(str(stage) for stage in maps_to_stages)
        tracks.add(str(method["track"]))
    _require(ALLOWED_STAGES.issubset(stages), "methods_mapped must cover S1-S4")
    _require("shortcut_invariance_leak_evaluation_gate" in tracks, "methods_mapped must include leak-robust eval gate")


def _validate_flags(flags: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(flags, Sequence) and not isinstance(flags, str | bytes) and bool(flags), "flagged_for_v440 required")
    cited = set(arxiv_ids_cited)
    for flag in flags:
        _require(isinstance(flag, Mapping), "each flagged_for_v440 entry must be a mapping")
        _require("candidate" in flag and "flag" in flag, "each flagged_for_v440 entry needs candidate and flag")
        _require("flagged_for_v439" not in json.dumps(flag, sort_keys=True), "flagged_for_v440 must not carry stale .439 flags")
        _require("flagged_for_v440" in str(flag["flag"]), "flagged_for_v440 entries must carry the .440 flag")
        _require(set(flag.get("source_ids", [])).issubset(cited), "flagged_for_v440 source_ids must be verified")


def _validate_preconditions(preconditions: object) -> None:
    _require(isinstance(preconditions, Mapping), "preconditions_checked must be a mapping")
    _require(set(preconditions) == REQUIRED_PRECONDITION_FIELDS, "preconditions_checked must match schema")
    _require(preconditions["research_studying_present"] is True, "research-studying precondition must pass")
    _require(preconditions["research_references_present"] is True, "research-references precondition must pass")
    _require(preconditions["sweep_clusters_used"] is True, "sweep_clusters must be used")
    _require(preconditions["sweep_cluster_ids"] == [1, 5, 6], "sweep cluster IDs must be [1, 5, 6]")
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
    _require(fresh_sweep["cluster_ids"] == [1, 5, 6], "fresh_sweep must record clusters 1, 5, and 6")
    sources = fresh_sweep["webfetch_top_sources"]
    _require(
        isinstance(sources, Sequence) and not isinstance(sources, str | bytes) and 5 <= len(sources) <= 8,
        "fresh_sweep must record top five to eight WebFetch sources",
    )
    _require(list(sources) == WEBSEARCH_WEBFETCH_TOP_SOURCES, "fresh_sweep sources must match verified source set")


def _validate_s0prime_context(s0prime_context: object) -> None:
    _require(isinstance(s0prime_context, Mapping), "s0prime_context must be a mapping")
    _require(s0prime_context.get("source_artifact") == S0PRIME_SOURCE_RELATIVE_PATH, "s0prime_context must cite S0'")
    _require(s0prime_context.get("stage") == "S0'", "s0prime_context stage must be S0'")
    _require(s0prime_context.get("s0prime_gate_passed") is True, "S0' gate must be imported as passed")
    _require(float(s0prime_context.get("origin_probe_auroc", 1.0)) <= 0.6, "origin probe must be leak-clean")
    _require(
        float(s0prime_context.get("shuffled_label_control_auroc", 1.0)) <= 0.55,
        "shuffled label control must be leak-clean",
    )
    _require("leak-robust" in str(s0prime_context.get("planning_constraint", "")), "S0' context must carry leak-robust constraint")


def _validate_leak_note(leak_note: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(leak_note, Mapping), "leak_robust_evaluation_note must be a mapping")
    _require(set(leak_note) == REQUIRED_LEAK_NOTE_FIELDS, "leak note must match schema")
    source_ids = leak_note["source_ids"]
    controls = leak_note["required_controls"]
    _require(
        isinstance(source_ids, Sequence)
        and not isinstance(source_ids, str | bytes)
        and bool(source_ids)
        and set(source_ids).issubset(set(arxiv_ids_cited)),
        "leak note source_ids must cite verified arXiv IDs",
    )
    _require(
        isinstance(controls, Sequence) and not isinstance(controls, str | bytes) and len(controls) >= 4,
        "leak note must define at least four required controls",
    )
    summary = str(leak_note["summary"])
    _require("origin/provenance" in summary, "leak note must mention origin/provenance controls")
    _require("counterfactual/invariance" in summary, "leak note must mention counterfactual/invariance probes")
    _require("flagged_for_v440" in str(leak_note["roadmap_gate"]), "leak note roadmap gate must be flagged_for_v440")


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
            f"{method['takes_over_from_current_stack']} Leak eval: {method['leak_robust_eval_role']} "
            f"Fails when: {method['fails_when']}"
        )
        for method in result["methods_mapped"]
    )
    control_lines = "\n".join(f"- {control}" for control in result["leak_robust_evaluation_note"]["required_controls"])
    flag_lines = "\n".join(flag["flag"] for flag in result["flagged_for_v440"])
    return f"""{STUDYING_SECTION_START}
## 2026-06-26 Exp 4778 - .440 structural-energy SOTA ingestion after S0' - INGESTED

**Status:** INGESTED into `results/experiment_4778_sota_ingestion_structural_energy.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted the EBM, action/effect, and
neural-guided/world-model cluster URLs. `scripts/sweep_semscholar.py` was run
on three focused queries and returned HTTP 429, so no S2-only source was
promoted. Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks
verified the top eight papers listed below. `/deep-research` was not invoked.
No model load, training, leaderboard submission, or solve claim was made; this
is a no solve claim ingestion note.

**S0' context imported:** `{S0PRIME_SOURCE_RELATIVE_PATH}` reports
`{result["s0prime_context"]["imported_honest_verdict"]}` with
`origin_probe_auroc={result["s0prime_context"]["origin_probe_auroc"]}` and
`shuffled_label_control_auroc={result["s0prime_context"]["shuffled_label_control_auroc"]}`.
Because the artifact is also adversarial-flagged, .440 should treat leak-robust
evaluation as the gate for every S1-S4 continuation.

**Verified source set:**
{citation_lines}

**SOTA -> S1-S4 structural-energy mapping:**
{method_lines}

**Leak-robust evaluation note:** {result["leak_robust_evaluation_note"]["summary"]}
Use ShortcutProbe and IRM as the explicit shortcut/invariance evaluation
templates.
{control_lines}

{flag_lines}

**Bottom line for .440:** prioritize the Slot Attention + C-SWM
slot-relational contrastive energy rerun under the explicit leak gate, then
connect PoE/code world-model trust only after origin/provenance and
shortcut/invariance controls stay clean.
{STUDYING_SECTION_END}"""


def update_research_studying_text(text: str, artifact: JsonMap | None = None) -> str:
    section = build_research_studying_section(artifact)
    start = text.find(STUDYING_SECTION_START)
    if start >= 0:
        end = text.find(STUDYING_SECTION_END, start)
        _require(end >= 0, "research-studying Exp 4778 section missing end marker")
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
    _require(start >= 0 and end >= 0, "research-studying missing Exp 4778 section markers")
    section = text[start : end + len(STUDYING_SECTION_END)]
    for phrase in (
        "SOTA -> S1-S4 structural-energy mapping",
        "Leak-robust evaluation note",
        "flagged_for_v440",
        "no solve claim",
        "Slot Attention",
        "C-SWM",
        "PoE",
        "ShortcutProbe",
        "IRM",
        "origin/provenance",
        "counterfactual/invariance",
    ):
        _require(phrase in section, f"research-studying section missing required phrase: {phrase}")
    missing_citations = sorted(citation for citation in NOTE_REQUIRED_SOURCE_CITATIONS if citation not in section)
    _require(not missing_citations, f"research-studying section missing citations: {missing_citations}")
    for method in result["methods_mapped"]:
        _require(method["method"] in section, f"research-studying section missing method: {method['method']}")
    for flag in result["flagged_for_v440"]:
        _require(flag["flag"] in section, "research-studying section missing flagged_for_v440 text")


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
    root = Path(os.environ.get("CARNOT_EXP4778_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        studying_path=root / STUDYING_RELATIVE_PATH,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
