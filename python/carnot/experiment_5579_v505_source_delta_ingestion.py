"""Exp5579: ingest the V505 execution-time source delta.

Spec refs: REQ-REPORT-5579, SCENARIO-REPORT-5579,
SCENARIO-REPORT-5579-NOOP, SCENARIO-REPORT-5579-FIELD-PRINCIPLES.

The live search and source reading happen before this module is run. This file
keeps the receipt reproducible: it records which source surfaces were checked,
deduplicates accepted candidates against the full reference history, appends a
short execution refresh only when a source is both new and locally actionable,
and writes the JSON artifact. The split is intentional because public search
rankings, paper mirrors, and citation APIs drift after the conductor has moved
on, while the research record needs a stable audit trail.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5579_v505_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")

EXPERIMENT = "experiment_5579_v505_source_delta_ingestion"
EXPERIMENT_ID = "exp5579-v505-source-delta-ingestion"
MILESTONE = "2026.07.505"
RUN_DATE = "20260714"
SEARCH_CUTOFF = "2026-07-14"
SCHEMA = "carnot.experiment_5579.v505_source_delta_ingestion.v1"
RANDOM_SEED = 5579
INFERENCE_SUBSTRATE = "web_and_repository_source_synthesis"
TERMINAL_PREFIXES = ("complete:", "blocked:")

PLANNER_MARKER = "## V505 Planner Refresh - 20260711"
PLANNER_MARKER_COMPACT = PLANNER_MARKER.replace("-", "")
EXECUTION_REFRESH_HEADING = "## V505 Execution Refresh - 20260714"
EXECUTION_REFRESH_END = "<!-- V505-EXECUTION-REFRESH-20260714-END -->"

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "planner_marker_found",
    "search_cutoff",
    "sources_checked",
    "primary_sources_checked",
    "new_references_added",
    "duplicates_suppressed",
    "citation_trails_checked",
    "research_references_updated",
    "experiment_mappings",
    "watch_only_items",
    "closed_scopes_reopened",
    "inference_substrate",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "One-line annotations for every required headline and gate field.",
    "planner_marker_found": "execution search begins after planning",
    "sources_checked": "coverage is auditable",
    "search_cutoff": (
        "Execution-time cutoff prevents later source drift from being attributed to this receipt."
    ),
    "primary_sources_checked": (
        "Primary source and mirror checks distinguish accepted evidence from secondary summaries."
    ),
    "new_references_added": "count only non-duplicates",
    "duplicates_suppressed": "repeated ideas do not create work",
    "citation_trails_checked": "Citation routes are recorded without fabricating stronger dependencies.",
    "research_references_updated": (
        "Reference mutation is a bare boolean separate from source acceptance."
    ),
    "experiment_mappings": "sources need executable hooks",
    "watch_only_items": "unavailable systems support no claim",
    "closed_scopes_reopened": "retirement needs explicit authority",
    "inference_substrate": "provenance identifies evidence sources",
    "honest_verdict": "no-op and accepted-delta outcomes are both terminal",
}

SPEC_REFS = (
    "REQ-REPORT-5579",
    "SCENARIO-REPORT-5579",
    "SCENARIO-REPORT-5579-NOOP",
    "SCENARIO-REPORT-5579-FIELD-PRINCIPLES",
)

SOURCES_CHECKED: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "queries": [
            "EBM verification and reasoning",
            "neural CSPs and constraint satisfaction",
            "Ising ML, p-bit, and thermodynamic sampling",
            "hallucination mitigation and verifier reliability",
            "Kolmogorov-Arnold Networks and online learning",
            "constrained generation and grammar decoding",
            "accelerated sampling and hardware samplers",
            "continual constraint learning and agent memory",
        ],
        "status": "checked_direct_api_and_primary_pages",
    },
    {
        "surface": "OpenReview",
        "queries": [
            "Gram2Token",
            "EvoPolicyGym",
            "self-evolving agents",
            "constrained decoding",
            "LLM verifier",
        ],
        "status": "checked_duplicates_or_no_stronger_local_hook",
    },
    {
        "surface": "Semantic Scholar",
        "queries": ["arXiv:2507.02092 citations", "arXiv:2512.15605 citations"],
        "status": "checked_http_200_duplicate_citation_context",
    },
    {
        "surface": "Hugging Face Papers",
        "queries": ["2607.09072", "2607.09349", "2607.09493", "2607.08959"],
        "status": "checked",
    },
    {
        "surface": "GitHub",
        "queries": [
            "Agentic Proof Property-Templates",
            "Deceptive Grounding entity attribution",
            "Shared Selective Persistent Memory",
            "XVada grammar inference",
        ],
        "status": "checked_no_new_local_dependency_adopted",
    },
    {
        "surface": "Extropic writing",
        "queries": ["X0", "XTR-0", "Z1", "TSU", "thermodynamic computing"],
        "status": "checked_watch_only_no_authenticated_local_tsu",
    },
    {
        "surface": "Logical Intelligence public pages",
        "queries": ["Kona", "Aleph", "automatic formal verification", "Sudoku EBM"],
        "status": "checked_watch_only_proprietary_no_local_baseline",
    },
    {
        "surface": "local Carnot reference history",
        "queries": [
            "full research-references.md",
            "V504 planner and execution blocks",
            "V505 planner block",
            "ops/exclusion_manifest.yaml",
            "ops/known-issues.md",
        ],
        "status": "checked",
    },
)

PRIMARY_SOURCES_CHECKED: tuple[JsonDict, ...] = (
    {
        "source_id": "agentic_property_templates_2607_09072",
        "url": "https://arxiv.org/abs/2607.09072",
        "status": "accepted_non_duplicate_actionable_delta",
    },
    {
        "source_id": "agentic_property_templates_hf_2607_09072",
        "url": "https://huggingface.co/papers/2607.09072",
        "status": "secondary_mirror_checked_http_200",
    },
    {
        "source_id": "deceptive_grounding_2607_09349",
        "url": "https://arxiv.org/abs/2607.09349",
        "status": "accepted_non_duplicate_actionable_delta",
    },
    {
        "source_id": "deceptive_grounding_hf_2607_09349",
        "url": "https://huggingface.co/papers/2607.09349",
        "status": "secondary_mirror_checked_http_404_no_claim_from_mirror",
    },
    {
        "source_id": "shared_selective_memory_2607_09493",
        "url": "https://arxiv.org/abs/2607.09493",
        "status": "watch_only_no_public_local_workspace_artifact",
    },
    {
        "source_id": "shared_selective_memory_hf_2607_09493",
        "url": "https://huggingface.co/papers/2607.09493",
        "status": "secondary_mirror_checked_http_200",
    },
    {
        "source_id": "xvada_cfg_inference_2607_08959",
        "url": "https://arxiv.org/abs/2607.08959",
        "status": "watch_only_not_a_v505_parser_repair_dependency",
    },
    {
        "source_id": "mosaic_multiagent_planning_2607_09603",
        "url": "https://arxiv.org/abs/2607.09603",
        "status": "watch_only_embodied_multiagent_not_ordinary_arc_hook",
    },
    {
        "source_id": "vocabulary_verifier_gaps_2607_09560",
        "url": "https://arxiv.org/abs/2607.09560",
        "status": "watch_only_position_context_no_executable_delta",
    },
    {
        "source_id": "semantic_scholar_ebt_citations",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations",
        "status": "http_200_duplicate_context",
    },
    {
        "source_id": "semantic_scholar_arm_ebm_citations",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations",
        "status": "http_200_duplicate_context",
    },
    {
        "source_id": "openreview_gram2token",
        "url": "https://openreview.net/forum?id=h3K23f6tLU",
        "status": "duplicate_planner_context",
    },
    {
        "source_id": "extropic_writing",
        "url": "https://extropic.ai/writing",
        "status": "watch_only_no_authenticated_local_tsu",
    },
    {
        "source_id": "logical_intelligence_formal_verification",
        "url": "https://logicalintelligence.com/blog/automatic-formal-verification-for-code-generation",
        "status": "watch_only_proprietary_no_local_baseline",
    },
)

CITATION_TRAILS_CHECKED: tuple[JsonDict, ...] = (
    {
        "paper": "EBT",
        "paper_id": "2507.02092",
        "route": "Semantic Scholar public citation API",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/"
            "arXiv:2507.02092/citations?fields=title,year,url,externalIds&limit=20"
        ),
        "status": "http_200",
        "sample_visible_citing_papers": [
            "Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers",
            "LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models",
            "Revisiting Transformer Layer Parameterization Through Causal Energy Minimization",
        ],
        "promoted_delta": False,
        "note": "Visible citation samples were already indexed or weaker than the V505 exact-verifier and two-timescale hooks.",
    },
    {
        "paper": "ARM-EBM",
        "paper_id": "2512.15605",
        "route": "Semantic Scholar public citation API",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/"
            "arXiv:2512.15605/citations?fields=title,year,url,externalIds&limit=20"
        ),
        "status": "http_200",
        "sample_visible_citing_papers": [
            "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
            "LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models",
            "Ontology-Constrained Neural Reasoning in Enterprise Agentic Systems",
        ],
        "promoted_delta": False,
        "note": "No visible citation displaced exact ASP/FSM authority, PACE/EvoPolicyGym, or the accepted Exp5582 deltas.",
    },
)

AGENTIC_PROPERTY_TEMPLATES_FINDING: JsonDict = {
    "source_id": "agentic_property_templates_2607_09072",
    "title": "Agentic Proof and Property-Based Testing via Property-Templates in Data-Intensive Computing",
    "arxiv_id": "2607.09072",
    "url": "https://arxiv.org/abs/2607.09072",
    "secondary_url": "https://huggingface.co/papers/2607.09072",
    "classification": "accepted_actionable_delta",
    "why_actionable": (
        "The paper makes recurring property templates the unit of verifier work and pairs formal "
        "proof with executable property-based tests. Exp5582 already mines human-auditable exact "
        "predicate candidates from clean residuals; this source sharpens that work by requiring "
        "candidate predicates to remain template-shaped, deterministic, and counterexample-backed "
        "when the model of a predicate and the implementation diverge."
    ),
    "experiment_ids": ["exp5582-exact-counterexample-verifier-extension"],
    "lanes": ["exact verifier residual extension"],
    "dedupe_tokens": [
        "2607.09072",
        "Agentic Proof and Property-Based Testing",
        "Property-Templates in Data-Intensive Computing",
    ],
}

DECEPTIVE_GROUNDING_FINDING: JsonDict = {
    "source_id": "deceptive_grounding_2607_09349",
    "title": "Deceptive Grounding: Entity Attribution Failure in Clinical Retrieval-Augmented Generation",
    "arxiv_id": "2607.09349",
    "url": "https://arxiv.org/abs/2607.09349",
    "secondary_url": "https://huggingface.co/papers/2607.09349",
    "classification": "accepted_actionable_delta",
    "why_actionable": (
        "The paper isolates a verifier blind spot where every cited fact can be real while the "
        "fact is bound to the wrong entity. Exp5582 can use this as a bounded exact-residual "
        "stress category: if parser-grounded ASP/FSM rows show variable, entity, or claim-subject "
        "swaps, exact predicate candidates must check attribution to the intended entity. This "
        "does not add a clinical RAG benchmark or an external faithfulness scorer."
    ),
    "experiment_ids": ["exp5582-exact-counterexample-verifier-extension"],
    "lanes": ["exact verifier residual extension"],
    "dedupe_tokens": [
        "2607.09349",
        "Deceptive Grounding",
        "Entity Attribution Failure",
    ],
}

CANDIDATE_FINDINGS: tuple[JsonDict, ...] = (
    AGENTIC_PROPERTY_TEMPLATES_FINDING,
    DECEPTIVE_GROUNDING_FINDING,
)

DUPLICATE_SUPPRESSED_BASE: tuple[JsonDict, ...] = (
    {
        "source_id": "pace_2605_23019",
        "title": "PACE: Two-Timescale Self-Evolution for Small Language Model Agents",
        "url": "https://arxiv.org/abs/2605.23019",
        "reason": "Already accepted in the V505 planner block for Exp5584 two-timescale control.",
    },
    {
        "source_id": "evopolicygym_2607_02440",
        "title": "EvoPolicyGym: Evaluating Autonomous Policy Evolution in Interactive Environments",
        "url": "https://arxiv.org/abs/2607.02440",
        "reason": "Already accepted in the V505 planner block for fixed budgets and decision ledgers.",
    },
    {
        "source_id": "llm_as_verifier_2607_05391",
        "title": "LLM-as-a-Verifier: A General-Purpose Verification Framework",
        "url": "https://arxiv.org/abs/2607.05391",
        "reason": "Already indexed in V504 and V505 as verifier-arm context without LLM judge authority.",
    },
    {
        "source_id": "verification_horizon_2606_26300",
        "title": "The Verification Horizon: No Silver Bullet for Coding Agent Rewards",
        "url": "https://arxiv.org/abs/2606.26300",
        "reason": "Already indexed for verifier co-evolution stress and exact-residual triggers.",
    },
    {
        "source_id": "blind_curator_2607_07436",
        "title": "The Blind Curator",
        "url": "https://arxiv.org/abs/2607.07436",
        "reason": "Already accepted in the V504 execution refresh for false-pass skill-retirement audits.",
    },
    {
        "source_id": "asp_energised_2607_08136_classiclogic_2607_05185",
        "title": "ASP Energised and ClassicLogic",
        "url": "https://arxiv.org/abs/2607.08136",
        "reason": "Already accepted in V502/V503 source history for exact ASP/FSM fixtures.",
    },
    {
        "source_id": "gram2token_xgrammar_llguidance",
        "title": "Gram2Token, XGrammar, and llguidance grammar-decoding context",
        "url": "https://openreview.net/forum?id=h3K23f6tLU",
        "reason": "Already covered as grammar-table and structured-output context; V505 parser repair is deterministic and cached.",
    },
    {
        "source_id": "selfmem_continual_harness_agent_memory_prior",
        "title": "SelfMem, Continual Harness, A-MEM, and prior memory-agent lines",
        "url": "https://arxiv.org/abs/2607.03726",
        "reason": "Already present in V504/V505 memory-policy context; no raw-trace persistence claim is reopened.",
    },
    {
        "source_id": "ebt_arm_ebm_routes",
        "title": "EBT 2507.02092 and ARM-EBM 2512.15605 citation routes",
        "url": "https://arxiv.org/abs/2507.02092",
        "reason": "Citation samples were checked and did not create a stronger V505 dependency.",
    },
    {
        "source_id": "game_theory_hallucination_2607_08403",
        "title": "Game Theory Driven Multi-Agent Framework Mitigates Language Model Hallucination",
        "url": "https://arxiv.org/abs/2607.08403",
        "reason": "Already excluded in V502 history; it depends on data synthesis and model training, not V505 exact-validator hooks.",
    },
)

WATCH_ONLY_ITEMS: tuple[JsonDict, ...] = (
    {
        "source_id": "shared_selective_memory_2607_09493",
        "title": "Shared Selective Persistent Memory for Agentic LLM Systems",
        "url": "https://arxiv.org/abs/2607.09493",
        "classification": "watch_only_no_public_local_workspace_artifact",
        "reason": (
            "The task/data-schema/tool-config/output-constraint memory taxonomy is useful context for "
            "Exp5583-Exp5586, but the deployed workspace platform is not a local Carnot artifact and "
            "does not supersede the PACE/EvoPolicyGym gates."
        ),
    },
    {
        "source_id": "xvada_cfg_inference_2607_08959",
        "title": "Toward Inferring Accurate Context-free Grammars for Big Languages in a Black-box Setting",
        "url": "https://arxiv.org/abs/2607.08959",
        "classification": "watch_only_parser_scope",
        "reason": (
            "XVada is relevant to grammar inference, but Exp5580 is a cached-response parser repair "
            "with strict schema, deterministic object extraction, and documented aliases. Inferring a "
            "new grammar would broaden the scope."
        ),
    },
    {
        "source_id": "mosaic_multiagent_planning_2607_09603",
        "title": "Mosaic: Runtime-Efficient Multi-Agent Embodied Planning",
        "url": "https://arxiv.org/abs/2607.09603",
        "classification": "watch_only_domain_mismatch",
        "reason": (
            "Object-relative memory plus ILP coordination is adjacent to ARC planning, but V505's "
            "ordinary ARC lane is a single-agent EOM-MCTS live path, not a multi-agent AI2-THOR or "
            "search-and-rescue planner."
        ),
    },
    {
        "source_id": "vocabulary_verifier_gaps_2607_09560",
        "title": "Beyond Fixed Representations: The Vocabulary and Verifier Gaps in Open-Ended AI",
        "url": "https://arxiv.org/abs/2607.09560",
        "classification": "watch_only_position_context",
        "reason": (
            "The verifier-gap framing is architecture context only. V505 needs executable parser, "
            "exact predicate, memory, ARC, and sampler hooks, not a new open-ended innovation metric."
        ),
    },
    {
        "source_id": "extropic_tsu_xtr_z1",
        "title": "Extropic X0/XTR-0/Z1 and TSU writing",
        "url": "https://extropic.ai/writing",
        "classification": "watch_only_unavailable_hardware",
        "reason": "No authenticated local TSU execution path or matched Carnot timing receipt exists.",
    },
    {
        "source_id": "logical_intelligence_kona_aleph",
        "title": "Logical Intelligence Kona and Aleph public pages",
        "url": "https://logicalintelligence.com/blog/automatic-formal-verification-for-code-generation",
        "classification": "watch_only_proprietary_system",
        "reason": "Public claims support architecture context only; no reproducible local Kona or Aleph baseline is available.",
    },
)


def _clone_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _research_references_text(root: Path) -> str:
    return (root / RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")


def _roadmap_context(root: Path) -> JsonDict:
    relative = (
        ROADMAP_NEXT_RELATIVE_PATH
        if (root / ROADMAP_NEXT_RELATIVE_PATH).exists()
        else ROADMAP_RELATIVE_PATH
    )
    parsed = yaml.safe_load((root / relative).read_text(encoding="utf-8")) or {}
    tasks = parsed.get("tasks", [])
    task_ids = [
        str(task.get("id")) for task in tasks if isinstance(task, Mapping) and task.get("id")
    ]
    return {
        "source": str(relative),
        "milestone": str(parsed.get("milestone", "")),
        "task_ids": task_ids,
    }


def _planner_marker_found(references_text: str) -> bool:
    compact_text = references_text.replace("-", "")
    return PLANNER_MARKER in references_text or PLANNER_MARKER_COMPACT in compact_text


def _execution_section(references_text: str) -> str:
    if EXECUTION_REFRESH_HEADING not in references_text:
        return ""
    section = references_text.split(EXECUTION_REFRESH_HEADING, 1)[1]
    return section.split(EXECUTION_REFRESH_END, 1)[0]


def _finding_present(references_text: str, finding: Mapping[str, Any]) -> bool:
    haystack = references_text.lower()
    return any(str(token).lower() in haystack for token in finding["dedupe_tokens"])


def _new_actionable_findings(references_text: str) -> list[JsonDict]:
    if not _planner_marker_found(references_text) or EXECUTION_REFRESH_HEADING in references_text:
        return []
    return [
        _clone_json(finding)
        for finding in CANDIDATE_FINDINGS
        if not _finding_present(references_text, finding)
    ]


def _existing_execution_findings(references_text: str) -> list[JsonDict]:
    section = _execution_section(references_text)
    return [
        _clone_json(finding)
        for finding in CANDIDATE_FINDINGS
        if section and _finding_present(section, finding)
    ]


def _duplicate_candidates(
    references_text: str, accepted_findings: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    accepted_ids = {str(finding["source_id"]) for finding in accepted_findings}
    duplicates = [_clone_json(row) for row in DUPLICATE_SUPPRESSED_BASE]
    for finding in CANDIDATE_FINDINGS:
        if finding["source_id"] not in accepted_ids and _finding_present(references_text, finding):
            duplicates.append(
                {
                    "source_id": finding["source_id"],
                    "title": finding["title"],
                    "url": finding["url"],
                    "reason": "Already present in research-references.md, so no V505 execution append was allowed.",
                }
            )
    return duplicates


def _mapping_sources(accepted_findings: Sequence[Mapping[str, Any]], lane: str) -> list[str]:
    return [
        str(finding["source_id"])
        for finding in accepted_findings
        if lane in finding.get("lanes", [])
    ]


def build_experiment_mappings(accepted_findings: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    accepted_verifier_sources = _mapping_sources(
        accepted_findings, "exact verifier residual extension"
    )
    return [
        {
            "lane": "parser forensics and positive control",
            "experiment_ids": ["exp5580-parser-forensics-positive-control"],
            "source_ids": ["xvada_cfg_inference_2607_08959", "gram2token_xgrammar_llguidance"],
            "source_status": "watch_only_or_duplicate_context",
            "mapping": (
                "Keep Exp5580 scoped to cached-response parsing. Do not infer a new grammar or "
                "reopen grammar-row completion from superficially similar grammar sources."
            ),
        },
        {
            "lane": "local SOTA solve-versus-verify remeasurement",
            "experiment_ids": ["exp5581-clean-sota-solve-verify-remeasurement"],
            "source_ids": ["llm_as_verifier_2607_05391", "verification_horizon_2606_26300"],
            "source_status": "duplicate_planner_context",
            "mapping": (
                "Retain exact ASP/FSM authority and parser failure ceilings before interpreting "
                "any local-SOTA solve or verifier arm."
            ),
        },
        {
            "lane": "exact verifier residual extension",
            "experiment_ids": ["exp5582-exact-counterexample-verifier-extension"],
            "source_ids": [
                "llm_as_verifier_2607_05391",
                "verification_horizon_2606_26300",
                *accepted_verifier_sources,
            ],
            "source_status": (
                "accepted_plus_planner_context"
                if accepted_verifier_sources
                else "duplicate_planner_context"
            ),
            "mapping": (
                "Use property-template discipline and entity-attribution stress only on clean "
                "Exp5581 residuals; every promoted predicate must compile to deterministic exact "
                "checks and avoid unsafe false accepts."
            ),
        },
        {
            "lane": "two-timescale continuous self-learning",
            "experiment_ids": [
                "exp5583-causal-memory-metric-corrigendum",
                "exp5584-two-timescale-exact-gated-controller",
                "exp5585-reset-free-live-local-sota-sessions",
                "exp5586-delayed-promotion-poisoning-gate",
            ],
            "source_ids": [
                "pace_2605_23019",
                "evopolicygym_2607_02440",
                "shared_selective_memory_2607_09493",
                "blind_curator_2607_07436",
            ],
            "source_status": "planner_context_plus_watch_only",
            "mapping": (
                "Keep memory-policy and active-spline updates exact-gated with fixed budgets, "
                "typed decision ledgers, stale-trace rejection, and rollback. Shared-memory "
                "workspace claims stay non-promoted."
            ),
        },
        {
            "lane": "PTRM leave-one-game-out adjudication",
            "experiment_ids": ["exp5587-ptrm-leave-one-game-out-adjudication"],
            "source_ids": ["ptrm_2605_19943_loop_2604_07822"],
            "source_status": "duplicate_reference_history",
            "mapping": "Use the existing checkpoint for LOO adjudication; do not retrain or count PTRM as ordinary ARC.",
        },
        {
            "lane": "ordinary ARC EOM-MCTS live path",
            "experiment_ids": [
                "exp5588-eom-mcts-live-precheck",
                "exp5589-gated-ordinary-arc-levelup",
            ],
            "source_ids": ["known_untried_eom_mcts_line", "mosaic_multiagent_planning_2607_09603"],
            "source_status": "planned_line_plus_watch_only",
            "mapping": (
                "Retain the known-untried object-model MCTS route. Mosaic is only adjacent "
                "constraint-guided planning context and does not replace live-agent self-discovery."
            ),
        },
        {
            "lane": "matched sampler crossover and board continuity",
            "experiment_ids": ["exp5590-matched-cpu-cuda-crossover-board-continuity"],
            "source_ids": ["thermodynamic_pbit_hardware_prior", "extropic_tsu_xtr_z1"],
            "source_status": "duplicate_or_watch_only",
            "mapping": (
                "Require matched CPU/CUDA sample quality and authenticated board receipts before "
                "any speedup claim; public TSU/Kona material remains non-executable context."
            ),
        },
    ]


def render_execution_refresh_block(findings: Sequence[Mapping[str, Any]], *, run_date: str) -> str:
    lines = [
        f"## V505 Execution Refresh - {run_date}",
        "",
        "Execution-time sweep after the `.505` planner refresh checked arXiv primary pages and "
        "direct API results, OpenReview public pages, Hugging Face Papers, Semantic Scholar "
        "EBT/ARM-EBM routes, GitHub discovery, Extropic writing, Logical Intelligence public "
        "pages, local duplicate history, the exclusion manifest, and known-issues scope notes. "
        "Only non-duplicate actionable deltas are listed below.",
        "",
        "### New actionable deltas",
    ]
    for finding in findings:
        if finding["source_id"] == "agentic_property_templates_2607_09072":
            lines.append(
                "- **{title}** (arXiv:{arxiv_id}, {url}; HF mirror {secondary_url}): "
                "Use property templates as the admissible shape for Exp5582 exact predicate "
                "candidates, and require executable counterexample evidence whenever a formal "
                "predicate model and implementation behavior diverge. This sharpens the existing "
                "exact-residual extension lane; it does not authorize an LLM judge, broad theorem "
                "prover buildout, or external Spark benchmark claim.".format(**finding)
            )
        elif finding["source_id"] == "deceptive_grounding_2607_09349":
            lines.append(
                "- **{title}** (arXiv:{arxiv_id}, {url}; HF mirror {secondary_url} checked but "
                "not present): Add entity/variable attribution as a bounded Exp5582 residual "
                "stress class when clean ASP/FSM rows expose a real wrong-entity or wrong-variable "
                "binding. This does not add a clinical RAG benchmark, citation-faithfulness scorer, "
                "or external generated-text detector.".format(**finding)
            )
    lines.extend(
        [
            "",
            "### Execution impact",
            "- **Plan impact:** No roadmap edit is required. The accepted deltas map only to the "
            "already-planned Exp5582 exact verifier extension after the Exp5581 clean-panel gate.",
            "- **Duplicates suppressed:** PACE, EvoPolicyGym, LLM-as-a-Verifier, Verification Horizon, "
            "Blind Curator, ASP Energised, ClassicLogic, Gram2Token/XGrammar/llguidance, SelfMem, "
            "Continual Harness, EBT, ARM-EBM, and prior memory/KAN/hardware lines were already covered "
            "or stayed non-promoted.",
            "- **Closed scope:** closed_scopes_reopened=false. Retired grammar-row completion, SGE, "
            "cross-family CSL, external generated-text scoring, broad GRPO/RL/fine-tuning, proprietary "
            "TSU/Kona/Aleph execution, and unmatched hardware speedup claims remain closed.",
            "- **Watch-only/excluded:** Shared Selective Persistent Memory, XVada grammar inference, "
            "Mosaic, open-ended verifier-gap position work, Extropic TSU/XTR/Z1, and Logical "
            "Intelligence Kona/Aleph were checked but not promoted as executable `.505` dependencies.",
            "",
            EXECUTION_REFRESH_END,
            "",
        ]
    )
    return "\n".join(lines)


def _honest_verdict(
    planner_marker_found: bool, accepted_findings: Sequence[Mapping[str, Any]]
) -> str:
    if not planner_marker_found:
        return "blocked: V505 planner refresh marker missing; source-delta append refused"
    if accepted_findings:
        return (
            f"complete: accepted {len(accepted_findings)} non-duplicate actionable V505 "
            "source deltas and kept closed scopes closed"
        )
    return "complete: no new non-duplicate actionable V505 source deltas; references left unchanged"


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
) -> JsonDict:
    references_text = _research_references_text(root)
    planner_marker_found = _planner_marker_found(references_text)
    existing_findings = _existing_execution_findings(references_text)
    new_findings = _new_actionable_findings(references_text)
    accepted_findings = existing_findings or new_findings
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "search_cutoff": SEARCH_CUTOFF,
        "result_path": str(RESULT_RELATIVE_PATH),
        "spec_refs": list(SPEC_REFS),
        "sources_checked": _clone_json(SOURCES_CHECKED),
        "primary_sources_checked": _clone_json(PRIMARY_SOURCES_CHECKED),
        "new_references_added": _clone_json(accepted_findings),
        "duplicates_suppressed": _duplicate_candidates(references_text, accepted_findings),
        "citation_trails_checked": _clone_json(CITATION_TRAILS_CHECKED),
        "research_references_updated": bool(accepted_findings),
        "planner_marker_found": planner_marker_found,
        "experiment_mappings": build_experiment_mappings(accepted_findings),
        "watch_only_items": _clone_json(WATCH_ONLY_ITEMS),
        "closed_scopes_reopened": False,
        "closed_scope_review": {
            "grammar_row_completion_reopened": False,
            "sge_reopened": False,
            "cross_family_csl_reopened": False,
            "external_text_scorer_reopened": False,
            "broad_grpo_rl_finetuning_reopened": False,
            "proprietary_tsu_kona_aleph_reopened": False,
            "hardware_speedup_claim_reopened": False,
            "operator_authorized_differentiator": None,
        },
        "roadmap_context": _roadmap_context(root),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "honest_verdict": _honest_verdict(planner_marker_found, accepted_findings),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required fields: {missing}")
    _require(
        isinstance(artifact["field_principles"], Mapping), "field_principles must be a mapping"
    )
    missing_principles = [
        field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact["field_principles"]
    ]
    _require(not missing_principles, f"field_principles missing: {missing_principles}")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "wrong inference_substrate")
    _require(artifact["closed_scopes_reopened"] is False, "closed_scopes_reopened must be false")
    _require(
        isinstance(artifact["planner_marker_found"], bool), "planner_marker_found must be bool"
    )
    _require(
        isinstance(artifact["research_references_updated"], bool),
        "research_references_updated must be bool",
    )
    _require(isinstance(artifact["sources_checked"], list), "sources_checked must be a list")
    _require(
        isinstance(artifact["primary_sources_checked"], list),
        "primary_sources_checked must be a list",
    )
    _require(
        isinstance(artifact["citation_trails_checked"], list),
        "citation_trails_checked must be a list",
    )
    _require(
        isinstance(artifact["new_references_added"], list), "new_references_added must be a list"
    )
    _require(
        isinstance(artifact["duplicates_suppressed"], list), "duplicates_suppressed must be a list"
    )
    _require(
        isinstance(artifact["experiment_mappings"], list), "experiment_mappings must be a list"
    )
    _require(isinstance(artifact["watch_only_items"], list), "watch_only_items must be a list")
    _require(
        str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES),
        "honest_verdict lacks terminal prefix",
    )


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
) -> JsonDict:
    started = time.monotonic()
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    references_text = references_path.read_text(encoding="utf-8")
    new_findings = _new_actionable_findings(references_text)
    if new_findings:
        references_path.write_text(
            references_text.rstrip()
            + "\n\n"
            + render_execution_refresh_block(new_findings, run_date=run_date),
            encoding="utf-8",
        )
    final_duration = duration_s + max(0.0, time.monotonic() - started)
    artifact = build_artifact(root=root, run_date=run_date, duration_s=round(final_duration, 6))
    validate_artifact(artifact)
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-date", default=RUN_DATE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    artifact = build_and_write_artifact(root=args.root, run_date=args.run_date)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
