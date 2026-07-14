"""Exp5626: ingest the V508 execution-time source delta.

Spec refs: REQ-REPORT-5626, SCENARIO-REPORT-5626-NOOP,
SCENARIO-REPORT-5626-BLOCKED-MARKER,
SCENARIO-REPORT-5626-FIELD-PRINCIPLES.

This module turns the execution-time literature sweep into a stable local
receipt. Public search indexes drift, so the code does not try to be a web
crawler. Instead, it preserves the decision that matters for downstream
experiments: the search window starts after the V508 planner marker, repeated
sources do not create new work, and unavailable or proprietary systems remain
watch-only unless there is an exact Carnot experiment hook.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5626_v508_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
EXP5625_RELATIVE_PATH = Path("results/experiment_5625_transition_v508.json")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5626_v508_source_delta_ingestion"
EXPERIMENT_ID = "exp5626-v508-source-delta-ingestion"
MILESTONE = "2026.07.508"
RUN_DATE = "20260714"
SEARCH_CUTOFF = "2026-07-14"
SCHEMA = "carnot.experiment_5626.v508_source_delta_ingestion.v1"
RANDOM_SEED = 5626
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CLOSED_SCOPES_REOPENED = False
TERMINAL_PREFIXES = ("complete:", "blocked:")

PLANNER_MARKER = "## V508 Planner Refresh - 20260714"
PLANNER_MARKER_COMPACT = PLANNER_MARKER.replace("-", "")
EXECUTION_REFRESH_HEADING = "## V508 Execution Refresh - 20260714"

ALLOWED_MAPPING_IDS = {
    "exp5627-online-conformal-kan-qualification",
    "exp5628-conformal-active-spline-kan-csl",
    "exp5629-conformal-kan-independent-audit",
    "exp5630-arc-epistemic-object-probe-prototype",
    "exp5631-arc-epistemic-probe-live-ab",
    "exp5632-arc-live-self-discovery-levelup-v508",
    "exp5633-temperature-exchange-cdls-exact-audit",
    "exp5634-temperature-exchange-cdls-quality",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "planner_marker_found",
    "sources_checked",
    "search_timestamp_utc",
    "new_references_added",
    "duplicates_suppressed",
    "experiment_mappings",
    "watch_only_items",
    "closed_scopes_reopened",
    "inference_substrate",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "One-line annotations for every required headline and gate field.",
    "planner_marker_found": "the search window is explicit",
    "sources_checked": "coverage is auditable",
    "search_timestamp_utc": "recency is reproducible",
    "new_references_added": "duplicates do not count",
    "duplicates_suppressed": "repeated ideas create no work",
    "experiment_mappings": "sources need executable hooks",
    "watch_only_items": "unavailable systems support no claim",
    "closed_scopes_reopened": "retirement requires authority",
    "inference_substrate": "this is source synthesis",
    "reproducibility_checksum": "the source set is stable",
    "honest_verdict": "a no-op is terminal",
}

SPEC_REFS = (
    "REQ-REPORT-5626",
    "SCENARIO-REPORT-5626-NOOP",
    "SCENARIO-REPORT-5626-BLOCKED-MARKER",
    "SCENARIO-REPORT-5626-FIELD-PRINCIPLES",
)

SOURCES_CHECKED: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "queries": [
            "EBM verification/reasoning",
            "neural constraint satisfaction",
            "Ising ML",
            "hallucination detection/mitigation",
            "KANs",
            "energy-guided decoding",
            "accelerated sampling",
            "continual/online constraint learning",
        ],
        "status": "checked_primary_pages_recent_lists_and_search",
        "decision": "V508 planner deltas remain the only exact local hooks",
    },
    {
        "surface": "OpenReview",
        "queries": [
            "Energy-Based Transformers",
            "Spilled Energy",
            "Distributional Energy-Based Models",
            "ConsFormer-LNS",
            "SafeMPO",
        ],
        "status": "checked_public_forum_search",
        "decision": "duplicates, workshop context, or no stronger Exp5627-Exp5634 hook",
    },
    {
        "surface": "Semantic Scholar",
        "queries": ["arXiv:2507.02092 citation route", "arXiv:2512.15605 citation route"],
        "status": "EBT API returned metadata; ARM-EBM API rate-limited during check",
        "decision": "citation routes do not supersede the V508 planner graph",
    },
    {
        "surface": "Hugging Face Papers",
        "queries": [
            "energy-based verification",
            "online conformal prediction",
            "hallucination detection",
            "energy-guided decoding",
        ],
        "status": "checked_papers_search_and_exact_EBT_page",
        "decision": "mirrors already-indexed arXiv/OpenReview items",
    },
    {
        "surface": "GitHub discovery/trending",
        "queries": [
            "energy-based transformer reasoning",
            "conformal KAN",
            "parallel tempering Ising",
            "hallucination detection LLM",
        ],
        "status": "checked_search_api_and_web_results",
        "decision": "no repository supplied a cleaner local substrate than current Carnot code",
    },
    {
        "surface": "Extropic writing",
        "queries": ["TSU", "XTR-0", "X0", "Z1", "thermodynamic computing"],
        "status": "checked_public_writing_index",
        "decision": "public TSU/XTR/Z1 context remains unavailable for Carnot execution",
    },
    {
        "surface": "Logical Intelligence public pages",
        "queries": ["Kona 1.0", "Aleph", "Sudoku EBM", "Raise summit 2026"],
        "status": "checked_public_site_search",
        "decision": "proprietary public updates do not provide local weights or receipts",
    },
    {
        "surface": "local Carnot ledgers",
        "queries": [
            "research-references.md after V508 marker",
            "research-complete.yaml",
            "openspec/change-proposals/*.md",
            "results/experiment_5625_transition_v508.json",
            "ops/exclusion_manifest.yaml",
            "ops/known-issues.md",
        ],
        "status": "checked",
        "decision": "duplicates and retired scopes are already represented locally",
    },
)

SOURCE_LINK_CHECKS: tuple[JsonDict, ...] = (
    {
        "source_id": "online_group_conformal_2606_00419",
        "url": "https://arxiv.org/abs/2606.00419",
        "status": "primary_arxiv_opened_duplicate_planner_delta",
    },
    {
        "source_id": "training_conditional_regret_2602_16537",
        "url": "https://arxiv.org/abs/2602.16537",
        "status": "primary_arxiv_opened_duplicate_planner_delta",
    },
    {
        "source_id": "snn_csp_parallel_tempering_2607_08897",
        "url": "https://arxiv.org/abs/2607.08897",
        "status": "primary_arxiv_opened_duplicate_planner_delta",
    },
    {
        "source_id": "semantic_scholar_ebt_route",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092",
        "status": "http_200_duplicate_citation_route",
        "evidence": {"citationCount": 27, "influentialCitationCount": 2},
    },
    {
        "source_id": "semantic_scholar_arm_ebm_route",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605",
        "status": "http_429_no_fresh_count_claim",
    },
    {
        "source_id": "openreview_ebt_zbj3qp1byg",
        "url": "https://openreview.net/forum?id=ZBj3Qp1bYg",
        "status": "public_openreview_page_checked_duplicate_context",
    },
    {
        "source_id": "huggingface_ebt_2507_02092",
        "url": "https://huggingface.co/papers/2507.02092",
        "status": "public_hf_paper_page_checked_duplicate_context",
    },
    {
        "source_id": "github_alexiglad_ebt",
        "url": "https://github.com/alexiglad/EBT",
        "status": "public_repository_duplicate_ebt_context",
    },
    {
        "source_id": "extropic_writing",
        "url": "https://extropic.ai/writing",
        "status": "http_200_watch_only_no_local_tsu",
    },
    {
        "source_id": "logical_intelligence_kona",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "status": "public_page_checked_watch_only_proprietary",
    },
)

CITATION_TRAILS_CHECKED: tuple[JsonDict, ...] = (
    {
        "paper": "Energy-Based Transformers",
        "paper_id": "2507.02092",
        "route": "Semantic Scholar direct API plus public web-search fallback",
        "status": "metadata_returned",
        "promoted_delta": False,
        "note": "The route repeated Fixed-Point Reasoners, LoopUS, causal-energy parameterization, NRGPT, and EBM workload items already represented in Carnot history.",
    },
    {
        "paper": "ARM-EBM",
        "paper_id": "2512.15605",
        "route": "Semantic Scholar direct API plus public web-search fallback",
        "status": "api_rate_limited_429",
        "promoted_delta": False,
        "note": "Search fallback repeated ARM-EBM theory context and did not expose a new local Exp5627-Exp5634 hook.",
    },
)

CANDIDATE_FINDINGS: tuple[JsonDict, ...] = ()

DUPLICATE_SUPPRESSED_BASE: tuple[JsonDict, ...] = (
    {
        "source_id": "online_group_conformal_2606_00419",
        "title": "Parameter-Free and Group Conditional Online Conformal Prediction",
        "url": "https://arxiv.org/abs/2606.00419",
        "reason": "Already accepted in the V508 planner block for Exp5627 group-conditional online coverage.",
    },
    {
        "source_id": "training_conditional_regret_2602_16537",
        "title": "Optimal Training-Conditional Regret for Online Conformal Prediction",
        "url": "https://arxiv.org/abs/2602.16537",
        "reason": "Already accepted in the V508 planner block for Exp5627/Exp5628 chronological regret gates.",
    },
    {
        "source_id": "snn_csp_parallel_tempering_2607_08897",
        "title": "Breaking Local-Minimum Traps in SNN-Based CSP Solvers via Parallel Tempering",
        "url": "https://arxiv.org/abs/2607.08897",
        "reason": "Already accepted in the V508 planner block for Exp5633-Exp5634 temperature-exchange quality checks.",
    },
    {
        "source_id": "ebt_arm_ebm_semantic_scholar_routes",
        "title": "EBT 2507.02092 and ARM-EBM 2512.15605 citation routes",
        "url": "https://arxiv.org/abs/2507.02092",
        "reason": "Direct citation routes were checked again; they repeat architecture context and older citation-trail items.",
    },
    {
        "source_id": "distributional_ebm_2605_18871",
        "title": "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
        "url": "https://arxiv.org/abs/2605.18871",
        "reason": "Already heavily indexed in Carnot verifier-moat history and outside the V508 exact conformal/ARC/cDLS graph.",
    },
    {
        "source_id": "consformer_lns_2603_20801",
        "title": "Large Neighborhood Search meets Iterative Neural Constraint Heuristics",
        "url": "https://arxiv.org/abs/2603.20801",
        "reason": "Already indexed as ConsFormer-LNS context and not a new V508 dependency.",
    },
    {
        "source_id": "march_claim_checking_2603_24579",
        "title": "MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination",
        "url": "https://arxiv.org/abs/2603.24579",
        "reason": "Already represented in Carnot claim-checking history; V508 does not reopen the external generated-text scorer line.",
    },
    {
        "source_id": "energy_guided_object_hallucination_2507_07731",
        "title": "Energy-Guided Decoding for Object Hallucination Mitigation",
        "url": "https://arxiv.org/abs/2507.07731",
        "reason": "Already indexed as VLM hidden-state decoding context and has no exact local V508 hook.",
    },
)

WATCH_ONLY_ITEMS: tuple[JsonDict, ...] = (
    {
        "source_id": "static_conformalized_kans_2504_15240",
        "title": "Conformalized-KANs",
        "url": "https://arxiv.org/abs/2504.15240",
        "classification": "watch_only_static_kan_uq",
        "evidence_status": "older_primary_arxiv_and_github_code",
        "reason": "Static KAN prediction intervals do not replace the causal group-conditional online conformal contract already planned for Exp5627.",
    },
    {
        "source_id": "github_claim_level_hallucination_repo",
        "title": "claim-level-hallucination-detection",
        "url": "https://github.com/Meher134/claim-level-hallucination-detection",
        "classification": "watch_only_external_generated_text_scorer",
        "evidence_status": "new_public_repository_zero_size_at_check_time",
        "reason": "A fresh external repository does not provide an exact local verifier hook and would reopen retired generated-text scorer scope.",
    },
    {
        "source_id": "github_parallel_tempering_examples",
        "title": "GitHub parallel-tempering Ising examples",
        "url": "https://github.com/search?q=%22parallel+tempering%22+Ising",
        "classification": "watch_only_generic_sampler_code",
        "evidence_status": "public_search_results_not_carnot_kernel",
        "reason": "Generic Ising examples are not the corrected cDLS kernel required by Exp5633.",
    },
    {
        "source_id": "logical_intelligence_public_updates",
        "title": "Logical Intelligence Kona and Aleph public updates",
        "url": "https://logicalintelligence.com/",
        "classification": "watch_only_proprietary_system",
        "evidence_status": "public_pages_no_local_weights_or_reproducible_receipt",
        "reason": "Kona and Aleph remain proprietary context and cannot be V508 baselines or execution evidence.",
    },
    {
        "source_id": "extropic_tsu_xtr_z1",
        "title": "Extropic TSU, XTR-0, X0, and Z1 writing",
        "url": "https://extropic.ai/writing",
        "classification": "watch_only_unavailable_hardware",
        "evidence_status": "public_writing_no_local_execution_path",
        "reason": "No authenticated local TSU path or matched Carnot sampler-speed receipt exists.",
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


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_text_if_present(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _planner_marker_found(references_text: str) -> bool:
    compact_text = references_text.replace("-", "")
    return PLANNER_MARKER in references_text or PLANNER_MARKER_COMPACT in compact_text


def _planner_marker_line(references_text: str) -> int | None:
    index = references_text.find(PLANNER_MARKER)
    if index < 0:
        return None
    return references_text[:index].count("\n") + 1


def _proposal_paths(root: Path) -> list[Path]:
    proposal_dir = root / "openspec/change-proposals"
    if not proposal_dir.exists():
        return []
    return sorted(proposal_dir.glob("*.md"))


def _dedupe_paths(root: Path) -> list[Path]:
    paths = [
        root / RESEARCH_REFERENCES_RELATIVE_PATH,
        root / RESEARCH_COMPLETE_RELATIVE_PATH,
        root / VNEXT_RELATIVE_PATH,
        root / EXCLUSION_MANIFEST_RELATIVE_PATH,
        root / KNOWN_ISSUES_RELATIVE_PATH,
        root / EXP5625_RELATIVE_PATH,
        root / CONDUCTOR_RELATIVE_PATH,
    ]
    paths.extend(_proposal_paths(root))
    return list(dict.fromkeys(paths))


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _dedupe_corpus(root: Path) -> tuple[str, list[JsonDict]]:
    chunks: list[str] = []
    checked: list[JsonDict] = []
    for path in _dedupe_paths(root):
        exists = path.exists()
        text = _read_text_if_present(path)
        if text:
            chunks.append(text)
        checked.append(
            {
                "path": _relative_path(root, path),
                "exists": exists,
                "sha256": path_sha256(path) if exists else None,
            }
        )
    return "\n".join(chunks), checked


def _roadmap_context(root: Path) -> JsonDict:
    relative = ROADMAP_NEXT_RELATIVE_PATH if (root / ROADMAP_NEXT_RELATIVE_PATH).exists() else ROADMAP_RELATIVE_PATH
    parsed = yaml.safe_load(_read_text_if_present(root / relative)) or {}
    tasks = parsed.get("tasks", []) if isinstance(parsed, Mapping) else []
    task_ids = [
        str(task.get("id")) for task in tasks if isinstance(task, Mapping) and task.get("id")
    ]
    milestone = str(parsed.get("milestone", "")) if isinstance(parsed, Mapping) else ""
    return {"source": relative.as_posix(), "milestone": milestone, "task_ids": task_ids}


def _duplicate_candidates() -> list[JsonDict]:
    return [_clone_json(row) for row in DUPLICATE_SUPPRESSED_BASE]


def _accepted_findings() -> list[JsonDict]:
    return [_clone_json(row) for row in CANDIDATE_FINDINGS]


def build_experiment_mappings() -> list[JsonDict]:
    return [
        {
            "lane": "online conformal KAN qualification and gated CSL",
            "experiment_ids": [
                "exp5627-online-conformal-kan-qualification",
                "exp5628-conformal-active-spline-kan-csl",
                "exp5629-conformal-kan-independent-audit",
            ],
            "source_ids": [
                "online_group_conformal_2606_00419",
                "training_conditional_regret_2602_16537",
                "static_conformalized_kans_2504_15240",
            ],
            "source_status": "duplicate_planner_context_or_watch_only",
            "mapping": "Use the planner-defined online conformal contract; static KAN UQ code is context only.",
        },
        {
            "lane": "epistemic-object ARC probe and live level attempt",
            "experiment_ids": [
                "exp5630-arc-epistemic-object-probe-prototype",
                "exp5631-arc-epistemic-probe-live-ab",
                "exp5632-arc-live-self-discovery-levelup-v508",
            ],
            "source_ids": [
                "epistemic_mcts_planner_context",
                "object_centric_world_models_planner_context",
                "logical_intelligence_public_updates",
            ],
            "source_status": "duplicate_planner_context_or_watch_only",
            "mapping": "Keep the live ARC hook generic and local; proprietary Kona/Aleph pages do not become comparators.",
        },
        {
            "lane": "temperature-exchange cDLS exact audit and quality trial",
            "experiment_ids": [
                "exp5633-temperature-exchange-cdls-exact-audit",
                "exp5634-temperature-exchange-cdls-quality",
            ],
            "source_ids": [
                "snn_csp_parallel_tempering_2607_08897",
                "github_parallel_tempering_examples",
                "extropic_tsu_xtr_z1",
            ],
            "source_status": "duplicate_planner_context_or_watch_only",
            "mapping": "Only the corrected Carnot cDLS kernel is executable; SNN, generic GitHub, and TSU context cannot create a speedup claim.",
        },
    ]


def _honest_verdict(
    planner_marker_found: bool, accepted_findings: Sequence[Mapping[str, Any]]
) -> str:
    if not planner_marker_found:
        return "blocked: V508 planner refresh marker missing; source-delta append refused"
    if accepted_findings:
        return (
            f"complete: accepted {len(accepted_findings)} non-duplicate actionable V508 "
            "source deltas and kept retired scopes closed"
        )
    return "complete: no new non-duplicate actionable V508 source deltas; references left unchanged"


def _closed_scope_review() -> JsonDict:
    return {
        "native_runtime_certificate_reopened": False,
        "solve_verify_scope_reopened": False,
        "arc_transition_cycle_proxy_reopened": False,
        "cdls_timing_crossover_reopened": False,
        "generated_text_scoring_reopened": False,
        "proprietary_tsu_kona_aleph_reopened": False,
        "unmatched_hardware_speedup_reopened": False,
        "operator_authorized_differentiator": None,
    }


def _normalize_timestamp(search_timestamp_utc: str | None) -> str:
    timestamp = search_timestamp_utc or datetime.now(UTC).replace(microsecond=0).isoformat()
    if timestamp.endswith("+00:00"):
        timestamp = timestamp[:-6] + "Z"
    return timestamp


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    search_timestamp_utc: str | None = None,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
) -> JsonDict:
    references_text = _read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    planner_marker_found = _planner_marker_found(references_text)
    accepted_findings = _accepted_findings()
    _corpus_text, dedupe_checked = _dedupe_corpus(root)
    duplicates = _duplicate_candidates()
    status = "complete" if planner_marker_found else "blocked"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "milestone": MILESTONE,
        "run_date": run_date,
        "search_cutoff": SEARCH_CUTOFF,
        "search_timestamp_utc": _normalize_timestamp(search_timestamp_utc),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "spec_refs": list(SPEC_REFS),
        "sources_checked": _clone_json(SOURCES_CHECKED),
        "source_link_checks": _clone_json(SOURCE_LINK_CHECKS),
        "citation_trails_checked": _clone_json(CITATION_TRAILS_CHECKED),
        "dedupe_corpus_checked": dedupe_checked,
        "marker_checks": {
            "planner_marker": PLANNER_MARKER,
            "planner_marker_found": planner_marker_found,
            "planner_marker_line": _planner_marker_line(references_text),
            "search_window": "strictly_after_planner_marker",
            "execution_refresh_heading": EXECUTION_REFRESH_HEADING,
            "execution_refresh_present": EXECUTION_REFRESH_HEADING in references_text,
        },
        "duplicate_checks": {
            "candidate_count": len(CANDIDATE_FINDINGS),
            "accepted_count": len(accepted_findings),
            "duplicates_suppressed_count": len(duplicates),
            "dedupe_sources_count": len(dedupe_checked),
        },
        "new_references_added": accepted_findings,
        "duplicates_suppressed": duplicates,
        "research_references_updated": bool(accepted_findings),
        "planner_marker_found": planner_marker_found,
        "experiment_mappings": build_experiment_mappings(),
        "watch_only_items": _clone_json(WATCH_ONLY_ITEMS),
        "closed_scopes_reopened": CLOSED_SCOPES_REOPENED,
        "closed_scope_review": _closed_scope_review(),
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
    _require(
        artifact["closed_scopes_reopened"] is CLOSED_SCOPES_REOPENED,
        "closed_scopes_reopened must be false",
    )
    _require(
        isinstance(artifact["planner_marker_found"], bool), "planner_marker_found must be bool"
    )
    _require(
        isinstance(artifact["research_references_updated"], bool),
        "research_references_updated must be bool",
    )
    for field in (
        "sources_checked",
        "new_references_added",
        "duplicates_suppressed",
        "experiment_mappings",
        "watch_only_items",
    ):
        _require(isinstance(artifact[field], list), f"{field} must be a list")
    mapping_ids = {
        experiment_id
        for row in artifact["experiment_mappings"]
        for experiment_id in row.get("experiment_ids", [])
    }
    _require(mapping_ids <= ALLOWED_MAPPING_IDS, "experiment_mappings outside exp5627-exp5634")
    _require(str(artifact["search_timestamp_utc"]).endswith("Z"), "timestamp must be UTC")
    _require(
        str(artifact["reproducibility_checksum"]).startswith("sha256:"),
        "checksum must be sha256-prefixed",
    )
    _require(
        str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES),
        "honest_verdict lacks terminal prefix",
    )


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    search_timestamp_utc: str | None = None,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
) -> JsonDict:
    started = time.monotonic()
    final_duration = duration_s + max(0.0, time.monotonic() - started)
    artifact = build_artifact(
        root=root,
        search_timestamp_utc=search_timestamp_utc,
        run_date=run_date,
        duration_s=round(final_duration, 6),
    )
    validate_artifact(artifact)
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-date", default=RUN_DATE)
    parser.add_argument("--search-timestamp-utc")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    artifact = build_and_write_artifact(
        root=args.root,
        run_date=args.run_date,
        search_timestamp_utc=args.search_timestamp_utc,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
