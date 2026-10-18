"""Exp5614: ingest the V507 execution-time source delta.

Spec refs: REQ-REPORT-5614, SCENARIO-REPORT-5614-NOOP,
SCENARIO-REPORT-5614-BLOCKED-MARKER,
SCENARIO-REPORT-5614-FIELD-PRINCIPLES.

This module turns a public-source freshness sweep into a stable repository
receipt. The web search itself is not repeated by tests because public indexes
drift; the durable behavior is the local evidence rule. A source only creates
work when it is new, non-duplicate, and has an exact non-retired Carnot hook in
Exp5615 through Exp5623. The July 14 execution sweep found no such source, so a
no-op artifact is the honest terminal result.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5614_v507_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
EXP5613_RELATIVE_PATH = Path("results/experiment_5613_transition_v507.json")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5614_v507_source_delta_ingestion"
EXPERIMENT_ID = "exp5614-v507-source-delta-ingestion"
MILESTONE = "2026.07.507"
RUN_DATE = "20260714"
SEARCH_CUTOFF = "2026-07-14"
SCHEMA = "carnot.experiment_5614.v507_source_delta_ingestion.v1"
RANDOM_SEED = 5614
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

PLANNER_MARKER = "## V507 Planner Refresh - 20260714"
PLANNER_MARKER_COMPACT = PLANNER_MARKER.replace("-", "")
EXECUTION_REFRESH_HEADING = "## V507 Execution Refresh - 20260714"

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
    "planner_marker_found": "the window is explicit",
    "sources_checked": "coverage is auditable",
    "search_timestamp_utc": "recency is reproducible",
    "new_references_added": "duplicates do not count",
    "duplicates_suppressed": "repeated ideas create no work",
    "experiment_mappings": "sources need executable hooks",
    "watch_only_items": "unavailable systems support no claim",
    "closed_scopes_reopened": "retirement requires authority",
    "inference_substrate": "this is source/repository synthesis, not model inference",
    "reproducibility_checksum": "the source set is stable",
    "honest_verdict": "no-op is terminal",
}

SPEC_REFS = (
    "REQ-REPORT-5614",
    "SCENARIO-REPORT-5614-NOOP",
    "SCENARIO-REPORT-5614-BLOCKED-MARKER",
    "SCENARIO-REPORT-5614-FIELD-PRINCIPLES",
)

SOURCES_CHECKED: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "queries": [
            "EBM verification/reasoning",
            "neural constraint satisfaction and neural CSPs",
            "Ising ML and hardware-accelerated sampling",
            "LLM hallucination detection/mitigation",
            "KANs and constrained KANs",
            "energy-guided and constrained decoding",
            "continual/online constraint learning",
        ],
        "status": "checked_arxiv_api_and_primary_pages",
        "decision": "newly visible items lacked a local exact non-retired Exp5615-Exp5623 hook",
    },
    {
        "surface": "OpenReview",
        "queries": [
            "Loss Smoothing for Continual Adaptation",
            "cDLS continuous intermediates",
            "Energy-Based Transformers",
            "constrained generation",
            "continual adaptation workshop updates",
        ],
        "status": "checked_public_search; forum pages behind browser challenge where direct open failed",
        "decision": "planner-indexed workshop or duplicate context only",
    },
    {
        "surface": "Semantic Scholar",
        "queries": ["arXiv:2507.02092 citation route", "arXiv:2512.15605 citation route"],
        "status": "direct_api_returned_http_429; web-search fallback checked",
        "decision": "no fresh citation-route dependency promoted",
    },
    {
        "surface": "Hugging Face Papers",
        "queries": [
            "energy based verification",
            "KAN continual learning",
            "hallucination detection LLM",
            "constrained decoding energy guided",
        ],
        "status": "checked_papers_index_and_search_fallback",
        "decision": "mirrors repeated arXiv or already-indexed EBT context",
    },
    {
        "surface": "GitHub discovery/trending",
        "queries": [
            "World Action Verifier repository",
            "studying-cl continual learning repository",
            "constrained-diffusion finite automata repository",
            "energy-based reasoning verification",
            "KAN continual learning",
        ],
        "status": "checked_web_discovery_and_topic_routes",
        "decision": "repositories were duplicate planner context or external systems",
    },
    {
        "surface": "Extropic writing",
        "queries": ["TSU", "XTR-0", "X0", "Z1", "thermodynamic computing"],
        "status": "checked_public_writing_index",
        "decision": "no authenticated Carnot-accessible TSU route",
    },
    {
        "surface": "Logical Intelligence public pages",
        "queries": ["Kona 1.0", "Aleph", "PutnamBench", "deterministic AI updates"],
        "status": "checked_public_site_and_blog/search_updates",
        "decision": "proprietary context only; no local weights or benchmark artifact",
    },
    {
        "surface": "local Carnot ledgers",
        "queries": [
            "full research-references.md",
            "research-complete.yaml",
            "openspec/change-proposals/*.md",
            "results/experiment_5613_transition_v507.json",
            "ops/exclusion_manifest.yaml",
            "ops/known-issues.md",
        ],
        "status": "checked",
        "decision": "V507 planner deltas already define the executable graph",
    },
)

SOURCE_LINK_CHECKS: tuple[JsonDict, ...] = (
    {
        "source_id": "retain_or_adapt_2607_05609",
        "url": "https://arxiv.org/abs/2607.05609",
        "status": "primary_arxiv_opened_duplicate_planner_delta",
    },
    {
        "source_id": "when_cl_requires_learning_2607_07847",
        "url": "https://arxiv.org/abs/2607.07847",
        "status": "primary_arxiv_opened_duplicate_planner_delta",
    },
    {
        "source_id": "loss_smoothing_openreview_puqcokv69j",
        "url": "https://openreview.net/forum?id=pUqcOkV69j",
        "status": "search_result_checked_openreview_challenge_on_direct_open",
    },
    {
        "source_id": "world_action_verifier_2604_01985",
        "url": "https://arxiv.org/abs/2604.01985",
        "status": "primary_arxiv_opened_duplicate_planner_delta",
    },
    {
        "source_id": "cdls_openreview_fni2fpyafq",
        "url": "https://openreview.net/forum?id=fNI2fPyAfQ",
        "status": "search_result_checked_openreview_challenge_on_direct_open",
    },
    {
        "source_id": "cycle_world_2607_11836",
        "url": "https://arxiv.org/abs/2607.11836",
        "status": "primary_arxiv_opened_watch_only_external_video_world_model",
    },
    {
        "source_id": "confidently_wrong_2607_11414",
        "url": "https://arxiv.org/abs/2607.11414",
        "status": "primary_arxiv_opened_watch_only_internal_state_probe",
    },
    {
        "source_id": "snn_csp_parallel_tempering_2607_08897",
        "url": "https://arxiv.org/abs/2607.08897",
        "status": "primary_arxiv_opened_watch_only_not_cdls_exact_kernel_hook",
    },
    {
        "source_id": "semantic_scholar_ebt_route",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092",
        "status": "http_429",
    },
    {
        "source_id": "semantic_scholar_arm_ebm_route",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605",
        "status": "http_429",
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
        "status": "api_rate_limited_429",
        "promoted_delta": False,
        "note": "Fallback repeated EBT project, arXiv, OpenReview, Hugging Face, and GitHub pages already covered.",
    },
    {
        "paper": "ARM-EBM",
        "paper_id": "2512.15605",
        "route": "Semantic Scholar direct API plus public web-search fallback",
        "status": "api_rate_limited_429",
        "promoted_delta": False,
        "note": "Fallback did not expose a stronger local Exp5615-Exp5623 dependency.",
    },
)

CANDIDATE_FINDINGS: tuple[JsonDict, ...] = ()

DUPLICATE_SUPPRESSED_BASE: tuple[JsonDict, ...] = (
    {
        "source_id": "retain_or_adapt_2607_05609",
        "title": "To Retain or to Adapt? Generalizing Continual Learning",
        "url": "https://arxiv.org/abs/2607.05609",
        "reason": "Already accepted in the V507 planner block for Exp5616-Exp5618 drift-duration KAN work.",
    },
    {
        "source_id": "when_cl_requires_learning_2607_07847",
        "title": "When Does Continual Learning Require Learning",
        "url": "https://arxiv.org/abs/2607.07847",
        "reason": "Already accepted in the V507 planner block for spatial-vs-temporal drift controls.",
    },
    {
        "source_id": "loss_smoothing_openreview_puqcokv69j",
        "title": "Loss Smoothing for Continual Adaptation",
        "url": "https://openreview.net/forum?id=pUqcOkV69j",
        "reason": "Already accepted as a bounded workshop ablation hypothesis, not performance authority.",
    },
    {
        "source_id": "world_action_verifier_2604_01985",
        "title": "World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry",
        "url": "https://arxiv.org/abs/2604.01985",
        "reason": "Already accepted in the V507 planner block for Exp5619-Exp5621 transition-cycle verification.",
    },
    {
        "source_id": "cdls_openreview_fni2fpyafq",
        "title": "Accelerating Discrete Langevin Samplers via Continuous Intermediates",
        "url": "https://openreview.net/forum?id=fNI2fPyAfQ",
        "reason": "Already indexed as cDLS hypothesis context; exactness gates precede any Exp5623 timing claim.",
    },
    {
        "source_id": "ebt_arm_ebm_routes",
        "title": "EBT 2507.02092 and ARM-EBM 2512.15605 citation routes",
        "url": "https://arxiv.org/abs/2507.02092",
        "reason": "Citation routes were rechecked; no stronger local V507 dependency appeared.",
    },
    {
        "source_id": "constrained_decoding_finite_automata_2607_07026",
        "title": "Constrained Decoding for Diffusion Language Models via Efficient Inference over Finite Automata",
        "url": "https://arxiv.org/abs/2607.07026",
        "reason": "Already belongs to constrained-decoding context; V507 runtime work is native llama.cpp readiness, not diffusion-LM decoding.",
    },
    {
        "source_id": "matched_fp16_speedup_2607_11368",
        "title": "Decomposing Runtime, Kernel, and Quantization Speedups",
        "url": "https://arxiv.org/abs/2607.11368",
        "reason": "Already covered in V506 execution watch-only methodology; it cannot replace local GPU/offload receipts.",
    },
)

WATCH_ONLY_ITEMS: tuple[JsonDict, ...] = (
    {
        "source_id": "cycle_world_2607_11836",
        "title": "Cycle-World: Mitigating Error Accumulation in Long-term Video World Models",
        "url": "https://arxiv.org/abs/2607.11836",
        "classification": "watch_only_external_video_world_model",
        "evidence_status": "primary_arxiv_available_eccv_2026",
        "reason": "Reverse-prediction cycle consistency is WAV-adjacent but targets video generation, not local ARC action observations or exact transition-cycle hooks.",
    },
    {
        "source_id": "confidently_wrong_2607_11414",
        "title": "Confidently Wrong: Detecting Hallucinations in Financial QA from LLM Internal States",
        "url": "https://arxiv.org/abs/2607.11414",
        "classification": "watch_only_internal_state_probe",
        "evidence_status": "primary_arxiv_available",
        "reason": "Financial residual-stream probes do not attach to the native llama.cpp runtime certificate or exact V507 validators.",
    },
    {
        "source_id": "verifier_guided_twelve_tone_2607_11334",
        "title": "Verifier-Guided Twelve-Tone Composition",
        "url": "https://arxiv.org/abs/2607.11334",
        "classification": "watch_only_domain_mismatch",
        "evidence_status": "primary_arxiv_available",
        "reason": "The symbolic generate-verify-repair loop is music-domain evidence and adds no exact Carnot V507 hook.",
    },
    {
        "source_id": "ugp_asr_continual_learning_2607_11163",
        "title": "Unified Gradient Projection for Multilingual Low-Resource ASR",
        "url": "https://arxiv.org/abs/2607.11163",
        "classification": "watch_only_domain_mismatch",
        "evidence_status": "primary_arxiv_available_interspeech_2026",
        "reason": "Gradient projection for ASR fine-tuning would reopen model-weight update work, outside the spline-local KAN controller scope.",
    },
    {
        "source_id": "snn_csp_parallel_tempering_2607_08897",
        "title": "Breaking Local-Minimum Traps in SNN-Based CSP Solvers via Parallel Tempering",
        "url": "https://arxiv.org/abs/2607.08897",
        "classification": "watch_only_sampler_context",
        "evidence_status": "primary_arxiv_available",
        "reason": "Parallel tempering in SNN CSP solvers is not the existing cDLS kernel; adding it would change the Exp5622-Exp5623 dependency graph.",
    },
    {
        "source_id": "constrained_kan_ide_2607_11110",
        "title": "Neural Discovery of Memory and Nonlocal Kernels with Constrained KANs",
        "url": "https://arxiv.org/abs/2607.11110",
        "classification": "watch_only_domain_mismatch",
        "evidence_status": "primary_arxiv_available",
        "reason": "Hard-constrained KAN kernel discovery is not the exact nonstationary constraint-stream fixture or predictive-window KAN controller.",
    },
    {
        "source_id": "extropic_tsu_xtr_z1",
        "title": "Extropic TSU, XTR-0, X0, and Z1 writing",
        "url": "https://extropic.ai/writing",
        "classification": "watch_only_unavailable_hardware",
        "evidence_status": "public_writing_no_local_execution_path",
        "reason": "No authenticated local TSU path or matched Carnot sampler-speed receipt exists.",
    },
    {
        "source_id": "logical_intelligence_public_updates",
        "title": "Logical Intelligence Kona and Aleph public updates",
        "url": "https://logicalintelligence.com/",
        "classification": "watch_only_proprietary_system",
        "evidence_status": "public_pages_no_local_weights_or_benchmark_artifact",
        "reason": "Kona and Aleph remain proprietary context and cannot be V507 baselines or execution evidence.",
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


def _proposal_paths(root: Path) -> list[Path]:
    proposal_dir = root / "openspec/change-proposals"
    if not proposal_dir.exists():
        return []
    return sorted(proposal_dir.glob("*.md"))


def _dedupe_paths(root: Path) -> list[Path]:
    base = [
        RESEARCH_REFERENCES_RELATIVE_PATH,
        RESEARCH_COMPLETE_RELATIVE_PATH,
        VNEXT_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
        KNOWN_ISSUES_RELATIVE_PATH,
        EXP5613_RELATIVE_PATH,
    ]
    paths = [root / relative for relative in base]
    paths.extend(_proposal_paths(root))
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path)] = path
    return list(unique.values())


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
                "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                "exists": exists,
                "sha256": path_sha256(path) if exists else None,
            }
        )
    return "\n".join(chunks), checked


def _roadmap_context(root: Path) -> JsonDict:
    relative = (
        ROADMAP_NEXT_RELATIVE_PATH
        if (root / ROADMAP_NEXT_RELATIVE_PATH).exists()
        else ROADMAP_RELATIVE_PATH
    )
    parsed = yaml.safe_load(_read_text_if_present(root / relative)) or {}
    tasks = parsed.get("tasks", [])
    task_ids = [
        str(task.get("id")) for task in tasks if isinstance(task, Mapping) and task.get("id")
    ]
    return {
        "source": str(relative),
        "milestone": str(parsed.get("milestone", "")),
        "task_ids": task_ids,
    }


def _new_actionable_findings(root: Path, references_text: str) -> list[JsonDict]:
    if not _planner_marker_found(references_text) or EXECUTION_REFRESH_HEADING in references_text:
        return []
    _corpus_text, _checked = _dedupe_corpus(root)
    return [_clone_json(finding) for finding in CANDIDATE_FINDINGS]


def _duplicate_candidates() -> list[JsonDict]:
    return [_clone_json(row) for row in DUPLICATE_SUPPRESSED_BASE]


def build_experiment_mappings() -> list[JsonDict]:
    return [
        {
            "lane": "native llama.cpp CUDA runtime certificate",
            "experiment_ids": ["exp5615-native-llamacpp-cuda-runtime-certificate"],
            "source_ids": [
                "matched_fp16_speedup_2607_11368",
                "extropic_tsu_xtr_z1",
                "logical_intelligence_public_updates",
            ],
            "source_status": "duplicate_or_watch_only",
            "mapping": "Keep runtime certification local: model hashes, native build, offload, PID/GPU memory, and replay evidence only.",
        },
        {
            "lane": "exact nonstationary constraint-stream fixture",
            "experiment_ids": ["exp5616-exact-nonstationary-constraint-stream"],
            "source_ids": [
                "retain_or_adapt_2607_05609",
                "when_cl_requires_learning_2607_07847",
            ],
            "source_status": "duplicate_planner_context",
            "mapping": "Use the already-planned spatial-shift, temporal-drift, and duration axes; no new execution-time source changes the fixture.",
        },
        {
            "lane": "KAN critical-duration and predictive-window controller",
            "experiment_ids": [
                "exp5617-kan-critical-task-duration-map",
                "exp5618-predictive-window-kan-self-learning",
            ],
            "source_ids": [
                "retain_or_adapt_2607_05609",
                "when_cl_requires_learning_2607_07847",
                "loss_smoothing_openreview_puqcokv69j",
                "constrained_kan_ide_2607_11110",
                "ugp_asr_continual_learning_2607_11163",
            ],
            "source_status": "duplicate_or_watch_only",
            "mapping": "Keep the spline-local exact-gated KAN controller; do not adopt ASR fine-tuning, domain-specific KAN kernels, or model-weight updates.",
        },
        {
            "lane": "ARC forward/inverse transition cycle and live level attempt",
            "experiment_ids": [
                "exp5619-arc-forward-inverse-transition-cycle",
                "exp5620-arc-cycle-guarded-live-update-ab",
                "exp5621-arc-live-self-discovery-levelup-v507",
            ],
            "source_ids": ["world_action_verifier_2604_01985", "cycle_world_2607_11836"],
            "source_status": "duplicate_or_watch_only",
            "mapping": "Use the already-planned WAV-style action-effect cycle on live ARC observations; video-world cycle guidance remains external context.",
        },
        {
            "lane": "cDLS exact kernel audit and multiseed crossover",
            "experiment_ids": [
                "exp5622-cdls-exact-kernel-audit",
                "exp5623-cdls-multiseed-cpu-cuda-crossover",
            ],
            "source_ids": [
                "cdls_openreview_fni2fpyafq",
                "snn_csp_parallel_tempering_2607_08897",
            ],
            "source_status": "duplicate_or_watch_only",
            "mapping": "Audit the existing cDLS kernel for exact parity before timing; parallel tempering SNN CSP work would be a new mechanism.",
        },
        {
            "lane": "verification and hallucination context",
            "experiment_ids": ["exp5615-native-llamacpp-cuda-runtime-certificate"],
            "source_ids": [
                "confidently_wrong_2607_11414",
                "verifier_guided_twelve_tone_2607_11334",
                "constrained_decoding_finite_automata_2607_07026",
            ],
            "source_status": "watch_only_no_local_hook",
            "mapping": "These sources do not attach to the runtime certificate or exact V507 validators, so they create no work.",
        },
    ]


def _honest_verdict(
    planner_marker_found: bool, accepted_findings: Sequence[Mapping[str, Any]]
) -> str:
    if not planner_marker_found:
        return "blocked: V507 planner refresh marker missing; source-delta append refused"
    if accepted_findings:
        return (
            f"complete: accepted {len(accepted_findings)} non-duplicate actionable V507 "
            "source deltas and kept retired scopes closed"
        )
    return "complete: no new non-duplicate actionable V507 source deltas; references left unchanged"


def _closed_scope_review() -> JsonDict:
    return {
        "parser_scope_reopened": False,
        "solve_verify_scope_reopened": False,
        "causal_memory_scope_reopened": False,
        "ptrm_scope_reopened": False,
        "sge_scope_reopened": False,
        "generated_text_scoring_reopened": False,
        "hardware_board_scope_reopened": False,
        "proprietary_tsu_kona_aleph_reopened": False,
        "unmatched_hardware_speedup_reopened": False,
        "operator_authorized_differentiator": None,
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    search_timestamp_utc: str | None = None,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
) -> JsonDict:
    references_text = _read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    planner_marker_found = _planner_marker_found(references_text)
    accepted_findings = _new_actionable_findings(root, references_text)
    _corpus_text, dedupe_checked = _dedupe_corpus(root)
    timestamp = search_timestamp_utc or datetime.now(UTC).replace(microsecond=0).isoformat()
    if timestamp.endswith("+00:00"):
        timestamp = timestamp[:-6] + "Z"
    duplicates = _duplicate_candidates()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "search_cutoff": SEARCH_CUTOFF,
        "search_timestamp_utc": timestamp,
        "result_path": str(RESULT_RELATIVE_PATH),
        "spec_refs": list(SPEC_REFS),
        "sources_checked": _clone_json(SOURCES_CHECKED),
        "source_link_checks": _clone_json(SOURCE_LINK_CHECKS),
        "citation_trails_checked": _clone_json(CITATION_TRAILS_CHECKED),
        "dedupe_corpus_checked": dedupe_checked,
        "marker_checks": {
            "planner_marker": PLANNER_MARKER,
            "planner_marker_found": planner_marker_found,
            "execution_refresh_heading": EXECUTION_REFRESH_HEADING,
            "execution_refresh_present": EXECUTION_REFRESH_HEADING in references_text,
        },
        "duplicate_checks": {
            "candidate_count": len(CANDIDATE_FINDINGS),
            "accepted_count": len(accepted_findings),
            "duplicates_suppressed_count": len(duplicates),
            "dedupe_sources_count": len(dedupe_checked),
        },
        "new_references_added": _clone_json(accepted_findings),
        "duplicates_suppressed": duplicates,
        "research_references_updated": bool(accepted_findings),
        "planner_marker_found": planner_marker_found,
        "experiment_mappings": build_experiment_mappings(),
        "watch_only_items": _clone_json(WATCH_ONLY_ITEMS),
        "closed_scopes_reopened": False,
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
        isinstance(artifact["new_references_added"], list), "new_references_added must be a list"
    )
    _require(
        isinstance(artifact["duplicates_suppressed"], list), "duplicates_suppressed must be a list"
    )
    _require(
        isinstance(artifact["experiment_mappings"], list), "experiment_mappings must be a list"
    )
    _require(isinstance(artifact["watch_only_items"], list), "watch_only_items must be a list")
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
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    references_text = references_path.read_text(encoding="utf-8")
    new_findings = _new_actionable_findings(root, references_text)
    if new_findings:  # pragma: no cover - no V507 execution candidate was accepted.
        raise RuntimeError("Exp5614 no-op receipt unexpectedly found accepted findings")
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
