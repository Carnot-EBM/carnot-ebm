"""Exp5637: ingest the V509 execution-time source delta.

Spec refs: REQ-REPORT-5637, SCENARIO-REPORT-5637-APPEND-DELTA,
SCENARIO-REPORT-5637-BLOCKED-MARKER,
SCENARIO-REPORT-5637-FIELD-PRINCIPLES.

This module records a literature freshness sweep as a deterministic artifact.
The web search itself is deliberately kept outside the code because public
indexes, citation counts, and search rankings drift. The code preserves the
research decision that downstream automation needs: the V509 planner marker
anchors the search window, one genuinely new executable source may be appended
once, duplicate ideas do not create work, and unavailable systems remain
watch-only rather than becoming claims.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5637_v509_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
EXP5636_RELATIVE_PATH = Path("results/experiment_5636_transition_v509.json")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5637_v509_source_delta_ingestion"
EXPERIMENT_ID = "exp5637-v509-source-delta-ingestion"
MILESTONE = "2026.07.509"
RUN_DATE = "20260714"
SEARCH_CUTOFF = "2026-07-14"
SCHEMA = "carnot.experiment_5637.v509_source_delta_ingestion.v1"
RANDOM_SEED = 5637
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CLOSED_SCOPES_REOPENED = False
TERMINAL_PREFIXES = ("complete:", "blocked:")

PLANNER_MARKER = "## V509 Planner Refresh - 20260714"
PLANNER_MARKER_COMPACT = PLANNER_MARKER.replace("-", "")
PLANNER_END_MARKER = "<!-- V509-PLANNER-REFRESH-20260714-END -->"
EXECUTION_REFRESH_HEADING = "## V509 Execution Refresh - 20260714"

ALLOWED_MAPPING_IDS = {
    "exp5638-fr11-gate-schema-corrigendum",
    "exp5639-anytime-valid-csl-independent-audit",
    "exp5640-fr11-shadow-pipeline-integration",
    "exp5641-arc-counterexample-executable-model",
    "exp5642-arc-executable-model-live-ab",
    "exp5643-arc-live-self-discovery-levelup-v509",
    "exp5644-two-axis-parallel-tempering-exact-audit",
    "exp5645-two-axis-tempering-hard-constraint-quality",
    "exp5646-two-axis-tempering-rust-parity",
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
    "reproducibility_checksum": "source set is stable",
    "honest_verdict": "a no-op is terminal",
}

SPEC_REFS = (
    "REQ-REPORT-5637",
    "SCENARIO-REPORT-5637-APPEND-DELTA",
    "SCENARIO-REPORT-5637-BLOCKED-MARKER",
    "SCENARIO-REPORT-5637-FIELD-PRINCIPLES",
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
            "executable world models counterexample repair",
        ],
        "status": "checked_arxiv_api_and_primary_abs_pages",
        "decision": "accepted Baba in Wonderland; other fresh hits were duplicate, non-local, or closed-scope",
    },
    {
        "surface": "OpenReview",
        "queries": [
            "executable world models counterexample",
            "world-model repair",
            "neural constraint satisfaction",
            "energy-based reasoning",
        ],
        "status": "checked_public_search_and_forum_results",
        "decision": "no cleaner V509 local dependency than arXiv primary sources and planner PatchWorld/ESBM pair",
    },
    {
        "surface": "Semantic Scholar",
        "queries": ["arXiv:2507.02092 citation route", "arXiv:2512.15605 citation route"],
        "status": "direct_api_returned_http_429_for_both_routes",
        "decision": "no citation-count or citation-delta claim is made",
    },
    {
        "surface": "Hugging Face Papers",
        "queries": [
            "Baba in Wonderland 2605.16725",
            "energy-based verification",
            "hallucination detection",
            "KAN conformal",
        ],
        "status": "checked_public_papers_search",
        "decision": "no HF-only executable V509 delta",
    },
    {
        "surface": "GitHub discovery/trending",
        "queries": [
            "Baba in Wonderland executable world models",
            "PatchWorld executable world models",
            "energy based model constraint satisfaction pushed after 2026-07-01",
        ],
        "status": "checked_github_api_search_and_direct_repository_links",
        "decision": "PatchWorld repository is duplicate planner evidence; no Baba implementation repository was found",
    },
    {
        "surface": "Extropic writing",
        "queries": ["TSU", "XTR-0", "X0", "Z1", "thermodynamic computing"],
        "status": "checked_public_writing_index_http_200",
        "decision": "public TSU context remains unavailable for Carnot execution",
    },
    {
        "surface": "Logical Intelligence public pages",
        "queries": ["Kona 1.0", "Aleph", "energy-based models", "constraint enforcement"],
        "status": "checked_public_site_http_200",
        "decision": "proprietary public updates do not provide local weights or receipts",
    },
    {
        "surface": "local Carnot ledgers",
        "queries": [
            "research-references.md after V509 marker",
            "research-complete.yaml",
            "openspec/change-proposals/*.md",
            "results/experiment_5636_transition_v509.json",
            "ops/exclusion_manifest.yaml",
            "ops/known-issues.md",
        ],
        "status": "checked",
        "decision": "Baba in Wonderland was absent locally; G-RRM and p-bit 2D-PT were duplicates",
    },
)

SOURCE_LINK_CHECKS: tuple[JsonDict, ...] = (
    {
        "source_id": "baba_in_wonderland_2605_16725",
        "url": "https://arxiv.org/abs/2605.16725",
        "status": "primary_arxiv_api_opened_accepted_non_duplicate",
    },
    {
        "source_id": "patchworld_2605_30880",
        "url": "https://arxiv.org/abs/2605.30880",
        "status": "primary_arxiv_api_opened_duplicate_planner_source",
    },
    {
        "source_id": "esbm_2606_07127",
        "url": "https://arxiv.org/abs/2606.07127",
        "status": "planner_source_duplicate_not_remapped",
    },
    {
        "source_id": "scratchworld_2606_31689",
        "url": "https://arxiv.org/abs/2606.31689",
        "status": "primary_arxiv_api_opened_watch_only_external_benchmark",
    },
    {
        "source_id": "confidently_wrong_internal_states_2607_11414",
        "url": "https://arxiv.org/abs/2607.11414",
        "status": "primary_arxiv_api_opened_watch_only_no_exact_v509_hook",
    },
    {
        "source_id": "g_rrm_2607_02491",
        "url": "https://arxiv.org/abs/2607.02491",
        "status": "primary_arxiv_api_opened_duplicate_solver_guidance_history",
    },
    {
        "source_id": "two_dimensional_pt_2506_14781",
        "url": "https://arxiv.org/abs/2506.14781",
        "status": "primary_arxiv_api_opened_duplicate_planner_source",
    },
    {
        "source_id": "pbit_2d_pt_2601_09037",
        "url": "https://arxiv.org/abs/2601.09037",
        "status": "primary_arxiv_api_opened_duplicate_prior_reference",
    },
    {
        "source_id": "semantic_scholar_ebt_route",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092",
        "status": "http_429_no_fresh_count_claim",
    },
    {
        "source_id": "semantic_scholar_arm_ebm_route",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605",
        "status": "http_429_no_fresh_count_claim",
    },
    {
        "source_id": "github_patchworld",
        "url": "https://github.com/HKBU-KnowComp/PatchWorld",
        "status": "http_200_duplicate_planner_repository",
    },
    {
        "source_id": "github_baba_query",
        "url": "https://api.github.com/search/repositories?q=%22Baba%20in%20Wonderland%22%20%22Executable%20World%20Models%22",
        "status": "total_count_0_no_repository_claim",
    },
    {
        "source_id": "extropic_writing",
        "url": "https://extropic.ai/writing",
        "status": "http_200_watch_only_no_local_tsu",
    },
    {
        "source_id": "logical_intelligence_public",
        "url": "https://logicalintelligence.com/",
        "status": "http_200_watch_only_proprietary",
    },
)

CITATION_TRAILS_CHECKED: tuple[JsonDict, ...] = (
    {
        "paper": "Energy-Based Transformers",
        "paper_id": "2507.02092",
        "route": "Semantic Scholar direct API plus web-search fallback",
        "status": "api_rate_limited_429",
        "promoted_delta": False,
        "note": "No citation-count update is claimed; existing EBT theory remains context only.",
    },
    {
        "paper": "ARM-EBM",
        "paper_id": "2512.15605",
        "route": "Semantic Scholar direct API plus web-search fallback",
        "status": "api_rate_limited_429",
        "promoted_delta": False,
        "note": "Search fallback did not expose a new exact local V509 hook.",
    },
)

CANDIDATE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "baba_in_wonderland_2605_16725",
        "title": "Baba in Wonderland: Online Self-Supervised Dynamics Discovery for Executable World Models",
        "url": "https://arxiv.org/abs/2605.16725",
        "published": "2026-05-16",
        "accepted_as": "non_duplicate_actionable_execution_delta",
        "evidence_boundary": "primary arXiv preprint; no external benchmark result is imported",
        "local_exact_hook": (
            "Use preservation-conflict counterexamples and class-aware transition coverage as "
            "controls inside the existing Exp5641 ARC transition-hypothesis patcher."
        ),
        "experiment_ids": [
            "exp5641-arc-counterexample-executable-model",
            "exp5642-arc-executable-model-live-ab",
        ],
        "dependency_graph_changed": False,
    },
)

DUPLICATE_SUPPRESSED_BASE: tuple[JsonDict, ...] = (
    {
        "source_id": "anytime_valid_crc_2602_04364",
        "title": "Anytime-Valid Conformal Risk Control",
        "url": "https://arxiv.org/abs/2602.04364",
        "reason": "Already accepted in the V509 planner block for Exp5639 anytime-valid audit.",
    },
    {
        "source_id": "conformal_selective_acting_2605_20270",
        "title": "Conformal Selective Acting",
        "url": "https://arxiv.org/abs/2605.20270",
        "reason": "Already promoted as V509 selective-release context; not remapped.",
    },
    {
        "source_id": "patchworld_2605_30880",
        "title": "PatchWorld: Gradient-Free Optimization of Executable World Models",
        "url": "https://arxiv.org/abs/2605.30880",
        "reason": "Already accepted in the V509 planner block for Exp5641.",
    },
    {
        "source_id": "esbm_2606_07127",
        "title": "Learning Explicit Behavioral Models with Adaptive Questions and World-Model Probes",
        "url": "https://arxiv.org/abs/2606.07127",
        "reason": "Already accepted in the V509 planner block for mechanism-question probes.",
    },
    {
        "source_id": "two_dimensional_pt_2506_14781",
        "title": "Two-dimensional Parallel Tempering for Constrained Optimization",
        "url": "https://arxiv.org/abs/2506.14781",
        "reason": "Already accepted in the V509 planner block for Exp5644-Exp5646.",
    },
    {
        "source_id": "pbit_2d_pt_2601_09037",
        "title": "Probabilistic Computers for MIMO Detection: From Sparsification to 2D Parallel Tempering",
        "url": "https://arxiv.org/abs/2601.09037",
        "reason": "Already indexed in Carnot's p-bit and 2D parallel tempering history.",
    },
    {
        "source_id": "g_rrm_2607_02491",
        "title": "G-RRM: Guiding Symbolic Solvers with Recurrent Reasoning Models",
        "url": "https://arxiv.org/abs/2607.02491",
        "reason": "Already indexed and completed as overwrite-capable solver guidance, outside the V509 dependency graph.",
    },
)

WATCH_ONLY_ITEMS: tuple[JsonDict, ...] = (
    {
        "source_id": "confidently_wrong_internal_states_2607_11414",
        "title": "Confidently Wrong: Detecting Hallucinations in Financial Question Answering from LLM Internal States",
        "url": "https://arxiv.org/abs/2607.11414",
        "classification": "watch_only_external_internal_state_probe",
        "evidence_status": "fresh_primary_arxiv_no_exact_local_v509_hook",
        "reason": "The result is a domain-specific residual-stream probe and would reopen internal-state hallucination scoring rather than support Exp5638-Exp5646 exact hooks.",
    },
    {
        "source_id": "scratchworld_2606_31689",
        "title": "ScratchWorld: Evaluating If World Models Compute Executable Consequences",
        "url": "https://arxiv.org/abs/2606.31689",
        "classification": "watch_only_external_benchmark",
        "evidence_status": "primary_arxiv_benchmark_no_local_scratch_vm_or_arc_receipts",
        "reason": "Changed-field F1 is useful context, but the benchmark substrate is external and not the agent-owned ARC receipt stream required by Exp5641.",
    },
    {
        "source_id": "github_baba_query",
        "title": "GitHub search for Baba in Wonderland implementation",
        "url": "https://api.github.com/search/repositories?q=%22Baba%20in%20Wonderland%22%20%22Executable%20World%20Models%22",
        "classification": "watch_only_no_public_repository_found",
        "evidence_status": "github_api_total_count_0",
        "reason": "No repository is needed or claimed; Carnot implements the local control idea if Exp5641 consumes the paper.",
    },
    {
        "source_id": "openreview_world_model_workshop_pages",
        "title": "OpenReview world-model and program-repair pages",
        "url": "https://openreview.net/search?term=executable%20world%20models%20counterexample",
        "classification": "watch_only_workshop_or_domain_context",
        "evidence_status": "public_openreview_search_no_cleaner_local_dependency",
        "reason": "Workshop and program-repair world-model pages do not supersede the accepted arXiv/local ARC hook.",
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
        "evidence_status": "public_pages_no_local_weights_or_reproducible_receipt",
        "reason": "Kona and Aleph remain proprietary context and cannot be V509 baselines or execution evidence.",
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
        root / EXP5636_RELATIVE_PATH,
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
    relative = (
        ROADMAP_NEXT_RELATIVE_PATH
        if (root / ROADMAP_NEXT_RELATIVE_PATH).exists()
        else ROADMAP_RELATIVE_PATH
    )
    parsed = yaml.safe_load(_read_text_if_present(root / relative)) or {}
    tasks = parsed.get("tasks", []) if isinstance(parsed, Mapping) else []
    task_ids = [
        str(task.get("id")) for task in tasks if isinstance(task, Mapping) and task.get("id")
    ]
    milestone = str(parsed.get("milestone", "")) if isinstance(parsed, Mapping) else ""
    return {"source": relative.as_posix(), "milestone": milestone, "task_ids": task_ids}


def _accepted_findings() -> list[JsonDict]:
    return [_clone_json(row) for row in CANDIDATE_FINDINGS]


def _duplicate_candidates() -> list[JsonDict]:
    return [_clone_json(row) for row in DUPLICATE_SUPPRESSED_BASE]


def build_experiment_mappings() -> list[JsonDict]:
    return [
        {
            "lane": "FR-11 gate contract, anytime audit, and shadow integration",
            "experiment_ids": [
                "exp5638-fr11-gate-schema-corrigendum",
                "exp5639-anytime-valid-csl-independent-audit",
                "exp5640-fr11-shadow-pipeline-integration",
            ],
            "source_ids": [
                "anytime_valid_crc_2602_04364",
                "conformal_selective_acting_2605_20270",
            ],
            "source_status": "duplicate_planner_context",
            "mapping": "Keep the planner-defined anytime certificate path; no execution-time source changed this lane.",
        },
        {
            "lane": "counterexample-patched ARC executable transition models",
            "experiment_ids": [
                "exp5641-arc-counterexample-executable-model",
                "exp5642-arc-executable-model-live-ab",
            ],
            "source_ids": [
                "patchworld_2605_30880",
                "esbm_2606_07127",
                "baba_in_wonderland_2605_16725",
                "scratchworld_2606_31689",
            ],
            "source_status": "one_new_actionable_plus_planner_duplicates_and_watch_only",
            "mapping": (
                "Baba's preservation-conflict control maps locally to Exp5641; Exp5642 may "
                "consume it only through the existing gated live-path A/B."
            ),
        },
        {
            "lane": "unconditional ARC live attempt",
            "experiment_ids": ["exp5643-arc-live-self-discovery-levelup-v509"],
            "source_ids": ["baba_in_wonderland_2605_16725"],
            "source_status": "advisory_only_no_dependency_change",
            "mapping": "The source delta does not gate or alter the mandatory live attempt.",
        },
        {
            "lane": "two-axis constrained tempering and Rust parity",
            "experiment_ids": [
                "exp5644-two-axis-parallel-tempering-exact-audit",
                "exp5645-two-axis-tempering-hard-constraint-quality",
                "exp5646-two-axis-tempering-rust-parity",
            ],
            "source_ids": [
                "two_dimensional_pt_2506_14781",
                "pbit_2d_pt_2601_09037",
            ],
            "source_status": "duplicate_planner_or_prior_reference",
            "mapping": "No new hardware or speedup claim is created; V509 remains exactness, quality, then parity.",
        },
    ]


def _honest_verdict(
    planner_marker_found: bool, accepted_findings: Sequence[Mapping[str, Any]]
) -> str:
    if not planner_marker_found:
        return "blocked: V509 planner refresh marker missing; source-delta append refused"
    if accepted_findings:
        return (
            f"complete: accepted {len(accepted_findings)} non-duplicate actionable V509 "
            "source delta and kept retired scopes closed"
        )
    return "complete: no new non-duplicate actionable V509 source deltas; references left unchanged"


def _closed_scope_review() -> JsonDict:
    return {
        "native_runtime_certificate_reopened": False,
        "external_generated_text_scorer_reopened": False,
        "solver_guidance_scope_reopened": False,
        "arc_epistemic_object_probe_reopened": False,
        "board_snn_or_hardware_speedup_reopened": False,
        "proprietary_tsu_kona_aleph_reopened": False,
        "operator_authorized_differentiator": None,
    }


def _normalize_timestamp(search_timestamp_utc: str | None) -> str:
    timestamp = search_timestamp_utc or datetime.now(UTC).replace(microsecond=0).isoformat()
    if timestamp.endswith("+00:00"):
        timestamp = timestamp[:-6] + "Z"
    return timestamp


def _execution_refresh_block(findings: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        EXECUTION_REFRESH_HEADING,
        "",
        (
            "Execution-time source check after the V509 planner marker accepted one "
            "non-duplicate local hook and left the dependency graph unchanged."
        ),
        "",
    ]
    for finding in findings:
        lines.extend(
            [
                (
                    f"- **{finding['title']}** - arXiv:2605.16725, {finding['url']}. "
                    "The paper treats preservation conflicts as structural signal when a "
                    "candidate update explains a new transition but breaks earlier ones. "
                    "Carnot hook: add preservation-conflict and class-aware counterexample "
                    "controls to Exp5641's generic ARC transition-hypothesis patcher using "
                    "only agent-owned observation/action receipts. Downstream Exp5642 may "
                    "consume the hook only if Exp5641 replay safety gates pass. Do not "
                    "import Baba rules, external code, or benchmark results as Carnot evidence."
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _insert_after_planner_block(references_text: str, block: str) -> str:
    end_index = references_text.find(PLANNER_END_MARKER)
    if end_index >= 0:
        insertion_index = end_index + len(PLANNER_END_MARKER)
        while insertion_index < len(references_text) and references_text[insertion_index] == "\n":
            insertion_index += 1
        prefix = references_text[:insertion_index].rstrip() + "\n\n"
        suffix = references_text[insertion_index:].lstrip("\n")
        return prefix + block + ("\n" + suffix if suffix else "")
    return references_text.rstrip() + "\n\n" + block


def _append_execution_refresh_if_needed(
    root: Path,
    planner_marker_found: bool,
    accepted_findings: Sequence[Mapping[str, Any]],
) -> bool:
    if not planner_marker_found or not accepted_findings:
        return False
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    references_text = _read_text_if_present(references_path)
    if EXECUTION_REFRESH_HEADING in references_text:
        return False
    updated = _insert_after_planner_block(references_text, _execution_refresh_block(accepted_findings))
    references_path.write_text(updated, encoding="utf-8")
    return True


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    search_timestamp_utc: str | None = None,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    research_references_updated: bool = False,
) -> JsonDict:
    references_text = _read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    planner_marker_found = _planner_marker_found(references_text)
    accepted_findings = _accepted_findings() if planner_marker_found else []
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
        "research_references_updated": bool(research_references_updated),
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
        isinstance(artifact["field_principles"], Mapping),
        "field_principles must be a mapping",
    )
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact["field_principles"], f"field_principles missing {field}")
        _require(str(artifact["field_principles"][field]).strip(), f"empty principle for {field}")
    _require(isinstance(artifact["planner_marker_found"], bool), "planner_marker_found bool")
    _require(artifact["closed_scopes_reopened"] is False, "closed_scopes_reopened must be false")
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate mismatch",
    )
    timestamp = str(artifact["search_timestamp_utc"])
    _require(timestamp.endswith("Z"), "timestamp must be UTC with Z suffix")
    _require(
        str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES),
        "honest_verdict must use terminal prefix",
    )
    for finding in artifact["new_references_added"]:
        experiment_ids = set(finding.get("experiment_ids", []))
        _require(experiment_ids <= ALLOWED_MAPPING_IDS, "finding maps outside Exp5638-Exp5646")
        _require(not finding.get("dependency_graph_changed", True), "finding changed dependency graph")
    for mapping in artifact["experiment_mappings"]:
        experiment_ids = set(mapping.get("experiment_ids", []))
        _require(experiment_ids <= ALLOWED_MAPPING_IDS, "mapping outside Exp5638-Exp5646")
    _require(artifact["reproducibility_checksum"] == payload_checksum(artifact), "checksum mismatch")
    _require(
        isinstance(artifact["sources_checked"], Sequence) and artifact["sources_checked"],
        "sources_checked must be non-empty",
    )
    _require(isinstance(artifact["duplicates_suppressed"], Sequence), "duplicates_suppressed list")
    _require(isinstance(artifact["watch_only_items"], Sequence), "watch_only_items list")


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    search_timestamp_utc: str | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    references_text = _read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_found = _planner_marker_found(references_text)
    accepted = _accepted_findings() if marker_found else []
    references_updated = _append_execution_refresh_if_needed(root, marker_found, accepted)
    artifact = build_artifact(
        root=root,
        search_timestamp_utc=search_timestamp_utc,
        duration_s=duration_s,
        research_references_updated=references_updated,
    )
    validate_artifact(artifact)
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--search-timestamp-utc", default=None)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    duration_s = max(0.0, time.perf_counter() - started)
    artifact = build_and_write_artifact(
        root=args.root,
        search_timestamp_utc=args.search_timestamp_utc,
        duration_s=duration_s,
    )
    print(json.dumps({"result_path": artifact["result_path"], "status": artifact["status"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
