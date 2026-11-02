"""Exp5604: ingest the V506 execution-time source delta.

Spec refs: REQ-REPORT-5604, SCENARIO-REPORT-5604,
SCENARIO-REPORT-5604-NOOP, SCENARIO-REPORT-5604-FIELD-PRINCIPLES.

This module records the execution-time literature sweep as an auditable receipt.
The live web search happens before this file is run; the code here preserves the
decision logic that matters afterward: start after the V506 planner marker,
deduplicate against Carnot's local ledgers, append at most one execution-refresh
block for truly actionable deltas, and keep retired scopes closed. That split is
deliberate because public search APIs drift, but the repo needs a stable record
of which sources were accepted, suppressed, or left watch-only.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5604_v506_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
EXP5603_RELATIVE_PATH = Path("results/experiment_5603_transition_v506.json")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5604_v506_source_delta_ingestion"
EXPERIMENT_ID = "exp5604-v506-source-delta-ingestion"
MILESTONE = "2026.07.506"
RUN_DATE = "20260714"
SEARCH_CUTOFF = "2026-07-14"
SCHEMA = "carnot.experiment_5604.v506_source_delta_ingestion.v1"
RANDOM_SEED = 5604
INFERENCE_SUBSTRATE = "web_and_repository_source_synthesis"
TERMINAL_PREFIXES = ("complete:", "blocked:")

PLANNER_MARKER = "## V506 Planner Refresh - 20260714"
PLANNER_MARKER_COMPACT = "## V506 Planner Refresh - 20260714".replace("-", "")
EXECUTION_REFRESH_HEADING = "## V506 Execution Refresh - 20260714"
EXECUTION_REFRESH_END = "<!-- V506-EXECUTION-REFRESH-20260714-END -->"

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
    "inference_substrate": "provenance names the evidence substrate",
    "honest_verdict": "no-op is terminal",
}

SPEC_REFS = (
    "REQ-REPORT-5604",
    "SCENARIO-REPORT-5604",
    "SCENARIO-REPORT-5604-NOOP",
    "SCENARIO-REPORT-5604-FIELD-PRINCIPLES",
)

SOURCES_CHECKED: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "queries": [
            "EBM verification and reasoning",
            "neural CSPs and constraint satisfaction",
            "Ising ML and hardware-accelerated sampling",
            "hallucination mitigation and verifier reasoning",
            "Kolmogorov-Arnold Networks and continual constraint learning",
            "constrained generation, constrained decoding, and grammar methods",
        ],
        "status": "checked_primary_pages_and_arxiv_api",
        "notable_new_items": [
            "arXiv:2607.11116",
            "arXiv:2607.10128",
            "arXiv:2607.10044",
            "arXiv:2607.11110",
            "arXiv:2607.11258",
            "arXiv:2607.11368",
        ],
    },
    {
        "surface": "OpenReview",
        "queries": [
            "cDLS OpenReview ProbML 2026 workshop",
            "Energy-Based Transformers",
            "Spilled Energy",
            "verifier-constrained flow expansion",
            "constrained generation",
        ],
        "status": "checked_duplicates_or_workshop_watch_context",
    },
    {
        "surface": "Semantic Scholar",
        "queries": ["arXiv:2507.02092 citations", "arXiv:2512.15605 citations"],
        "status": "direct_api_returned_429_during_execution; search_fallback_checked",
    },
    {
        "surface": "Hugging Face Papers",
        "queries": [
            "energy-based models",
            "constrained decoding",
            "KAN",
            "hallucination mitigation",
            "arXiv:2607.11116",
        ],
        "status": "checked_no_stronger_local_artifact_than_arxiv_primary",
    },
    {
        "surface": "GitHub discovery",
        "queries": [
            "lazy-identity-deq",
            "FlashTrie",
            "TreeThink",
            "constrained decoding GPU",
            "energy-based reasoning",
        ],
        "status": "checked_code_availability_and_local_fit",
    },
    {
        "surface": "Extropic writing",
        "queries": ["TSU", "XTR-0", "X0", "Z1", "thermodynamic computing"],
        "status": "checked_watch_only_no_authenticated_local_tsu",
    },
    {
        "surface": "Logical Intelligence public pages",
        "queries": ["Kona", "Aleph", "automatic formal verification", "Sudoku EBM"],
        "status": "checked_watch_only_proprietary_no_local_baseline",
    },
    {
        "surface": "local Carnot ledgers",
        "queries": [
            "research-references.md",
            "research-complete.yaml",
            "openspec/change-proposals/*.md",
            "ops/exclusion_manifest.yaml",
            "ops/known-issues.md",
            "results/experiment_5603_transition_v506.json",
        ],
        "status": "checked",
    },
)

SOURCE_LINK_CHECKS: tuple[JsonDict, ...] = (
    {
        "source_id": "lazy_identity_deq_2607_11116",
        "url": "https://arxiv.org/abs/2607.11116",
        "status": "primary_arxiv_opened",
    },
    {
        "source_id": "lazy_identity_deq_code",
        "url": "https://github.com/joyjeet-singh/lazy-identity-deq",
        "status": "github_repository_opened",
    },
    {
        "source_id": "energy_guided_recursive_model_2607_10128",
        "url": "https://arxiv.org/abs/2607.10128",
        "status": "primary_arxiv_opened_watch_only_retired_ptrm_adjacent",
    },
    {
        "source_id": "flashtrie_2607_10044",
        "url": "https://arxiv.org/abs/2607.10044",
        "status": "primary_arxiv_opened_watch_only_code_unavailable",
    },
    {
        "source_id": "constrained_kan_ide_2607_11110",
        "url": "https://arxiv.org/abs/2607.11110",
        "status": "primary_arxiv_opened_watch_only_domain_mismatch",
    },
    {
        "source_id": "treethink_2607_11258",
        "url": "https://arxiv.org/abs/2607.11258",
        "status": "primary_arxiv_opened_watch_only_nonlocal_theorem_proving",
    },
    {
        "source_id": "matched_fp16_speedup_2607_11368",
        "url": "https://arxiv.org/abs/2607.11368",
        "status": "primary_arxiv_opened_watch_only_methodology_context",
    },
    {
        "source_id": "semantic_scholar_ebt_route",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092",
        "status": "http_429_rate_limited_no_citation_count_claim",
    },
    {
        "source_id": "semantic_scholar_arm_ebm_route",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605",
        "status": "http_429_rate_limited_no_citation_count_claim",
    },
)

CITATION_TRAILS_CHECKED: tuple[JsonDict, ...] = (
    {
        "paper": "Energy-Based Transformers",
        "paper_id": "2507.02092",
        "route": "Semantic Scholar public API and web-search fallback",
        "status": "api_rate_limited_429",
        "promoted_delta": False,
        "note": (
            "The route was checked directly, but rate limiting prevented a fresh count. "
            "Search fallback exposed only already-indexed EBT project, arXiv, OpenReview, "
            "Hugging Face, and code pages."
        ),
    },
    {
        "paper": "ARM-EBM",
        "paper_id": "2512.15605",
        "route": "Semantic Scholar public API and web-search fallback",
        "status": "api_rate_limited_429",
        "promoted_delta": False,
        "note": (
            "The route was checked directly, but rate limiting prevented a fresh count. "
            "Search fallback did not produce a stronger local Exp5605-Exp5611 dependency."
        ),
    },
)

LAZY_IDENTITY_FINDING: JsonDict = {
    "source_id": "lazy_identity_deq_2607_11116",
    "title": (
        "The Equilibrium Is the Initialization: Lazy Identity Collapse in "
        "Physics-Structured Deep Equilibrium Reasoning"
    ),
    "arxiv_id": "2607.11116",
    "url": "https://arxiv.org/abs/2607.11116",
    "code_url": "https://github.com/joyjeet-singh/lazy-identity-deq",
    "classification": "accepted_actionable_delta",
    "evidence_status": "primary_arxiv_with_released_code_and_logs",
    "why_actionable": (
        "The paper turns a silent no-op iterative-inference failure into a concrete "
        "audit protocol: compare solved state to initialization, bypass the solver, "
        "substitute the start state, and verify iteration count correlates with "
        "difficulty. Exp5608 already has a KAN-only longitudinal self-learning slot; "
        "adding this diagnostic prevents a spline update from being credited when the "
        "downstream decision is unchanged or the learned component is ignored."
    ),
    "experiment_ids": ["exp5608-kan-longitudinal-self-learning"],
    "lanes": ["KAN-only longitudinal self-learning"],
    "dedupe_tokens": [
        "2607.11116",
        "The Equilibrium Is the Initialization",
        "Lazy Identity Collapse",
        "lazy-identity-deq",
    ],
}

CANDIDATE_FINDINGS: tuple[JsonDict, ...] = (LAZY_IDENTITY_FINDING,)

DUPLICATE_SUPPRESSED_BASE: tuple[JsonDict, ...] = (
    {
        "source_id": "scientistone_2605_26340",
        "title": "ScientistOne: Towards Human-Level Autonomous Research via Chain-of-Evidence",
        "url": "https://arxiv.org/abs/2605.26340",
        "reason": "Already accepted in the V506 planner block for Exp5605 response envelopes.",
    },
    {
        "source_id": "aggregate_invariants_2606_24421",
        "title": "Can Aggregate Invariants Accelerate Continuous Subgraph Matching?",
        "url": "https://arxiv.org/abs/2606.24421",
        "reason": "Already accepted in the V506 planner block for Exp5609 intermediate ledgers.",
    },
    {
        "source_id": "cdls_openreview_rgs15pixcl",
        "title": "Accelerating Discrete Langevin Samplers via Continuous Intermediates",
        "url": "https://openreview.net/forum?id=Rgs15piXcl",
        "reason": (
            "Already accepted in the V506 planner block as workshop-level motivation for "
            "Exp5611; it cannot be counted again as a demonstrated speedup."
        ),
    },
    {
        "source_id": "agentic_property_templates_2607_09072",
        "title": "Agentic Proof and Property-Based Testing via Property-Templates",
        "url": "https://arxiv.org/abs/2607.09072",
        "reason": "Already indexed in V505 execution and vNEXT context for exact predicate work.",
    },
    {
        "source_id": "deceptive_grounding_2607_09349",
        "title": "Deceptive Grounding",
        "url": "https://arxiv.org/abs/2607.09349",
        "reason": "Already indexed in V505 execution and vNEXT context as attribution stress.",
    },
    {
        "source_id": "asp_energised_classiclogic",
        "title": "ASP Energised and ClassicLogic",
        "url": "https://arxiv.org/abs/2607.08136",
        "reason": "Already covered in V502-V505 exact ASP/FSM source history.",
    },
    {
        "source_id": "gram2token_xgrammar_llguidance",
        "title": "Gram2Token, XGrammar, and llguidance",
        "url": "https://openreview.net/forum?id=h3K23f6tLU",
        "reason": "Already covered for grammar-table and structured-output context.",
    },
    {
        "source_id": "ebt_arm_ebm_routes",
        "title": "EBT 2507.02092 and ARM-EBM 2512.15605 citation routes",
        "url": "https://arxiv.org/abs/2507.02092",
        "reason": "Citation routes were checked; no stronger local Exp5605-Exp5611 hook appeared.",
    },
)

WATCH_ONLY_ITEMS: tuple[JsonDict, ...] = (
    {
        "source_id": "energy_guided_recursive_model_2607_10128",
        "title": "Energy-guided Recursive Model",
        "url": "https://arxiv.org/abs/2607.10128",
        "classification": "watch_only_retired_scope_adjacent",
        "evidence_status": "primary_arxiv_available",
        "reason": (
            "Explicit Hopfield-energy trajectory selection is relevant to recursive reasoning, "
            "but promoting it would reopen the PTRM/recursive-generator lane outside Exp5608's "
            "KAN-only scope. No Exp5605-Exp5611 dependency is created."
        ),
    },
    {
        "source_id": "flashtrie_2607_10044",
        "title": "FlashTrie: A GPU-Accelerated Constrained Beam Search for Generative Retrieval",
        "url": "https://arxiv.org/abs/2607.10044",
        "classification": "watch_only_unavailable_code_domain_mismatch",
        "evidence_status": "primary_arxiv_available_code_future_release",
        "reason": (
            "The CUDA trie design is useful constrained-decoding context, but its code is not "
            "available yet and the task is large-scale generative retrieval, not Carnot's local "
            "schema/llama.cpp evidence envelope."
        ),
    },
    {
        "source_id": "constrained_kan_ide_2607_11110",
        "title": "Neural Discovery of Memory and Nonlocal Kernels with Constrained KANs",
        "url": "https://arxiv.org/abs/2607.11110",
        "classification": "watch_only_domain_mismatch",
        "evidence_status": "primary_arxiv_available",
        "reason": (
            "Hard-constrained KANs are aligned with Carnot's long-term KAN discipline, but this "
            "paper targets integro-differential kernel discovery rather than Exp5608's ordered "
            "verification-session self-learning fixture."
        ),
    },
    {
        "source_id": "treethink_2607_11258",
        "title": "TreeThink: A Modular Tree Search Library for Mathematical Reasoning with LLMs",
        "url": "https://arxiv.org/abs/2607.11258",
        "classification": "watch_only_nonlocal_theorem_proving_system",
        "evidence_status": "primary_arxiv_and_code_available",
        "reason": (
            "The verifier-connected tree-search library is local-code-available, but adopting it "
            "would start a theorem-proving lane rather than the already-planned ASP/FSM exact "
            "residual extension. It stays context only."
        ),
    },
    {
        "source_id": "matched_fp16_speedup_2607_11368",
        "title": "Decomposing Runtime, Kernel, and Quantization Speedups",
        "url": "https://arxiv.org/abs/2607.11368",
        "classification": "watch_only_methodology_context",
        "evidence_status": "primary_arxiv_available",
        "reason": (
            "The matched-intermediate timing lesson is good methodology context, but it is not "
            "a sampler target and does not replace Exp5611's exact Ising quality-matched CPU/CUDA "
            "benchmark requirements."
        ),
    },
    {
        "source_id": "extropic_tsu_xtr_z1",
        "title": "Extropic TSU, XTR-0, X0, and Z1 writing",
        "url": "https://extropic.ai/writing",
        "classification": "watch_only_unavailable_hardware",
        "evidence_status": "public_writing_no_local_execution_path",
        "reason": "No authenticated local TSU path or matched Carnot speedup receipt exists.",
    },
    {
        "source_id": "logical_intelligence_kona_aleph",
        "title": "Logical Intelligence Kona and Aleph public pages",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "classification": "watch_only_proprietary_system",
        "evidence_status": "public_pages_no_local_weights_or_benchmark_artifact",
        "reason": (
            "Kona/Aleph remain proprietary architecture context. They cannot be baselines, "
            "execution routes, or speedup evidence for Exp5605-Exp5611."
        ),
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


def _execution_section(references_text: str) -> str:
    if EXECUTION_REFRESH_HEADING not in references_text:
        return ""
    section = references_text.split(EXECUTION_REFRESH_HEADING, 1)[1]
    return section.split(EXECUTION_REFRESH_END, 1)[0]


def _finding_present(haystack_text: str, finding: Mapping[str, Any]) -> bool:
    haystack = haystack_text.lower()
    return any(str(token).lower() in haystack for token in finding["dedupe_tokens"])


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
        EXP5603_RELATIVE_PATH,
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
    corpus_text, _checked = _dedupe_corpus(root)
    return [
        _clone_json(finding)
        for finding in CANDIDATE_FINDINGS
        if not _finding_present(corpus_text, finding)
    ]


def _existing_execution_findings(references_text: str) -> list[JsonDict]:
    section = _execution_section(references_text)
    return [
        _clone_json(finding)
        for finding in CANDIDATE_FINDINGS
        if section and _finding_present(section, finding)
    ]


def _duplicate_candidates(
    root: Path, accepted_findings: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    accepted_ids = {str(finding["source_id"]) for finding in accepted_findings}
    corpus_text, _checked = _dedupe_corpus(root)
    duplicates = [_clone_json(row) for row in DUPLICATE_SUPPRESSED_BASE]
    for finding in CANDIDATE_FINDINGS:
        if finding["source_id"] not in accepted_ids and _finding_present(corpus_text, finding):
            duplicates.append(
                {
                    "source_id": finding["source_id"],
                    "title": finding["title"],
                    "url": finding["url"],
                    "reason": (
                        "Already present in the full dedupe corpus, so no V506 execution "
                        "append was allowed."
                    ),
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
    kan_sources = _mapping_sources(accepted_findings, "KAN-only longitudinal self-learning")
    return [
        {
            "lane": "response evidence envelope",
            "experiment_ids": ["exp5605-raw-response-evidence-envelope"],
            "source_ids": ["scientistone_2605_26340"],
            "source_status": "duplicate_planner_context",
            "mapping": (
                "Preserve raw or losslessly compressed responses, hashes, prompts, parser "
                "versions, and exact-validator outcomes before expensive SOTA inference."
            ),
        },
        {
            "lane": "clean local-SOTA solve-versus-verify panel",
            "experiment_ids": ["exp5606-clean-sota-solve-verify-evidence-panel"],
            "source_ids": [
                "ebt_arm_ebm_routes",
                "flashtrie_2607_10044",
                "tree_search_and_formal_verifier_context",
            ],
            "source_status": "duplicate_or_watch_only",
            "mapping": (
                "Retain exact validators and evidence envelopes. Do not replace local "
                "llama.cpp/schema evidence with unavailable constrained-decoding engines or "
                "formal theorem-proving systems."
            ),
        },
        {
            "lane": "property-template exact residual extension",
            "experiment_ids": ["exp5607-property-template-exact-residual-extension"],
            "source_ids": [
                "agentic_property_templates_2607_09072",
                "deceptive_grounding_2607_09349",
                "treethink_2607_11258",
            ],
            "source_status": "duplicate_planner_context_plus_watch_only",
            "mapping": (
                "Keep deterministic predicate mining bounded to residual rows from a clean "
                "Exp5606 panel; no theorem-proving lane or generated-text scorer is introduced."
            ),
        },
        {
            "lane": "KAN-only longitudinal self-learning",
            "experiment_ids": ["exp5608-kan-longitudinal-self-learning"],
            "source_ids": kan_sources,
            "source_status": (
                "accepted_execution_delta" if kan_sources else "duplicate_or_watch_only"
            ),
            "mapping": (
                "Add lazy-identity/no-op diagnostics to the existing frozen, shuffled-order, "
                "always-update, exact-gated, poison, and rollback arms. Credit learning only "
                "when update substitution changes independently measured held-out behavior."
            ),
        },
        {
            "lane": "ARC filter intermediate-invariance A/B",
            "experiment_ids": ["exp5609-arc-filter-intermediate-invariance-ab"],
            "source_ids": ["aggregate_invariants_2606_24421"],
            "source_status": "duplicate_planner_context",
            "mapping": (
                "Measure candidates, actions, states, expansions, level gains, and wall time; "
                "candidate reduction alone remains non-claim evidence."
            ),
        },
        {
            "lane": "unconditional ARC live-agent levelup",
            "experiment_ids": ["exp5610-unconditional-live-agent-levelup-attempt"],
            "source_ids": ["current_live_baseline", "energy_guided_recursive_model_2607_10128"],
            "source_status": "planned_baseline_plus_watch_only",
            "mapping": (
                "Run the live self-discovery attempt without source reads or per-game adapters. "
                "Energy-guided recursive-model context does not reopen PTRM as a generator."
            ),
        },
        {
            "lane": "cDLS matched CPU/CUDA benchmark",
            "experiment_ids": ["exp5611-cdls-matched-cpu-cuda-benchmark"],
            "source_ids": ["cdls_openreview_rgs15pixcl", "matched_fp16_speedup_2607_11368"],
            "source_status": "duplicate_planner_context_plus_watch_only_methodology",
            "mapping": (
                "Use identical targets, seeds, schedules, samples, quality diagnostics, and "
                "device receipts. Watch-only timing-methodology context cannot create a "
                "sampler speedup claim."
            ),
        },
    ]


def render_execution_refresh_block(findings: Sequence[Mapping[str, Any]], *, run_date: str) -> str:
    lines = [
        f"## V506 Execution Refresh - {run_date}",
        "",
        "Execution-time sweep after the `.506` planner marker checked arXiv primary "
        "pages/API results, OpenReview, Semantic Scholar EBT/ARM-EBM routes, Hugging "
        "Face Papers, GitHub discovery, Extropic writing, Logical Intelligence public "
        "pages, the full local reference/completion/proposal ledger, the exclusion "
        "manifest, and known-issues scope notes. Only non-duplicate actionable deltas "
        "are listed below.",
        "",
        "### New actionable deltas",
    ]
    for finding in findings:
        if finding["source_id"] == "lazy_identity_deq_2607_11116":
            lines.append(
                "- **{title}** - arXiv:{arxiv_id}, {url}; code/logs {code_url}. "
                "The paper documents a failure mode where an implicit solver is "
                "effectively ignored, and recommends substitution/no-op diagnostics. "
                "Carnot hook: Exp5608 should include a lazy-identity guard for the "
                "KAN-only longitudinal learner: compare updated-vs-substituted spline "
                "state, verify held-out behavior changes under update substitution, and "
                "refuse self-learning credit when the active component is bypassed. "
                "This does not reopen causal memory, PTRM, SGE, generated-text scoring, "
                "or hardware-board scopes.".format(**finding)
            )
    lines.extend(
        [
            "",
            "### Execution impact",
            "- **Plan impact:** No experiment-graph edit is required. The accepted delta maps "
            "only to the already-planned Exp5608 KAN-only longitudinal self-learning lane.",
            "- **Duplicates suppressed:** ScientistOne, aggregate invariants, cDLS, "
            "property-template testing, Deceptive Grounding, ASP/FSM, grammar-decoding, "
            "EBT/ARM-EBM, prior KAN, and prior hardware contexts were already covered.",
            "- **Closed scope:** closed_scopes_reopened=false. Retired parser, causal-memory, "
            "PTRM, SGE, generated-text scoring, proprietary TSU/Kona/Aleph execution, and "
            "unmatched hardware-speedup lanes remain closed.",
            "- **Watch-only/excluded:** Energy-guided Recursive Model, FlashTrie, constrained "
            "KAN IDE kernel discovery, TreeThink, matched-FP16 speedup attribution, Extropic, "
            "and Logical Intelligence were checked but not promoted as executable `.506` "
            "dependencies.",
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
        return "blocked: V506 planner refresh marker missing; source-delta append refused"
    if accepted_findings:
        return (
            f"complete: accepted {len(accepted_findings)} non-duplicate actionable V506 "
            "source delta and kept retired scopes closed"
        )
    return "complete: no new non-duplicate actionable V506 source deltas; references left unchanged"


def _closed_scope_review() -> JsonDict:
    return {
        "parser_scope_reopened": False,
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
    existing_findings = _existing_execution_findings(references_text)
    new_findings = _new_actionable_findings(root, references_text)
    accepted_findings = existing_findings or new_findings
    _corpus_text, dedupe_checked = _dedupe_corpus(root)
    timestamp = search_timestamp_utc or datetime.now(UTC).replace(microsecond=0).isoformat()
    if timestamp.endswith("+00:00"):
        timestamp = timestamp[:-6] + "Z"
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
        "new_references_added": _clone_json(accepted_findings),
        "duplicates_suppressed": _duplicate_candidates(root, accepted_findings),
        "research_references_updated": bool(accepted_findings),
        "planner_marker_found": planner_marker_found,
        "experiment_mappings": build_experiment_mappings(accepted_findings),
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
    if new_findings:
        references_path.write_text(
            references_text.rstrip()
            + "\n\n"
            + render_execution_refresh_block(new_findings, run_date=run_date),
            encoding="utf-8",
        )
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
