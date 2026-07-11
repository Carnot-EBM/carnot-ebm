"""Exp5565: ingest the V504 execution-time source delta.

Spec refs: REQ-REPORT-5565, SCENARIO-REPORT-5565,
SCENARIO-REPORT-5565-NOOP, SCENARIO-REPORT-5565-FIELD-PRINCIPLES.

The live literature search happens before this module is run. This module keeps
the result reproducible: it records which source surfaces were checked,
deduplicates the accepted finding against the full reference file, appends the
short execution refresh only when the finding is still new, and writes the JSON
receipt. That split matters because search rankings and public APIs drift, but
the conductor needs a stable artifact it can audit later.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5565_v504_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")

EXPERIMENT = "experiment_5565_v504_source_delta_ingestion"
EXPERIMENT_ID = "exp5565-v504-source-delta-ingestion"
MILESTONE = "2026.07.504"
RUN_DATE = "20260711"
SEARCH_CUTOFF = "2026-07-11"
SCHEMA = "carnot.experiment_5565.v504_source_delta_ingestion.v1"
RANDOM_SEED = 5565
INFERENCE_SUBSTRATE = "web_and_repository_source_synthesis"
TERMINAL_PREFIXES = ("complete:", "blocked:")

PLANNER_MARKER = "## V504 Planner Refresh - 20260711"
PLANNER_MARKER_COMPACT = PLANNER_MARKER.replace("-", "")
EXECUTION_REFRESH_HEADING = "## V504 Execution Refresh - 20260711"
EXECUTION_REFRESH_END = "<!-- V504-EXECUTION-REFRESH-20260711-END -->"

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
    "planner_marker_found": (
        "Bare boolean proving the V504 planner baseline was found before dedupe or reference mutation."
    ),
    "search_cutoff": (
        "Exact execution-time cutoff prevents later source drift from being misattributed to this receipt."
    ),
    "sources_checked": (
        "Lists each public and local source surface checked so absence or acceptance of deltas is auditable."
    ),
    "primary_sources_checked": (
        "Separates primary source URLs from secondary mirrors so actionability does not depend on summaries alone."
    ),
    "new_references_added": (
        "Contains only non-duplicate actionable findings accepted into the V504 execution refresh."
    ),
    "duplicates_suppressed": (
        "Names source hits already covered by earlier Carnot reference history or the V504 planner block."
    ),
    "citation_trails_checked": (
        "Records EBT/ARM-EBM citation-route status without fabricating citation deltas during public API rate limits."
    ),
    "research_references_updated": (
        "Bare boolean saying whether an execution refresh block is present because at least one accepted delta exists."
    ),
    "experiment_mappings": (
        "Maps accepted or retained source context to planned exp5566-exp5576 lanes without changing the roadmap."
    ),
    "watch_only_items": (
        "Keeps unavailable hardware, proprietary systems, and non-local or training-only work visible without promoting them to execution dependencies."
    ),
    "closed_scopes_reopened": (
        "Bare boolean proving duplicate, watch-only, proprietary, retired, and excluded scopes stayed closed."
    ),
    "inference_substrate": (
        "Must equal web_and_repository_source_synthesis because Exp5565 combines live source lookup with repository dedupe and emits no model or hardware measurement."
    ),
    "honest_verdict": (
        "Terminal summary starting with complete: or blocked: that distinguishes accepted deltas from no-op dedupe."
    ),
}

SPEC_REFS = (
    "REQ-REPORT-5565",
    "SCENARIO-REPORT-5565",
    "SCENARIO-REPORT-5565-NOOP",
    "SCENARIO-REPORT-5565-FIELD-PRINCIPLES",
)

SOURCES_CHECKED: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "queries": [
            "EBM reasoning and verification",
            "neural CSPs and lattice deduction",
            "Ising ML and accelerated sampling",
            "hallucination mitigation and verifier reliability",
            "KAN verification and online learning",
            "energy-guided decoding",
            "continual learning and self-evolving agents",
        ],
        "status": "checked",
    },
    {
        "surface": "OpenReview",
        "queries": ["EBT", "Energy-Guided Test-Time Scaling", "LLM-as-a-Verifier", "neural CSP"],
        "status": "checked_browser_challenge_public_snippets_only",
    },
    {
        "surface": "Semantic Scholar",
        "queries": ["arXiv:2507.02092 citations", "arXiv:2512.15605 citations"],
        "status": "checked_api_rate_limited_http_429",
    },
    {
        "surface": "Hugging Face Papers",
        "queries": ["2607.05391", "2607.03726", "2605.09998", "2607.07436"],
        "status": "checked",
    },
    {
        "surface": "GitHub",
        "queries": [
            "lattice-deduction-transformers",
            "llm-as-a-verifier",
            "continual-harness",
            "SelfMem",
        ],
        "status": "checked",
    },
    {
        "surface": "Extropic writing",
        "queries": ["X0", "XTR-0", "Z1", "TSU", "THRML"],
        "status": "checked_watch_only",
    },
    {
        "surface": "Logical Intelligence public pages",
        "queries": ["Kona", "Aleph", "automatic formal verification", "formal reasoning benchmarks"],
        "status": "checked_watch_only",
    },
    {
        "surface": "local Carnot reference history",
        "queries": ["full research-references.md", "V504 planner block", "exclusion manifest"],
        "status": "checked",
    },
)

PRIMARY_SOURCES_CHECKED: tuple[JsonDict, ...] = (
    {
        "source_id": "blind_curator_2607_07436",
        "url": "https://arxiv.org/abs/2607.07436",
        "status": "accepted_non_duplicate_actionable_delta",
    },
    {
        "source_id": "blind_curator_hf_2607_07436",
        "url": "https://huggingface.co/papers/2607.07436",
        "status": "secondary_confirmation_checked",
    },
    {
        "source_id": "lattice_deduction_transformers_2605_08605",
        "url": "https://arxiv.org/abs/2605.08605",
        "status": "watch_only_scope_expansion",
    },
    {
        "source_id": "lattice_deduction_transformers_github",
        "url": "https://github.com/lcrh/lattice-deduction-transformers",
        "status": "watch_only_reproduction_code_present",
    },
    {
        "source_id": "fineverify_2606_00660",
        "url": "https://arxiv.org/abs/2606.00660",
        "status": "watch_only_verifier_variant_not_v504_arm",
    },
    {
        "source_id": "caps_2605_15513",
        "url": "https://arxiv.org/abs/2605.15513",
        "status": "watch_only_verifier_variant_not_v504_arm",
    },
    {
        "source_id": "ruverbench_2606_29920",
        "url": "https://arxiv.org/abs/2606.29920",
        "status": "watch_only_external_rubric_benchmark",
    },
    {
        "source_id": "semantic_scholar_ebt_citations",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations",
        "status": "http_429_rate_limited",
    },
    {
        "source_id": "semantic_scholar_arm_ebm_citations",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations",
        "status": "http_429_rate_limited",
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
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations?fields=title,year,url,externalIds&limit=20",
        "status": "http_429_rate_limited",
        "promoted_delta": False,
        "note": "No citation-count or citing-paper delta was fabricated while the public API was rate-limited.",
    },
    {
        "paper": "ARM-EBM",
        "paper_id": "2512.15605",
        "route": "Semantic Scholar public citation API",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations?fields=title,year,url,externalIds&limit=20",
        "status": "http_429_rate_limited",
        "promoted_delta": False,
        "note": "No citation-count or citing-paper delta was fabricated while the public API was rate-limited.",
    },
)

BLIND_CURATOR_FINDING: JsonDict = {
    "source_id": "blind_curator_2607_07436",
    "title": "The Blind Curator: How a Biased Judge Silently Disables Skill Retirement in Self-Evolving Agents",
    "arxiv_id": "2607.07436",
    "url": "https://arxiv.org/abs/2607.07436",
    "secondary_url": "https://huggingface.co/papers/2607.07436",
    "classification": "accepted_actionable_delta",
    "why_actionable": (
        "The paper isolates false-pass judge bias as a hidden failure mode for self-evolving skill "
        "retirement and proposes a cheap defect-injection audit. V504's memory-policy tournament "
        "and delayed promotion gate already require poisoning, stale-memory, rollback, and false-accept "
        "stress, so the accepted action is to add a false-pass retirement audit without reopening "
        "cross-family CSL or external LLM-judge authority."
    ),
    "experiment_ids": [
        "exp5569-causal-memory-policy-tournament",
        "exp5572-gated-delayed-regression-promotion",
    ],
    "lanes": ["reset-free continuous self-learning stress"],
    "dedupe_tokens": [
        "The Blind Curator",
        "2607.07436",
        "biased judge",
        "skill retirement",
    ],
}

CANDIDATE_FINDINGS: tuple[JsonDict, ...] = (BLIND_CURATOR_FINDING,)

DUPLICATE_SUPPRESSED_BASE: tuple[JsonDict, ...] = (
    {
        "source_id": "selfmem_2607_03726",
        "title": "SelfMem: Self-Optimizing Memory for AI Agents",
        "url": "https://arxiv.org/abs/2607.03726",
        "reason": "Already accepted in the V504 planner block for Exp5569 memory-policy search.",
    },
    {
        "source_id": "continual_harness_2605_09998",
        "title": "Continual Harness: Online Adaptation for Self-Improving Foundation Agents",
        "url": "https://arxiv.org/abs/2605.09998",
        "reason": "Already accepted in the V504 planner block for reset-free harness adaptation.",
    },
    {
        "source_id": "llm_as_verifier_2607_05391",
        "title": "LLM-as-a-Verifier: A General-Purpose Verification Framework",
        "url": "https://arxiv.org/abs/2607.05391",
        "reason": "Already accepted in the V504 planner block for Exp5567 verifier arm design.",
    },
    {
        "source_id": "verification_horizon_2606_26300",
        "title": "The Verification Horizon: No Silver Bullet for Coding Agent Rewards",
        "url": "https://arxiv.org/abs/2606.26300",
        "reason": "Already accepted in the V504 planner block for Exp5568 co-evolution triggers.",
    },
    {
        "source_id": "ultrafast_kan_online_2602_02056",
        "title": "Ultrafast On-chip Online Learning via Spline Locality in KANs",
        "url": "https://arxiv.org/abs/2602.02056",
        "reason": "Already accepted in V504 planner context for Exp5570 active-spline updates.",
    },
    {
        "source_id": "ptrm_2605_19943_loop_2604_07822_sge_2603_02045",
        "title": "PTRM, Loop Think Generalize, and SGE",
        "url": "https://arxiv.org/abs/2605.19943",
        "reason": "Already accepted in the V504 planner block for Exp5574 and Exp5575-5576.",
    },
    {
        "source_id": "thermodynamic_pbit_hardware_prior",
        "title": "p-bit FPGA, thermodynamic scaling, Extropic, and Ising hardware prior",
        "url": "https://arxiv.org/abs/2607.00170",
        "reason": "Already present in Carnot hardware reference history or V504 planner context.",
    },
    {
        "source_id": "constraintbench_2602_22465",
        "title": "ConstraintBench",
        "url": "https://arxiv.org/abs/2602.22465",
        "reason": "Already indexed repeatedly as solver-verified constraint-reasoning context.",
    },
    {
        "source_id": "budget_curated_memory_2606_25115",
        "title": "Forget to Improve: Budget-Curated Memory",
        "url": "https://arxiv.org/abs/2606.25115",
        "reason": "Already indexed in V489/V490 history as memory-governance context.",
    },
    {
        "source_id": "seva_2606_29713_march_2603_24579",
        "title": "SEVA and MARCH hallucination verifier surfaces",
        "url": "https://arxiv.org/abs/2606.29713",
        "reason": "Already indexed or training/RL-scoped; not a new exact-validator V504 dependency.",
    },
    {
        "source_id": "ebt_arm_ebm_routes",
        "title": "EBT 2507.02092 and ARM-EBM 2512.15605 citation routes",
        "url": "https://arxiv.org/abs/2507.02092",
        "reason": "Already architecture context; execution-time Semantic Scholar API calls were rate-limited.",
    },
)

WATCH_ONLY_ITEMS: tuple[JsonDict, ...] = (
    {
        "source_id": "lattice_deduction_transformers_2605_08605",
        "title": "Lattice Deduction Transformers",
        "url": "https://arxiv.org/abs/2605.08605",
        "classification": "watch_only_scope_expansion",
        "reason": (
            "Reproduction code exists, but adopting a new trained puzzle/deduction model would expand V504 "
            "beyond exact ASP/FSM corpus and local-SOTA verification arms. Do not reopen WOPR/Sudoku-style "
            "benchmark lanes; retain only as future abstract-lattice architecture context."
        ),
    },
    {
        "source_id": "fineverify_caps_ruverbench_gradesql",
        "title": "FineVerify, CAPS, RuVerBench, and GradeSQL verifier variants",
        "url": "https://arxiv.org/abs/2606.00660",
        "classification": "watch_only_or_excluded",
        "reason": (
            "These add pairwise, partial-evidence, rubric, or domain-specific verifier variants, but Exp5567 "
            "is preregistered around exact ASP/FSM labels plus discrete, criteria, granular, and repeated arms. "
            "They should not silently broaden the panel."
        ),
    },
    {
        "source_id": "arch_eval_2607_03601",
        "title": "ArchEval: Measuring AI Agents as Computer Architects",
        "url": "https://arxiv.org/abs/2607.03601",
        "classification": "watch_only_external_benchmark",
        "reason": (
            "Useful artifact-integrity context, but it is an external benchmark rather than a matched CPU/CUDA "
            "sampler-quality receipt or board-local evidence for Exp5573."
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
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "classification": "watch_only_proprietary_system",
        "reason": "Public claims support architecture context only; no reproducible local Kona or Aleph baseline is available.",
    },
    {
        "source_id": "ac0_graphical_models_2607_08303_and_nqs_2607_02292",
        "title": "Learning AC0 under locally sampleable graphical models and NQS/PWO Ising optimization",
        "url": "https://arxiv.org/abs/2607.08303",
        "classification": "watch_only_theory_or_nonlocal_training",
        "reason": "Interesting Ising/sampling theory, but not an executable V504 matched sampler-quality or exact-verifier delta.",
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
    relative = ROADMAP_NEXT_RELATIVE_PATH if (root / ROADMAP_NEXT_RELATIVE_PATH).exists() else ROADMAP_RELATIVE_PATH
    parsed = yaml.safe_load((root / relative).read_text(encoding="utf-8")) or {}
    tasks = parsed.get("tasks", [])
    task_ids = [str(task.get("id")) for task in tasks if isinstance(task, Mapping) and task.get("id")]
    return {"source": str(relative), "milestone": str(parsed.get("milestone", "")), "task_ids": task_ids}


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
    return [_clone_json(finding) for finding in CANDIDATE_FINDINGS if not _finding_present(references_text, finding)]


def _existing_execution_findings(references_text: str) -> list[JsonDict]:
    section = _execution_section(references_text)
    return [_clone_json(finding) for finding in CANDIDATE_FINDINGS if section and _finding_present(section, finding)]


def _duplicate_candidates(references_text: str, accepted_findings: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    accepted_ids = {str(finding["source_id"]) for finding in accepted_findings}
    duplicates = [_clone_json(row) for row in DUPLICATE_SUPPRESSED_BASE]
    for finding in CANDIDATE_FINDINGS:
        if finding["source_id"] not in accepted_ids and _finding_present(references_text, finding):
            duplicates.append(
                {
                    "source_id": finding["source_id"],
                    "title": finding["title"],
                    "url": finding["url"],
                    "reason": "Already present in research-references.md, so no V504 execution append was allowed.",
                }
            )
    return duplicates


def _mapping_sources(accepted_findings: Sequence[Mapping[str, Any]], lane: str) -> list[str]:
    return [str(finding["source_id"]) for finding in accepted_findings if lane in finding.get("lanes", [])]


def build_experiment_mappings(accepted_findings: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    self_learning_sources = [
        "selfmem_2607_03726",
        "continual_harness_2605_09998",
        *_mapping_sources(accepted_findings, "reset-free continuous self-learning stress"),
    ]
    return [
        {
            "lane": "exact ASP/FSM near-miss corpus",
            "experiment_ids": ["exp5566-exact-asp-fsm-near-miss-corpus"],
            "source_ids": ["llm_as_verifier_2607_05391", "verification_horizon_2606_26300"],
            "source_status": "duplicate_planner_context",
            "mapping": "Use planner sources to shape exact solve-versus-verify rows; no new execution delta changed the corpus gate.",
        },
        {
            "lane": "local SOTA solve-versus-verify panel",
            "experiment_ids": [
                "exp5567-gated-local-sota-solve-verify-asymmetry",
                "exp5568-gated-verifier-coevolution-trigger",
            ],
            "source_ids": ["llm_as_verifier_2607_05391", "verification_horizon_2606_26300"],
            "source_status": "duplicate_planner_context",
            "mapping": "Keep exact-validator authority and the preregistered discrete/criteria/granular/repeated arms.",
        },
        {
            "lane": "reset-free continuous self-learning stress",
            "experiment_ids": [
                "exp5569-causal-memory-policy-tournament",
                "exp5571-gated-reset-free-sota-continual-harness",
                "exp5572-gated-delayed-regression-promotion",
            ],
            "source_ids": self_learning_sources,
            "source_status": (
                "accepted_plus_planner_context"
                if any(source == "blind_curator_2607_07436" for source in self_learning_sources)
                else "duplicate_planner_context"
            ),
            "mapping": (
                "Add a false-pass defect-injection and genuine-retirement-vs-cap-eviction audit to the memory "
                "policy and delayed promotion gates; keep exact feedback and rollback authoritative."
            ),
        },
        {
            "lane": "spline-local KAN update",
            "experiment_ids": ["exp5570-spline-local-kan-online-energy"],
            "source_ids": ["ultrafast_kan_online_2602_02056"],
            "source_status": "duplicate_planner_context",
            "mapping": "Retain active-spline-only update and rollback gates from the V504 planner.",
        },
        {
            "lane": "matched sampler hardware continuity",
            "experiment_ids": ["exp5573-matched-sampler-hardware-continuity"],
            "source_ids": ["thermodynamic_pbit_hardware_prior", "extropic_tsu_xtr_z1"],
            "source_status": "duplicate_or_watch_only",
            "mapping": "Require matched CPU/CUDA sample-quality evidence; no TSU, board, or paper-only speedup claim is promoted.",
        },
        {
            "lane": "PTRM and live ARC diversity",
            "experiment_ids": [
                "exp5574-ptrm-stochastic-generator-stage1",
                "exp5575-sge-anti-stagnation-live-precheck",
                "exp5576-gated-sge-live-levelup",
            ],
            "source_ids": ["ptrm_2605_19943_loop_2604_07822_sge_2603_02045"],
            "source_status": "duplicate_planner_context",
            "mapping": "Keep PTRM separate from the ordinary ARC floor and require live-agent self-discovery for Exp5576 credit.",
        },
    ]


def render_execution_refresh_block(findings: Sequence[Mapping[str, Any]], *, run_date: str) -> str:
    lines = [
        f"## V504 Execution Refresh - {run_date}",
        "",
        "Execution-time sweep after the `.504` planner refresh checked arXiv primary pages, "
        "OpenReview public surfaces, Hugging Face Papers, Semantic Scholar EBT/ARM-EBM routes, "
        "GitHub, Extropic writing, Logical Intelligence public pages, local duplicate history, "
        "the exclusion manifest, and known-issues scope notes. Only non-duplicate actionable "
        "deltas are listed below.",
        "",
        "### New actionable delta",
    ]
    for finding in findings:
        lines.append(
            "- **{title}** (arXiv:{arxiv_id}, {url}; HF mirror {secondary_url}): "
            "Add a false-pass defect-injection audit to Exp5569 and Exp5572 so memory-skill "
            "retirement cannot be credited by cap eviction or a biased judge that silently lets "
            "bad skills pass. This sharpens the existing poisoning, stale-memory, rollback, and "
            "unsafe-false-accept gates; it does not reopen cross-family CSL, external LLM-judge "
            "authority, GRPO/RL/fine-tuning, or any retired grammar scope.".format(**finding)
        )
    lines.extend(
        [
            "",
            "### Execution impact",
            "- **Plan impact:** No roadmap edit is required. The accepted delta maps to the already-planned "
            "reset-free memory-policy and delayed-promotion gates.",
            "- **Duplicates suppressed:** SelfMem, Continual Harness, LLM-as-a-Verifier, Verification Horizon, "
            "ultrafast KAN online learning, PTRM, Loop/Think/Generalize, SGE, p-bit/thermodynamic hardware, "
            "ConstraintBench, Budget-Curated Memory, SEVA, MARCH, EBT, and ARM-EBM were already covered or "
            "stayed non-promoted.",
            "- **Closed scope:** closed_scopes_reopened=false. Retired grammar-row completion, cross-family CSL, "
            "external generated-text scoring, broad GRPO/RL/fine-tuning, proprietary TSU/Kona/Aleph execution, "
            "and hardware speedup claims without matched timing remain closed.",
            "- **Watch-only/excluded:** Lattice Deduction Transformers, FineVerify/CAPS/RuVerBench/GradeSQL, "
            "ArchEval, Extropic TSU/XTR/Z1, Logical Intelligence Kona/Aleph, and fresh Ising/sampling theory "
            "were checked but not promoted as executable `.504` dependencies.",
            "",
            EXECUTION_REFRESH_END,
            "",
        ]
    )
    return "\n".join(lines)


def _honest_verdict(planner_marker_found: bool, accepted_findings: Sequence[Mapping[str, Any]]) -> str:
    if not planner_marker_found:
        return "blocked: V504 planner refresh marker missing; source-delta append refused"
    if accepted_findings:
        return "complete: accepted 1 non-duplicate actionable V504 source delta and kept closed scopes closed"
    return "complete: no new non-duplicate actionable V504 source deltas; references left unchanged"


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
            "cross_family_csl_reopened": False,
            "external_text_scorer_reopened": False,
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
    _require(isinstance(artifact["field_principles"], Mapping), "field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact["field_principles"]]
    _require(not missing_principles, f"field_principles missing: {missing_principles}")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "wrong inference_substrate")
    _require(artifact["closed_scopes_reopened"] is False, "closed_scopes_reopened must be false")
    _require(isinstance(artifact["planner_marker_found"], bool), "planner_marker_found must be bool")
    _require(isinstance(artifact["research_references_updated"], bool), "research_references_updated must be bool")
    _require(isinstance(artifact["sources_checked"], list), "sources_checked must be a list")
    _require(isinstance(artifact["primary_sources_checked"], list), "primary_sources_checked must be a list")
    _require(isinstance(artifact["citation_trails_checked"], list), "citation_trails_checked must be a list")
    _require(isinstance(artifact["watch_only_items"], list), "watch_only_items must be a list")
    _require(str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES), "honest_verdict lacks terminal prefix")


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
