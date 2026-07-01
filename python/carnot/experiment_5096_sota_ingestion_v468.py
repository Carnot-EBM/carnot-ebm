"""Exp 5096 V468 SOTA ingestion verifier.

Spec refs: REQ-REPORT-5096, SCENARIO-REPORT-5096,
SCENARIO-REPORT-5096-MISSING-REFERENCE.

This module verifies the V468 planner source set already present in
research-references.md and emits a deterministic JSON artifact. It does not run
local model inference, call Semantic Scholar, or edit conductor scripts.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any


RESULT_RELATIVE_PATH = "results/experiment_5096_sota_ingestion_v468.json"
REFERENCES_RELATIVE_PATH = "research-references.md"
HONEST_VERDICT = "success_sota_ingestion_v468_references_verified"
INFERENCE_SUBSTRATE = "literature_review_and_repo_inspection"
DURATION_S = 0.001
V468_SECTION_START = "<!-- V468-PLANNER-REFERENCES-START -->"
V468_SECTION_END = "<!-- V468-PLANNER-REFERENCES-END -->"
TERMINAL_PREFIXES = ("blocked_", "complete:", "complete_", "success:", "success_")
SPEC_REFS = [
    "REQ-REPORT-5096",
    "SCENARIO-REPORT-5096",
    "SCENARIO-REPORT-5096-MISSING-REFERENCE",
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "sources_checked",
    "references_section_found",
    "references_added_count",
    "semantic_scholar_status",
    "task_mapping",
    "planning_hooks",
    "background_only_sources",
    "flagged_adversarial",
    "field_principles",
    "spec_refs",
)

REQUIRED_USER_FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal prefix; success only when the V468 source set is verified."
    },
    "duration_s": {
        "principle": (
            "bounded repo/literature inspection duration; not a model-inference runtime claim."
        )
    },
    "inference_substrate": {
        "principle": (
            "literature_review_and_repo_inspection; no local model inference or live LLM claim."
        )
    },
    "sources_checked": {
        "principle": (
            "per-source evidence for arXiv, OpenReview, Hugging Face Papers/GitHub, "
            "Extropic, Semantic Scholar citation-lineage, and Logical Intelligence coverage."
        )
    },
    "references_section_found": {
        "principle": (
            "true only when V468 planner markers bracket the checked research-references.md "
            "section."
        )
    },
    "references_added_count": {
        "principle": (
            "zero when the V468 section already contains the required high-value sources; "
            "otherwise equals appended references."
        )
    },
    "semantic_scholar_status": {
        "principle": (
            "records the V468 section's Semantic Scholar EBT/ARM citation-lineage status "
            "without implying a fresh local API call."
        )
    },
    "task_mapping": {
        "principle": "maps each verified V468 source to one .468 task or background_only."
    },
    "planning_hooks": {"principle": "groups verified sources into concrete .468 execution hooks."},
    "background_only_sources": {
        "principle": (
            "names verified V468 sources intentionally kept as context rather than scheduled "
            "experiments."
        )
    },
    "flagged_adversarial": {
        "principle": "true if required sources are missing, fabricated, or claim local model inference."
    },
}

FIELD_PRINCIPLES = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "field_principles": {"principle": "principle annotations are part of the artifact contract."},
    "spec_refs": {"principle": "OpenSpec requirements and scenarios verified by this artifact."},
}

REQUIRED_CHANNELS = frozenset(
    {
        "arXiv",
        "OpenReview",
        "GitHub",
        "Hugging Face Papers",
        "Extropic",
        "Semantic Scholar",
        "Logical Intelligence",
    }
)

COVERAGE_MATCH_TOKENS = [
    "Search coverage: arXiv, OpenReview, Hugging Face Papers, GitHub",
    "Extropic, Logical Intelligence, and",
    "Scholar API returned HTTP 429",
    "citation-lineage notes below come from public",
]

REQUIRED_REFERENCE_CHECKS = [
    {
        "source_id": "beaver_prefix_bounds",
        "required_token": "arXiv:2512.05439",
        "title": "BEAVER: Efficient Deterministic LLM Verifier",
        "channels": ["arXiv", "GitHub"],
        "urls": [
            "https://arxiv.org/abs/2512.05439",
            "https://github.com/uiuc-focal-lab/Beaver",
        ],
        "spec_tokens": ["arXiv:2512.05439", "uiuc-focal-lab/Beaver"],
        "match_tokens": [
            "### BEAVER: Efficient Deterministic LLM Verifier",
            "- **Source:** arXiv:2512.05439 - https://arxiv.org/abs/2512.05439",
            "- **Code:** https://github.com/uiuc-focal-lab/Beaver",
            "deterministic verification, prefix constraints",
            "- **Carnot hook:** BEAVER computes deterministic probability bounds",
            "- **Actionability:** Build a small BEAVER-style prefix-bound verifier",
        ],
    },
    {
        "source_id": "graph_evidence_grounding",
        "required_token": "arXiv:2606.30247",
        "title": "Grounding LLM Reasoning Under Incomplete Graph Evidence",
        "channels": ["arXiv"],
        "urls": ["https://arxiv.org/abs/2606.30247"],
        "spec_tokens": ["arXiv:2606.30247", "graph-evidence grounding"],
        "match_tokens": [
            "### Grounding LLM Reasoning Under Incomplete Graph Evidence",
            "- **Source:** arXiv:2606.30247 - https://arxiv.org/abs/2606.30247",
            "evidence-relative reasoning, graph energies",
            "- **Carnot hook:** The paper frames reasoning under incomplete evidence",
            "- **Actionability:** Add a graph-evidence energy experiment",
        ],
    },
    {
        "source_id": "severa_self_evolving_agents",
        "required_token": "arXiv:2603.25111",
        "title": "SEVerA: Verified Synthesis of Self-Evolving Agents",
        "channels": ["arXiv"],
        "urls": ["https://arxiv.org/abs/2603.25111"],
        "spec_tokens": ["arXiv:2603.25111", "SEVerA"],
        "match_tokens": [
            "### SEVerA: Verified Synthesis of Self-Evolving Agents",
            "- **Source:** arXiv:2603.25111 - https://arxiv.org/abs/2603.25111",
            "self-evolving agents, formal guards",
            "- **Carnot hook:** SEVerA's formally guarded generative model pattern",
            "- **Actionability:** Use the idea for a no-weight-update",
        ],
    },
    {
        "source_id": "taco_adaptive_csp",
        "required_token": "arXiv:2601.21048",
        "title": "Test-Time Adaptation for Unsupervised Combinatorial Optimization",
        "channels": ["arXiv"],
        "urls": ["https://arxiv.org/abs/2601.21048"],
        "spec_tokens": ["arXiv:2601.21048", "TACO"],
        "match_tokens": [
            "### Test-Time Adaptation for Unsupervised Combinatorial Optimization (TACO)",
            "- **Source:** arXiv:2601.21048 - https://arxiv.org/abs/2601.21048",
            "neural combinatorial optimization, instance-wise adaptation",
            "- **Carnot hook:** TACO's strategic warm-start",
            "- **Actionability:** Test whether adaptation improves exact-solver effort",
        ],
    },
    {
        "source_id": "hubo_pspin_planck",
        "required_token": "arXiv:2602.16665",
        "title": "PLANCK: p-Spin Optimization With Hypergraph Neural Networks And DRL",
        "channels": ["arXiv"],
        "urls": ["https://arxiv.org/abs/2602.16665"],
        "spec_tokens": ["arXiv:2602.16665", "PLANCK p-spin/HUBO"],
        "match_tokens": [
            "### PLANCK: p-Spin Optimization With Hypergraph Neural Networks And DRL",
            "- **Source:** arXiv:2602.16665 - https://arxiv.org/abs/2602.16665",
            "high-order Ising/p-spin models, HUBO",
            "- **Carnot hook:** Carnot has pairwise p-bit and QUBO paths",
            "- **Actionability:** Add a direct high-order energy vs QUBO-gadget experiment",
        ],
    },
    {
        "source_id": "neuromorphic_csp_hardware",
        "required_token": "arXiv:2603.01150",
        "title": "Implicitly Parallel Neuromorphic Solver Design For CSPs",
        "channels": ["arXiv"],
        "urls": ["https://arxiv.org/abs/2603.01150"],
        "spec_tokens": ["arXiv:2603.01150", "neuromorphic CSP hardware"],
        "match_tokens": [
            "### Implicitly Parallel Neuromorphic Solver Design For CSPs",
            "- **Source:** arXiv:2603.01150 - https://arxiv.org/abs/2603.01150",
            "neuromorphic CSP solving, parallelism",
            "- **Carnot hook:** The reported parallel CSP speedups",
            "- **Actionability:** Fold partition/update telemetry",
        ],
    },
    {
        "source_id": "cfg_constrained_diffusion",
        "required_token": "arXiv:2606.00722+arXiv:2602.00612+arXiv:2508.10111",
        "title": "CFG-Constrained Diffusion Language Models",
        "channels": ["arXiv", "GitHub"],
        "urls": [
            "https://arxiv.org/abs/2606.00722",
            "https://arxiv.org/abs/2602.00612",
            "https://arxiv.org/abs/2508.10111",
            "https://github.com/eth-sri/constrained-diffusion",
        ],
        "spec_tokens": [
            "arXiv:2606.00722",
            "arXiv:2602.00612",
            "arXiv:2508.10111",
            "eth-sri/constrained-diffusion",
        ],
        "match_tokens": [
            "### CFG-Constrained Diffusion Language Models: EPIC, LAVE, And Constrained Diffusion",
            "EPIC arXiv:2606.00722 - https://arxiv.org/abs/2606.00722",
            "LAVE arXiv:2602.00612 -",
            "Constrained Diffusion arXiv:2508.10111 -",
            "code - https://github.com/eth-sri/constrained-diffusion",
            "constrained generation, CFG validity",
            "- **Carnot hook:** Even if Carnot is not using diffusion LMs locally",
            "- **Actionability:** Use prefix-reachability and completion-existence checks",
        ],
    },
    {
        "source_id": "grammar_aligned_decoding",
        "required_token": "arXiv:2405.21047",
        "title": "Grammar-Aligned Decoding And Distribution-Distortion Caveat",
        "channels": ["arXiv"],
        "urls": ["https://arxiv.org/abs/2405.21047"],
        "spec_tokens": ["arXiv:2405.21047", "Grammar-Aligned Decoding"],
        "match_tokens": [
            "### Grammar-Aligned Decoding And Distribution-Distortion Caveat",
            "- **Source:** arXiv:2405.21047 - https://arxiv.org/abs/2405.21047",
            "constrained decoding, grammar alignment",
            "- **Carnot hook:** The paper is older than the 2025-2026 scan window",
            "- **Actionability:** Include semantic no-op/tautology controls",
        ],
    },
    {
        "source_id": "constrainprompt_code_assurance",
        "required_token": "OpenReview O3Kg4dLdpg",
        "title": "Code-Based Assurance Of Prompt-Defined Constraints",
        "channels": ["OpenReview"],
        "urls": ["https://openreview.net/forum?id=O3Kg4dLdpg"],
        "spec_tokens": ["CONSTRAINPROMPT OpenReview", "O3Kg4dLdpg"],
        "match_tokens": [
            "### Code-Based Assurance Of Prompt-Defined Constraints (CONSTRAINPROMPT)",
            "- **Source:** OpenReview - https://openreview.net/forum?id=O3Kg4dLdpg",
            "prompt-defined constraints, executable checks",
            "- **Carnot hook:** Translating prompt constraints into code-verifiable",
            "- **Actionability:** Prototype prompt-to-code constraint assurance",
        ],
    },
    {
        "source_id": "halt_logprob_timeseries",
        "required_token": "arXiv:2602.02888",
        "title": "HALT: Hallucination Assessment Via Log-Probabilities As Time Series",
        "channels": ["arXiv"],
        "urls": ["https://arxiv.org/abs/2602.02888"],
        "spec_tokens": ["arXiv:2602.02888", "HALT"],
        "match_tokens": [
            "### HALT: Hallucination Assessment Via Log-Probabilities As Time Series",
            "- **Source:** arXiv:2602.02888 - https://arxiv.org/abs/2602.02888",
            "hallucination detection, logprob telemetry",
            "- **Carnot hook:** HALT is a good process-signal candidate",
            "- **Actionability:** Keep as a downstream process-verifier option",
        ],
    },
    {
        "source_id": "genericagent_alma_memory",
        "required_token": "arXiv:2604.17091+arXiv:2602.07755",
        "title": "GenericAgent, ALMA, And Memory Design Search",
        "channels": ["arXiv"],
        "urls": [
            "https://arxiv.org/abs/2604.17091",
            "https://arxiv.org/abs/2602.07755",
        ],
        "spec_tokens": ["arXiv:2604.17091", "arXiv:2602.07755"],
        "match_tokens": [
            "### GenericAgent, ALMA, And Memory Design Search",
            "GenericAgent arXiv:2604.17091 - https://arxiv.org/abs/2604.17091",
            "ALMA arXiv:2602.07755 -",
            "agent memory, verified trajectories",
            "- **Carnot hook:** These papers support the FR-11 direction",
            "- **Actionability:** Use evidence snapshots, provenance hashes",
        ],
    },
    {
        "source_id": "ebt_arm_citation_lineage",
        "required_token": "arXiv:2511.00907+arXiv:2505.11081",
        "title": "EBT / ARM-EBM Citation-Lineage Candidates",
        "channels": ["arXiv", "Semantic Scholar"],
        "urls": [
            "https://arxiv.org/abs/2511.00907",
            "https://arxiv.org/abs/2505.11081",
        ],
        "spec_tokens": ["arXiv:2511.00907", "arXiv:2505.11081"],
        "match_tokens": [
            "### EBT / ARM-EBM Citation-Lineage Candidates",
            "Transformers as Intrinsic Optimizers arXiv:2511.00907 -",
            "ShiQ arXiv:2505.11081 - https://arxiv.org/abs/2505.11081",
            "energy-principle inference, Bellman-style reasoning",
            "- **Carnot hook:** These are medium-term architecture signals",
            "- **Actionability:** Record as architecture pressure",
        ],
    },
    {
        "source_id": "llguidance_baseline",
        "required_token": "guidance-ai/llguidance",
        "title": "llguidance And Constraint-Decoding Engineering Baselines",
        "channels": ["GitHub"],
        "urls": ["https://github.com/guidance-ai/llguidance"],
        "spec_tokens": ["llguidance"],
        "match_tokens": [
            "### llguidance And Constraint-Decoding Engineering Baselines",
            "- **Source:** https://github.com/guidance-ai/llguidance",
            "structured outputs, grammar-constrained decoding",
            "- **Carnot hook:** llguidance gives a practical baseline",
            "- **Actionability:** Compare any finite-schema constrained-generation result",
        ],
    },
    {
        "source_id": "extropic_logical_updates",
        "required_token": "Extropic XTR-0 / TSU And Logical Intelligence Kona/Aleph",
        "title": "Extropic XTR-0 / TSU And Logical Intelligence Kona/Aleph Updates",
        "channels": ["Extropic", "Logical Intelligence"],
        "urls": [
            "https://extropic.ai/writing/inside-x0-and-xtr-0",
            "https://extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware",
            "https://logicalintelligence.com/blog/energy-based-model-sudoku-demo",
            "https://logicalintelligence.com/blog/automatic-formal-verification-for-code-generation",
            "https://logicalintelligence.com/blog/aleph-leading-benchmarks",
        ],
        "spec_tokens": [
            "Extropic XTR-0/TSU pages",
            "Logical Intelligence Kona/Aleph/formal",
            "verification updates",
        ],
        "match_tokens": [
            "### Extropic XTR-0 / TSU And Logical Intelligence Kona/Aleph Updates",
            "https://extropic.ai/writing/inside-x0-and-xtr-0",
            "https://extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware",
            "https://logicalintelligence.com/blog/energy-based-model-sudoku-demo",
            "https://logicalintelligence.com/blog/automatic-formal-verification-for-code-generation",
            "https://logicalintelligence.com/blog/aleph-leading-benchmarks",
            "thermodynamic accelerators, EBM reasoning products",
            "- **Carnot hook:** Extropic still appears to be an early-access hardware path",
            "- **Actionability:** Keep TSU work in architecture/simulation",
        ],
    },
]

EXPECTED_SOURCE_HOOK_COUNT = len(REQUIRED_REFERENCE_CHECKS)
EXPECTED_ACTIONABILITY_COUNT = len(REQUIRED_REFERENCE_CHECKS)

SOURCE_CHECK_OVERRIDES = [
    {
        "source_id": "huggingface_papers_coverage",
        "title": "Hugging Face Papers query coverage",
        "channels": ["Hugging Face Papers"],
        "urls": ["https://huggingface.co/papers"],
        "status": "coverage_declared_in_v468_section_no_hf_only_source_added",
        "reference_found": True,
    },
]

SEMANTIC_SCHOLAR_STATUS = {
    "attempted_by_planner": True,
    "checked_on": "2026-07-01",
    "status": "planner_api_429_public_lineage_notes_recorded",
    "fresh_api_call_in_exp5096": False,
    "targets": [
        {
            "label": "EBT",
            "paper_id": "arXiv:2507.02092",
            "citation_count_recorded": None,
            "lineage_source_ids": [
                "arXiv:2511.00907",
                "arXiv:2505.11081",
            ],
        },
        {
            "label": "ARM-EBM",
            "paper_id": "arXiv:2512.15605",
            "citation_count_recorded": None,
            "lineage_source_ids": [
                "arXiv:2511.00907",
                "arXiv:2505.11081",
            ],
        },
    ],
}

TASK_MAPPING = [
    {
        "source_id": "beaver_prefix_bounds",
        "source_tokens": ["arXiv:2512.05439", "github.com/uiuc-focal-lab/Beaver"],
        "task_id": "exp5099",
        "mapping_status": "mapped_to_task",
        "rationale": "deterministic prefix probability bounds are Exp5099's exact-verifier surface.",
    },
    {
        "source_id": "graph_evidence_grounding",
        "source_tokens": ["arXiv:2606.30247"],
        "task_id": "exp5101",
        "mapping_status": "mapped_to_task",
        "rationale": "incomplete graph support maps directly to the evidence-energy audit.",
    },
    {
        "source_id": "severa_self_evolving_agents",
        "source_tokens": ["arXiv:2603.25111"],
        "task_id": "exp5105",
        "mapping_status": "mapped_to_task",
        "rationale": "Search-Verify-Learn guards define the FR-11 memory promotion contract.",
    },
    {
        "source_id": "taco_adaptive_csp",
        "source_tokens": ["arXiv:2601.21048"],
        "task_id": "exp5103",
        "mapping_status": "mapped_to_task",
        "rationale": "instance adaptation becomes a solver-effort heuristic under exact checking.",
    },
    {
        "source_id": "hubo_pspin_planck",
        "source_tokens": ["arXiv:2602.16665"],
        "task_id": "exp5102",
        "mapping_status": "mapped_to_task",
        "rationale": "direct high-order energy tests whether HUBO avoids QUBO gadget blowup.",
    },
    {
        "source_id": "neuromorphic_csp_hardware",
        "source_tokens": ["arXiv:2603.01150"],
        "task_id": "exp5106",
        "mapping_status": "mapped_to_task",
        "rationale": "partition/update telemetry constrains hardware-continuity claims.",
    },
    {
        "source_id": "cfg_constrained_diffusion",
        "source_tokens": ["arXiv:2606.00722", "arXiv:2602.00612", "arXiv:2508.10111"],
        "task_id": "exp5104",
        "mapping_status": "mapped_to_task",
        "rationale": "prefix reachability and completion existence are constrained-decoding controls.",
    },
    {
        "source_id": "grammar_aligned_decoding",
        "source_tokens": ["arXiv:2405.21047"],
        "task_id": "exp5104",
        "mapping_status": "mapped_to_task",
        "rationale": "distribution-distortion controls prevent syntactic validity from becoming the claim.",
    },
    {
        "source_id": "constrainprompt_code_assurance",
        "source_tokens": ["openreview:O3Kg4dLdpg"],
        "task_id": "exp5100",
        "mapping_status": "mapped_to_task",
        "rationale": "prompt-defined constraints are accepted only through executable code checks.",
    },
    {
        "source_id": "halt_logprob_timeseries",
        "source_tokens": ["arXiv:2602.02888"],
        "task_id": "background_only",
        "mapping_status": "background_only",
        "rationale": "HALT needs clean logprob substrate, so it is deferred until after Exp5097.",
    },
    {
        "source_id": "genericagent_alma_memory",
        "source_tokens": ["arXiv:2604.17091", "arXiv:2602.07755"],
        "task_id": "exp5105",
        "mapping_status": "mapped_to_task",
        "rationale": "memory design ideas are used only under SEVerA-style exact promotion guards.",
    },
    {
        "source_id": "ebt_arm_citation_lineage",
        "source_tokens": ["arXiv:2511.00907", "arXiv:2505.11081"],
        "task_id": "background_only",
        "mapping_status": "background_only",
        "rationale": "architecture pressure only; .468 avoids training-heavy EBT/ARM reproduction.",
    },
    {
        "source_id": "llguidance_baseline",
        "source_tokens": ["github.com/guidance-ai/llguidance"],
        "task_id": "exp5104",
        "mapping_status": "mapped_to_task",
        "rationale": "real grammar-engine baselines bound toy constrained-decoding claims.",
    },
    {
        "source_id": "extropic_logical_updates",
        "source_tokens": ["Extropic XTR-0", "Logical Intelligence Kona", "Logical Intelligence Aleph"],
        "task_id": "exp5106",
        "mapping_status": "mapped_to_task",
        "rationale": "TSU/Kona context informs hardware telemetry and exact-verifier product framing.",
    },
]

PLANNING_HOOKS = [
    {
        "hook_id": "exp5099_beaver_prefix_bounds",
        "source_ids": ["beaver_prefix_bounds"],
        "hook": "Prototype BEAVER-style deterministic prefix bounds over finite verifier schemas.",
    },
    {
        "hook_id": "exp5100_prompt_code_assurance",
        "source_ids": ["constrainprompt_code_assurance"],
        "hook": "Translate prompt-defined constraints into executable checker code and tests.",
    },
    {
        "hook_id": "exp5101_graph_evidence_energy",
        "source_ids": ["graph_evidence_grounding"],
        "hook": "Separate contradiction rejection from unsupported-claim abstention on a graph.",
    },
    {
        "hook_id": "exp5102_hubo_pspin_direct_energy",
        "source_ids": ["hubo_pspin_planck"],
        "hook": "Compare direct HUBO/p-spin energy against QUBO gadgets by exact enumeration.",
    },
    {
        "hook_id": "exp5103_taco_adaptive_csp",
        "source_ids": ["taco_adaptive_csp"],
        "hook": "Measure whether adaptation reduces exact-solver effort without owning correctness.",
    },
    {
        "hook_id": "exp5104_constrained_decoding_semantic_audit",
        "source_ids": [
            "cfg_constrained_diffusion",
            "grammar_aligned_decoding",
            "llguidance_baseline",
        ],
        "hook": "Audit constrained decoding with reachability, semantic controls, and llguidance.",
    },
    {
        "hook_id": "exp5105_severa_fr11_memory",
        "source_ids": ["severa_self_evolving_agents", "genericagent_alma_memory"],
        "hook": "Gate FR-11 memory/SOP promotion through contracts, provenance, and non-regression.",
    },
    {
        "hook_id": "exp5106_hardware_partition_telemetry",
        "source_ids": ["neuromorphic_csp_hardware", "extropic_logical_updates"],
        "hook": "Report partition/update telemetry before any board or TSU acceleration claim.",
    },
    {
        "hook_id": "background_runtime_and_architecture_context",
        "source_ids": ["halt_logprob_timeseries", "ebt_arm_citation_lineage"],
        "hook": "Keep HALT and EBT/ARM lineage as deferred context for runtime and architecture work.",
    },
]


def _require(condition: bool, message: str) -> None:
    if not condition:  # pragma: no cover
        raise ValueError(message)


def extract_v468_section(text: str) -> str:
    """Return the V468 managed reference section without the outer markers."""

    _require(V468_SECTION_START in text, "V468 planner section start marker missing")
    after_start = text.split(V468_SECTION_START, 1)[1]
    _require(V468_SECTION_END in after_start, "V468 planner section end marker missing")
    return after_start.split(V468_SECTION_END, 1)[0]


def verify_v468_references(section: str) -> dict[str, Any]:
    """Check the V468 section for source IDs, URLs, hooks, and actionability."""

    present: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    coverage_absent = [token for token in COVERAGE_MATCH_TOKENS if token not in section]
    if coverage_absent:
        missing.append(
            {
                "source_id": "Semantic Scholar citation-lineage status",
                "required_token": "Semantic Scholar",
                "title": "V468 search coverage and citation-lineage status",
                "missing_tokens": coverage_absent,
            }
        )

    for check in REQUIRED_REFERENCE_CHECKS:
        absent = [token for token in check["match_tokens"] if token not in section]
        if absent:
            missing.append(
                {
                    "source_id": check["source_id"],
                    "required_token": check["required_token"],
                    "title": check["title"],
                    "missing_tokens": absent,
                }
            )
        else:
            present.append(
                {
                    "source_id": check["source_id"],
                    "required_token": check["required_token"],
                    "title": check["title"],
                    "channels": list(check["channels"]),
                    "urls": list(check["urls"]),
                }
            )

    hook_count = section.count("- **Carnot hook:**")
    actionability_count = section.count("- **Actionability:**")
    if hook_count < EXPECTED_SOURCE_HOOK_COUNT:
        missing.append(
            {
                "source_id": "per_source_carnot_hooks",
                "required_token": "Carnot hook",
                "title": "per-source Carnot hooks",
                "missing_tokens": [f"{EXPECTED_SOURCE_HOOK_COUNT} hooks, observed {hook_count}"],
            }
        )
    if actionability_count < EXPECTED_ACTIONABILITY_COUNT:
        missing.append(
            {
                "source_id": "per_source_actionability",
                "required_token": "Actionability",
                "title": "per-source actionability notes",
                "missing_tokens": [
                    f"{EXPECTED_ACTIONABILITY_COUNT} actionability notes, observed {actionability_count}"
                ],
            }
        )

    return {
        "references_section_found": True,
        "present": present,
        "missing": missing,
        "carnot_hook_count": hook_count,
        "actionability_count": actionability_count,
    }


def _validate_reference_text(reference_text: str) -> dict[str, Any]:
    section = extract_v468_section(reference_text)
    verification = verify_v468_references(section)
    if verification["missing"]:
        first_missing = verification["missing"][0]
        missing_tokens = ", ".join(first_missing["missing_tokens"])
        raise ValueError(
            "missing V468 reference evidence for "
            f"{first_missing['source_id']}: {missing_tokens}"
        )
    return verification


def _build_sources_checked(verification: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "source_id": row["source_id"],
            "required_token": row["required_token"],
            "title": row["title"],
            "channels": list(row["channels"]),
            "urls": list(row["urls"]),
            "status": "present_in_v468_section",
            "reference_found": True,
        }
        for row in verification["present"]
    ]
    rows.extend(dict(row) for row in SOURCE_CHECK_OVERRIDES)
    return rows


def _background_only_sources() -> list[dict[str, Any]]:
    return [dict(row) for row in TASK_MAPPING if row["mapping_status"] == "background_only"]


def build_artifact(*, reference_text: str) -> dict[str, Any]:
    """Build and validate the Exp 5096 V468 SOTA ingestion artifact."""

    verification = _validate_reference_text(reference_text)
    artifact: dict[str, Any] = {
        "honest_verdict": HONEST_VERDICT,
        "duration_s": DURATION_S,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "sources_checked": _build_sources_checked(verification),
        "references_section_found": verification["references_section_found"],
        "references_added_count": 0,
        "semantic_scholar_status": dict(SEMANTIC_SCHOLAR_STATUS),
        "task_mapping": [dict(row) for row in TASK_MAPPING],
        "planning_hooks": [dict(hook) for hook in PLANNING_HOOKS],
        "background_only_sources": _background_only_sources(),
        "flagged_adversarial": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the artifact drifts from REQ-REPORT-5096."""

    _require(set(artifact) == set(REQUIRED_ARTIFACT_FIELDS), "artifact fields mismatch")
    verdict = artifact["honest_verdict"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict must use a terminal prefix",
    )
    _require(verdict == HONEST_VERDICT, "unexpected Exp 5096 honest_verdict")
    _require(artifact["duration_s"] == DURATION_S, "duration_s mismatch")
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must be literature_review_and_repo_inspection",
    )
    _require(artifact["references_section_found"] is True, "V468 section was not found")
    _require(artifact["references_added_count"] == 0, "unexpected V468 reference addition")
    _require(artifact["flagged_adversarial"] is False, "clean V468 audit cannot be flagged")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles mismatch")
    _require(artifact["spec_refs"] == SPEC_REFS, "spec_refs mismatch")

    sources_checked = artifact["sources_checked"]
    _require(isinstance(sources_checked, list), "sources_checked must be a list")
    channels = {
        channel
        for row in sources_checked
        for channel in (row.get("channels") if isinstance(row, Mapping) else [])
    }
    _require(REQUIRED_CHANNELS.issubset(channels), "required source channel missing")
    for row in sources_checked:
        _require(isinstance(row, Mapping), "source row must be a mapping")
        _require(row.get("reference_found") is True, "source reference must be marked found")
        _require(bool(row.get("status")), "source status is required")
        urls = row.get("urls")
        _require(isinstance(urls, list) and urls, "source URLs are required")
        _require(all(str(url).startswith("https://") for url in urls), "source URL must be https")

    semantic = artifact["semantic_scholar_status"]
    _require(semantic == SEMANTIC_SCHOLAR_STATUS, "semantic_scholar_status mismatch")
    _require(
        semantic["status"] == "planner_api_429_public_lineage_notes_recorded",
        "Semantic Scholar status mismatch",
    )
    _require(semantic["fresh_api_call_in_exp5096"] is False, "fresh API call must not be claimed")
    semantic_targets = {target["label"]: target for target in semantic["targets"]}
    _require(
        semantic_targets["EBT"]["lineage_source_ids"] == ["arXiv:2511.00907", "arXiv:2505.11081"],
        "EBT lineage mismatch",
    )
    _require(
        semantic_targets["ARM-EBM"]["lineage_source_ids"]
        == ["arXiv:2511.00907", "arXiv:2505.11081"],
        "ARM-EBM lineage mismatch",
    )

    mapping = artifact["task_mapping"]
    _require(isinstance(mapping, list), "task_mapping must be a list")
    expected_source_ids = [row["source_id"] for row in REQUIRED_REFERENCE_CHECKS]
    observed_source_ids = [row.get("source_id") for row in mapping if isinstance(row, Mapping)]
    _require(observed_source_ids == expected_source_ids, "task_mapping source order mismatch")
    allowed_tasks = {
        "exp5099",
        "exp5100",
        "exp5101",
        "exp5102",
        "exp5103",
        "exp5104",
        "exp5105",
        "exp5106",
    }
    for row in mapping:
        _require(isinstance(row, Mapping), "task mapping row must be a mapping")
        task_id = row.get("task_id")
        status = row.get("mapping_status")
        if task_id == "background_only":
            _require(status == "background_only", "background mapping status mismatch")
        else:
            _require(task_id in allowed_tasks, "unexpected .468 task mapping")
            _require(status == "mapped_to_task", "task mapping status mismatch")
        _require(bool(row.get("source_tokens")), "task mapping source tokens missing")
        _require(bool(row.get("rationale")), "task mapping rationale missing")

    background = artifact["background_only_sources"]
    _require(isinstance(background, list), "background_only_sources must be a list")
    _require(background == _background_only_sources(), "background_only_sources mismatch")
    _require(
        {row["source_id"] for row in background}
        == {"halt_logprob_timeseries", "ebt_arm_citation_lineage"},
        "unexpected background-only source set",
    )

    hooks = artifact["planning_hooks"]
    _require(isinstance(hooks, list) and len(hooks) >= 8, "planning_hooks too small")
    hook_ids = {hook.get("hook_id") for hook in hooks if isinstance(hook, Mapping)}
    _require("exp5099_beaver_prefix_bounds" in hook_ids, "BEAVER hook missing")
    _require("exp5100_prompt_code_assurance" in hook_ids, "CONSTRAINPROMPT hook missing")
    _require("exp5101_graph_evidence_energy" in hook_ids, "graph evidence hook missing")
    _require("exp5102_hubo_pspin_direct_energy" in hook_ids, "HUBO/p-spin hook missing")
    _require("exp5103_taco_adaptive_csp" in hook_ids, "TACO hook missing")
    _require("exp5104_constrained_decoding_semantic_audit" in hook_ids, "CFG hook missing")
    _require("exp5105_severa_fr11_memory" in hook_ids, "SEVerA hook missing")
    _require("exp5106_hardware_partition_telemetry" in hook_ids, "hardware hook missing")
    for hook in hooks:
        _require(isinstance(hook, Mapping), "planning hook row must be a mapping")
        _require(bool(hook.get("source_ids")), "planning hook source IDs missing")
        _require(bool(hook.get("hook")), "planning hook text missing")


def write_outputs(*, artifact_path: Path, references_path: Path) -> dict[str, Any]:
    """Write the stable JSON artifact after validating V468 references."""

    reference_text = references_path.read_text(encoding="utf-8")
    artifact = build_artifact(reference_text=reference_text)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> int:
    root = Path(os.environ.get("CARNOT_EXP5096_ROOT", Path(__file__).resolve().parents[2]))
    write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        references_path=root / REFERENCES_RELATIVE_PATH,
    )
    print(HONEST_VERDICT)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
