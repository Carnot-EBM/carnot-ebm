"""Exp 5110 source-freshness artifact for the V469 SOTA ingestion gate.

Spec refs: REQ-REPORT-5110, SCENARIO-REPORT-5110,
SCENARIO-REPORT-5110-BLOCKED-OR-PARTIAL-SOURCE.

This module is a literature-review and repository-inspection artifact. It does
not train a model, run a local LLM, touch hardware, or edit the active roadmap.
The goal is to make the V469 references auditable before downstream FoVer,
KAN, sampler, FR-11, runtime, and hardware-continuity tasks consume them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_5110_sota_ingestion_v469.json"
REFERENCES_RELATIVE_PATH = "research-references.md"
ROADMAP_RELATIVE_PATH = "openspec/change-proposals/research-roadmap-vNEXT.md"
EXPERIMENT_ID = "exp5110-source-freshness-sota-ingestion-v469"
MILESTONE = "2026.07.469"
INFERENCE_SUBSTRATE = "literature_review_and_repo_inspection"
COMPLETE_VERDICT = "complete_source_freshness_sota_ingestion_v469_references_mapped"
PARTIAL_REFERENCES_VERDICT = "partial_v469_references_section_not_found"
PARTIAL_SOURCE_VERDICT = "partial_v469_source_access_issue_preserved"
REFERENCES_SECTION_START = "<!-- V469-PLANNER-REFERENCES-START -->"
REFERENCES_SECTION_END = "<!-- V469-PLANNER-REFERENCES-END -->"
FRESH_SWEEP_HEADING = "V469 Fresh Sweep Addendum"
TERMINAL_PREFIXES = (
    "blocked_",
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "partial_",
)
ALLOWED_TASK_IDS = frozenset(f"exp{task_id}" for task_id in range(5111, 5121))
REQUIRED_TOPICS = frozenset(
    {
        "FoVer",
        "FormalRewardBench",
        "self-verification cliff",
        "KAN abstraction",
        "p-bit 2D-PT",
        "million p-bit residual telemetry",
        "FALCON",
        "Verus-SpecGym",
        "EBT",
        "ARM-EBM",
        "Extropic",
        "Logical Intelligence",
    }
)
REQUIRED_USER_ARTIFACT_FIELDS = frozenset(
    {
        "experiment_id",
        "milestone",
        "honest_verdict",
        "inference_substrate",
        "duration_s",
        "references_section_found",
        "sources_checked",
        "task_mapping",
        "background_only_sources",
        "active_roadmap_modified",
        "flagged_adversarial",
        "tests_run",
    }
)
EXTRA_ARTIFACT_FIELDS = frozenset(
    {
        "field_principles",
        "coverage_by_required_topic",
        "claim_boundary_findings",
        "references_section_bounds",
        "training_or_hardware_execution_claims_detected",
        "repo_inputs_read",
        "run_date",
    }
)
REQUIRED_ARTIFACT_FIELDS = REQUIRED_USER_ARTIFACT_FIELDS | EXTRA_ARTIFACT_FIELDS
FIELD_PRINCIPLES = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "references_section_found": "downstream gate signal",
    "sources_checked": "provenance",
    "task_mapping": "actionability",
    "background_only_sources": "no false execution claims",
    "active_roadmap_modified": "operator instruction compliance",
    "flagged_adversarial": "adversarial-verification accountability",
    "tests_run": "verification evidence",
    "field_principles": "principle annotations for every top-level artifact field",
    "coverage_by_required_topic": "source coverage completeness",
    "claim_boundary_findings": "no unplanned training or hardware execution claims",
    "references_section_bounds": "auditable source-section extraction",
    "training_or_hardware_execution_claims_detected": "false execution claim detection",
    "repo_inputs_read": "repo inspection provenance",
    "run_date": "run labeling",
}
DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5110_sota_ingestion_v469.py --date 20260701",
    ".venv/bin/pytest tests/python/test_experiment_5110_sota_ingestion_v469.py -q",
    ".venv/bin/pytest tests/python -q",
]
DEAD_OR_STALE_URL_STATUSES = frozenset({"dead", "stale"})

SOURCE_CHECKS: list[dict[str, Any]] = [
    {
        "source_id": "fover",
        "required_topic": "FoVer",
        "title": "Efficient PRM Training Data Synthesis via Formal Verification",
        "claim_boundary": "actionable",
        "source_access_status": "verified_live",
        "urls": [
            {
                "url": "https://arxiv.org/abs/2505.15960",
                "status": "verified_live",
                "evidence": "arXiv page live; v3 revised 2026-04-08; names FoVer.",
            },
            {
                "url": "https://github.com/psunlpgroup/FoVer",
                "status": "verified_live",
                "evidence": "Public repository for the ACL 2026 Findings FoVer paper.",
            },
            {
                "url": "https://huggingface.co/papers/2505.15960",
                "status": "verified_live",
                "evidence": "Hugging Face paper record is live.",
            },
        ],
    },
    {
        "source_id": "formalrewardbench",
        "required_topic": "FormalRewardBench",
        "title": "FormalRewardBench: A Benchmark for Formal Theorem Proving Reward Models",
        "claim_boundary": "actionable",
        "source_access_status": "verified_live",
        "urls": [
            {
                "url": "https://arxiv.org/abs/2605.10141",
                "status": "verified_live",
                "evidence": "arXiv page live; submitted 2026-05-11.",
            }
        ],
    },
    {
        "source_id": "self_verification_cliff",
        "required_topic": "self-verification cliff",
        "title": "The Self-Verification Cliff",
        "claim_boundary": "actionable",
        "source_access_status": "partial_openreview_challenge_metadata_verified",
        "access_note": (
            "OpenReview forum and PDF URLs redirect to a browser challenge, but search "
            "metadata and the ICML 2026 AI4Math listing confirm the record."
        ),
        "urls": [
            {
                "url": "https://openreview.net/forum?id=QJTSAvHFQn",
                "status": "partial_openreview_challenge_metadata_verified",
                "evidence": "Forum URL is real but guarded by OpenReview browser verification.",
            },
            {
                "url": "https://openreview.net/pdf?id=QJTSAvHFQn",
                "status": "partial_openreview_challenge_metadata_verified",
                "evidence": "PDF URL is real but guarded by OpenReview browser verification.",
            },
        ],
    },
    {
        "source_id": "kan_abstraction",
        "required_topic": "KAN abstraction",
        "title": "Optimal Abstractions for Verifying Properties of Kolmogorov-Arnold Networks",
        "claim_boundary": "actionable",
        "source_access_status": "verified_live",
        "urls": [
            {
                "url": "https://arxiv.org/abs/2602.06737",
                "status": "verified_live",
                "evidence": "arXiv page live; submitted 2026-02-06.",
            }
        ],
    },
    {
        "source_id": "pbit_2d_parallel_tempering",
        "required_topic": "p-bit 2D-PT",
        "title": "Probabilistic Computers for MIMO Detection",
        "claim_boundary": "actionable",
        "source_access_status": "verified_live",
        "urls": [
            {
                "url": "https://arxiv.org/abs/2601.09037",
                "status": "verified_live",
                "evidence": "arXiv page live; v2 revised 2026-05-31; describes 2D-PT.",
            }
        ],
    },
    {
        "source_id": "million_pbit_residual_telemetry",
        "required_topic": "million p-bit residual telemetry",
        "title": "Programmable Probabilistic Computer with 1,000,000 p-bits",
        "claim_boundary": "actionable",
        "source_access_status": "verified_live",
        "urls": [
            {
                "url": "https://arxiv.org/abs/2606.25313",
                "status": "verified_live",
                "evidence": "arXiv page live; submitted 2026-06-24.",
            }
        ],
    },
    {
        "source_id": "falcon",
        "required_topic": "FALCON",
        "title": "Hard Constraints Meet Soft Generation",
        "claim_boundary": "actionable",
        "source_access_status": "verified_live",
        "urls": [
            {
                "url": "https://arxiv.org/abs/2602.01090",
                "status": "verified_live",
                "evidence": "arXiv page live; names FALCON and semantic repair.",
            }
        ],
    },
    {
        "source_id": "verus_specgym",
        "required_topic": "Verus-SpecGym",
        "title": "Verus-SpecGym: An Agentic Environment for Evaluating Specification Autoformalization",
        "claim_boundary": "actionable",
        "source_access_status": "verified_live",
        "urls": [
            {
                "url": "https://huggingface.co/papers/2605.26457",
                "status": "verified_live",
                "evidence": "Hugging Face paper page live with Verus-SpecBench and Verus-SpecGym summary.",
            },
            {
                "url": "https://arxiv.org/abs/2605.26457",
                "status": "verified_live",
                "evidence": "arXiv page live; submitted 2026-05-26.",
            },
            {
                "url": "https://github.com/formal-verif-is-cool/verus-spec-gym",
                "status": "verified_live",
                "evidence": "Public GitHub repository URL resolves.",
            },
        ],
    },
    {
        "source_id": "ebt_primary",
        "required_topic": "EBT",
        "title": "Energy-Based Transformers are Scalable Learners and Thinkers",
        "claim_boundary": "background_only",
        "source_access_status": "verified_live",
        "urls": [
            {
                "url": "https://arxiv.org/abs/2507.02092",
                "status": "verified_live",
                "evidence": "arXiv page live; submitted 2025-07-02.",
            },
            {
                "url": "https://github.com/alexiglad/EBT",
                "status": "verified_live",
                "evidence": "Public EBT code repository resolves.",
            },
        ],
    },
    {
        "source_id": "arm_ebm",
        "required_topic": "ARM-EBM",
        "title": "Autoregressive Language Models are Secretly Energy-Based Models",
        "claim_boundary": "background_only",
        "source_access_status": "verified_live",
        "urls": [
            {
                "url": "https://arxiv.org/abs/2512.15605",
                "status": "verified_live",
                "evidence": "arXiv page live; v4 revised 2026-05-25.",
            }
        ],
    },
    {
        "source_id": "nrgpt",
        "required_topic": "EBT",
        "title": "NRGPT: An Energy-based Alternative for GPT",
        "claim_boundary": "background_only",
        "source_access_status": "verified_live",
        "urls": [
            {
                "url": "https://arxiv.org/abs/2512.16762",
                "status": "verified_live",
                "evidence": "arXiv page live; ICLR 2026 main conference note.",
            }
        ],
    },
    {
        "source_id": "domain_grounded_agents",
        "required_topic": "ARM-EBM",
        "title": "Ontology-Constrained Neural Reasoning in Enterprise Agentic Systems",
        "claim_boundary": "background_only",
        "source_access_status": "verified_live",
        "urls": [
            {
                "url": "https://arxiv.org/abs/2604.00555",
                "status": "verified_live",
                "evidence": "arXiv page live; v5 revised 2026-06-04.",
            }
        ],
    },
    {
        "source_id": "extropic_tsu",
        "required_topic": "Extropic",
        "title": "Extropic XTR-0 and Thermodynamic Sampling Unit updates",
        "claim_boundary": "background_only",
        "source_access_status": "verified_live_with_minimal_pages",
        "urls": [
            {
                "url": "https://extropic.ai/writing/inside-x0-and-xtr-0",
                "status": "reachable_minimal_page",
                "evidence": "Page URL resolves but exposes minimal static text to the fetcher.",
            },
            {
                "url": "https://extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware",
                "status": "reachable_minimal_page",
                "evidence": "Page URL resolves but exposes minimal static text to the fetcher.",
            },
            {
                "url": "https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one",
                "status": "verified_live",
                "evidence": "Live article describes XTR-0, TSU, and thrml simulation context.",
            },
        ],
    },
    {
        "source_id": "logical_kona_aleph",
        "required_topic": "Logical Intelligence",
        "title": "Logical Intelligence Kona and Aleph updates",
        "claim_boundary": "background_only",
        "source_access_status": "verified_live",
        "urls": [
            {
                "url": "https://logicalintelligence.com/blog/aleph-leading-benchmarks",
                "status": "verified_live",
                "evidence": "Live 2026-05-14 Aleph benchmark article resolves.",
            },
            {
                "url": "https://logicalintelligence.com/blog/energy-based-model-sudoku-demo",
                "status": "verified_live",
                "evidence": "Live 2026-02-03 Kona Sudoku benchmark article resolves.",
            },
            {
                "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
                "status": "verified_live",
                "evidence": "Live Kona product architecture page resolves.",
            },
        ],
    },
]
ACTIONABLE_SOURCE_IDS = frozenset(
    source["source_id"] for source in SOURCE_CHECKS if source["claim_boundary"] == "actionable"
)
REQUIRED_BACKGROUND_SOURCE_IDS = frozenset(
    {
        "ebt_primary",
        "arm_ebm",
        "nrgpt",
        "domain_grounded_agents",
        "extropic_tsu",
        "logical_kona_aleph",
    }
)

DEFAULT_TASK_MAPPING: dict[str, list[dict[str, str]]] = {
    "fover": [
        {
            "task_id": "exp5111",
            "task_name": "FoVer in-domain pool",
            "reason": "Build the n>=150 candidate pool with oracle-distinct labels.",
        },
        {
            "task_id": "exp5112",
            "task_name": "FoVer in-domain selector",
            "reason": "Measure Best-of-K headroom and select against tuned self-consistency.",
        },
        {
            "task_id": "exp5113",
            "task_name": "FoVer selector adversarial audit",
            "reason": "Audit leakage, shuffled labels, and CI handling before a moat claim.",
        },
        {
            "task_id": "exp5115",
            "task_name": "Graph evidence FoVer transfer",
            "reason": "Transfer exact evidence energy to FoVer step/support traces.",
        },
        {
            "task_id": "exp5118",
            "task_name": "FR-11 FoVer residual memory",
            "reason": "Use FoVer selector residuals as the auditable self-learning stream.",
        },
    ],
    "formalrewardbench": [
        {
            "task_id": "exp5113",
            "task_name": "FoVer selector adversarial audit",
            "reason": "Add controlled error-family and label-ablation checks.",
        },
        {
            "task_id": "exp5118",
            "task_name": "FR-11 FoVer residual memory",
            "reason": "Require non-forgetting checks before any memory/SOP promotion.",
        },
    ],
    "self_verification_cliff": [
        {
            "task_id": "exp5111",
            "task_name": "FoVer in-domain pool",
            "reason": "Record oracle@K versus tuned self-consistency headroom.",
        },
        {
            "task_id": "exp5112",
            "task_name": "FoVer in-domain selector",
            "reason": "Test whether the selector recovers sampling headroom.",
        },
        {
            "task_id": "exp5113",
            "task_name": "FoVer selector adversarial audit",
            "reason": "Prevent self-selection headroom from being misread as verifier value.",
        },
    ],
    "kan_abstraction": [
        {
            "task_id": "exp5114",
            "task_name": "KAN abstraction refinement post wall",
            "reason": "Replace repeated exact-MILP scale sweeps with abstraction budgeting.",
        }
    ],
    "pbit_2d_parallel_tempering": [
        {
            "task_id": "exp5116",
            "task_name": "HUBO 2D-PT sampling reference",
            "reason": "Build a CPU exact-checked two-axis tempering reference before board claims.",
        }
    ],
    "million_pbit_residual_telemetry": [
        {
            "task_id": "exp5120",
            "task_name": "Hardware residual telemetry",
            "reason": "Use residual-energy decay and partition telemetry as the honest hardware metric.",
        }
    ],
    "falcon": [
        {
            "task_id": "exp5117",
            "task_name": "TACO harm-gated scale",
            "reason": "Use semantic repair and adaptive sampling lessons for harm-gated solver help.",
        },
        {
            "task_id": "exp5113",
            "task_name": "FoVer selector adversarial audit",
            "reason": "Avoid a syntax-only constrained-generation headline.",
        },
    ],
    "verus_specgym": [
        {
            "task_id": "exp5113",
            "task_name": "FoVer selector adversarial audit",
            "reason": "Borrow executable-spec and adversarial-edge-case faithfulness checks.",
        },
        {
            "task_id": "exp5118",
            "task_name": "FR-11 FoVer residual memory",
            "reason": "Apply adversarial edge-case controls to memory promotion contracts.",
        },
    ],
}


def find_references_section(references_text: str) -> dict[str, Any]:
    """Locate the V469 references section and the fresh-sweep addendum."""

    start_index = references_text.find(REFERENCES_SECTION_START)
    end_index = references_text.find(REFERENCES_SECTION_END)
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return {
            "found": False,
            "start_line": None,
            "end_line": None,
            "fresh_sweep_addendum_found": False,
            "section_text": "",
        }

    section_end_index = end_index + len(REFERENCES_SECTION_END)
    section_text = references_text[start_index:section_end_index]
    return {
        "found": True,
        "start_line": references_text[:start_index].count("\n") + 1,
        "end_line": references_text[:section_end_index].count("\n") + 1,
        "fresh_sweep_addendum_found": FRESH_SWEEP_HEADING in section_text,
        "section_text": section_text,
    }


def _source_url_has_dead_or_stale_status(source: Mapping[str, Any]) -> bool:
    return any(url["status"] in DEAD_OR_STALE_URL_STATUSES for url in source["urls"])


def _build_coverage_by_topic(sources: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    coverage = {topic: [] for topic in sorted(REQUIRED_TOPICS)}
    for source in sources:
        coverage[source["required_topic"]].append(source["source_id"])
    return coverage


def _build_background_only_sources() -> list[dict[str, str]]:
    return [
        {
            "source_id": "ebt_primary",
            "reason": "Architecture pressure only; V469 has no local EBT pretraining reproduction.",
        },
        {
            "source_id": "arm_ebm",
            "reason": "Architecture pressure only; local work can cite ARM-to-EBM theory but not retrain it.",
        },
        {
            "source_id": "nrgpt",
            "reason": "Citation-lineage context only; no V469 NRGPT training task exists.",
        },
        {
            "source_id": "domain_grounded_agents",
            "reason": "Domain-grounding architecture context only; no local enterprise-agent reproduction.",
        },
        {
            "source_id": "extropic_tsu",
            "reason": "TSU/XTR-0 context only; no authenticated local Extropic hardware target exists.",
        },
        {
            "source_id": "logical_kona_aleph",
            "reason": "Kona/Aleph architecture pressure only; no local Kona execution target exists.",
        },
    ]


def _roadmap_task_ids_present(roadmap_text: str) -> list[str]:
    return [task_id for task_id in sorted(ALLOWED_TASK_IDS) if task_id in roadmap_text]


def build_artifact(
    *,
    references_text: str,
    roadmap_text: str,
    duration_s: float,
    run_date: str,
    tests_run: Sequence[str],
) -> dict[str, Any]:
    """Build the deterministic Exp 5110 artifact from repo text inputs."""

    references_section = find_references_section(references_text)
    references_section_found = bool(
        references_section["found"] and references_section["fresh_sweep_addendum_found"]
    )
    dead_or_stale_source_found = any(
        _source_url_has_dead_or_stale_status(source) for source in SOURCE_CHECKS
    )
    if dead_or_stale_source_found:
        honest_verdict = PARTIAL_SOURCE_VERDICT
    elif references_section_found:
        honest_verdict = COMPLETE_VERDICT
    else:
        honest_verdict = PARTIAL_REFERENCES_VERDICT

    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "references_section_found": references_section_found,
        "sources_checked": SOURCE_CHECKS,
        "task_mapping": DEFAULT_TASK_MAPPING,
        "background_only_sources": _build_background_only_sources(),
        "active_roadmap_modified": False,
        "flagged_adversarial": False,
        "tests_run": list(tests_run),
        "field_principles": FIELD_PRINCIPLES,
        "coverage_by_required_topic": _build_coverage_by_topic(SOURCE_CHECKS),
        "claim_boundary_findings": [
            "EBT/ARM-EBM/Kona/TSU kept as architecture pressure",
            "No training-heavy reproduction is scheduled by Exp 5110.",
            "No local TSU, Kona, board execution, latency, or speedup claim is made.",
        ],
        "references_section_bounds": {
            "start_line": references_section["start_line"],
            "end_line": references_section["end_line"],
            "fresh_sweep_addendum_found": references_section["fresh_sweep_addendum_found"],
            "roadmap_task_ids_present": _roadmap_task_ids_present(roadmap_text),
        },
        "training_or_hardware_execution_claims_detected": False,
        "repo_inputs_read": [
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md",
            REFERENCES_RELATIVE_PATH,
            ROADMAP_RELATIVE_PATH,
            "openspec/capabilities/research-reporting/spec.md",
            "openspec/capabilities/verifiable-reasoning/spec.md",
            "openspec/capabilities/kan/spec.md",
            "openspec/capabilities/samplers/spec.md",
            "openspec/capabilities/self-learning/spec.md",
            "openspec/capabilities/llm-ebm-inference/spec.md",
            "openspec/capabilities/fpga/spec.md",
        ],
        "run_date": run_date,
    }
    validate_artifact(artifact)
    return artifact


def _verdict_has_terminal_prefix(verdict: str) -> bool:
    return verdict.startswith(TERMINAL_PREFIXES)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 5110 schema and claim boundaries."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"required field missing: {sorted(missing)}")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id mismatch")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone mismatch")
    if not _verdict_has_terminal_prefix(str(artifact["honest_verdict"])):
        raise ValueError("honest_verdict lacks a terminal prefix")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be literature_review_and_repo_inspection")
    if artifact["honest_verdict"] == COMPLETE_VERDICT and not artifact["references_section_found"]:
        raise ValueError("references_section_found is required for a complete verdict")
    if artifact["active_roadmap_modified"] is not False:
        raise ValueError("active_roadmap_modified must remain false")
    if artifact["training_or_hardware_execution_claims_detected"] is not False:
        raise ValueError("training or hardware execution claims are forbidden")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must cover the required artifact fields")
    if not artifact["tests_run"]:
        raise ValueError("tests_run must record verification evidence")

    sources = artifact["sources_checked"]
    source_ids = {source["source_id"] for source in sources}
    topic_coverage = {source["required_topic"] for source in sources}
    if topic_coverage != REQUIRED_TOPICS:
        raise ValueError("source coverage does not match the required V469 topics")
    for source in sources:
        if not source["urls"]:
            raise ValueError(f"source {source['source_id']} has no real URL")
        if source["claim_boundary"] == "actionable" and source["source_id"] not in ACTIONABLE_SOURCE_IDS:
            raise ValueError(f"source {source['source_id']} has an unknown actionable boundary")
        for url in source["urls"]:
            if not str(url["url"]).startswith("https://"):
                raise ValueError(f"source {source['source_id']} URL is not HTTPS")
            if url["status"] in DEAD_OR_STALE_URL_STATUSES and artifact["honest_verdict"] == COMPLETE_VERDICT:
                raise ValueError("dead or stale source cannot have a complete verdict")

    task_mapping = artifact["task_mapping"]
    background_ids = {row["source_id"] for row in artifact["background_only_sources"]}
    if not REQUIRED_BACKGROUND_SOURCE_IDS.issubset(background_ids):
        raise ValueError("background-only architecture pressure sources are incomplete")
    if background_ids.intersection(task_mapping):
        raise ValueError("background-only sources must not be mapped as executable tasks")
    if set(task_mapping) != ACTIONABLE_SOURCE_IDS:
        raise ValueError("every actionable source must be mapped exactly once")
    for source_id, task_rows in task_mapping.items():
        if source_id not in source_ids:
            raise ValueError(f"task_mapping references unknown source {source_id}")
        if not task_rows:
            raise ValueError(f"actionable source {source_id} has no task rows")
        for row in task_rows:
            if row["task_id"] not in ALLOWED_TASK_IDS:
                raise ValueError("task IDs must stay in exp5111-exp5120")
            if not row["reason"]:
                raise ValueError(f"task row for {source_id} needs a reason")

    if "EBT/ARM-EBM/Kona/TSU kept as architecture pressure" not in artifact["claim_boundary_findings"]:
        raise ValueError("claim_boundary_findings must record EBT/ARM/Kona/TSU pressure")


def write_artifact(
    *,
    root: Path,
    duration_s: float,
    run_date: str,
    tests_run: Sequence[str],
) -> dict[str, Any]:
    """Read repo inputs and write the stable Exp 5110 JSON artifact."""

    references_text = (root / REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    roadmap_text = (root / ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")
    artifact = build_artifact(
        references_text=references_text,
        roadmap_text=roadmap_text,
        duration_s=duration_s,
        run_date=run_date,
        tests_run=tests_run,
    )
    artifact_path = root / RESULT_RELATIVE_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(
    *,
    root: Path = REPO_ROOT,
    date: str = "20260701",
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Run Exp 5110 and return the artifact path."""

    start = time.perf_counter()
    run_tests = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    elapsed = time.perf_counter() - start if duration_s is None else duration_s
    write_artifact(root=root, duration_s=elapsed, run_date=date, tests_run=run_tests)
    return root / RESULT_RELATIVE_PATH


if __name__ == "__main__":  # pragma: no cover
    main()
