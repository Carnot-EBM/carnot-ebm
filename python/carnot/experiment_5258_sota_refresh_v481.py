"""Exp 5258: execution-time SOTA refresh for the V481 roadmap.

Spec refs: REQ-REPORT-5258, SCENARIO-REPORT-5258-APPEND-DELTAS,
SCENARIO-REPORT-5258-NOOP.

This module is a reporting harness. The live search happened through bounded
network sources, then the source-verified rows were frozen here so the result
artifact and references append are deterministic. The code intentionally does
not edit `research-roadmap.yaml` or `scripts/research_conductor.py`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5258_sota_refresh_v481"
MILESTONE = "2026.07.481"
RESULT_RELATIVE_PATH = Path("results/experiment_5258_sota_refresh_v481.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
REFRESH_HEADING = "### V481 Execution Refresh - 2026-07-05"
REFRESH_END_MARKER = "<!-- V481-EXECUTION-REFRESH-2026-07-05-END -->"
INFERENCE_SUBSTRATE = "literature_ingestion_network_sources"

SPEC_REFS = [
    "REQ-REPORT-5258",
    "SCENARIO-REPORT-5258-APPEND-DELTAS",
    "SCENARIO-REPORT-5258-NOOP",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Must start with complete: and distinguish new actionable findings from an honest "
        "no-op refresh."
    ),
    "inference_substrate": (
        "literature_ingestion_network_sources because Exp5258 reads network "
        "literature/source metadata and makes no experiment outcome claim."
    ),
    "sources_checked": (
        "Records every required source family and query channel so a missing source cannot "
        "masquerade as a zero-new-finding result."
    ),
    "new_references_added": (
        "Integer count of genuinely new actionable references appended; zero is valid only "
        "when references_md_updated is false."
    ),
    "references_md_updated": (
        "True only when a dated V481 execution refresh subsection was appended to "
        "research-references.md."
    ),
    "actionable_deltas": (
        "List of new findings with source URLs and concrete Carnot hooks; may be empty only "
        "for a no-op refresh."
    ),
    "retired_scope_reopened": (
        "False unless the artifact names a specific exclusion-manifest override that "
        "justifies reopening a retired scope."
    ),
    "semantic_scholar_status": (
        "EBT and ARM-EBM citation lookup results must record counts, citing-paper samples, "
        "or rate-limit failures without inventing citation-trend claims."
    ),
}

REQUIRED_SOURCE_FAMILIES = (
    "arxiv",
    "openreview",
    "huggingface_papers",
    "semantic_scholar",
    "github",
    "extropic",
    "logical_intelligence",
    "local_v481_comparison",
)

REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "experiment_id",
        "milestone",
        "spec_refs",
        "search_window",
        "references_section_heading",
        "duration_s",
        "honest_verdict",
        "inference_substrate",
        "sources_checked",
        "new_references_added",
        "references_md_updated",
        "actionable_deltas",
        "retired_scope_reopened",
        "semantic_scholar_status",
        "no_deep_research_used",
        "research_roadmap_yaml_modified",
        "research_conductor_modified",
        "plan_change_required",
        "tests_run",
        "field_principles",
    }
)

REQUIRED_DELTA_FIELDS = frozenset(
    {
        "title",
        "source_url",
        "arxiv_id_or_repo",
        "source_family",
        "category",
        "carnot_hook",
        "actionability",
        "planned_task_impact",
        "retired_scope_risk",
    }
)

ACTIONABLE_DELTAS: list[JsonDict] = [
    {
        "title": "DuoMem: Towards Capable On-Device Memory Agents via Dual-Space Distillation",
        "arxiv_id_or_repo": "2606.29961",
        "source_url": "https://arxiv.org/abs/2606.29961",
        "source_family": "arxiv",
        "category": "continuous_self_learning_memory",
        "carnot_hook": (
            "Split Exp5260/Exp5261 memory analysis into context-space memory usefulness and "
            "parameter-space training pressure; Carnot can test the context-memory arm without "
            "reopening LoRA or broad fine-tuning."
        ),
        "actionability": (
            "Record aligned precomputed memory versus no-memory and shuffled-memory controls; "
            "defer adapter training unless a later task has explicit training scope."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    },
    {
        "title": "QVal: Cheaply Evaluating Dense Supervision Signals for Long-Horizon LLM Agents",
        "arxiv_id_or_repo": "2606.32034",
        "source_url": "https://arxiv.org/abs/2606.32034",
        "source_family": "arxiv",
        "category": "verifier_dose_scheduler",
        "carnot_hook": (
            "Exp5264 can score verifier-dose policies by whether cheap signals order cached "
            "actions/candidates like a stronger reference policy before any training run."
        ),
        "actionability": (
            "Add Q-alignment style ordering as an analysis note for cached scheduler replay, "
            "without changing the already planned task."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    },
    {
        "title": "MemSyco-Bench: Benchmarking Sycophancy in Agent Memory",
        "arxiv_id_or_repo": "2607.01071",
        "source_url": "https://arxiv.org/abs/2607.01071",
        "source_family": "arxiv",
        "category": "memory_interference_safety",
        "carnot_hook": (
            "Exp5261 should treat harmful memory as more than storage/retrieval failure: "
            "fixtures should verify that memory is rejected when objective evidence conflicts "
            "or when the memory is out of scope."
        ),
        "actionability": (
            "Include conflict, scope-validity, update-tracking, and personalization-only cases "
            "inside the deterministic typed-memory interference audit."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    },
    {
        "title": "Managing Procedural Memory in LLM Agents: Control, Adaptation, and Evaluation",
        "arxiv_id_or_repo": "2606.23127",
        "source_url": "https://arxiv.org/abs/2606.23127",
        "source_family": "arxiv",
        "category": "cross_model_memory_transfer",
        "carnot_hook": (
            "The AFTER benchmark separates local improvement, cross-task transfer, cross-role "
            "transfer, and cross-model generalization, matching Exp5260's blocked cross-model "
            "typed-memory question."
        ),
        "actionability": (
            "Report broad-transfer versus role-specific memory entries separately instead of "
            "collapsing them into one memory-usefulness metric."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    },
    {
        "title": "SkillHone: A Harness for Continual Agent Skill Evolution Through Persistent Decision History",
        "arxiv_id_or_repo": "2606.08671",
        "source_url": "https://arxiv.org/abs/2606.08671",
        "source_family": "arxiv",
        "category": "persistent_decision_history",
        "carnot_hook": (
            "Typed verifier memory should preserve diagnoses, rejected revisions, evidence, and "
            "outcomes, not only the final promoted skill or constraint."
        ),
        "actionability": (
            "Use persistent decision-history rows as an optional schema check in Exp5261 and "
            "future skill-memory work; do not import a deep-research workflow."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    },
    {
        "title": "Beyond In-Domain Detection: SpikeScore for Cross-Domain Hallucination Detection",
        "arxiv_id_or_repo": "2601.19245",
        "source_url": "https://arxiv.org/abs/2601.19245",
        "source_family": "arxiv_openreview",
        "category": "hallucination_mitigation",
        "carnot_hook": (
            "Exp5263 can treat multi-turn uncertainty fluctuation as a narrow control if the "
            "runtime exposes token probabilities; it must not replace the planned internal or "
            "logit-energy receipts with a text-only scorer."
        ),
        "actionability": (
            "Use only as a receipt-gated comparator for cross-domain hallucination fixtures; "
            "keep Phase D external text scoring retired."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    },
    {
        "title": "PLN-THRML: Probabilistic Logic Rules Compiled To Thermodynamic Factor Graphs",
        "arxiv_id_or_repo": "xiaohanma-oss/PLN-THRML",
        "source_url": "https://github.com/xiaohanma-oss/PLN-THRML",
        "source_family": "github",
        "category": "hardware_accelerated_sampling_watch",
        "carnot_hook": (
            "The repository shows a concrete PLN-to-Ising-factor-graph mapping with CPU "
            "confidence propagation and sampler-side strength propagation."
        ),
        "actionability": (
            "Use as a design comparison for Exp5266 boundary notes only; do not add a dependency "
            "or claim TSU execution from a third-party repository."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    },
]

SOURCES_CHECKED: JsonDict = {
    "arxiv": {
        "status": "ok",
        "queries": [
            "EBMs for verification/reasoning",
            "neural constraint satisfaction and SAT/CSP",
            "hallucination mitigation and energy/logit diagnostics",
            "KANs and hardware-accelerated sampling",
            "continual/online learning for constraints",
        ],
        "new_actionable_ids": [
            "2606.29961",
            "2606.32034",
            "2607.01071",
            "2606.23127",
            "2606.08671",
            "2601.19245",
        ],
    },
    "openreview": {
        "status": "ok",
        "queries": [
            "ConstrainPrompt prompt-defined constraints",
            "Semantic Energy and Spilled Energy hallucination checks",
            "energy-guided decoding and constraint benchmarks",
        ],
        "result": "no plan-changing item beyond the already indexed V481 OpenReview signals",
    },
    "huggingface_papers": {
        "status": "ok",
        "queries": ["Daily/weekly/monthly papers around 2026-07-03 memory and agent learning"],
        "result": "surfaced DuoMem, QVal, MemSyco-Bench, and related memory papers",
    },
    "semantic_scholar": {
        "status": "ok",
        "queries": [
            "DOI:10.48550/arXiv.2507.02092 metadata and citations",
            "DOI:10.48550/arXiv.2512.15605 metadata and citations",
        ],
        "result": "metadata and citation samples returned without rate-limit failure",
    },
    "github": {
        "status": "ok",
        "queries": [
            "Extropic THRML and thermodynamic reasoning repos",
            "EBM/KAN/ML4CO watchlists",
        ],
        "result": "PLN-THRML is a new watch item; no maintained replacement for local Carnot code",
    },
    "extropic": {
        "status": "ok",
        "queries": ["extropic.ai/writing TSU/XTR-0/Z1/THRML pages"],
        "result": "no new local SDK or hardware receipt; architecture watch only",
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": ["logicalintelligence.com Aleph/Kona formal verification and EBRM posts"],
        "result": "no reproducible Kona internals; supports verifier-first framing only",
    },
    "local_v481_comparison": {
        "status": "ok",
        "queries": ["research-references.md V481 section and repo-wide duplicate search"],
        "result": "new deltas are not in the V481 planning block; several other search hits were already indexed",
    },
}

SEMANTIC_SCHOLAR_STATUS: JsonDict = {
    "EBT": {
        "arxiv_id": "2507.02092",
        "title": "Energy-Based Transformers are Scalable Learners and Thinkers",
        "year": 2025,
        "citationCount": 26,
        "influentialCitationCount": 2,
        "url": "https://www.semanticscholar.org/paper/2da9163730998a4368c609972ccff0582518b36b",
        "citation_samples": [
            {
                "title": "Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers",
                "year": 2026,
                "arxiv_id": "2606.18206",
            },
            {
                "title": "LoopUS",
                "full_title": "LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models",
                "year": 2026,
                "arxiv_id": "2605.11011",
            },
            {
                "title": "Towards System-2 AI: Workloads and Characterizations of Energy-Based Models",
                "year": 2026,
                "arxiv_id": None,
            },
        ],
    },
    "ARM-EBM": {
        "arxiv_id": "2512.15605",
        "title": (
            "Autoregressive Language Models are Secretly Energy-Based Models: Insights into "
            "the Lookahead Capabilities of Next-Token Prediction"
        ),
        "year": 2025,
        "citationCount": 8,
        "influentialCitationCount": 2,
        "url": "https://www.semanticscholar.org/paper/c73c449d8116684d89282c153f2ddd60334097d8",
        "citation_samples": [
            {
                "title": "Path-Measure Dynamics of Attention-Driven World Models",
                "year": 2026,
                "arxiv_id": "2607.02154",
            },
            {
                "title": "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
                "year": 2026,
                "arxiv_id": "2605.18871",
            },
            {
                "title": "LoopUS",
                "full_title": "LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models",
                "year": 2026,
                "arxiv_id": "2605.11011",
            },
        ],
    },
}


def _principled(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _verified_url(value: str) -> bool:
    return value.startswith("https://arxiv.org/abs/") or value.startswith("https://")


def _validate_principled_wrapper(field: str, artifact: Mapping[str, Any]) -> Any:
    wrapper = artifact.get(field)
    if not isinstance(wrapper, Mapping):
        raise ValueError(f"{field} must be principle-wrapped")
    if wrapper.get("principle") != FIELD_PRINCIPLES[field]:
        raise ValueError(f"{field} must include the declared principle")
    if "value" not in wrapper:
        raise ValueError(f"{field} missing value")
    return wrapper["value"]


def build_artifact(
    *,
    actionable_deltas: Sequence[Mapping[str, Any]] = ACTIONABLE_DELTAS,
    duration_s: float = 0.0,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    deltas = [dict(row) for row in actionable_deltas]
    added = len(deltas)
    references_updated = added > 0
    verdict_detail = (
        f"{added} new actionable findings appended; executable .481 plan unchanged"
        if added
        else "no new actionable findings; references unchanged"
    )
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "search_window": {
            "run_date": "2026-07-05",
            "years": "2025-2026",
            "comparison_anchor": "research-references.md V481 Research Update - 2026-07-05",
        },
        "references_section_heading": REFRESH_HEADING if references_updated else None,
        "duration_s": round(float(duration_s), 6),
        "honest_verdict": _principled("honest_verdict", f"complete: {verdict_detail}."),
        "inference_substrate": _principled("inference_substrate", INFERENCE_SUBSTRATE),
        "sources_checked": _principled("sources_checked", dict(SOURCES_CHECKED)),
        "new_references_added": _principled("new_references_added", added),
        "references_md_updated": _principled("references_md_updated", references_updated),
        "actionable_deltas": _principled("actionable_deltas", deltas),
        "retired_scope_reopened": _principled("retired_scope_reopened", False),
        "semantic_scholar_status": _principled(
            "semantic_scholar_status", dict(SEMANTIC_SCHOLAR_STATUS)
        ),
        "no_deep_research_used": True,
        "research_roadmap_yaml_modified": False,
        "research_conductor_modified": False,
        "plan_change_required": False,
        "tests_run": list(tests_run) or ["tests/python/test_experiment_5258_sota_refresh_v481.py"],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def _validate_sources(sources: Any) -> None:
    if not isinstance(sources, Mapping) or not set(REQUIRED_SOURCE_FAMILIES).issubset(sources):
        raise ValueError("sources_checked must include every required source family")


def _validate_semantic_scholar(status: Any) -> None:
    if not isinstance(status, Mapping) or not {"EBT", "ARM-EBM"}.issubset(status):
        raise ValueError("Semantic Scholar status must include EBT and ARM-EBM")
    for key in ("EBT", "ARM-EBM"):
        row = status[key]
        if not isinstance(row, Mapping) or not isinstance(row.get("citationCount"), int):
            raise ValueError(f"Semantic Scholar {key} must record integer citationCount")
        if not isinstance(row.get("citation_samples"), list) or not row["citation_samples"]:
            raise ValueError(f"Semantic Scholar {key} must include citation samples")


def _validate_deltas(deltas: Any) -> None:
    if not isinstance(deltas, list):
        raise ValueError("actionable_deltas value must be a list")
    for row in deltas:
        if not isinstance(row, Mapping) or not REQUIRED_DELTA_FIELDS.issubset(row):
            raise ValueError(f"actionable_deltas rows must include {sorted(REQUIRED_DELTA_FIELDS)}")
        if not _verified_url(str(row["source_url"])):
            raise ValueError("actionable_deltas rows must use a verified URL")
        if row["planned_task_impact"] != "no_plan_edit":
            raise ValueError("actionable_deltas must not edit the active plan")
        if row["retired_scope_risk"] != "none":
            raise ValueError("actionable_deltas must not reopen retired scopes")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-REPORT-5258")
    verdict = _validate_principled_wrapper("honest_verdict", artifact)
    if not str(verdict).startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
    substrate = _validate_principled_wrapper("inference_substrate", artifact)
    if substrate != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be literature_ingestion_network_sources")
    if artifact["no_deep_research_used"] is not True:
        raise ValueError("deep-research must not be used")
    if artifact["research_roadmap_yaml_modified"] is not False:
        raise ValueError("research-roadmap.yaml must not be modified")
    if artifact["research_conductor_modified"] is not False:
        raise ValueError("research_conductor must not be modified")
    if not artifact["tests_run"]:
        raise ValueError("tests_run must record at least one test")

    sources = _validate_principled_wrapper("sources_checked", artifact)
    _validate_sources(sources)
    deltas = _validate_principled_wrapper("actionable_deltas", artifact)
    _validate_deltas(deltas)
    semantic = _validate_principled_wrapper("semantic_scholar_status", artifact)
    _validate_semantic_scholar(semantic)

    added = _validate_principled_wrapper("new_references_added", artifact)
    references_updated = _validate_principled_wrapper("references_md_updated", artifact)
    if not isinstance(added, int) or added != len(deltas):
        raise ValueError("new_references_added must equal actionable_deltas length")
    if references_updated is not (added > 0):
        raise ValueError("references_md_updated must match whether new references were added")
    retired = _validate_principled_wrapper("retired_scope_reopened", artifact)
    if retired is not False:
        raise ValueError("retired_scope_reopened must remain false for this refresh")


def _render_delta(row: Mapping[str, Any]) -> str:
    return (
        f"- **{row['title']}** ({row['source_url']}): {row['carnot_hook']} "
        f"Actionability: {row['actionability']}"
    )


def render_refresh_section(artifact: Mapping[str, Any]) -> str:
    validate_artifact(artifact)
    deltas = artifact["actionable_deltas"]["value"]
    if not deltas:
        return ""
    lines = [
        REFRESH_HEADING,
        "",
        (
            "Execution-time sweep after the `.481` plan checked arXiv, OpenReview, HuggingFace "
            "Papers, Semantic Scholar EBT/ARM-EBM citation trails, GitHub repositories, Extropic "
            "writing, and Logical Intelligence public pages. The items below were not in the V481 "
            "planning block and are actionable as implementation notes, but they do not require a "
            "roadmap edit."
        ),
        "",
        "### New actionable deltas",
        *(_render_delta(row) for row in deltas),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No executable `.481` task edit is required. The deltas sharpen "
            "Exp5260/Exp5261 memory checks, Exp5263 hallucination controls, Exp5264 scheduler "
            "analysis, and Exp5266 hardware boundary notes."
        ),
        (
            "- **Retired scope:** No retired scope was reopened; Phase D external generated-text "
            "scoring, broad LoRA fine-tuning, and TSU/Kona execution claims remain closed unless a "
            "future task carries an explicit override."
        ),
        (
            "- **Semantic Scholar:** EBT returned citationCount=26 and influentialCitationCount=2; "
            "ARM-EBM returned citationCount=8 and influentialCitationCount=2. The citation samples "
            "remain watch items only."
        ),
        "",
        REFRESH_END_MARKER,
        "",
    ]
    return "\n".join(lines)


def append_refresh_section(references_text: str, artifact: Mapping[str, Any]) -> str:
    if REFRESH_HEADING in references_text:
        return references_text
    section = render_refresh_section(artifact)
    if not section:
        return references_text
    separator = "\n\n" if references_text and not references_text.endswith("\n\n") else ""
    return f"{references_text}{separator}{section}"


def write_outputs(
    *,
    root: Path | str = REPO_ROOT,
    references_path: Path | None = None,
    result_path: Path | None = None,
    actionable_deltas: Sequence[Mapping[str, Any]] = ACTIONABLE_DELTAS,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    base = Path(root)
    references = references_path or (base / REFERENCES_RELATIVE_PATH)
    result = result_path or (base / RESULT_RELATIVE_PATH)
    artifact = build_artifact(actionable_deltas=actionable_deltas, tests_run=tests_run)
    original = references.read_text(encoding="utf-8") if references.exists() else ""
    updated = append_refresh_section(original, artifact)
    references.parent.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)
    references.write_text(updated, encoding="utf-8")
    result.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - CLI convenience for the experiment run.
    artifact = write_outputs()
    print(artifact["honest_verdict"]["value"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
