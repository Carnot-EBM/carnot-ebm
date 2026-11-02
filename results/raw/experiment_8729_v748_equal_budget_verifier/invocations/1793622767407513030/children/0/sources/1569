"""Exp 5234 reserved SOTA ingestion for the V479 execution handoff.

Spec refs: REQ-REPORT-5234, SCENARIO-REPORT-5234,
SCENARIO-REPORT-5234-BLOCKED-METADATA.

This module turns the live literature refresh into a deterministic receipt. The
important behavior is conservative: a reachable citation API can update
metadata, but it cannot create roadmap work unless the source changes an
existing .479 experiment decision.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5234_sota_ingestion_v479"
MILESTONE = "2026.07.479"
RUN_DATE = "2026-07-04"
SCHEMA = "carnot.experiment_5234_sota_ingestion_v479.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_5234_sota_ingestion_v479.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
INFERENCE_SUBSTRATE = "literature_ingestion"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
SPEC_REFS = [
    "REQ-REPORT-5234",
    "SCENARIO-REPORT-5234",
    "SCENARIO-REPORT-5234-BLOCKED-METADATA",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "sources_checked": (
        "The ingestion is only useful if it names the concrete source groups checked "
        "immediately before execution."
    ),
    "new_references_added": (
        "A zero value is valid when the refresh finds no genuinely new actionable "
        "finding beyond V479."
    ),
    "sota_to_experiment_mapping": (
        "Each fresh or rechecked source must either map to an existing .479 task or "
        "say defer/no action; no new roadmap task may be created here."
    ),
    "ebt_semantic_scholar_status": (
        "EBT metadata must come from the Semantic Scholar response and retry log, "
        "not memory; blocked queries must be labeled blocked."
    ),
    "arm_ebm_semantic_scholar_status": (
        "ARM-EBM metadata must come from the Semantic Scholar response and retry log, "
        "not memory; blocked queries must be labeled blocked."
    ),
    "retired_scope_reopened": (
        "A SOTA citation cannot override the exclusion manifest by implication."
    ),
    "inference_substrate": (
        "This workflow ingests literature and metadata only; it does not run model "
        "training or benchmarks."
    ),
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and distinguish new "
        "actionable findings from no-op refresh."
    ),
}

EXTRA_FIELD_PRINCIPLES: dict[str, str] = {
    "references_md_updated": (
        "research-references.md is updated only when new genuinely actionable references are added."
    ),
    "no_deep_research_used": (
        "The prompt forbids /deep-research, so the artifact must affirm it was not used."
    ),
    "research_conductor_py_untouched_confirmed": (
        "The reserved ingestion task must not modify scripts/research_conductor.py."
    ),
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_SCHEMA_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "spec_refs",
        "result_path",
        "references_path",
        "duration_s",
        "field_principles",
        "source_urls",
        "source_api_responses",
        "references_md_updated",
        "no_deep_research_used",
        "research_conductor_py_untouched_confirmed",
        "tests_run",
        "reproducibility_checksum",
        *REQUIRED_ARTIFACT_FIELDS,
    }
)

ALLOWED_479_EXPERIMENTS = {f"exp{idx}" for idx in range(5235, 5244)}

EBT_SEMANTIC_SCHOLAR_STATUS: JsonDict = {
    "raw_status": "attempt 1 HTTP/2 429 Too Many Requests; attempt 2 HTTP 200",
    "paper_id": "2da9163730998a4368c609972ccff0582518b36b",
    "title": "Energy-Based Transformers are Scalable Learners and Thinkers",
    "year": 2025,
    "citation_count": 26,
    "blocked": False,
    "backoff_behavior": ["attempt=1 http_code=429", "attempt=2 http_code=200"],
}

ARM_EBM_SEMANTIC_SCHOLAR_STATUS: JsonDict = {
    "raw_status": "attempt 1 HTTP/2 429 Too Many Requests; attempt 2 HTTP 200",
    "paper_id": "c73c449d8116684d89282c153f2ddd60334097d8",
    "title": (
        "Autoregressive Language Models are Secretly Energy-Based Models: "
        "Insights into the Lookahead Capabilities of Next-Token Prediction"
    ),
    "year": 2025,
    "citation_count": 8,
    "blocked": False,
    "backoff_behavior": ["attempt=1 http_code=429", "attempt=2 http_code=200"],
}

SOURCE_URLS = [
    "https://export.arxiv.org/api/query?search_query=all:(energy-based OR EBM OR constraint OR Ising OR KAN OR constrained decoding OR hallucination OR continual learning) AND submittedDate:[202607040000 TO 202607042359]",
    "https://arxiv.org/abs/2606.19404",
    "https://arxiv.org/abs/2606.00301",
    "https://arxiv.org/abs/2603.03748",
    "https://arxiv.org/abs/2602.18419",
    "https://arxiv.org/abs/2606.26476",
    "https://arxiv.org/abs/2606.02461",
    "https://arxiv.org/abs/2606.27892",
    "https://openreview.net/forum?id=LYBs6f3jlK",
    "https://openreview.net/pdf?id=E5mL07Fbq8",
    "https://openreview.net/pdf?id=EXFKk4Y3yc",
    "https://huggingface.co/papers/date/2026-07-03",
    "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092?fields=title,year,citationCount,externalIds,url",
    "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605?fields=title,year,citationCount,externalIds,url",
    "https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one",
    "https://extropic.ai/writing/inside-x0-and-xtr-0",
    "https://logicalintelligence.com/blog/energy-based-models-for-reasoning",
    "https://github.com/facebookresearch/CRV",
    "https://github.com/youtube/static-constraint-decoding",
    "https://github.com/eth-sri/constrained-diffusion",
    "https://github.com/Saibo-creator/Awesome-LLM-Constrained-Decoding",
]

SOURCE_API_RESPONSES: JsonDict = {
    "arxiv_date_window_total_results": {
        "updated": "2026-07-04T18:26:03Z",
        "submitted_date_window": "[202607040000 TO 202607042359]",
        "total_results": 0,
    },
    "openreview_pgcd_status": "HTTP/2 307 to /challenge followed by challenge page HTTP/2 200",
    "openreview_semantic_energy_status": "challenge-gated: /challenge?redirect=%2Fpdf%3Fid%3DE5mL07Fbq8",
    "openreview_spilled_energy_status": "challenge-gated: /challenge?redirect=%2Fpdf%3Fid%3DEXFKk4Y3yc",
    "huggingface_2026_07_03_status": "HTTP/2 200; page lists AgenticSTS, SkillCoach, AutoMem, DuoMem, and related memory/skill papers",
    "semantic_scholar_ebt_retry_log": ["attempt=1 http_code=429", "attempt=2 http_code=200"],
    "semantic_scholar_ebt_success_body": {
        "paperId": EBT_SEMANTIC_SCHOLAR_STATUS["paper_id"],
        "title": EBT_SEMANTIC_SCHOLAR_STATUS["title"],
        "year": EBT_SEMANTIC_SCHOLAR_STATUS["year"],
        "citationCount": EBT_SEMANTIC_SCHOLAR_STATUS["citation_count"],
    },
    "semantic_scholar_arm_ebm_retry_log": ["attempt=1 http_code=429", "attempt=2 http_code=200"],
    "semantic_scholar_arm_ebm_success_body": {
        "paperId": ARM_EBM_SEMANTIC_SCHOLAR_STATUS["paper_id"],
        "title": ARM_EBM_SEMANTIC_SCHOLAR_STATUS["title"],
        "year": ARM_EBM_SEMANTIC_SCHOLAR_STATUS["year"],
        "citationCount": ARM_EBM_SEMANTIC_SCHOLAR_STATUS["citation_count"],
    },
    "extropic_status": "HTTP/2 200 for thermodynamic-computing and inside-x0 pages",
    "logical_intelligence_status": "HTTP/2 200 for energy-based-models-for-reasoning",
    "github_repo_status": "HTTP/2 200 for CRV, STATIC, constrained-diffusion, and Awesome-LLM-Constrained-Decoding",
}

SOURCE_CHECKS: list[JsonDict] = [
    {
        "group": "local V478/V479 planning context",
        "checked": [
            "research-references.md V478 and V479 sections",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            "research-program.md Continuous Self-Learning",
            "ops/exclusion_manifest.yaml",
            "ops/known-issues.md",
        ],
        "evidence": (
            "The .479 roadmap already maps artifact QA, typed-memory controls, KAN "
            "certificate scaling, and hardware continuity to Exp 5235-5243. No "
            "retired scope was reopened by the refresh."
        ),
    },
    {
        "group": "arXiv API and primary abstract refresh",
        "checked": [
            "EBMs for verification",
            "constraint satisfaction",
            "Ising/hardware sampling",
            "hallucination mitigation",
            "KANs",
            "constrained or energy-guided decoding",
            "continual/self-learning for constraints",
        ],
        "evidence": (
            "The 2026-07-04 arXiv submitted-date window returned "
            "opensearch:totalResults=0. V479 primary arXiv anchors were rechecked "
            "as source URLs, but no post-planning actionable reference appeared."
        ),
    },
    {
        "group": "OpenReview page status",
        "checked": [
            "P-GCD forum LYBs6f3jlK",
            "Semantic Energy PDF E5mL07Fbq8",
            "Spilled Energy PDF EXFKk4Y3yc",
        ],
        "evidence": (
            "OpenReview still redirects direct pages/PDFs to a browser challenge. "
            "The search-visible titles remain watchlist-only and do not change .479."
        ),
    },
    {
        "group": "Hugging Face Papers",
        "checked": ["Daily Papers 2026-07-03", "memory and skill-transfer paper pages"],
        "evidence": (
            "The HF page is reachable and surfaces AgenticSTS, SkillCoach, AutoMem, "
            "DuoMem, and related skill/memory items. AgenticSTS, SkillCoach, and "
            "AutoMem were already in the V478/V479 references; DuoMem is related "
            "background but not a new experiment driver."
        ),
    },
    {
        "group": "Semantic Scholar Graph API",
        "checked": ["arXiv:2507.02092", "arXiv:2512.15605"],
        "evidence": (
            "Both metadata checks hit HTTP 429 on the first attempt and returned "
            "HTTP 200 on the second attempt. EBT has 26 citations; ARM-EBM has 8."
        ),
    },
    {
        "group": "Extropic public writing",
        "checked": [
            "Thermodynamic Computing From Zero to One",
            "Inside X0 and XTR-0",
        ],
        "evidence": (
            "Extropic pages remain reachable and continue to support TSU/EBM sampler "
            "watchlist status, not a local Carnot speedup claim."
        ),
    },
    {
        "group": "Logical Intelligence public posts",
        "checked": ["Energy-Based Models for Reasoning", "Kona/Aleph public links"],
        "evidence": (
            "Logical Intelligence still frames EBRMs as verifier/reasoning layers "
            "under LLM interfaces. It does not expose reproducible Kona internals "
            "that would change the .479 baseline set."
        ),
    },
    {
        "group": "GitHub repository refresh",
        "checked": [
            "facebookresearch/CRV",
            "youtube/static-constraint-decoding",
            "eth-sri/constrained-diffusion",
            "Awesome-LLM-Constrained-Decoding",
        ],
        "evidence": (
            "The watched repos are reachable. They remain implementation references "
            "for existing verifier-node and constrained-generation ideas, with no "
            "new roadmap task required."
        ),
    },
]

SOTA_TO_EXPERIMENT_MAPPING: list[JsonDict] = [
    {
        "source": "arXiv 2026-07-04 date-window refresh",
        "mapped_task_or_defer": "defer/no action",
        "reason": "The immediate submitted-date window returned zero results.",
    },
    {
        "source": "Free-Energy / FLaG hallucination diagnostics",
        "mapped_task_or_defer": "defer/no action",
        "reason": (
            "They remain future frozen-model diagnostic ideas. The .479 work is QA "
            "calibration and decision repair, not a new hidden-signal probe."
        ),
    },
    {
        "source": "JANUS / Hard-CSP constraint refresh",
        "mapped_task_or_defer": "exp5236",
        "reason": (
            "They reinforce the existing GAP-4 status decision discipline: do not "
            "regenerate ad hoc pools, and require deterministic baselines."
        ),
    },
    {
        "source": "RW-EBR / AgentCL controlled memory refresh",
        "mapped_task_or_defer": "exp5239",
        "reason": (
            "The five-arm and controlled-stream methods are the direct design input "
            "for the typed-memory ablation."
        ),
    },
    {
        "source": "HF July 3 memory and skill papers",
        "mapped_task_or_defer": "exp5239",
        "reason": (
            "AgenticSTS, SkillCoach, AutoMem, and related pages reinforce the "
            "existing controlled memory and skill-rubric tasks without adding scope."
        ),
    },
    {
        "source": "Analog KAN / KAN abstraction refresh",
        "mapped_task_or_defer": "exp5242",
        "reason": (
            "These sources remain bounded certificate/abstraction inputs, not a "
            "hardware-speedup claim."
        ),
    },
    {
        "source": "Extropic / p-bit hardware refresh",
        "mapped_task_or_defer": "exp5243",
        "reason": (
            "They support the hardware boundary/watchlist task while preserving "
            "no-speedup discipline."
        ),
    },
    {
        "source": "Logical Intelligence Kona/Aleph posts",
        "mapped_task_or_defer": "defer/no action",
        "reason": (
            "The posts support verifier-first architecture but do not expose a "
            "reproducible local baseline for .479."
        ),
    },
    {
        "source": "GitHub CRV/STATIC/constrained-diffusion repos",
        "mapped_task_or_defer": "exp5238",
        "reason": (
            "CRV and related repos remain implementation references for verifier "
            "node and solver-feedback methodology checks."
        ),
    },
    {
        "source": "Semantic Scholar EBT metadata retry",
        "mapped_task_or_defer": "defer/no action",
        "reason": "The citation count is metadata only and does not change any .479 experiment.",
    },
    {
        "source": "Semantic Scholar ARM-EBM metadata retry",
        "mapped_task_or_defer": "defer/no action",
        "reason": "The citation count is metadata only and does not change any .479 experiment.",
    },
]


def value_of(value: Any) -> Any:
    """Unwrap the value/principle records used in result artifacts."""

    if isinstance(value, Mapping) and "value" in value:
        return value_of(value["value"])
    return value


def _principled(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _extra_principled(field: str, value: Any) -> JsonDict:
    return {"principle": EXTRA_FIELD_PRINCIPLES[field], "value": value}


def _stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact content while excluding the checksum field itself."""

    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def build_artifact(
    *,
    tests_run: Sequence[str] | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Build the Exp 5234 receipt from verified V479 refresh evidence."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "references_path": str(REFERENCES_RELATIVE_PATH),
        "duration_s": duration_s,
        "field_principles": copy.deepcopy(FIELD_PRINCIPLES),
        "source_urls": list(SOURCE_URLS),
        "source_api_responses": copy.deepcopy(SOURCE_API_RESPONSES),
        "sources_checked": _principled("sources_checked", copy.deepcopy(SOURCE_CHECKS)),
        "new_references_added": _principled("new_references_added", 0),
        "sota_to_experiment_mapping": _principled(
            "sota_to_experiment_mapping", copy.deepcopy(SOTA_TO_EXPERIMENT_MAPPING)
        ),
        "ebt_semantic_scholar_status": _principled(
            "ebt_semantic_scholar_status", copy.deepcopy(EBT_SEMANTIC_SCHOLAR_STATUS)
        ),
        "arm_ebm_semantic_scholar_status": _principled(
            "arm_ebm_semantic_scholar_status",
            copy.deepcopy(ARM_EBM_SEMANTIC_SCHOLAR_STATUS),
        ),
        "retired_scope_reopened": _principled("retired_scope_reopened", False),
        "inference_substrate": _principled("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": _principled(
            "honest_verdict",
            "complete: V479 SOTA execution refresh found no new actionable findings "
            "beyond the planning section; research-references.md unchanged; Semantic "
            "Scholar metadata was reachable after one 429 retry for both EBT and "
            "ARM-EBM.",
        ),
        "references_md_updated": _extra_principled("references_md_updated", False),
        "no_deep_research_used": _extra_principled("no_deep_research_used", True),
        "research_conductor_py_untouched_confirmed": _extra_principled(
            "research_conductor_py_untouched_confirmed", True
        ),
        "tests_run": list(tests_run or ["not_yet_run"]),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _require_principled(artifact: Mapping[str, Any], field: str, principle: str) -> Any:
    wrapped = artifact[field]
    if not isinstance(wrapped, Mapping) or "value" not in wrapped or "principle" not in wrapped:
        raise ValueError(f"{field} must be principle-wrapped")
    if wrapped["principle"] != principle:
        raise ValueError(f"{field} does not match its declared principle")
    return wrapped["value"]


def _semantic_status_is_exact_or_blocked(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    backoff = value.get("backoff_behavior")
    if not isinstance(backoff, list) or not backoff:
        return False
    if value.get("blocked") is True:
        raw_status = value.get("raw_status")
        return (
            value.get("citation_count") == "blocked"
            and isinstance(raw_status, str)
            and bool(raw_status.strip())
        )
    if value.get("blocked") is not False:
        return False
    citation_count = value.get("citation_count")
    return (
        isinstance(value.get("raw_status"), str)
        and bool(str(value.get("raw_status")).strip())
        and isinstance(value.get("title"), str)
        and bool(str(value.get("title")).strip())
        and isinstance(value.get("year"), int)
        and not isinstance(value.get("year"), bool)
        and value["year"] >= 1900
        and isinstance(citation_count, int)
        and not isinstance(citation_count, bool)
        and citation_count >= 0
        and any("http_code=200" in str(item) for item in backoff)
    )


def _mapping_targets_existing_task(mapping: str) -> bool:
    if mapping == "defer/no action":
        return True
    return mapping.split(" ", 1)[0] in ALLOWED_479_EXPERIMENTS


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject artifacts that blur metadata refresh into new research progress."""

    missing = REQUIRED_SCHEMA_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match module principles")

    values = {
        field: _require_principled(artifact, field, principle)
        for field, principle in FIELD_PRINCIPLES.items()
    }
    references_updated = _require_principled(
        artifact, "references_md_updated", EXTRA_FIELD_PRINCIPLES["references_md_updated"]
    )
    no_deep_research_used = _require_principled(
        artifact, "no_deep_research_used", EXTRA_FIELD_PRINCIPLES["no_deep_research_used"]
    )
    conductor_untouched = _require_principled(
        artifact,
        "research_conductor_py_untouched_confirmed",
        EXTRA_FIELD_PRINCIPLES["research_conductor_py_untouched_confirmed"],
    )

    verdict = values["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal complete/success prefix")
    if values["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be literature_ingestion")
    if no_deep_research_used is not True:
        raise ValueError("deep-research must not be used")
    if conductor_untouched is not True:
        raise ValueError("conductor must remain untouched")

    sources_checked = values["sources_checked"]
    if not isinstance(sources_checked, list) or not sources_checked:
        raise ValueError("sources_checked value must be a non-empty list")
    for row in sources_checked:
        if not isinstance(row, Mapping) or not {"group", "checked", "evidence"}.issubset(row):
            raise ValueError("sources_checked rows must include group, checked, and evidence")

    new_references = values["new_references_added"]
    if (
        not isinstance(new_references, int)
        or isinstance(new_references, bool)
        or new_references < 0
    ):
        raise ValueError("new_references_added must be a non-negative int")
    if new_references > 0 and references_updated is not True:
        raise ValueError("references_md_updated must be true when new references are added")
    if new_references == 0 and references_updated is True:
        raise ValueError("new_references_added must be positive when references_md_updated is true")
    if values["retired_scope_reopened"] is not False:
        raise ValueError("retired_scope_reopened must stay false")

    api_responses = artifact.get("source_api_responses", {})
    if not isinstance(api_responses, Mapping):
        raise ValueError("source_api_responses must record exact API snippets")
    if not api_responses:
        raise ValueError("source_api_responses must record exact API snippets")

    if not _semantic_status_is_exact_or_blocked(values["ebt_semantic_scholar_status"]):
        raise ValueError(
            "EBT Semantic Scholar status must include exact metadata or blocked status"
        )
    if not _semantic_status_is_exact_or_blocked(values["arm_ebm_semantic_scholar_status"]):
        raise ValueError(
            "ARM-EBM Semantic Scholar status must include exact metadata or blocked status"
        )

    mappings = values["sota_to_experiment_mapping"]
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("sota_to_experiment_mapping value must be a non-empty list")
    for row in mappings:
        if not isinstance(row, Mapping):
            raise ValueError("mapping rows must be objects")
        if not {"source", "mapped_task_or_defer", "reason"}.issubset(row):
            raise ValueError("mapping rows must include source, mapped_task_or_defer, and reason")
        if not _mapping_targets_existing_task(str(row["mapped_task_or_defer"])):
            raise ValueError("mapping must target an existing .479 task or defer/no action")

    if not isinstance(artifact["source_urls"], list) or not artifact["source_urls"]:
        raise ValueError("source_urls must be a non-empty list")
    if not artifact["tests_run"]:
        raise ValueError("tests_run must be non-empty")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum does not match artifact content")


def maybe_append_references(existing_text: str, artifact: Mapping[str, Any]) -> str:
    """Return updated references text, or the original text for the no-op case."""

    new_references = value_of(artifact["new_references_added"])
    if new_references == 0:
        return existing_text
    append_text = artifact.get("reference_appendix")
    if not isinstance(append_text, str) or not append_text.strip():
        raise ValueError("reference_appendix is required when new references are added")
    if append_text in existing_text:
        return existing_text
    suffix = "" if existing_text.endswith("\n") else "\n"
    return f"{existing_text}{suffix}{append_text.rstrip()}\n"


def write_outputs(
    *,
    root: Path = REPO_ROOT,
    references_path: Path | None = None,
    result_path: Path | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Write the Exp 5234 JSON artifact and leave references unchanged on no-op."""

    root = Path(root)
    references_path = references_path or root / REFERENCES_RELATIVE_PATH
    result_path = result_path or root / RESULT_RELATIVE_PATH
    original_references = (
        references_path.read_text(encoding="utf-8") if references_path.exists() else ""
    )
    artifact = build_artifact(tests_run=tests_run)
    validate_artifact(artifact)

    updated_references = maybe_append_references(original_references, artifact)
    if updated_references != original_references:
        references_path.parent.mkdir(parents=True, exist_ok=True)
        references_path.write_text(updated_references, encoding="utf-8")

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(_stable_json(artifact) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--test-run", action="append", default=[])
    args = parser.parse_args(argv)

    artifact = write_outputs(root=args.root, tests_run=args.test_run or None)
    print(_stable_json(artifact))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
