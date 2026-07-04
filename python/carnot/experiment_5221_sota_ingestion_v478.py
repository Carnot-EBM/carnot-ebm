"""Exp 5221 reserved SOTA ingestion for the V478 handoff.

Spec refs: REQ-REPORT-5221, SCENARIO-REPORT-5221,
SCENARIO-REPORT-5221-BLOCKED-METADATA.

This module records the live literature refresh that happens immediately before
the `.478` execution tasks consume the planning references. It is intentionally
small and deterministic: the web/API checks are represented as explicit evidence
rows, and the artifact refuses to convert blocked citation metadata or strategic
architecture posts into new roadmap work.
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
EXPERIMENT_ID = "experiment_5221_sota_ingestion_v478"
MILESTONE = "2026.07.478"
RUN_DATE = "2026-07-04"
SCHEMA = "carnot.experiment_5221_sota_ingestion_v478.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_5221_sota_ingestion_v478.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
INFERENCE_SUBSTRATE = "literature_ingestion"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
SPEC_REFS = [
    "REQ-REPORT-5221",
    "SCENARIO-REPORT-5221",
    "SCENARIO-REPORT-5221-BLOCKED-METADATA",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "sources_checked": (
        "The ingestion is only useful if it names the concrete source groups checked "
        "immediately before execution."
    ),
    "new_references_added": (
        "A zero value is valid when the refresh finds no genuinely new actionable "
        "finding beyond V478."
    ),
    "sota_to_experiment_mapping": (
        "Each fresh or rechecked source must either map to an existing .478 task or "
        "say defer/no action; no new roadmap task may be created here."
    ),
    "ebt_semantic_scholar_citation_count": (
        "EBT citation counts must come from the Semantic Scholar response, not memory; "
        "blocked queries must be labeled blocked."
    ),
    "arm_ebm_semantic_scholar_status": (
        "ARM-EBM rate limits or API errors must be recorded exactly rather than "
        "converted into a citation trail."
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

ALLOWED_478_EXPERIMENTS = {f"exp{idx}" for idx in range(5222, 5232)}

EBT_SEMANTIC_SCHOLAR_CITATION_COUNT = "blocked"

EBT_SEMANTIC_SCHOLAR_STATUS = (
    "HTTP/2 429\n"
    "content-type: application/json\n"
    "content-length: 174\n"
    "date: Sat, 04 Jul 2026 09:46:11 GMT\n"
    "x-amz-apigw-id: f-ZvmEb3PHcEeNg=\n"
    "x-amzn-requestid: 9de12c5c-0b50-415c-8319-9d8c1e104c76\n"
    "x-amzn-errortype: TooManyRequestsException\n"
    "x-cache: Error from cloudfront\n"
    "via: 1.1 e1c42f7e378e3bdce50f3034fd2550f4.cloudfront.net (CloudFront)\n"
    "x-amz-cf-pop: MIA3-P5\n"
    "x-amz-cf-id: jBENh58Rtx55Rb1UeoSPhh3WjevfuF5ZL_xCl9ljQQUbnbNKXJY84Q==\n\n"
    '{"message": "Too Many Requests. Please wait and try again or apply for a key '
    'for higher rate limits. https://www.semanticscholar.org/product/api#api-key-form", '
    '"code": "429"}'
)

ARM_EBM_SEMANTIC_SCHOLAR_STATUS = (
    "HTTP/2 429\n"
    "content-type: application/json\n"
    "content-length: 174\n"
    "date: Sat, 04 Jul 2026 09:46:11 GMT\n"
    "x-amz-apigw-id: f-ZvoFCfPHcEdIg=\n"
    "x-amzn-requestid: 7d82125d-10f3-4382-b704-63b793898ced\n"
    "x-amzn-errortype: TooManyRequestsException\n"
    "x-cache: Error from cloudfront\n"
    "via: 1.1 456dd60f1399d8458ed20abe4eae33a0.cloudfront.net (CloudFront)\n"
    "x-amz-cf-pop: MIA3-P5\n"
    "x-amz-cf-id: _cDDgKoZZVKWHViK8FkWvHEupHBXP32QcQVor622pRLVZF5INa-jFQ==\n\n"
    '{"message": "Too Many Requests. Please wait and try again or apply for a key '
    'for higher rate limits. https://www.semanticscholar.org/product/api#api-key-form", '
    '"code": "429"}'
)

SOURCE_URLS = [
    "https://export.arxiv.org/api/query?search_query=all:(EBM OR energy-based OR constraint OR Ising OR KAN OR constrained decoding OR hallucination OR continual learning) AND submittedDate:[202607040000 TO 202607042359]",
    "https://arxiv.org/abs/2606.16886",
    "https://arxiv.org/abs/2601.20055",
    "https://arxiv.org/abs/2510.09312",
    "https://arxiv.org/abs/2512.05439",
    "https://arxiv.org/abs/2606.01926",
    "https://arxiv.org/abs/2602.22647",
    "https://arxiv.org/abs/2607.01236",
    "https://arxiv.org/html/2606.18037v1",
    "https://arxiv.org/abs/2607.01224",
    "https://arxiv.org/abs/2607.01523",
    "https://arxiv.org/abs/2607.01874",
    "https://arxiv.org/abs/2607.02255",
    "https://arxiv.org/abs/2602.06737",
    "https://arxiv.org/abs/2607.01449",
    "https://arxiv.org/abs/2606.25313",
    "https://arxiv.org/abs/2602.15985",
    "https://openreview.net/forum?id=LYBs6f3jlK",
    "https://huggingface.co/papers/2512.05439",
    "https://huggingface.co/papers/2606.01926",
    "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092?fields=title,citationCount,externalIds,url",
    "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605?fields=title,citationCount,externalIds,url",
    "https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one",
    "https://extropic.ai/writing/inside-x0-and-xtr-0",
    "https://extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware",
    "https://logicalintelligence.com/blog/aleph-solves-putnambench",
    "https://logicalintelligence.com/blog/energy-based-models-for-reasoning",
    "https://logicalintelligence.com/blog/energy-based-model-sudoku-demo",
    "https://logicalintelligence.com/blog/automatic-formal-verification-for-code-generation",
    "https://github.com/facebookresearch/CRV",
    "https://github.com/youtube/static-constraint-decoding",
    "https://github.com/eth-sri/constrained-diffusion",
]

SOURCE_CHECKS: list[JsonDict] = [
    {
        "group": "local V477/V478 planning context",
        "checked": [
            "research-references.md V477 and V478 sections",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            "research-program.md Continuous Self-Learning",
            "ops/exclusion_manifest.yaml",
            "ops/known-issues.md",
        ],
        "evidence": (
            "V478 already maps verifier feedback, constrained generation, typed memory, "
            "KAN certificates, hardware continuity, and provenance repair to existing "
            "Exp 5223-5231 tasks. Retired Phase-D text scoring, MMLU hidden-state, "
            "ARC source-reading/BFS, and hardware speedup scopes remain closed."
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
            "arXiv API checked on 2026-07-04T09:46:11Z returned "
            "opensearch:totalResults=0 for the 20260704 submittedDate window. "
            "Primary arXiv URLs from the V478 block were rechecked for the selected "
            "anchors and did not reveal a new post-planning actionable reference."
        ),
    },
    {
        "group": "OpenReview page status",
        "checked": [
            "P-GCD forum LYBs6f3jlK",
            "constrained diffusion forum 7Sph4KyeYO",
            "CRV forum CxiNICq0Rr",
        ],
        "evidence": (
            "OpenReview forum HEAD for LYBs6f3jlK returned HTTP/2 307 to "
            "/challenge; the page remains a known source URL but did not expose new "
            "metadata that changes the .478 plan."
        ),
    },
    {
        "group": "Hugging Face Papers",
        "checked": [
            "BEAVER 2512.05439",
            "P-GCD 2606.01926",
            "constrained decoding and verifier paper pages",
        ],
        "evidence": (
            "HF Papers check found BEAVER's paper page as a V478 source. The P-GCD "
            "HF paper URL returned 404 with Hugging Face's 'Paper not found' message, "
            "so the arXiv/OpenReview URLs remain the authoritative sources."
        ),
    },
    {
        "group": "Semantic Scholar Graph API",
        "checked": ["arXiv:2507.02092", "arXiv:2512.15605"],
        "evidence": (
            "EBT arXiv:2507.02092 and ARM-EBM arXiv:2512.15605 both returned "
            "HTTP/2 429 with Too Many Requests bodies at 2026-07-04 09:46:11 UTC. "
            "No citation count is inferred."
        ),
    },
    {
        "group": "Extropic public writing",
        "checked": [
            "Thermodynamic Computing From Zero to One",
            "Inside X0 and XTR-0",
            "TSU 101",
        ],
        "evidence": (
            "Extropic writing URLs returned HTTP/2 200 during the refresh. They still "
            "support programmable stochastic sampler framing, not a Carnot speedup claim."
        ),
    },
    {
        "group": "Logical Intelligence public posts",
        "checked": [
            "Aleph solves PutnamBench",
            "Energy-Based Models for Reasoning",
            "Kona Sudoku demo",
            "Automatic formal verification for code generation",
        ],
        "evidence": (
            "Logical Intelligence posts returned HTTP/2 200 and continue to support "
            "verifier-first positioning. They do not expose a reproducible Carnot "
            "baseline or alter the VerIbmc-style Exp 5226 plan."
        ),
    },
    {
        "group": "GitHub repository refresh",
        "checked": [
            "facebookresearch/CRV",
            "youtube/static-constraint-decoding",
            "eth-sri/constrained-diffusion",
        ],
        "evidence": (
            "CRV and STATIC GitHub repository URLs returned HTTP/2 200. The repos "
            "remain implementation references for explicit verifier nodes and "
            "constrained-generation protocols already mapped into .478."
        ),
    },
]

SOTA_TO_EXPERIMENT_MAPPING: list[JsonDict] = [
    {
        "source": "arXiv 2026-07-04 date-window refresh",
        "mapped_task_or_defer": "defer/no action",
        "reason": (
            "The immediate post-planning arXiv date-window returned zero results, so "
            "there is no new reference to append."
        ),
    },
    {
        "source": "VerIbmc / VERGE / CRV / BEAVER verifier refresh",
        "mapped_task_or_defer": "exp5226",
        "reason": (
            "These sources remain the existing VerIbmc/local solver-feedback and "
            "explicit verifier-node design inputs; no new task is needed."
        ),
    },
    {
        "source": "P-GCD / STATIC constrained generation refresh",
        "mapped_task_or_defer": "exp5224",
        "reason": (
            "They continue to constrain GAP-4 canonical candidate-pool generation and "
            "protocol fields. STATIC's GitHub URL is live; P-GCD's OpenReview page is "
            "challenge-gated, so no new claim is added."
        ),
    },
    {
        "source": "ProvenanceGuard / source-aware factuality refresh",
        "mapped_task_or_defer": "exp5223",
        "reason": (
            "The sources remain protocol/provenance guardrails for the GAP-4 audit "
            "and canonical schema; they do not require roadmap changes."
        ),
    },
    {
        "source": "AutoMem / Multi-Head Memory refresh",
        "mapped_task_or_defer": "exp5227",
        "reason": (
            "They still map to typed constraint/provenance/failure/skill memory with "
            "promotion and rollback gates."
        ),
    },
    {
        "source": "SkillCoach / AgenticSTS refresh",
        "mapped_task_or_defer": "exp5228",
        "reason": (
            "They remain process-rubric and retention-test references for ARC live "
            "trace diagnosis before any gated live patch."
        ),
    },
    {
        "source": "KAN abstraction / GRS-KAN / KANFIS refresh",
        "mapped_task_or_defer": "exp5230",
        "reason": (
            "The KAN references still support a small PWA/MILP certificate pilot, not "
            "a broad KAN verification claim."
        ),
    },
    {
        "source": "p-bit / FPGA Ising / Extropic refresh",
        "mapped_task_or_defer": "exp5231",
        "reason": (
            "The sources remain hardware continuity and boundary-planning inputs. "
            "They do not justify a .478 speedup claim."
        ),
    },
    {
        "source": "EBT Semantic Scholar API refresh",
        "mapped_task_or_defer": "defer/no action",
        "reason": (
            "The exact response is HTTP/2 429 Too Many Requests. Citation count is "
            "recorded as blocked rather than inferred."
        ),
    },
    {
        "source": "ARM-EBM Semantic Scholar refresh",
        "mapped_task_or_defer": "defer/no action",
        "reason": (
            "The exact response is HTTP/2 429 Too Many Requests. No ARM-EBM citation "
            "trail is inferred."
        ),
    },
    {
        "source": "Logical Intelligence public posts refresh",
        "mapped_task_or_defer": "defer/no action",
        "reason": (
            "The posts reinforce verifier-first strategy but do not expose enough "
            "reproducible internals to become a local baseline."
        ),
    },
    {
        "source": "Hugging Face P-GCD page refresh",
        "mapped_task_or_defer": "defer/no action",
        "reason": (
            "HF returned a paper-not-found page for 2606.01926, so arXiv/OpenReview "
            "remain the source of record and no new Carnot action follows."
        ),
    },
]

SOURCE_API_RESPONSES = {
    "semantic_scholar_ebt_arxiv_2507_02092": EBT_SEMANTIC_SCHOLAR_STATUS,
    "semantic_scholar_arm_ebm_arxiv_2512_15605": ARM_EBM_SEMANTIC_SCHOLAR_STATUS,
    "arxiv_date_window_total_results": {
        "updated": "2026-07-04T09:46:11Z",
        "submitted_date_window": "[202607040000 TO 202607042359]",
        "total_results": 0,
    },
    "openreview_pgcd_status": (
        "HTTP/2 307 to /challenge?redirect=%2Fforum%3Fid%3DLYBs6f3jlK"
    ),
    "huggingface_pgcd_status": (
        "HTTP/2 404; x-error-message: Paper not found. For an arXiv paper to "
        "appear on Hugging Face, its arxiv.org URL needs to be mentioned in at "
        "least one model, dataset or Space's README.md."
    ),
}


def value_of(value: Any) -> Any:
    """Unwrap principle/value wrappers used by Carnot result artifacts.

    The repo stores important artifact values with a principle next to them so a
    future reader knows why the field exists. Validators need the raw value, but
    the wrapper must still be present to keep the research record auditable.
    """

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
    """Hash the artifact while excluding the checksum field itself."""

    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def build_artifact(
    *,
    tests_run: Sequence[str] | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Build the Exp 5221 receipt from verified V478 refresh evidence."""

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
        "ebt_semantic_scholar_citation_count": _principled(
            "ebt_semantic_scholar_citation_count",
            EBT_SEMANTIC_SCHOLAR_CITATION_COUNT,
        ),
        "arm_ebm_semantic_scholar_status": _principled(
            "arm_ebm_semantic_scholar_status", ARM_EBM_SEMANTIC_SCHOLAR_STATUS
        ),
        "retired_scope_reopened": _principled("retired_scope_reopened", False),
        "inference_substrate": _principled("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": _principled(
            "honest_verdict",
            "complete: V478 SOTA refresh found no new actionable findings beyond "
            "the planning section; research-references.md unchanged; Semantic "
            "Scholar returned HTTP/2 429 for both EBT and ARM-EBM, so no fresh "
            "citation trail was inferred.",
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


def _arm_status_is_exact(value: Any) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return value >= 0
    if not isinstance(value, str):
        return False
    if "citation_count" in value and any(char.isdigit() for char in value):
        return True
    rate_limited = (
        "HTTP/2 429" in value and "Too Many Requests" in value and '"code": "429"' in value
    )
    explicit_error = value.startswith("error:") and len(value) > len("error:")
    return rate_limited or explicit_error


def _ebt_count_is_exact_or_blocked(value: Any, api_responses: Mapping[str, Any]) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return value >= 0
    if value != "blocked":
        return False
    response = str(api_responses.get("semantic_scholar_ebt_arxiv_2507_02092", ""))
    return (
        "HTTP/2 429" in response and "Too Many Requests" in response and '"code": "429"' in response
    )


def _mapping_targets_existing_task(mapping: str) -> bool:
    if mapping == "defer/no action":
        return True
    head = mapping.split(" ", 1)[0]
    return head in ALLOWED_478_EXPERIMENTS


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject artifacts that blur blocked metadata into research progress."""

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

    ebt_count = values["ebt_semantic_scholar_citation_count"]
    if not _ebt_count_is_exact_or_blocked(ebt_count, api_responses):
        raise ValueError(
            "EBT Semantic Scholar citation count must be a non-negative int or "
            "blocked with exact API details"
        )
    if not _arm_status_is_exact(values["arm_ebm_semantic_scholar_status"]):
        raise ValueError("ARM-EBM status must include exact rate-limit/error details")

    mappings = values["sota_to_experiment_mapping"]
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("sota_to_experiment_mapping value must be a non-empty list")
    for row in mappings:
        if not isinstance(row, Mapping):
            raise ValueError("mapping rows must be objects")
        if not {"source", "mapped_task_or_defer", "reason"}.issubset(row):
            raise ValueError("mapping rows must include source, mapped_task_or_defer, and reason")
        if not _mapping_targets_existing_task(str(row["mapped_task_or_defer"])):
            raise ValueError("mapping must target an existing .478 task or defer/no action")

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
    """Write the Exp 5221 JSON artifact and leave references unchanged on no-op."""

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
