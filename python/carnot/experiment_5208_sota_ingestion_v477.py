"""Exp 5208 reserved SOTA ingestion for the V477 handoff.

Spec refs: REQ-REPORT-5208, SCENARIO-REPORT-5208,
SCENARIO-REPORT-5208-BLOCKED-METADATA.

This module is deliberately deterministic. The live literature checks happen
outside unit tests through low-fanout web/API requests, then this file records
the evidence rows that were actually observed. That separation matters because
the result artifact is a planning receipt, not a benchmark: future conductor
steps need to know which sources were checked, what changed, and whether any
roadmap task should move. If nothing actionable changed, the honest output is a
no-op refresh with `research-references.md` left untouched.
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
EXPERIMENT_ID = "experiment_5208_sota_ingestion_v477"
MILESTONE = "2026.07.477"
RUN_DATE = "2026-07-04"
SCHEMA = "carnot.experiment_5208_sota_ingestion_v477.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_5208_sota_ingestion_v477.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
INFERENCE_SUBSTRATE = "literature_ingestion"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
SPEC_REFS = [
    "REQ-REPORT-5208",
    "SCENARIO-REPORT-5208",
    "SCENARIO-REPORT-5208-BLOCKED-METADATA",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "sources_checked": (
        "The ingestion is only useful if it names the concrete source groups checked "
        "immediately before execution."
    ),
    "new_references_added": (
        "A zero value is valid when the refresh finds no genuinely new actionable "
        "finding beyond V477."
    ),
    "sota_to_experiment_mapping": (
        "Each fresh or rechecked source must either map to an existing .477 task or "
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

ALLOWED_477_EXPERIMENTS = {f"exp{idx}" for idx in range(5209, 5220)}

EBT_SEMANTIC_SCHOLAR_CITATION_COUNT = "blocked"

EBT_SEMANTIC_SCHOLAR_STATUS = (
    "HTTP/2 429\n"
    "content-type: application/json\n"
    "content-length: 174\n"
    "date: Sat, 04 Jul 2026 01:54:45 GMT\n"
    "x-amz-apigw-id: f9Ur6FDhPHcEZiQ=\n"
    "x-amzn-requestid: 9245e85a-c598-44ad-80c2-f05f9db7162b\n"
    "x-amzn-errortype: TooManyRequestsException\n"
    "x-cache: Error from cloudfront\n"
    "via: 1.1 456dd60f1399d8458ed20abe4eae33a0.cloudfront.net (CloudFront)\n"
    "x-amz-cf-pop: MIA3-P5\n"
    "x-amz-cf-id: SbytvRMZm0ZiHqapWmZXvkOUt7mfBKLUR1BRSmNU6U4uh7iTdtpPYA==\n\n"
    '{"message": "Too Many Requests. Please wait and try again or apply for a key '
    'for higher rate limits. https://www.semanticscholar.org/product/api#api-key-form", '
    '"code": "429"}'
)

ARM_EBM_SEMANTIC_SCHOLAR_STATUS = (
    "HTTP/2 429\n"
    "content-type: application/json\n"
    "content-length: 174\n"
    "date: Sat, 04 Jul 2026 01:54:45 GMT\n"
    "x-amz-apigw-id: f9Ur8EmkPHcEQEQ=\n"
    "x-amzn-requestid: 71253428-4ec4-4425-8dc5-b6e67336f62f\n"
    "x-amzn-errortype: TooManyRequestsException\n"
    "x-cache: Error from cloudfront\n"
    "via: 1.1 068df0c205693925392105783899e172.cloudfront.net (CloudFront)\n"
    "x-amz-cf-pop: MIA3-P5\n"
    "x-amz-cf-id: BnSCH9LvR-Rar4Mx8SAbeHRaE20UFPSQGnDmtGJlVuQq5CSa9mayvg==\n\n"
    '{"message": "Too Many Requests. Please wait and try again or apply for a key '
    'for higher rate limits. https://www.semanticscholar.org/product/api#api-key-form", '
    '"code": "429"}'
)

SOURCE_URLS = [
    "https://export.arxiv.org/api/query?search_query=(EBM+OR+constraint+OR+Ising+OR+KAN+OR+constrained+decoding+OR+hallucination+OR+continual+learning)+AND+submittedDate:20260703-20260704",
    "https://arxiv.org/abs/2607.02512",
    "https://arxiv.org/abs/2607.01585",
    "https://arxiv.org/abs/2605.28020",
    "https://arxiv.org/abs/2507.02092",
    "https://arxiv.org/abs/2512.15605",
    "https://huggingface.co/papers/2607.02512",
    "https://huggingface.co/papers/2505.23061",
    "https://openreview.net/forum?id=ZBj3Qp1bYg",
    "https://openreview.net/forum?id=7Sph4KyeYO",
    "https://openreview.net/forum?id=QYrzaPAqnX",
    "https://openreview.net/forum?id=PAQsJGXtnV",
    "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092?fields=title,citationCount,externalIds,url",
    "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605?fields=title,citationCount,externalIds,url",
    "https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one",
    "https://extropic.ai/writing/inside-x0-and-xtr-0",
    "https://logicalintelligence.com/blog/automatic-formal-verification-for-code-generation",
    "https://logicalintelligence.com/blog/energy-based-models-for-reasoning",
    "https://logicalintelligence.com/blog/aleph-leading-benchmarks",
]

SOURCE_CHECKS: list[JsonDict] = [
    {
        "group": "local planning context",
        "checked": [
            "research-references.md V476/V477 sections",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            "research-program.md Continuous Self-Learning",
            "ops/exclusion_manifest.yaml",
        ],
        "evidence": (
            "The V477 section already maps FALCON, distributional EBMs, EBD, "
            "STV/DeepVerifier, ADVENT, PAW, EBT, KAN variants, Extropic, and "
            "Logical Intelligence into .477 task design. The exclusion manifest "
            "keeps retired scopes closed."
        ),
    },
    {
        "group": "arXiv API date-window refresh",
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
            "arXiv API updated 2026-07-04T01:54:45Z and returned "
            "opensearch:totalResults=0 for the submittedDate window "
            "[202607030000 TO 202607042359]."
        ),
    },
    {
        "group": "OpenReview search/API status",
        "checked": [
            "Energy-Based Transformers page",
            "constrained diffusion decoding page",
            "KAN pages",
            "hallucination mitigation pages",
        ],
        "evidence": (
            "Search resurfaced EBT, constrained diffusion decoding, KAN, and "
            "hallucination-mitigation pages. Browser fetch read the premise-"
            "verification OpenReview page, while selected forum/API reads still "
            "redirected to challenge-gated pages. No OpenReview result changed "
            "the V477 task mapping."
        ),
    },
    {
        "group": "Hugging Face Papers",
        "checked": [
            "energy-based decoding",
            "constrained decoding",
            "hallucination mitigation",
            "KAN constraints",
            "continual self-learning constraints",
        ],
        "evidence": (
            "HF Papers showed Program-as-Weights as the July 3 submitted #1 paper "
            "with models and a dataset, already covered by V477's PAW entry. The "
            "constrained-decoding query resurfaced DINGO and older constrained "
            "decoding work, but nothing created a new .477 task."
        ),
    },
    {
        "group": "Semantic Scholar Graph API",
        "checked": ["arXiv:2507.02092", "arXiv:2512.15605"],
        "evidence": (
            "EBT arXiv:2507.02092 and ARM-EBM arXiv:2512.15605 both returned "
            "HTTP/2 429 with Too Many Requests bodies, so the planning-time EBT "
            "citation count was not reused as a fresh metadata claim."
        ),
    },
    {
        "group": "Extropic public writing",
        "checked": [
            "Thermodynamic Computing From Zero to One",
            "Inside X0 and XTR-0",
        ],
        "evidence": (
            "The public TSU/XTR-0 framing remains sampling from programmable EBMs "
            "with thrml simulation/replication links. No new public hardware "
            "architecture update changes the .477 hardware-continuity task."
        ),
    },
    {
        "group": "Logical Intelligence public posts",
        "checked": [
            "automatic formal verification for code generation",
            "energy-based models for reasoning",
            "Aleph leading benchmarks",
            "Kona/Aleph search results",
        ],
        "evidence": (
            "The posts still support the verifier-first/product-strategy stance: "
            "Kona as an EBRM-style reasoning model and Aleph as orchestration. "
            "They do not expose a reproducible baseline that would change .477."
        ),
    },
]

SOTA_TO_EXPERIMENT_MAPPING: list[JsonDict] = [
    {
        "source": "arXiv date-window 2026-07-03..2026-07-04",
        "mapped_task_or_defer": "defer/no action",
        "reason": (
            "The fresh submittedDate window returned zero relevant entries, so "
            "there is no reference to append and no experiment to retarget."
        ),
    },
    {
        "source": "Program-as-Weights HF/arXiv refresh",
        "mapped_task_or_defer": "exp5215",
        "reason": (
            "PAW remains the amortization-gate input already selected by V477; "
            "the HF models/dataset links confirm availability but do not require "
            "a new task."
        ),
    },
    {
        "source": "ADVENT arXiv refresh",
        "mapped_task_or_defer": "exp5214",
        "reason": (
            "Predicate invention with Prolog verification is already the knowledge "
            "pool promotion pattern for the verifier-memory task."
        ),
    },
    {
        "source": "Energy-Based Decoding arXiv refresh",
        "mapped_task_or_defer": "exp5211 optional arm",
        "reason": (
            "EBD remains an optional decoding arm for SOTA local candidate "
            "generation if token-level control is available; no roadmap change."
        ),
    },
    {
        "source": "EBT arXiv refresh",
        "mapped_task_or_defer": "exp5213",
        "reason": (
            "The EBT source paper remains unchanged and continues to support "
            "intermediate/chunk/halting signals for hidden-state v3. The blocked "
            "Semantic Scholar call adds no new citation trail."
        ),
    },
    {
        "source": "EBT Semantic Scholar API refresh",
        "mapped_task_or_defer": "defer/no action",
        "reason": (
            "The exact API response is HTTP/2 429 Too Many Requests. The "
            "planning-time citation count is not reused as fresh evidence."
        ),
    },
    {
        "source": "ARM-EBM Semantic Scholar refresh",
        "mapped_task_or_defer": "defer/no action",
        "reason": (
            "The exact API response is HTTP/2 429 Too Many Requests. No citation "
            "trail is inferred from a blocked metadata query."
        ),
    },
    {
        "source": "Extropic TSU/XTR-0 refresh",
        "mapped_task_or_defer": "exp5217",
        "reason": (
            "Extropic continues to support hardware continuity as correctness/hash "
            "smokes only; no speedup claim or new hardware task is justified."
        ),
    },
    {
        "source": "Logical Intelligence Kona/Aleph refresh",
        "mapped_task_or_defer": "defer/no action",
        "reason": (
            "Public posts reinforce verifier-first strategy but still lack enough "
            "reproducible technical detail to become a Carnot baseline."
        ),
    },
    {
        "source": "OpenReview constrained diffusion decoding refresh",
        "mapped_task_or_defer": "defer/no action",
        "reason": (
            "The page/API were challenge-blocked and the search result points at "
            "diffusion constrained decoding while DiffusionGemma loading is a "
            "retired thread until upstream loader status materially changes."
        ),
    },
    {
        "source": "Hugging Face DINGO constrained dLLM refresh",
        "mapped_task_or_defer": "defer/no action",
        "reason": (
            "DINGO is useful background for constrained diffusion inference, but "
            "it does not override the DiffusionGemma loader retirement or create "
            "a .477 task."
        ),
    },
]

SOURCE_API_RESPONSES = {
    "semantic_scholar_ebt_arxiv_2507_02092": EBT_SEMANTIC_SCHOLAR_STATUS,
    "semantic_scholar_arm_ebm_arxiv_2512_15605": ARM_EBM_SEMANTIC_SCHOLAR_STATUS,
    "arxiv_date_window_total_results": {
        "updated": "2026-07-04T01:54:45Z",
        "submitted_date_window": "[202607030000 TO 202607042359]",
        "total_results": 0,
    },
    "openreview_api_status": {
        "selected_note_ids": ["7Sph4KyeYO", "JEWqDI4BEx", "QYrzaPAqnX", "PAQsJGXtnV"],
        "status": (
            "forum HEAD requests redirected to /challenge; browser fetch read "
            "PAQsJGXtnV, while other selected forum/API reads were challenge-gated"
        ),
    },
}


def value_of(value: Any) -> Any:
    """Unwrap principle/value fields so validators handle repo artifact variants.

    Many Carnot artifacts wrap important values with a human-readable principle.
    Tests and downstream checks should judge the value itself, while still
    verifying that the principle text is present and correct.
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
    """Hash the artifact without the hash field itself.

    The checksum lets a later reconciliation step detect accidental edits. The
    hash excludes `reproducibility_checksum`, otherwise calculating the checksum
    would change the artifact being hashed.
    """

    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def build_artifact(
    *,
    tests_run: Sequence[str] | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Build the Exp 5208 receipt from verified literature-refresh evidence."""

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
            "complete: V477 SOTA refresh found no new actionable findings beyond "
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
    return head in ALLOWED_477_EXPERIMENTS


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject artifacts that blur a blocked metadata check into progress.

    The validator is intentionally stricter than plain JSON-schema validation.
    This task is a literature ingestion receipt, so the dangerous failure modes
    are subtle: inventing a citation count, creating a new task by implication,
    or reopening a retired scope because a paper sounds relevant.
    """

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
            raise ValueError("mapping must target an existing .477 task or defer/no action")

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
    """Write the stable Exp 5208 JSON artifact and preserve references on no-op."""

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
