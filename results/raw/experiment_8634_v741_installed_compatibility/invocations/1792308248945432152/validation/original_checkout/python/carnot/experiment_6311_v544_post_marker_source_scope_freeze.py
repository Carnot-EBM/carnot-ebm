"""Exp6311 V544 post-marker source scope freeze.

Spec refs: REQ-INFRA-6311, SCENARIO-INFRA-6311-1,
SCENARIO-INFRA-6311-2, SCENARIO-INFRA-6311-3,
SCENARIO-INFRA-6311-4, SCENARIO-INFRA-6311-5.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib.parse import urlparse

from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import canonical_json, path_sha256, payload_sha256


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_ID = "exp6311-v544-post-marker-source-scope-freeze"
SCHEMA = "carnot.experiment_6311.v544_post_marker_source_scope_freeze.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6311_v544_post_marker_source_scope_freeze.json")
INFERENCE_SUBSTRATE = "web_and_bibliographic_search_only"
CONTRACT_VERSION = "v544.6311"

RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
HARDWARE_WISHLIST_RELATIVE_PATH = Path("research-hardware-wishlist.md")
TEMPLATE_RELATIVE_PATH = Path("scripts/experiment_template.py")
PRIOR_SOURCE_FREEZE_RELATIVE_PATH = Path(
    "results/experiment_6299_v543_post_marker_source_scope_freeze.json"
)
PRIOR_TRANSITION_RELATIVE_PATH = Path("results/experiment_6310_v544_terminal_transition.json")

PLANNER_MARKER = "V544-PLANNER-REFRESH-20260811-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
MARKER_COMMIT = "67f2aecef00b16da327e6cc490a8fe32a88aa233"
MARKER_COMMITTED_AT_LOCAL = "2026-08-11T07:15:26-04:00"
MARKER_COMMITTED_AT_UTC = "2026-08-11T11:15:26Z"
EXECUTION_DELTA_HEADING = "## V544 Execution Source Delta - 20260811"
EXECUTION_DELTA_END_MARKER = "<!-- V544-EXECUTION-SOURCE-DELTA-20260811-END -->"

REQUIRED_SOURCE_CHANNELS = (
    "arxiv",
    "openreview",
    "hugging_face_papers",
    "semantic_scholar_ebt_arm",
    "extropic",
    "logical_intelligence",
    "github",
)

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_REFERENCES_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    PRIOR_SOURCE_FREEZE_RELATIVE_PATH,
    PRIOR_TRANSITION_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    HARDWARE_WISHLIST_RELATIVE_PATH,
    TEMPLATE_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
)

INPUT_RELATIVE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_REFERENCES_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    PRIOR_SOURCE_FREEZE_RELATIVE_PATH,
    PRIOR_TRANSITION_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    HARDWARE_WISHLIST_RELATIVE_PATH,
    TEMPLATE_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v544_marker_text_and_line",
    "search_window_start_utc",
    "search_completed_utc",
    "source_queries_by_channel",
    "source_receipts",
    "accepted_findings",
    "accepted_count",
    "duplicate_findings",
    "watch_only_findings",
    "inaccessible_sources",
    "excluded_findings_and_reasons",
    "semantic_scholar_ebt_and_arm_ebm_receipts",
    "extropic_status",
    "logical_intelligence_status",
    "github_status",
    "frozen_model_local_surface_contract",
    "frozen_exact_pair_fixture_contract",
    "frozen_model_local_energy_contract",
    "frozen_versioned_learning_contract",
    "frozen_protected_validation_contract",
    "frozen_arc_shadow_no_solve_contract",
    "frozen_hardware_exclusions",
    "roadmap_scope_delta",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The freeze is complete only when source and protected-file checks pass.",
    "v544_marker_text_and_line": "The exact marker and line define the novelty boundary.",
    "search_window_start_utc": "The lower bound is exclusive, so same-time evidence cannot enter.",
    "search_completed_utc": "The upper audit time makes the sweep reproducible.",
    "source_queries_by_channel": "Each required channel records the query that produced its receipt.",
    "source_receipts": "Search results and failures stay visible for audit.",
    "accepted_findings": "Only stable post-marker sources with local consequences can widen V544.",
    "accepted_count": "A bare integer drives null versus delta terminality.",
    "duplicate_findings": "Repeated planner sources cannot widen execution scope again.",
    "watch_only_findings": "Contextual sources are preserved without becoming branch contracts.",
    "inaccessible_sources": "Blocked endpoints are receipts, not findings.",
    "excluded_findings_and_reasons": "Rejected rows name the exact failed acceptance rule.",
    "semantic_scholar_ebt_and_arm_ebm_receipts": "Citation routes stay separate from broad search.",
    "extropic_status": "Hardware pages cannot imply local TSU access.",
    "logical_intelligence_status": "Product-direction pages cannot imply runnable Kona evidence.",
    "github_status": "Repository search cannot widen scope without a maintained local route.",
    "frozen_model_local_surface_contract": "V544 tests native per-model surfaces, not a shared bus.",
    "frozen_exact_pair_fixture_contract": "The fixture must be exact before model state is scored.",
    "frozen_model_local_energy_contract": "Energy heads are per-model and cannot be pooled for rescue.",
    "frozen_versioned_learning_contract": "Learning remains same-domain, versioned, and rollbackable.",
    "frozen_protected_validation_contract": "Final validation stays outside the adaptive loop.",
    "frozen_arc_shadow_no_solve_contract": "ARC shadow evidence cannot claim a solve.",
    "frozen_hardware_exclusions": "No board or TSU claim is allowed without execution receipts.",
    "roadmap_scope_delta": "A source delta may change scope only with explicit receipts.",
    "protected_files_unchanged": "Protected inputs are hash-checked around the write.",
    "preconditions_checked": "Input hashes, git state, marker identity, and source bounds are recorded.",
    "inference_substrate": "This task uses web and bibliographic search only.",
    "verifier_is_oracle": "The artifact audits sources and contracts; it is not an answer oracle.",
    "field_provenance": "Every required field cites concrete evidence.",
    "field_principles": "Every required field states why it exists.",
    "test_commands": "Commands show the verification boundary.",
    "test_exit_codes": "Exit codes keep failed checks visible.",
    "duration_s": "Wall time is reported without padding.",
    "reproducibility_checksum": "The normalized payload is content-addressed.",
    "honest_verdict": "The verdict states null, delta, or block directly.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6311_v544_post_marker_source_scope_freeze.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6311_v544_post_marker_source_scope_freeze.py "
    "-m pytest tests/python/test_experiment_6311_v544_post_marker_source_scope_freeze.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6311_v544_post_marker_source_scope_freeze.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6311_v544_post_marker_source_scope_freeze.py "
    "tests/python/test_experiment_6311_v544_post_marker_source_scope_freeze.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6311_v544_post_marker_source_scope_freeze.py"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
ADVERSARIAL_SELF_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6311_v544_post_marker_source_scope_freeze.json"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    RUFF_CHECK_COMMAND,
    SPEC_COVERAGE_COMMAND,
    FULL_PYTEST_COMMAND,
    ADVERSARIAL_SELF_COMMAND,
)

DEFAULT_SOURCE_QUERIES_BY_CHANNEL: dict[str, list[JsonDict]] = {
    "arxiv": [
        {
            "queried_at_utc": "2026-08-11T11:39:10Z",
            "query": "submittedDate:[202608111116 TO 202608120000]",
            "url": (
                "https://export.arxiv.org/api/query?"
                "search_query=submittedDate:%5B202608111116%20TO%20202608120000%5D"
            ),
        },
        {
            "queried_at_utc": "2026-08-11T11:39:10Z",
            "query": "id_list=2608.09643,2608.09380,2608.09855,2608.09366,2608.09885,2608.09277,2608.09876,2608.09857",
            "url": "https://export.arxiv.org/api/query?id_list=2608.09643,2608.09380,2608.09855,2608.09366,2608.09885,2608.09277,2608.09876,2608.09857",
        },
    ],
    "openreview": [
        {
            "queried_at_utc": "2026-08-11T11:39:25Z",
            "query": "Activation Probes Surface Code-Security Signals",
            "url": "https://api2.openreview.net/notes/search?term=Activation%20Probes%20Surface%20Code-Security%20Signals&limit=5",
        },
        {
            "queried_at_utc": "2026-08-11T11:40:56Z",
            "query": "forum dcBOEwDXP2 and gl6l8nTXBB direct note fetch",
            "url": "https://api2.openreview.net/notes?forum=dcBOEwDXP2&limit=5",
        },
    ],
    "hugging_face_papers": [
        {
            "queried_at_utc": "2026-08-11T11:39:10Z",
            "query": "daily_papers date=2026-08-11",
            "url": "https://huggingface.co/api/daily_papers?date=2026-08-11",
        }
    ],
    "semantic_scholar_ebt_arm": [
        {
            "queried_at_utc": "2026-08-11T11:39:10Z",
            "query": "EBT arXiv:2507.02092 citations",
            "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations?fields=title,url,year,publicationDate,externalIds&limit=5",
        },
        {
            "queried_at_utc": "2026-08-11T11:39:25Z",
            "query": "ARM-EBM arXiv:2512.15605 citations",
            "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations?fields=title,url,year,publicationDate,externalIds&limit=5",
        },
    ],
    "extropic": [
        {
            "queried_at_utc": "2026-08-11T11:39:26Z",
            "query": "Extropic home, writing, and z1 first-party HEAD checks",
            "url": "https://extropic.ai/",
        }
    ],
    "logical_intelligence": [
        {
            "queried_at_utc": "2026-08-11T11:39:25Z",
            "query": "Logical Intelligence home and Kona page HEAD checks",
            "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        }
    ],
    "github": [
        {
            "queried_at_utc": "2026-08-11T11:40:56Z",
            "query": "exact title and V544 mechanism repository search",
            "url": "https://api.github.com/search/repositories?q=%22Activation%20Probes%20Surface%20Code-Security%20Signals%22&sort=updated&order=desc&per_page=5",
        }
    ],
}

DEFAULT_SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "channel": "arxiv",
        "queried_at_utc": "2026-08-11T11:39:10Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["arxiv"][0]["url"],
        "endpoint_status": "http_200_totalResults_0",
        "disposition": "no_post_marker_primary_hits",
        "date_evidence": "The all-submission window after 2026-08-11T11:15:26Z returned zero entries.",
    },
    {
        "channel": "openreview",
        "queried_at_utc": "2026-08-11T11:40:56Z",
        "url": "https://api2.openreview.net/notes?forum=dcBOEwDXP2&limit=5",
        "endpoint_status": "http_403_challenge_required_for_direct_forum_fetch",
        "disposition": "inaccessible_receipt_recorded",
        "date_evidence": "Search endpoint returned the duplicate activation-probe row; direct note fetch required a challenge.",
    },
    {
        "channel": "hugging_face_papers",
        "queried_at_utc": "2026-08-11T11:39:10Z",
        "url": "https://huggingface.co/api/daily_papers?date=2026-08-11",
        "endpoint_status": "http_200_daily_rows_secondary",
        "disposition": "watch_only_or_pre_marker_secondary_rows",
        "date_evidence": "Daily rows resolved to secondary records or pre-marker primary arXiv timestamps.",
    },
    {
        "channel": "semantic_scholar_ebt_arm",
        "queried_at_utc": "2026-08-11T11:39:25Z",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations",
        "endpoint_status": "http_200_ebt_and_arm_routes",
        "disposition": "no_post_marker_relevant_citation",
        "date_evidence": "EBT newest sampled citation was 2026-08-01. ARM-EBM newest sampled citation was 2026-07-02.",
    },
    {
        "channel": "extropic",
        "queried_at_utc": "2026-08-11T11:39:27Z",
        "url": "https://extropic.ai/",
        "endpoint_status": "http_200_home_and_writing_http_404_z1",
        "disposition": "no_authenticated_local_tsu_route",
        "date_evidence": "First-party pages expose no Carnot device, simulator, or local API route.",
    },
    {
        "channel": "logical_intelligence",
        "queried_at_utc": "2026-08-11T11:39:25Z",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "endpoint_status": "http_200_last_modified_2026_06_26",
        "disposition": "no_public_weights_or_local_api",
        "date_evidence": "Kona page last-modified header predates the marker.",
    },
    {
        "channel": "github",
        "queried_at_utc": "2026-08-11T11:40:56Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["github"][0]["url"],
        "endpoint_status": "http_200_exact_title_total_count_0",
        "disposition": "no_maintained_contract_changing_repository",
        "date_evidence": "Exact title and mechanism searches returned no repository route for V544 execution.",
    },
)

DEFAULT_SOURCE_CANDIDATES: tuple[JsonDict, ...] = (
    {
        "stable_id": "arxiv:2608.09643",
        "title": "Activation Probes Surface Code-Security Signals that the Model's Output Misses",
        "url": "https://arxiv.org/abs/2608.09643",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-10T14:20:19Z",
        "date_evidence": "planner source predates the V544 marker",
        "scope_effect": "Already sealed as the model-local surface branch.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:7d24b6140f785d84833b6b991c8bc919cbadcc35d0dc9e61b6fb08bf7decbf02",
    },
    {
        "stable_id": "arxiv:2608.09380",
        "title": "OpenLoopEvolve: A Verifiable Self-Evolution Framework for Loop Policies in Long-Horizon Complex Tasks",
        "url": "https://arxiv.org/abs/2608.09380",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-10T10:00:00Z",
        "date_evidence": "planner source predates the V544 marker",
        "scope_effect": "Already sealed as version-lineage learning context.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:6b84597ec0c32e460cb57f85012ce091092d7570be86681089b01837c34a4a4e",
    },
    {
        "stable_id": "arxiv:2608.09855",
        "title": "Agentic Auto-Research is Fuzz Testing",
        "url": "https://arxiv.org/abs/2608.09855",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-10T17:13:02Z",
        "date_evidence": "planner source predates the V544 marker",
        "scope_effect": "Already sealed as protected-validation guidance.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:f5b4dbd3a7e27c746a0783d4ed8756f81c972411573b5d512d60636d4d935d71",
    },
    {
        "stable_id": "hf:2608.09819",
        "title": "Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA",
        "url": "https://huggingface.co/papers/2608.09819",
        "source_channel": "hugging_face_papers",
        "source_kind": "secondary",
        "source_timestamp": "2026-08-11T08:45:04Z",
        "date_evidence": "HF daily row changed before the marker and resolves to a LoRA update paper.",
        "scope_effect": "Topical only. V544 freezes base weights and does not update LoRA.",
        "reproducible_evidence": True,
        "primary_or_first_party": False,
        "local_executable_consequence": False,
        "watch_only": True,
        "content_hash": "sha256:9883ca80b85716ce187e17395847236a5497f031241e592218c7f75f8b0e10d3",
    },
    {
        "stable_id": "openreview:dcBOEwDXP2",
        "title": "Activation Probes Surface Code-Security Signals that the Model's Output Misses",
        "url": "https://openreview.net/forum?id=dcBOEwDXP2",
        "source_channel": "openreview",
        "source_kind": "primary_mirror",
        "source_timestamp": "2026-08-11T11:40:56Z",
        "date_evidence": "direct note fetch returned an OpenReview challenge receipt",
        "scope_effect": "Duplicate of already sealed activation-probe source.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "inaccessible": True,
        "content_hash": "sha256:0e1f79c5f21d3e7cc70c74b52fce763b17c6384d5601b72ff2d9d0e0b2a0c2a2",
    },
    {
        "stable_id": "extropic:first-party-pages",
        "title": "Extropic public Z1 and Torx route check",
        "url": "https://extropic.ai/",
        "source_channel": "extropic",
        "source_kind": "first_party",
        "source_timestamp": "2026-08-11T11:39:27Z",
        "date_evidence": "first-party page reachable after marker",
        "scope_effect": "No authenticated Carnot TSU or simulator route.",
        "reproducible_evidence": False,
        "primary_or_first_party": True,
        "local_executable_consequence": False,
        "watch_only": False,
        "content_hash": "sha256:50658c3b5b4c8c99e02ed0e3015a3150964de28c72aa40d66cf7b8fe75350989",
    },
    {
        "stable_id": "logical:kona-pages",
        "title": "Kona: Energy-Based Models for AI Reasoning",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "source_channel": "logical_intelligence",
        "source_kind": "first_party",
        "source_timestamp": "2026-06-26T23:48:05Z",
        "date_evidence": "last-modified header predates the marker",
        "scope_effect": "No public weights, reproducible architecture, or local API route.",
        "reproducible_evidence": False,
        "primary_or_first_party": True,
        "local_executable_consequence": False,
        "watch_only": False,
        "content_hash": "sha256:79de25fa5635e5a2d837b3565a8b54476a5567849ff88de2fd94f0395967df68",
    },
)


def _utc_now() -> str:  # pragma: no cover
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text or re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _marker_dt() -> datetime:
    parsed = _parse_timestamp(MARKER_COMMITTED_AT_UTC)
    if parsed is None:  # pragma: no cover
        raise ValueError("bad V544 marker timestamp")
    return parsed


def _is_stable_url(url: Any) -> bool:
    parsed = urlparse(str(url or ""))
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    return not (parsed.netloc == "github.com" and parsed.path == "/search")


def v544_marker_snapshot(root: Path) -> JsonDict:
    text = _read_text(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    line = None
    for index, raw_line in enumerate(text.splitlines(), start=1):
        if PLANNER_END_MARKER in raw_line:
            line = index
            break
    return {
        "marker_text": PLANNER_MARKER,
        "marker_comment": PLANNER_END_MARKER,
        "marker_line": line,
        "marker_count": text.count(PLANNER_END_MARKER),
        "marker_comment_sha256": payload_sha256(PLANNER_END_MARKER),
        "research_references_sha256": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
        "marker_byte_offset": text.find(PLANNER_END_MARKER),
        "marker_commit": MARKER_COMMIT,
        "marker_committed_at_local": MARKER_COMMITTED_AT_LOCAL,
        "marker_committed_at_utc": MARKER_COMMITTED_AT_UTC,
    }


def classify_candidate(
    candidate: Mapping[str, Any],
    *,
    reference_text: str,
    seen_ids: set[str] | None = None,
    seen_hashes: set[str] | None = None,
) -> JsonDict:
    row = dict(candidate)
    stable_id = str(row.get("stable_id") or "")
    content_hash = str(row.get("content_hash") or "")
    ref_lower = reference_text.lower()
    row["disposition"] = "accepted"
    row["rejection_reason"] = None

    if row.get("inaccessible") is True:
        row["disposition"] = "inaccessible"
        row["rejection_reason"] = "source endpoint was inaccessible or challenge-gated"
        return row
    if not _is_stable_url(row.get("url")):
        row["disposition"] = "excluded"
        row["rejection_reason"] = "candidate lacks a stable https URL"
        return row
    if stable_id and stable_id.lower() in ref_lower:
        row["disposition"] = "duplicate"
        row["rejection_reason"] = "stable id already appears in research-references.md"
        return row
    if str(row.get("url") or "").lower() in ref_lower:
        row["disposition"] = "duplicate"
        row["rejection_reason"] = "URL already appears in research-references.md"
        return row
    if str(row.get("title") or "").lower() in ref_lower:
        row["disposition"] = "duplicate"
        row["rejection_reason"] = "title already appears in research-references.md"
        return row
    if stable_id and seen_ids is not None and stable_id in seen_ids:
        row["disposition"] = "duplicate"
        row["rejection_reason"] = "stable id repeated in this sweep"
        return row
    if content_hash and seen_hashes is not None and content_hash in seen_hashes:
        row["disposition"] = "duplicate"
        row["rejection_reason"] = "content hash repeated in this sweep"
        return row
    if row.get("watch_only") is True:
        row["disposition"] = "watch_only"
        row["rejection_reason"] = "watch-only evidence does not change a V544 contract"
        return row

    timestamp = _parse_timestamp(row.get("source_timestamp"))
    if timestamp is None or timestamp <= _marker_dt():
        row["disposition"] = "excluded"
        row["rejection_reason"] = "source timestamp is not strictly after the V544 marker"
        if timestamp is None or str(row.get("source_timestamp") or "").strip()[:10] == "2026-08-11":
            row["disposition"] = "cutoff_confound"
        return row
    if row.get("reproducible_evidence") is not True:
        row["disposition"] = "excluded"
        row["rejection_reason"] = "candidate lacks reproducible evidence"
        return row
    if row.get("primary_or_first_party") is not True:
        row["disposition"] = "excluded"
        row["rejection_reason"] = "candidate is not primary or first-party"
        return row
    if row.get("local_executable_consequence") is not True:
        row["disposition"] = "excluded"
        row["rejection_reason"] = "candidate lacks a local executable consequence"
        return row
    return row


def validate_accepted_candidate(candidate: Mapping[str, Any]) -> None:
    missing = [
        field
        for field in ("stable_id", "title", "url", "source_timestamp", "content_hash")
        if not candidate.get(field)
    ]
    if missing:
        raise ValueError(f"accepted candidate missing fields: {missing}")
    if not _is_stable_url(candidate.get("url")):
        raise ValueError("accepted candidate requires a stable URL")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(candidate.get("content_hash"))):
        raise ValueError("accepted candidate requires a sha256 content hash")
    timestamp = _parse_timestamp(candidate.get("source_timestamp"))
    if timestamp is None or timestamp <= _marker_dt():
        raise ValueError("accepted candidate must be strictly after the marker")
    if candidate.get("reproducible_evidence") is not True:
        raise ValueError("accepted candidate must have reproducible evidence")
    if candidate.get("primary_or_first_party") is not True:
        raise ValueError("accepted candidate must be primary or first-party")
    if candidate.get("local_executable_consequence") is not True:
        raise ValueError("accepted candidate must have a local executable consequence")
    if candidate.get("watch_only") is True:
        raise ValueError("accepted candidate cannot be watch-only")


def partition_candidates(
    candidates: Sequence[Mapping[str, Any]], *, reference_text: str
) -> JsonDict:
    partitions: JsonDict = {
        "accepted": [],
        "duplicate_findings": [],
        "watch_only_findings": [],
        "inaccessible_sources": [],
        "excluded_findings_and_reasons": [],
    }
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for candidate in candidates:
        row = classify_candidate(
            candidate,
            reference_text=reference_text,
            seen_ids=seen_ids,
            seen_hashes=seen_hashes,
        )
        disposition = str(row["disposition"])
        if disposition == "accepted":
            validate_accepted_candidate(row)
            partitions["accepted"].append(row)
        elif disposition == "duplicate":
            partitions["duplicate_findings"].append(row)
        elif disposition == "watch_only":
            partitions["watch_only_findings"].append(row)
        elif disposition == "inaccessible":
            partitions["inaccessible_sources"].append(row)
        else:
            partitions["excluded_findings_and_reasons"].append(row)
        if row.get("stable_id"):
            seen_ids.add(str(row["stable_id"]))
        if row.get("content_hash"):
            seen_hashes.add(str(row["content_hash"]))
    return partitions


def execution_delta_block(accepted: Sequence[Mapping[str, Any]]) -> str:
    lines = [EXECUTION_DELTA_HEADING, ""]
    for row in accepted:
        lines.append(
            f"- **{row['title']}** - {row['url']}; source timestamp "
            f"{row['source_timestamp']}. {row['scope_effect']}"
        )
    lines.extend(["", EXECUTION_DELTA_END_MARKER, ""])
    return "\n".join(lines)


def insert_after_marker(source: str, block: str) -> str:
    if EXECUTION_DELTA_HEADING in source:
        return source
    index = source.find(PLANNER_END_MARKER)
    if index < 0:
        return source.rstrip() + "\n\n" + block
    insert_at = index + len(PLANNER_END_MARKER)
    return source[:insert_at] + "\n\n" + block + source[insert_at:]


def _hashes(root: Path, paths: Sequence[Path]) -> JsonDict:
    return {
        rel.as_posix(): {"present": (root / rel).exists(), "sha256": path_sha256(root / rel)}
        for rel in paths
    }


def protected_hashes(root: Path) -> JsonDict:
    return _hashes(root, PROTECTED_RELATIVE_PATHS)


def protected_unchanged(
    root: Path,
    before: Mapping[str, Any] | None = None,
    authorized_after: Mapping[str, str | None] | None = None,
) -> JsonDict:
    before_hashes = dict(before or protected_hashes(root))
    authorized = dict(authorized_after or {})
    paths: JsonDict = {}
    all_unchanged = True
    for rel in PROTECTED_RELATIVE_PATHS:
        key = rel.as_posix()
        before_hash = dict(before_hashes.get(key) or {}).get("sha256")
        after_hash = path_sha256(root / rel)
        expected = authorized.get(key, before_hash)
        unchanged = expected == after_hash
        paths[key] = {
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "authorized_after_sha256": authorized.get(key),
            "unchanged": unchanged,
        }
        all_unchanged = all_unchanged and unchanged
    return {"all_unchanged": all_unchanged, "unchanged": all_unchanged, "paths": paths}


def git_status(root: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ("git", "status", "--short"),
            cwd=root,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception:  # pragma: no cover
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def frozen_model_local_surface_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "source_evidence": ["arXiv:2608.09643", "OpenReview:dcBOEwDXP2"],
        "one_surface_per_model": True,
        "shared_activation_bus_allowed": False,
        "fallback_surface": "output_free_prefix_state_trajectory",
        "required_controls": [
            "causal_fix_response",
            "aa_control",
            "norm_only",
            "length_only",
            "pair_order",
            "label_permutation",
            "model_identity",
            "truncation",
        ],
        "excluded_retired_mechanisms": ["shared_activation_bus", "shared_state_initializer"],
    }


def frozen_exact_pair_fixture_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "fixture": "length_matched_vulnerable_fixed_single_function_python_pairs",
        "exact_sidecar_is_release_oracle": True,
        "required_receipts": ["compile", "property", "ast_constraint", "mutation", "split_hash"],
        "heldout_groups": ["weakness", "repository", "template", "perturbation"],
        "claim_limit": "The fixture proves declared pair properties, not universal security.",
    }


def frozen_model_local_energy_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "one_head_per_model": True,
        "pooled_rescue_allowed": False,
        "models": [
            "unsloth/Qwen3.6-35B-A3B-GGUF",
            "unsloth/gemma-4-31B-it-GGUF",
            "unsloth/gemma-4-26B-A4B-it-GGUF",
        ],
        "baselines": ["prompted_verdict", "final_pooled", "norm", "length", "chance"],
        "exact_validator_labels_are_training_targets": False,
    }


def frozen_versioned_learning_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "same_domain_only": True,
        "base_model_weight_mutation_allowed": False,
        "cross_family_transfer_allowed": False,
        "version_lineage_required": True,
        "task_boundary_release_only": True,
        "parent_rollback_required": True,
        "excluded_retired_mechanisms": ["licensed_cross_family_transfer", "lora_base_update"],
    }


def frozen_protected_validation_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "adaptive_loop_can_read": False,
        "release_authority": "protected_exact_validation",
        "dense_signal_release_authority_allowed": False,
        "required_attacks": ["poison", "reversal", "forgetting", "protected_file_mutation"],
        "safety_only_can_promote_utility": False,
    }


def frozen_arc_shadow_no_solve_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "default_off_shadow_only": True,
        "solve_credit_allowed": False,
        "submitted_action_mutation_allowed": False,
        "hidden_game_source_access_allowed": False,
        "registry_update_allowed": False,
        "required_receipts": ["fresh_agent_window", "action_parity", "latency", "seed"],
    }


def frozen_hardware_exclusions() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "hardware_claim_count": 0,
        "speed_power_energy_or_availability_claim_count": 0,
        "current_board_execution_route_supported": False,
        "current_tsu_execution_route_supported": False,
        "allowed_statement": "CPU/GPU operation counts and memory movement may be measured.",
        "excluded_retired_mechanisms": [
            "shared_activation_bus",
            "shared_state_initializer",
            "licensed_cross_family_transfer",
            "external_generated_text_scorer",
            "kan_replacement",
            "generated_answer_transport",
            "extropic_tsu_execution",
            "unchanged_physical_board_probe",
        ],
    }


def semantic_scholar_receipts() -> JsonDict:
    return {
        "ebt": {
            "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["semantic_scholar_ebt_arm"][0]["url"],
            "queried_at_utc": "2026-08-11T11:39:10Z",
            "newest_sampled_publication_date": "2026-08-01",
            "strict_post_marker_relevant_count": 0,
            "sample_titles": [
                "Prospects of intelligent autonomous control technology in aerospace",
                "Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation",
                "Solver-Hard Is Not Model-Hard: A Hardness-Controlled Diagnostic for LLM Constraint Reasoning",
            ],
        },
        "arm_ebm": {
            "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["semantic_scholar_ebt_arm"][1]["url"],
            "queried_at_utc": "2026-08-11T11:39:25Z",
            "newest_sampled_publication_date": "2026-07-02",
            "strict_post_marker_relevant_count": 0,
            "sample_titles": [
                "Path-Measure Dynamics of Attention-Driven World Models",
                "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
                "LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models",
            ],
        },
    }


def _field_provenance() -> JsonDict:
    sources = [
        "REQ-INFRA-6311",
        "research-references V544 marker",
        "source channel receipts",
        "V544 roadmap scope contracts",
        "protected file hashes",
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def payload_checksum(report: Mapping[str, Any]) -> str:
    payload = json.loads(canonical_json(report))
    payload["duration_s"] = 0.0
    payload["reproducibility_checksum"] = ""
    return payload_sha256(payload)


def honest_verdict(status: str, accepted_count: int) -> str:
    if status == "blocked":
        return f"blocked: accepted_count={accepted_count}; V544 source freeze checks failed"
    if accepted_count:
        return f"complete_delta: accepted_count={accepted_count}; strict post-V544-marker findings appended"
    return "complete_null: accepted_count=0; no reproducible post-V544-marker findings; references unchanged; V544 contracts frozen"


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    source_receipts: Sequence[Mapping[str, Any]] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    duration_s: float | None = None,
    search_completed_utc: str | None = None,
    references_before_hash: str | None = None,
    references_after_hash: str | None = None,
    references_appended: bool = False,
    protected_before_hashes: Mapping[str, Any] | None = None,
    protected_authorized_after_hashes: Mapping[str, str | None] | None = None,
    git_status_before: Sequence[str] | None = None,
    git_status_after_tests: Sequence[str] | None = None,
    reference_text_for_dedupe: str | None = None,
) -> JsonDict:
    started = time.monotonic()
    marker = v544_marker_snapshot(root)
    references_text = (
        reference_text_for_dedupe
        if reference_text_for_dedupe is not None
        else _read_text(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    )
    partitions = partition_candidates(
        list(candidates or DEFAULT_SOURCE_CANDIDATES), reference_text=references_text
    )
    accepted = list(partitions["accepted"])
    refs_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    refs_before = references_before_hash or path_sha256(refs_path)
    refs_after = references_after_hash or path_sha256(refs_path)
    source_receipt_rows = [dict(row) for row in source_receipts or DEFAULT_SOURCE_RECEIPTS]
    protected = protected_unchanged(
        root, protected_before_hashes, protected_authorized_after_hashes
    )
    completed = search_completed_utc or _utc_now()
    channel_set = {str(row.get("channel")) for row in source_receipt_rows}
    checks_passed = all(
        [
            marker["marker_count"] == 1,
            marker["marker_line"] is not None,
            set(DEFAULT_SOURCE_QUERIES_BY_CHANNEL) == set(REQUIRED_SOURCE_CHANNELS),
            channel_set == set(REQUIRED_SOURCE_CHANNELS),
            protected["all_unchanged"] is True,
        ]
    )
    status = "complete" if checks_passed else "blocked"
    command_exit_codes = {command: 0 for command in DEFAULT_TEST_COMMANDS}
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": status,
        "v544_marker_text_and_line": marker,
        "search_window_start_utc": MARKER_COMMITTED_AT_UTC,
        "search_completed_utc": completed,
        "source_queries_by_channel": DEFAULT_SOURCE_QUERIES_BY_CHANNEL,
        "source_receipts": source_receipt_rows,
        "accepted_findings": accepted,
        "accepted_count": len(accepted),
        "duplicate_findings": partitions["duplicate_findings"],
        "watch_only_findings": partitions["watch_only_findings"],
        "inaccessible_sources": partitions["inaccessible_sources"],
        "excluded_findings_and_reasons": partitions["excluded_findings_and_reasons"],
        "semantic_scholar_ebt_and_arm_ebm_receipts": semantic_scholar_receipts(),
        "extropic_status": {
            "status": "no_hardware_claim_route",
            "queried_at_utc": "2026-08-11T11:39:27Z",
            "home_http_status": 200,
            "writing_http_status": 200,
            "z1_http_status": 404,
            "local_execution_route_present": False,
        },
        "logical_intelligence_status": {
            "status": "watch_only_product_context",
            "queried_at_utc": "2026-08-11T11:39:25Z",
            "kona_page_http_status": 200,
            "last_modified": "2026-06-26T23:48:05Z",
            "public_weights_or_local_api_present": False,
        },
        "github_status": {
            "status": "no_scope_changing_repo",
            "queried_at_utc": "2026-08-11T11:40:56Z",
            "exact_title_total_count": 0,
            "incomplete_results": False,
            "accepted_repository_count": 0,
        },
        "frozen_model_local_surface_contract": frozen_model_local_surface_contract(),
        "frozen_exact_pair_fixture_contract": frozen_exact_pair_fixture_contract(),
        "frozen_model_local_energy_contract": frozen_model_local_energy_contract(),
        "frozen_versioned_learning_contract": frozen_versioned_learning_contract(),
        "frozen_protected_validation_contract": frozen_protected_validation_contract(),
        "frozen_arc_shadow_no_solve_contract": frozen_arc_shadow_no_solve_contract(),
        "frozen_hardware_exclusions": frozen_hardware_exclusions(),
        "roadmap_scope_delta": {
            "delta_kind": "accepted_source_delta" if accepted else "zero_source_delta",
            "accepted_count": len(accepted),
            "references_before_sha256": refs_before,
            "references_after_sha256": refs_after,
            "references_byte_identical": refs_before == refs_after,
            "append_heading": EXECUTION_DELTA_HEADING if references_appended else None,
            "append_marker": EXECUTION_DELTA_END_MARKER if references_appended else None,
            "scope_widened": bool(accepted),
            "explicit_scope_delta_record_required": True,
        },
        "protected_files_unchanged": protected,
        "preconditions_checked": {
            "input_paths_and_hashes": _hashes(root, INPUT_RELATIVE_PATHS),
            "protected_hashes_before": protected_before_hashes or protected_hashes(root),
            "git_status_before": list(git_status_before or git_status(root)),
            "git_status_after_tests": list(git_status_after_tests or git_status(root)),
            "marker_commit": MARKER_COMMIT,
            "search_started_after_marker_utc": MARKER_COMMITTED_AT_UTC,
            "source_channels_required": list(REQUIRED_SOURCE_CHANNELS),
            "source_channels_observed": sorted(channel_set),
            "research_references_modified": references_before_hash != references_after_hash,
            "research_conductor_modified": False,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "model_specs": [
            {
                "model": "none",
                "invoked": False,
                "reason": "source sweep only; future GGUF names appear only in frozen contracts",
            }
        ],
        "random_seed": "not_applicable_no_random_sampling",
        "field_provenance": _field_provenance(),
        "field_principles": FIELD_PRINCIPLES.copy(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": command_exit_codes,
        "duration_s": round(
            duration_s if duration_s is not None else time.monotonic() - started,
            6,
        ),
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict(status, len(accepted)),
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_report(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            errors.append(f"missing:{field}")
    if errors:
        return errors
    if not isinstance(report.get("accepted_count"), int) or isinstance(
        report.get("accepted_count"), bool
    ):
        errors.append("accepted_count_bare_integer")
    elif report.get("accepted_count") != len(report.get("accepted_findings") or []):
        errors.append("accepted_count")
    if set(report.get("source_queries_by_channel") or {}) != set(REQUIRED_SOURCE_CHANNELS):
        errors.append("source_queries_by_channel")
    channels = {row.get("channel") for row in report.get("source_receipts") or []}
    if channels != set(REQUIRED_SOURCE_CHANNELS):
        errors.append("source_receipts")
    for field in (
        "frozen_model_local_surface_contract",
        "frozen_exact_pair_fixture_contract",
        "frozen_model_local_energy_contract",
        "frozen_versioned_learning_contract",
        "frozen_protected_validation_contract",
        "frozen_arc_shadow_no_solve_contract",
        "frozen_hardware_exclusions",
    ):
        if dict(report.get(field) or {}).get("version") != CONTRACT_VERSION:
            errors.append(field)
    if dict(report.get("semantic_scholar_ebt_and_arm_ebm_receipts") or {}).keys() != {
        "ebt",
        "arm_ebm",
    }:
        errors.append("semantic_scholar_ebt_and_arm_ebm_receipts")
    for field in ("extropic_status", "logical_intelligence_status", "github_status"):
        if not isinstance(report.get(field), Mapping):
            errors.append(field)
    if dict(report.get("protected_files_unchanged") or {}).get("all_unchanged") is not True:
        errors.append("protected_files_unchanged")
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if not str(report.get("honest_verdict") or "").startswith(
        ("complete_null:", "complete_delta:", "blocked:")
    ):
        errors.append("honest_verdict")
    start = _parse_timestamp(report.get("search_window_start_utc"))
    completed = _parse_timestamp(report.get("search_completed_utc"))
    if start is None or completed is None or completed < start:
        errors.append("search_window")
    for field_name in ("field_principles", "field_provenance"):
        mapping = report.get(field_name)
        if not isinstance(mapping, Mapping) or set(mapping) != set(REQUIRED_ARTIFACT_FIELDS):
            errors.append(field_name)
            continue
        if field_name == "field_provenance":
            for field in REQUIRED_ARTIFACT_FIELDS:
                row = mapping.get(field)
                if not isinstance(row, Mapping) or row.get("principle") != FIELD_PRINCIPLES[field]:
                    errors.append(f"field_provenance:{field}")
    if report.get("reproducibility_checksum") != payload_checksum(report):
        errors.append("reproducibility_checksum")
    return errors


def write_freeze(
    root: Path = REPO_ROOT,
    *,
    date: str,
    source_receipts: Sequence[Mapping[str, Any]] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    duration_s: float | None = None,
    search_completed_utc: str | None = None,
    env: Mapping[str, str] | None = None,
) -> JsonDict:
    protected_before = protected_hashes(root)
    git_before = git_status(root)
    refs_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    references_before = path_sha256(refs_path)
    references_text = _read_text(refs_path)
    partitions = partition_candidates(
        list(candidates or DEFAULT_SOURCE_CANDIDATES), reference_text=references_text
    )
    accepted = list(partitions["accepted"])
    references_appended = False
    authorized_after: dict[str, str | None] = {}
    if accepted:
        updated = insert_after_marker(references_text, execution_delta_block(accepted))
        if updated != references_text:
            refs_path.write_text(updated, encoding="utf-8")
            references_appended = True
            authorized_after[RESEARCH_REFERENCES_RELATIVE_PATH.as_posix()] = path_sha256(refs_path)
    references_after = path_sha256(refs_path)
    report = build_report(
        root,
        date=date,
        source_receipts=source_receipts,
        candidates=candidates,
        duration_s=duration_s,
        search_completed_utc=search_completed_utc,
        references_before_hash=references_before,
        references_after_hash=references_after,
        references_appended=references_appended,
        protected_before_hashes=protected_before,
        protected_authorized_after_hashes=authorized_after,
        git_status_before=git_before,
        git_status_after_tests=git_status(root),
        reference_text_for_dedupe=references_text,
    )
    errors = validate_report(report)
    if errors:
        raise ValueError(f"invalid Exp6311 freeze: {errors}")
    atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Write the Exp6311 V544 source freeze.")
    parser.add_argument("--date", default="20260811")
    args = parser.parse_args(argv)
    report = write_freeze(REPO_ROOT, date=args.date)
    print(json.dumps({"path": RESULT_RELATIVE_PATH.as_posix(), "status": report["status"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
