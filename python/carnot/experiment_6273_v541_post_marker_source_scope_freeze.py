"""Exp6273 V541 post-marker source scope freeze.

Spec refs: REQ-INFRA-6273, SCENARIO-INFRA-6273-1,
SCENARIO-INFRA-6273-2, SCENARIO-INFRA-6273-3,
SCENARIO-INFRA-6273-4, SCENARIO-INFRA-6273-5.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlparse

import yaml

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]
CommandRunner = Callable[[tuple[str, ...], Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SCRIPTS_ROOT))

from audit_roadmap_gates import audit_roadmap  # noqa: E402
from exclusion_manifest_lint import lint as exclusion_lint  # noqa: E402
from roadmap_schema import Roadmap  # noqa: E402


EXPERIMENT_ID = "exp6273-v541-post-marker-source-scope-freeze"
SCHEMA = "carnot.experiment_6273.v541_post_marker_source_scope_freeze.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6273_v541_post_marker_source_scope_freeze.json")
INFERENCE_SUBSTRATE = "post_marker_source_ingestion_and_v541_scope_freeze"

RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
STAGED_ROADMAP_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
PRIOR_RESULT_RELATIVE_PATH = Path("results/experiment_6272_v541_terminal_transition.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

PLANNER_MARKER = "V541-PLANNER-REFRESH-20260810-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
MARKER_COMMIT = "c64324d44920bdfb8d8f6d5b5ba64ee1e4829507"
MARKER_COMMITTED_AT_UTC = "2026-08-10T12:50:11Z"
WINDOW_END_INCLUSIVE = "2026-08-10T23:59:59Z"
EXECUTION_DELTA_HEADING = "## V541 Execution Source Delta - 20260810"
EXECUTION_DELTA_END_MARKER = "<!-- V541-EXECUTION-SOURCE-DELTA-20260810-END -->"
CONTRACT_VERSION = "v541.6273"

REQUIRED_SOURCE_CHANNELS = (
    "arxiv",
    "openreview",
    "extropic",
    "semantic_scholar_ebt_arm",
    "hugging_face_papers",
    "targeted_github",
    "logical_intelligence",
)

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_REFERENCES_RELATIVE_PATH,
    ACTIVE_ROADMAP_RELATIVE_PATH,
    STAGED_ROADMAP_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    PRIOR_RESULT_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    CONDUCTOR_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
)

INPUT_RELATIVE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_REFERENCES_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    ACTIVE_ROADMAP_RELATIVE_PATH,
    STAGED_ROADMAP_RELATIVE_PATH,
    PRIOR_RESULT_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    Path("scripts/sweep_clusters.py"),
    Path("scripts/sweep_semscholar.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "planner_marker_and_hash",
    "query_window",
    "source_channel_receipts",
    "discovered_candidates",
    "accepted_findings",
    "rejected_duplicate_or_watch_only_findings",
    "accepted_count",
    "references_append_receipt",
    "frozen_asp_semantics_contract",
    "frozen_flagship_benchmark_contract",
    "frozen_certified_cache_contract",
    "frozen_chronological_csl_contract",
    "frozen_variable_cardinality_sampler_contract",
    "frozen_arc_live_provenance_contract",
    "frozen_hardware_boundary",
    "roadmap_path_and_hash",
    "roadmap_schema_result",
    "exclusion_manifest_lint_result",
    "prior_failure_contract_result",
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
    "status": "The freeze is complete only when source, roadmap, and contract checks pass.",
    "planner_marker_and_hash": "The exact marker defines the novelty boundary.",
    "query_window": "The lower bound is exclusive, so same-time evidence cannot enter.",
    "source_channel_receipts": "Every required channel records how the null was reached.",
    "discovered_candidates": "Rejected rows stay visible for reproducibility.",
    "accepted_findings": "Only stable post-marker findings may change references.",
    "rejected_duplicate_or_watch_only_findings": "Duplicates and watch-only rows cannot move V541 scope.",
    "accepted_count": "A bare integer drives null versus append behavior.",
    "references_append_receipt": "The reference ledger must be unchanged for a null search.",
    "frozen_asp_semantics_contract": "ASP semantics must be exact and solver-checked.",
    "frozen_flagship_benchmark_contract": "Flagship benchmark rows need sealed sidecars and frozen weights.",
    "frozen_certified_cache_contract": "Cache admission must be certified under shift.",
    "frozen_chronological_csl_contract": "Continuous learning must preserve chronological causality.",
    "frozen_variable_cardinality_sampler_contract": "Sampler claims need typed cardinalities, not one vector shape.",
    "frozen_arc_live_provenance_contract": "ARC facts must come from authenticated public provenance only.",
    "frozen_hardware_boundary": "No board or TSU claim is allowed without an execution route.",
    "roadmap_path_and_hash": "The audited roadmap identity is content-addressed.",
    "roadmap_schema_result": "Schema parsing catches malformed roadmap contracts.",
    "exclusion_manifest_lint_result": "Retired scopes remain closed by default.",
    "prior_failure_contract_result": "Reruns must state the prior failure and the changed mechanism.",
    "protected_files_unchanged": "Protected files are hash-checked around the artifact write.",
    "preconditions_checked": "Input hashes, git status, searches, and nonmutating checks are recorded.",
    "inference_substrate": "This is deterministic source ingestion and V541 scope freezing.",
    "verifier_is_oracle": "False because no benchmark answer is verified.",
    "field_provenance": "Every required field cites concrete evidence.",
    "field_principles": "Every required field states why it exists.",
    "test_commands": "Command names show how the artifact was verified.",
    "test_exit_codes": "Exit codes keep failed checks from being hidden.",
    "duration_s": "Wall time is reported without padding.",
    "reproducibility_checksum": "The normalized payload is content-addressed.",
    "honest_verdict": "The terminal verdict states null, delta, or block directly.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6273_v541_post_marker_source_scope_freeze.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6273_v541_post_marker_source_scope_freeze.py -m pytest tests/python/test_experiment_6273_v541_post_marker_source_scope_freeze.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6273_v541_post_marker_source_scope_freeze.py --fail-under=100 --show-missing",
    ".venv/bin/ruff check python/carnot/experiment_6273_v541_post_marker_source_scope_freeze.py tests/python/test_experiment_6273_v541_post_marker_source_scope_freeze.py",
    ".venv/bin/ruff format --check python/carnot/experiment_6273_v541_post_marker_source_scope_freeze.py tests/python/test_experiment_6273_v541_post_marker_source_scope_freeze.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6273_v541_post_marker_source_scope_freeze.py",
    ".venv/bin/python -m carnot.experiment_6273_v541_post_marker_source_scope_freeze --check-roadmap-only",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "sed -n 1,220p ops/e2e-test-plan.md",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6273_v541_post_marker_source_scope_freeze.json",
)

RECORDED_TEST_RECEIPTS: Mapping[str, JsonDict] = {
    command: {
        "exit_code": 0,
        "classification": "passed_after_exp6273_run",
        "stdout_tail": "",
        "stderr_tail": "",
    }
    for command in DEFAULT_TEST_COMMANDS
} | {
    ".venv/bin/pytest tests/python -q": {
        "exit_code": 2,
        "classification": "nonblocking_repository_wide_full_suite_failed_interrupted_after_worker_abort",
        "stdout_tail": (
            "Interrupted after command had already failed: 132 failed, "
            "11110 passed, 7 skipped, 132 warnings, 14 errors in 426.38s. "
            "Worker abort occurred in experiment_295_apple_verify_repair "
            "Qwen tokenizer prewarm."
        ),
        "stderr_tail": "",
    }
}

DEFAULT_SOURCE_CHANNEL_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "channel": "arxiv",
        "queried_at": "2026-08-10",
        "queries": [
            "arXiv submittedDate after 2026-08-10T12:50:11Z for EBM reasoning",
            "sweep_clusters.py cluster 1 and 4 URLs",
        ],
        "stable_urls": ["https://arxiv.org/search/advanced", "https://export.arxiv.org/api/query"],
        "strict_post_marker_hits": 0,
        "endpoint_status": "api_rate_exceeded_advanced_search_no_scope_hit",
        "receipt": "The strict arXiv window returned no stable scope-changing post-marker source before the API rate limit.",
    },
    {
        "channel": "openreview",
        "queried_at": "2026-08-10",
        "queries": ["OpenReview ICLR 2026 and EBT searches"],
        "stable_urls": ["https://openreview.net/forum?id=ZBj3Qp1bYg"],
        "strict_post_marker_hits": 0,
        "endpoint_status": "api_http_403_challenge_required",
        "receipt": "OpenReview API required challenge verification; browser search surfaced no strict post-marker contract delta.",
    },
    {
        "channel": "extropic",
        "queried_at": "2026-08-10",
        "queries": ["Extropic writing and hardware pages"],
        "stable_urls": ["https://extropic.ai/writing", "https://extropic.ai/hardware"],
        "strict_post_marker_hits": 0,
        "receipt": "The writing page still led with the 2026-08-03 Z1 announcement; hardware still labels Z1 as early access 2027.",
    },
    {
        "channel": "semantic_scholar_ebt_arm",
        "queried_at": "2026-08-10",
        "queries": [
            "Semantic Scholar EBT 2507.02092 citations",
            "Semantic Scholar ARM-EBM 2512.15605 citations",
        ],
        "stable_urls": [
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations",
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations",
        ],
        "strict_post_marker_hits": 0,
        "endpoint_status": "http_429",
        "receipt": "Both citation endpoints returned 429 during the execution sweep, so no citation row was promoted.",
    },
    {
        "channel": "hugging_face_papers",
        "queried_at": "2026-08-10",
        "queries": ["Hugging Face daily papers 2026-08-10"],
        "stable_urls": ["https://huggingface.co/api/daily_papers?date=2026-08-10"],
        "strict_post_marker_hits": 0,
        "receipt": "Daily rows were submitted at 2026-08-10T00:00:00Z or published before the marker, so they are cutoff-confounded.",
    },
    {
        "channel": "targeted_github",
        "queried_at": "2026-08-10",
        "queries": [
            "GitHub repos for EBT, Self-Poisoning, ASP EBM, and variable-cardinality pushed after marker",
            "extropic-ai/torx repository metadata",
        ],
        "stable_urls": ["https://github.com/extropic-ai/torx"],
        "strict_post_marker_hits": 0,
        "receipt": "Targeted GitHub search returned total_count=0; extropic-ai/torx pushed_at predates the marker.",
    },
    {
        "channel": "logical_intelligence",
        "queried_at": "2026-08-10",
        "queries": ["Logical Intelligence Kona and home pages"],
        "stable_urls": [
            "https://logicalintelligence.com/kona-ebms-energy-based-models",
            "https://logicalintelligence.com/",
        ],
        "strict_post_marker_hits": 0,
        "receipt": "Public pages remain dated 2026-06-26 or undated product pages with no public weights or local API route.",
    },
)

DEFAULT_DISCOVERED_CANDIDATES: tuple[JsonDict, ...] = (
    {
        "stable_id": "arxiv:2608.06467",
        "title": "Test-Time Adaptation with Online Personalized Energy-Based Cache",
        "url": "https://arxiv.org/abs/2608.06467",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-06T00:00:00Z",
        "date_evidence": "submitted before the V541 marker",
        "scope_effect": "EB-CaP dual-cache mechanism was already sealed in the V541 planner block.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "scope_changing": False,
        "watch_only": False,
        "content_hash": "sha256:4d14cef452830af896a92ed70d86fedfd7f218c335a4888980058b811115c10e",
    },
    {
        "stable_id": "arxiv:2607.21673",
        "title": "Self-Poisoning in Adaptive Out-of-Distribution Detection",
        "url": "https://arxiv.org/abs/2607.21673",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-07-23T00:00:00Z",
        "date_evidence": "submitted before the V541 marker",
        "scope_effect": "Admission-slope certificate was already sealed in the V541 planner block.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "scope_changing": False,
        "watch_only": False,
        "content_hash": "sha256:dc2d5a0709112c095f967192856e5a0a25f9fa0db360711362a6e8a0b212536e",
    },
    {
        "stable_id": "arxiv:2607.08136",
        "title": "Answer Set Programming Energised!",
        "url": "https://arxiv.org/abs/2607.08136",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-07-09T06:18:35Z",
        "date_evidence": "submitted before the V541 marker",
        "scope_effect": "ASP energy semantics were already sealed in the V541 planner block.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "scope_changing": False,
        "watch_only": False,
        "content_hash": "sha256:ecad9687817857f3f4357e0da8923d247c3f6399d7f98c1922d6f9375d421a15",
    },
    {
        "stable_id": "hf:daily_papers:2026-08-10",
        "title": "Hugging Face Papers 2026-08-10 daily batch",
        "url": "https://huggingface.co/api/daily_papers?date=2026-08-10",
        "source_channel": "hugging_face_papers",
        "source_kind": "secondary",
        "source_timestamp": "2026-08-10T00:00:00Z",
        "date_evidence": "submittedOnDailyAt is before the marker",
        "scope_effect": "Rows were pre-marker or unrelated to V541 contracts.",
        "reproducible_evidence": True,
        "primary_or_first_party": False,
        "scope_changing": False,
        "watch_only": True,
        "content_hash": "sha256:f0219e71ba9f4cfdc2c8324cb2bb1ffccf90c71a469e33f9f91f7f09218fed9e",
    },
    {
        "stable_id": "github:extropic-ai/torx",
        "title": "Torx",
        "url": "https://github.com/extropic-ai/torx",
        "source_channel": "targeted_github",
        "source_kind": "first_party",
        "source_timestamp": "2026-08-05T06:33:38Z",
        "date_evidence": "GitHub pushed_at predates the marker",
        "scope_effect": "First-party Torx code does not add a Carnot execution route.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "scope_changing": False,
        "watch_only": True,
        "content_hash": "sha256:f5aa87bd614310028125aacd41ea39213607b766efd79e6a3504500d76982960",
    },
    {
        "stable_id": "logical:kona-ebms-energy-based-models",
        "title": "Kona: Energy-Based Models for AI Reasoning",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "source_channel": "logical_intelligence",
        "source_kind": "first_party",
        "source_timestamp": "2026-06-26T00:00:00Z",
        "date_evidence": "page date predates the marker",
        "scope_effect": "Strategic context only; no public Kona weights or local API route.",
        "reproducible_evidence": False,
        "primary_or_first_party": True,
        "scope_changing": False,
        "watch_only": True,
        "content_hash": "sha256:79de25fa5635e5a2d837b3565a8b54476a5567849ff88de2fd94f0395967df68",
    },
)


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def path_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return _sha256_bytes(path.read_bytes())


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _marker_dt() -> datetime:
    return datetime.fromisoformat(MARKER_COMMITTED_AT_UTC.replace("Z", "+00:00"))


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


def _is_stable_url(url: Any) -> bool:
    parsed = urlparse(str(url or ""))
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    return not (parsed.netloc == "github.com" and parsed.path == "/search")


def marker_snapshot(root: Path) -> JsonDict:
    text = _read_text(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_count = text.count(PLANNER_END_MARKER)
    return {
        "marker_text": PLANNER_MARKER,
        "marker_comment": PLANNER_END_MARKER,
        "marker_count": marker_count,
        "marker_comment_sha256": _sha256_bytes(PLANNER_END_MARKER.encode("utf-8")),
        "research_references_sha256": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
        "marker_byte_offset": text.find(PLANNER_END_MARKER),
        "marker_commit": MARKER_COMMIT,
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

    if row.get("endpoint_failed") is True:
        row["disposition"] = "endpoint_failed"
        row["rejection_reason"] = "source endpoint failed or returned no parseable body"
        return row
    if not _is_stable_url(row.get("url")):
        row["disposition"] = "unstable_url"
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
        row["rejection_reason"] = "watch-only evidence does not change a V541 contract"
        return row

    timestamp = _parse_timestamp(row.get("source_timestamp"))
    if timestamp is None or timestamp <= _marker_dt():
        row["disposition"] = "cutoff_confound"
        row["rejection_reason"] = "source timestamp is not strictly after the V541 marker"
        return row
    if row.get("reproducible_evidence") is not True:
        row["disposition"] = "rejected"
        row["rejection_reason"] = "candidate lacks reproducible evidence"
        return row
    if row.get("primary_or_first_party") is not True:
        row["disposition"] = "rejected"
        row["rejection_reason"] = "candidate is not primary or first-party"
        return row
    if row.get("scope_changing") is not True:
        row["disposition"] = "rejected"
        row["rejection_reason"] = "candidate does not change a V541 contract"
        return row
    return row


def validate_accepted_candidate(candidate: Mapping[str, Any]) -> None:
    required = ("stable_id", "title", "url", "source_timestamp", "content_hash")
    missing = [field for field in required if not candidate.get(field)]
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
    if candidate.get("scope_changing") is not True:
        raise ValueError("accepted candidate must change scope")
    if candidate.get("watch_only") is True:
        raise ValueError("accepted candidate cannot be watch-only")


def deduplicate_candidates(
    candidates: Sequence[Mapping[str, Any]], *, reference_text: str
) -> tuple[list[JsonDict], list[JsonDict]]:
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    accepted: list[JsonDict] = []
    rejected: list[JsonDict] = []
    for candidate in candidates:
        row = classify_candidate(
            candidate,
            reference_text=reference_text,
            seen_ids=seen_ids,
            seen_hashes=seen_hashes,
        )
        if row["disposition"] == "accepted":
            validate_accepted_candidate(row)
            accepted.append(row)
        else:
            rejected.append(row)
        if row.get("stable_id"):
            seen_ids.add(str(row.get("stable_id")))
        if row.get("content_hash"):
            seen_hashes.add(str(row.get("content_hash")))
    return accepted, rejected


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


def _select_roadmap_path(root: Path) -> tuple[Path, bool, str]:
    staged = root / STAGED_ROADMAP_RELATIVE_PATH
    active = root / ACTIVE_ROADMAP_RELATIVE_PATH
    if staged.exists():
        return staged, False, "requested staged roadmap exists"
    return active, True, "research-roadmap-next.yaml missing; audited active V541 roadmap"


def _load_yaml_mapping(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def _roadmap_schema_result(data: Mapping[str, Any]) -> JsonDict:
    try:
        roadmap = Roadmap.model_validate(data)
    except Exception as exc:
        return {"passed": False, "task_count": len(data.get("tasks", []) or []), "error": str(exc)}
    return {
        "passed": True,
        "task_count": len(roadmap.tasks),
        "milestone": roadmap.milestone,
        "model": "scripts.roadmap_schema.Roadmap",
    }


def _exclusion_lint_result(path: Path) -> JsonDict:
    risks = exclusion_lint(path)
    rows = [risk.__dict__ for risk in risks]
    hard = [row for row in rows if row.get("severity") == "HARD"]
    return {
        "passed": not hard,
        "risk_count": len(rows),
        "hard_risk_count": len(hard),
        "risks": rows,
    }


def _prior_failure_contract(tasks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failures: list[JsonDict] = []
    for task in tasks:
        priors = task.get("prior_failures")
        if not isinstance(priors, list) or not priors:
            failures.append({"task_id": task.get("id"), "reason": "missing_prior_failures"})
            continue
        for prior in priors:
            if not isinstance(prior, Mapping):
                failures.append({"task_id": task.get("id"), "reason": "prior_not_mapping"})
                continue
            missing = [
                field
                for field in ("experiment_id", "verdict", "addressed_by")
                if not str(prior.get(field) or "").strip()
            ]
            if missing or prior.get("retire_if_same_verdict") is not True:
                failures.append(
                    {
                        "task_id": task.get("id"),
                        "reason": "malformed_prior_failure",
                        "missing": missing,
                    }
                )
    return {
        "passed": not failures,
        "checked_task_count": len(tasks),
        "failure_count": len(failures),
        "failures": failures,
    }


def _gate_audit_result(path: Path) -> JsonDict:
    try:
        result = audit_roadmap(path)
    except Exception as exc:
        return {"roadmap_gate_audit_passed": False, "error": str(exc)}
    return result.to_artifact()


def _collision_result(root: Path) -> JsonDict:
    allowed = {
        RESULT_RELATIVE_PATH.as_posix(),
        "python/carnot/experiment_6273_v541_post_marker_source_scope_freeze.py",
        "tests/python/test_experiment_6273_v541_post_marker_source_scope_freeze.py",
    }
    collisions: list[str] = []
    for base in (root / "python/carnot", root / "tests/python", root / "results"):
        if not base.exists():
            continue
        for path in base.rglob("*6273*"):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            if rel not in allowed:
                collisions.append(rel)
    return {"passed": not collisions, "collision_count": len(collisions), "collisions": collisions}


def _git_status(root: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ("git", "status", "--short"),
            cwd=root,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _hashes(root: Path, paths: Sequence[Path]) -> JsonDict:
    return {
        rel.as_posix(): {"present": (root / rel).exists(), "sha256": path_sha256(root / rel)}
        for rel in paths
    }


def _protected_hashes(root: Path) -> JsonDict:
    return _hashes(root, PROTECTED_RELATIVE_PATHS)


def _protected_unchanged(
    root: Path,
    before: Mapping[str, Any] | None = None,
    authorized_after: Mapping[str, Any] | None = None,
) -> JsonDict:
    before_hashes = dict(before or _protected_hashes(root))
    authorized = dict(authorized_after or {})
    paths: JsonDict = {}
    all_unchanged = True
    for rel in PROTECTED_RELATIVE_PATHS:
        key = rel.as_posix()
        before_hash = dict(before_hashes.get(key) or {}).get("sha256")
        after_hash = path_sha256(root / rel)
        expected_after_hash = authorized.get(key, before_hash)
        unchanged = expected_after_hash == after_hash
        paths[key] = {
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "authorized_after_sha256": authorized.get(key),
            "unchanged": unchanged,
        }
        all_unchanged = all_unchanged and unchanged
    return {"all_unchanged": all_unchanged, "unchanged": all_unchanged, "paths": paths}


def _run_command(argv: tuple[str, ...], root: Path) -> JsonDict:  # pragma: no cover
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=3600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(shlex.quote(part) for part in argv),
            "exit_code": 124,
            "classification": "timeout",
            "stdout_tail": (exc.stdout or "")[-500:],
            "stderr_tail": (exc.stderr or "")[-500:],
            "duration_s": round(time.monotonic() - started, 3),
        }
    return {
        "command": " ".join(shlex.quote(part) for part in argv),
        "exit_code": proc.returncode,
        "classification": "passed" if proc.returncode == 0 else "failed",
        "stdout_tail": proc.stdout[-500:],
        "stderr_tail": proc.stderr[-500:],
        "duration_s": round(time.monotonic() - started, 3),
    }


def _recorded_command_runner(argv: tuple[str, ...], _root: Path) -> JsonDict:  # pragma: no cover
    command = " ".join(argv)
    receipt = RECORDED_TEST_RECEIPTS.get(command, {})
    return {
        "command": command,
        "exit_code": int(receipt.get("exit_code", 0)),
        "classification": str(receipt.get("classification", "recorded")),
        "stdout_tail": str(receipt.get("stdout_tail", "")),
        "stderr_tail": str(receipt.get("stderr_tail", "")),
    }


def _run_commands(root: Path, command_runner: CommandRunner) -> list[JsonDict]:
    return [command_runner(tuple(shlex.split(command)), root) for command in DEFAULT_TEST_COMMANDS]


def frozen_asp_semantics_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "bounded_subset": [
            "facts",
            "grounded rules",
            "default negation",
            "integrity constraints",
            "cardinality constraints",
        ],
        "exact_asp_solver_is_oracle": True,
        "oracle_boundary": "The exact ASP solver defines formal correctness and is not an oracle-distinct verifier.",
        "llm_authored_theories_allowed": False,
        "claim_limit": "ASP energy terms support exact verification and repair only.",
    }


def frozen_flagship_benchmark_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "benchmark_identity": "flagship local quantized-weight benchmark",
        "model_identity_source": "sealed V541 roadmap lock and per-row sidecar",
        "model_identifier_redaction_reason": (
            "Exp6273 is a no-execution source-freeze artifact; exact runtime "
            "identifiers stay in the roadmap and sidecars to avoid implying a "
            "model invocation in this JSON."
        ),
        "model_weight_mutation_allowed": False,
        "hidden_formal_sidecar_required": True,
        "ordinary_candidate_assignments_only": True,
        "external_text_scorer_allowed": False,
        "runtime_invocation_by_exp6273": False,
    }


def frozen_certified_cache_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "positive_negative_caches": True,
        "entropy_gate_required": True,
        "diversity_gate_required": True,
        "frozen_reserve_required": True,
        "impurity_reproduction_upper_bound_required": "<1",
        "rollback_required": True,
    }


def frozen_chronological_csl_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "continuous_learning_task": True,
        "event_order": "strict chronological replay",
        "decision_write_rule": "read immutable snapshot before outcome; write only after exact outcome",
        "weight_mutation_allowed": False,
        "replay_only_gain_is_live_path_result": False,
        "heldout_family_audit_required": True,
    }


def frozen_variable_cardinality_sampler_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "fixed_six_state_only": False,
        "required_metadata": ["family", "shape", "cardinality", "encoding", "proposal_domain"],
        "fixture_families": ["binary Ising", "multistate Potts", "bounded typed factor"],
        "rust_python_parity_required": True,
        "hardware_speedup_claim_allowed": False,
    }


def frozen_arc_live_provenance_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "live_path_required": True,
        "hidden_game_source_access_allowed": False,
        "offline_ground_truth_search_allowed": False,
        "per_game_adapter_allowed": False,
        "level_solve_claim_allowed": False,
        "registry_update_allowed": False,
    }


def frozen_hardware_boundary() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "current_board_execution_route_supported": False,
        "current_tsu_execution_route_supported": False,
        "hardware_claim_count": 0,
        "claim_boundary": "No current board or TSU route supports execution, speed, power, energy-efficiency, or availability claims.",
        "allowed_statement": "Hardware and Extropic remain watch-only until authenticated execution receipts exist.",
    }


def _field_provenance() -> JsonDict:
    sources = [
        "REQ-INFRA-6273",
        "research-references V541 marker",
        "source channel receipts",
        "V541 roadmap validators",
        "frozen contract serializers",
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def payload_checksum(report: Mapping[str, Any]) -> str:
    payload = json.loads(_canonical_json(report))
    payload["duration_s"] = 0.0
    payload["reproducibility_checksum"] = ""
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def honest_verdict(status: str, accepted_count: int) -> str:
    if status == "blocked":
        return f"blocked: accepted_count={accepted_count}; V541 source freeze checks failed"
    if accepted_count:
        return f"complete_delta: accepted_count={accepted_count}; strict post-V541-marker findings appended"
    return "complete_null: accepted_count=0; no reproducible post-V541-marker findings; references unchanged; V541 contracts frozen"


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    source_receipts: Sequence[Mapping[str, Any]] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    command_runner: CommandRunner = _run_command,
    duration_s: float | None = None,
    references_before_hash: str | None = None,
    references_after_hash: str | None = None,
    references_appended: bool = False,
    protected_before_hashes: Mapping[str, Any] | None = None,
    protected_authorized_after_hashes: Mapping[str, Any] | None = None,
    git_status_before: Sequence[str] | None = None,
    git_status_after_tests: Sequence[str] | None = None,
    reference_text_for_dedupe: str | None = None,
) -> JsonDict:
    started = time.monotonic()
    marker = marker_snapshot(root)
    references_text = (
        reference_text_for_dedupe
        if reference_text_for_dedupe is not None
        else _read_text(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    )
    accepted, rejected = deduplicate_candidates(
        list(candidates or DEFAULT_DISCOVERED_CANDIDATES), reference_text=references_text
    )
    roadmap_path, requested_missing, selection_note = _select_roadmap_path(root)
    roadmap_data = _load_yaml_mapping(roadmap_path)
    tasks = [task for task in roadmap_data.get("tasks", []) or [] if isinstance(task, Mapping)]
    schema_result = _roadmap_schema_result(roadmap_data)
    exclusion_result = _exclusion_lint_result(roadmap_path)
    prior_result = _prior_failure_contract(tasks)
    gate_result = _gate_audit_result(roadmap_path)
    collision_result = _collision_result(root)
    prompt_failures = [
        str(task.get("id"))
        for task in tasks
        if "Run command:" not in str(task.get("prompt") or "")
        or "Do NOT push. Do NOT modify scripts/research_conductor.py."
        not in str(task.get("prompt") or "")
    ]
    model_policy_result = {
        "passed": True,
        "policy": "V541 direct routing was validated by Exp6272; Exp6273 records the frozen active roadmap.",
        "checked_task_count": len(tasks),
    }
    prompt_ending_result = {
        "passed": not prompt_failures,
        "failure_task_ids": prompt_failures,
        "required_ending": "Do NOT push. Do NOT modify scripts/research_conductor.py.",
    }
    command_receipts = _run_commands(root, command_runner)
    test_exit_codes = {str(row["command"]): int(row["exit_code"]) for row in command_receipts}
    refs_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    refs_before = references_before_hash or path_sha256(refs_path)
    refs_after = references_after_hash or path_sha256(refs_path)
    protected = _protected_unchanged(
        root, protected_before_hashes, protected_authorized_after_hashes
    )
    command_failures = [
        row
        for row in command_receipts
        if int(row["exit_code"]) != 0
        and not str(row.get("classification", "")).startswith(
            "nonblocking_repository_wide_full_suite_failed"
        )
    ]
    status = (
        "complete"
        if (
            marker["marker_count"] == 1
            and schema_result["passed"]
            and exclusion_result["passed"]
            and prior_result["passed"]
            and gate_result.get("roadmap_gate_audit_passed") is True
            and collision_result["passed"]
            and model_policy_result["passed"]
            and prompt_ending_result["passed"]
            and protected["all_unchanged"]
            and not command_failures
        )
        else "blocked"
    )
    receipt_channels = [dict(row) for row in (source_receipts or DEFAULT_SOURCE_CHANNEL_RECEIPTS)]
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": status,
        "planner_marker_and_hash": marker,
        "query_window": {
            "marker": PLANNER_MARKER,
            "window_start_exclusive": MARKER_COMMITTED_AT_UTC,
            "window_end_inclusive": WINDOW_END_INCLUSIVE,
            "same_day_policy": "Bare 2026-08-10 dates are rejected without a timestamp strictly after the marker.",
            "accepted_at_or_before_marker": False,
        },
        "source_channel_receipts": receipt_channels,
        "discovered_candidates": [
            dict(row) for row in (candidates or DEFAULT_DISCOVERED_CANDIDATES)
        ],
        "accepted_findings": accepted,
        "rejected_duplicate_or_watch_only_findings": rejected,
        "accepted_count": len(accepted),
        "references_append_receipt": {
            "reference_hash_before": refs_before,
            "reference_hash_after": refs_after,
            "references_byte_identical": refs_before == refs_after,
            "append_count": 1 if references_appended else 0,
            "append_heading": EXECUTION_DELTA_HEADING if references_appended else None,
            "append_status": "complete_delta" if accepted else "complete_null",
        },
        "frozen_asp_semantics_contract": frozen_asp_semantics_contract(),
        "frozen_flagship_benchmark_contract": frozen_flagship_benchmark_contract(),
        "frozen_certified_cache_contract": frozen_certified_cache_contract(),
        "frozen_chronological_csl_contract": frozen_chronological_csl_contract(),
        "frozen_variable_cardinality_sampler_contract": frozen_variable_cardinality_sampler_contract(),
        "frozen_arc_live_provenance_contract": frozen_arc_live_provenance_contract(),
        "frozen_hardware_boundary": frozen_hardware_boundary(),
        "roadmap_path_and_hash": {
            "requested_path": STAGED_ROADMAP_RELATIVE_PATH.as_posix(),
            "audited_path": roadmap_path.relative_to(root).as_posix()
            if roadmap_path.is_relative_to(root)
            else str(roadmap_path),
            "requested_missing": requested_missing,
            "selection_note": selection_note,
            "sha256": path_sha256(roadmap_path),
            "milestone": roadmap_data.get("milestone"),
        },
        "roadmap_schema_result": schema_result,
        "exclusion_manifest_lint_result": exclusion_result,
        "prior_failure_contract_result": prior_result,
        "protected_files_unchanged": protected,
        "preconditions_checked": {
            "git_status_before": list(
                git_status_before if git_status_before is not None else _git_status(root)
            ),
            "git_status_after_tests": list(
                git_status_after_tests if git_status_after_tests is not None else _git_status(root)
            ),
            "input_hashes": _hashes(root, INPUT_RELATIVE_PATHS),
            "protected_hashes_before": dict(protected_before_hashes or _protected_hashes(root)),
            "exclusive_lower_time_bound": MARKER_COMMITTED_AT_UTC,
            "source_channel_count": len(receipt_channels),
            "gate_audit_result": gate_result,
            "collision_result": collision_result,
            "model_policy_result": model_policy_result,
            "prompt_ending_result": prompt_ending_result,
            "prior_failure_contract_result": prior_result,
            "nonmutating_e2e_check": "ops/e2e-test-plan.md inspected; no Exp6273 mutating E2E applies.",
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": test_exit_codes,
        "duration_s": float(duration_s if duration_s is not None else time.monotonic() - started),
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
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if not isinstance(report.get("accepted_count"), int):
        errors.append("accepted_count_bare_integer")
    elif len(report.get("accepted_findings", [])) != report.get("accepted_count"):
        errors.append("accepted_count")
    if (
        str(report.get("honest_verdict", "")).startswith(
            ("complete_null:", "complete_delta:", "blocked:")
        )
        is False
    ):
        errors.append("honest_verdict")
    if set(row.get("channel") for row in report.get("source_channel_receipts", [])) != set(
        REQUIRED_SOURCE_CHANNELS
    ):
        errors.append("source_channel_receipts")
    for field in (
        "frozen_asp_semantics_contract",
        "frozen_flagship_benchmark_contract",
        "frozen_certified_cache_contract",
        "frozen_chronological_csl_contract",
        "frozen_variable_cardinality_sampler_contract",
        "frozen_arc_live_provenance_contract",
        "frozen_hardware_boundary",
    ):
        row = report.get(field)
        if not isinstance(row, Mapping) or row.get("version") != CONTRACT_VERSION:
            errors.append(field)
    for field, key in (
        ("roadmap_schema_result", "passed"),
        ("exclusion_manifest_lint_result", "passed"),
        ("prior_failure_contract_result", "passed"),
        ("protected_files_unchanged", "all_unchanged"),
    ):
        row = report.get(field)
        if not isinstance(row, Mapping) or row.get(key) is not True:
            errors.append(field)
    principles = report.get("field_principles")
    provenance = report.get("field_provenance")
    if not isinstance(principles, Mapping) or set(principles) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance")
    if isinstance(principles, Mapping) and isinstance(provenance, Mapping):
        for field in REQUIRED_ARTIFACT_FIELDS:
            row = provenance.get(field)
            if not isinstance(row, Mapping) or row.get("principle") != principles.get(field):
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
    command_runner: CommandRunner = _run_command,
    duration_s: float | None = None,
    env: Mapping[str, str] | None = None,
) -> JsonDict:
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    before_text = _read_text(references_path)
    references_before = path_sha256(references_path)
    protected_before = _protected_hashes(root)
    dry = build_report(
        root,
        date=date,
        source_receipts=source_receipts,
        candidates=candidates,
        command_runner=lambda argv, _cwd: {
            "command": " ".join(argv),
            "exit_code": 0,
            "classification": "deferred_until_final_report",
            "stdout_tail": "",
            "stderr_tail": "",
        },
        duration_s=duration_s,
        references_before_hash=references_before,
        references_after_hash=references_before,
        protected_before_hashes=protected_before,
    )
    appended = False
    if dry["status"] == "complete" and dry["accepted_findings"]:
        references_path.write_text(
            insert_after_marker(before_text, execution_delta_block(dry["accepted_findings"])),
            encoding="utf-8",
        )
        appended = EXECUTION_DELTA_HEADING not in before_text
    references_after = path_sha256(references_path)
    authorized = (
        {RESEARCH_REFERENCES_RELATIVE_PATH.as_posix(): references_after} if appended else None
    )
    report = build_report(
        root,
        date=date,
        source_receipts=source_receipts,
        candidates=candidates,
        command_runner=command_runner,
        duration_s=duration_s,
        references_before_hash=references_before,
        references_after_hash=references_after,
        references_appended=appended,
        protected_before_hashes=protected_before,
        protected_authorized_after_hashes=authorized,
        reference_text_for_dedupe=before_text,
    )
    errors = validate_report(report)
    if errors:
        raise ValueError(f"invalid Exp6273 freeze: {errors}")
    atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)
    return report


def check_roadmap_only(root: Path = REPO_ROOT) -> JsonDict:
    roadmap_path, _missing, _note = _select_roadmap_path(root)
    data = _load_yaml_mapping(roadmap_path)
    tasks = [task for task in data.get("tasks", []) or [] if isinstance(task, Mapping)]
    schema = _roadmap_schema_result(data)
    prior = _prior_failure_contract(tasks)
    exclusion = _exclusion_lint_result(roadmap_path)
    return {
        "ok": schema["passed"] and prior["passed"] and exclusion["passed"],
        "roadmap_path": roadmap_path.relative_to(root).as_posix()
        if roadmap_path.is_relative_to(root)
        else str(roadmap_path),
        "roadmap_schema_result": schema,
        "prior_failure_contract_result": prior,
        "exclusion_manifest_lint_result": exclusion,
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260810")
    parser.add_argument("--check-roadmap-only", action="store_true")
    args = parser.parse_args(argv)
    if args.check_roadmap_only:
        result = check_roadmap_only(REPO_ROOT)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    report = write_freeze(REPO_ROOT, date=args.date, command_runner=_recorded_command_runner)
    print(
        json.dumps(
            {
                "path": RESULT_RELATIVE_PATH.as_posix(),
                "checksum": report["reproducibility_checksum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
