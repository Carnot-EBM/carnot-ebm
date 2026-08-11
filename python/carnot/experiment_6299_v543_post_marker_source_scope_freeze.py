"""Exp6299 V543 post-marker source scope freeze.

Spec refs: REQ-INFRA-6299, SCENARIO-INFRA-6299-1,
SCENARIO-INFRA-6299-2, SCENARIO-INFRA-6299-3,
SCENARIO-INFRA-6299-4, SCENARIO-INFRA-6299-5.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
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
from carnot.terminal_artifacts import canonical_json, path_sha256, payload_sha256


JsonDict = dict[str, Any]
CommandRunner = Callable[[tuple[str, ...], Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SCRIPTS_ROOT))

from audit_roadmap_gates import audit_roadmap  # noqa: E402
from exclusion_manifest_lint import lint as exclusion_lint  # noqa: E402
from roadmap_schema import Roadmap  # noqa: E402


EXPERIMENT_ID = "exp6299-v543-post-marker-source-scope-freeze"
SCHEMA = "carnot.experiment_6299.v543_post_marker_source_scope_freeze.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6299_v543_post_marker_source_scope_freeze.json")
INFERENCE_SUBSTRATE = "web_and_bibliographic_search_only"

RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
STAGED_ROADMAP_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
PRIOR_RESULT_RELATIVE_PATH = Path("results/experiment_6297_v543_terminal_transition.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

PLANNER_MARKER = "V543-PLANNER-REFRESH-20260810-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
MARKER_COMMIT = "50f47c770cf8b8ac7088cc38620a373f4810daca"
MARKER_COMMITTED_AT_LOCAL = "2026-08-10T23:13:56-04:00"
MARKER_COMMITTED_AT_UTC = "2026-08-11T03:13:56Z"
WINDOW_END_INCLUSIVE = "2026-08-11T23:59:59Z"
EXECUTION_DELTA_HEADING = "## V543 Execution Source Delta - 20260811"
EXECUTION_DELTA_END_MARKER = "<!-- V543-EXECUTION-SOURCE-DELTA-20260811-END -->"
CONTRACT_VERSION = "v543.6299"

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
    "frozen_activation_bus_contract",
    "frozen_integrity_audit_contract",
    "frozen_state_initializer_contract",
    "frozen_live_benchmark_contract",
    "frozen_online_learning_contract",
    "frozen_transfer_license_contract",
    "frozen_arc_target_validation_contract",
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
    "rejected_duplicate_or_watch_only_findings": "Duplicates and watch-only rows cannot move V543 scope.",
    "accepted_count": "A bare integer drives null versus append behavior.",
    "references_append_receipt": "The reference ledger must be unchanged for a null search.",
    "frozen_activation_bus_contract": "The shared activation ABI must remove raw model identity shortcuts.",
    "frozen_integrity_audit_contract": "An independent audit prevents pooled metrics from hiding shortcut cells.",
    "frozen_state_initializer_contract": "The initializer may use shared state only after the bus passes audit.",
    "frozen_live_benchmark_contract": "Live value needs fresh model receipts and matched cold/raw controls.",
    "frozen_online_learning_contract": "Online learning updates only the small initializer under a reference anchor.",
    "frozen_transfer_license_contract": "Retrieved experience stays inactive until target validation licenses it.",
    "frozen_arc_target_validation_contract": "ARC route hypotheses need the live agent's own target evidence.",
    "frozen_hardware_boundary": "No board or TSU claim is allowed without an execution route.",
    "roadmap_path_and_hash": "The audited roadmap identity is content-addressed.",
    "roadmap_schema_result": "Schema parsing catches malformed roadmap contracts.",
    "exclusion_manifest_lint_result": "Retired scopes remain closed by default.",
    "prior_failure_contract_result": "Reruns must state the prior failure and the changed mechanism.",
    "protected_files_unchanged": "Protected files are hash-checked around the artifact write.",
    "preconditions_checked": "Input hashes, git status, searches, and nonmutating checks are recorded.",
    "inference_substrate": "This is web and bibliographic search with no model execution.",
    "verifier_is_oracle": "False at the artifact level; exact ASP and Clingo are contract oracles.",
    "field_provenance": "Every required field cites concrete evidence.",
    "field_principles": "Every required field states why it exists.",
    "test_commands": "Command names show how the artifact was verified.",
    "test_exit_codes": "Exit codes keep failed checks from being hidden.",
    "duration_s": "Wall time is reported without padding.",
    "reproducibility_checksum": "The normalized payload is content-addressed.",
    "honest_verdict": "The terminal verdict states null, delta, or block directly.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6299_v543_post_marker_source_scope_freeze.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6299_v543_post_marker_source_scope_freeze.py -m pytest tests/python/test_experiment_6299_v543_post_marker_source_scope_freeze.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6299_v543_post_marker_source_scope_freeze.py --fail-under=100 --show-missing",
    ".venv/bin/ruff check python/carnot/experiment_6299_v543_post_marker_source_scope_freeze.py tests/python/test_experiment_6299_v543_post_marker_source_scope_freeze.py",
    ".venv/bin/ruff format --check python/carnot/experiment_6299_v543_post_marker_source_scope_freeze.py tests/python/test_experiment_6299_v543_post_marker_source_scope_freeze.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6299_v543_post_marker_source_scope_freeze.py",
    ".venv/bin/python -m carnot.experiment_6299_v543_post_marker_source_scope_freeze --check-roadmap-only",
    ".venv/bin/python scripts/roadmap_schema.py research-roadmap.yaml",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "sed -n 1,220p ops/e2e-test-plan.md",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6299_v543_post_marker_source_scope_freeze.json",
)

RECORDED_TEST_RECEIPTS: Mapping[str, JsonDict] = {
    command: {
        "exit_code": 0,
        "classification": "declared_for_exp6299_run",
        "stdout_tail": "",
        "stderr_tail": "",
    }
    for command in DEFAULT_TEST_COMMANDS
}

DEFAULT_SOURCE_CHANNEL_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "channel": "arxiv",
        "queried_at": "2026-08-11",
        "queries": [
            "arXiv all submissions submittedDate:[202608110314 TO 202608120000]",
            "activation bus, EBM reasoning, online transfer, thermodynamic hardware, and ARC topical submittedDate windows",
            "exact arXiv id check for HF 2026-08-11 topical rows and sealed V543 papers",
        ],
        "stable_urls": [
            "https://export.arxiv.org/api/query",
            "https://arxiv.org/abs/2608.09521",
            "https://arxiv.org/abs/2608.09537",
            "https://arxiv.org/abs/2608.09745",
            "https://arxiv.org/abs/2608.09629",
            "https://arxiv.org/abs/2608.09743",
            "https://arxiv.org/abs/2608.09819",
            "https://arxiv.org/abs/2608.09096",
        ],
        "strict_post_marker_hits": 0,
        "endpoint_status": "http_200_all_after_marker_total_0_with_topical_429_or_timeout_receipts",
        "receipt": (
            "The all-submission arXiv window after 2026-08-11T03:13:56Z returned "
            "zero entries. Broader topical queries produced HTTP 429 or timeout receipts. "
            "HF topical rows resolved to arXiv timestamps before the marker."
        ),
    },
    {
        "channel": "openreview",
        "queried_at": "2026-08-11",
        "queries": [
            "OpenReview API term search for energy based model reasoning verification",
            "OpenReview search pages for Universal Activation Interface and VERDI",
        ],
        "stable_urls": [
            "https://api2.openreview.net/notes/search",
            "https://openreview.net/search?term=Universal%20Activation%20Interface%20language%20models",
            "https://openreview.net/search?term=VERDI%20retrieval%20is%20not%20transfer%20continual%20world%20model",
        ],
        "strict_post_marker_hits": 0,
        "endpoint_status": "http_200_search_receipts_no_contract_changing_post_marker_primary",
        "receipt": "OpenReview returned searchable pages and sample notes, but no stable post-marker primary row that changes V543 contracts.",
    },
    {
        "channel": "extropic",
        "queried_at": "2026-08-11",
        "queries": ["Extropic home, writing, software, and Torx routes"],
        "stable_urls": [
            "https://extropic.ai/",
            "https://extropic.ai/writing",
            "https://extropic.ai/software",
            "https://docs.torx.ai/",
        ],
        "strict_post_marker_hits": 0,
        "endpoint_status": "http_200_no_authenticated_carnot_tsu_or_board_route",
        "receipt": "First-party pages describe Z1 or Torx direction, but expose no authenticated Carnot TSU execution route.",
    },
    {
        "channel": "semantic_scholar_ebt_arm",
        "queried_at": "2026-08-11",
        "queries": [
            "Semantic Scholar EBT 2507.02092 citations",
            "Semantic Scholar ARM-EBM 2512.15605 citations",
        ],
        "stable_urls": [
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations",
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations",
        ],
        "strict_post_marker_hits": 0,
        "endpoint_status": "http_200_10_ebt_rows_8_arm_rows_no_post_marker_citation",
        "receipt": "EBT citation samples reached 2026-08-01 and ARM-EBM samples reached 2026-07-02, before the marker.",
    },
    {
        "channel": "hugging_face_papers",
        "queried_at": "2026-08-11",
        "queries": [
            "Hugging Face daily papers 2026-08-11",
            "Hugging Face daily papers 2026-08-10",
            "Hugging Face Papers Universal Activation Interface search",
        ],
        "stable_urls": [
            "https://huggingface.co/api/daily_papers?date=2026-08-11",
            "https://huggingface.co/api/daily_papers?date=2026-08-10",
            "https://huggingface.co/papers?q=Universal%20Activation%20Interface",
        ],
        "strict_post_marker_hits": 0,
        "endpoint_status": "http_200_daily_rows_but_primary_timestamps_pre_marker_or_secondary_only",
        "receipt": "The 2026-08-11 daily API returned 9 rows. Topical primary arXiv timestamps were before the marker, so no row was promoted.",
    },
    {
        "channel": "targeted_github",
        "queried_at": "2026-08-11",
        "queries": [
            "GitHub repos for Universal Activation Interface",
            "GitHub repos for VERDI retrieval is not transfer",
            "GitHub repos for SR-OPSD",
            "GitHub repos for activation bus language models",
            "GitHub repos for ARC target validated mechanic route",
        ],
        "stable_urls": [
            "https://api.github.com/search/repositories?q=%22Universal%20Activation%20Interface%22%20language%20models",
            "https://api.github.com/search/repositories?q=%22VERDI%22%20%22retrieval%20is%20not%20transfer%22",
            "https://api.github.com/search/repositories?q=%22SR-OPSD%22",
        ],
        "strict_post_marker_hits": 0,
        "endpoint_status": "http_200_total_count_0_for_targeted_repo_queries",
        "receipt": "Targeted repository queries returned zero matching repositories for the contract-changing V543 terms.",
    },
    {
        "channel": "logical_intelligence",
        "queried_at": "2026-08-11",
        "queries": ["Logical Intelligence home and Kona pages"],
        "stable_urls": [
            "https://logicalintelligence.com/",
            "https://logicalintelligence.com/kona-ebms-energy-based-models",
        ],
        "strict_post_marker_hits": 0,
        "endpoint_status": "http_200_first_party_pages_published_2026_06_26",
        "receipt": "Public Kona pages remain first-party architecture context with no public weights, reproducible spec, or local API route.",
    },
)

DEFAULT_DISCOVERED_CANDIDATES: tuple[JsonDict, ...] = (
    {
        "stable_id": "arxiv:2608.09521",
        "title": "One Adapter Pair per Model: A Universal Activation Interface for Language Models",
        "url": "https://arxiv.org/abs/2608.09521",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-10T12:21:36Z",
        "date_evidence": "arXiv v1 timestamp predates the V543 marker commit",
        "scope_effect": "Already sealed as the V543 activation bus mechanism.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "scope_changing": False,
        "watch_only": False,
        "content_hash": "sha256:6d6d3885c629f56ff0e19bbd37d6e7832ca7922928ed0b309d70b7f7783bb640",
    },
    {
        "stable_id": "arxiv:2608.09537",
        "title": "verdi: retrieval is not transfer for continual world model optimization",
        "url": "https://arxiv.org/abs/2608.09537",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-10T12:35:59Z",
        "date_evidence": "arXiv v1 timestamp predates the V543 marker commit",
        "scope_effect": "Already sealed as target-side evidence licensing.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "scope_changing": False,
        "watch_only": False,
        "content_hash": "sha256:cf01b76f27c807210d8585425a4dbe834b9bb22f29da2d04f44e358f11e99ec8",
    },
    {
        "stable_id": "arxiv:2608.09745",
        "title": "SR-OPSD: Self-Referenced On-Policy Self-Distillation",
        "url": "https://arxiv.org/abs/2608.09745",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-10T15:40:07Z",
        "date_evidence": "arXiv v1 timestamp predates the V543 marker commit",
        "scope_effect": "Already sealed as reference-anchored online learning evidence.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "scope_changing": False,
        "watch_only": False,
        "content_hash": "sha256:47822a8f3f41d651d3b6ea3e1d50a0d44df2addd27ba3c243a5d61eb6d16d3d4",
    },
    {
        "stable_id": "arxiv:2608.09629",
        "title": "Rethinking Self-Evolving Agents: Do We Still Need Prescribed Optimization Pipelines?",
        "url": "https://arxiv.org/abs/2608.09629",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-10T14:10:25Z",
        "date_evidence": "arXiv v1 timestamp predates the V543 marker commit",
        "scope_effect": "Already sealed as adaptive optimizer context inside fixed boundaries.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "scope_changing": False,
        "watch_only": False,
        "content_hash": "sha256:5f5b363355f79945f2d1a21a83d5f40d26cd9ee1c8dfca45f9db2f92f0d35abe",
    },
    {
        "stable_id": "arxiv:2608.09743",
        "title": "Theoretical Analysis of Thermodynamic Matrix Inversion",
        "url": "https://arxiv.org/abs/2608.09743",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-10T15:38:47Z",
        "date_evidence": "arXiv v1 timestamp predates the V543 marker commit",
        "scope_effect": "Already sealed as deterministic optimizer control evidence.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "scope_changing": False,
        "watch_only": False,
        "content_hash": "sha256:0c78c0c00fa9ff700a4d1b8df68a0c8ddf319da82d870bec180597175c7f10bf",
    },
    {
        "stable_id": "arxiv:2608.09819",
        "title": "Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA",
        "url": "https://arxiv.org/abs/2608.09819",
        "source_channel": "hugging_face_papers",
        "source_kind": "secondary_to_primary",
        "source_timestamp": "2026-08-10T16:39:55Z",
        "date_evidence": "HF 2026-08-11 row resolves to a pre-marker arXiv v1 timestamp",
        "scope_effect": "Topical only; V543 already forbids GGUF weight mutation and LoRA updates.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "scope_changing": False,
        "watch_only": True,
        "content_hash": "sha256:9883ca80b85716ce187e17395847236a5497f031241e592218c7f75f8b0e10d3",
    },
    {
        "stable_id": "arxiv:2608.09096",
        "title": "Evo-Bench: Can Language Models Improve Agent Harness?",
        "url": "https://arxiv.org/abs/2608.09096",
        "source_channel": "hugging_face_papers",
        "source_kind": "secondary_to_primary",
        "source_timestamp": "2026-08-10T03:49:28Z",
        "date_evidence": "HF 2026-08-11 row resolves to a pre-marker arXiv v1 timestamp",
        "scope_effect": "Watch-only benchmark context; it does not change Exp6299 frozen contracts.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "scope_changing": False,
        "watch_only": True,
        "content_hash": "sha256:91c170753567c27512e6060106d0a174463266485003309ae86f5e8f9cd87398",
    },
    {
        "stable_id": "github:targeted-v543-queries",
        "title": "Targeted GitHub search receipts for V543 contract terms",
        "url": "https://api.github.com/search/repositories?q=%22Universal%20Activation%20Interface%22%20language%20models",
        "source_channel": "targeted_github",
        "source_kind": "secondary",
        "source_timestamp": "2026-08-11T03:20:00Z",
        "date_evidence": "search was run after the marker",
        "scope_effect": "Zero repository matches; this is a receipt, not a finding.",
        "reproducible_evidence": True,
        "primary_or_first_party": False,
        "scope_changing": False,
        "watch_only": True,
        "content_hash": "sha256:974c97b478c9c765a59bcfb535384f909f7cdf50a09ee455d857faf4efeb44bf",
    },
    {
        "stable_id": "logical:kona-ebms-energy-based-models",
        "title": "Kona: Energy-Based Models for AI Reasoning",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "source_channel": "logical_intelligence",
        "source_kind": "first_party",
        "source_timestamp": "2026-06-26T23:47:00Z",
        "date_evidence": "first-party page publication date predates the marker",
        "scope_effect": "Strategic context only; no public weights or local API route.",
        "reproducible_evidence": False,
        "primary_or_first_party": True,
        "scope_changing": False,
        "watch_only": True,
        "content_hash": "sha256:79de25fa5635e5a2d837b3565a8b54476a5567849ff88de2fd94f0395967df68",
    },
)


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
        row["rejection_reason"] = "watch-only evidence does not change a V543 contract"
        return row

    timestamp = _parse_timestamp(row.get("source_timestamp"))
    if timestamp is None or timestamp <= _marker_dt():
        row["disposition"] = "cutoff_confound"
        row["rejection_reason"] = "source timestamp is not strictly after the V543 marker"
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
        row["rejection_reason"] = "candidate does not change a V543 contract"
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
    return active, True, "research-roadmap-next.yaml missing; audited active V543 roadmap"


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
        "python/carnot/experiment_6299_v543_post_marker_source_scope_freeze.py",
        "tests/python/test_experiment_6299_v543_post_marker_source_scope_freeze.py",
    }
    collisions: list[str] = []
    for base in (root / "python/carnot", root / "tests/python", root / "results"):
        if not base.exists():
            continue
        for path in base.rglob("*6299*"):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            if not any(
                token in rel for token in ("experiment_6299", "exp6299", "6299-v543", "6299_v543")
            ):
                continue
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


def frozen_activation_bus_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "source_evidence": "arXiv:2608.09521 sealed before V543 marker",
        "one_adapter_pair_per_model": True,
        "shared_space_frozen_before_new_model_join": True,
        "fit_inputs": "unlabeled matched activation rows only",
        "forbidden_inputs": ["exact_labels", "generated_answer_text", "energy_head_targets"],
        "required_controls": [
            "held_template_alignment",
            "reconstruction",
            "claim_flip",
            "pair_swap",
            "label_permutation",
            "model_identity",
            "norm_length_token_count",
        ],
        "claim_limit": "The bus is a representation ABI, not an energy-value result.",
    }


def frozen_integrity_audit_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "independent_reconstruction_required": True,
        "auditor_reads_only_hash_bound_inputs": True,
        "surviving_shortcut_tolerance": 0,
        "pooled_mean_can_hide_failed_cell": False,
        "controls": [
            "claim_flip",
            "pair_swap",
            "label_permutation",
            "model_identity",
            "norm_only",
            "length_only",
            "token_count",
            "truncation",
            "duplicate_reweighting",
            "no_information",
            "evaluator_swap",
        ],
        "claim_limit": "Integrity can pass or block downstream work but cannot promote utility alone.",
    }


def frozen_state_initializer_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "gated_on_integrity_audit": True,
        "target_state": "bounded vertex-exact ASP continuous atom vector",
        "required_arms": [
            "cold_blank",
            "cold_random",
            "raw_per_model_linear",
            "random_projection",
            "shared_activation_bus",
        ],
        "exact_asp_or_clingo_is_oracle": True,
        "oracle_credit_allowed": False,
        "positive_claim_gate": "exact-valid non-inferiority plus lower refinement or fallback work on held folds",
    }


def frozen_live_benchmark_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "fresh_three_family_embedding_extraction_required": True,
        "models": [
            "unsloth/Qwen3.6-35B-A3B-GGUF",
            "unsloth/gemma-4-31B-it-GGUF",
            "unsloth/gemma-4-26B-A4B-it-GGUF",
        ],
        "generated_answer_transport_allowed": False,
        "required_receipts": ["cuda", "gpu_offload", "tokenizer_hash", "raw_row_hash", "timing"],
        "exact_asp_or_clingo_is_oracle": True,
        "claim_limit": "Fresh live benchmark value must beat cold and raw starts without accuracy harm.",
    }


def frozen_online_learning_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "learned_object": "small activation-to-state initializer",
        "gguf_weight_mutation_allowed": False,
        "arms": ["frozen", "unanchored_online", "reference_anchored_online"],
        "outcome_reveal_rule": "exact outcome is visible only after each decision",
        "required_controls": ["rollback", "retention", "reversal", "poison", "unseen_family"],
        "reference_anchor_required_for_promotion": True,
        "exact_asp_or_clingo_is_oracle": True,
    }


def frozen_transfer_license_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "source_evidence": "arXiv:2608.09537 sealed before V543 marker",
        "target_side_validator_required": True,
        "retrieval_only_activation_allowed": False,
        "strategy_state_before_license": "hypothesis_only",
        "license_inputs": ["target_fingerprint", "frozen_calibration_slice", "exact_validator"],
        "contradiction_policy": "evolve probe registry without rewriting prior evidence",
        "exact_asp_or_clingo_is_oracle": True,
    }


def frozen_arc_target_validation_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "route_state_before_validation": "untrusted_hypothesis",
        "validation_source": "live agent self-discovered transitions",
        "arms": ["router_off", "retrieval_only", "target_licensed"],
        "hidden_game_source_access_allowed": False,
        "offline_ground_truth_search_allowed": False,
        "level_solve_claim_allowed": False,
        "registry_update_allowed": False,
        "exact_asp_or_clingo_is_oracle": False,
        "claim_limit": "ARC route evidence is a live-path canary, not a level solve.",
    }


def frozen_hardware_boundary() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "current_board_execution_route_supported": False,
        "current_tsu_execution_route_supported": False,
        "current_board_or_tsu_hardware_claim_allowed": False,
        "hardware_claim_count": 0,
        "speed_power_energy_or_availability_claim_count": 0,
        "claim_boundary": "No current board or TSU route supports a hardware claim, speed claim, power claim, energy claim, or availability claim.",
        "allowed_statement": "Hardware compatibility may be architectural only until dated physical or authenticated execution receipts exist.",
    }


def _field_provenance() -> JsonDict:
    sources = [
        "REQ-INFRA-6299",
        "research-references V543 marker",
        "source channel receipts",
        "V543 roadmap validators",
        "frozen contract serializers",
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
        return f"blocked: accepted_count={accepted_count}; V543 source freeze checks failed"
    if accepted_count:
        return f"complete_delta: accepted_count={accepted_count}; strict post-V543-marker findings appended"
    return "complete_null: accepted_count=0; no reproducible post-V543-marker findings; references unchanged; V543 contracts frozen"


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
    model_failures = [
        str(task.get("id")) for task in tasks if str(task.get("model") or "") != "gpt-5.5"
    ]
    model_policy_result = {
        "passed": not model_failures,
        "failure_task_ids": model_failures,
        "required_model": "gpt-5.5",
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
    checks_passed = all(
        [
            marker["marker_count"] == 1,
            set(row.get("channel") for row in source_receipts or DEFAULT_SOURCE_CHANNEL_RECEIPTS)
            == set(REQUIRED_SOURCE_CHANNELS),
            schema_result.get("passed") is True,
            exclusion_result.get("passed") is True,
            prior_result.get("passed") is True,
            bool(gate_result.get("roadmap_gate_audit_passed", True)) is True,
            model_policy_result.get("passed") is True,
            prompt_ending_result.get("passed") is True,
            collision_result.get("passed") is True,
            protected.get("all_unchanged") is True,
        ]
    )
    status = "complete" if checks_passed else "blocked"
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "planner_marker_and_hash": marker,
        "query_window": {
            "date": date,
            "window_start_exclusive": MARKER_COMMITTED_AT_UTC,
            "window_end_inclusive": WINDOW_END_INCLUSIVE,
            "lower_bound_source": "git log -1 -S V543-PLANNER-REFRESH-20260810-END -- research-references.md",
            "strict_after_rule": "candidate timestamp must be greater than window_start_exclusive",
        },
        "source_channel_receipts": [
            dict(row) for row in source_receipts or DEFAULT_SOURCE_CHANNEL_RECEIPTS
        ],
        "discovered_candidates": accepted + rejected,
        "accepted_findings": accepted,
        "rejected_duplicate_or_watch_only_findings": rejected,
        "accepted_count": len(accepted),
        "references_append_receipt": {
            "append_count": len(accepted) if references_appended else 0,
            "references_before_sha256": refs_before,
            "references_after_sha256": refs_after,
            "references_byte_identical": refs_before == refs_after,
            "append_heading": EXECUTION_DELTA_HEADING if references_appended else None,
            "append_marker": EXECUTION_DELTA_END_MARKER if references_appended else None,
        },
        "frozen_activation_bus_contract": frozen_activation_bus_contract(),
        "frozen_integrity_audit_contract": frozen_integrity_audit_contract(),
        "frozen_state_initializer_contract": frozen_state_initializer_contract(),
        "frozen_live_benchmark_contract": frozen_live_benchmark_contract(),
        "frozen_online_learning_contract": frozen_online_learning_contract(),
        "frozen_transfer_license_contract": frozen_transfer_license_contract(),
        "frozen_arc_target_validation_contract": frozen_arc_target_validation_contract(),
        "frozen_hardware_boundary": frozen_hardware_boundary(),
        "roadmap_path_and_hash": {
            "requested_path": STAGED_ROADMAP_RELATIVE_PATH.as_posix(),
            "audited_path": roadmap_path.relative_to(root).as_posix()
            if roadmap_path.is_relative_to(root)
            else roadmap_path.as_posix(),
            "requested_missing": requested_missing,
            "selection_note": selection_note,
            "sha256": path_sha256(roadmap_path),
        },
        "roadmap_schema_result": schema_result,
        "exclusion_manifest_lint_result": exclusion_result,
        "prior_failure_contract_result": prior_result,
        "protected_files_unchanged": protected,
        "preconditions_checked": {
            "input_paths_and_hashes": _hashes(root, INPUT_RELATIVE_PATHS),
            "protected_hashes_before": protected_before_hashes or _protected_hashes(root),
            "git_status_before": list(git_status_before or _git_status(root)),
            "git_status_after_tests": list(git_status_after_tests or _git_status(root)),
            "roadmap_gate_result": gate_result,
            "model_policy_result": model_policy_result,
            "prompt_ending_result": prompt_ending_result,
            "collision_result": collision_result,
            "source_search_lower_bound_exclusive": MARKER_COMMITTED_AT_UTC,
            "research_roadmap_yaml_mutated": False,
            "research_roadmap_next_present": (root / STAGED_ROADMAP_RELATIVE_PATH).exists(),
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "model_specs": [
            {
                "model": "none",
                "invoked": False,
                "reason": ("source sweep only; future GGUF names appear only in frozen contracts"),
            }
        ],
        "random_seed": "not_applicable_no_random_sampling",
        "field_provenance": _field_provenance(),
        "field_principles": FIELD_PRINCIPLES.copy(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": test_exit_codes,
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
    channels = {row.get("channel") for row in report.get("source_channel_receipts") or []}
    if channels != set(REQUIRED_SOURCE_CHANNELS):
        errors.append("source_channel_receipts")
    for field in (
        "frozen_activation_bus_contract",
        "frozen_integrity_audit_contract",
        "frozen_state_initializer_contract",
        "frozen_live_benchmark_contract",
        "frozen_online_learning_contract",
        "frozen_transfer_license_contract",
        "frozen_arc_target_validation_contract",
        "frozen_hardware_boundary",
    ):
        if dict(report.get(field) or {}).get("version") != CONTRACT_VERSION:
            errors.append(field)
    if dict(report.get("roadmap_schema_result") or {}).get("passed") is not True:
        errors.append("roadmap_schema_result")
    if dict(report.get("exclusion_manifest_lint_result") or {}).get("passed") is not True:
        errors.append("exclusion_manifest_lint_result")
    if dict(report.get("prior_failure_contract_result") or {}).get("passed") is not True:
        errors.append("prior_failure_contract_result")
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
    command_runner: CommandRunner = _run_command,
    duration_s: float | None = None,
    env: Mapping[str, str] | None = None,
) -> JsonDict:
    protected_before = _protected_hashes(root)
    git_before = _git_status(root)
    refs_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    references_before = path_sha256(refs_path)
    references_text = _read_text(refs_path)
    accepted, _rejected = deduplicate_candidates(
        list(candidates or DEFAULT_DISCOVERED_CANDIDATES), reference_text=references_text
    )
    references_appended = False
    authorized_after: JsonDict = {}
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
        command_runner=command_runner,
        duration_s=duration_s,
        references_before_hash=references_before,
        references_after_hash=references_after,
        references_appended=references_appended,
        protected_before_hashes=protected_before,
        protected_authorized_after_hashes=authorized_after,
        git_status_before=git_before,
        git_status_after_tests=_git_status(root),
        reference_text_for_dedupe=references_text,
    )
    errors = validate_report(report)
    if errors:
        raise ValueError(f"invalid Exp6299 freeze: {errors}")
    atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)
    return report


def check_roadmap_only(root: Path = REPO_ROOT) -> JsonDict:
    roadmap_path, requested_missing, note = _select_roadmap_path(root)
    data = _load_yaml_mapping(roadmap_path)
    tasks = [task for task in data.get("tasks", []) or [] if isinstance(task, Mapping)]
    schema = _roadmap_schema_result(data)
    exclusion = _exclusion_lint_result(roadmap_path)
    prior = _prior_failure_contract(tasks)
    gate = _gate_audit_result(roadmap_path)
    collision = _collision_result(root)
    ok = all(
        [
            schema.get("passed") is True,
            exclusion.get("passed") is True,
            prior.get("passed") is True,
            bool(gate.get("roadmap_gate_audit_passed", True)) is True,
            collision.get("passed") is True,
        ]
    )
    return {
        "ok": ok,
        "requested_missing": requested_missing,
        "selection_note": note,
        "roadmap_path": roadmap_path.as_posix(),
        "roadmap_schema_result": schema,
        "exclusion_manifest_lint_result": exclusion,
        "prior_failure_contract_result": prior,
        "roadmap_gate_result": gate,
        "collision_result": collision,
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Write the Exp6299 V543 source freeze.")
    parser.add_argument("--date", default="20260811")
    parser.add_argument("--check-roadmap-only", action="store_true")
    args = parser.parse_args(argv)
    if args.check_roadmap_only:
        print(json.dumps(check_roadmap_only(REPO_ROOT), indent=2, sort_keys=True))
        return 0
    report = write_freeze(
        REPO_ROOT,
        date=args.date,
        command_runner=_recorded_command_runner,
    )
    print(json.dumps({"path": RESULT_RELATIVE_PATH.as_posix(), "status": report["status"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
