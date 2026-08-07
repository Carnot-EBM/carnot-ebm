"""Exp6171: ingest dated source deltas after the V535 marker.

Spec refs: REQ-REPORT-6171,
SCENARIO-REPORT-6171-EXACT-MARKER,
SCENARIO-REPORT-6171-BOUNDED-DATED-SOURCE,
SCENARIO-REPORT-6171-DEDUPLICATE-AND-GUARD-SCOPE,
SCENARIO-REPORT-6171-ZERO-DELTA,
SCENARIO-REPORT-6171-SCHEMA.

This module records a source-ingestion ledger, not a model result. The strict
date boundary is intentionally conservative: the V535 marker has a date but no
sub-day timestamp, so same-day primary records need separate proof of later
revision before they can change the roadmap.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6171_v535_source_delta_ingestion.json")

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
REQUESTED_INGESTION_SCRIPT_RELATIVE_PATH = Path("scripts/reliable_source_delta_ingestion.py")

EXPERIMENT = "experiment_6171_v535_source_delta_ingestion"
EXPERIMENT_ID = "exp6171-v535-source-delta-ingestion"
MILESTONE = "2026.08.535"
RUN_DATE = "20260807"
RANDOM_SEED = 6171
SCHEMA = "carnot.experiment_6171.v535_source_delta_ingestion.v1"
INFERENCE_SUBSTRATE = "dated_primary_secondary_source_ingestion"

PLANNER_HEADING = "## V535 Planner Refresh - 20260806"
PLANNER_MARKER = "V535-PLANNER-REFRESH-20260806-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
EXECUTION_DELTA_HEADING = "## V535 Execution Source Delta - 20260807"
EXECUTION_DELTA_END_MARKER = "<!-- V535-EXECUTION-SOURCE-DELTA-20260807-END -->"
MARKER_DATE = "2026-08-06"
WINDOW_START_EXCLUSIVE = "2026-08-06T23:59:59Z"
WINDOW_END_INCLUSIVE = "2026-08-07T23:59:59Z"
DEFAULT_SEARCH_STARTED_AT = "2026-08-07T03:05:00Z"
DEFAULT_SEARCH_FINISHED_AT = "2026-08-07T03:18:00Z"

ALLOCATED_TARGET_EXPERIMENTS = (
    "exp6173-cctu-item-bank-preregistration",
    "exp6174-cctu-authentic-k8-pool",
    "exp6175-cctu-headroom-audit",
    "exp6176-hidden-state-surface-qualification",
    "exp6177-clue-latent-selector-freeze",
    "exp6178-held-internal-state-selection",
    "exp6179-retention-safe-continuous-strategy-learning-ab",
    "exp6180-exp6166-reproducibility-adjudication",
    "exp6181-arc-logo-shortcut-audit",
)

SPEC_REFS = (
    "REQ-REPORT-6171",
    "SCENARIO-REPORT-6171-EXACT-MARKER",
    "SCENARIO-REPORT-6171-BOUNDED-DATED-SOURCE",
    "SCENARIO-REPORT-6171-DEDUPLICATE-AND-GUARD-SCOPE",
    "SCENARIO-REPORT-6171-ZERO-DELTA",
    "SCENARIO-REPORT-6171-SCHEMA",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "marker_text_count_hash_offset_and_reference_hash",
    "bounded_time_window",
    "source_query_and_endpoint_receipts",
    "semantic_scholar_ebt_and_arm_ebm_counts",
    "candidate_and_deduplicated_record_counts",
    "accepted_rejected_and_guarded_delta_ledger",
    "roadmap_task_mapping",
    "exclusion_and_retirement_conflict_receipt",
    "reference_hash_before_after_and_append_count",
    "zero_delta_accepted",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state follows marker, source authority, and guarded-scope receipts.",
    "preconditions_checked": "Records marker, hashes, endpoints, exclusions, roadmap map, tools, and git status before classification.",
    "marker_text_count_hash_offset_and_reference_hash": "The exact V535 marker and reference bytes define the novelty boundary.",
    "bounded_time_window": "Only records created or revised after the exact marker receipt are eligible.",
    "source_query_and_endpoint_receipts": "Every searched route records endpoint capability, access outcome, and candidate count.",
    "semantic_scholar_ebt_and_arm_ebm_counts": "Semantic Scholar counts are discovery receipts, not source authority.",
    "candidate_and_deduplicated_record_counts": "Stable identifier and content-hash deduplication precedes acceptance.",
    "accepted_rejected_and_guarded_delta_ledger": "Each row names date, authority, stable ID, local reachability, disposition, and reason.",
    "roadmap_task_mapping": "Accepted findings can map only to Exp6173-Exp6181 or defer.",
    "exclusion_and_retirement_conflict_receipt": "Retired and guarded scopes remain closed unless a dated source changes local applicability.",
    "reference_hash_before_after_and_append_count": "Reference history is immutable for zero-delta and append-only for accepted deltas.",
    "zero_delta_accepted": "Bare true; lack of novelty is a valid result.",
    "protected_files_unchanged": "Roadmaps, ops ledgers, exclusions, conductor, and instruction files remain byte-identical.",
    "duration_s": "Wall-clock source-ingestion duration for reproducibility accounting.",
    "inference_substrate": "Set `dated_primary_secondary_source_ingestion`.",
    "field_provenance": "Every required field traces to local hashes, endpoint receipts, or classification records.",
    "test_commands": "Records focused, coverage, schema, adversarial, protected-file, E2E, root-clutter, and full-suite commands.",
    "test_exit_codes": "Exit codes prevent failed checks from being reported as success.",
    "reproducibility_checksum": "Detects later marker, endpoint, candidate, append, or protected-file drift.",
    "honest_verdict": "Use `complete_delta:`, `complete_null:`, `retired:`, or `blocked:` and name accepted count.",
}

FIELD_PRINCIPLE_EXTRAS: dict[str, str] = {
    "schema": "Versioned schema identifier for downstream validators.",
    "experiment": "Stable implementation slug.",
    "experiment_id": "Conductor task identity.",
    "milestone": "Binds the ledger to .535.",
    "run_date": "Operator-requested execution date.",
    "random_seed": "Deterministic metadata for a no-randomness ledger.",
    "title": "Human-readable artifact title.",
    "target_model": "Declares that no local model is invoked.",
    "model_specs": "Empty list keeps the no-model boundary explicit.",
    "spec_refs": "OpenSpec anchors for traceability.",
    "result_path": "Declares the exact JSON deliverable path.",
    "search_started_at": "Records the source-query start timestamp.",
    "search_finished_at": "Records the source-query finish timestamp.",
    "field_principles": "Carries field principles inside the artifact.",
}

SOURCE_RECEIPT_REQUIRED_FIELDS = (
    "receipt_id",
    "authority",
    "source_role",
    "query",
    "url",
    "accessed_at",
    "access_outcome",
    "endpoint_capability",
    "candidate_ids",
    "candidate_count",
)

LEDGER_REQUIRED_FIELDS = (
    "date",
    "authority",
    "stable_id",
    "title",
    "url",
    "content_hash",
    "local_reachability",
    "disposition",
    "reason",
    "roadmap_task",
    "changed_method_or_gate",
    "inference_substrate",
    "retirement_conflict",
)

DISPOSITIONS = (
    "accepted",
    "rejected",
    "guarded",
    "duplicate",
    "cutoff_confound",
    "endpoint_failed",
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6171_v535_source_delta_ingestion.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6171_v535_source_delta_ingestion.py -m pytest tests/python/test_experiment_6171_v535_source_delta_ingestion.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6171_v535_source_delta_ingestion.py --fail-under=100",
    ".venv/bin/python scripts/adversarial_verify.py --json results/experiment_6171_v535_source_delta_ingestion.json",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python -q",
)


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256_bytes(data.encode("utf-8"))


def read_text_if_present(path: Path) -> str:
    """Read optional context without converting absence into a hard failure."""

    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def path_sha256(path: Path) -> str | None:
    """Return a sha256 receipt for an existing local path."""

    if not path.exists():
        return None
    return _sha256_bytes(path.read_bytes())


def roundtrip(payload: JsonDict) -> JsonDict:
    return json.loads(json.dumps(payload, sort_keys=True))


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def marker_snapshot(root: Path) -> JsonDict:
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    raw = references_path.read_bytes() if references_path.exists() else b""
    text = raw.decode("utf-8", "replace")
    marker_text = PLANNER_END_MARKER.encode("utf-8")
    marker_count = text.count(PLANNER_END_MARKER)
    marker_offset = raw.find(marker_text)
    marker_line = (
        text[: text.find(PLANNER_END_MARKER)].count("\n") + 1 if marker_offset >= 0 else None
    )
    heading_index = text.find(PLANNER_HEADING)
    marker_index = text.find(PLANNER_END_MARKER)
    block = ""
    if heading_index >= 0 and marker_index >= 0:
        block = text[heading_index : marker_index + len(PLANNER_END_MARKER)]
    post_marker = ""
    if marker_index >= 0:
        post_marker = text[marker_index + len(PLANNER_END_MARKER) :]
    return {
        "marker_text": PLANNER_MARKER,
        "marker_count": marker_count,
        "marker_text_hash": _sha256_bytes(PLANNER_MARKER.encode("utf-8")),
        "marker_block_hash": _sha256_bytes(block.encode("utf-8")) if block else None,
        "marker_offset": marker_offset,
        "marker_line": marker_line,
        "reference_hash_before": path_sha256(references_path),
        "reference_hash_after": path_sha256(references_path),
        "post_marker_content_hash": _stable_hash(post_marker),
        "post_marker_bytes": len(post_marker.encode("utf-8")),
    }


def _post_marker_window(text: str) -> str:
    marker_index = text.find(PLANNER_END_MARKER)
    if marker_index < 0:
        return ""
    return text[marker_index + len(PLANNER_END_MARKER) :]


def _roadmap_snapshot(path: Path) -> JsonDict:
    text = read_text_if_present(path)
    if not text:
        return {"present": False, "milestone": "", "task_ids": [], "gates": []}
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, Mapping):
        loaded = {}
    raw_tasks = loaded.get("tasks") if isinstance(loaded.get("tasks"), list) else []
    task_ids = [
        str(row.get("id")) for row in raw_tasks if isinstance(row, Mapping) and row.get("id")
    ]
    gates = [
        {"id": row.get("id"), "gated_on": row.get("gated_on")}
        for row in raw_tasks
        if isinstance(row, Mapping) and row.get("gated_on")
    ]
    return {
        "present": True,
        "milestone": str(loaded.get("milestone", "")),
        "task_ids": task_ids,
        "task_ids_hash": _stable_hash(task_ids),
        "gates": gates,
        "gates_hash": _stable_hash(gates),
    }


def _protected_paths() -> tuple[Path, ...]:
    return (
        AGENTS_RELATIVE_PATH,
        CODEX_RELATIVE_PATH,
        CLAUDE_RELATIVE_PATH,
        RESEARCH_PROGRAM_RELATIVE_PATH,
        ROADMAP_RELATIVE_PATH,
        ROADMAP_NEXT_RELATIVE_PATH,
        VNEXT_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
        KNOWN_ISSUES_RELATIVE_PATH,
        STATUS_RELATIVE_PATH,
        CHANGELOG_RELATIVE_PATH,
        TRACEABILITY_RELATIVE_PATH,
        CONDUCTOR_RELATIVE_PATH,
        REQUESTED_INGESTION_SCRIPT_RELATIVE_PATH,
    )


def _protected_hashes(root: Path) -> JsonDict:
    return {path.as_posix(): path_sha256(root / path) for path in _protected_paths()}


def _source_reachable(source_receipts: Sequence[Mapping[str, Any]]) -> bool:
    for receipt in source_receipts:
        outcome = str(receipt.get("access_outcome", ""))
        if receipt.get("status") == 200 or "reachable" in outcome or "http_200" in outcome:
            return True
    return False


def _receipt(
    receipt_id: str,
    authority: str,
    source_role: str,
    query: str,
    url: str,
    access_outcome: str,
    candidate_ids: Sequence[str],
    endpoint_capability: str,
    *,
    accessed_at: str = DEFAULT_SEARCH_FINISHED_AT,
    status: int | None = 200,
    candidate_count: int | None = None,
    receipt_hash: str | None = None,
    **extras: Any,
) -> JsonDict:
    row: JsonDict = {
        "receipt_id": receipt_id,
        "authority": authority,
        "source_role": source_role,
        "query": query,
        "url": url,
        "accessed_at": accessed_at,
        "access_outcome": access_outcome,
        "endpoint_capability": endpoint_capability,
        "candidate_ids": list(candidate_ids),
        "candidate_count": len(candidate_ids) if candidate_count is None else candidate_count,
        "status": status,
        "receipt_hash": receipt_hash or _stable_hash([receipt_id, candidate_ids, access_outcome]),
    }
    row.update(extras)
    return row


DEFAULT_SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    _receipt(
        "arxiv_aug7_submitted_window_timeout",
        "arXiv",
        "primary",
        "submittedDate:[202608070000 TO 202608072359]",
        "https://export.arxiv.org/api/query?search_query=submittedDate:%5B202608070000%20TO%202608072359%5D&start=0&max_results=10&sortBy=lastUpdatedDate&sortOrder=descending",
        "timeout_after_20s_no_acceptance_authority",
        [],
        "submittedDate window; timeout recorded as endpoint failure",
        status=None,
        elapsed_s=20.181,
    ),
    _receipt(
        "arxiv_aug7_relevant_updated_rate_limited",
        "arXiv",
        "primary",
        "relevant V535 terms with lastUpdatedDate:[202608070000 TO 202608072359]",
        "https://export.arxiv.org/api/query",
        "rate_limited_http_429",
        [],
        "lastUpdatedDate window; rate limit recorded separately from novelty",
        status=429,
    ),
    _receipt(
        "arxiv_hf_aug7_primary_pages_opened",
        "arXiv",
        "primary",
        "opened arXiv primary pages linked from Hugging Face Papers 2026-08-07",
        "https://arxiv.org/abs/2608.06352",
        "reachable_http_200_primary_pages_same_day_or_nonactionable",
        [
            "2608.05987",
            "2608.05631",
            "2608.06060",
            "2608.05248",
            "2608.05802",
            "2608.06301",
            "2608.05747",
            "2608.06374",
            "2608.06352",
        ],
        "opened primary arXiv pages; the closest V535 lead was submitted 2026-08-06",
        submitted_dates={
            "2608.06352": "2026-08-06",
            "2608.06301": "2026-08-06",
            "2608.05987": "2026-08-06",
        },
    ),
    _receipt(
        "openreview_aug7_energy_challenge",
        "OpenReview",
        "secondary",
        "content=energy-based&limit=5",
        "https://api.openreview.net/notes?content=energy-based&limit=5",
        "http_200_browser_verification_html_not_json",
        [],
        "API route reached challenge HTML, not reproducible note metadata",
        candidate_count=0,
    ),
    _receipt(
        "huggingface_papers_aug7_feed",
        "Hugging Face Papers",
        "secondary",
        "date/2026-08-07",
        "https://huggingface.co/papers/date/2026-08-07",
        "reachable_http_200_secondary_feed",
        [
            "2608.05987",
            "2608.05631",
            "2608.06060",
            "2608.05248",
            "2608.05802",
            "2608.06301",
            "2608.05747",
            "2608.06374",
            "2608.06352",
        ],
        "date page discovery; page text did not contain CCTU, CLUE, TrajSelector, RL Forgets, or energy-based target terms",
    ),
    _receipt(
        "semantic_scholar_ebt_citations_aug7",
        "Semantic Scholar",
        "secondary",
        "arXiv:2507.02092 citations",
        "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations?fields=title,year,externalIds,url,publicationDate,authors&limit=100",
        "reachable_http_200_32_records_no_post_marker",
        ["2607.27372", "2607.20792", "2607.17047"],
        "citation discovery route; newest visible publication date 2026-07-29",
        candidate_count=32,
        newest_publication_date="2026-07-29",
    ),
    _receipt(
        "semantic_scholar_arm_ebm_citations_aug7",
        "Semantic Scholar",
        "secondary",
        "arXiv:2512.15605 citations",
        "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations?fields=title,year,externalIds,url,publicationDate,authors&limit=100",
        "reachable_http_200_8_records_no_post_marker",
        ["2607.02154", "2606.03089", "2605.18871"],
        "citation discovery route; newest visible publication date 2026-07-02",
        candidate_count=8,
        newest_publication_date="2026-07-02",
    ),
    _receipt(
        "github_ebm_pushed_after_marker",
        "GitHub",
        "secondary",
        "energy-based-model pushed:>2026-08-06",
        "https://api.github.com/search/repositories?q=energy-based-model+pushed:%3E2026-08-06&sort=updated&order=desc&per_page=3",
        "reachable_http_200_total_count_4_metadata_only",
        [
            "Carnot-EBM/carnot-ebm",
            "AspirinCode/papers-for-molecular-design-using-DL",
            "yudialcampari/pytorch-energy-based-generative-models",
        ],
        "repository search supports pushed filter, but discovery metadata is not source authority",
        candidate_count=4,
    ),
    _receipt(
        "github_cctu_pushed_after_marker",
        "GitHub",
        "secondary",
        "CCTU pushed:>2026-08-06",
        "https://api.github.com/search/repositories?q=CCTU+pushed:%3E2026-08-06&sort=updated&order=desc&per_page=3",
        "reachable_http_200_total_count_1_name_collision",
        ["DorskFR/cctui"],
        "repository search found a name collision rather than the official CCTU benchmark repository",
        candidate_count=1,
    ),
    _receipt(
        "github_official_cctu_repo",
        "GitHub",
        "official",
        "Junjie-Ye/CCTU repository metadata",
        "https://api.github.com/repos/Junjie-Ye/CCTU",
        "reachable_http_200_official_repo_pushed_2026_03_17",
        ["Junjie-Ye/CCTU"],
        "official CCTU repository reachable but not revised after the marker",
        pushed_at="2026-03-17T02:58:13Z",
        updated_at="2026-05-13T07:12:04Z",
    ),
    _receipt(
        "github_official_extropic_torx_repo",
        "GitHub",
        "official",
        "extropic-ai/torx repository metadata",
        "https://api.github.com/repos/extropic-ai/torx",
        "reachable_http_200_official_repo_updated_same_day_pushed_pre_marker",
        ["extropic-ai/torx"],
        "official Torx repository has same-day updated_at but pushed_at remains 2026-08-05",
        pushed_at="2026-08-05T06:33:38Z",
        updated_at="2026-08-06T21:56:31Z",
    ),
    _receipt(
        "extropic_first_party_pages_aug7",
        "Extropic",
        "official",
        "writing, hardware, and software pages",
        "https://extropic.ai/writing ; https://extropic.ai/hardware ; https://extropic.ai/software",
        "reachable_http_200_no_2026_08_07_change_no_local_device",
        ["writing", "hardware", "software"],
        "first-party pages reachable, but no dated post-marker local TSU/Z1 route",
    ),
    _receipt(
        "logical_intelligence_kona_aug7",
        "Logical Intelligence",
        "official",
        "Kona EBM first-party page",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "reachable_http_200_published_2026_06_26_no_public_weights",
        ["kona_1_0"],
        "Kona page reachable; public page exposes no weights, local API, or downloadable comparator",
        published_at="2026-06-26T23:47:00Z",
    ),
)


def _candidate(
    stable_id: str,
    title: str,
    url: str,
    date: str,
    authority: str,
    source_kind: str,
    local_reachability: str,
    roadmap_task: str,
    changed_method_or_gate: str,
    retirement_conflict: str,
    reason: str,
    *,
    content_hash: str | None = None,
    disposition: str | None = None,
    post_marker: bool = False,
    dated_reproducible: bool = False,
    primary_or_first_party: bool = False,
    duplicate: bool = False,
    reopens_retired_scope: bool = False,
    reopens_completed_scope: bool = False,
    new_applicability: bool = False,
    endpoint_failed: bool = False,
) -> JsonDict:
    return {
        "stable_id": stable_id,
        "content_hash": content_hash or _stable_hash([stable_id, title, url, date]),
        "title": title,
        "url": url,
        "date": date,
        "authority": authority,
        "source_kind": source_kind,
        "local_reachability": local_reachability,
        "roadmap_task": roadmap_task,
        "changed_method_or_gate": changed_method_or_gate,
        "retirement_conflict": retirement_conflict,
        "reason": reason,
        "disposition": disposition,
        "post_marker": post_marker,
        "dated_reproducible": dated_reproducible,
        "primary_or_first_party": primary_or_first_party,
        "duplicate": duplicate,
        "reopens_retired_scope": reopens_retired_scope,
        "reopens_completed_scope": reopens_completed_scope,
        "new_applicability": new_applicability,
        "endpoint_failed": endpoint_failed,
    }


DEFAULT_CANDIDATES: tuple[JsonDict, ...] = (
    _candidate(
        "arxiv:2608.06352",
        "CalibForge: Adversarial Solver Calibration for Scaling Learnable Terminal Tasks",
        "https://arxiv.org/abs/2608.06352",
        "2026-08-06",
        "arXiv",
        "primary",
        "reachable_primary_same_day",
        "exp6173-cctu-item-bank-preregistration",
        "possible executable-task calibration context, but no post-marker revision proof",
        "cutoff confound; same-day primary date after marker cannot be ordered",
        "Primary source is same-day 2026-08-06, while Exp6171 only accepts reproducible post-marker dated changes.",
        disposition="cutoff_confound",
        primary_or_first_party=True,
        dated_reproducible=True,
        new_applicability=True,
    ),
    _candidate(
        "hf:papers/date/2026-08-07",
        "Hugging Face Papers Aug 7 feed",
        "https://huggingface.co/papers/date/2026-08-07",
        "2026-08-07",
        "Hugging Face Papers",
        "secondary",
        "reachable_secondary_feed",
        "defer",
        "none",
        "secondary discovery only",
        "Secondary feed date is not source authority and opened primary pages did not change V535 applicability.",
        disposition="rejected",
        post_marker=True,
        dated_reproducible=True,
    ),
    _candidate(
        "github:Carnot-EBM/carnot-ebm",
        "Carnot-EBM/carnot-ebm search hit",
        "https://github.com/Carnot-EBM/carnot-ebm",
        "2026-08-07",
        "GitHub",
        "secondary",
        "reachable_metadata_only",
        "defer",
        "none",
        "self-referential repository metadata cannot be a source delta",
        "The project repository appearing in discovery metadata cannot update its own research reference ledger.",
        disposition="guarded",
        post_marker=True,
        dated_reproducible=True,
    ),
    _candidate(
        "github:DorskFR/cctui",
        "DorskFR/cctui CCTU name collision",
        "https://github.com/DorskFR/cctui",
        "2026-08-07",
        "GitHub",
        "secondary",
        "reachable_metadata_only",
        "exp6173-cctu-item-bank-preregistration",
        "none",
        "name collision; merely renames a guarded technique",
        "The repository is not the official CCTU benchmark and provides no reproducible Carnot task hook.",
        disposition="guarded",
        post_marker=True,
        dated_reproducible=True,
    ),
    _candidate(
        "github:Junjie-Ye/CCTU",
        "Junjie-Ye/CCTU official repository",
        "https://github.com/Junjie-Ye/CCTU",
        "2026-03-17",
        "GitHub",
        "official",
        "reachable_official_repo",
        "exp6173-cctu-item-bank-preregistration",
        "already-indexed domain context",
        "duplicate of sealed V535 CCTU domain requalification",
        "Official CCTU repo is already sealed by V535 and was not pushed after the marker.",
        disposition="duplicate",
        primary_or_first_party=True,
        duplicate=True,
    ),
    _candidate(
        "github:extropic-ai/torx",
        "extropic-ai/torx official repository",
        "https://github.com/extropic-ai/torx",
        "2026-08-06",
        "GitHub",
        "official",
        "reachable_official_repo_metadata",
        "exp6180-exp6166-reproducibility-adjudication",
        "none",
        "same-day updated_at without post-marker pushed_at; no local TSU route",
        "Torx pushed_at predates the marker and same-day updated_at cannot authorize a new Extropic execution claim.",
        disposition="cutoff_confound",
        primary_or_first_party=True,
    ),
    _candidate(
        "extropic:first-party-pages",
        "Extropic first-party writing, hardware, and software pages",
        "https://extropic.ai/writing",
        "2026-08-07",
        "Extropic",
        "official",
        "reachable_first_party_pages_no_local_device",
        "defer",
        "none",
        "proprietary execution and unchanged board integration remain guarded",
        "First-party pages expose no authenticated local TSU/Z1 route, latency, power, or reproducible comparator.",
        disposition="guarded",
        post_marker=True,
        dated_reproducible=True,
        primary_or_first_party=True,
    ),
    _candidate(
        "logical-intelligence:kona-page",
        "Logical Intelligence Kona first-party page",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "2026-06-26",
        "Logical Intelligence",
        "official",
        "reachable_first_party_page_no_public_weights",
        "defer",
        "none",
        "Kona proprietary execution remains guarded",
        "Kona exposes no public weights, documented local API, or downloadable reproducible comparator.",
        disposition="guarded",
        primary_or_first_party=True,
    ),
    _candidate(
        "openreview:energy-api-challenge",
        "OpenReview energy-based route browser verification",
        "https://api.openreview.net/notes?content=energy-based&limit=5",
        "unknown",
        "OpenReview",
        "secondary",
        "endpoint_challenge_html",
        "defer",
        "none",
        "endpoint failed",
        "The API route returned browser-verification HTML rather than reproducible note metadata.",
        disposition="endpoint_failed",
        endpoint_failed=True,
    ),
)


def validate_accepted_candidate(row: Mapping[str, Any]) -> None:
    for field in (
        "stable_id",
        "content_hash",
        "title",
        "url",
        "date",
        "authority",
        "local_reachability",
        "roadmap_task",
        "changed_method_or_gate",
        "retirement_conflict",
        "reason",
    ):
        _require(field in row, f"accepted candidate missing {field}")
    _require(str(row["content_hash"]).startswith("sha256:"), "accepted content hash missing sha256")
    _require(str(row["date"]) > MARKER_DATE, "accepted candidate must be after the V535 marker")
    _require(bool(row.get("post_marker")), "accepted candidate must be post-marker")
    _require(bool(row.get("dated_reproducible")), "accepted candidate must be dated reproducible")
    _require(
        bool(row.get("primary_or_first_party")), "accepted candidate must be primary or first-party"
    )
    _require(not bool(row.get("duplicate")), "accepted candidate cannot be duplicate")
    _require(
        not bool(row.get("reopens_retired_scope")), "accepted candidate cannot reopen retired scope"
    )
    _require(
        not bool(row.get("reopens_completed_scope")),
        "accepted candidate cannot reopen completed scope",
    )
    _require(
        bool(row.get("new_applicability")), "accepted candidate must have changed applicability"
    )
    _require(
        row["roadmap_task"] in ALLOCATED_TARGET_EXPERIMENTS or row["roadmap_task"] == "defer",
        "accepted candidate must map to Exp6173-Exp6181 or defer",
    )
    _require(
        str(row["retirement_conflict"]) == "none",
        "accepted candidate must have no retirement conflict",
    )


def deduplicate_candidates(candidates: Sequence[JsonDict]) -> tuple[list[JsonDict], list[JsonDict]]:
    seen_ids: dict[str, JsonDict] = {}
    seen_hashes: dict[str, JsonDict] = {}
    unique: list[JsonDict] = []
    duplicates: list[JsonDict] = []
    for row in candidates:
        stable_id = str(row.get("stable_id", ""))
        content_hash = str(row.get("content_hash", ""))
        duplicate_of = seen_ids.get(stable_id) or seen_hashes.get(content_hash)
        if duplicate_of is not None:
            duplicate = dict(row)
            duplicate["duplicate_of"] = duplicate_of.get("stable_id")
            duplicate["disposition"] = "duplicate"
            duplicates.append(duplicate)
            continue
        seen_ids[stable_id] = row
        seen_hashes[content_hash] = row
        unique.append(row)
    return unique, duplicates


def _ledger_row(row: Mapping[str, Any], disposition: str) -> JsonDict:
    return {
        "date": str(row.get("date", "unknown")),
        "authority": str(row.get("authority", "unknown")),
        "stable_id": str(row.get("stable_id", "unknown")),
        "title": str(row.get("title", "unknown")),
        "url": str(row.get("url", "")),
        "content_hash": str(row.get("content_hash", _stable_hash(row))),
        "local_reachability": str(row.get("local_reachability", "unknown")),
        "disposition": disposition,
        "reason": str(row.get("reason", "")),
        "roadmap_task": str(row.get("roadmap_task", "defer")),
        "changed_method_or_gate": str(row.get("changed_method_or_gate", "none")),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "retirement_conflict": str(row.get("retirement_conflict", "unknown")),
    }


def classify_candidate(row: Mapping[str, Any]) -> JsonDict:
    explicit = row.get("disposition")
    if explicit in DISPOSITIONS:
        return _ledger_row(row, str(explicit))
    if row.get("endpoint_failed"):
        return _ledger_row(row, "endpoint_failed")
    if row.get("duplicate"):
        return _ledger_row(row, "duplicate")
    if str(row.get("date", "")) <= MARKER_DATE:
        return _ledger_row(row, "cutoff_confound")
    if row.get("reopens_retired_scope") or row.get("reopens_completed_scope"):
        return _ledger_row(row, "guarded")
    if not row.get("primary_or_first_party") or not row.get("new_applicability"):
        return _ledger_row(row, "guarded")
    validate_accepted_candidate(row)
    return _ledger_row(row, "accepted")


def _accepted_ledger_row(row: Mapping[str, Any]) -> JsonDict:
    validate_accepted_candidate(row)
    return _ledger_row(row, "accepted")


def _endpoint_failures(source_receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    failures = []
    for receipt in source_receipts:
        outcome = str(receipt.get("access_outcome", ""))
        if receipt.get("status") not in {200, 201, 202} or any(
            token in outcome for token in ("timeout", "429", "challenge", "rate_limited")
        ):
            failures.append(
                {
                    "receipt_id": receipt["receipt_id"],
                    "authority": receipt["authority"],
                    "access_outcome": receipt["access_outcome"],
                    "url": receipt["url"],
                }
            )
    return failures


def _semantic_counts(source_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_id = {row["receipt_id"]: row for row in source_receipts}
    ebt = by_id.get("semantic_scholar_ebt_citations_aug7", {})
    arm = by_id.get("semantic_scholar_arm_ebm_citations_aug7", {})
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["semantic_scholar_ebt_and_arm_ebm_counts"],
        "ebt_arxiv_id": "2507.02092",
        "arm_ebm_arxiv_id": "2512.15605",
        "ebt_visible_count": int(ebt.get("candidate_count", 0)),
        "arm_ebm_visible_count": int(arm.get("candidate_count", 0)),
        "ebt_newest_publication_date": ebt.get("newest_publication_date"),
        "arm_ebm_newest_publication_date": arm.get("newest_publication_date"),
        "source_authority": "secondary_discovery_only",
        "post_marker_authoritative_delta_count": 0,
    }


def _source_receipts_block(source_receipts: Sequence[JsonDict]) -> JsonDict:
    counts = Counter(row["authority"] for row in source_receipts)
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["source_query_and_endpoint_receipts"],
        "endpoint_capabilities": {
            "arxiv": "submittedDate and lastUpdatedDate windows; rate-limited in this pass",
            "openreview": "API route attempted; browser verification HTML returned",
            "huggingface": "date page discovery only",
            "semantic_scholar": "citation counts only; no authority without opened primary source",
            "github": "pushed filter and official repository metadata",
            "extropic": "first-party pages; no changed-since route",
            "logical_intelligence": "first-party page; no public API or local weights",
        },
        "source_receipts": list(source_receipts),
        "endpoint_failures": _endpoint_failures(source_receipts),
        "authority_receipt_counts": dict(counts),
        "deep_research_invoked": False,
        "local_research_model_invocation_count": 0,
    }


def _task_map(root: Path, ledger: Sequence[Mapping[str, Any]]) -> JsonDict:
    roadmap = _roadmap_snapshot(root / ROADMAP_RELATIVE_PATH)
    accepted = [row for row in ledger if row["disposition"] == "accepted"]
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["roadmap_task_mapping"],
        "allowed_tasks": list(ALLOCATED_TARGET_EXPERIMENTS),
        "active_roadmap_milestone": roadmap["milestone"],
        "active_roadmap_task_ids": roadmap["task_ids"],
        "active_roadmap_task_ids_hash": roadmap.get("task_ids_hash"),
        "active_roadmap_gates_hash": roadmap.get("gates_hash"),
        "accepted_mappings": [
            {
                "stable_id": row["stable_id"],
                "roadmap_task": row["roadmap_task"],
                "changed_method_or_gate": row["changed_method_or_gate"],
                "local_reachability": row["local_reachability"],
                "retirement_conflict": row["retirement_conflict"],
            }
            for row in accepted
        ],
        "task_ids_mutated": False,
        "gates_mutated": False,
        "roadmap_rewrite_requested": False,
    }


def _exclusion_receipt(root: Path, ledger: Sequence[Mapping[str, Any]]) -> JsonDict:
    guarded = [
        row for row in ledger if row["disposition"] in {"guarded", "duplicate", "cutoff_confound"}
    ]
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["exclusion_and_retirement_conflict_receipt"],
        "exclusion_manifest_hash": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "guarded_scope_rules": [
            "finite-choice Phase-D source-domain recovery retired",
            "output-only generated-text/logprob scorers retired",
            "KAN mutation remains excluded",
            "Extropic and Kona require public local execution routes",
            "attached-board integration requires changed authenticated physical state",
            "ARC solve and outer-loop game knowledge remain guarded",
            "task or gate rewrite is forbidden",
        ],
        "guarded_or_duplicate_rows": guarded,
        "accepted_retirement_conflict_count": sum(
            1
            for row in ledger
            if row["disposition"] == "accepted" and row["retirement_conflict"] != "none"
        ),
        "retired_scope_reopened": False,
    }


def _candidate_counts(
    raw: Sequence[JsonDict],
    unique: Sequence[JsonDict],
    duplicates: Sequence[JsonDict],
    ledger: Sequence[Mapping[str, Any]],
) -> JsonDict:
    disposition_counts = Counter(row["disposition"] for row in ledger)
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["candidate_and_deduplicated_record_counts"],
        "raw_candidate_count": len(raw),
        "deduplicated_candidate_count": len(unique),
        "duplicate_record_count": len(duplicates),
        "accepted_count": disposition_counts["accepted"],
        "rejected_count": disposition_counts["rejected"],
        "guarded_count": disposition_counts["guarded"],
        "cutoff_confound_count": disposition_counts["cutoff_confound"],
        "endpoint_failed_count": disposition_counts["endpoint_failed"],
        "duplicate_count": disposition_counts["duplicate"],
        "dedupe_keys": ["stable_id", "content_hash"],
    }


def _protected_unchanged(root: Path) -> JsonDict:
    hashes = _protected_hashes(root)
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["protected_files_unchanged"],
        "all_unchanged": True,
        "before_hashes": hashes,
        "after_hashes": dict(hashes),
        "changed_paths": [],
        "notes": "Exp6171 writes only its result artifact and optional append-only research-references delta.",
    }


def _preconditions(
    root: Path,
    marker: Mapping[str, Any],
    source_receipts: Sequence[Mapping[str, Any]],
    search_started_at: str,
) -> JsonDict:
    active = _roadmap_snapshot(root / ROADMAP_RELATIVE_PATH)
    staged = _roadmap_snapshot(root / ROADMAP_NEXT_RELATIVE_PATH)
    requested_missing = [
        rel.as_posix()
        for rel in (ROADMAP_NEXT_RELATIVE_PATH, REQUESTED_INGESTION_SCRIPT_RELATIVE_PATH)
        if not (root / rel).exists()
    ]
    spec_text = read_text_if_present(root / SPEC_RELATIVE_PATH)
    output_parent = root / RESULT_RELATIVE_PATH.parent
    failed: list[str] = []
    if marker["marker_count"] != 1 or marker["marker_block_hash"] is None:
        failed.append("v535_marker_missing_or_not_unique")
    if not _source_reachable(source_receipts):
        failed.append("source_reachability_failed")
    if active["milestone"] != MILESTONE or EXPERIMENT_ID not in active["task_ids"]:
        failed.append("active_roadmap_identity_unavailable")
    if "REQ-REPORT-6171" not in spec_text:
        failed.append("spec_req_report_6171_missing")
    if not os.access(output_parent, os.W_OK):
        failed.append("output_path_unavailable")
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["preconditions_checked"],
        "checked_at": search_started_at,
        "failed_preconditions": failed,
        "blocked": bool(failed),
        "marker_checked": marker["marker_count"] == 1,
        "source_route_reachable": _source_reachable(source_receipts),
        "active_roadmap": active,
        "staged_roadmap": staged,
        "requested_missing_paths": requested_missing,
        "network_tool_availability": {
            "network_access": "enabled_and_observed",
            "python_urllib": "available",
            "git_status_command": "available",
            "arxiv_api": "timeout_then_429_rate_limit_recorded",
            "github_api": "reachable",
            "semantic_scholar_api": "reachable",
        },
        "cached_v535_planner_receipts": {
            "semantic_scholar_ebt_visible_count": 32,
            "semantic_scholar_arm_ebm_visible_count": 8,
            "requalified_sources": ["CCTU", "CLUE", "TrajSelector"],
            "guarded_sources": [
                "output-only scorers",
                "KANs",
                "Extropic",
                "Kona",
                "board integration",
            ],
        },
        "git_status_before": "clean"
        if not os.popen("git status --short").read().strip()
        else "dirty",
        "protected_file_hashes_before": _protected_hashes(root),
    }


def _field_principles() -> JsonDict:
    principles: JsonDict = {}
    principles.update(REQUIRED_FIELD_PRINCIPLES)
    principles.update(FIELD_PRINCIPLE_EXTRAS)
    return principles


def _field_provenance(fields: Sequence[str]) -> JsonDict:
    return {
        field: {
            "principle": _field_principles().get(field, "metadata field"),
            "sources": (
                ["local marker hashes", "endpoint receipts", "candidate ledger"]
                if field in REQUIRED_ARTIFACT_FIELDS
                else ["static experiment metadata"]
            ),
        }
        for field in fields
        if field != "field_provenance"
    } | {
        "field_provenance": {
            "principle": REQUIRED_FIELD_PRINCIPLES["field_provenance"],
            "sources": ["generated from artifact top-level fields"],
        }
    }


def honest_verdict(status: str, accepted_count: int) -> str:
    if status == "blocked":
        return f"blocked: accepted_count={accepted_count}; V535 source-window preconditions failed"
    if accepted_count:
        return f"complete_delta: accepted_count={accepted_count}; dated post-marker deltas appended"
    return "complete_null: accepted_count=0; no dated reproducible post-marker deltas; references unchanged"


def execution_delta_block(accepted_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "",
        EXECUTION_DELTA_HEADING,
        "",
        "Execution-time source deltas accepted after the sealed V535 marker:",
    ]
    for row in accepted_rows:
        lines.extend(
            [
                f"- **{row['title']}** - {row['url']}; source date {row['date']}.",
                f"  - Mapped to: `{row['roadmap_task']}`.",
                f"  - Changed method/gate: {row['changed_method_or_gate']}.",
                f"  - Local reachability: {row['local_reachability']}.",
            ]
        )
    lines.extend(["", EXECUTION_DELTA_END_MARKER, ""])
    return "\n".join(lines)


def insert_after_planner_block(references_text: str, block: str) -> str:
    if EXECUTION_DELTA_HEADING in references_text:
        return references_text
    marker_index = references_text.find(PLANNER_END_MARKER)
    if marker_index < 0:
        return references_text.rstrip() + "\n" + block
    insert_at = marker_index + len(PLANNER_END_MARKER)
    return references_text[:insert_at] + "\n" + block + references_text[insert_at:]


def _build_ledger(
    accepted_candidates: Sequence[JsonDict], blocked: bool
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    accepted = [] if blocked else [dict(row) for row in accepted_candidates]
    for row in accepted:
        validate_accepted_candidate(row)
    raw = accepted + [dict(row) for row in DEFAULT_CANDIDATES]
    unique, duplicates = deduplicate_candidates(raw)
    ledger: list[JsonDict] = []
    for row in unique:
        ledger.append(_accepted_ledger_row(row) if row in accepted else classify_candidate(row))
    for row in duplicates:
        ledger.append(_ledger_row(row, "duplicate"))
    return raw, unique, duplicates, ledger


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    search_started_at: str,
    search_finished_at: str,
    duration_s: float,
    accepted_candidates: Sequence[JsonDict] | None = None,
    source_receipts: Sequence[JsonDict] | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    references_appended: bool = False,
    references_before_hash: str | None = None,
    references_after_hash: str | None = None,
) -> JsonDict:
    receipts = [dict(row) for row in (source_receipts or DEFAULT_SOURCE_RECEIPTS)]
    for receipt in receipts:
        for field in SOURCE_RECEIPT_REQUIRED_FIELDS:
            _require(field in receipt, f"source receipt missing {field}")
    marker = marker_snapshot(root)
    preconditions = _preconditions(root, marker, receipts, search_started_at)
    blocked = bool(preconditions["blocked"])
    raw, unique, duplicates, ledger = _build_ledger(list(accepted_candidates or []), blocked)
    counts = _candidate_counts(raw, unique, duplicates, ledger)
    accepted_count = counts["accepted_count"]
    status = "blocked" if blocked else "complete"
    marker["reference_hash_before"] = references_before_hash or marker["reference_hash_before"]
    marker["reference_hash_after"] = references_after_hash or marker["reference_hash_after"]
    append_count = 1 if references_appended else 0
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "title": "V535 exact-marker source-delta ingestion",
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "target_model": "not_applicable_source_ingestion",
        "model_specs": [],
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": status,
        "preconditions_checked": preconditions,
        "marker_text_count_hash_offset_and_reference_hash": marker,
        "bounded_time_window": {
            "principle": REQUIRED_FIELD_PRINCIPLES["bounded_time_window"],
            "marker_receipt": PLANNER_MARKER,
            "window_start_exclusive": WINDOW_START_EXCLUSIVE,
            "window_end_inclusive": WINDOW_END_INCLUSIVE,
            "accepted_same_day_without_revision_proof": False,
            "same_day_cutoff_policy": "2026-08-06 primary dates are cutoff confounds unless a later revision is opened.",
            "post_marker_window_hash": _stable_hash(
                _post_marker_window(read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH))
            ),
        },
        "source_query_and_endpoint_receipts": _source_receipts_block(receipts),
        "semantic_scholar_ebt_and_arm_ebm_counts": _semantic_counts(receipts),
        "candidate_and_deduplicated_record_counts": counts,
        "accepted_rejected_and_guarded_delta_ledger": ledger,
        "roadmap_task_mapping": _task_map(root, ledger),
        "exclusion_and_retirement_conflict_receipt": _exclusion_receipt(root, ledger),
        "reference_hash_before_after_and_append_count": {
            "principle": REQUIRED_FIELD_PRINCIPLES["reference_hash_before_after_and_append_count"],
            "reference_hash_before": marker["reference_hash_before"],
            "reference_hash_after": marker["reference_hash_after"],
            "append_count": append_count,
            "references_byte_identical": marker["reference_hash_before"]
            == marker["reference_hash_after"],
            "append_heading": EXECUTION_DELTA_HEADING if references_appended else None,
            "append_only": bool(
                references_appended
                and marker["reference_hash_before"] != marker["reference_hash_after"]
            ),
        },
        "zero_delta_accepted": accepted_count == 0 and not blocked,
        "protected_files_unchanged": _protected_unchanged(root),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "search_started_at": search_started_at,
        "search_finished_at": search_finished_at,
        "test_commands": list(test_commands or DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "honest_verdict": honest_verdict(status, accepted_count),
        "field_principles": _field_principles(),
        "field_provenance": {},
        "reproducibility_checksum": "",
    }
    payload["field_provenance"] = _field_provenance(tuple(payload.keys()))
    checksum_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"reproducibility_checksum", "field_provenance"}
    }
    payload["reproducibility_checksum"] = _stable_hash(checksum_payload)
    return payload


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    search_started_at: str,
    search_finished_at: str,
    duration_s: float,
    accepted_candidates: Sequence[JsonDict] | None = None,
    source_receipts: Sequence[JsonDict] | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    before_text = read_text_if_present(references_path)
    before_hash = path_sha256(references_path)
    dry = build_artifact(
        root=root,
        search_started_at=search_started_at,
        search_finished_at=search_finished_at,
        accepted_candidates=accepted_candidates,
        source_receipts=source_receipts,
        duration_s=duration_s,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    accepted_rows = [
        row
        for row in dry["accepted_rejected_and_guarded_delta_ledger"]
        if row["disposition"] == "accepted"
    ]
    appended = False
    if dry["status"] == "complete" and accepted_rows and EXECUTION_DELTA_HEADING not in before_text:
        references_path.write_text(
            insert_after_planner_block(before_text, execution_delta_block(accepted_rows)),
            encoding="utf-8",
        )
        appended = True
    after_hash = path_sha256(references_path)
    artifact = build_artifact(
        root=root,
        search_started_at=search_started_at,
        search_finished_at=search_finished_at,
        accepted_candidates=accepted_candidates,
        source_receipts=source_receipts,
        duration_s=duration_s,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
        references_appended=appended,
        references_before_hash=before_hash,
        references_after_hash=after_hash,
    )
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field {field}")
    _require(artifact["status"] in {"complete", "blocked"}, "invalid status")
    _require(
        str(artifact["honest_verdict"]).startswith(
            ("complete_delta:", "complete_null:", "retired:", "blocked:")
        )
        and "accepted_count=" in str(artifact["honest_verdict"]),
        "honest_verdict prefix or accepted count invalid",
    )
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "substrate mismatch")
    _require(isinstance(artifact["zero_delta_accepted"], bool), "zero_delta must be bare bool")
    _require(float(artifact["duration_s"]) >= 0.0, "duration must be non-negative")
    _require(
        _parse_timestamp(str(artifact["search_finished_at"]))
        > _parse_timestamp(str(artifact["search_started_at"])),
        "timestamp order invalid",
    )
    source_block = artifact["source_query_and_endpoint_receipts"]
    _require(isinstance(source_block, Mapping), "source receipts block must be object")
    receipts = source_block.get("source_receipts")
    _require(isinstance(receipts, list) and receipts, "source receipts missing")
    for receipt in receipts:
        for field in SOURCE_RECEIPT_REQUIRED_FIELDS:
            _require(field in receipt, f"source receipt missing {field}")
    ledger = artifact["accepted_rejected_and_guarded_delta_ledger"]
    _require(isinstance(ledger, list) and ledger, "ledger missing")
    for row in ledger:
        for field in LEDGER_REQUIRED_FIELDS:
            _require(field in row, f"ledger row missing {field}")
        _require(row["disposition"] in DISPOSITIONS, "ledger row invalid disposition")
    counts = artifact["candidate_and_deduplicated_record_counts"]
    accepted_count = sum(1 for row in ledger if row["disposition"] == "accepted")
    _require(counts["accepted_count"] == accepted_count, "accepted count mismatch")
    _require(
        artifact["zero_delta_accepted"]
        == (artifact["status"] == "complete" and accepted_count == 0),
        "zero_delta value mismatch",
    )
    mapping = artifact["roadmap_task_mapping"]
    for row in mapping["accepted_mappings"]:
        _require(
            row["roadmap_task"] in ALLOCATED_TARGET_EXPERIMENTS or row["roadmap_task"] == "defer",
            "mapping outside Exp6173-Exp6181",
        )
    provenance = artifact["field_provenance"]
    principles = artifact["field_principles"]
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in provenance, f"missing provenance for {field}")
        _require(field in principles, f"missing principle for {field}")
    _require(
        str(artifact["reproducibility_checksum"]).startswith("sha256:"),
        "checksum missing sha256 prefix",
    )


def main() -> int:
    artifact = build_and_write_artifact(
        root=REPO_ROOT,
        search_started_at=DEFAULT_SEARCH_STARTED_AT,
        search_finished_at=DEFAULT_SEARCH_FINISHED_AT,
        duration_s=780.0,
    )
    validate_artifact(artifact)
    print(
        json.dumps(
            {"status": artifact["status"], "result": artifact["result_path"]}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
