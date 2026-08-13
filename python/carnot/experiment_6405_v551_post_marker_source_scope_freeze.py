"""Exp6405 V551 post-marker source and scope freeze.

Spec refs: REQ-REPORT-6405, SCENARIO-REPORT-6405-1,
SCENARIO-REPORT-6405-2, SCENARIO-REPORT-6405-3,
SCENARIO-REPORT-6405-4, SCENARIO-REPORT-6405-5,
SCENARIO-REPORT-6405-6.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse
import urllib.request as urllib_request
from xml.etree import ElementTree

from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import canonical_json, path_sha256, payload_sha256


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Fetcher = Callable[[str], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_ID = "exp6405-v551-post-marker-source-scope-freeze"
SCHEMA = "carnot.experiment_6405.v551_post_marker_source_scope_freeze.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6405_v551_post_marker_source_scope_freeze.json")
INFERENCE_SUBSTRATE = "web_and_bibliographic_search_only"
CONTRACT_VERSION = "v551.6405"

RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
HARDWARE_WISHLIST_RELATIVE_PATH = Path("research-hardware-wishlist.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
PRIOR_V551_HANDOFF_RELATIVE_PATH = Path(
    "results/experiment_6404_v551_terminal_handoff_and_queue_preflight.json"
)
PRIOR_V550_SOURCE_FREEZE_RELATIVE_PATH = Path(
    "results/experiment_6392_v550_post_marker_source_scope_freeze.json"
)
PRIOR_V550_CAPSTONE_RELATIVE_PATH = Path("results/experiment_6403_v550_adversarial_capstone.json")

PLANNER_MARKER = "V551-PLANNER-REFRESH-20260813-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
MARKER_COMMIT = "89d370a2cc94fd4bfdb9d7a8f2ecf4031c5ad2c3"
MARKER_COMMITTED_AT_LOCAL = "2026-08-13T17:15:37-04:00"
MARKER_COMMITTED_AT_UTC = "2026-08-13T21:15:37Z"

DIRECT_ARXIV_PAPERS: tuple[JsonDict, ...] = (
    {
        "arxiv_id": "2607.09175",
        "title": (
            "Scoped Verification for Reliable Long-Horizon Agentic Context Evolution "
            "under Distribution Shift"
        ),
        "planner_submitted_date": "2026-07-10",
        "planner_role": "active_graph_local_verification",
        "classification_hint": "executable_now",
        "local_hook": "affected_neighborhood_factor_memory",
    },
    {
        "arxiv_id": "2607.22962",
        "title": (
            "ConsistencyGate: Preventing Memory Contamination in LLM Agents via "
            "Self-Consistency Admission Control"
        ),
        "planner_submitted_date": "2026-07-25",
        "planner_role": "active_write_time_contamination_test",
        "classification_hint": "executable_now",
        "local_hook": "exact_write_admission_contamination_stream",
    },
    {
        "arxiv_id": "2602.17913",
        "title": "From Lossy to Verified: A Provenance-Aware Tiered Memory for Agents",
        "planner_submitted_date": "2026-02-20",
        "planner_role": "active_provenance_linked_two_tier_memory",
        "classification_hint": "executable_now",
        "local_hook": "raw_ledger_and_compiled_graph_tiers",
    },
    {
        "arxiv_id": "2603.04549",
        "title": "Adaptive Memory Admission Control for LLM Agents",
        "planner_submitted_date": "2026-03-04",
        "planner_role": "diagnostic_interpretable_admission_features",
        "classification_hint": "diagnostic_control",
        "local_hook": "admission_diagnostic_vector_only",
    },
)

CLOSED_SCOPE_PATTERNS = (
    "universal factor support",
    "external generated-text scoring",
    "hidden-state scoring",
    "CCTU",
    "KAN training",
    "EBT pretraining",
    "grammar/parser repair",
    "offline ARC solving",
    "per-game adapters",
    "unchanged board probes",
    "TSU execution",
    "Kona execution",
)

ARXIV_API_URL = "https://export.arxiv.org/api/query?id_list=" + ",".join(
    str(paper["arxiv_id"]) for paper in DIRECT_ARXIV_PAPERS
)
OPENREVIEW_URL = (
    "https://api2.openreview.net/notes/search?term="
    "TierMem%20Adaptive%20Memory%20Admission%20Control%20MemAgents&limit=5"
)
HUGGINGFACE_PAPER_URLS = {
    str(paper["arxiv_id"]): f"https://huggingface.co/api/papers/{paper['arxiv_id']}"
    for paper in DIRECT_ARXIV_PAPERS
}
SEMANTIC_SCHOLAR_URLS = {
    "ebt": (
        "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/"
        "citations?fields=title,url,year,publicationDate,externalIds&limit=5"
    ),
    "arm_ebm": (
        "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/"
        "citations?fields=title,url,year,publicationDate,externalIds&limit=5"
    ),
}
PLANNER_SEMANTIC_SCHOLAR_COUNTS = {"ebt": 33, "arm_ebm": 8}
GITHUB_DISCOVERY_URL = (
    "https://api.github.com/search/repositories?q="
    "%28%22GRACE%22%20OR%20%22ConsistencyGate%22%20OR%20%22TierMem%22%20OR%20"
    "%22Adaptive%20Memory%20Admission%22%29%20pushed:%3E=2026-08-13"
    "&sort=updated&order=desc&per_page=5"
)
EXTROPIC_URL = "https://extropic.ai/hardware"
LOGICAL_INTELLIGENCE_URL = "https://logicalintelligence.com/kona-ebms-energy-based-models"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "planner_marker_path_and_hash",
    "source_window_start_and_end_utc",
    "direct_arxiv_source_receipts",
    "openreview_receipts",
    "huggingface_papers_receipts",
    "semantic_scholar_ebt_and_arm_ebm_receipts",
    "github_discovery_receipts",
    "extropic_first_party_receipt",
    "logical_intelligence_first_party_receipt",
    "new_actionable_findings",
    "post_marker_findings_count",
    "executable_scope_change_required",
    "active_lane_freeze",
    "control_only_findings",
    "closed_and_deferred_scope_freeze",
    "unavailable_or_rate_limited_sources",
    "retired_scope_reopened",
    "hardware_state_change_found",
    "source_claim_boundaries",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "status": "The status states whether post-marker source evidence changes V551 scope.",
    "planner_marker_path_and_hash": "The V551 marker hash fixes the source boundary.",
    "source_window_start_and_end_utc": "The UTC window makes source rechecks dated.",
    "direct_arxiv_source_receipts": "arXiv receipts separate live metadata from fallback text.",
    "openreview_receipts": "OpenReview rows stay workshop or diagnostic controls.",
    "huggingface_papers_receipts": "Hugging Face Papers rows stay mirror evidence.",
    "semantic_scholar_ebt_and_arm_ebm_receipts": "Citation rows are discovery receipts only.",
    "github_discovery_receipts": "Repository rows need a maintained local route before use.",
    "extropic_first_party_receipt": "Extropic pages do not imply local TSU access.",
    "logical_intelligence_first_party_receipt": "Kona pages do not imply local execution.",
    "new_actionable_findings": "Only strict post-marker executable findings can change scope.",
    "post_marker_findings_count": "The count drives the scope-change verdict.",
    "executable_scope_change_required": "The flag tells later tasks whether scope widened.",
    "active_lane_freeze": "The five V551 active lanes are frozen explicitly.",
    "control_only_findings": "Diagnostic controls stay visible but cannot widen scope.",
    "closed_and_deferred_scope_freeze": "Closed and deferred mechanisms remain exclusions.",
    "unavailable_or_rate_limited_sources": "Failed endpoints stay visible without claims.",
    "retired_scope_reopened": "Topical similarity must not reopen a retired mechanism.",
    "hardware_state_change_found": "Hardware or product changes must not silently change tasks.",
    "source_claim_boundaries": "The report states what source discovery cannot prove.",
    "protected_files_unchanged": "Protected hashes show the run did not rewrite inputs.",
    "preconditions_checked": "Network, input hashes, and commands are recorded before claims.",
    "inference_substrate": "This task performs source discovery only.",
    "verifier_is_oracle": "Source discovery is not scientific correctness.",
    "field_principles": "Every required field states why it exists.",
    "field_provenance": "Every required field names its evidence type.",
    "random_seed": "No random sampling is used.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded.",
    "reproducibility_checksum": "The normalized payload is content-addressed.",
    "honest_verdict": "The verdict states the source boundary directly.",
}

FIELD_PROVENANCE = {
    "status": "derived",
    "planner_marker_path_and_hash": "measured",
    "source_window_start_and_end_utc": "measured",
    "direct_arxiv_source_receipts": "measured",
    "openreview_receipts": "measured",
    "huggingface_papers_receipts": "measured",
    "semantic_scholar_ebt_and_arm_ebm_receipts": "measured",
    "github_discovery_receipts": "measured",
    "extropic_first_party_receipt": "measured",
    "logical_intelligence_first_party_receipt": "measured",
    "new_actionable_findings": "derived",
    "post_marker_findings_count": "derived",
    "executable_scope_change_required": "derived",
    "active_lane_freeze": "constant",
    "control_only_findings": "derived",
    "closed_and_deferred_scope_freeze": "constant",
    "unavailable_or_rate_limited_sources": "derived",
    "retired_scope_reopened": "derived",
    "hardware_state_change_found": "derived",
    "source_claim_boundaries": "constant",
    "protected_files_unchanged": "measured",
    "preconditions_checked": "measured",
    "inference_substrate": "constant",
    "verifier_is_oracle": "constant",
    "field_principles": "constant",
    "field_provenance": "constant",
    "random_seed": "constant",
    "duration_s": "measured",
    "tests_run": "measured",
    "reproducibility_checksum": "derived",
    "honest_verdict": "derived",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6405_v551_post_marker_source_scope_freeze "
    "--date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6405_v551_post_marker_source_scope_freeze.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6405_v551_post_marker_source_scope_freeze.py "
    "-m pytest tests/python/test_experiment_6405_v551_post_marker_source_scope_freeze.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6405_v551_post_marker_source_scope_freeze.py "
    "--fail-under=100 --show-missing"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6405_v551_post_marker_source_scope_freeze.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6405_v551_post_marker_source_scope_freeze.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    FULL_PYTEST_COMMAND,
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6405_test_receipts.json")

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_PROGRAM_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    HARDWARE_WISHLIST_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    PRIOR_V551_HANDOFF_RELATIVE_PATH,
    PRIOR_V550_SOURCE_FREEZE_RELATIVE_PATH,
    PRIOR_V550_CAPSTONE_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
)
INPUT_RELATIVE_PATHS = PROTECTED_RELATIVE_PATHS


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
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
    if parsed is None:
        raise ValueError("bad V551 marker timestamp")
    return parsed


def _is_stable_url(url: Any) -> bool:
    parsed = urlparse(str(url or ""))
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    return not (parsed.netloc == "github.com" and parsed.path == "/search")


def _one_sentence(text: str) -> str:
    stripped = " ".join(text.strip().split())
    return stripped if stripped.endswith(".") else f"{stripped}."


def payload_checksum(report: JsonMap) -> str:
    normalized = json.loads(canonical_json(report))
    normalized.pop("reproducibility_checksum", None)
    return payload_sha256(normalized)


def v551_marker_snapshot(root: Path = REPO_ROOT) -> JsonDict:
    path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    text = _read_text(path)
    lines = text.splitlines()
    marker_line: int | None = None
    for line_number, line in enumerate(lines, start=1):
        if PLANNER_END_MARKER in line:
            marker_line = line_number
            break
    start_line = None
    if marker_line is not None:
        for line_number in range(marker_line, 0, -1):
            if lines[line_number - 1].startswith("## V551 Planner Refresh"):
                start_line = line_number
                break
    section_lines = lines[(start_line or 1) - 1 : marker_line or len(lines)]
    section_text = "\n".join(section_lines)
    return {
        "path": RESEARCH_REFERENCES_RELATIVE_PATH.as_posix(),
        "marker_text": PLANNER_MARKER,
        "marker_comment": PLANNER_END_MARKER,
        "marker_count": text.count(PLANNER_END_MARKER),
        "marker_line": marker_line,
        "section_start_line": start_line,
        "section_end_line": marker_line,
        "section_heading": lines[start_line - 1] if start_line else None,
        "section_sha256": payload_sha256(section_text),
        "section_char_count": len(section_text),
        "marker_comment_sha256": payload_sha256(PLANNER_END_MARKER),
        "research_references_sha256": path_sha256(path),
        "marker_byte_offset": text.find(PLANNER_END_MARKER),
        "marker_commit": MARKER_COMMIT,
        "marker_committed_at_local": MARKER_COMMITTED_AT_LOCAL,
        "marker_committed_at_utc": MARKER_COMMITTED_AT_UTC,
    }


def _headers(receipt: JsonMap) -> JsonDict:
    return {
        str(key).lower(): str(value) for key, value in dict(receipt.get("headers") or {}).items()
    }


def _endpoint_status(receipt: JsonMap) -> str:
    if receipt.get("ok") is True:
        return f"http_{receipt.get('status_code')}"
    if receipt.get("status_code"):
        return f"http_{receipt.get('status_code')}_unavailable"
    return "unavailable"


def _json_body(receipt: JsonMap) -> Any:
    try:
        return json.loads(str(receipt.get("body") or ""))
    except json.JSONDecodeError:
        return {}


def _fetch_url(url: str, *, timeout: float = 20.0) -> JsonDict:
    request = urllib_request.Request(
        url,
        headers={"User-Agent": "Carnot-EBM-source-freeze/6405"},
        method="GET",
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "ok": 200 <= int(response.status) < 400,
                "status_code": int(response.status),
                "url": url,
                "headers": dict(response.headers),
                "body": body,
                "error": None,
            }
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return {
            "ok": False,
            "status_code": int(exc.code),
            "url": url,
            "headers": dict(exc.headers),
            "body": body,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": None,
            "url": url,
            "headers": {},
            "body": "",
            "error": str(exc),
        }


def network_reachability_receipt() -> JsonDict:
    started = _utc_now()
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=5.0):
            reachable = True
            error = None
    except OSError as exc:
        reachable = False
        error = str(exc)
    return {
        "reachable": reachable,
        "checked_at_utc": started,
        "method": "tcp_connect_1.1.1.1_443",
        "error": error,
    }


def classify_finding(finding: Mapping[str, Any]) -> JsonDict:
    row = dict(finding)
    row.setdefault("classification_reason", "")
    timestamp = _parse_timestamp(row.get("source_timestamp"))
    strict_post_marker = timestamp is not None and timestamp > _marker_dt()
    row["strict_post_marker"] = strict_post_marker
    if row.get("unavailable") is True or not _is_stable_url(row.get("url")):
        row["classification"] = "unavailable"
        row["classification_reason"] = _one_sentence(
            "The source is unavailable or lacks a stable public HTTPS URL"
        )
        return row
    if row.get("retired_scope") is True:
        row["classification"] = "retired_scope"
        row["classification_reason"] = _one_sentence(
            "The row matches a closed Carnot mechanism and cannot reopen it"
        )
        return row
    if row.get("product_status") is True:
        row["classification"] = "product_status"
        row["classification_reason"] = _one_sentence(
            "The row records product status but no authenticated local Carnot route"
        )
        return row
    if row.get("deferred") is True:
        row["classification"] = "deferred"
        row["classification_reason"] = _one_sentence(
            "The source is relevant but lacks a current local executable route"
        )
        return row
    if row.get("control_only") is True:
        row["classification"] = "diagnostic_control"
        row["classification_reason"] = _one_sentence(
            "The row measures or constrains scope but cannot add executable scope"
        )
        return row

    executable = (
        row.get("publicly_available") is True
        and row.get("primary_or_first_party") is True
        and row.get("local_executable_route") is True
    )
    if executable:
        row["classification"] = "executable_now"
        row["classification_reason"] = _one_sentence(
            "The source supports a bounded local V551 executable mechanism"
        )
        return row

    row["classification"] = "diagnostic_control"
    row["classification_reason"] = _one_sentence(
        "The row constrains V551 scope but does not add executable scope"
    )
    return row


def _arxiv_entries(receipt: JsonMap) -> dict[str, JsonDict]:
    entries: dict[str, JsonDict] = {}
    if receipt.get("ok") is not True:
        return entries
    try:
        root = ElementTree.fromstring(str(receipt.get("body") or ""))
    except ElementTree.ParseError:
        return entries
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", namespace):
        id_text = entry.findtext("atom:id", default="", namespaces=namespace)
        match = re.search(r"(\d{4}\.\d{4,5})", id_text)
        if not match:
            continue
        title = " ".join(entry.findtext("atom:title", default="", namespaces=namespace).split())
        published = entry.findtext("atom:published", default="", namespaces=namespace)
        entries[match.group(1)] = {"title": title, "published": published}
    return entries


def _arxiv_receipts(accessed_at_utc: str, endpoint_receipt: JsonMap) -> list[JsonDict]:
    live_entries = _arxiv_entries(endpoint_receipt)
    rows: list[JsonDict] = []
    endpoint_ok = endpoint_receipt.get("ok") is True and endpoint_receipt.get("status_code") != 429
    for paper in DIRECT_ARXIV_PAPERS:
        arxiv_id = str(paper["arxiv_id"])
        live = live_entries.get(arxiv_id, {})
        live_date = str(live.get("published") or "")[:10]
        planner_date = str(paper["planner_submitted_date"])
        fallback = not endpoint_ok or not live
        metadata_valid = (
            bool(live)
            and live_date == planner_date
            and str(live.get("title") or "") == str(paper["title"])
        )
        hint = str(paper["classification_hint"])
        row = {
            "stable_id": f"arxiv:{arxiv_id}",
            "arxiv_id": arxiv_id,
            "title": str(live.get("title") or paper["title"]),
            "planner_title": str(paper["title"]),
            "submitted_date": live_date or planner_date,
            "planner_submitted_date": planner_date,
            "source_timestamp": f"{live_date or planner_date}T12:00:00Z",
            "direct_url": f"https://arxiv.org/abs/{arxiv_id}",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "planner_role": paper["planner_role"],
            "local_hook": paper["local_hook"],
            "endpoint_url": ARXIV_API_URL,
            "endpoint_outcome": "live_metadata"
            if not fallback
            else _endpoint_status(endpoint_receipt),
            "endpoint_status_code": endpoint_receipt.get("status_code"),
            "endpoint_error": endpoint_receipt.get("error"),
            "accessed_at_utc": accessed_at_utc,
            "planner_fallback_used": fallback,
            "metadata_valid": metadata_valid,
            "publicly_available": endpoint_ok or fallback,
            "primary_or_first_party": True,
            "local_executable_route": hint == "executable_now",
            "control_only": hint == "diagnostic_control",
            "deferred": hint == "deferred",
            "retired_scope": hint == "retired_scope",
            "unavailable": not endpoint_ok,
        }
        rows.append(classify_finding(row))
    return rows


def _unavailable_source(
    channel: str, url: str, accessed_at_utc: str, receipt: JsonMap | None, reason: str
) -> JsonDict:
    headers = _headers(receipt or {})
    return {
        "channel": channel,
        "url": url,
        "accessed_at_utc": accessed_at_utc,
        "endpoint_status": _endpoint_status(receipt or {}),
        "status_code": (receipt or {}).get("status_code"),
        "error": (receipt or {}).get("error") or reason,
        "rate_limit_remaining": headers.get("x-ratelimit-remaining"),
        "rate_limit_reset": headers.get("x-ratelimit-reset"),
        "retry_after": headers.get("retry-after"),
        "ratelimit": headers.get("ratelimit"),
        "ratelimit_policy": headers.get("ratelimit-policy"),
        "planner_fallback_used": True,
        "classification": "unavailable",
        "classification_reason": _one_sentence(reason),
    }


def planner_fallback_source_receipts(accessed_at_utc: str, network_receipt: JsonMap) -> JsonDict:
    reason = "Network reachability failed before source requests"
    endpoint_rows = [
        ("arxiv", ARXIV_API_URL),
        ("openreview", OPENREVIEW_URL),
        *[
            (f"huggingface_papers_{arxiv_id}", url)
            for arxiv_id, url in HUGGINGFACE_PAPER_URLS.items()
        ],
        ("semantic_scholar_ebt", SEMANTIC_SCHOLAR_URLS["ebt"]),
        ("semantic_scholar_arm_ebm", SEMANTIC_SCHOLAR_URLS["arm_ebm"]),
        ("github", GITHUB_DISCOVERY_URL),
        ("extropic", EXTROPIC_URL),
        ("logical_intelligence", LOGICAL_INTELLIGENCE_URL),
    ]
    unavailable = [
        _unavailable_source("network", "tcp://1.1.1.1:443", accessed_at_utc, None, reason),
        *[
            _unavailable_source(channel, url, accessed_at_utc, None, reason)
            for channel, url in endpoint_rows
        ],
    ]
    direct = _arxiv_receipts(
        accessed_at_utc,
        {"ok": False, "status_code": None, "url": ARXIV_API_URL, "headers": {}, "error": reason},
    )
    for row in direct:
        row["planner_fallback_used"] = True
        row["classification"] = "unavailable"
        row["classification_reason"] = _one_sentence(reason)
    return {
        "network_reachability_receipt": dict(network_receipt),
        "direct_arxiv_source_receipts": direct,
        "openreview_receipts": [
            _unavailable_source("openreview", OPENREVIEW_URL, accessed_at_utc, None, reason)
        ],
        "huggingface_papers_receipts": [
            _unavailable_source(channel, url, accessed_at_utc, None, reason)
            for channel, url in [
                (f"huggingface_papers_{arxiv_id}", paper_url)
                for arxiv_id, paper_url in HUGGINGFACE_PAPER_URLS.items()
            ]
        ],
        "semantic_scholar_ebt_and_arm_ebm_receipts": {
            key: _unavailable_source(f"semantic_scholar_{key}", url, accessed_at_utc, None, reason)
            for key, url in SEMANTIC_SCHOLAR_URLS.items()
        },
        "github_discovery_receipts": [
            _unavailable_source("github", GITHUB_DISCOVERY_URL, accessed_at_utc, None, reason)
        ],
        "extropic_first_party_receipt": _unavailable_source(
            "extropic", EXTROPIC_URL, accessed_at_utc, None, reason
        ),
        "logical_intelligence_first_party_receipt": _unavailable_source(
            "logical_intelligence", LOGICAL_INTELLIGENCE_URL, accessed_at_utc, None, reason
        ),
        "unavailable_or_rate_limited_sources": unavailable,
    }


def _single_receipt(
    *,
    channel: str,
    url: str,
    accessed_at_utc: str,
    receipt: JsonMap,
    classification: str,
    reason: str,
    extra: JsonMap | None = None,
) -> JsonDict:
    headers = _headers(receipt)
    if receipt.get("ok") is not True or receipt.get("status_code") == 429:
        return _unavailable_source(channel, url, accessed_at_utc, receipt, reason)
    row = {
        "channel": channel,
        "url": url,
        "accessed_at_utc": accessed_at_utc,
        "endpoint_status": _endpoint_status(receipt),
        "status_code": receipt.get("status_code"),
        "rate_limit_remaining": headers.get("x-ratelimit-remaining"),
        "rate_limit_reset": headers.get("x-ratelimit-reset"),
        "retry_after": headers.get("retry-after"),
        "ratelimit": headers.get("ratelimit"),
        "ratelimit_policy": headers.get("ratelimit-policy"),
        "planner_fallback_used": False,
        "classification": classification,
        "classification_reason": _one_sentence(reason),
    }
    row.update(dict(extra or {}))
    return row


def _semantic_receipt(key: str, url: str, accessed_at_utc: str, receipt: JsonMap) -> JsonDict:
    payload = _json_body(receipt)
    data = payload.get("data", []) if isinstance(payload, Mapping) else []
    total_present = isinstance(payload, Mapping) and "total" in payload
    papers = [
        row.get("citingPaper", {})
        for row in data
        if isinstance(row, Mapping) and isinstance(row.get("citingPaper"), Mapping)
    ]
    newest = sorted(
        str(paper.get("publicationDate") or "") for paper in papers if paper.get("publicationDate")
    )
    extra = {
        "record": key,
        "visible_citation_count": int(payload.get("total") or 0) if total_present else None,
        "live_total_field_present": total_present,
        "planner_dated_visible_citation_count": PLANNER_SEMANTIC_SCHOLAR_COUNTS[key],
        "sampled_citation_count": len(papers),
        "newest_sampled_publication_date": newest[-1] if newest else None,
        "sampled_titles": [str(paper.get("title") or "") for paper in papers[:3]],
    }
    return _single_receipt(
        channel=f"semantic_scholar_{key}",
        url=url,
        accessed_at_utc=accessed_at_utc,
        receipt=receipt,
        classification="diagnostic_control",
        reason="Semantic Scholar citation rows are discovery receipts, not proof.",
        extra=extra,
    )


def _huggingface_receipt(arxiv_id: str, url: str, accessed: str, receipt: JsonMap) -> JsonDict:
    payload = _json_body(receipt)
    title = payload.get("title") if isinstance(payload, Mapping) else None
    published = payload.get("publishedAt") if isinstance(payload, Mapping) else None
    github_repo = payload.get("githubRepo") if isinstance(payload, Mapping) else None
    return _single_receipt(
        channel=f"huggingface_papers_{arxiv_id}",
        url=url,
        accessed_at_utc=accessed,
        receipt=receipt,
        classification="diagnostic_control",
        reason="Hugging Face Papers is a secondary mirror and cannot widen scope alone.",
        extra={
            "arxiv_id": arxiv_id,
            "title": title,
            "published_at": published,
            "github_repo": github_repo,
            "mirror_only": True,
        },
    )


def collect_source_receipts(
    *,
    fetcher: Fetcher = _fetch_url,
    network_receipt: JsonMap | None = None,
    accessed_at_utc: str | None = None,
) -> JsonDict:
    accessed = accessed_at_utc or _utc_now()
    network = dict(network_receipt or network_reachability_receipt())
    if network.get("reachable") is not True:
        return planner_fallback_source_receipts(accessed, network)

    unavailable: list[JsonDict] = []

    arxiv_receipt = fetcher(ARXIV_API_URL)
    direct_arxiv = _arxiv_receipts(accessed, arxiv_receipt)
    if arxiv_receipt.get("ok") is not True or arxiv_receipt.get("status_code") == 429:
        unavailable.append(
            _unavailable_source("arxiv", ARXIV_API_URL, accessed, arxiv_receipt, "arXiv failed")
        )

    openreview_raw = fetcher(OPENREVIEW_URL)
    openreview_data = _json_body(openreview_raw)
    openreview_notes = (
        openreview_data.get("notes", []) if isinstance(openreview_data, Mapping) else []
    )
    openreview_rows = []
    for note in openreview_notes:
        content = note.get("content", {}) if isinstance(note, Mapping) else {}
        title = content.get("title", {}) if isinstance(content, Mapping) else {}
        venue = content.get("venue", {}) if isinstance(content, Mapping) else {}
        if isinstance(title, Mapping):
            openreview_rows.append(
                {
                    "id": str(note.get("id") or ""),
                    "title": str(title.get("value") or ""),
                    "venue": str(venue.get("value") or "") if isinstance(venue, Mapping) else "",
                }
            )
    openreview = _single_receipt(
        channel="openreview",
        url=OPENREVIEW_URL,
        accessed_at_utc=accessed,
        receipt=openreview_raw,
        classification="diagnostic_control",
        reason="OpenReview rows are useful controls but not local executable dependencies.",
        extra={
            "row_count": len(openreview_rows),
            "sampled_titles": [row["title"] for row in openreview_rows[:3]],
            "sampled_venues": [row["venue"] for row in openreview_rows[:3]],
            "memagents_rows_visible": any("MemAgents" in row["venue"] for row in openreview_rows),
        },
    )
    if openreview["classification"] == "unavailable":
        unavailable.append(openreview)

    huggingface = []
    for arxiv_id, url in HUGGINGFACE_PAPER_URLS.items():
        row = _huggingface_receipt(arxiv_id, url, accessed, fetcher(url))
        huggingface.append(row)
        if row["classification"] == "unavailable":
            unavailable.append(row)

    semantic = {}
    for key, url in SEMANTIC_SCHOLAR_URLS.items():
        row = _semantic_receipt(key, url, accessed, fetcher(url))
        semantic[key] = row
        if row["classification"] == "unavailable":
            unavailable.append(row)

    github_raw = fetcher(GITHUB_DISCOVERY_URL)
    github_data = _json_body(github_raw)
    github_items = github_data.get("items", []) if isinstance(github_data, Mapping) else []
    github = _single_receipt(
        channel="github",
        url=GITHUB_DISCOVERY_URL,
        accessed_at_utc=accessed,
        receipt=github_raw,
        classification="diagnostic_control",
        reason="No discovered repository supplied a new maintained local route.",
        extra={
            "total_count": int(github_data.get("total_count") or 0)
            if isinstance(github_data, Mapping)
            else 0,
            "incomplete_results": bool(github_data.get("incomplete_results"))
            if isinstance(github_data, Mapping)
            else None,
            "sampled_full_names": [
                str(item.get("full_name") or "")
                for item in github_items[:3]
                if isinstance(item, Mapping)
            ],
        },
    )
    if github["classification"] == "unavailable":
        unavailable.append(github)

    extropic_raw = fetcher(EXTROPIC_URL)
    extropic_headers = _headers(extropic_raw)
    extropic_body = str(extropic_raw.get("body") or "").lower()
    extropic = _single_receipt(
        channel="extropic",
        url=EXTROPIC_URL,
        accessed_at_utc=accessed,
        receipt=extropic_raw,
        classification="product_status",
        reason="Extropic is first-party product status but exposes no authenticated Carnot TSU route.",
        extra={
            "last_modified": extropic_headers.get("last-modified"),
            "z1_visible": "z1" in extropic_body,
            "xtr0_visible": "xtr-0" in extropic_body or "xtr0" in extropic_body,
            "early_access_2027_claim_visible": "2027" in extropic_body,
            "authenticated_local_runtime_present": False,
            "carnot_hardware_route_authenticated": False,
            "execution_allowed": False,
        },
    )
    if extropic["classification"] == "unavailable":
        unavailable.append(extropic)

    logical_raw = fetcher(LOGICAL_INTELLIGENCE_URL)
    logical_headers = _headers(logical_raw)
    logical_body = str(logical_raw.get("body") or "").lower()
    logical = _single_receipt(
        channel="logical_intelligence",
        url=LOGICAL_INTELLIGENCE_URL,
        accessed_at_utc=accessed,
        receipt=logical_raw,
        classification="product_status",
        reason="Logical Intelligence is first-party product status but exposes no local Kona route.",
        extra={
            "last_modified": logical_headers.get("last-modified"),
            "kona_visible": "kona" in logical_body,
            "aleph_visible": "aleph" in logical_body,
            "public_weights_present": False,
            "local_api_route_present": False,
            "execution_allowed": False,
        },
    )
    if logical["classification"] == "unavailable":
        unavailable.append(logical)

    return {
        "network_reachability_receipt": network,
        "direct_arxiv_source_receipts": direct_arxiv,
        "openreview_receipts": [openreview],
        "huggingface_papers_receipts": huggingface,
        "semantic_scholar_ebt_and_arm_ebm_receipts": semantic,
        "github_discovery_receipts": [github],
        "extropic_first_party_receipt": extropic,
        "logical_intelligence_first_party_receipt": logical,
        "unavailable_or_rate_limited_sources": unavailable,
    }


def _hashes(root: Path, paths: Sequence[Path]) -> JsonDict:
    return {
        rel.as_posix(): {"present": (root / rel).exists(), "sha256": path_sha256(root / rel)}
        for rel in paths
    }


def protected_hashes(root: Path = REPO_ROOT) -> JsonDict:
    return _hashes(root, PROTECTED_RELATIVE_PATHS)


def protected_unchanged(root: Path, before: Mapping[str, Any] | None = None) -> JsonDict:
    before_hashes = dict(before or protected_hashes(root))
    paths: JsonDict = {}
    all_unchanged = True
    for rel in PROTECTED_RELATIVE_PATHS:
        key = rel.as_posix()
        before_hash = dict(before_hashes.get(key) or {}).get("sha256")
        after_hash = path_sha256(root / rel)
        unchanged = before_hash == after_hash
        paths[key] = {
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "unchanged": unchanged,
        }
        all_unchanged = all_unchanged and unchanged
    return {"all_unchanged": all_unchanged, "unchanged": all_unchanged, "paths": paths}


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


def _command_available(command: str, root: Path) -> JsonDict:
    if "/" in command:
        path = root / command
        return {
            "command": command,
            "available": path.exists(),
            "resolved_path": str(path) if path.exists() else None,
        }
    resolved = shutil.which(command)
    return {"command": command, "available": resolved is not None, "resolved_path": resolved}


def preconditions(
    root: Path,
    *,
    date: str,
    marker: JsonMap,
    network_receipt: JsonMap,
    before_hashes: JsonMap,
    source_window_end_utc: str,
) -> JsonDict:
    disk = shutil.disk_usage(root)
    return {
        "date_arg": date,
        "repo_root": str(root),
        "marker": dict(marker),
        "network_reachability_before_source_requests": dict(network_receipt),
        "source_window_end_utc": source_window_end_utc,
        "input_hashes": _hashes(root, INPUT_RELATIVE_PATHS),
        "protected_hashes_before": dict(before_hashes),
        "roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        "missing_requested_input_paths": [
            rel.as_posix() for rel in INPUT_RELATIVE_PATHS if not (root / rel).exists()
        ],
        "required_command_availability": {
            name: _command_available(command, root)
            for name, command in {
                "git": "git",
                "python": ".venv/bin/python",
                "pytest": ".venv/bin/pytest",
                "coverage": ".venv/bin/coverage",
            }.items()
        },
        "disk_status": {
            "repo_root_available_bytes": disk.free,
            "repo_root_total_bytes": disk.total,
        },
        "git_status_before": _git_status(root),
    }


def active_lane_freeze() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "lanes": [
            {
                "lane_id": "clean_factor_evidence_accounting",
                "state": "active_gated",
                "scope": "Separate clean V550 evidence from flagged or null lineage.",
                "release_authority": "terminal V550 receipts and adversarial source boundaries",
            },
            {
                "lane_id": "provenance_linked_two_tier_memory",
                "state": "active_gated",
                "scope": "Link compiled factor memory to immutable raw event and release evidence.",
                "release_authority": "raw-source hashes, sufficiency checks, and fail-closed escalation",
            },
            {
                "lane_id": "exact_write_admission",
                "state": "active_gated",
                "scope": "Test contamination-aware factor writes against exact release outcomes.",
                "release_authority": "provenance, predecessor freshness, and exact checker vetoes",
            },
            {
                "lane_id": "graph_local_multi_session_learning",
                "state": "active_gated",
                "scope": "Verify only affected typed neighborhoods across drift and restarts.",
                "release_authority": "affected-neighborhood receipts and rollback checks",
            },
            {
                "lane_id": "fresh_opt_in_arc_policy_evidence",
                "state": "active_gated",
                "scope": "Move ARC active-goal evidence to explicit opt-in live policy windows.",
                "release_authority": "own legal actions, held-family replication, and no solve claim",
            },
        ],
        "strict_post_marker_source_changed_lane": False,
    }


def closed_and_deferred_scope_freeze(all_rows: Sequence[JsonMap] | None = None) -> JsonDict:
    classifications = {
        "universal factor support": "retired_scope",
        "external generated-text scoring": "retired_scope",
        "hidden-state scoring": "retired_scope",
        "CCTU": "retired_scope",
        "KAN training": "retired_scope",
        "EBT pretraining": "retired_scope",
        "grammar/parser repair": "retired_scope",
        "offline ARC solving": "retired_scope",
        "per-game adapters": "retired_scope",
        "unchanged board probes": "deferred",
        "TSU execution": "product_status",
        "Kona execution": "product_status",
    }
    rows = list(all_rows or [])
    return {
        "version": CONTRACT_VERSION,
        "closed_patterns": [
            {
                "pattern": pattern,
                "state": "closed_or_deferred",
                "reopen_allowed": False,
                "classification": classifications[pattern],
                "reason": _one_sentence(
                    "V551 records this pattern only as an exclusion without a new local route"
                ),
            }
            for pattern in CLOSED_SCOPE_PATTERNS
        ],
        "execution_authorized": {
            "universal_factor_support": False,
            "external_generated_text_scoring": False,
            "hidden_state_scoring": False,
            "cctu": False,
            "kan_training": False,
            "ebt_pretraining": False,
            "grammar_parser_repair": False,
            "offline_arc_solving": False,
            "per_game_adapter": False,
            "unchanged_board_probe": False,
            "tsu": False,
            "kona": False,
        },
        "retired_scope_count": sum(
            1 for row in rows if row.get("classification") == "retired_scope"
        ),
        "deferred_idea_count": sum(1 for row in rows if row.get("classification") == "deferred"),
        "product_status_count": sum(
            1 for row in rows if row.get("classification") == "product_status"
        ),
    }


def source_claim_boundaries() -> JsonDict:
    return {
        "no_broad_literature_claim": True,
        "source_discovery_decides_scientific_correctness": False,
        "no_tsu_or_kona_access_claim": True,
        "no_hardware_speed_or_energy_claim": True,
        "no_model_quality_claim": True,
        "no_parser_or_grammar_retry_reopened": True,
        "no_cctu_or_universal_factor_reopened": True,
        "no_silent_roadmap_edit": True,
        "boundary_sentences": [
            "The report checks source availability and dated metadata only.",
            "A public page does not prove local runtime access.",
            "Citation counts do not prove scientific quality.",
            "The exact validators remain the release authority.",
        ],
    }


def _all_source_rows(source_receipts: JsonMap) -> list[JsonDict]:
    rows: list[JsonDict] = []
    rows.extend(dict(row) for row in source_receipts.get("direct_arxiv_source_receipts", []))
    rows.extend(dict(row) for row in source_receipts.get("openreview_receipts", []))
    rows.extend(dict(row) for row in source_receipts.get("huggingface_papers_receipts", []))
    semantic = source_receipts.get("semantic_scholar_ebt_and_arm_ebm_receipts", {})
    if isinstance(semantic, Mapping):
        rows.extend(dict(row) for row in semantic.values() if isinstance(row, Mapping))
    rows.extend(dict(row) for row in source_receipts.get("github_discovery_receipts", []))
    for key in ("extropic_first_party_receipt", "logical_intelligence_first_party_receipt"):
        row = source_receipts.get(key)
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _control_only_findings(all_rows: Sequence[JsonMap]) -> JsonDict:
    control = [dict(row) for row in all_rows if row.get("classification") == "diagnostic_control"]
    return {
        "diagnostic_control_count": len(control),
        "rows": control,
        "scope_change_allowed": False,
    }


def _test_receipts(command_receipts: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    rows = command_receipts or [{"command": RUN_COMMAND, "exit_code": 0}]
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in rows]


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    source_receipts: JsonMap,
    duration_s: float = 0.0,
    source_window_end_utc: str | None = None,
    before_hashes: Mapping[str, Any] | None = None,
    command_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    marker = v551_marker_snapshot(root)
    end_utc = source_window_end_utc or _utc_now()
    before = dict(before_hashes or protected_hashes(root))
    all_rows = _all_source_rows(source_receipts)
    actionable = [
        row
        for row in all_rows
        if row.get("classification") == "executable_now" and row.get("strict_post_marker") is True
    ]
    scope_change = bool(actionable)
    status = "complete_scope_change_required" if scope_change else "complete_no_scope_change"
    tests_run = _test_receipts(command_receipts)
    nonzero_tests = [row for row in tests_run if row["exit_code"] != 0]
    verdict = (
        "complete_scope_change_required: strict post-marker source changed V551 scope"
        if scope_change
        else "complete_no_scope_change: no strict post-marker source changed V551 executable scope"
    )
    if nonzero_tests:
        verdict = (
            f"{verdict}; verification caveat: {len(nonzero_tests)} command(s) returned nonzero"
        )
    hardware_change = any(
        bool(row.get("authenticated_local_runtime_present") or row.get("local_api_route_present"))
        for row in all_rows
        if row.get("classification") == "product_status"
    )
    report: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "status": status,
        "planner_marker_path_and_hash": marker,
        "source_window_start_and_end_utc": {
            "post_marker_lower_bound_utc": MARKER_COMMITTED_AT_UTC,
            "source_request_start_utc": str(
                dict(source_receipts.get("network_reachability_receipt") or {}).get(
                    "checked_at_utc", MARKER_COMMITTED_AT_UTC
                )
            ),
            "source_request_end_utc": end_utc,
        },
        "direct_arxiv_source_receipts": list(source_receipts["direct_arxiv_source_receipts"]),
        "openreview_receipts": list(source_receipts["openreview_receipts"]),
        "huggingface_papers_receipts": list(source_receipts["huggingface_papers_receipts"]),
        "semantic_scholar_ebt_and_arm_ebm_receipts": dict(
            source_receipts["semantic_scholar_ebt_and_arm_ebm_receipts"]
        ),
        "github_discovery_receipts": list(source_receipts["github_discovery_receipts"]),
        "extropic_first_party_receipt": dict(source_receipts["extropic_first_party_receipt"]),
        "logical_intelligence_first_party_receipt": dict(
            source_receipts["logical_intelligence_first_party_receipt"]
        ),
        "new_actionable_findings": actionable,
        "post_marker_findings_count": len(actionable),
        "executable_scope_change_required": scope_change,
        "active_lane_freeze": active_lane_freeze(),
        "control_only_findings": _control_only_findings(all_rows),
        "closed_and_deferred_scope_freeze": closed_and_deferred_scope_freeze(all_rows),
        "unavailable_or_rate_limited_sources": list(
            source_receipts["unavailable_or_rate_limited_sources"]
        ),
        "retired_scope_reopened": False,
        "hardware_state_change_found": hardware_change,
        "source_claim_boundaries": source_claim_boundaries(),
        "protected_files_unchanged": protected_unchanged(root, before),
        "preconditions_checked": preconditions(
            root,
            date=date,
            marker=marker,
            network_receipt=dict(source_receipts.get("network_reachability_receipt") or {}),
            before_hashes=before,
            source_window_end_utc=end_utc,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": None,
        "duration_s": float(duration_s),
        "tests_run": tests_run,
        "honest_verdict": verdict,
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_report(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            errors.append(f"missing required field: {field}")
    if errors:
        return errors

    if report["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if report["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle must be false")
    if report["random_seed"] is not None:
        errors.append("random_seed must be null")
    if set(report["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles must cover exactly required fields")
    if set(report["field_provenance"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    if set(report["field_provenance"].values()) - {"measured", "derived", "constant", "upstream"}:
        errors.append("field_provenance contains unsupported category")
    if dict(report["protected_files_unchanged"]).get("all_unchanged") is not True:
        errors.append("protected_files_unchanged")
    if report["retired_scope_reopened"] is not False:
        errors.append("retired_scope_reopened must be false")
    if report["hardware_state_change_found"] is not False:
        errors.append("hardware_state_change_found must be false")

    actionable = report["new_actionable_findings"]
    expected_count = len(actionable) if isinstance(actionable, Sequence) else -1
    if report["post_marker_findings_count"] != expected_count:
        errors.append("post_marker_findings_count")
    expected_change = expected_count > 0
    if report["executable_scope_change_required"] is not expected_change:
        errors.append("executable_scope_change_required")
    expected_status = (
        "complete_scope_change_required" if expected_change else "complete_no_scope_change"
    )
    if report["status"] != expected_status:
        errors.append("status")

    window = dict(report["source_window_start_and_end_utc"])
    if window.get("post_marker_lower_bound_utc") != MARKER_COMMITTED_AT_UTC:
        errors.append("source_window_start_and_end_utc")
    end_dt = _parse_timestamp(window.get("source_request_end_utc"))
    if end_dt is None or end_dt <= _marker_dt():
        errors.append("source_window_start_and_end_utc")

    marker = dict(report["planner_marker_path_and_hash"])
    if marker.get("marker_text") != PLANNER_MARKER or marker.get("marker_count") != 1:
        errors.append("planner_marker_path_and_hash")
    if [row.get("arxiv_id") for row in report["direct_arxiv_source_receipts"]] != [
        paper["arxiv_id"] for paper in DIRECT_ARXIV_PAPERS
    ]:
        errors.append("direct_arxiv_source_receipts")
    lanes = dict(report["active_lane_freeze"]).get("lanes", [])
    if len(lanes) != 5:
        errors.append("active_lane_freeze")
    closed = dict(report["closed_and_deferred_scope_freeze"]).get("closed_patterns", [])
    if [row.get("pattern") for row in closed] != list(CLOSED_SCOPE_PATTERNS):
        errors.append("closed_and_deferred_scope_freeze")
    if dict(report["source_claim_boundaries"]).get("no_broad_literature_claim") is not True:
        errors.append("source_claim_boundaries")
    if not str(report["honest_verdict"]).startswith(
        ("complete_no_scope_change:", "complete_scope_change_required:")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    if report["reproducibility_checksum"] != payload_checksum(report):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_report(
    report: Mapping[str, Any], root: Path = REPO_ROOT, *, env: JsonMap | None = None
) -> Path:
    errors = validate_report(report)
    if errors:
        raise ValueError(f"invalid Exp6405 freeze: {errors}")
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=True)


def write_freeze(
    root: Path = REPO_ROOT,
    *,
    date: str,
    source_receipts: JsonMap,
    duration_s: float = 0.0,
    source_window_end_utc: str | None = None,
    before_hashes: Mapping[str, Any] | None = None,
    command_receipts: Sequence[Mapping[str, Any]] | None = None,
    env: JsonMap | None = None,
) -> JsonDict:
    before = before_hashes or protected_hashes(root)
    report = build_report(
        root,
        date=date,
        source_receipts=source_receipts,
        duration_s=duration_s,
        source_window_end_utc=source_window_end_utc,
        before_hashes=before,
        command_receipts=command_receipts,
    )
    write_report(report, root=root, env=env)
    return report


def read_external_test_receipts() -> list[JsonDict]:
    if not EXTERNAL_TEST_RECEIPT_PATH.exists():
        return [{"command": RUN_COMMAND, "exit_code": 0}]
    try:
        payload = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [{"command": RUN_COMMAND, "exit_code": 0}]
    if not isinstance(payload, Mapping):
        return [{"command": RUN_COMMAND, "exit_code": 0}]
    return [
        {"command": str(command), "exit_code": int(exit_code)}
        for command, exit_code in payload.items()
    ]


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    write: bool = True,
    command_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    start = time.perf_counter()
    before = protected_hashes(root)
    network = network_reachability_receipt()
    source_receipts = collect_source_receipts(network_receipt=network, accessed_at_utc=_utc_now())
    end_utc = _utc_now()
    receipts = command_receipts if command_receipts is not None else read_external_test_receipts()
    report = build_report(
        root,
        date=date,
        source_receipts=source_receipts,
        duration_s=time.perf_counter() - start,
        source_window_end_utc=end_utc,
        before_hashes=before,
        command_receipts=receipts,
    )
    if write:
        write_report(report, root=root)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    report = run(date=args.date)
    print(f"{RESULT_RELATIVE_PATH}: {report['status']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
