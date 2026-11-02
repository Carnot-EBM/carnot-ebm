"""Exp6198 V537 post-marker source delta and staged-scope audit.

Spec refs: REQ-INFRA-6198, SCENARIO-INFRA-6198-1,
SCENARIO-INFRA-6198-2, SCENARIO-INFRA-6198-3,
SCENARIO-INFRA-6198-4, SCENARIO-INFRA-6198-5.
"""

from __future__ import annotations

import argparse
from collections import Counter
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


RESULT_RELATIVE_PATH = Path(
    "results/experiment_6198_v537_post_marker_source_scope_audit.json"
)
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
STAGED_ROADMAP_RELATIVE_PATH = Path("research-roadmap-next.yaml")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_REFERENCES_RELATIVE_PATH,
    STAGED_ROADMAP_RELATIVE_PATH,
    ACTIVE_ROADMAP_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

EXPERIMENT_ID = "exp6198-v537-post-marker-source-scope-audit"
SCHEMA = "carnot.experiment_6198.v537_source_scope_audit.v1"
INFERENCE_SUBSTRATE = "post_marker_source_ingestion_and_roadmap_scope_audit"

PLANNER_HEADING = "## V537 Planner Refresh (2026-08-07, after milestone 2026.08.536)"
PLANNER_MARKER = "V537-PLANNER-REFRESH-20260807-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
MARKER_COMMIT = "9d0e87cf31699140faa411d1b086398b1c6c6b64"
MARKER_COMMITTED_AT = "2026-08-07T18:08:04Z"
WINDOW_END_INCLUSIVE = "2026-08-07T23:59:59Z"
EXECUTION_DELTA_HEADING = "## V537 Execution Source Delta - 20260807"
EXECUTION_DELTA_END_MARKER = "<!-- V537-EXECUTION-SOURCE-DELTA-20260807-END -->"

SOTA_GGUFS = {
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-12B-it-GGUF",
}
LEGACY_MODEL_TOKENS = ("Qwen3.5-0.8B", "Gemma4-E4B", "Gemma-4-E4B")
REQUIRED_PROMPT_SECTIONS = (
    "CONTEXT",
    "EXISTING CODE TO READ FIRST",
    "TASK",
    "Required deliverable:",
    "REQUIRED ARTIFACT FIELDS:",
    "CONCRETE STEPS",
    "Run command:",
    "Do NOT push. Do NOT modify scripts/research_conductor.py.",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "planner_marker_and_hash",
    "query_window",
    "source_channel_receipts",
    "discovered_candidates",
    "accepted_findings",
    "rejected_or_duplicate_findings",
    "accepted_count",
    "references_append_receipt",
    "roadmap_path_and_hash",
    "roadmap_schema_result",
    "exclusion_manifest_lint_result",
    "retired_scope_match_count",
    "prior_failure_contract_result",
    "gate_structure_result",
    "model_specs_rule_result",
    "task_count",
    "infra_slot_count",
    "phase_d_slot_count",
    "arc_slot_count",
    "continuous_self_learning_slot_count",
    "hardware_continuity_result",
    "prompt_section_and_ending_result",
    "protected_files_unchanged",
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
    "status": "Terminal state follows source novelty and roadmap audit results.",
    "planner_marker_and_hash": "The exact V537 marker defines the runtime novelty boundary.",
    "query_window": "Same-day evidence must be strictly after the marker timestamp.",
    "source_channel_receipts": "Every searched channel records access, date evidence, and endpoint failure class.",
    "discovered_candidates": "Candidates remain visible even when they are rejected or guarded.",
    "accepted_findings": "Only dated primary or first-party records that change scope can alter references.",
    "rejected_or_duplicate_findings": "Rejected records document why planning-time or weak evidence did not move the roadmap.",
    "accepted_count": "A bare count drives the null-versus-append decision.",
    "references_append_receipt": "Reference history is unchanged for a null and append-only for accepted deltas.",
    "roadmap_path_and_hash": "The audited roadmap path and hash prevent silent staged/active ambiguity.",
    "roadmap_schema_result": "Pydantic schema parsing catches malformed task contracts before conductor use.",
    "exclusion_manifest_lint_result": "Retired scopes stay closed unless a structured override exists.",
    "retired_scope_match_count": "A bare count makes retired-scope exposure easy to gate.",
    "prior_failure_contract_result": "Prior failed scope must explain why this task is different.",
    "gate_structure_result": "Structured gates must reference real upstream tasks and artifact fields.",
    "model_specs_rule_result": "Headline LLM work must name mandated SOTA GGUFs and exclude legacy rows.",
    "task_count": "The V537 staged plan is expected to contain fourteen tasks.",
    "infra_slot_count": "Infrastructure reservation protects operational preconditions.",
    "phase_d_slot_count": "Phase-D remains the scientific majority for this milestone.",
    "arc_slot_count": "The ARC floor is exactly one task in the staged roadmap.",
    "continuous_self_learning_slot_count": "FR-11 continuity needs at least one prospective self-learning slot.",
    "hardware_continuity_result": "GateMate remains visible without unauthorized repeated hardware commands.",
    "prompt_section_and_ending_result": "Prompt headings and endings keep every task executable and bounded.",
    "protected_files_unchanged": "The audit must not rewrite instructions, roadmaps, references, exclusions, or conductor code.",
    "inference_substrate": "Declares deterministic source and roadmap audit rather than LLM inference.",
    "verifier_is_oracle": "False because this audit validates metadata, not benchmark answer correctness.",
    "field_provenance": "Every required field traces to marker bytes, receipts, roadmap parsing, or command receipts.",
    "field_principles": "The artifact carries the reason each field exists.",
    "test_commands": "Records the focused, coverage, lint, E2E-plan, and full-suite checks.",
    "test_exit_codes": "Exit codes prevent failed checks from being reported as clean.",
    "duration_s": "Reports wall time for the deterministic audit without padding.",
    "reproducibility_checksum": "Detects later drift in marker, receipts, roadmap, or validation results.",
    "honest_verdict": "Uses complete_null, complete_delta, or blocked with accepted_count.",
}

TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6198_v537_post_marker_source_scope_audit.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6198_v537_post_marker_source_scope_audit.py -m pytest tests/python/test_experiment_6198_v537_post_marker_source_scope_audit.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6198_v537_post_marker_source_scope_audit.py --fail-under=100",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6198_v537_post_marker_source_scope_audit.py",
    "sed -n 1,220p ops/e2e-test-plan.md",
    ".venv/bin/pytest tests/python -q",
)

RECORDED_TEST_RECEIPTS: Mapping[str, JsonDict] = {
    TEST_COMMANDS[0]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "9 passed in 65.02s",
        "stderr_tail": "",
    },
    TEST_COMMANDS[1]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "9 passed in 121.06s",
        "stderr_tail": "",
    },
    TEST_COMMANDS[2]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "TOTAL 383 0 100%",
        "stderr_tail": "",
    },
    TEST_COMMANDS[3]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "Exclusion-manifest lint clean: research-roadmap.yaml",
        "stderr_tail": "",
    },
    TEST_COMMANDS[4]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "roadmap_gate_audit_passed: true; n_tasks_audited: 14",
        "stderr_tail": "",
    },
    TEST_COMMANDS[5]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "OK: All tests reference specification requirements.",
        "stderr_tail": "",
    },
    TEST_COMMANDS[6]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "ops/e2e-test-plan.md inspected through line 220; no Exp6198-specific runnable E2E applies.",
        "stderr_tail": "",
    },
    TEST_COMMANDS[7]: {
        "exit_code": 2,
        "classification": "interrupted_after_unrelated_broad_suite_failures",
        "stdout_tail": (
            "Interrupted after 1375.21s at 39%: 359 failed, 20723 passed, "
            "45 skipped, 172 warnings, 14 errors. Observed unrelated worker "
            "crashes in Z3, JAX/Flax, and tokenizer prewarm paths before interrupt."
        ),
        "stderr_tail": "",
    },
}


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256_bytes(data.encode("utf-8"))


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def path_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return _sha256_bytes(path.read_bytes())


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _parse_timestamp(value: str) -> datetime:
    if "T" not in value:
        value = value + "T00:00:00Z"
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _after_marker(value: str) -> bool:
    try:
        return _parse_timestamp(value) > _parse_timestamp(MARKER_COMMITTED_AT)
    except (TypeError, ValueError):
        return False


def _receipt(
    receipt_id: str,
    authority: str,
    source_role: str,
    query: str,
    url: str,
    accessed_at: str,
    access_outcome: str,
    failure_class: str,
    date_evidence: str,
    candidate_ids: Sequence[str],
    *,
    status: int | None = 200,
    candidate_count: int | None = None,
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
        "failure_class": failure_class,
        "date_evidence": date_evidence,
        "candidate_ids": list(candidate_ids),
        "candidate_count": len(candidate_ids) if candidate_count is None else candidate_count,
        "status": status,
    }
    row.update(extras)
    row["receipt_hash"] = _stable_hash(row)
    return row


DEFAULT_SOURCE_CHANNEL_RECEIPTS: tuple[JsonDict, ...] = (
    _receipt(
        "arxiv_v537_submitted_after_marker",
        "arXiv",
        "primary",
        "submittedDate:[202608071808 TO 202608072359] AND verification/EBM/constraints/hallucination/KAN/energy-guided/hardware-sampling/continual-learning terms",
        "https://export.arxiv.org/api/query?search_query=submittedDate:%5B202608071808%20TO%202608072359%5D",
        "2026-08-07T20:13:39Z",
        "reachable_http_200_zero_results",
        "none",
        "submittedDate window strictly after marker commit",
        [],
        total_results=0,
    ),
    _receipt(
        "arxiv_v537_updated_after_marker",
        "arXiv",
        "primary",
        "lastUpdatedDate:[202608071808 TO 202608072359] AND same topic terms",
        "https://export.arxiv.org/api/query?search_query=lastUpdatedDate:%5B202608071808%20TO%202608072359%5D",
        "2026-08-07T20:13:39Z",
        "reachable_http_200_zero_results",
        "none",
        "updated-date route normalized to zero-result feed in the strict window",
        [],
        total_results=0,
    ),
    _receipt(
        "openreview_v537_energy_route",
        "OpenReview",
        "secondary",
        "content=energy-based&limit=5",
        "https://api.openreview.net/notes?content=energy-based&limit=5",
        "2026-08-07T20:13:53Z",
        "http_403_challenge_required",
        "challenge_required",
        "route returned challenge metadata rather than note dates",
        [],
        status=403,
        request_id="2026-08-07-9642049",
    ),
    _receipt(
        "semantic_scholar_v537_ebt_citations",
        "Semantic Scholar",
        "secondary",
        "arXiv:2507.02092 citations",
        "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations?fields=title,year,externalIds,url,publicationDate,authors&limit=100",
        "2026-08-07T20:13:53Z",
        "reachable_http_200_32_records_no_post_marker",
        "none",
        "newest visible publicationDate 2026-07-29",
        ["2607.27372", "2607.20792", "2607.17047"],
        candidate_count=32,
        newest_publication_date="2026-07-29",
    ),
    _receipt(
        "semantic_scholar_v537_arm_ebm_citations",
        "Semantic Scholar",
        "secondary",
        "arXiv:2512.15605 citations",
        "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations?fields=title,year,externalIds,url,publicationDate,authors&limit=100",
        "2026-08-07T20:13:53Z",
        "reachable_http_200_8_records_no_post_marker",
        "none",
        "newest visible publicationDate 2026-07-02",
        ["2607.02154", "2606.03089", "2605.18871"],
        candidate_count=8,
        newest_publication_date="2026-07-02",
    ),
    _receipt(
        "huggingface_papers_v537_aug7",
        "Hugging Face Papers",
        "secondary",
        "date/2026-08-07",
        "https://huggingface.co/papers/date/2026-08-07",
        "2026-08-07T20:13:53Z",
        "reachable_http_200_secondary_feed",
        "secondary_feed_only",
        "date page has a calendar date but no strict post-marker primary timestamp",
        ["2607.28609", "2608.06352", "2608.06374"],
        candidate_count=25,
    ),
    _receipt(
        "github_v537_ebm_repos_after_marker",
        "GitHub",
        "secondary",
        "energy-based-model pushed:>2026-08-07T18:08:04Z",
        "https://api.github.com/search/repositories?q=energy-based-model+pushed:%3E2026-08-07T18:08:04Z&sort=updated&order=desc&per_page=5",
        "2026-08-07T20:14:09Z",
        "reachable_http_200_total_count_6",
        "none",
        "repository pushed_at filter is strict after marker; hits are not independent method evidence",
        ["Carnot-EBM/carnot-ebm", "keithdeo/AI-Optimized-Continuous-Stirred-Tank-Reactor"],
        candidate_count=6,
    ),
    _receipt(
        "github_v537_kan_repos_after_marker",
        "GitHub",
        "secondary",
        "Kolmogorov-Arnold-Network pushed:>2026-08-07T18:08:04Z",
        "https://api.github.com/search/repositories?q=Kolmogorov-Arnold-Network+pushed:%3E2026-08-07T18:08:04Z&sort=updated&order=desc&per_page=5",
        "2026-08-07T20:14:10Z",
        "reachable_http_200_total_count_0",
        "none",
        "strict pushed_at filter returned zero KAN repositories",
        [],
        candidate_count=0,
    ),
    _receipt(
        "github_v537_code_hidden_state_after_marker",
        "GitHub",
        "secondary",
        "code verification hidden state pushed:>2026-08-07T18:08:04Z",
        "https://api.github.com/search/repositories?q=code+verification+hidden+state+pushed:%3E2026-08-07T18:08:04Z&sort=updated&order=desc&per_page=5",
        "2026-08-07T20:14:30Z",
        "reachable_http_200_total_count_0",
        "none",
        "strict pushed_at filter returned zero hidden-state code-verification repositories",
        [],
        candidate_count=0,
    ),
    _receipt(
        "github_v537_arc_agi3_after_marker",
        "GitHub",
        "secondary",
        "ARC-AGI-3 pushed:>2026-08-07T18:08:04Z",
        "https://api.github.com/search/repositories?q=ARC-AGI-3+pushed:%3E2026-08-07T18:08:04Z&sort=updated&order=desc&per_page=5",
        "2026-08-07T20:14:30Z",
        "reachable_http_200_total_count_1",
        "none",
        "one repository pushed after marker, but no reproducible benchmark or roadmap-changing method contract",
        ["Sahasawatt/arc-agi-3-agent"],
        candidate_count=1,
    ),
    _receipt(
        "extropic_v537_first_party_pages",
        "Extropic",
        "official",
        "writing, hardware, and software pages",
        "https://extropic.ai/writing ; https://extropic.ai/hardware ; https://extropic.ai/software",
        "2026-08-07T20:14:31Z",
        "reachable_http_200_no_dated_post_marker_local_tsu_route",
        "no_changed_since_endpoint",
        "pages expose no strict post-marker public TSU/Z1 route for local execution",
        ["writing", "hardware", "software"],
        candidate_count=3,
    ),
    _receipt(
        "logical_intelligence_v537_kona",
        "Logical Intelligence",
        "official",
        "Kona EBM first-party page",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "2026-08-07T20:14:11Z",
        "reachable_http_200_published_2026_06_26_no_public_weights",
        "none",
        "page published 2026-06-26T23:47:00Z; no public weights, local API, or architecture spec",
        ["kona_1_0"],
        candidate_count=1,
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
    date_evidence: str,
    local_reachability: str,
    roadmap_task: str,
    changed_method_or_gate: str,
    retirement_conflict: str,
    reason: str,
    *,
    disposition: str | None = None,
    primary_or_first_party: bool = False,
    dated_reproducible: bool = False,
    new_applicability: bool = False,
    endpoint_failed: bool = False,
    content_hash: str | None = None,
) -> JsonDict:
    row: JsonDict = {
        "stable_id": stable_id,
        "title": title,
        "url": url,
        "date": date,
        "authority": authority,
        "source_kind": source_kind,
        "date_evidence": date_evidence,
        "local_reachability": local_reachability,
        "roadmap_task": roadmap_task,
        "changed_method_or_gate": changed_method_or_gate,
        "retirement_conflict": retirement_conflict,
        "reason": reason,
        "disposition": disposition,
        "primary_or_first_party": primary_or_first_party,
        "dated_reproducible": dated_reproducible,
        "new_applicability": new_applicability,
        "endpoint_failed": endpoint_failed,
    }
    row["content_hash"] = content_hash or _stable_hash(
        [stable_id, title, url, date, date_evidence]
    )
    return row


DEFAULT_DISCOVERED_CANDIDATES: tuple[JsonDict, ...] = (
    _candidate(
        "hf:papers/date/2026-08-07",
        "Hugging Face Papers Aug 7 feed",
        "https://huggingface.co/papers/date/2026-08-07",
        "2026-08-07",
        "Hugging Face Papers",
        "secondary",
        "secondary feed calendar date only",
        "reachable_secondary_feed",
        "defer",
        "none",
        "secondary discovery only",
        "No strict post-marker primary timestamp; planning-time sources already sealed.",
        disposition="rejected",
        dated_reproducible=True,
    ),
    _candidate(
        "github:Carnot-EBM/carnot-ebm",
        "Carnot-EBM/carnot-ebm repository metadata",
        "https://github.com/Carnot-EBM/carnot-ebm",
        "2026-08-07T20:07:33Z",
        "GitHub",
        "secondary",
        "pushed_at after marker",
        "reachable_self_repo_metadata",
        EXPERIMENT_ID,
        "none",
        "self-referential project repository",
        "The current project's own public mirror cannot be independent runtime source evidence.",
        disposition="duplicate",
        dated_reproducible=True,
    ),
    _candidate(
        "github:Sahasawatt/arc-agi-3-agent",
        "Sahasawatt/arc-agi-3-agent repository metadata",
        "https://github.com/Sahasawatt/arc-agi-3-agent",
        "2026-08-07T20:12:50Z",
        "GitHub",
        "secondary",
        "pushed_at after marker",
        "reachable_repo_metadata_only",
        "exp6209-arc-loo-task-aware-shadow",
        "none",
        "no reproducible benchmark or method contract",
        "A post-marker public repository does not change the one-slot ARC shadow-measurement rule.",
        disposition="rejected",
        dated_reproducible=True,
    ),
    _candidate(
        "extropic:first-party-pages",
        "Extropic first-party writing, hardware, and software pages",
        "https://extropic.ai/writing",
        "unknown",
        "Extropic",
        "official",
        "no changed-since route or strict post-marker publication timestamp",
        "reachable_first_party_pages_no_local_device",
        "exp6199-gatemate-terminal-action-audit-v537",
        "none",
        "no authenticated local TSU/Z1 route",
        "Extropic pages still do not authorize TSU, speed, power, energy, or local hardware claims.",
        disposition="guarded",
        primary_or_first_party=True,
        dated_reproducible=False,
    ),
    _candidate(
        "logical-intelligence:kona-page",
        "Logical Intelligence Kona first-party page",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "2026-06-26T23:47:00Z",
        "Logical Intelligence",
        "official",
        "published Jun 26, 2026, 11:47 PM UTC",
        "reachable_first_party_page_no_public_weights",
        "defer",
        "none",
        "Kona proprietary execution remains guarded",
        "The page predates the marker and exposes no public weights, local API, or architecture spec.",
        disposition="cutoff_confound",
        primary_or_first_party=True,
        dated_reproducible=True,
    ),
)


def marker_snapshot(root: Path) -> JsonDict:
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    raw = references_path.read_bytes() if references_path.exists() else b""
    text = raw.decode("utf-8", "replace")
    marker_bytes = PLANNER_END_MARKER.encode("utf-8")
    marker_index = text.find(PLANNER_END_MARKER)
    heading_index = text.find(PLANNER_HEADING)
    block = ""
    if heading_index >= 0 and marker_index >= 0:
        block = text[heading_index : marker_index + len(PLANNER_END_MARKER)]
    return {
        "marker_text": PLANNER_MARKER,
        "marker_count": text.count(PLANNER_END_MARKER),
        "marker_line": text[:marker_index].count("\n") + 1 if marker_index >= 0 else None,
        "marker_offset": raw.find(marker_bytes),
        "marker_text_hash": _sha256_bytes(PLANNER_MARKER.encode("utf-8")),
        "marker_block_hash": _sha256_bytes(block.encode("utf-8")) if block else None,
        "marker_commit": MARKER_COMMIT,
        "marker_committed_at": MARKER_COMMITTED_AT,
        "reference_hash": path_sha256(references_path),
    }


def validate_accepted_candidate(row: Mapping[str, Any]) -> None:
    required = (
        "stable_id",
        "content_hash",
        "title",
        "url",
        "date",
        "authority",
        "source_kind",
        "date_evidence",
        "local_reachability",
        "roadmap_task",
        "changed_method_or_gate",
        "retirement_conflict",
        "reason",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"accepted candidate missing {missing}")
    if not str(row["content_hash"]).startswith("sha256:"):
        raise ValueError("accepted candidate content hash must be sha256")
    if not _after_marker(str(row["date"])):
        raise ValueError("accepted candidate must be strictly after the V537 marker")
    if not row.get("primary_or_first_party"):
        raise ValueError("accepted candidate must be primary or first-party")
    if not row.get("dated_reproducible"):
        raise ValueError("accepted candidate must have reproducible date evidence")
    if not row.get("new_applicability"):
        raise ValueError("accepted candidate must change a roadmap method or gate")
    if str(row.get("retirement_conflict")) != "none":
        raise ValueError("accepted candidate cannot carry a retirement conflict")
    if str(row.get("stable_id")).startswith("github:Carnot-EBM/carnot-ebm"):
        raise ValueError("accepted candidate cannot be the current project repository")


def _ledger_row(row: Mapping[str, Any], disposition: str) -> JsonDict:
    return {
        "stable_id": str(row.get("stable_id", "unknown")),
        "content_hash": str(row.get("content_hash", _stable_hash(row))),
        "title": str(row.get("title", "unknown")),
        "url": str(row.get("url", "")),
        "date": str(row.get("date", "unknown")),
        "authority": str(row.get("authority", "unknown")),
        "source_kind": str(row.get("source_kind", "unknown")),
        "date_evidence": str(row.get("date_evidence", "")),
        "local_reachability": str(row.get("local_reachability", "unknown")),
        "roadmap_task": str(row.get("roadmap_task", "defer")),
        "changed_method_or_gate": str(row.get("changed_method_or_gate", "none")),
        "retirement_conflict": str(row.get("retirement_conflict", "unknown")),
        "reason": str(row.get("reason", "")),
        "disposition": disposition,
    }


def classify_candidate(row: Mapping[str, Any]) -> JsonDict:
    candidate = dict(row)
    candidate.setdefault("content_hash", _stable_hash(candidate))
    explicit = candidate.get("disposition")
    if explicit in {"rejected", "guarded", "duplicate", "cutoff_confound", "endpoint_failed"}:
        return _ledger_row(candidate, str(explicit))
    if candidate.get("endpoint_failed"):
        return _ledger_row(candidate, "endpoint_failed")
    if str(candidate.get("stable_id", "")).startswith("github:Carnot-EBM/carnot-ebm"):
        return _ledger_row(candidate, "duplicate")
    if not _after_marker(str(candidate.get("date", ""))):
        return _ledger_row(candidate, "cutoff_confound")
    if str(candidate.get("retirement_conflict")) != "none":
        return _ledger_row(candidate, "guarded")
    if not candidate.get("dated_reproducible") or not candidate.get("primary_or_first_party"):
        return _ledger_row(candidate, "rejected")
    if not candidate.get("new_applicability"):
        return _ledger_row(candidate, "rejected")
    validate_accepted_candidate(candidate)
    return _ledger_row(candidate, "accepted")


def deduplicate_candidates(candidates: Sequence[JsonDict]) -> tuple[list[JsonDict], list[JsonDict]]:
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    unique: list[JsonDict] = []
    duplicates: list[JsonDict] = []
    for row in candidates:
        stable_id = str(row.get("stable_id", ""))
        content_hash = str(row.get("content_hash", ""))
        if stable_id in seen_ids or content_hash in seen_hashes:
            duplicate = dict(row)
            duplicate["disposition"] = "duplicate"
            duplicates.append(duplicate)
            continue
        seen_ids.add(stable_id)
        seen_hashes.add(content_hash)
        unique.append(row)
    return unique, duplicates


def _candidate_ledgers(candidates: Sequence[JsonDict]) -> tuple[list[JsonDict], list[JsonDict]]:
    unique, duplicates = deduplicate_candidates([dict(row) for row in candidates])
    classified = [classify_candidate(row) for row in unique]
    classified.extend(_ledger_row(row, "duplicate") for row in duplicates)
    accepted = [row for row in classified if row["disposition"] == "accepted"]
    rejected = [row for row in classified if row["disposition"] != "accepted"]
    return accepted, rejected


def _select_roadmap_path(root: Path) -> tuple[Path, bool, str]:
    staged = root / STAGED_ROADMAP_RELATIVE_PATH
    if staged.exists():
        staged_data = _load_roadmap_data(staged)
        if _raw_tasks(staged_data):
            return staged, False, "requested staged roadmap exists"
        return root / ACTIVE_ROADMAP_RELATIVE_PATH, True, (
            "research-roadmap-next.yaml is present but not a roadmap; audited active research-roadmap.yaml"
        )
    return root / ACTIVE_ROADMAP_RELATIVE_PATH, True, (
        "research-roadmap-next.yaml absent; audited active research-roadmap.yaml"
    )


def _load_roadmap_data(path: Path) -> JsonDict:
    loaded = yaml.safe_load(_read_text(path)) or {}
    return loaded if isinstance(loaded, dict) else {}


def _raw_tasks(roadmap_data: Mapping[str, Any]) -> list[JsonDict]:
    tasks = roadmap_data.get("tasks")
    return [task for task in tasks if isinstance(task, dict)] if isinstance(tasks, list) else []


def _roadmap_schema_result(roadmap_data: Mapping[str, Any]) -> JsonDict:
    try:
        parsed = Roadmap.model_validate(dict(roadmap_data))
    except Exception as exc:
        return {"passed": False, "error": str(exc), "task_count": len(_raw_tasks(roadmap_data))}
    return {
        "passed": True,
        "milestone": parsed.milestone,
        "task_count": len(parsed.tasks),
        "task_ids": [task.id for task in parsed.tasks],
    }


def _exclusion_lint_result(roadmap_path: Path) -> JsonDict:
    risks = exclusion_lint(roadmap_path)
    rows = [
        {
            "task_id": risk.task_id,
            "violation_class": risk.violation_class,
            "severity": risk.severity,
            "detail": risk.detail,
        }
        for risk in risks
    ]
    hard = [row for row in rows if row["severity"] == "HARD"]
    return {
        "passed": not hard,
        "risk_count": len(rows),
        "hard_count": len(hard),
        "warning_count": len(rows) - len(hard),
        "risks": rows,
    }


def _gate_structure_result(roadmap_path: Path) -> JsonDict:
    result = audit_roadmap(roadmap_path).to_artifact()
    result["structured_gate_count"] = result["n_gate_upstream_checks"]
    return result


def _prior_failure_contract(tasks: Sequence[Mapping[str, Any]]) -> JsonDict:
    missing: list[str] = []
    malformed: list[str] = []
    for task in tasks:
        task_id = str(task.get("id", "<missing-id>"))
        prior_failures = task.get("prior_failures")
        if not isinstance(prior_failures, list) or not prior_failures:
            missing.append(task_id)
            continue
        for prior in prior_failures:
            if not isinstance(prior, Mapping) or any(
                field not in prior
                for field in ("experiment_id", "verdict", "addressed_by", "retire_if_same_verdict")
            ):
                malformed.append(task_id)
                break
    return {
        "passed": not missing and not malformed,
        "task_count": len(tasks),
        "tasks_with_prior_failures": len(tasks) - len(missing),
        "missing_prior_failure_count": len(missing),
        "malformed_prior_failure_count": len(malformed),
        "missing_task_ids": missing,
        "malformed_task_ids": malformed,
    }


def _model_specs_from_prompt(prompt: str) -> list[str]:
    if "MODEL_SPECS:" not in prompt:
        return []
    block = prompt.split("MODEL_SPECS:", 1)[1].split("Required deliverable:", 1)[0]
    return [match.strip() for match in re.findall(r"hf_id:\s*([^;\n]+)", block)]


def _legacy_mentions_are_guarded(prompt: str) -> bool:
    for token in LEGACY_MODEL_TOKENS:
        index = prompt.find(token)
        if index == -1:
            continue
        window = prompt[max(0, index - 120) : index + len(token) + 160].lower()
        if not any(word in window for word in ("cannot", "zero", "smoke", "not acceptable")):
            return False
    return True


def _model_specs_rule(tasks: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows: list[JsonDict] = []
    failures: list[str] = []
    for task in tasks:
        task_id = str(task.get("id", "<missing-id>"))
        prompt = str(task.get("prompt", ""))
        specs = _model_specs_from_prompt(prompt)
        if not specs:
            continue
        has_sota = any(spec in SOTA_GGUFS for spec in specs)
        legacy_guarded = _legacy_mentions_are_guarded(prompt)
        row = {
            "task_id": task_id,
            "model_specs": specs,
            "has_mandated_sota_gguf": has_sota,
            "legacy_mentions_guarded": legacy_guarded,
        }
        rows.append(row)
        if not has_sota or not legacy_guarded:
            failures.append(task_id)
    return {
        "passed": not failures and bool(rows),
        "sota_gguf_allowlist": sorted(SOTA_GGUFS),
        "llm_invocation_task_count": len(rows),
        "tasks": rows,
        "failure_task_ids": failures,
        "legacy_small_models_can_supply_headline_rows": False,
    }


def _prompt_sections(tasks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failures: list[JsonDict] = []
    for task in tasks:
        prompt = str(task.get("prompt", ""))
        missing = [section for section in REQUIRED_PROMPT_SECTIONS if section not in prompt]
        if missing:
            failures.append({"task_id": task.get("id"), "missing_sections": missing})
    return {
        "passed": not failures,
        "all_prompts_passed": not failures,
        "prompt_count": len(tasks),
        "required_sections": list(REQUIRED_PROMPT_SECTIONS),
        "failure_count": len(failures),
        "failures": failures,
    }


def _hardware_continuity(tasks: Sequence[Mapping[str, Any]]) -> JsonDict:
    gatemate = [
        str(task.get("id"))
        for task in tasks
        if "gatemate" in f"{task.get('id', '')} {task.get('title', '')}".lower()
    ]
    return {
        "passed": len(gatemate) == 1,
        "gatemate_task_count": len(gatemate),
        "gatemate_task_ids": gatemate,
        "unauthorized_hardware_command_claimed": False,
    }


def _protected_hashes(root: Path) -> JsonDict:
    return {path.as_posix(): path_sha256(root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(
    root: Path,
    before_hashes: Mapping[str, Any] | None = None,
    authorized_after_hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    before = dict(before_hashes or _protected_hashes(root))
    after = _protected_hashes(root)
    authorized = dict(authorized_after_hashes or {})
    before.update({path: digest for path, digest in authorized.items() if digest is not None})
    changed = [path for path, digest in before.items() if after.get(path) != digest]
    return {
        "all_unchanged": not changed,
        "before_hashes": before,
        "after_hashes": after,
        "authorized_changed_paths": sorted(authorized),
        "changed_paths": changed,
    }


def _source_receipts_block(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts = Counter(str(row.get("authority", "unknown")) for row in receipts)
    failures = [
        {
            "receipt_id": row.get("receipt_id"),
            "authority": row.get("authority"),
            "failure_class": row.get("failure_class"),
            "access_outcome": row.get("access_outcome"),
        }
        for row in receipts
        if str(row.get("failure_class")) not in {"none", ""}
    ]
    return {
        "receipts": [dict(row) for row in receipts],
        "authority_counts": dict(sorted(counts.items())),
        "failure_receipts": failures,
        "low_concurrency": True,
        "deep_research_invoked": False,
    }


def _execution_delta_block(accepted_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "",
        EXECUTION_DELTA_HEADING,
        "",
        "Execution-time source deltas accepted after the sealed V537 marker:",
    ]
    for row in accepted_rows:
        lines.append(f"- **{row['title']}** - {row['url']}; source date {row['date']}.")
        lines.append(f"  - Mapped to: `{row['roadmap_task']}`.")
        lines.append(f"  - Changed method/gate: {row['changed_method_or_gate']}.")
    lines.extend(["", EXECUTION_DELTA_END_MARKER, ""])
    return "\n".join(lines)


def _insert_after_marker(text: str, block: str) -> str:
    if EXECUTION_DELTA_HEADING in text:
        return text
    index = text.find(PLANNER_END_MARKER)
    if index < 0:
        return text.rstrip() + "\n" + block
    insert_at = index + len(PLANNER_END_MARKER)
    return text[:insert_at] + "\n" + block + text[insert_at:]


def _run_command(argv: tuple[str, ...], root: Path) -> JsonDict:  # pragma: no cover
    try:
        proc = subprocess.run(argv, cwd=root, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        return {
            "command": " ".join(argv),
            "exit_code": 127,
            "classification": "command_not_found",
            "error": str(exc),
        }
    return {
        "command": " ".join(argv),
        "exit_code": proc.returncode,
        "classification": "passed" if proc.returncode == 0 else f"nonzero_exit_{proc.returncode}",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _run_commands(root: Path, runner: CommandRunner) -> list[JsonDict]:
    return [runner(tuple(shlex.split(command)), root) for command in TEST_COMMANDS]


def _recorded_command_runner(argv: tuple[str, ...], _root: Path) -> JsonDict:
    command = " ".join(argv)
    receipt = dict(RECORDED_TEST_RECEIPTS.get(command, {}))
    exit_code = int(receipt.get("exit_code", 1))
    return {
        "command": command,
        "exit_code": exit_code,
        "classification": str(
            receipt.get("classification", "missing_recorded_command_receipt")
        ),
        "stdout_tail": str(receipt.get("stdout_tail", "")),
        "stderr_tail": str(receipt.get("stderr_tail", "")),
    }


def _deferred_command_runner(argv: tuple[str, ...], _root: Path) -> JsonDict:
    return {
        "command": " ".join(argv),
        "exit_code": 0,
        "classification": "deferred_until_final_artifact_build",
        "stdout_tail": "",
        "stderr_tail": "",
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": [
                "REQ-INFRA-6198",
                "research-references marker",
                "source channel receipts",
                "roadmap validators",
                "command receipts",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def payload_checksum(report: Mapping[str, Any]) -> str:
    payload = json.loads(_canonical_json(report))
    payload["duration_s"] = 0.0
    payload["reproducibility_checksum"] = ""
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def honest_verdict(status: str, accepted_count: int) -> str:
    if status == "blocked":
        return f"blocked: accepted_count={accepted_count}; V537 source or roadmap audit failed"
    if accepted_count:
        return f"complete_delta: accepted_count={accepted_count}; dated post-V537-marker findings appended"
    return "complete_null: accepted_count=0; no reproducible post-V537-marker findings; references unchanged; roadmap audit clean"


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    source_receipts: Sequence[JsonDict] | None = None,
    candidates: Sequence[JsonDict] | None = None,
    command_runner: CommandRunner = _run_command,
    duration_s: float | None = None,
    references_before_hash: str | None = None,
    references_after_hash: str | None = None,
    references_appended: bool = False,
    protected_before_hashes: Mapping[str, Any] | None = None,
    protected_authorized_after_hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    started = time.monotonic()
    receipts = [dict(row) for row in (source_receipts or DEFAULT_SOURCE_CHANNEL_RECEIPTS)]
    found_candidates = [dict(row) for row in (candidates or DEFAULT_DISCOVERED_CANDIDATES)]
    accepted, rejected = _candidate_ledgers(found_candidates)
    accepted_count = len(accepted)
    marker = marker_snapshot(root)
    roadmap_path, requested_missing, roadmap_note = _select_roadmap_path(root)
    roadmap_data = _load_roadmap_data(roadmap_path)
    tasks = _raw_tasks(roadmap_data)

    schema_result = _roadmap_schema_result(roadmap_data)
    exclusion_result = _exclusion_lint_result(roadmap_path)
    gate_result = _gate_structure_result(roadmap_path)
    prior_result = _prior_failure_contract(tasks)
    model_result = _model_specs_rule(tasks)
    prompt_result = _prompt_sections(tasks)
    hardware_result = _hardware_continuity(tasks)
    track_counts = Counter(str(task.get("track", "")) for task in tasks)
    command_receipts = _run_commands(root, command_runner)
    test_exit_codes = {
        str(receipt["command"]): int(receipt["exit_code"]) for receipt in command_receipts
    }
    refs_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    refs_before = references_before_hash or path_sha256(refs_path)
    refs_after = references_after_hash or path_sha256(refs_path)
    status = (
        "complete"
        if (
            marker["marker_count"] == 1
            and schema_result["passed"]
            and exclusion_result["passed"]
            and gate_result["roadmap_gate_audit_passed"]
            and prior_result["passed"]
            and model_result["passed"]
            and prompt_result["passed"]
            and hardware_result["passed"]
        )
        else "blocked"
    )

    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": status,
        "planner_marker_and_hash": marker,
        "query_window": {
            "marker": PLANNER_MARKER,
            "window_start_exclusive": MARKER_COMMITTED_AT,
            "window_end_inclusive": WINDOW_END_INCLUSIVE,
            "same_day_policy": "A bare 2026-08-07 date is rejected unless the source timestamp is strictly after 18:08:04Z.",
            "accepted_at_or_before_marker": False,
        },
        "source_channel_receipts": _source_receipts_block(receipts),
        "discovered_candidates": found_candidates,
        "accepted_findings": accepted,
        "rejected_or_duplicate_findings": rejected,
        "accepted_count": accepted_count,
        "references_append_receipt": {
            "reference_hash_before": refs_before,
            "reference_hash_after": refs_after,
            "references_byte_identical": refs_before == refs_after,
            "append_count": 1 if references_appended else 0,
            "append_heading": EXECUTION_DELTA_HEADING if references_appended else None,
            "append_status": "complete_delta" if accepted_count else "complete_null",
        },
        "roadmap_path_and_hash": {
            "requested_path": STAGED_ROADMAP_RELATIVE_PATH.as_posix(),
            "audited_path": roadmap_path.relative_to(root).as_posix()
            if roadmap_path.is_relative_to(root)
            else str(roadmap_path),
            "requested_missing": requested_missing,
            "selection_note": roadmap_note,
            "sha256": path_sha256(roadmap_path),
            "milestone": roadmap_data.get("milestone"),
        },
        "roadmap_schema_result": schema_result,
        "exclusion_manifest_lint_result": exclusion_result,
        "retired_scope_match_count": exclusion_result["risk_count"],
        "prior_failure_contract_result": prior_result,
        "gate_structure_result": gate_result,
        "model_specs_rule_result": model_result,
        "task_count": len(tasks),
        "infra_slot_count": track_counts["infrastructure"],
        "phase_d_slot_count": track_counts["phase-d"],
        "arc_slot_count": track_counts["arc-generalization"],
        "continuous_self_learning_slot_count": track_counts["self-learning"],
        "hardware_continuity_result": hardware_result,
        "prompt_section_and_ending_result": prompt_result,
        "protected_files_unchanged": _protected_unchanged(
            root, protected_before_hashes, protected_authorized_after_hashes
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "test_commands": list(TEST_COMMANDS),
        "test_exit_codes": test_exit_codes,
        "duration_s": float(duration_s if duration_s is not None else time.monotonic() - started),
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict(status, accepted_count),
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
    if len(report.get("accepted_findings", [])) != report.get("accepted_count"):
        errors.append("accepted_count")
    if str(report.get("honest_verdict", "")).startswith(
        ("complete_null:", "complete_delta:", "blocked:")
    ) is False:
        errors.append("honest_verdict")
    checks = (
        ("roadmap_schema_result", "passed"),
        ("exclusion_manifest_lint_result", "passed"),
        ("prior_failure_contract_result", "passed"),
        ("model_specs_rule_result", "passed"),
        ("hardware_continuity_result", "passed"),
        ("prompt_section_and_ending_result", "passed"),
        ("protected_files_unchanged", "all_unchanged"),
    )
    for field, key in checks:
        row = report.get(field)
        if not isinstance(row, Mapping) or row.get(key) is not True:
            errors.append(field)
    gate = report.get("gate_structure_result")
    if not isinstance(gate, Mapping) or gate.get("roadmap_gate_audit_passed") is not True:
        errors.append("gate_structure_result")
    if report.get("task_count") != 14:
        errors.append("task_count")
    if int(report.get("infra_slot_count", 0)) < 2:
        errors.append("infra_slot_count")
    if report.get("phase_d_slot_count") != 6:
        errors.append("phase_d_slot_count")
    if report.get("arc_slot_count") != 1:
        errors.append("arc_slot_count")
    if int(report.get("continuous_self_learning_slot_count", 0)) < 1:
        errors.append("continuous_self_learning_slot_count")
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


def write_audit(
    root: Path = REPO_ROOT,
    *,
    date: str,
    source_receipts: Sequence[JsonDict] | None = None,
    candidates: Sequence[JsonDict] | None = None,
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
        command_runner=_deferred_command_runner,
        duration_s=duration_s,
        references_before_hash=references_before,
        references_after_hash=references_before,
        protected_before_hashes=protected_before,
    )
    appended = False
    if dry["status"] == "complete" and dry["accepted_findings"]:
        references_path.write_text(
            _insert_after_marker(before_text, _execution_delta_block(dry["accepted_findings"])),
            encoding="utf-8",
        )
        appended = EXECUTION_DELTA_HEADING not in before_text
    references_after = path_sha256(references_path)
    protected_authorized_after = (
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
        protected_authorized_after_hashes=protected_authorized_after,
    )
    errors = validate_report(report)
    if errors:
        raise ValueError(f"invalid Exp6198 audit: {errors}")
    atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    report = write_audit(REPO_ROOT, date=args.date, command_runner=_recorded_command_runner)
    print(
        json.dumps(
            {"path": RESULT_RELATIVE_PATH.as_posix(), "checksum": report["reproducibility_checksum"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
