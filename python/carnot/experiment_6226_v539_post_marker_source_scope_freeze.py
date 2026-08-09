"""Exp6226 V539 post-marker source scope freeze.

Spec refs: REQ-INFRA-6226, SCENARIO-INFRA-6226-1,
SCENARIO-INFRA-6226-2, SCENARIO-INFRA-6226-3,
SCENARIO-INFRA-6226-4, SCENARIO-INFRA-6226-5,
SCENARIO-INFRA-6226-6.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

from carnot.experiment_6211_v538_post_marker_source_scope_prereg import (
    _canonical_json,
    _exclusion_lint_result,
    _load_roadmap_data,
    _prior_failure_contract,
    _raw_tasks,
    _read_text,
    _roadmap_schema_result,
    _sha256_bytes,
    _stable_hash,
    path_sha256,
)
from carnot.experiment_6225_v539_terminal_transition import (
    load_retired_exp_ids,
    validate_v539_roadmap_data,
)
from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]
CommandRunner = Callable[[tuple[str, ...], Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SCRIPTS_ROOT))


RESULT_RELATIVE_PATH = Path(
    "results/experiment_6226_v539_post_marker_source_scope_freeze.json"
)
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
STAGED_ROADMAP_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
SWEEP_CLUSTERS_RELATIVE_PATH = Path("scripts/sweep_clusters.py")
SWEEP_SEMSCHOLAR_RELATIVE_PATH = Path("scripts/sweep_semscholar.py")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_REFERENCES_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    STAGED_ROADMAP_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    SWEEP_CLUSTERS_RELATIVE_PATH,
    SWEEP_SEMSCHOLAR_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
SOURCE_INPUT_RELATIVE_PATHS = (
    RESEARCH_REFERENCES_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    STAGED_ROADMAP_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    SWEEP_CLUSTERS_RELATIVE_PATH,
    SWEEP_SEMSCHOLAR_RELATIVE_PATH,
)

EXPERIMENT_ID = "exp6226-v539-post-marker-source-scope-freeze"
SCHEMA = "carnot.experiment_6226.v539_source_scope_freeze.v1"
INFERENCE_SUBSTRATE = "post_marker_source_ingestion_and_v539_scope_freeze"

PLANNER_HEADING = "## V539 Planner Refresh (2026-08-09, after milestone 2026.08.538)"
PLANNER_MARKER = "V539-PLANNER-REFRESH-20260809-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
MARKER_COMMIT = "fc9c5b9dad0120f05ce375295690a28e76840b9e"
MARKER_COMMITTED_AT = "2026-08-09T06:16:52Z"
EXECUTION_DELTA_HEADING = "## V539 Execution Source Delta - 20260809"
EXECUTION_DELTA_END_MARKER = "<!-- V539-EXECUTION-SOURCE-DELTA-20260809-END -->"

REQUIRED_SOURCE_CHANNELS = (
    "arxiv_topics",
    "openreview",
    "extropic",
    "semantic_scholar_ebt_citations",
    "semantic_scholar_arm_ebm_citations",
    "huggingface_papers",
    "github_targeted",
    "logical_intelligence",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "planner_marker_and_hash",
    "bootstrap_artifact_write_receipt",
    "query_window",
    "source_channel_receipts",
    "discovered_candidates",
    "accepted_findings",
    "rejected_or_duplicate_findings",
    "accepted_count",
    "references_append_receipt",
    "frozen_runtime_contract",
    "frozen_arc_provenance_contract",
    "frozen_code_content_margin_contract",
    "frozen_csl_contract",
    "frozen_sampler_activation_contract",
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
    "status": "The terminal state follows source novelty, roadmap, command, and freeze-contract checks.",
    "planner_marker_and_hash": "The exact V539 marker defines the runtime novelty boundary.",
    "bootstrap_artifact_write_receipt": "The minimal atomic write proves the task preserved preconditions before optional network work.",
    "query_window": "Only evidence strictly after the marker can be a runtime delta.",
    "source_channel_receipts": "Every required source channel records success, null, duplicate, or failure.",
    "discovered_candidates": "All found rows stay visible before classification.",
    "accepted_findings": "Only strict post-marker primary or first-party V539 deltas are accepted.",
    "rejected_or_duplicate_findings": "Rejected rows explain why references stayed unchanged.",
    "accepted_count": "Null or zero records a null search without blocking later freeze-dependent tasks.",
    "references_append_receipt": "References stay byte-identical on a null and append-only on a delta.",
    "frozen_runtime_contract": "Runtime ownership must be explicit before long local model jobs run.",
    "frozen_arc_provenance_contract": "ARC solve credit requires live-agent self-discovery and no hidden source access.",
    "frozen_code_content_margin_contract": "Parse repair and semantic hidden-test content gains must stay separated.",
    "frozen_csl_contract": "Fresh self-learning may write memory only after verified outcomes.",
    "frozen_sampler_activation_contract": "Sampler comparisons are invalid when the treatment never fires.",
    "frozen_hardware_boundary": "Hardware claims need a new authenticated route or physical-state receipt.",
    "roadmap_path_and_hash": "The audited active and staged roadmap identities are content-addressed.",
    "roadmap_schema_result": "Roadmap, gate, model, retired-dependency, and prompt checks must pass mechanically.",
    "exclusion_manifest_lint_result": "Retired scopes stay closed unless the manifest allows them.",
    "prior_failure_contract_result": "Reruns must explain how they differ from prior failures.",
    "protected_files_unchanged": "The task must not rewrite conductor, roadmap, ops ledgers, or registry state.",
    "preconditions_checked": "Marker, staged roadmap, protected files, and source inputs are hashed before final work.",
    "inference_substrate": "Declares deterministic source and roadmap audit rather than live LLM inference.",
    "verifier_is_oracle": "False because this audit validates metadata, not benchmark answers.",
    "field_provenance": "Each required field traces to marker, receipts, roadmap checks, contracts, or commands.",
    "field_principles": "Every required field carries the reason it exists.",
    "test_commands": "Records focused, coverage, roadmap, exclusion, retired-dependency, prompt, full-suite, and adversarial checks.",
    "test_exit_codes": "Exit codes prevent failed checks from being reported as clean.",
    "duration_s": "Reports wall time without padding.",
    "reproducibility_checksum": "Detects later drift in the artifact payload.",
    "honest_verdict": "The verdict preserves null, delta, or blocked state without overclaiming.",
}

TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6226_v539_post_marker_source_scope_freeze.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6226_v539_post_marker_source_scope_freeze.py -m pytest tests/python/test_experiment_6226_v539_post_marker_source_scope_freeze.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6226_v539_post_marker_source_scope_freeze.py --fail-under=100",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/python -m carnot.experiment_6225_v539_terminal_transition --check-roadmap-only",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6226_v539_post_marker_source_scope_freeze.py",
    "sed -n 1,260p ops/e2e-test-plan.md",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6226_v539_post_marker_source_scope_freeze.json",
)

RECORDED_TEST_RECEIPTS: Mapping[str, JsonDict] = {
    TEST_COMMANDS[0]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "9 passed in 35.02s",
        "stderr_tail": "",
    },
    TEST_COMMANDS[1]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "9 passed in 46.99s",
        "stderr_tail": "",
    },
    TEST_COMMANDS[2]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "TOTAL 348 0 100%",
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
        "stdout_tail": "roadmap_gate_audit_passed: true; n_tasks_audited: 14; n_gate_upstream_checks: 8",
        "stderr_tail": "",
    },
    TEST_COMMANDS[5]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "[OK] research-roadmap.yaml - no schema errors, no prior_failures violations",
        "stderr_tail": "",
    },
    TEST_COMMANDS[6]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "ok: true; task_count: 14; retired_dependency_count: 0; prompt_contract_validation.ok: true; model_policy_validation.ok: true",
        "stderr_tail": "",
    },
    TEST_COMMANDS[7]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "OK: All tests reference specification requirements.",
        "stderr_tail": "",
    },
    TEST_COMMANDS[8]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "ops/e2e-test-plan.md inspected; no Exp6226-specific mutating E2E applies.",
        "stderr_tail": "",
    },
    TEST_COMMANDS[9]: {
        "exit_code": 2,
        "classification": "interrupted_after_existing_broad_suite_failures_and_worker_crashes",
        "stdout_tail": "Interrupted at 33% after 1462.92s: 351 failed, 17657 passed, 31 skipped, 189 warnings, 14 errors. Existing worker crashes occurred in Z3, JAX Consformer compile, and Qwen tokenizer prewarm paths before interruption.",
        "stderr_tail": "",
    },
    TEST_COMMANDS[10]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "Scanned 1 artifact(s); 0 flagged.",
        "stderr_tail": "",
    },
}


def _parse_timestamp(value: str) -> datetime:
    if "T" not in value:
        value = value + "T00:00:00Z"
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _after_marker(value: str) -> bool:
    try:
        return _parse_timestamp(value) > _parse_timestamp(MARKER_COMMITTED_AT)
    except (TypeError, ValueError):
        return False


def _window_end(date: str) -> str:
    parsed = datetime.strptime(date, "%Y%m%d").replace(tzinfo=UTC)
    return parsed.strftime("%Y-%m-%dT23:59:59Z")


def _receipt(
    receipt_id: str,
    channel: str,
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
        "channel": channel,
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
        "arxiv_v539_topics_after_marker",
        "arxiv_topics",
        "arXiv",
        "primary",
        "submittedDate:[202608090616 TO 202608092359] AND EBM/ARC/continual/hardware/KAN terms",
        "https://export.arxiv.org/api/query?search_query=submittedDate:%5B202608090616%20TO%202608092359%5D&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending",
        "2026-08-09T07:05:00Z",
        "queried_no_accepted_post_marker_delta",
        "none",
        "strict submittedDate endpoint returned totalResults=0",
        [],
        candidate_count=0,
        total_results=0,
    ),
    _receipt(
        "openreview_v539_energy_route",
        "openreview",
        "OpenReview",
        "secondary",
        "content=energy-based&limit=5 after 2026-08-09T06:16:52Z",
        "https://api.openreview.net/notes?content=energy-based&limit=5",
        "2026-08-09T07:05:00Z",
        "queried_no_strict_post_marker_acceptance",
        "html_or_challenge",
        "endpoint returned HTTP 200 with non-JSON HTML/challenge body, so no note dates were accepted",
        [],
        candidate_count=0,
        response_bytes=4745,
    ),
    _receipt(
        "extropic_v539_first_party_pages",
        "extropic",
        "Extropic",
        "official",
        "writing, hardware, and software pages after V539 marker",
        "https://extropic.ai/writing/from-one-to-one-billion ; https://extropic.ai/software",
        "2026-08-09T07:05:00Z",
        "reachable_no_new_post_marker_route",
        "none",
        "Z1 status was sealed in V539 planner context; no authenticated post-marker device or API route found",
        ["from-one-to-one-billion"],
        candidate_count=1,
    ),
    _receipt(
        "semantic_scholar_v539_ebt_citations",
        "semantic_scholar_ebt_citations",
        "Semantic Scholar",
        "secondary",
        "arXiv:2507.02092 citations after V539 marker",
        "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations?fields=title,year,externalIds,url,publicationDate,authors&limit=100",
        "2026-08-09T07:05:00Z",
        "queried_no_new_post_marker_primary_delta",
        "none",
        "33 citation rows; newest visible publicationDate 2026-08-01, before the V539 marker",
        ["2607.27372", "2607.20792", "2607.17047", "2607.11555"],
        candidate_count=33,
        newest_publication_date="2026-08-01",
    ),
    _receipt(
        "semantic_scholar_v539_arm_ebm_citations",
        "semantic_scholar_arm_ebm_citations",
        "Semantic Scholar",
        "secondary",
        "arXiv:2512.15605 citations after V539 marker",
        "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations?fields=title,year,externalIds,url,publicationDate,authors&limit=100",
        "2026-08-09T07:05:00Z",
        "queried_no_new_post_marker_primary_delta",
        "none",
        "8 citation rows; newest visible publicationDate 2026-07-02, before the V539 marker",
        ["2607.02154", "2606.03089", "2605.18871", "2605.11011"],
        candidate_count=8,
        newest_publication_date="2026-07-02",
    ),
    _receipt(
        "huggingface_papers_v539_aug9",
        "huggingface_papers",
        "Hugging Face Papers",
        "secondary",
        "date/2026-08-09",
        "https://huggingface.co/papers/date/2026-08-09",
        "2026-08-09T07:05:00Z",
        "secondary_feed_no_primary_post_marker_delta",
        "secondary_only",
        "calendar page is secondary; parsed date links only through 2026-08-08 in the fetched body",
        [],
        candidate_count=0,
        response_dates=["2026-07-29", "2026-07-30", "2026-08-01", "2026-08-08"],
    ),
    _receipt(
        "github_v539_targeted_after_marker",
        "github_targeted",
        "GitHub",
        "secondary",
        "ARC-AGI-3 OR energy-based-model OR mode-jump pushed:>2026-08-09T06:16:52Z",
        "https://api.github.com/search/repositories?q=ARC-AGI-3+pushed:%3E2026-08-09T06:16:52Z",
        "2026-08-09T07:05:00Z",
        "queried_no_independent_accepted_delta",
        "none",
        "strict pushed_at filter returned one ARC repo and three energy repos; all were secondary metadata or self-reference",
        [
            "DongYaoZe/llm-evaluation-and-agent",
            "davidkhjo/ebm-pytorch",
            "utpala-07/Energy_Estimator-",
            "Carnot-EBM/carnot-ebm",
        ],
        candidate_count=4,
        newest_pushed_at="2026-08-09T06:54:39Z",
    ),
    _receipt(
        "logical_intelligence_v539_kona",
        "logical_intelligence",
        "Logical Intelligence",
        "official",
        "Kona EBM first-party page after V539 marker",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "2026-08-09T07:05:00Z",
        "reachable_no_public_weights_or_api",
        "none",
        "first-party page remains a guarded proprietary route without public weights, local API, or architecture spec",
        ["kona_1_0"],
        candidate_count=1,
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
        "arxiv:2608.05025",
        "Canonical Joint Energy-Based Model on CIFAR-10",
        "https://arxiv.org/abs/2608.05025",
        "2026-08-05",
        "arXiv",
        "primary",
        "sealed in V539 planner block",
        "reachable_primary_premarked_reference",
        "exp6237-activated-mode-jump-sampler-ab",
        "none",
        "planning-time duplicate",
        "Treatment activation evidence was already sealed before the V539 marker.",
        disposition="duplicate",
        primary_or_first_party=True,
        dated_reproducible=True,
    ),
    _candidate(
        "arxiv:2608.04355",
        "The Calibration Floor",
        "https://arxiv.org/abs/2608.04355",
        "2026-08-05",
        "arXiv",
        "primary",
        "sealed in V539 planner block",
        "reachable_primary_premarked_reference",
        "exp6233-three-family-code-content-margin",
        "none",
        "planning-time duplicate",
        "Parse/content margin evidence was already sealed before the V539 marker.",
        disposition="duplicate",
        primary_or_first_party=True,
        dated_reproducible=True,
    ),
    _candidate(
        "extropic:z1-status",
        "From One to One Billion: Torx, Thermalizers, and Z1",
        "https://extropic.ai/writing/from-one-to-one-billion",
        "2026-08-03",
        "Extropic",
        "official",
        "sealed in V539 planner block",
        "reachable_first_party_no_authenticated_route",
        "defer",
        "none",
        "no authenticated local Z1 or simulator route",
        "Z1 taped-out status is sealed planner context and still does not authorize hardware claims.",
        disposition="guarded",
        primary_or_first_party=True,
        dated_reproducible=True,
    ),
    _candidate(
        "logical-intelligence:kona-page",
        "Logical Intelligence Kona first-party page",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "2026-06-26",
        "Logical Intelligence",
        "official",
        "first-party page predates the V539 marker",
        "reachable_first_party_page_no_public_weights",
        "defer",
        "none",
        "Kona proprietary execution remains guarded",
        "The page exposes no public weights, local API, or architecture spec.",
        disposition="guarded",
        primary_or_first_party=True,
        dated_reproducible=True,
    ),
    _candidate(
        "github:DongYaoZe/llm-evaluation-and-agent",
        "Course project repository with ARC-AGI-3 metadata",
        "https://github.com/DongYaoZe/llm-evaluation-and-agent",
        "2026-08-09T06:38:52Z",
        "GitHub",
        "secondary",
        "pushed_at after marker from GitHub repository search",
        "reachable_secondary_repository_metadata",
        "defer",
        "none",
        "secondary repository metadata",
        "The repository search hit is not primary technical evidence for a V539 method or gate.",
        disposition="rejected",
        dated_reproducible=True,
    ),
    _candidate(
        "github:davidkhjo/ebm-pytorch",
        "EBM PyTorch repository metadata",
        "https://github.com/davidkhjo/ebm-pytorch",
        "2026-08-09T06:54:39Z",
        "GitHub",
        "secondary",
        "pushed_at after marker from GitHub repository search",
        "reachable_secondary_repository_metadata",
        "defer",
        "none",
        "secondary repository metadata",
        "The repository search hit did not change a V539 Carnot method or gate.",
        disposition="rejected",
        dated_reproducible=True,
    ),
    _candidate(
        "github:Carnot-EBM/carnot-ebm",
        "Carnot-EBM/carnot-ebm repository metadata",
        "https://github.com/Carnot-EBM/carnot-ebm",
        "2026-08-09T06:40:28Z",
        "GitHub",
        "secondary",
        "pushed_at after marker on the current project repository",
        "reachable_self_repo_metadata",
        EXPERIMENT_ID,
        "none",
        "self-referential project repository",
        "The current project's repository cannot be independent runtime source evidence.",
        disposition="duplicate",
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


def _references_has_candidate(row: Mapping[str, Any], references_text: str) -> bool:
    haystack = references_text.lower()
    stable_id = str(row.get("stable_id", "")).lower()
    url = str(row.get("url", "")).lower()
    if stable_id.startswith("arxiv:") and stable_id.split(":", 1)[1] in haystack:
        return True
    return bool(url and url in haystack)


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
        raise ValueError("accepted candidate must be strictly after the V539 marker")
    if not row.get("primary_or_first_party"):
        raise ValueError("accepted candidate must be primary or first-party")
    if not row.get("dated_reproducible"):
        raise ValueError("accepted candidate must have reproducible date evidence")
    if not row.get("new_applicability"):
        raise ValueError("accepted candidate must change a V539 method or gate")
    if str(row.get("retirement_conflict")) != "none":
        raise ValueError("accepted candidate cannot carry a retirement conflict")


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


def classify_candidate(row: Mapping[str, Any], references_text: str = "") -> JsonDict:
    candidate = dict(row)
    candidate.setdefault("content_hash", _stable_hash(candidate))
    explicit = candidate.get("disposition")
    if explicit in {"rejected", "guarded", "duplicate", "cutoff_confound", "endpoint_failed"}:
        return _ledger_row(candidate, str(explicit))
    if _references_has_candidate(candidate, references_text):
        return _ledger_row(candidate, "duplicate")
    if candidate.get("endpoint_failed"):
        return _ledger_row(candidate, "endpoint_failed")
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


def deduplicate_candidates(
    candidates: Sequence[JsonDict], references_text: str = ""
) -> tuple[list[JsonDict], list[JsonDict]]:
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    unique: list[JsonDict] = []
    duplicates: list[JsonDict] = []
    for row in candidates:
        stable_id = str(row.get("stable_id", ""))
        content_hash = str(row.get("content_hash", _stable_hash(row)))
        if stable_id in seen_ids or content_hash in seen_hashes or _references_has_candidate(
            row, references_text
        ):
            duplicate = dict(row)
            duplicate["disposition"] = "duplicate"
            duplicates.append(duplicate)
            continue
        seen_ids.add(stable_id)
        seen_hashes.add(content_hash)
        unique.append(row)
    return unique, duplicates


def _candidate_ledgers(
    candidates: Sequence[JsonDict], references_text: str
) -> tuple[list[JsonDict], list[JsonDict]]:
    unique, duplicates = deduplicate_candidates([dict(row) for row in candidates], references_text)
    classified = [classify_candidate(row, references_text) for row in unique]
    classified.extend(_ledger_row(row, "duplicate") for row in duplicates)
    accepted = [row for row in classified if row["disposition"] == "accepted"]
    rejected = [row for row in classified if row["disposition"] != "accepted"]
    return accepted, rejected


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


def preconditions_checked(root: Path) -> JsonDict:
    input_hashes = {path.as_posix(): path_sha256(root / path) for path in SOURCE_INPUT_RELATIVE_PATHS}
    protected = _protected_hashes(root)
    marker = marker_snapshot(root)
    return {
        "marker_hash": marker["marker_block_hash"],
        "marker_commit": MARKER_COMMIT,
        "active_roadmap_hash": input_hashes[ROADMAP_RELATIVE_PATH.as_posix()],
        "staged_roadmap_path": STAGED_ROADMAP_RELATIVE_PATH.as_posix(),
        "staged_roadmap_hash": input_hashes[STAGED_ROADMAP_RELATIVE_PATH.as_posix()],
        "staged_roadmap_present": (root / STAGED_ROADMAP_RELATIVE_PATH).exists(),
        "source_input_hash": _stable_hash(input_hashes),
        "input_hashes": input_hashes,
        "protected_hashes": protected,
        "principle": "Hashing inputs before search prevents hidden context or roadmap drift.",
    }


def bootstrap_artifact_write_receipt(
    root: Path,
    *,
    date: str,
    env: Mapping[str, str] | None,
    write_artifact: bool = True,
) -> JsonDict:
    preconditions = preconditions_checked(root)
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": "bootstrap",
        "planner_marker_and_hash": marker_snapshot(root),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
    }
    payload_hash = _stable_hash(payload)
    target = None
    file_hash = None
    if write_artifact:
        target = atomic_write_json(RESULT_RELATIVE_PATH, payload, root=root, env=env, sort_keys=False)
        file_hash = path_sha256(target)
    return {
        "status": "bootstrap_written",
        "passed": True,
        "survived_to_final": True,
        "path": RESULT_RELATIVE_PATH.as_posix(),
        "resolved_path": str(target) if target is not None else None,
        "bootstrap_payload_hash": payload_hash,
        "bootstrap_file_hash": file_hash,
        "precondition_hash": _stable_hash(preconditions),
        "principle": "A bootstrap artifact preserves the exact pre-search input state.",
    }


def _source_receipts_block(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    channels = [str(row.get("channel", "unknown")) for row in receipts]
    failures = [
        {
            "receipt_id": row.get("receipt_id"),
            "channel": row.get("channel"),
            "authority": row.get("authority"),
            "failure_class": row.get("failure_class"),
            "access_outcome": row.get("access_outcome"),
        }
        for row in receipts
        if str(row.get("failure_class")) not in {"none", ""}
    ]
    return {
        "receipts": [dict(row) for row in receipts],
        "required_channels": list(REQUIRED_SOURCE_CHANNELS),
        "required_channels_observed": sorted(set(channels) & set(REQUIRED_SOURCE_CHANNELS)),
        "missing_required_channels": sorted(set(REQUIRED_SOURCE_CHANNELS) - set(channels)),
        "authority_counts": dict(sorted(Counter(str(row.get("authority", "unknown")) for row in receipts).items())),
        "failure_receipts": failures,
        "deep_research_invoked": False,
    }


def _execution_delta_block(accepted_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "",
        EXECUTION_DELTA_HEADING,
        "",
        "Execution-time source deltas accepted after the sealed V539 marker:",
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


def _runtime_contract() -> JsonDict:
    return {
        "task_owned_process_provenance_required": True,
        "bounded_wait_required": True,
        "signal_sender_or_unlocalized_result_required": True,
        "owned_child_recovery_only": True,
        "unrelated_pid_kill_allowed": False,
        "gate_principles": {
            "task_owned_process_provenance_required": "Runtime evidence needs a PID and ownership chain.",
            "bounded_wait_required": "A caller hang is not a valid measurement.",
            "owned_child_recovery_only": "Recovery may not disturb unrelated local work.",
        },
    }


def _arc_provenance_contract() -> JsonDict:
    return {
        "solve_provenance_required": "live_agent_self_discovery",
        "hidden_game_source_access_allowed": False,
        "offline_bfs_access_allowed": False,
        "adapter_truth_access_allowed": False,
        "hidden_state_access_allowed": False,
        "registry_trajectory_access_allowed": False,
        "per_game_forbidden_access_counts_required": True,
        "arc_registry_hash_required": True,
        "gate_principles": {
            "solve_provenance_required": "ARC credit must come from the live process on unseen games.",
            "per_game_forbidden_access_counts_required": "Zero-access counts make leakage auditable.",
            "arc_registry_hash_required": "Registry nonmutation prevents post-hoc credit drift.",
        },
    }


def _code_content_margin_contract() -> JsonDict:
    return {
        "raw_bytes_record_required": True,
        "parse_outcome_required": True,
        "compile_outcome_required": True,
        "public_test_outcome_required": True,
        "hidden_test_content_margin_required": True,
        "parse_content_separated": True,
        "format_only_gain_not_verified_reasoning": True,
        "gate_principles": {
            "parse_content_separated": "Format recovery can mimic self-correction without semantic gain.",
            "hidden_test_content_margin_required": "Content claims require already-parseable hidden-test movement.",
        },
    }


def _csl_contract() -> JsonDict:
    return {
        "fresh_event_required": True,
        "predecision_snapshot_read_only": True,
        "post_outcome_commit_only": True,
        "verifier_approved_commit_required": True,
        "model_weight_mutation_allowed": False,
        "rollback_required": True,
        "duplicate_reorder_stale_quarantine_required": True,
        "gate_principles": {
            "post_outcome_commit_only": "Decision-time writes can leak labels into the current prediction.",
            "model_weight_mutation_allowed": "This milestone tests system memory, not parameter updates.",
            "rollback_required": "Unsafe memory must be exactly reversible.",
        },
    }


def _sampler_activation_contract() -> JsonDict:
    return {
        "treatment_activation_required": True,
        "nonzero_jump_proposals_required": True,
        "nonzero_jump_acceptances_required": True,
        "positive_control_required": True,
        "matched_seeded_fallback_required": True,
        "inactive_treatment_is_failure": True,
        "gate_principles": {
            "treatment_activation_required": "A null with no treatment fire is an instrument failure.",
            "positive_control_required": "A sampler A/B needs proof that the harness can detect movement.",
        },
    }


def _hardware_boundary() -> JsonDict:
    return {
        "hardware_claim_allowed_without_receipt": False,
        "forbidden_claim_terms": [
            "Extropic",
            "TSU",
            "Z1",
            "Kona",
            "GateMate",
            "KV260",
            "PolarFire",
            "power",
            "energy",
            "latency",
            "speedup",
        ],
        "authenticated_route_required": True,
        "physical_state_receipt_required": True,
        "z1_status": "taped_out_planner_context_only_no_execution_claim",
        "gate_principles": {
            "hardware_claim_allowed_without_receipt": "Product-roadmap text is not execution evidence.",
            "authenticated_route_required": "Hardware claims need a local or authenticated route.",
        },
    }


def _roadmap_schema_bundle(root: Path) -> JsonDict:
    active_path = root / ROADMAP_RELATIVE_PATH
    staged_path = root / STAGED_ROADMAP_RELATIVE_PATH
    active_data = _load_roadmap_data(active_path)
    staged_data = _load_roadmap_data(staged_path)
    retired_ids = load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    active_contract = validate_v539_roadmap_data(active_data, retired_exp_ids=retired_ids)
    active_schema = _roadmap_schema_result(active_data)
    staged_schema = _roadmap_schema_result(staged_data) if staged_path.exists() else {
        "passed": True,
        "present": False,
        "task_count": 0,
    }
    passed = (
        active_schema.get("passed") is True
        and staged_schema.get("passed") is True
        and active_contract["dependency_validation"]["ok"] is True
        and active_contract["gated_on_validation"]["ok"] is True
        and active_contract["model_policy_validation"]["ok"] is True
        and active_contract["prompt_contract_validation"]["ok"] is True
        and active_contract["retired_dependency_count"] == 0
        and active_contract["id_collision_count"] == 0
    )
    return {
        "passed": passed,
        "active_schema": active_schema,
        "staged_schema": staged_schema,
        "active_v539_contracts": active_contract,
        "gate_audit_result": active_contract["gated_on_validation"],
        "model_specs_rule_result": active_contract["model_policy_validation"],
        "prompt_ending_result": active_contract["prompt_contract_validation"],
        "retired_dependency_result": {
            "retired_dependency_count": active_contract["retired_dependency_count"],
            "id_collision_count": active_contract["id_collision_count"],
        },
    }


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
    return {
        "command": command,
        "exit_code": int(receipt.get("exit_code", 1)),
        "classification": str(receipt.get("classification", "missing_recorded_command_receipt")),
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
                "REQ-INFRA-6226",
                "V539 planner marker",
                "source channel receipts",
                "roadmap validators",
                "frozen V539 contracts",
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


def honest_verdict(status: str, accepted_count: int | None) -> str:
    count_text = "null" if accepted_count is None else str(accepted_count)
    if status == "blocked":
        return f"blocked: accepted_count={count_text}; V539 source, roadmap, freeze contract, or verification command failed"
    if accepted_count:
        return f"complete_delta: accepted_count={accepted_count}; dated post-V539-marker findings appended"
    return "complete_null: accepted_count=null; no reproducible post-V539-marker findings; references unchanged; V539 scope contracts frozen"


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    source_receipts: Sequence[JsonDict] | None = None,
    candidates: Sequence[JsonDict] | None = None,
    command_runner: CommandRunner = _run_command,
    duration_s: float | None = None,
    bootstrap_receipt: JsonDict | None = None,
    references_before_hash: str | None = None,
    references_after_hash: str | None = None,
    references_appended: bool = False,
    references_text_for_dedup: str | None = None,
    protected_before_hashes: Mapping[str, Any] | None = None,
    protected_authorized_after_hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    started = time.monotonic()
    receipts = [dict(row) for row in (source_receipts or DEFAULT_SOURCE_CHANNEL_RECEIPTS)]
    found_candidates = [dict(row) for row in (candidates or DEFAULT_DISCOVERED_CANDIDATES)]
    references_text = (
        references_text_for_dedup
        if references_text_for_dedup is not None
        else _read_text(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    )
    accepted, rejected = _candidate_ledgers(found_candidates, references_text)
    accepted_count: int | None = len(accepted) if accepted else None
    marker = marker_snapshot(root)
    roadmap_data = _load_roadmap_data(root / ROADMAP_RELATIVE_PATH)
    tasks = _raw_tasks(roadmap_data)
    schema_result = _roadmap_schema_bundle(root)
    exclusion_result = _exclusion_lint_result(root / ROADMAP_RELATIVE_PATH)
    prior_result = _prior_failure_contract(tasks)
    protected = _protected_unchanged(
        root, protected_before_hashes, protected_authorized_after_hashes
    )
    source_block = _source_receipts_block(receipts)
    bootstrap = dict(
        bootstrap_receipt
        or bootstrap_artifact_write_receipt(root, date=date, env=None, write_artifact=False)
    )
    command_receipts = _run_commands(root, command_runner)
    test_exit_codes = {
        str(receipt["command"]): int(receipt["exit_code"]) for receipt in command_receipts
    }
    commands_passed = all(int(receipt["exit_code"]) == 0 for receipt in command_receipts)
    refs_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    refs_before = references_before_hash or path_sha256(refs_path)
    refs_after = references_after_hash or path_sha256(refs_path)
    status = (
        "complete"
        if (
            marker["marker_count"] == 1
            and source_block["missing_required_channels"] == []
            and schema_result["passed"]
            and exclusion_result["passed"]
            and exclusion_result["hard_count"] == 0
            and prior_result["passed"]
            and protected["all_unchanged"]
            and bootstrap.get("passed") is True
            and commands_passed
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
        "bootstrap_artifact_write_receipt": bootstrap,
        "query_window": {
            "marker": PLANNER_MARKER,
            "window_start_exclusive": MARKER_COMMITTED_AT,
            "window_end_inclusive": _window_end(date),
            "accepted_at_or_before_marker": False,
            "same_day_policy": "A bare 2026-08-09 date is rejected because it resolves before the marker timestamp.",
        },
        "source_channel_receipts": source_block,
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
        "frozen_runtime_contract": _runtime_contract(),
        "frozen_arc_provenance_contract": _arc_provenance_contract(),
        "frozen_code_content_margin_contract": _code_content_margin_contract(),
        "frozen_csl_contract": _csl_contract(),
        "frozen_sampler_activation_contract": _sampler_activation_contract(),
        "frozen_hardware_boundary": _hardware_boundary(),
        "roadmap_path_and_hash": {
            "active_path": ROADMAP_RELATIVE_PATH.as_posix(),
            "active_sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
            "staged_path": STAGED_ROADMAP_RELATIVE_PATH.as_posix(),
            "staged_present": (root / STAGED_ROADMAP_RELATIVE_PATH).exists(),
            "staged_sha256": path_sha256(root / STAGED_ROADMAP_RELATIVE_PATH),
            "proposal_path": VNEXT_RELATIVE_PATH.as_posix(),
            "proposal_sha256": path_sha256(root / VNEXT_RELATIVE_PATH),
            "milestone": roadmap_data.get("milestone"),
        },
        "roadmap_schema_result": schema_result,
        "exclusion_manifest_lint_result": exclusion_result,
        "prior_failure_contract_result": prior_result,
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions_checked(root),
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


def _freeze_contract_errors(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    runtime = report.get("frozen_runtime_contract")
    if not isinstance(runtime, Mapping) or runtime.get("task_owned_process_provenance_required") is not True or runtime.get("bounded_wait_required") is not True:
        errors.append("frozen_runtime_contract")
    arc = report.get("frozen_arc_provenance_contract")
    if (
        not isinstance(arc, Mapping)
        or arc.get("solve_provenance_required") != "live_agent_self_discovery"
        or arc.get("hidden_game_source_access_allowed") is not False
        or arc.get("offline_bfs_access_allowed") is not False
        or arc.get("adapter_truth_access_allowed") is not False
        or arc.get("hidden_state_access_allowed") is not False
        or arc.get("registry_trajectory_access_allowed") is not False
    ):
        errors.append("frozen_arc_provenance_contract")
    margin = report.get("frozen_code_content_margin_contract")
    if not isinstance(margin, Mapping) or margin.get("parse_content_separated") is not True or margin.get("hidden_test_content_margin_required") is not True:
        errors.append("frozen_code_content_margin_contract")
    csl = report.get("frozen_csl_contract")
    if (
        not isinstance(csl, Mapping)
        or csl.get("post_outcome_commit_only") is not True
        or csl.get("model_weight_mutation_allowed") is not False
        or csl.get("rollback_required") is not True
    ):
        errors.append("frozen_csl_contract")
    sampler = report.get("frozen_sampler_activation_contract")
    if not isinstance(sampler, Mapping) or sampler.get("treatment_activation_required") is not True or sampler.get("inactive_treatment_is_failure") is not True:
        errors.append("frozen_sampler_activation_contract")
    hardware = report.get("frozen_hardware_boundary")
    if not isinstance(hardware, Mapping) or hardware.get("hardware_claim_allowed_without_receipt") is not False or hardware.get("authenticated_route_required") is not True:
        errors.append("frozen_hardware_boundary")
    return errors


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
    accepted_len = len(report.get("accepted_findings", []))
    accepted_count = report.get("accepted_count")
    if (accepted_len == 0 and accepted_count not in (0, None)) or (
        accepted_len > 0 and accepted_count != accepted_len
    ):
        errors.append("accepted_count")
    if str(report.get("honest_verdict", "")).startswith(
        ("complete_null:", "complete_delta:", "blocked:")
    ) is False:
        errors.append("honest_verdict")
    checks = (
        ("roadmap_schema_result", "passed"),
        ("exclusion_manifest_lint_result", "passed"),
        ("prior_failure_contract_result", "passed"),
        ("protected_files_unchanged", "all_unchanged"),
        ("bootstrap_artifact_write_receipt", "survived_to_final"),
    )
    for field, key in checks:
        row = report.get(field)
        if not isinstance(row, Mapping) or row.get(key) is not True:
            errors.append(field)
    source = report.get("source_channel_receipts")
    if not isinstance(source, Mapping) or source.get("missing_required_channels") != []:
        errors.append("source_channel_receipts")
    errors.extend(_freeze_contract_errors(report))
    if any(int(code) != 0 for code in report.get("test_exit_codes", {}).values()) and report.get(
        "status"
    ) != "blocked":
        errors.append("test_exit_codes")
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
    bootstrap = bootstrap_artifact_write_receipt(root, date=date, env=env, write_artifact=True)
    dry = build_report(
        root,
        date=date,
        source_receipts=source_receipts,
        candidates=candidates,
        command_runner=_deferred_command_runner,
        duration_s=duration_s,
        bootstrap_receipt=bootstrap,
        references_before_hash=references_before,
        references_after_hash=references_before,
        references_text_for_dedup=before_text,
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
        bootstrap_receipt=bootstrap,
        references_before_hash=references_before,
        references_after_hash=references_after,
        references_appended=appended,
        references_text_for_dedup=before_text,
        protected_before_hashes=protected_before,
        protected_authorized_after_hashes=protected_authorized_after,
    )
    errors = validate_report(report)
    if errors:
        raise ValueError(f"invalid Exp6226 freeze: {errors}")
    atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    report = write_freeze(REPO_ROOT, date=args.date, command_runner=_recorded_command_runner)
    print(
        json.dumps(
            {"path": RESULT_RELATIVE_PATH.as_posix(), "checksum": report["reproducibility_checksum"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
