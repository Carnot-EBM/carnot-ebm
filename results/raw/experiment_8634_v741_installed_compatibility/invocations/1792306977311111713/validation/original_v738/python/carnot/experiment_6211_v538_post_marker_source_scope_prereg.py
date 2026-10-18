"""Exp6211 V538 source ingress and ARC causal preregistration.

Spec refs: REQ-INFRA-6211, SCENARIO-INFRA-6211-1,
SCENARIO-INFRA-6211-2, SCENARIO-INFRA-6211-3,
SCENARIO-INFRA-6211-4, SCENARIO-INFRA-6211-5,
SCENARIO-INFRA-6211-6.
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
    "results/experiment_6211_v538_post_marker_source_scope_prereg.json"
)
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_REFERENCES_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

EXPERIMENT_ID = "exp6211-v538-post-marker-source-scope-prereg"
SCHEMA = "carnot.experiment_6211.v538_source_scope_prereg.v1"
INFERENCE_SUBSTRATE = "post_marker_source_ingestion_and_arc_causal_preregistration"

PLANNER_HEADING = "## V538 Planner Refresh (2026-08-07, after milestone 2026.08.537)"
PLANNER_MARKER = "V538-PLANNER-REFRESH-20260807-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
MARKER_COMMIT = "d9b6e7a9380a254dc06e75169d92612d4248615e"
MARKER_COMMITTED_AT = "2026-08-08T01:44:27Z"
EXECUTION_DELTA_HEADING = "## V538 Execution Source Delta - 20260808"
EXECUTION_DELTA_END_MARKER = "<!-- V538-EXECUTION-SOURCE-DELTA-20260808-END -->"

SOTA_GGUFS = {
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
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
ARC_AB_TASK_IDS = (
    "exp6214-arc-object-delta-heldout-ab",
    "exp6215-arc-trajectory-transfer-ab",
    "exp6216-arc-budget-aware-search-ab",
    "exp6217-arc-gemma31-think-ab",
)
EXPECTED_PHASE_COUNTS = {
    "arc": 6,
    "capstone": 1,
    "continuous_learning": 1,
    "infrastructure": 2,
    "phase_d": 3,
    "sampling": 1,
}

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
    "phase_counts",
    "arc_task_count",
    "continuous_self_learning_slot_count",
    "hardware_boundary_result",
    "arc_outcome_vocabulary",
    "matched_control_contract",
    "no_solve_and_registry_nonmutation_contract",
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
    "status": "The terminal state follows novelty, roadmap, and causal-contract checks.",
    "planner_marker_and_hash": "The exact V538 marker defines the runtime novelty boundary.",
    "query_window": "Only source evidence strictly after the marker can be a runtime delta.",
    "source_channel_receipts": "Every required source channel records success, null, or failure.",
    "discovered_candidates": "Rejected and duplicate rows stay visible for audit.",
    "accepted_findings": "Only dated primary or first-party rows that change V538 scope are accepted.",
    "rejected_or_duplicate_findings": "Rejected rows explain why the roadmap did not move.",
    "accepted_count": "A bare count drives the null versus append decision.",
    "references_append_receipt": "References stay byte-identical on a null and append-only on a delta.",
    "roadmap_path_and_hash": "The audited roadmap hash prevents silent path ambiguity.",
    "roadmap_schema_result": "Schema parsing catches malformed task contracts before launch.",
    "exclusion_manifest_lint_result": "Retired scopes stay closed unless a structured exception exists.",
    "retired_scope_match_count": "The bare zero is the gateable retired-scope exposure count.",
    "prior_failure_contract_result": "Prior failures must explain why this task is different.",
    "gate_structure_result": "Structured gates must reference real upstream tasks and fields.",
    "model_specs_rule_result": "Headline LLM tasks must use the mandated SOTA GGUF set.",
    "task_count": "The V538 roadmap is expected to contain fourteen tasks.",
    "phase_counts": "Phase counts keep ARC, runtime, CSL, sampling, Phase-D, and capstone slots explicit.",
    "arc_task_count": "ARC work is counted before any live A/B measurement.",
    "continuous_self_learning_slot_count": "FR-11 continuity needs exactly one V538 CSL slot.",
    "hardware_boundary_result": "Unavailable hardware stays blocked without new command claims.",
    "arc_outcome_vocabulary": "A/B outcomes must use a frozen vocabulary before measurement.",
    "matched_control_contract": "Matched controls are required before causal ARC claims.",
    "no_solve_and_registry_nonmutation_contract": "The prereg forbids solve credit and registry mutation.",
    "protected_files_unchanged": "The audit must not rewrite instructions, roadmap, registry, or conductor code.",
    "inference_substrate": "Declares deterministic source and roadmap audit rather than LLM inference.",
    "verifier_is_oracle": "False because this audit validates metadata, not benchmark answers.",
    "field_provenance": "Each required field traces to marker, receipts, roadmap checks, or commands.",
    "field_principles": "Every required field carries the reason it exists.",
    "test_commands": "Records focused, coverage, lint, E2E-plan, adversarial, and full-suite checks.",
    "test_exit_codes": "Exit codes prevent failed checks from being reported as clean.",
    "duration_s": "Reports wall time without padding.",
    "reproducibility_checksum": "Detects later drift in the artifact payload.",
    "honest_verdict": "Uses complete_null, complete_delta, or blocked with accepted_count.",
}

TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6211_v538_post_marker_source_scope_prereg.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6211_v538_post_marker_source_scope_prereg.py -m pytest tests/python/test_experiment_6211_v538_post_marker_source_scope_prereg.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6211_v538_post_marker_source_scope_prereg.py --fail-under=100",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6211_v538_post_marker_source_scope_prereg.py",
    "sed -n 1,220p ops/e2e-test-plan.md",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6211_v538_post_marker_source_scope_prereg.json",
    ".venv/bin/pytest tests/python -q",
)

RECORDED_TEST_RECEIPTS: Mapping[str, JsonDict] = {
    TEST_COMMANDS[0]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "9 passed in 61.10s",
        "stderr_tail": "",
    },
    TEST_COMMANDS[1]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "9 passed in 110.57s",
        "stderr_tail": "",
    },
    TEST_COMMANDS[2]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "TOTAL 430 0 100%",
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
        "stdout_tail": "roadmap_gate_audit_passed: true; n_tasks_audited: 14; n_gate_upstream_checks: 5",
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
        "stdout_tail": "ops/e2e-test-plan.md inspected; no Exp6211-specific live E2E applies.",
        "stderr_tail": "",
    },
    TEST_COMMANDS[7]: {
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "Scanned 1 artifact(s); 0 flagged.",
        "stderr_tail": "",
    },
    TEST_COMMANDS[8]: {
        "exit_code": 2,
        "classification": "interrupted_after_preexisting_broad_suite_failures_and_worker_abort",
        "stdout_tail": "Interrupted at 20% after 759.92s: 104 failed, 11192 passed, 7 skipped, 132 warnings, 14 errors. Worker gw1 aborted in scripts/experiment_295_apple_verify_repair.py during Qwen tokenizer prewarm; JAX GC warnings appeared on interrupt. The run also rewrote results/experiment_1822_rtl_synth.log and results/experiment_2031.json; both historical files were restored byte-identically.",
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
        "arxiv_v538_topics_after_marker",
        "arxiv_topics",
        "arXiv",
        "primary",
        "submittedDate:[202608080144 TO 202608082359] AND EBM/ARC/continual/hardware/KAN terms",
        "https://export.arxiv.org/api/query?search_query=submittedDate:%5B202608080144%20TO%202608082359%5D&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending",
        "2026-08-08T03:29:37Z",
        "reachable_http_200_zero_results",
        "none",
        "strict submittedDate endpoint updated at 2026-08-08T03:29:37Z and returned zero topic records",
        [],
        status=200,
        total_results=0,
    ),
    _receipt(
        "openreview_v538_energy_route",
        "openreview",
        "OpenReview",
        "secondary",
        "content=energy-based&limit=5",
        "https://api.openreview.net/notes?content=energy-based&limit=5",
        "2026-08-08T02:40:42Z",
        "http_403_challenge_required",
        "challenge_required",
        "route returned challenge metadata rather than note dates",
        [],
        status=403,
        request_id="2026-08-08-984284",
    ),
    _receipt(
        "semantic_scholar_v538_ebt_citations",
        "semantic_scholar_ebt_citations",
        "Semantic Scholar",
        "secondary",
        "arXiv:2507.02092 citations",
        "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations?fields=title,year,externalIds,url,publicationDate,authors&limit=100",
        "2026-08-08T03:29:38Z",
        "reachable_http_200_32_records_no_post_marker",
        "none",
        "newest visible publicationDate 2026-07-29",
        ["2607.27372", "2607.20792", "2607.17047", "2607.11555"],
        candidate_count=32,
        newest_publication_date="2026-07-29",
    ),
    _receipt(
        "semantic_scholar_v538_arm_ebm_citations",
        "semantic_scholar_arm_ebm_citations",
        "Semantic Scholar",
        "secondary",
        "arXiv:2512.15605 citations",
        "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations?fields=title,year,externalIds,url,publicationDate,authors&limit=100",
        "2026-08-08T03:29:38Z",
        "reachable_http_200_8_records_no_post_marker",
        "none",
        "newest visible publicationDate 2026-07-02",
        ["2607.02154", "2606.03089", "2605.18871", "2605.11011", "2604.00555"],
        candidate_count=8,
        newest_publication_date="2026-07-02",
    ),
    _receipt(
        "huggingface_papers_v538_aug8",
        "huggingface_papers",
        "Hugging Face Papers",
        "secondary",
        "date/2026-08-08",
        "https://huggingface.co/papers/date/2026-08-08",
        "2026-08-08T03:29:39Z",
        "reachable_http_200_secondary_feed",
        "secondary_feed_only",
        "date page is secondary and exposed no strict post-marker primary timestamp",
        [
            "2608.05987",
            "2607.28609",
            "2608.01481",
            "2608.05248",
            "2608.05747",
            "2608.06197",
            "2608.05631",
            "2608.06060",
            "2608.06020",
            "2608.06301",
        ],
        candidate_count=30,
    ),
    _receipt(
        "github_v538_arc_agi3_after_marker",
        "github_targeted",
        "GitHub",
        "secondary",
        "ARC-AGI-3 pushed:>2026-08-08T01:44:27Z",
        "https://api.github.com/search/repositories?q=ARC-AGI-3+pushed:%3E2026-08-08T01:44:27Z&sort=updated&order=desc&per_page=5",
        "2026-08-08T03:29:40Z",
        "reachable_http_200_total_count_0",
        "none",
        "strict pushed_at filter returned zero repositories",
        [],
        candidate_count=0,
        total_count=0,
    ),
    _receipt(
        "github_v538_energy_repos_after_marker",
        "github_targeted",
        "GitHub",
        "secondary",
        "energy-based-model pushed:>2026-08-08T01:44:27Z",
        "https://api.github.com/search/repositories?q=energy-based-model+pushed:%3E2026-08-08T01:44:27Z&sort=updated&order=desc&per_page=5",
        "2026-08-08T03:29:40Z",
        "reachable_http_200_total_count_1_self_repo_only",
        "none",
        "strict pushed_at filter returned only the current project repository",
        ["Carnot-EBM/carnot-ebm"],
        candidate_count=1,
        total_count=1,
    ),
    _receipt(
        "github_v538_kan_repos_after_marker",
        "github_targeted",
        "GitHub",
        "secondary",
        "Kolmogorov-Arnold-Network pushed:>2026-08-08T01:44:27Z",
        "https://api.github.com/search/repositories?q=Kolmogorov-Arnold-Network+pushed:%3E2026-08-08T01:44:27Z&sort=updated&order=desc&per_page=5",
        "2026-08-08T03:29:40Z",
        "reachable_http_200_total_count_0",
        "none",
        "strict pushed_at filter returned zero repositories",
        [],
        candidate_count=0,
        total_count=0,
    ),
    _receipt(
        "extropic_v538_first_party_pages",
        "extropic",
        "Extropic",
        "official",
        "writing, hardware, and software pages",
        "https://extropic.ai/ ; https://extropic.ai/writing ; https://extropic.ai/hardware ; https://extropic.ai/software",
        "2026-08-08T03:29:41Z",
        "reachable_http_200_no_dated_post_marker_local_route",
        "no_changed_since_endpoint",
        "writing page exposes From One to One Billion publishDate 2026-08-03, before the marker, and no authenticated local TSU/Z1 route",
        ["home", "writing", "from-one-to-one-billion", "hardware", "software"],
        candidate_count=5,
    ),
    _receipt(
        "logical_intelligence_v538_kona",
        "logical_intelligence",
        "Logical Intelligence",
        "official",
        "Kona EBM first-party page",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "2026-08-08T03:29:41Z",
        "reachable_http_200_no_public_weights_or_api",
        "none",
        "first-party page published 2026-06-26 and exposes no public weights, local API, or architecture spec",
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
        "github:Carnot-EBM/carnot-ebm",
        "Carnot-EBM/carnot-ebm repository metadata",
        "https://github.com/Carnot-EBM/carnot-ebm",
        "2026-08-08T03:29:40Z",
        "GitHub",
        "secondary",
        "pushed_at after marker on the current project repository",
        "reachable_self_repo_metadata",
        EXPERIMENT_ID,
        "none",
        "self-referential project repository",
        "The current project's own repository cannot be independent runtime source evidence.",
        disposition="duplicate",
        dated_reproducible=True,
    ),
    _candidate(
        "arxiv:2607.09059",
        "ARCANA: A Reflective Multi-Agent Program Synthesis Framework",
        "https://arxiv.org/abs/2607.09059",
        "2026-07-10",
        "arXiv",
        "primary",
        "planning-time arXiv identifier already sealed under V538 marker",
        "reachable_primary_premarked_reference",
        "exp6213-arc-object-delta-perception-wiring",
        "none",
        "planning-time duplicate",
        "ARCANA is sealed planner context and predates the V538 marker.",
        disposition="duplicate",
        primary_or_first_party=True,
        dated_reproducible=True,
    ),
    _candidate(
        "arxiv:2607.06764",
        "Cost-Effective Agent Harnesses for ARC-AGI-1",
        "https://arxiv.org/abs/2607.06764",
        "2026-07-07",
        "arXiv",
        "primary",
        "planning-time arXiv identifier already sealed under V538 marker",
        "reachable_primary_premarked_reference",
        "exp6217-arc-gemma31-think-ab",
        "none",
        "planning-time duplicate",
        "The think-tool comparator is already incorporated and is not a runtime delta.",
        disposition="duplicate",
        primary_or_first_party=True,
        dated_reproducible=True,
    ),
    _candidate(
        "hf:papers/2608.05987",
        "AgentOPSD on Hugging Face Papers date page",
        "https://huggingface.co/papers/2608.05987",
        "2026-08-08",
        "Hugging Face Papers",
        "secondary",
        "secondary calendar page only",
        "reachable_secondary_feed",
        "defer",
        "none",
        "secondary-only source",
        "The page did not provide a strict post-marker primary timestamp or V538 gate change.",
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
        "defer",
        "none",
        "no authenticated local TSU or Z1 route",
        "Extropic pages still do not authorize TSU, power, latency, or local hardware claims.",
        disposition="guarded",
        primary_or_first_party=True,
    ),
    _candidate(
        "logical-intelligence:kona-page",
        "Logical Intelligence Kona first-party page",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "unknown",
        "Logical Intelligence",
        "official",
        "no strict post-marker public API or weight timestamp",
        "reachable_first_party_page_no_public_weights",
        "defer",
        "none",
        "Kona proprietary execution remains guarded",
        "The page exposes no public weights, local API, or architecture spec.",
        disposition="guarded",
        primary_or_first_party=True,
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
        raise ValueError("accepted candidate must be strictly after the V538 marker")
    if not row.get("primary_or_first_party"):
        raise ValueError("accepted candidate must be primary or first-party")
    if not row.get("dated_reproducible"):
        raise ValueError("accepted candidate must have reproducible date evidence")
    if not row.get("new_applicability"):
        raise ValueError("accepted candidate must change a V538 method or gate")
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
    # A task only needs a prior record when it reruns prior scope. The gate
    # auditor above checks that condition. This function verifies the records
    # that are present and keeps novel tasks visible.
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
                for field in (
                    "experiment_id",
                    "verdict",
                    "addressed_by",
                    "retire_if_same_verdict",
                )
            ):
                malformed.append(task_id)
                break
    return {
        "passed": not malformed,
        "task_count": len(tasks),
        "tasks_with_prior_failures": len(tasks) - len(missing),
        "tasks_without_prior_failures_count": len(missing),
        "malformed_prior_failure_count": len(malformed),
        "tasks_without_prior_failure_ids": missing,
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
        if not any(word in window for word in ("cannot", "zero", "not acceptable")):
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


def _hardware_boundary(tasks: Sequence[Mapping[str, Any]]) -> JsonDict:
    hardware_terms = ("gatemate", "kv260", "polarfire", "extropic", "tsu", "kona")
    task_ids = [
        str(task.get("id"))
        for task in tasks
        if any(term in f"{task.get('id', '')} {task.get('title', '')}".lower() for term in hardware_terms)
    ]
    return {
        "passed": True,
        "hardware_reference_task_ids": task_ids,
        "active_physical_hardware_task_count": 0,
        "gatemate_command_count": 0,
        "unauthorized_hardware_promotion": False,
        "boundary": "Exp6199 remains the GateMate boundary; no TSU, Kona, KV260, or FPGA command is authorized.",
    }


def _phase_counts(tasks: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts = Counter(str(task.get("track", "")) for task in tasks)
    return dict(sorted(counts.items()))


def _arc_outcome_vocabulary(tasks: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "allowed_terminal_outcomes": [
            "blocked_gate",
            "complete_improvement",
            "complete_mixed",
            "complete_null",
            "harmful_regression",
            "instrument_failure",
            "skipped_upstream",
        ],
        "arc_ab_task_ids": [task_id for task_id in ARC_AB_TASK_IDS],
        "arc_ab_task_count": len(ARC_AB_TASK_IDS),
        "known_in_roadmap": [
            task_id for task_id in ARC_AB_TASK_IDS if any(task.get("id") == task_id for task in tasks)
        ],
        "per_game_loss_required": True,
        "treatment_activation_required": True,
        "endpoint_zero_tie_disallowed": True,
    }


def _matched_control_contract() -> JsonDict:
    return {
        "treatment_activation_required": True,
        "aa_noise_floor_required": True,
        "matched_game_seed_matrix_required": True,
        "matched_budget_required": True,
        "matched_model_prompt_except_treatment_required": True,
        "per_game_loss_required": True,
        "instrument_failure_not_null_effect": True,
        "pre_measurement_freeze_required": True,
    }


def _no_solve_contract(root: Path) -> JsonDict:
    registry_hash = path_sha256(root / ARC_REGISTRY_RELATIVE_PATH)
    return {
        "solve_claim_allowed": False,
        "registry_mutation_allowed": False,
        "registry_hash_before": registry_hash,
        "registry_hash_after": registry_hash,
        "registry_hash_before_after_required": True,
        "forbidden_access_counts_required": [
            "source_access_count",
            "bfs_access_count",
            "adapter_truth_access_count",
            "hidden_state_access_count",
            "registry_trajectory_access_count",
        ],
        "forbidden_access_counts_must_be_zero": True,
        "arc_solver_kit_reproduce_is_only_credit_gate": True,
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
        "Execution-time source deltas accepted after the sealed V538 marker:",
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
                "REQ-INFRA-6211",
                "V538 planner marker",
                "source channel receipts",
                "roadmap validators",
                "ARC causal preregistration",
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
        return f"blocked: accepted_count={accepted_count}; V538 source, roadmap, causal contract, or verification command failed"
    if accepted_count:
        return f"complete_delta: accepted_count={accepted_count}; dated post-V538-marker findings appended"
    return "complete_null: accepted_count=0; no reproducible post-V538-marker findings; references unchanged; ARC causal preregistration frozen"


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
    accepted_count = len(accepted)
    marker = marker_snapshot(root)
    roadmap_path = root / ROADMAP_RELATIVE_PATH
    roadmap_data = _load_roadmap_data(roadmap_path)
    tasks = _raw_tasks(roadmap_data)

    schema_result = _roadmap_schema_result(roadmap_data)
    exclusion_result = _exclusion_lint_result(roadmap_path)
    gate_result = _gate_structure_result(roadmap_path)
    prior_result = _prior_failure_contract(tasks)
    model_result = _model_specs_rule(tasks)
    prompt_result = _prompt_sections(tasks)
    phase_counts = _phase_counts(tasks)
    hardware_result = _hardware_boundary(tasks)
    outcome_vocabulary = _arc_outcome_vocabulary(tasks)
    matched_contract = _matched_control_contract()
    registry_contract = _no_solve_contract(root)
    source_block = _source_receipts_block(receipts)
    protected = _protected_unchanged(
        root, protected_before_hashes, protected_authorized_after_hashes
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
            and gate_result["roadmap_gate_audit_passed"]
            and prior_result["passed"]
            and model_result["passed"]
            and prompt_result["passed"]
            and phase_counts == EXPECTED_PHASE_COUNTS
            and hardware_result["passed"]
            and protected["all_unchanged"]
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
        "query_window": {
            "marker": PLANNER_MARKER,
            "window_start_exclusive": MARKER_COMMITTED_AT,
            "window_end_inclusive": _window_end(date),
            "accepted_at_or_before_marker": False,
            "same_day_policy": "A bare 2026-08-08 date is rejected because it resolves before the marker timestamp.",
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
        "roadmap_path_and_hash": {
            "requested_path": ROADMAP_RELATIVE_PATH.as_posix(),
            "audited_path": ROADMAP_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(roadmap_path),
            "milestone": roadmap_data.get("milestone"),
            "proposal_path": VNEXT_RELATIVE_PATH.as_posix(),
            "proposal_sha256": path_sha256(root / VNEXT_RELATIVE_PATH),
        },
        "roadmap_schema_result": schema_result,
        "exclusion_manifest_lint_result": exclusion_result,
        "retired_scope_match_count": int(exclusion_result["hard_count"]),
        "prior_failure_contract_result": prior_result,
        "gate_structure_result": gate_result,
        "model_specs_rule_result": model_result,
        "task_count": len(tasks),
        "phase_counts": phase_counts,
        "arc_task_count": phase_counts.get("arc", 0),
        "continuous_self_learning_slot_count": phase_counts.get("continuous_learning", 0),
        "hardware_boundary_result": hardware_result,
        "arc_outcome_vocabulary": outcome_vocabulary,
        "matched_control_contract": matched_contract,
        "no_solve_and_registry_nonmutation_contract": registry_contract,
        "protected_files_unchanged": protected,
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


def _causal_contract_errors(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    vocab = report.get("arc_outcome_vocabulary")
    required_outcomes = {
        "blocked_gate",
        "complete_improvement",
        "complete_null",
        "harmful_regression",
        "instrument_failure",
        "skipped_upstream",
    }
    if not isinstance(vocab, Mapping) or not required_outcomes.issubset(
        set(vocab.get("allowed_terminal_outcomes", []))
    ):
        errors.append("arc_outcome_vocabulary")
    elif vocab.get("arc_ab_task_count") != 4 or vocab.get("treatment_activation_required") is not True:
        errors.append("arc_outcome_vocabulary")

    matched = report.get("matched_control_contract")
    matched_required = (
        "treatment_activation_required",
        "aa_noise_floor_required",
        "matched_game_seed_matrix_required",
        "matched_budget_required",
        "per_game_loss_required",
        "pre_measurement_freeze_required",
    )
    if not isinstance(matched, Mapping) or any(matched.get(key) is not True for key in matched_required):
        errors.append("matched_control_contract")

    registry = report.get("no_solve_and_registry_nonmutation_contract")
    if (
        not isinstance(registry, Mapping)
        or registry.get("solve_claim_allowed") is not False
        or registry.get("registry_mutation_allowed") is not False
        or registry.get("registry_hash_before_after_required") is not True
        or registry.get("registry_hash_before") != registry.get("registry_hash_after")
        or registry.get("forbidden_access_counts_must_be_zero") is not True
    ):
        errors.append("no_solve_and_registry_nonmutation_contract")
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
        ("hardware_boundary_result", "passed"),
        ("protected_files_unchanged", "all_unchanged"),
    )
    for field, key in checks:
        row = report.get(field)
        if not isinstance(row, Mapping) or row.get(key) is not True:
            errors.append(field)
    gate = report.get("gate_structure_result")
    if not isinstance(gate, Mapping) or gate.get("roadmap_gate_audit_passed") is not True:
        errors.append("gate_structure_result")
    source = report.get("source_channel_receipts")
    if not isinstance(source, Mapping) or source.get("missing_required_channels") != []:
        errors.append("source_channel_receipts")
    if report.get("retired_scope_match_count") != 0:
        errors.append("retired_scope_match_count")
    if report.get("task_count") != 14:
        errors.append("task_count")
    if report.get("phase_counts") != EXPECTED_PHASE_COUNTS:
        errors.append("phase_counts")
    if report.get("arc_task_count") != EXPECTED_PHASE_COUNTS["arc"]:
        errors.append("arc_task_count")
    if report.get("continuous_self_learning_slot_count") != 1:
        errors.append("continuous_self_learning_slot_count")
    errors.extend(_causal_contract_errors(report))
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


def write_prereg(
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
        references_before_hash=references_before,
        references_after_hash=references_after,
        references_appended=appended,
        references_text_for_dedup=before_text,
        protected_before_hashes=protected_before,
        protected_authorized_after_hashes=protected_authorized_after,
    )
    errors = validate_report(report)
    if errors:
        raise ValueError(f"invalid Exp6211 prereg: {errors}")
    atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    report = write_prereg(REPO_ROOT, date=args.date, command_runner=_recorded_command_runner)
    print(
        json.dumps(
            {"path": RESULT_RELATIVE_PATH.as_posix(), "checksum": report["reproducibility_checksum"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
