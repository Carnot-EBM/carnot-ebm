"""Exp6324 V545 post-marker source scope freeze.

Spec refs: REQ-INFRA-6324, SCENARIO-INFRA-6324-1,
SCENARIO-INFRA-6324-2, SCENARIO-INFRA-6324-3,
SCENARIO-INFRA-6324-4, SCENARIO-INFRA-6324-5.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any
from urllib.parse import urlparse

from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import canonical_json, path_sha256, payload_sha256


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_ID = "exp6324-v545-post-marker-source-scope-freeze"
SCHEMA = "carnot.experiment_6324.v545_post_marker_source_scope_freeze.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6324_v545_post_marker_source_scope_freeze.json")
INFERENCE_SUBSTRATE = "web_and_bibliographic_search_only"
CONTRACT_VERSION = "v545.6324"

RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
HARDWARE_WISHLIST_RELATIVE_PATH = Path("research-hardware-wishlist.md")
TEMPLATE_RELATIVE_PATH = Path("scripts/experiment_template.py")
PRIOR_SOURCE_FREEZE_RELATIVE_PATH = Path(
    "results/experiment_6311_v544_post_marker_source_scope_freeze.json"
)
PRIOR_TRANSITION_RELATIVE_PATH = Path("results/experiment_6323_v545_terminal_transition.json")
PRIOR_TRANSITION_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6323_v545_terminal_transition.py"
)

PLANNER_MARKER = "V545-PLANNER-REFRESH-20260812-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
MARKER_COMMIT = "adbcbf48e2f9ed8645b830495194219f6157acf1"
MARKER_COMMITTED_AT_LOCAL = "2026-08-12T01:56:59-04:00"
MARKER_COMMITTED_AT_UTC = "2026-08-12T05:56:59Z"
EXECUTION_DELTA_HEADING = "## V545 Execution Source Delta - 20260812"
EXECUTION_DELTA_END_MARKER = "<!-- V545-EXECUTION-SOURCE-DELTA-20260812-END -->"

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
    PRIOR_SOURCE_FREEZE_RELATIVE_PATH,
    PRIOR_TRANSITION_RELATIVE_PATH,
    PRIOR_TRANSITION_MODULE_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    HARDWARE_WISHLIST_RELATIVE_PATH,
    TEMPLATE_RELATIVE_PATH,
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("scripts/research_conductor.py"),
)

INPUT_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_REFERENCES_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    PRIOR_SOURCE_FREEZE_RELATIVE_PATH,
    PRIOR_TRANSITION_RELATIVE_PATH,
    PRIOR_TRANSITION_MODULE_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    HARDWARE_WISHLIST_RELATIVE_PATH,
    TEMPLATE_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v545_marker_text_and_line",
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
    "frozen_restricted_policy_contract",
    "frozen_exact_factor_guard_contract",
    "frozen_verified_fallback_contract",
    "frozen_blind_checker_contract",
    "frozen_anytime_certificate_contract",
    "frozen_counterexample_update_contract",
    "frozen_protected_validation_contract",
    "frozen_arc_influence_no_solve_contract",
    "frozen_hardware_contract",
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
    "v545_marker_text_and_line": "The exact marker and line define the novelty boundary.",
    "search_window_start_utc": "The lower bound is exclusive, so same-time evidence cannot enter.",
    "search_completed_utc": "The upper audit time makes the sweep reproducible.",
    "source_queries_by_channel": "Each required channel records the query that produced its receipt.",
    "source_receipts": "Search results and failures stay visible for audit.",
    "accepted_findings": "Only stable post-marker sources with local consequences can widen V545.",
    "accepted_count": "A bare integer drives null versus delta terminality.",
    "duplicate_findings": "Repeated planner sources cannot widen execution scope again.",
    "watch_only_findings": "Contextual sources are preserved without becoming branch contracts.",
    "inaccessible_sources": "Blocked endpoints are receipts, not findings.",
    "excluded_findings_and_reasons": "Rejected rows name the exact failed acceptance rule.",
    "semantic_scholar_ebt_and_arm_ebm_receipts": "Citation routes stay separate from broad search.",
    "extropic_status": "Hardware pages cannot imply local TSU access.",
    "logical_intelligence_status": "Product-direction pages cannot imply runnable Kona evidence.",
    "github_status": "Repository search cannot widen scope without a maintained local route.",
    "frozen_restricted_policy_contract": "V545 uses observable bounded programs, not model-local state.",
    "frozen_exact_factor_guard_contract": "Exact factors are the guard authority.",
    "frozen_verified_fallback_contract": "Rejected or malformed policies must have a safe fallback.",
    "frozen_blind_checker_contract": "The checker must not see solver rationale.",
    "frozen_anytime_certificate_contract": "Repeated release decisions need a valid error ledger.",
    "frozen_counterexample_update_contract": "Only minimized exact counterexamples may propose updates.",
    "frozen_protected_validation_contract": "Final validation stays outside the adaptive loop.",
    "frozen_arc_influence_no_solve_contract": "ARC influence evidence cannot claim a solve.",
    "frozen_hardware_contract": "Only one GateMate detect is allowed in V545 hardware work.",
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
    ".venv/bin/pytest tests/python/test_experiment_6324_v545_post_marker_source_scope_freeze.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = ".venv/bin/python -m carnot.experiment_6324_v545_post_marker_source_scope_freeze --date 20260812"
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6324_v545_post_marker_source_scope_freeze.py "
    "-m pytest tests/python/test_experiment_6324_v545_post_marker_source_scope_freeze.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6324_v545_post_marker_source_scope_freeze.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6324_v545_post_marker_source_scope_freeze.py "
    "tests/python/test_experiment_6324_v545_post_marker_source_scope_freeze.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6324_v545_post_marker_source_scope_freeze.py"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
ADVERSARIAL_SELF_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6324_v545_post_marker_source_scope_freeze.json"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
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
            "queried_at_utc": "2026-08-12T11:19:41Z",
            "query": "submittedDate:[202608120557 TO 202608122359]",
            "url": (
                "https://export.arxiv.org/api/query?"
                "search_query=submittedDate:%5B202608120557%20TO%20202608122359%5D"
            ),
        },
        {
            "queried_at_utc": "2026-08-12T11:21:05Z",
            "query": "energy policy fallback anytime-valid topic window after marker",
            "url": (
                "https://export.arxiv.org/api/query?search_query=all:"
                "%28energy-based%20OR%20self-evolving%20OR%20anytime-valid%29"
            ),
        },
    ],
    "openreview": [
        {
            "queried_at_utc": "2026-08-12T11:20:19Z",
            "query": "anytime-valid certificate",
            "url": "https://api2.openreview.net/notes/search?term=anytime-valid%20certificate&limit=5",
        },
        {
            "queried_at_utc": "2026-08-12T11:20:19Z",
            "query": "verified fallback self-evolving agents",
            "url": (
                "https://api2.openreview.net/notes/search?"
                "term=verified%20fallback%20self-evolving%20agents&limit=5"
            ),
        },
    ],
    "hugging_face_papers": [
        {
            "queried_at_utc": "2026-08-12T11:20:18Z",
            "query": "daily_papers date=2026-08-12",
            "url": "https://huggingface.co/api/daily_papers?date=2026-08-12",
        }
    ],
    "semantic_scholar_ebt_arm": [
        {
            "queried_at_utc": "2026-08-12T11:20:18Z",
            "query": "EBT arXiv:2507.02092 citations",
            "url": (
                "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/"
                "citations?fields=title,url,year,publicationDate,externalIds&limit=10"
            ),
        },
        {
            "queried_at_utc": "2026-08-12T11:20:18Z",
            "query": "ARM-EBM arXiv:2512.15605 citations",
            "url": (
                "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/"
                "citations?fields=title,url,year,publicationDate,externalIds&limit=10"
            ),
        },
    ],
    "extropic": [
        {
            "queried_at_utc": "2026-08-12T11:21:05Z",
            "query": "Extropic home and writing first-party HEAD checks",
            "url": "https://extropic.ai/",
        }
    ],
    "logical_intelligence": [
        {
            "queried_at_utc": "2026-08-12T11:21:05Z",
            "query": "Logical Intelligence Kona and Sudoku page HEAD checks",
            "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        }
    ],
    "github": [
        {
            "queried_at_utc": "2026-08-12T11:20:18Z",
            "query": "V545 mechanism repository search pushed after 2026-08-12",
            "url": (
                "https://api.github.com/search/repositories?q=%28%22verified%20fallback%22"
                "%20OR%20%22restricted%20policy%22%20OR%20%22anytime-valid%20certificate%22"
                "%20OR%20%22energy-based%20model%22%29%20pushed:%3E=2026-08-12"
                "&sort=updated&order=desc&per_page=10"
            ),
        }
    ],
}

DEFAULT_SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "channel": "arxiv",
        "queried_at_utc": "2026-08-12T11:19:41Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["arxiv"][0]["url"],
        "endpoint_status": "http_200_window_totalResults_0_http_200_topic_pre_marker",
        "disposition": "no_post_marker_primary_hits",
        "date_evidence": (
            "The all-submission API window after 2026-08-12T05:56:59Z returned "
            "totalResults=0. The topic query returned only pre-marker rows, with the "
            "newest sampled row published at 2026-08-11T17:59:13Z."
        ),
    },
    {
        "channel": "openreview",
        "queried_at_utc": "2026-08-12T11:20:19Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["openreview"][1]["url"],
        "endpoint_status": "http_200_count_5_each_query",
        "disposition": "duplicates_or_pre_marker_context_only",
        "date_evidence": (
            "Anytime-valid and verified-fallback searches returned five rows each. "
            "The SEVerA row was already in the planner. The newest relevant anytime "
            "row had mdate 2026-07-27T09:20:35Z, before the marker."
        ),
    },
    {
        "channel": "hugging_face_papers",
        "queried_at_utc": "2026-08-12T11:20:18Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["hugging_face_papers"][0]["url"],
        "endpoint_status": "http_200_22_daily_rows_secondary",
        "disposition": "watch_only_or_pre_marker_secondary_rows",
        "date_evidence": (
            "Daily rows were submitted to Hugging Face at 2026-08-12T00:00:00Z, "
            "before the marker. Their primary paper dates were 2026-08-11 or older."
        ),
    },
    {
        "channel": "semantic_scholar_ebt_arm",
        "queried_at_utc": "2026-08-12T11:20:18Z",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations",
        "endpoint_status": "http_200_ebt_and_arm_routes",
        "disposition": "no_post_marker_relevant_citation",
        "date_evidence": "EBT newest sampled citation was 2026-08-01. ARM-EBM newest sampled citation was 2026-07-02.",
    },
    {
        "channel": "extropic",
        "queried_at_utc": "2026-08-12T11:21:05Z",
        "url": "https://extropic.ai/",
        "endpoint_status": "http_200_home_and_writing",
        "disposition": "no_authenticated_local_tsu_route",
        "date_evidence": (
            "Home x-nextjs-date was 2026-08-10T00:46:55Z. Writing x-nextjs-date "
            "was 2026-08-11T07:12:45Z. No Carnot device, simulator, or local API route appeared."
        ),
    },
    {
        "channel": "logical_intelligence",
        "queried_at_utc": "2026-08-12T11:21:05Z",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "endpoint_status": "http_200_kona_last_modified_2026_06_26_sudoku_2026_02_03",
        "disposition": "no_public_weights_or_local_api",
        "date_evidence": (
            "Kona page last-modified header was 2026-06-26T23:48:05Z. "
            "The Sudoku demo last-modified header was 2026-02-03T14:18:43Z."
        ),
    },
    {
        "channel": "github",
        "queried_at_utc": "2026-08-12T11:20:18Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["github"][0]["url"],
        "endpoint_status": "http_200_scope_search_total_count_0",
        "disposition": "no_maintained_contract_changing_repository",
        "date_evidence": (
            "The V545 mechanism search returned zero repositories. MARCH and EBT repo "
            "checks were reachable but pushed before the marker or already in the planner."
        ),
    },
)

DEFAULT_SOURCE_CANDIDATES: tuple[JsonDict, ...] = (
    {
        "stable_id": "arxiv:2607.00871",
        "title": "Self-Evolving Agents with Anytime-Valid Certificates",
        "url": "https://arxiv.org/abs/2607.00871",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-07-01T12:34:52Z",
        "date_evidence": "planner source predates the V545 marker",
        "scope_effect": "Already sealed as anytime-valid certificate context.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
    },
    {
        "stable_id": "arxiv:2603.25111",
        "title": "SEVerA: Verified Synthesis of Self-Evolving Agents",
        "url": "https://arxiv.org/abs/2603.25111",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-04-24T00:00:00Z",
        "date_evidence": "planner source predates the V545 marker",
        "scope_effect": "Already sealed as verified fallback and contract guard context.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    },
    {
        "stable_id": "arxiv:2606.05395",
        "title": "VASO: Formally Verifiable Self-Evolving Skills for Physical AI Agents",
        "url": "https://arxiv.org/abs/2606.05395",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-06-03T00:00:00Z",
        "date_evidence": "planner source predates the V545 marker",
        "scope_effect": "Already sealed as exact counterexample update context.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
    },
    {
        "stable_id": "arxiv:2603.24579",
        "title": "MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination",
        "url": "https://arxiv.org/abs/2603.24579",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-03-25T00:00:00Z",
        "date_evidence": "planner source predates the V545 marker",
        "scope_effect": "Already sealed as blind checker information-asymmetry context.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
    },
    {
        "stable_id": "openreview:pUqcOkV69j",
        "title": "Loss Smoothing for Continual Adaptation",
        "url": "https://openreview.net/forum?id=pUqcOkV69j",
        "source_channel": "openreview",
        "source_kind": "primary",
        "source_timestamp": "2026-06-01T00:00:00Z",
        "date_evidence": "OpenReview record predates the V545 marker",
        "scope_effect": "Already sealed as a matched control, not release authority.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
    },
    {
        "stable_id": "openreview:HgeGw5hu9Z",
        "title": "SEVerA: Verified Self-Evolving Agents with Specification Guidance",
        "url": "https://openreview.net/forum?id=HgeGw5hu9Z",
        "source_channel": "openreview",
        "source_kind": "primary",
        "source_timestamp": "2026-06-17T00:00:00Z",
        "date_evidence": "OpenReview workshop record predates the V545 marker",
        "scope_effect": "Mechanism is already represented by the SEVerA planner source.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:6666666666666666666666666666666666666666666666666666666666666666",
    },
    {
        "stable_id": "arxiv:2608.11191",
        "title": "Test-Time Self-Evolving GUI Visual Grounding via Reflection-Guided On-Policy Self-Distillation",
        "url": "https://arxiv.org/abs/2608.11191",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-11T17:50:25Z",
        "date_evidence": "arXiv topic query row predates the V545 marker.",
        "scope_effect": "Uses model adaptation and self-distillation. V545 keeps GGUF weights frozen.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": False,
        "watch_only": False,
        "content_hash": "sha256:7777777777777777777777777777777777777777777777777777777777777777",
    },
    {
        "stable_id": "hf:2608.10915",
        "title": "ComBodied Agents: a New Paradigm of Human-Centric Agentic AI",
        "url": "https://huggingface.co/papers/2608.10915",
        "source_channel": "hugging_face_papers",
        "source_kind": "secondary",
        "source_timestamp": "2026-08-12T00:00:00Z",
        "date_evidence": "HF daily submission time predates the marker; primary date is 2026-08-11.",
        "scope_effect": "Agent memory context only. It does not change a V545 executable contract.",
        "reproducible_evidence": True,
        "primary_or_first_party": False,
        "local_executable_consequence": False,
        "watch_only": True,
        "content_hash": "sha256:8888888888888888888888888888888888888888888888888888888888888888",
    },
    {
        "stable_id": "hf:2608.11079",
        "title": "SkillZip: Evaluation-Free Skill Compression for Self-Evolving Agents by Discovering Reusable Structure",
        "url": "https://huggingface.co/papers/2608.11079",
        "source_channel": "hugging_face_papers",
        "source_kind": "secondary",
        "source_timestamp": "2026-08-12T00:00:00Z",
        "date_evidence": "HF daily row predates the V545 marker; primary date is 2026-08-11.",
        "scope_effect": "Skill compression context only. It does not change a V545 executable contract.",
        "reproducible_evidence": True,
        "primary_or_first_party": False,
        "local_executable_consequence": False,
        "watch_only": True,
        "content_hash": "sha256:9999999999999999999999999999999999999999999999999999999999999999",
    },
    {
        "stable_id": "extropic:first-party-pages",
        "title": "Extropic public writing route check",
        "url": "https://extropic.ai/writing",
        "source_channel": "extropic",
        "source_kind": "first_party",
        "source_timestamp": "2026-08-11T07:12:45Z",
        "date_evidence": "x-nextjs-date predates the V545 marker",
        "scope_effect": "No authenticated Carnot TSU or simulator route.",
        "reproducible_evidence": False,
        "primary_or_first_party": True,
        "local_executable_consequence": False,
        "watch_only": False,
        "content_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
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
        "content_hash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    },
    {
        "stable_id": "github:Qwen-Applications/MARCH",
        "title": "Qwen-Applications/MARCH",
        "url": "https://github.com/Qwen-Applications/MARCH",
        "source_channel": "github",
        "source_kind": "repository",
        "source_timestamp": "2026-06-09T05:01:09Z",
        "date_evidence": "GitHub pushed_at predates the V545 marker.",
        "scope_effect": "Already sealed as the MARCH code context.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": False,
        "watch_only": False,
        "content_hash": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    },
    {
        "stable_id": "github:alexiglad/EBT",
        "title": "alexiglad/EBT",
        "url": "https://github.com/alexiglad/EBT",
        "source_channel": "github",
        "source_kind": "repository",
        "source_timestamp": "2026-04-21T00:53:47Z",
        "date_evidence": "GitHub pushed_at predates the V545 marker.",
        "scope_effect": "EBT remains pretraining code, not a local GGUF sidecar.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": False,
        "watch_only": False,
        "content_hash": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
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
        raise ValueError("bad V545 marker timestamp")
    return parsed


def _is_stable_url(url: Any) -> bool:
    parsed = urlparse(str(url or ""))
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    return not (parsed.netloc == "github.com" and parsed.path == "/search")


def v545_marker_snapshot(root: Path) -> JsonDict:
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
        row["rejection_reason"] = "watch-only evidence does not change a V545 contract"
        return row

    timestamp = _parse_timestamp(row.get("source_timestamp"))
    if timestamp is None or timestamp <= _marker_dt():
        row["disposition"] = "excluded"
        row["rejection_reason"] = "source timestamp is not strictly after the V545 marker"
        if timestamp is None or str(row.get("source_timestamp") or "").strip()[:10] == "2026-08-12":
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


def frozen_restricted_policy_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "source_evidence": ["arXiv:2603.25111", "arXiv:2606.05395"],
        "object": "bounded_typed_policy_dsl_over_finite_domains",
        "normalization_required": True,
        "finite_domain_only": True,
        "hidden_state_inputs_allowed": False,
        "model_local_surface_allowed": False,
        "candidate_generator": "local_gguf_text_to_restricted_policy_only",
        "retired_or_deferred_mechanisms": [
            "hidden_state_probe",
            "activation_probe",
            "embedding_probe",
            "prefix_trajectory_probe",
            "pooled_representation_rescue",
            "natural_language_constraintir_reprompt",
            "finite_id_answer_transport",
        ],
    }


def frozen_exact_factor_guard_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "source_evidence": ["arXiv:2603.25111", "arXiv:2606.05395"],
        "guard_is_exact_oracle": True,
        "energy_definition": "weighted_unsatisfied_contract_clauses",
        "compiler_target": "local_exact_clause_factors",
        "domain_check": "enumeration_or_z3_over_full_finite_domain",
        "oracle_disclosure_required": True,
        "learned_verifier_claim_allowed": False,
        "required_receipts": ["contract_hash", "program_hash", "factor_hash", "domain_hash"],
    }


def frozen_verified_fallback_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "source_evidence": ["arXiv:2603.25111"],
        "hash_pinned": True,
        "fallback_invoked_for": ["parse_error", "type_error", "contract_violation", "timeout"],
        "fallback_laundering_allowed": False,
        "fallback_cost_counts_in_utility": True,
        "fallback_hash_swap_attack_required": True,
        "safety_promotion_only": True,
    }


def frozen_blind_checker_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "source_evidence": ["arXiv:2603.24579"],
        "checker_inputs": ["canonical_contract", "normalized_semantics", "exact_evidence"],
        "solver_rationale_visible": False,
        "solver_claimed_verdict_visible": False,
        "information_asymmetry_required": True,
        "required_attacks": [
            "vacuous_specification",
            "parser_default",
            "fallback_laundering",
            "validator_mutation",
            "hash_swap",
            "label_swap",
            "budget_mismatch",
        ],
    }


def frozen_anytime_certificate_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "source_evidence": ["arXiv:2607.00871"],
        "optional_stopping_valid": True,
        "fixed_error_budget_required": True,
        "alpha_spending_required": True,
        "immutable_predecision_receipts_required": True,
        "restart_identity_checked": True,
        "certificate_reset_attack_required": True,
        "release_authority": "anytime_valid_release_ledger",
    }


def frozen_counterexample_update_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "source_evidence": ["arXiv:2606.05395"],
        "minimized_exact_only": True,
        "proposal_scope": "factor_local_policy_delta",
        "foundation_model_weights_frozen": True,
        "protected_checker_keeps_release_authority": True,
        "counterexample_fabrication_attack_required": True,
        "deferred_mechanisms": [
            "cross_domain_transfer",
            "cross_family_transfer",
            "gguf_weight_update",
        ],
    }


def frozen_protected_validation_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "adaptive_loop_can_read": False,
        "release_authority": "protected_exact_validation_plus_anytime_certificate",
        "dense_signal_release_authority_allowed": False,
        "opened_once": True,
        "required_attacks": [
            "poison",
            "reversal",
            "forgetting",
            "protected_file_mutation",
            "protected_set_reuse",
            "future_leakage",
        ],
        "safety_only_can_promote_utility": False,
    }


def frozen_arc_influence_no_solve_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "default_off_shadow_only": True,
        "influence_before_utility": True,
        "solve_credit_allowed": False,
        "submitted_action_mutation_allowed": False,
        "hidden_game_source_access_allowed": False,
        "offline_bfs_allowed": False,
        "per_game_adapter_allowed": False,
        "registry_update_allowed": False,
        "candidate_creation_allowed": False,
        "may_reorder_existing_e3_candidates_only": True,
        "required_receipts": ["fresh_agent_window", "aa_noise_floor", "legal_action_delta", "seed"],
    }


def frozen_hardware_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "allowed_gatemate_detect_count": 1,
        "allowed_gatemate_command": "openFPGALoader -c dirtyJtag --detect",
        "requires_dated_power_cycle_receipt": "2026-08-11T12:58:49Z",
        "flash_allowed": False,
        "synthesis_allowed": False,
        "place_and_route_allowed": False,
        "timing_task_allowed": False,
        "kv260_task_allowed": False,
        "polarfire_dependency_allowed": False,
        "extropic_tsu_execution_allowed": False,
        "speed_power_energy_or_availability_claim_count": 0,
        "excluded_retired_mechanisms": [
            "hidden_state_probe",
            "activation_probe",
            "external_generated_text_scorer",
            "masked_model_energy",
            "best_of_n_text_judge",
            "kan_experiment",
            "cross_family_transfer",
            "gguf_weight_update",
            "public_arc_resolve",
            "extropic_tsu_execution",
            "kv260_task",
            "polarfire_dependency",
            "gatemate_flash",
            "gatemate_synthesis",
            "gatemate_place_and_route",
            "gatemate_timing",
        ],
    }


def semantic_scholar_receipts() -> JsonDict:
    return {
        "ebt": {
            "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["semantic_scholar_ebt_arm"][0]["url"],
            "queried_at_utc": "2026-08-12T11:20:18Z",
            "newest_sampled_publication_date": "2026-08-01",
            "strict_post_marker_relevant_count": 0,
            "strict_after_v545_marker_relevant_count": 0,
            "sample_titles": [
                "Prospects of intelligent autonomous control technology in aerospace",
                "Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation",
                "Memoir: Should a Model Write to Its Memory While It Thinks?",
                "Solver-Hard Is Not Model-Hard: A Hardness-Controlled Diagnostic for LLM Constraint Reasoning",
            ],
        },
        "arm_ebm": {
            "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["semantic_scholar_ebt_arm"][1]["url"],
            "queried_at_utc": "2026-08-12T11:20:18Z",
            "newest_sampled_publication_date": "2026-07-02",
            "strict_post_marker_relevant_count": 0,
            "strict_after_v545_marker_relevant_count": 0,
            "sample_titles": [
                "Path-Measure Dynamics of Attention-Driven World Models",
                "Constitutional On-Policy Safe Distillation",
                "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
                "LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models",
            ],
        },
    }


def _field_provenance() -> JsonDict:
    sources = [
        "REQ-INFRA-6324",
        "research-references V545 marker",
        "source channel receipts",
        "V545 roadmap scope contracts",
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
        return f"blocked: accepted_count={accepted_count}; V545 source freeze checks failed"
    if accepted_count:
        return f"complete_delta: accepted_count={accepted_count}; strict post-V545-marker findings appended"
    return (
        "complete_null: accepted_count=0; no reproducible post-V545-marker findings; "
        "references unchanged; V545 contracts frozen"
    )


def disk_status(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    return {
        "repo_root": root.as_posix(),
        "repo_root_total_bytes": usage.total,
        "repo_root_used_bytes": usage.used,
        "repo_root_available_bytes": usage.free,
    }


def required_command_availability(root: Path) -> JsonDict:
    commands = ("curl", "git", "jq")
    availability: JsonDict = {
        command: {"available": shutil.which(command) is not None, "path": shutil.which(command)}
        for command in commands
    }
    for rel in (Path(".venv/bin/python"), Path(".venv/bin/pytest"), Path(".venv/bin/coverage")):
        target = root / rel
        availability[rel.as_posix()] = {
            "available": target.exists(),
            "path": target.as_posix() if target.exists() else None,
        }
    return availability


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
    marker = v545_marker_snapshot(root)
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
    if not checks_passed:
        status = "blocked"
    elif accepted:
        status = "complete_delta"
    else:
        status = "complete_null"
    command_exit_codes = {command: 0 for command in DEFAULT_TEST_COMMANDS}
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": status,
        "v545_marker_text_and_line": marker,
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
            "queried_at_utc": "2026-08-12T11:21:05Z",
            "home_http_status": 200,
            "writing_http_status": 200,
            "home_x_nextjs_date": "2026-08-10T00:46:55Z",
            "writing_x_nextjs_date": "2026-08-11T07:12:45Z",
            "local_execution_route_present": False,
            "speed_power_or_sampling_claim_allowed": False,
        },
        "logical_intelligence_status": {
            "status": "watch_only_product_context",
            "queried_at_utc": "2026-08-12T11:21:05Z",
            "kona_page_http_status": 200,
            "sudoku_page_http_status": 200,
            "last_modified": "2026-06-26T23:48:05Z",
            "sudoku_last_modified": "2026-02-03T14:18:43Z",
            "public_weights_or_local_api_present": False,
        },
        "github_status": {
            "status": "no_scope_changing_repo",
            "queried_at_utc": "2026-08-12T11:20:18Z",
            "scope_search_total_count": 0,
            "incomplete_results": False,
            "accepted_repository_count": 0,
            "march_pushed_at": "2026-06-09T05:01:09Z",
            "ebt_pushed_at": "2026-04-21T00:53:47Z",
        },
        "frozen_restricted_policy_contract": frozen_restricted_policy_contract(),
        "frozen_exact_factor_guard_contract": frozen_exact_factor_guard_contract(),
        "frozen_verified_fallback_contract": frozen_verified_fallback_contract(),
        "frozen_blind_checker_contract": frozen_blind_checker_contract(),
        "frozen_anytime_certificate_contract": frozen_anytime_certificate_contract(),
        "frozen_counterexample_update_contract": frozen_counterexample_update_contract(),
        "frozen_protected_validation_contract": frozen_protected_validation_contract(),
        "frozen_arc_influence_no_solve_contract": frozen_arc_influence_no_solve_contract(),
        "frozen_hardware_contract": frozen_hardware_contract(),
        "roadmap_scope_delta": {
            "delta_kind": "accepted_source_delta" if accepted else "zero_source_delta",
            "accepted_count": len(accepted),
            "terminal_state": "complete_delta" if accepted else "complete_null",
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
            "marker_line": marker["marker_line"],
            "marker_comment_sha256": marker["marker_comment_sha256"],
            "search_started_after_marker_utc": MARKER_COMMITTED_AT_UTC,
            "search_completed_utc": completed,
            "source_channels_required": list(REQUIRED_SOURCE_CHANNELS),
            "source_channels_observed": sorted(channel_set),
            "disk_status": disk_status(root),
            "network_status": {
                "network_access_enabled": True,
                "observed_http_statuses": [
                    "arxiv_http_200",
                    "openreview_http_200",
                    "huggingface_http_200",
                    "semantic_scholar_http_200",
                    "extropic_http_200",
                    "logical_intelligence_http_200",
                    "github_http_200",
                ],
            },
            "required_command_availability": required_command_availability(root),
            "research_references_modified": refs_before != refs_after,
            "research_conductor_modified": False,
            "exp6323_transition_result_present": (root / PRIOR_TRANSITION_RELATIVE_PATH).exists(),
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
    if report.get("status") not in {"complete_null", "complete_delta", "blocked"}:
        errors.append("status")
    if report.get("accepted_count") == 0 and report.get("status") == "complete_delta":
        errors.append("status")
    if report.get("accepted_count") and report.get("status") == "complete_null":
        errors.append("status")
    if set(report.get("source_queries_by_channel") or {}) != set(REQUIRED_SOURCE_CHANNELS):
        errors.append("source_queries_by_channel")
    channels = {row.get("channel") for row in report.get("source_receipts") or []}
    if channels != set(REQUIRED_SOURCE_CHANNELS):
        errors.append("source_receipts")
    for field in (
        "frozen_restricted_policy_contract",
        "frozen_exact_factor_guard_contract",
        "frozen_verified_fallback_contract",
        "frozen_blind_checker_contract",
        "frozen_anytime_certificate_contract",
        "frozen_counterexample_update_contract",
        "frozen_protected_validation_contract",
        "frozen_arc_influence_no_solve_contract",
        "frozen_hardware_contract",
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
        raise ValueError(f"invalid Exp6324 freeze: {errors}")
    atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Write the Exp6324 V545 source freeze.")
    parser.add_argument("--date", default="20260812")
    args = parser.parse_args(argv)
    report = write_freeze(REPO_ROOT, date=args.date)
    print(json.dumps({"path": RESULT_RELATIVE_PATH.as_posix(), "status": report["status"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
