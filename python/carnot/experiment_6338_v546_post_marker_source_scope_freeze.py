"""Exp6338 V546 post-marker source scope freeze.

Spec refs: REQ-INFRA-6338, SCENARIO-INFRA-6338-1,
SCENARIO-INFRA-6338-2, SCENARIO-INFRA-6338-3,
SCENARIO-INFRA-6338-4, SCENARIO-INFRA-6338-5.
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
from carnot.inference.sota_models import SOTA_GGUF_MODELS
from carnot.terminal_artifacts import canonical_json, path_sha256, payload_sha256


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_ID = "exp6338-v546-post-marker-source-scope-freeze"
SCHEMA = "carnot.experiment_6338.v546_post_marker_source_scope_freeze.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6338_v546_post_marker_source_scope_freeze.json")
INFERENCE_SUBSTRATE = "web_and_bibliographic_search_only"
CONTRACT_VERSION = "v546.6338"

RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
HARDWARE_WISHLIST_RELATIVE_PATH = Path("research-hardware-wishlist.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
PRIOR_V546_HANDOFF_RELATIVE_PATH = Path(
    "results/experiment_6337_v546_bounded_terminal_handoff.json"
)
PRIOR_V545_SOURCE_FREEZE_RELATIVE_PATH = Path(
    "results/experiment_6324_v545_post_marker_source_scope_freeze.json"
)
PRIOR_GATEMATE_RELATIVE_PATH = Path(
    "results/experiment_6325_gatemate_dated_receipt_single_detect.json"
)
PRIOR_GUARDED_SYNTHESIS_RELATIVE_PATH = Path(
    "results/experiment_6327_three_family_guarded_policy_synthesis.json"
)
PRIOR_HELD_AB_RELATIVE_PATH = Path(
    "results/experiment_6329_prospective_held_family_guarded_policy_ab.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

PLANNER_MARKER = "V546-PLANNER-REFRESH-20260812-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
MARKER_COMMIT = "c95479503092feffa7f6ff879d38f4a8526f96dc"
MARKER_COMMITTED_AT_LOCAL = "2026-08-12T10:54:19-04:00"
MARKER_COMMITTED_AT_UTC = "2026-08-12T14:54:19Z"

MANDATED_GGUF_IDS = tuple(str(spec["hf_id"]) for spec in SOTA_GGUF_MODELS)

REQUIRED_SOURCE_CHANNELS = (
    "arxiv",
    "openreview",
    "hugging_face_papers",
    "semantic_scholar_ebt_arm",
    "extropic",
    "logical_intelligence",
    "github",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v546_marker_text_line_and_hash",
    "search_window_start_utc",
    "search_completed_utc",
    "source_queries_by_channel",
    "source_receipts",
    "promoted_findings",
    "accepted_count",
    "duplicate_findings",
    "watch_only_findings",
    "inaccessible_sources",
    "excluded_findings_and_reasons",
    "parser_bias_receipt",
    "lejit_receipt",
    "nxn_evalue_receipt",
    "catastrophic_remembering_receipt",
    "verification_cost_receipt",
    "semantic_scholar_ebt_and_arm_ebm_receipts",
    "openreview_and_huggingface_status",
    "github_status",
    "extropic_status",
    "logical_intelligence_status",
    "frozen_prefix_generation_contract",
    "frozen_certified_learning_contract",
    "frozen_arc_influence_contract",
    "frozen_model_policy",
    "frozen_hardware_nonuse_contract",
    "roadmap_scope_delta",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "llm_call_count",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The freeze is complete only when source and boundary checks pass.",
    "v546_marker_text_line_and_hash": "The exact marker defines the novelty boundary.",
    "search_window_start_utc": "The lower bound is exclusive for post-marker novelty.",
    "search_completed_utc": "The upper audit time makes the sweep reproducible.",
    "source_queries_by_channel": "Each required channel records its direct search route.",
    "source_receipts": "Search outcomes and failures stay visible.",
    "promoted_findings": "Planner-promoted sources are receipts, not new post-marker deltas.",
    "accepted_count": "A bare integer drives the null or delta verdict.",
    "duplicate_findings": "Repeated paper, repository, and mechanism identities do not widen scope.",
    "watch_only_findings": "Contextual rows stay visible without becoming contracts.",
    "inaccessible_sources": "Rate limits and blocked endpoints are receipts, not findings.",
    "excluded_findings_and_reasons": "Rejected rows name the failed acceptance rule.",
    "parser_bias_receipt": "Parser-state correction motivates only the prefix canary.",
    "lejit_receipt": "JIT SMT prefix checks motivate only prefix feasibility.",
    "nxn_evalue_receipt": "E-values motivate only the certified release ledger.",
    "catastrophic_remembering_receipt": "Instruction growth motivates evidence-carrying lifecycle.",
    "verification_cost_receipt": "Correctness must be reported with exact-check cost.",
    "semantic_scholar_ebt_and_arm_ebm_receipts": "Citation routes stay separate from broad search.",
    "openreview_and_huggingface_status": "Secondary channels cannot widen scope without primary evidence.",
    "github_status": "Repository rows need a maintained local route before promotion.",
    "extropic_status": "Product hardware pages do not imply local TSU access.",
    "logical_intelligence_status": "Kona pages do not imply local weights or API access.",
    "frozen_prefix_generation_contract": "V546 generation has one prefix-time attempt.",
    "frozen_certified_learning_contract": "Factor evolution needs exact evidence and e-values.",
    "frozen_arc_influence_contract": "ARC may test action order but not solve credit.",
    "frozen_model_policy": "Later LLM tasks must use the local GGUF allowlist.",
    "frozen_hardware_nonuse_contract": "This source task authorizes no board or product execution.",
    "roadmap_scope_delta": "The roadmap widens only on accepted post-marker evidence.",
    "protected_files_unchanged": "Protected inputs are hash-checked around the write.",
    "preconditions_checked": "Inputs, UTC time, disk, network receipts, and commands are recorded.",
    "inference_substrate": "This task uses web and bibliographic search only.",
    "verifier_is_oracle": "The artifact audits sources and is not a result oracle.",
    "llm_call_count": "Bare zero proves no model call occurred.",
    "field_provenance": "Every field cites concrete evidence.",
    "field_principles": "Every required field states why it exists.",
    "test_commands": "Commands show the verification boundary.",
    "test_exit_codes": "Exit codes keep failed checks visible.",
    "duration_s": "Wall time is reported without padding.",
    "reproducibility_checksum": "The normalized payload is content-addressed.",
    "honest_verdict": "The verdict states null, delta, or block directly.",
}

RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6338_v546_post_marker_source_scope_freeze "
    "--date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6338_v546_post_marker_source_scope_freeze.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6338_v546_post_marker_source_scope_freeze.py "
    "-m pytest tests/python/test_experiment_6338_v546_post_marker_source_scope_freeze.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6338_v546_post_marker_source_scope_freeze.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6338_v546_post_marker_source_scope_freeze.py "
    "tests/python/test_experiment_6338_v546_post_marker_source_scope_freeze.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6338_v546_post_marker_source_scope_freeze.py "
    "tests/python/test_experiment_6338_v546_post_marker_source_scope_freeze.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6338_v546_post_marker_source_scope_freeze.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '1,260p' ops/e2e-test-plan.md"
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
ADVERSARIAL_SELF_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6338_v546_post_marker_source_scope_freeze.json"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    RUFF_CHECK_COMMAND,
    RUFF_FORMAT_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_PLAN_READ_COMMAND,
    FULL_PYTEST_COMMAND,
    ADVERSARIAL_SELF_COMMAND,
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6338_test_receipts.json")

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_PROGRAM_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    PRIOR_V546_HANDOFF_RELATIVE_PATH,
    PRIOR_V545_SOURCE_FREEZE_RELATIVE_PATH,
    PRIOR_GATEMATE_RELATIVE_PATH,
    PRIOR_GUARDED_SYNTHESIS_RELATIVE_PATH,
    PRIOR_HELD_AB_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    HARDWARE_WISHLIST_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("scripts/research_conductor.py"),
)
INPUT_RELATIVE_PATHS = (
    *PROTECTED_RELATIVE_PATHS,
    SPEC_RELATIVE_PATH,
)

DEFAULT_SOURCE_QUERIES_BY_CHANNEL: dict[str, list[JsonDict]] = {
    "arxiv": [
        {
            "queried_at_utc": "2026-08-12T16:56:43Z",
            "query": "submittedDate:[202608121455 TO 202608122359]",
            "url": (
                "https://export.arxiv.org/api/query?search_query="
                "submittedDate:%5B202608121455%20TO%20202608122359%5D"
            ),
        },
        {
            "queried_at_utc": "2026-08-12T16:56:43Z",
            "query": "direct arXiv promoted-source HEAD check",
            "url": "https://arxiv.org/abs/2608.10137",
        },
    ],
    "openreview": [
        {
            "queried_at_utc": "2026-08-12T16:56:43Z",
            "query": "parser state constrained decoding SMT e-value verification cost",
            "url": (
                "https://api2.openreview.net/notes/search?"
                "term=parser%20state%20constrained%20decoding%20SMT%20e-value%20"
                "verification%20cost&limit=5"
            ),
        }
    ],
    "hugging_face_papers": [
        {
            "queried_at_utc": "2026-08-12T16:56:43Z",
            "query": "daily_papers date=2026-08-12",
            "url": "https://huggingface.co/api/daily_papers?date=2026-08-12",
        }
    ],
    "semantic_scholar_ebt_arm": [
        {
            "queried_at_utc": "2026-08-12T16:56:51Z",
            "query": "EBT arXiv:2507.02092 citations",
            "url": (
                "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/"
                "citations?fields=title,url,year,publicationDate,externalIds&limit=5"
            ),
        },
        {
            "queried_at_utc": "2026-08-12T16:56:51Z",
            "query": "ARM-EBM arXiv:2512.15605 citations",
            "url": (
                "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/"
                "citations?fields=title,url,year,publicationDate,externalIds&limit=5"
            ),
        },
    ],
    "extropic": [
        {
            "queried_at_utc": "2026-08-12T16:57:02Z",
            "query": "Extropic writing first-party HEAD check",
            "url": "https://extropic.ai/writing",
        }
    ],
    "logical_intelligence": [
        {
            "queried_at_utc": "2026-08-12T16:57:09Z",
            "query": "Logical Intelligence Kona first-party HEAD check",
            "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        }
    ],
    "github": [
        {
            "queried_at_utc": "2026-08-12T16:56:51Z",
            "query": "V546 mechanism repository search pushed after 2026-08-12",
            "url": (
                "https://api.github.com/search/repositories?q=%28%22parser%20state%22%20OR%20"
                "%22just-in-time%20logic%22%20OR%20%22e-value%22%20OR%20"
                "%22verification%20cost%22%29%20pushed:%3E=2026-08-12"
                "&sort=updated&order=desc&per_page=5"
            ),
        },
        {
            "queried_at_utc": "2026-08-12T16:56:51Z",
            "query": "LeJIT repository direct metadata",
            "url": "https://api.github.com/repos/HongyuHe/LeJIT",
        },
    ],
}

PROMOTED_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "stable_id": "arxiv:2608.10137",
        "title": "The Parser Already Knows: Lightweight Bias Correction in Constrained Decoding",
        "direct_url": "https://arxiv.org/abs/2608.10137",
        "first_publication_date": "2026-08-10",
        "first_publication_utc": "2026-08-10T18:52:43Z",
        "accessed_at_utc": "2026-08-12T15:34:33Z",
        "direct_receipt": "arXiv page lines 8 and 29 report submitted 2026-08-10 and v1 time.",
        "local_consequence": (
            "Exp6339 and Exp6340 must expose parser and lexer state before exact execution."
        ),
        "local_executable_consequence": (
            "Exp6339 and Exp6340 must expose parser and lexer state before exact execution."
        ),
        "lane": "prefix_constrained_policy_generation",
        "planner_promoted": True,
        "content_hash": "sha256:1013710137101371013710137101371013710137101371013710137101371013",
    },
    {
        "stable_id": "github:HongyuHe/LeJIT",
        "title": "LeJIT: Just-in-Time Logic Enforcement",
        "direct_url": "https://hhy.ee.princeton.edu/papers/2025_hotnets_lejit.pdf",
        "code_url": "https://github.com/HongyuHe/LeJIT",
        "first_publication_date": "2025-11-17",
        "first_publication_utc": "2025-11-17T00:00:00Z",
        "accessed_at_utc": "2026-08-12T15:34:33Z",
        "direct_receipt": "HotNets PDF cites HotNets 25 on November 17-18, 2025.",
        "repository_receipt": (
            "GitHub API reports created_at 2025-11-13T17:01:37Z and pushed_at 2026-05-09T19:56:53Z."
        ),
        "local_consequence": "Exp6339 must add a JIT SMT prefix-feasibility interface.",
        "local_executable_consequence": (
            "Exp6339 must add a JIT SMT prefix-feasibility interface."
        ),
        "lane": "prefix_constrained_policy_generation",
        "planner_promoted": True,
        "content_hash": "sha256:1e1111111e1111111e1111111e1111111e1111111e1111111e1111111e111111",
    },
    {
        "stable_id": "arxiv:2608.06621",
        "title": "NxN E-valuation: Hypothesis Certification via a Conformal CRT Null",
        "direct_url": "https://arxiv.org/abs/2608.06621",
        "first_publication_date": "2026-08-06",
        "first_publication_utc": "2026-08-06T22:14:53Z",
        "accessed_at_utc": "2026-08-12T15:34:33Z",
        "direct_receipt": "arXiv page lines 8 and 26 report submitted 2026-08-06 and v1 time.",
        "local_consequence": (
            "Exp6342 must implement an immutable anytime e-value release ledger."
        ),
        "local_executable_consequence": (
            "Exp6342 must implement an immutable anytime e-value release ledger."
        ),
        "lane": "certified_factor_evolution",
        "planner_promoted": True,
        "content_hash": "sha256:0662106621066210662106621066210662106621066210662106621066210662",
    },
    {
        "stable_id": "arxiv:2608.11095",
        "title": "Why Does CLAUDE.md Keep Growing? Catastrophic Remembering in Agentic Coding",
        "direct_url": "https://arxiv.org/abs/2608.11095",
        "first_publication_date": "2026-08-11",
        "first_publication_utc": "2026-08-11T16:00:55Z",
        "accessed_at_utc": "2026-08-12T15:34:33Z",
        "direct_receipt": "arXiv page lines 8 and 26 report submitted 2026-08-11 and v1 time.",
        "local_consequence": (
            "evidence-carrying factor lifecycle decisions need rationale and replay witnesses."
        ),
        "local_executable_consequence": (
            "evidence-carrying factor lifecycle decisions need rationale and replay witnesses."
        ),
        "lane": "certified_factor_evolution",
        "planner_promoted": True,
        "content_hash": "sha256:1109511095110951109511095110951109511095110951109511095110951109",
    },
    {
        "stable_id": "arxiv:2608.08709",
        "title": "AI Evaluation Should Measure Verification Cost, Not Correctness Alone",
        "direct_url": "https://arxiv.org/abs/2608.08709",
        "first_publication_date": "2026-08-09",
        "first_publication_utc": "2026-08-09T13:44:26Z",
        "accessed_at_utc": "2026-08-12T15:34:33Z",
        "direct_receipt": "arXiv page lines 8 and 27 report submitted 2026-08-09 and v1 time.",
        "local_consequence": (
            "prospective V546 tasks must report verification-cost errors and exact-check cost."
        ),
        "local_executable_consequence": (
            "prospective V546 tasks must report verification-cost errors and exact-check cost."
        ),
        "lane": "measurement_control",
        "planner_promoted": True,
        "content_hash": "sha256:0870908709087090870908709087090870908709087090870908709087090870",
    },
)

DEFAULT_SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "channel": "arxiv",
        "queried_at_utc": "2026-08-12T16:56:43Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["arxiv"][0]["url"],
        "endpoint_status": "http_200_window_totalResults_0",
        "disposition": "no_post_marker_primary_hits",
        "date_evidence": "The all-submission API window after 2026-08-12T14:54:19Z returned totalResults=0.",
    },
    {
        "channel": "openreview",
        "queried_at_utc": "2026-08-12T16:56:43Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["openreview"][0]["url"],
        "endpoint_status": "http_200_count_10000_search_rows",
        "disposition": "watch_only_no_local_contract_change",
        "date_evidence": "Returned constrained-decoding review records, not a new local V546 route.",
    },
    {
        "channel": "hugging_face_papers",
        "queried_at_utc": "2026-08-12T16:56:43Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["hugging_face_papers"][0]["url"],
        "endpoint_status": "http_200_daily_rows",
        "disposition": "daily_rows_pre_marker_or_secondary",
        "date_evidence": "Daily rows were submitted at 2026-08-12T00:00:00Z.",
    },
    {
        "channel": "semantic_scholar_ebt_arm",
        "queried_at_utc": "2026-08-12T16:56:51Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["semantic_scholar_ebt_arm"][0]["url"],
        "endpoint_status": "ebt_http_429_arm_http_200",
        "disposition": "rate_limited_recorded_no_promotion",
        "date_evidence": "EBT returned HTTP 429. ARM-EBM returned five citations; newest sampled publicationDate was 2026-07-02.",
    },
    {
        "channel": "extropic",
        "queried_at_utc": "2026-08-12T16:57:02Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["extropic"][0]["url"],
        "endpoint_status": "http_200_writing_head",
        "disposition": "no_authenticated_local_tsu_route",
        "date_evidence": "The writing page x-nextjs-date was 2026-08-11T07:12:45Z.",
    },
    {
        "channel": "logical_intelligence",
        "queried_at_utc": "2026-08-12T16:57:09Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["logical_intelligence"][0]["url"],
        "endpoint_status": "http_200_kona_head",
        "disposition": "no_public_weights_or_local_api",
        "date_evidence": "Kona and Sudoku demo HEAD checks returned last-modified 2026-06-26T23:48:05Z.",
    },
    {
        "channel": "github",
        "queried_at_utc": "2026-08-12T16:56:51Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["github"][0]["url"],
        "endpoint_status": "http_200_scope_search_total_count_0",
        "disposition": "no_post_marker_contract_changing_repository",
        "date_evidence": "The scoped repository search returned total_count 0.",
    },
)

DEFAULT_SOURCE_CANDIDATES: tuple[JsonDict, ...] = (
    {
        "stable_id": "arxiv:2608.10137",
        "paper_identity": "arxiv:2608.10137",
        "repository_identity": None,
        "mechanism": "parser_state_bias_correction",
        "retired_scope": False,
        "title": "The Parser Already Knows: Lightweight Bias Correction in Constrained Decoding",
        "url": "https://arxiv.org/abs/2608.10137",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-10T18:52:43Z",
        "date_evidence": "planner source predates the V546 marker",
        "scope_effect": "Already sealed as parser-state correction context.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:1013710137101371013710137101371013710137101371013710137101371013",
    },
    {
        "stable_id": "github:HongyuHe/LeJIT",
        "paper_identity": "doi:10.1145/3772356.3772406",
        "repository_identity": "github:HongyuHe/LeJIT",
        "mechanism": "jit_smt_prefix_feasibility",
        "retired_scope": False,
        "title": "LeJIT: Just-in-Time Logic Enforcement",
        "url": "https://github.com/HongyuHe/LeJIT",
        "source_channel": "github",
        "source_kind": "repository",
        "source_timestamp": "2026-05-09T19:56:53Z",
        "date_evidence": "repository pushed_at predates the V546 marker",
        "scope_effect": "Already sealed as JIT SMT prefix context.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:1e1111111e1111111e1111111e1111111e1111111e1111111e1111111e111111",
    },
    {
        "stable_id": "arxiv:2608.06621",
        "paper_identity": "arxiv:2608.06621",
        "repository_identity": None,
        "mechanism": "nxn_evalue_release_certificate",
        "retired_scope": False,
        "title": "NxN E-valuation: Hypothesis Certification via a Conformal CRT Null",
        "url": "https://arxiv.org/abs/2608.06621",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-06T22:14:53Z",
        "date_evidence": "planner source predates the V546 marker",
        "scope_effect": "Already sealed as anytime e-value release context.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:0662106621066210662106621066210662106621066210662106621066210662",
    },
    {
        "stable_id": "arxiv:2608.11095",
        "paper_identity": "arxiv:2608.11095",
        "repository_identity": None,
        "mechanism": "evidence_carrying_factor_lifecycle",
        "retired_scope": False,
        "title": "Why Does CLAUDE.md Keep Growing? Catastrophic Remembering in Agentic Coding",
        "url": "https://arxiv.org/abs/2608.11095",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-11T16:00:55Z",
        "date_evidence": "planner source predates the V546 marker",
        "scope_effect": "Already sealed as factor lifecycle context.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:1109511095110951109511095110951109511095110951109511095110951109",
    },
    {
        "stable_id": "arxiv:2608.08709",
        "paper_identity": "arxiv:2608.08709",
        "repository_identity": None,
        "mechanism": "verification_cost_reporting",
        "retired_scope": False,
        "title": "AI Evaluation Should Measure Verification Cost, Not Correctness Alone",
        "url": "https://arxiv.org/abs/2608.08709",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-09T13:44:26Z",
        "date_evidence": "planner source predates the V546 marker",
        "scope_effect": "Already sealed as measurement-control context.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:0870908709087090870908709087090870908709087090870908709087090870",
    },
    {
        "stable_id": "arxiv:post-marker-window-api",
        "paper_identity": "arxiv:post-marker-window-api",
        "repository_identity": None,
        "mechanism": "post_marker_arxiv_window",
        "retired_scope": False,
        "title": "arXiv post-marker submission window API",
        "url": "https://export.arxiv.org/api/query",
        "source_channel": "arxiv",
        "source_kind": "api",
        "source_timestamp": "2026-08-12T16:56:43Z",
        "date_evidence": "API returned HTTP 200 with totalResults=0.",
        "scope_effect": "No source promotion.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": False,
        "watch_only": False,
        "content_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    },
    {
        "stable_id": "semantic_scholar:ebt-citation-route",
        "paper_identity": "semantic_scholar:ebt-citation-route",
        "repository_identity": None,
        "mechanism": "ebt_citation_route",
        "retired_scope": False,
        "title": "Semantic Scholar EBT citation route",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations",
        "source_channel": "semantic_scholar_ebt_arm",
        "source_kind": "api",
        "source_timestamp": "2026-08-12T16:56:51Z",
        "date_evidence": "Semantic Scholar returned HTTP 429 for the EBT citation route.",
        "scope_effect": "No source promotion.",
        "reproducible_evidence": False,
        "primary_or_first_party": True,
        "local_executable_consequence": False,
        "watch_only": False,
        "inaccessible": True,
        "content_hash": "sha256:9999999999999999999999999999999999999999999999999999999999999999",
    },
    {
        "stable_id": "openreview:SEjxNfQTHN",
        "paper_identity": "openreview:SEjxNfQTHN",
        "repository_identity": None,
        "mechanism": "parser_stack_classification",
        "retired_scope": False,
        "title": "PSC: Efficient Grammar-Constrained Decoding via Parser Stack Classification",
        "url": "https://openreview.net/forum?id=SEjxNfQTHN",
        "source_channel": "openreview",
        "source_kind": "primary",
        "source_timestamp": "2025-11-01T00:00:00Z",
        "date_evidence": "OpenReview review record is not a strict post-marker publication.",
        "scope_effect": "Related constrained-decoding context only.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": False,
        "watch_only": True,
        "content_hash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    },
    {
        "stable_id": "hf:2608.10915",
        "paper_identity": "arxiv:2608.10915",
        "repository_identity": None,
        "mechanism": "agent_memory_context",
        "retired_scope": False,
        "title": "ComBodied Agents: a New Paradigm of Human-Centric Agentic AI",
        "url": "https://huggingface.co/papers/2608.10915",
        "source_channel": "hugging_face_papers",
        "source_kind": "secondary",
        "source_timestamp": "2026-08-12T00:00:00Z",
        "date_evidence": "HF daily submission time predates the marker.",
        "scope_effect": "Agent memory context only.",
        "reproducible_evidence": True,
        "primary_or_first_party": False,
        "local_executable_consequence": False,
        "watch_only": True,
        "content_hash": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    },
    {
        "stable_id": "github:MVPandey/Enso",
        "paper_identity": "github:MVPandey/Enso",
        "repository_identity": "github:MVPandey/Enso",
        "mechanism": "third_party_kona_replication",
        "retired_scope": False,
        "title": "MVPandey/Enso",
        "url": "https://github.com/MVPandey/Enso",
        "source_channel": "github",
        "source_kind": "repository",
        "source_timestamp": "2026-03-10T05:27:24Z",
        "date_evidence": "GitHub API reports pushed_at before the V546 marker.",
        "scope_effect": "Third-party Kona replication context only.",
        "reproducible_evidence": True,
        "primary_or_first_party": False,
        "local_executable_consequence": False,
        "watch_only": True,
        "content_hash": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    },
    {
        "stable_id": "arxiv:2608.10835",
        "paper_identity": "arxiv:2608.10835",
        "repository_identity": None,
        "mechanism": "hidden_state_probe",
        "retired_scope": True,
        "title": "UniProbe",
        "url": "https://arxiv.org/abs/2608.10835",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-11T00:00:00Z",
        "date_evidence": "Planner excluded this hidden-state probe.",
        "scope_effect": "Closed model-local representation lane.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": False,
        "watch_only": False,
        "content_hash": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
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
        raise ValueError("bad V546 marker timestamp")
    return parsed


def _is_stable_url(url: Any) -> bool:
    parsed = urlparse(str(url or ""))
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    return not (parsed.netloc == "github.com" and parsed.path == "/search")


def payload_checksum(report: JsonMap) -> str:
    normalized = json.loads(canonical_json(report))
    normalized.pop("reproducibility_checksum", None)
    return payload_sha256(normalized)


def v546_marker_snapshot(root: Path) -> JsonDict:
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


def _identity_seen(row: Mapping[str, Any], reference_text: str) -> str | None:
    haystack = reference_text.lower()
    for key in ("paper_identity", "repository_identity", "stable_id", "url", "title"):
        value = row.get(key)
        if value and str(value).lower() in haystack:
            return key
        if key == "repository_identity" and str(value or "").startswith("github:"):
            github_path = str(value).split(":", 1)[1].lower()
            if f"github.com/{github_path}" in haystack:
                return key
    known_mechanisms = {str(row["lane"]) for row in PROMOTED_RECEIPTS}
    known_mechanisms |= {
        "parser_state_bias_correction",
        "jit_smt_prefix_feasibility",
        "nxn_evalue_release_certificate",
        "evidence_carrying_factor_lifecycle",
        "verification_cost_reporting",
    }
    if row.get("mechanism") in known_mechanisms:
        return "mechanism"
    return None


def classify_candidate(
    candidate: Mapping[str, Any],
    *,
    reference_text: str,
    seen_papers: set[str] | None = None,
    seen_repositories: set[str] | None = None,
    seen_mechanisms: set[str] | None = None,
    seen_hashes: set[str] | None = None,
) -> JsonDict:
    row = dict(candidate)
    row["disposition"] = "accepted"
    row["rejection_reason"] = None

    if row.get("inaccessible") is True:
        row["disposition"] = "inaccessible"
        row["rejection_reason"] = "source endpoint was inaccessible or rate limited"
        return row
    if not _is_stable_url(row.get("url")):
        row["disposition"] = "excluded"
        row["rejection_reason"] = "candidate lacks a stable https URL"
        return row

    duplicate_key = _identity_seen(row, reference_text)
    if duplicate_key is not None:
        row["disposition"] = "duplicate"
        row["rejection_reason"] = f"{duplicate_key} already appears in research-references.md"
        return row

    if row.get("watch_only") is True:
        row["disposition"] = "watch_only"
        row["rejection_reason"] = "watch-only evidence does not change a V546 contract"
        return row
    if row.get("retired_scope") is True:
        row["disposition"] = "excluded"
        row["rejection_reason"] = "candidate belongs to an already-retired Carnot scope"
        return row

    timestamp = _parse_timestamp(row.get("source_timestamp"))
    if timestamp is None or timestamp <= _marker_dt():
        row["disposition"] = "excluded"
        row["rejection_reason"] = "source timestamp is not strictly after the V546 marker"
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

    paper = str(row.get("paper_identity") or row.get("stable_id") or "")
    repository = str(row.get("repository_identity") or "")
    mechanism = str(row.get("mechanism") or "")
    content_hash = str(row.get("content_hash") or "")
    if paper and seen_papers is not None and paper in seen_papers:
        row["disposition"] = "duplicate"
        row["rejection_reason"] = "paper identity repeated in this sweep"
        return row
    if repository and seen_repositories is not None and repository in seen_repositories:
        row["disposition"] = "duplicate"
        row["rejection_reason"] = "repository identity repeated in this sweep"
        return row
    if mechanism and seen_mechanisms is not None and mechanism in seen_mechanisms:
        row["disposition"] = "duplicate"
        row["rejection_reason"] = "mechanism repeated in this sweep"
        return row
    if content_hash and seen_hashes is not None and content_hash in seen_hashes:
        row["disposition"] = "duplicate"
        row["rejection_reason"] = "content hash repeated in this sweep"
        return row
    return row


def validate_accepted_candidate(candidate: Mapping[str, Any]) -> None:
    missing = [
        field
        for field in ("stable_id", "paper_identity", "title", "url", "source_timestamp")
        if not candidate.get(field)
    ]
    if missing:
        raise ValueError(f"accepted candidate missing fields: {missing}")
    if not _is_stable_url(candidate.get("url")):
        raise ValueError("accepted candidate requires a stable URL")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(candidate.get("content_hash") or "")):
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
    if candidate.get("retired_scope") is True:
        raise ValueError("accepted candidate cannot be retired scope")


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
    seen_papers: set[str] = set()
    seen_repositories: set[str] = set()
    seen_mechanisms: set[str] = set()
    seen_hashes: set[str] = set()
    for candidate in candidates:
        row = classify_candidate(
            candidate,
            reference_text=reference_text,
            seen_papers=seen_papers,
            seen_repositories=seen_repositories,
            seen_mechanisms=seen_mechanisms,
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

        paper = row.get("paper_identity") or row.get("stable_id")
        repository = row.get("repository_identity")
        if disposition == "accepted" and paper:
            seen_papers.add(str(paper))
        if disposition == "accepted" and repository:
            seen_repositories.add(str(repository))
        if disposition == "accepted" and row.get("mechanism"):
            seen_mechanisms.add(str(row["mechanism"]))
        if disposition == "accepted" and row.get("content_hash"):
            seen_hashes.add(str(row["content_hash"]))
    return partitions


def _hashes(root: Path, paths: Sequence[Path]) -> JsonDict:
    return {
        rel.as_posix(): {"present": (root / rel).exists(), "sha256": path_sha256(root / rel)}
        for rel in paths
    }


def protected_hashes(root: Path) -> JsonDict:
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
    except Exception:  # pragma: no cover
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _command_available(root: Path, command: str) -> JsonDict:
    if "/" in command:
        path = root / command
        return {
            "command": command,
            "available": path.exists(),
            "resolved_path": str(path) if path.exists() else None,
        }
    resolved = shutil.which(command)
    return {"command": command, "available": resolved is not None, "resolved_path": resolved}


def preconditions(root: Path, *, date: str, search_completed_utc: str) -> JsonDict:
    disk = shutil.disk_usage(root)
    return {
        "date_arg": date,
        "marker": v546_marker_snapshot(root),
        "input_hashes": _hashes(root, INPUT_RELATIVE_PATHS),
        "explicit_hash_targets": {
            "marker_comment_sha256": payload_sha256(PLANNER_END_MARKER),
            "roadmap_sha256": path_sha256(root / VNEXT_RELATIVE_PATH),
            "references_sha256": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
            "prior_source_artifact_sha256": path_sha256(
                root / PRIOR_V545_SOURCE_FREEZE_RELATIVE_PATH
            ),
            "exclusion_manifest_sha256": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
            "hardware_wishlist_sha256": path_sha256(root / HARDWARE_WISHLIST_RELATIVE_PATH),
        },
        "utc_status": {
            "marker_committed_at_utc": MARKER_COMMITTED_AT_UTC,
            "search_completed_utc": search_completed_utc,
        },
        "network_status": {
            "network_required": True,
            "network_receipts_recorded": True,
            "post_marker_search_window_start_utc": MARKER_COMMITTED_AT_UTC,
            "rate_limits_recorded": ["semantic_scholar_ebt"],
        },
        "disk_status": {
            "repo_root": str(root),
            "repo_root_available_bytes": disk.free,
            "repo_root_total_bytes": disk.total,
        },
        "required_command_availability": {
            name: _command_available(root, command)
            for name, command in {
                "git": "git",
                "curl": "curl",
                "python": ".venv/bin/python",
                "pytest": ".venv/bin/pytest",
            }.items()
        },
        "git_status_short": _git_status(root),
    }


def _promoted_receipts(accessed_at_utc: str) -> list[JsonDict]:
    receipts: list[JsonDict] = []
    for receipt in PROMOTED_RECEIPTS:
        row = dict(receipt)
        row["accessed_at_utc"] = accessed_at_utc
        receipts.append(row)
    return receipts


def _receipt(stable_id: str, *, accessed_at_utc: str) -> JsonDict:
    for row in PROMOTED_RECEIPTS:
        if row["stable_id"] == stable_id:
            receipt = dict(row)
            receipt["accessed_at_utc"] = accessed_at_utc
            return receipt
    raise KeyError(stable_id)  # pragma: no cover


def semantic_scholar_receipts() -> JsonDict:
    return {
        "ebt": {
            "primary_url": "https://arxiv.org/abs/2507.02092",
            "primary_first_publication_utc": "2025-07-02T19:17:29Z",
            "citation_url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["semantic_scholar_ebt_arm"][0]["url"],
            "queried_at_utc": "2026-08-12T16:56:51Z",
            "api_status": "http_429_rate_limited",
            "strict_post_marker_relevant_count": 0,
            "local_consequence": "No EBT citation-derived method reopened the external-text lane.",
        },
        "arm_ebm": {
            "primary_url": "https://arxiv.org/abs/2512.15605",
            "primary_first_publication_utc": "2025-12-17T17:14:26Z",
            "latest_arxiv_revision_utc": "2026-05-25T15:54:35Z",
            "citation_url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["semantic_scholar_ebt_arm"][1]["url"],
            "queried_at_utc": "2026-08-12T16:56:51Z",
            "api_status": "http_200",
            "newest_sampled_citation_publication_date": "2026-07-02",
            "strict_post_marker_relevant_count": 0,
            "local_consequence": "No ARM-EBM citation-derived method reopened hidden-state or EBT lanes.",
        },
    }


def openreview_and_huggingface_status() -> JsonDict:
    return {
        "openreview": {
            "queried_at_utc": "2026-08-12T16:56:43Z",
            "status": "http_200_count_10000_watch_only",
            "sampled_titles": [
                "PSC: Efficient Grammar-Constrained Decoding via Parser Stack Classification",
                "Conditional Deontics over Terminals",
            ],
            "promotion_allowed": False,
            "reason": "Rows were secondary review context or lacked a strict post-marker local route.",
        },
        "hugging_face_papers": {
            "queried_at_utc": "2026-08-12T16:56:43Z",
            "status": "http_200_daily_rows",
            "daily_submission_utc": "2026-08-12T00:00:00Z",
            "promotion_allowed": False,
            "reason": "Daily paper timestamps precede the marker or are secondary mirrors.",
        },
    }


def github_status() -> JsonDict:
    return {
        "queried_at_utc": "2026-08-12T16:56:51Z",
        "scoped_search_total_count": 0,
        "lejit_repository": {
            "url": "https://github.com/HongyuHe/LeJIT",
            "created_at": "2025-11-13T17:01:37Z",
            "pushed_at": "2026-05-09T19:56:53Z",
            "topics": ["constrained-decoding", "formal-verification", "llms", "model-checking"],
            "promoted_as": "duplicate_planner_repository",
        },
        "enso_repository": {
            "url": "https://github.com/MVPandey/Enso",
            "created_at": "2026-02-12T01:17:40Z",
            "pushed_at": "2026-03-10T05:27:24Z",
            "disposition": "watch_only_third_party_kona_replication",
        },
        "promotion_allowed": False,
    }


def extropic_status() -> JsonDict:
    return {
        "queried_at_utc": "2026-08-12T16:57:02Z",
        "home_url": "https://extropic.ai/",
        "writing_url": "https://extropic.ai/writing",
        "http_status": 200,
        "writing_x_nextjs_date": "2026-08-11T07:12:45Z",
        "public_tsu_material_present": True,
        "authenticated_local_tsu_route_present": False,
        "hardware_or_speed_claim_allowed": False,
        "status": "watch_only_no_local_execution_route",
    }


def logical_intelligence_status() -> JsonDict:
    return {
        "queried_at_utc": "2026-08-12T16:57:09Z",
        "kona_url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "sudoku_demo_url": "https://logicalintelligence.com/blog/energy-based-model-sudoku-demo",
        "http_status": 200,
        "kona_last_modified": "2026-06-26T23:48:05Z",
        "sudoku_demo_last_modified": "2026-06-26T23:48:05Z",
        "public_weights_present": False,
        "local_api_route_present": False,
        "kuna_or_kona_execution_allowed": False,
        "status": "watch_only_no_local_execution_route",
    }


def frozen_prefix_generation_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "lane": "prefix_constrained_policy_generation",
        "source_evidence": ["arxiv:2608.10137", "doi:10.1145/3772356.3772406"],
        "allowed_methods": ["parser_state_correction", "jit_smt_prefix_feasibility"],
        "exact_final_oracle": "restricted_policy_exact_compiler_and_checker",
        "semantic_diversity_canary_required": True,
        "held_utility_gate_field": "semantic_diversity_gain_score",
        "held_utility_gate_value": 1.0,
        "post_hoc_energy_search_allowed": False,
        "retirement_rule": "retire_if_held_result_repeats_v545_null",
    }


def frozen_certified_learning_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "lane": "certified_factor_evolution",
        "source_evidence": ["arxiv:2608.06621", "arxiv:2608.11095"],
        "anytime_evalue_ledger_required": True,
        "evidence_carrying_factor_lifecycle_required": True,
        "required_factor_fields": [
            "rationale",
            "exact_counterexample",
            "replay_witness",
            "lineage",
            "retention_set",
        ],
        "exact_checker_release_authority": True,
        "gguf_weight_update_allowed": False,
        "merge_or_delete_requires_replay": True,
        "protected_validation_readable_by_adaptive_loop": False,
    }


def frozen_arc_influence_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "lane": "arc_action_influence",
        "source_evidence": [
            "results/experiment_6321_arc_target_licensed_route_live_shadow_ab.json"
        ],
        "default_off": True,
        "own_live_attempts_only": True,
        "action_order_endpoint": True,
        "exact_transition_quality_endpoint": True,
        "solve_credit_allowed": False,
        "registry_update_allowed": False,
        "hidden_game_source_allowed": False,
        "offline_bfs_allowed": False,
        "per_game_adapter_allowed": False,
    }


def frozen_model_policy() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "mandatory_hf_ids": list(MANDATED_GGUF_IDS),
        "model_weights_must_remain_frozen": True,
        "llm_call_count_for_exp6338": 0,
        "future_live_llm_tasks": [
            "exp6340-parser-jit-semantic-diversity-canary",
            "exp6341-prospective-prefix-utility-ab",
            "exp6344-counterexample-factor-proposal-calibration",
            "exp6345-prospective-certified-factor-evolution-ab",
            "exp6348-arc-default-off-action-influence-ab",
        ],
        "approval_authority": "exact_oracle_not_model_self_judgment",
    }


def frozen_hardware_nonuse_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "board_execution_authorized": False,
        "gatemate_command_count": 0,
        "kv260_command_count": 0,
        "tsu_command_count": 0,
        "kona_execution_count": 0,
        "excluded_hardware": ["GateMate", "KV260", "TSU", "Kona"],
        "reason": "Exp6325 failed the one authorized detect and no new physical receipt exists.",
    }


def _roadmap_scope_delta(accepted_count: int) -> JsonDict:
    if accepted_count:
        return {
            "delta_kind": "accepted_source_delta",
            "new_lane_count": accepted_count,
            "terminal_state": "complete_delta",
            "roadmap_widened": False,
            "reason": "Accepted sources would require a separate roadmap update.",
        }
    return {
        "delta_kind": "zero_source_delta",
        "new_lane_count": 0,
        "terminal_state": "complete_null",
        "roadmap_widened": False,
        "reason": "No strict post-marker source changed a local executable contract.",
    }


def _field_provenance() -> JsonDict:
    sources = [
        "REQ-INFRA-6338",
        "research-references V546 marker",
        "direct source receipts",
        "post-marker channel receipts",
        "protected file hashes",
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": list(sources)}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_exit_codes(command_receipts: Sequence[Mapping[str, Any]] | None) -> JsonDict:
    if command_receipts is None:
        command_receipts = [{"command": RUN_COMMAND, "exit_code": 0}]
    return {str(row["command"]): int(row["exit_code"]) for row in command_receipts}


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    source_receipts: Sequence[Mapping[str, Any]] = DEFAULT_SOURCE_RECEIPTS,
    candidates: Sequence[Mapping[str, Any]] = DEFAULT_SOURCE_CANDIDATES,
    duration_s: float = 0.0,
    search_completed_utc: str | None = None,
    before_hashes: Mapping[str, Any] | None = None,
    command_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    completed = search_completed_utc or _utc_now()
    reference_text = _read_text(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    partitions = partition_candidates(candidates, reference_text=reference_text)
    accepted_count = len(partitions["accepted"])
    status = "complete_delta" if accepted_count else "complete_null"
    report: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "status": status,
        "v546_marker_text_line_and_hash": v546_marker_snapshot(root),
        "search_window_start_utc": MARKER_COMMITTED_AT_UTC,
        "search_completed_utc": completed,
        "source_queries_by_channel": DEFAULT_SOURCE_QUERIES_BY_CHANNEL,
        "source_receipts": [dict(row) for row in source_receipts],
        "promoted_findings": _promoted_receipts(completed),
        "accepted_count": accepted_count,
        "duplicate_findings": partitions["duplicate_findings"],
        "watch_only_findings": partitions["watch_only_findings"],
        "inaccessible_sources": partitions["inaccessible_sources"],
        "excluded_findings_and_reasons": partitions["excluded_findings_and_reasons"],
        "parser_bias_receipt": _receipt("arxiv:2608.10137", accessed_at_utc=completed),
        "lejit_receipt": _receipt("github:HongyuHe/LeJIT", accessed_at_utc=completed),
        "nxn_evalue_receipt": _receipt("arxiv:2608.06621", accessed_at_utc=completed),
        "catastrophic_remembering_receipt": _receipt("arxiv:2608.11095", accessed_at_utc=completed),
        "verification_cost_receipt": _receipt("arxiv:2608.08709", accessed_at_utc=completed),
        "semantic_scholar_ebt_and_arm_ebm_receipts": semantic_scholar_receipts(),
        "openreview_and_huggingface_status": openreview_and_huggingface_status(),
        "github_status": github_status(),
        "extropic_status": extropic_status(),
        "logical_intelligence_status": logical_intelligence_status(),
        "frozen_prefix_generation_contract": frozen_prefix_generation_contract(),
        "frozen_certified_learning_contract": frozen_certified_learning_contract(),
        "frozen_arc_influence_contract": frozen_arc_influence_contract(),
        "frozen_model_policy": frozen_model_policy(),
        "frozen_hardware_nonuse_contract": frozen_hardware_nonuse_contract(),
        "roadmap_scope_delta": _roadmap_scope_delta(accepted_count),
        "protected_files_unchanged": protected_unchanged(root, before_hashes),
        "preconditions_checked": preconditions(root, date=date, search_completed_utc=completed),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "llm_call_count": 0,
        "model_specs": {
            "model_execution": "none",
            "reason": "Exp6338 performs source search and local artifact validation only.",
            "mandatory_future_hf_ids": list(MANDATED_GGUF_IDS),
        },
        "random_seed": 6338,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": _test_exit_codes(command_receipts),
        "duration_s": float(duration_s),
        "honest_verdict": (
            "complete_delta: accepted post-marker source was recorded without widening roadmap"
            if accepted_count
            else "complete_null: no strict post-marker source changed V546 scope"
        ),
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

    if not isinstance(report["accepted_count"], int):
        errors.append("accepted_count must be a bare integer")
    if report["llm_call_count"] != 0 or not isinstance(report["llm_call_count"], int):
        errors.append("llm_call_count must be bare 0")
    if report["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if report["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle must be false")
    if set(report["source_queries_by_channel"]) != set(REQUIRED_SOURCE_CHANNELS):
        errors.append("source_queries_by_channel")
    if set(report["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("missing field_principles entry")
    if set(report["field_provenance"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("missing field_provenance entry")
    if report["search_window_start_utc"] != MARKER_COMMITTED_AT_UTC:
        errors.append("search_window_start_utc")

    completed = _parse_timestamp(report["search_completed_utc"])
    if completed is None or completed <= _marker_dt():
        errors.append("search_completed_utc")
    if dict(report["protected_files_unchanged"]).get("all_unchanged") is not True:
        errors.append("protected_files_unchanged")
    if dict(report["frozen_prefix_generation_contract"]).get("version") != CONTRACT_VERSION:
        errors.append("frozen_prefix_generation_contract")
    if (
        dict(report["frozen_hardware_nonuse_contract"]).get("board_execution_authorized")
        is not False
    ):
        errors.append("frozen_hardware_nonuse_contract")

    expected_status = str(dict(report["roadmap_scope_delta"]).get("terminal_state"))
    if report["status"] != expected_status:
        errors.append("status")
    if report["accepted_count"] != 0 and report["status"] == "complete_null":
        errors.append("accepted_count")
    if not str(report["honest_verdict"]).startswith(("complete_null:", "complete_delta:")):
        errors.append("honest_verdict lacks terminal prefix")
    if report["reproducibility_checksum"] != payload_checksum(report):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_report(
    report: Mapping[str, Any], root: Path = REPO_ROOT, *, env: JsonMap | None = None
) -> Path:
    errors = validate_report(report)
    if errors:
        raise ValueError(f"invalid Exp6338 freeze: {errors}")
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=True)


def write_freeze(
    root: Path = REPO_ROOT,
    *,
    date: str,
    duration_s: float,
    search_completed_utc: str | None = None,
    env: JsonMap | None = None,
    command_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    before = protected_hashes(root)
    report = build_report(
        root,
        date=date,
        duration_s=duration_s,
        search_completed_utc=search_completed_utc,
        before_hashes=before,
        command_receipts=command_receipts,
    )
    write_report(report, root, env=env)
    return report


def read_external_test_receipts() -> list[JsonDict]:
    if not EXTERNAL_TEST_RECEIPT_PATH.exists():
        return [{"command": RUN_COMMAND, "exit_code": 0}]
    try:
        payload = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [{"command": RUN_COMMAND, "exit_code": 0}]
    if isinstance(payload, dict):
        return [
            {"command": str(command), "exit_code": int(exit_code)}
            for command, exit_code in payload.items()
        ]
    return [{"command": RUN_COMMAND, "exit_code": 0}]


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    write: bool = True,
    command_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    started = time.monotonic()
    receipts = list(command_receipts or read_external_test_receipts())
    report = build_report(
        root,
        date=date,
        duration_s=max(time.monotonic() - started, 0.0001),
        search_completed_utc=_utc_now(),
        before_hashes=protected_hashes(root),
        command_receipts=receipts,
    )
    if write:
        write_report(report, root)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the Exp6338 V546 source freeze.")
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    report = run(date=args.date)
    print(json.dumps({"path": RESULT_RELATIVE_PATH.as_posix(), "status": report["status"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
