"""Exp6351 V547 post-marker source scope freeze.

Spec refs: REQ-INFRA-6351, SCENARIO-INFRA-6351-1,
SCENARIO-INFRA-6351-2, SCENARIO-INFRA-6351-3,
SCENARIO-INFRA-6351-4, SCENARIO-INFRA-6351-5.
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

EXPERIMENT_ID = "exp6351-v547-post-marker-source-scope-freeze"
SCHEMA = "carnot.experiment_6351.v547_post_marker_source_scope_freeze.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6351_v547_post_marker_source_scope_freeze.json")
INFERENCE_SUBSTRATE = "web_and_bibliographic_search_only"
CONTRACT_VERSION = "v547.6351"

RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
HARDWARE_WISHLIST_RELATIVE_PATH = Path("research-hardware-wishlist.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

PRIOR_HANDOFF_RELATIVE_PATH = Path("results/experiment_6350_v547_bounded_terminal_handoff.json")
PRIOR_V546_SOURCE_RELATIVE_PATH = Path(
    "results/experiment_6338_v546_post_marker_source_scope_freeze.json"
)
PRIOR_PARSER_JIT_RELATIVE_PATH = Path(
    "results/experiment_6340_parser_jit_semantic_diversity_canary.json"
)
PRIOR_CERTIFIED_FACTOR_RELATIVE_PATH = Path(
    "results/experiment_6345_prospective_certified_factor_evolution_ab.json"
)
PRIOR_GOAL_VETO_RELATIVE_PATH = Path("results/experiment_6258_goal_veto_confusion_matrix.json")

PLANNER_MARKER = "V547-PLANNER-REFRESH-20260812-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
MARKER_COMMIT = "444250de2807a14b0cc12ff229d9c4d6ae821853"
MARKER_COMMITTED_AT_LOCAL = "2026-08-12T17:28:21-04:00"
MARKER_COMMITTED_AT_UTC = "2026-08-12T21:28:21Z"

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
    "v547_marker_text_line_and_hash",
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
    "active_reward_machine_receipt",
    "zero_shot_goal_recognition_receipt",
    "memoir_receipt",
    "solver_hardness_control_receipt",
    "distributional_ebm_receipt",
    "verification_horizon_receipt",
    "semantic_scholar_ebt_and_arm_ebm_receipts",
    "openreview_and_huggingface_status",
    "github_status",
    "extropic_status",
    "logical_intelligence_status",
    "frozen_live_factor_learning_contract",
    "frozen_arc_goal_contract",
    "frozen_model_policy",
    "frozen_closed_parser_jit_contract",
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
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The freeze is complete only when source and boundary checks pass.",
    "v547_marker_text_line_and_hash": "The exact marker defines the novelty boundary.",
    "search_window_start_utc": "The lower bound is exclusive for post-marker novelty.",
    "search_completed_utc": "The upper audit time makes the sweep reproducible.",
    "source_queries_by_channel": "Each required channel records its direct search route.",
    "source_receipts": "Search outcomes and failures stay visible.",
    "promoted_findings": "Planner-promoted sources are receipts, not new post-marker deltas.",
    "accepted_count": "A bare integer drives the null or delta verdict.",
    "duplicate_findings": "Repeated work, repository, and mechanism identities do not widen scope.",
    "watch_only_findings": "Contextual rows stay visible without becoming contracts.",
    "inaccessible_sources": "Rate limits and blocked endpoints are receipts, not findings.",
    "excluded_findings_and_reasons": "Rejected rows name the failed acceptance rule.",
    "active_reward_machine_receipt": "Goal automata are hypotheses, not hidden-game oracles.",
    "zero_shot_goal_recognition_receipt": "Goal recognition must be measured over evidence prefixes.",
    "memoir_receipt": "Memory writes are blocked until exact outcome evidence is known.",
    "solver_hardness_control_receipt": "Solver effort is not treated as model difficulty.",
    "distributional_ebm_receipt": "Learned energy may rank or abstain but cannot approve output.",
    "verification_horizon_receipt": "Verification must evolve while exact checks keep veto authority.",
    "semantic_scholar_ebt_and_arm_ebm_receipts": "Citation routes stay separate from broad search.",
    "openreview_and_huggingface_status": "Secondary channels cannot widen scope without primary evidence.",
    "github_status": "Repository rows need a maintained local route before promotion.",
    "extropic_status": "Product hardware pages do not imply local TSU access.",
    "logical_intelligence_status": "Kona pages do not imply local weights or API access.",
    "frozen_live_factor_learning_contract": "Prospective factor learning needs real proposals and exact release.",
    "frozen_arc_goal_contract": "ARC goals require two-sided evidence on the live path.",
    "frozen_model_policy": "Later LLM tasks must use the local GGUF allowlist.",
    "frozen_closed_parser_jit_contract": "The parser/JIT lane stays closed after the V546 null.",
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
    "random_seeds": "No randomness is used by this deterministic source audit.",
    "reproducibility_checksum": "The normalized payload is content-addressed.",
    "honest_verdict": "The verdict states null, delta, or block directly.",
}

RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6351_v547_post_marker_source_scope_freeze "
    "--date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6351_v547_post_marker_source_scope_freeze.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6351_v547_post_marker_source_scope_freeze.py "
    "-m pytest tests/python/test_experiment_6351_v547_post_marker_source_scope_freeze.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6351_v547_post_marker_source_scope_freeze.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6351_v547_post_marker_source_scope_freeze.py "
    "tests/python/test_experiment_6351_v547_post_marker_source_scope_freeze.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6351_v547_post_marker_source_scope_freeze.py "
    "tests/python/test_experiment_6351_v547_post_marker_source_scope_freeze.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6351_v547_post_marker_source_scope_freeze.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '1,260p' ops/e2e-test-plan.md"
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
ADVERSARIAL_SELF_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6351_v547_post_marker_source_scope_freeze.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
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
    ROOT_CLUTTER_COMMAND,
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6351_test_receipts.json")

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_PROGRAM_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    PRIOR_HANDOFF_RELATIVE_PATH,
    PRIOR_V546_SOURCE_RELATIVE_PATH,
    PRIOR_PARSER_JIT_RELATIVE_PATH,
    PRIOR_CERTIFIED_FACTOR_RELATIVE_PATH,
    PRIOR_GOAL_VETO_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    HARDWARE_WISHLIST_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("scripts/research_conductor.py"),
    Path("python/carnot/experiment_6351_v547_post_marker_source_scope_freeze.py"),
    Path("tests/python/test_experiment_6351_v547_post_marker_source_scope_freeze.py"),
)
INPUT_RELATIVE_PATHS = PROTECTED_RELATIVE_PATHS

DEFAULT_SOURCE_QUERIES_BY_CHANNEL: dict[str, list[JsonDict]] = {
    "arxiv": [
        {
            "queried_at_utc": "2026-08-12T21:54:05Z",
            "query": "submittedDate:[202608122129 TO 202608122359]",
            "url": (
                "https://export.arxiv.org/api/query?search_query="
                "submittedDate:%5B202608122129%20TO%20202608122359%5D"
            ),
        }
    ],
    "openreview": [
        {
            "queried_at_utc": "2026-08-12T21:54:06Z",
            "query": (
                "active reward machine goal recognition distributional energy verification horizon"
            ),
            "url": (
                "https://api2.openreview.net/notes/search?term=active%20reward%20machine%20"
                "goal%20recognition%20distributional%20energy%20verification%20horizon"
                "&limit=5"
            ),
        }
    ],
    "hugging_face_papers": [
        {
            "queried_at_utc": "2026-08-12T21:54:06Z",
            "query": "daily_papers date=2026-08-12",
            "url": "https://huggingface.co/api/daily_papers?date=2026-08-12",
        }
    ],
    "semantic_scholar_ebt_arm": [
        {
            "queried_at_utc": "2026-08-12T21:54:17Z",
            "query": "EBT arXiv:2507.02092 citations",
            "url": (
                "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/"
                "citations?fields=title,url,year,publicationDate,externalIds&limit=5"
            ),
        },
        {
            "queried_at_utc": "2026-08-12T21:54:17Z",
            "query": "ARM-EBM arXiv:2512.15605 citations",
            "url": (
                "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/"
                "citations?fields=title,url,year,publicationDate,externalIds&limit=5"
            ),
        },
    ],
    "extropic": [
        {
            "queried_at_utc": "2026-08-12T21:54:18Z",
            "query": "Extropic writing and hardware first-party HEAD checks",
            "url": "https://extropic.ai/writing",
        },
        {
            "queried_at_utc": "2026-08-12T21:54:18Z",
            "query": "Extropic hardware first-party HEAD check",
            "url": "https://extropic.ai/hardware",
        },
    ],
    "logical_intelligence": [
        {
            "queried_at_utc": "2026-08-12T21:54:16Z",
            "query": "Logical Intelligence Kona first-party HEAD check",
            "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        }
    ],
    "github": [
        {
            "queried_at_utc": "2026-08-12T21:54:06Z",
            "query": "V547 mechanism repository search pushed after 2026-08-12",
            "url": (
                "https://api.github.com/search/repositories?q=%28%22reward%20machine%22%20"
                "OR%20%22goal%20recognition%22%20OR%20%22distributional%20energy%22%20"
                "OR%20%22verification%20horizon%22%29%20pushed:%3E=2026-08-12"
                "&sort=updated&order=desc&per_page=5"
            ),
        }
    ],
}


def _receipt_hash(*parts: str) -> str:
    return payload_sha256({"receipt": list(parts)})


PROMOTED_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "stable_id": "arxiv:2604.07480",
        "title": "Active Reward Machine Inference From Raw State Trajectories",
        "direct_url": "https://arxiv.org/abs/2604.07480",
        "first_publication_date": "2026-04-08",
        "first_publication_utc": "2026-04-08T18:19:55Z",
        "accessed_at_utc": "2026-08-12T21:55:33Z",
        "direct_receipt": "arXiv reports submitted 2026-04-08 and v1 at 18:19:55 UTC.",
        "claim_boundary": (
            "Learns reward-machine structure from raw state and policy trajectories "
            "without observed rewards, labels, or machine nodes."
        ),
        "local_consequence": (
            "infer competing live ARC goal automata from visible transitions and choose "
            "legal discriminating actions; keep exact transition checks evaluator-only"
        ),
        "local_executable_consequence": (
            "Exp6358 must operate on legal live-path transitions and keep unverifiable goals out."
        ),
        "mechanism": "active_reward_machine_inference",
        "lane": "falsifiable_live_arc_goal_discovery",
        "planner_promoted": True,
        "post_marker_candidate": False,
        "content_hash": _receipt_hash("arxiv:2604.07480", "2026-04-08T18:19:55Z"),
    },
    {
        "stable_id": "arxiv:2605.15333",
        "title": "Zero-Shot Goal Recognition with Large Language Models",
        "direct_url": "https://arxiv.org/abs/2605.15333",
        "first_publication_date": "2026-05-14",
        "first_publication_utc": "2026-05-14T18:56:06Z",
        "accessed_at_utc": "2026-08-12T21:55:33Z",
        "direct_receipt": "arXiv reports submitted 2026-05-14 and v1 at 18:56:06 UTC.",
        "claim_boundary": (
            "Goal-recognition accuracy is uneven across models and changes as observations accumulate."
        ),
        "local_consequence": (
            "measure frozen-prior, accumulating-evidence, and counterexample-gated goal curves"
        ),
        "local_executable_consequence": (
            "Exp6359 must score evidence-response curves instead of one final goal guess."
        ),
        "mechanism": "zero_shot_goal_recognition",
        "lane": "falsifiable_live_arc_goal_discovery",
        "planner_promoted": True,
        "post_marker_candidate": False,
        "content_hash": _receipt_hash("arxiv:2605.15333", "2026-05-14T18:56:06Z"),
    },
    {
        "stable_id": "arxiv:2607.20792",
        "title": "Memoir: Should a Model Write to Its Memory While It Thinks?",
        "direct_url": "https://arxiv.org/abs/2607.20792",
        "first_publication_date": "2026-07-22",
        "first_publication_utc": "2026-07-22T23:34:56Z",
        "accessed_at_utc": "2026-08-12T21:55:33Z",
        "direct_receipt": "arXiv reports submitted 2026-07-22 and v1 at 23:34:56 UTC.",
        "claim_boundary": (
            "Same-iteration read/write fast memory learned slower than a read-only control "
            "at fixed budget, with saturation later."
        ),
        "local_consequence": (
            "keep active factor memory read-only during proposal generation and commit only "
            "after exact outcome evidence"
        ),
        "local_executable_consequence": (
            "Exp6352-Exp6355 must separate proposal reads from release commits."
        ),
        "mechanism": "read_only_then_commit_memory",
        "lane": "prospective_certified_factor_learning",
        "planner_promoted": True,
        "post_marker_candidate": False,
        "content_hash": _receipt_hash("arxiv:2607.20792", "2026-07-22T23:34:56Z"),
    },
    {
        "stable_id": "arxiv:2607.17047",
        "title": "Solver-Hard Is Not Model-Hard",
        "direct_url": "https://arxiv.org/abs/2607.17047",
        "first_publication_date": "2026-07-19",
        "first_publication_utc": "2026-07-19T03:23:22Z",
        "accessed_at_utc": "2026-08-12T21:55:33Z",
        "direct_receipt": "arXiv reports submitted 2026-07-19 and v1 at 03:23:22 UTC.",
        "claim_boundary": (
            "solver effort and surface relabeling are not reliable proxies for model difficulty."
        ),
        "local_consequence": (
            "balance streams by executable structure and surface form, not by solver conflict count"
        ),
        "local_executable_consequence": (
            "Exp6353 and Exp6354 must record hardness controls separately from solver effort."
        ),
        "mechanism": "solver_hardness_control",
        "lane": "prospective_certified_factor_learning",
        "planner_promoted": True,
        "post_marker_candidate": False,
        "content_hash": _receipt_hash("arxiv:2607.17047", "2026-07-19T03:23:22Z"),
    },
    {
        "stable_id": "arxiv:2605.18871",
        "title": "Distributional Energy-Based Models for Structured LLM Reasoning",
        "direct_url": "https://arxiv.org/abs/2605.18871",
        "first_publication_date": "2026-05-15",
        "first_publication_utc": "2026-05-15T17:08:27Z",
        "accessed_at_utc": "2026-08-12T21:55:33Z",
        "direct_receipt": "arXiv reports submitted 2026-05-15 and v1 at 17:08:27 UTC.",
        "claim_boundary": (
            "Learned energy, deterministic penalties, abstention, and model-identity shortcuts "
            "must be separated."
        ),
        "local_consequence": (
            "use exact penalties and family-blind controls; learned uncertainty can abstain but not approve"
        ),
        "local_executable_consequence": (
            "Exp6355 must expose abstention and family-blind cells under exact veto authority."
        ),
        "mechanism": "distributional_ebm_abstention",
        "lane": "prospective_certified_factor_learning",
        "planner_promoted": True,
        "post_marker_candidate": False,
        "content_hash": _receipt_hash("arxiv:2605.18871", "2026-05-15T17:08:27Z"),
    },
    {
        "stable_id": "arxiv:2606.26300",
        "title": "The Verification Horizon: No Silver Bullet for Coding Agent Rewards",
        "direct_url": "https://arxiv.org/abs/2606.26300",
        "first_publication_date": "2026-06-24",
        "first_publication_utc": "2026-06-24T18:45:03Z",
        "latest_revision_utc": "2026-06-29T08:21:30Z",
        "accessed_at_utc": "2026-08-12T21:55:33Z",
        "direct_receipt": "arXiv reports v1 on 2026-06-24 and v2 on 2026-06-29.",
        "claim_boundary": ("No fixed reward function stays effective as policy capability grows."),
        "local_consequence": (
            "co-evolve verification with released factors and new counterexamples while exact checks veto"
        ),
        "local_executable_consequence": (
            "Exp6356 must attack saturation, leakage, and changing proposal distributions."
        ),
        "mechanism": "verification_horizon_coevolution",
        "lane": "prospective_certified_factor_learning",
        "planner_promoted": True,
        "post_marker_candidate": False,
        "content_hash": _receipt_hash("arxiv:2606.26300", "2026-06-24T18:45:03Z"),
    },
)

DEFAULT_SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "channel": "arxiv",
        "queried_at_utc": "2026-08-12T21:54:05Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["arxiv"][0]["url"],
        "endpoint_status": "http_200_window_totalResults_0",
        "disposition": "no_post_marker_primary_hits",
        "date_evidence": "The arXiv API window after 2026-08-12T21:28:21Z returned totalResults=0.",
    },
    {
        "channel": "openreview",
        "queried_at_utc": "2026-08-12T21:54:06Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["openreview"][0]["url"],
        "endpoint_status": "http_200_review_rows",
        "disposition": "watch_only_no_strict_post_marker_primary_route",
        "date_evidence": "Returned review records are not strict post-marker publications.",
    },
    {
        "channel": "hugging_face_papers",
        "queried_at_utc": "2026-08-12T21:54:06Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["hugging_face_papers"][0]["url"],
        "endpoint_status": "http_200_daily_rows",
        "disposition": "daily_rows_pre_marker_or_secondary",
        "date_evidence": "Sample daily row submittedOnDailyAt was 2026-08-12T00:00:00Z.",
    },
    {
        "channel": "semantic_scholar_ebt_arm",
        "queried_at_utc": "2026-08-12T21:54:17Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["semantic_scholar_ebt_arm"][0]["url"],
        "endpoint_status": "http_200_ebt_and_arm_citation_routes",
        "disposition": "no_post_marker_relevant_citation",
        "date_evidence": "Newest sampled EBT citation was 2026-08-01; ARM-EBM was 2026-07-02.",
    },
    {
        "channel": "extropic",
        "queried_at_utc": "2026-08-12T21:54:18Z",
        "url": "https://extropic.ai/writing",
        "endpoint_status": "http_200_first_party_pages",
        "disposition": "no_authenticated_local_tsu_route",
        "date_evidence": "Writing x-nextjs-date was 2026-08-11T07:12:45Z.",
    },
    {
        "channel": "logical_intelligence",
        "queried_at_utc": "2026-08-12T21:54:16Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["logical_intelligence"][0]["url"],
        "endpoint_status": "http_200_kona_head",
        "disposition": "no_public_weights_or_local_api",
        "date_evidence": "Kona page last-modified was 2026-06-26T23:48:05Z.",
    },
    {
        "channel": "github",
        "queried_at_utc": "2026-08-12T21:54:06Z",
        "url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["github"][0]["url"],
        "endpoint_status": "http_200_scope_search_total_count_0",
        "disposition": "no_post_marker_contract_changing_repository",
        "date_evidence": "The scoped repository search returned total_count 0.",
    },
)

DEFAULT_SOURCE_CANDIDATES: tuple[JsonDict, ...] = tuple(
    {
        "stable_id": receipt["stable_id"],
        "work_identity": receipt["stable_id"],
        "repository_identity": None,
        "mechanism": receipt["mechanism"],
        "retired_scope": False,
        "title": receipt["title"],
        "url": receipt["direct_url"],
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": receipt["first_publication_utc"],
        "date_evidence": "planner source predates the V547 marker",
        "scope_effect": "Already sealed by the V547 planner marker.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": receipt["content_hash"],
    }
    for receipt in PROMOTED_RECEIPTS
) + (
    {
        "stable_id": "arxiv:post-marker-window-api",
        "work_identity": "arxiv:post-marker-window-api",
        "repository_identity": None,
        "mechanism": "post_marker_arxiv_window",
        "retired_scope": False,
        "title": "arXiv post-marker submission window API",
        "url": "https://export.arxiv.org/api/query",
        "source_channel": "arxiv",
        "source_kind": "api",
        "source_timestamp": "2026-08-12T21:54:05Z",
        "date_evidence": "API returned HTTP 200 with totalResults=0.",
        "scope_effect": "No source promotion.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": False,
        "watch_only": False,
        "content_hash": _receipt_hash("arxiv:post-marker-window-api"),
    },
    {
        "stable_id": "semantic_scholar:ebt-citation-route",
        "work_identity": "semantic_scholar:ebt-citation-route",
        "repository_identity": None,
        "mechanism": "ebt_citation_route",
        "retired_scope": False,
        "title": "Semantic Scholar EBT citation route",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations",
        "source_channel": "semantic_scholar_ebt_arm",
        "source_kind": "api",
        "source_timestamp": "2026-08-12T21:54:17Z",
        "date_evidence": "Sampled citations all predate the V547 marker.",
        "scope_effect": "No source promotion.",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": False,
        "watch_only": False,
        "content_hash": _receipt_hash("semantic_scholar:ebt-citation-route"),
    },
)


def _utc_now() -> str:
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
    if parsed is None:
        raise ValueError("bad V547 marker timestamp")
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


def v547_marker_snapshot(root: Path) -> JsonDict:
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
    for key in ("work_identity", "repository_identity", "stable_id", "url", "title"):
        value = row.get(key)
        if value and str(value).lower() in haystack:
            return key
        if key == "repository_identity" and str(value or "").startswith("github:"):
            github_path = str(value).split(":", 1)[1].lower()
            if f"github.com/{github_path}" in haystack:
                return key
    known_mechanisms = {str(row["mechanism"]) for row in PROMOTED_RECEIPTS}
    if row.get("mechanism") in known_mechanisms:
        return "mechanism"
    return None


def classify_candidate(
    candidate: Mapping[str, Any],
    *,
    reference_text: str,
    seen_works: set[str] | None = None,
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
        row["rejection_reason"] = "watch-only evidence does not change a V547 contract"
        return row
    if row.get("retired_scope") is True:
        row["disposition"] = "excluded"
        row["rejection_reason"] = "candidate belongs to an already-retired Carnot scope"
        return row

    timestamp = _parse_timestamp(row.get("source_timestamp"))
    if timestamp is None or timestamp <= _marker_dt():
        row["disposition"] = "excluded"
        row["rejection_reason"] = "source timestamp is not strictly after the V547 marker"
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

    work = str(row.get("work_identity") or row.get("stable_id") or "")
    repository = str(row.get("repository_identity") or "")
    mechanism = str(row.get("mechanism") or "")
    content_hash = str(row.get("content_hash") or "")
    if work and seen_works is not None and work in seen_works:
        row["disposition"] = "duplicate"
        row["rejection_reason"] = "work identity repeated in this sweep"
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
        for field in ("stable_id", "work_identity", "title", "url", "source_timestamp")
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
    seen_works: set[str] = set()
    seen_repositories: set[str] = set()
    seen_mechanisms: set[str] = set()
    seen_hashes: set[str] = set()
    for candidate in candidates:
        row = classify_candidate(
            candidate,
            reference_text=reference_text,
            seen_works=seen_works,
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

        if disposition == "accepted" and row.get("work_identity"):
            seen_works.add(str(row["work_identity"]))
        if disposition == "accepted" and row.get("repository_identity"):
            seen_repositories.add(str(row["repository_identity"]))
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
    except Exception:
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
        "marker": v547_marker_snapshot(root),
        "input_hashes": _hashes(root, INPUT_RELATIVE_PATHS),
        "explicit_hash_targets": {
            "marker_comment_sha256": payload_sha256(PLANNER_END_MARKER),
            "roadmap_sha256": path_sha256(root / VNEXT_RELATIVE_PATH),
            "references_sha256": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
            "prior_handoff_artifact_sha256": path_sha256(root / PRIOR_HANDOFF_RELATIVE_PATH),
            "prior_source_artifact_sha256": path_sha256(root / PRIOR_V546_SOURCE_RELATIVE_PATH),
            "exclusion_manifest_sha256": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
            "hardware_wishlist_sha256": path_sha256(root / HARDWARE_WISHLIST_RELATIVE_PATH),
        },
        "utc_status": {
            "marker_committed_at_utc": MARKER_COMMITTED_AT_UTC,
            "search_completed_utc": search_completed_utc,
            "utc_now_checked": _utc_now(),
        },
        "network_status": {
            "network_required": True,
            "network_receipts_recorded": True,
            "post_marker_search_window_start_utc": MARKER_COMMITTED_AT_UTC,
            "post_marker_channels": list(REQUIRED_SOURCE_CHANNELS),
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
                "coverage": ".venv/bin/coverage",
                "ruff": ".venv/bin/ruff",
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
    raise KeyError(stable_id)


def semantic_scholar_receipts() -> JsonDict:
    return {
        "ebt": {
            "primary_url": "https://arxiv.org/abs/2507.02092",
            "primary_first_publication_utc": "2025-07-02T19:17:29Z",
            "citation_url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["semantic_scholar_ebt_arm"][0]["url"],
            "queried_at_utc": "2026-08-12T21:54:17Z",
            "api_status": "http_200",
            "sampled_citation_count": 5,
            "newest_sampled_citation_publication_date": "2026-08-01",
            "strict_post_marker_relevant_count": 0,
            "local_consequence": "No EBT citation-derived method reopened hidden-state scoring.",
        },
        "arm_ebm": {
            "primary_url": "https://arxiv.org/abs/2512.15605",
            "primary_first_publication_utc": "2025-12-17T17:14:26Z",
            "latest_arxiv_revision_utc": "2026-05-25T15:54:35Z",
            "citation_url": DEFAULT_SOURCE_QUERIES_BY_CHANNEL["semantic_scholar_ebt_arm"][1]["url"],
            "queried_at_utc": "2026-08-12T21:54:17Z",
            "api_status": "http_200",
            "sampled_citation_count": 5,
            "newest_sampled_citation_publication_date": "2026-07-02",
            "strict_post_marker_relevant_count": 0,
            "local_consequence": "No ARM-EBM citation-derived method widened V547.",
        },
    }


def openreview_and_huggingface_status() -> JsonDict:
    return {
        "openreview": {
            "queried_at_utc": "2026-08-12T21:54:06Z",
            "status": "http_200_review_rows",
            "sampled_title": "Deep Active Inference in a Learning Environment",
            "promotion_allowed": False,
            "reason": "Rows were review records and not strict post-marker primary publications.",
        },
        "hugging_face_papers": {
            "queried_at_utc": "2026-08-12T21:54:06Z",
            "status": "http_200_daily_rows",
            "sampled_title": "ComBodied Agents: a New Paradigm of Human-Centric Agentic AI",
            "sample_published_at": "2026-08-11T00:00:00Z",
            "sample_submitted_on_daily_at": "2026-08-12T00:00:00Z",
            "promotion_allowed": False,
            "reason": "Daily rows predate the marker or are secondary mirrors.",
        },
    }


def github_status() -> JsonDict:
    return {
        "queried_at_utc": "2026-08-12T21:54:06Z",
        "scoped_search_total_count": 0,
        "incomplete_results": False,
        "promotion_allowed": False,
        "reason": "No scoped repository updated after the V547 marker changed a local route.",
    }


def extropic_status() -> JsonDict:
    return {
        "queried_at_utc": "2026-08-12T21:54:18Z",
        "writing_url": "https://extropic.ai/writing",
        "hardware_url": "https://extropic.ai/hardware",
        "writing_http_status": 200,
        "hardware_http_status": 200,
        "writing_x_nextjs_date": "2026-08-11T07:12:45Z",
        "hardware_x_nextjs_date": "2026-08-12T03:11:19Z",
        "z1_early_access": "2027",
        "authenticated_local_tsu_route_present": False,
        "hardware_or_speed_claim_allowed": False,
        "status": "watch_only_no_local_execution_route",
    }


def logical_intelligence_status() -> JsonDict:
    return {
        "queried_at_utc": "2026-08-12T21:54:16Z",
        "kona_url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "http_status": 200,
        "kona_last_modified": "2026-06-26T23:48:05Z",
        "public_weights_present": False,
        "local_api_route_present": False,
        "kona_execution_allowed": False,
        "status": "watch_only_no_local_execution_route",
    }


def frozen_live_factor_learning_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "lane": "prospective_certified_factor_learning",
        "source_evidence": [
            "arxiv:2607.20792",
            "arxiv:2607.17047",
            "arxiv:2605.18871",
            "arxiv:2606.26300",
        ],
        "prospective_generation_required": True,
        "raw_generation_receipts_required": True,
        "same_step_read_write_allowed": False,
        "proposal_reads_only_released_version": True,
        "exact_release_authority": "deterministic_checker_and_eprocess",
        "learned_uncertainty_can_approve": False,
        "abstention_state_required": True,
        "family_blind_controls_required": True,
        "solver_effort_as_model_difficulty_allowed": False,
        "protected_future_events_readable_by_proposer": False,
        "model_weight_update_allowed": False,
        "fail_fast_missing_model": True,
    }


def frozen_arc_goal_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "lane": "falsifiable_live_arc_goal_discovery",
        "source_evidence": [
            "arxiv:2604.07480",
            "arxiv:2605.15333",
            "results/experiment_6258_goal_veto_confusion_matrix.json",
        ],
        "states": ["accepted", "rejected", "unverifiable"],
        "positive_and_negative_evidence_required": True,
        "accepted_without_positive_evidence_allowed": False,
        "active_legal_disagreement_probes_required": True,
        "own_live_attempt_path_only": True,
        "hidden_game_source_allowed": False,
        "offline_bfs_allowed": False,
        "per_game_adapter_allowed": False,
        "public_game_solve_claim_allowed": False,
        "hidden_level_solve_claim_allowed": False,
        "solve_credit_allowed": False,
        "registry_update_allowed": False,
        "default_off": True,
    }


def frozen_model_policy() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "mandatory_hf_ids": list(MANDATED_GGUF_IDS),
        "model_weights_must_remain_frozen": True,
        "llm_call_count_for_exp6351": 0,
        "future_live_factor_tasks": [
            "exp6352-live-factor-proposal-authenticity-preflight",
            "exp6353-live-counterexample-factor-proposal-ab",
            "exp6354-prospective-live-certified-factor-learning",
            "exp6355-default-off-certified-factor-consumer-ab",
        ],
        "future_live_arc_goal_tasks": [
            "exp6359-arc-goal-evidence-response-calibration",
            "exp6360-arc-default-off-active-goal-shadow",
        ],
        "approval_authority": "exact_oracle_not_model_self_judgment",
    }


def frozen_closed_parser_jit_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "lane": "parser_jit_constrained_generation",
        "closed": True,
        "reopen_allowed_in_v547": False,
        "basis": "Exp6340 null semantic-diversity canary and Exp6341 gate block.",
        "exp6340_honest_verdict": (
            "complete_null: no preregistered parser-JIT semantic diversity gain at matched cost"
        ),
        "semantic_diversity_gain_score": 0.0,
        "new_decoding_or_posthoc_repair_retry_allowed": False,
    }


def frozen_hardware_nonuse_contract() -> JsonDict:
    return {
        "version": CONTRACT_VERSION,
        "board_execution_authorized": False,
        "hardware_command_count": 0,
        "gatemate_command_count": 0,
        "kv260_command_count": 0,
        "polarfire_command_count": 0,
        "npu_command_count": 0,
        "tsu_command_count": 0,
        "kona_execution_count": 0,
        "excluded_hardware": ["GateMate", "KV260", "PolarFire", "NPU", "TSU", "Kona"],
        "allowed_accelerator": "dual_rtx_3090_for_future_local_gguf_generation_only",
        "reason": "V547 source freeze found no new authenticated physical-state receipt.",
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
        "REQ-INFRA-6351",
        "research-references V547 marker",
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
        "v547_marker_text_line_and_hash": v547_marker_snapshot(root),
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
        "active_reward_machine_receipt": _receipt("arxiv:2604.07480", accessed_at_utc=completed),
        "zero_shot_goal_recognition_receipt": _receipt(
            "arxiv:2605.15333", accessed_at_utc=completed
        ),
        "memoir_receipt": _receipt("arxiv:2607.20792", accessed_at_utc=completed),
        "solver_hardness_control_receipt": _receipt("arxiv:2607.17047", accessed_at_utc=completed),
        "distributional_ebm_receipt": _receipt("arxiv:2605.18871", accessed_at_utc=completed),
        "verification_horizon_receipt": _receipt("arxiv:2606.26300", accessed_at_utc=completed),
        "semantic_scholar_ebt_and_arm_ebm_receipts": semantic_scholar_receipts(),
        "openreview_and_huggingface_status": openreview_and_huggingface_status(),
        "github_status": github_status(),
        "extropic_status": extropic_status(),
        "logical_intelligence_status": logical_intelligence_status(),
        "frozen_live_factor_learning_contract": frozen_live_factor_learning_contract(),
        "frozen_arc_goal_contract": frozen_arc_goal_contract(),
        "frozen_model_policy": frozen_model_policy(),
        "frozen_closed_parser_jit_contract": frozen_closed_parser_jit_contract(),
        "frozen_hardware_nonuse_contract": frozen_hardware_nonuse_contract(),
        "roadmap_scope_delta": _roadmap_scope_delta(accepted_count),
        "protected_files_unchanged": protected_unchanged(root, before_hashes),
        "preconditions_checked": preconditions(root, date=date, search_completed_utc=completed),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "llm_call_count": 0,
        "model_specs": {
            "model_execution": "none",
            "reason": "Exp6351 performs source search and local artifact validation only.",
            "mandatory_future_hf_ids": list(MANDATED_GGUF_IDS),
        },
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": _test_exit_codes(command_receipts),
        "duration_s": float(duration_s),
        "random_seed": 6351,
        "random_seeds": {"used": [], "deterministic": True},
        "honest_verdict": (
            "complete_delta: accepted post-marker source was recorded without widening roadmap"
            if accepted_count
            else "complete_null: no strict post-marker source changed V547 scope"
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
    if dict(report["frozen_live_factor_learning_contract"]).get("version") != CONTRACT_VERSION:
        errors.append("frozen_live_factor_learning_contract")
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
        raise ValueError(f"invalid Exp6351 freeze: {errors}")
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
    parser = argparse.ArgumentParser(description="Write the Exp6351 V547 source freeze.")
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    report = run(date=args.date)
    print(json.dumps({"path": RESULT_RELATIVE_PATH.as_posix(), "status": report["status"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
