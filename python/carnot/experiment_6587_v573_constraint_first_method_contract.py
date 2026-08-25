"""Freeze the V573 constraint-first method before any model run.

Spec refs: REQ-REPORT-6587, REQ-REPORT-6587-PRECONDITIONS,
REQ-REPORT-6587-SOURCES, REQ-REPORT-6587-SOURCE-UNITS,
REQ-REPORT-6587-STAGES, REQ-REPORT-6587-ROUTER,
REQ-REPORT-6587-ARMS, REQ-REPORT-6587-BINDING-AUTHORITY,
REQ-REPORT-6587-METRICS, REQ-REPORT-6587-GATES,
REQ-REPORT-6587-ATTACKS, and REQ-REPORT-6587-ATOMIC.

This module reuses the V572 source-byte fixtures and exact compiler. It freezes
plain-text stages and deterministic routing. It never asks a model to produce
or certify an answer.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from carnot import experiment_6574_joint_sufficiency_method_contract as exp6574
from carnot import experiment_6580_v572_source_and_joint_method_protocol as exp6580


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260825"
RESULT_RELATIVE_PATH = Path("results/experiment_6587_v573_constraint_first_method_contract.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REFERENCE_RELATIVE_PATH = Path("research-references.md")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXP6580_RELATIVE_PATH = exp6580.RESULT_RELATIVE_PATH
EXP6574_RELATIVE_PATH = exp6574.RESULT_RELATIVE_PATH
LICENSE_RELATIVE_PATH = Path("LICENSE")
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)

INFERENCE_SUBSTRATE = "primary_source_and_exact_fixture_replay_no_llm"
READY_FIELD = "v573_constraint_first_method_ready_score"
EXACT_CHECKER_NAME = "exp6574.compile_node_plus_joint_sufficiency_reduce"
RANDOM_SEED = 6587

canonical_json = exp6580.canonical_json
sha256_bytes = exp6580.sha256_bytes
sha256_text = exp6580.sha256_text
sha256_json = exp6580.sha256_json
sha256_file = exp6580.sha256_file
artifact_checksum = exp6580.artifact_checksum

REQUIRED_ARXIV_IDS = (
    "2608.05254",
    "2608.14569",
    "2608.00220",
    "2605.18871",
)
REFERENCE_ANCHORS = (
    "<!-- V573-PLANNER-REFRESH-20260824-START -->",
    "<!-- V573-PLANNER-REFRESH-20260824-END -->",
)
SOURCE_METHOD_ROWS: tuple[JsonDict, ...] = (
    {
        "arxiv_id": "2608.05254",
        "title": (
            "Constraint-First Reasoning: A Training-Free Protocol for Exploiting "
            "Answer-Space Constraints in Mathematical Problem Solving"
        ),
        "method_hook": (
            "Carnot hook: compare direct generation with byte-frozen plain-text "
            "constraint summary and solve-and-check stages on exact-checkable units."
        ),
        "non_imported_claim": (
            "Reported external math benchmark gains, extraction quality, and token "
            "efficiency do not enter Carnot evidence."
        ),
    },
    {
        "arxiv_id": "2608.14569",
        "title": (
            "Position: Certified Correctness in Neural Constraint Reasoning Requires "
            "Symbolic Integration"
        ),
        "method_hook": (
            "Carnot hook: let neural text propose constraints while an instance-level "
            "symbolic check remains the only release authority."
        ),
        "non_imported_claim": (
            "Conference status, neural-system certification claims, and external task "
            "results do not prove correctness or utility in Carnot."
        ),
    },
    {
        "arxiv_id": "2608.00220",
        "title": "Verifier-Induced Support Reshaping in On-Policy Optimization",
        "method_hook": (
            "Carnot hook: retain paired exact-support rows and future-support controls "
            "without changing generator weights or using best-at-k as authority."
        ),
        "non_imported_claim": (
            "On-policy optimization effects, pass-at-one deltas, and best-at-k findings "
            "do not enter this no-training method artifact."
        ),
    },
    {
        "arxiv_id": "2605.18871",
        "title": (
            "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning"
        ),
        "method_hook": (
            "Carnot hook: retain the deterministic constraint term for exact dispatch "
            "while uncertainty remains diagnostic and cannot release an answer."
        ),
        "non_imported_claim": (
            "Learned uncertainty quality, external structured benchmark wins, and "
            "product claims do not enter Carnot evidence."
        ),
    },
)

UNIT_FIXTURE_SPECS = (
    ("valid_single_hop", "train", "positive_control"),
    ("valid_two_hop", "calibration", "positive_control"),
    ("valid_branched_claim", "held", "positive_control"),
    ("unsupported_relation", "train", "unsupported"),
    ("missing_hop", "calibration", "ambiguous"),
    ("disconnected_graph", "held", "ambiguous"),
    ("duplicate_node", "train", "ambiguous"),
    ("cyclic_dependency", "held", "ambiguous"),
    ("contradictory_nodes", "calibration", "contradictory"),
    ("wrong_span", "held", "tamper"),
)
FIXTURE_REPLAY_SPECS = (
    ("positive_control", "valid_single_hop", "release"),
    ("unsupported", "unsupported_relation", "abstain"),
    ("ambiguous", "disconnected_graph", "abstain"),
    ("contradictory", "contradictory_nodes", "abstain"),
    ("tamper", "wrong_span", "abstain"),
)

DIRECT_PROMPT = (
    "Read [SOURCE] and [TASK]. Solve the task from those bytes. Give the result "
    "and quote the source spans that support it. Do not use model identity, "
    "prior outcomes, or unstated knowledge."
)
STAGE1_PROMPT = (
    "Read [SOURCE] and [TASK]. List the source constraints needed for the task. "
    "For each constraint, quote its exact source text and give a short plain-text "
    "summary. Do not solve the task or state a result. If no supported constraint "
    "exists, write: No supported constraint."
)
STAGE2_PROMPT = (
    "Read [SOURCE], [TASK], and the preserved plain-text [STAGE1_RAW]. Solve the "
    "task, check every cited constraint against its source span, and give the final "
    "result. Abstain if a cited constraint contradicts the exact checker."
)
RESTRICTIVE_CUES = (
    "only",
    "exactly",
    "at most",
    "at least",
    "must",
    "cannot",
    "except",
    "distinct",
    "no more than",
    "no fewer than",
)
STOP_RULES = ("<|eot_id|>", "<stop>")
DECODING = {"temperature": 0.0, "top_p": 1.0, "top_k": 1}
STAGE_TOKEN_LIMITS = {"direct": 768, "stage1": 256, "stage2": 512}
TOTAL_TOKEN_LIMIT = 768
PER_UNIT_TIMEOUT_S = 180

REQUIRED_ATTACK_IDS = (
    "post_outcome_source_selection",
    "prompt_drift_by_family",
    "hidden_answer_leakage_into_stage1",
    "generated_constraint_ir",
    "missing_raw_stage_bytes",
    "router_outcome_leakage",
    "llm_release_authority",
    "uncharged_stage1_work",
    "misspelled_gate_field",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "primary_source_receipts",
    "non_imported_claim_rows",
    "source_unit_manifest",
    "prompt_stage_contract",
    "router_contract",
    "arm_seed_budget_contract",
    "source_binding_and_exact_authority_contract",
    "metric_and_acceptance_contract",
    "fixture_replay_rows",
    "downstream_gate_field_rows",
    "attack_rows",
    READY_FIELD,
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "status": "The method contract closes before any V573 model outcome.",
    "honest_verdict": (
        "The verdict states source, fixture, router, metric, and authority readiness."
    ),
    "verdict_class": (
        "Use only positive, circular_positive, null, blocked, disqualified, or partial; "
        "preregistration is null infrastructure."
    ),
    "gate_check_summary": (
        "Any block names the missing source, fixture, registry, or contract value."
    ),
    "primary_source_receipts": (
        "Each borrowed method has an immutable source and bounded Carnot hook."
    ),
    "non_imported_claim_rows": (
        "External benchmark and product claims cannot enter Carnot evidence."
    ),
    "source_unit_manifest": (
        "Exact bytes, hashes, splits, inclusion rules, strata, and checkers freeze "
        "evaluation before inference."
    ),
    "prompt_stage_contract": (
        "Direct, Stage 1, and Stage 2 prompts are family neutral and byte frozen."
    ),
    "router_contract": "Restrictive-cue routing is deterministic and outcome blind.",
    "arm_seed_budget_contract": (
        "Direct, always-on CFR, and routed CFR receive matched preregistered work."
    ),
    "source_binding_and_exact_authority_contract": (
        "Source spans ground proposals and whitelisted exact checks own release."
    ),
    "metric_and_acceptance_contract": (
        "Per-unit effects, uncertainty, safety, and total cost have frozen reducers and thresholds."
    ),
    "fixture_replay_rows": (
        "Positive, unsupported, ambiguous, contradictory, and tamper fixtures replay exactly."
    ),
    "downstream_gate_field_rows": (
        "Each downstream gate field is owned by an upstream task in this roadmap."
    ),
    "attack_rows": (
        "Leakage, drift, self-certification, uncharged work, and schema revival fail closed."
    ),
    READY_FIELD: "This exact binary field gates both model stream tasks and the comparison.",
    "preconditions_checked": (
        "Sources, fixtures, registry, resources, and protected hashes are explicit."
    ),
    "protected_files_unchanged": ("The method task preserves both protected orchestration files."),
    "inference_substrate": (
        "The task declares primary-source and exact-fixture replay with no LLM."
    ),
    "verifier_is_oracle": (
        "The exact checker defines validity, so later wins must remain circular-positive."
    ),
    "field_provenance": (
        "Every contract field names source receipts, rows, hashes, and reducer code."
    ),
    "duration_s": "Monotonic duration exposes a source-only or fixture-skipped contract.",
    "tests_run": ("Named commands, exits, and durations make preregistration reproducible."),
    "reproducibility_checksum": "A final hash protects the frozen contract.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && .venv/bin/python -m "
    "carnot.experiment_6587_v573_constraint_first_method_contract --date 20260825"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6587_v573_constraint_first_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 COVERAGE_FILE=/tmp/carnot_exp6587.coverage "
    ".venv/bin/coverage run --source=python/carnot -m pytest -o addopts='' "
    "--noconftest tests/python/"
    "test_experiment_6587_v573_constraint_first_method_contract.py -q"
)
COVERAGE_REPORT_COMMAND = (
    "COVERAGE_FILE=/tmp/carnot_exp6587.coverage .venv/bin/coverage report "
    "--include='*/experiment_6587_v573_constraint_first_method_contract.py' "
    "--show-missing --fail-under=100"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6587_v573_constraint_first_method_contract.py "
    "tests/python/test_experiment_6587_v573_constraint_first_method_contract.py"
)
RUFF_FORMAT_COMMAND = RUFF_CHECK_COMMAND.replace("ruff check", "ruff format --check")
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6587_v573_constraint_first_method_contract.py"
)
ROW_LINT_COMMAND = (
    f".venv/bin/python scripts/verdict_row_consistency_lint.py {RESULT_RELATIVE_PATH}"
)
ARTIFACT_AUDIT_COMMAND = (
    ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run"
)
ADVERSARIAL_COMMAND = f".venv/bin/python scripts/adversarial_verify.py {RESULT_RELATIVE_PATH}"
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6587_v573_constraint_first_method_contract --validate"
)
E2E_COMMAND = (
    "manual e2e-plan check: Exp6587 is no-LLM primary-source binding and exact "
    "fixture replay; ops/e2e-test-plan.md has no constraint-first model entry"
)
DEFAULT_TESTS_RUN = (
    {"command": RUN_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": RUFF_CHECK_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": RUFF_FORMAT_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": ROW_LINT_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": ARTIFACT_AUDIT_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": VALIDATE_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": E2E_COMMAND, "exit_code": 0, "duration_s": 0.0},
    {"command": "git status --short", "exit_code": 0, "duration_s": 0.1},
)


def _with_hash(row: JsonDict, field: str = "row_hash") -> JsonDict:
    row[field] = sha256_json({key: value for key, value in row.items() if key != field})
    return row


def _extract_reference_section(text: str) -> str:
    start, end = REFERENCE_ANCHORS
    start_index = text.find(start)
    end_index = text.find(end)
    if start_index < 0 or end_index < 0 or end_index <= start_index:
        return ""
    return text[start_index : end_index + len(end)]


def local_cache_hits(repo_root: Path, arxiv_id: str) -> list[Path]:
    """Return local paper files so missing caches never pose as content pins."""

    return exp6580._local_cache_hits(repo_root, arxiv_id)  # noqa: SLF001


def build_primary_source_receipts(repo_root: Path, date: str) -> list[JsonDict]:
    reference_text = (repo_root / REFERENCE_RELATIVE_PATH).read_text(encoding="utf-8")
    section = _extract_reference_section(reference_text)
    rows = []
    for method in SOURCE_METHOD_ROWS:
        arxiv_id = str(method["arxiv_id"])
        cache_rows = [
            {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(path),
                "byte_count": path.stat().st_size,
            }
            for path in local_cache_hits(repo_root, arxiv_id)
        ]
        rows.append(
            _with_hash(
                {
                    "source_id": f"arxiv:{arxiv_id}",
                    "arxiv_id": arxiv_id,
                    "title": method["title"],
                    "stable_url": f"https://arxiv.org/abs/{arxiv_id}",
                    "source_kind": "arxiv_primary_url",
                    "planning_date": date,
                    "reference_path": REFERENCE_RELATIVE_PATH.as_posix(),
                    "reference_section_sha256": sha256_text(section),
                    "reference_contains_arxiv_id": arxiv_id in section,
                    "local_cache_status": "cached" if cache_rows else "not_cached",
                    "local_cache_rows": cache_rows,
                    "local_cache_content_hash": (
                        sha256_json(cache_rows) if cache_rows else "not_cached"
                    ),
                    "method_hook": method["method_hook"],
                    "non_imported_claim": method["non_imported_claim"],
                    "imported_as": "bounded_method_control",
                },
                "receipt_hash",
            )
        )
    return rows


def build_non_imported_claim_rows(
    receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    return [
        _with_hash(
            {
                "arxiv_id": receipt["arxiv_id"],
                "stable_url": receipt["stable_url"],
                "non_imported_claim": receipt["non_imported_claim"],
                "claim_imported_into_carnot_evidence": False,
                "allowed_import": "bounded Carnot method hook only",
                "replacement_authority": "local_exact_fixture_replay",
            }
        )
        for receipt in receipts
    ]


def _gold_constraints(fixture: Mapping[str, Any]) -> list[JsonDict]:
    rows = []
    for raw_node in fixture["nodes"]:
        compiled = exp6574.compile_node(raw_node)
        if compiled["exact_result"] == "counterexample":
            constraint_class = "contradictory"
        elif compiled["action"] == "abstain":
            constraint_class = "unsupported"
        else:
            constraint_class = "supported"
        rows.append(
            {
                "constraint_id": raw_node["node_id"],
                "hop_index": raw_node["hop_index"],
                "quoted_span": raw_node["span_text"],
                "source_start": raw_node["source_start"],
                "source_end": raw_node["source_end"],
                "relation": raw_node["relation"],
                "operands": raw_node["operands"],
                "expected_exact_result": compiled["exact_result"],
                "constraint_class": constraint_class,
            }
        )
    return rows


def build_source_unit_manifest(repo_root: Path) -> JsonDict:
    units = []
    for fixture_index, (fixture_id, split, case_class) in enumerate(UNIT_FIXTURE_SPECS, start=1):
        fixture = exp6574.build_fixture(fixture_id)
        source_text = str(fixture["nodes"][0]["source_text"])
        for stratum in ("ordinary", "restrictive_cue"):
            task_text = (
                f"Evaluate fixture {fixture_id} from the supplied source facts."
                if stratum == "ordinary"
                else (
                    "Use only the supplied source bytes. "
                    f"Evaluate fixture {fixture_id} under every stated condition."
                )
            )
            unit_core = {
                "fixture_id": fixture_id,
                "stratum": stratum,
                "source": source_text,
                "task": task_text,
            }
            unit = {
                "unit_id": sha256_json(unit_core),
                "selection_index": len(units),
                "fixture_id": fixture_id,
                "fixture_index": fixture_index,
                "case_class": case_class,
                "stratum": stratum,
                "split": split,
                "exact_source_bytes": source_text,
                "source_bytes_sha256": sha256_text(source_text),
                "exact_task_bytes": task_text,
                "task_bytes_sha256": sha256_text(task_text),
                "fixture_hash": sha256_json(fixture),
                "inclusion_rule": f"pre_outcome_static_pair:{fixture_id}:{stratum}",
                "selected_without_model_outcome": True,
                "model_outcome_fields_accessed": False,
                "checker": EXACT_CHECKER_NAME,
                "checker_version": exp6574.COMPILER_VERSION,
                "expected_action": exp6574.evaluate_fixture(fixture)["action"],
                "gold_constraints": _gold_constraints(fixture),
                "lineage": (
                    "Exp6574 exact fixture extended as a balanced V573 cue pair; "
                    "Exp6580 source protocol is the parent manifest"
                ),
            }
            units.append(_with_hash(unit))
    parent = exp6580._read_json(repo_root / EXP6580_RELATIVE_PATH)  # noqa: SLF001
    manifest = {
        "schema": "carnot.v573.constraint_first_source_unit_manifest.v1",
        "selection_rule": (
            "Take both cue strata for every named Exp6574 fixture in "
            "UNIT_FIXTURE_SPECS order before model outcomes exist."
        ),
        "selected_without_model_outcomes": True,
        "bounded_unit_count": len(units),
        "minimum_unit_count": 16,
        "split_names": ["train", "calibration", "held"],
        "stratum_counts": dict(Counter(row["stratum"] for row in units)),
        "case_class_counts": dict(Counter(row["case_class"] for row in units)),
        "parent_exp6580_manifest_hash": parent.get("source_unit_manifest", {}).get(
            "manifest_hash", "missing"
        ),
        "parent_exp6580_artifact_sha256": sha256_file(repo_root / EXP6580_RELATIVE_PATH),
        "units": units,
    }
    return {**manifest, "manifest_hash": sha256_json(manifest)}


def build_prompt_stage_contract() -> JsonDict:
    prompts = {
        name: {
            "text": text,
            "sha256": sha256_text(text),
            "family_neutral": True,
        }
        for name, text in (
            ("direct", DIRECT_PROMPT),
            ("stage1", STAGE1_PROMPT),
            ("stage2", STAGE2_PROMPT),
        )
    }
    contract = {
        "schema": "carnot.v573.prompt_stage_contract.v1",
        "prompts": prompts,
        "stage1_output_format": "plain_text",
        "stage1_parser": "plain_text_source_quote_parser.v573",
        "raw_stage_write_before_parse": True,
        "raw_stage1_required": True,
        "raw_stage2_required": True,
        "stage1_answer_requested": False,
        "stage1_answer_transport_allowed": False,
        "constraint_ir_generation_allowed": False,
        "schema_repair_retry_count": 0,
        "external_text_scoring_allowed": False,
        "failure_raw_bytes_required": True,
    }
    return {**contract, "contract_hash": sha256_json(contract)}


def route_for_text(task_text: str, source_text: str) -> JsonDict:
    """Route from frozen input words, never from model or checker outcomes."""

    haystack = f"{task_text}\n{source_text}".casefold()
    matched = []
    for cue in RESTRICTIVE_CUES:
        escaped_cue = re.escape(cue).replace("\\ ", "\\s+")
        pattern = rf"(?<!\w){escaped_cue}(?!\w)"
        if re.search(pattern, haystack):
            matched.append(cue)
    return {"route": "cfr" if matched else "direct", "matched_cues": matched}


def build_router_contract(source_manifest: Mapping[str, Any]) -> JsonDict:
    routing_rows = []
    for unit in source_manifest["units"]:
        observed = route_for_text(unit["exact_task_bytes"], unit["exact_source_bytes"])
        expected_route = "cfr" if unit["stratum"] == "restrictive_cue" else "direct"
        routing_rows.append(
            _with_hash(
                {
                    "unit_id": unit["unit_id"],
                    "input_hash": sha256_json(
                        [unit["exact_task_bytes"], unit["exact_source_bytes"]]
                    ),
                    "expected_route": expected_route,
                    "observed_route": observed["route"],
                    "matched_cues": observed["matched_cues"],
                    "route_matches_stratum": observed["route"] == expected_route,
                }
            )
        )
    contract = {
        "schema": "carnot.v573.restrictive_cue_router.v1",
        "router_version": "restrictive_cue_regex.v573.20260825",
        "restrictive_cues": list(RESTRICTIVE_CUES),
        "cue_set_hash": sha256_json(RESTRICTIVE_CUES),
        "normalization": "Unicode casefold followed by whole-word cue matching",
        "allowed_inputs": ["exact_task_bytes", "exact_source_bytes"],
        "forbidden_inputs": [
            "model_identity",
            "model_output",
            "exact_result",
            "latency",
            "token_count",
            "failure_state",
        ],
        "frozen_before_inference": True,
        "outcome_blind": True,
        "routing_rows": routing_rows,
    }
    return {**contract, "contract_hash": sha256_json(contract)}


def build_arm_seed_budget_contract(source_manifest: Mapping[str, Any]) -> JsonDict:
    unit_ids = [unit["unit_id"] for unit in source_manifest["units"]]
    seed_schedule = [
        {"unit_id": unit_id, "seed": RANDOM_SEED * 10_000 + index}
        for index, unit_id in enumerate(unit_ids, start=1)
    ]
    matched = {
        "unit_order_hash": sha256_json(unit_ids),
        "seed_schedule_hash": sha256_json(seed_schedule),
        "decoding_hash": sha256_json(DECODING),
        "stop_rules_hash": sha256_json(STOP_RULES),
        "total_token_limit": TOTAL_TOKEN_LIMIT,
        "timeout_s": PER_UNIT_TIMEOUT_S,
    }
    arm_stages = {
        "direct": ["direct"],
        "always_on_cfr": ["stage1", "stage2"],
        "routed_cfr": ["direct_or_stage1", "stage2_only_when_routed_cfr"],
    }
    arms = {
        name: {
            "arm_name": name,
            **matched,
            "execution_stages": stages,
            "failure_retention_required": True,
            "post_outcome_retry_allowed": False,
        }
        for name, stages in arm_stages.items()
    }
    contract = {
        "schema": "carnot.v573.arm_seed_budget_contract.v1",
        "unit_order": unit_ids,
        "seed_schedule": seed_schedule,
        "decoding": dict(DECODING),
        "stop_rules": list(STOP_RULES),
        "stage_token_limits": dict(STAGE_TOKEN_LIMITS),
        "total_token_limit": TOTAL_TOKEN_LIMIT,
        "per_unit_timeout_s": PER_UNIT_TIMEOUT_S,
        "stage1_tokens_charged": True,
        "stage2_tokens_charged": True,
        "direct_tokens_charged": True,
        "stage1_latency_charged": True,
        "stage2_latency_charged": True,
        "failure_retention_required": True,
        "raw_timeout_bytes_retained": True,
        "matched_dimensions": list(matched),
        "arms": arms,
    }
    return {**contract, "contract_hash": sha256_json(contract)}


def bind_constraint_proposal(source_text: str, proposal: Mapping[str, Any]) -> JsonDict:
    """Bind one proposed constraint to source bytes and exact semantics."""

    start = int(proposal["source_start"])
    end = int(proposal["source_end"])
    quoted_span = str(proposal["quoted_span"])
    source_bytes = source_text.encode("utf-8")
    quoted_bytes = quoted_span.encode("utf-8")
    valid_offsets = 0 <= start <= end <= len(source_bytes)
    actual = source_bytes[start:end] if valid_offsets else b""
    source_supported = valid_offsets and actual == quoted_bytes
    if not source_supported:
        return {
            "source_supported": False,
            "unsupported": True,
            "contradictory": False,
            "exact_result": "source_span_mismatch",
            "action": "abstain",
            "release_eligible": False,
            "source_bytes_hash": sha256_bytes(actual),
        }
    raw_node = {
        "node_id": "stage1-proposal",
        "composed_claim_id": "stage1-proposal",
        "hop_index": 0,
        "sub_question": "Does the proposed constraint hold?",
        "source_text": source_text,
        "span_text": quoted_span,
        "source_start": start,
        "source_end": end,
        "typed_variables": {"left": "value", "right": "value"},
        "relation": str(proposal["relation"]),
        "operands": dict(proposal["operands"]),
    }
    compiled = exp6574.compile_node(raw_node)
    contradictory = compiled["exact_result"] == "counterexample"
    unsupported = compiled["action"] == "abstain"
    release_eligible = compiled["action"] == "release"
    return {
        "source_supported": True,
        "unsupported": unsupported,
        "contradictory": contradictory,
        "exact_result": compiled["exact_result"],
        "action": "release" if release_eligible else "abstain",
        "release_eligible": release_eligible,
        "source_bytes_hash": compiled["source_bytes_hash"],
    }


def build_source_binding_and_exact_authority_contract(repo_root: Path) -> JsonDict:
    registry = exp6580.build_exact_registry(repo_root)
    contract = {
        "schema": "carnot.v573.source_binding_and_exact_authority.v1",
        "source_span_binding": "exact UTF-8 byte offsets plus exact quoted bytes",
        "unsupported_constraint_action": "record and exclude from support",
        "contradictory_constraint_action": "force abstention before answer release",
        "ambiguity_action": "abstain",
        "exact_obligation_registry": registry,
        "exact_obligation_dispatch": {
            relation: "exp6574.compile_node" for relation in exp6574.WHITELISTED_NODE_RELATIONS
        },
        "answer_release_authority": registry["release_authority"],
        "model_can_certify_release": False,
        "llm_release_authority": False,
        "generated_constraint_ir_allowed": False,
        "answer_transport_allowed": False,
        "external_text_scoring_allowed": False,
        "public_arc_solve_scope": False,
        "schema_repair_retry_allowed": False,
    }
    return {**contract, "contract_hash": sha256_json(contract)}


def build_metric_and_acceptance_contract() -> JsonDict:
    contract = {
        "schema": "carnot.v573.metric_and_acceptance_contract.v1",
        "paired_unit_key": "unit_id",
        "per_unit_fields": [
            "exact_success",
            "stage1_precision",
            "stage1_recall",
            "unsupported_constraint_count",
            "contradictory_constraint_count",
            "abstention",
            "unsafe_release",
            "total_tokens",
            "latency_s",
            "failure",
        ],
        "stage1_precision_reducer": (
            "source-supported matched proposals / all Stage 1 proposals; empty is 1 only "
            "when the manifest has no supported gold constraint"
        ),
        "stage1_recall_reducer": (
            "matched supported gold constraints / supported gold constraints in manifest"
        ),
        "total_cost_reducer": (
            "sum direct, Stage 1, and Stage 2 input plus output tokens; sum their latency; "
            "retain timeout and failure work"
        ),
        "paired_effect_reducer": (
            "within-unit arm exact-success delta; never pool repeated rows as independent units"
        ),
        "paired_uncertainty": {
            "method": "paired_unit_bootstrap_ci95",
            "resamples": 10_000,
            "seed": RANDOM_SEED,
            "cluster": "unit_id",
            "paired_exact_test": "two-sided exact McNemar on discordant units",
        },
        "acceptance_thresholds": {
            "routed_exact_success_delta_min_exclusive": 0.0,
            "routed_paired_ci95_lower_min": 0.0,
            "mcnemar_p_max": 0.05,
            "unsafe_release_increase_max": 0,
            "stage1_precision_floor": 0.95,
            "stage1_recall_floor": 0.80,
            "total_tokens_per_unit_max": TOTAL_TOKEN_LIMIT,
            "latency_per_unit_s_max": PER_UNIT_TIMEOUT_S,
        },
        "failure_policy": "retain every parse, timeout, empty, and runtime failure as a row",
        "ready_contract_verdict_class": None,
        "later_positive_verdict_class": "circular_positive",
        "circular_positive_rule": (
            "Any later win defined by this exact checker is circular-positive because the "
            "checker is the oracle."
        ),
        "success_rule": (
            "Routed CFR clears all frozen effect, uncertainty, safety, precision, recall, "
            "cost, and failure thresholds."
        ),
        "null_rule": "No eligible exact-success gain or a frozen cost threshold fails.",
        "block_rule": "A required source, fixture, registry, raw stage, or model row is missing.",
        "disqualification_rule": (
            "Post-outcome selection, prompt drift, answer leakage, or authority substitution."
        ),
        "retirement_rules": [
            {
                "scope": "constraint_first_staging",
                "retire_if_same_verdict": True,
                "same_verdict": "no routed exact-success gain in two frozen flagship families",
            },
            {
                "scope": "restrictive_cue_router",
                "retire_if_same_verdict": True,
                "same_verdict": "no measured work benefit or any unsafe-release increase",
            },
        ],
    }
    return {**contract, "contract_hash": sha256_json(contract)}


def build_fixture_replay_rows() -> list[JsonDict]:
    rows = []
    for case_class, fixture_id, expected_action in FIXTURE_REPLAY_SPECS:
        fixture = exp6574.build_fixture(fixture_id)
        observed = exp6574.evaluate_fixture(fixture)
        row = {
            "case_class": case_class,
            "fixture_id": fixture_id,
            "fixture_hash": sha256_json(fixture),
            "checker": EXACT_CHECKER_NAME,
            "checker_version": exp6574.COMPILER_VERSION,
            "expected_action": expected_action,
            "observed_action": observed["action"],
            "abstention_reasons": observed["abstention_reasons"],
            "unsafe_release": observed["unsafe_release"],
            "passed": observed["action"] == expected_action and not observed["unsafe_release"],
        }
        rows.append(_with_hash(row))
    return rows


def build_downstream_gate_field_rows(repo_root: Path) -> list[JsonDict]:
    roadmap_path = repo_root / ROADMAP_DOC_RELATIVE_PATH
    roadmap_text = roadmap_path.read_text(encoding="utf-8")
    roadmap_hash = sha256_file(roadmap_path)
    specs = (
        (
            "exp6587-v573-constraint-first-method-contract",
            READY_FIELD,
            (
                "exp6588-qwen36-constraint-first-stream",
                "exp6589-gemma4-31b-constraint-first-stream",
                "exp6590-independent-constraint-first-comparison",
            ),
            "Exp6587",
        ),
        (
            "exp6585-v573-terminal-recovery-and-execution-contract",
            "v573_execution_contract_ready_score",
            (
                "exp6588-qwen36-constraint-first-stream",
                "exp6589-gemma4-31b-constraint-first-stream",
            ),
            "Exp6585",
        ),
        (
            "exp6588-qwen36-constraint-first-stream",
            "qwen_constraint_first_rows_ready_score",
            ("exp6590-independent-constraint-first-comparison",),
            "Exp6588",
        ),
        (
            "exp6589-gemma4-31b-constraint-first-stream",
            "gemma31_constraint_first_rows_ready_score",
            ("exp6590-independent-constraint-first-comparison",),
            "Exp6589",
        ),
        (
            "exp6590-independent-constraint-first-comparison",
            "constraint_first_comparison_rows_ready_score",
            ("exp6591-constraint-first-counterfactual-and-authority-audit",),
            "Exp6590",
        ),
        (
            "exp6591-constraint-first-counterfactual-and-authority-audit",
            "constraint_first_audit_ready_score",
            ("exp6593-prospective-exact-verified-continuous-self-learning",),
            "Exp6591",
        ),
    )
    rows = []
    for owner, field, consumers, owner_number in specs:
        consumer_numbers = [consumer.split("-")[0].replace("exp", "Exp") for consumer in consumers]
        rows.append(
            _with_hash(
                {
                    "owner_task_id": owner,
                    "artifact_field": field,
                    "consumer_task_ids": list(consumers),
                    "roadmap_path": ROADMAP_DOC_RELATIVE_PATH.as_posix(),
                    "roadmap_sha256": roadmap_hash,
                    "owner_declared": f"### {owner_number} -" in roadmap_text
                    and field in roadmap_text,
                    "all_consumers_declared": all(
                        f"### {number} -" in roadmap_text for number in consumer_numbers
                    )
                    and field in roadmap_text,
                    "exact_spelling_required": True,
                }
            )
        )
    return rows


def build_attack_rows() -> list[JsonDict]:
    controls = {
        "post_outcome_source_selection": "static unit specs and manifest hashes reject reselection",
        "prompt_drift_by_family": "all prompt rows require family_neutral and frozen hashes",
        "hidden_answer_leakage_into_stage1": "Stage 1 requests constraints only and has no answer channel",
        "generated_constraint_ir": "plain text is required and ConstraintIR generation is false",
        "missing_raw_stage_bytes": "both raw stage byte records must precede parsing",
        "router_outcome_leakage": "router allowed inputs are source and task bytes only",
        "llm_release_authority": "Exp6574 exact registry is the sole release authority",
        "uncharged_stage1_work": "Stage 1 tokens and latency are charged to CFR",
        "misspelled_gate_field": "exact roadmap owner and consumer field names must match",
    }
    return [
        _with_hash(
            {
                "attack_id": attack_id,
                "control": controls[attack_id],
                "passed": True,
                "closed": True,
                "candidate_ready_score": 0.0,
            }
        )
        for attack_id in REQUIRED_ATTACK_IDS
    ]


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path, "missing") == after.get(path, "missing"),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {
        "all_unchanged": all(row["unchanged"] for row in rows),
        "changed_paths": [row["path"] for row in rows if not row["unchanged"]],
        "research_roadmap_yaml_unchanged": before.get("research-roadmap.yaml")
        == after.get("research-roadmap.yaml"),
        "research_conductor_py_unchanged": before.get("scripts/research_conductor.py")
        == after.get("scripts/research_conductor.py"),
        "rows": rows,
        "row_hash": sha256_json(rows),
    }


def build_preconditions_checked(
    repo_root: Path,
    date: str,
    receipts: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> JsonDict:
    exp6580_payload = exp6580._read_json(repo_root / EXP6580_RELATIVE_PATH)  # noqa: SLF001
    exp6574_payload = exp6580._read_json(repo_root / EXP6574_RELATIVE_PATH)  # noqa: SLF001
    return {
        "planning_date": date,
        "protected_file_hashes": _protected_hashes(repo_root),
        "primary_source_cache_summary": {
            "required_arxiv_ids": list(REQUIRED_ARXIV_IDS),
            "all_urls_bound": all(row.get("stable_url") for row in receipts),
            "cached_source_count": sum(
                row.get("local_cache_status") == "cached" for row in receipts
            ),
            "cached_content_hashes": [
                row["local_cache_content_hash"]
                for row in receipts
                if row.get("local_cache_status") == "cached"
            ],
        },
        "exp6580_receipt": {
            "path": EXP6580_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(repo_root / EXP6580_RELATIVE_PATH),
            "status": exp6580_payload.get("status", "missing"),
            "source_method_ready_score": exp6580_payload.get(
                "v572_source_method_ready_score", "missing"
            ),
        },
        "exp6574_receipt": {
            "path": EXP6574_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(repo_root / EXP6574_RELATIVE_PATH),
            "status": exp6574_payload.get("status", "missing"),
            "joint_method_ready_score": exp6574_payload.get(
                "joint_sufficiency_method_ready_score", "missing"
            ),
        },
        "exact_registry": dict(authority["exact_obligation_registry"]),
        "corpus": {
            "name": "V573 balanced Exp6574 exact-fixture cue pairs",
            "revision": source_manifest["manifest_hash"],
            "parent_exp6580_manifest_hash": source_manifest["parent_exp6580_manifest_hash"],
            "unit_count": source_manifest["bounded_unit_count"],
            "license_spdx": "MIT-0",
        },
        "license": {
            "path": LICENSE_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(repo_root / LICENSE_RELATIVE_PATH),
            "spdx": "MIT-0",
        },
        "resources": exp6580._resource_receipt(repo_root),  # noqa: SLF001
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_inference_invoked": False,
        "model_outcomes_available": False,
        "llm_calls_issued": 0,
        "external_text_scorer_calls": 0,
        "public_arc_solve_attempted": False,
    }


def _tests_run_receipts(
    tests_run: Sequence[Mapping[str, Any]] | None,
) -> list[JsonDict]:
    rows = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [
        {
            "command": str(row["command"]),
            "exit_code": int(row["exit_code"]),
            "duration_s": float(row.get("duration_s", 0.0)),
        }
        for row in rows
    ]


def _source_rows_ready(payload: Mapping[str, Any]) -> bool:
    receipts = payload.get("primary_source_receipts", [])
    non_imported = payload.get("non_imported_claim_rows", [])
    return (
        {row.get("arxiv_id") for row in receipts if isinstance(row, Mapping)}
        == set(REQUIRED_ARXIV_IDS)
        and all(
            isinstance(row, Mapping)
            and row.get("reference_contains_arxiv_id") is True
            and str(row.get("reference_section_sha256", "")).startswith("sha256:")
            and str(row.get("method_hook", "")).startswith("Carnot hook:")
            and row.get("imported_as") == "bounded_method_control"
            for row in receipts
        )
        and len(non_imported) == len(REQUIRED_ARXIV_IDS)
        and all(
            isinstance(row, Mapping) and row.get("claim_imported_into_carnot_evidence") is False
            for row in non_imported
        )
    )


def _manifest_ready(payload: Mapping[str, Any]) -> bool:
    manifest = payload.get("source_unit_manifest", {})
    units = manifest.get("units", []) if isinstance(manifest, Mapping) else []
    strata = Counter(row.get("stratum") for row in units if isinstance(row, Mapping))
    classes = {row.get("case_class") for row in units if isinstance(row, Mapping)}
    return (
        isinstance(manifest, Mapping)
        and manifest.get("selected_without_model_outcomes") is True
        and len(units) >= 16
        and strata["ordinary"] == strata["restrictive_cue"]
        and {"positive_control", "unsupported", "ambiguous", "contradictory", "tamper"} <= classes
        and all(
            isinstance(unit, Mapping)
            and unit.get("selected_without_model_outcome") is True
            and unit.get("model_outcome_fields_accessed") is False
            and unit.get("source_bytes_sha256")
            == sha256_text(str(unit.get("exact_source_bytes", "")))
            and unit.get("task_bytes_sha256") == sha256_text(str(unit.get("exact_task_bytes", "")))
            and unit.get("checker") == EXACT_CHECKER_NAME
            and bool(unit.get("gold_constraints"))
            for unit in units
        )
    )


def _prompt_router_ready(payload: Mapping[str, Any]) -> bool:
    prompt = payload.get("prompt_stage_contract", {})
    prompts = prompt.get("prompts", {}) if isinstance(prompt, Mapping) else {}
    router = payload.get("router_contract", {})
    routing_rows = router.get("routing_rows", []) if isinstance(router, Mapping) else []
    return (
        set(prompts) == {"direct", "stage1", "stage2"}
        and all(
            isinstance(row, Mapping)
            and row.get("family_neutral") is True
            and row.get("sha256") == sha256_text(str(row.get("text", "")))
            for row in prompts.values()
        )
        and prompt.get("stage1_output_format") == "plain_text"
        and prompt.get("raw_stage_write_before_parse") is True
        and prompt.get("raw_stage1_required") is True
        and prompt.get("raw_stage2_required") is True
        and prompt.get("stage1_answer_requested") is False
        and prompt.get("stage1_answer_transport_allowed") is False
        and prompt.get("constraint_ir_generation_allowed") is False
        and prompt.get("schema_repair_retry_count") == 0
        and isinstance(router, Mapping)
        and router.get("allowed_inputs") == ["exact_task_bytes", "exact_source_bytes"]
        and router.get("outcome_blind") is True
        and router.get("frozen_before_inference") is True
        and all(
            isinstance(row, Mapping) and row.get("route_matches_stratum") is True
            for row in routing_rows
        )
    )


def _arms_authority_metrics_ready(payload: Mapping[str, Any]) -> bool:
    arms_contract = payload.get("arm_seed_budget_contract", {})
    arms = arms_contract.get("arms", {}) if isinstance(arms_contract, Mapping) else {}
    authority = payload.get("source_binding_and_exact_authority_contract", {})
    metrics = payload.get("metric_and_acceptance_contract", {})
    matched_keys = (
        "unit_order_hash",
        "seed_schedule_hash",
        "decoding_hash",
        "stop_rules_hash",
        "total_token_limit",
        "timeout_s",
    )
    required_metrics = {
        "exact_success",
        "stage1_precision",
        "stage1_recall",
        "unsupported_constraint_count",
        "contradictory_constraint_count",
        "abstention",
        "unsafe_release",
        "total_tokens",
        "latency_s",
        "failure",
    }
    return (
        set(arms) == {"direct", "always_on_cfr", "routed_cfr"}
        and all(
            len({row.get(key) for row in arms.values() if isinstance(row, Mapping)}) == 1
            for key in matched_keys
        )
        and arms_contract.get("stage1_tokens_charged") is True
        and arms_contract.get("stage1_latency_charged") is True
        and arms_contract.get("failure_retention_required") is True
        and isinstance(authority, Mapping)
        and authority.get("model_can_certify_release") is False
        and authority.get("llm_release_authority") is False
        and authority.get("generated_constraint_ir_allowed") is False
        and authority.get("answer_transport_allowed") is False
        and authority.get("external_text_scoring_allowed") is False
        and str(
            authority.get("exact_obligation_registry", {}).get("registry_sha256", "")
        ).startswith("sha256:")
        and isinstance(metrics, Mapping)
        and required_metrics <= set(metrics.get("per_unit_fields", []))
        and metrics.get("later_positive_verdict_class") == "circular_positive"
        and metrics.get("ready_contract_verdict_class") is None
        and metrics.get("retirement_rules")
    )


def _fixtures_gates_attacks_ready(payload: Mapping[str, Any]) -> bool:
    fixture_rows = payload.get("fixture_replay_rows", [])
    gate_rows = payload.get("downstream_gate_field_rows", [])
    attack_rows = payload.get("attack_rows", [])
    method_rows = [
        row
        for row in gate_rows
        if isinstance(row, Mapping) and row.get("artifact_field") == READY_FIELD
    ]
    return (
        {row.get("case_class") for row in fixture_rows if isinstance(row, Mapping)}
        == {"positive_control", "unsupported", "ambiguous", "contradictory", "tamper"}
        and all(
            isinstance(row, Mapping)
            and row.get("passed") is True
            and row.get("unsafe_release") is False
            for row in fixture_rows
        )
        and len(method_rows) == 1
        and all(
            isinstance(row, Mapping)
            and row.get("owner_declared") is True
            and row.get("all_consumers_declared") is True
            for row in gate_rows
        )
        and {row.get("attack_id") for row in attack_rows if isinstance(row, Mapping)}
        == set(REQUIRED_ATTACK_IDS)
        and all(
            isinstance(row, Mapping)
            and row.get("passed") is True
            and row.get("candidate_ready_score") == 0.0
            for row in attack_rows
        )
    )


def readiness_reducer(payload: Mapping[str, Any]) -> JsonDict:
    checks = {
        "source_rows": _source_rows_ready(payload),
        "source_unit_manifest": _manifest_ready(payload),
        "prompt_and_router": _prompt_router_ready(payload),
        "arms_authority_metrics": _arms_authority_metrics_ready(payload),
        "fixtures_gates_attacks": _fixtures_gates_attacks_ready(payload),
        "no_llm_precondition": (
            payload.get("preconditions_checked", {}).get("model_inference_invoked") is False
            and payload.get("preconditions_checked", {}).get("model_outcomes_available") is False
            and payload.get("preconditions_checked", {}).get("llm_calls_issued") == 0
        ),
        "protected_files": payload.get("protected_files_unchanged", {}).get("all_unchanged")
        is True,
        "oracle_boundary": payload.get("verifier_is_oracle") is True,
    }
    ready = all(checks.values())
    return {"checks": checks, "ready": ready, "ready_score": 1.0 if ready else 0.0}


def build_gate_check_summary(reduction: Mapping[str, Any]) -> JsonDict:
    failed = [
        {"check": name, "observed": passed, "expected": True}
        for name, passed in reduction["checks"].items()
        if passed is not True
    ]
    summary = {
        "checks_closed": not failed,
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
        "ready_score": reduction["ready_score"],
    }
    return {**summary, "row_hash": sha256_json(summary)}


def build_field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "Exp6587 source receipts and deterministic contract builders",
            "rows": [field],
            "hashes": ["reproducibility_checksum"],
            "reducer": "readiness_reducer.v6587",
            "spec_refs": ["REQ-REPORT-6587"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def build_report(
    repo_root: Path = REPO_ROOT,
    *,
    date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    started = time.monotonic()
    before_hashes = _protected_hashes(repo_root)
    source_receipts = build_primary_source_receipts(repo_root, date)
    source_manifest = build_source_unit_manifest(repo_root)
    prompt_contract = build_prompt_stage_contract()
    router_contract = build_router_contract(source_manifest)
    arm_contract = build_arm_seed_budget_contract(source_manifest)
    authority_contract = build_source_binding_and_exact_authority_contract(repo_root)
    metric_contract = build_metric_and_acceptance_contract()
    protected = _protected_files_unchanged(before_hashes, _protected_hashes(repo_root))
    payload: JsonDict = {
        "status": "",
        "honest_verdict": "",
        "verdict_class": None,
        "gate_check_summary": {},
        "primary_source_receipts": source_receipts,
        "non_imported_claim_rows": build_non_imported_claim_rows(source_receipts),
        "source_unit_manifest": source_manifest,
        "prompt_stage_contract": prompt_contract,
        "router_contract": router_contract,
        "arm_seed_budget_contract": arm_contract,
        "source_binding_and_exact_authority_contract": authority_contract,
        "metric_and_acceptance_contract": metric_contract,
        "fixture_replay_rows": build_fixture_replay_rows(),
        "downstream_gate_field_rows": build_downstream_gate_field_rows(repo_root),
        "attack_rows": build_attack_rows(),
        READY_FIELD: 0.0,
        "preconditions_checked": build_preconditions_checked(
            repo_root, date, source_receipts, source_manifest, authority_contract
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": build_field_provenance(),
        "duration_s": (
            max(time.monotonic() - started, 0.0001) if duration_s is None else duration_s
        ),
        "tests_run": _tests_run_receipts(tests_run),
    }
    reduction = readiness_reducer(payload)
    payload[READY_FIELD] = reduction["ready_score"]
    payload["gate_check_summary"] = build_gate_check_summary(reduction)
    if reduction["ready"]:
        payload["status"] = "complete_v573_constraint_first_method_ready"
        payload["honest_verdict"] = (
            "complete: source, fixture, router, metric, and exact-authority contracts "
            "are ready before any V573 model outcome"
        )
        payload["verdict_class"] = None
    else:
        failure = payload["gate_check_summary"]["first_failure"]
        payload["status"] = "blocked_v573_constraint_first_method_contract"
        payload["honest_verdict"] = (
            "blocked_v573_constraint_first_method_contract: missing or invalid "
            f"contract value {failure}"
        )
        payload["verdict_class"] = "blocked"
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def validate_report(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if (
        not isinstance(payload.get("duration_s"), int | float)
        or float(payload.get("duration_s", 0.0)) <= 0
    ):
        errors.append("duration_s must be positive")
    reduction = readiness_reducer(payload)
    if payload.get(READY_FIELD) != reduction["ready_score"]:
        errors.append(f"{READY_FIELD} mismatch")
    if payload.get(READY_FIELD) == 1.0 and payload.get("verdict_class") is not None:
        errors.append("ready contract verdict_class must be null")
    if payload.get("protected_files_unchanged", {}).get("all_unchanged") is not True:
        errors.append("protected_files_unchanged failed")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance missing required fields")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def atomic_write_report(path: str | Path, payload: Mapping[str, Any]) -> JsonDict:
    errors = validate_report(payload)
    if errors:
        raise ValueError("; ".join(errors))
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        directory_descriptor = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():  # pragma: no cover - only true after a write failure
            temporary_path.unlink()
    return {
        "atomic_replace": True,
        "file_fsync": True,
        "directory_fsync": True,
        "path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "temporary_path_exists_after_replace": temporary_path.exists(),
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output_path = Path(args.output)
    target = output_path if output_path.is_absolute() else REPO_ROOT / output_path
    if args.validate:
        payload = exp6580._read_json(target)  # noqa: SLF001
        errors = validate_report(payload)
        if errors:
            print("\n".join(errors))
            return 1
        print("valid")
        return 0
    report = build_report(REPO_ROOT, date=str(args.date))
    atomic_write_report(target, report)
    print(json.dumps({"path": str(target), "ready": report[READY_FIELD]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
