"""Preserve dated board scopes and size an eligible host service boundary.

The task reads historical evidence. It never contacts a board. Host timing can
describe placement only when one measured denominator separates all stages.

Spec: REQ-HW-7599 and SCENARIO-HW-7599-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from carnot import experiment_6559_gatemate_changed_state_continuity as receipt_authority
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260924"
MILESTONE = "2026.09.663"
EXPERIMENT_ID = "exp7599-v663-board-continuity"
SCHEMA = "carnot.exp7599.v663.board_continuity.v1"
RANDOM_SEED = 7_599_001
REPO_ROOT = Path(__file__).resolve().parents[2]

RESULT_PATH = Path("results/experiment_7599_v663_board_continuity.json")
RAW_DIR = Path("results/raw/experiment_7599_v663_board_continuity")
MODULE_PATH = Path("python/carnot/experiment_7599_v663_board_continuity.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7599_v663_board_continuity.py")
TEST_PATH = Path("tests/python/test_experiment_7599_v663_board_continuity.py")
SPEC_PATH = Path("openspec/capabilities/hardware/spec.md")
PRIOR_BOARD_PATH = Path("results/experiment_7314_v642_board_continuity.json")
CONSUMER_PATH = Path("results/experiment_7598_v663_rust_consumer.json")
GATEMATE_SEARCH_NAME = "gatemate_receipt_search.json"

BOARD_EVIDENCE_PATHS: dict[str, Path] = {
    "KV260": Path("results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json"),
    "PolarFire": Path("results/experiment_7231_v636_board_continuity.json"),
    "GateMate": Path("results/experiment_6559_gatemate_changed_state_continuity.json"),
}
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = {
    "model_loads": 0,
    "forward_calls": 0,
    "generation_calls": 0,
    "current_llm_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}
SPEC_REFS = (
    "REQ-HW-7599",
    "SCENARIO-HW-7599-BOARD-SCOPES",
    "SCENARIO-HW-7599-PLACEMENT",
    "SCENARIO-HW-7599-PLACEMENT-UNMEASURED",
)
VALIDATION_NAMES = (
    "focused_pytest",
    "changed_module_coverage",
    "changed_module_coverage_report",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
)
TERMINAL_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
GATEMATE_REQUIRED_ACTION = (
    "operator records a dated GateMate cable, port, power, board, JTAG, or DirtyJTAG "
    "physical change after Exp6559"
)
GATEMATE_MISSING = "no qualifying operator physical-change receipt after Exp6559"


def load_object(path: Path) -> JsonDict:
    """Return one JSON object, or an empty object for unusable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _path_label(path: Path, root: Path) -> str:
    """Use stable worktree labels and retain external test paths exactly."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def source_row(path: Path, root: Path) -> JsonDict:
    """Bind exact bytes and keep the producer's terminal declarations."""

    value = load_object(path)
    return {
        "path": _path_label(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "original_honest_verdict": value.get("honest_verdict"),
        "original_verdict_class": value.get("verdict_class"),
        "original_flagged_adversarial": value.get("flagged_adversarial"),
    }


def _check(
    check: str,
    upstream: str,
    path: str,
    field: str,
    op: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Keep both operands so an absent input cannot look like zero."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "op": op,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
    }


def collect_preconditions(root: Path) -> JsonDict:
    """Read all board dispositions before considering optional placement."""

    repo = root.resolve()
    checks: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    loaded: dict[str, JsonDict] = {}

    required = {"prior board ledger": PRIOR_BOARD_PATH, **BOARD_EVIDENCE_PATHS}
    for name, relative in required.items():
        path = relative if relative.is_absolute() else repo / relative
        readable = path.is_file() and path.stat().st_size > 0
        checks.append(
            _check(
                "dated_board_disposition",
                f"{name} dated disposition",
                str(relative),
                "path",
                "exists",
                "readable nonempty file",
                "readable nonempty file" if readable else "missing",
                readable,
            )
        )
        if readable:
            sources[_path_label(path, repo)] = source_row(path, repo)
            loaded[name] = load_object(path)

    spec_path = repo / SPEC_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    spec_ok = all(reference in spec_text for reference in SPEC_REFS)
    if spec_path.is_file():
        sources[SPEC_PATH.as_posix()] = source_row(spec_path, repo)
    checks.append(
        _check(
            "capability_requirement",
            "hardware capability",
            SPEC_PATH.as_posix(),
            "REQ-HW-7599 and scenarios",
            "contains",
            list(SPEC_REFS),
            [reference for reference in SPEC_REFS if reference in spec_text],
            spec_ok,
        )
    )

    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH):
        path = repo / relative
        if path.is_file():
            sources[relative.as_posix()] = source_row(path, repo)

    consumer_path = repo / CONSUMER_PATH
    if consumer_path.is_file() and consumer_path.stat().st_size > 0:
        sources[CONSUMER_PATH.as_posix()] = source_row(consumer_path, repo)
        consumer = load_object(consumer_path)
    else:
        consumer = {}
    checks.append(
        _check(
            "optional_exp7598_presence",
            "Exp7598 public client",
            CONSUMER_PATH.as_posix(),
            "path",
            "optional",
            "present or explicit absent",
            "present" if consumer else "absent",
            True,
        )
    )

    mandatory = [row for row in checks if row["check"] != "optional_exp7598_presence"]
    blocker = next((deepcopy(row) for row in mandatory if row["passed"] is not True), None)
    return {
        "checks": checks,
        "blocker": blocker,
        "sources": sources,
        "loaded": loaded,
        "consumer": consumer,
        "resource_ownership": {
            "root": str(repo),
            "result_parent_writable": (repo / RESULT_PATH).parent.is_dir(),
            "raw_parent_creatable": (repo / RAW_DIR).parent.is_dir(),
        },
    }


def search_gatemate_receipt(
    root: Path,
    raw_path: Path,
    *,
    receipt_candidates: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Search approved prose, or explicit private fixtures, without board I/O."""

    rows = (
        receipt_authority.search_dated_receipts(root, "20260823")
        if receipt_candidates is None
        else receipt_authority.rows_from_overrides(receipt_candidates, "20260823")
    )
    selected = receipt_authority.select_physical_state_receipt(rows)
    accepted = [row for row in rows if row.get("valid") is True]
    payload = {
        "schema": "carnot.exp7599.gatemate_receipt_search.v1",
        "run_date": RUN_DATE,
        "cutoff_experiment": "Exp6559",
        "cutoff_date": "20260823",
        "approved_sources": [
            path.as_posix() for path in receipt_authority.APPROVED_RECEIPT_LOCATIONS
        ],
        "candidate_count": len(rows),
        "accepted_receipt_count": len(accepted),
        "selected_receipt": selected,
        "hardware_operations_issued": [],
    }
    atomic_json(raw_path, payload)
    return {
        **selected,
        "accepted_receipt_count": len(accepted),
        "search_path": str(raw_path),
        "search_sha256": sha256_file(raw_path),
        "required_operator_action": GATEMATE_REQUIRED_ACTION,
    }


def _prior_rows(prior: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Index exactly one prior row per named board."""

    rows = prior.get("board_rows")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("board")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("board") in BOARD_EVIDENCE_PATHS
    }


def build_board_rows(
    root: Path,
    prior: Mapping[str, Any],
    physical: Mapping[str, Any],
) -> list[JsonDict]:
    """Reduce three distinct venues without claiming present reachability."""

    indexed = _prior_rows(prior)
    if set(indexed) != set(BOARD_EVIDENCE_PATHS):
        return []
    rows: list[JsonDict] = []
    for board in ("KV260", "PolarFire", "GateMate"):
        source = indexed[board]
        evidence_path = BOARD_EVIDENCE_PATHS[board]
        resolved = evidence_path if evidence_path.is_absolute() else root / evidence_path
        common: JsonDict = {
            "unit_id": f"board:{board}",
            "row_type": "board_continuity",
            "board": board,
            "arm": "historical_scope",
            "seed": None,
            "direction": "documentation_only",
            "missing": False,
            "censored": False,
            "source_path": f"{PRIOR_BOARD_PATH.as_posix()}#board={board}",
            "source_sha256": canonical_hash(source),
            "source_artifact_sha256": sha256_file(root / PRIOR_BOARD_PATH),
            "evidence_date": source.get("latest_receipt_date") or prior.get("run_date"),
            "evidence_artifact_path": str(evidence_path),
            "evidence_artifact_sha256": sha256_file(resolved),
            "current_reachability": "unknown_not_probed",
            "historical_hash_establishes_current_reachability": False,
            "hardware_operations_issued": [],
            "numerator": 1 if source.get("terminal_criterion_met") is True else 0,
            "denominator": 1,
            "metric": bool(source.get("terminal_criterion_met")),
            "provenance": "authenticated historical disposition; no current board command",
        }
        if board == "KV260":
            common.update(
                {
                    "disposition": "graduated_fpga_fabric_scope_preserved",
                    "processor_class": "fpga_fabric",
                    "execution_venue": "kv260_fpga_fabric_historical",
                    "claim_scope": (
                        "historical programmable-logic latency transcript and synthesis only"
                    ),
                    "future_access": "ssh kria",
                    "host_block_device_precondition_allowed": False,
                    "k_max": 5,
                    "exact_next_condition": "none; future access is separately scoped",
                    "abstention": False,
                    "error": None,
                }
            )
        elif board == "PolarFire":
            common.update(
                {
                    "disposition": "graduated_linux_cpu_dispatch_scope_preserved",
                    "processor_class": "linux_cpu",
                    "execution_venue": "polarfire_linux_cpu_historical",
                    "claim_scope": "historical hash-matched Linux CPU dispatch only",
                    "fpga_sampling_measured": False,
                    "exact_next_condition": "FPGA sampling needs a separate measured task",
                    "abstention": False,
                    "error": None,
                }
            )
        else:
            changed = physical.get("exists") is True
            common.update(
                {
                    "disposition": (
                        "changed_physical_state_separate_task_eligible"
                        if changed
                        else "blocked_unchanged_physical_prerequisite"
                    ),
                    "processor_class": "not_executed",
                    "execution_venue": "none_read_only",
                    "claim_scope": "physical and JTAG scope remains unexecuted",
                    "terminal_criterion_met": False,
                    "metric": False,
                    "numerator": 0,
                    "operator_receipt": deepcopy(dict(physical)) if changed else None,
                    "exact_missing_receipt": None if changed else GATEMATE_MISSING,
                    "exact_next_condition": (
                        "separately reviewed GateMate execution task"
                        if changed
                        else GATEMATE_REQUIRED_ACTION
                    ),
                    "current_hardware_execution_authorized": False,
                    "last_diagnostic": {
                        "scope": "v477 physical/JTAG block",
                        "observed": "0xffffffff",
                        "meaning": "all-ones TDO; open or unpowered target remained possible",
                        "source": "results/experiment_6559_gatemate_changed_state_continuity.json",
                    },
                    "abstention": not changed,
                    "error": None if changed else GATEMATE_MISSING,
                }
            )
        rows.append(common)
    return rows


def _placement_failure(field: str, observed: str) -> JsonDict:
    """Return an explicit unmeasured placement instead of a residual guess."""

    return {
        "placement_scope": "placement_unmeasured",
        "fractions": {},
        "amdahl_upper_bound": None,
        "formula": "whole_client / (whole_client - arithmetic)",
        "projected_latency_is_measurement": False,
        "failed_check": _check(
            "eligible_public_client_stage_timings",
            "Exp7598",
            CONSUMER_PATH.as_posix(),
            field,
            "contains",
            "measured whole-client arithmetic, IPC, persistence, and other times",
            observed,
            False,
        ),
    }


def reduce_placement(consumer: Mapping[str, Any]) -> JsonDict:
    """Compute stage shares only from one eligible measured denominator."""

    if not consumer:
        return _placement_failure("path", "Exp7598 artifact absent")
    if consumer.get("flagged_adversarial") is True:
        return _placement_failure("flagged_adversarial", "true")
    if consumer.get("consumer_ready_score") != 1:
        return _placement_failure("consumer_ready_score", str(consumer.get("consumer_ready_score")))
    timings = consumer.get("public_client_stage_timings")
    if not isinstance(timings, Mapping):
        return _placement_failure(
            "public_client_stage_timings",
            "missing separate measured IPC and persistence fields",
        )
    names = ("whole_client", "arithmetic", "ipc", "persistence", "other")
    values = {name: timings.get(name) for name in names}
    numeric = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
        for value in values.values()
    )
    scope_ok = timings.get("scope") == "whole_public_client"
    measured = timings.get("measured") is True
    whole = float(values["whole_client"]) if numeric else 0.0
    stages = sum(float(values[name]) for name in names[1:]) if numeric else 0.0
    arithmetic = float(values["arithmetic"]) if numeric else 0.0
    total_ok = whole > 0.0 and arithmetic < whole and math.isclose(stages, whole)
    if not (numeric and scope_ok and measured and total_ok):
        return _placement_failure(
            "public_client_stage_timings",
            f"ineligible measured={measured} scope={timings.get('scope')} values={values}",
        )
    fractions = {name: float(values[name]) / whole for name in names[1:]}
    return {
        "placement_scope": "whole_public_client_measured_decomposition",
        "fractions": fractions,
        "stage_timings": {**values, "unit": timings.get("unit")},
        "amdahl_upper_bound": whole / (whole - arithmetic),
        "formula": "whole_client / (whole_client - arithmetic)",
        "projected_latency_is_measurement": False,
        "failed_check": None,
    }


def acquisition_decision() -> JsonDict:
    """State evidence prerequisites without authorizing a purchase."""

    return {
        "purchase_authorized": False,
        "new_accelerator_justified_by_small_host_update": False,
        "decision_note": (
            "No new accelerator purchase is justified solely by this small host update."
        ),
        "extropic_thrml": {
            "status": "compatibility_or_future_access",
            "required_evidence": "authenticated device access and a concrete workload",
        },
        "xdna": {
            "status": "deferred",
            "required_evidence": "concrete workload plus an installed and qualified execution provider",
        },
        "larger_fpga": {
            "status": "deferred",
            "required_evidence": "concrete workload, tool access, and transfer-orchestration timing",
        },
        "literature_comparison": {
            "fpga_orchestration": {
                "source": "arXiv:2602.15985v2",
                "project_index": "docs/research-notes/v661-method-map.md",
                "finding": "transfer and control costs need whole-client measurement",
            },
            "local_kan_update": {
                "source": "arXiv:2602.02056v4",
                "project_index": "docs/research-notes/v663-method-map.md",
                "finding": "small local arithmetic does not establish accelerator value",
            },
        },
        "thermodynamic_sampler_required": False,
        "reason": "exact binary normalization is deterministic arithmetic",
    }


FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": "Use a complete_ prefix; completed aggregation does not prove benefit.",
    "verdict_class": "Use one closed class; only unfinished owned work is partial.",
    "flagged_adversarial": "Persist the terminal reader outcome; flagged evidence opens no gate.",
    "gate_check_summary": "A failed check names upstream, path, field, operator, expected, and observed.",
    "acceptance_gate_results": "Keep validity, readiness, benefit, retention, and freshness separate.",
    "rows": "Keep one auditable row for each board and venue.",
    "sample_size_budget": "Count boards as units; seeds and windows do not multiply them.",
    "inference_substrate": "State that current work is host aggregation, not inherited board execution.",
    "inference_substrate_class": "Keep actual aggregation separate from planned hardware classes.",
    "MODEL_SPECS": "No current LLM call means an empty model list.",
    "invocation_counts": "Count current loads, forwards, generations, and tokens independently.",
    "duration_s": "Measure current monotonic work without padding or inherited time.",
    "random_seed": "Record the fixed seed even though current reduction is deterministic.",
    "reproducibility_checksum": "Bind immutable evidence, configuration, and terminal reduction.",
    "source_artifact_hashes": "Separate authenticated producer bytes from missing evidence.",
    "validation_receipts": "Bind command, worktree, exit code, and raw log hash.",
    "verifier_is_oracle": "Exact historical labels cannot support an oracle-distinct benefit claim.",
    "field_principles": "Carry each reporting principle in the emitted artifact.",
    "board_continuity_complete_score": (
        "One requires three authenticated dated scopes and explicit unresolved prerequisites."
    ),
    "board_rows": (
        "KV260 FPGA history, PolarFire CPU history, and GateMate physical prerequisites remain separate."
    ),
    "hardware_operations_issued": (
        "An empty list proves this task made no present board execution claim."
    ),
    "placement_scope": (
        "Use whole-client measured decomposition when eligible; otherwise report unmeasured."
    ),
    "amdahl_upper_bound": (
        "Compute a bound from eligible measured stage fractions; never report measured speedup."
    ),
}


def _acceptance_gates(
    board_score: int, placement: Mapping[str, Any], blocked: bool
) -> list[JsonDict]:
    """Keep evidence validity distinct from readiness and benefit."""

    values = (
        ("validity", "dated_sources_authenticated", True, not blocked),
        ("readiness", "board_continuity_complete", 1, board_score),
        ("benefit", "new_hardware_benefit_measured", 1, 0),
        ("retention", "zero_current_hardware_operations", 0, 0),
        ("freshness", "historical_hash_is_current_reachability", False, False),
    )
    return [
        {
            "category": category,
            "check": check,
            "upstream": "current Exp7599 reduction",
            "path": "board_rows" if category != "benefit" else "placement_scope",
            "field": check,
            "op": "eq",
            "expected": expected,
            "observed": observed,
            "passed": observed == expected,
            "principle": FIELD_PRINCIPLES["acceptance_gate_results"],
        }
        for category, check, expected, observed in values
    ]


def _gate_summary(
    blocker: Mapping[str, Any] | None,
    rows: Sequence[Mapping[str, Any]],
    physical: Mapping[str, Any] | None,
    placement: Mapping[str, Any],
) -> list[JsonDict]:
    """List each failed external or optional input without merging scopes."""

    if blocker is not None:
        return [deepcopy(dict(blocker))]
    summary: list[JsonDict] = []
    gate = next((row for row in rows if row.get("board") == "GateMate"), None)
    if gate and gate.get("disposition") == "blocked_unchanged_physical_prerequisite":
        summary.append(
            _check(
                "gatemate_receipt",
                "operator physical-change receipt",
                str((physical or {}).get("search_path")),
                "accepted_receipt_count",
                "gt",
                0,
                (physical or {}).get("accepted_receipt_count", 0),
                False,
            )
        )
    failed = placement.get("failed_check")
    if isinstance(failed, Mapping):
        summary.append(deepcopy(dict(failed)))
    return summary


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable evidence and reduction while excluding clocks and receipts."""

    keys = (
        "schema",
        "run_date",
        "milestone",
        "random_seed",
        "source_artifact_hashes",
        "board_rows",
        "placement_scope",
        "placement_fractions",
        "amdahl_upper_bound",
        "acquisition_decision",
        "hardware_operations_issued",
        "independent_reduction",
    )
    return canonical_hash({key: deepcopy(value.get(key)) for key in keys})


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute comparative counts and placement only from emitted rows."""

    rows = value.get("board_rows")
    board_rows = [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
    fractions = value.get("placement_fractions")
    return {
        "board_count": len(board_rows),
        "authenticated_board_count": sum(
            bool(row.get("source_sha256")) and bool(row.get("evidence_artifact_sha256"))
            for row in board_rows
        ),
        "blocked_board_count": sum(
            str(row.get("disposition", "")).startswith("blocked_") for row in board_rows
        ),
        "hardware_operation_count": len(value.get("hardware_operations_issued") or []),
        "placement_scope": value.get("placement_scope"),
        "fractions": deepcopy(dict(fractions)) if isinstance(fractions, Mapping) else {},
        "amdahl_upper_bound": value.get("amdahl_upper_bound"),
    }


def _provisional_receipts() -> list[JsonDict]:
    """Give pure tests complete receipt names without claiming real subprocesses."""

    return [
        {
            "name": name,
            "command": f"provisional {name}",
            "command_argv": ["provisional", name],
            "scope": "private_test_candidate",
            "worktree": str(REPO_ROOT),
            "exit_code": 0,
            "duration_s": 0.0,
            "log_path": "private_test_candidate",
            "log_sha256": "sha256:" + "0" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in (*VALIDATION_NAMES, *TERMINAL_NAMES)
    ]


def _principles(keys: Sequence[str]) -> dict[str, str]:
    """Carry a short reason for every top-level artifact field."""

    return {
        key: FIELD_PRINCIPLES.get(
            key, f"Retain {key} so this terminal scope cannot disappear silently."
        )
        for key in keys
    }


def build_artifact(
    root: Path,
    raw_dir: Path,
    *,
    consumer_override: Mapping[str, Any] | None = None,
    receipt_candidates: Sequence[Mapping[str, Any]] | None = None,
    validation_receipts: Sequence[Mapping[str, Any]] | None = None,
    duration_s: float = 0.0,
    phase_spans: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build one complete terminal artifact without any hardware operation."""

    repo = root.resolve()
    context = collect_preconditions(repo)
    blocker = context["blocker"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    physical: JsonDict | None = None
    board_rows: list[JsonDict] = []
    if blocker is None:
        physical = search_gatemate_receipt(
            repo,
            raw_dir / GATEMATE_SEARCH_NAME,
            receipt_candidates=receipt_candidates,
        )
        board_rows = build_board_rows(
            repo,
            context["loaded"]["prior board ledger"],
            physical,
        )

    consumer = (
        deepcopy(dict(consumer_override))
        if consumer_override is not None
        else deepcopy(context["consumer"])
    )
    placement = reduce_placement(consumer)
    sources = deepcopy(context["sources"])
    if not consumer:
        sources[CONSUMER_PATH.as_posix()] = {
            "path": CONSUMER_PATH.as_posix(),
            "sha256": None,
            "bytes": 0,
            "missing_producer": True,
            "original_honest_verdict": None,
            "original_verdict_class": None,
            "original_flagged_adversarial": None,
        }
    if physical is not None:
        search_path = Path(str(physical["search_path"]))
        sources[_path_label(search_path, repo)] = source_row(search_path, repo)

    board_score = int(
        blocker is None
        and len(board_rows) == 3
        and all(
            row.get("source_sha256") and row.get("evidence_artifact_sha256") for row in board_rows
        )
        and any(row.get("board") == "GateMate" for row in board_rows)
    )
    gates = _acceptance_gates(board_score, placement, blocker is not None)
    gate_summary = _gate_summary(blocker, board_rows, physical, placement)
    receipts = [deepcopy(dict(row)) for row in (validation_receipts or _provisional_receipts())]
    receipt_outcomes = {
        str(row.get("name")): {
            "passed": row.get("passed") is True,
            "exit_code": row.get("exit_code"),
            "log_sha256": row.get("log_sha256"),
        }
        for row in receipts
        if row.get("name") in TERMINAL_NAMES
    }

    if blocker is not None:
        verdict = f"complete_blocked_{str(blocker['check']).replace(' ', '_')}"
        verdict_class = "blocked"
    elif placement["placement_scope"] == "placement_unmeasured":
        verdict = "complete_null_board_continuity_placement_unmeasured"
        verdict_class = "null"
    else:
        verdict = "complete_null_board_continuity_host_placement_only"
        verdict_class = "null"

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "worktree": str(repo),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": RANDOM_SEED,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_identity": {
            "current_identity": [],
            "scope": "historical source declarations only; not current inference",
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "preconditions_checked": deepcopy(context["checks"]),
        "resource_ownership": deepcopy(context["resource_ownership"]),
        "source_artifact_hashes": sources,
        "rows": deepcopy(board_rows),
        "board_rows": deepcopy(board_rows),
        "sample_size_budget": {
            "intended_independent_units": 3,
            "observed_independent_units": len(board_rows),
            "excluded_independent_units": 0 if blocker is None else 3,
            "censored_independent_units": 0,
            "seeds": [],
            "windows_multiply_source_groups": False,
        },
        "board_continuity_complete_score": board_score,
        "hardware_operations_issued": [],
        "hardware_command_count": 0,
        "physical_state_receipt": deepcopy(physical),
        "placement_scope": placement["placement_scope"],
        "placement_fractions": deepcopy(placement["fractions"]),
        "placement_evidence": deepcopy(placement),
        "amdahl_upper_bound": placement["amdahl_upper_bound"],
        "amdahl_upper_bound_is_measured_speedup": False,
        "host_speed_reported_as_board_or_tsu_speed": False,
        "acquisition_decision": acquisition_decision(),
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_summary,
        "independent_reduction": {},
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": receipts,
        "terminal_reader_outcomes": receipt_outcomes,
        "retire_if_same_verdict": True,
        "prior_verdict_retirement": {
            "prior_experiment": "exp7571-portable-calibration",
            "prior_verdict": "complete_blocked_exp7561_recalibration_ready_score",
            "literal_verdict_repeated": False,
            "disposition": "not_retired; board continuity remains independent",
        },
    }
    artifact["independent_reduction"] = independent_reduce(artifact)
    artifact["field_principles"] = _principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_test_artifact(
    root: Path,
    scratch: Path,
    *,
    receipt_candidates: Sequence[Mapping[str, Any]] | None = (),
) -> JsonDict:
    """Build deterministic private evidence for unit and mutation tests."""

    return build_artifact(
        root,
        scratch / "raw",
        receipt_candidates=receipt_candidates,
        validation_receipts=_provisional_receipts(),
    )


def _verify_sources(value: Mapping[str, Any], root: Path) -> list[str]:
    """Reject any changed authenticated source byte."""

    errors: list[str] = []
    rows = value.get("source_artifact_hashes")
    if not isinstance(rows, Mapping):
        return ["source_artifact_hashes"]
    for key, receipt in rows.items():
        if not isinstance(receipt, Mapping):
            errors.append(f"source_receipt:{key}")
            continue
        expected = receipt.get("sha256")
        if expected is None and receipt.get("missing_producer") is True:
            continue
        raw_path = Path(str(receipt.get("path", key)))
        path = raw_path if raw_path.is_absolute() else root / raw_path
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"source_hash:{key}")
    return errors


def validate_artifact(value: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Reject custody drift, claim broadening, or inconsistent reduction."""

    errors: list[str] = []
    required = {
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "gate_check_summary",
        "acceptance_gate_results",
        "rows",
        "sample_size_budget",
        "inference_substrate",
        "inference_substrate_class",
        "MODEL_SPECS",
        "invocation_counts",
        "duration_s",
        "random_seed",
        "reproducibility_checksum",
        "source_artifact_hashes",
        "validation_receipts",
        "verifier_is_oracle",
        "field_principles",
        "board_continuity_complete_score",
        "board_rows",
        "hardware_operations_issued",
        "placement_scope",
        "amdahl_upper_bound",
    }
    missing = sorted(required.difference(value))
    if missing:
        errors.append("required_fields:" + ",".join(missing))
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_declaration")
    counts = value.get("invocation_counts")
    if not isinstance(counts, Mapping) or any(counts.values()):
        errors.append("invocation_counts")
    if value.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("inference_substrate")
    if value.get("inference_substrate_class") != "aggregation":
        errors.append("inference_substrate_class")
    if value.get("hardware_operations_issued") != [] or value.get("hardware_command_count") != 0:
        errors.append("hardware_operations")
    if value.get("host_speed_reported_as_board_or_tsu_speed") is not False:
        errors.append("host_speed_claim")
    if value.get("rows") != value.get("board_rows"):
        errors.append("rows")

    reduction = independent_reduce(value)
    if value.get("independent_reduction") != reduction:
        errors.append("placement_reduction")
    blocked = value.get("verdict_class") == "blocked"
    expected_score = (
        0
        if blocked
        else int(reduction["board_count"] == 3 and reduction["authenticated_board_count"] == 3)
    )
    if value.get("board_continuity_complete_score") != expected_score:
        errors.append("board_continuity_score")
    if not blocked:
        rows = value.get("board_rows")
        by_board = (
            {row.get("board"): row for row in rows if isinstance(row, Mapping)}
            if isinstance(rows, list)
            else {}
        )
        if set(by_board) != set(BOARD_EVIDENCE_PATHS):
            errors.append("board_identity")
        else:
            if by_board["KV260"].get("processor_class") != "fpga_fabric":
                errors.append("kv260_scope")
            if by_board["KV260"].get("future_access") != "ssh kria":
                errors.append("kv260_access")
            if by_board["KV260"].get("k_max", 6) > 5:
                errors.append("kv260_k_max")
            if by_board["PolarFire"].get("processor_class") != "linux_cpu":
                errors.append("polarfire_scope")
            if by_board["PolarFire"].get("fpga_sampling_measured") is not False:
                errors.append("polarfire_fpga_claim")
            if by_board["GateMate"].get("current_hardware_execution_authorized") is not False:
                errors.append("gatemate_authorization")

    receipt_names = {
        row.get("name")
        for row in value.get("validation_receipts") or []
        if isinstance(row, Mapping) and row.get("passed") is True and row.get("exit_code") == 0
    }
    if not set((*VALIDATION_NAMES, *TERMINAL_NAMES)).issubset(receipt_names):
        errors.append("validation_receipts")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or any(key not in principles for key in value):
        errors.append("field_principles")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum")
    if root is not None:
        errors.extend(_verify_sources(value, root.resolve()))
    return sorted(set(errors))


def atomic_publish(path: Path, value: Mapping[str, Any], *, root: Path) -> None:
    """Write only complete bytes that pass the strict local reader."""

    errors = validate_artifact(value, root=root)
    if errors:
        raise ValueError("invalid Exp7599 artifact: " + ",".join(errors))
    atomic_json(path, dict(value))


def cold_replay(path: Path, *, root: Path) -> list[str]:
    """Load the candidate in a fresh process and verify source custody."""

    value = load_object(path)
    return validate_artifact(value, root=root) if value else ["artifact_unreadable"]


def independent_replay(path: Path) -> list[str]:
    """Recompute terminal counts without trusting producer summaries."""

    value = load_object(path)
    if not value:
        return ["artifact_unreadable"]
    return (
        []
        if value.get("independent_reduction") == independent_reduce(value)
        else ["independent_reduction"]
    )


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze exact tests and static checks with private pytest state."""

    basetemp = private_root / "basetemp"
    coverage_file = private_root / "coverage" / ".coverage"
    basetemp.mkdir(parents=True, exist_ok=True)
    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    commands = validation_scope.build_scoped_commands(
        root,
        (TEST_PATH.as_posix(),),
        (MODULE_PATH.as_posix(),),
        static_paths=(WRAPPER_PATH.as_posix(),),
        basetemp=basetemp,
        coverage_file=coverage_file,
    )
    return [command for command in commands if command.name != "worktree_imports"]


def private_basetemps(commands: Sequence[validation_scope.CommandSpec]) -> list[str]:
    """Create every private pytest parent before its subprocess starts."""

    paths = [
        argument.split("=", 1)[1]
        for command in commands
        for argument in command.argv
        if argument.startswith("--basetemp=")
    ]
    for value in paths:
        Path(value).parent.mkdir(parents=True, exist_ok=True)
    return paths


def terminal_commands(candidate: Path, root: Path) -> list[validation_scope.CommandSpec]:
    """Build four exact readers for one immutable candidate path."""

    common = ("--root", str(root.resolve()))
    python = str(root / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            TERMINAL_NAMES[0],
            (python, "-u", WRAPPER_PATH.as_posix(), *common, "--cold-replay", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            TERMINAL_NAMES[1],
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--independent-reduce",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            TERMINAL_NAMES[2],
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            TERMINAL_NAMES[3],
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
    ]


def _run_commands(
    root: Path,
    commands: Sequence[validation_scope.CommandSpec],
    log_dir: Path,
) -> list[JsonDict]:  # pragma: no cover - exercised by the declared entrypoint.
    """Run bounded owned children with a heartbeat for each unit."""

    receipts: list[JsonDict] = []
    for index, command in enumerate(commands):
        private_basetemps([command])
        receipts.extend(
            validation_scope.run_commands(
                root,
                [command],
                log_dir=log_dir / f"{index:02d}_{command.name}",
                heartbeat_s=60.0,
            )
        )
        receipts[-1]["worktree"] = str(root.resolve())
    return receipts


def progress(  # pragma: no cover - production heartbeat.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Emit one flushed monotonic phase boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7599] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - production timing receipt.
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:
    """Record one real monotonic phase span."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_offset_s": phase_started - run_started,
        "ended_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def run_experiment(  # pragma: no cover - the declared entrypoint is the task E2E.
    root: Path,
    run_date: str,
    *,
    output_path: Path | None = None,
) -> JsonDict:
    """Aggregate, validate, and publish one exact read-only receipt."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    repo = root.resolve()
    destination = output_path or repo / RESULT_PATH
    raw_dir = repo / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7599-", dir="/tmp"))
    candidate = private_root / "candidate.json"
    exact_candidate = private_root / "exact-candidate.json"
    started = time.monotonic()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    preview = build_artifact(repo, raw_dir)
    spans.append(
        _span("preconditions", phase_started, started, len(preview["preconditions_checked"]))
    )
    progress(
        started,
        "preconditions",
        "complete",
        blocked=preview["verdict_class"] == "blocked",
    )

    for phase in ("model_load", "generation", "benchmark"):
        progress(started, phase, "before", planned=0)
        phase_started = time.monotonic()
        spans.append(_span(phase, phase_started, started, 0))
        progress(started, phase, "after", completed=0)

    commands = build_validation_commands(repo, private_root / "validation")
    progress(started, "scoped_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = _run_commands(repo, commands, private_root / "logs" / "affected")
    spans.append(_span("scoped_validation", phase_started, started, len(affected)))
    progress(
        started,
        "scoped_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=all(row.get("passed") is True for row in affected),
    )
    if not all(row.get("passed") is True for row in affected):
        raise RuntimeError("required_scoped_validation_failed")

    terminal_placeholders = [
        row for row in _provisional_receipts() if row["name"] in TERMINAL_NAMES
    ]
    provisional = build_artifact(
        repo,
        raw_dir,
        validation_receipts=[*affected, *terminal_placeholders],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    progress(started, "candidate", "before_atomic_write", path=candidate)
    atomic_json(candidate, provisional)
    progress(started, "candidate", "after_atomic_write", bytes=candidate.stat().st_size)

    commands = terminal_commands(candidate, repo)
    progress(started, "terminal_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    terminal = _run_commands(repo, commands, private_root / "logs" / "terminal")
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=all(row.get("passed") is True for row in terminal),
    )
    if not all(row.get("passed") is True for row in terminal):
        raise RuntimeError("terminal_candidate_validation_failed")

    final = build_artifact(
        repo,
        raw_dir,
        validation_receipts=[*affected, *terminal],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    errors = validate_artifact(final, root=repo)
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    atomic_json(exact_candidate, final)

    commands = terminal_commands(exact_candidate, repo)
    progress(started, "exact_terminal_validation", "before_subprocesses", planned=len(commands))
    exact = _run_commands(repo, commands, private_root / "logs" / "exact")
    progress(
        started,
        "exact_terminal_validation",
        "after_subprocesses",
        completed=len(exact),
        passed=all(row.get("passed") is True for row in exact),
    )
    if not all(row.get("passed") is True for row in exact):
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    atomic_json(raw_dir / "exact_terminal_validation_receipts.json", {"receipts": exact})

    progress(started, "publish", "before_atomic_write", path=destination)
    atomic_publish(destination, final, root=repo)
    if sha256_file(destination) != sha256_file(exact_candidate):
        raise RuntimeError("published_bytes_differ_from_exact_candidate")
    progress(
        started,
        "publish",
        "after_atomic_write",
        bytes=destination.stat().st_size,
        verdict=final["honest_verdict"],
    )
    shutil.rmtree(private_root)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse producer and strict read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the producer or one strict reader."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay, root=root)
        print(
            json.dumps({"validation_passed": not errors, "errors": errors}, sort_keys=True),
            flush=True,
        )
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = independent_replay(args.independent_reduce)
        print(
            json.dumps({"validation_passed": not errors, "errors": errors}, sort_keys=True),
            flush=True,
        )
        return int(bool(errors))
    if args.date != RUN_DATE:  # pragma: no cover - CLI misuse guard.
        print(f"run_date_must_equal_{RUN_DATE}", flush=True)
        return 2
    progress(time.monotonic(), "startup", "resolved_root", root=root)
    run_experiment(root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
