"""Produce the V640 host-only board-state continuation.

This task reads receipts and does not contact hardware. It advances provenance
from the V639 receipt while the shipped reader and validator retain authority
over board evidence.

Spec: REQ-ISING-7286 and SCENARIO-ISING-7286-ARTIFACT.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, cast

import yaml

from carnot import experiment_7272_v639_board_state as current


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260913"
EXPERIMENT_ID = 7286
TASK_ID = "exp7286-board-state"
MILESTONE = "2026.09.640"
SCHEMA = "carnot.exp7286.v640.board_state.v1"
RANDOM_SEED = 7286

RESULT_PATH = Path("results/experiment_7286_v640_board_state.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7286_v640_board_state.json")
RAW_DIR = Path("results/raw/experiment_7286")
OPERATOR_SEARCH_PATH = RAW_DIR / "gatemate_operator_receipt_search.json"
HISTORICAL_MODELS_PATH = RAW_DIR / "historical_model_receipts.json"
NEGATIVE_FIXTURES_PATH = RAW_DIR / "negative_fixture_receipts.json"
VALIDATION_DIR = RAW_DIR / "validation"
ENTRYPOINT_PATH = Path("scripts/experiments/experiment_7286_v640_board_state.py")
MODULE_PATH = Path("python/carnot/experiment_7286_v640_board_state.py")
TEST_PATH = Path("tests/python/test_experiment_7286_v640_board_state.py")
SPEC_PATH = current.SPEC_PATH
ROADMAP_PATH = current.ROADMAP_PATH
EXCLUSION_PATH = current.EXCLUSION_PATH
UPSTREAM_PATH = current.RESULT_PATH
GATEMATE_CUTOFF_PATH = current.GATEMATE_CUTOFF_PATH

KV260_TERMINAL_CRITERION = current.KV260_TERMINAL_CRITERION
POLARFIRE_TERMINAL_CRITERION = current.POLARFIRE_TERMINAL_CRITERION
GATEMATE_CUTOFF_DATE = current.GATEMATE_CUTOFF_DATE
MISSING_RECEIPT = current.MISSING_RECEIPT
GATEMATE_OPERATOR_ACTION = current.GATEMATE_OPERATOR_ACTION
GATEMATE_FUTURE_ACTION = current.GATEMATE_FUTURE_ACTION

EXPECTED_TASK_CONTRACT: JsonDict = {
    "id": TASK_ID,
    "title": "GateMate changed-state review and graduated-board continuity",
    "track": "hardware",
    "priority": "high",
    "requires_gpu": False,
    "max_turns": 20,
    "estimated_wall_time_min": 20,
    "per_unit_rows": True,
    "milestone": MILESTONE,
    "deliverable": RESULT_PATH.as_posix(),
    "prior_failure_ids": ["exp6559-gatemate-changed-state-continuity"],
    "prompt_sha256": "sha256:6322a824a1b141bc8925fce9d4224e595150b4dd6486ef0486eb7da3a375d2c2",
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    MODULE_PATH,
    ENTRYPOINT_PATH,
    TEST_PATH,
    UPSTREAM_PATH,
    GATEMATE_CUTOFF_PATH,
    Path("research-hardware-wishlist.md"),
    Path("ops/known-issues.md"),
    SPEC_PATH,
    ROADMAP_PATH,
)

FIELD_PRINCIPLES = {
    **current.FIELD_PRINCIPLES,
    "schema": "Version the result and retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind this receipt to Exp7286.",
    "task_id": "Bind this receipt to the exact V640 roadmap task.",
    "milestone": "Bind this receipt to milestone 2026.09.640.",
    "spec_refs": "Connect the artifact and tests to REQ-ISING-7286 scenarios.",
    "status": "Use complete or blocked for terminal evidence; keep unfinished work in separate checkpoints.",
    "run_date": "Use 20260913 and actual UTC start and end times.",
    "field_principles": "Store explanations here; consumer values remain ordinary top-level fields.",
    "preconditions_checked": "Record actual input hashes, authority separation, resource ownership, and failures.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical identities in hashed sidecars.",
    "model_invoked": "Derive from actual calls, including failed or unusable generation.",
    "invocation_counts": "Separate attempted and completed loads and generation from usable answers.",
    "inference_substrate": "Use the recognized literal for actual computation, not an invented task label.",
    "inference_substrate_class": "Use the correct no-LLM class and never pad duration.",
    "execution_venue": "Host orchestration is host; identify real device execution separately.",
    "duration_s": "Measure monotonic elapsed time and disjoint phase spans.",
    "random_seed": "Freeze independent-unit seeds before observing results.",
    "reproducibility_checksum": "Bind code, configuration, input manifests, and raw evidence.",
    "source_artifact_hashes": "Preserve exact input identity, retirement, and quarantine status.",
    "rows": "Keep each unit, arm, seed, error, abstention, cost, metric, and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, and censored units and the stopping rule.",
    "acceptance_gate_results": "Each criterion records expected, observed, passed, and principle; separate completeness and value.",
    "gate_check_summary": "For blocked_* name upstream, exact field or check, observed value, and expected value.",
    "verifier_is_oracle": "Expose shared verifier or evaluator authority; exact conformance is not learned correctness.",
    "honest_verdict": "Completed findings start complete_ or complete:; external absence starts blocked_; retain the measured finding.",
    "verdict_class": "Use the closed verdict class; only incomplete work from this task is partial.",
    "validation_receipts": "Retain command, exit code, timing, and log hash; do not hide failures.",
    "board_disposition_complete_score": "One means three authenticated dispositions and exact next conditions are present.",
    "board_rows": "Separate KV260 fabric, PolarFire CPU, and GateMate physical-state evidence.",
    "operator_state_receipt": "Only a real later physical-state change can enable a future GateMate probe.",
    "hardware_operations_issued": "Empty for this host-only receipt.",
}
REQUIRED_ARTIFACT_FIELDS = current.REQUIRED_ARTIFACT_FIELDS

sha256_file = current.sha256_file
sha256_text = current.sha256_text
artifact_checksum = current.artifact_checksum
_read_json = current._read_json
_atomic_json = current._atomic_json
_writable_destination = current._writable_destination
_check = current._check
_first_failed = current._first_failed
_gate_summary = current._gate_summary
_gate_result = current._gate_result
_display_path = current._display_path
progress = current.progress
reference_observation = current.reference_observation
search_gatemate_operator_receipts = current.search_gatemate_operator_receipts
write_negative_fixture_receipt = current.write_negative_fixture_receipt
read_validation_receipts = current.read_validation_receipts
reduce_board_rows = current.reduce_board_rows


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep provisional, raw, and terminal outputs in separate locations."""

    artifact: Path
    checkpoint: Path
    operator_search: Path
    historical_models: Path
    negative_fixtures: Path
    validation_dir: Path

    @classmethod
    def defaults(cls, root: Path = REPO_ROOT) -> ExperimentPaths:
        """Resolve all producer outputs below one repository root."""

        return cls(
            root / RESULT_PATH,
            root / CHECKPOINT_PATH,
            root / OPERATOR_SEARCH_PATH,
            root / HISTORICAL_MODELS_PATH,
            root / NEGATIVE_FIXTURES_PATH,
            root / VALIDATION_DIR,
        )

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Give tests private paths that cannot replace research evidence."""

        return cls.defaults(root)


def _task_contract(root: Path) -> JsonDict | None:
    """Read the fields that fix the identity of only the V640 board task."""

    try:
        roadmap = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    tasks = roadmap.get("tasks") if isinstance(roadmap, Mapping) else None
    if not isinstance(tasks, list):
        return None
    task = next(
        (row for row in tasks if isinstance(row, Mapping) and row.get("id") == TASK_ID),
        None,
    )
    if task is None:
        return None
    result = {
        key: deepcopy(task.get(key))
        for key in EXPECTED_TASK_CONTRACT
        if key not in {"prior_failure_ids", "prompt_sha256"}
    }
    failures = task.get("prior_failures")
    result["prior_failure_ids"] = (
        [row.get("experiment_id") for row in failures if isinstance(row, Mapping)]
        if isinstance(failures, list)
        else None
    )
    prompt = task.get("prompt")
    result["prompt_sha256"] = sha256_text(prompt) if isinstance(prompt, str) else None
    return result


def _board_row(receipt: Mapping[str, Any], board: str) -> Mapping[str, Any] | None:
    """Use the latest reader to select exactly one row for one board."""

    return current._board_row(receipt, board)


def collect_preconditions(
    root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str], dict[str, Any]]:
    """Authenticate the current receipt, original evidence, and output ownership."""

    print("[phase 0 check start] source bytes, spec, and V640 task contract", flush=True)
    checks: list[JsonDict] = []
    sizes = {
        path.as_posix(): (root / path).stat().st_size if (root / path).is_file() else None
        for path in REQUIRED_SOURCE_PATHS
    }
    checks.append(
        _check(
            "required_source_bytes",
            "repository",
            "REQUIRED_SOURCE_PATHS",
            "all nonempty",
            sizes,
            all(size is not None and size > 0 for size in sizes.values()),
        )
    )
    hashes = {
        path.as_posix(): sha256_file(root / path)
        for path in REQUIRED_SOURCE_PATHS
        if sizes[path.as_posix()] not in (None, 0)
    }
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    spec_state = {
        "requirement": "REQ-ISING-7286" in spec_text,
        "scenarios": "SCENARIO-ISING-7286-" in spec_text,
    }
    checks.append(
        _check(
            "driving_capability_spec",
            SPEC_PATH.as_posix(),
            "REQ-ISING-7286 and scenarios",
            {"requirement": True, "scenarios": True},
            spec_state,
            all(spec_state.values()),
        )
    )
    contract = _task_contract(root)
    checks.append(
        _check(
            "roadmap_task_contract",
            ROADMAP_PATH.as_posix(),
            TASK_ID,
            EXPECTED_TASK_CONTRACT,
            contract if contract is not None else "missing_task_contract",
            contract == EXPECTED_TASK_CONTRACT,
        )
    )

    print("[phase 0 check start] authority separation and writable outputs", flush=True)
    resources = {
        "python": str(Path(sys.executable).absolute()),
        "execution_host": platform.node(),
        "latest_board_reader": callable(current.reference_observation),
        "latest_artifact_validator": callable(current.validate_artifact),
        "operator_receipt_parser": callable(current.search_gatemate_operator_receipts),
        "row_reducer": callable(current.reduce_board_rows),
        "artifact_writable": _writable_destination(paths.artifact),
        "checkpoint_writable": _writable_destination(paths.checkpoint),
        "operator_search_writable": _writable_destination(paths.operator_search),
        "historical_models_writable": _writable_destination(paths.historical_models),
        "negative_fixtures_writable": _writable_destination(paths.negative_fixtures),
    }
    checks.append(
        _check(
            "imports_resource_ownership_and_outputs",
            "host",
            "shipped authorities and output ownership",
            "all authorities available and outputs writable",
            resources,
            bool(resources["python"] and resources["execution_host"])
            and all(
                value is True
                for key, value in resources.items()
                if key.endswith(("reader", "validator", "parser", "reducer", "writable"))
            ),
        )
    )

    print("[phase 0 check start] exclusion state and current receipt", flush=True)
    try:
        manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        manifest = None
    checks.append(
        _check(
            "exclusion_manifest_loaded",
            EXCLUSION_PATH.as_posix(),
            "YAML mapping",
            True,
            isinstance(manifest, Mapping),
            isinstance(manifest, Mapping),
        )
    )
    receipt = _read_json(root / UPSTREAM_PATH)
    quarantine = current.current.exp7244._quarantine(receipt, manifest, "7272")
    checks.append(
        _check(
            "exp7272_not_quarantined_or_retired",
            UPSTREAM_PATH.as_posix(),
            "quarantined_or_retired",
            False,
            quarantine,
            bool(receipt) and quarantine.get("quarantined") is False,
        )
    )
    upstream_errors = current.validate_artifact(receipt) if receipt else ["not_json_object"]
    checks.append(
        _check(
            "exp7272_latest_validator",
            UPSTREAM_PATH.as_posix(),
            "validator_errors",
            [],
            upstream_errors,
            upstream_errors == [],
        )
    )
    kv260 = _board_row(receipt, "KV260")
    polarfire = _board_row(receipt, "PolarFire")
    references = reference_observation(root, kv260, polarfire)
    checks.append(
        _check(
            "graduated_board_reference_hashes",
            UPSTREAM_PATH.as_posix(),
            "KV260 and PolarFire original evidence paths and hashes",
            "all current bytes hash-match",
            references,
            references["all_match"] is True,
        )
    )
    observed_scope = {
        "kv260_criterion": kv260.get("terminal_criterion") if kv260 else None,
        "kv260_met": kv260.get("terminal_criterion_met") if kv260 else None,
        "kv260_processor": kv260.get("processor_class") if kv260 else None,
        "polarfire_criterion": polarfire.get("terminal_criterion") if polarfire else None,
        "polarfire_met": polarfire.get("terminal_criterion_met") if polarfire else None,
        "polarfire_processor": polarfire.get("processor_class") if polarfire else None,
        "polarfire_fpga_sampling": (
            polarfire.get("programmable_logic_sampling_observed") if polarfire else None
        ),
    }
    expected_scope = {
        "kv260_criterion": KV260_TERMINAL_CRITERION,
        "kv260_met": True,
        "kv260_processor": "fpga_fabric",
        "polarfire_criterion": POLARFIRE_TERMINAL_CRITERION,
        "polarfire_met": True,
        "polarfire_processor": "cpu",
        "polarfire_fpga_sampling": False,
    }
    checks.append(
        _check(
            "graduated_board_terminal_definitions",
            UPSTREAM_PATH.as_posix(),
            "exact terminal criteria and processor boundaries",
            expected_scope,
            observed_scope,
            observed_scope == expected_scope,
        )
    )
    operator = receipt.get("operator_state_receipt") if receipt else None
    cutoff_hash = hashes.get(GATEMATE_CUTOFF_PATH.as_posix())
    cutoff_observed = {
        "receipt_present": (root / GATEMATE_CUTOFF_PATH).is_file(),
        "expected_by_exp7272": (
            operator.get("cutoff_source_hash") if isinstance(operator, Mapping) else None
        ),
        "observed_sha256": cutoff_hash,
    }
    checks.append(
        _check(
            "exp6559_boundary_authenticated",
            GATEMATE_CUTOFF_PATH.as_posix(),
            "presence and hash",
            {"present": True, "hash_match": True},
            cutoff_observed,
            cutoff_observed["receipt_present"] is True
            and cutoff_observed["expected_by_exp7272"] == cutoff_hash,
        )
    )
    return (
        checks,
        hashes,
        {
            "receipt": receipt,
            "kv260_row": kv260,
            "polarfire_row": polarfire,
            "manifest": manifest,
            "reference_observation": references,
        },
    )


def write_historical_model_receipt(root: Path, path: Path) -> JsonDict:
    """Keep upstream model declarations outside this invocation identity."""

    source = _read_json(root / UPSTREAM_PATH)
    receipt = {
        "schema": "carnot.exp7286.historical_model_receipts.v1",
        "current_invocation_model_specs": [],
        "current_invocation_model_count": 0,
        "source_receipts": [
            {
                "source_path": UPSTREAM_PATH.as_posix(),
                "source_sha256": sha256_file(root / UPSTREAM_PATH),
                "historical_MODEL_SPECS": source.get("MODEL_SPECS"),
                "historical_model_invoked": source.get("model_invoked"),
                "historical_inference_substrate": source.get("inference_substrate"),
            }
        ],
    }
    _atomic_json(path, receipt)
    return receipt


def build_board_rows(
    root: Path, upstreams: Mapping[str, Any], operator: Mapping[str, Any]
) -> list[JsonDict]:
    """Advance provenance while retaining each board's measured evidence scope."""

    receipt = cast(Mapping[str, Any], upstreams["receipt"])
    upstream_hash = sha256_file(root / UPSTREAM_PATH)
    changed = operator.get("exists") is True
    rows: list[JsonDict] = []
    for board, source_key in (
        ("KV260", "kv260_row"),
        ("GateMate", "gatemate_row"),
        ("PolarFire", "polarfire_row"),
    ):
        source = cast(
            Mapping[str, Any],
            _board_row(receipt, board) if source_key == "gatemate_row" else upstreams[source_key],
        )
        row = deepcopy(dict(source))
        row.pop("row_sha256", None)
        if board == "GateMate":
            row.update(
                {
                    "processor_class": "not_executed" if changed else "unavailable",
                    "latest_receipt_path": operator.get("search_receipt_path"),
                    "latest_receipt_date": RUN_DATE,
                    "latest_receipt_hash": operator.get("search_receipt_hash"),
                    "latest_receipt_authenticated": True,
                    "operator_source_path": operator.get("source_path"),
                    "operator_author_evidence": operator.get("author_evidence"),
                    "operator_date_evidence": operator.get("date_evidence"),
                    "operator_evidence_hash": operator.get("evidence_hash"),
                    "operator_changed_conditions": deepcopy(operator.get("changed_conditions", {})),
                    "observed_state": (
                        "operator_changed_physical_state_recorded"
                        if changed
                        else "operator_changed_physical_state_receipt_missing"
                    ),
                    "observed_missing_receipt": operator.get("observed_missing_receipt"),
                    "disposition": (
                        "changed_physical_state_future_action_enabled"
                        if changed
                        else "blocked_changed_physical_state"
                    ),
                    "exact_next_condition": (
                        str(operator.get("newly_enabled_next_action"))
                        if changed
                        else GATEMATE_OPERATOR_ACTION
                    ),
                    "metric": changed,
                    "error": None if changed else MISSING_RECEIPT,
                    "abstention": not changed,
                }
            )
        else:
            evidence = [
                {
                    "path": source.get("latest_receipt_path"),
                    "sha256": source.get("latest_receipt_hash"),
                },
                *deepcopy(source.get("referenced_evidence", [])),
            ]
            row.update(
                {
                    "latest_receipt_path": UPSTREAM_PATH.as_posix(),
                    "latest_receipt_date": receipt.get("run_date"),
                    "latest_receipt_hash": upstream_hash,
                    "latest_receipt_authenticated": True,
                    "referenced_evidence": evidence,
                }
            )
        row["hardware_operations_issued"] = []
        row["hardware_command_count"] = 0
        rows.append(current.current._finish_row(row))
    return rows


def _base_artifact(
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
    phase_spans: Mapping[str, float],
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
) -> JsonDict:
    """Create the complete field shape before selecting a terminal state."""

    artifact = current._base_artifact(
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
        phase_spans=phase_spans,
        checks=checks,
        hashes=hashes,
    )
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "task_id": TASK_ID,
            "milestone": MILESTONE,
            "spec_refs": [
                "REQ-ISING-7286",
                "SCENARIO-ISING-7286-PREFLIGHT",
                "SCENARIO-ISING-7286-BOARDS",
                "SCENARIO-ISING-7286-GATEMATE",
                "SCENARIO-ISING-7286-FIXTURES",
                "SCENARIO-ISING-7286-ARTIFACT",
            ],
            "field_principles": FIELD_PRINCIPLES,
            "random_seed": RANDOM_SEED,
            "invocation_counts": {
                "model_loads_attempted": 0,
                "model_loads_completed": 0,
                "generation_calls_attempted": 0,
                "generation_calls_completed": 0,
                "usable_answers": 0,
            },
        }
    )
    return artifact


def build_artifact(
    root: Path,
    paths: ExperimentPaths,
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> JsonDict:
    """Aggregate authenticated receipt bytes in memory without a board command."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: dict[str, float] = {}

    phase_started = time.monotonic()
    progress(0, "start", "authenticate V640 sources before aggregation")
    checks, hashes, upstreams = collect_preconditions(root, paths)
    spans["phase_0_preconditions"] = time.monotonic() - phase_started
    failed = _first_failed(checks)
    progress(0, "end", f"preconditions failed={int(failed is not None)}")
    if failed is not None:
        artifact = _base_artifact(
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - started,
            phase_spans=spans,
            checks=checks,
            hashes=hashes,
        )
        artifact["honest_verdict"] = (
            f"blocked_{failed['check']}: expected {failed['expected_value']!r}; "
            f"observed {failed['observed_value']!r}"
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return artifact

    _atomic_json(
        paths.checkpoint,
        {
            "schema": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "in_progress",
            "started_at_utc": started_at,
            "terminal_artifact_path": str(paths.artifact),
        },
    )

    phase_started = time.monotonic()
    progress(1, "start", "write historical-model and negative-fixture sidecars")
    write_historical_model_receipt(root, paths.historical_models)
    negative = write_negative_fixture_receipt(paths.negative_fixtures)
    for path in (paths.historical_models, paths.negative_fixtures):
        hashes[_display_path(root, path)] = sha256_file(path)
    spans["phase_1_hashed_sidecars"] = time.monotonic() - phase_started
    progress(1, "end", f"sidecars=2 negative-fixtures={negative['fixture_count']}")

    phase_started = time.monotonic()
    progress(2, "start", "search later operator GateMate physical-state receipts")
    operator = search_gatemate_operator_receipts(
        root, paths.operator_search, candidate_paths=candidate_paths
    )
    hashes[_display_path(root, paths.operator_search)] = operator["search_receipt_hash"]
    spans["phase_2_operator_receipt"] = time.monotonic() - phase_started
    progress(2, "end", f"accepted-receipts={int(operator['exists'])} hardware-commands=0")

    phase_started = time.monotonic()
    progress(3, "start", "reduce current receipt into three board dispositions")
    rows = build_board_rows(root, upstreams, operator)
    reduced = reduce_board_rows(rows)
    spans["phase_3_board_reducer"] = time.monotonic() - phase_started
    progress(
        3,
        "end",
        f"completed-units={reduced['board_count']} elapsed={time.monotonic() - started:.6f}s",
    )

    phase_started = time.monotonic()
    progress(4, "start", "assemble terminal host-only receipt in memory")
    validation_rows = read_validation_receipts(paths.validation_dir)
    for row in validation_rows:
        log_path = Path(str(row["log_path"]))
        hashes[_display_path(root, log_path)] = str(row["log_hash"])
    baseline_rows = read_validation_receipts(paths.validation_dir.parent / "baseline_failures")
    for row in baseline_rows:
        log_path = Path(str(row["log_path"]))
        hashes[_display_path(root, log_path)] = str(row["log_hash"])
    validation_rows.append(
        {
            "command": "internal current.reduce_board_rows(board_rows)",
            "exit_code": 0,
            "log_hash": sha256_text(json.dumps(reduced, sort_keys=True)),
        }
    )
    spans["phase_4_artifact_assembly"] = time.monotonic() - phase_started
    artifact = _base_artifact(
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        checks=checks,
        hashes=hashes,
    )
    summary = _gate_summary(checks)
    summary["board_blocks"] = (
        []
        if operator["exists"]
        else [
            {
                "verdict": "blocked_changed_physical_state",
                "upstream": "operator_state_receipt",
                "field": "operator-authored physical change after Exp6559",
                "expected_value": GATEMATE_OPERATOR_ACTION,
                "observed_value": MISSING_RECEIPT,
            }
        ]
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "inference_substrate_class": "aggregation",
            "rows": rows,
            "sample_size_budget": {
                "planned": 3,
                "attempted": 3,
                "completed": 3,
                "censored": 0,
                "independent_units_planned": 3,
                "independent_units_completed": 3,
                "stopping_rule": "one read-only disposition per board; zero board commands",
            },
            "acceptance_gate_results": {
                "current_receipt_and_references_authenticated": _gate_result(
                    "Use the shipped V639 validator and original referenced hashes.",
                    True,
                    upstreams["reference_observation"]["all_match"],
                    upstreams["reference_observation"]["all_match"] is True,
                ),
                "kv260_exact_terminal_preserved": _gate_result(
                    "Preserve synthesis and programmable-logic latency graduation.",
                    KV260_TERMINAL_CRITERION,
                    rows[0]["terminal_criterion"],
                    rows[0]["terminal_criterion"] == KV260_TERMINAL_CRITERION,
                ),
                "polarfire_cpu_not_fpga_preserved": _gate_result(
                    "Keep CPU dispatch distinct from FPGA sampling.",
                    {"processor_class": "cpu", "fpga_sampling": False},
                    {
                        "processor_class": rows[2]["processor_class"],
                        "fpga_sampling": rows[2]["programmable_logic_sampling_observed"],
                    },
                    rows[2]["processor_class"] == "cpu"
                    and rows[2]["programmable_logic_sampling_observed"] is False,
                ),
                "gatemate_disposition_recorded": _gate_result(
                    "Record a later change or its exact missing receipt.",
                    "changed receipt or blocked_changed_physical_state",
                    rows[1]["disposition"],
                    rows[1]["disposition"]
                    in {
                        "blocked_changed_physical_state",
                        "changed_physical_state_future_action_enabled",
                    },
                ),
                "negative_receipt_fixtures_fail_closed": _gate_result(
                    "Missing and malformed receipts authorize no command.",
                    {"count": 2, "all_failed_closed": True},
                    {
                        "count": negative["fixture_count"],
                        "all_failed_closed": negative["all_failed_closed"],
                    },
                    negative["fixture_count"] == 2 and negative["all_failed_closed"] is True,
                ),
                "hardware_operations_issued_count": _gate_result(
                    "This host-only task issues no board command.", 0, 0, True
                ),
                "three_dispositions_reduce": _gate_result(
                    "Every board has authenticated evidence and an exact next condition.",
                    1,
                    reduced["board_disposition_complete_score"],
                    reduced["board_disposition_complete_score"] == 1,
                ),
            },
            "gate_check_summary": summary,
            "verdict_class": "positive",
            "honest_verdict": (
                "complete: three authenticated board dispositions and exact next conditions "
                "are recorded. KV260 fabric graduation and PolarFire hash-matched CPU dispatch "
                "remain preserved. PolarFire CPU dispatch is not FPGA sampling. "
                + (
                    "A later operator GateMate physical-state receipt enables one future probe. "
                    if operator["exists"]
                    else "GateMate remains blocked_changed_physical_state because the required "
                    "operator receipt is missing. "
                )
                + "This invocation issued zero hardware operations."
            ),
            "validation_receipts": validation_rows,
            "baseline_validation_failures": baseline_rows,
            "board_disposition_complete_score": reduced["board_disposition_complete_score"],
            "board_rows": rows,
            "operator_state_receipt": operator,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    progress(4, "end", "terminal receipt assembled; hardware-commands=0")
    return artifact


def validate_artifact(artifact: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Check V640 identity, then delegate board semantics to the V639 validator."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        return ["missing_fields:" + ",".join(missing)]
    errors: list[str] = []
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("task_id") != TASK_ID
        or artifact.get("milestone") != MILESTONE
    ):
        errors.append("identity")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("invocation_counts") != {
        "model_loads_attempted": 0,
        "model_loads_completed": 0,
        "generation_calls_attempted": 0,
        "generation_calls_completed": 0,
        "usable_answers": 0,
    }:
        errors.append("invocation_counts")
    try:
        checksum_matches = artifact.get("reproducibility_checksum") == artifact_checksum(artifact)
    except (TypeError, ValueError):
        checksum_matches = False
    if not checksum_matches:
        errors.append("reproducibility_checksum")
    try:
        compatible = deepcopy(dict(artifact))
        compatible.update(
            {
                "schema": current.SCHEMA,
                "experiment_id": current.EXPERIMENT_ID,
                "task_id": current.TASK_ID,
                "milestone": current.MILESTONE,
                "field_principles": current.FIELD_PRINCIPLES,
            }
        )
        compatible["reproducibility_checksum"] = artifact_checksum(compatible)
    except (TypeError, ValueError):
        return errors
    for error in current.validate_artifact(compatible, root=root):
        if error not in errors:
            errors.append(error)
    return errors


def atomic_write(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Publish only bytes accepted by the independent validator chain."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp7286 artifact: {errors}")
    _atomic_json(path, artifact)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "atomic_replace": True,
    }


def run_experiment(root: Path, paths: ExperimentPaths) -> JsonDict:
    """Build, reduce, validate, and atomically publish the host receipt."""

    artifact = build_artifact(root, paths)
    progress(5, "before", "final independent raw-row reducer and artifact validation")
    errors = validate_artifact(artifact, root=root)
    progress(5, "after", f"validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7286 artifact: {errors}")
    progress(6, "before", "atomic terminal write")
    receipt = atomic_write(paths.artifact, artifact)
    progress(6, "after", f"atomic terminal write bytes={receipt['bytes']}")
    return artifact


def _parser() -> argparse.ArgumentParser:
    """Keep production replay and read-only validation on one thin CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the receipt producer or validate existing bytes without mutation."""

    progress(0, "entry", "Exp7286 host aggregation; model-loads=0 hardware-commands=0")
    args = _parser().parse_args(argv)
    try:
        if args.validate is not None:
            progress(5, "before", f"read-only validation path={args.validate}")
            artifact = _read_json(args.validate)
            errors = ["artifact_not_json_object"] if not artifact else validate_artifact(artifact)
            progress(5, "after", f"read-only validation errors={len(errors)}")
            if errors:
                print(f"validation_failed errors={errors}", flush=True)
                return 2
            print("validation_passed", flush=True)
            return 0
        if args.date != RUN_DATE:
            raise ValueError(f"run date must be {RUN_DATE}")
        root = args.root.resolve()
        paths = ExperimentPaths.defaults(root)
        if args.output is not None:
            output = args.output if args.output.is_absolute() else root / args.output
            paths = ExperimentPaths(
                output,
                paths.checkpoint,
                paths.operator_search,
                paths.historical_models,
                paths.negative_fixtures,
                paths.validation_dir,
            )
        artifact = run_experiment(root, paths)
        print(
            f"experiment_complete status={artifact['status']} "
            f"score={artifact['board_disposition_complete_score']} output={paths.artifact}",
            flush=True,
        )
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"experiment_error type={type(exc).__name__} message={exc}", flush=True)
        return 2
