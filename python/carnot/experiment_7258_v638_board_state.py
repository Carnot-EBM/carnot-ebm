"""Produce the V638 host-only board-state receipt.

The task reads checked-in receipts and never contacts a board. It reuses the
Exp7244 validator for graduated-board evidence and the Exp6559 parser for the
operator-only GateMate boundary.

Spec: REQ-ISING-7258 and SCENARIO-ISING-7258-ARTIFACT.
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
import re
import time
from typing import Any

import yaml

from carnot import experiment_6559_gatemate_changed_state_continuity as exp6559
from carnot import experiment_7244_v637_board_disposition as exp7244


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260913"
EXPERIMENT_ID = 7258
TASK_ID = "exp7258-board-state"
MILESTONE = "2026.09.638"
SCHEMA = "carnot.exp7258.v638.board_state.v1"
RANDOM_SEED = 7258

RESULT_PATH = Path("results/experiment_7258_v638_board_state.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7258_v638_board_state.json")
RAW_DIR = Path("results/raw/experiment_7258")
OPERATOR_SEARCH_PATH = RAW_DIR / "gatemate_operator_receipt_search.json"
HISTORICAL_MODELS_PATH = RAW_DIR / "historical_model_receipts.json"
NEGATIVE_FIXTURES_PATH = RAW_DIR / "negative_fixture_receipts.json"
VALIDATION_DIR = RAW_DIR / "validation"
ENTRYPOINT_PATH = Path("scripts/experiments/experiment_7258_v638_board_state.py")
MODULE_PATH = Path("python/carnot/experiment_7258_v638_board_state.py")
TEST_PATH = Path("tests/python/test_experiment_7258_v638_board_state.py")
SPEC_PATH = exp7244.SPEC_PATH
ROADMAP_PATH = exp7244.ROADMAP_PATH
EXCLUSION_PATH = exp7244.EXCLUSION_PATH
UPSTREAM_PATH = exp7244.RESULT_PATH
GATEMATE_CUTOFF_PATH = exp7244.GATEMATE_CUTOFF_PATH
KV260_PATH = exp7244.KV260_PATH
KV260_CONFIRM_PATH = exp7244.KV260_CONFIRM_PATH
CONTINUITY_PATH = exp7244.CONTINUITY_PATH
POLARFIRE_RAW_PATH = exp7244.POLARFIRE_RAW_PATH

KV260_TERMINAL_CRITERION = exp7244.KV260_TERMINAL_CRITERION
POLARFIRE_TERMINAL_CRITERION = exp7244.POLARFIRE_TERMINAL_CRITERION
GATEMATE_TERMINAL_CRITERION = exp7244.GATEMATE_TERMINAL_CRITERION
GATEMATE_CUTOFF_DATE = exp7244.GATEMATE_CUTOFF_DATE
MISSING_RECEIPT = (
    "no operator-authored dated GateMate cable, port, board, power, or DirtyJTAG "
    "change after Exp6559"
)
GATEMATE_OPERATOR_ACTION = (
    "operator records a dated GateMate cable, port, board, power, or DirtyJTAG change after Exp6559"
)
GATEMATE_FUTURE_ACTION = "one bounded GateMate DirtyJTAG detect in a future hardware task"

EXPECTED_TASK_CONTRACT: JsonDict = {
    "id": TASK_ID,
    "title": "GateMate physical-state condition and graduated-board record",
    "track": "hardware",
    "priority": "high",
    "requires_gpu": False,
    "max_turns": 20,
    "estimated_wall_time_min": 10,
    "per_unit_rows": True,
    "milestone": MILESTONE,
    "deliverable": RESULT_PATH.as_posix(),
    "prior_failure_ids": [
        "exp6559-gatemate-changed-state-continuity",
        "exp5861-attached-board-state-receipts",
    ],
    "prompt_sha256": "sha256:b316e8f63ecae3edc1850b5bc640d10748663b86a319aad483505202666e7048",
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
    KV260_PATH,
    KV260_CONFIRM_PATH,
    CONTINUITY_PATH,
    POLARFIRE_RAW_PATH,
)

HISTORICAL_SOURCE_PATHS = (
    UPSTREAM_PATH,
    KV260_PATH,
    KV260_CONFIRM_PATH,
    CONTINUITY_PATH,
    POLARFIRE_RAW_PATH,
    GATEMATE_CUTOFF_PATH,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Version this artifact; also emit experiment_id and milestone as ordinary top-level values.",
    "experiment_id": "Bind this receipt to Exp7258.",
    "task_id": "Bind this receipt to the exact V638 roadmap task.",
    "milestone": "Bind this receipt to milestone 2026.09.638.",
    "spec_refs": "Connect the artifact and tests to REQ-ISING-7258 scenarios.",
    "status": "A terminal artifact records complete or blocked; unfinished work belongs in a separate checkpoint.",
    "run_date": "Use 20260913 and retain actual UTC start and end timestamps.",
    "started_at_utc": "Record the actual UTC start timestamp.",
    "completed_at_utc": "Record the actual UTC end timestamp.",
    "field_principles": "Store explanations here while leaving ordinary values at top level for consumers.",
    "preconditions_checked": "Record exact observed inputs, resource ownership, hashes and failed checks before expensive work.",
    "MODEL_SPECS": "Name only models this invocation may execute; source-model history lives in hashed sidecars.",
    "model_invoked": "Derive from actual current calls, not usable-answer count or a nested control arm.",
    "model_invocation_count": "Count only model calls made by this invocation.",
    "model_load_count": "Count only model loads made by this invocation.",
    "generation_count": "Count only generations made by this invocation.",
    "inference_substrate": "Describe the compute actually performed with a recognized literal.",
    "inference_substrate_class": "Use the closed compute class and its duration floor; never sleep or relabel to pass.",
    "execution_venue": "Use host for orchestration; actual board receipts separately name kv260, gatemate or polarfire.",
    "execution_host": "Record the real hostname separately from the closed venue vocabulary.",
    "duration_s": "Measure monotonic elapsed time for this invocation, with disjoint phase spans and no invented time.",
    "phase_spans_s": "Retain the disjoint measured phase spans that make up the invocation.",
    "random_seed": "Freeze seeds before seeing outcomes so replay cannot select favorable runs.",
    "reproducibility_checksum": "Bind source code, input manifests, configuration and raw rows to the result.",
    "source_artifact_hashes": "Authenticate input bytes and preserve quarantine; readiness alone is insufficient.",
    "rows": "Retain every unit, arm, seed, metric, error, abstention and censoring state; aggregates must be recomputable.",
    "sample_size_budget": "State planned, attempted, completed and censored independent units and the fixed stopping rule.",
    "acceptance_gate_results": "For each frozen criterion record expected, observed and passed, plus its principle.",
    "gate_check_summary": "For every blocked_* verdict name the upstream, exact field or check, observed and expected values.",
    "verifier_is_oracle": "Expose exact-oracle use; oracle conformance cannot become learned verification evidence.",
    "honest_verdict": "Use complete_* for terminal measured findings and blocked_* for absent external prerequisites.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive; a failed scientific acceptance gate forbids positive.",
    "validation_receipts": "Record actual command, exit code and log hash; no skipped, weakened, deleted or reverted tests.",
    "baseline_validation_failures": "Keep unrelated nonzero suite receipts separate from the passing changed-scope gates.",
    "board_disposition_complete_score": "One records three authenticated dispositions and exact next conditions, not three working samplers.",
    "board_rows": "Separate KV260 fabric, PolarFire CPU and GateMate physical-state boundaries.",
    "operator_state_receipt": "Only a real later operator-authored physical change can reopen GateMate probing.",
    "hardware_operations_issued": "Must be an empty list for this host-only task.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)

sha256_file = exp7244.sha256_file
sha256_text = exp6559.sha256_text
artifact_checksum = exp7244.artifact_checksum
_read_json = exp7244._read_json
_atomic_json = exp7244._atomic_json
_writable_destination = exp7244._writable_destination
_check = exp7244._check
_first_failed = exp7244._first_failed
_gate_summary = exp7244._gate_summary


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
        """Resolve all production outputs below one repository root."""

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
        """Give a test private paths that cannot replace research evidence."""

        return cls.defaults(root)


def progress(phase: int, boundary: str, operation: str) -> None:
    """Flush every real phase boundary so the task stays observable."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def _display_path(root: Path, path: Path) -> str:
    """Use a stable repository-relative path when the path belongs to this run."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _task_contract(root: Path) -> JsonDict | None:
    """Read the fixed identity and prompt hash for only the Exp7258 task."""

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


def _reference_rows(
    receipt: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """Select exactly one KV260 row and one PolarFire row from Exp7244."""

    rows = receipt.get("board_rows")
    kv260 = exp7244._row_for_board(rows, "KV260")
    polarfire = exp7244._row_for_board(rows, "PolarFire")
    return kv260, polarfire


def _immutable_reference_observation(
    root: Path,
    kv260: Mapping[str, Any] | None,
    polarfire: Mapping[str, Any] | None,
) -> JsonDict:
    """Recheck the exact immutable paths and hashes cited by the board rows."""

    references: list[Mapping[str, Any]] = []
    if kv260 is not None:
        references.append(
            {"path": kv260.get("latest_receipt_path"), "sha256": kv260.get("latest_receipt_hash")}
        )
        supporting = kv260.get("supporting_evidence")
        if isinstance(supporting, list):
            references.extend(row for row in supporting if isinstance(row, Mapping))
    if polarfire is not None:
        references.append(
            {
                "path": polarfire.get("latest_receipt_path"),
                "sha256": polarfire.get("latest_receipt_hash"),
            }
        )
        supporting = polarfire.get("supporting_evidence")
        if isinstance(supporting, list):
            references.extend(row for row in supporting if isinstance(row, Mapping))
    rows: list[JsonDict] = []
    for reference in references:
        rel = reference.get("path")
        path = root / str(rel) if rel else root / "__missing_reference__"
        current = sha256_file(path) if path.is_file() else None
        rows.append(
            {
                "path": rel,
                "expected_sha256": reference.get("sha256"),
                "observed_sha256": current,
                "passed": current is not None and current == reference.get("sha256"),
            }
        )
    return {"rows": rows, "all_match": bool(rows) and all(row["passed"] for row in rows)}


def collect_preconditions(
    root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str], dict[str, Any]]:
    """Authenticate sources, quarantine state, board chains, imports, and outputs."""

    print("[phase 0 check start] source bytes, spec, and task contract", flush=True)
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
        "requirement": "REQ-ISING-7258" in spec_text,
        "scenarios": "SCENARIO-ISING-7258-" in spec_text,
    }
    checks.append(
        _check(
            "driving_capability_spec",
            SPEC_PATH.as_posix(),
            "REQ-ISING-7258 and scenarios",
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

    print("[phase 0 check start] imports, ownership, and writable outputs", flush=True)
    resources = {
        "python": str(Path(exp7244.os.sys.executable).absolute()),
        "execution_host": platform.node(),
        "exp7244_validator": callable(exp7244.validate_artifact),
        "exp6559_operator_parser": callable(exp6559.search_dated_receipts),
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
            "python,hostname,shipped readers,and output ownership",
            "all readers available and outputs writable",
            resources,
            bool(resources["python"] and resources["execution_host"])
            and all(
                value is True
                for key, value in resources.items()
                if key.endswith(("validator", "parser", "writable"))
            ),
        )
    )

    print("[phase 0 check start] quarantine and shipped receipt validation", flush=True)
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
    upstream_quarantine = exp7244._quarantine(receipt, manifest, "7244")
    checks.append(
        _check(
            "exp7244_not_quarantined",
            UPSTREAM_PATH.as_posix(),
            "quarantined",
            False,
            upstream_quarantine,
            bool(receipt) and upstream_quarantine.get("quarantined") is False,
        )
    )
    shipped_errors = exp7244.validate_artifact(receipt) if receipt else ["not_json_object"]
    checks.append(
        _check(
            "exp7244_shipped_validator",
            UPSTREAM_PATH.as_posix(),
            "validator_errors",
            [],
            shipped_errors,
            shipped_errors == [],
        )
    )
    kv260_row, polarfire_row = _reference_rows(receipt)
    immutable = _immutable_reference_observation(root, kv260_row, polarfire_row)
    checks.append(
        _check(
            "graduated_board_reference_hashes",
            UPSTREAM_PATH.as_posix(),
            "KV260 and PolarFire referenced transcript hashes",
            "all current bytes hash-match",
            immutable,
            immutable["all_match"] is True,
        )
    )
    board_state = {
        "kv260_criterion": kv260_row.get("terminal_criterion") if kv260_row else None,
        "kv260_met": kv260_row.get("terminal_criterion_met") if kv260_row else None,
        "kv260_processor": kv260_row.get("processor_class") if kv260_row else None,
        "polarfire_criterion": (polarfire_row.get("terminal_criterion") if polarfire_row else None),
        "polarfire_met": polarfire_row.get("terminal_criterion_met") if polarfire_row else None,
        "polarfire_processor": polarfire_row.get("processor_class") if polarfire_row else None,
        "polarfire_fpga_sampling": (
            polarfire_row.get("programmable_logic_sampling_observed") if polarfire_row else None
        ),
    }
    board_expected = {
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
            board_expected,
            board_state,
            board_state == board_expected,
        )
    )
    cutoff = _read_json(root / GATEMATE_CUTOFF_PATH)
    cutoff_quarantine = exp7244._quarantine(cutoff, manifest, "6559")
    cutoff_hash = hashes.get(GATEMATE_CUTOFF_PATH.as_posix())
    cutoff_state = {
        "receipt_present": bool(cutoff),
        "quarantine": cutoff_quarantine,
        "expected_by_exp7244": receipt.get("operator_state_receipt", {}).get("cutoff_source_hash"),
        "observed_sha256": cutoff_hash,
    }
    checks.append(
        _check(
            "exp6559_boundary_authenticated",
            GATEMATE_CUTOFF_PATH.as_posix(),
            "presence,quarantine,and hash",
            {"present": True, "quarantined": False, "hash_match": True},
            cutoff_state,
            bool(cutoff)
            and cutoff_quarantine.get("quarantined") is False
            and cutoff_state["expected_by_exp7244"] == cutoff_hash,
        )
    )
    return (
        checks,
        hashes,
        {
            "receipt": receipt,
            "kv260_row": kv260_row,
            "polarfire_row": polarfire_row,
            "manifest": manifest,
            "immutable_references": immutable,
        },
    )


def _changed_conditions(row: Mapping[str, Any]) -> dict[str, str]:
    """Extract only the five physical fields approved by the Exp6559 parser."""

    raw = row.get("raw_receipt")
    receipt = raw if isinstance(raw, Mapping) else {}
    changed: dict[str, str] = {}
    changes = receipt.get("changes")
    if isinstance(changes, list):
        for item in changes:
            if not isinstance(item, Mapping):
                continue
            field = str(item.get("field", "")).lower()
            if field in exp6559.MATERIAL_PHYSICAL_FIELDS:
                changed[field] = str(item.get("description") or receipt.get(field) or "changed")
    if not changed:
        for field in exp6559.MATERIAL_PHYSICAL_FIELDS - {"board"}:
            if receipt.get(field):
                changed[field] = str(receipt[field])
    return dict(sorted(changed.items()))


def _candidate_rows_from_paths(candidate_paths: Sequence[Path]) -> list[JsonDict]:
    """Turn explicit fixture files into parser rows while retaining parse failures."""

    rows: list[JsonDict] = []
    for index, path in enumerate(candidate_paths, start=1):
        present = path.is_file()
        try:
            value = json.loads(path.read_text(encoding="utf-8")) if present else None
        except (OSError, json.JSONDecodeError):
            value = None
        if isinstance(value, Mapping):
            parsed = exp6559.rows_from_overrides([value], GATEMATE_CUTOFF_DATE)[0]
            row = dict(parsed)
            row["row_id"] = f"explicit-{index:03d}"
        else:
            row = {
                "row_id": f"explicit-{index:03d}",
                "receipt_date": None,
                "operator_authored": False,
                "material_physical_fields": [],
                "target_ok": False,
                "valid": False,
                "reject_reason": "missing" if not present else "malformed_json",
                "source": "explicit fixture",
                "raw_receipt": {},
            }
        row["path"] = str(path)
        row["source_path"] = str(path)
        row["source_sha256"] = sha256_file(path) if present else None
        rows.append(row)
    return rows


def search_gatemate_operator_receipts(
    root: Path,
    raw_path: Path,
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> JsonDict:
    """Search the approved operator sources and write a raw, zero-command receipt."""

    if candidate_paths is None:
        parsed = exp6559.search_dated_receipts(root, GATEMATE_CUTOFF_DATE)
        rows = [
            dict(row)
            for row in parsed
            if row.get("operator_authored") is True
            and row.get("target_ok") is True
            and bool(row.get("material_physical_fields"))
        ]
        for row in rows:
            path = root / str(row["path"])
            row["source_path"] = str(row["path"])
            row["source_sha256"] = sha256_file(path) if path.is_file() else None
    else:
        rows = _candidate_rows_from_paths(candidate_paths)
    accepted = [row for row in rows if row.get("valid") is True]
    selected = max(accepted, key=lambda row: str(row.get("receipt_date"))) if accepted else None
    raw = {
        "schema": "carnot.exp7258.gatemate_operator_receipt_search.v1",
        "run_date": RUN_DATE,
        "cutoff_experiment": "Exp6559",
        "cutoff_date": GATEMATE_CUTOFF_DATE,
        "search_scope": (
            "operator-authored GateMate cable, port, board, power, or DirtyJTAG changes"
        ),
        "candidate_rows": rows,
        "accepted_receipt_count": len(accepted),
        "selected_row_id": selected.get("row_id") if selected else None,
        "explicit_absence": selected is None,
        "hardware_operations_issued": [],
        "external_commands_issued": [],
    }
    _atomic_json(raw_path, raw)
    changed = _changed_conditions(selected) if selected else {}
    return {
        "exists": selected is not None,
        "newer_than_exp6559": selected is not None,
        "cutoff_experiment": "Exp6559",
        "cutoff_date": GATEMATE_CUTOFF_DATE,
        "cutoff_source_path": GATEMATE_CUTOFF_PATH.as_posix(),
        "cutoff_source_hash": sha256_file(root / GATEMATE_CUTOFF_PATH),
        "source_path": selected.get("source_path") if selected else None,
        "source_text": selected.get("source") if selected else None,
        "author_evidence": selected.get("operator_authored") if selected else None,
        "date_evidence": selected.get("receipt_date") if selected else None,
        "evidence_hash": selected.get("source_sha256") if selected else None,
        "changed_conditions": changed,
        "observed_missing_receipt": None if selected else MISSING_RECEIPT,
        "newly_enabled_next_action": GATEMATE_FUTURE_ACTION if selected else None,
        "search_receipt_path": str(raw_path),
        "search_receipt_hash": sha256_file(raw_path),
        "hardware_operations_issued": [],
        "hardware_command_count": 0,
        "external_commands_issued": [],
    }


def write_negative_fixture_receipt(path: Path) -> JsonDict:
    """Prove that separate missing and malformed receipts cannot authorize work."""

    fixtures: tuple[tuple[str, str | None], ...] = (
        ("missing_receipt", None),
        ("malformed_receipt", "{"),
    )
    rows: list[JsonDict] = []
    for name, content in fixtures:
        try:
            value = json.loads(content) if content is not None else None
        except json.JSONDecodeError:
            value = None
        accepted = isinstance(value, Mapping) and bool(
            exp6559.rows_from_overrides([value], GATEMATE_CUTOFF_DATE)[0].get("valid")
        )
        rows.append(
            {
                "fixture": name,
                "input_present": content is not None,
                "parsed_object": isinstance(value, Mapping),
                "accepted": accepted,
                "failed_closed": not accepted,
                "hardware_operations_issued": [],
                "external_commands_issued": [],
            }
        )
    receipt = {
        "schema": "carnot.exp7258.negative_fixture_receipts.v1",
        "fixture_count": len(rows),
        "rows": rows,
        "all_failed_closed": all(row["failed_closed"] for row in rows),
        "hardware_operations_issued": [],
        "external_commands_issued": [],
    }
    _atomic_json(path, receipt)
    return receipt


def write_historical_model_receipt(root: Path, path: Path) -> JsonDict:
    """Keep source model declarations outside the current invocation identity."""

    rows: list[JsonDict] = []
    for rel_path in HISTORICAL_SOURCE_PATHS:
        value = _read_json(root / rel_path)
        rows.append(
            {
                "source_path": rel_path.as_posix(),
                "source_sha256": sha256_file(root / rel_path),
                "historical_MODEL_SPECS": value.get("MODEL_SPECS"),
                "historical_model_invoked": value.get("model_invoked"),
                "historical_inference_substrate": value.get("inference_substrate"),
            }
        )
    receipt = {
        "schema": "carnot.exp7258.historical_model_receipts.v1",
        "current_invocation_model_specs": [],
        "current_invocation_model_count": 0,
        "source_receipts": rows,
    }
    _atomic_json(path, receipt)
    return receipt


def read_validation_receipts(directory: Path) -> list[JsonDict]:
    """Read exact commands and exit markers from validation log sidecars."""

    rows: list[JsonDict] = []
    for path in sorted(directory.glob("*.log")) if directory.is_dir() else []:
        text = path.read_text(encoding="utf-8")
        command_match = re.search(r"(?m)^\$ (.+)$", text)
        exit_match = re.search(r"(?m)^\[exit_code\] (-?\d+)$", text)
        rows.append(
            {
                "command": command_match.group(1) if command_match else "unknown",
                "exit_code": int(exit_match.group(1)) if exit_match else None,
                "log_path": str(path),
                "log_hash": sha256_file(path),
            }
        )
    return rows


def _finish_row(row: JsonDict) -> JsonDict:
    """Apply the shipped row hash after every required comparison field exists."""

    row.setdefault("arm", "aggregation_from_upstream_artifacts")
    row.setdefault("seed", None)
    row.setdefault("metric", False)
    row.setdefault("error", None)
    row.setdefault("abstention", False)
    row.setdefault("censored", False)
    return exp7244._finish_row(row)


def build_board_rows(
    root: Path, upstreams: Mapping[str, Any], operator: Mapping[str, Any]
) -> list[JsonDict]:
    """Build three dispositions from authenticated receipt bytes only."""

    receipt = upstreams["receipt"]
    upstream_hash = sha256_file(root / UPSTREAM_PATH)
    kv_source = upstreams["kv260_row"]
    pf_source = upstreams["polarfire_row"]
    changed = operator.get("exists") is True
    rows = [
        {
            "unit_id": "board:KV260",
            "board": "KV260",
            "execution_venue": "kv260",
            "processor_class": "fpga_fabric",
            "latest_receipt_path": UPSTREAM_PATH.as_posix(),
            "latest_receipt_date": receipt.get("run_date"),
            "latest_receipt_hash": upstream_hash,
            "latest_receipt_authenticated": True,
            "referenced_evidence": deepcopy(kv_source.get("supporting_evidence")),
            "terminal_criterion": KV260_TERMINAL_CRITERION,
            "terminal_criterion_met": True,
            "observed_state": deepcopy(kv_source.get("observed_state")),
            "disposition": "graduated_preserved",
            "exact_next_condition": "none; any future access uses ssh kria only",
            "hardware_operations_issued": [],
            "hardware_command_count": 0,
            "metric": True,
        },
        {
            "unit_id": "board:GateMate",
            "board": "GateMate",
            "execution_venue": "gatemate",
            "processor_class": "not_executed" if changed else "unavailable",
            "latest_receipt_path": operator.get("search_receipt_path"),
            "latest_receipt_date": RUN_DATE,
            "latest_receipt_hash": operator.get("search_receipt_hash"),
            "latest_receipt_authenticated": True,
            "operator_source_path": operator.get("source_path"),
            "operator_evidence_hash": operator.get("evidence_hash"),
            "terminal_criterion": GATEMATE_TERMINAL_CRITERION,
            "terminal_criterion_met": False,
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
            "hardware_operations_issued": [],
            "hardware_command_count": 0,
            "metric": changed,
            "error": None if changed else MISSING_RECEIPT,
            "abstention": not changed,
        },
        {
            "unit_id": "board:PolarFire",
            "board": "PolarFire",
            "execution_venue": "polarfire",
            "processor_class": "cpu",
            "latest_receipt_path": UPSTREAM_PATH.as_posix(),
            "latest_receipt_date": receipt.get("run_date"),
            "latest_receipt_hash": upstream_hash,
            "latest_receipt_authenticated": True,
            "referenced_evidence": [
                {
                    "path": pf_source.get("latest_receipt_path"),
                    "sha256": pf_source.get("latest_receipt_hash"),
                },
                *deepcopy(pf_source.get("supporting_evidence", [])),
            ],
            "terminal_criterion": POLARFIRE_TERMINAL_CRITERION,
            "terminal_criterion_met": True,
            "observed_state": deepcopy(pf_source.get("observed_state")),
            "disposition": "graduated_cpu_dispatch_preserved",
            "exact_next_condition": (
                "none for CPU dispatch; FPGA sampling remains a separate future task"
            ),
            "programmable_logic_sampling_observed": False,
            "smoke_repeated": False,
            "hardware_operations_issued": [],
            "hardware_command_count": 0,
            "metric": True,
        },
    ]
    return [_finish_row(row) for row in rows]


def reduce_board_rows(rows: Any) -> JsonDict:
    """Independently reduce raw rows without using the artifact's claimed score."""

    values = [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
    by_board = {row.get("board"): row for row in values if isinstance(row.get("board"), str)}
    row_hashes_match = all(
        row.get("row_sha256")
        == exp7244.exp7217.sha256_json(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
        for row in values
    )
    complete = (
        len(values) == 3
        and set(by_board) == {"KV260", "GateMate", "PolarFire"}
        and row_hashes_match
        and all(row.get("latest_receipt_authenticated") is True for row in values)
        and all(bool(row.get("exact_next_condition")) for row in values)
        and all(row.get("hardware_operations_issued") == [] for row in values)
        and all(row.get("hardware_command_count") == 0 for row in values)
        and by_board["KV260"].get("terminal_criterion") == KV260_TERMINAL_CRITERION
        and by_board["KV260"].get("terminal_criterion_met") is True
        and by_board["PolarFire"].get("terminal_criterion") == POLARFIRE_TERMINAL_CRITERION
        and by_board["PolarFire"].get("terminal_criterion_met") is True
        and by_board["PolarFire"].get("processor_class") == "cpu"
        and by_board["PolarFire"].get("programmable_logic_sampling_observed") is False
        and by_board["GateMate"].get("disposition")
        in {"blocked_changed_physical_state", "changed_physical_state_future_action_enabled"}
    )
    return {
        "board_count": len(values),
        "board_names": sorted(str(name) for name in by_board),
        "row_hashes_match": row_hashes_match,
        "hardware_command_count": sum(int(row.get("hardware_command_count", 0)) for row in values),
        "board_disposition_complete_score": int(complete),
    }


def _base_artifact(
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
    phase_spans: Mapping[str, float],
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
) -> JsonDict:
    """Create all required fields before choosing complete or blocked state."""

    summary = _gate_summary(checks)
    summary["board_blocks"] = []
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "spec_refs": [
            "REQ-ISING-7258",
            "SCENARIO-ISING-7258-PREFLIGHT",
            "SCENARIO-ISING-7258-BOARDS",
            "SCENARIO-ISING-7258-GATEMATE",
            "SCENARIO-ISING-7258-FIXTURES",
            "SCENARIO-ISING-7258-ARTIFACT",
        ],
        "status": "blocked_external_precondition",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": [dict(row) for row in checks],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "model_invocation_count": 0,
        "model_load_count": 0,
        "generation_count": 0,
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node(),
        "duration_s": duration_s,
        "phase_spans_s": dict(phase_spans),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(hashes),
        "rows": [],
        "sample_size_budget": {
            "planned": 3,
            "attempted": 0,
            "completed": 0,
            "censored": 3,
            "independent_units_planned": 3,
            "independent_units_completed": 0,
            "stopping_rule": "one read-only disposition per attached board; zero board commands",
        },
        "acceptance_gate_results": {},
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "honest_verdict": "blocked_external_precondition",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "baseline_validation_failures": [],
        "board_disposition_complete_score": 0,
        "board_rows": [],
        "operator_state_receipt": {},
        "hardware_operations_issued": [],
        "external_actions": {
            "purchases": [],
            "signups": [],
            "firmware_actions": [],
            "uploads": [],
            "publications": [],
            "external_messages": [],
        },
    }


def _gate_result(principle: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Give each frozen gate the same independently readable fields."""

    return {
        "principle": principle,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def build_artifact(
    root: Path,
    paths: ExperimentPaths,
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> JsonDict:
    """Aggregate authenticated inputs in memory without issuing a board command."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: dict[str, float] = {}

    phase_started = time.monotonic()
    progress(0, "start", "authenticate sources before aggregation")
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
    progress(2, "start", "search operator-authored GateMate physical-state receipts")
    operator = search_gatemate_operator_receipts(
        root, paths.operator_search, candidate_paths=candidate_paths
    )
    hashes[_display_path(root, paths.operator_search)] = operator["search_receipt_hash"]
    spans["phase_2_operator_receipt"] = time.monotonic() - phase_started
    progress(
        2,
        "end",
        f"accepted-receipts={int(operator['exists'])} hardware-commands=0",
    )

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
            "command": "internal reduce_board_rows(board_rows)",
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
                "stopping_rule": (
                    "one read-only disposition per attached board; zero board commands"
                ),
            },
            "acceptance_gate_results": {
                "exp7244_current_receipt_authenticated": _gate_result(
                    "Use the shipped validator and current immutable hashes.", True, True, True
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
                    "A later operator GateMate physical-state receipt enables one future detect. "
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
    """Recompute the new identity and rows while the shipped reader owns old evidence."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        return ["missing_fields:" + ",".join(missing)]
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(artifact.get("schema") != SCHEMA, "schema")
    add(
        artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("task_id") != TASK_ID
        or artifact.get("milestone") != MILESTONE,
        "identity",
    )
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles")
    for field in ("started_at_utc", "completed_at_utc"):
        try:
            datetime.fromisoformat(str(artifact.get(field)))
        except ValueError:
            add(True, field)
    duration = artifact.get("duration_s")
    spans = artifact.get("phase_spans_s")
    add(not isinstance(duration, (int, float)) or duration < 0, "duration_s")
    add(
        not isinstance(spans, Mapping)
        or any(not isinstance(value, (int, float)) or value < 0 for value in spans.values())
        or (isinstance(duration, (int, float)) and sum(spans.values()) > duration + 1e-6),
        "phase_spans_s",
    )
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("model_invocation_count") != 0
        or artifact.get("model_load_count") != 0
        or artifact.get("generation_count") != 0,
        "model_declaration",
    )
    add(
        artifact.get("execution_venue") != "host" or not artifact.get("execution_host"),
        "execution_identity",
    )
    add(artifact.get("verifier_is_oracle") is not False, "verifier_authority")
    add(artifact.get("hardware_operations_issued") != [], "hardware_operations")
    try:
        checksum_ok = artifact.get("reproducibility_checksum") == artifact_checksum(artifact)
    except (TypeError, ValueError):
        checksum_ok = False
    add(not checksum_ok, "reproducibility_checksum")

    if artifact.get("verdict_class") == "blocked":
        summary = artifact.get("gate_check_summary")
        add(artifact.get("status") != "blocked_external_precondition", "blocked_status")
        add(
            artifact.get("inference_substrate") != "blocked_no_run"
            or artifact.get("inference_substrate_class") != "blocked_no_run",
            "blocked_substrate",
        )
        add(artifact.get("rows") != [] or artifact.get("board_rows") != [], "blocked_rows")
        add(artifact.get("board_disposition_complete_score") != 0, "blocked_score")
        add(
            not isinstance(summary, Mapping)
            or summary.get("passed") is not False
            or any(
                summary.get(name) is None
                for name in (
                    "failed_check",
                    "upstream",
                    "field",
                    "expected_value",
                    "observed_value",
                )
            ),
            "blocked_gate_summary",
        )
        return errors

    add(
        artifact.get("status") != "complete"
        or artifact.get("verdict_class") != "positive"
        or not str(artifact.get("honest_verdict", "")).startswith("complete:"),
        "terminal_state",
    )
    add(
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation",
        "substrate",
    )
    rows = artifact.get("board_rows")
    reduced = reduce_board_rows(rows)
    add(
        artifact.get("rows") != rows
        or reduced["board_disposition_complete_score"] != 1
        or artifact.get("board_disposition_complete_score") != 1
        or reduced["hardware_command_count"] != 0,
        "board_reducer",
    )
    by_board = (
        {row.get("board"): row for row in rows if isinstance(row, Mapping)}
        if isinstance(rows, list)
        else {}
    )
    operator = artifact.get("operator_state_receipt")
    operator_ok = (
        isinstance(operator, Mapping)
        and operator.get("cutoff_experiment") == "Exp6559"
        and operator.get("cutoff_date") == GATEMATE_CUTOFF_DATE
        and operator.get("hardware_operations_issued") == []
        and operator.get("hardware_command_count") == 0
        and bool(operator.get("search_receipt_path"))
        and bool(operator.get("search_receipt_hash"))
    )
    if operator_ok and operator.get("exists") is True:
        operator_ok = bool(
            operator.get("source_path")
            and operator.get("author_evidence") is True
            and operator.get("date_evidence")
            and operator.get("evidence_hash")
            and operator.get("changed_conditions")
            and operator.get("newly_enabled_next_action") == GATEMATE_FUTURE_ACTION
            and by_board.get("GateMate", {}).get("disposition")
            == "changed_physical_state_future_action_enabled"
        )
    elif operator_ok:
        operator_ok = (
            operator.get("source_path") is None
            and operator.get("evidence_hash") is None
            and operator.get("observed_missing_receipt") == MISSING_RECEIPT
            and operator.get("newly_enabled_next_action") is None
            and by_board.get("GateMate", {}).get("disposition") == "blocked_changed_physical_state"
        )
    add(not operator_ok, "operator_state_receipt")
    budget = artifact.get("sample_size_budget")
    add(
        not isinstance(budget, Mapping)
        or any(
            budget.get(name) != value
            for name, value in {
                "planned": 3,
                "attempted": 3,
                "completed": 3,
                "censored": 0,
                "independent_units_planned": 3,
                "independent_units_completed": 3,
            }.items()
        ),
        "sample_size_budget",
    )
    gates = artifact.get("acceptance_gate_results")
    add(
        not isinstance(gates, Mapping)
        or len(gates) != 7
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"principle", "expected", "observed", "passed"}
            or row.get("passed") is not True
            for row in gates.values()
        ),
        "acceptance_gate_results",
    )
    summary = artifact.get("gate_check_summary")
    add(
        not isinstance(summary, Mapping)
        or summary.get("passed") is not True
        or not isinstance(summary.get("board_blocks"), list),
        "gate_check_summary",
    )
    validation = artifact.get("validation_receipts")
    add(
        not isinstance(validation, list)
        or not validation
        or any(
            not isinstance(row, Mapping)
            or not row.get("command")
            or row.get("exit_code") != 0
            or not str(row.get("log_hash", "")).startswith("sha256:")
            for row in validation
        ),
        "validation_receipts",
    )
    baseline = artifact.get("baseline_validation_failures")
    add(not isinstance(baseline, list), "baseline_validation_failures")
    if root is not None:
        hashes = artifact.get("source_artifact_hashes")
        add(
            not isinstance(hashes, Mapping)
            or any(
                not (Path(path) if Path(path).is_absolute() else root / path).is_file()
                or sha256_file(Path(path) if Path(path).is_absolute() else root / path) != digest
                for path, digest in hashes.items()
            ),
            "source_artifact_hashes",
        )
    return errors


def atomic_write(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Publish only an independently valid terminal receipt."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp7258 artifact: {errors}")
    _atomic_json(path, artifact)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "atomic_replace": True,
    }


def run_experiment(root: Path, paths: ExperimentPaths) -> JsonDict:
    """Build, independently reduce, and atomically publish the host receipt."""

    artifact = build_artifact(root, paths)
    progress(5, "before", "final independent raw-row reducer and artifact validation")
    errors = validate_artifact(artifact, root=root)
    progress(5, "after", f"validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7258 artifact: {errors}")
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

    progress(0, "entry", "Exp7258 host-only aggregation; model-loads=0 hardware-commands=0")
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
