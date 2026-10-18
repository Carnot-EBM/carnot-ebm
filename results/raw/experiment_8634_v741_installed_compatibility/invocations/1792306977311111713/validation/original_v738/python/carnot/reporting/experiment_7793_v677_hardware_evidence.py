"""Reduce dated board evidence without promoting old verdicts or vendor estimates."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


INVENTORY = "results/experiment_7751_v674_hardware_continuity.json"
INVENTORY_RAW = "results/raw/experiment_7751_v674_hardware_continuity/rows.json"
HISTORY = "results/experiment_7779_v676_hardware_evidence.json"
HISTORY_RAW = "results/raw/experiment_7779_v676_hardware_evidence/rows.json"
SERVICE = "results/experiment_7792_v677_service_cost.json"
PRE_GATE = "results/experiment_7792_service_cost.json"
BOARDS = ("KV260", "PolarFire", "GateMate")
CONTEXT = ("AMD XDNA NPU", "Extropic Z1T/TSU", "Kona")


def progress(started: float, phase: str, event: str, units: int) -> None:
    """Flush real elapsed time so a stalled child is visible to an operator."""
    print(
        f"[exp7793] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def _check(
    upstream: str, path: str, digest: str | None, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep both operands so a missing external input differs from a scientific null."""
    return {
        "upstream_id": upstream,
        "artifact_path": path,
        "artifact_hash": digest,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def _read(root: Path, rel: str, scope: str, fields: list[str], sources: dict[str, Any]) -> Any:
    """Bind an imported field list to the exact producer bytes or record absence."""
    path = root / rel
    digest = sha256_file(path) if path.is_file() else None
    value = json.loads(path.read_text()) if digest else None
    sources[rel] = {
        "path": rel,
        "sha256": digest,
        "date": value.get("run_date") if isinstance(value, dict) else None,
        "imported_fields": fields,
        "scope": scope,
        "eligible": digest is not None,
    }
    return value


def _git_blob(root: Path, blob: str | None) -> str | None:
    """Read a named immutable Git object; today's reference file has no authority."""
    if not blob:
        return None
    started = time.monotonic()
    print(
        f"[exp7793] phase=git_blob event=before_subprocess elapsed_s={time.monotonic() - started:.3f} completed_units=0",
        file=sys.stderr,
        flush=True,
    )
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", blob],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        print(
            f"[exp7793] phase=git_blob event=after_subprocess elapsed_s={time.monotonic() - started:.3f} completed_units=1",
            file=sys.stderr,
            flush=True,
        )
        return None
    print(
        f"[exp7793] phase=git_blob event=after_subprocess elapsed_s={time.monotonic() - started:.3f} completed_units=1",
        file=sys.stderr,
        flush=True,
    )
    return "sha256:" + hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def _source_hash(root: Path, rel: str) -> str | None:
    """Hash a local immutable artifact only when it exists as a regular file."""
    path = root / rel
    return sha256_file(path) if path.is_file() else None


def _reduce(
    root: Path, date: str
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]
]:
    """SCENARIO-REPORT-7793-CUSTODY: check both raw histories before board rows."""
    sources: dict[str, Any] = {}
    inventory = _read(
        root, INVENTORY, "historical_inventory", ["rows", "hardware_continuity"], sources
    )
    inventory_raw = _read(root, INVENTORY_RAW, "historical_raw_rows", [], sources)
    history = _read(
        root, HISTORY, "disqualified_historical_context", ["rows", "validation_receipts"], sources
    )
    sources[HISTORY]["eligible"] = False
    history_raw = _read(root, HISTORY_RAW, "disqualified_historical_raw_rows", [], sources)
    service = _read(root, SERVICE, "declared_science_producer", ["service_fractions"], sources)
    checks = [
        _check(
            "Exp7751",
            INVENTORY,
            sources[INVENTORY]["sha256"],
            "raw_rows",
            True,
            bool(inventory and inventory_raw == inventory.get("rows")),
        ),
        _check(
            "Exp7779",
            HISTORY,
            sources[HISTORY]["sha256"],
            "raw_rows",
            True,
            bool(history and history_raw == history.get("rows")),
        ),
        _check(
            "Exp7779",
            HISTORY,
            sources[HISTORY]["sha256"],
            "verdict_class",
            "disqualified",
            history.get("verdict_class") if history else None,
        ),
    ]
    old_rows = {row.get("board"): row for row in history_raw or [] if isinstance(row, dict)}
    inventory_rows = {
        row.get("substrate"): row for row in inventory_raw or [] if isinstance(row, dict)
    }
    old_inventory = inventory.get("hardware_continuity", {}) if inventory else {}
    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for name in (*BOARDS, *CONTEXT):
        old = old_inventory.get(name, {})
        old_row = old_rows.get(name, {})
        expected = old.get("evidence_hash")
        path = (
            old_row.get("source_path")
            if name in BOARDS
            else f"git:{(old_row.get('git_identity') or {}).get('blob')}"
        )
        blob = (old_row.get("git_identity") or {}).get("blob")
        observed = _source_hash(root, path) if name in BOARDS and path else _git_blob(root, blob)
        checks.append(
            _check(
                f"Exp7751:{name}",
                path or "missing",
                observed,
                "historical_evidence_sha256",
                expected,
                observed,
            )
        )
        checks.append(
            _check(
                f"Exp7779:{name}",
                path or "missing",
                observed,
                "raw_expected_hash",
                expected,
                old_row.get("expected_hash"),
            )
        )
        if name in BOARDS:
            checks.append(
                _check(
                    f"Exp7751:{name}",
                    INVENTORY_RAW,
                    sources[INVENTORY_RAW]["sha256"],
                    "inventory_raw_evidence_sha256",
                    expected,
                    inventory_rows.get(name, {}).get("evidence_hash"),
                )
            )
            inventory_digest = (
                (inventory or {})
                .get("source_artifact_hashes", {})
                .get(path or "", {})
                .get("sha256")
            )
            checks.append(
                _check(
                    f"Exp7751:{name}",
                    path or "missing",
                    observed,
                    "inventory_source_sha256",
                    expected,
                    inventory_digest,
                )
            )
            checks.append(
                _check(
                    f"Exp7779:{name}",
                    path or "missing",
                    observed,
                    "raw_source_sha256",
                    expected,
                    old_row.get("source_hash"),
                )
            )
            checks.append(
                _check(
                    f"Exp7751:{name}",
                    path or "missing",
                    observed,
                    "raw_venue",
                    old.get("venue"),
                    old_row.get("venue"),
                )
            )
        if name in BOARDS and path:
            sources[path] = {
                "path": path,
                "sha256": observed,
                "expected_sha256": expected,
                "date": old.get("evidence_date"),
                "imported_fields": ["board", "venue"],
                "scope": "dated_board_receipt",
                "eligible": observed == expected,
            }
        if name in CONTEXT and blob:
            sources[path] = {
                "path": path,
                "sha256": observed,
                "expected_sha256": expected,
                "date": old.get("evidence_date"),
                "imported_fields": ["historical_disclosure"],
                "scope": "immutable_git_blob",
                "eligible": observed == expected,
            }
        resolved = (
            observed is not None
            and observed == expected
            and old_row.get("expected_hash") == expected
        )
        if not resolved:
            unresolved.append(
                {"board": name, "path": path, "expected_hash": expected, "observed_hash": observed}
            )
        state = (
            "evidence_unresolved"
            if not resolved
            else "historical_fabric_k_max<=5"
            if name == "KV260"
            else "historical_linux_cpu_only"
            if name == "PolarFire"
            else "blocked_physical_change_0xffffffff"
            if name == "GateMate"
            else "no_local_execution"
        )
        row = {
            "board": name,
            "arm": "historical_accounting",
            "seed": None,
            "state": state,
            "historical_verdict": old_row.get("historical_verdict"),
            "last_authenticated_evidence_date": old.get("evidence_date") if resolved else None,
            "source_path": path,
            "source_hash": observed,
            "expected_hash": expected,
            "venue": old.get("venue"),
            "scope": old.get("supported_workload"),
            "processor_class": "linux_cpu"
            if name == "PolarFire"
            else "fpga_fabric"
            if name == "KV260"
            else "none",
            "k_max": 5 if name == "KV260" else None,
            "blocker": "0xffffffff" if name == "GateMate" else old.get("blocker"),
            "next_missing_prerequisite": old.get("changed_prerequisite"),
            "acquisition_relevance": "defer: complete service cost and local benefit unmeasured",
            "service_fraction": None,
            "metric": None,
            "started": False,
            "completed": False,
            "excluded": not resolved or name in CONTEXT or name == "GateMate",
            "censored": False,
            "raw_provenance": {"inventory": INVENTORY_RAW, "history": HISTORY_RAW},
        }
        rows.append(row)
    eligible_service = bool(
        service
        and service.get("experiment_id") == 7792
        and service.get("service_evidence_ready_score") == 1
        and service.get("flagged_adversarial") is False
        and service.get("verdict_class") in {"positive", "null", "circular_positive"}
        and isinstance(service.get("service_fractions"), dict)
    )
    checks.append(
        _check(
            "Exp7792",
            SERVICE,
            sources[SERVICE]["sha256"],
            "eligible_service_producer",
            True,
            eligible_service,
        )
    )
    sources[SERVICE]["eligible"] = eligible_service
    if eligible_service:
        fractions = service["service_fractions"]
        for row in rows:
            row["service_fraction"] = fractions.get(row["board"])
    history_checks = history.get("validation_receipts", {}).get("checks", []) if history else []
    history_full_suite_exit = next(
        (
            item.get("exit_code")
            for item in history_checks
            if item.get("name") == "full_python_suite"
        ),
        None,
    )
    context = {
        "eligible_service": eligible_service,
        "history_full_suite_exit": history_full_suite_exit,
        "historical_verdict": history.get("honest_verdict") if history else None,
        "run_date": date,
    }
    return rows, checks, sources, unresolved, context


def build_candidate(root: Path, run_date: str) -> dict[str, Any]:
    """Reduce only present source bytes; absent science ends in a blocked state."""
    started = time.monotonic()
    root = root.resolve()
    progress(started, "preflight", "start", 0)
    rows, checks, sources, unresolved, context = _reduce(root, run_date)
    progress(started, "preflight", "complete", len(checks))
    board_rows = [row for row in rows if row["board"] in BOARDS]
    custody_ok = not unresolved and all(
        check["passed"]
        for check in checks
        if check["upstream_id"].startswith(("Exp7751", "Exp7779"))
    )
    service_measured = context["eligible_service"]
    verdict = (
        "complete_blocked_historical_evidence"
        if not custody_ok
        else "complete_blocked_missing_service_evidence"
        if not service_measured
        else "complete_null_hardware_accounting_only"
    )
    ended = time.monotonic()
    source_digest = canonical_hash({key: value.get("sha256") for key, value in sources.items()})
    principles = {
        "experiment_id": "Each record needs a unique owner.",
        "milestone": "The milestone binds the current acceptance contract.",
        "run_date": "Dates separate historical and current observations.",
        "honest_verdict": "Unchanged external inputs must not consume retries.",
        "verdict_class": "Claim strength travels with the result.",
        "flagged_adversarial": "Invalid evidence cannot open a downstream gate.",
        "gate_check_summary": "A missing producer differs from a scientific null.",
        "rows": "Individual units make the headline reproducible.",
        "acceptance_gate_results": "Working custody does not establish benefit.",
        "duration_s": "Only actual elapsed work determines duration.",
        "phase_spans": "Real phase times expose stalled work.",
        "random_seed": "A deterministic audit draws no random sample.",
        "reproducibility_checksum": "Exact code, inputs, roles and parameters bind replay.",
        "sample_size_budget": "Historical views do not multiply independent boards.",
        "source_artifact_hashes": "Old bytes cannot be replaced by current prose.",
        "preconditions_checked": "Cheap custody and resource failures precede computation.",
        "validation_receipts": "Required checks control readiness.",
        "verifier_is_oracle": "Fixture truth does not establish semantic accuracy.",
        "claim_scope": "Historical access does not prove current speedup.",
        "inference_substrate": "Aggregation invokes no model or board.",
        "inference_substrate_class": "Duration floors follow actual invoked work.",
        "MODEL_SPECS": "Cited models are not current invocations.",
        "model_specs": "Only current calls enter model metadata.",
        "model_invocation_counts": "Loaded-file hashes and tokens are zero without calls.",
        "board_rows": "Every attached board retains its physical prerequisite.",
        "evidence_unresolved": "Unrecovered bytes stay unresolved with both hashes.",
        "service_measured": "Vendor estimates cannot open a local speed gate.",
        "hardware_continuity_accounted": "Inventory completeness is not fresh execution.",
    }
    artifact: dict[str, Any] = {
        "schema": "carnot.experiment_7793.v1",
        "experiment_id": 7793,
        "milestone": "2026.09.677",
        "run_date": run_date,
        "honest_verdict": verdict,
        "verdict_class": "blocked" if verdict.startswith("complete_blocked") else "null",
        "flagged_adversarial": False,
        "gate_check_summary": [check for check in checks if not check["passed"]],
        "rows": rows,
        "rows_checksum": canonical_hash(rows),
        "board_rows": board_rows,
        "evidence_unresolved": unresolved,
        "service_measured": service_measured,
        "hardware_continuity_accounted": len(board_rows) == len(BOARDS),
        "acceptance_gate_results": {
            "validity": custody_ok,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": ended - started,
        "phase_spans": [
            {
                "phase": "preflight_and_reduction",
                "start_monotonic_s": started,
                "end_monotonic_s": ended,
                "duration_s": ended - started,
                "completed_units": len(rows),
            }
        ],
        "random_seed": None,
        "sample_size_budget": {
            "intended": 6,
            "eligible": sum(not row["excluded"] for row in rows),
            "started": 0,
            "completed": 0,
            "excluded": sum(row["excluded"] for row in rows),
            "censored": 0,
            "effective_independent_n": 2 if custody_ok else 0,
        },
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "root": str(root),
            "custody_checks": checks,
            "backend": "host_cpu_read_only",
            "board_operations_issued": [],
            "current_board_reachability": "not_probed",
            "historical_exp7779_verdict": context["historical_verdict"],
            "historical_exp7779_full_suite_exit": context["history_full_suite_exit"],
            "service_pre_gate_is_producer": False,
            "declared_science_producer": SERVICE,
            "conductor_pre_gate_receipt": {
                "path": PRE_GATE,
                "sha256": _source_hash(root, PRE_GATE),
                "is_science_producer": False,
            },
            "resource_status": "local files and Git object only",
        },
        "validation_receipts": {
            "frozen_scope_path": "/tmp/carnot-7793/frozen_scope.json",
            "checks": [],
            "required_checks_passed": False,
            "cold_replay": None,
            "terminal_readers": [],
        },
        "verifier_is_oracle": False,
        "claim_scope": {
            "hardware_advantage": "unmeasured",
            "probability": "unmeasured",
            "decision": "unmeasured",
            "learning": "unmeasured",
            "NPU": "no qualified local device run",
            "TSU": "vendor projection only; no local silicon or SDK run",
            "acquisition": "defer expensive board or NPU install until complete service fractions and local benefit are measured",
        },
        "literature_to_future_requirements": [
            {
                "source": "https://extropic.ai/writing/z1t",
                "historical_git_blob": "7766c92e07b4cee74ece8d9e0ddfbea14fb69173",
                "requirement": "Measure local preparation, transfer, sampling, dense readout and durable result on an accessible TSU.",
            },
            {
                "source": "https://doi.org/10.1038/s41467-026-75119-0",
                "historical_git_blob": "7766c92e07b4cee74ece8d9e0ddfbea14fb69173",
                "requirement": "Measure sparse Boltzmann setup, transfer, readout and complete local service cost before proposing FPGA acquisition.",
            },
        ],
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "actual_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"loads": 0, "generations": 0, "tokens": 0, "loaded_files": []},
        "hardware_operations_issued": [],
        "hardware_advantage_claimed": False,
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "code": sha256_file(Path(__file__)),
            "cli_code": sha256_file(
                Path(__file__).resolve().parents[3]
                / "scripts/experiments/experiment_7793_v677_hardware_evidence.py"
            ),
            "data": source_digest,
            "roles": "read_only_board_reducer",
            "parameters": {"boards": BOARDS, "context": CONTEXT},
            "seed": None,
        }
    )
    artifact["field_principles"] = {
        key: principles.get(key, "Exact provenance and scope remain visible.") for key in artifact
    }
    artifact["field_principles"]["field_principles"] = (
        "The rationale for every field and gate travels with the result."
    )
    for gate in artifact["acceptance_gate_results"]:
        artifact["field_principles"][f"{gate}_gate"] = (
            "Unmeasured benefit remains null; invalid work cannot qualify."
        )
    return artifact


def cold_reduce(candidate: Path, root: Path) -> dict[str, Any]:
    """SCENARIO-REPORT-7793-REPLAY: regenerate every row from exact source bytes."""
    artifact = json.loads(candidate.read_text())
    rows, checks, sources, unresolved, context = _reduce(root.resolve(), artifact["run_date"])
    for name, receipt in artifact["source_artifact_hashes"].items():
        actual = sources.get(name, {}).get("sha256")
        if actual != receipt.get("sha256") or receipt.get("eligible") != sources.get(name, {}).get(
            "eligible"
        ):
            raise ValueError(f"source_hash_mismatch:{name}")
    if set(sources) != set(artifact["source_artifact_hashes"]):
        raise ValueError("source_set_mismatch")
    if artifact["rows"] != rows or artifact["rows_checksum"] != canonical_hash(rows):
        raise ValueError("row_mismatch")
    if artifact["board_rows"] != [row for row in rows if row["board"] in BOARDS]:
        raise ValueError("board_row_mismatch")
    if (
        artifact["evidence_unresolved"] != unresolved
        or artifact["service_measured"] != context["eligible_service"]
    ):
        raise ValueError("summary_mismatch")
    if [check for check in checks if not check["passed"]] != [
        check
        for check in artifact["gate_check_summary"]
        if not check["upstream_id"].startswith("Exp7793:validation")
    ]:
        raise ValueError("gate_mismatch")
    return {
        "row_count": len(rows),
        "rows_checksum": canonical_hash(rows),
        "source_count": len(sources),
    }


def main(argv: list[str] | None = None) -> int:
    """Prepare private bytes, replay them, or publish a validated exact candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    root = args.root.resolve()
    progress(started, "entrypoint", "start", 0)
    if args.cold_replay:
        progress(started, "cold_replay", "before_reduction", 0)
        print(json.dumps(cold_reduce(args.cold_replay, root), sort_keys=True), flush=True)
        progress(started, "cold_replay", "after_reduction", 6)
        return 0
    if args.prepare:
        artifact = build_candidate(root, args.date)
        atomic_json(args.prepare, artifact)
        progress(started, "prepare", "complete", len(artifact["rows"]))
        return 0
    candidate_path = Path(
        os.environ.get("CARNOT_7793_CANDIDATE", "/tmp/carnot-7793/candidate.json")
    )
    artifact = json.loads(candidate_path.read_text())
    if (
        artifact["run_date"] != args.date
        or not artifact["validation_receipts"]["required_checks_passed"]
    ):
        raise ValueError("validated_candidate_required")
    if any(check["exit_code"] != 0 for check in artifact["validation_receipts"]["checks"]):
        raise ValueError("failed_validation_cannot_publish")
    progress(started, "publish", "before_cold_replay", 0)
    cold_reduce(candidate_path, root)
    progress(started, "publish", "after_cold_replay", len(artifact["rows"]))
    output = root / "results/experiment_7793_v677_hardware_evidence.json"
    progress(started, "publish", "before_atomic_write", len(artifact["rows"]))
    atomic_json(output, artifact)
    progress(started, "publish", "after_atomic_write", len(artifact["rows"]))
    return 0
