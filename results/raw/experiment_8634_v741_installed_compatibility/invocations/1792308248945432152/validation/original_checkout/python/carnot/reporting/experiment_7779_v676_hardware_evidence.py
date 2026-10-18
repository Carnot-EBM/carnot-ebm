"""Authenticate historical hardware claims without binding them to mutable prose.

REQ-REPORT-7779 keeps dated board measurements separate from current access,
literature, and missing complete-service science. It issues no board command.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


INVENTORY = "results/experiment_7751_v674_hardware_continuity.json"
PRIOR_COST = "results/experiment_7765_v675_service_cost.json"
SERVICE = "results/experiment_7778_v676_service_cost.json"
OUTPUT = "results/experiment_7779_v676_hardware_evidence.json"
RAW = "results/raw/experiment_7779_v676_hardware_evidence"
SCOPE = f"{RAW}/validation_scope.json"
BOARDS = ("KV260", "PolarFire", "GateMate")
CONTEXT = ("AMD XDNA NPU", "Extropic Z1T/TSU", "Kona")
PRINCIPLES = {
    "experiment_id": "An artifact has one current owner.",
    "honest_verdict": "An unchanged external block ends this attempt.",
    "verdict_class": "The claim class travels with evidence.",
    "flagged_adversarial": "Invalid evidence closes downstream gates.",
    "gate_check_summary": "A missing producer differs from a failed threshold.",
    "rows": "Raw units permit independent recomputation.",
    "acceptance_gate_results": "A working audit is not acceleration benefit.",
    "duration_s": "Only measured work time is recorded.",
    "phase_spans": "Each phase has an actual time boundary.",
    "random_seed": "This read-only reduction has no stochastic draw.",
    "reproducibility_checksum": "Exact input and code bytes identify this run.",
    "sample_size_budget": "Repeated views do not increase independent N.",
    "source_artifact_hashes": "Upstream producers retain distinct custody.",
    "preconditions_checked": "Access and validity precede any computation.",
    "validation_receipts": "Required checks control readiness.",
    "verifier_is_oracle": "Historical execution is not semantic verification.",
    "claim_scope": "Only the authenticated venue and workload qualify.",
    "inference_substrate": "Host aggregation makes no model invocation.",
    "MODEL_SPECS": "A cited model is not a current model call.",
    "hardware_continuity": "Prior board access does not prove current speedup.",
    "immutable_evidence_ready_score": "Original bytes or explicit unresolved exclusion are required.",
}


def progress(started: float, phase: str, event: str, units: int) -> None:
    """Print a flushed real-time boundary for each owned phase and child."""
    print(
        f"[exp7779] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def _check(
    upstream: str, path: str, digest: str | None, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Retain both operands, including a missing observed value."""
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


def resolve_historical_blob(root: Path, expected: str) -> dict[str, str] | None:
    """Find the original reference bytes by digest in a retained Git commit."""
    started = time.monotonic()
    argv = ["git", "-C", str(root), "log", "--all", "--format=%H", "--", "research-references.md"]
    progress(started, "git_history", "before_subprocess", 0)
    history = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=30)
    progress(started, "git_history", "after_subprocess", 0)
    if history.returncode:
        return None
    for count, commit in enumerate(history.stdout.splitlines(), 1):
        progress(started, "git_blob", "before_subprocess", count - 1)
        blob = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:research-references.md"],
            capture_output=True,
            check=False,
            timeout=30,
        )
        progress(started, "git_blob", "after_subprocess", count)
        if blob.returncode == 0 and "sha256:" + hashlib.sha256(blob.stdout).hexdigest() == expected:
            progress(started, "git_blob_id", "before_subprocess", count)
            blob_id = subprocess.run(
                ["git", "-C", str(root), "rev-parse", f"{commit}:research-references.md"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout.strip()
            progress(started, "git_blob_id", "after_subprocess", count)
            return {"commit": commit, "blob": blob_id, "sha256": expected}
    return None


def _source(
    root: Path, name: str, fields: list[str], scope: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read a declared JSON producer without replacing it with nearby prose."""
    path = root / name
    receipt = {
        "path": name,
        "sha256": sha256_file(path) if path.is_file() else None,
        "date": None,
        "imported_fields": fields,
        "scope": scope,
        "eligible": False,
    }
    if not path.is_file():
        return receipt, None
    value = json.loads(path.read_text())
    receipt["date"] = value.get("run_date")
    return receipt, value


def build_audit(root: Path, run_date: str, snapshots: Path) -> dict[str, Any]:
    """SCENARIO-REPORT-7779-CUSTODY: account for six dated substrates."""
    started = time.monotonic()
    progress(started, "preflight", "start", 0)
    sources: dict[str, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    for name, fields, scope in (
        (INVENTORY, ["hardware_continuity", "source_artifact_hashes"], "dated_inventory"),
        (PRIOR_COST, ["service_evidence_ready_score", "gate_check_summary"], "prior_cost_context"),
        (
            SERVICE,
            ["service_evidence_ready_score", "hardware_opportunity"],
            "declared_service_producer",
        ),
    ):
        sources[name], value = _source(root, name, fields, scope)
        if name == INVENTORY:
            inventory = value
        elif name == PRIOR_COST:
            prior_cost = value
        else:
            service = value
    checks.append(
        _check(
            "Exp7751",
            INVENTORY,
            sources[INVENTORY]["sha256"],
            "hardware_continuity_complete_score",
            1,
            inventory.get("hardware_continuity_complete_score") if inventory else None,
        )
    )
    continuity = inventory.get("hardware_continuity", {}) if inventory else {}
    original_hash = continuity.get("Kona", {}).get("evidence_hash")
    historical = resolve_historical_blob(root, original_hash) if original_hash else None
    rows: list[dict[str, Any]] = []
    for name in (*BOARDS, *CONTEXT):
        item = continuity.get(name, {})
        expected = item.get("evidence_hash")
        if name in BOARDS:
            matches = (
                [
                    rel
                    for rel, receipt in inventory.get("source_artifact_hashes", {}).items()
                    if receipt.get("sha256") == expected
                ]
                if inventory
                else []
            )
            path = matches[0] if matches else f"unresolved:{name}"
            actual = sha256_file(root / path) if (root / path).is_file() else None
            identity = None
        else:
            path = "research-references.md"
            actual = historical["sha256"] if historical and expected == original_hash else None
            identity = historical
        check = _check(
            f"Exp7751:{name}", path, actual, "historical_evidence_sha256", expected, actual
        )
        checks.append(check)
        state = (
            "evidence_unresolved"
            if not check["passed"]
            else "historical_fabric_k_max<=5"
            if name == "KV260"
            else "historical_linux_cpu_only"
            if name == "PolarFire"
            else "blocked_physical_change_0xffffffff"
            if name == "GateMate"
            else "no_local_execution"
        )
        scope = (
            "quadratic Ising FPGA fabric k_max<=5; no complete-service speedup"
            if name == "KV260"
            else "hash-matched PolarFire Linux CPU dispatch only; no FPGA sampling"
            if name == "PolarFire"
            else "JTAG all-ones 0xffffffff; no execution"
            if name == "GateMate"
            else item.get("supported_workload", "unresolved disclosure")
        )
        row = {
            "unit_type": "substrate",
            "board": name,
            "arm": "historical_custody",
            "seed": None,
            "source_path": path,
            "source_hash": actual,
            "expected_hash": expected,
            "git_identity": identity,
            "evidence_date": item.get("evidence_date"),
            "historical_verdict": state,
            "state": state,
            "venue": item.get("venue"),
            "scope": scope,
            "blocker": item.get("blocker"),
            "unresolved_prerequisite": item.get("changed_prerequisite"),
            "next_action": item.get("bounded_next_experiment"),
            "measured_service_opportunity": None,
            "started": True,
            "completed": True,
            "excluded": name in CONTEXT or not check["passed"],
            "censored": False,
            "metric": None,
        }
        rows.append(row)
        if name in BOARDS:
            sources[path] = {
                "path": path,
                "sha256": actual,
                "date": item.get("evidence_date"),
                "imported_fields": ["board", "venue"],
                "scope": "dated_board_receipt",
                "eligible": check["passed"],
            }
    if historical:
        sources[f"git:{historical['commit']}:research-references.md"] = {
            "path": "research-references.md",
            "sha256": historical["sha256"],
            "git_blob": historical["blob"],
            "date": "20260926",
            "scope": "original_literature",
            "imported_fields": ["historical_disclosure"],
            "eligible": True,
        }
    for name in ("research-references.md", "research-hardware-wishlist.md"):
        source = root / name
        snap = snapshots / name
        snap.parent.mkdir(parents=True, exist_ok=True)
        current = source.read_bytes() if source.is_file() else None
        if current is not None and not snap.exists():
            snap.write_bytes(current)
        observed = sha256_file(snap) if snap.is_file() else None
        current_hash = (
            "sha256:" + hashlib.sha256(current).hexdigest() if current is not None else None
        )
        checks.append(
            _check("current_literature", name, observed, "snapshot_sha256", current_hash, observed)
        )
        sources[name] = {
            "path": name,
            "sha256": current_hash,
            "snapshot_path": str(snap.relative_to(root)),
            "snapshot_sha256": observed,
            "date": run_date,
            "imported_fields": ["current_context"],
            "scope": "immutable_current_snapshot",
            "eligible": current_hash == observed,
        }
    service_eligible = bool(
        service
        and service.get("service_evidence_ready_score") == 1
        and service.get("flagged_adversarial") is False
        and service.get("verdict_class") in {"positive", "null", "circular_positive"}
    )
    checks.append(
        _check(
            "Exp7778",
            SERVICE,
            sources[SERVICE]["sha256"],
            "eligible_service_producer",
            True,
            service_eligible,
        )
    )
    sources[SERVICE]["eligible"] = service_eligible
    opportunity = service.get("hardware_opportunity") if service_eligible else None
    failed = [check for check in checks if not check["passed"]]
    history_ok = all(
        check["passed"] for check in checks if check["upstream_id"].startswith("Exp7751")
    )
    snapshot_ok = all(
        check["passed"] for check in checks if check["upstream_id"] == "current_literature"
    )
    verdict = (
        "complete_blocked_historical_evidence"
        if not history_ok or not snapshot_ok
        else "complete_blocked_missing_service_evidence"
        if not service_eligible
        else "complete_null_hardware_continuity_only"
    )
    ended = time.monotonic()
    progress(started, "preflight", "complete", len(checks))
    summary = {
        "intended": 6,
        "eligible": sum(not row["excluded"] for row in rows),
        "started": 6,
        "completed": 6,
        "excluded": sum(row["excluded"] for row in rows),
        "censored": 0,
        "effective_independent_n": 2 if history_ok else 0,
    }
    artifact: dict[str, Any] = {
        "schema": "carnot.experiment_7779.v1",
        "experiment_id": 7779,
        "milestone": "2026.09.676",
        "run_date": run_date,
        "honest_verdict": verdict,
        "verdict_class": "blocked" if failed else "null",
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "rows": rows,
        "rows_checksum": canonical_hash(rows),
        "hardware_continuity_rows": rows,
        "hardware_continuity": {
            row["board"]: {
                "evidence_date": row["evidence_date"],
                "evidence_hash": row["expected_hash"],
                "venue": row["venue"],
                "scope": row["scope"],
                "blocker": row["blocker"],
                "reopening_condition": row["unresolved_prerequisite"],
                "state": row["state"],
            }
            for row in rows
        },
        "immutable_evidence_ready_score": int(history_ok),
        "acceptance_gate_results": {
            "validity": history_ok and snapshot_ok,
            "readiness": int(history_ok and snapshot_ok and service_eligible),
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "sample_size_budget": summary,
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "root": str(root),
            "custody_checks": checks,
            "backend": "host_cpu",
            "board_operations_issued": [],
            "service_pre_gate_receipt": "results/experiment_7757_view_energy_fit.json",
            "service_pre_gate_is_producer": False,
            "prior_cost_verdict": prior_cost.get("honest_verdict") if prior_cost else None,
            "current_board_reachability": "not_probed",
        },
        "service_opportunities": opportunity,
        "claim_scope": {
            "hardware": "dated continuity only; no current acceleration",
            "NPU": "no qualified local NPU run",
            "TSU": "vendor Z1T figures only; no local access",
            "Kona": "public disclosure; no local weights or runner",
            "probability": "unmeasured",
            "decision": "unmeasured",
        },
        "verifier_is_oracle": False,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "actual_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"loads": 0, "generations": 0, "tokens": 0, "loaded_files": []},
        "random_seed": None,
        "duration_s": ended - started,
        "phase_spans": [
            {
                "phase": "preflight_and_reduction",
                "start_monotonic_s": started,
                "end_monotonic_s": ended,
                "duration_s": ended - started,
                "completed_units": len(checks),
            }
        ],
        "validation_receipts": {
            "frozen_scope_path": SCOPE,
            "frozen_scope_sha256": sha256_file(root / SCOPE) if (root / SCOPE).is_file() else None,
            "checks": [],
            "required_checks_passed": False,
            "repository_debt": [],
            "cold_replay": None,
            "terminal_reader": None,
        },
        "hardware_operations_issued": [],
        "hardware_advantage_claimed": False,
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "sources": sources,
            "rows": rows,
            "date": run_date,
            "code": sha256_file(Path(__file__)),
            "roles": "historical_evidence_audit",
            "parameters": {"boards": BOARDS, "contexts": CONTEXT},
        }
    )
    artifact["field_principles"] = {
        key: PRINCIPLES.get(key, "Exact scope and value are retained.") for key in artifact
    }
    for gate in artifact["acceptance_gate_results"]:
        artifact["field_principles"][f"{gate}_gate"] = (
            "Historical custody may pass; unmeasured benefit stays null."
        )
    return artifact


def cold_reduce(candidate: Path, raw: Path, root: Path) -> dict[str, Any]:
    """SCENARIO-REPORT-7779-TERMINAL: independently check source and row bytes."""
    artifact = json.loads(candidate.read_text())
    rows = json.loads(raw.read_text())
    for name, receipt in artifact["source_artifact_hashes"].items():
        if name.startswith("git:"):
            resolved = resolve_historical_blob(root, receipt["sha256"])
            if resolved is None or resolved["blob"] != receipt["git_blob"]:
                raise ValueError(f"source_hash_mismatch:{name}")
        elif receipt["scope"] == "immutable_current_snapshot":
            path = root / receipt["snapshot_path"]
            if not path.is_file() or sha256_file(path) != receipt["snapshot_sha256"]:
                raise ValueError(f"source_hash_mismatch:{name}")
        elif receipt["sha256"] is not None:
            path = root / receipt["path"]
            if not path.is_file() or sha256_file(path) != receipt["sha256"]:
                raise ValueError(f"source_hash_mismatch:{name}")
    if rows != artifact["rows"] or canonical_hash(rows) != artifact["rows_checksum"]:
        raise ValueError("row_mismatch")
    if rows != artifact["hardware_continuity_rows"] or len(rows) != 6:
        raise ValueError("row_mismatch")
    for row in rows:
        if artifact["hardware_continuity"][row["board"]]["state"] != row["state"]:
            raise ValueError("row_mismatch")
    return {
        "row_count": len(rows),
        "rows_checksum": canonical_hash(rows),
        "source_count": len(artifact["source_artifact_hashes"]),
    }


def main(argv: list[str] | None = None) -> int:
    """Prepare raw rows or publish the validated terminal artifact atomically."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    raw_dir = root / RAW
    raw = raw_dir / "rows.json"
    started = time.monotonic()
    progress(started, "entrypoint", "start", 0)
    if args.cold_reduce:
        progress(started, "cold_replay", "before_reduction", 0)
        print(json.dumps(cold_reduce(args.cold_reduce, raw, root)), flush=True)
        progress(started, "cold_replay", "after_reduction", 6)
        return 0
    artifact = build_audit(root, args.date, raw_dir / "snapshots")
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(raw, artifact["rows"])
    if args.prepare:
        atomic_json(raw_dir / "candidate.json", artifact)
        progress(started, "prepare", "complete", 6)
        return 0
    receipt_path = raw_dir / "validation_receipts.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"validation receipts required: {receipt_path}")
    receipts = json.loads(receipt_path.read_text())
    artifact["validation_receipts"].update(receipts)
    failed = [entry for entry in receipts["checks"] if entry["exit_code"] != 0]
    artifact["gate_check_summary"].extend(
        _check(
            "Exp7779:validation",
            entry["log_path"],
            entry["log_sha256"],
            entry["name"],
            0,
            entry["exit_code"],
        )
        for entry in failed
    )
    artifact["flagged_adversarial"] = bool(receipts["terminal_reader"]["flagged_adversarial"])
    if failed or artifact["flagged_adversarial"] or not receipts["required_checks_passed"]:
        artifact["honest_verdict"] = "complete_disqualified_required_checks"
        artifact["verdict_class"] = "disqualified"
        artifact["acceptance_gate_results"]["readiness"] = 0
    artifact["phase_spans"].extend(receipts["phase_spans"])
    artifact["duration_s"] = time.monotonic() - receipts["run_started_monotonic_s"]
    progress(started, "publish", "before_atomic_write", 6)
    atomic_json(root / OUTPUT, artifact)
    progress(started, "publish", "after_atomic_write", 6)
    return 0
