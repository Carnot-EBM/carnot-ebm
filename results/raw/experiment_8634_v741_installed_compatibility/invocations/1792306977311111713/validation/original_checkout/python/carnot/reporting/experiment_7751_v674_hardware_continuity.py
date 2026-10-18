"""Read-only V674 hardware custody and cold reduction (REQ-REPORT-7751)."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


BOARD_SOURCE = "results/experiment_7599_v663_board_continuity.json"
COST_SOURCE = "results/experiment_7750_v674_service_cost.json"
RAW_NAME = "results/raw/experiment_7751_v674_hardware_continuity/rows.json"
OUTPUT_NAME = "results/experiment_7751_v674_hardware_continuity.json"
REQUIRED = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-program.md",
    "research-hardware-wishlist.md",
    "research-references.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/hardware/spec.md",
    BOARD_SOURCE,
)
PRINCIPLES = {
    "honest_verdict": "Completion and scientific benefit are different facts.",
    "verdict_class": "Unchanged external failures do not trigger identical attempts.",
    "flagged_adversarial": "A clean label cannot replace independent checks.",
    "gate_check_summary": "A missing field is a broken contract, not a scientific null.",
    "acceptance_gate_results": "Negative science may be valid while invalid execution never qualifies.",
    "rows": "Every comparison must be independently recomputable.",
    "sample_size_budget": "Board and service counts are not multiplied by requests.",
    "claim_scope": "Historical exposure cannot become fresh generalization.",
    "inference_substrate": "Cached evidence does not claim live generation.",
    "inference_substrate_class": "Duration checks match actual work.",
    "MODEL_SPECS": "The experiment model is distinct from the coding backend.",
    "model_invoked": "Names alone are not invocation evidence.",
    "execution_venue": "Host work is not fabric work.",
    "phase_spans": "Real timings and bounded silence expose stalled work.",
    "random_seed": "Replay uses fixed inputs and parameters.",
    "reproducibility_checksum": "Replay binds bytes, parameters, and reducer code.",
    "source_artifact_hashes": "An artifact cannot authenticate itself as upstream.",
    "preconditions_checked": "Missing access blocks before measurement.",
    "validation_receipts": "Registered checks control use of the result.",
    "verifier_is_oracle": "Circular success cannot establish independent value.",
    "field_principles": "The rationale travels with the contract.",
    "hardware_continuity_complete_score": "Inventory completeness is not acceleration.",
    "hardware_continuity": "Hardware credit belongs to the measured boundary.",
    "cost_input_eligibility": "Missing service timing cannot become projection.",
}


def default_root() -> Path:
    """Resolve the repository from this installed source file."""
    return Path(__file__).resolve().parents[3]


def _check(root: Path, rel: str) -> dict[str, Any]:
    """Record exact bytes or a complete missing-input operand."""
    path = root / rel
    exists = path.is_file() and path.stat().st_size > 0
    return {
        "check": "required_source_bytes",
        "upstream_id": rel,
        "artifact_path": str(path),
        "field": "byte_count",
        "op": ">",
        "expected": 0,
        "observed": path.stat().st_size if exists else None,
        "passed": exists,
    }


def _board_row(source: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep a board's historical venue separate from this host audit."""
    name = source["board"]
    path = source["evidence_artifact_path"]
    expected = source["evidence_artifact_sha256"]
    actual = sha256_file(root / path) if (root / path).is_file() else None
    check = {
        "check": "dated_board_evidence_hash",
        "upstream_id": name,
        "artifact_path": str(root / path),
        "field": "sha256",
        "op": "==",
        "expected": expected,
        "observed": actual,
        "passed": actual == expected,
    }
    supported = {
        "KV260": "SSH authenticated quadratic Ising fabric, k_max<=5; transport limits preserved",
        "PolarFire": "hash-matched Linux CPU dispatch only",
        "GateMate": "no qualified execution; physical JTAG IDCODE blocked",
    }[name]
    reopen = {
        "KV260": "A new workload with k<=5, authenticated SSH fabric transcript, transport and complete service timing",
        "PolarFire": "A distinct FPGA workload with device-side execution and timing transcript",
        "GateMate": "Dated operator cable, port, power, board or DirtyJTAG change, then valid GM1Ax IDCODE",
    }[name]
    blocker = "0xffffffff" if name == "GateMate" else None
    row = {
        "substrate": name,
        "evidence_date": source["evidence_date"],
        "evidence_path": path,
        "evidence_hash": expected,
        "venue": source["execution_venue"],
        "supported_workload": supported,
        "blocker": blocker,
        "changed_prerequisite": reopen,
        "bounded_next_experiment": reopen,
        "measured_service_fraction": None,
        "numerator": 1 if actual == expected and name != "GateMate" else 0,
        "denominator": 1,
        "excluded": name == "GateMate",
        "censored": False,
        "input_hash": actual,
        "arm": "historical_inventory",
        "seed": None,
    }
    return row, check


def _context_row(name: str, path: str, source_hash: str) -> dict[str, Any]:
    """Represent proprietary or unavailable substrates without execution credit."""
    detail = {
        "Extropic Z1T/TSU": (
            "vendor projected sparse energy; dense readout and transfers excluded",
            "No local silicon or reproducible SDK execution",
            "Operator access plus end-to-end SDK run including dense readout and transfers",
        ),
        "Kona": (
            "public product disclosure only",
            "No weights, recipe, or local runner",
            "Reproducible local runner, weights, recipe and matched benchmark",
        ),
        "AMD XDNA NPU": (
            "no qualified local NPU execution",
            "No qualified provider or device run",
            "Operator-provided SDK/provider and device-side workload receipt",
        ),
    }[name]
    evidence_date = {
        "Extropic Z1T/TSU": "20260904",
        "Kona": "20260203",
        "AMD XDNA NPU": "20260926",
    }[name]
    return {
        "substrate": name,
        "evidence_date": evidence_date,
        "evidence_path": path,
        "evidence_hash": source_hash,
        "venue": "none",
        "supported_workload": detail[0],
        "blocker": detail[1],
        "changed_prerequisite": detail[2],
        "bounded_next_experiment": detail[2],
        "measured_service_fraction": None,
        "numerator": 0,
        "denominator": 1,
        "excluded": True,
        "censored": False,
        "input_hash": source_hash,
        "arm": "disclosure_inventory",
        "seed": None,
    }


def build_audit(root: Path, run_date: str) -> dict[str, Any]:
    """Build the immutable inventory facts before any terminal validation."""
    root = root.resolve()
    started = time.monotonic()
    checks = [_check(root, rel) for rel in REQUIRED]
    sources = {
        rel: {"path": rel, "sha256": sha256_file(root / rel), "scope": "eligible_inventory"}
        for rel in REQUIRED
        if (root / rel).is_file()
    }
    board_rows: list[dict[str, Any]] = []
    if (root / BOARD_SOURCE).is_file():
        try:
            board = json.loads((root / BOARD_SOURCE).read_text())
            by_name = {row["board"]: row for row in board["board_rows"]}
            if set(by_name) != {"KV260", "PolarFire", "GateMate"}:
                raise ValueError("board roster")
            for name in ("KV260", "PolarFire", "GateMate"):
                row, check = _board_row(by_name[name], root)
                board_rows.append(row)
                checks.append(check)
                if check["observed"] is not None:
                    sources[row["evidence_path"]] = {
                        "path": row["evidence_path"],
                        "sha256": check["observed"],
                        "scope": "historical_board_evidence",
                    }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            checks.append(
                {
                    "check": "board_schema",
                    "upstream_id": "Exp7599",
                    "artifact_path": str(root / BOARD_SOURCE),
                    "field": "board_rows",
                    "op": "valid",
                    "expected": "three named board rows",
                    "observed": str(error),
                    "passed": False,
                }
            )
    references = "research-references.md"
    if references in sources:
        for name in ("Extropic Z1T/TSU", "Kona", "AMD XDNA NPU"):
            board_rows.append(_context_row(name, references, sources[references]["sha256"]))
    cost_path = root / COST_SOURCE
    cost = None
    cost_reason = "missing"
    if cost_path.is_file():
        sources[COST_SOURCE] = {
            "path": COST_SOURCE,
            "sha256": sha256_file(cost_path),
            "scope": "candidate_cost_producer",
        }
        try:
            candidate = json.loads(cost_path.read_text())
            if (
                candidate.get("verdict_class") in {"positive", "null", "circular_positive"}
                and str(candidate.get("honest_verdict", "")).startswith("complete")
                and candidate.get("flagged_adversarial") is False
            ):
                cost = candidate
                cost_reason = "eligible"
                sources[COST_SOURCE]["scope"] = "eligible_cost_producer"
            else:
                cost_reason = "producer_ineligible"
        except (ValueError, TypeError):
            cost_reason = "invalid_json"
    fractions = cost.get("service_fractions") if cost else None
    if not isinstance(fractions, dict):
        fractions = {}
    # Only a specifically measured quadratic kernel can map to the KV260 fabric.
    kernel = fractions.get("quadratic_ising_kernel")
    if isinstance(kernel, (float, int)) and not isinstance(kernel, bool) and 0 <= kernel <= 1:
        for row in board_rows:
            if row["substrate"] == "KV260":
                row["measured_service_fraction"] = float(kernel)
    cost_gate = {
        "check": "eligible_service_fractions",
        "upstream_id": "Exp7750",
        "artifact_path": str(cost_path),
        "field": "service_fractions.quadratic_ising_kernel",
        "op": "measured",
        "expected": "eligible measured quadratic kernel fraction",
        "observed": kernel if cost is not None else cost_reason,
        "passed": cost is not None and kernel is not None,
    }
    checks.append(cost_gate)
    required_ok = all(check["passed"] for check in checks if check is not cost_gate)
    complete = required_ok and len(board_rows) == 6
    summary = {
        "intended": 6,
        "started": len(board_rows),
        "completed": len(board_rows),
        "eligible": sum(not row["excluded"] for row in board_rows),
        "excluded": sum(row["excluded"] for row in board_rows),
        "censored": 0,
        "effective_independent_n": 3 if complete else 0,
    }
    result: dict[str, Any] = {
        "schema": "carnot.experiment_7751.v1",
        "experiment_id": 7751,
        "run_date": run_date,
        "honest_verdict": "complete_null_hardware_continuity_only"
        if complete
        else "complete_blocked_required_inventory_evidence",
        "verdict_class": "null" if complete else "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": [check for check in checks if not check["passed"]],
        "acceptance_gate_results": {
            "validity": complete,
            "readiness": None,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "rows": board_rows,
        "rows_checksum": canonical_hash(board_rows),
        "sample_size_budget": summary,
        "claim_scope": {
            "RAGTruth": "development_only",
            "constructed_truth": "fixture_only",
            "ARC": "adapter_withheld_public",
        },
        "fresh_generalization_eligible": False,
        "inference_substrate": "aggregation",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": {
            name: 0
            for name in ("loads", "forwards", "generations", "tokens", "failures", "cancellations")
        },
        "execution_venue": "host",
        "execution_venue_details": {"pid": os.getpid(), "gpu_uuid": None},
        "phase_spans": [
            {
                "phase": "inventory",
                "start_monotonic_s": started,
                "end_monotonic_s": time.monotonic(),
                "duration_s": time.monotonic() - started,
                "run_date": run_date,
                "heartbeat_times": [],
                "completed_units": len(board_rows),
                "checkpoint_hash": canonical_hash(board_rows),
            }
        ],
        "random_seed": {"inventory": None, "reason": "deterministic read-only reduction"},
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "root": str(root),
            "required_checks": checks,
            "coding_backend": os.getenv("CARNOT_CODING_BACKEND", "codex"),
            "host_storage_required_for_kv260": False,
        },
        "validation_receipts": {
            "frozen_scope": [
                "tests/python/test_experiment_7751_v674_hardware_continuity.py",
                "python/carnot/reporting/experiment_7751_v674_hardware_continuity.py",
            ],
            "checks": [],
            "repository_debt": [],
        },
        "verifier_is_oracle": False,
        "hardware_continuity_complete_score": int(complete),
        "hardware_continuity": {
            row["substrate"]: {
                key: row[key]
                for key in (
                    "evidence_date",
                    "evidence_hash",
                    "venue",
                    "supported_workload",
                    "blocker",
                    "changed_prerequisite",
                    "bounded_next_experiment",
                )
            }
            | (
                {"k_max": 5, "transport_limit": "SSH/UIO bounded; no complete-service speedup"}
                if row["substrate"] == "KV260"
                else {}
            )
            for row in board_rows
        },
        "cost_input_eligibility": {
            "path": COST_SOURCE,
            "sha256": sources.get(COST_SOURCE, {}).get("sha256"),
            "eligible": cost is not None,
            "reason": cost_reason,
            "observed_missing_field": None
            if kernel is not None
            else "service_fractions.quadratic_ising_kernel",
        },
        "hardware_operations_issued": [],
        "hardware_advantage_claimed": False,
        "continuous_learning_throughput_claimed": False,
        "nfr01_claimed": False,
    }
    result["reproducibility_checksum"] = canonical_hash(
        {
            "sources": sources,
            "date": run_date,
            "rows": board_rows,
            "reducer": sha256_file(Path(__file__)),
        }
    )
    result["field_principles"] = {
        key: PRINCIPLES.get(key, "Exact observed value and scope are retained.") for key in result
    }
    result["field_principles"]["validity_gate"] = (
        "Inventory validity is separate from scientific value."
    )
    result["field_principles"]["readiness_gate"] = (
        "Administrative accounting does not promote hardware."
    )
    for name in ("probability_quality", "decision_benefit", "retention", "efficiency"):
        result["field_principles"][name + "_gate"] = "Unmeasured science stays null."
    return result


def cold_reduce(candidate: Path, raw: Path, root: Path) -> dict[str, Any]:
    """SCENARIO-REPORT-7751-REPLAY: reject changed bytes and row summaries."""
    artifact = json.loads(candidate.read_text())
    rows = json.loads(raw.read_text())
    for rel, receipt in artifact["source_artifact_hashes"].items():
        path = root / rel
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"source_hash_mismatch:{rel}")
    if rows != artifact["rows"] or canonical_hash(rows) != artifact["rows_checksum"]:
        raise ValueError("summary_mismatch:rows")
    expected = {
        "intended": 6,
        "started": len(rows),
        "completed": len(rows),
        "eligible": sum(not row["excluded"] for row in rows),
        "excluded": sum(row["excluded"] for row in rows),
        "censored": 0,
        "effective_independent_n": 3 if len(rows) == 6 else 0,
    }
    if artifact["sample_size_budget"] != expected:
        raise ValueError("summary_mismatch:sample_size_budget")
    if artifact["hardware_continuity_complete_score"] != int(len(rows) == 6):
        raise ValueError("summary_mismatch:continuity_score")
    for row in rows:
        if (
            artifact["hardware_continuity"][row["substrate"]]["evidence_hash"]
            != row["evidence_hash"]
        ):
            raise ValueError("summary_mismatch:hardware_continuity")
    return {
        "rows": len(rows),
        "rows_checksum": canonical_hash(rows),
        "source_count": len(artifact["source_artifact_hashes"]),
    }


def main(argv: list[str] | None = None) -> int:
    """Produce a private fixture or the host-only terminal candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    raw = args.raw or root / RAW_NAME
    if args.cold_reduce:
        print(f"[exp7751] before cold_reduce path={args.cold_reduce}", flush=True)
        print(json.dumps(cold_reduce(args.cold_reduce, raw, root)), flush=True)
        print("[exp7751] after cold_reduce", flush=True)
        return 0
    print(f"[exp7751] preconditions root={root}", flush=True)
    artifact = build_audit(root, args.date)
    if args.fixture and artifact["verdict_class"] == "null":
        artifact["honest_verdict"] = "complete_circular_positive_fixture_inventory"
        artifact["verdict_class"] = "circular_positive"
        artifact["verifier_is_oracle"] = True
    print(f"[exp7751] inventory complete units={len(artifact['rows'])}", flush=True)
    raw.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(raw, artifact["rows"])
    output = args.output or root / OUTPUT_NAME
    atomic_json(output, artifact)
    print(f"[exp7751] candidate written path={output}", flush=True)
    print(json.dumps(cold_reduce(output, raw, root)), flush=True)
    return 0
