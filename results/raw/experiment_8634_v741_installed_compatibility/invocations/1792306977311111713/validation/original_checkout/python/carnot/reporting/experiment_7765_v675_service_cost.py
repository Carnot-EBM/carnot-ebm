"""Account for V675 service timing only when fitted response heads exist.

The current host can preserve dated hardware evidence while refusing to turn a
conductor pre-gate receipt into a measured deployment path (REQ-REPORT-7765).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


FIT = "results/experiment_7757_view_energy_fit.json"
ONLINE = "results/experiment_7761_v675_online_state.json"
INVENTORY = "results/experiment_7751_v674_hardware_continuity.json"
RAW = "results/raw/experiment_7765_v675_service_cost/rows.json"
OUTPUT = "results/experiment_7765_v675_service_cost.json"
ARM_NAMES = ("canonical_energy", "paired_energy", "canonical_mlp", "paired_mlp", "local_logistic")
IMPORTED_FIELDS = {
    FIT: ["fit_ready_score", "heads", "verdict_class", "flagged_adversarial", "blocked_at_layer"],
    ONLINE: ["online_state_ready_score", "verdict_class", "flagged_adversarial"],
    INVENTORY: [
        "hardware_continuity_complete_score",
        "hardware_continuity",
        "source_artifact_hashes",
    ],
}


def aggregate_risk(
    arm: str,
    values: Sequence[Any],
    temperature: float,
    *,
    logistic_head: Callable[[list[float]], float] | None = None,
) -> float:
    """Apply one temperature after the response risk has been assembled."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if arm == "local_logistic":
        if logistic_head is None:
            raise ValueError("logistic head required")
        features = [sum(column) / len(values) for column in zip(*values, strict=True)]
        risk = logistic_head(features)
    elif arm.startswith("canonical_"):
        risk = float(values[0])
    else:
        risk = sum(float(value) for value in values) / len(values)
    if not 0 < risk < 1:
        raise ValueError("risk must be strictly inside (0,1)")
    logit = math.log(risk / (1 - risk)) / temperature
    return 1 / (1 + math.exp(-logit))


def score_rows(rows: Sequence[dict[str, Any]], temperature: float) -> list[dict[str, Any]]:
    """Keep forced escalations in overall denominators with their fixed cost."""
    scored = []
    for row in rows:
        invalid = bool(row["invalid"])
        risk = 0.5 if invalid else aggregate_risk(row["arm"], row["risks"], temperature)
        decision = "escalate" if invalid or risk >= 0.5 else "accept"
        scored.append(
            {
                **row,
                "risk": risk,
                "decision": decision,
                "brier": 0.25 if invalid else (risk - row["label"]) ** 2,
                "realized_cost": 0.25
                if invalid
                else (0.25 if decision == "escalate" else float(row["label"])),
            }
        )
    return scored


def _source(root: Path, rel: str, scope: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Retain exact source bytes and report an absent declared file explicitly."""
    path = root / rel
    if not path.is_file():
        return {
            "path": rel,
            "sha256": None,
            "eligible": False,
            "scope": scope,
            "imported_fields": IMPORTED_FIELDS[rel],
        }, None
    value = json.loads(path.read_text())
    return {
        "path": rel,
        "sha256": sha256_file(path),
        "date": value.get("run_date"),
        "eligible": False,
        "scope": scope,
        "imported_fields": IMPORTED_FIELDS[rel],
    }, value


def _check(
    upstream: str, receipt: dict[str, Any], field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """A failed operand names the producer, its bytes, and the exact field."""
    return {
        "upstream_id": upstream,
        "artifact_path": receipt["path"],
        "artifact_hash": receipt["sha256"],
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def build_candidate(root: Path, run_date: str) -> dict[str, Any]:
    """Record a terminal block when no qualified fitted response head exists."""
    started = time.monotonic()
    print(f"[exp7765] phase=preflight elapsed_s=0.000 completed_units=0", flush=True)
    sources: dict[str, dict[str, Any]] = {}
    values = {}
    for rel, scope in (
        (FIT, "declared_fit_or_pre_gate"),
        (ONLINE, "optional_online_state"),
        (INVENTORY, "dated_inventory"),
    ):
        sources[rel], values[rel] = _source(root, rel, scope)
    fit = values[FIT]
    inventory = values[INVENTORY]
    checks = [
        _check("Exp7757", sources[FIT], "artifact_exists", True, fit is not None),
        _check(
            "Exp7757",
            sources[FIT],
            "fitted_heads",
            "qualified_current_fit",
            (
                "conductor_pre_gate"
                if fit and fit.get("blocked_at_layer") == "conductor_pre_gate"
                else "qualified_current_fit"
                if fit and fit.get("fit_ready_score") == 1 and fit.get("heads")
                else "missing_or_disqualified"
            ),
        ),
        _check(
            "Exp7757",
            sources[FIT],
            "verdict_class",
            "qualified",
            (
                "qualified"
                if fit
                and fit.get("verdict_class") in {"positive", "null", "circular_positive"}
                and fit.get("flagged_adversarial") is False
                else fit.get("verdict_class")
                if fit
                else None
            ),
        ),
        _check(
            "Exp7751",
            sources[INVENTORY],
            "hardware_continuity_complete_score",
            1,
            inventory.get("hardware_continuity_complete_score") if inventory else None,
        ),
    ]
    online = values[ONLINE]
    online_eligible = bool(
        online
        and online.get("online_state_ready_score") == 1
        and online.get("flagged_adversarial") is False
    )
    sources[ONLINE]["eligible"] = online_eligible
    hardware = (
        {name: dict(entry) for name, entry in inventory.get("hardware_continuity", {}).items()}
        if inventory
        else {}
    )
    for name, entry in hardware.items():
        expected = entry.get("evidence_hash")
        if not expected:
            entry["authenticated"] = False
            continue
        matches = [
            (rel, source)
            for rel, source in inventory.get("source_artifact_hashes", {}).items()
            if source.get("sha256") == expected
        ]
        rel = matches[0][0] if matches else None
        actual = sha256_file(root / rel) if rel and (root / rel).is_file() else None
        receipt = {"path": rel or "unresolved_historical_receipt", "sha256": actual}
        entry["evidence_path"] = rel
        entry["observed_hash"] = actual
        entry["authenticated"] = actual == expected
        checks.append(_check(f"Exp7751:{name}", receipt, "evidence_hash", expected, actual))
        if rel:
            sources[rel] = {
                "path": rel,
                "sha256": actual,
                "date": entry.get("evidence_date"),
                "scope": "historical_board_evidence",
                "eligible": actual == expected,
                "imported_fields": [
                    "evidence_date",
                    "evidence_hash",
                    "venue",
                    "supported_workload",
                ],
            }
    failed = [check for check in checks if not check["passed"]]
    sources[INVENTORY]["eligible"] = bool(
        inventory
        and inventory.get("hardware_continuity_complete_score") == 1
        and all(check["passed"] for check in checks if check["upstream_id"].startswith("Exp7751"))
    )
    print(
        f"[exp7765] phase=preflight_complete elapsed_s={time.monotonic() - started:.3f} completed_units={len(checks)}",
        flush=True,
    )
    rows: list[dict[str, Any]] = [
        {
            "unit_type": "check",
            "arm": None,
            "batch": None,
            "block": None,
            "source_path": check["artifact_path"],
            "source_hash": check["artifact_hash"],
            "field": check["field"],
            "observed": check["observed"],
            "started": True,
            "completed": True,
            "excluded": not check["passed"],
            "censored": False,
            "label": None,
            "metric": None,
        }
        for check in checks
    ]
    rows.extend(
        {
            "unit_type": "timing",
            "arm": arm,
            "batch": batch,
            "block": block,
            "source_path": FIT,
            "source_hash": sources[FIT]["sha256"],
            "field": "complete_service_ms",
            "observed": None,
            "started": False,
            "completed": False,
            "excluded": True,
            "censored": False,
            "label": None,
            "metric": None,
            "reason": "qualified_fit_missing",
        }
        for arm in ARM_NAMES
        for batch in (1, 8, 32)
        for block in range(30)
    )
    elapsed = time.monotonic() - started
    print(
        f"[exp7765] phase=accounting elapsed_s={elapsed:.3f} completed_units={len(rows)}",
        flush=True,
    )
    result: dict[str, Any] = {
        "schema": "carnot.experiment_7765.v1",
        "experiment_id": 7765,
        "milestone": "2026.09.675",
        "run_date": run_date,
        "honest_verdict": "complete_blocked_missing_qualified_fit",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "rows": rows,
        "rows_checksum": canonical_hash(rows),
        "service_cost_rows": [],
        "acceptance_gate_results": {
            "validity": not any(
                check["upstream_id"].startswith("Exp7751") and not check["passed"]
                for check in checks
            ),
            "readiness": False,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "service_evidence_ready_score": 0,
        "cold_load_ms": None,
        "update_to_durable_ack_ms": None,
        "batch_1_p95_target_ms": 100,
        "batch_1_p95_observed_ms": None,
        "matched_mlp_p95_ratio_target": 2,
        "matched_mlp_p95_ratio_observed": None,
        "amdahl_upper_bound": None,
        "hardware_opportunity": None,
        "hardware_continuity": hardware,
        "cost_input_eligibility": {
            FIT: {
                "path": FIT,
                "sha256": sources[FIT]["sha256"],
                "field": "fitted_heads",
                "value": fit.get("heads") if fit else None,
                "eligible": False,
                "reason": "conductor_pre_gate_not_fitted_heads" if fit else "missing_producer",
            },
            ONLINE: {
                "path": ONLINE,
                "sha256": sources[ONLINE]["sha256"],
                "field": "online_state_ready_score",
                "value": online.get("online_state_ready_score") if online else None,
                "eligible": online_eligible,
                "reason": "optional_absent"
                if online is None
                else "qualified"
                if online_eligible
                else "ineligible",
            },
        },
        "sample_size_budget": {
            "intended": len(ARM_NAMES) * 3 * 30,
            "eligible": 0,
            "started": 0,
            "completed": 0,
            "excluded": len(ARM_NAMES) * 3 * 30,
            "censored": 0,
            "effective_independent_n": 0,
        },
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "root": str(root),
            "coding_backend": os.getenv("CARNOT_CODING_BACKEND", "codex"),
            "checks": checks,
            "owned_backend": "host_cpu",
            "board_operations_issued": [],
            "complete_service_available": False,
        },
        "validation_receipts": {
            "frozen_scope_path": "results/raw/experiment_7765_v675_service_cost/validation_scope.json",
            "checks": [],
            "repository_debt": [],
        },
        "phase_spans": [
            {
                "phase": "preflight_and_accounting",
                "duration_s": elapsed,
                "completed_units": len(rows),
                "checkpoint_hash": canonical_hash(rows),
            }
        ],
        "duration_s": elapsed,
        "random_seed": {"timing": 7765, "used": False},
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "host_cpu_complete_service",
        "actual_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"loads": 0, "generations": 0, "tokens": 0, "loaded_files": []},
        "verifier_is_oracle": False,
        "claim_scope": {
            "service_timing": "blocked_unmeasured",
            "hardware": "dated_inventory_only",
            "natural_annotation": "unmeasured",
            "online_update": "unmeasured",
        },
        "hardware_operations_issued": [],
        "hardware_advantage_claimed": False,
        "vendor_z1t_and_fpga_asic": "external_projections_only",
    }
    result["reproducibility_checksum"] = canonical_hash(
        {
            "source_artifact_hashes": sources,
            "code": sha256_file(Path(__file__)),
            "roles": {"fit": FIT, "online": ONLINE, "inventory": INVENTORY},
            "parameters": {"seed": 7765, "warmups": 10, "paired_blocks": 30, "batches": [1, 8, 32]},
        }
    )
    result["field_principles"] = {
        key: "Keep exact observed evidence and its use limit." for key in result
    }
    result["field_principles"].update(
        {
            "honest_verdict": "A terminal record avoids repeating unchanged external failures.",
            "gate_check_summary": "Missing producers and scientific thresholds are different causes.",
            "rows": "Raw units let another process recompute aggregates.",
            "acceptance_gate_results": "Protocol validity does not prove scientific benefit.",
            "service_evidence_ready_score": "A kernel alone cannot establish complete service cost.",
            "hardware_continuity": "Prior board access does not prove current acceleration.",
        }
    )
    return result


def cold_reduce(candidate: Path, root: Path, raw: Path | None = None) -> dict[str, Any]:
    """Reject changed source bytes and recompute every check and timing unit."""
    artifact = json.loads(candidate.read_text())
    raw_path = raw or root / RAW
    if raw_path.is_file():
        raw_rows = json.loads(raw_path.read_text())
        if raw_rows != {"rows": artifact["rows"], "rows_checksum": artifact["rows_checksum"]}:
            raise ValueError("raw_rows_mismatch")
    for rel, receipt in artifact["source_artifact_hashes"].items():
        path = root / rel
        observed = sha256_file(path) if path.is_file() else None
        if observed != receipt["sha256"]:
            raise ValueError(f"source_hash_mismatch:{rel}")
    expected = build_candidate(root, artifact["run_date"])
    for field in (
        "rows",
        "rows_checksum",
        "gate_check_summary",
        "sample_size_budget",
        "hardware_continuity",
        "service_evidence_ready_score",
        "service_cost_rows",
        "cost_input_eligibility",
        "reproducibility_checksum",
    ):
        if artifact[field] != expected[field]:
            raise ValueError(f"summary_mismatch:{field}")
    return {
        "row_count": len(artifact["rows"]),
        "rows_checksum": artifact["rows_checksum"],
        "failed_checks": len(artifact["gate_check_summary"]),
    }


def main(argv: list[str] | None = None) -> int:
    """Write the current terminal candidate and replay it from disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.cold_reduce:
        print("[exp7765] phase=cold_replay_start elapsed_s=0 completed_units=0", flush=True)
        print(json.dumps(cold_reduce(args.cold_reduce, root, args.raw)), flush=True)
        print("[exp7765] phase=cold_replay_complete elapsed_s=0 completed_units=1", flush=True)
        return 0
    started = time.monotonic()
    artifact = build_candidate(root, args.date)
    raw = args.raw or root / RAW
    print(
        f"[exp7765] phase=serialize_start elapsed_s={time.monotonic() - started:.3f} completed_units=0",
        flush=True,
    )
    atomic_json(raw, {"rows": artifact["rows"], "rows_checksum": artifact["rows_checksum"]})
    output = args.output or root / OUTPUT
    atomic_json(output, artifact)
    print(
        f"[exp7765] phase=serialize_complete elapsed_s={time.monotonic() - started:.3f} completed_units={len(artifact['rows'])}",
        flush=True,
    )
    print(json.dumps(cold_reduce(output, root, raw)), flush=True)
    print(
        f"[exp7765] phase=complete elapsed_s={time.monotonic() - started:.3f} completed_units={len(artifact['rows'])}",
        flush=True,
    )
    return 0
