"""Reconcile exact V682 evidence without turning queue completion into benefit.

REQ-REPORT-7864-V682. Each task keeps its declared deliverable identity, so a
conductor skip or a similarly named older result cannot stand in for science.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.roadmap_contract import compare_contract, verify_snapshots


AUTHORITY = Path("docs/research-notes/v682-authority-snapshots")
CONTRACT = Path("results/experiment_7851_v682_contract_methods.json")
AUDIT = Path("results/experiment_7863_v682_independent_audit.json")
QUALIFIED = {"positive", "circular_positive", "null"}
SCIENCE = set(range(7852, 7863))


def failure(
    number: int,
    path: Path,
    digest: str | None,
    field: str,
    expected: Any,
    observed: Any,
    op: str = "==",
) -> dict[str, Any]:
    """Keep an absent field distinct from a present field with the wrong value."""
    return {
        "upstream_id": f"Exp{number}",
        "path": str(path),
        "hash": digest,
        "artifact_field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
    }


def _data(path: Path) -> dict[str, Any]:
    """A malformed external artifact is unavailable evidence, not a result."""
    try:
        value = json.loads(path.read_bytes())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def authority_tasks(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Require the saved bytes and live sources named by Exp7851 to agree."""
    receipt = _data(root / CONTRACT)
    snapshots = receipt.get("authority_snapshot_paths", [])
    if not verify_snapshots(snapshots):
        raise ValueError("V682 authority snapshot bytes differ")
    staged = yaml.safe_load((root / AUTHORITY / "staged.yaml").read_text())
    active = yaml.safe_load((root / AUTHORITY / "active.yaml").read_text())
    design = (root / AUTHORITY / "design.md").read_text()
    comparison = compare_contract(design, staged, active, milestone="2026.09.682", first_id=7851)
    if not comparison["passed"]:
        raise ValueError(f"V682 task contract differs: {comparison['errors']}")
    return active["tasks"], snapshots


def _required_failures(
    number: int, path: Path, digest: str | None, value: dict[str, Any]
) -> list[dict[str, Any]]:
    """A failed required child remains a failure even under a complete prefix."""
    raw = value.get("validation_receipts", [])
    receipts = raw.get("checks", []) if isinstance(raw, dict) else raw
    found = []
    for item in receipts if isinstance(receipts, list) else []:
        if (
            not isinstance(item, dict)
            or item.get("classification", item.get("class", item.get("scope"))) != "required"
        ):
            continue
        if item.get("exit_code") != 0 or item.get("passed") is False or item.get("timed_out"):
            found.append(
                failure(
                    number,
                    path,
                    digest,
                    f"validation_receipts.{item.get('name', 'unnamed')}",
                    0,
                    item.get("exit_code"),
                )
            )
    return found


def build_ledger(root: Path) -> dict[str, Any]:
    """Read one declared path per task and retain every prior comparison."""
    tasks, snapshots = authority_tasks(root)
    dispositions: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    retirements: list[dict[str, Any]] = []
    primitive: list[dict[str, Any]] = []
    for number, task in zip(range(7851, 7865), tasks, strict=True):
        path = root / task["deliverable"]
        current = number != 7864 and path.is_file()
        digest = sha256_file(path) if current else None
        value = _data(path) if current else {}
        verdict = value.get("honest_verdict")
        verdict_class = value.get("verdict_class")
        own = number == 7864
        status = (
            "own_reconciliation"
            if own
            else "missing"
            if not value
            else str(verdict_class or "invalid")
        )
        errors: list[dict[str, Any]] = []
        if number in SCIENCE:
            if not value:
                errors.append(
                    failure(
                        number,
                        path,
                        digest,
                        "science_producer",
                        "current qualified artifact",
                        "missing" if not current else "invalid_json",
                    )
                )
            else:
                for field, expected in (
                    ("experiment_id", number),
                    ("task_id", task["id"]),
                    ("milestone", "2026.09.682"),
                    ("run_date", "20260929"),
                    ("flagged_adversarial", False),
                ):
                    if value.get(field) != expected:
                        errors.append(
                            failure(number, path, digest, field, expected, value.get(field))
                        )
                if verdict_class not in QUALIFIED:
                    errors.append(
                        failure(
                            number,
                            path,
                            digest,
                            "verdict_class",
                            sorted(QUALIFIED),
                            verdict_class,
                            "in",
                        )
                    )
                errors.extend(_required_failures(number, path, digest, value))
        gates.extend(errors)
        sample = value.get("sample_size_budget", {})
        if not isinstance(sample, dict):
            sample = {}
        counts = {
            key: sample.get(key, 0)
            for key in ("intended", "eligible", "started", "completed", "censored", "excluded")
        }
        row = {
            "upstream_id": f"Exp{number}",
            "task_id": task["id"],
            "path": str(path),
            "sha256": digest,
            "honest_verdict": verdict,
            "verdict_class": verdict_class,
            "status": status,
            "eligible": bool(number in SCIENCE and value and not errors),
            "arm": task["id"],
            "source_family": None,
            "seed": value.get("random_seed"),
            "independent": sample.get("independent", sample.get("independent_n", 0)),
            **counts,
        }
        dispositions.append(row)
        primitive.append(
            {
                "unit_id": task["id"],
                "arm": task["id"],
                "source_family": None,
                "seed": row["seed"],
                "status": status,
                "metric": None,
                "eligible": row["eligible"],
                **counts,
                "independent": row["independent"],
            }
        )
        sources.append(
            {
                "path": str(path),
                "sha256": digest,
                "date": value.get("run_date"),
                "role": "own_output"
                if own
                else "science_producer"
                if number in SCIENCE
                else "administrative_or_audit",
                "exposure_status": "current" if number == 7851 else "exposed_development",
                "eligible": row["eligible"],
            }
        )
        for prior in task.get("prior_failures", []):
            prior_number = str(prior["experiment_id"]).split("-", 1)[0].removeprefix("exp")
            prior_paths = (
                sorted((root / "results").glob(f"experiment_{prior_number}_*.json"))
                if prior_number.isdigit()
                else []
            )
            matching = [
                candidate
                for candidate in prior_paths
                if _data(candidate).get("honest_verdict") == prior["verdict"]
            ]
            prior_path = matching[0] if len(matching) == 1 else None
            match = bool(
                current
                and prior_path
                and verdict == prior["verdict"]
                and prior.get("retire_if_same_verdict")
            )
            retirements.append(
                {
                    "task_id": task["id"],
                    "prior_experiment_id": prior["experiment_id"],
                    "prior_verdict": prior["verdict"],
                    "prior_observed_verdict": _data(prior_path).get("honest_verdict")
                    if prior_path
                    else None,
                    "prior_path": str(prior_path) if prior_path else None,
                    "prior_sha256": sha256_file(prior_path) if prior_path else None,
                    "current_verdict": verdict,
                    "identical_verdict": match,
                    "retire_scope": match,
                    "current_path": str(path),
                    "current_sha256": digest,
                }
            )
    for item in snapshots:
        sources.append(
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "date": "20260929",
                "role": "authority_snapshot",
                "exposure_status": "current",
                "eligible": True,
            }
        )
    return {
        "task_disposition_rows": dispositions,
        "gate_check_summary": gates,
        "source_artifact_hashes": sources,
        "retirement_rows": retirements,
        "rows": primitive,
    }


def build_candidate(
    root: Path, date: str, elapsed_s: float, publication: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Separate a finished reconciliation from missing scientific evidence."""
    if date != "20260929":
        raise ValueError("V682 run date must be 20260929")
    ledger = build_ledger(root)
    module = Path(__file__).resolve()
    closure = [
        module,
        root / "scripts/experiments/experiment_7864_v682_capstone.py",
        root / "tests/python/test_experiment_7864_v682_capstone.py",
        root / "python/carnot/reporting/current_work_receipt.py",
        root / "python/carnot/reporting/roadmap_contract.py",
    ]
    for path in closure:
        ledger["source_artifact_hashes"].append(
            {
                "path": str(path),
                "sha256": sha256_file(path) if path.is_file() else None,
                "date": date,
                "role": "code_or_test_closure",
                "exposure_status": "current",
                "eligible": path.is_file(),
            }
        )
    rows = ledger["task_disposition_rows"]
    science = [row for row in rows if 7852 <= int(row["upstream_id"][3:]) <= 7862]
    complete = all(row["eligible"] for row in science)
    budget = {
        key: sum(int(row.get(key) or 0) for row in rows)
        for key in ("eligible", "started", "completed", "censored", "excluded")
    }
    budget.update(
        {
            "intended": 14,
            "independent": sum(int(row.get("independent") or 0) for row in science),
            "independent_unit": "source_family_not_seed_or_view",
        }
    )
    pub = publication or {}
    publication_rows = [
        {
            "gate": gate,
            "pass": pub.get("gates", {}).get(gate, {}).get("pass"),
            "detail": pub.get("gates", {}).get(gate, {}).get("detail"),
            "scope": "stable_publication_gate_not_v682_benefit",
        }
        for gate in ("G1", "G2", "G3", "G4")
    ]
    decisions = [
        {
            "branch": "source_boundary",
            "decision": "Repair required coverage and import receipts before source qualification.",
        },
        {
            "branch": "sufficiency",
            "decision": "Repair protocol validation before any Qwen generation or context claim.",
        },
        {
            "branch": "calibrated_decision",
            "decision": "Wait for qualified natural labels and fit; retain fixture agreement as circular.",
        },
        {
            "branch": "causal_learning",
            "decision": "Run prediction-before-feedback and no-write controls only after qualified producers exist.",
        },
        {
            "branch": "arc",
            "decision": "Keep the zero new-supervisor-outcome null; require live independent firings before renewal.",
        },
        {
            "branch": "service_cost",
            "decision": "Wait for complete service producer with paired costs before efficiency claims.",
        },
        {
            "branch": "hardware",
            "decision": "Keep historical board facts read-only; require authenticated new board execution.",
        },
    ]
    sources = ledger["source_artifact_hashes"]
    repro = canonical_hash(
        {
            "sources": [(s["path"], s["sha256"]) for s in sources],
            "configuration": "v682_capstone_v1",
            "seed": 68201,
        }
    )
    audit = _data(root / AUDIT)
    history = audit.get("repository_health", {}).get("historical_required_failures", [])
    result: dict[str, Any] = {
        "experiment_id": 7864,
        "task_id": "exp7864-capstone",
        "milestone": "2026.09.682",
        "run_date": date,
        "honest_verdict": "complete_blocked_required_v682_science"
        if not complete
        else "complete_null_no_registered_v682_benefit",
        "verdict_class": "blocked" if not complete else "null",
        "flagged_adversarial": False,
        **ledger,
        "sample_size_budget": budget,
        "acceptance_gate_results": {
            "validity": None,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": elapsed_s,
        "phase_spans": [
            {"phase": "preconditions_and_ledger", "duration_s": elapsed_s, "completed_units": 14}
        ],
        "random_seed": 68201,
        "reproducibility_checksum": repro,
        "preconditions_checked": [
            {
                "path": item["path"],
                "hash": item["sha256"],
                "source_role": item["role"],
                "exposure_status": item["exposure_status"],
                "date": item["date"],
                "schema_version": "v682_capstone_v1",
                "resource_ownership": "external"
                if item["role"] in {"science_producer", "administrative_or_audit"}
                else "current_task",
                "observed": item["sha256"] is not None,
            }
            for item in sources
        ],
        "validation_receipts": [],
        "validation_command_manifest_path": None,
        "observed_child_commands": [],
        "repository_health": {"status": "not_run", "historical_required_failures": history},
        "verifier_is_oracle": False,
        "claim_scope": {
            "fixtures": "circular_only",
            "natural_annotations": "exposed_development_only",
            "fresh_holdout_generalization": False,
            "llm_weight_training": False,
            "new_arc_solve": False,
            "new_board_execution": False,
            "september_28_corrigendum": "preserved",
            "gap_oracle_distinct": "open",
            "diffusiongemma": "pending",
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "actual_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0, "model_file_hashes": []},
        "capstone_execution_ready_score": 0,
        "milestone_evidence_complete_score": int(complete),
        "milestone_benefit_score": 0,
        "next_decision_rows": decisions,
        "publication_gate_rows": publication_rows,
        "capstone_doc_path": "docs/research-notes/v682-capstone.md",
        "resolved_imports": {
            "carnot.reporting.v682_capstone": str(module),
            "carnot.reporting.current_work_receipt": str(
                (root / "python/carnot/reporting/current_work_receipt.py").resolve()
            ),
            "carnot.reporting.roadmap_contract": str(
                (root / "python/carnot/reporting/roadmap_contract.py").resolve()
            ),
        },
    }
    default = "Keep exact evidence, status, and claim limits independently readable."
    result["field_principles"] = {key: default for key in result}
    result["field_principles"]["acceptance_gate_results"] = {
        gate: "An unmeasured scientific gate remains null; readiness records own validation only."
        for gate in result["acceptance_gate_results"]
    }
    result["field_principles"]["task_disposition_rows"] = (
        "All fourteen current tasks must be visible, including unstarted producers and this capstone."
    )
    result["field_principles"]["retirement_rows"] = (
        "Only an exact repeat of a declared prior verdict can retire that narrow scope."
    )
    result["field_principles"]["milestone_benefit_score"] = (
        "Queue completion and fixture validity do not establish empirical benefit."
    )
    result["field_principles"]["field_principles"] = "Persist why each reported field exists."
    return result


def cold_replay(candidate: Path, root: Path) -> list[str]:
    """Re-read source bytes and primitive identities after the candidate exists."""
    value = _data(candidate)
    if not value:
        return ["candidate_unreadable"]
    expected = build_candidate(root, "20260929", 0.0)
    errors = [
        key
        for key in (
            "task_disposition_rows",
            "gate_check_summary",
            "source_artifact_hashes",
            "retirement_rows",
            "rows",
        )
        if value.get(key) != expected[key]
    ]
    for key in (
        "reproducibility_checksum",
        "milestone_benefit_score",
        "milestone_evidence_complete_score",
        "verdict_class",
    ):
        if value.get(key) != expected[key]:
            errors.append(key)
    return errors
