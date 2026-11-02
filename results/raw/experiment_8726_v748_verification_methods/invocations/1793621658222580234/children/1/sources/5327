"""Close V683 from exact current artifacts (REQ-REPORT-7878)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting import v683_independent_audit as audit
from carnot.reporting import current_work_receipt as receipt_module
from carnot.reporting.roadmap_contract import compare_contract
from scripts import publication_gate


SCIENCE = {7869, 7870, 7871, 7872, 7873, 7875}
QUALIFIED = {"positive", "circular_positive", "null"}


def _read(path: Path) -> dict[str, Any]:
    """Keep malformed external bytes distinct from an eligible result."""
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _failure(
    number: int, path: Path, field: str, expected: Any, observed: Any, op: str = "=="
) -> dict[str, Any]:
    """Name the exact operand so an upstream block can be repaired."""
    return {
        "upstream_id": f"Exp{number}",
        "path": str(path),
        "hash": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
    }


def build_ledger(root: Path) -> dict[str, Any]:
    """Read all contract paths; a skip receipt only explains an absent producer."""
    authority = root / "docs/research-notes/v683-authority-snapshots/staged.yaml"
    tasks = yaml.safe_load(authority.read_text())["tasks"]
    if len(tasks) != 14 or [t["id"].split("-", 1)[0] for t in tasks] != [
        f"exp{n}" for n in range(7865, 7879)
    ]:
        raise ValueError("v683_contract_order_changed")
    rows, failures, sources = audit.inspect_sources(root)
    snapshot_root = root / "docs/research-notes/v683-authority-snapshots"
    design = snapshot_root / "design.md"
    staged = snapshot_root / "staged.yaml"
    active = snapshot_root / "active.yaml"
    comparison = compare_contract(
        design.read_text(),
        yaml.safe_load(staged.read_text()),
        yaml.safe_load(active.read_text()),
        milestone="2026.09.683",
        first_id=7865,
        count=14,
    )
    if not comparison["passed"]:
        failures.append(
            _failure(7865, design, "contract_comparison.passed", True, comparison["errors"])
        )
    for path in (design, staged, active):
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "date": "20260929",
                "role": "authority_snapshot",
                "exposure_status": "current",
            }
        )
    units, comparisons = audit._primitive_rows(root, rows)
    for number in (7877, 7878):
        task = tasks[number - 7865]
        path = root / task["deliverable"]
        own = number == 7878
        data = {} if own else _read(path)
        state = "own_reconciliation" if own else str(data.get("verdict_class", "missing_producer"))
        if not own:
            for field, expected in (
                ("experiment_id", number),
                ("task_id", task["id"]),
                ("milestone", "2026.09.683"),
                ("run_date", "20260929"),
                ("flagged_adversarial", False),
            ):
                if data.get(field) != expected:
                    failures.append(_failure(number, path, field, expected, data.get(field)))
            if data.get("verdict_class") not in QUALIFIED:
                failures.append(
                    _failure(
                        number,
                        path,
                        "verdict_class",
                        sorted(QUALIFIED),
                        data.get("verdict_class"),
                        "in",
                    )
                )
            for receipt in audit._receipts(data):
                if receipt.get("classification", receipt.get("scope")) == "required" and (
                    receipt.get("exit_code") != 0
                    or receipt.get("passed") is False
                    or receipt.get("timed_out")
                ):
                    failures.append(
                        _failure(
                            number,
                            path,
                            f"validation_receipts.{receipt.get('name', 'unnamed')}",
                            0,
                            receipt.get("exit_code"),
                        )
                    )
        eligible = bool(data) and not any(
            f["upstream_id"] == f"Exp{number}" and f["path"] == str(path) for f in failures
        )
        rows.append(
            {
                "upstream_id": f"Exp{number}",
                "task_id": task["id"],
                "path": str(path),
                "hash": sha256_file(path) if data else None,
                "status": state,
                "eligible": eligible,
                "role": "independent_audit" if not own else "own_reconciliation",
                "source_exposure": "administrative",
                "label_authority": "none",
                "skip_receipts": [],
            }
        )
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path) if data else None,
                "date": data.get("run_date"),
                "role": rows[-1]["role"],
                "exposure_status": "administrative",
            }
        )
        units.append(
            {
                "upstream_id": f"Exp{number}",
                "arm": task["id"],
                "family_id": None,
                "seed": None,
                "status": state,
                "metric": {},
                "eligible": int(eligible),
                "started": int(not own and bool(data)),
                "completed": int(eligible),
                "censored": 0,
                "excluded": int(not eligible),
                "independent": 0,
            }
        )
    sources.insert(
        0,
        {
            "path": str(authority),
            "sha256": sha256_file(authority),
            "date": "20260929",
            "role": "active_contract",
            "exposure_status": "current",
        },
    )
    return {
        "task_evidence_rows": rows,
        "gate_check_summary": failures,
        "source_artifact_hashes": sources,
        "rows": units,
        "recomputed_comparison_rows": comparisons,
        "contract_tasks": tasks,
    }


def _retirements(root: Path, ledger: dict[str, Any]) -> list[dict[str, Any]]:
    """Retire only an exact repeated verdict on the same named protocol."""
    decisions: list[dict[str, Any]] = []
    for task, row in zip(ledger["contract_tasks"], ledger["task_evidence_rows"], strict=True):
        current = _read(Path(row["path"])) if row["upstream_id"] != "Exp7878" else {}
        for prior in task.get("prior_failures", []):
            number = prior["experiment_id"].split("-", 1)[0].removeprefix("exp")
            matches = sorted((root / "results").glob(f"experiment_{number}_*.json"))
            exact = [p for p in matches if _read(p).get("honest_verdict") == prior["verdict"]]
            prior_data = _read(exact[0]) if len(exact) == 1 else {}
            current_scope = current.get("claim_scope")
            prior_scope = prior_data.get("claim_scope")
            same_scope = bool(
                task["id"].split("-", 1)[1] == prior["experiment_id"].split("-", 1)[1]
                and current_scope
                and current_scope == prior_scope
            )
            repeated = bool(
                len(exact) == 1
                and same_scope
                and current.get("honest_verdict") == prior["verdict"]
                and prior.get("retire_if_same_verdict")
            )
            decisions.append(
                {
                    "task_id": task["id"],
                    "prior_experiment_id": prior["experiment_id"],
                    "prior_verdict": prior["verdict"],
                    "current_verdict": current.get("honest_verdict"),
                    "same_scope": same_scope,
                    "identical_verdict": repeated,
                    "decision": "retire_exact_scope" if repeated else "continue_or_blocked",
                    "prior_path": str(exact[0]) if len(exact) == 1 else None,
                    "prior_sha256": sha256_file(exact[0]) if len(exact) == 1 else None,
                    "current_path": row["path"],
                    "current_sha256": row["hash"],
                }
            )
    return decisions


def build_candidate(root: Path, date: str) -> dict[str, Any]:
    """Keep a finished reconciliation separate from missing scientific work."""
    if date != "20260929":
        raise ValueError("v683_date_changed")
    start = time.monotonic()
    start_ns = time.monotonic_ns()
    ledger = build_ledger(root)
    decisions = _retirements(root, ledger)
    sources = ledger.pop("source_artifact_hashes")
    tasks = ledger.pop("contract_tasks")
    for module_path in (
        Path(__file__),
        Path(audit.__file__),
        Path(receipt_module.__file__),
        root / "python/carnot/reporting/roadmap_contract.py",
        Path(publication_gate.__file__),
        root / "scripts/experiments/experiment_7878_v683_capstone.py",
        root / "tests/python/test_experiment_7878_v683_capstone.py",
    ):
        sources.append(
            {
                "path": str(module_path),
                "sha256": sha256_file(module_path) if module_path.is_file() else None,
                "date": date,
                "role": "code_or_test_closure",
                "exposure_status": "current",
            }
        )
    eligible_science = {
        int(row["upstream_id"][3:]) for row in ledger["task_evidence_rows"] if row["eligible"]
    } & SCIENCE
    complete = eligible_science == SCIENCE
    units = ledger["rows"]
    budget = {
        key: sum(int(row[key]) for row in units)
        for key in ("eligible", "started", "completed", "censored", "excluded", "independent")
    }
    budget.update({"intended": len(units), "independent_unit": "source_family_not_seed_or_view"})
    pub = publication_gate.evaluate()
    gap = {
        "FR-12": {
            "decision": "blocked",
            "reason": "No qualified current energy fit and calibrated natural-label decision evidence.",
            "required_producers": [7869, 7870, 7871],
            "authority": "original human labels; exposed development only",
            "next_question": "Can a fitted head improve Brier and paired decision cost on independent source families?",
        },
        "FR-11": {
            "decision": "blocked",
            "reason": "No qualified prediction-before-feedback causal update or persistent retention producer.",
            "required_producers": [7872, 7873],
            "authority": "delayed feedback with no-write control",
            "next_question": "Does one admitted constraint persist and improve later held-out decisions after feedback?",
        },
        "FR-05/FR-08/ARC": {
            "decision": "blocked",
            "reason": "Service producer absent; ARC and board continuity artifacts fail required checks.",
            "required_producers": [7874, 7875, 7876],
            "authority": "live route and authenticated device custody",
            "next_question": "Can a paired live-route run and new board receipt establish usable cost and custody?",
        },
    }
    elapsed = time.monotonic() - start
    historical = [
        f
        for f in ledger["gate_check_summary"]
        if f["artifact_field"].startswith("validation_receipts.")
    ]
    result: dict[str, Any] = {
        "experiment_id": 7878,
        "task_id": tasks[-1]["id"],
        "milestone": "2026.09.683",
        "run_date": date,
        "honest_verdict": "complete_blocked_required_v683_science"
        if not complete
        else "complete_null_no_registered_v683_benefit",
        "verdict_class": "blocked" if not complete else "null",
        "flagged_adversarial": False,
        **ledger,
        "source_artifact_hashes": sources,
        "sample_size_budget": budget,
        "acceptance_gate_results": {
            "validity": None,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": elapsed,
        "phase_spans": [
            {"phase": "preconditions_and_ledger", "duration_s": elapsed, "completed_units": 14}
        ],
        "random_seed": 7878,
        "preconditions_checked": [
            {
                "path": s["path"],
                "hash": s["sha256"],
                "date": s["date"],
                "source_role": s["role"],
                "exposure_status": s["exposure_status"],
                "authority": "external"
                if s["role"]
                not in {"code_or_test_closure", "active_contract", "own_reconciliation"}
                else "current_task",
                "schema": _read(Path(s["path"])).get("schema") if s["sha256"] else None,
                "observed": s["sha256"] is not None,
            }
            for s in sources
        ],
        "validation_receipts": [],
        "validation_command_manifest_path": None,
        "observed_child_commands": [],
        "repository_health": {"status": "pending", "historical_required_failures": historical},
        "verifier_is_oracle": False,
        "claim_scope": {
            "fixtures": "circular_only",
            "natural_annotations": "exposed_development_only",
            "fresh_holdout_generalization": False,
            "gap_oracle_distinct": "open",
            "september_28_retractions": [4245, 5160, 5171],
            "diffusiongemma": "not_promoted",
            "new_arc_solve": False,
            "new_board_execution": False,
            "llm_weight_training": False,
        },
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none",
        "trained_head_specs": [],
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0, "model_file_hashes": []},
        "capstone_execution_ready_score": 0,
        "milestone_evidence_complete_score": int(complete),
        "milestone_benefit_score": 0,
        "publication_gate_results": pub,
        "prd_gap_decisions": gap,
        "retirement_decisions": decisions,
        "literature_adoption_decisions": {
            "source_sufficiency": "continue only with independent source-family labels",
            "causal_feedback": "continue only after delayed-feedback control qualification",
            "diffusiongemma": "defer until independent scorer and local control pass",
        },
        "report_path": "ops/research-milestone-v683.md",
        "resolved_imports": {
            "carnot.reporting.v683_capstone": str(Path(__file__).resolve()),
            "carnot.reporting.v683_independent_audit": str(Path(audit.__file__).resolve()),
            "carnot.reporting.current_work_receipt": str(Path(receipt_module.__file__).resolve()),
            "carnot.reporting.roadmap_contract": str(
                (root / "python/carnot/reporting/roadmap_contract.py").resolve()
            ),
            "scripts.publication_gate": str(Path(publication_gate.__file__).resolve()),
        },
    }
    result["reproducibility_checksum"] = canonical_hash(
        {"source_hashes": sources, "seed": 7878, "configuration": "v683_capstone_v1"}
    )
    result["current_work_receipt"] = receipt_module.build_current_work_receipt(
        run_id=f"exp7878-{os.getpid()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_details={"scope": "current_cpu_capstone"},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=result["phase_spans"],
    )
    result["field_principles"] = {
        key: "Keep the current byte identity, evidence status and claim limit auditable."
        for key in result
    }
    result["field_principles"]["acceptance_gate_results"] = {
        key: "Unmeasured scientific benefit stays null; own validation controls readiness."
        for key in result["acceptance_gate_results"]
    }
    result["field_principles"]["retirement_decisions"] = (
        "Only an identical verdict on the same named protocol can retire its exact scope."
    )
    result["field_principles"]["field_principles"] = "Explain why each field and gate exists."
    return result


def cold_replay(path: Path, root: Path) -> list[str]:
    """Re-read current bytes and rows before any terminal publication."""
    value = _read(path)
    if not value:
        return ["candidate_unreadable"]
    expected = build_candidate(root, "20260929")
    errors: list[str] = []
    for source in value.get("source_artifact_hashes", []):
        if source.get("role") == "own_reconciliation":
            continue
        candidate = Path(source["path"])
        if (sha256_file(candidate) if candidate.is_file() else None) != source["sha256"]:
            errors.append("source_bytes_changed")
    for key in (
        "task_evidence_rows",
        "rows",
        "recomputed_comparison_rows",
        "retirement_decisions",
        "sample_size_budget",
        "reproducibility_checksum",
        "milestone_evidence_complete_score",
    ):
        if value.get(key) != expected[key]:
            errors.append(f"{key}_changed")
    if (
        value.get("gate_check_summary", [])[: len(expected["gate_check_summary"])]
        != expected["gate_check_summary"]
    ):
        errors.append("gate_operands_changed")
    for item in value.get("validation_receipts", []):
        log = Path(item["log_path"])
        if not log.is_file() or sha256_file(log) != item["log_sha256"]:
            errors.append("validation_log_changed")
    return sorted(set(errors))
