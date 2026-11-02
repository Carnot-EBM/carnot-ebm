"""Direct V681 capstone reader with typed evidence and byte custody.

REQ-REPORT-7850: account for declared current artifacts without running old
experiment dispatchers or treating a completed queue as scientific evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.roadmap_contract import compare_contract

DESIGN = Path("docs/research-notes/v681-authority-snapshots/design.md")
STAGED = Path("docs/research-notes/v681-authority-snapshots/staged.yaml")
ACTIVE = Path("docs/research-notes/v681-authority-snapshots/active.yaml")
OUTPUT = Path("results/experiment_7850_v681_capstone.json")
BASE = Path("results/raw/experiment_7850_v681_capstone")
MANIFEST = BASE / "validation_command_manifest.json"
REQUIRED = {
    "worktree_imports",
    "affected_pytest",
    "changed_coverage",
    "ruff_check",
    "ruff_format",
    "mypy",
    "scoped_spec",
    "cli_e2e",
    "cold_replay",
    "adversarial_verify",
    "strict_rows",
}
QUALIFIED = {"positive", "circular_positive", "null"}
SCIENCE = {7838, 7840, 7841, 7842, 7843, 7844, 7846, 7848}
GATES = (
    "validity",
    "readiness",
    "probability_quality",
    "decision_benefit",
    "retention",
    "efficiency",
)
FIELDS = (
    "experiment_id",
    "task_id",
    "milestone",
    "run_date",
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "preconditions_checked",
    "validation_receipts",
    "validation_command_manifest_path",
    "observed_child_commands",
    "repository_health",
    "verifier_is_oracle",
    "claim_scope",
    "inference_substrate",
    "inference_substrate_class",
    "MODEL_SPECS",
    "model_specs",
    "model_invocation_counts",
    "milestone_evidence_ready_score",
    "task_dispositions",
    "continuation_decisions",
    "g1",
    "g2",
    "g3",
    "g4",
    "paper_ready",
    "unmet_gates",
)
PRINCIPLE = (
    "A claim is limited by exact current evidence, independent units and required validation."
)


def load_tasks(root: Path) -> list[dict[str, Any]]:
    """Check all three independent task representations before reading results."""
    directory = root / STAGED.parent
    staged = yaml.safe_load((directory / STAGED.name).read_text())
    active = yaml.safe_load((directory / ACTIVE.name).read_text())
    comparison = compare_contract((root / DESIGN).read_text(), staged, active)
    if not comparison["passed"]:
        raise ValueError(f"V681 authority mismatch: {comparison['errors']}")
    return staged["tasks"]


def load_manifest(root: Path) -> dict[str, Any]:
    """Require the preregistered named checks and one separate diagnostic."""
    value = json.loads((root / MANIFEST).read_bytes())
    commands = value["commands"]
    if (
        value.get("schema") != "carnot.exp7850.validation.v1"
        or {c["name"] for c in commands if c["classification"] == "required"} != REQUIRED
        or {c["name"] for c in commands if c["classification"] == "diagnostic"}
        != {"repository_health_180s"}
    ):
        raise ValueError("frozen validation manifest mismatch")
    return value


def failure(
    upstream: str, path: str, digest: str | None, field: str, op: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Retain the exact path, byte identity and comparison operands."""
    return dict(
        upstream_id=upstream,
        path=path,
        hash=digest,
        artifact_field=field,
        op=op,
        expected=expected,
        observed=observed,
    )


def source(
    root: Path, path: Path, role: str, eligible: bool, date: str | None = None
) -> dict[str, Any]:
    """Name exact source bytes and their role without promoting a side receipt."""
    return dict(
        path=str(path),
        sha256=sha256_file(root / path) if (root / path).is_file() else None,
        role=role,
        eligibility=eligible,
        date=date,
    )


def account(
    root: Path, tasks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read all fourteen current declared paths and every same-roadmap gate."""
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    values: dict[str, dict[str, Any]] = {}
    conductor = root / "ops/conductor-log.md"
    conductor_text = conductor.read_text() if conductor.is_file() else ""
    for order, task in enumerate(tasks, 1):
        number = 7836 + order
        label = Path(task["deliverable"])
        exists = number != 7850 and (root / label).is_file()
        digest = sha256_file(root / label) if exists else None
        try:
            value = json.loads((root / label).read_bytes()) if exists else {}
        except (ValueError, OSError):
            value = {}
        if not isinstance(value, dict):
            value = {}
        values[task["id"]] = value
        typed = (
            value.get("experiment_id") == number
            and value.get("task_id") == task["id"]
            and value.get("milestone") == "2026.09.681"
            and value.get("run_date") in {"20260928", "20260929"}
            and value.get("inference_substrate_class") == task["inference_substrate_class"]
            and value.get("MODEL_SPECS") == task["MODEL_SPECS"]
        )
        verdict = value.get("verdict_class")
        eligible = bool(
            exists
            and typed
            and verdict in QUALIFIED
            and value.get("flagged_adversarial") is False
            and str(value.get("honest_verdict", "")).startswith("complete_")
            and value.get("validation_receipts") is not None
        )
        availability = "planned_output" if number == 7850 else "producer" if exists else "absent"
        conductor_lines = [
            line for line in conductor_text.splitlines() if task["title"][:35] in line
        ][-3:]
        budget = value.get("sample_size_budget") or {}
        row = dict(
            order=order,
            unit_id=task["id"],
            task_id=task["id"],
            experiment_id=number,
            title=task["title"],
            phase=task["phase"],
            arm="task_accounting",
            family=task["id"],
            seed=None,
            producer_path=str(label),
            producer_hash=digest,
            availability=availability,
            status="unstarted" if not exists else "completed",
            verdict_class=verdict if exists else "absent" if number != 7850 else "planned",
            honest_verdict=value.get("honest_verdict"),
            flagged_adversarial=value.get("flagged_adversarial"),
            producer_eligible=eligible,
            conductor_receipts=conductor_lines,
            raw_metrics={k: v for k, v in value.items() if k.endswith("_score")},
            raw_rows_path=value.get("raw_rows_path"),
            intended=1,
            eligible=int(eligible),
            started=int(exists),
            completed=int(exists and verdict is not None),
            censored=int(not exists and number != 7850),
            excluded=int(exists and not eligible),
            independent_n=budget.get("independent_n", 0) if eligible and number in SCIENCE else 0,
        )
        rows.append(row)
        if number == 7850:
            continue
        sources.append(
            source(
                root,
                label,
                "science_producer" if number in SCIENCE else "administrative_or_other_producer",
                eligible,
                value.get("run_date"),
            )
        )
        if not exists:
            failures.append(
                failure(
                    task["id"],
                    str(label),
                    None,
                    "producer_path",
                    "==",
                    "existing declared producer",
                    "missing",
                )
            )
        elif not eligible:
            for field, expected, observed in (
                ("experiment_id", number, value.get("experiment_id")),
                ("task_id", task["id"], value.get("task_id")),
                (
                    "inference_substrate_class",
                    task["inference_substrate_class"],
                    value.get("inference_substrate_class"),
                ),
                ("MODEL_SPECS", task["MODEL_SPECS"], value.get("MODEL_SPECS")),
                ("verdict_class", sorted(QUALIFIED), verdict),
                ("flagged_adversarial", False, value.get("flagged_adversarial")),
            ):
                if observed not in expected if field == "verdict_class" else observed != expected:
                    failures.append(
                        failure(
                            task["id"],
                            str(label),
                            digest,
                            field,
                            "in" if field == "verdict_class" else "==",
                            expected,
                            observed,
                        )
                    )
    for task in tasks:
        for gate in task.get("gated_on", []):
            upstream = gate["upstream"]
            upstream_row = next(row for row in rows if row["task_id"] == upstream)
            observed = values[upstream].get(gate["artifact_field"])
            expected = gate["value"]
            passed = (
                observed == expected
                if gate["op"] == "=="
                else observed in expected
                if gate["op"] == "in"
                else False
            )
            if not passed:
                failures.append(
                    failure(
                        upstream,
                        upstream_row["producer_path"],
                        upstream_row["producer_hash"],
                        gate["artifact_field"],
                        gate["op"],
                        expected,
                        observed,
                    )
                )
    return rows, sources, failures


def decisions(rows: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose a bounded action from each measured defect or missing resource."""
    triggers = {
        7837: "Repair current contract validation and show identical fourteen-row authority with passing required checks.",
        7838: "Supply qualified public source projection with unknown-label masking and independent label isolation.",
        7839: "Repair required protocol checks and emit resolved worktree imports with complete CLI coverage.",
        7840: "Obtain qualified Exp7838 boundary, then measure natural source-conditioned fit against frozen length control.",
        7841: "Obtain qualified Exp7840 heads and show lower family-level Brier and typed cost than matched controls.",
        7842: "Obtain qualified Exp7839 protocol and measure source deletion against matched intervention controls.",
        7843: "Obtain qualified Exp7840 fit and show delayed-constraint benefit with retained decisions after restart.",
        7844: "Obtain qualified Exp7840 fit and show retained utility gain at matched abstention coverage.",
        7845: "Collect new authenticated supervisor receipt hashes before a changed-board probe; no solve credit from empty ledger.",
        7846: "Obtain qualified Exp7840 fit and measure complete update and inference service cost on the same workload.",
        7847: "Supply owned board evidence with passing required validation before comparing any acceleration.",
        7848: "Repair required length-control validation; make new source predictions beyond response/source length.",
        7849: "Supply eight qualified current science producers and independently reconstruct primitive family rows.",
        7850: "Reopen only after all current science and own frozen required checks qualify.",
    }
    result = []
    for row, task in zip(rows, tasks, strict=True):
        matched = [
            prior["experiment_id"]
            for prior in task["prior_failures"]
            if prior["retire_if_same_verdict"] is True and prior["verdict"] == row["honest_verdict"]
        ]
        decision = (
            "retire"
            if matched
            else "continue"
            if row["producer_eligible"]
            else "await_named_prerequisite"
        )
        result.append(
            dict(
                task_id=task["id"],
                experiment_id=row["experiment_id"],
                decision=decision,
                trigger=triggers[row["experiment_id"]],
                measured_defect_or_effect=row["raw_metrics"]
                or row["honest_verdict"]
                or "declared producer absent",
                primary_artifact=row["producer_path"],
                primary_artifact_hash=row["producer_hash"],
                matched_prior_ids=matched,
            )
        )
    return result


def artifact(root: Path, label: str) -> dict[str, Any]:
    """Treat a missing or malformed external producer as absent evidence."""
    path = root / label
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def build_candidate(root: Path, date: str, publication: dict[str, Any]) -> dict[str, Any]:
    """Build a terminal evidence candidate from exact current source bytes."""
    tasks = load_tasks(root)
    rows, sources, failures = account(root, tasks)
    contract = artifact(root, tasks[0]["deliverable"])
    contract_ok = rows[0]["producer_eligible"] and contract.get("contract_ready_score") == 1
    if not contract_ok:
        failures.append(
            failure(
                tasks[0]["id"],
                tasks[0]["deliverable"],
                rows[0]["producer_hash"],
                "contract_ready_score",
                "==",
                1,
                contract.get("contract_ready_score"),
            )
        )
    audit = artifact(root, tasks[12]["deliverable"])
    audit_ok = rows[12]["producer_eligible"] and audit.get("independent_evidence_ready_score") == 1
    if not audit_ok:
        failures.append(
            failure(
                tasks[12]["id"],
                tasks[12]["deliverable"],
                rows[12]["producer_hash"],
                "independent_evidence_ready_score",
                "==",
                1,
                audit.get("independent_evidence_ready_score"),
            )
        )
    length = artifact(root, tasks[11]["deliverable"])
    if not rows[11]["producer_eligible"] or length.get("length_control_ready_score") != 1:
        failures.append(
            failure(
                tasks[11]["id"],
                tasks[11]["deliverable"],
                rows[11]["producer_hash"],
                "length_control_ready_score",
                "==",
                1,
                length.get("length_control_ready_score"),
            )
        )
    science_ready = all(rows[number - 7837]["producer_eligible"] for number in SCIENCE)
    ready = bool(contract_ok and audit_ok and science_ready and not failures)
    verdict = (
        "complete_null_v681_accounting" if ready else "complete_blocked_required_v681_evidence"
    )
    rows[-1].update(
        status="completed",
        verdict_class="null" if ready else "blocked",
        honest_verdict=verdict,
        completed=1,
        censored=0,
        raw_metrics={"milestone_evidence_ready_score": int(ready)},
    )
    for label in (
        DESIGN,
        STAGED,
        ACTIVE,
        MANIFEST,
        Path("ops/conductor-log.md"),
        Path("results/experiment_7836_v680_capstone.json"),
        Path("python/carnot/reporting/v681_capstone.py"),
        Path("scripts/experiments/experiment_7850_v681_capstone.py"),
    ):
        sources.append(
            source(root, label, "authority_or_provenance", (root / label).is_file(), date)
        )
    gates = publication.get("gates", {})
    budget = {
        key: sum(int(row[key] or 0) for row in rows)
        for key in (
            "intended",
            "eligible",
            "started",
            "completed",
            "censored",
            "excluded",
            "independent_n",
        )
    }
    budget["independent_unit"] = "source_family_not_seed_or_view"
    value: dict[str, Any] = dict(
        schema="carnot.exp7850.v681.capstone.v1",
        experiment_id=7850,
        task_id=tasks[-1]["id"],
        milestone="2026.09.681",
        run_date=date,
        honest_verdict=verdict,
        verdict_class="null" if ready else "blocked",
        flagged_adversarial=False,
        gate_check_summary=failures,
        rows=rows,
        task_dispositions=[dict(row) for row in rows],
        continuation_decisions=decisions(rows, tasks),
        sample_size_budget=budget,
        milestone_evidence_ready_score=int(ready),
        acceptance_gate_results={
            "validity": contract_ok,
            "readiness": int(ready),
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        duration_s=0.0,
        phase_spans=[],
        random_seed={"seeds": [], "purpose": "deterministic_aggregation"},
        source_artifact_hashes=sources,
        preconditions_checked={
            "authority_pair": [str(DESIGN), str(STAGED)],
            "exp7837_snapshot_qualified": contract_ok,
            "resource_ownership": "CPU/local files; no board, model or external job owned",
            "failed_operands": failures,
        },
        validation_receipts=[],
        validation_command_manifest_path=str(MANIFEST),
        validation_command_manifest_sha256=sha256_file(root / MANIFEST),
        observed_child_commands=[],
        repository_health={
            "status": "pending_diagnostic",
            "historical_required_failures": [
                {"path": str(tasks[0]["deliverable"]), "verdict": contract.get("honest_verdict")},
                {
                    "path": "results/experiment_7836_v680_capstone.json",
                    "verdict": "complete_disqualified_required_validation",
                },
            ],
        },
        verifier_is_oracle=False,
        claim_scope={
            "fresh_generalization_eligible": False,
            "source_families": "historically_exposed_development",
            "fixtures": "circular_positive_only",
            "September_oracle_distinct_corrigendum": "open",
            "publication": "legacy_G1_G4_only",
            "activation": False,
            "registry_solve_credit": False,
        },
        field_principles={field: PRINCIPLE for field in FIELDS},
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="aggregation",
        planned_inference_substrate_class="aggregation",
        actual_inference_substrate_class="aggregation",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts={
            "model_loads_attempted": 0,
            "generation_calls_attempted": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "loaded_files": [],
        },
        publication_gate_results=publication,
        paper_ready=publication.get("paper_ready"),
        unmet_gates=publication.get("unmet_gates", []),
        prd_gaps={
            "FR-12_independent_verification": "blocked: no qualified source-conditioned decision gain",
            "FR-11_continuous_learning": "blocked: no qualified delayed retention result",
            "FR-05_08_09_10_NFR-01_execution": "blocked: required validation and board/service evidence missing",
        },
    )
    value.update({f"g{i}": gates.get(f"G{i}", {}).get("pass") for i in range(1, 5)})
    value["reproducibility_checksum"] = canonical_hash(
        {
            "sources": [(s["path"], s["sha256"]) for s in sources],
            "config": value["validation_command_manifest_sha256"],
            "seeds": [],
            "date": date,
        }
    )
    return value


def cold_replay(root: Path, path: Path) -> list[str]:
    """Recompute declared hashes and primitive task rows from cold bytes."""
    value = json.loads(path.read_bytes())
    errors = []
    for item in value["source_artifact_hashes"]:
        source_path = root / item["path"]
        observed = sha256_file(source_path) if source_path.is_file() else None
        if observed != item["sha256"]:
            errors.append(item["path"])
    if sha256_file(root / MANIFEST) != value["validation_command_manifest_sha256"]:
        errors.append(str(MANIFEST))
    for receipt in value["validation_receipts"]:
        log = Path(receipt["log_path"])
        if not log.is_absolute():
            log = root / log
        if not log.is_file() or sha256_file(log) != receipt["log_sha256"]:
            errors.append(str(log))
    rows, _, _ = account(root, load_tasks(root))
    for observed, fresh in zip(value["rows"][:-1], rows[:-1], strict=True):
        if observed != fresh:
            errors.append("rows")
            break
    return errors
