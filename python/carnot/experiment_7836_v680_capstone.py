"""Reconcile V680 artifacts without promoting queue messages to science.

REQ-REPORT-7836. The capstone is a deterministic reader of named bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import yaml

from carnot.experiment_7823_v680_contract_methods import compare_contract
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[2]
DESIGN = Path("docs/research-notes/v680-authority-snapshots/design.md")
ROADMAP = Path("docs/research-notes/v680-authority-snapshots/roadmap.yaml")
OUTPUT = Path("results/experiment_7836_v680_capstone.json")
RAW = Path("results/raw/experiment_7836_v680_capstone")
MANIFEST = RAW / "validation_command_manifest.json"
MANIFEST_SHA = "12ef461633c280ae6dabcf6e23b064b0921c00970b61e504c6159c6641f6b347"
PRE_GATE = {
    7826: "results/experiment_7826_view_energy_fit.json",
    7829: "results/experiment_7829_qwen_counter_evidence.json",
}
ELIGIBLE = {"positive", "circular_positive", "null"}
GATE_NAMES = (
    "validity",
    "readiness",
    "probability_quality",
    "decision_benefit",
    "retention",
    "efficiency",
)
PRINCIPLE = "Preserve the stated evidence boundary."
FIELDS = (
    "experiment_id",
    "milestone",
    "run_date",
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "rows",
    "acceptance_gate_results",
    "duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "sample_size_budget",
    "source_artifact_hashes",
    "preconditions_checked",
    "validation_receipts",
    "validation_command_manifest_path",
    "observed_child_commands",
    "repository_health",
    "verifier_is_oracle",
    "claim_scope",
    "field_principles",
    "inference_substrate",
    "inference_substrate_class",
    "MODEL_SPECS",
    "model_specs",
    "model_invocation_counts",
    "capstone_complete_score",
    "task_dispositions",
    "paper_ready",
    "unmet_gates",
    "mechanism_decisions",
    "outcome_note_path",
)


def failed(
    upstream: str, path: str, digest: str | None, field: str, op: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Retain the exact two operands of one failed precondition."""
    return dict(
        upstream_id=upstream,
        path=path,
        hash=digest,
        field=field,
        op=op,
        expected=expected,
        observed=observed,
    )


def load_authority(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare immutable visible design, machine JSON and YAML independently."""
    design = (root / DESIGN).read_text()
    roadmap = yaml.safe_load((root / ROADMAP).read_text())
    comparison = compare_contract(design, roadmap)
    tasks = roadmap["tasks"]
    if not comparison["passed"] or [t["id"].split("-", 1)[0] for t in tasks] != [
        f"exp{n}" for n in range(7823, 7837)
    ]:
        raise ValueError("V680 authority mismatch")
    return tasks, comparison


def account(
    root: Path, tasks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Account for each producer, receipt and declared exact gate in order."""
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    values: dict[str, dict[str, Any]] = {}
    for order, task in enumerate(tasks, 1):
        number = 7822 + order
        label = task["deliverable"]
        path = root / label
        exists = number != 7836 and path.is_file()
        digest = sha256_file(path) if exists else None
        value = json.loads(path.read_bytes()) if exists else {}
        if not isinstance(value, dict):
            value = {}
        values[task["id"]] = value
        receipt_label = PRE_GATE.get(number)
        receipt = root / receipt_label if receipt_label else None
        receipt_hash = sha256_file(receipt) if receipt and receipt.is_file() else None
        availability = (
            "planned_output"
            if number == 7836
            else "producer"
            if exists
            else "pre_gate_receipt"
            if receipt_hash
            else "absent"
        )
        eligible = bool(
            exists
            and value.get("experiment_id") in (number, task["id"])
            and value.get("milestone") == "2026.09.680"
            and value.get("run_date") == "20260928"
            and value.get("flagged_adversarial") is False
            and value.get("verdict_class") in ELIGIBLE
            and str(value.get("honest_verdict", "")).startswith("complete_")
        )
        row = dict(
            order=order,
            unit_id=task["id"],
            task_id=task["id"],
            experiment_id=number,
            title=task["title"],
            phase=task["phase"],
            arm="task_accounting",
            producer_path=label,
            producer_hash=digest,
            availability=availability,
            pre_gate_receipt_path=receipt_label if receipt_hash else None,
            pre_gate_receipt_hash=receipt_hash,
            producer_eligible=eligible,
            verdict_class=value.get("verdict_class"),
            honest_verdict=value.get("honest_verdict"),
            flagged_adversarial=value.get("flagged_adversarial"),
            benefit=None,
            raw_metrics={k: v for k, v in value.items() if k.endswith("_score")},
            raw_rows_path=value.get("raw_rows_path"),
            censored=availability != "producer",
            exclusions=[]
            if eligible
            else [availability if availability != "producer" else "ineligible_producer"],
            effective_independent_N=(value.get("sample_size_budget") or {}).get("independent_n"),
        )
        rows.append(row)
        if number == 7836:
            continue
        sources.append(
            dict(
                upstream_id=f"Exp{number}",
                path=label,
                sha256=digest,
                date=value.get("run_date"),
                role="science_producer"
                if task["inference_substrate_class"] != "aggregation"
                else "administrative_or_audit_producer",
                eligibility=eligible,
                availability=availability,
            )
        )
        if receipt_hash:
            sources.append(
                dict(
                    upstream_id=f"Exp{number}",
                    path=receipt_label,
                    sha256=receipt_hash,
                    date="20260928",
                    role="conductor_pre_gate_receipt",
                    eligibility=False,
                    availability="receipt",
                )
            )
        if not exists:
            failures.append(
                failed(
                    f"Exp{number}",
                    label,
                    None,
                    "producer_path",
                    "==",
                    "existing declared producer",
                    "missing",
                )
            )
        elif not eligible:
            failures.append(
                failed(
                    f"Exp{number}",
                    label,
                    digest,
                    "verdict_class",
                    "in",
                    sorted(ELIGIBLE),
                    value.get("verdict_class"),
                )
            )
        for check in value.get("gate_check_summary", []):
            if isinstance(check, dict) and check.get("passed") is not True:
                failures.append(
                    failed(
                        str(check.get("upstream_id", f"Exp{number}")),
                        str(check.get("path", check.get("artifact_path", label))),
                        check.get(
                            "hash", check.get("artifact_hash", check.get("artifact_sha256", digest))
                        ),
                        str(check.get("field", "upstream_gate")),
                        str(check.get("op", check.get("operator", "=="))),
                        check.get("expected"),
                        check.get("observed"),
                    )
                )
    for task in tasks:
        for gate in task.get("gated_on", []):
            upstream = gate["upstream"]
            source = rows[int(upstream.split("-", 1)[0][3:]) - 7823]
            observed, expected = values[upstream].get(gate["artifact_field"]), gate["value"]
            passed = (
                observed == expected
                if gate["op"] == "=="
                else observed in expected
                if gate["op"] == "in"
                else False
            )
            if not passed:
                failures.append(
                    failed(
                        upstream,
                        source["producer_path"],
                        source["producer_hash"],
                        gate["artifact_field"],
                        gate["op"],
                        expected,
                        observed,
                    )
                )
    return rows, sources, failures


def decisions(rows: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retire only a listed same-verdict scope and name a measurable reopening."""
    result = []
    for row, task in zip(rows, tasks, strict=True):
        matched = [
            prior["experiment_id"]
            for prior in task.get("prior_failures", [])
            if prior.get("retire_if_same_verdict") is True
            and prior.get("verdict") == row["honest_verdict"]
        ]
        decision = (
            "retire"
            if matched
            else "continue"
            if row["producer_eligible"]
            else "await_named_prerequisite"
        )
        trigger = (
            f"Change the exact {row['honest_verdict']} scope with new independent primitive rows."
            if matched
            else "Repeat this qualified mechanism on independent unexposed families with fixed controls."
            if row["producer_eligible"]
            else f"Produce {row['producer_path']} with passing required checks and primitive rows."
        )
        result.append(
            dict(
                task_id=task["id"],
                decision=decision,
                trigger=trigger,
                matched_prior_ids=matched,
                evidence_path=row["producer_path"],
                evidence_hash=row["producer_hash"],
            )
        )
    return result


def build_result(root: Path, publication: dict[str, Any]) -> dict[str, Any]:
    """Build one terminal, prevalidation accounting from current named bytes."""
    tasks, authority = load_authority(root)
    rows, sources, failures = account(root, tasks)
    for label in (DESIGN, ROADMAP):
        sources.append(
            dict(
                upstream_id="V680 authority",
                path=str(label),
                sha256=sha256_file(root / label),
                date="20260928",
                role="immutable_authority",
                eligibility=True,
            )
        )
    audit = (
        json.loads((root / tasks[12]["deliverable"]).read_bytes())
        if rows[12]["producer_hash"]
        else {}
    )
    audit_ready = (
        rows[12]["producer_eligible"] and audit.get("independent_evidence_ready_score") == 1
    )
    if not audit_ready:
        failures.append(
            failed(
                "Exp7835",
                tasks[12]["deliverable"],
                rows[12]["producer_hash"],
                "independent_evidence_ready_score",
                "==",
                1,
                audit.get("independent_evidence_ready_score"),
            )
        )
    source_by_number = {row["experiment_id"]: row for row in rows}
    branch_map = {
        "source_isolation": 7824,
        "calibration": 7826,
        "source_sensitivity": 7829,
        "selective_utility": 7832,
        "causal_acquisition": 7830,
        "retention": 7830,
        "complete_service_cost": 7833,
    }
    branches = {
        name: dict(
            state="blocked",
            producer=row["producer_path"],
            producer_hash=row["producer_hash"],
            benefit=None,
            independent_n=None,
            audit_state=next(
                (
                    a.get("state")
                    for a in audit.get("branch_dispositions", [])
                    if a.get("upstream_id") == f"Exp{number}"
                ),
                None,
            ),
        )
        for name, number in branch_map.items()
        for row in [source_by_number[number]]
    }
    verdict = (
        "complete_blocked_required_v680_evidence" if failures else "complete_null_v680_accounting"
    )
    rows[-1].update(
        verdict_class="blocked" if failures else "null",
        honest_verdict=verdict,
        raw_metrics={"capstone_complete_score": int(not failures)},
        censored=False,
        exclusions=["external_evidence_missing"] if failures else [],
    )
    budget = dict(
        intended=14,
        eligible=sum(row["producer_eligible"] for row in rows[:-1]),
        started=sum(row["availability"] == "producer" for row in rows[:-1]),
        completed=sum(row["verdict_class"] is not None for row in rows),
        excluded=sum(
            row["availability"] == "producer" and not row["producer_eligible"] for row in rows[:-1]
        ),
        censored=sum(row["censored"] for row in rows),
        independent_n=0,
    )
    value: dict[str, Any] = dict(
        schema="carnot.exp7836.v680.capstone.v1",
        experiment_id=7836,
        milestone="2026.09.680",
        run_date="20260928",
        honest_verdict=verdict,
        verdict_class=rows[-1]["verdict_class"],
        flagged_adversarial=False,
        gate_check_summary=failures,
        rows=rows,
        task_dispositions=[dict(row) for row in rows],
        acceptance_gate_results=dict(
            validity=True,
            readiness=int(not failures),
            probability_quality=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
        ),
        duration_s=0.0,
        phase_spans=[],
        random_seed=dict(seeds=[], purpose="deterministic aggregation"),
        sample_size_budget=budget,
        source_artifact_hashes=sources,
        preconditions_checked=dict(
            authority_match=authority["passed"],
            authority_errors=authority["errors"],
            declared_paths=[task["deliverable"] for task in tasks],
            input_presence=[
                dict(
                    task_id=row["task_id"],
                    availability=row["availability"],
                    hash=row["producer_hash"],
                )
                for row in rows
            ],
            failed_operands=failures,
            resources="CPU and local files; no model or board load",
            required_resources=dict(cpu=True, model=False, board=False),
        ),
        validation_receipts={"checks": [], "required_checks_passed": None},
        validation_command_manifest_path=str(MANIFEST),
        observed_child_commands=[],
        repository_health={"status": "historical_failures_open"},
        verifier_is_oracle=False,
        claim_scope=dict(
            all_640_source_families_exposed=True,
            fresh_generalization_eligible=False,
            fixtures="circular_positive_only",
            September_28_oracle_distinct_corrigendum="open",
            publication="operator_owned_G1_G4",
            hidden_game="no_new_level_claim",
        ),
        field_principles={field: PRINCIPLE for field in FIELDS},
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="aggregation",
        planned_inference_substrate_class="aggregation",
        actual_inference_substrate_class="aggregation",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(
            loads=0, generations=0, forwards=0, input_tokens=0, output_tokens=0, loaded_files=[]
        ),
        capstone_complete_score=int(not failures),
        mechanism_decisions=decisions(rows, tasks),
        outcome_note_path="docs/research-notes/v680-outcomes.md",
        publication_gates={
            name: publication.get("gates", {}).get(name, {}).get("pass")
            for name in ("G1", "G2", "G3", "G4")
        },
        paper_ready=publication.get("paper_ready"),
        unmet_gates=publication.get("unmet_gates"),
        publication_gate_results=publication,
        branch_outcomes=branches,
        audit_branch_dispositions=audit.get("branch_dispositions", []),
        arc_selector_promoted=False,
        arc_plain_rerun_scheduled=False,
        arc_outcome_ledger=dict(
            state="refinement_only", new_level_solve=None, organic_selector="not_promoted"
        ),
    )
    value["validation_command_manifest_sha256"] = sha256_file(root / MANIFEST)
    value["reproducibility_checksum"] = canonical_hash(
        dict(
            seed=[],
            code=sha256_file(root / "python/carnot/experiment_7836_v680_capstone.py"),
            cli=sha256_file(root / "scripts/experiments/experiment_7836_v680_capstone.py"),
            manifest=value["validation_command_manifest_sha256"],
            sources=[(item["path"], item["sha256"]) for item in sources],
        )
    )
    return value


def load_manifest(root: Path, path: Path | None = None) -> dict[str, Any]:
    """Reject changed command names, argv or classes before any child starts."""
    selected = path or root / MANIFEST
    if sha256_file(selected) != f"sha256:{MANIFEST_SHA}":
        raise ValueError("frozen validation manifest mismatch")
    value = json.loads(selected.read_bytes())
    if (
        value.get("schema") != "carnot.exp7836.validation.v1"
        or len(value.get("commands", [])) != 16
    ):
        raise ValueError("frozen validation manifest schema mismatch")
    return value


def dispatch(
    root: Path,
    manifest: dict[str, Any],
    executor: Callable[[Path, dict[str, Any], Path], dict[str, Any]],
    before_child: Callable[[int, list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Dispatch only the frozen commands and compare each observed child."""
    receipts: list[dict[str, Any]] = []
    durable = root / RAW / "sealed_logs"
    for index, command in enumerate(manifest["commands"]):
        if before_child:
            before_child(index, receipts)
        spec = dict(
            command,
            private_root=str(root / RAW / "validation" / f"attempt-{index:02d}-{uuid4().hex}"),
        )
        receipt = executor(root, spec, durable)
        if (receipt.get("name"), receipt.get("command_argv"), receipt.get("classification")) != (
            command["name"],
            command["argv"],
            command["classification"],
        ):
            raise ValueError("undeclared child receipt")
        receipts.append(receipt)
    return receipts


def check_logs(receipts: list[dict[str, Any]]) -> list[str]:
    """A retry may add logs but must not change earlier sealed bytes."""
    errors = []
    for receipt in receipts:
        path = Path(receipt["log_path"])
        if not path.is_file() or sha256_file(path) != receipt["log_sha256"]:
            errors.append(str(path))
    return errors


def cold_replay(candidate: Path, root: Path = ROOT) -> list[str]:
    """Reopen source and log bytes in a fresh process after publication."""
    value = json.loads(candidate.read_bytes())
    errors = check_logs(value.get("observed_child_commands", []))
    for source in value.get("source_artifact_hashes", []):
        path = root / source["path"]
        if (sha256_file(path) if path.is_file() else None) != source["sha256"]:
            errors.append(source["path"])
    if value.get("validation_command_manifest_sha256") != sha256_file(root / MANIFEST):
        errors.append(str(MANIFEST))
    tasks, _ = load_authority(root)
    rows, _, _ = account(root, tasks)
    if value.get("task_dispositions", [])[:-1] != rows[:-1]:
        errors.append("task_dispositions")
    return sorted(set(errors))
