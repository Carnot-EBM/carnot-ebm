"""Account for V678 evidence without turning queue progress into science.

REQ-REPORT-7808; SCENARIO-REPORT-7808-CUSTODY/DECISIONS/TERMINAL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot.experiment_7795_v678_contract_methods import compare_contract
from carnot.experiment_7807_v678_independent_evidence_audit import cold_replay as audit_cold_replay
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[2]
DESIGN = Path("docs/research-notes/v678-authority-snapshots/design.md")
ROADMAP = Path("docs/research-notes/v678-authority-snapshots/roadmap.yaml")
OUTPUT = Path("results/experiment_7808_v678_capstone.json")
MODULE = Path("python/carnot/experiment_7808_v678_capstone.py")
CLI = Path("scripts/experiments/experiment_7808_v678_capstone.py")
TEST = Path("tests/python/test_experiment_7808_v678_capstone.py")
ELIGIBLE = {"positive", "null", "circular_positive"}
GATES = (
    "validity",
    "readiness",
    "probability_quality",
    "decision_benefit",
    "retention",
    "efficiency",
)
PRINCIPLES = {
    "experiment_id": "Each result has one owner.",
    "milestone": "The result belongs to V678.",
    "run_date": "The requested date binds this run.",
    "honest_verdict": "External incompleteness cannot be fixed by retrying owned work.",
    "verdict_class": "Claim strength travels with the record.",
    "flagged_adversarial": "Invalid evidence cannot open a gate.",
    "gate_check_summary": "Missing evidence differs from a scientific null.",
    "rows": "Recompute every comparison from its units.",
    "acceptance_gate_results": "A working fixture proves no scientific gain.",
    "duration_s": "Duration reflects actual work.",
    "phase_spans": "Monotonic phase spans describe actual work.",
    "random_seed": "The aggregation is deterministic.",
    "reproducibility_checksum": "Replay requires identical inputs.",
    "sample_size_budget": "Views and seeds are not new families.",
    "source_artifact_hashes": "Old files cannot replace current producers.",
    "preconditions_checked": "Cheap failures precede compute.",
    "validation_receipts": "Every required check must pass.",
    "verifier_is_oracle": "Fixture truth cannot prove oracle-distinct benefit.",
    "claim_scope": "Exposed data cannot prove hidden generalization.",
    "inference_substrate": "Floors follow invoked work.",
    "inference_substrate_class": "The capstone only aggregates evidence.",
    "MODEL_SPECS": "A cited model was not invoked here.",
    "model_specs": "A cited model was not invoked here.",
    "model_invocation_counts": "Calls and tokens count actual invocations.",
    "task_dispositions": "Queue completion and science completion differ.",
    "publication_gates": "The unchanged stable gate controls publication.",
    "paper_ready": "Every fixed publication gate must pass.",
    "unmet_gates": "Do not improve readiness by recounting blockers.",
    "mechanism_decisions": "Continuation needs a falsifiable trigger.",
    "next_prerequisites": "Each blocked branch names its next input.",
    "outcome_note_path": "The interpretation is kept with the result.",
}


def failure(
    number: int,
    path: str,
    digest: str | None,
    field: str,
    expected: Any,
    observed: Any,
    operator: str = "==",
) -> dict[str, Any]:
    """Name the file and operand so the blocker can be checked again."""
    return dict(
        upstream_id=f"Exp{number}",
        artifact_path=path,
        artifact_hash=digest,
        field=field,
        operator=operator,
        expected=expected,
        observed=observed,
    )


def check_authority(design: bytes, roadmap: bytes) -> dict[str, Any]:
    """Compare the displayed table, embedded JSON, and frozen YAML values."""
    try:
        value = yaml.safe_load(roadmap)
        if not isinstance(value, dict):
            raise ValueError("roadmap is not an object")
        return compare_contract(design.decode(), value)
    except (UnicodeError, ValueError, KeyError, TypeError, IndexError) as exc:
        return {"passed": False, "errors": [str(exc)], "rows": []}


def account_tasks(
    root: Path, tasks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep one task row even when its science producer never started."""
    if len(tasks) != 14 or [t["id"].split("-", 1)[0] for t in tasks] != [
        f"exp{i}" for i in range(7795, 7809)
    ]:
        raise ValueError("V678 fourteen-task order required")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    values: dict[str, dict[str, Any]] = {}
    for index, task in enumerate(tasks):
        number = 7795 + index
        label = task["deliverable"]
        path = root / label
        exists = number != 7808 and path.is_file()
        digest = sha256_file(path) if exists else None
        try:
            value = json.loads(path.read_bytes()) if exists else {}
            if not isinstance(value, dict):
                raise ValueError("producer must be an object")
        except (ValueError, UnicodeError):
            value = {}
            failures.append(failure(number, label, digest, "schema", "JSON object", "invalid"))
        queue = (
            root
            / f"results/experiment_{number}_{task['id'].split('-', 1)[1].replace('-', '_')}.json"
        )
        queue_exists = not exists and number != 7808 and queue != path and queue.is_file()
        queue_label = queue.relative_to(root).as_posix() if queue_exists else None
        state = (
            "planned_output"
            if number == 7808
            else "producer"
            if exists
            else "pre_gate_receipt"
            if queue_exists
            else "absent"
        )
        eligible = bool(
            exists
            and value.get("experiment_id") == number
            and value.get("milestone") == "2026.09.678"
            and value.get("run_date") == "20260928"
            and value.get("verdict_class") in ELIGIBLE
            and value.get("flagged_adversarial") is False
            and str(value.get("honest_verdict", "")).startswith("complete_")
        )
        row = dict(
            unit_id=task["id"],
            task_id=task["id"],
            experiment_id=number,
            order=index + 1,
            phase=task["phase"],
            arm="task_accounting",
            producer_path=label,
            producer_hash=digest,
            availability=state,
            pre_gate_receipt_path=queue_label,
            pre_gate_receipt_hash=sha256_file(queue) if queue_exists else None,
            producer_eligible=eligible,
            verdict_class=value.get("verdict_class"),
            honest_verdict=value.get("honest_verdict"),
            flagged_adversarial=value.get("flagged_adversarial"),
            raw_metrics={k: v for k, v in value.items() if k.endswith("_score")},
            benefit=None,
            censored=state in {"absent", "pre_gate_receipt"},
            exclusions=[] if eligible else [state],
            effective_independent_N=None,
        )
        rows.append(row)
        values[task["id"]] = value
        if number == 7808:
            continue
        sources.append(
            dict(
                upstream_id=f"Exp{number}",
                path=label,
                sha256=digest,
                date=value.get("run_date"),
                imported_fields=["verdict_class", "honest_verdict", "gate_check_summary"]
                if exists
                else [],
                eligible=eligible,
                pre_gate_path=queue_label,
                pre_gate_sha256=row["pre_gate_receipt_hash"],
            )
        )
        if not exists:
            failures.append(
                failure(
                    number,
                    label,
                    None,
                    "producer_path",
                    "existing declared science producer",
                    "missing",
                )
            )
        elif not eligible:
            failures.append(
                failure(
                    number,
                    label,
                    digest,
                    "verdict_class",
                    sorted(ELIGIBLE),
                    value.get("verdict_class"),
                    "in",
                )
            )
        for check in value.get("gate_check_summary", []):
            failures.append(
                failure(
                    number,
                    str(check.get("artifact_path", label)),
                    check.get("artifact_hash", digest),
                    str(check.get("field", "upstream_gate")),
                    check.get("expected"),
                    check.get("observed"),
                    str(check.get("operator", "==")),
                )
            )
    for task in tasks:
        number = int(task["id"].split("-", 1)[0][3:])
        for gate in task.get("gated_on", []):
            upstream = gate["upstream"]
            source_number = int(upstream.split("-", 1)[0][3:])
            source = rows[source_number - 7795]
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
                        source_number,
                        source["producer_path"],
                        source["producer_hash"],
                        gate["artifact_field"],
                        expected,
                        observed,
                        gate["op"],
                    )
                )
    return rows, sources, failures


def mechanism_decisions(
    rows: list[dict[str, Any]], tasks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Retire repeated exact scopes; name a new input for every other task."""
    decisions = []
    for row, task in zip(rows, tasks, strict=True):
        same = [
            prior["experiment_id"]
            for prior in task.get("prior_failures", [])
            if prior.get("retire_if_same_verdict") is True
            and prior.get("verdict") == row.get("honest_verdict")
        ]
        if same:
            action = "retire"
            trigger = "New mechanism or independent evidence changes the repeated verdict."
        elif row.get("producer_eligible"):
            action = "continue"
            trigger = "A distinct, independently labeled source changes the measured outcome."
        else:
            action = "await_named_prerequisite"
            trigger = f"Qualify {row['producer_path']} with required validation and primitive rows."
        decisions.append(
            dict(
                task_id=task["id"],
                decision=action,
                trigger=trigger,
                matched_prior_ids=same,
                evidence_path=row["producer_path"],
                evidence_hash=row["producer_hash"],
            )
        )
    return decisions


def replay_candidate(value: dict[str, Any], root: Path) -> list[str]:
    """Cold-check every cited byte and refuse a circular output dependency."""
    errors = []
    for source in value.get("source_artifact_hashes", []):
        label = source.get("path")
        if label == str(OUTPUT):
            errors.append("self_input")
            continue
        path = root / str(label)
        observed = sha256_file(path) if path.is_file() else None
        if observed != source.get("sha256"):
            errors.append(str(label))
    if [row.get("experiment_id") for row in value.get("task_dispositions", [])] != list(
        range(7795, 7809)
    ):
        errors.append("task_order")
    return sorted(set(errors))


def build_artifact(
    root: Path,
    publication: dict[str, Any],
    validation: dict[str, Any],
    phase_spans: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """Reduce current evidence; leave scientific gains unmeasured when absent."""
    design = (root / DESIGN).read_bytes()
    roadmap = (root / ROADMAP).read_bytes()
    authority = check_authority(design, roadmap)
    tasks = yaml.safe_load(roadmap)["tasks"]
    rows, sources, failures = account_tasks(root, tasks)
    for label in (DESIGN, ROADMAP):
        sources.append(
            dict(
                upstream_id="V678 authority",
                path=str(label),
                sha256=sha256_file(root / label),
                date="20260928",
                imported_fields=["contract_bytes"],
                eligible=True,
            )
        )
    if not authority["passed"]:
        failures.append(
            failure(
                7808,
                str(DESIGN),
                sha256_file(root / DESIGN),
                "table_json_yaml_match",
                True,
                authority["errors"],
            )
        )
    for label, snapshot in (
        ("openspec/change-proposals/research-roadmap-vNEXT.md", design),
        ("research-roadmap.yaml", roadmap),
    ):
        path = root / label
        if path.is_file() and path.read_bytes() != snapshot:
            failures.append(
                failure(
                    7808,
                    label,
                    sha256_file(path),
                    "authority_bytes",
                    canonical_hash(snapshot.hex()),
                    sha256_file(path),
                )
            )
    audit = root / tasks[12]["deliverable"]
    audit_value = json.loads(audit.read_bytes()) if audit.is_file() else {}
    audit_recheck_errors = audit_cold_replay(audit) if audit.is_file() else ["audit_missing"]
    audit_source = sources[12]
    audit_ready = (
        audit_source["eligible"]
        and not audit_recheck_errors
        and audit_value.get("independent_evidence_ready_score") == 1
    )
    if not audit_ready:
        failures.append(
            failure(
                7807,
                tasks[12]["deliverable"],
                audit_source["sha256"],
                "independent_evidence_ready_score",
                1,
                audit_value.get("independent_evidence_ready_score"),
            )
        )
    checks_passed = bool(validation.get("required_checks_passed"))
    if not checks_passed:
        failures.append(
            failure(
                7808,
                "results/raw/experiment_7808_v678_capstone/validation",
                None,
                "required_checks_passed",
                True,
                checks_passed,
            )
        )
    own_valid = authority["passed"] and checks_passed
    verdict_class = "disqualified" if not own_valid else "blocked" if failures else "null"
    honest_verdict = (
        "complete_disqualified_required_validation"
        if not own_valid
        else "complete_blocked_required_v678_evidence"
        if failures
        else "complete_null_v678_scientific_accounting"
    )
    rows[-1].update(
        verdict_class=verdict_class,
        honest_verdict=honest_verdict,
        raw_metrics={"capstone_complete_score": int(own_valid and not failures)},
    )
    decisions = mechanism_decisions(rows, tasks)
    hardware = root / tasks[11]["deliverable"]
    hardware_value = json.loads(hardware.read_bytes()) if hardware.is_file() else {}
    board_prerequisites = [
        dict(board=row.get("board"), prerequisite=row.get("next_missing_prerequisite"))
        for row in hardware_value.get("rows", [])
        if row.get("next_missing_prerequisite")
    ]
    source_digest = canonical_hash([(s["path"], s["sha256"]) for s in sources])
    pub_gates = {gate: publication.get(gate) for gate in ("G1", "G2", "G3", "G4")}
    artifact: dict[str, Any] = dict(
        schema="carnot.exp7808.v678.capstone.v1",
        experiment_id=7808,
        milestone="2026.09.678",
        run_date="20260928",
        honest_verdict=honest_verdict,
        verdict_class=verdict_class,
        flagged_adversarial=False,
        gate_check_summary=failures,
        rows=rows,
        task_dispositions=[dict(row) for row in rows],
        acceptance_gate_results=dict(
            validity=own_valid,
            readiness=int(own_valid and not failures),
            probability_quality=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
        ),
        duration_s=duration_s,
        phase_spans=phase_spans,
        random_seed=dict(seeds=[], purpose="deterministic aggregation"),
        sample_size_budget=dict(
            intended=14,
            eligible=sum(r["producer_eligible"] for r in rows),
            started=sum(r["availability"] == "producer" for r in rows),
            completed=sum(r["verdict_class"] is not None for r in rows),
            excluded=sum(
                r["availability"] == "producer" and not r["producer_eligible"] for r in rows
            ),
            censored=sum(r["censored"] for r in rows),
            independent_N=None,
        ),
        source_artifact_hashes=sources,
        preconditions_checked=dict(
            authority_match=authority["passed"],
            authority_errors=authority["errors"],
            declared_paths=[task["deliverable"] for task in tasks],
            backend="host_cpu_aggregation",
            resources="CPU and local files",
            external_inputs_present=all(r["availability"] == "producer" for r in rows[:-1]),
        ),
        validation_receipts=validation,
        verifier_is_oracle=False,
        claim_scope=dict(
            source="640 exposed development families; no hidden generalization",
            fixture="circular_positive_only",
            qwen="current science absent",
            ARC="current scored organic measurement absent",
            hardware="complete device service absent",
            hidden_game="unmeasured",
            publication="stable_G1_to_G4_only",
            oracle_distinct_gap="reopened",
        ),
        field_principles=PRINCIPLES,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="aggregation",
        planned_inference_substrate_class="aggregation",
        actual_inference_substrate_class="aggregation",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(
            loads=0,
            generations=0,
            forwards=0,
            input_tokens=0,
            output_tokens=0,
            failures=0,
            cancellations=0,
            loaded_files=[],
        ),
        mechanism_decisions=decisions,
        next_prerequisites=[
            d["trigger"] for d in decisions if d["decision"] == "await_named_prerequisite"
        ]
        + board_prerequisites,
        branch_outcomes=dict(
            source_learning="current decision and learning science absent",
            qwen="pre-gate receipt only; no current producer",
            ARC="no eligible current scored organic SDK rows",
            service="no complete current service rows",
            hardware="no current complete service acceleration bound",
        ),
        audit_raw_recheck_errors=audit_recheck_errors,
        arc_sdk_score_recomputed=None,
        hardware_complete_service_bound=None,
        outcome_note_path="docs/research-notes/v678-outcomes.md",
        publication_gates=pub_gates,
        paper_ready=publication.get("paper_ready"),
        unmet_gates=publication.get("unmet_gates"),
        publication_gate_results=publication,
        G1=pub_gates["G1"],
        G2=pub_gates["G2"],
        G3=pub_gates["G3"],
        G4=pub_gates["G4"],
        capstone_complete_score=int(own_valid and not failures),
    )
    artifact["reproducibility_checksum"] = canonical_hash(
        dict(
            code=sha256_file(root / MODULE),
            cli=sha256_file(root / CLI),
            inputs=source_digest,
            roles="V678 capstone",
            seeds=[],
        )
    )
    return artifact
