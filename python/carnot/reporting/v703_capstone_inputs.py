"""REQ-REPORT-8135: preserve exact external operands before reducing any branch.

An immutable activation receipt survives later roadmap edits. Each branch still
has to qualify its own primary; custody cannot turn missing science into a null.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from carnot.reporting import v702_capstone_inputs as previous
from carnot.reporting import v703_contract_custody as custody
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import tasks_digest

Json = dict[str, Any]
ROOT = previous.ROOT
INPUT = "results/experiment_8123_v703_contract_custody.json"
reference = previous.reference
failure = previous.failure
NAMED = list(
    dict.fromkeys(
        [
            *previous.NAMED,
            "scripts/recurring_blocker_ledger.py",
            "scripts/verdict_row_consistency_lint.py",
            INPUT,
            "scripts/experiments/experiment_8122_v702_capstone.py",
        ]
    )
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real accounting counts because this process never performs model work."""
    print(f"[exp8135] phase={phase} completed={completed} pending={pending}", flush=True)


def authorities(root: Path, raw: Path) -> tuple[list[Json], Json, list[Json]]:
    """Authenticate table, full JSON, digest and sealed activation separately."""
    receipt = json.loads((root / INPUT).read_bytes())
    snaps = receipt["authority_snapshots"]
    active, design = [
        checked(dict(path=snaps[k]["snapshot_path"], sha256=snaps[k]["sha256"]))
        for k in ("active", "design")
    ]
    assessment = custody.assess(design, root / "research-roadmap-next.yaml", active, raw)
    table, tasks = parse_design(design.read_text(), milestone="2026.10.703")
    if tasks_digest(tasks) != receipt["canonical_tasks_sha256"]:
        raise ValueError("receipt_tasks_digest_drift")
    assessment["visible_table"] = table
    assessment["embedded_tasks"] = tasks
    refs = [reference(root / INPUT), reference(active), reference(design)]
    live = root / "research-roadmap.yaml"
    if live.is_file():
        current = yaml.safe_load(live.read_bytes())
        if current.get("milestone") == "2026.10.703" and current["tasks"] != tasks:
            assessment["gate_check_summary"].append(
                failure(
                    live,
                    "active_roadmap",
                    "tasks_digest",
                    tasks_digest(tasks),
                    tasks_digest(current["tasks"]),
                )
            )
            assessment["activated"] = False
        refs.append(reference(live))
    return tasks, assessment, refs


def deduplicate(gates: list[Json]) -> list[Json]:
    """One failed operand is recorded once even when several consumers repeat it."""
    seen, result = set(), []
    for gate in gates:
        path = Path(gate.get("path", gate.get("artifact_path", "missing")))
        if not path.is_absolute():
            path = ROOT / path
        row = dict(
            check=gate.get("check", gate.get("artifact_field")),
            upstream=gate.get("upstream", gate.get("upstream_id")),
            path=str(path),
            hash=gate.get("hash", gate.get("artifact_hash")),
            artifact_field=gate.get("artifact_field", gate.get("field")),
            op=gate.get("op", "=="),
            expected=gate.get("expected"),
            observed=gate.get("observed"),
            passed=gate.get("passed", False),
        )
        key = canonical_hash({k: v for k, v in row.items() if k not in ("hash", "check")})
        if key not in seen:
            result.append(row)
            seen.add(key)
    return result


def primitive_audit(value: Json, number: int) -> Json:
    """Reuse qualified equations on original cached rows without timing a new service."""
    if number == 8134 and "replay_input_reference" in value:
        from carnot.reporting.hardware_service_8134 import reduce

        ref = value["replay_input_reference"]
        fresh = reduce(json.loads(checked(ref).read_bytes()))
        keys = ("board_rows", "workload_operation_rows", "quantization_rows", "amdahl_bounds")
        result = {key: fresh[key] for key in keys}
        if any(result[key] != value[key] for key in keys):
            raise ValueError("hardware_primitive_reduction_drift")
        return dict(available=True, primitive_reference=ref, result=result, kind="hardware")
    if number != 8132 or "primitive_rows" not in value:
        return dict(available=False, reason="No available comparative primitive receipt")
    from carnot.experiment_8132_v703_service_cost import reduce_rows

    ref = value["primitive_rows"]
    evidence = json.loads(checked(ref).read_bytes())
    result = reduce_rows(evidence, evidence["config"])
    if result != value["reduction"]:
        raise ValueError("service_primitive_reduction_drift")
    return dict(available=True, primitive_reference=ref, result=result)


def load(root: Path, raw: Path) -> Json:
    """Retain missing slots, rejected primaries and independently usable subclaims."""
    try:
        tasks, authority, refs = authorities(root, raw / "authority")
        failures = authority["gate_check_summary"]
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        tasks = json.loads((ROOT / INPUT).read_bytes())["task_contract"]
        failures = [failure(root / INPUT, "V703_authority", "authority", True, str(error))]
        authority, refs = dict(activated=False), [reference(root / INPUT)]
    preconditions = []
    for path in [
        *(ROOT / p for p in NAMED),
        *(ROOT / ".venv/bin" / p for p in ("python", "pytest", "coverage", "ruff", "mypy")),
    ]:
        gate = failure(path, "prerequisites", "resource_exists", True, path.is_file())
        gate["passed"] = path.is_file()
        preconditions.append(gate)
        refs.append(reference(path))
        if not gate["passed"]:
            failures.append(gate)
    rows, primaries, audits = [], {}, {}
    for index, task in enumerate(tasks[:-1]):
        progress("before_branch", index, 12 - index)
        path = root / task["deliverable"]
        alternate = sorted((root / "results").glob(f"experiment_{8123 + index}_*.json"))
        selected = path if path.is_file() or not alternate else alternate[0]
        refs.extend([reference(path), reference(selected)])
        issues = []
        value = json.loads(selected.read_bytes()) if selected.is_file() else {}
        present = path.is_file()
        if not present:
            issues.append(failure(path, task["id"], "primary_exists", True, False))
        conductor = value.get("schema") == "blocked_gate_check_v1"
        if conductor:
            issues.append(
                dict(
                    check=value["failed_field"],
                    upstream=value["failed_upstream"],
                    path=value["failed_evidence_path"],
                    hash=value["failed_evidence_sha256"],
                    artifact_field=value["failed_field"],
                    op=value["failed_operator"],
                    expected=value["failed_expected"],
                    observed=value["failed_observed"],
                    passed=False,
                )
            )
        elif value:
            for field, expected in [
                ("task_id", task["id"]),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                if value.get(field) != expected:
                    issues.append(failure(selected, task["id"], field, expected, value.get(field)))
            try:
                terminal = Path(value["terminal_validation_sidecar_path"])
                side = json.loads(terminal.read_bytes())
                pub = side.get("publication", side)
                report = read_bound_sidecar(selected, Path(pub["sidecar_path"]))
                refs.extend([reference(terminal), reference(Path(pub["sidecar_path"]))])
                if report["report"]["passed"] is not True:
                    raise ValueError("terminal_report_failed")
            except (ValueError, KeyError, OSError, TypeError) as error:
                issues.append(
                    failure(selected, task["id"], "terminal_validation", True, str(error))
                )
        for gate in task["gated_on"]:
            upstream_task = next(t for t in tasks if t["id"] == gate["upstream"])
            upstream = primaries.get(gate["upstream"], {})
            observed = upstream.get(gate["artifact_field"])
            if observed != gate["value"]:
                issues.append(
                    failure(
                        root / upstream_task["deliverable"],
                        gate["upstream"],
                        gate["artifact_field"],
                        gate["value"],
                        observed,
                    )
                )
        issues.extend(g for g in value.get("gate_check_summary", []) if not g.get("passed"))
        audits[task["id"]] = dict(available=False, reason="missing_or_disqualified")
        owned_ok = (
            value.get("required_checks_passed") is True
            and value.get("flagged_adversarial") is False
        )
        if owned_ok:
            try:
                audits[task["id"]] = primitive_audit(value, 8123 + index)
                if audits[task["id"]].get("available"):
                    refs.append(audits[task["id"]]["primitive_reference"])
            except (ValueError, KeyError, OSError, TypeError) as error:
                issues.append(
                    failure(selected, task["id"], "primitive_reduction", True, str(error))
                )
        qualified = (
            present
            and owned_ok
            and not any(
                g["artifact_field"] in {"task_id", "terminal_validation", "primitive_reduction"}
                for g in issues
            )
        )
        state = value.get("verdict_class", "blocked")
        eligible = bool(qualified and state not in {"blocked", "disqualified", "partial"})
        kind = (
            "alternate_conductor_block"
            if conductor
            else "missing_primary"
            if not present
            else "disqualified_measurement"
            if not owned_ok or state == "disqualified"
            else "external_block"
            if not eligible
            else "legitimate_null"
            if state == "null"
            else "legitimate_result"
        )
        issues = deduplicate(issues)
        verdict = value.get("honest_verdict", "complete_blocked_primary_exists")
        if conductor:
            verdict, state = "complete_blocked_" + value["failed_field"], "blocked"
        if not qualified and present and state not in {"blocked", "disqualified"}:
            verdict, state = "complete_blocked_upstream_qualification", "blocked"
        row = dict(
            task_id=task["id"],
            unit_id=task["id"],
            source_cluster_id=task["id"],
            arm="task_disposition",
            condition="terminal_accounting",
            metric="qualified_branch",
            numerator=int(eligible),
            denominator=1,
            raw_numerator=int(eligible),
            raw_denominator=1,
            status="completed",
            completed=True,
            failed=state == "disqualified",
            censored=False,
            primary_present=present,
            qualified=bool(qualified),
            eligible=eligible,
            excluded=not eligible,
            honest_verdict=verdict,
            verdict_class=state,
            disposition=kind,
            exclusion_reason=None if eligible else kind,
            gate_check_summary=issues,
            path=str(selected),
            sha256=reference(selected)["sha256"],
        )
        rows.append(row)
        primaries[task["id"]] = {
            k: v
            for k, v in value.items()
            if k.endswith("_score")
            or k
            in {
                "decision_rows",
                "retention_rows",
                "board_rows",
                "new_outcome_count",
                "current_frontier",
                "event_frontier",
                "reader_conformance_rows",
                "required_checks_passed",
                "flagged_adversarial",
                "trained_head_specs",
                "MODEL_SPECS",
                "model_invocation_counts",
                "nfr01_met",
                "claim_scope",
                "exposure_scope",
                "verifier_is_oracle",
                "retirement_decisions",
                "retirement_rows",
                "cited_upstream_artifacts",
            }
        }
        failures.extend(issues)
        progress("after_branch", index + 1, 11 - index)
    priors = {}
    for task in tasks:
        for prior in task["prior_failures"]:
            number = prior["experiment_id"].split("-")[0][3:]
            paths = sorted((root / "results").glob(f"experiment_{number}_*.json"))
            path = paths[0] if paths else root / "results" / f"missing_{number}.json"
            old = json.loads(path.read_bytes()) if path.is_file() else {}
            priors[prior["experiment_id"]] = dict(
                reference(path), honest_verdict=old.get("honest_verdict")
            )
            priors[prior["experiment_id"]]["retirement_history"] = old.get(
                "retirement_decisions",
                old.get("retirement_candidates", old.get("retirement_rows", [])),
            )
            refs.append(reference(path))
    preserved = []
    unique_refs = list({r["path"]: r for r in refs}.values())
    for index, ref in enumerate(unique_refs):
        path = Path(ref["path"])
        saved = dict(ref)
        if path.is_file():
            target = raw / "custody" / (ref["sha256"].split(":")[-1] + path.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(path, target)
            saved.update(source_path=str(path), path=str(target))
        preserved.append(saved)
        progress("custody", index + 1, len(unique_refs) - index - 1)
    return dict(
        tasks=tasks,
        authority=authority,
        dispositions=rows,
        primaries=primaries,
        references=preserved,
        failures=deduplicate(failures),
        preconditions=preconditions,
        independent_reductions=audits,
        prior_evidence=priors,
    )
