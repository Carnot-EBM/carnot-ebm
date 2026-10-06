"""REQ-REPORT-8191: preserve input bytes before independent terminal accounting.

A scheduled skip cannot supply a scientific result. Each branch retains its own
qualification, so failed learning does not erase qualified source decisions.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from carnot.reporting import v707_contract_custody as custody
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import tasks_digest
from carnot.reporting.v707_capstone_science import primitive, reduce
from carnot.reporting.v703_capstone_inputs import deduplicate, failure, reference

Json = dict[str, Any]
ROOT = custody.ROOT
INPUT = "results/experiment_8178_v707_contract_custody.json"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts because accounting performs no model or benchmark work."""
    print(f"[exp8191] phase={phase} completed={completed} pending={pending}", flush=True)


def load(root: Path, raw: Path) -> Json:
    """Bind immutable activation and each actual producer or conductor disposition."""
    receipt = json.loads((ROOT / INPUT).read_bytes())
    tasks = receipt["task_contract"]
    refs: list[Json] = []
    issues: list[Json] = []
    authority: Json = dict(activated=False)
    historical: Json = {}
    try:
        receipt = json.loads((root / INPUT).read_bytes())
        historical = {r["task_id"]: r for r in receipt.get("historical_dispositions", [])}
        snaps = receipt["authority_snapshots"]
        active, design = [
            checked(dict(path=snaps[k]["snapshot_path"], sha256=snaps[k]["sha256"]))
            for k in ("active", "design")
        ]
        authority = custody.assess(
            design, root / "research-roadmap-next.yaml", active, raw / "authority"
        )
        table, tasks = parse_design(design.read_text(), milestone="2026.10.707")
        authority.update(visible_table=table, embedded_tasks=tasks)
        if tasks_digest(tasks) != receipt["canonical_tasks_sha256"]:
            raise ValueError("receipt_tasks_digest_drift")
        refs.extend([reference(active), reference(design)])
        issues.extend(authority["gate_check_summary"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        authority["activated"] = False
        issues.append(failure(root / INPUT, "V707_authority", "authority", True, str(error)))
    manifest = root / "ops/exclusion_manifest.yaml"
    retired = yaml.safe_load(manifest.read_bytes()) if manifest.is_file() else {}
    retired_ids = {
        str(r.get("experiment_id"))
        for k in ("retired", "retired_experiments")
        for r in retired.get(k, [])
    }
    retired_ids.update(
        str(i).split("-")[0].removeprefix("exp")
        for r in retired.get("retired", [])
        for i in r.get("experiment_ids", [])
    )
    log_path = root / "ops/conductor-log.md"
    log = log_path.read_text() if log_path.is_file() else ""
    refs.extend([reference(root / INPUT), reference(manifest), reference(log_path)])
    primaries: Json = {}
    audits: Json = {}
    rows = []
    for index, task in enumerate(tasks[:-1]):
        progress("before_task", index, 13 - index)
        number = 8178 + index
        path = root / task["deliverable"]
        alternatives = sorted((root / "results").glob(f"experiment_{number}_*.json"))
        selected = path
        for alternative in alternatives:
            if (
                alternative != path
                and json.loads(alternative.read_bytes()).get("schema") == "blocked_gate_check_v1"
            ):
                selected = alternative
                break
        value = json.loads(selected.read_bytes()) if selected.is_file() else {}
        conductor = value.get("schema") == "blocked_gate_check_v1"
        gates = []
        qualified = bool(value and not conductor)
        operands = dict(
            primary_exists=path.is_file(),
            required_checks_passed=value.get("required_checks_passed"),
            flagged_adversarial=value.get("flagged_adversarial"),
            task_id=value.get("task_id"),
            not_retired=str(number) not in retired_ids,
        )
        for field, expected in dict(
            primary_exists=True,
            required_checks_passed=True,
            flagged_adversarial=False,
            task_id=task["id"],
            not_retired=True,
        ).items():
            if operands[field] != expected:
                qualified = False
                gates.append(failure(selected, task["id"], field, expected, operands[field]))
        if value and not conductor:
            try:
                terminal = Path(value["terminal_validation_sidecar_path"])
                pub = json.loads(terminal.read_bytes())
                pub = pub.get("publication", pub)
                report = read_bound_sidecar(selected, Path(pub["sidecar_path"]))
                if report["report"]["passed"] is not True:
                    raise ValueError("terminal_report_failed")
                refs.extend([reference(terminal), reference(Path(pub["sidecar_path"]))])
            except (OSError, ValueError, KeyError, TypeError) as error:
                qualified = False
                gates.append(failure(selected, task["id"], "terminal_validation", True, str(error)))
        for gate in task["gated_on"]:
            parent = next(t for t in tasks if t["id"] == gate["upstream"])
            observed = primaries.get(parent["id"], {}).get(gate["artifact_field"], "<missing>")
            if observed != gate["value"]:
                gates.append(
                    failure(
                        root / parent["deliverable"],
                        parent["id"],
                        gate["artifact_field"],
                        gate["value"],
                        observed,
                    )
                )
        summary = value.get("gate_check_summary", [])
        if isinstance(summary, list):
            gates.extend(g for g in summary if isinstance(g, dict) and not g.get("passed"))
        if conductor:
            gates.append(
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
        audit = dict(available=False)
        if qualified and value.get("verdict_class") not in ("disqualified", "partial"):
            try:
                audit = primitive(value, number)
                if audit["available"]:
                    refs.extend(audit["references"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                qualified = False
                gates.append(failure(selected, task["id"], "primitive_reduction", True, str(error)))
        state = value.get("verdict_class", "blocked")
        if gates and state != "disqualified":
            state = "blocked"
        eligible = qualified and not gates and state not in ("blocked", "disqualified", "partial")
        lines = [s for s in log.splitlines() if task["title"][:48] in s]
        statuses = [s.split("|")[3].strip() for s in lines]
        skipped = conductor or "GATE_BLOCK" in statuses and not value
        verdict = value.get(
            "honest_verdict",
            "complete_blocked_" + str(gates[0]["check"] if gates else "primary_exists"),
        )
        if conductor:
            verdict = "complete_blocked_" + value["failed_field"]
        rows.append(
            dict(
                task_id=task["id"],
                unit_id=task["id"],
                source_cluster_id=task["id"],
                arm="task_disposition",
                condition="terminal_accounting",
                metric="eligible_branch",
                numerator=int(eligible),
                denominator=1,
                raw_numerator=int(eligible),
                raw_denominator=1,
                status="completed",
                completed=True,
                failed=state == "disqualified",
                censored=False,
                eligible=eligible,
                excluded=not eligible,
                qualified=qualified,
                primary_present=bool(value) and not conductor,
                primary_honest_verdict=value.get("honest_verdict", "<missing>")
                if not conductor
                else "<missing>",
                honest_verdict=verdict,
                verdict_class=state,
                disposition="conductor_skip"
                if skipped
                else "producer"
                if value
                else "missing_primary",
                conductor_log_rows=lines,
                conductor_statuses=statuses,
                exclusion_reason=None if eligible else state,
                gate_check_summary=deduplicate(gates),
                **reference(selected),
            )
        )
        primaries[task["id"]] = {
            k: v
            for k, v in value.items()
            if (
                number != 8178
                or k
                in (
                    "experiment_id",
                    "task_id",
                    "honest_verdict",
                    "verdict_class",
                    "MODEL_SPECS",
                    "model_invocation_counts",
                    "trained_head_specs",
                )
            )
            and k
            not in (
                "field_principles",
                "validation_receipts",
                "code_config_hashes",
                "source_artifact_hashes",
                "gate_check_summary",
            )
        }
        audits[task["id"]] = audit
        refs.append(reference(selected))
        for r in value.get("validation_receipts", []):
            if r.get("log_path") and Path(r["log_path"]).is_file():
                refs.append(reference(Path(r["log_path"])))
        issues.extend(gates)
        progress("after_task", index + 1, 12 - index)
    priors: Json = {}
    for task in tasks:
        for prior in task["prior_failures"]:
            key = prior["experiment_id"]
            number = key.split("-")[0][3:]
            ledger = next(
                r
                for r in receipt["prior_scope_ledger"]
                if r["prior_failure"]["experiment_id"] == key
            )
            original = Path(ledger["path"])
            path = root / "results" / original.name
            old = json.loads(path.read_bytes()) if path.is_file() else {}
            priors[key] = dict(
                reference(path),
                expected_sha256=ledger["sha256"],
                honest_verdict=old.get("honest_verdict", "<missing>")
                if old.get("schema") != "blocked_gate_check_v1"
                else "<missing>",
                retired=str(number) in retired_ids,
                conductor_statuses=historical.get(key, {}).get("conductor_statuses", []),
                conductor_log_rows=historical.get(key, {}).get("conductor_log_rows", []),
                earlier_logged_verdicts=historical.get(key, {}).get("earlier_logged_verdicts", []),
            )
            refs.append(reference(path))
    archived: Json = dict(available=False, scope="historical_only")
    preserved = []
    for ref in {r["path"]: r for r in refs}.values():
        path = Path(ref["path"])
        saved = dict(ref)
        if path.is_file():
            target = raw / "custody" / (ref["sha256"].split(":")[-1] + path.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            saved.update(source_path=str(path), path=str(target))
        preserved.append(saved)
    return dict(
        literature_mapping=receipt.get("literature_mapping", {})
        if authority.get("activated")
        else {},
        prior_scope_ledger=receipt.get("prior_scope_ledger", [])
        if authority.get("activated")
        else [],
        tasks=tasks,
        authority=authority,
        historical_hash_failures=receipt.get("historical_hash_failures", [])
        if authority.get("activated")
        else [],
        dispositions=rows,
        primaries=primaries,
        audits=audits,
        failures=deduplicate(issues),
        references=preserved,
        prior_evidence=priors,
        archived_learning=archived,
    )
