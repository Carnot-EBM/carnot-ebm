"""REQ-REPORT-8204: preserve real dispositions before drawing scientific conclusions.

Frozen activation survives roadmap movement. Historical failed checks remain
failures even when the current task list is administratively authentic.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import v708_contract_custody as custody
from carnot.reporting import v708_capstone_science as science
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.v698_fixture_consumer_contract import failure
from carnot.reporting.v703_capstone_inputs import deduplicate, reference
from carnot.reporting.v685_authority_lifecycle import tasks_digest

Json = dict[str, Any]
ROOT = custody.ROOT
INPUT = "results/experiment_8192_v708_contract_custody.json"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual completed counts keep quiet child waits visible without fake activity."""
    print(f"[exp8204] phase={phase} completed={completed} pending={pending}", flush=True)


def load(root: Path, raw: Path) -> Json:
    """Authenticate each branch separately so a learning block cannot erase H1."""
    binder = custody.Binder(raw / "custody", task="exp8204-capstone")
    fallback = json.loads((ROOT / INPUT).read_bytes())
    tasks = deepcopy(fallback["task_contract"])
    receipt: Json = {}
    authority: Json = dict(activated=False)
    try:
        receipt = binder.read(root / INPUT)
        snaps = receipt["authority_snapshots"]
        active, design = [
            checked(dict(path=snaps[k]["snapshot_path"], sha256=snaps[k]["sha256"]))
            for k in ("active", "design")
        ]
        binder.bind(active)
        binder.bind(design)
        authority = custody.assess(
            design, root / "research-roadmap-next.yaml", active, raw / "authority"
        )
        tasks = authority["tasks"]
        binder.failures.extend(authority["gate_check_summary"])
        binder.require(
            root / INPUT,
            "receipt_tasks_digest",
            receipt["canonical_tasks_sha256"],
            tasks_digest(tasks),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        authority["activated"] = False
        binder.failures.append(failure(root / INPUT, "V708_authority", True, str(error)))
    manifest = root / "ops/exclusion_manifest.yaml"
    retired = yaml.safe_load(manifest.read_bytes()) if manifest.is_file() else {}
    retired_ids = {
        str(r.get("experiment_id"))
        for key in ("retired", "retired_experiments")
        for r in retired.get(key, [])
    }
    retired_ids.update(
        str(i).split("-")[0].removeprefix("exp")
        for r in retired.get("retired", [])
        for i in r.get("experiment_ids", [])
    )
    log_path = root / "ops/conductor-log.md"
    log = log_path.read_text() if log_path.is_file() else ""
    for path in (
        manifest,
        log_path,
        root / "research-roadmap.yaml",
        root / "research-roadmap-next.yaml",
        root / "research-references.md",
    ):
        if path.is_file():
            binder.bind(path)
    primaries: Json = {}
    audits: Json = {}
    rows = []
    for index, task in enumerate(tasks[:-1]):
        progress("before_task", index, 12 - index)
        number, path = 8192 + index, root / task["deliverable"]
        skips = []
        for alternative in sorted((root / "results").glob(f"experiment_{number}_*.json")):
            observed = binder.read(alternative)
            if observed.get("schema") == "blocked_gate_check_v1":
                skips.append(dict(reference(alternative), artifact=observed))
        value = binder.read(path) if path.is_file() else {}
        if value.get("schema") == "blocked_gate_check_v1":
            value = {}
        before = len(binder.failures)
        operands = dict(
            primary_exists=bool(value),
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
            try:
                binder.require(path, field, expected, operands[field])
            except ValueError:
                pass
        if value:
            try:
                terminal = binder.terminal_evidence(path, value)
                binder.require(path, "terminal_validation", True, terminal["passed"])
                for ref in value.get("code_config_hashes", []):
                    method = Path(ref["path"])
                    if method.suffix == ".py":
                        binder.bind(method, ref["sha256"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                binder.failures.append(
                    failure(path, "terminal_validation", True, str(error), task["id"])
                )
        authenticated = bool(value) and not binder.failures[before:]
        for gate in task["gated_on"]:
            parent = next(t for t in tasks if t["id"] == gate["upstream"])
            operand = failure(
                root / parent["deliverable"],
                gate["artifact_field"],
                gate["value"],
                primaries.get(parent["id"], {}).get(gate["artifact_field"]),
                parent["id"],
            )
            operand["op"] = gate["op"]
            operand["passed"] = operand["observed"] == operand["expected"]
            if not operand["passed"]:
                binder.failures.append(operand)
        for skip in skips:
            v = skip["artifact"]
            binder.failures.append(
                dict(
                    check=v["failed_field"],
                    upstream=v["failed_upstream"],
                    path=v["failed_evidence_path"],
                    hash=v["failed_evidence_sha256"],
                    artifact_field=v["failed_field"],
                    op=v["failed_operator"],
                    expected=v["failed_expected"],
                    observed=v["failed_observed"],
                    passed=False,
                )
            )
        if value.get("verdict_class") == "blocked" and number != 8192:
            binder.failures.extend(
                g for g in value.get("gate_check_summary", []) if not g.get("passed")
            )
        gates = binder.failures[before:]
        qualified = (
            authenticated
            and (not gates or value.get("verdict_class") == "blocked")
            and value.get("verdict_class") not in ("partial", "disqualified")
        )
        audit = dict(available=False)
        if qualified:
            try:
                audit = science.primitive(value, number)
                for ref in audit.get("references", []):
                    binder.bind(Path(ref["path"]), ref["sha256"])
            except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
                qualified = False
                gates.append(failure(path, "primitive_reduction", True, str(error), task["id"]))
        state = value.get("verdict_class", "blocked")
        eligible = qualified and state not in ("blocked", "disqualified", "partial")
        lines = [line for line in log.splitlines() if task["title"][:48] in line]
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
                status="completed",
                completed=True,
                eligible=eligible,
                excluded=not eligible,
                failed=state == "disqualified",
                censored=False,
                qualified=qualified,
                primary_present=bool(value),
                primary_honest_verdict=value.get("honest_verdict"),
                honest_verdict=value.get("honest_verdict", "complete_blocked_primary_exists"),
                verdict_class=state,
                exclusion_reason=None if eligible else state,
                disposition="producer"
                if value
                else "conductor_skip"
                if skips
                else "missing_primary",
                conductor_skips=skips,
                conductor_log_rows=lines,
                gate_check_summary=deduplicate(gates),
                **reference(path),
            )
        )
        primaries[task["id"]] = {
            k: v
            for k, v in value.items()
            if k
            not in (
                "field_principles",
                "source_artifact_hashes",
                "code_config_hashes",
                "validation_receipts",
            )
        }
        if number == 8192:
            primaries[task["id"]] = {
                k: v
                for k, v in primaries[task["id"]].items()
                if k in ("experiment_id", "task_id", "honest_verdict", "verdict_class")
            }
        audits[task["id"]] = audit
        progress("after_task", index + 1, 11 - index)
    priors: Json = {}
    for task in tasks:
        for prior in task["prior_failures"]:
            key = prior["experiment_id"]
            ledger = next(
                (
                    r
                    for r in receipt.get("prior_scope_ledger", [])
                    if r["prior_failure"]["experiment_id"] == key
                ),
                {},
            )
            path = root / "results" / Path(ledger.get("path", "absent")).name
            old = binder.read(path) if path.is_file() else {}
            priors[key] = dict(
                reference(path),
                expected_sha256=ledger.get("sha256"),
                honest_verdict=old.get("honest_verdict"),
                documented_scope=prior["addressed_by"],
            )
    historical: Json = {}
    for number in (8174, 8188):
        paths = sorted((root / "results").glob(f"experiment_{number}_*.json"))
        if paths:
            try:
                old = binder.read(paths[0])
                binder.terminal_evidence(paths[0], old)
                if old["required_checks_passed"] and old["flagged_adversarial"] is False:
                    audit = science.primitive(old, number)
                    historical[str(number)] = audit
                    for ref in audit.get("references", []):
                        binder.bind(Path(ref["path"]), ref["sha256"])
            except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
                binder.failures.append(failure(paths[0], "historical_service", True, str(error)))
    return dict(
        tasks=tasks,
        authority=authority,
        primaries=primaries,
        audits=audits,
        dispositions=rows,
        failures=deduplicate(binder.failures),
        references=[dict(r, source_path=r["path"], path=r["snapshot_path"]) for r in binder.refs],
        prior_evidence=priors,
        historical_service=historical,
        historical_hash_failures=receipt.get("historical_hash_failures", []),
        literature_mapping=receipt.get("literature_mapping", {}),
        prior_scope_ledger=receipt.get("prior_scope_ledger", []),
        preconditions=binder.observations,
        source_comparisons=[
            dict(path=r["path"], sha256=r["sha256"], current_sha256=sha256_file(Path(r["path"])))
            for r in binder.refs
            if Path(r["path"]).is_file()
        ],
    )
