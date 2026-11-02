"""REQ-VERIFY-8178: freeze history without turning an old failure into new work.

Saved producer bytes establish the old verdict. Today's copy only establishes
what is available now; it cannot change the provenance of the original run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import v706_contract_context as context
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.v685_authority_lifecycle import tasks_digest
from carnot.reporting.v698_fixture_consumer_contract import failure

Json = dict[str, Any]
SKIPS = {8168, 8169, 8170}
PRESERVED = "openspec/change-proposals/research-roadmap-v706-preserved-20261005.md"
TASK = "exp8178-contract-custody"


def capture(binder: Any, path: Path) -> Json | None:
    """Keep available bytes even if another old reference expects a different hash."""
    try:
        return dict(binder.bind(path))
    except (OSError, ValueError):
        return None


def log_refs(value: Any) -> list[Json]:
    """Find every completed log so a first bad log cannot hide later receipts."""
    rows: list[Json] = []
    if isinstance(value, dict):
        if value.get("log_path") and value.get("log_sha256"):
            rows.append(dict(path=value["log_path"], sha256=value["log_sha256"]))
        for child in value.values():
            rows.extend(log_refs(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(log_refs(child))
    return rows


def historical(root: Path, binder: Any) -> Json:
    """Preserve exact task paths and original failures, including absent science.

    Read the old activation copy, never today's vNEXT headings. Continue past
    unavailable producer evidence so the fourteen administrative outcomes survive.
    """
    result: Json = dict(
        historical_dispositions=[],
        historical_hash_failures=[],
        authorities={},
        older_scope_dispositions=[],
        qualified_inputs={},
        qualified={},
        historical_inputs_ready=False,
        diagnostic_8071={},
    )
    admin = root / "results/experiment_8164_v706_contract_custody.json"
    try:
        old = binder.read(admin)
        for role in ["active", "design"]:
            ref = old["authority_snapshots"][role]
            result["authorities"][role] = binder.bind(Path(ref["snapshot_path"]), ref["sha256"])
        active = yaml.safe_load(Path(result["authorities"]["active"]["snapshot_path"]).read_bytes())
        tasks = active["tasks"]
        binder.require(admin, "historical_milestone", "2026.10.706", active["milestone"])
        binder.require(
            admin, "historical_tasks_digest", old["canonical_tasks_sha256"], tasks_digest(tasks)
        )
        binder.require(
            admin,
            "historical_sequence",
            list(range(8164, 8178)),
            [int(t["id"].split("-")[0][3:]) for t in tasks],
        )
        result["authorities"]["preserved_design"] = binder.bind(root / PRESERVED)
        result["older_scope_dispositions"] = old.get("historical_dispositions", [])
        log = Path(binder.bind(root / "ops/conductor-log.md")["snapshot_path"]).read_text()
    except (OSError, ValueError, KeyError, TypeError) as error:
        binder.failures.append(
            failure(admin, "historical_authority_readable", True, str(error), TASK)
        )
        return result
    values: Json = {}
    for task in tasks:
        path = root / task["deliverable"]
        if int(task["id"].split("-")[0][3:]) in SKIPS:
            continue
        ref = capture(binder, path)
        if ref and int(task["id"].split("-")[0][3:]) not in SKIPS:
            values[task["id"]] = binder.read(path)
    saved = context.hash_refs(old)
    requests: list[Json] = []
    for index, task in enumerate(tasks):
        print(f"[exp8178] phase=historical completed={index} pending={14 - index}", flush=True)
        n = 8164 + index
        path = root / task["deliverable"]
        value = values.get(task["id"], {})
        ref = next((r for r in binder.refs if r["path"] == str(path)), None)
        lines = [line for line in log.splitlines() if task["title"][:48] in line]
        statuses = [line.split("|")[3 if len(line.split("|")) > 5 else 2].strip() for line in lines]
        try:
            binder.require(
                root / "ops/conductor-log.md",
                f"conductor_disposition_{n}",
                True,
                ("GATE_BLOCK" if n in SKIPS else "OK") in statuses,
            )
        except ValueError:
            pass
        side_ref = validator_ref = None
        if value:
            try:
                binder.require(path, "task_id", task["id"], value.get("task_id"))
                side_path = Path(value["terminal_validation_sidecar_path"])
                side_ref = binder.bind(side_path)
                side = binder.read(side_path)
                publication = side.get("publication", side)
                binder.require(
                    side_path, "primary_sha256", ref["sha256"], publication.get("primary_sha256")
                )
                validator_path = Path(publication["sidecar_path"])
                validator_ref = binder.bind(validator_path)
                report = binder.read(validator_path)
                requests.extend(log_refs(side) + log_refs(report))
            except (OSError, ValueError, KeyError, TypeError) as error:
                binder.failures.append(
                    failure(path, "historical_terminal_readable", True, str(error), task["id"])
                )
        gates = []
        for gate in task.get("gated_on", []):
            producer = next(t for t in tasks if t["id"] == gate["upstream"])
            row = failure(
                root / producer["deliverable"],
                gate["artifact_field"],
                gate["value"],
                values.get(gate["upstream"], {}).get(gate["artifact_field"]),
                gate["upstream"],
            )
            row["op"] = gate["op"]
            gates.append(row)
        result["historical_dispositions"].append(
            dict(
                task_id=task["id"],
                experiment_id=n,
                path=str(path),
                sha256=ref["sha256"] if ref else None,
                primary_present=bool(value),
                disposition="gate_skipped"
                if n in SKIPS
                else "primary"
                if value
                else "missing_primary",
                conductor_statuses=statuses,
                conductor_log_rows=lines,
                honest_verdict=value.get("honest_verdict"),
                verdict_class=value.get("verdict_class"),
                required_checks_passed=value.get("required_checks_passed"),
                flagged_adversarial=value.get("flagged_adversarial"),
                gate_check_summary=value.get("gate_check_summary", gates),
                failed_receipts=[
                    r for r in value.get("validation_receipts", []) if not r["passed"]
                ],
                validation_sidecar_snapshot=side_ref,
                validator_snapshot=validator_ref,
                historical_MODEL_SPECS=value.get("MODEL_SPECS", []),
                historical_model_invocation_counts=value.get("model_invocation_counts", {}),
                no_retry_unchanged_outcome=True,
            )
        )
        requests.extend(log_refs(value))
        requests.extend(dict(r, original_failure=False) for r in context.hash_refs(value))
        requests.extend(
            dict(r, sha256=r["expected"], original_failure=True)
            for r in value.get("gate_check_summary", [])
            if r.get("check") == "sha256"
        )
        saved.extend(context.hash_refs(value))
    unique: Json = {}
    for request in requests:
        key = str((request.get("path"), request["sha256"]))
        if key not in unique or request.get("original_failure"):
            unique[key] = request
    for index, request in enumerate(unique.values()):
        if index % 64 == 0:
            print(
                f"[exp8178] phase=historical_hashes completed={index} pending={len(unique) - index}",
                flush=True,
            )
        path = Path(request.get("path", request.get("snapshot_path", "")))
        path = path if path.is_absolute() else root / path
        captured = capture(binder, path)
        observed = captured["sha256"] if captured else None
        if observed != request["sha256"] or request.get("original_failure"):
            authentic = []
            for ref in saved:
                candidate = ref.get("snapshot_path")
                if candidate and ref["sha256"] == request["sha256"]:
                    old_copy = capture(binder, Path(candidate))
                    if (
                        old_copy
                        and old_copy["sha256"] == request["sha256"]
                        and old_copy not in authentic
                    ):
                        authentic.append(old_copy)
            row = dict(
                check="sha256",
                upstream=TASK,
                path=str(path),
                hash=observed,
                artifact_field="sha256",
                field="sha256",
                op="==",
                expected=request["sha256"],
                observed=observed,
                passed=False,
                mutable_original_path=str(path),
                original_observed=request.get("observed"),
                captured_version=captured,
                authentic_saved_snapshots=authentic,
                old_provenance_repaired=False,
            )
            result["historical_hash_failures"].append(row)
    return result


def literature(root: Path, binder: Any, contract: Json) -> Json:
    """Freeze the already written literature scan; this run makes no new paper claim."""
    path = root / "research-references.md"
    try:
        ref = binder.bind(path)
        text = Path(ref["snapshot_path"]).read_text()
        entry = (
            text.split("## 2026-10-05 — V707 planning scan:", 1)[1]
            .split("\n", 1)[1]
            .split("\n## ", 1)[0]
        )
        design = Path(contract["authority_snapshots"]["design"]["snapshot_path"]).read_text()
        return dict(
            source=ref,
            entry=entry,
            frozen_design=contract["authority_snapshots"]["design"],
            method_to_tasks=[
                dict(method=method, tasks=tasks)
                for method, tasks in [
                    ("bounded sentence transport and energy decisions", list(range(8179, 8186))),
                    ("calibration and delayed energy memory", [8180, 8186, 8187]),
                    ("exact-request reuse and hardware obligations", [8188, 8190]),
                    ("ARC supervisor outcomes", [8189]),
                ]
            ],
            frozen_protocol=design.split("## Phase 1", 1)[-1].split("## Exact task contract", 1)[0],
            independent_generalization_credit=0,
        )
    except (OSError, ValueError, KeyError, IndexError) as error:
        binder.failures.append(
            failure(path, "V707_reference_entry_readable", True, str(error), TASK)
        )
        return {}
