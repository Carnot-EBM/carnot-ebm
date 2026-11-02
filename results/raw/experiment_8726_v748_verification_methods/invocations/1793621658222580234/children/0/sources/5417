"""REQ-VERIFY-8192: keep exact historical outcomes without repairing provenance.

Old methods come from authenticated versioned bytes. A new snapshot can keep
those bytes available, but cannot qualify a producer that failed its own checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import v706_contract_context as context
from carnot.reporting.v685_authority_lifecycle import tasks_digest
from carnot.reporting.v698_fixture_consumer_contract import failure
from carnot.reporting.v707_contract_history import capture, log_refs

Json = dict[str, Any]
TASK = "exp8192-contract-custody"
ADMIN = "results/experiment_8178_v707_contract_custody.json"
PRESERVED = "openspec/change-proposals/research-roadmap-v707-preserved-20261006.md"
SKIPS = {8186, 8187}


def historical(root: Path, binder: Any) -> Json:
    """Keep all fourteen outcomes, even when individual old inputs are unavailable."""
    result: Json = dict(
        historical_dispositions=[],
        historical_hash_failures=[],
        authorities={},
        qualified={},
        historical_inputs_ready=False,
        diagnostic_8071={},
        older_scope_dispositions=[],
        prior_scope_ledger=[],
    )
    admin = root / ADMIN
    try:
        old = binder.read(admin)
        result["historical_hash_failures"] = old.get("historical_hash_failures", [])
        result["older_scope_dispositions"] = old.get("historical_dispositions", [])
        result["prior_scope_ledger"] = old.get("prior_scope_ledger", [])
        for role in ["active", "design"]:
            ref = old["authority_snapshots"][role]
            result["authorities"][role] = binder.bind(Path(ref["snapshot_path"]), ref["sha256"])
        active = yaml.safe_load(Path(result["authorities"]["active"]["snapshot_path"]).read_bytes())
        tasks = active["tasks"]
        binder.require(admin, "historical_milestone", "2026.10.707", active["milestone"])
        binder.require(
            admin, "historical_tasks_digest", old["canonical_tasks_sha256"], tasks_digest(tasks)
        )
        binder.require(
            admin,
            "historical_sequence",
            list(range(8178, 8192)),
            [int(t["id"].split("-")[0][3:]) for t in tasks],
        )
        result["authorities"]["preserved_design"] = binder.bind(root / PRESERVED)
        log = Path(binder.bind(root / "ops/conductor-log.md")["snapshot_path"]).read_text()
    except (OSError, ValueError, KeyError, TypeError) as error:
        binder.failures.append(
            failure(admin, "historical_authority_readable", True, str(error), TASK)
        )
        return result
    values: Json = {}
    requests = []
    for index, task in enumerate(tasks):
        print(f"[exp8192] phase=historical completed={index} pending={14 - index}", flush=True)
        n = 8178 + index
        path = root / task["deliverable"]
        ref = None if n in SKIPS else capture(binder, path)
        value = binder.read(path) if ref else {}
        values[task["id"]] = value
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
                requests.extend(log_refs(side) + log_refs(binder.read(validator_path)))
            except (OSError, ValueError, KeyError, TypeError) as error:
                binder.failures.append(
                    failure(path, "historical_terminal_readable", True, str(error), task["id"])
                )
        gates = []
        for gate in task.get("gated_on", []):
            producer = next(t for t in tasks if t["id"] == gate["upstream"])
            operand = failure(
                root / producer["deliverable"],
                gate["artifact_field"],
                gate["value"],
                values.get(gate["upstream"], {}).get(gate["artifact_field"]),
                gate["upstream"],
            )
            operand.update(op=gate["op"], artifact_field=gate["artifact_field"])
            gates.append(operand)
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
                gate_check_summary=value.get("gate_check_summary") or gates,
                failed_receipts=[
                    r for r in value.get("validation_receipts", []) if not r["passed"]
                ],
                validation_sidecar_snapshot=side_ref,
                validator_snapshot=validator_ref,
                historical_MODEL_SPECS=value.get("MODEL_SPECS", []),
                historical_trained_head_specs=value.get("trained_head_specs", []),
                historical_model_invocation_counts=value.get("model_invocation_counts", {}),
                exposure_scope="exposed development",
                no_retry_unchanged_outcome=True,
            )
        )
        requests.extend(log_refs(value))
    unique = {(r["path"], r["sha256"]): r for r in requests}
    for index, request in enumerate(unique.values()):
        print(
            f"[exp8192] phase=completed_log_custody completed={index} pending={len(unique) - index}",
            flush=True,
        )
        ref = capture(binder, Path(request["path"]))
        observed = ref["sha256"] if ref else None
        if observed != request["sha256"]:
            result["historical_hash_failures"].append(
                dict(
                    failure(Path(request["path"]), "sha256", request["sha256"], observed, TASK),
                    artifact_field="sha256",
                    hash=observed,
                    old_provenance_repaired=False,
                )
            )
    return result


def literature(root: Path, binder: Any, contract: Json) -> Json:
    """Freeze the existing scan and PRD threshold without making new paper claims."""
    path = root / "research-references.md"
    try:
        ref = binder.bind(path)
        text = Path(ref["snapshot_path"]).read_text()
        entry = (
            text.split("## 2026-10-06 — V708 planning scan", 1)[1]
            .split("\n", 1)[1]
            .split("\n## ", 1)[0]
        )
        design = Path(contract["authority_snapshots"]["design"]["snapshot_path"]).read_text()
        return dict(
            source=ref,
            entry=entry,
            frozen_design=contract["authority_snapshots"]["design"],
            frozen_protocol=design.split("## Phase 1", 1)[-1].split("## Exact task contract", 1)[0],
            method_to_tasks=[
                dict(method=m, tasks=t)
                for m, t in [
                    ("real child coverage and calibrated memory", [8193, 8198, 8199]),
                    ("energy-conformal selective decisions", [8194, 8195, 8196, 8197]),
                    ("ordinary request census and service boundary", [8200, 8201]),
                    ("ARC supervisor frontier", [8202]),
                    ("precision bounds and attached boards", [8203]),
                ]
            ],
            board_obligations=["KV260", "PolarFire", "GateMate"],
            nfr01_threshold=10,
            nfr01_measure="complete Rust/Python throughput ratio",
            all_old_cohorts="exposed development",
            independent_generalization_credit=0,
        )
    except (OSError, ValueError, KeyError, IndexError) as error:
        binder.failures.append(
            failure(path, "V708_reference_entry_readable", True, str(error), TASK)
        )
        return {}
