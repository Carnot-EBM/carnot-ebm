"""REQ-REPORT-8985: preserve failed science and authenticate terminal scheduler evidence."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v766_branch_evidence as previous_arc
from carnot.reporting import v767_capstone as capstone
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v755_branch_contract import reference

Json = dict[str, Any]
ROOT = capstone.ROOT
ARCHIVE = "openspec/change-proposals/research-roadmap-v767-preserved-20261110.md"
QUEUE = "openspec/change-proposals/research-roadmap-v767-queue-preserved-20261110.yaml"
CAPSTONE = "results/experiment_8984_v767_capstone.json"
PINS = {
    ARCHIVE: "sha256:f6b4cd77f27fc64054a2dcb345eb300a4e694838ec541f00b868fee3331a203d",
    QUEUE: "sha256:03d7d8ce109e2372cbaf55d799553fcf099874c6673bc1b8cc176d20351bfea7",
    CAPSTONE: "sha256:df2cfe785ec32fe95e0d29c2a49d4a5fad4d19c337dfaae3598574586d0bc5e9",
    "results/experiment_8971_v767_branch_contract.json": capstone.PINS[8971],
    "results/experiment_8972_v767_encoder_head_diagnosis.json": capstone.PINS[8972],
    "results/experiment_8973_v767_shared_feature_methods.json": capstone.PINS[8973],
    "results/experiment_8975_v767_natural_label_eligibility.json": capstone.PINS[8975],
    "results/experiment_8974_feature_online_methods.json": "sha256:3f6e717933d5534c02c24c2feac4a77be02e21d84b9c43adff33363c4817d426",
    "results/experiment_8976_graph_corpus_a.json": "sha256:280d98da49ee01991d8694cbcc961594f230a712f211269ac0a765bd902d5f3f",
    "results/experiment_8983_arc_transfer_evidence.json": "sha256:4a6286d70721fae54ed8b4d4cfa0c06ada3e77ab7bbd2525611c766f66adbe99",
    "research-complete.yaml": "sha256:b9c44a0393314c55dda7d07f845005bf963247df398ec846febd64abfeb8a233",
    "ops/conductor-log.md": "sha256:2c6766a98693f30eeb8dc7f34fd10cd5eb9d36bb2767961283d2a076d5af5fc7",
}


def history(root: Path) -> Json:
    """Read pinned original bytes; missing historical science never becomes a new result."""
    refs, rows, failures, values = [], [], [], {}
    for name, pin in PINS.items():
        path = root / name
        ref = reference(path)
        if ref["sha256"] != pin:
            failures.append(
                dict(
                    ref,
                    upstream="v767_original_evidence",
                    check="original_bytes",
                    artifact_field="sha256",
                    op="==",
                    expected=pin,
                    observed=ref["sha256"] or "missing_file",
                    passed=False,
                )
            )
        else:
            value = json.loads(path.read_bytes()) if path.suffix == ".json" else {}
            refs.append(
                dict(
                    ref,
                    experiment_id=value.get("experiment_id", value.get("experiment")),
                    fields_imported=list(value) or ["original_authority_bytes"],
                )
            )
            values[name] = value
    if not failures:
        try:
            table, tasks = parse_design((root / ARCHIVE).read_text(), milestone="2026.11.767")
            queue = yaml.load((root / QUEUE).read_bytes(), Loader=yaml.CSafeLoader)
            original = values[CAPSTONE]
            branch = values["results/experiment_8971_v767_branch_contract.json"]
            if tasks != branch["full_task_objects"] or len(table) != 14:
                raise ValueError("preserved_design_objects")
            if queue["milestone"] != "2026.11.767" or queue["tasks"] != tasks:
                raise ValueError("preserved_active_queue")
            log = (root / "ops/conductor-log.md").read_text()
            archive = yaml.load(
                (root / "research-complete.yaml").read_bytes(), Loader=yaml.CSafeLoader
            )
            scheduled = next(m for m in archive["milestones"] if m["id"] == "2026.11.767")["tasks"]
            for task, terminal, recorded in zip(
                tasks, scheduled, original["slot_rows"], strict=True
            ):
                name = next(
                    (n for n in PINS if n.startswith(f"results/experiment_{task['id'][3:7]}_")),
                    None,
                )
                value = values.get(name, {})
                kind = (
                    "primary"
                    if name == task["deliverable"]
                    else "structured_skip"
                    if name
                    else "missing_primary"
                )
                if terminal["id"] != task["id"] or recorded["task_id"] != task["id"]:
                    raise ValueError("scheduler_identity")
                if kind == "primary":
                    validate_primary(value, root / task["deliverable"])
                else:
                    if not recorded["conductor_rows"] or not all(
                        line in log for line in recorded["conductor_rows"]
                    ):
                        raise ValueError("scheduler_skip_receipt")
                    if (
                        kind == "structured_skip"
                        and value["gate_check_summary"] != recorded["conductor_gate_check_summary"]
                    ):
                        raise ValueError("scheduler_gate_operand")
                rows.append(
                    dict(
                        recorded,
                        kind=kind,
                        primary_reference=reference(root / task["deliverable"]),
                        conductor_reference=reference(root / name)
                        if name
                        else reference(root / "ops/conductor-log.md"),
                        scheduler_outcome=terminal["result"],
                        scientific_results_admitted=False,
                    )
                )
            for ref in [original["work_reference"]]:
                if sha256_file(Path(ref["path"])) != ref["sha256"]:
                    raise ValueError("historical_work_hash")
                refs.append(
                    dict(ref, experiment_id=8984, fields_imported=["retained_primitive_work"])
                )
            work = json.loads(Path(original["work_reference"]["path"]).read_bytes())
            old_archive = next(
                r
                for r in work["references"]
                if r.get("original_path", "").endswith("/research-complete.yaml")
            )
            if sha256_file(Path(old_archive["path"])) != old_archive["sha256"]:
                raise ValueError("historical_archive_hash")
            refs.append(
                dict(
                    old_archive,
                    experiment_id=None,
                    fields_imported=["archive_lag_at_v767_capstone"],
                )
            )
            archived = yaml.load(Path(old_archive["path"]).read_bytes(), Loader=yaml.CSafeLoader)
            archive_lag = not any(m["id"] == "2026.11.767" for m in archived["milestones"])
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            failures.append(
                dict(
                    reference(root / CAPSTONE),
                    upstream="v767_original_evidence",
                    check="history_schema",
                    artifact_field="terminal_history",
                    op="==",
                    expected="authenticated originals",
                    observed=str(error),
                    passed=False,
                )
            )
    return dict(
        authenticated=not failures,
        rows=rows,
        references=refs,
        failures=failures,
        quarantine={},
        original_child_qualified=not failures,
        capstone_verdict=values.get(CAPSTONE, {}).get("honest_verdict"),
        archive_lag_at_v767_capstone=archive_lag if not failures else None,
    )


def arc_evidence(ledger_path: Path | None = None) -> Json:
    """Read the ledger once and exclude outcomes already held by the V767 frontier."""
    branch = ROOT / "results/experiment_8971_v767_branch_contract.json"
    producer = json.loads(branch.read_bytes())
    frontier = producer["arc_frontier"]["frontier_after"]
    snapshot = next(
        s
        for s in producer["raw_shard_hashes"]
        if s.get("source", {}).get("path")
        == str(ROOT.joinpath(*previous_arc.arc.DEFAULT_LEDGER_PARTS))
    )
    held = previous_arc.arc.read(snapshot)
    delta = previous_arc.arc.delta

    def newer(baseline: Json, ledger: Json, closed: Json) -> tuple[list[Json], Json]:
        """Retained Exp8720 and V767 identities must both survive in the new ledger."""
        if sorted(held["entries"]) != frontier["ledger_receipt_identities"] or any(
            held["entries"].get(k) != v for k, v in baseline["entries"].items()
        ):
            raise ValueError("v767_frontier_identity_mismatch")
        return delta(held, ledger, closed)

    with patch.object(previous_arc.arc, "delta", newer):
        value = previous_arc.arc_evidence(ledger_path=ledger_path)
    value["v767_frontier"] = frontier
    value["v767_frontier_reference"] = reference(branch)
    return value


arc = SimpleNamespace(arc=previous_arc.arc, arc_evidence=arc_evidence)
