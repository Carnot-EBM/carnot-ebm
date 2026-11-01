"""REQ-VERIFY-8246: compare physical evidence with the last qualified frontier.

The previous receipt already proves historical custody. Reusing it avoids
turning an unchanged setup into another device experiment.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import gatemate_continuity_8232 as previous
from carnot.reporting import gatemate_execution_8232 as _previous_cli
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.request_trace_inventory_8200 import operand
from scripts.experiments import experiment_7146_v627_gatemate_changed_state as legacy

Json = dict[str, Any]
ROOT = previous.ROOT
NAME = "experiment_8246_v712_gatemate_change_ledger"
CLI = "scripts/experiments/" + NAME + ".py"
UPSTREAM = "results/experiment_8232_v711_gatemate_continuity.json"
PIN = "sha256:3e350368f93f951d871c2d4fa1bda4a4a4853e73a780d09d7af838300d5e052f"
REOPEN = "docs/research-notes/v712-gatemate-change-ledger.md"
CONFIG: Json = dict(seed=7128246, run_date="20261007", previous_primary_sha256=PIN)
DOCS = [*previous.DOCS, REOPEN]
previous_cli = _previous_cli
freeze = previous.freeze


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so the conductor can distinguish work from a stall."""
    print(f"[exp8246] phase={phase} completed={completed} pending={pending}", flush=True)


def empty_data() -> Json:
    """A missing source still produces one explicit, unavailable board obligation."""
    return dict(
        board={},
        history_ready=False,
        checks=[],
        references=[],
        cited=[],
        fixture=False,
        receipt_rows=[],
        candidate_rows=[],
        physical_change=dict(exists=False),
        frontier={},
        document_rows=[],
        previous_reference=None,
    )


def frontier(value: Json, root: Path, observed_wall_ns: int) -> Json:
    """Bind the same-day boundary to actual prior clocks, rather than calendar age."""
    invocation = value["invocation"]
    primitive = json.loads(checked(value["replay_input_reference"]).read_bytes())
    documents = {
        str(Path(r["original_path"]).relative_to(root)): r["sha256"]
        for r in value["source_artifact_hashes"]
        if r.get("original_path") and Path(r["original_path"]).is_relative_to(root)
    }
    return dict(
        cutoff_date=value["run_date"],
        cutoff_wall_ns=invocation["started_wall_ns"]
        + invocation["ended_monotonic_ns"]
        - invocation["started_monotonic_ns"],
        observed_wall_ns=observed_wall_ns,
        seen_receipt_hashes=[canonical_hash(r["raw_receipt"]) for r in primitive["receipt_rows"]],
        documents=documents,
    )


def scan(documents: list[Json]) -> list[Json]:
    """Use the qualified dry-run parser only for document bytes that changed."""
    selected = [r for r in documents if r["receipt_source"] and r["disposition"] == "changed"]
    if not selected:
        return []
    paths = {str(checked(r["snapshot"])): r["relative_path"] for r in selected}
    with patch.object(legacy, "RECEIPT_PATHS", tuple(Path(p) for p in paths)):
        rows, _ = legacy.audit_receipts(
            Path("/"),
            cutoff_date=previous.CONFIG["cutoff_date"],
            run_date=CONFIG["run_date"],
            dry_run=True,
        )
    for row in rows:
        row["source_path"] = paths[row["source_path"]]
    return list(rows)


def select_change(data: Json) -> tuple[list[Json], Json]:
    """A new document or a new date alone cannot prove a physical change."""
    boundary = data["frontier"]
    rows = []
    for candidate in data["candidate_rows"]:
        if canonical_hash(candidate["raw_receipt"]) in boundary["seen_receipt_hashes"]:
            continue
        row = deepcopy(candidate)
        reason = row["reject_reason"]
        if row["valid"]:
            date = row["receipt_date"]
            if date < boundary["cutoff_date"]:
                reason = "not_newer_than_exp8232"
            elif date == boundary["cutoff_date"]:
                try:
                    stamp = datetime.fromisoformat(row["raw_receipt"]["receipt_timestamp"])
                    wall_ns = int(stamp.timestamp() * 1e9)
                    if stamp.tzinfo is None or stamp.strftime("%Y%m%d") != date:
                        raise ValueError("ambiguous_timestamp")
                    if not boundary["cutoff_wall_ns"] < wall_ns <= boundary["observed_wall_ns"]:
                        raise ValueError("timestamp_outside_frontier")
                except (ValueError, KeyError, TypeError):
                    reason = "missing_or_invalid_post_exp8232_timestamp"
        row.update(valid=reason is None, reject_reason=reason)
        rows.append(row)
    valid = [r for r in rows if r["valid"]]
    change = dict(exists=False, reason="no authenticated physical change since Exp8232")
    if valid:
        change = dict(exists=True, **max(valid, key=lambda r: (r["receipt_date"], r["row_id"])))
    return rows, change


def load(root: Path, raw: Path) -> Json:
    """Authenticate the existing obligation rather than introduce another hardware reader."""
    data = empty_data()
    path = root / UPSTREAM
    progress("previous_receipt_before", 0, 1)
    try:
        ref = freeze(path, raw, data)
        value = json.loads(checked(ref).read_bytes())
        for field, expected in [
            ("sha256", PIN),
            ("experiment_id", 8232),
            ("schema", "carnot.gatemate_continuity.v711.v1"),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
            ("fixture_mode", False),
        ]:
            observed = ref["sha256"] if field == "sha256" else value.get(field)
            data["checks"].append(operand(field, path, expected, observed))
        if not all(c["passed"] for c in data["checks"]):
            raise ValueError("previous_receipt_authentication")
        side = path.parent / "raw" / path.stem / "validators" / (PIN.split(":")[1] + ".json")
        report = read_bound_sidecar(path, side)
        freeze(side, raw, data)
        data["checks"].append(operand("report.passed", side, True, report["report"]["passed"]))
        previous_cli.replay(path)
        if value["gatemate_obligation"]["board"] != "GateMate":
            raise ValueError("gatemate_row_identity")
        freeze(checked(value["replay_input_reference"]), raw, data)
        for item in value["source_artifact_hashes"]:
            freeze(checked(item), raw, data)
        data.update(
            board=value["gatemate_obligation"],
            history_ready=all(c["passed"] for c in data["checks"]),
            previous_reference=ref,
            source_root=str(root),
            frontier=frontier(value, root, time.time_ns()),
        )
        data["cited"].append(
            dict(
                experiment_id=8232,
                path=str(path),
                sha256=ref["sha256"],
                fields_imported=[
                    "gatemate_obligation",
                    "invocation",
                    "replay_input_reference",
                    "source_artifact_hashes",
                    "physical_change_receipt_rows",
                ],
            )
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        data["checks"].append(
            operand("gatemate_history", path, "authenticated Exp8232 frontier", str(error))
        )
    progress("previous_receipt_after", 1, 0)
    receipts = set(map(str, legacy.RECEIPT_PATHS))
    documents = list(dict.fromkeys([*DOCS, *sorted(receipts)]))
    for i, relative in enumerate(documents):
        document = root / relative
        required = relative in DOCS or relative in data["frontier"].get("documents", {})
        if required:
            data["checks"].append(operand("named_input.exists", document, True, document.is_file()))
        snapshot = freeze(document, raw, data) if document.is_file() else None
        old_hash = data["frontier"].get("documents", {}).get(relative)
        digest = snapshot["sha256"] if snapshot else None
        data["document_rows"].append(
            dict(
                relative_path=relative,
                receipt_source=relative in receipts,
                required=required,
                previous_sha256=old_hash,
                current_sha256=digest,
                snapshot=snapshot,
                disposition="missing"
                if snapshot is None
                else "unchanged"
                if digest == old_hash
                else "changed",
            )
        )
        progress("document_frontier", i + 1, len(documents) - i - 1)
    if data["history_ready"]:
        data["candidate_rows"] = scan(data["document_rows"])
        data["receipt_rows"], data["physical_change"] = select_change(data)
    data["checks"].append(
        operand("physical_change_evidence.exists", path, True, data["physical_change"]["exists"])
    )
    return data


def verify_primitives(data: Json) -> None:
    """Reconstruct receipts and the prior boundary so rehashed summaries cannot invent change."""
    if data["history_ready"]:
        if data["board"].get("blocked_idcode") != "0xffffffff":
            raise ValueError("board_drift")
        if not data["fixture"]:
            value = json.loads(checked(data["previous_reference"]).read_bytes())
            if data["previous_reference"]["sha256"] != PIN:
                raise ValueError("previous_pin_drift")
            expected = frontier(
                value, Path(data["source_root"]), data["frontier"]["observed_wall_ns"]
            )
            if expected != data["frontier"] or data["board"] != value["gatemate_obligation"]:
                raise ValueError("frontier_drift")
            required_paths = list(dict.fromkeys([*DOCS, *sorted(map(str, legacy.RECEIPT_PATHS))]))
            if [r["relative_path"] for r in data["document_rows"]] != required_paths:
                raise ValueError("document_roster_drift")
            for row in data["document_rows"]:
                previous_hash = expected["documents"].get(row["relative_path"])
                digest = sha256_file(checked(row["snapshot"])) if row["snapshot"] else None
                disposition = (
                    "missing"
                    if digest is None
                    else "unchanged"
                    if digest == previous_hash
                    else "changed"
                )
                if (previous_hash, digest, disposition) != (
                    row["previous_sha256"],
                    row["current_sha256"],
                    row["disposition"],
                ):
                    raise ValueError("document_metadata_drift")
            if scan(data["document_rows"]) != data["candidate_rows"]:
                raise ValueError("candidate_drift")
        for candidate in data["candidate_rows"]:
            fresh = legacy.receipt_row(
                candidate["raw_receipt"],
                source_path=candidate["source_path"],
                row_index=int(candidate["row_id"].split("-")[1]),
                cutoff_date=previous.CONFIG["cutoff_date"],
                run_date=CONFIG["run_date"],
                structured=candidate["structured_receipt"],
            )
            if fresh != candidate:
                raise ValueError("candidate_parser_drift")
        rows, change = select_change(data)
        if rows != data["receipt_rows"] or change != data["physical_change"]:
            raise ValueError("receipt_drift")


def reduce(data: Json) -> Json:
    """Retain the single board obligation; documentation grants no device or learning benefit."""
    value = previous.reduce(data)
    changed = bool(data["physical_change"].get("exists"))
    row = value["rows"][0]
    row.update(
        reopen_contract_path=REOPEN,
        numerator=1 if changed else None,
        missing_status=None if changed else "physical_receipt_absent",
        metric="authenticated_new_physical_receipt",
    )
    value.update(
        reopen_contract_path=REOPEN,
        previous_receipt=data["previous_reference"],
        physical_change_frontier=data["frontier"],
        change_ledger=data["document_rows"],
        future_probe_contract=dict(
            eligible=changed and bool(value["gatemate_obligation_ready_score"]),
            contract_path=REOPEN,
            board=legacy.EXPECTED_BOARD,
            receipt=data["physical_change"] if changed else None,
            detect_argv=list(legacy.DETECT_COMMAND),
            detect_deadline_s=30,
            expected_idcode=legacy.EXPECTED_IDCODE,
            n16_acceptance="authenticated existing tile/constraints/bitstream/CPU hashes; bounded flash then device hash-smoke parity and transfer/readout clocks",
            first_failure_stop=True,
            current_execution_authorized=False,
        ),
    )
    return value
