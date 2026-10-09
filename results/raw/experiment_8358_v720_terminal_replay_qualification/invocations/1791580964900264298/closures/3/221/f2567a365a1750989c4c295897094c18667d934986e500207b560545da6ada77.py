"""REQ-VERIFY-8316: advance a physical evidence frontier without a device retry.

The earlier ledger already qualified the parser and historical custody. Small
adapters retain those checks while making the new frontier explicit.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import gatemate_change_ledger_8246 as prior
from carnot.reporting import gatemate_delta_execution_8302 as upstream_cli
from carnot.reporting import gatemate_physical_delta_8302 as predecessor
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.request_trace_inventory_8200 import operand
from scripts.experiments import experiment_7146_v627_gatemate_changed_state as legacy

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8316_v717_gatemate_obligation"
CLI = "scripts/experiments/" + NAME + ".py"
UPSTREAM = "results/experiment_8302_v716_gatemate_physical_delta.json"
PIN = "sha256:fb847291bb18a922ed944e08f38b98d486a6dbc1dce29cf322155a7f61ee6f12"
BOUND_PIN = "sha256:a8eab8ed696f0bb378cef5f9fcee8a4bcbda2b16e5a594e1d8e0b8f5a083520d"
TERMINAL_PIN = "sha256:29e45246632bc896096ee949a142a5db72b093cb5497ca6db2c797ddf3511ab9"
REOPEN = "docs/research-notes/v717-gatemate-obligation.md"
CONFIG: Json = dict(seed=7178316, run_date="20261008", previous_primary_sha256=PIN)
DOCS = [*predecessor.DOCS, REOPEN]
previous_cli = prior.previous_cli
freeze = prior.freeze
empty_data = prior.empty_data
frontier = predecessor.frontier
_select = prior.select_change
_upstream_replay = upstream_cli.replay


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts keep a documentation audit visible to the supervisor."""
    print(f"[exp8316] phase={phase} completed={completed} pending={pending}", flush=True)


def select_change(data: Json) -> tuple[list[Json], Json]:
    """The qualified timestamp checks use the advanced boundary, never calendar age."""
    from copy import deepcopy

    restricted = deepcopy(data)
    for row in restricted["candidate_rows"]:
        fields = row["raw_receipt"].get("changed_physical_fields", [])
        if not set(fields) & {
            "cable",
            "usb_jtag_cable_state",
            "port",
            "host_path",
            "power",
            "board",
            "board_presence",
        }:
            row.update(valid=False, reject_reason="no_cable_port_power_or_board_change")
    rows, change = _select(restricted)
    for row in rows:
        if row["reject_reason"]:
            row["reject_reason"] = row["reject_reason"].replace("exp8232", "exp8302")
    if change["exists"]:
        return rows, change
    return rows, dict(exists=False, reason="no authenticated physical change since Exp8302")


@contextmanager
def adapters() -> Iterator[None]:
    """Temporary bindings reuse qualified functions without changing historical code."""
    with (
        patch.object(prior, "PIN", PIN),
        patch.object(prior, "DOCS", DOCS),
        patch.object(prior, "CONFIG", CONFIG),
        patch.object(prior, "REOPEN", REOPEN),
        patch.object(prior, "frontier", frontier),
        patch.object(prior, "select_change", select_change),
    ):
        yield


def load(root: Path, raw: Path) -> Json:
    """Authenticate the actual predecessor before comparing any operator documents."""
    data = empty_data()
    path = root / UPSTREAM
    progress("authenticate_before", 0, 1)
    try:
        ref = freeze(path, raw, data)
        value = json.loads(checked(ref).read_bytes())
        for field, expected in [
            ("sha256", PIN),
            ("experiment_id", 8302),
            ("schema", "carnot.gatemate_physical_delta.v716.v1"),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
            ("fixture_mode", False),
        ]:
            observed = ref["sha256"] if field == "sha256" else value.get(field)
            data["checks"].append(operand(field, path, expected, observed))
        if not all(c["passed"] for c in data["checks"]):
            raise ValueError("exp8302_authentication")
        side = path.parent / "raw" / path.stem / "validators" / (PIN.split(":")[1] + ".json")
        bound = read_bound_sidecar(path, side)
        bound_ref = freeze(side, raw, data)
        data["checks"].append(operand("bound_sidecar.sha256", side, BOUND_PIN, bound_ref["sha256"]))
        data["checks"].append(operand("report.passed", side, True, bound["report"]["passed"]))
        terminal = freeze(Path(value["terminal_validation_sidecar_path"]), raw, data)
        data["checks"].append(
            operand(
                "terminal_sidecar.sha256", Path(terminal["path"]), TERMINAL_PIN, terminal["sha256"]
            )
        )
        terminal_value = json.loads(checked(terminal).read_bytes())
        for field, expected, observed in [
            ("publication.primary_sha256", PIN, terminal_value["publication"]["primary_sha256"]),
            (
                "publication.primary_path",
                str(path.absolute()),
                terminal_value["publication"]["primary_path"],
            ),
            (
                "private_candidate_validation.passed",
                True,
                terminal_value["private_candidate_validation"]["passed"],
            ),
        ]:
            data["checks"].append(operand(field, Path(terminal["path"]), expected, observed))
        if not all(c["passed"] for c in data["checks"]):
            raise ValueError("terminal_sidecar_authentication")
        _upstream_replay(path)
        if value["gatemate_obligation"]["board"] != "GateMate":
            raise ValueError("gatemate_row_identity")
        freeze(checked(value["replay_input_reference"]), raw, data)
        for item in value["source_artifact_hashes"]:
            freeze(checked(item), raw, data)
        data.update(
            board=value["gatemate_obligation"],
            terminal_reference=terminal,
            bound_reference=bound_ref,
            history_ready=all(c["passed"] for c in data["checks"]),
            previous_reference=ref,
            source_root=str(root),
            frontier=frontier(value, root, time.time_ns()),
        )
        data["cited"].append(
            dict(
                experiment_id=8302,
                path=str(path),
                sha256=ref["sha256"],
                fields_imported=[
                    "gatemate_obligation",
                    "invocation",
                    "physical_change_frontier",
                    "replay_input_reference",
                    "source_artifact_hashes",
                ],
            )
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        data["checks"].append(
            operand("gatemate_history", path, "authenticated Exp8302", str(error))
        )
    progress("authenticate_after", 1, 0)
    receipts = set(map(str, legacy.RECEIPT_PATHS))
    documents = list(dict.fromkeys([*DOCS, *sorted(receipts)]))
    for i, relative in enumerate(documents):
        document = root / relative
        required = relative in DOCS or relative in data["frontier"].get("documents", {})
        if required:
            data["checks"].append(operand("named_input.exists", document, True, document.is_file()))
        snapshot = freeze(document, raw, data) if document.is_file() else None
        old = data["frontier"].get("documents", {}).get(relative)
        digest = snapshot["sha256"] if snapshot else None
        data["document_rows"].append(
            dict(
                relative_path=relative,
                receipt_source=relative in receipts,
                required=required,
                previous_sha256=old,
                current_sha256=digest,
                snapshot=snapshot,
                disposition="missing"
                if snapshot is None
                else "unchanged"
                if digest == old
                else "changed",
            )
        )
        progress("document_frontier", i + 1, len(documents) - i - 1)
    if data["history_ready"]:
        with adapters():
            data["candidate_rows"] = prior.scan(data["document_rows"])
            data["receipt_rows"], data["physical_change"] = select_change(data)
    data["checks"].append(
        operand("physical_change_evidence.exists", path, True, data["physical_change"]["exists"])
    )
    return data


def verify_primitives(data: Json) -> None:
    """Rebuild the frontier and parser rows so rehashing cannot invent physical progress."""
    with adapters():
        prior.verify_primitives(data)
    if data["history_ready"] and not data["fixture"]:
        for key, pin in [("terminal_reference", TERMINAL_PIN), ("bound_reference", BOUND_PIN)]:
            checked(data[key])
            if data[key]["sha256"] != pin:
                raise ValueError("historical_sidecar_drift:" + key)


def reduce(data: Json) -> Json:
    """Carry a falsifiable board obligation while keeping its execution gates unpassed."""
    with adapters():
        value = prior.reduce(data)
        value["physical_change_rows"] = value["physical_change_receipt_rows"]
        value["original_transcript_sha256"] = data["board"].get("source_transcript_sha256")
        value["historical_model_provenance"] = []
        value["sample_size_budget"] = dict(
            intended_obligations=1, independent_samples=0, measured_probes=0
        )
        return value
