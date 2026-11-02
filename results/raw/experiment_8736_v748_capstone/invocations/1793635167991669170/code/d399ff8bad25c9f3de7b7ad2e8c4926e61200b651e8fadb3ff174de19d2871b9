"""REQ-VERIFY-8288: advance a physical evidence frontier without a device retry.

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
from carnot.reporting import gatemate_delta_execution_8274 as upstream_cli
from carnot.reporting import gatemate_physical_delta_8274 as predecessor
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.request_trace_inventory_8200 import operand
from scripts.experiments import experiment_7146_v627_gatemate_changed_state as legacy

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8288_v715_gatemate_physical_delta"
CLI = "scripts/experiments/" + NAME + ".py"
UPSTREAM = "results/experiment_8274_v714_gatemate_physical_delta.json"
PIN = "sha256:46e9174e6fe6949b1e88d991909a66e25477fa23225c44faa1c50e4bcfa0d241"
REOPEN = "docs/research-notes/v715-gatemate-physical-delta.md"
CONFIG: Json = dict(seed=7158288, run_date="20261008", previous_primary_sha256=PIN)
DOCS = [*predecessor.DOCS, REOPEN]
previous_cli = prior.previous_cli
freeze = prior.freeze
empty_data = prior.empty_data
frontier = predecessor.frontier
_select = prior.select_change
_upstream_replay = upstream_cli.replay


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts keep a documentation audit visible to the supervisor."""
    print(f"[exp8288] phase={phase} completed={completed} pending={pending}", flush=True)


def select_change(data: Json) -> tuple[list[Json], Json]:
    """The qualified timestamp checks use the advanced boundary, never calendar age."""
    rows, change = _select(data)
    for row in rows:
        if row["reject_reason"]:
            row["reject_reason"] = row["reject_reason"].replace("exp8232", "exp8274")
    if change["exists"]:
        return rows, change
    return rows, dict(exists=False, reason="no authenticated physical change since Exp8274")


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
            ("experiment_id", 8274),
            ("schema", "carnot.gatemate_physical_delta.v714.v1"),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
            ("fixture_mode", False),
        ]:
            observed = ref["sha256"] if field == "sha256" else value.get(field)
            data["checks"].append(operand(field, path, expected, observed))
        if not all(c["passed"] for c in data["checks"]):
            raise ValueError("exp8274_authentication")
        side = path.parent / "raw" / path.stem / "validators" / (PIN.split(":")[1] + ".json")
        bound = read_bound_sidecar(path, side)
        freeze(side, raw, data)
        data["checks"].append(operand("report.passed", side, True, bound["report"]["passed"]))
        _upstream_replay(path)
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
                experiment_id=8274,
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
            operand("gatemate_history", path, "authenticated Exp8274", str(error))
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


def reduce(data: Json) -> Json:
    """Carry a falsifiable board obligation while keeping its execution gates unpassed."""
    with adapters():
        return prior.reduce(data)
