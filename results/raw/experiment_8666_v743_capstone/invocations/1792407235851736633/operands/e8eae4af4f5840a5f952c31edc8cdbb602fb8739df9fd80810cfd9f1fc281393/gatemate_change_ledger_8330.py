"""REQ-VERIFY-8330: retain physical evidence without making a device call.

The existing reader already qualifies receipt parsing and frontier semantics.
These adapters change the authenticated predecessor and keep that enforcement.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import gatemate_obligation_8316 as previous
from carnot.reporting import gatemate_change_ledger_8246 as prior
from carnot.reporting import gatemate_obligation_execution_8316 as upstream_cli
from carnot.reporting import v718_contract_replay as contract
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.request_trace_inventory_8200 import operand
from carnot.reporting.v717_contract_methods import PIN as PROTOCOL_PIN
from scripts.experiments import experiment_7146_v627_gatemate_changed_state as legacy

Json = dict[str, Any]
ROOT = previous.ROOT
NAME = "experiment_8330_v718_gatemate_change_ledger"
CLI = "scripts/experiments/" + NAME + ".py"
UPSTREAM = "results/experiment_8316_v717_gatemate_obligation.json"
PIN = "sha256:927a69b20c76a04f63672cd84d9ae6ca50ffb4bc97eb76a39a8b5cd798226f3c"
BOUND_PIN = "sha256:e1a94b3b4b2f3e6c87383e95a46b170b4193a0221145a949b95e80f7858cb7f1"
TERMINAL_PIN = "sha256:da029c975d1cb302a54ea2d442050c07d2ecc56e812673eae491c0c3c1af347e"
TRANSCRIPT_PIN = "sha256:59a76f8ab46fa24b1ebe9aa038dde2ccf35a32a348e02696409b03ff096c8e66"
REOPEN = "docs/research-notes/v718-gatemate-obligation.md"
CONFIG: Json = dict(seed=7188330, run_date="20261009", previous_primary_sha256=PIN)
DOCS = [*previous.DOCS, REOPEN, "docs/jtag-wiring-gatemate-dirtyjtag.md"]
freeze, frontier = previous.freeze, previous.frontier
previous_cli = previous.previous_cli
_select = previous.select_change
_historical_replay = upstream_cli.replay


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts make each documentation phase visible to the supervisor."""
    print(f"[exp8330] phase={phase} completed={completed} pending={pending}", flush=True)


def empty_data() -> Json:
    """Missing history still retains a single obligation with unavailable evidence."""
    return dict(previous.empty_data(), authority={}, authority_refs=[])


def select_change(data: Json) -> tuple[list[Json], Json]:
    """Reuse physical field and timestamp checks against the advanced frontier."""
    rows, change = _select(data)
    for row in rows:
        if row["reject_reason"]:
            row["reject_reason"] = row["reject_reason"].replace("exp8302", "exp8316")
    if not change["exists"]:
        change["reason"] = "no authenticated physical change since Exp8316"
    return rows, change


@contextmanager
def adapters() -> Iterator[None]:
    """Temporary bindings preserve the historical reader and its test contracts."""
    with (
        patch.object(previous, "PIN", PIN),
        patch.object(previous, "BOUND_PIN", BOUND_PIN),
        patch.object(previous, "TERMINAL_PIN", TERMINAL_PIN),
        patch.object(previous, "CONFIG", CONFIG),
        patch.object(previous, "DOCS", DOCS),
        patch.object(previous, "REOPEN", REOPEN),
        patch.object(previous, "select_change", select_change),
    ):
        yield


def load(root: Path, raw: Path) -> Json:
    """Authenticate exact upstream bytes before inspecting changed operator documents."""
    data = empty_data()
    path = root / UPSTREAM
    progress("authentication_before", 0, 1)
    try:
        for relative in [
            contract.DESIGN,
            contract.ACTIVE,
            contract.PROTOCOL,
            "ops/exclusion_manifest.yaml",
        ]:
            data["authority_refs"].append(
                dict(freeze(root / relative, raw, data), relative=relative)
            )
        actual = contract.authority(root, raw / "authority")
        data["checks"].append(
            operand(
                "protocol.sha256",
                root / contract.PROTOCOL,
                PROTOCOL_PIN,
                data["authority_refs"][2]["sha256"],
            )
        )
        task = next(t for t in actual["tasks"] if t["id"] == "exp8330-gatemate-change-ledger")
        data["authority"] = dict(
            activated=actual["activated"], digest=actual["active_tasks_sha256"], task=task
        )
        data["checks"].extend(
            [
                operand("authority.activated", root / contract.ACTIVE, True, actual["activated"]),
                operand(
                    "task.deliverable",
                    root / contract.DESIGN,
                    "results/" + NAME + ".json",
                    task["deliverable"],
                ),
            ]
        )
        ref = freeze(path, raw, data)
        value = json.loads(checked(ref).read_bytes())
        for field, expected, observed in [
            ("sha256", PIN, ref["sha256"]),
            ("experiment_id", 8316, value.get("experiment_id")),
            ("schema", "carnot.gatemate_obligation.v717.v1", value.get("schema")),
            ("required_checks_passed", True, value.get("required_checks_passed")),
            ("flagged_adversarial", False, value.get("flagged_adversarial")),
            ("fixture_mode", False, value.get("fixture_mode")),
        ]:
            data["checks"].append(operand(field, path, expected, observed))
        side = path.parent / "raw" / path.stem / "validators" / (PIN[7:] + ".json")
        bound = read_bound_sidecar(path, side)
        bound_ref = freeze(side, raw, data)
        terminal = freeze(Path(value["terminal_validation_sidecar_path"]), raw, data)
        report = json.loads(checked(terminal).read_bytes())
        for field, expected, observed in [
            ("bound_sidecar.sha256", BOUND_PIN, bound_ref["sha256"]),
            ("report.passed", True, bound["report"]["passed"]),
            ("terminal_sidecar.sha256", TERMINAL_PIN, terminal["sha256"]),
            ("publication.primary_sha256", PIN, report["publication"]["primary_sha256"]),
            (
                "publication.primary_path",
                str(path.absolute()),
                report["publication"]["primary_path"],
            ),
            (
                "private_candidate_validation.passed",
                True,
                report["private_candidate_validation"]["passed"],
            ),
        ]:
            data["checks"].append(operand(field, side, expected, observed))
        if not all(c["passed"] for c in data["checks"]):
            raise ValueError("upstream_or_authority_authentication")
        with patch.object(upstream_cli, "h", previous):
            _historical_replay(path)
        transcript = freeze(Path(value["gatemate_obligation"]["source_transcript"]), raw, data)
        if transcript["sha256"] != TRANSCRIPT_PIN:
            raise ValueError("original_transcript_drift")
        freeze(checked(value["replay_input_reference"]), raw, data)
        for item in value["source_artifact_hashes"]:
            freeze(checked(item), raw, data)
        data.update(
            board=value["gatemate_obligation"],
            history_ready=True,
            previous_reference=ref,
            terminal_reference=terminal,
            bound_reference=bound_ref,
            source_root=str(root),
            frontier=frontier(value, root, time.time_ns()),
        )
        data["cited"].append(
            dict(
                experiment_id=8316,
                path=str(path),
                sha256=PIN,
                fields_imported=[
                    "gatemate_obligation",
                    "invocation",
                    "replay_input_reference",
                    "source_artifact_hashes",
                ],
            )
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        data["checks"].append(
            operand(
                "gatemate_history", path, "authenticated Exp8316 and V718 authority", str(error)
            )
        )
    progress("authentication_after", 1, 0)
    receipts = set(map(str, legacy.RECEIPT_PATHS))
    documents = list(dict.fromkeys([*DOCS, *sorted(receipts)]))
    for index, relative in enumerate(documents):
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
        progress("document_frontier", index + 1, len(documents) - index - 1)
    if data["history_ready"]:
        with adapters(), previous.adapters():
            data["candidate_rows"] = prior.scan(data["document_rows"])
            data["receipt_rows"], data["physical_change"] = select_change(data)
    data["checks"].append(
        operand("physical_change_evidence.exists", path, True, data["physical_change"]["exists"])
    )
    return data


def verify_primitives(data: Json) -> None:
    """Rebuild frozen authority and receipt semantics so a new hash cannot bless drift."""
    with adapters():
        previous.verify_primitives(data)
    if data["history_ready"] and not data["fixture"]:
        with TemporaryDirectory(prefix="exp8330-replay-") as directory:
            root = Path(directory)
            for ref in data["authority_refs"]:
                output = root / ref["relative"]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(checked(ref).read_bytes())
            actual = contract.authority(root, root / "scratch")
            task = next(t for t in actual["tasks"] if t["id"] == "exp8330-gatemate-change-ledger")
            if data["authority"] != dict(
                activated=actual["activated"], digest=actual["active_tasks_sha256"], task=task
            ):
                raise ValueError("authority_drift")


def reduce(data: Json) -> Json:
    """Ledger readiness certifies the obligation while execution remains unavailable."""
    with adapters():
        value = previous.reduce(data)
    value["execution_authority"] = data.get("authority", {})
    return value
