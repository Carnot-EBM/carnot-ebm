"""REQ-VERIFY-8344: carry physical custody without a hardware retry.

The qualified reader owns physical receipt semantics. These bindings advance
its immutable predecessor while authenticating current authority separately.
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
from carnot.reporting import gatemate_change_ledger_8330 as historical
from carnot.reporting import gatemate_ledger_execution_8330 as historical_cli
from carnot.reporting import v719_contract_replay as contract
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.request_trace_inventory_8200 import operand
from carnot.reporting.v717_contract_methods import PIN as PROTOCOL_PIN
from scripts.experiments import experiment_7146_v627_gatemate_changed_state as legacy

Json = dict[str, Any]
ROOT = previous.ROOT
NAME = "experiment_8344_v719_gatemate_change_ledger"
CLI = "scripts/experiments/" + NAME + ".py"
UPSTREAM = "results/experiment_8330_v718_gatemate_change_ledger.json"
PIN = "sha256:6f79a6d2374c41baec7bf5ecb792b8534bc59b91e8b0c8cbe5720305d8545512"
BOUND_PIN = "sha256:11ad525107a42f9017513b49207fb20f18c412a9dc0482c743e6aac770b7d8f8"
TERMINAL_PIN = "sha256:1a954d66f6202d5487dfd4b784ceeb0b8aac07d88df63227bec03485fbfb8e65"
TRANSCRIPT_PIN = historical.TRANSCRIPT_PIN
REOPEN = "docs/research-notes/v719-gatemate-obligation.md"
CONFIG: Json = dict(seed=7198344, run_date="20261009", previous_primary_sha256=PIN)
DOCS = list(dict.fromkeys([*historical.DOCS, REOPEN]))
MODEL_SPECS: list[Json] = []
freeze, frontier, previous_cli = previous.freeze, previous.frontier, previous.previous_cli
_historical_replay = historical_cli.replay
_select = previous.select_change


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so the supervisor can observe this bounded documentation work."""
    print(f"[exp8344] phase={phase} completed={completed} pending={pending}", flush=True)


def empty_data() -> Json:
    """Absent operands retain the intended obligation without fabricated observations."""
    return dict(previous.empty_data(), authority={}, authority_refs=[])


def select_change(data: Json) -> tuple[list[Json], Json]:
    """Only the qualified cable, port, power and board reader can prepare preflight."""
    rows, change = _select(data)
    for row in rows:
        if row["reject_reason"]:
            row["reject_reason"] = row["reject_reason"].replace("exp8302", "exp8330")
    if not change["exists"]:
        change["reason"] = "no authenticated physical change since Exp8330"
    return rows, change


@contextmanager
def adapters() -> Iterator[None]:
    """Scoped bindings preserve historical parser and reduction enforcement."""
    with (
        patch.object(previous, "PIN", PIN),
        patch.object(previous, "BOUND_PIN", BOUND_PIN),
        patch.object(previous, "TERMINAL_PIN", TERMINAL_PIN),
        patch.object(previous, "DOCS", DOCS),
        patch.object(previous, "CONFIG", CONFIG),
        patch.object(previous, "REOPEN", REOPEN),
        patch.object(previous, "select_change", select_change),
    ):
        yield


def authority(root: Path, raw: Path) -> Json:
    """Read the full current task directly, without requiring any future producer."""
    actual = contract.authority(root, raw)
    task = next(t for t in actual["tasks"] if t["id"] == "exp8344-gatemate-change-ledger")
    return dict(activated=actual["activated"], digest=actual["active_tasks_sha256"], task=task)


def load(root: Path, raw: Path) -> Json:
    """Authenticate immutable history before scanning changed operator documents."""
    data, path = empty_data(), root / UPSTREAM
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
        data["authority"] = authority(root, raw / "authority")
        ref = freeze(path, raw, data)
        value = json.loads(checked(ref).read_bytes())
        side = path.parent / "raw" / path.stem / "validators" / (PIN[7:] + ".json")
        bound = read_bound_sidecar(path, side)
        data["bound_reference"] = freeze(side, raw, data)
        data["terminal_reference"] = freeze(
            Path(value["terminal_validation_sidecar_path"]), raw, data
        )
        terminal = json.loads(checked(data["terminal_reference"]).read_bytes())
        for field, source, expected, observed in [
            (
                "protocol.sha256",
                root / contract.PROTOCOL,
                PROTOCOL_PIN,
                data["authority_refs"][2]["sha256"],
            ),
            ("authority.activated", root / contract.ACTIVE, True, data["authority"]["activated"]),
            (
                "task.deliverable",
                root / contract.DESIGN,
                "results/" + NAME + ".json",
                data["authority"]["task"]["deliverable"],
            ),
            ("sha256", path, PIN, ref["sha256"]),
            ("experiment_id", path, 8330, value.get("experiment_id")),
            ("required_checks_passed", path, True, value.get("required_checks_passed")),
            ("flagged_adversarial", path, False, value.get("flagged_adversarial")),
            ("fixture_mode", path, False, value.get("fixture_mode")),
            ("bound_sidecar.sha256", side, BOUND_PIN, data["bound_reference"]["sha256"]),
            ("report.passed", side, True, bound["report"]["passed"]),
            ("terminal_sidecar.sha256", side, TERMINAL_PIN, data["terminal_reference"]["sha256"]),
            ("publication.primary_sha256", side, PIN, terminal["publication"]["primary_sha256"]),
            (
                "publication.primary_path",
                side,
                str(path.absolute()),
                terminal["publication"]["primary_path"],
            ),
            (
                "private_candidate_validation.passed",
                side,
                True,
                terminal["private_candidate_validation"]["passed"],
            ),
        ]:
            data["checks"].append(operand(field, source, expected, observed))
        if not all(c["passed"] for c in data["checks"]):
            raise ValueError("history_or_authority_authentication")
        with patch.object(historical_cli, "h", historical):
            _historical_replay(path)
        transcript = freeze(Path(value["gatemate_obligation"]["source_transcript"]), raw, data)
        if transcript["sha256"] != TRANSCRIPT_PIN:
            raise ValueError("original_transcript_drift")
        for item in [value["replay_input_reference"], *value["source_artifact_hashes"]]:
            freeze(checked(item), raw, data)
        data.update(
            board=value["gatemate_obligation"],
            history_ready=True,
            previous_reference=ref,
            source_root=str(root),
            frontier=frontier(value, root, time.time_ns()),
        )
        data["cited"].append(
            dict(
                experiment_id=8330,
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
                "gatemate_history", path, "authenticated Exp8330 and V719 authority", str(error)
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
    """Reconstruct history and authority so new hashes cannot bless altered meaning."""
    with adapters():
        previous.verify_primitives(data)
    if data["history_ready"] and not data["fixture"]:
        transcript = Path(data["board"]["source_transcript"])
        if freeze_hash(transcript) != TRANSCRIPT_PIN:
            raise ValueError("original_transcript_drift")
        with TemporaryDirectory(prefix="exp8344-replay-") as directory:
            root = Path(directory)
            for ref in data["authority_refs"]:
                output = root / ref["relative"]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(checked(ref).read_bytes())
            actual = authority(root, root / "scratch")
            if actual != data["authority"] or freeze_hash(root / contract.PROTOCOL) != PROTOCOL_PIN:
                raise ValueError("authority_drift")


def freeze_hash(path: Path) -> str:
    """Use the same qualified byte hash for original transcript and frozen protocol."""
    from carnot.reporting.current_work_receipt import sha256_file

    return str(sha256_file(path))


def reduce(data: Json) -> Json:
    """Ledger quality cannot satisfy device execution or establish scientific benefit."""
    with adapters():
        value = previous.reduce(data)
    value["execution_authority"] = data["authority"]
    return value
