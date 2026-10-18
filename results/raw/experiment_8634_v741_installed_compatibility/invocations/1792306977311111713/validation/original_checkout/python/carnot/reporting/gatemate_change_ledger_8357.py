"""REQ-VERIFY-8357: preserve immutable history before reading physical changes.

Historical closure quality and physical execution have separate gates. This
adapter reuses the established receipt parser without any device operation.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import gatemate_change_ledger_8330 as old
from carnot.reporting import gatemate_change_ledger_8344 as previous
from carnot.reporting import gatemate_change_ledger_8246 as prior
from carnot.reporting import gatemate_obligation_8316 as obligation
from carnot.reporting import gatemate_history_8357 as history
from carnot.reporting import v720_frozen_input_contract as contract
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.request_trace_inventory_8200 import operand
from carnot.reporting.v717_contract_methods import PIN as PROTOCOL_PIN
from scripts.experiments import experiment_7146_v627_gatemate_changed_state as legacy

Json = dict[str, Any]
ROOT = previous.ROOT
NAME = "experiment_8357_v720_gatemate_change_ledger"
CLI = "scripts/experiments/" + NAME + ".py"
REOPEN = "docs/research-notes/v720-gatemate-obligation.md"
UPSTREAM = "results/experiment_8330_v718_gatemate_change_ledger.json"
PIN = previous.PIN
V719 = "results/experiment_8344_v719_gatemate_change_ledger.json"
V719_PIN = "sha256:48eb3ad7bccb2d869d561c3e021b50348fed3339bddbce2d493442ce7108f73b"
CONFIG: Json = dict(seed=7208357, run_date="20261009", previous_primary_sha256=PIN)
MODEL_SPECS: list[Json] = []
DOCS = list(dict.fromkeys([*previous.DOCS, "ops/operator-followup.md", REOPEN]))
freeze, frontier, previous_cli = previous.freeze, previous.frontier, previous.previous_cli
empty_data = previous.empty_data
select_change = previous.select_change


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so an evidence audit remains visible without padded elapsed time."""
    print(f"[exp8357] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def adapters() -> Iterator[None]:
    """Reuse parser and reduction enforcement with only this invocation's bindings."""
    with (
        patch.object(old, "CONFIG", CONFIG),
        patch.object(old, "DOCS", DOCS),
        patch.object(old, "REOPEN", REOPEN),
        patch.object(old, "select_change", select_change),
    ):
        with old.adapters(), obligation.adapters():
            yield


def authority(root: Path, raw: Path) -> Json:
    """Current authority is authenticated independently from historical snapshots."""
    actual = contract.authority(root, raw)
    task = next(t for t in actual["tasks"] if t["id"] == "exp8357-gatemate-change-ledger")
    return dict(activated=actual["activated"], digest=actual["active_tasks_sha256"], task=task)


def load(root: Path, raw: Path) -> Json:
    """Recover exact seals first; incomplete history suppresses physical inspection."""
    data = empty_data()
    progress("history_before", 0, 2)
    index: dict[str, list[Path]] = {}
    for parent in [
        root / "results/raw/experiment_8330_v718_gatemate_change_ledger",
        root / "results/raw/experiment_8344_v719_gatemate_change_ledger",
    ]:
        for path in parent.rglob("*"):
            if path.is_file() and len(path.stem) == 64:
                index.setdefault("sha256:" + path.stem, []).append(path)
    bundles = []
    for i, (relative, pin) in enumerate([(UPSTREAM, PIN), (V719, V719_PIN)]):
        try:
            bundle = history.authenticate(root / relative, pin, raw / "history" / str(i), index)
            bundles.append(bundle)
            if bundle["passed"]:
                bundle["replay"] = history.replay_bundle(bundle, raw / "history-replay" / str(i))
            for row in bundle["rows"]:
                if row["reference"]:
                    data["references"].append(row["reference"])
                data["checks"].append(
                    operand(
                        "source_closure.sha256",
                        Path(row["original_path"]),
                        row["expected_sha256"],
                        row["expected_sha256"] if row["available"] else None,
                    )
                )
            data["cited"].append(
                dict(
                    experiment_id=8330 if i == 0 else 8344,
                    path=str(root / relative),
                    sha256=pin,
                    fields_imported=[
                        "code_config_hashes",
                        "gate_check_summary",
                        "gatemate_obligation",
                    ],
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            data["checks"].append(operand("gatemate_history", root / relative, pin, str(error)))
        progress("history_closure", i + 1, 1 - i)
    data["history_authentication"] = dict(
        passed=len(bundles) == 2 and all(b["passed"] and b["replay"]["passed"] for b in bundles),
        bundles=bundles,
        missing_hashes=[
            dict(path=r["original_path"], sha256=r["expected_sha256"])
            for b in bundles
            for r in b["rows"]
            if not r["available"]
        ],
    )
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
        data["checks"].append(
            operand(
                "authority.activated", root / contract.ACTIVE, True, data["authority"]["activated"]
            )
        )
        data["checks"].append(
            operand(
                "task.deliverable",
                root / contract.DESIGN,
                "results/" + NAME + ".json",
                data["authority"]["task"]["deliverable"],
            )
        )
        data["checks"].append(
            operand(
                "protocol.sha256",
                root / contract.PROTOCOL,
                PROTOCOL_PIN,
                data["authority_refs"][2]["sha256"],
            )
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        data["checks"].append(
            operand("authority", root / contract.DESIGN, "activated V720 task", str(error))
        )
    if bundles:
        value = json.loads(checked(bundles[0]["primary"]).read_bytes())
        data["board"] = value["gatemate_obligation"]
        transcript = freeze(
            root / "results/experiment_6559_gatemate_changed_state_continuity.json", raw, data
        )
        data["checks"].append(
            operand(
                "original_transcript.sha256",
                Path(transcript["path"]),
                old.TRANSCRIPT_PIN,
                transcript["sha256"],
            )
        )
    data["history_ready"] = data["history_authentication"]["passed"] and all(
        c["passed"] for c in data["checks"]
    )
    progress("history_after", 2, 0)
    if data["history_ready"]:
        data.update(
            previous_reference=bundles[0]["primary"],
            source_root=str(root),
            frontier=frontier(value, root, time.time_ns()),
        )
        for relative in list(dict.fromkeys([*DOCS, *sorted(map(str, legacy.RECEIPT_PATHS))])):
            path = root / relative
            snapshot = freeze(path, raw, data) if path.is_file() else None
            old_hash = data["frontier"]["documents"].get(relative)
            digest = snapshot["sha256"] if snapshot else None
            data["document_rows"].append(
                dict(
                    relative_path=relative,
                    receipt_source=relative in map(str, legacy.RECEIPT_PATHS),
                    required=relative in DOCS,
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
            data["checks"].append(operand("named_input.exists", path, True, path.is_file()))
            progress("physical_documents", len(data["document_rows"]), 0)
        with adapters():
            data["candidate_rows"] = prior.scan(data["document_rows"])
            data["receipt_rows"], data["physical_change"] = select_change(data)
    data["checks"].append(
        operand(
            "physical_change_evidence.exists",
            root / UPSTREAM,
            True,
            data["physical_change"].get("exists", False) if data["history_ready"] else None,
        )
    )
    return data


def verify_primitives(data: Json) -> None:
    """Rebuild closure membership and authority; new hashes cannot bless new meaning."""
    auth = data["history_authentication"]
    if not data["fixture"]:
        missing: list[Json] = []
        for bundle in auth["bundles"]:
            value = json.loads(checked(bundle["primary"]).read_bytes())
            pin = PIN if value["experiment_id"] == 8330 else V719_PIN
            if bundle["primary_sha256"] != pin or bundle["primary"]["sha256"] != pin:
                raise ValueError("historical_primary_drift")
            seals = (pin, *history.SEALS[pin])
            side = json.loads(checked(bundle["rows"][1]["reference"]).read_bytes())
            terminal = json.loads(checked(bundle["rows"][2]["reference"]).read_bytes())
            if tuple(r["expected_sha256"] for r in bundle["rows"][:3]) != seals or tuple(
                r["original_path"] for r in bundle["rows"][:3]
            ) != (
                side["primary_path"],
                terminal["publication"]["sidecar_path"],
                value["terminal_validation_sidecar_path"],
            ):
                raise ValueError("terminal_membership_drift")
            expected = [(r["path"], r["sha256"]) for r in history.requirements(value)]
            observed = [(r["original_path"], r["expected_sha256"]) for r in bundle["rows"][3:]]
            if expected != observed or bundle["passed"] != all(
                r["available"] for r in bundle["rows"]
            ):
                raise ValueError("closure_membership_drift")
            history.verify(bundle["rows"])
            for receipt in bundle["receipts"]:
                for stream in ["stdout", "stderr"]:
                    checked(
                        dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                    )
            missing.extend(
                dict(path=r["original_path"], sha256=r["expected_sha256"])
                for r in bundle["rows"]
                if not r["available"]
            )
        passed = len(auth["bundles"]) == 2 and all(
            b["passed"] and b["replay"]["passed"] for b in auth["bundles"]
        )
        if auth["passed"] != passed or auth["missing_hashes"] != missing:
            raise ValueError("history_authentication_drift")
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="exp8357-authority-") as directory:
            root = Path(directory)
            for ref in data["authority_refs"]:
                output = root / ref["relative"]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(checked(ref).read_bytes())
            if data["authority"] and authority(root, root / "scratch") != data["authority"]:
                raise ValueError("authority_drift")
        if data["board"] and data["board"]["source_transcript_sha256"] != old.TRANSCRIPT_PIN:
            raise ValueError("original_transcript_drift")
        for bundle in auth["bundles"]:
            if bundle["passed"]:
                with TemporaryDirectory(prefix="exp8357-history-control-") as directory:
                    replayed = history.replay_bundle(deepcopy(bundle), Path(directory))
                if replayed["passed"] != bundle["replay"]["passed"]:
                    raise ValueError("historical_replay_drift")
    if data["history_ready"]:
        with adapters():
            prior.verify_primitives(data)
    elif data["document_rows"] or data["candidate_rows"] or data["physical_change"].get("exists"):
        raise ValueError("physical_scan_before_history")


def reduce(data: Json) -> Json:
    """Ledger quality cannot create hardware execution or a scientific measurement."""
    with adapters():
        value: Json = obligation.reduce(data)
    value.update(
        execution_authority=data["authority"],
        history_authentication=data["history_authentication"],
        original_transcript_sha256=data["board"].get("source_transcript_sha256"),
    )
    return value
