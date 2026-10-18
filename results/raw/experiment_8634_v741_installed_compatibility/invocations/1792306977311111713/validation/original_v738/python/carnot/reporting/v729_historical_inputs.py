"""REQ-REPORT-8471: original source and service authority occupy separate roots."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from carnot.reporting import v727_historical_inputs as source
from carnot.reporting import v728_capstone_evidence as service
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import child
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
ROOT = source.ROOT
FIELDS = {
    **source.FIELDS,
    8464: "persistence_profile_ready_score",
    8465: "journal_ready_score",
    8466: "service_cost_ready_score",
}


def model_provenance(value: Json, primary: Json) -> list[Json]:
    """Keep model identity and imported counts beside a hash-bound reference to full cached receipts.

    Historical service receipts contain complete old traces. Repeating those
    trees in every family and citation makes administrative validation needlessly
    expensive. The immutable primary retains every original operand for readers.
    """
    return [
        dict(
            {
                k: row[k]
                for k in (
                    "experiment_id",
                    "producer",
                    "model",
                    "models",
                    "MODEL_SPECS",
                    "imported_counts",
                    "scope",
                )
                if k in row
            },
            provenance_reference=dict(
                path=primary["snapshot_path"],
                sha256=primary["sha256"],
                field="historical_model_provenance",
            ),
        )
        for row in value.get("historical_model_provenance", [])
    ]


def qualify(args: Json, scratch: Path) -> Json:
    """Shipped consumers reconstruct original operands; a ready score alone cannot qualify."""
    n, primary = args["number"], args["primary"]
    if n in source.FIELDS:
        result = dict(source.qualify(n, primary, args["authority"], scratch))
        if result["qualified"]:
            result["historical_model_provenance"] = model_provenance(service.load(primary), primary)
        return result
    observed = None
    try:
        if primary.get("sha256") != "sha256:" + service.PINS[n]:
            raise ValueError("original_producer_pin")
        pins = [
            "sha256:191897315bde0250ea57d967b046e5b91919dca699c7dd1c99ee69ac9dbaac14",
            "sha256:54a139617d9bfff745b72f20094468d43dd1b50fbfd29e8a5cbda66b80c26651",
        ]
        for ref, pin in zip(args["authority"], pins, strict=True):
            if ref["sha256"] != pin or sha256_file(Path(ref["snapshot_path"])) != pin:
                raise ValueError("historical_authority_pin")
        plan = yaml.safe_load(Path(args["authority"][1]["snapshot_path"]).read_bytes())
        task = next(t for t in plan["tasks"] if t["id"].startswith(f"exp{n}-"))
        value = service.load(primary)
        observed = value.get(FIELDS[n])
        with frozen_inputs(args["authority"], scratch):
            checked = service.outcome(task, primary, scratch)["eligible"]
        if not checked or observed != 1:
            raise ValueError("original_producer_replay_and_readiness")
        failures = []
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as error:
        checked = False
        failures = [
            dict(
                check="historical_producer_qualification",
                upstream_id=f"exp{n}",
                artifact_path=primary["path"],
                artifact_hash=primary.get("sha256"),
                artifact_field=FIELDS[n],
                op="==",
                expected=1,
                observed=observed,
                reason=str(error),
                passed=False,
            )
        ]
    return dict(
        experiment_id=n,
        field=FIELDS[n],
        observed=observed,
        qualified=bool(checked),
        primary_sha256=primary.get("sha256"),
        qualification_scope="original V728 producer primitive replay; no new benchmark",
        gate_check_summary=failures,
        historical_model_provenance=model_provenance(value, primary) if checked else [],
    )


def worker(request: Path, output: Path) -> int:
    """A fresh bounded process owns replay memory and puts reconstruction writes on disk."""
    with TemporaryDirectory(prefix="exp8471-family-", dir="/var/tmp") as directory:
        result = qualify(json.loads(request.read_bytes()), Path(directory))
    atomic_json(output, result)
    return 0


def historical(root: Path, raw: Path, cli: str, protocol: Json) -> Json:
    """Freeze separate original roots before invoking each isolated family reader."""
    refs: list[Json] = []
    authorities: dict[str, list[Json]] = {}
    for family, inputs in protocol["historical_authorities"].items():
        authorities[family] = []
        for original in inputs:
            saved = service.freeze(Path(original["path"]), raw / family / "authority")
            saved["source_path"] = original["source_path"]
            if saved["sha256"] != original["sha256"]:
                saved["exists"] = False
            authorities[family].append(saved)
            refs.append(saved)
    families, receipts = [], []
    for n in FIELDS:
        family = "V726" if n in source.FIELDS else "V728"
        path = next(
            (root / "results").glob(f"experiment_{n}_*.json"),
            root / "results" / f"experiment_{n}_missing.json",
        )
        primary = service.freeze(path, raw / family / "primaries")
        refs.append(primary)
        request, output = raw / f"request-{n}.json", raw / f"summary-{n}.json"
        atomic_json(request, dict(number=n, primary=primary, authority=authorities[family]))
        receipt = child(
            f"historical_{n}",
            [
                sys.executable,
                "-u",
                str(ROOT / cli),
                "--history-request",
                str(request),
                "--history-output",
                str(output),
            ],
            raw / "logs",
            deadline=120,
            scope="upstream",
        )
        receipts.append(receipt)
        result = (
            json.loads(output.read_bytes())
            if receipt["passed"]
            else dict(
                experiment_id=n,
                field=FIELDS[n],
                qualified=False,
                primary_sha256=primary.get("sha256"),
                gate_check_summary=[
                    dict(
                        check="bounded_historical_reader",
                        upstream_id=f"exp{n}",
                        artifact_path=str(path),
                        artifact_hash=primary.get("sha256"),
                        artifact_field=FIELDS[n],
                        op="==",
                        expected=0,
                        observed=receipt["actual_exit"],
                        passed=False,
                    )
                ],
            )
        )
        families.append(result)
    return dict(families=families, references=refs, receipts=receipts)
