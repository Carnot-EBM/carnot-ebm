"""REQ-REPORT-8526: original consumers keep separate historical authority and dates."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from carnot.reporting import v732_historical_inputs as old
from carnot.reporting import v730_capstone_evidence as v730
from carnot.reporting import v732_capstone_evidence as v732
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import child
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
FIELDS = {
    **old.FIELDS,
    8486: "source_controls_ready_score",
    8487: "head_controls_ready_score",
    8516: "learning_qualification_ready_score",
    8520: "service_cost_ready_score",
}
EXTRA = [8486, 8487, 8516, 8520]


def qualify(args: Json, scratch: Path) -> Json:
    """Replay each imported primary using its shipped consumer, never its score alone."""
    n, primary = args["number"], args["primary"]
    if n in old.FIELDS:
        return dict(old.qualify(args, scratch))
    cap = v730 if n < 8512 else v732
    observed = None
    try:
        if primary.get("sha256") != "sha256:" + cap.PINS[n]:
            raise ValueError("original_producer_pin")
        original = v732.methods.protocol()
        refs = (
            original["v730_authority_references"]
            if n < 8512
            else [original["v732_design_reference"], original["v732_activation_reference"]]
        )
        for ref, pin in zip(args["authority"], refs, strict=True):
            if (
                ref["sha256"] != pin["sha256"]
                or sha256_file(Path(ref["snapshot_path"])) != pin["sha256"]
            ):
                raise ValueError("historical_authority_pin")
        task = next(
            t
            for t in yaml.safe_load(Path(args["authority"][1]["snapshot_path"]).read_bytes())[
                "tasks"
            ]
            if t["id"].startswith(f"exp{n}-")
        )
        value = cap.load(primary)
        observed = value.get(FIELDS[n])
        with frozen_inputs(args["authority"], scratch):
            checked = cap.outcome(task, primary, scratch)["eligible"]
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
        original_run_date=value.get("run_date") if checked else None,
        qualification_scope="original frozen-family publication and primitive consumer; no current fitting or timing",
        gate_check_summary=failures,
        historical_model_provenance=old.old.model_provenance(value, primary) if checked else [],
    )


def worker(request: Path, output: Path) -> int:
    """A fresh process confines original reader memory and reconstruction files."""
    with TemporaryDirectory(prefix="exp8526-family-", dir="/var/tmp") as directory:
        result = qualify(json.loads(request.read_bytes()), Path(directory))
    atomic_json(output, result)
    return 0


def historical(root: Path, raw: Path, cli: str, protocol: Json) -> Json:
    """Separate immutable family roots prevent current activation from rewriting history."""
    result = dict(old.historical(root, raw, cli, protocol))
    authorities = {}
    for family in ["v730", "v732"]:
        refs = []
        for original in protocol[family + "_authority_references"]:
            saved = v732.freeze(Path(original["path"]), raw / family / "authority")
            saved["source_path"] = original["source_path"]
            refs.append(saved)
            result["references"].append(saved)
        authorities[family] = refs
    for i, n in enumerate(EXTRA):
        print(
            f"[exp8526] phase=family_{n}_before completed={i} pending={len(EXTRA) - i}", flush=True
        )
        primary = v732.freeze(
            next(
                (root / "results").glob(f"experiment_{n}_*.json"),
                root / "results" / f"experiment_{n}_missing.json",
            ),
            raw / "extra" / "primaries",
        )
        result["references"].append(primary)
        request, output = raw / f"request-{n}.json", raw / f"summary-{n}.json"
        atomic_json(
            request,
            dict(number=n, primary=primary, authority=authorities["v730" if n < 8512 else "v732"]),
        )
        receipt = child(
            f"historical_{n}",
            [
                sys.executable,
                "-u",
                str(old.old.ROOT / cli),
                "--history-request",
                str(request),
                "--history-output",
                str(output),
            ],
            raw / "logs",
            deadline=120,
            scope="upstream",
        )
        result["receipts"].append(receipt)
        row = (
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
                        artifact_path=primary["path"],
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
        result["families"].append(row)
        print(
            f"[exp8526] phase=family_{n}_after completed={i + 1} pending={len(EXTRA) - i - 1}",
            flush=True,
        )
    return result
