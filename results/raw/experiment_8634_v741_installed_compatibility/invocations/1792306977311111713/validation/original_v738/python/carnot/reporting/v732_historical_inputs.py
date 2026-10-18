"""REQ-REPORT-8512: original V730 store qualification does not import speed benefit."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from carnot.reporting import v729_historical_inputs as old
from carnot.reporting import v730_capstone_evidence as cap
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import child
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
FIELDS = {**old.FIELDS, 8492: "append_protocol_ready_score", 8493: "append_store_ready_score"}


def qualify(args: Json, scratch: Path) -> Json:
    """Original publication and primitive consumers must qualify each imported family."""
    n, primary = args["number"], args["primary"]
    if n in old.FIELDS:
        return dict(old.qualify(args, scratch))
    observed = None
    try:
        if primary.get("sha256") != "sha256:" + cap.PINS[n]:
            raise ValueError("original_producer_pin")
        for ref, pin in zip(
            args["authority"],
            [
                "16895212c69e7ec0e916b6a65aeca584c6e036aacdb343f3ec3170ede2d0c827",
                "e021d835a85910871c387ca88d5c7daf192847aff5d48edb8bd94447b8e0f9c4",
            ],
            strict=True,
        ):
            if (
                ref["sha256"] != "sha256:" + pin
                or sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]
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
        qualification_scope="original V730 primitive/publication consumers; no new timing or speed benefit",
        gate_check_summary=failures,
        historical_model_provenance=old.model_provenance(value, primary) if checked else [],
    )


def worker(request: Path, output: Path) -> int:
    """Fresh readers isolate replay memory and place reconstruction files on disk."""
    with TemporaryDirectory(prefix="exp8512-family-", dir="/var/tmp") as directory:
        result = qualify(json.loads(request.read_bytes()), Path(directory))
    atomic_json(output, result)
    return 0


def historical(root: Path, raw: Path, cli: str, protocol: Json) -> Json:
    """Preserve original source families and add independently qualified V730 append inputs."""
    result = dict(old.historical(root, raw, cli, protocol))
    authority = []
    for original in protocol["v730_authority_references"]:
        saved = cap.freeze(Path(original["path"]), raw / "V730" / "authority")
        saved["source_path"] = original["source_path"]
        authority.append(saved)
        result["references"].append(saved)
    for i, n in enumerate([8492, 8493]):
        print(f"[exp8512] phase=V730_family_before completed={i} pending={2 - i}", flush=True)
        path = next(
            (root / "results").glob(f"experiment_{n}_*.json"),
            root / "results" / f"experiment_{n}_missing.json",
        )
        primary = cap.freeze(path, raw / "V730" / "primaries")
        result["references"].append(primary)
        request, output = raw / f"request-{n}.json", raw / f"summary-{n}.json"
        atomic_json(request, dict(number=n, primary=primary, authority=authority))
        receipt = child(
            f"historical_{n}",
            [
                sys.executable,
                "-u",
                str(old.ROOT / cli),
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
        result["families"].append(row)
        print(f"[exp8512] phase=V730_family_after completed={i + 1} pending={1 - i}", flush=True)
    return result
