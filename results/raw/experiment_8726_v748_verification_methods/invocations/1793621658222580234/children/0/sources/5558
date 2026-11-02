"""REQ-REPORT-8540: original scientific authority survives current task activation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

from carnot.reporting import v733_historical_inputs as old
from carnot.reporting import v733_capstone_evidence as cap
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
FIELDS = {
    **old.FIELDS,
    8529: "source_audit_ready_score",
    8536: "arc_capability_ready_score",
}


def qualify(args: Json, scratch: Path) -> Json:
    """The original consumer must qualify a family before its ready scalar can be used."""
    n = args["number"]
    if n in old.FIELDS:
        return dict(old.qualify(args, scratch))
    primary = args["primary"]
    try:
        with frozen_inputs(args["authority"], scratch):
            outcome = cap.outcome(args["task"], primary, scratch)
        value = cap.load(primary)
        observed = value.get(FIELDS[n])
        ready = outcome["eligible"] and observed == 1
        failures = outcome["checks"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        ready, observed = False, None
        failures = [cap.gate(primary, FIELDS[n], 1, str(error), f"exp{n}")]
    return dict(
        experiment_id=n,
        field=FIELDS[n],
        observed=observed,
        qualified=bool(ready),
        primary_sha256=primary.get("sha256"),
        gate_check_summary=failures,
        qualification_scope="original V733 publication and primitive consumer; no fit",
        historical_model_provenance=[],
    )


def worker(request: Path, output: Path) -> int:
    """Fresh processes keep historical reconstruction memory and writes in disk scratch."""
    with TemporaryDirectory(prefix="exp8540-family-", dir="/var/tmp") as directory:
        row = qualify(json.loads(request.read_bytes()), Path(directory))
    atomic_json(output, row)
    return 0


def historical(root: Path, raw: Path, cli: str, protocol: Json) -> Json:
    """Separate original source and ARC qualification from failed learning and panel producers."""
    if not (root / "results").is_dir():
        return dict(
            families=[
                dict(
                    experiment_id=n,
                    field=field,
                    qualified=False,
                    primary_sha256=None,
                    gate_check_summary=[
                        cap.gate(
                            dict(path=str(root / "results" / f"experiment_{n}_missing.json")),
                            field,
                            1,
                            None,
                            f"exp{n}",
                        )
                    ],
                )
                for n, field in FIELDS.items()
            ],
            references=[],
            receipts=[],
        )
    result = dict(old.historical(root, raw, cli, protocol))
    authority = []
    for key in ["v733_design_reference", "v733_activation_reference"]:
        original = protocol[key]
        saved = cap.freeze(Path(original["path"]), raw / "V733" / "authority")
        saved["source_path"] = original["source_path"]
        authority.append(saved)
        result["references"].append(saved)
    for i, n in enumerate([8529, 8536]):
        cap.progress(f"original_family_{n}_before", i, 2 - i)
        task = next(t for t in protocol["task_map_v733"] if t["id"].startswith(f"exp{n}-"))
        primary = cap.freeze(root / task["deliverable"], raw / "V733" / "primaries")
        result["references"].append(primary)
        request, output = raw / f"request-{n}.json", raw / f"summary-{n}.json"
        atomic_json(request, dict(number=n, task=task, primary=primary, authority=authority))
        receipt = child(
            f"historical_{n}",
            [
                sys.executable,
                "-u",
                str(cap.ROOT / cli),
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
                    cap.gate(
                        primary, "bounded_original_reader", 0, receipt["actual_exit"], f"exp{n}"
                    )
                ],
            )
        )
        result["families"].append(row)
        cap.progress(f"original_family_{n}_after", i + 1, 1 - i)
    return result
