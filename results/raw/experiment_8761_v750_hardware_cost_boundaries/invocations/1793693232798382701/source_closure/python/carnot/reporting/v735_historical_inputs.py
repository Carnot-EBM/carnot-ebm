"""REQ-REPORT-8552: retain original dates and isolate input families in disk scratch."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

from carnot.reporting import v727_historical_inputs as source
from carnot.reporting import v734_historical_inputs as previous
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
FIELDS = {n: previous.FIELDS[n] for n in [8430, 8431, 8433, 8434, 8529]}


def worker(request: Path, output: Path) -> int:
    """A fresh process confines original reconstruction memory and writes to disk scratch."""
    with TemporaryDirectory(prefix="exp8552-original-", dir="/var/tmp") as directory:
        result = previous.qualify(json.loads(request.read_bytes()), Path(directory))
    atomic_json(output, result)
    return 0


def historical(root: Path, raw: Path, cli: str, protocol: Json) -> Json:
    """Qualify each original consumer independently; missing families retain explicit gates."""
    result: Json = dict(families=[], references=[], receipts=[])
    authorities: dict[str, list[Json]] = dict(V726=[], V733=[])
    present = (root / "results").is_dir()
    if present:
        authorities["V726"] = source.custody(source.ROOT, raw / "V726" / "authority")
        for key in ["v733_design_reference", "v733_activation_reference"]:
            ref = protocol[key]
            saved = previous.cap.freeze(Path(ref["path"]), raw / "V733" / "authority")
            saved["source_path"] = ref["source_path"]
            authorities["V733"].append(saved)
        result["references"] += [r for refs in authorities.values() for r in refs]
    for i, (number, field) in enumerate(FIELDS.items()):
        previous.cap.progress(f"original_{number}_before", i, len(FIELDS) - i)
        family = "V733" if number == 8529 else "V726"
        task: Json = next(
            (t for t in protocol["task_map_v733"] if t["id"].startswith(f"exp{number}-")), {}
        )
        path = next(
            (root / "results").glob(f"experiment_{number}_*.json"),
            root / "results" / f"experiment_{number}_missing.json",
        )
        primary = previous.cap.freeze(path, raw / family / "primaries")
        result["references"].append(primary)
        request, output = raw / f"request-{number}.json", raw / f"summary-{number}.json"
        atomic_json(
            request, dict(number=number, primary=primary, authority=authorities[family], task=task)
        )
        receipt = (
            child(
                f"historical_{number}",
                [
                    sys.executable,
                    "-u",
                    str(source.ROOT / cli),
                    "--history-request",
                    str(request),
                    "--history-output",
                    str(output),
                ],
                raw / "logs",
                deadline=120,
                scope="upstream",
            )
            if present
            else None
        )
        if receipt is not None:
            result["receipts"].append(receipt)
        if receipt and receipt["passed"]:
            row = json.loads(output.read_bytes())
        else:
            row = dict(
                experiment_id=number,
                field=field,
                qualified=False,
                primary_sha256=primary.get("sha256"),
                gate_check_summary=[
                    previous.cap.gate(
                        primary,
                        field,
                        1,
                        receipt["actual_exit"] if receipt else None,
                        f"exp{number}",
                    )
                ],
            )
        result["families"].append(row)
        previous.cap.progress(f"original_{number}_after", i + 1, len(FIELDS) - i - 1)
    return result
