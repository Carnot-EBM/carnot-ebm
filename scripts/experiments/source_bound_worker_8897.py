"""REQ-REPORT-8897: a private installation receives typed provenance without hidden labels."""

import hashlib
import json
from pathlib import Path
import sys

import carnot.verify.source_bound_witness as installed
import carnot.verify.size_aware_witness as exact


def main(argv: list[str]) -> None:
    """Retain deterministic decisions and imported bytes so a checkout import cannot pass."""
    requests = json.loads(Path(argv[0]).read_bytes())["requests"]
    rows = []
    for index, request in enumerate(requests):
        if request["arm"] == "view_only":
            value = exact.verify_size_aware_witness(request["payload"])
            result = dict(
                record_decision=value["decision"],
                budget_exhausted=value["budget_exhausted"],
                source_coverage_status="unknown",
                source_semantics_status="unknown",
                ledger_errors=[],
            )
        else:
            value = installed.verify_source_bound_witness(
                request["payload"],
                record_ids=request["record_ids"],
                source_ledger=request["source_ledger"],
                source_bytes=bytes.fromhex(request["source_hex"])
                if request["source_hex"] is not None
                else None,
            )
            result = {
                key: value[key]
                for key in [
                    "record_decision",
                    "source_coverage_status",
                    "source_semantics_status",
                    "ledger_errors",
                    "budget_exhausted",
                ]
            }
        rows.append(result)
        if (index + 1) % 100 == 0 or index + 1 == len(requests):
            print(
                f"[exp8897-installed] completed={index + 1} pending={len(requests) - index - 1}",
                flush=True,
            )
    imported = []
    for module in (installed, exact):
        path = Path(str(module.__file__)).resolve()
        imported.append(
            dict(path=str(path), sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
        )
    Path(argv[1]).write_text(json.dumps(dict(rows=rows, imported=imported)))


if __name__ == "__main__":
    main(sys.argv[1:])
