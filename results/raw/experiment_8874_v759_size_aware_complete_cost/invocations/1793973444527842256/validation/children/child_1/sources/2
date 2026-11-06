"""REQ-REPORT-8874: installed complete request clocks without checkout imports."""

import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
from types import FunctionType
from typing import Any, Callable

from carnot.verify.size_aware_witness import verify_size_aware_witness

Json = dict[str, Any]
HISTORICAL = runpy.run_path(str(Path(__file__).with_name("installed_cost_worker_8861.py")))
compile_baseline = HISTORICAL["compile_baseline"]
ARMS = ("scan", "indexed", "size_aware")


def attempt(encoded: str, arm: str, baseline: Callable[..., Any], cap: int) -> Json:
    """Reuse the authenticated complete meter, changing only the selected public entry."""
    if arm not in ARMS:
        raise ValueError("arm")
    scope = dict(
        HISTORICAL["attempt"].__globals__, ARMS=("scan", arm if arm != "scan" else "indexed")
    )
    if arm == "size_aware":
        scope["verify_indexed_witness"] = verify_size_aware_witness
    evaluate = FunctionType(HISTORICAL["attempt"].__code__, scope)
    return dict(evaluate(encoded, arm, baseline, cap))


def main() -> int:
    """Pay resident serialization through disposal and retain every repeated observation."""
    started = time.monotonic_ns()
    baseline = compile_baseline(Path(sys.argv[1]).read_text())
    request = json.load(sys.stdin)
    rows = []
    for world in request["worlds"]:
        for repeat in request["repeats"]:
            for arm in world["arm_order"]:
                for unit in world["requests"]:
                    start = time.monotonic_ns()
                    encoded = json.dumps(unit["input"], sort_keys=True)
                    result = attempt(encoded, arm, baseline, request["cap"])
                    response = json.dumps(
                        dict(decision=result["decision"], witness=result["witness"]), sort_keys=True
                    )
                    response_bytes = len(response.encode())
                    del encoded, response
                    end = time.monotonic_ns()
                    rows.append(
                        dict(
                            result,
                            unit_id=unit["unit_id"],
                            cluster_id=world["cluster_id"],
                            arm=arm,
                            repeat=repeat,
                            size=world["size"],
                            population=world["population"],
                            seed=request["seed"],
                            denominator=1,
                            metric=int(result["decision"] != "insufficient"),
                            started_monotonic_ns=start,
                            ended_monotonic_ns=end,
                            complete_wall_time_s=(end - start) / 1e9,
                            response_bytes=response_bytes,
                        )
                    )
        print(
            f"[exp8874-worker] completed={len(rows)} cluster={world['cluster_id']}",
            file=sys.stderr,
            flush=True,
        )
    imports = []
    for name in (
        "_indexed_witness_core",
        "_indexed_witness_meter",
        "indexed_witness",
        "size_aware_witness",
    ):
        path = Path(sys.modules["carnot.verify." + name].__file__ or "")
        imports.append(
            dict(path=str(path), sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
        )
    json.dump(
        dict(
            rows=rows,
            installed_imports=imports,
            startup_and_request_s=(time.monotonic_ns() - started) / 1e9,
        ),
        sys.stdout,
    )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
