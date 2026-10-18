"""REQ-REPORT-8584: batch frames reuse the qualified complete numerical evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

from carnot.reporting import resident_worker_8572 as w
from carnot.reporting.complete_native_worker_8559 import batch

Json = dict[str, Any]


def serve(path: Path, digest: str, arm: str) -> int:
    """Load immutable operands once; recompute all text features for every received batch."""
    try:
        config, native, loaded = w.configuration(path, digest, arm)
        w.write_frame(
            1, dict(status="ready", config_hash=digest, loaded_extension=loaded, config_loads=1)
        )
        seen: set[str] = set()
        for index in range(1025):
            value = json.loads(w.read_frame(0, 30) or b"null")
            if not isinstance(value, dict) or value.get("config_hash") != digest:
                raise ValueError("request_configuration")
            identifier = value["request_id"]
            if identifier in seen or time.monotonic_ns() >= value["deadline_ns"]:
                raise ValueError("duplicate_or_expired_request")
            seen.add(identifier)
            if value.get("operation") == "shutdown":
                w.write_frame(
                    1,
                    dict(
                        status="shutdown",
                        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    ),
                )
                return 0
            rows = value["row"]["rows"]
            if len(rows) not in {1, 8}:
                raise ValueError("batch_size")
            responses, spans = batch(config["head"], rows, native)
            w.write_frame(
                1, dict(status="ok", request_id=identifier, responses=responses, spans=spans)
            )
            if (index + 1) % 64 == 0:
                print(f"[exp8584] batches_completed={index + 1}", file=sys.stderr, flush=True)
        raise ValueError("request_limit")
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, ImportError) as error:
        print(f"[exp8584] worker_error={error}", file=sys.stderr, flush=True)
        return 1
