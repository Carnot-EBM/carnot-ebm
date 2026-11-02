"""REQ-VERIFY-8622: identical resident framing surrounds unchanged arithmetic."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def main(argv: list[str] | None = None) -> int:
    """Import boundaries are measured inside the fresh interpreter, before either graph."""
    args = sys.argv[1:] if argv is None else argv
    began = time.monotonic_ns()
    try:
        path, digest, label = args
        if label not in {"lean_python", "lean_native", "legacy_python", "legacy_native"}:
            raise ValueError("worker_arm")
        root = Path(__file__).resolve().parents[3]
        sys.path[:0] = [str(root / "python/carnot/inference"), str(root / "python"), str(root)]
        print(f"[8622] before_imports {label}", file=sys.stderr, flush=True)
        if label.startswith("lean"):
            wire = importlib.import_module("resident_protocol")
            numeric = importlib.import_module("resident_energy")
        else:
            wire = importlib.import_module("carnot.reporting.resident_worker_8572")
            numeric = importlib.import_module("carnot.reporting.complete_native_worker_8559")
        imported = time.monotonic_ns()
        print(f"[8622] after_imports {label}", file=sys.stderr, flush=True)
        config, native, loaded = wire.configuration(Path(path), digest, label.split("_")[-1])
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        wire.write_frame(
            1,
            dict(
                status="ready",
                imports_start_ns=began,
                imports_end_ns=imported,
                config_hash=digest,
                config_loads=1,
                loaded_extension=loaded,
                ready_rss_kib=rss,
            ),
        )
        seen: set[str] = set()
        for index in range(1025):
            value = json.loads(wire.read_frame(0, 30) or b"null")
            if (
                not isinstance(value, dict)
                or value.get("config_hash") != digest
                or value["request_id"] in seen
                or time.monotonic_ns() >= value["deadline_ns"]
            ):
                raise ValueError("request_identity_or_deadline")
            seen.add(value["request_id"])
            if value["operation"] == "shutdown":
                wire.write_frame(
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
            responses, spans = numeric.batch(config["head"], rows, native)
            wire.write_frame(
                1,
                dict(status="ok", request_id=value["request_id"], responses=responses, spans=spans),
            )
            if (index + 1) % 64 == 0:
                print(
                    f"[8622] completed={index + 1} pending={1024 - index}",
                    file=sys.stderr,
                    flush=True,
                )
        raise ValueError("request_limit")
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, ImportError) as error:
        print(f"[8622] worker_error={error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
