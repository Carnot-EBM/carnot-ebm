"""REQ-REPORT-8733: opt-in local spans around the unchanged installed service.

The profiler writes a separate file after shutdown. Successful and rejected wire
responses therefore keep the original service format and 65536-byte limit.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

from . import resident_batch as batch
from . import resident_protocol as wire

Json = dict[str, Any]


class Recorder:
    """Pause the enclosing span while a nested operation owns the local clock."""

    def __init__(self) -> None:
        self.stack: list[Json] = []
        self.spans: list[Json] = []

    def call(self, operation: str, function: Callable[..., Any], *args: Any, **kw: Any) -> Any:
        """Finally restores the caller even when the original operation rejects input."""
        start = time.monotonic_ns()
        if self.stack:
            parent = self.stack[-1]
            self.spans.append(
                dict(operation=parent["operation"], start_ns=parent["cursor"], end_ns=start)
            )
        frame = dict(operation=operation, cursor=start)
        self.stack.append(frame)
        try:
            return function(*args, **kw)
        finally:
            end = time.monotonic_ns()
            self.spans.append(dict(operation=operation, start_ns=frame["cursor"], end_ns=end))
            self.stack.pop()
            if self.stack:
                self.stack[-1]["cursor"] = end


def install(recorder: Recorder) -> None:
    """Wrap function boundaries without replacing validation or numerical assertions."""
    numeric = wire.numeric
    for owner, name, label in [
        (wire, "read_frame", "transport_read"),
        (wire, "write_frame", "transport_write"),
        (wire, "sha256_file", "configuration_validation"),
        (wire, "configuration", "configuration"),
        (wire, "decide", "protocol_validation"),
        (batch, "evaluate", "batch_validation"),
        (numeric, "batch", "kernel_and_numerical_scoring"),
    ]:
        original = getattr(owner, name)

        def wrapped(*args: Any, _fn: Any = original, _label: str = label, **kw: Any) -> Any:
            return recorder.call(_label, _fn, *args, **kw)

        setattr(owner, name, wrapped)
    extract = numeric.extract

    def features(*args: Any, **kw: Any) -> Any:
        label = (
            "scoring_feature_extraction"
            if any(f["operation"] == "kernel_and_numerical_scoring" for f in recorder.stack)
            else "duplicate_feature_digest_extraction"
        )
        return recorder.call(label, extract, *args, **kw)

    numeric.extract = features
    for owner in (wire, batch, numeric):

        def dumps(*args: Any, **kw: Any) -> Any:
            return recorder.call("serialization", json.dumps, *args, **kw)

        def loads(*args: Any, **kw: Any) -> Any:
            return recorder.call("deserialization", json.loads, *args, **kw)

        owner.json = SimpleNamespace(dumps=dumps, loads=loads)


def main(argv: list[str] | None = None) -> int:
    """Both modes use the same installed imports; timing is an explicit opt-in."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("digest")
    parser.add_argument("arm", choices=["python", "native"])
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--instrument", action="store_true")
    args = parser.parse_args(argv)
    recorder = Recorder()
    before = list(sys.path)
    forbidden = [
        n for n in sys.modules if n.startswith(("carnot.training", "carnot.reporting", "jax"))
    ]
    assert not forbidden and wire.MAX_FRAME == 65536 and wire.MAX_REQUESTS == 1025
    if args.instrument:
        install(recorder)
    start = time.monotonic_ns()
    result = wire.serve(args.config, args.digest, args.arm)
    end = time.monotonic_ns()
    assert sys.path == before
    args.profile_output.write_text(
        json.dumps(
            dict(
                start_ns=start,
                end_ns=end,
                spans=recorder.spans,
                instrumented=args.instrument,
                actual_exit=result,
                forbidden_imports=forbidden,
            )
        )
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
