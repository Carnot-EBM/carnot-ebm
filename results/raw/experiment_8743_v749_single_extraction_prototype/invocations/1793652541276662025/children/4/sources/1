"""REQ-VERIFY-8743: count extraction calls in the actual installed service.

The separate sidecar leaves wire decisions and errors unchanged. Failed
extractions never contribute a successful vector to a later request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from . import resident_protocol as wire


def main(argv: list[str] | None = None) -> int:
    """Restore instrumentation even when a malformed frame terminates the service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("digest")
    parser.add_argument("arm", choices=["python", "native"])
    parser.add_argument("--counts-output", type=Path, required=True)
    parser.add_argument("--idle-timeout", type=float, default=2)
    args = parser.parse_args(argv)
    before = list(sys.path)
    forbidden = [
        n for n in sys.modules if n.startswith(("carnot.training", "carnot.reporting", "jax"))
    ]
    assert not forbidden and wire.MAX_FRAME == 65536 and wire.MAX_REQUESTS == 1025
    original = wire.numeric.extract
    extractions = []

    def extract(row: Any) -> Any:
        """Count attempts before validation and mark success only after original extraction."""
        record = dict(public_hash=wire.numeric.canonical_hash(row), successful=False)
        extractions.append(record)
        feature = original(row)
        record.update(successful=True, feature_hash=feature["feature_hash"])
        return feature

    wire.numeric.extract = extract
    try:
        result = wire.serve(args.config, args.digest, args.arm, args.idle_timeout)
    finally:
        wire.numeric.extract = original
    assert sys.path == before
    args.counts_output.write_text(
        json.dumps(
            dict(extractions=extractions, actual_exit=result, forbidden_imports=forbidden),
            sort_keys=True,
        )
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
