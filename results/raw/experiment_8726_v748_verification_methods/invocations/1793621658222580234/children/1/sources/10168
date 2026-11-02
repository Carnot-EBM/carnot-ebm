"""REQ-VERIFY-8608: bypass eager package imports for the private serving process."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/carnot/inference"))
from resident_protocol import serve


def main(argv: list[str] | None = None) -> int:
    """Explicit paths bind one immutable configuration to one bounded session."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("digest")
    parser.add_argument("arm", choices=["python", "native"])
    parser.add_argument("--idle-timeout", type=float, default=15)
    args = parser.parse_args(argv)
    return int(serve(args.config, args.digest, args.arm, args.idle_timeout))


if __name__ == "__main__":
    raise SystemExit(main())
