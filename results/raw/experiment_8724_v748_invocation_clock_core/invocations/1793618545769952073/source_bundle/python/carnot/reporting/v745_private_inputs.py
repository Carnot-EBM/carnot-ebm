"""REQ-VERIFY-8681: replay the exact historical ARC input without reverting its live repair."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import v744_contract_methods as original

ROOT = original.ROOT
ARCHIVE, SOURCE, PIN = original.ARC_ARCHIVE, original.ARC_SOURCE, original.ARC_PIN
CLI = "scripts/experiments/experiment_8681_original_private.py"
CHECKED = original.checked


def checked(ref: dict[str, Any]) -> bool:
    """Substitute only the declared old input after checking the retained bytes themselves."""
    if ref == dict(path=str(SOURCE), sha256=PIN):
        return bool(CHECKED(dict(path=str(ARCHIVE), sha256=PIN)))
    return bool(CHECKED(ref))


@contextmanager
def binding() -> Iterator[None]:
    """Restore the original producer immediately after a private historical read."""
    with patch.multiple(original, checked=checked, CLI=CLI):
        yield


def pytest_configure(config: Any) -> None:
    """Keep unchanged original nodes and assertions under an explicit private input binding."""
    config._v745_original_binding = binding()
    config._v745_original_binding.__enter__()


def pytest_unconfigure(config: Any) -> None:
    """Release the private substitution without editing any shared or historical file."""
    config._v745_original_binding.__exit__(None, None, None)


def main(argv: list[str] | None = None) -> int:
    """The original CLI can only publish private controls or replay existing evidence here."""
    from carnot.reporting.v744_contract_runner import main as run

    args = list(sys.argv[1:] if argv is None else argv)
    if (
        "--cold-replay" not in args
        and "--deliberate-error" not in args
        and "--private-control" not in args
    ):
        args.append("--private-control")
    with binding():
        return int(run(args))
