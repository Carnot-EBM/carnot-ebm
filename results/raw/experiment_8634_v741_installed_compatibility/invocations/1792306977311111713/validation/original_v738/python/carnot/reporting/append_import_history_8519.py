"""REQ-VERIFY-8519: private historical pytest reads retain their original authority."""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from carnot.reporting import v732_contract_methods as methods
from carnot.reporting.v721_capstone_evidence import frozen_inputs

_context: AbstractContextManager[None] | None = None


def pytest_sessionstart(session: Any) -> None:
    """Original gates must see their sealed task contract rather than a later roadmap."""
    global _context
    refs = []
    for name, original in zip(
        (methods.DESIGN, methods.ACTIVE),
        methods.protocol()["v730_authority_references"],
        strict=True,
    ):
        if methods.sha256_file(Path(original["path"])) != original["sha256"]:
            raise ValueError("original_authority_hash")
        refs.append(
            dict(
                path=str(methods.ROOT / name),
                snapshot_path=original["path"],
                exists=True,
                sha256=original["sha256"],
            )
        )
    _context = frozen_inputs(refs, Path(session.config.option.basetemp) / "authority_writes")
    _context.__enter__()


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Release only this process's read adapter so current authority stays unchanged."""
    global _context
    if _context is not None:
        _context.__exit__(None, None, None)
        _context = None
