"""Pure checks for measured source custody (REQ-REPORT-7880)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from coverage import CoverageData


def policy_subroles(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Freeze policy use from public identity before an evaluator opens labels."""
    ids = [row["family_id"] for row in rows if row["role"] == "policy"]
    if len(ids) != 64 or len(set(ids)) != 64:
        raise ValueError("policy_family_count")
    ranked = sorted(
        ids, key=lambda family: hashlib.sha256(f"v684-policy:{family}".encode()).digest()
    )
    return {
        family: "policy_design" if index < 32 else "calibration_replay"
        for index, family in enumerate(ranked)
    }


def measured_counts(data_path: Path, required_paths: list[Path]) -> dict[str, int]:
    """Require real executed statements from every explicit file in one coverage run."""
    data = CoverageData(basename=str(data_path))
    if not data_path.is_file():
        raise ValueError("coverage_missing_statements")
    data.read()
    measured = {
        str(Path(path).resolve()): len(data.lines(path) or ()) for path in data.measured_files()
    }
    expected = {
        str(path.resolve()): measured.get(str(path.resolve()), 0) for path in required_paths
    }
    if not expected or any(count == 0 for count in expected.values()):
        raise ValueError("coverage_missing_statements")
    return expected


def valid_venue(value: str) -> bool:
    """Keep the venue in the experiment artifact's closed vocabulary."""
    return value in {"host", "gatemate", "kv260", "polarfire"}
