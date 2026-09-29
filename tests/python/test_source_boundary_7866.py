"""REQ-REPORT-7866: current public source boundary qualification."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import source_boundary_7866 as boundary


SCRIPT = "scripts/experiments/experiment_7866_v683_source_boundary.py"


def _fixture(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _cli(public: Path, output: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            sys.executable,
            "-u",
            SCRIPT,
            "--fixture-public",
            str(public),
            "--fixture-output",
            str(output),
        ],
        capture_output=True,
        check=False,
    )


def test_public_cli_success_with_spaces_and_direct_import(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7866-CLI: real script accepts private public bytes."""
    root = tmp_path / "fixture with spaces"
    root.mkdir()
    public, output = root / "public.jsonl", root / "features.jsonl"
    row = {"family_id": "one", "source_bytes": b"A. B.".hex(), "answer_bytes": b"A.".hex()}
    _fixture(public, [row])
    assert _cli(public, output).returncode == 0
    assert output.is_file()
    assert boundary.qualified_imports() == {
        "carnot.verify.source_projection": str(boundary.source_projection.__file__),
        "carnot.verify.source_alignment": str(boundary.source_alignment.__file__),
    }


@pytest.mark.parametrize("fault", ["missing", "label", "duplicate"])
def test_public_cli_rejects_invalid_source(tmp_path: Path, fault: str) -> None:
    """SCENARIO-REPORT-7866-CLI: missing, labeled, or repeated sources fail."""
    public, output = tmp_path / "public.jsonl", tmp_path / "features.jsonl"
    row = {"family_id": "one", "source_bytes": b"A.".hex(), "answer_bytes": b"B.".hex()}
    if fault != "missing":
        _fixture(public, [{**row, "label": "bad"}] if fault == "label" else [row, row])
    assert _cli(public, output).returncode != 0


def test_alias_import_receipt_is_rejected() -> None:
    """SCENARIO-REPORT-7866-CLI: the old short aliases cannot qualify imports."""
    resolved = boundary.qualified_imports()
    assert boundary.imports_valid(resolved)
    assert not boundary.imports_valid({"projection": resolved["carnot.verify.source_projection"]})


def test_duplicate_grouping_and_budget() -> None:
    """SCENARIO-REPORT-7866-CUSTODY: normalized copies share one family group."""
    assert boundary.group_key(" One  TWO ", "Answer") == boundary.group_key("one two", " answer ")
    rows = [
        {"family_id": "a", "role": "fit", "status": "completed", "source_group": "g"},
        {"family_id": "b", "role": "tune", "status": "excluded", "source_group": "h"},
    ]
    assert boundary.reduce_budget(rows, 2) == {
        "intended": 2,
        "eligible": 1,
        "started": 2,
        "completed": 1,
        "censored": 0,
        "excluded": 1,
        "independent": 2,
    }
    with pytest.raises(ValueError, match="duplicate_group_role_leakage"):
        boundary.check_role_groups(
            [{**rows[0], "source_group": "same"}, {**rows[1], "source_group": "same"}]
        )
