"""REQ-VERIFY-8565: actual script-path children cover private publication and rejection."""

import json
from pathlib import Path
import sys

from carnot.reporting import v736_contract_methods as e
from carnot.reporting import v736_contract_runner as runner
from carnot.reporting.v709_execution import child


def test_private_cli(tmp_path):
    """SCENARIO-VERIFY-8565-CONTROLS: real publication and cold children retain terminal logs."""
    output = tmp_path / "results" / (e.NAME + ".json")
    common = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    r = child(
        "private",
        common
        + [
            "--date",
            "20261016",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
            "--private-control",
        ],
        tmp_path / "logs",
        deadline=120,
    )
    assert r["passed"], Path(r["stderr_path"]).read_text()
    v = json.loads(output.read_bytes())
    assert v["verdict_class"] == "blocked"
    from carnot.reporting.primary_publication import read_bound_sidecar

    assert read_bound_sidecar(output, Path(v["terminal_validation_sidecar_path"]))[
        "primary_sha256"
    ] == e.sha256_file(output)
    assert child("cold", common + ["--cold-replay", str(output)], tmp_path / "logs", deadline=120)[
        "passed"
    ]
    v["completed_count"] += 1
    v["reproducibility_checksum"] = e.checksum(v)
    e.atomic_json(output, v)
    assert child(
        "tamper",
        common + ["--cold-replay", str(output)],
        tmp_path / "logs",
        deadline=120,
        expected=1,
    )["passed"]
    assert child("date", common + ["--date", "bad"], tmp_path / "logs", deadline=30, expected=2)[
        "passed"
    ]
    assert child(
        "owned", common + ["--deliberate-error"], tmp_path / "logs", deadline=30, expected=7
    )["passed"]
    assert child(
        "unsafe", common + ["--private-control"], tmp_path / "logs", deadline=30, expected=2
    )["passed"]


def test_validation_plan(tmp_path, monkeypatch):
    """REQ-VERIFY-8565: only scoped checks and exact original node unions are authorized."""
    monkeypatch.setattr(
        runner.collection,
        "collect",
        lambda private, suite: (
            [suite + "::private"],
            dict(name="collect_" + Path(suite).stem, argv=[]),
        ),
    )
    plan = runner.manifest(tmp_path)
    assert {"ruff_check", "ruff_format", "strict_mypy", "spec_coverage", "coverage_report"} <= {
        r["name"] for r in plan
    }
    assert all("tests/python" not in r["argv"] for r in plan)


def test_validation_scratch_rejects_checkout():
    """SCENARIO-VERIFY-8565-OWNED: child scratch must not collide with guarded result paths."""
    import pytest

    with pytest.raises(ValueError, match="private"):
        runner.manifest(e.ROOT / "results/raw/forbidden-validation-scratch")
