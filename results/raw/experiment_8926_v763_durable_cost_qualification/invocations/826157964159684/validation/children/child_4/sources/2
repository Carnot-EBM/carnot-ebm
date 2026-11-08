"""REQ-REPORT-8926 / SCENARIO-REPORT-8926-CONTRACT: qualify retained costs."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import durable_cost_qualification_8926 as m
from scripts import verdict_row_consistency_lint as lint


@pytest.fixture(scope="module")
def retained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Authenticate actual historical work; fixtures do not claim a new benchmark."""
    raw = tmp_path_factory.mktemp("retained8926")
    pre = m.preconditions(raw, m.ROOT / "results")
    assert pre["external_ready"], pre["diagnostics"]
    return dict(raw=raw, pre=pre, result=m.reduce_retained(raw, pre))


def test_retained_costs(retained: dict[str, Any]) -> None:
    """REQ-REPORT-8926: preserve48 cells and all durable work without process scalars."""
    value = retained["result"]
    assert value["completed_count"] == 48 and value["independent_count"] == 12
    assert len(value["feedback_cost_rows"]) == 48 * 48
    assert len(value["durability_rows"]) == 48 * 97
    assert all(r["feedback_cursor"] == 24 for r in value["restart_rows"])
    assert all(not r["additive_partition"] for r in value["inclusive_timer_rows"])
    assert len(value["process_receipt_references"]) == 48
    assert all("control_receipt" not in r and "commands" not in r for r in value["rows"])
    assert value["rows"] == value["comparison_rows"]
    path = retained["raw"] / "typed.json"
    m.atomic_json(path, dict(value, honest_verdict="complete_null_historical_costs"))
    assert lint.check_artifact(path) == ("ok", [])
    m.check_comparisons(value, value)


@pytest.mark.parametrize(
    "fault", ["missing_link", "changed_time", "equal_arms", "missing_cell", "swapped_arms"]
)
def test_comparison_boundary(retained: dict[str, Any], fault: str) -> None:
    """SCENARIO-REPORT-8926-CONTRACT: administrative repair must not hide bad science."""
    original = retained["result"]
    bad = deepcopy(original)
    if fault == "missing_link":
        bad["process_receipt_references"].pop()
    elif fault == "changed_time":
        bad["rows"][0]["metric"] += 1
    elif fault == "equal_arms":
        bad["honest_verdict"] = "complete_positive_improved"
        for row in bad["rows"]:
            row["baseline_cost"] = row["control_cost"] = row["metric"]
    elif fault == "missing_cell":
        bad["rows"].pop()
    else:
        bad["rows"][0]["arm"], bad["rows"][1]["arm"] = (
            bad["rows"][1]["arm"],
            bad["rows"][0]["arm"],
        )
    with pytest.raises(ValueError):
        m.check_comparisons(bad, original)


def test_strict_lint_still_rejects_equal_controls(tmp_path: Path) -> None:
    """REQ-REPORT-8926: real degenerate controls still fail the unchanged scanner."""
    path = tmp_path / "bad.json"
    m.atomic_json(
        path,
        dict(
            honest_verdict="complete_positive_improved",
            rows=[
                dict(unit_id=str(i), baseline_cost=100 + i, control_cost=100 + i) for i in range(12)
            ],
        ),
    )
    assert lint.main(["--strict", str(path)]) == 1


def test_missing_external_bytes(tmp_path: Path) -> None:
    """REQ-REPORT-8926: absent upstream bytes name the exact terminal external gate."""
    pre = m.preconditions(tmp_path / "raw", tmp_path)
    assert not pre["external_ready"]
    assert pre["diagnostics"][0]["check"] == "missing_file"
    assert {"check", "upstream", "artifact_field", "expected", "observed", "path", "hash"} <= pre[
        "diagnostics"
    ][0].keys()


def test_changed_original_bytes(tmp_path: Path) -> None:
    """REQ-REPORT-8926: a replacement primary cannot authenticate historical work."""
    (tmp_path / (m.SOURCE + ".json")).write_text("{}")
    assert (
        m.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"]
        == "rejected_producer"
    )


def validation_fixture() -> dict[str, Any]:
    """Use explicit test-only readiness without claiming a production acceptance result."""
    return dict(
        coverage=dict(
            complete=True,
            percent=100,
            files=[dict(path=p, missing=[], excluded=[]) for p in m.OWNED],
        ),
        union=dict(passed=True),
        receipts=[dict(name="unit_fixture", passed=True)],
        raw_references=[],
    )


def test_preserved_receipts(
    retained: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8926: immutable copies retain original source hashes and process bytes."""
    pre, raw = retained["pre"], retained["raw"]
    assert m.reduce_retained(raw, pre) == retained["result"]
    missing = m.preconditions(tmp_path / "missing", m.ROOT / "results")
    assert missing["external_ready"]
    target = Path(missing["original_primary"]["path"])
    target.write_text("changed")
    assert not m.preconditions(tmp_path / "missing", m.ROOT / "results")["external_ready"]
    monkeypatch.setattr(m, "ARCHIVE", tmp_path / "unavailable")
    absent = m.preconditions(tmp_path / "absent", m.ROOT / "results")
    assert not absent["external_ready"] and absent["diagnostics"][0]["check"] == "missing_file"


@pytest.mark.parametrize("payload", ["{}", "[]", "{"])
def test_malformed_producer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: str) -> None:
    """REQ-REPORT-8926: malformed producer bytes cannot become qualification evidence."""
    path = tmp_path / (m.SOURCE + ".json")
    path.write_text(payload)
    monkeypatch.setattr(m, "PIN", m.sha256_file(path))
    assert not m.preconditions(tmp_path / "raw", tmp_path)["external_ready"]


def test_wrappers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8926: wrappers retain private E2E and scanner execution reports."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        m,
        "namespace",
        lambda module, name: SimpleNamespace(
            validate=lambda raw, private: validation_fixture(),
            terminal=lambda candidate, raw: [dict(passed=True)],
        ),
    )
    monkeypatch.setattr(m.previous, "e2e018", lambda raw, private: dict(passed=True))
    assert m.validate(tmp_path, tmp_path)["receipts"][-1]["passed"]
    assert m.terminal(tmp_path / "candidate.json", tmp_path)[0]["passed"]


@pytest.fixture(scope="module")
def candidate(retained: dict[str, Any]) -> Path:
    """Produce a private complete historical reduction; no benchmark is executed."""
    from unittest.mock import patch

    path = retained["raw"] / (m.NAME + ".json")
    with (
        patch.object(m, "validate", return_value=validation_fixture()),
        patch.object(m, "terminal", return_value=[]),
    ):
        assert m.main(["--output", str(path)]) == 0
    return path


def test_cold_replay(candidate: Path, tmp_path: Path) -> None:
    """SCENARIO-REPORT-8926-CONTRACT: actual foreign-directory CLI accepts historical bytes."""
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-u", str(m.ROOT / m.CLI), "--replay", str(candidate)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "replay_passed" in result.stdout
    assert m.main(["--replay", str(tmp_path / "missing")]) == 1
    assert m.main(["--replay", str(candidate)]) == 0
    with pytest.raises(SystemExit):
        m.main(["--date", "20260101"])
    with pytest.raises(SystemExit):
        m.main(["--private-output", str(m.ROOT / "forbidden.json")])


@pytest.mark.parametrize(
    "fault", ["metric", "readiness", "identity", "duration", "speed", "manifest", "link"]
)
def test_rehashed_tampering(candidate: Path, tmp_path: Path, fault: str) -> None:
    """REQ-REPORT-8926: rehashing false headlines or swapped provenance cannot pass."""
    value = json.loads(candidate.read_bytes())
    if fault == "metric":
        value["rows"][0]["metric"] += 1
    elif fault == "readiness":
        value["adaptation_cost_ready_score"] = 0
    elif fault == "identity":
        value["task_id"] = "wrong"
    elif fault == "duration":
        value["duration_s"] = 0
    elif fault == "speed":
        value["speed_quality_recommendation_score"] = 1
    elif fault == "manifest":
        value["original_receipt_manifest"][0]["original"]["sha256"] = "changed"
    else:
        value["process_receipt_references"][0]["sha256"] = "changed"
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = m.canonical_hash(value)
    path = tmp_path / "bad.json"
    m.atomic_json(path, value)
    with pytest.raises(ValueError):
        m.replay(path)


def test_external_cli_block(tmp_path: Path) -> None:
    """REQ-REPORT-8926: actual private publication records external absence as blocked."""
    output = tmp_path / (m.NAME + ".json")
    assert m.main(["--private-output", str(output), "--sources", str(tmp_path)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    assert m.replay(output)["passed"]


def test_failed_owned_validation(
    candidate: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8926: failed owned controls disqualify complete historical measurements."""
    monkeypatch.setattr(m, "validate", lambda raw, private: m.unqualified())
    monkeypatch.setattr(m, "terminal", lambda path, raw: [])
    output = tmp_path / (m.NAME + ".json")
    assert m.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"


def test_historical_validation_source_view(tmp_path: Path) -> None:
    """REQ-REPORT-8926: original tests read authentic archived code without replacing live files."""
    import sys

    path = m.ROOT / "python/carnot/memory/guarded_memory_8905.py"
    before = m.sha256_file(path)
    code = (
        "from pathlib import Path; import hashlib; print(hashlib.sha256(Path("
        + repr(str(path))
        + ").read_bytes()).hexdigest(),flush=True)"
    )
    report = m.validation_child(
        "child_legacy",
        [
            sys.executable,
            "-u",
            "-c",
            code,
            "tests/python/test_guarded_durable_cost_8914.py::test_roster",
        ],
        tmp_path,
        deadline=30,
    )
    assert report["passed"]
    assert (
        "379c78d6c15c75ca6e4410bf1b08a8c65b9276f27c562cd23cd6b8faf1dd6033"
        in Path(report["stdout_path"]).read_text()
    )
    assert m.sha256_file(path) == before
    assert m.validation_child("ordinary", ["true"], tmp_path, deadline=30)["passed"]
