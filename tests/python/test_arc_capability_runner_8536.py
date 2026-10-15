"""REQ-REPORT-8536: private CLI publication must bind exact terminal evidence."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import arc_capability_qualification_8536 as e
from carnot.reporting import arc_capability_runner_8536 as runner
from carnot.reporting.current_work_receipt import atomic_json


@pytest.mark.parametrize(
    "args,expected",
    [
        (["--date", "20261014"], 2),
        (["--deliberate-error"], 7),
        (["--cold-replay", "/does/not/exist"], 1),
    ],
)
def test_standalone_cli_rejections(tmp_path, args, expected):
    """SCENARIO-REPORT-8536-REPLAY: execute actual script outside the checkout."""
    result = subprocess.run(
        [sys.executable, str(e.ROOT / e.CLI), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == expected, result.stdout + result.stderr


@pytest.mark.parametrize("failed", [False, True])
def test_private_blocked_failure_publish_and_consumer(tmp_path, monkeypatch, failed):
    """REQ-REPORT-8536: actual validators and consumers see a private completed candidate."""
    monkeypatch.setattr(
        runner, "validate_checks", lambda *_: [dict(name="private", passed=not failed)]
    )
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("disqualified" if failed else "blocked")
    assert value["arc_capability_ready_score"] == 0
    assert runner.main(["--cold-replay", str(output)]) == 0
    side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert side["passed"] and side["consumer"]["passed"]
    assert side["consumer"]["gates"][0]["artifact_field"] == "arc_capability_ready_score"


def test_cli_malformed_and_owned_error(tmp_path):
    """SCENARIO-REPORT-8536-REPLAY: malformed input is refused; owned errors stay failures."""
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    assert runner.main(["--cold-replay", str(malformed)]) == 1
    assert runner.main(["--deliberate-error"]) == 7


def test_validation_adapter_preserves_collected_union(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8536-WRAPPER: keep original nodes and check added statements only."""
    calls = []

    def fake(name, argv, logs, **kwargs):
        calls.append((name, argv))
        logs.mkdir(parents=True, exist_ok=True)
        out = logs / (name + ".stdout")
        out.write_text(
            "tests/python/test_private.py::test_original\n" if name.startswith("collect") else ""
        )
        if name == "coverage_json":
            atomic_json(
                logs / "owned_coverage.json",
                dict(
                    files={
                        p: dict(
                            executed_lines=list(
                                range(1, len((e.ROOT / p).read_text().splitlines()) + 1)
                            ),
                            missing_lines=[],
                        )
                        for p in [e.POLICY, e.SUPERVISOR]
                    }
                ),
            )
        return dict(name=name, passed=True, stdout_path=str(out))

    monkeypatch.setattr(runner, "child", fake)
    rows = runner.validate_checks(tmp_path / "raw", tmp_path / "private")
    assert rows and all(r["passed"] for r in rows)
    assert rows[-1]["name"] == "policy_added_coverage"
    assert all("--no-cov" in a and "0" in a for n, a in calls if n.startswith("historical"))
    assert any(n == "coverage_report" and "--fail-under=100" in a for n, a in calls)
    union = json.loads((tmp_path / "raw/validation/historical_node_union.json").read_bytes())
    assert all(r["union_equals_collection"] and r["disjoint"] for r in union.values())
    monkeypatch.setattr(runner, "child", lambda *_a, **_k: dict(passed=False))
    assert all(
        not r["passed"] for r in runner.validate_checks(tmp_path / "failed", tmp_path / "scratch")
    )
