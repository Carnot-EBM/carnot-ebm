"""REQ-REPORT-8846: sealed operands, clustered inference and private CLI custody."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import future_benefit_audit_8846 as r
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    """Use preserved producer bytes so a fixture cannot invent scientific inputs."""
    upstream = r.load(dict(path=str(r.SOURCE), sha256=r.PIN))
    return r.load(upstream["raw_shard_hashes"][0])


def test_independent_stream_reduction(
    source: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8846-CONTRACT: independently reconstruct the complete census."""
    result = r.reduce(source)
    assert len(result["rows"]) == 12096
    assert len(result["causal_integrity_rows"]) == 144
    assert len(result["per_stream_results"]) == 144
    assert len(result["paired_interval_rows"]) == 5
    assert len(result["influence_rows"]) == 120
    assert all(x["feedback_ordinal"] < x["use_ordinal"] for x in result["template_reuse_rows"])
    assert all(
        x["draws"] == 10000 and x["tail_probability"] == 0.001
        for x in result["paired_interval_rows"]
    )
    assert result["independent_count"] == 24
    assert result["next_head_recommendation"]["read_only"]
    upstream = r.load(dict(path=str(r.SOURCE), sha256=r.PIN))
    candidate = tmp_path / "candidate.json"
    value = dict(
        result,
        verdict_class="null",
        raw_shard_hashes=upstream["raw_shard_hashes"],
        source_artifact_hashes=[r.reference(r.SOURCE)],
        code_config_hashes=[],
        validation_receipts=[],
    )
    atomic_json(candidate, value)
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    child = subprocess.run(
        [r.sys.executable, "-u", str(r.ROOT / r.CLI), "--replay", str(candidate)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert child.returncode == 0 and '"replay_passed": true' in child.stdout, (
        child.stdout + child.stderr
    )
    monkeypatch.setattr(r, "reduce", lambda _: deepcopy(result))
    value["verdict_class"] = "disqualified"
    value.update({k: 0 for k in result if k.endswith("_score")})
    atomic_json(candidate, value)
    assert r.replay(candidate)["passed"]
    evidence = tmp_path / "changed.json"
    atomic_json(evidence, dict(changed=True))
    value["raw_shard_hashes"] = [r.reference(evidence)]
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="original_evidence_hash"):
        r.replay(candidate)


@pytest.mark.parametrize(
    "mutation", ["missing", "early", "truth", "retention", "seal", "use", "restart", "gradient"]
)
def test_integrity_rejections(source: dict[str, Any], mutation: str) -> None:
    """REQ-REPORT-8846: sealed state and independent truth reject internal drift."""
    bad = deepcopy(source)
    if mutation == "missing":
        bad["per_event_results"].pop()
    elif mutation == "early":
        bad["label_release_rows"][0]["feedback_ordinal"] -= 1
    elif mutation == "truth":
        bad["per_event_results"][0]["truth"] = "support"
    elif mutation == "retention":
        bad["retention_rows"][0]["learning_state_unchanged"] = False
    elif mutation == "seal":
        bad["per_event_results"][0]["sealed"]["prediction"][0] = 0
    elif mutation == "use":
        bad["template_use_rows"][0]["template_uses"] = ["unadmitted"]
    elif mutation == "restart":
        bad["restart_rows"][0]["after"] = "changed"
    else:
        bad["gradient_rows"][0]["parameters_before"] = {}
    with pytest.raises(ValueError):
        r.reduce(bad)


def test_intervals_and_gates() -> None:
    """REQ-REPORT-8846: frozen margins distinguish calibration, addition and retention."""
    rows = [dict(comparison=str(i), estimate=-0.02, lower=-0.03, upper=-0.011) for i in range(5)]
    retained = [dict(upper=0.005)]
    gates = r.gates(rows, retained)
    assert gates["causal_benefit_score"] == gates["addition_benefit_score"] == 1
    rows[1]["upper"] = -0.001
    assert r.gates(rows, retained)["addition_benefit_score"] == 0
    rows[2]["upper"] = 0.001
    rows[4]["upper"] = 0.006
    retained[0]["upper"] = 0.02
    gates = r.gates(rows, retained)
    assert (
        gates["causal_benefit_score"]
        == gates["retention_pass_score"]
        == gates["categorical_noninferiority_score"]
        == 0
    )


def test_preconditions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8846: authenticate real authorities; absent input cannot become a fixture."""
    source, checked, refs = r.preflight(tmp_path / "private", r.SOURCE)
    assert source["integrity"]["online"] == 6912 and checked["time_budget_s"] == 4200
    assert refs and checked["dependencies"]["numpy"]
    with pytest.raises(FileNotFoundError):
        r.preflight(tmp_path / "missing", tmp_path / "absent.json")
    with patch.object(r.sys, "prefix", "/tmp/wrong-venv"), pytest.raises(RuntimeError):
        r.preflight(tmp_path / "env", r.SOURCE)
    monkeypatch.setattr(r, "load", lambda ref: dict(stream_rows_ready_score=0))
    with pytest.raises(PermissionError):
        r.preflight(tmp_path / "gate", r.SOURCE)


def test_replay_rehash_and_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8846: recomputation defeats repaired outer aggregate hashes."""
    evidence = tmp_path / "evidence.json"
    atomic_json(evidence, {})
    monkeypatch.setattr(r, "reduce", lambda _: dict(rows=[], learning_audit_ready_score=1))
    value = dict(
        raw_shard_hashes=[r.reference(evidence)],
        source_artifact_hashes=[],
        code_config_hashes=[],
        validation_receipts=[],
        rows=[],
        learning_audit_ready_score=1,
    )
    log = tmp_path / "validation.log"
    log.write_text("private test log")
    value["validation_receipts"] = [dict(log_path=str(log), log_sha256=r.reference(log)["sha256"])]
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert r.replay(candidate)["replay_passed"]
    value["rows"] = [dict(metric=123)]
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(candidate)
    value["code_config_hashes"] = [dict(path=str(evidence), sha256="changed")]
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="operand_hash"):
        r.replay(candidate)


def test_cli_date_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8846-CONTRACT: direct CLI imports outside the checkout."""
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for args, marker in [
        (["--date", "wrong"], "date_rejected"),
        (["--replay", str(tmp_path / "absent")], "replay_rejected"),
    ]:
        child = subprocess.run(
            [r.sys.executable, "-u", str(r.ROOT / r.CLI), *args],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert child.returncode == 1 and marker in child.stdout


@pytest.mark.parametrize(
    "failure",
    [
        "",
        "null",
        "missing",
        "gate",
        "rejected",
        "field",
        "integrity",
        "owned",
        "terminal",
        "deadline",
    ],
)
def test_runner_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """REQ-REPORT-8846: completion, external blocking and owned disqualification differ."""
    from carnot.reporting import future_benefit_runner_8846 as runner

    minimal = dict(
        rows=[],
        independent_count=24,
        intended_count=12096,
        completed_count=12096,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        learning_audit_ready_score=1,
        causal_benefit_score=1,
        addition_benefit_score=1,
        retention_pass_score=1,
        categorical_noninferiority_score=1,
        per_stream_results=[],
        paired_interval_rows=[],
        retention_interval_rows=[],
        causal_integrity_rows=[],
        untouched_retention_rows=[],
        template_reuse_rows=[],
        influence_rows=[],
        subgroup_descriptives=[],
        next_head_recommendation={},
    )

    def preflight(raw: Path, path: Path) -> Any:
        raw.mkdir(parents=True, exist_ok=True)
        if failure in {"missing", "gate", "rejected", "field", "integrity"}:
            raise {
                "missing": FileNotFoundError("absent"),
                "gate": PermissionError("observed_zero"),
                "rejected": PermissionError("rejected_producer"),
                "field": KeyError("authority"),
                "integrity": ValueError("drift"),
            }[failure]
        return {}, {}, []

    monkeypatch.setattr(r, "preflight", preflight)
    if failure == "null":
        minimal["causal_benefit_score"] = 0
    monkeypatch.setattr(r, "reduce", lambda _: deepcopy(minimal))
    monkeypatch.setattr(
        runner,
        "validate",
        lambda _: dict(
            passed=failure != "owned", receipts=[], coverage=dict(complete=failure != "owned")
        ),
    )
    monkeypatch.setattr(
        runner, "terminal", lambda *a: dict(passed=failure != "terminal", receipts=[])
    )
    output = tmp_path / (r.NAME + ".json")
    args = ["--private-output", str(output), "--raw-dir", str(tmp_path / "raw")]
    if failure == "deadline":
        monkeypatch.setattr(runner, "BUDGET", -1)
    assert runner.main(args) == 0
    value = json.loads(output.read_text())
    assert value["honest_verdict"].startswith("complete_")
    expected = (
        "blocked"
        if failure in {"missing", "gate", "rejected", "field"}
        else "null"
        if failure == "null"
        else "disqualified"
        if failure
        else "circular_positive"
    )
    assert value["verdict_class"] == expected
    if failure and failure != "null":
        assert value["learning_audit_ready_score"] == 0
    else:
        assert runner.main(["--replay", str(output)]) == 0
    assert runner.main(["--date", "bad"]) == 1
    assert runner.main(["--replay", str(tmp_path / "absent")]) == 1


def test_runner_delegates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8846: owned validation uses the existing bounded child supervisor."""
    from carnot.reporting import future_benefit_runner_8846 as runner

    monkeypatch.setattr(runner.validation, "validate", lambda _: dict(passed=True))
    monkeypatch.setattr(runner.validation, "terminal", lambda *a: dict(passed=True))
    assert runner.validate(tmp_path)["passed"]
    assert runner.terminal(tmp_path / "candidate", tmp_path)["passed"]
