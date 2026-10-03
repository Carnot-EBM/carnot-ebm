"""REQ-REPORT-7990, REQ-VERIFY-7990: custody and estimates cannot imply device work."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

from carnot.reporting import experiment_7990_v692_hardware_evidence as q
from carnot.reporting import validation_7990 as plan
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7990_v692_hardware_evidence as cli


def authority(root: Path) -> Path:
    """Copy authenticated inputs so negative controls preserve source history."""
    labels = set()
    pins = {}
    for label in (q.PRIOR, q.SERVICE):
        value = json.loads((q.ROOT / label).read_text())
        labels.add(label)
        refs = value["source_artifact_hashes"]
        refs = list(refs.values()) if isinstance(refs, dict) else refs
        refs += value.get("code_config_hashes", []) + value.get("raw_shard_hashes", [])
        if value.get("input_checkpoint"):
            refs.append(value["input_checkpoint"])
        labels.update(r["path"] for r in refs if r.get("sha256"))
        pins.update({r["path"]: r["sha256"] for r in refs if r.get("sha256")})
        labels.update(value.get("terminal_receipt_hashes", {}))
        terminal_label = value["terminal_validation_sidecar_path"]
        labels.add(terminal_label)
        terminal = json.loads(Path(terminal_label).read_text())
        if terminal.get("sidecar_path"):
            labels.add(terminal["sidecar_path"])
            terminal = json.loads(Path(terminal["sidecar_path"]).read_text())["report"]
        labels.update(r["log_path"] for r in terminal.get("reports", terminal.get("receipts", [])))
    for label in labels:
        source = q.ROOT / label
        archived = (
            q.ROOT
            / "results/raw/experiment_7989_v692_service_cost/storage_attempts/tmpfs/code"
            / source.relative_to(q.ROOT)
        )
        if label in pins and sha256_file(source) != pins[label] and archived.is_file():
            assert sha256_file(archived) == pins[label]
            source = archived
        target = q.history.bound_path(root, str(q.ROOT / label))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return root


def test_valid_custody_and_mapping(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7990-MAPPING: zero compatible kernel is a ready null."""
    root = authority(tmp_path)
    value = q.read_evidence(root, "20261001")
    assert value["verdict_class"] == "null" and value["hardware_evidence_ready_score"] == 1
    assert value["experiment_id"] == 7990 and value["milestone"] == "2026.10.692"
    assert value["execution_date"] == value["run_date"] == "20261001"
    boards = value["board_rows"]
    assert [r["board"] for r in boards] == ["KV260", "PolarFire", "GateMate"]
    assert [r["evidence_age_days"] for r in boards] == [18, 18, 16]
    assert boards[0]["k_max"] == 5 and boards[1]["processor_class"] == "linux_cpu"
    assert boards[2]["blocker"] == "0xffffffff"
    assert all(not r["compatible_workload"] for r in boards)
    assert len(value["workload_placement_rows"]) == 12 * 7 * 3
    assert value["compatible_fraction"] == 0
    assert value["ideal_amdahl_bound"] == value["modeled_100x_bound"] == 1
    assert value["spline_accounting"]["coefficient_touches"]["fsync"] > 0
    assert value["spline_accounting"]["online_update_touches"] is None
    assert value["trained_head_specs"] and value["MODEL_SPECS"] == []
    assert not any(value["model_invocation_counts"].values())
    assert not value["hardware_speedup_claimed"]
    assert q.cold_reduce(root, value)["row_count"] == 3
    with pytest.raises(ValueError, match="run_date_mismatch"):
        q.read_evidence(q.ROOT, "20260930")


@pytest.mark.parametrize(
    "kind",
    [
        "missing_prior",
        "board_hash",
        "service_missing",
        "service_hash",
        "terminal",
        "checkpoint",
        "retired",
    ],
)
def test_external_block(tmp_path: Path, kind: str) -> None:
    """SCENARIO-REPORT-7990-CUSTODY: failed bytes block without device retries."""
    root = authority(tmp_path)
    prior = json.loads((root / q.PRIOR).read_text())
    service = json.loads((root / q.SERVICE).read_text())
    labels = dict(
        missing_prior=q.PRIOR,
        board_hash=prior["board_rows"][0]["source_path"],
        service_missing=q.SERVICE,
        service_hash=q.SERVICE,
        terminal=service["terminal_validation_sidecar_path"],
        checkpoint=service["input_checkpoint"]["path"],
    )
    if kind == "retired":
        (root / "ops").mkdir()
        (root / "ops/exclusion_manifest.yaml").write_text("retired:\n- experiment_id: 7989\n")
    else:
        path = q.history.bound_path(root, labels[kind])
        if kind in {"missing_prior", "service_missing"}:
            path.unlink()
        else:
            path.write_text("{}")
    value = q.read_evidence(root, "20261001")
    assert value["verdict_class"] == "blocked" and value["hardware_evidence_ready_score"] == 0
    assert value["gate_check_summary"] and len(value["board_rows"]) == 3
    assert all(
        set(("artifact_path", "artifact_hash", "artifact_field", "op", "expected", "observed"))
        <= set(r)
        for r in value["gate_check_summary"]
    )
    assert not value["workload_placement_rows"]
    assert q.cold_reduce(root, value)["row_count"] == 3


@pytest.mark.parametrize(
    "field",
    [
        "board_rows",
        "workload_placement_rows",
        "modeled_100x_bound",
        "hardware_evidence_ready_score",
        "gate_check_summary",
    ],
)
def test_claim_drift(field: str) -> None:
    """SCENARIO-VERIFY-7990-REPLAY: reductions must reject edited claims."""
    value = q.read_evidence(q.ROOT, "20261001")
    value[field] = "changed"
    with pytest.raises(ValueError, match="claims_changed"):
        q.cold_reduce(q.ROOT, value)


def test_bounds() -> None:
    """SCENARIO-VERIFY-7990-BOUNDS: transfer stays in the service denominator."""
    assert q.bounds(0.5, 10, None)["modeled_100x_bound"] == pytest.approx(1 / 0.505)
    assert q.bounds(0.5, 10, 1)["modeled_100x_bound"] == pytest.approx(1 / 0.605)
    assert q.bounds(1, 10, 0)["ideal_amdahl_bound"] is None
    assert q.bounds(0, 10, None)["comparison_class"] == "optimistic_estimate"
    for args in ((-1, 10, 0), (0.5, 0, 0), (0.5, 10, -1), (float("nan"), 10, 0)):
        with pytest.raises(ValueError, match="invalid_bound_operand"):
            q.bounds(*args)


def test_owned_replay() -> None:
    """SCENARIO-VERIFY-7990-REPLAY: disqualification needs failed owned receipts."""
    value = q.read_evidence(q.ROOT, "20261001")
    value.update(verdict_class="disqualified", hardware_evidence_ready_score=0)
    with pytest.raises(ValueError, match="unsubstantiated_disqualification"):
        q.cold_reduce(q.ROOT, value)
    value["validation_receipts"]["checks"] = [{"classification": "required", "passed": False}]
    assert q.cold_reduce(q.ROOT, value)["row_count"] == 3
    value["hardware_evidence_ready_score"] = 1
    with pytest.raises(ValueError, match="claims_changed:readiness"):
        q.cold_reduce(q.ROOT, value)


def test_replay_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7990-CUSTODY: invalid upstream reductions are external gates."""

    def reject(value: Any) -> None:
        raise ValueError("upstream_span_drift")

    monkeypatch.setattr(q.service, "replay", reject)
    value = q.read_evidence(authority(tmp_path), "20261001")
    assert value["verdict_class"] == "blocked"
    assert any(r["observed"] == "upstream_span_drift" for r in value["gate_check_summary"])


def test_private_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7990-VALIDATION: all CLI routes leave outputs private."""
    output = tmp_path / "experiment_7990_fixture.json"
    assert cli.main(["--evidence-only", "--output", str(output)]) == 0
    assert cli.main(["--cold-replay", str(output)]) == 0
    assert cli.main(["--terminal-recheck", str(output)]) == 0
    blocked = tmp_path / "blocked/experiment_7990_fixture.json"
    assert (
        cli.main(["--evidence-only", "--root", str(tmp_path / "absent"), "--output", str(blocked)])
        == 0
    )
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    with pytest.raises(SystemExit):
        cli.main(["--date", "20260930"])
    monkeypatch.setattr(plan, "qualify", lambda *a: 0)
    assert cli.main(["--output", str(output)]) == 0


def test_real_private_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7990-VALIDATION: cold script replay needs no PYTHONPATH."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [str(q.ROOT / ".venv/bin/python")]
    if env.get("CARNOT_7990_COVERAGE_DIR"):
        command += [
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--data-file=" + env["CARNOT_7990_COVERAGE_DIR"] + "/.coverage",
            plan.INCLUDE,
        ]
    command += [str(q.ROOT / plan.SCRIPT), "--root", str(authority(tmp_path / "authority"))]
    output = tmp_path / "experiment_7990_fixture.json"
    for args in (["--evidence-only", "--output", str(output)], ["--cold-replay", str(output)]):
        print("[exp7990-test] before_subprocess private CLI", flush=True)
        result = subprocess.run(
            command + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        print("[exp7990-test] after_subprocess private CLI", flush=True)
        assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    value["modeled_100x_bound"] = 100
    atomic_json(output, value)
    print("[exp7990-test] before_subprocess tamper replay", flush=True)
    result = subprocess.run(
        command + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    print("[exp7990-test] after_subprocess tamper replay", flush=True)
    assert result.returncode != 0 and "claims_changed" in result.stderr


@pytest.mark.parametrize("mode", ["pass", "required_failure", "reader_failure", "terminal_failure"])
def test_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """SCENARIO-REPORT-7990-VALIDATION: failures cannot supply ready primary bytes."""

    def child(
        spec: dict[str, Any], private: Path, durable: Path, started: float, units: int
    ) -> dict[str, Any]:
        log = private / (spec["name"] + ".log")
        log.write_text("{}")
        if spec["name"] == "coverage_json":
            atomic_json(
                private / "coverage.json",
                {
                    "files": {
                        name: {"summary": {"covered_lines": 1, "num_statements": 1}}
                        for name in plan.OWNED
                    }
                },
            )
        passed = not (
            mode == "required_failure"
            and spec["name"] == "owned_tests"
            or mode == "terminal_failure"
            and spec["classification"] == "terminal_validator"
        )
        return dict(
            spec,
            passed=passed,
            exit_code=0 if passed else 1,
            timed_out=False,
            log_path=str(log),
            log_sha256=sha256_file(log),
        )

    monkeypatch.setattr(plan, "run_child", child)
    fixture_root = authority(tmp_path / "authority")
    actual_read = q.read_evidence
    monkeypatch.setattr(q, "read_evidence", lambda root, date: actual_read(fixture_root, date))
    if mode == "reader_failure":
        monkeypatch.setattr(plan, "reader_receipt", lambda *a, **kw: {"passed": False})
    output = tmp_path / "experiment_7990_fixture.json"
    if mode in {"reader_failure", "terminal_failure"}:
        with pytest.raises(ValueError, match="reader_identity|candidate_rejected"):
            plan.qualify(q.ROOT, "20261001", output)
    else:
        assert plan.qualify(q.ROOT, "20261001", output) == 0
        value = json.loads(output.read_text())
        assert value["verdict_class"] == ("null" if mode == "pass" else "disqualified")
        assert value["hardware_evidence_ready_score"] == (1 if mode == "pass" else 0)
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
        assert terminal["candidate_sha256"] == sha256_file(output)
    with pytest.raises(ValueError, match="worktree_root"):
        plan.qualify(tmp_path, "20261001", output)


def test_manifest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7990-VALIDATION: coverage scope and validation are frozen."""
    commands = plan.manifest(tmp_path)
    assert (
        next(r for r in commands if r["name"] == "full_pytest")["classification"]
        == "repository_health"
    )
    assert any(r["name"] == "owned_tests" for r in commands)
    assert all(r["deadline_s"] > 0 for r in commands)
    assert len(plan.terminal_manifest(tmp_path)) == 3
    with pytest.raises(ValueError, match="private_tmp_required"):
        plan.manifest(q.ROOT / "results")


def test_current_code_drift_blocks_mapping() -> None:
    """SCENARIO-REPORT-7990-CUSTODY: readiness cannot override producer hash drift."""
    value = q.read_evidence(q.ROOT, "20261001")
    assert value["required_custody_valid"]
    assert value["verdict_class"] == "blocked"
    assert value["hardware_evidence_ready_score"] == 0
    changed = {
        r["artifact_path"] for r in value["gate_check_summary"] if r["artifact_field"] == "sha256"
    }
    expected = {
        "python/carnot/experiment_7989_v692_service_cost.py",
        "python/carnot/reporting/service_cost_7989.py",
        "tests/python/test_service_cost_7989.py",
        "tests/python/test_experiment_7989_v692_service_cost.py",
    }
    assert changed == {str(q.ROOT / path) for path in expected}
    assert not value["workload_placement_rows"]


def test_unpinned_optional_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7990-CUSTODY: an optional unpinned reference supplies no evidence."""
    authenticated = q.history.authenticate

    def optional(
        root: Path, label: str, pin: str, checks: list[dict[str, Any]], sources: dict[str, Any]
    ) -> dict[str, Any]:
        value = authenticated(root, label, pin, checks, sources)
        if label == q.PRIOR:
            value["source_artifact_hashes"]["unqualified_tsu"] = dict(
                path="unqualified-tsu.json", sha256=None
            )
        return value

    monkeypatch.setattr(q.history, "authenticate", optional)
    checks: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    q.input_evidence(q.ROOT, q.PRIOR, q.PRIOR_HASH, 7977, checks, sources)
    assert all(r["passed"] for r in checks)
    assert "unqualified-tsu.json" not in sources
