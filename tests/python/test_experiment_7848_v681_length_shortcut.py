"""REQ-REPORT-7848 and SCENARIO-REPORT-7848-* length-control checks."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from carnot.reporting import length_shortcut as length
from scripts.experiments import experiment_7848_v681_length_shortcut as cli


ROOT = Path(__file__).resolve().parents[2]


def _row(family: str, answer: str, source: str) -> dict:
    return {"family_id": family, "complete_response": answer, "complete_source": source}


def test_public_features_use_utf8_bytes_and_ignore_metadata() -> None:
    """SCENARIO-REPORT-7848-FIT: bytes, including non-ASCII, determine inputs."""
    a = _row("one", "é", "abcd")
    b = {**a, "label": 1, "role": "evaluation", "annotations": ["private"]}
    assert length.public_features(a) == length.public_features(b)
    x = length.public_features(a)
    assert x[0] == pytest.approx(length.math.log1p(2))
    assert x[1] == pytest.approx(length.math.log1p(4))
    assert x[2] == pytest.approx(x[0] / x[1])


def test_fit_uses_only_fit_labels_and_fixed_grid() -> None:
    """SCENARIO-REPORT-7848-FIT: repeated fit is deterministic and tune only calibrates."""
    fit = [_row(str(i), "a" * (i + 1), "s" * 20) for i in range(8)]
    labels = {str(i): int(i >= 4) for i in range(8)}
    model = length.fit_model(fit, labels)
    assert model == length.fit_model(fit, labels)
    assert model["steps"] == 200
    assert model["regularization"] == 1.0
    assert len(model["weights"]) == 3
    assert length.predict(model, fit[-1]) > length.predict(model, fit[0])
    assert length.TEMPERATURE_GRID == (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    assert length.choose_temperature(model, fit, labels) in length.TEMPERATURE_GRID


def test_strata_cost_and_paired_bootstrap() -> None:
    """SCENARIO-REPORT-7848-FIT: fit quartiles and one family per resample unit."""
    fit = [_row(str(i), "a" * (i + 1), "s") for i in range(8)]
    edges = length.fit_strata(fit)
    assert edges == [2, 4, 6]
    assert [length.stratum(i, edges) for i in (1, 2, 3, 4, 8)] == [0, 0, 1, 1, 3]
    assert length.action(0.01) == "accept"
    assert length.action(0.1) == "escalate"
    assert length.action(0.9) == "reject"
    assert length.realized_cost("accept", 1) == 5
    assert length.realized_cost("reject", 0) == 1
    assert length.realized_cost("escalate", 1) == 0.25
    assert length.paired_bootstrap([1.0, 2.0], 10000, 7848) == length.paired_bootstrap(
        [1.0, 2.0], 10000, 7848
    )


def test_authenticate_all_roles_and_reject_changed_source(tmp_path: Path) -> None:
    """REQ-REPORT-7848: qualified canonical roles and raw hashes are mandatory."""
    custody = length.authenticate(ROOT)
    assert custody["role_counts"] == {
        "fit": 256,
        "tune": 64,
        "evaluation": 64,
        "policy": 64,
        "retention": 32,
        "online_admission": 64,
        "online_update": 96,
    }
    assert sum(map(len, custody["public"].values())) == 640
    assert len({row["family_id"] for rows in custody["public"].values() for row in rows}) == 640
    row = custody["public"]["fit"][0]
    assert length.digest_text(row["complete_source"]) == row["source_sha256"]
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        length.verify_raw_row({**row, "complete_source": "altered"})
    altered = tmp_path / "receipt.json"
    altered.write_text(json.dumps({"verdict_class": "blocked"}))
    with pytest.raises(length.CustodyError):
        length.authenticate(ROOT, qualification_path=altered)


def test_energy_gate_rejects_conductor_receipt() -> None:
    """SCENARIO-REPORT-7848-ENERGY: current blocked gate is no fitted head."""
    energy = length.energy_gate(ROOT)
    assert energy["qualified"] is False
    assert energy["failures"]
    assert energy["failures"][0]["upstream_id"] == "exp7840"
    assert energy["permutation_rows"] == []


def test_real_baseline_seals_predictions_before_evaluation_labels(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7848-FIT: raw current rows and cold replay agree."""
    custody = length.authenticate(ROOT)
    result = length.run_baseline(custody, tmp_path, seed=7848)
    assert len(result["rows"]) == 64
    assert result["sample_size_budget"]["independent_n"] == 64
    assert result["prediction_seal_sha256"].startswith("sha256:")
    assert result["bootstrap"]["draws"] == 10000
    assert sum(item["count"] for item in result["strata_metrics"]) == 64
    assert length.cold_replay(result, custody, tmp_path)["passed"] is True
    (tmp_path / "evaluation_predictions.json").write_text("{}")
    assert length.cold_replay(result, custody, tmp_path)["passed"] is False


def test_private_cli_and_cold_replay() -> None:
    """SCENARIO-REPORT-7848-TERMINAL: real entrypoint keeps child validation off."""
    script = ROOT / "scripts/experiments/experiment_7848_v681_length_shortcut.py"
    with tempfile.TemporaryDirectory(prefix="exp7848-cli-", dir="/tmp") as private:
        command = [
            sys.executable,
            "-u",
            str(script),
            "--date",
            "20260929",
            "--output-root",
            private,
            "--science-only",
        ]
        child = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert child.returncode == 0, child.stdout + child.stderr
        candidate = Path(private) / "candidate.json"
        value = json.loads(candidate.read_text())
        assert value["experiment_id"] == 7848
        assert value["task_id"] == "exp7848-length-shortcut"
        assert value["length_control_ready_score"] == 1
        assert len(value["rows"]) == 64
        replay = subprocess.run(
            [sys.executable, "-u", str(script), "--cold-replay", str(candidate)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert replay.returncode == 0, replay.stdout + replay.stderr


def test_custody_rejects_missing_and_wrong_qualification(tmp_path: Path) -> None:
    """REQ-REPORT-7848: absent and wrong-valued operands remain distinct."""
    with pytest.raises(length.CustodyError) as absent:
        length.authenticate(ROOT, qualification_path=tmp_path / "missing.json")
    assert absent.value.failures[0]["artifact_field"] == "is_file"
    original = json.loads((ROOT / length.QUALIFICATION).read_text())
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps({**original, "verdict_class": "blocked"}))
    with pytest.raises(length.CustodyError) as wrong:
        length.authenticate(ROOT, qualification_path=path)
    assert wrong.value.failures[0]["artifact_field"] == "verdict_class"
    path.write_text(json.dumps({**original, "source_view_manifest_path": str(tmp_path / "absent")}))
    with pytest.raises(length.CustodyError) as missing_manifest:
        length.authenticate(ROOT, qualification_path=path)
    assert missing_manifest.value.failures[0]["artifact_field"] == "is_file"


def test_custody_rejects_changed_manifest_and_public_row(tmp_path: Path) -> None:
    """REQ-REPORT-7848: manifest hash and raw response bytes cannot drift."""
    original = json.loads((ROOT / length.QUALIFICATION).read_text())
    manifest = json.loads(Path(original["source_view_manifest_path"]).read_text())
    mpath = tmp_path / "manifest.json"
    qpath = tmp_path / "qualification.json"
    mpath.write_text(json.dumps({**manifest, "role_hashes": {}}))
    qpath.write_text(json.dumps({**original, "source_view_manifest_path": str(mpath)}))
    with pytest.raises(length.CustodyError) as changed:
        length.authenticate(ROOT, qualification_path=qpath)
    assert changed.value.failures[0]["artifact_field"] == "role_hashes"
    mpath.write_text(json.dumps({**manifest, "rows_sha256": "sha256:wrong"}))
    with pytest.raises(length.CustodyError) as changed_rows:
        length.authenticate(ROOT, qualification_path=qpath)
    assert changed_rows.value.failures[0]["artifact_field"] == "rows_sha256"
    custody = length.authenticate(ROOT)
    row = custody["public"]["fit"][0]
    with pytest.raises(ValueError, match="response_hash_mismatch"):
        length.verify_raw_row({**row, "complete_response": "changed"})


def test_replay_rejects_changed_metrics_and_roster(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7848-TERMINAL: independent replay rejects row drift."""
    custody = length.authenticate(ROOT)
    result = length.run_baseline(custody, tmp_path)
    altered = json.loads(json.dumps(result))
    altered["rows"][0]["label"] = 1 - altered["rows"][0]["label"]
    assert length.cold_replay(altered, custody, tmp_path)["reason"] == "label_join_mismatch"
    altered = json.loads(json.dumps(result))
    altered["rows"][0]["arms"]["length"]["cost"] = 99
    assert length.cold_replay(altered, custody, tmp_path)["reason"] == "length_metric_mismatch"
    altered = json.loads(json.dumps(result))
    altered["rows"][0]["arms"]["always_escalate"]["cost"] = 99
    assert length.cold_replay(altered, custody, tmp_path)["reason"] == "escalation_mismatch"
    altered = json.loads(json.dumps(result))
    altered["config_sha256"] = "sha256:wrong"
    assert length.cold_replay(altered, custody, tmp_path)["reason"] == "config_mismatch"
    altered = json.loads(json.dumps(result))
    altered["rows"].pop()
    assert length.cold_replay(altered, custody, tmp_path)["reason"] == "row_count_mismatch"


def test_energy_gate_missing_and_headless(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7848-ENERGY: missing and headless science stay blocked."""
    absent = length.energy_gate(tmp_path)
    assert absent["failures"][0]["artifact_field"] == "is_file"
    path = tmp_path / length.ENERGY
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {"status": "completed", "energy_fit_ready_score": 1, "flagged_adversarial": False}
        )
    )
    headless = length.energy_gate(tmp_path)
    assert headless["failures"][0]["artifact_field"] == "constrained_set_heads"


def test_fit_rejects_roster_and_checkpoint_replacement(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7848-FIT: retries cannot replace a fitted observation."""
    with pytest.raises(ValueError, match="fit_roster_mismatch"):
        length.fit_model([_row("one", "a", "s")], {})
    path = tmp_path / "checkpoint.json"
    first = length._freeze(path, {"risk": 0.1})
    assert length._freeze(path, {"risk": 0.1}) == first
    with pytest.raises(ValueError, match="checkpoint_mismatch"):
        length._freeze(path, {"risk": 0.9})


def test_owned_child_deadline_and_sealed_log(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7848-TERMINAL: only owned children expire; logs are immutable."""
    spec = dict(
        name="worktree_imports",
        argv=[sys.executable, "-c", 'print("{\\"resolved_imports\\": {\\"x\\": \\"/tmp/x\\"}}")'],
        deadline_s=5,
        classification="required",
        env={"PYTHONPATH": "python:."},
    )
    receipt = cli.run_child(spec, tmp_path, cli.time.monotonic())
    assert receipt["passed"] is True
    assert receipt["resolved_imports"] == {"x": "/tmp/x"}
    assert length.sha256_file(Path(receipt["log_path"])) == receipt["log_sha256"]
    Path(receipt["log_path"]).write_text("tampered")
    with pytest.raises(ValueError, match="sealed_log_mismatch"):
        cli.run_child(spec, tmp_path, cli.time.monotonic())
    sleeper = dict(
        name="deadline",
        argv=[sys.executable, "-c", "import time;time.sleep(2)"],
        deadline_s=0.05,
        classification="diagnostic",
        env={},
    )
    expired = cli.run_child(sleeper, tmp_path, cli.time.monotonic())
    assert expired["timed_out"] is True
    assert expired["passed"] is False


def test_science_blocks_missing_external_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7848: external absence has a complete blocked terminal artifact."""
    missing = length.failure("exp7810", tmp_path / "missing", "is_file", True, False)

    def blocked(_root: Path) -> None:
        raise length.CustodyError([missing])

    monkeypatch.setattr(cli, "authenticate", blocked)
    result = cli.science(tmp_path, "20260929")
    assert result["verdict_class"] == "blocked"
    assert result["gate_check_summary"] == [missing]
    assert json.loads((tmp_path / "candidate.json").read_text())["length_control_ready_score"] == 0


def test_science_rejects_failed_cold_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7848-TERMINAL: a bad private replay cannot write a success."""
    monkeypatch.setattr(cli, "cold_replay", lambda *_args: {"passed": False})
    with pytest.raises(ValueError, match="cold_replay_failed"):
        cli.science(tmp_path, "20260929")


def test_validation_reducer_keeps_required_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7848-TERMINAL: health never erases a failed required command."""
    real_root = cli.ROOT
    candidate = cli.science(tmp_path / "science", "20260929")
    frozen = cli.commands(tmp_path / "validation")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "commands", lambda _output: frozen)

    def fake_child(spec: dict, output: Path, _start: float) -> dict:
        log = output / f"{spec['name']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('{"flagged_count": 1}' if spec["name"] == "adversarial_verify" else "ok")
        return dict(
            name=spec["name"],
            command_argv=spec["argv"],
            classification=spec["classification"],
            passed=spec["name"] != "affected_pytest",
            exit_code=1 if spec["name"] == "affected_pytest" else 0,
            log_path=str(log),
            log_sha256=length.sha256_file(log),
        )

    monkeypatch.setattr(cli, "run_child", fake_child)
    result = cli.validate(candidate, tmp_path / "validation")
    assert result["verdict_class"] == "disqualified"
    assert result["required_validation_failures"] == ["affected_pytest"]
    assert result["repository_health"]["status"] == "healthy"
    assert result["flagged_adversarial"] is True
    assert (tmp_path / "results/experiment_7848_v681_length_shortcut.json").is_file()
    monkeypatch.setattr(cli, "ROOT", real_root)


def test_main_cold_replay_rejects_log_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7848-TERMINAL: terminal reader reopens sealed child logs."""
    candidate = cli.science(tmp_path, "20260929")
    candidate["validation_receipts"] = [
        dict(log_path=str(tmp_path / "missing.log"), log_sha256="sha256:missing")
    ]
    path = tmp_path / "with_receipt.json"
    path.write_text(json.dumps(candidate))
    monkeypatch.setattr(sys, "argv", ["exp7848", "--cold-replay", str(path)])
    assert cli.main() == 1
