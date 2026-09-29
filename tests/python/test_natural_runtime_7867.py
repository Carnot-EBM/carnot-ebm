"""REQ-VERIFY-7867: real byte fixtures and a causal advisory bank."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import pytest

from carnot.verify import evidence_views, natural_predicates, natural_training, training_runtime
from carnot.verify.natural_bank import NaturalBank
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from scripts.experiments import experiment_7867_v683_natural_runtime as experiment


def test_fixture_runtime_and_split(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7867-TRAIN: the callable path fits every declared arm."""
    output = tmp_path / "fixture.json"
    run = subprocess.run(
        [
            sys.executable,
            str(Path(experiment.__file__)),
            "--fixture",
            "--output",
            str(output),
            "--private-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    record = json.loads(output.read_text())
    assert record["verdict_class"] == "circular_positive"
    assert record["verifier_is_oracle"] is True
    assert record["natural_measurement_performed"] is False
    assert {row["arm"] for row in record["rows"]} == set(evidence_views.ARMS)
    assert all(row["feature_dim"] == 132 for row in record["rows"])
    assert all(row["temperature"] in training_runtime.TEMPERATURES for row in record["rows"])
    assert all(row["view_divergence"] >= 0 for row in record["rows"])
    assert record["natural_training_ready_score"] == 1
    assert any(
        row["artifact_field"] == "source_boundary_ready_score"
        for row in record["scientific_gate_check_summary"]
    )
    assert record["sample_size_budget"]["independent"] == 2
    assert record["MODEL_SPECS"] == []
    assert Path(record["runtime_manifest_path"]).is_file()


def test_source_bytes_and_unknown_labels() -> None:
    """SCENARIO-VERIFY-7867-TRAIN: labels never become view features."""
    fit, tune = experiment.fixture_records()
    assert fit[0]["source"] != tune[0]["source"]
    assert (
        len(
            evidence_views.prepare_views(fit[0]["source"].encode(), fit[0]["answer"].encode())["a"][
                "windows"
            ]
        )
        > 2
    )
    batch, _ = natural_training.prepare(fit, "local_set")
    changed = [{**fit[0], "id": "changed", "label": 1, "known": [-1]}]
    changed_batch, _ = natural_training.prepare(changed, "local_set")
    np.testing.assert_array_equal(batch["a"]["x"], changed_batch["a"]["x"])
    params = training_runtime.init_params("energy_local", 67801)
    divergence = float(training_runtime.constraints(params, batch, "energy_local")[0])
    swapped = {**batch, "a": batch["b"], "b": batch["a"]}
    assert divergence == pytest.approx(
        float(training_runtime.constraints(params, swapped, "energy_local")[0])
    )
    all_unknown = {**batch, "known": batch["known"].at[:].set(-1)}
    assert float(training_runtime._local_loss(params, all_unknown, "energy_local", "a")) == 0
    gradients = jax.grad(training_runtime._local_loss)(params, all_unknown, "energy_local", "a")
    assert all(np.all(np.asarray(value) == 0) for value in gradients.values())
    with pytest.raises(ValueError, match="overlap"):
        natural_training.fit(fit, [{**fit[0], "role": "tune"}], "local_set", 67801, 0.01, 1)


def test_admission_restart_and_no_write(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7867-BANK: admission changes later action without label training."""
    path = tmp_path / "bank.json"
    bank = NaturalBank(path)
    feature = natural_predicates.features(b"Lumen has 12.", b"Lumen has 13.")
    name = "and_unmatched_decimal"
    bank.predict("train", 0, feature, 0.1)
    bank.release("train", 1, 1)
    before = bank.predict("probe", 2, feature, 0.045, read_only=True)
    bytes_before = path.read_bytes()
    assert bank.predict("probe", 2, feature, 0.045, read_only=True) == before
    assert path.read_bytes() == bytes_before
    bank.predict("admission", 3, feature, 0.045)
    with pytest.raises(ValueError, match="early"):
        bank.release("admission", 3, 1)
    coefficients = list(bank.state["coefficients"])
    bank.release("admission", 4, 1, admission_only=True)
    bank.admit("admission", name)
    assert bank.state["coefficients"] == coefficients
    after = bank.predict("later", 5, feature, 0.045, read_only=True)
    assert after["probability"] > before["probability"]
    assert after["action"] != before["action"]
    assert NaturalBank(path).predict("later", 5, feature, 0.045, read_only=True) == after
    with pytest.raises(ValueError, match="order"):
        bank.predict("past", 1, feature, 0.04)
    assert all("previous" in event and "hash" in event for event in bank.state["ledger"])


def test_real_cli_and_blocked_manifest(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7867-RECEIPT: both real CLI branches publish honest records."""
    output = tmp_path / "fixture.json"
    script = str(Path(experiment.__file__))
    good = subprocess.run(
        [
            sys.executable,
            script,
            "--date",
            "20260929",
            "--fixture",
            "--output",
            str(output),
            "--private-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert good.returncode == 0, good.stderr
    assert json.loads(output.read_text())["verdict_class"] == "circular_positive"
    manifest = tmp_path / "missing.json"
    blocked = tmp_path / "blocked.json"
    bad = subprocess.run(
        [
            sys.executable,
            script,
            "--date",
            "20260929",
            "--qualified-manifest",
            str(manifest),
            "--output",
            str(blocked),
            "--private-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert bad.returncode == 0, bad.stderr
    value = json.loads(blocked.read_text())
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"]
    assert value["natural_training_ready_score"] == 0


def test_hash_bound_qualified_fixture_and_overlap(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7867-TRAIN: a qualified circular source reaches fitting."""
    fit, tune = experiment.fixture_records()
    entries = []
    for row in [*fit, *tune]:
        path = tmp_path / f"{row['role']}.json"
        path.write_text(
            json.dumps({key: row[key] for key in ("source", "answer", "label", "known")})
        )
        digest = sha256_file(path)
        entries.append(
            {
                "id": row["id"],
                "group": row["group"],
                "role": row["role"],
                "path": str(path),
                "sha256": digest,
                "split_token": experiment.split_token(row["group"], row["role"], digest),
            }
        )
    manifest = tmp_path / "qualified.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "carnot.natural_runtime.qualified.v1",
                "fixture_oracle": True,
                "records": entries,
            }
        )
    )
    checks, _, records = experiment.preflight(manifest)
    assert all(check["passed"] for check in checks)
    assert {row["role"] for row in records} == {"fit", "tune"}
    output = tmp_path / "qualified_output.json"
    run = subprocess.run(
        [
            sys.executable,
            str(Path(experiment.__file__)),
            "--qualified-manifest",
            str(manifest),
            "--output",
            str(output),
            "--private-root",
            str(tmp_path / "private"),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text())["verdict_class"] == "circular_positive"
    entries[1]["group"] = entries[0]["group"]
    manifest.write_text(
        json.dumps(
            {
                "schema": "carnot.natural_runtime.qualified.v1",
                "fixture_oracle": True,
                "records": entries,
            }
        )
    )
    checks, _, _ = experiment.preflight(manifest)
    assert {check["artifact_field"] for check in checks if not check["passed"]} >= {
        "split_token",
        "group_unique_role",
    }


def test_manifest_failures_and_authority(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7867-TRAIN: malformed bytes and missing authority block."""
    manifest = tmp_path / "qualified.json"
    manifest.write_text("[invalid")
    checks, _, _ = experiment.preflight(manifest)
    assert any(row["artifact_field"] == "schema" and not row["passed"] for row in checks)
    manifest.write_text("[]")
    checks, _, _ = experiment.preflight(manifest)
    assert any(not row["passed"] for row in checks)
    manifest.write_text(
        json.dumps(
            {
                "schema": "carnot.natural_runtime.qualified.v1",
                "fixture_oracle": True,
                "records": "invalid",
            }
        )
    )
    checks, _, _ = experiment.preflight(manifest)
    assert any(row["artifact_field"] == "has_fit" and not row["passed"] for row in checks)
    manifest.write_text(
        json.dumps(
            {
                "schema": "carnot.natural_runtime.qualified.v1",
                "fixture_oracle": False,
                "records": [None],
            }
        )
    )
    checks, _, _ = experiment.preflight(manifest)
    assert any(
        row["artifact_field"] == "source_boundary_ready_score" and not row["passed"]
        for row in checks
    )
    broken = tmp_path / "broken.json"
    broken.write_text("{invalid")
    digest = sha256_file(broken)
    manifest.write_text(
        json.dumps(
            {
                "schema": "carnot.natural_runtime.qualified.v1",
                "fixture_oracle": True,
                "records": [
                    {
                        "group": "bad",
                        "role": "fit",
                        "path": str(broken),
                        "sha256": digest,
                        "split_token": experiment.split_token("bad", "fit", digest),
                    }
                ],
            }
        )
    )
    checks, _, _ = experiment.preflight(manifest)
    assert any(row["artifact_field"] == "record_schema" and not row["passed"] for row in checks)
    scientific, source = experiment.science_gate(tmp_path)
    assert source["sha256"] is None
    assert all(not row["passed"] for row in scientific)


def test_bank_lineage_and_mixed_admission(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7867-BANK: a state checksum cannot hide broken ledger links."""
    path = tmp_path / "bank.json"
    bank = NaturalBank(path)
    feature = natural_predicates.features(b"Lumen has 12.", b"Lumen has 13.")
    bank.predict("first", 0, feature, 0.2)
    bank.release("first", 1, 1)
    bank.predict("second", 2, feature, 0.2)
    bank.release("second", 3, 1)
    with pytest.raises(ValueError, match="mixed"):
        bank.admit("first", "and_unmatched_decimal")
    state = json.loads(path.read_text())
    state["ledger"][-1]["previous"] = "forged"
    state["checksum"] = canonical_hash(
        {key: value for key, value in state.items() if key != "checksum"}
    )
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="lineage"):
        NaturalBank(path)


@pytest.mark.memory_watchdog_skip
def test_qualified_authority_and_cold_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7867-RECEIPT: a private authority fixture exercises the callable branch."""
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    producer_path = tmp_path / "results/experiment_7866_v683_source_boundary.json"
    atomic_json(
        producer_path,
        {
            "experiment_id": 7866,
            "milestone": "2026.09.683",
            "source_boundary_ready_score": 1,
            "flagged_adversarial": False,
            "verdict_class": "circular_positive",
            "cohort_manifest_sha256": "sha256:private-fixture",
        },
    )
    entries = []
    fit, tune = experiment.fixture_records()
    for row in [*fit, *tune]:
        path = tmp_path / f"{row['role']}.json"
        atomic_json(path, {key: row[key] for key in ("source", "answer", "label", "known")})
        digest = sha256_file(path)
        entries.append(
            {
                "id": row["id"],
                "group": row["group"],
                "role": row["role"],
                "path": str(path),
                "sha256": digest,
                "split_token": experiment.split_token(row["group"], row["role"], digest),
            }
        )
    manifest = tmp_path / "qualified.json"
    atomic_json(
        manifest,
        {
            "schema": "carnot.natural_runtime.qualified.v1",
            "fixture_oracle": False,
            "producer_path": str(producer_path),
            "producer_sha256": sha256_file(producer_path),
            "cohort_manifest_sha256": "sha256:private-fixture",
            "records": entries,
        },
    )
    checks, _, _ = experiment.preflight(manifest)
    assert all(row["passed"] for row in checks)
    record = experiment.run_fixture(tmp_path / "private", "20260929", qualified_manifest=manifest)
    assert record["verdict_class"] == "null"
    assert record["natural_measurement_performed"] is True
    path = tmp_path / "candidate.json"
    atomic_json(path, record)
    assert experiment.main(["--cold-replay", str(path)]) == 0
    record["rows"][0]["brier"] = -1
    atomic_json(path, record)
    assert experiment.main(["--cold-replay", str(path)]) == 1


@pytest.mark.memory_watchdog_skip
def test_runner_rejects_inactive_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7867-BANK: a no-op admission cannot qualify online readiness."""
    monkeypatch.setattr(NaturalBank, "admit", lambda self, event_id, name: None)
    with pytest.raises(ValueError, match="causal admission"):
        experiment.run_fixture(tmp_path, "20260929")
    with pytest.raises(SystemExit) as error:
        experiment.main(["--date", "20260928", "--output", str(tmp_path / "unused.json")])
    assert error.value.code == 2
