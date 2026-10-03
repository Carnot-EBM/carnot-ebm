"""Private fitting qualification checks REQ-VERIFY-7967 and REQ-REPORT-7967."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import private_training_7967 as core


def test_freeze_private_roots(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7967-ROOTS: mutable paths cannot enter archived results."""
    scratch, archive = tmp_path / "scratch", tmp_path / "archive"
    manifest, commands = core.freeze(scratch, archive)
    value = json.loads(manifest.read_text())
    assert value["scratch_root"] == str(scratch)
    assert value["archive_root"] == str(archive)
    assert value["execution_date"] == "20261001"
    assert value["coverage_includes"] == core.INCLUDES
    assert all(value["dependencies"].values())
    assert len([c for c in commands if c.scope in ("route", "exception", "control")]) >= 11
    for command in commands:
        for arg in command.argv:
            assert not arg.startswith(str(archive)) or command.name == "unsafe_scratch"
        if command.name.startswith("e2e_016"):
            assert command.argv[command.argv.index("--date") + 1] == "20260929"
    for unsafe in (core.ROOT, core.ROOT / "results/raw/new", archive, scratch / "overlap"):
        with pytest.raises(ValueError, match="unsafe_scratch"):
            core.freeze(unsafe, scratch if unsafe != archive else archive)


def test_guarded_mirror_and_atomic_storage(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7967-ROOTS: the actual guard reproduces the source rename bug."""
    receipt = core.guard_probe(tmp_path)
    assert receipt["passed"]
    assert receipt["guard_enabled"]
    assert receipt["protected_move_errno"] == 18
    assert receipt["private_atomic_same_device"]
    assert receipt["historical_failure_count"] == 13


def test_reduction_requires_every_route() -> None:
    """SCENARIO-VERIFY-7967-QUALIFICATION: coverage alone cannot open readiness."""
    commands = [core.CommandSpec("success", ("python",), "route", 30)]
    rows = [{"name": "success", "passed": True, "argv": ["python"]}]
    counts = {n: {"num_statements": 1, "missing_lines": 0} for n in core.COVERED}
    selected = [{"passed": True}] * 3
    fixture = {
        "fixture_ready_score": 1,
        "energy_fit_ready_score": 0,
        "trained_head_specs": [{}],
        "rows": [{}, {}, {}],
    }
    assert core.reduce_ready(commands, rows, counts, selected, fixture)
    for change in ({}, {core.COVERED[0]: {"num_statements": 0, "missing_lines": 0}}):
        assert not core.reduce_ready(commands, rows, change, selected, fixture)
    assert not core.reduce_ready(commands, [], counts, selected, fixture)
    assert not core.reduce_ready(commands, rows + rows, counts, selected, fixture)
    assert not core.reduce_ready(commands, rows, counts, [], fixture)
    assert not core.reduce_ready(commands, rows, counts, selected, {})


def test_cli_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7967: current dates and exception exits remain explicit."""
    from scripts.experiments import experiment_7967_v691_private_training_qualification as cli

    monkeypatch.setattr(core, "qualify", lambda *a: 0)
    monkeypatch.setattr(core, "replay", lambda *a: None)
    monkeypatch.setattr(core, "check_roots", lambda *a: None)
    monkeypatch.setattr(core, "guard_probe", lambda *a: {"passed": True})
    base = ["--date", "20261001"]
    assert cli.main(base) == 0
    assert cli.main(base + ["--cold-replay", str(tmp_path)]) == 0
    assert cli.main(base + ["--check-private-root", str(tmp_path)]) == 0
    assert cli.main(base + ["--guard-fixture", str(tmp_path)]) == 0
    monkeypatch.setattr(core, "replay", lambda *a: (_ for _ in ()).throw(ValueError("drift")))
    assert cli.main(base + ["--cold-replay", str(tmp_path)]) == 2


@pytest.mark.parametrize(
    "mode", ["ready", "failed", "blocked", "absent", "source_absent", "source_malformed"]
)
def test_qualification_terminal_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """SCENARIO-REPORT-7967-REPLAY: primitive evidence decides terminal readiness."""
    output = tmp_path / "experiment_7967_fixture.json"
    counts = {n: {"num_statements": 1, "missing_lines": 0} for n in core.COVERED}
    if mode == "blocked":

        def authenticate(*a: object) -> None:
            raise core.fit.old.custody.InputBlocked(
                [{"upstream_id": "exp7941", "artifact_field": "exists", "observed": False}]
            )

        monkeypatch.setattr(core.fit, "authenticate", authenticate)
    else:
        monkeypatch.setattr(core.fit, "authenticate", lambda *a: ([], {}, {}))
    monkeypatch.setattr(core, "authenticate_sources", lambda *a: [])
    if mode.startswith("source_"):
        source = tmp_path / "missing_source.json"
        if mode == "source_malformed":
            source.write_text("{")
        monkeypatch.setattr(core, "SOURCE_IDENTITIES", ((7954, source, "20260930"),))

        def blocked_sources() -> list:
            raise core.fit.old.custody.InputBlocked(
                [
                    {
                        "upstream_id": "exp7954",
                        "artifact_path": str(source),
                        "artifact_sha256": None,
                        "artifact_field": "valid_json",
                        "op": "==",
                        "expected": True,
                        "observed": False,
                    }
                ]
            )

        monkeypatch.setattr(core, "authenticate_sources", blocked_sources)

    def execute(commands: list, scratch: Path) -> list:
        if mode != "absent":
            (scratch / ".coverage").write_bytes(b"mock measured data")
            atomic_json(
                scratch / "coverage.json", {"files": {k: {"summary": v} for k, v in counts.items()}}
            )
        if mode != "absent":
            atomic_json(
                core.prior.route(scratch, "success"),
                {
                    "fixture_ready_score": 1,
                    "energy_fit_ready_score": 0,
                    "rows": [{}, {}, {}],
                    "fixture_prediction_rows": [{}, {}, {}],
                    "trained_head_specs": [{}],
                    "source_artifact_hashes": [],
                },
            )
        (scratch / "logs").mkdir()
        if mode == "ready":
            atomic_json(scratch / "guard_probe/receipt.json", {"passed": True})
        log = scratch / "logs/receipt.log"
        log.write_text("closed child")
        return [
            {
                "name": c.name,
                "passed": mode == "ready",
                "argv": list(c.argv),
                "scope": c.scope,
                "log_path": str(log),
                "log_sha256": sha256_file(log),
                "actual_exit": 2 if c.name == "unsafe_scratch" else 0,
                "expected_exit": 2 if c.name == "unsafe_scratch" else 0,
                "output_tail": "unsafe_scratch" if c.name == "unsafe_scratch" else "",
            }
            for c in commands
        ]

    monkeypatch.setattr(core.prior, "execute", execute)
    monkeypatch.setattr(core.prior, "negative_mutation", lambda *a: {"passed": True})
    monkeypatch.setattr(core.prior, "selections", lambda *a: [{"passed": True}] * 3)
    monkeypatch.setattr(core, "seal", lambda v, p: atomic_json(p, v))
    monkeypatch.setattr(core, "replay", lambda *a: None)
    assert core.qualify(output) == 0
    value = json.loads(output.read_text())
    assert value["experiment_id"] == 7967 and value["run_date"] == "20261001"
    assert value["training_execution_ready_score"] == int(mode == "ready")
    assert value["verdict_class"] == (
        "blocked"
        if mode == "blocked" or mode.startswith("source_")
        else "circular_positive"
        if mode == "ready"
        else "disqualified"
    )
    assert value["MODEL_SPECS"] == [] and value["execution_venue"] == "host"
    assert not Path(value["scratch_root_receipt"]["path"]).exists()
    assert value["historical_required_failures"]
    if mode.startswith("source_"):
        assert not value["validation_receipts"]
        assert value["gate_check_summary"][0]["upstream_id"] == "exp7954"
        for row in value["historical_required_failures"]:
            assert not row["passed"]
            assert sha256_file(Path(row["log_path"])) == row["log_sha256"]
        assert all(not row["fields_imported"] for row in value["cited_upstream_artifacts"])


def test_archive_replay_and_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7967-REPLAY: final hashes, rows and newer sidecars bind readers."""
    output = tmp_path / "experiment_7967_fixture.json"
    raw = output.parent / "raw" / output.stem
    primitive = raw / "primitive.json"
    atomic_json(primitive, {"measured": True})
    value = core.prior.base([], [], {})
    value.update(
        experiment_id=7967,
        task_id=core.TASK,
        training_execution_ready_score=0,
        scratch_root_receipt={"path": "/tmp/no-longer-live", "archive_path": str(raw / "evidence")},
    )
    value["source_artifact_hashes"] = [
        {"path": str(primitive), "sha256": sha256_file(primitive)},
        {"path": core.OWNED[0], "sha256": sha256_file(core.ROOT / core.OWNED[0])},
    ]
    monkeypatch.setattr(
        core, "terminal_checks", lambda *a: {"passed": True, "flagged_adversarial": False}
    )
    core.seal(value, output)
    core.replay(output)
    assert core.archived_path(value, Path("/tmp/no-longer-live/log")) == raw / "evidence/log"
    atomic_json(primitive, {"measured": False})
    with pytest.raises(ValueError, match="primitive hash drift"):
        core.replay(output)
    monkeypatch.setattr(
        core, "terminal_checks", lambda *a: {"passed": False, "flagged_adversarial": True}
    )
    with pytest.raises(ValueError, match="candidate_rejected"):
        core.seal(value, output)


def test_external_source_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7967: each source retains its own original invocation date."""
    original = core.SOURCE_IDENTITIES
    path = tmp_path / "source.json"
    shard = tmp_path / "shard.jsonl"
    shard.write_text("{}\n")
    monkeypatch.setattr(core, "SOURCE_IDENTITIES", ((7892, path, "20260929"),))
    atomic_json(
        path,
        {
            "experiment_id": 7892,
            "run_date": "20260929",
            "source_boundary_ready_score": 1,
            "flagged_adversarial": False,
            "public_shards": [{"path": str(shard), "sha256": sha256_file(shard)}],
        },
    )
    assert core.authenticate_sources()[-1]["producer_run_date"] == "20260929"
    shard.write_text("changed")
    with pytest.raises(core.fit.old.custody.InputBlocked):
        core.authenticate_sources()
    atomic_json(path, {"experiment_id": 7892, "run_date": "20261001"})
    with pytest.raises(core.fit.old.custody.InputBlocked):
        core.authenticate_sources()
    atomic_json(
        path, {"experiment_id": 7892, "run_date": "20260929", "source_boundary_ready_score": 0}
    )
    with pytest.raises(core.fit.old.custody.InputBlocked):
        core.authenticate_sources()
    path.write_text("{")
    with pytest.raises(core.fit.old.custody.InputBlocked):
        core.authenticate_sources()


def test_archived_numerical_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7967-REPLAY: recompute actual small-head predictions from checkpoints."""
    lib = core.fit.fixture_library()
    upstream = lib.make_fixture(tmp_path / "input")
    fitted = lib.fit_score(upstream, tmp_path / "attempt.json", tmp_path / "fit", 7943, "20260930")
    value = {
        "scratch_root_receipt": {"path": "/tmp/absent-original", "archive_path": str(tmp_path)},
        "fixture_checkpoint_manifest": {
            "path": fitted["checkpoint_manifest_path"],
            "sha256": fitted["checkpoint_manifest_sha256"],
        },
        "fixture_prediction_path": fitted["prediction_rows_path"],
        "fixture_prediction_sha256": fitted["prediction_rows_sha256"],
        "fixture_prediction_rows": fitted["fixture_prediction_rows"],
    }
    core.replay_fit(value)
    value["fixture_prediction_rows"] = []
    with pytest.raises(ValueError, match="fixture primitive drift"):
        core.replay_fit(value)
    value["fixture_prediction_rows"] = fitted["fixture_prediction_rows"]
    original = core.fit.old.natural_training.predict

    def wrong(*args: object) -> list:
        predicted = original(*args)
        predicted[0]["probability_unsupported"] += 0.01
        return predicted

    monkeypatch.setattr(core.fit.old.natural_training, "predict", wrong)
    with pytest.raises(ValueError, match="cold fixture prediction drift"):
        core.replay_fit(value)


def test_repository_health_preserves_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7967: a global deadline never becomes a current passing receipt."""
    monkeypatch.setattr(core, "HEALTH_LOG", tmp_path / "absent.log")
    assert core.archive_health(tmp_path / "raw") == {}
    core.HEALTH_LOG.write_text("pre-existing tests failed before deadline")
    row = core.archive_health(tmp_path / "raw")
    assert row["exit_code"] == 124 and not row["passed"] and row["timed_out"]
    assert sha256_file(Path(row["log_path"])) == row["log_sha256"]


def test_ready_replay_requires_primitive_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7967-REPLAY: missing commands and changed selected bytes fail cold reduction."""
    output = tmp_path / "experiment_7967_fixture.json"
    raw = output.parent / "raw" / output.stem
    evidence = raw / "evidence"
    counts = {n: {"num_statements": 1, "missing_lines": 0} for n in core.COVERED}
    fixture_path = core.prior.route(evidence, "success")
    atomic_json(
        fixture_path,
        {
            "fixture_ready_score": 1,
            "energy_fit_ready_score": 0,
            "trained_head_specs": [{}],
            "rows": [{}, {}, {}],
        },
    )
    atomic_json(
        evidence / "coverage.json", {"files": {n: {"summary": v} for n, v in counts.items()}}
    )
    manifest = evidence / "validation_command_manifest.json"
    atomic_json(
        manifest,
        {"commands": [{"name": "success", "argv": ["python"], "scope": "route", "timeout_s": 30}]},
    )
    value = core.prior.base([], [], {})
    value.update(
        experiment_id=7967,
        task_id=core.TASK,
        training_execution_ready_score=1,
        scratch_root_receipt={"path": "/tmp/gone7967", "archive_path": str(evidence)},
        validation_receipts=[{"name": "success", "argv": ["python"], "passed": True}],
        validation_command_manifest_path=str(manifest),
        coverage_statement_counts=counts,
        consumer_selection_rows=[
            {
                "passed": True,
                "gate_path": str(fixture_path),
                "gate_sha256": sha256_file(fixture_path),
            }
        ]
        * 3,
        negative_mutation_receipt={"passed": True},
        training_dependency_hashes={core.OWNED[0]: sha256_file(core.ROOT / core.OWNED[0])},
    )
    value["sample_size_budget"].update(completed=1, independent=0)
    monkeypatch.setattr(
        core, "terminal_checks", lambda *a: {"passed": True, "flagged_adversarial": False}
    )
    monkeypatch.setattr(core, "replay_fit", lambda *a: None)
    core.seal(value, output)
    core.replay(output)
    value["sample_size_budget"]["completed"] = 2
    core.seal(value, output)
    with pytest.raises(ValueError, match="readiness drift"):
        core.replay(output)
    value["sample_size_budget"]["completed"] = 1
    value["consumer_selection_rows"][0]["gate_sha256"] = "sha256:changed"
    core.seal(value, output)
    with pytest.raises(ValueError, match="route primary hash drift"):
        core.replay(output)
    value["training_dependency_hashes"][core.OWNED[0]] = "sha256:changed"
    core.seal(value, output)
    with pytest.raises(ValueError, match="dependency hash drift"):
        core.replay(output)


def test_private_child_environment(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7967-ROOTS: library-created temporary files inherit one private root."""
    import tempfile
    import os

    previous = os.environ.get("TMPDIR")
    with core.private_environment(tmp_path):
        assert Path(tempfile.gettempdir()).is_relative_to(tmp_path)
        assert Path(os.environ["TMPDIR"]).is_relative_to(tmp_path)
    assert os.environ.get("TMPDIR") == previous
    with core.private_environment(tmp_path):
        assert Path(tempfile.gettempdir()).is_relative_to(tmp_path)


@pytest.mark.parametrize("inherited", [False, True])
def test_private_environment_restores_after_child_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inherited: bool
) -> None:
    """SCENARIO-VERIFY-7967-ROOTS: child storage and exceptional cleanup stay private."""
    import os
    import subprocess
    import sys
    import tempfile

    original_cache = tempfile.tempdir
    names = ("TMPDIR", "TMP", "TEMP")
    for name in names:
        if inherited:
            monkeypatch.setenv(name, str(tmp_path / "original"))
        else:
            monkeypatch.delenv(name, raising=False)
    previous = {name: os.environ.get(name) for name in names}
    with pytest.raises(RuntimeError, match="child failed"):
        with core.private_environment(tmp_path):
            result = subprocess.run(
                [sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"],
                check=True,
                capture_output=True,
                text=True,
            )
            assert Path(result.stdout.strip()).is_relative_to(tmp_path)
            assert all(Path(os.environ[name]).is_relative_to(tmp_path) for name in names)
            raise RuntimeError("child failed")
    assert {name: os.environ.get(name) for name in names} == previous
    assert tempfile.tempdir == original_cache
