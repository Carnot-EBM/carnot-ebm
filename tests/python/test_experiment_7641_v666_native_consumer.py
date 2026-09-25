"""Focused checks for REQ-REPORT-7641 and REQ-PIPELINE-7641."""

from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import pytest

from carnot import experiment_7626_v665_native_service as exp7626
from carnot import experiment_7641_v666_native_consumer as exp
from carnot.pipeline import native_calibrated_decision_service as native


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def extension_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the real extension when no authenticated producer path remains."""

    configured = os.environ.get("CARNOT_EXP7626_EXTENSION")
    if configured and Path(configured).is_file():
        return Path(configured).resolve()
    target = Path("/tmp/carnot-exp7641-pytest-target")
    lock_path = Path("/tmp/carnot-exp7641-pytest-build.lock")
    suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
    destination = tmp_path_factory.mktemp("exp7641-native") / f"_rust{suffix}"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        completed = subprocess.run(
            (
                "cargo",
                "build",
                "--release",
                "-p",
                "carnot-python",
                "--target-dir",
                str(target),
            ),
            cwd=ROOT,
            env={**os.environ, "PYO3_PYTHON": sys.executable},
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        shutil.copy2(target / "release/libcarnot_python.so", destination)
    return destination


def test_scenario_pipeline_7641_native_lifecycle_is_durable(
    extension_path: Path, tmp_path: Path
) -> None:
    """SCENARIO-PIPELINE-7641-NATIVE crosses real PyO3 and cold reload."""

    state = tmp_path / "state.json"
    client = native.NativeServiceClient.from_extension(
        state_path=state, extension_path=extension_path
    )
    decision = client.predict("one", 0.31)
    acknowledgment = client.release_feedback("one", 1)
    assert isinstance(decision, native.CalibratedDecision)
    assert isinstance(acknowledgment, native.FeedbackAcknowledgment)
    assert decision.available and decision.verified is False
    assert acknowledgment.available and acknowledgment.durable
    assert client.state_summary()["sample_count"] == 1
    client.close()
    assert client.predict("closed", 0.2).error == "native_client_closed"

    reopened = native.NativeServiceClient.from_extension(
        state_path=state, extension_path=extension_path
    )
    summary = reopened.state_summary()
    assert summary["available"] is True
    assert summary["processed_event_ids"] == ["one"]


def test_scenario_report_7641_lifecycle_rejects_invalid_and_duplicate_calls(
    extension_path: Path, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7641-LIFECYCLE enforces pending and released IDs."""

    client = native.NativeServiceClient.from_extension(
        state_path=tmp_path / "lifecycle.json", extension_path=extension_path
    )
    for value in (float("nan"), -0.1, 1.1, "not-a-number"):
        decision = client.predict("invalid", value)  # type: ignore[arg-type]
        assert not decision.available
        assert decision.action == "escalate"
        assert decision.error == "finite_probability_required"
    assert client.predict("", 0.2).error == "event_id_required"
    assert client.predict("pending", 0.2).available
    assert client.predict("pending", 0.2).error == "duplicate_prediction:pending"
    assert client.release_feedback("missing", 1).error == "unknown_prediction:missing"
    assert client.release_feedback("pending", True).error == "binary_label_required"
    assert client.release_feedback("pending", 1).durable
    assert client.release_feedback("pending", 1).error == "duplicate_feedback:pending"
    assert client.predict("pending", 0.2).error == "duplicate_feedback:pending"


def test_scenario_pipeline_7641_failure_has_no_python_substitute(tmp_path: Path) -> None:
    """SCENARIO-PIPELINE-7641-FAILURE returns typed unavailable results."""

    client = native.NativeServiceClient.from_extension(
        state_path=tmp_path / "missing.json", extension_path=tmp_path / "missing.so"
    )
    decision = client.predict("event", 0.4)
    acknowledgment = client.release_feedback("event", 1)
    assert decision.available is False and decision.verified is False
    assert decision.action == "escalate"
    assert decision.error and decision.error.startswith("native_extension_unavailable:")
    assert acknowledgment.available is False
    assert acknowledgment.acknowledged is False and acknowledgment.durable is False
    assert client.state_summary()["available"] is False


def test_scenario_report_7641_unavailable_covers_corrupt_state(
    extension_path: Path, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7641-UNAVAILABLE keeps corrupt state explicit."""

    state = tmp_path / "corrupt.json"
    state.write_text("not-json", encoding="utf-8")
    client = native.NativeServiceClient.from_extension(
        state_path=state, extension_path=extension_path
    )
    result = client.predict("event", 0.3)
    assert result.available is False
    assert result.error and result.error.startswith("native_service_unavailable:")


def test_scenario_report_7641_unavailable_rejects_native_contract_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7641-UNAVAILABLE rejects mismatched IDs and false acks."""

    class BadService:
        def __init__(self, _state: str) -> None:
            pass

        def predict(self, event_id: str, _probability: float) -> tuple[str, float, str]:
            if event_id == "raises":
                raise ValueError("native_predict_failed")
            return "wrong-id", 0.2, "accept"

        def release_feedback(self, event_id: str, _label: int) -> tuple[str, bool, bool, None]:
            return f"{event_id}-wrong", True, True, None

        def state_summary(self) -> tuple[int, list[str], str]:
            return 0, [], "carnot.recalibration.sufficient_statistics.v1"

    class BadBinding:
        RustPortableRecalibrationService = BadService

    client = native.NativeServiceClient(binding=BadBinding, state_path=tmp_path / "state.json")
    assert client.predict("drift", 0.2).error == "prediction_contract_invalid"
    assert client.predict("raises", 0.2).error == "native_predict_failed"

    class BadAckService(BadService):
        def predict(self, event_id: str, probability: float) -> tuple[str, float, str]:
            return event_id, probability, "escalate"

    class BadAckBinding:
        RustPortableRecalibrationService = BadAckService

    ack_client = native.NativeServiceClient(binding=BadAckBinding, state_path=tmp_path / "ack.json")
    assert ack_client.predict("ack", 0.2).available
    result = ack_client.release_feedback("ack", 1)
    assert result.error == "durable_acknowledgment_invalid"
    assert result.durable is False


def test_scenario_report_7641_interrupted_write_preserves_state(
    extension_path: Path, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7641-PACKAGE retains Exp7626 interruption behavior."""

    state = tmp_path / "interrupted.json"
    client = native.NativeServiceClient.from_extension(
        state_path=state, extension_path=extension_path
    )
    assert client.predict("kept", 0.21).available
    assert client.release_feedback("kept", 1).durable
    before = exp7626.sha256_file(state)
    row = exp7626.run_interrupted_write_probe(ROOT, extension_path, state)
    assert row["exit_code"] == exp7626.INTERRUPTED_WRITE_EXIT
    assert row["prior_state_survived"] is True
    assert exp7626.sha256_file(state) == before


def test_req_report_7641_reducer_requires_every_integration_unit() -> None:
    """REQ-REPORT-7641 reduces only complete typed integration rows."""

    rows = exp.synthetic_integration_rows()
    reduced = exp.reduce_integration_rows(rows)
    assert reduced == {
        "complete": True,
        "independent_units": 12,
        "passed_units": 12,
        "failed_units": 0,
        "real_native_units": 4,
        "unavailable_units": 4,
        "durability_units": 4,
    }
    missing = deepcopy(rows[:-1])
    with pytest.raises(ValueError, match="integration_units_missing"):
        exp.reduce_integration_rows(missing)
    wrong = deepcopy(rows)
    wrong[0]["passed"] = False
    with pytest.raises(ValueError, match="integration_unit_failed"):
        exp.reduce_integration_rows(wrong)


def test_req_report_7641_artifact_contract_and_mutations(tmp_path: Path) -> None:
    """REQ-REPORT-7641 keeps readiness separate from historical speed."""

    artifact = exp.build_test_artifact(ROOT, tmp_path)
    assert exp.validate_artifact(artifact, check_files=False) == []
    assert artifact["honest_verdict"] == "complete_null_native_consumer_ready"
    assert artifact["verdict_class"] == "null"
    assert artifact["native_consumer_ready_score"] == 1
    assert artifact["new_speed_claim"] is False
    assert artifact["production_defaults_changed"] is False
    assert artifact["historical_speed_evidence"]["python_over_native"] == pytest.approx(7.8270)
    assert artifact["historical_speed_evidence"]["nfr_10x_met"] is False
    assert {row["category"] for row in artifact["acceptance_gate_results"]} == {
        "validity",
        "readiness",
        "probability_benefit",
        "utility",
        "retention",
        "freshness",
    }

    for field, value, expected_error in (
        ("new_speed_claim", True, "new_speed_claim"),
        ("native_consumer_ready_score", 0, "ready_score"),
        ("MODEL_SPECS", [{"name": "forbidden"}], "model_specs"),
        ("production_defaults_changed", True, "production_defaults"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert expected_error in exp.validate_artifact(changed, check_files=False)


def test_req_report_7641_hardware_and_blocked_contract(tmp_path: Path) -> None:
    """REQ-REPORT-7641 preserves board scopes and exact blocked operands."""

    dispositions = exp.hardware_dispositions(ROOT)
    by_name = {row["hardware"]: row for row in dispositions}
    assert by_name["KV260"]["k_max"] == 5
    assert by_name["PolarFire"]["claim_scope"] == "linux_cpu_dispatch_only"
    assert by_name["GateMate"]["last_observed"] == "0xffffffff"
    assert by_name["NPU"]["qualified"] is False
    assert by_name["TSU"]["qualified"] is False

    blocker = {
        "check": "native_extension",
        "upstream": "Exp7626",
        "path": "/tmp/missing.so",
        "field": "exists",
        "operator": "eq",
        "expected": True,
        "observed": False,
    }
    blocked = exp.build_blocked_artifact(ROOT, blocker, tmp_path)
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"] == blocker


def test_req_report_7641_preconditions_authenticate_upstream_and_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7641 names exact receipt and extension checks before work."""

    context = exp.collect_preconditions(ROOT)
    assert context["blocker"] is None
    assert Path(context["native_extension"]).is_file()
    assert all(row["passed"] for row in context["rows"])
    assert exp.historical_speed_evidence(context["exp7627"])["python_over_native"] == pytest.approx(
        7.827002746229495
    )
    assert exp._check("truthy", "test", "path", "field", "truthy", True, 1)["passed"]

    monkeypatch.setattr(exp, "EXP7626_PATH", Path("results/absent-exp7626.json"))
    blocked = exp.collect_preconditions(ROOT)
    assert blocked["blocker"]["check"] == "named_input"
    assert blocked["blocker"]["observed"] is False


def test_req_report_7641_reducer_and_json_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7641 rejects wrong groups and non-object artifact bytes."""

    rows = exp.synthetic_integration_rows()
    rows[0]["group"] = "wrong"
    with pytest.raises(ValueError, match="integration_group_invalid"):
        exp.reduce_integration_rows(rows)
    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object_required"):
        exp._load_object(path)


def test_req_report_7641_validator_rejects_all_claim_boundaries(tmp_path: Path) -> None:
    """REQ-REPORT-7641 terminal validation fails closed for every governed class."""

    baseline = exp.build_test_artifact(ROOT, tmp_path)

    def rejected(mutator: object, expected: str, *, refresh: bool = True) -> None:
        changed = deepcopy(baseline)
        assert callable(mutator)
        mutator(changed)
        if refresh:
            changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert expected in exp.validate_artifact(changed, check_files=False)

    rejected(lambda value: value.__setitem__("honest_verdict", "unfinished"), "honest_verdict")
    rejected(lambda value: value.__setitem__("verdict_class", "other"), "verdict_class")
    rejected(lambda value: value.__setitem__("model_invoked", True), "model_invocation")
    rejected(lambda value: value.__setitem__("verifier_is_oracle", False), "oracle_declaration")
    rejected(lambda value: value["integration_rows"].pop(), "integration_rows")
    rejected(
        lambda value: value["integration_reduction"].__setitem__("passed_units", 11),
        "integration_reduction",
    )
    rejected(lambda value: value.__setitem__("native_consumer_ready_score", 0), "ready_score")
    rejected(lambda value: value.__setitem__("historical_speed_evidence", None), "historical_speed")
    rejected(
        lambda value: value["historical_speed_evidence"].__setitem__("python_over_native", 9.0),
        "historical_python_ratio",
    )
    rejected(
        lambda value: value["historical_speed_evidence"].__setitem__("jsonl_over_native", 9.0),
        "historical_jsonl_ratio",
    )
    rejected(
        lambda value: value["historical_speed_evidence"].__setitem__("nfr_10x_met", True),
        "historical_nfr",
    )
    rejected(lambda value: value["acceptance_gate_results"].pop(), "acceptance_gate_categories")
    rejected(lambda value: value["hardware_dispositions"].pop(), "hardware_dispositions")
    rejected(
        lambda value: value["hardware_dispositions"][0].__setitem__("k_max", 6),
        "hardware_scope",
    )
    rejected(lambda value: value.__setitem__("field_principles", {}), "field_principles")
    rejected(
        lambda value: value.__setitem__("reproducibility_checksum", "sha256:wrong"),
        "reproducibility_checksum",
        refresh=False,
    )

    blocked = exp.build_blocked_artifact(
        ROOT,
        {
            "check": "missing",
            "upstream": "test",
            "path": "missing",
            "field": "exists",
            "operator": "eq",
            "expected": True,
            "observed": False,
        },
        tmp_path,
    )
    blocked["gate_check_summary"] = None
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    assert "blocked_gate_summary" in exp.validate_artifact(blocked, check_files=False)


def test_scenario_report_7641_terminal_replays_and_commands(
    extension_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-7641-TERMINAL freezes scoped commands and cold readers."""

    artifact = exp.build_test_artifact(ROOT, tmp_path)
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.cold_replay(path)["valid"] is True
    assert exp.independent_replay(path)["valid"] is True

    broken = deepcopy(artifact)
    broken.pop("integration_rows")
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    assert exp.independent_replay(broken_path)["valid"] is False

    commands = exp.build_validation_commands(ROOT, tmp_path / "private", extension_path)
    names = {command.name for command in commands}
    assert {
        "focused_pytest",
        "changed_module_coverage_report",
        "private_package_install",
        "installed_style_native_import",
        "e2e_003_real_pyo3",
        "e2e_004_json_interrupt_reload",
    }.issubset(names)
    assert len(exp.terminal_commands(path, ROOT)) == 4
    span = exp._span("unit", 1.0, 0.5, 2)
    assert span["completed_units"] == 2 and span["pending_operations"] == 0
    assert exp._all_passed([]) is False
    assert exp._all_passed([{"passed": True}]) is True

    assert exp.main(["--cold-replay", str(path)]) == 0
    assert "artifact_sha256" in capsys.readouterr().out
    assert exp.main(["--independent-replay", str(path)]) == 0
    assert "integration_reduction" in capsys.readouterr().out
    calls: list[tuple[Path, str, Path]] = []
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root, date, output: calls.append((root, date, output)),
    )
    assert exp.main(["--root", str(tmp_path), "--output", "out.json"]) == 0
    assert calls == [(tmp_path.resolve(), exp.RUN_DATE, Path("out.json"))]


def test_req_report_7641_source_hash_failure_is_visible(tmp_path: Path) -> None:
    """REQ-REPORT-7641 rejects changed producer bytes during exact replay."""

    artifact = exp.build_test_artifact(ROOT, tmp_path)
    source = next(
        receipt
        for receipt in artifact["source_artifact_hashes"].values()
        if receipt["role"] != "planned_output_not_input" and receipt["exists"]
    )
    source["sha256"] = "sha256:" + "0" * 64
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    errors = exp.validate_artifact(artifact)
    assert any(error.startswith("source_hash:") for error in errors)


def test_scenario_pipeline_7641_native_loader_and_exception_edges(
    extension_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-PIPELINE-7641-FAILURE covers loader and native call exceptions."""

    real_module = native.load_native_extension(extension_path)
    monkeypatch.setattr(native.importlib, "import_module", lambda _name: real_module)
    assert native.load_native_extension() is real_module

    monkeypatch.setattr(native.importlib.util, "spec_from_file_location", lambda *_args: None)
    with pytest.raises(ImportError, match="native_loader_unavailable"):
        native.load_native_extension(extension_path)

    class EmptyModule:
        pass

    monkeypatch.setattr(native.importlib, "import_module", lambda _name: EmptyModule())
    with pytest.raises(ImportError, match="native_service_class_missing"):
        native.load_native_extension()

    absent = native.NativeServiceClient(None, tmp_path / "absent.json")
    assert (
        absent.predict("event", 0.2).error
        == "native_service_unavailable:native_service_class_missing"
    )

    class RaisingService:
        def __init__(self, _path: str) -> None:
            pass

        def state_summary(self) -> tuple[int, list[str], str]:
            raise ValueError("summary_failed")

    class RaisingBinding:
        RustPortableRecalibrationService = RaisingService

    raising = native.NativeServiceClient(RaisingBinding, tmp_path / "raising.json")
    assert raising.state_summary()["error"] == "native_service_unavailable:summary_failed"


def test_scenario_pipeline_7641_release_and_summary_exceptions(tmp_path: Path) -> None:
    """SCENARIO-PIPELINE-7641-FAILURE converts live native exceptions to typed errors."""

    class Service:
        def __init__(self, _path: str) -> None:
            self.raise_summary = False

        def state_summary(self) -> tuple[int, list[str], str]:
            if self.raise_summary:
                raise ValueError("summary_broke")
            return 0, [], "schema"

        def predict(self, event_id: str, probability: float) -> tuple[str, float, str]:
            return event_id, probability, "escalate"

        def release_feedback(self, _event_id: str, _label: int) -> tuple[str, bool, bool, None]:
            raise ValueError("release_broke")

    class Binding:
        RustPortableRecalibrationService = Service

    client = native.NativeServiceClient(Binding, tmp_path / "state.json")
    assert client.predict("released", 0.2).available
    assert client.release_feedback("released", 1).error == "release_broke"
    client._inner.raise_summary = True
    assert client.state_summary()["error"] == "summary_broke"
