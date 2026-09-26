"""REQ-REPORT-7710 and SCENARIO-PYBIND-7710-ROUNDTRIP."""

import json
import os
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import pytest

from carnot.pipeline.native_calibrated_decision_service import load_native_extension
from carnot.pipeline.native_record_decision import predict_reference


FEATURES = (
    "tuple_supported",
    "tuple_contradicted",
    "path_line_supported",
    "path_line_contradicted",
    "unknown_propositions",
    "residual_unknown_bytes",
    "checked_fraction",
    "source_records",
)
PARAMETERS = {
    "schema": "carnot.exp7710.record_binary_energy.v1",
    "weights": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
    "bias": -0.25,
    "thresholds": [0.2, 0.8],
}


@pytest.fixture(scope="module")
def extension_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the current PyO3 code for standalone and conductor test runs."""

    configured = os.environ.get("CARNOT_7710_EXTENSION")
    if configured:
        extension = Path(configured).resolve()
    else:
        root = Path(__file__).resolve().parents[2]
        target = Path("/tmp/carnot-exp7710-pytest-target")
        lock_path = Path("/tmp/carnot-exp7710-pytest-build.lock")
        suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
        extension = tmp_path_factory.mktemp("exp7710-native") / f"_rust{suffix}"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            completed = subprocess.run(
                ("cargo", "build", "--release", "-p", "carnot-python", "--target-dir", str(target)),
                cwd=root,
                env={**os.environ, "PYO3_PYTHON": sys.executable},
                check=False,
                capture_output=True,
                text=True,
                timeout=900,
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr
            shutil.copy2(target / "release/libcarnot_python.so", extension)
    assert extension.is_file()
    return extension


def payload(index: int) -> dict:
    """SCENARIO-REPORT-7710-PARITY covers empty and finite extreme evidence."""

    counts = {name: (index * (position + 1)) % 1000 for position, name in enumerate(FEATURES)}
    counts["checked_fraction"] = (index % 101) / 100
    if index == 0:
        counts = dict.fromkeys(FEATURES, 0)
    if index == 1:
        counts["residual_unknown_bytes"] = 1_000_000_000
    return {"schema": "carnot.exp7700.record_features.v1", "counts": counts}


def test_scenario_report_7710_reference_128_cases() -> None:
    """REQ-REPORT-7710 needs finite normalized binary probabilities."""

    for index in range(128):
        probability, action = predict_reference(payload(index), PARAMETERS)
        assert 0 <= probability <= 1
        assert action in {"accept", "escalate", "reject"}


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"schema": "wrong"}, "record_schema_invalid"),
        ({"counts": {}}, "record_fields_invalid"),
        (
            {"counts": {**payload(2)["counts"], "tuple_supported": float("nan")}},
            "record_field_invalid",
        ),
    ],
)
def test_scenario_report_7710_invalid_payloads(change: dict, error: str) -> None:
    """SCENARIO-REPORT-7710-PARITY preserves typed error codes."""

    sample = {**payload(2), **change}
    with pytest.raises(ValueError, match=f"^{error}$"):
        predict_reference(sample, PARAMETERS)


@pytest.mark.parametrize(
    ("sample", "params", "error"),
    [
        (
            {**payload(2), "counts": {**payload(2)["counts"], "checked_fraction": 1.1}},
            PARAMETERS,
            "record_field_invalid",
        ),
        (
            {**payload(2), "counts": {**payload(2)["counts"], "source_records": -1}},
            PARAMETERS,
            "record_field_invalid",
        ),
        (payload(2), {**PARAMETERS, "schema": "wrong"}, "parameter_schema_invalid"),
        (payload(2), {**PARAMETERS, "weights": []}, "parameter_fields_invalid"),
        (payload(2), {**PARAMETERS, "weights": [1e308] * 8}, "energy_not_finite"),
    ],
)
def test_scenario_report_7710_reference_rejects_invalid_fields(
    sample: dict, params: dict, error: str
) -> None:
    """REQ-REPORT-7710 rejects invalid domains and nonfinite energy."""

    with pytest.raises(ValueError, match=f"^{error}$"):
        predict_reference(sample, params)


def test_scenario_pybind_7710_real_extension_and_restart(
    extension_path: Path, tmp_path: Path
) -> None:
    """SCENARIO-PYBIND-7710-ROUNDTRIP and SCENARIO-REPORT-7710-RESTART."""

    extension = extension_path
    binding = load_native_extension(extension)
    assert Path(binding.__file__).resolve() == extension
    assert binding.RustPortableRecalibrationService.__module__ in {"builtins", "carnot._rust"}
    assert (
        binding.__dict__["RustPortableRecalibrationService"]
        is binding.RustPortableRecalibrationService
    )
    state = tmp_path / "service.json"
    native = binding.RustPortableRecalibrationService(str(state))
    assert not state.with_suffix(".records.json").exists()
    for index in range(128):
        sample = payload(index)
        expected = predict_reference(sample, PARAMETERS)
        _, observed, action = native.predict_record(
            f"event-{index}", json.dumps(sample), json.dumps(PARAMETERS)
        )
        assert abs(observed - expected[0]) <= 1e-6
        assert action == expected[1]
    del native
    native = binding.RustPortableRecalibrationService(str(state))
    assert native.record_state_summary()[0] == 0
    assert len(native.record_state_summary()[1]) == 128
    assert native.release_record_feedback("event-0", 1)[1:3] == (True, True)
    assert native.release_record_feedback("event-0", 1)[3] == "duplicate_feedback:event-0"
    assert native.record_state_summary()[0] == 1
    for sample, params, error in [
        ({**payload(2), "schema": "wrong"}, PARAMETERS, "record_schema_invalid"),
        (payload(2), {**PARAMETERS, "schema": "wrong"}, "parameter_schema_invalid"),
        ({**payload(2), "counts": {}}, PARAMETERS, "record_fields_invalid"),
    ]:
        with pytest.raises(ValueError, match=f"^{error}$"):
            native.predict_record("invalid", json.dumps(sample), json.dumps(params))


def test_scenario_report_7710_crash_replay_and_version(
    extension_path: Path, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7710-RESTART reopens an owned child after hard exit."""

    extension = extension_path
    state = tmp_path / "crash.json"
    script = (
        "import os,sys\n"
        "from carnot.pipeline.native_calibrated_decision_service import load_native_extension\n"
        "b=load_native_extension(sys.argv[1])\n"
        "s=b.RustPortableRecalibrationService(sys.argv[2])\n"
        "s.predict_record('crash',sys.argv[3],sys.argv[4])\n"
        "os._exit(86)\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(extension),
            str(state),
            json.dumps(payload(2)),
            json.dumps(PARAMETERS),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 86
    binding = load_native_extension(extension)
    native = binding.RustPortableRecalibrationService(str(state))
    assert native.record_state_summary()[1] == ["crash"]
    assert native.release_record_feedback("crash", 1)[1:3] == (True, True)
    assert native.release_record_feedback("crash", 1)[3] == "duplicate_feedback:crash"
    sidecar = state.with_suffix(".records.json")
    data = json.loads(sidecar.read_text())
    data["schema"] = "wrong"
    sidecar.write_text(json.dumps(data))
    reopened = binding.RustPortableRecalibrationService(str(state))
    with pytest.raises(ValueError, match="record_state_version_mismatch"):
        reopened.record_state_summary()
