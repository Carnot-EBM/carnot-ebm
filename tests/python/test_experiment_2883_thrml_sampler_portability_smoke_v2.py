"""Tests for Exp 2883 THRML sampler portability smoke v2.

Spec traces: REQ-SAMPLE-067, SCENARIO-SAMPLE-095.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import carnot.samplers.thrml_sampler_portability_smoke_v2 as exp2883


def _fake_sampler(
    case: Any,
    *,
    seed: int,
    n_samples: int,
    schedule: dict[str, Any],
) -> np.ndarray:
    del schedule
    rng = np.random.default_rng(int(seed))
    return rng.random((int(n_samples), int(case.n_spins))) > 0.5


def _base_preconditions(*, thrml_available: bool) -> dict[str, Any]:
    return {
        "python_version": "3.test",
        "python_executable": "/venv/bin/python",
        "platform": "test-platform",
        "jax_available": True,
        "jax_version": "0.test",
        "jax_devices": ["cpu:0"],
        "jax_default_backend": "cpu",
        "thrml_import_available": thrml_available,
        "thrml_import_error": None
        if thrml_available
        else "ModuleNotFoundError: No module named 'thrml'",
        "thrml_version": "0.test" if thrml_available else None,
        "thrml_import_path": "/venv/thrml/__init__.py" if thrml_available else None,
        "local_fallback_available": True,
        "local_fallback_error": None,
    }


def test_req_sample_067_spec_anchor_exists() -> None:
    """REQ-SAMPLE-067, SCENARIO-SAMPLE-095: Exp2883 is spec-anchored."""

    spec = (exp2883.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-067" in spec
    assert "SCENARIO-SAMPLE-095" in spec
    assert "experiment_2883_thrml_sampler_portability_smoke_v2.json" in spec
    assert "blocked_thrml_unavailable" in spec


def test_probe_preconditions_records_available_thrml_and_import_errors() -> None:
    """REQ-SAMPLE-067: preconditions capture success and failure details."""

    def importer(name: str) -> Any:
        if name == "jax":
            raise RuntimeError("jax failed")
        if name == "thrml":
            return SimpleNamespace(__version__="0.fake", __file__="/fake/thrml/__init__.py")
        raise ImportError("backend failed")

    preconditions = exp2883.probe_preconditions(importer=importer)

    assert exp2883._device_label("cpu:0") == "cpu:0"
    assert preconditions["jax_available"] is False
    assert preconditions["jax_error"] == "RuntimeError: jax failed"
    assert preconditions["thrml_import_available"] is True
    assert preconditions["thrml_version"] == "0.fake"
    assert preconditions["thrml_import_path"] == "/fake/thrml/__init__.py"
    assert preconditions["local_fallback_available"] is False
    assert preconditions["local_fallback_error"] == "ImportError: backend failed"


def test_probe_preconditions_records_thrml_absence_and_jax_devices() -> None:
    """REQ-SAMPLE-067: preconditions record Python, JAX, THRML, and fallback state."""

    def importer(name: str) -> Any:
        if name == "jax":
            return SimpleNamespace(
                __version__="0.test",
                devices=lambda: [SimpleNamespace(platform="cpu", id=0)],
                default_backend=lambda: "cpu",
            )
        if name == "carnot.samplers.backend":
            return SimpleNamespace(CpuBackend=object)
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    preconditions = exp2883.probe_preconditions(importer=importer)

    assert preconditions["thrml_import_available"] is False
    assert preconditions["thrml_import_error"] == "ModuleNotFoundError: No module named 'thrml'"
    assert preconditions["jax_available"] is True
    assert preconditions["jax_devices"] == ["cpu:0"]
    assert preconditions["local_fallback_available"] is True
    assert "python_version" in preconditions


def test_scenario_sample_095_thrml_absent_writes_blocked_artifact_with_local_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAMPLE-095: absent THRML blocks cleanly and still runs fallback."""

    artifact = exp2883.run_sampler_portability_smoke(
        output_path=tmp_path / "experiment_2883.json",
        preconditions=_base_preconditions(thrml_available=False),
        local_sampler=_fake_sampler,
        thrml_sampler=lambda *args, **kwargs: pytest.fail("THRML lane must not run"),
        sample_count=12,
    )

    exp2883.validate_artifact(artifact)
    assert artifact["thrml_portability_ready"] is False
    assert artifact["blocked_reason"] == "blocked_thrml_unavailable"
    assert artifact["thrml_import_available"] is False
    assert artifact["local_fallback_ran"] is True
    assert artifact["hardware_claim_made"] is False
    assert artifact["sample_count"] == 12
    assert artifact["problem_spec"]["n_spins"] == 4
    assert artifact["parity_metrics"]["local"]["sample_shape"] == [12, 4]
    assert artifact["parity_metrics"]["thrml"]["ran"] is False
    assert artifact["parity_metrics"]["shape_match"] is None
    assert artifact["field_principles"]["no_pip_install_attempted"] is True


def test_scenario_sample_095_thrml_available_compares_shapes_histograms_and_runtime(
    tmp_path: Path,
) -> None:
    """REQ-SAMPLE-067: available THRML lane is compared against the fallback lane."""

    artifact = exp2883.run_sampler_portability_smoke(
        output_path=tmp_path / "experiment_2883.json",
        preconditions=_base_preconditions(thrml_available=True),
        local_sampler=_fake_sampler,
        thrml_sampler=_fake_sampler,
        sample_count=16,
        tests_run=[
            ".venv/bin/pytest tests/python/test_experiment_2883_thrml_sampler_portability_smoke_v2.py -q"
        ],
    )
    written = json.loads((tmp_path / "experiment_2883.json").read_text(encoding="utf-8"))

    exp2883.validate_artifact(artifact)
    assert written == artifact
    assert artifact["thrml_portability_ready"] is True
    assert artifact["blocked_reason"] == "none"
    assert artifact["thrml_import_available"] is True
    assert artifact["local_fallback_ran"] is True
    assert artifact["parity_metrics"]["thrml"]["ran"] is True
    assert artifact["parity_metrics"]["shape_match"] is True
    assert artifact["parity_metrics"]["histogram_sanity_passed"] is True
    assert artifact["parity_metrics"]["mean_energy_delta_abs"] is not None
    assert artifact["parity_metrics"]["local"]["scheduled_spin_updates"] > 0
    assert artifact["parity_metrics"]["thrml"]["acceptance_count_available"] is False
    assert artifact["tests_run"]


def test_req_sample_067_local_sampler_failure_is_terminal_blocker(tmp_path: Path) -> None:
    """REQ-SAMPLE-067: fallback failures are recorded without hardware claims."""

    def failing_sampler(*args: Any, **kwargs: Any) -> np.ndarray:
        del args, kwargs
        raise RuntimeError("fallback exploded")

    artifact = exp2883.run_sampler_portability_smoke(
        output_path=tmp_path / "experiment_2883.json",
        preconditions=_base_preconditions(thrml_available=True),
        local_sampler=failing_sampler,
        thrml_sampler=_fake_sampler,
        sample_count=8,
    )

    exp2883.validate_artifact(artifact)
    assert artifact["thrml_portability_ready"] is False
    assert artifact["blocked_reason"] == "local_fallback_failed"
    assert artifact["local_fallback_ran"] is False
    assert "fallback exploded" in artifact["parity_metrics"]["local"]["error"]
    assert artifact["hardware_claim_made"] is False


def test_req_sample_067_thrml_failure_and_shape_mismatch_are_terminal_blocks(
    tmp_path: Path,
) -> None:
    """REQ-SAMPLE-067: THRML run failures and mismatched shapes are not readiness."""

    def failing_thrml(*args: Any, **kwargs: Any) -> np.ndarray:
        del args, kwargs
        raise RuntimeError("thrml exploded")

    def wrong_shape(
        case: Any, *, seed: int, n_samples: int, schedule: dict[str, Any]
    ) -> np.ndarray:
        del case, seed, schedule
        return np.zeros((int(n_samples), 3), dtype=bool)

    failed = exp2883.run_sampler_portability_smoke(
        output_path=tmp_path / "failed.json",
        preconditions=_base_preconditions(thrml_available=True),
        local_sampler=_fake_sampler,
        thrml_sampler=failing_thrml,
        sample_count=8,
    )
    malformed = exp2883.run_sampler_portability_smoke(
        output_path=tmp_path / "mismatched.json",
        preconditions=_base_preconditions(thrml_available=True),
        local_sampler=_fake_sampler,
        thrml_sampler=wrong_shape,
        sample_count=8,
    )

    assert failed["blocked_reason"] == "thrml_sampler_failed"
    assert (
        failed["honest_verdict"]
        == "complete: thrml_sampler_portability_smoke_blocked_no_hardware_claim"
    )
    assert malformed["blocked_reason"] == "thrml_sampler_failed"
    assert "expected (8, 4)" in malformed["parity_metrics"]["thrml"]["error"]
    assert (
        exp2883._blocked_reason(
            preconditions={"thrml_import_available": True},
            local_row={"ran": True},
            thrml_row={"ran": True},
            comparison={"shape_match": False, "histogram_sanity_passed": True},
        )
        == "parity_sanity_failed"
    )


def test_req_sample_067_constant_samples_cover_degenerate_histogram_path(
    tmp_path: Path,
) -> None:
    """REQ-SAMPLE-067: a degenerate local chain still reports a valid histogram."""

    def constant_sampler(
        case: Any, *, seed: int, n_samples: int, schedule: dict[str, Any]
    ) -> np.ndarray:
        del seed, schedule
        return np.zeros((int(n_samples), int(case.n_spins)), dtype=bool)

    artifact = exp2883.run_sampler_portability_smoke(
        output_path=tmp_path / "constant.json",
        preconditions=_base_preconditions(thrml_available=False),
        local_sampler=constant_sampler,
        sample_count=6,
    )

    histogram = artifact["parity_metrics"]["local"]["energy_histogram"]
    assert histogram["nonempty_bins"] == 1
    assert histogram["total_count"] == 6


def test_req_sample_067_default_local_sampler_wrapper_uses_carnot_sampler(monkeypatch: Any) -> None:
    """REQ-SAMPLE-067: the default local wrapper delegates to Carnot's CPU sampler."""

    def fake_carnot(
        case: Any, *, seed: int, n_samples: int, schedule: dict[str, Any]
    ) -> np.ndarray:
        assert seed == 7
        assert n_samples == 3
        assert schedule["beta"] == case.beta
        return np.ones((3, int(case.n_spins)), dtype=bool)

    monkeypatch.setattr(exp2883, "carnot_cpu_sampler", fake_carnot)
    case = exp2883.tiny_portability_case()
    samples = exp2883.local_fallback_sampler(
        case,
        seed=7,
        n_samples=3,
        schedule={"beta": case.beta},
    )

    assert samples.shape == (3, 4)


def test_req_sample_067_validator_rejects_invalid_artifacts() -> None:
    """REQ-SAMPLE-067: validation protects schema and no-hardware boundaries."""

    artifact = exp2883.run_sampler_portability_smoke(
        output_path=None,
        preconditions=_base_preconditions(thrml_available=False),
        local_sampler=_fake_sampler,
        sample_count=4,
    )
    exp2883.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("duration_s")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp2883.validate_artifact(missing)

    hardware = dict(artifact)
    hardware["hardware_claim_made"] = True
    with pytest.raises(ValueError, match="hardware_claim_made"):
        exp2883.validate_artifact(hardware)

    bad_ready = dict(artifact)
    bad_ready["thrml_portability_ready"] = True
    with pytest.raises(ValueError, match="requires THRML import"):
        exp2883.validate_artifact(bad_ready)

    for field, value, message in [
        ("run_date", "20250101", "run_date"),
        ("preconditions_checked", "python", "preconditions_checked"),
        ("jax_devices", "cpu:0", "jax_devices"),
        ("tests_run", "pytest", "tests_run"),
        ("duration_s", -1.0, "duration_s"),
    ]:
        invalid = dict(artifact)
        invalid[field] = value
        with pytest.raises(ValueError, match=message):
            exp2883.validate_artifact(invalid)

    ready_missing_local = dict(artifact)
    ready_missing_local.update(
        {
            "thrml_portability_ready": True,
            "thrml_import_available": True,
            "local_fallback_ran": False,
        }
    )
    with pytest.raises(ValueError, match="local fallback"):
        exp2883.validate_artifact(ready_missing_local)

    ready_missing_thrml = dict(artifact)
    ready_missing_thrml.update(
        {
            "thrml_portability_ready": True,
            "thrml_import_available": True,
            "local_fallback_ran": True,
        }
    )
    with pytest.raises(ValueError, match="THRML sampler"):
        exp2883.validate_artifact(ready_missing_thrml)

    ready_bad_shape = dict(ready_missing_thrml)
    ready_bad_shape["parity_metrics"] = {
        **artifact["parity_metrics"],
        "thrml": {"ran": True},
        "shape_match": False,
        "histogram_sanity_passed": True,
    }
    with pytest.raises(ValueError, match="matching sample shape"):
        exp2883.validate_artifact(ready_bad_shape)

    ready_bad_histogram = dict(ready_bad_shape)
    ready_bad_histogram["parity_metrics"] = {
        **ready_bad_shape["parity_metrics"],
        "shape_match": True,
        "histogram_sanity_passed": False,
    }
    with pytest.raises(ValueError, match="histogram sanity"):
        exp2883.validate_artifact(ready_bad_histogram)
