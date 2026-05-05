"""Tests for Exp 1347 THRML compatibility parity audit.

Spec refs: REQ-SAMPLE-041, SCENARIO-SAMPLE-069.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import carnot.hardware.thrml_compatibility_audit as audit_module
from carnot.hardware.thrml_compatibility_audit import (
    REQUIRED_ARTIFACT_FIELDS,
    ThrmlProbeResult,
    build_artifact,
    probe_thrml,
    run_experiment,
    validate_artifact,
)


class _FakeSpinNode:
    pass


class _FakeBlock:
    def __init__(self, nodes: list[_FakeSpinNode]) -> None:
        self.nodes = list(nodes)


class _FakeIsingEBM:
    def __init__(
        self,
        nodes: list[_FakeSpinNode],
        edges: list[tuple[_FakeSpinNode, _FakeSpinNode]],
        biases: np.ndarray,
        weights: np.ndarray,
        beta: float,
    ) -> None:
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.biases = np.asarray(biases, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.beta = float(beta)
        self._node_index = {node: idx for idx, node in enumerate(self.nodes)}

    def energy(self, state: list[np.ndarray], blocks: list[_FakeBlock]) -> float:
        spins_by_node: dict[_FakeSpinNode, float] = {}
        for block_state, block in zip(state, blocks):
            spin_values = 2.0 * np.asarray(block_state, dtype=np.int8) - 1.0
            for node, spin in zip(block.nodes, spin_values):
                spins_by_node[node] = float(spin)
        spins = np.array([spins_by_node[node] for node in self.nodes], dtype=np.float64)
        edge_total = 0.0
        for weight, (left, right) in zip(self.weights, self.edges):
            edge_total += float(weight) * spins[self._node_index[left]] * spins[self._node_index[right]]
        return -self.beta * (float(self.biases @ spins) + edge_total)


def _fake_thrml_module() -> SimpleNamespace:
    return SimpleNamespace(
        __version__="fake-local",
        SpinNode=_FakeSpinNode,
        Block=_FakeBlock,
        models=SimpleNamespace(IsingEBM=_FakeIsingEBM),
    )


def _available_probe() -> ThrmlProbeResult:
    return ThrmlProbeResult(
        import_available=True,
        module=_fake_thrml_module(),
        version="fake-local",
        import_source="test_fake_thrml",
        local_package_path=None,
        missing_api_or_dependency=None,
    )


def _missing_probe() -> ThrmlProbeResult:
    return ThrmlProbeResult(
        import_available=False,
        module=None,
        version=None,
        import_source="not_importable",
        local_package_path="/home/ianblenke/github.com/ianblenke/thrml",
        missing_api_or_dependency="missing dependency while importing local THRML: equinox",
    )


def test_req_sample_041_missing_thrml_writes_mapping_notes_only() -> None:
    """REQ-SAMPLE-041: missing THRML leaves parity unset and blocks hardware claims."""
    artifact = build_artifact(
        probe=_missing_probe(),
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
    )

    validate_artifact(artifact)
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["thrml_import_available"] is False
    assert artifact["energy_parity_max_abs_error"] is None
    assert artifact["sample_quality_proxy"] is None
    assert "equinox" in artifact["missing_api_or_dependency"]
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["honest_verdict"] == "thrml_unavailable_mapping_notes_only_no_hardware_claim"
    assert {case["case_type"] for case in artifact["cases_attempted"]} == {"ising", "kan"}


def test_req_sample_041_fake_thrml_measures_tiny_ising_energy_parity() -> None:
    """REQ-SAMPLE-041: importable THRML-like API measures exact tiny Ising parity."""
    artifact = build_artifact(
        probe=_available_probe(),
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
    )

    validate_artifact(artifact)
    ising_case = next(case for case in artifact["cases_attempted"] if case["case_type"] == "ising")
    kan_case = next(case for case in artifact["cases_attempted"] if case["case_type"] == "kan")

    assert artifact["thrml_import_available"] is True
    assert artifact["energy_parity_max_abs_error"] == pytest.approx(0.0, abs=1e-12)
    assert artifact["sample_quality_proxy"]["kl_to_local_exact"] == pytest.approx(0.0, abs=1e-12)
    assert artifact["hardware_claim_allowed"] is True
    assert artifact["honest_verdict"] == "thrml_energy_parity_measured_no_tsu_hardware_execution"
    assert ising_case["status"] == "parity_measured"
    assert ising_case["state_count"] == 16
    assert kan_case["status"] == "mapping_notes_only"
    assert "spline" in kan_case["notes"]


def test_scenario_sample_069_run_experiment_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-069: runner writes the validated Exp 1347 JSON artifact."""
    deliverable = tmp_path / "experiment_1347_thrml_compatibility_parity_audit.json"

    artifact = run_experiment(
        deliverable_path=deliverable,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
        probe_func=lambda: _available_probe(),
    )

    payload = json.loads(deliverable.read_text(encoding="utf-8"))
    assert payload == artifact
    assert payload["metadata"]["run_date"] == "20260505"
    assert payload["hardware_claim_allowed"] is True
    assert payload["energy_parity_max_abs_error"] == pytest.approx(0.0, abs=1e-12)


def test_req_sample_041_probe_reports_missing_import_without_install() -> None:
    """REQ-SAMPLE-041: probe uses imports only and records a missing THRML module."""

    def _raise_missing(_name: str) -> object:
        raise ModuleNotFoundError("No module named 'thrml'", name="thrml")

    result = probe_thrml(importer=_raise_missing, local_package_candidates=())

    assert result.import_available is False
    assert result.module is None
    assert result.local_package_path is None
    assert "thrml" in str(result.missing_api_or_dependency)


def test_req_sample_041_probe_uses_default_local_candidate_without_scanning(
    tmp_path: Path,
) -> None:
    """REQ-SAMPLE-041: default local candidate probing is bounded and deterministic."""

    def _raise_missing(_name: str) -> object:
        raise ModuleNotFoundError("No module named 'thrml'", name="thrml")

    project_root = tmp_path / "carnot"
    project_root.mkdir()
    result = probe_thrml(importer=_raise_missing, project_root=project_root)

    assert result.import_available is False
    assert result.local_package_path is None
    assert result.import_source == "not_importable"


def test_req_sample_041_probe_reports_direct_import_success_without_version() -> None:
    """REQ-SAMPLE-041: direct import success does not require package metadata."""
    result = probe_thrml(
        importer=lambda _name: SimpleNamespace(),
        local_package_candidates=(),
    )

    assert result.import_available is True
    assert result.version == "unknown"
    assert result.import_source == "python_import_path"
    assert result.missing_api_or_dependency is None


def test_req_sample_041_probe_can_use_local_checkout_candidate(tmp_path: Path) -> None:
    """REQ-SAMPLE-041: a local checkout can satisfy the probe without installation."""
    checkout = tmp_path / "thrml"
    package = checkout / "thrml"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# local fake THRML\n", encoding="utf-8")
    fake_module = _fake_thrml_module()
    calls = {"count": 0}

    def _missing_then_local(_name: str) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ModuleNotFoundError("No module named 'thrml'", name="thrml")
        return fake_module

    result = probe_thrml(
        importer=_missing_then_local,
        local_package_candidates=(checkout,),
    )

    assert result.import_available is True
    assert result.module is fake_module
    assert result.local_package_path == str(checkout)
    assert result.import_source == "local_source_checkout"


def test_req_sample_041_probe_reports_local_checkout_dependency_failure(tmp_path: Path) -> None:
    """REQ-SAMPLE-041: a local checkout with missing dependencies is not importable."""
    checkout = tmp_path / "thrml"
    package = checkout / "thrml"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# local fake THRML\n", encoding="utf-8")

    def _missing_dependency(_name: str) -> Any:
        raise ModuleNotFoundError("No module named 'equinox'", name="equinox")

    result = probe_thrml(
        importer=_missing_dependency,
        local_package_candidates=(checkout,),
    )

    assert result.import_available is False
    assert result.local_package_path == str(checkout)
    assert "equinox" in str(result.missing_api_or_dependency)


def test_req_sample_041_probe_reports_local_checkout_metadata_and_generic_failures(
    tmp_path: Path,
) -> None:
    """REQ-SAMPLE-041: local import failures keep the blocking reason explicit."""
    checkout = tmp_path / "thrml"
    package = checkout / "thrml"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# local fake THRML\n", encoding="utf-8")

    def _metadata_failure(_name: str) -> Any:
        raise audit_module.importlib.metadata.PackageNotFoundError("thrml")

    metadata_failure = probe_thrml(
        importer=_metadata_failure,
        local_package_candidates=(checkout,),
    )

    def _generic_failure(_name: str) -> Any:
        raise RuntimeError("bad local api")

    generic_failure = probe_thrml(
        importer=_generic_failure,
        local_package_candidates=(checkout,),
    )

    assert "package metadata" in str(metadata_failure.missing_api_or_dependency)
    assert "RuntimeError: bad local api" in str(generic_failure.missing_api_or_dependency)


def test_req_sample_041_local_import_failure_cleans_partial_thrml_modules(
    tmp_path: Path,
) -> None:
    """REQ-SAMPLE-041: failed local imports do not leave partial THRML modules loaded."""
    checkout = tmp_path / "thrml"
    package = checkout / "thrml"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# local fake THRML\n", encoding="utf-8")

    sys.modules.pop("thrml.partial", None)
    calls = {"count": 0}

    def _polluting_failure(_name: str) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ModuleNotFoundError("No module named 'thrml'", name="thrml")
        sys.modules["thrml.partial"] = SimpleNamespace()
        raise RuntimeError("partial import failed")

    result = probe_thrml(
        importer=_polluting_failure,
        local_package_candidates=(checkout,),
    )

    assert result.import_available is False
    assert "partial import failed" in str(result.missing_api_or_dependency)
    assert "thrml.partial" not in sys.modules


def test_req_sample_041_importable_thrml_with_missing_api_is_blocked() -> None:
    """REQ-SAMPLE-041: importable THRML still blocks when Ising APIs are absent."""
    probe = ThrmlProbeResult(
        import_available=True,
        module=SimpleNamespace(),
        version="bad-api",
        import_source="test",
        local_package_path=None,
        missing_api_or_dependency=None,
    )

    artifact = build_artifact(
        probe=probe,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
    )

    assert artifact["hardware_claim_allowed"] is False
    assert artifact["energy_parity_max_abs_error"] is None
    assert artifact["honest_verdict"] == "thrml_import_available_but_api_incompatible_no_hardware_claim"
    assert "SpinNode" in str(artifact["missing_api_or_dependency"])
    assert "Block" in str(artifact["missing_api_or_dependency"])
    assert "models.IsingEBM" in str(artifact["missing_api_or_dependency"])


def test_req_sample_041_missing_thrml_without_local_checkout_uses_import_path_notes() -> None:
    """REQ-SAMPLE-041: notes distinguish no local checkout from local dependency failure."""
    artifact = build_artifact(
        probe=ThrmlProbeResult(
            import_available=False,
            module=None,
            version=None,
            import_source="not_importable",
            local_package_path=None,
            missing_api_or_dependency="missing Python module while importing THRML: thrml",
        ),
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
    )

    assert "no importable local package" in artifact["tsu_mapping_notes"]


def test_req_sample_041_validation_rejects_dishonest_or_incomplete_artifacts() -> None:
    """REQ-SAMPLE-041: validator rejects missing fields and unsupported claims."""
    artifact = build_artifact(
        probe=_missing_probe(),
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
    )

    missing = dict(artifact)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing"):
        validate_artifact(missing)

    dishonest = dict(artifact)
    dishonest["hardware_claim_allowed"] = True
    with pytest.raises(ValueError, match="parity"):
        validate_artifact(dishonest)

    no_dependency = dict(artifact)
    no_dependency["missing_api_or_dependency"] = None
    with pytest.raises(ValueError, match="missing_api_or_dependency"):
        validate_artifact(no_dependency)

    bad_status = dict(artifact)
    bad_status["status"] = "done"
    with pytest.raises(ValueError, match="invalid status"):
        validate_artifact(bad_status)

    in_progress = dict(artifact)
    in_progress["status"] = "in_progress"
    validate_artifact(in_progress)

    bad_verdict = dict(artifact)
    bad_verdict["honest_verdict"] = "claimed_tsu_hardware_without_run"
    with pytest.raises(ValueError, match="invalid honest_verdict"):
        validate_artifact(bad_verdict)

    empty_cases = dict(artifact)
    empty_cases["cases_attempted"] = []
    with pytest.raises(ValueError, match="cases_attempted"):
        validate_artifact(empty_cases)

    parity_without_import = dict(artifact)
    parity_without_import["energy_parity_max_abs_error"] = 0.0
    with pytest.raises(ValueError, match="energy parity"):
        validate_artifact(parity_without_import)

    proxy_without_import = dict(artifact)
    proxy_without_import["sample_quality_proxy"] = {"proxy": "bad"}
    with pytest.raises(ValueError, match="sample_quality_proxy"):
        validate_artifact(proxy_without_import)

    negative_parity = build_artifact(
        probe=_available_probe(),
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
    )
    negative_parity["energy_parity_max_abs_error"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        validate_artifact(negative_parity)
