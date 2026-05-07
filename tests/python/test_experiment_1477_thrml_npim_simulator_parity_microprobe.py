"""Tests for Exp 1477 THRML/NPIM simulator parity microprobe.

Spec traces: REQ-SAMPLE-042, SCENARIO-SAMPLE-070.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import carnot.samplers.thrml_npim_microprobe as exp1477
from carnot.samplers.thrml_npim_microprobe import (
    REQUIRED_ARTIFACT_FIELDS,
    ThrmlImportProbe,
    build_toy_ising_cases,
    exact_case_reference,
    measure_thrml_parity_cases,
    probe_thrml_import,
    run_carnot_sampler_cases,
    run_microprobe,
    run_npim_probe,
    validate_artifact,
    write_in_progress_artifact,
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
        self._node_index = {node: index for index, node in enumerate(nodes)}

    def energy(self, state: list[np.ndarray], blocks: list[_FakeBlock]) -> float:
        spins_by_node: dict[_FakeSpinNode, float] = {}
        for block_state, block in zip(state, blocks):
            spins = 2.0 * np.asarray(block_state, dtype=np.int8) - 1.0
            for node, spin in zip(block.nodes, spins):
                spins_by_node[node] = float(spin)
        spin_vec = np.asarray([spins_by_node[node] for node in self.nodes], dtype=np.float64)
        edge_total = 0.0
        for weight, (left, right) in zip(self.weights, self.edges):
            edge_total += float(weight) * spin_vec[self._node_index[left]] * spin_vec[
                self._node_index[right]
            ]
        return -self.beta * (float(self.biases @ spin_vec) + edge_total)


def _fake_thrml_module() -> SimpleNamespace:
    return SimpleNamespace(
        __version__="fake-thrml",
        SpinNode=_FakeSpinNode,
        Block=_FakeBlock,
        models=SimpleNamespace(IsingEBM=_FakeIsingEBM),
    )


def _available_probe() -> ThrmlImportProbe:
    return ThrmlImportProbe(
        available=True,
        module=_fake_thrml_module(),
        version="fake-thrml",
        missing_api_or_dependency=None,
    )


def _missing_probe() -> ThrmlImportProbe:
    return ThrmlImportProbe(
        available=False,
        module=None,
        version=None,
        missing_api_or_dependency="missing Python module while importing THRML: thrml",
    )


def _ground_state_sampler(
    case: exp1477.IsingCase,
    *,
    seed: int,
    n_samples: int,
    n_warmup: int,
    steps_per_sample: int,
) -> np.ndarray:
    del seed, n_warmup, steps_per_sample
    reference = exact_case_reference(case)
    return np.tile(np.asarray(reference["ground_state"], dtype=np.int8), (n_samples, 1))


def test_req_sample_042_spec_anchor_exists() -> None:
    """REQ-SAMPLE-042, SCENARIO-SAMPLE-070: the microprobe is spec-anchored."""

    spec = (exp1477.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-042" in spec
    assert "SCENARIO-SAMPLE-070" in spec
    assert "experiment_1477_thrml_npim_simulator_parity_microprobe.json" in spec


def test_req_sample_042_writes_in_progress_artifact(tmp_path: Path) -> None:
    """REQ-SAMPLE-042: the deliverable starts with an in-progress marker."""

    output = tmp_path / "results" / "experiment_1477_thrml_npim_simulator_parity_microprobe.json"

    marker = write_in_progress_artifact(output)

    assert REQUIRED_ARTIFACT_FIELDS <= set(marker)
    assert marker["status"] == "in_progress"
    assert marker["hardware_claim_allowed"] is False
    assert marker["simulator_only"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == marker


def test_req_sample_042_toy_cases_have_exact_cpu_reference() -> None:
    """REQ-SAMPLE-042: every toy Ising case has exact enumerated reference energy."""

    cases = build_toy_ising_cases()

    assert len(cases) == 3
    for case in cases:
        reference = exact_case_reference(case)
        assert reference["state_count"] == 2 ** case.n_spins
        assert reference["exact_ground_energy"] <= reference["exact_mean_energy"]
        assert len(reference["ground_state"]) == case.n_spins


def test_req_sample_042_carnot_rows_compare_sampler_to_exact_reference() -> None:
    """REQ-SAMPLE-042: Carnot sampler rows report exact-reference energy gaps."""

    rows = run_carnot_sampler_cases(
        build_toy_ising_cases()[:2],
        sample_func=_ground_state_sampler,
        seed=17,
        n_samples=5,
        n_warmup=4,
        steps_per_sample=1,
    )

    assert len(rows) == 2
    for row in rows:
        assert row["status"] == "sampled"
        assert row["sample_count"] == 5
        assert row["carnot_best_energy"] == pytest.approx(row["exact_ground_energy"])
        assert row["best_energy_gap_to_exact"] == pytest.approx(0.0)


def test_req_sample_042_default_carnot_sampler_path_runs_tiny_case() -> None:
    """REQ-SAMPLE-042: the default Carnot sampler path returns +/-1 samples."""

    rows = run_carnot_sampler_cases(
        build_toy_ising_cases()[:1],
        seed=3,
        n_samples=1,
        n_warmup=1,
        steps_per_sample=1,
    )

    assert rows[0]["status"] == "sampled"
    assert rows[0]["sample_count"] == 1
    assert rows[0]["exact_reference_state_count"] == 8


def test_scenario_sample_070_missing_thrml_blocks_parity_without_hardware_claim() -> None:
    """SCENARIO-SAMPLE-070: unavailable THRML records a blocker and no parity rows."""

    parity_rows, parity_metric, blockers = measure_thrml_parity_cases(
        build_toy_ising_cases(),
        _missing_probe(),
    )

    assert parity_rows == []
    assert parity_metric["status"] == "blocked"
    assert parity_metric["value"] is None
    assert blockers[0]["blocker"] == "thrml_not_importable"


def test_scenario_sample_070_fake_thrml_measures_energy_parity() -> None:
    """SCENARIO-SAMPLE-070: importable THRML-like API measures exact energy parity."""

    parity_rows, parity_metric, blockers = measure_thrml_parity_cases(
        build_toy_ising_cases()[:2],
        _available_probe(),
    )

    assert blockers == []
    assert parity_metric["status"] == "measured"
    assert parity_metric["value"] == pytest.approx(0.0, abs=1e-12)
    assert len(parity_rows) == 2
    assert all(row["status"] == "parity_measured" for row in parity_rows)
    assert all(row["max_abs_energy_error"] == pytest.approx(0.0, abs=1e-12) for row in parity_rows)


def test_scenario_sample_070_available_thrml_with_missing_api_blocks_parity() -> None:
    """SCENARIO-SAMPLE-070: import success is not enough without Ising APIs."""

    parity_rows, parity_metric, blockers = measure_thrml_parity_cases(
        build_toy_ising_cases()[:1],
        ThrmlImportProbe(
            available=True,
            module=SimpleNamespace(),
            version="api-missing",
            missing_api_or_dependency=None,
        ),
    )

    assert parity_rows == []
    assert parity_metric["status"] == "blocked"
    assert "SpinNode" in parity_metric["reason"]
    assert blockers == [{"blocker": "thrml_api_incompatible", "detail": parity_metric["reason"]}]


def test_req_sample_042_probe_import_reports_missing_thrml_without_install() -> None:
    """REQ-SAMPLE-042: the THRML probe uses import only and records failure."""

    def _raise_missing(_name: str) -> Any:
        raise ModuleNotFoundError("No module named 'thrml'", name="thrml")

    probe = probe_thrml_import(importer=_raise_missing)

    assert probe.available is False
    assert probe.module is None
    assert probe.version is None
    assert "thrml" in str(probe.missing_api_or_dependency)


def test_req_sample_042_probe_import_reports_success_and_generic_failure() -> None:
    """REQ-SAMPLE-042: import probing records success and non-module errors."""

    success = probe_thrml_import(importer=lambda _name: SimpleNamespace(__version__="0.test"))
    assert success.available is True
    assert success.version == "0.test"
    assert success.missing_api_or_dependency is None

    def _raise_value_error(_name: str) -> Any:
        raise ValueError("broken import")

    failure = probe_thrml_import(importer=_raise_value_error)
    assert failure.available is False
    assert failure.missing_api_or_dependency == "ValueError: broken import"


def test_req_sample_042_npim_probe_reports_energy_and_time_deltas() -> None:
    """REQ-SAMPLE-042: the NPIM-style simulator reports bounded heuristic deltas."""

    summary = run_npim_probe(build_toy_ising_cases(), seed=123, chains=8, steps=12)

    assert summary["attempted"] is True
    assert len(summary["cases"]) == 3
    assert summary["energy_delta"]["negative_is_better"] is True
    assert summary["energy_delta"]["value"] <= 0.0
    assert summary["time_to_energy_delta"]["unit"] == "sweeps"
    assert summary["time_to_energy_delta"]["value"] is None or summary["time_to_energy_delta"]["value"] >= 0


def test_req_sample_042_npim_probe_records_first_improvement_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-SAMPLE-042: NPIM time-to-energy reports the first improving sweep."""

    calls = {"count": 0}

    def _fake_policy(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "momentum": 0.0,
                "beta_final": 1.0,
                "best_energy": 2.0,
                "best_step": 0,
                "best_by_step": [2.0, 2.0, 2.0],
            }
        return {
            "momentum": 0.5,
            "beta_final": 2.0,
            "best_energy": 1.0,
            "best_step": 1,
            "best_by_step": [2.0, 1.5, 1.0],
        }

    monkeypatch.setattr(exp1477, "_run_update_policy", _fake_policy)

    summary = run_npim_probe(build_toy_ising_cases()[:1], seed=1, chains=2, steps=3)

    assert summary["energy_delta"]["value"] == pytest.approx(-1.0)
    assert summary["time_to_energy_delta"]["value"] == 1


def test_scenario_sample_070_run_microprobe_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-070: the runner writes the required terminal schema."""

    output = tmp_path / "results" / "experiment_1477_thrml_npim_simulator_parity_microprobe.json"

    artifact = run_microprobe(
        output_path=output,
        run_date="20260507",
        probe_func=lambda: _missing_probe(),
        sample_func=_ground_state_sampler,
        n_samples=4,
        n_warmup=3,
        steps_per_sample=1,
        npim_steps=8,
        npim_chains=6,
    )

    validate_artifact(artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["thrml_available"] is False
    assert artifact["thrml_parity_cases"] == []
    assert artifact["parity_metric"]["status"] == "blocked"
    assert artifact["npim_probe_attempted"] is True
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["simulator_only"] is True
    assert artifact["blockers"][0]["blocker"] == "thrml_not_importable"
    assert artifact["honest_verdict"] == "complete_thrml_unavailable_npim_simulator_probe_recorded"


def test_scenario_sample_070_run_microprobe_records_thrml_success_and_api_block(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAMPLE-070: terminal verdicts distinguish success from API block."""

    success = run_microprobe(
        output_path=tmp_path / "success.json",
        run_date="20260507",
        probe_func=lambda: _available_probe(),
        sample_func=_ground_state_sampler,
        n_samples=2,
        n_warmup=1,
        steps_per_sample=1,
        npim_steps=4,
        npim_chains=4,
    )
    assert success["thrml_available"] is True
    assert success["parity_metric"]["status"] == "measured"
    assert success["honest_verdict"] == "complete_thrml_parity_measured_npim_simulator_probe_recorded"

    blocked = run_microprobe(
        output_path=tmp_path / "blocked.json",
        run_date="20260507",
        probe_func=lambda: ThrmlImportProbe(
            available=True,
            module=SimpleNamespace(),
            version="api-missing",
            missing_api_or_dependency=None,
        ),
        sample_func=_ground_state_sampler,
        n_samples=2,
        n_warmup=1,
        steps_per_sample=1,
        npim_steps=4,
        npim_chains=4,
    )
    assert blocked["thrml_available"] is True
    assert blocked["parity_metric"]["status"] == "blocked"
    assert blocked["honest_verdict"] == "complete_thrml_api_blocked_npim_simulator_probe_recorded"


def test_req_sample_042_validation_enforces_no_hardware_claim() -> None:
    """REQ-SAMPLE-042: validation rejects hardware claims for this simulator probe."""

    valid = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    valid.update(
        {
            "status": "complete",
            "thrml_available": False,
            "carnot_sampler_cases": [{"status": "sampled"}],
            "thrml_parity_cases": [],
            "parity_metric": {"status": "blocked", "value": None},
            "npim_probe_attempted": True,
            "npim_energy_delta": {"value": 0.0},
            "npim_time_to_energy_delta": {"value": None},
            "hardware_claim_allowed": False,
            "simulator_only": True,
            "blockers": [{"blocker": "thrml_not_importable"}],
            "honest_verdict": "complete_thrml_unavailable_npim_simulator_probe_recorded",
        }
    )
    validate_artifact(valid)

    invalid = dict(valid)
    invalid["hardware_claim_allowed"] = True
    with pytest.raises(ValueError, match="hardware_claim_allowed"):
        validate_artifact(invalid)

    invalid = dict(valid)
    invalid["simulator_only"] = False
    with pytest.raises(ValueError, match="simulator_only"):
        validate_artifact(invalid)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "failed", "invalid status"),
        ("carnot_sampler_cases", [], "carnot_sampler_cases"),
        ("parity_metric", None, "parity_metric"),
        ("npim_probe_attempted", False, "npim_probe_attempted"),
        ("npim_energy_delta", None, "npim_energy_delta"),
        ("npim_time_to_energy_delta", None, "npim_time_to_energy_delta"),
        ("honest_verdict", "not_allowed", "invalid honest_verdict"),
        ("thrml_parity_cases", [{"status": "parity_measured"}], "thrml_parity_cases"),
    ],
)
def test_req_sample_042_validation_rejects_incomplete_terminal_artifacts(
    field: str,
    value: Any,
    message: str,
) -> None:
    """REQ-SAMPLE-042: terminal validation catches partial or unsafe artifacts."""

    artifact = {
        "status": "complete",
        "thrml_available": False,
        "carnot_sampler_cases": [{"status": "sampled"}],
        "thrml_parity_cases": [],
        "parity_metric": {"status": "blocked", "value": None},
        "npim_probe_attempted": True,
        "npim_energy_delta": {"value": 0.0},
        "npim_time_to_energy_delta": {"value": None},
        "hardware_claim_allowed": False,
        "simulator_only": True,
        "blockers": [{"blocker": "thrml_not_importable"}],
        "honest_verdict": "complete_thrml_unavailable_npim_simulator_probe_recorded",
    }
    artifact[field] = value

    with pytest.raises(ValueError, match=message):
        validate_artifact(artifact)


def test_req_sample_042_validation_rejects_missing_required_fields() -> None:
    """REQ-SAMPLE-042: required schema fields must be present."""

    with pytest.raises(ValueError, match="missing required artifact fields"):
        validate_artifact({"status": "complete"})


def test_req_sample_042_private_metric_helpers_cover_null_and_empty_errors() -> None:
    """REQ-SAMPLE-042: metric helpers keep null and empty exception text stable."""

    assert exp1477._round_metric(None) is None
    assert exp1477._describe_exception(RuntimeError()) == "RuntimeError"
