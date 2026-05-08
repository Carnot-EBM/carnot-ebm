"""Tests for Exp 1561 kinetic-defense zero-coupling null-space audit.

Spec refs: REQ-SAMPLE-057, SCENARIO-SAMPLE-085.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot.samplers import kinetic_defense_zero_coupling as exp1561


def test_spec_mentions_exp1561_contract() -> None:
    """REQ-SAMPLE-057, SCENARIO-SAMPLE-085: Exp1561 is spec-anchored."""

    spec = (exp1561.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-057" in spec
    assert "SCENARIO-SAMPLE-085" in spec
    assert "experiment_1561_kinetic_defense_zero_coupling_test.json" in spec
    assert "thrml_security_parity_with_single_site_gibbs" in spec


def test_zero_coupling_landscape_energy_and_null_space() -> None:
    """REQ-SAMPLE-057: landscape has 15 target blocks and four free null bits."""

    config = exp1561.ZeroCouplingConfig()
    states = np.zeros((3, config.n_blocks), dtype=np.uint8)
    states[0, :] = exp1561.target_block_state(config)
    states[1, :3] = exp1561.target_block_state(config)
    states[2, :14] = exp1561.target_block_state(config)

    energies = exp1561.energy_from_block_states(states, config)
    in_null = exp1561.in_null_space(states, config)

    assert config.n_bits == 64
    assert config.n_blocks == 15
    assert config.block_size == 4
    assert config.free_bits == 4
    assert exp1561.target_block_state(config) == 15
    assert energies.tolist() == [-150.0, -30.0, -140.0]
    assert in_null.tolist() == [True, False, False]


def test_single_block_hitting_times_expose_mh_gibbs_thrml_ordering() -> None:
    """REQ-SAMPLE-057: single-block hitting estimates are empirical and separated."""

    config = exp1561.ZeroCouplingConfig(n_chains=4000, single_block_max_steps=400)

    mh = exp1561.simulate_single_block_hitting("mh", config, seed=101)
    gibbs = exp1561.simulate_single_block_hitting("single_site_gibbs", config, seed=102)
    thrml = exp1561.simulate_single_block_hitting("thrml_block_gibbs", config, seed=103)

    assert 18.0 <= mh["mean_hitting_time_steps"] <= 24.0
    assert 24.0 <= gibbs["mean_hitting_time_steps"] <= 32.0
    assert 12.0 <= thrml["mean_hitting_time_steps"] <= 20.0
    assert mh["mean_hitting_time_steps"] < gibbs["mean_hitting_time_steps"]
    assert thrml["mean_hitting_time_steps"] < gibbs["mean_hitting_time_steps"]
    assert mh["censored_fraction"] == 0.0
    assert gibbs["censored_fraction"] == 0.0
    assert thrml["censored_fraction"] == 0.0


def test_gate_classifier_records_validated_and_falsified_cases() -> None:
    """REQ-SAMPLE-057: THRML faster-than-Gibbs cases are explicit falsifications."""

    validated = exp1561.classify_kinetic_defense(
        mh_steps=21.3,
        gibbs_steps=32.9,
        thrml_steps=33.0,
    )
    falsified = exp1561.classify_kinetic_defense(
        mh_steps=21.3,
        gibbs_steps=27.7,
        thrml_steps=15.1,
    )

    assert validated["kinetic_defense_in_depth_validated"] is True
    assert validated["thrml_security_parity_with_single_site_gibbs"] is True
    assert validated["honest_verdict"].startswith("complete_")
    assert falsified["kinetic_defense_in_depth_validated"] is False
    assert falsified["thrml_security_parity_with_single_site_gibbs"] is False
    assert falsified["thrml_hits_at_mh_class_rate"] is True
    assert "mitigation" in falsified
    assert falsified["honest_verdict"].startswith("complete_")


def test_thrml_metadata_probe_success_and_failure() -> None:
    """REQ-SAMPLE-057: THRML import provenance is recorded without hardware claims."""

    def fake_importer(name: str) -> Any:
        assert name == "thrml"
        return SimpleNamespace(__version__="0.1.3", __file__="/fake/thrml/__init__.py")

    def missing_importer(name: str) -> Any:
        raise ModuleNotFoundError(name)

    ready = exp1561.probe_thrml_metadata(fake_importer)
    missing = exp1561.probe_thrml_metadata(missing_importer)

    assert ready["thrml_import_ready"] is True
    assert ready["thrml_version"] == "0.1.3"
    assert ready["thrml_execution_mode"] == "thrml_0_1_3_graph_color_semantics"
    assert ready["hardware_claim_allowed"] is False
    assert missing["thrml_import_ready"] is False
    assert "ModuleNotFoundError" in missing["thrml_import_error"]


def test_run_experiment_writes_complete_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-085: terminal JSON has required fields and energy rows."""

    output_path = tmp_path / "experiment_1561.json"
    config = exp1561.ZeroCouplingConfig(
        n_chains=256,
        checkpoints=(10, 50, 100),
        single_block_max_steps=400,
        seed=1561,
    )

    artifact = exp1561.run_experiment(
        output_path=output_path,
        config=config,
        importer=lambda _name: SimpleNamespace(__version__="0.1.3", __file__="/fake/thrml.py"),
    )

    assert artifact["status"] == "complete"
    assert artifact["n_chains"] == 256
    assert artifact["checkpoints"] == [10, 50, 100]
    assert artifact["honest_verdict"].startswith("complete_")
    assert exp1561.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["thrml_security_parity_with_single_site_gibbs"] is False
    assert artifact["kinetic_defense_in_depth_validated"] is False
    assert artifact["p_n_at_k100_thrml_block_gibbs"] >= artifact["p_n_at_k100_single_site_gibbs"]
    assert set(artifact["sampler_results"]) == {"mh", "single_site_gibbs", "thrml_block_gibbs"}
    assert artifact["sampler_results"]["mh"]["energy_distributions"]["100"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_validate_artifact_rejects_missing_fields_and_bad_verdict() -> None:
    """REQ-SAMPLE-057: artifact schema requires terminal complete verdicts."""

    valid = {
        field: None for field in exp1561.REQUIRED_ARTIFACT_FIELDS
    }
    valid.update(
        {
            "status": "complete",
            "honest_verdict": "complete_test",
            "kinetic_defense_in_depth_validated": False,
            "thrml_security_parity_with_single_site_gibbs": False,
            "blockers": [],
        }
    )

    assert exp1561.validate_artifact(valid) is None

    missing = dict(valid)
    missing.pop("p_n_at_k100_mh")
    with pytest.raises(ValueError, match="missing required fields"):
        exp1561.validate_artifact(missing)

    bad_status = dict(valid)
    bad_status["status"] = "in_progress"
    with pytest.raises(ValueError, match="status must be complete"):
        exp1561.validate_artifact(bad_status)

    bad_prefix = dict(valid)
    bad_prefix["honest_verdict"] = "blocked"
    with pytest.raises(ValueError, match="terminal prefix"):
        exp1561.validate_artifact(bad_prefix)
