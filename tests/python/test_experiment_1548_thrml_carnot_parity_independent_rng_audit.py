"""Tests for Exp 1548 THRML/Carnot independent-RNG parity audit.

Spec refs: REQ-SAMPLE-056, SCENARIO-SAMPLE-084.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.samplers import thrml_carnot_parity_independent_rng_audit as exp1548


def _fake_sampler(
    case: exp1548.AuditIsingCase,
    *,
    seed: int,
    n_samples: int,
    schedule: dict[str, Any],
) -> np.ndarray:
    del schedule
    rng = np.random.default_rng(int(seed))
    return rng.random((int(n_samples), case.n_spins)) > 0.5


def test_spec_mentions_exp1548_contract() -> None:
    """REQ-SAMPLE-056, SCENARIO-SAMPLE-084: Exp1548 is spec-anchored."""

    spec = (exp1548.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-056" in spec
    assert "SCENARIO-SAMPLE-084" in spec
    assert "experiment_1548_thrml_carnot_parity_independent_rng_audit.json" in spec
    assert "rng_seed_manifest_path" in spec
    assert "byte-identical stochastic" in spec


def test_seed_manifest_requires_disjoint_roots_and_lineages(tmp_path: Path) -> None:
    """REQ-SAMPLE-056: seed manifests must not reuse roots or shared-key derivation."""

    manifest = exp1548.build_seed_manifest(
        n_values=(32, 64),
        topologies=("complete", "lattice"),
        carnot_root_seed=2026050801,
        thrml_root_seed=2026050802,
    )

    exp1548.validate_seed_manifest(manifest)
    assert manifest["samplers"]["carnot"]["root_seed"] != manifest["samplers"]["thrml"]["root_seed"]
    assert manifest["samplers"]["carnot"]["derivation"] == "numpy_seed_sequence_spawn"
    assert manifest["samplers"]["thrml"]["derivation"] == "numpy_seed_sequence_spawn"
    assert len(manifest["case_seeds"]) == 4

    manifest_path = tmp_path / "seed_manifest.json"
    written = exp1548.write_seed_manifest(manifest, manifest_path)
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == written

    reused_root = json.loads(json.dumps(manifest))
    reused_root["samplers"]["thrml"]["root_seed"] = reused_root["samplers"]["carnot"]["root_seed"]
    with pytest.raises(ValueError, match="disjoint root seeds"):
        exp1548.validate_seed_manifest(reused_root)

    shared_key = json.loads(json.dumps(manifest))
    shared_key["samplers"]["carnot"]["derivation"] = "shared_key_split"
    with pytest.raises(ValueError, match="shared key"):
        exp1548.validate_seed_manifest(shared_key)


def test_byte_identical_hash_detection_flags_hash_and_histogram_matches() -> None:
    """REQ-SAMPLE-056: byte-identical stochastic summaries are failure evidence."""

    samples = np.array([[True, False, True], [False, True, False]], dtype=bool)
    shifted = np.logical_not(samples)
    same_hash = exp1548.sample_path_hash(samples)
    other_hash = exp1548.sample_path_hash(shifted)
    per_case = [
        {
            "case_id": "identical_hash",
            "carnot_sample_hash": same_hash,
            "thrml_sample_hash": same_hash,
            "histogram_counts": {"carnot_counts": [1, 2], "thrml_counts": [1, 2]},
        },
        {
            "case_id": "histogram_only",
            "carnot_sample_hash": same_hash,
            "thrml_sample_hash": other_hash,
            "histogram_counts": {"carnot_counts": [3, 4], "thrml_counts": [3, 4]},
        },
    ]

    identical = exp1548.detect_byte_identical_pairs(per_case)

    assert identical == [
        {"case_id": "identical_hash", "match_type": "sample_path_hash_and_histogram_counts"},
        {"case_id": "histogram_only", "match_type": "histogram_counts"},
    ]


def test_metric_computation_reports_bounded_nonzero_kl_and_ks_pass() -> None:
    """REQ-SAMPLE-056: metric rows distinguish agreement from exact equality."""

    carnot = np.array([-2.0, -1.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0])
    thrml = np.array([-2.0, -1.0, -0.5, 0.0, 0.2, 1.0, 1.8, 2.5])

    metrics = exp1548.compute_distribution_metrics(carnot, thrml, energy_bin_count=5)

    assert metrics["mean_energy_delta_abs"] > 0.0
    assert 0.0 < metrics["kl_divergence"] <= 0.05
    assert metrics["ks_statistic"] >= 0.0
    assert metrics["ks_p_value"] > 0.01
    assert metrics["histogram_counts"]["carnot_counts"] != metrics["histogram_counts"]["thrml_counts"]


def test_run_audit_with_fake_independent_samplers_writes_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-084: complete audit writes all required terminal fields."""

    output_path = tmp_path / "experiment_1548.json"
    seed_manifest_path = tmp_path / "seed_manifest.json"
    manifest_path = tmp_path / "audit_manifest.jsonl"

    artifact = exp1548.run_independent_rng_audit(
        output_path=output_path,
        seed_manifest_path=seed_manifest_path,
        manifest_path=manifest_path,
        n_values=(4,),
        topologies=("complete",),
        carnot_sampler=_fake_sampler,
        thrml_sampler=_fake_sampler,
        sample_count_per_case=64,
        n_warmup=4,
        steps_per_sample=1,
        thresholds={
            "mean_energy_delta_abs_max": 10.0,
            "kl_divergence_max": 10.0,
            "ks_p_value_min": 0.0,
        },
        energy_bin_count=8,
        focused_tests_passed=True,
    )
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.119"
    assert artifact["independent_rng_audit_ready"] is True
    assert artifact["rng_path_independent"] is True
    assert artifact["code_path_independent"] is True
    assert artifact["rng_seed_manifest_path"] == str(seed_manifest_path)
    assert artifact["n_values_tested"] == [4]
    assert artifact["topologies_tested"] == ["complete"]
    assert artifact["byte_identical_pairs"] == []
    assert artifact["nonzero_stochastic_delta_observed"] is True
    assert artifact["bounded_kl_passed"] is True
    assert artifact["ks_test_passed"] is True
    assert artifact["rng_path_not_independent"] is False
    assert artifact["simulator_only"] is True
    assert artifact["no_tsu_hardware_claim"] is True
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert rows[-1]["case_type"] == "independent_rng_audit_summary"
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_validate_artifact_rejects_ready_with_byte_identical_pair() -> None:
    """REQ-SAMPLE-056: readiness cannot be true with byte-identical evidence."""

    artifact = {
        "status": "complete",
        "milestone": "2026.04.119",
        "independent_rng_audit_ready": True,
        "rng_path_independent": True,
        "code_path_independent": True,
        "rng_seed_manifest_path": "results/seed_manifest.json",
        "n_values_tested": [32],
        "topologies_tested": ["complete"],
        "sample_path_hashes": {"n32_complete": {"carnot": "a", "thrml": "a"}},
        "byte_identical_pairs": [{"case_id": "n32_complete", "match_type": "sample_path_hash"}],
        "nonzero_stochastic_delta_observed": True,
        "per_case_results": [{"case_id": "n32_complete"}],
        "max_mean_energy_delta_abs": 0.1,
        "max_kl_divergence": 0.01,
        "min_ks_p_value": 0.8,
        "bounded_kl_passed": True,
        "ks_test_passed": True,
        "rng_path_not_independent": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "focused_tests_passed": True,
        "honest_verdict": "complete: invalid",
    }

    with pytest.raises(ValueError, match="byte-identical"):
        exp1548.validate_artifact(artifact)
