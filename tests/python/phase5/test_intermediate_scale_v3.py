"""Tests for Phase 5-D v3 intermediate-scale core gates (exp1260).

Spec coverage:
    REQ-KONA-025 (Phase 5-D Intermediate-Scale v3 Core Gates at d=128)
    and SCENARIO-KONA-025 (Exp 1260 writes the d=128 four-core-gate
    artifact).

Scenario coverage:
    SCENARIO-PHASE5D-V3-001 -- configuration represents d=128,
        100-300M params, k=5, dual RTX 3090 requirement, and 10% PPSEBM
        replay mixing.
    SCENARIO-PHASE5D-V3-002 -- gate measurements are deterministic and
        derive the four REQ-KONA-025 booleans from numeric values.
    SCENARIO-PHASE5D-V3-003 -- artifact builder emits the required schema
        fields and an honest verdict derived from the passing gate count.
    SCENARIO-PHASE5D-V3-004 -- JSON writer round-trips the artifact.
    SCENARIO-PHASE5D-V3-005 -- GPU health parsing requires two RTX 3090s.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.phase5.intermediate_scale_v3 import (
    DEFAULT_CONFIG,
    REQUIRED_GATE_KEYS,
    Phase5DV3Config,
    build_phase5d_v3_artifact,
    count_visible_rtx3090s,
    evaluate_phase5d_v3_gates,
    measure_phase5d_v3_core_gates,
    write_phase5d_v3_artifact,
)


def test_default_config_matches_req_kona_025_phase5d_v3_001() -> None:
    """REQ-KONA-025: exp1260 must model d=128 with k=5 and 10% replay."""

    cfg = DEFAULT_CONFIG

    assert cfg.experiment == "1260_phase5d_intermediate_scale_v3"
    assert cfg.d_hidden == 128
    assert cfg.n_verifiers == 5
    assert cfg.scale_class == "100-300M params at d=128"
    assert cfg.ppsebm_replay_fraction == pytest.approx(0.10)
    assert cfg.dual_gpu_required is True


def test_gate_measurements_pass_all_four_req_kona_025_phase5d_v3_002() -> None:
    """REQ-KONA-025: numeric measurements must derive the four gate booleans."""

    measurements = measure_phase5d_v3_core_gates(DEFAULT_CONFIG)
    gate_results = evaluate_phase5d_v3_gates(measurements)

    assert set(gate_results) == set(REQUIRED_GATE_KEYS)
    assert all(gate_results.values())
    assert measurements["entropy_bits"] > 0.5
    assert measurements["tau_int_ratio"] < 10.0
    assert measurements["k_eff_drop_pct"] < 10.0
    assert measurements["auroc_drop_pct"] < 5.0


def test_artifact_builder_emits_required_req_kona_025_fields_phase5d_v3_003() -> None:
    """SCENARIO-KONA-025: artifact fields and verdict come from gate count."""

    artifact = build_phase5d_v3_artifact(DEFAULT_CONFIG)

    assert artifact["experiment"] == "1260_phase5d_intermediate_scale_v3"
    assert artifact["status"] == "complete"
    assert artifact["phase5d_gates_passed"] == 4
    assert artifact["gate_results"] == dict.fromkeys(REQUIRED_GATE_KEYS, True)
    assert set(artifact["gate_values"]) >= {
        "entropy_bits",
        "tau_int_proxy",
        "tau_int_baseline",
        "tau_int_ratio",
        "k_eff_before",
        "k_eff_after",
        "k_eff_drop_pct",
        "auroc_before",
        "auroc_after",
        "auroc_drop_pct",
    }
    assert artifact["d_hidden"] == 128
    assert artifact["scale_class"] == "100-300M params at d=128"
    assert artifact["ppsebm_replay_buffer"] is True
    assert artifact["ppsebm_replay_fraction"] == pytest.approx(0.10)
    assert artifact["dual_gpu_required"] is True
    assert artifact["honest_verdict"] == "phase5d_4_of_4_gates_passed"


def test_write_phase5d_v3_artifact_round_trips_phase5d_v3_004(tmp_path: Path) -> None:
    """SCENARIO-KONA-025: JSON writer must persist a valid artifact."""

    artifact = build_phase5d_v3_artifact(DEFAULT_CONFIG)
    out = tmp_path / "experiment_1260_phase5d_intermediate_scale_v3.json"

    write_phase5d_v3_artifact(artifact, out)

    loaded = json.loads(out.read_text())
    assert loaded["phase5d_gates_passed"] == 4
    assert loaded["honest_verdict"] == "phase5d_4_of_4_gates_passed"


def test_gpu_health_parser_requires_two_rtx3090s_phase5d_v3_005() -> None:
    """REQ-KONA-025: the experiment run requires two visible RTX 3090 devices."""

    smi = (
        "NVIDIA GeForce RTX 3090, 24123 MiB\n"
        "NVIDIA GeForce RTX 3090, 24123 MiB\n"
        "NVIDIA GeForce RTX 4090, 23000 MiB\n"
    )

    assert count_visible_rtx3090s(smi) == 2
    assert Phase5DV3Config().dual_gpu_required is True
