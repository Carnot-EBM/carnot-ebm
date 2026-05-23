"""Tests for Exp 2942 KV260 continuation n-scaling.

REQ-HW-074: the hardware-smoke artifact must report real KV260 latency rows
or an explicit active-bitstream fixed-n limitation instead of extrapolated
crossover data.
SCENARIO-HW-074: supported spin counts produce positive 1000-sample median and
p95 per-sample timing rows with bitstream and seed provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.hardware import kv260_continuation_n_scaling as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"


def test_req_hw_074_spec_anchor_exists() -> None:
    """REQ-HW-074: OpenSpec anchors the n-scaling artifact schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-074" in spec
    assert "SCENARIO-HW-074" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "bitstream_supports_variable_n" in spec
    assert "per_n_results" in spec


def test_detect_bitstream_support_documents_n64_alias_boundary() -> None:
    """REQ-HW-074: legacy n64 overlays are treated as fixed-n bitstreams."""

    support = exp.detect_bitstream_support("carnot_ising_v2_n64")

    assert support.variable is False
    assert support.supported_n == [64]
    assert "fixed n=64" in support.detail
    assert exp.select_measured_n_values(support) == [64]


def test_detect_bitstream_support_allows_v4_variable_rows_through_n128() -> None:
    """SCENARIO-HW-074: v4 metadata can measure only requested rows it supports."""

    support = exp.detect_bitstream_support("carnot_ising_v4")

    assert support.variable is True
    assert support.supported_n == [64, 128]
    assert exp.select_measured_n_values(support) == [64, 128]


def test_unknown_or_empty_bitstream_support_falls_back_safely() -> None:
    """REQ-HW-074: unknown bitstream metadata never invents large-n support."""

    unknown = exp.detect_bitstream_support("mystery")
    empty = exp.BitstreamNSupport(variable=True, supported_n=[], detail="none")

    assert unknown.variable is False
    assert unknown.supported_n == [64]
    assert exp.select_measured_n_values(empty) == []


def test_problem_payload_is_deterministic_and_sparse() -> None:
    """REQ-HW-074: generated Ising problems carry deterministic seed checksums."""

    first = exp.build_problem_payload([64, 128])
    second = exp.build_problem_payload([64, 128])

    assert first == second
    assert first["n_samples_per_n"] == 1000
    assert [row["n"] for row in first["problem_specs"]] == [64, 128]
    assert [row["random_seed"] for row in first["problem_specs"]] == [
        exp.RANDOM_SEED_BY_N[64],
        exp.RANDOM_SEED_BY_N[128],
    ]
    assert len(first["problems"][0]["upload"]["adjacency"]) == 64
    assert len(first["problems"][1]["upload"]["adjacency"]) == 128
    assert len(first["problem_specs"][0]["sparse_upload_sha256"]) == 64


def test_problem_generation_rejects_bad_inputs_and_pads_small_graphs() -> None:
    """REQ-HW-074: payload generation fails closed for invalid n/max-degree."""

    with pytest.raises(ValueError, match="n_spins"):
        exp.generate_sparse_ising_problem(1, seed=1)
    with pytest.raises(ValueError, match="max_degree"):
        exp.generate_sparse_ising_problem(4, seed=1, max_degree=0)

    problem = exp.generate_sparse_ising_problem(4, seed=1, max_degree=6)

    assert problem["upload"]["adjacency"][0][-3:] == [-1, -1, -1]
    assert problem["upload"]["couplings_q88"][0][-3:] == [0, 0, 0]


def test_summarize_board_payload_returns_required_per_n_shape() -> None:
    """SCENARIO-HW-074: board rows collapse to the required per_n_results shape."""

    board_payload: dict[str, Any] = {
        "runs": [
            {
                "n": 64,
                "n_samples": 1000,
                "per_sample_us_median": 24.0,
                "per_sample_us_p95": 25.0,
                "per_sample_us_min": 23.5,
                "per_sample_us_max": 26.0,
                "failed_samples": 0,
            },
            {
                "n": 128,
                "n_samples": 1000,
                "per_sample_us_median": 47.0,
                "per_sample_us_p95": 49.0,
                "failed_samples": 0,
            },
        ]
    }

    rows = exp.summarize_board_payload(board_payload, measured_n_values=[64, 128])

    assert rows == [
        {"n": 64, "per_sample_us_median": 24.0, "per_sample_us_p95": 25.0},
        {"n": 128, "per_sample_us_median": 47.0, "per_sample_us_p95": 49.0},
    ]


def test_summarize_board_payload_rejects_malformed_runs() -> None:
    """REQ-HW-074: malformed board payloads do not become timing evidence."""

    with pytest.raises(ValueError, match="missing runs"):
        exp.summarize_board_payload({}, measured_n_values=[64])

    with pytest.raises(ValueError, match="missing n=64"):
        exp.summarize_board_payload({"runs": []}, measured_n_values=[64])

    with pytest.raises(ValueError, match="failed samples"):
        exp.summarize_board_payload(
            {
                "runs": [
                    {
                        "n": 64,
                        "per_sample_us_median": 24.0,
                        "per_sample_us_p95": 25.0,
                        "failed_samples": 1,
                    }
                ]
            },
            measured_n_values=[64],
        )

    with pytest.raises(ValueError, match="positive"):
        exp.summarize_board_payload(
            {
                "runs": [
                    {
                        "n": 64,
                        "per_sample_us_median": 0.0,
                        "per_sample_us_p95": 25.0,
                        "failed_samples": 0,
                    }
                ]
            },
            measured_n_values=[64],
        )


def test_run_experiment_with_fake_fixed_n_hardware_writes_artifact(tmp_path: Path) -> None:
    """REQ-HW-074: fixed-n hardware success records limitation and no extrapolation."""

    def fake_hardware_runner(_: dict[str, Any]) -> exp.HardwareRunResult:
        return exp.HardwareRunResult(
            preconditions_checked=[
                {"resource": "kv260_ssh", "available": True, "detail": "test"},
                {"resource": "kv260_overlay", "available": True, "detail": "carnot_ising_v2_n64"},
                {"resource": "kv260_uio0", "available": True, "detail": "test"},
                {"resource": "bitstream_n_support", "available": True, "detail": "fixed n=64"},
            ],
            bitstream_sha256="a" * 64,
            bitstream_support=exp.BitstreamNSupport(
                variable=False,
                supported_n=[64],
                detail="active overlay exposes fixed n=64",
            ),
            per_n_results=[
                {"n": 64, "per_sample_us_median": 24.2, "per_sample_us_p95": 24.9}
            ],
            board_summary={"selected_uio": "/dev/uio4"},
        )

    artifact = exp.run_experiment(
        root_path=tmp_path,
        hardware_runner=fake_hardware_runner,
        started_s=10.0,
        now_s=12.5,
    )

    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"] == "complete: kv260_fixed_n64_latency_profile_recorded"
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["bitstream_supports_variable_n"] is False
    assert artifact["per_n_results"] == [
        {"n": 64, "per_sample_us_median": 24.2, "per_sample_us_p95": 24.9}
    ]
    assert artifact["unsupported_n_values"] == [128, 256, 512, 1024]
    assert artifact["random_seeds_used"] == [exp.RANDOM_SEED_BY_N[64]]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] == pytest.approx(2.5)

    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    assert saved == artifact


def test_success_artifact_verdicts_cover_variable_partial_and_full_profiles() -> None:
    """SCENARIO-HW-074: variable bitstreams distinguish partial and full target coverage."""

    problem_payload = exp.build_problem_payload(exp.TARGET_N_VALUES)
    support = exp.BitstreamNSupport(variable=True, supported_n=exp.TARGET_N_VALUES, detail="all")
    partial = exp.build_success_artifact(
        preconditions_checked=[],
        bitstream_sha256="b" * 64,
        support=support,
        per_n_results=[
            {"n": 64, "per_sample_us_median": 24.0, "per_sample_us_p95": 25.0}
        ],
        problem_payload=problem_payload,
        board_summary={},
        duration_s=1.0,
    )
    full = exp.build_success_artifact(
        preconditions_checked=[],
        bitstream_sha256="b" * 64,
        support=support,
        per_n_results=[
            {"n": n, "per_sample_us_median": float(n), "per_sample_us_p95": float(n + 1)}
            for n in exp.TARGET_N_VALUES
        ],
        problem_payload=problem_payload,
        board_summary={},
        duration_s=1.0,
    )

    assert partial["honest_verdict"] == "complete: kv260_variable_n_partial_latency_profile_recorded"
    assert full["honest_verdict"] == "complete: kv260_variable_n_latency_profile_recorded"


def test_run_experiment_blocks_without_hardware_preconditions(tmp_path: Path) -> None:
    """REQ-HW-074: blocked hardware preconditions still write the required schema."""

    def blocked_runner(_: dict[str, Any]) -> exp.HardwareRunResult:
        return exp.HardwareRunResult(
            preconditions_checked=[
                {"resource": "kv260_ssh", "available": False, "detail": "ssh rc=255"}
            ],
            bitstream_sha256="",
            bitstream_support=exp.BitstreamNSupport(
                variable=False,
                supported_n=[],
                detail="ssh unreachable",
            ),
            per_n_results=[],
            blocked_verdict="blocked_kv260_ssh_unreachable",
        )

    artifact = exp.run_experiment(
        root_path=tmp_path,
        hardware_runner=blocked_runner,
        started_s=0.0,
        now_s=1.0,
    )

    assert artifact["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    assert artifact["bitstream_supports_variable_n"] is False
    assert artifact["per_n_results"] == []
    assert artifact["random_seeds_used"] == []
    assert (tmp_path / exp.OUTPUT_REL_PATH).exists()


def test_validate_artifact_rejects_bad_success_rows() -> None:
    """REQ-HW-074: successful timing rows must be positive and correctly shaped."""

    artifact = exp.blocked_artifact(
        verdict="blocked_test",
        preconditions_checked=[],
        duration_s=0.1,
    )
    artifact.update(
        {
            "honest_verdict": "complete: bad",
            "bitstream_sha256": "a" * 64,
            "per_n_results": [{"n": 64, "per_sample_us_median": 0.0}],
        }
    )

    with pytest.raises(ValueError, match="per_n_results"):
        exp.validate_artifact(artifact)


def test_validate_artifact_rejects_other_incomplete_success_shapes() -> None:
    """REQ-HW-074: complete artifacts require bitstream, rows, seeds, and checksum."""

    base = exp.build_success_artifact(
        preconditions_checked=[],
        bitstream_sha256="a" * 64,
        support=exp.BitstreamNSupport(variable=False, supported_n=[64], detail="fixed"),
        per_n_results=[
            {"n": 64, "per_sample_us_median": 24.0, "per_sample_us_p95": 25.0}
        ],
        problem_payload=exp.build_problem_payload([64]),
        board_summary={},
        duration_s=1.0,
    )

    missing = dict(base)
    missing.pop("duration_s")
    with pytest.raises(ValueError, match="missing"):
        exp.validate_artifact(missing)

    bad_sha = dict(base, bitstream_sha256="not-a-sha")
    with pytest.raises(ValueError, match="bitstream"):
        exp.validate_artifact(bad_sha)

    empty_rows = dict(base, per_n_results=[])
    with pytest.raises(ValueError, match="non-empty"):
        exp.validate_artifact(empty_rows)

    bad_n = dict(base, per_n_results=[{"n": 0, "per_sample_us_median": 1.0, "per_sample_us_p95": 2.0}])
    with pytest.raises(ValueError, match="n must be positive"):
        exp.validate_artifact(bad_n)

    bad_latency = dict(base, per_n_results=[{"n": 64, "per_sample_us_median": 1.0, "per_sample_us_p95": 0.0}])
    with pytest.raises(ValueError, match="latency values"):
        exp.validate_artifact(bad_latency)

    no_seeds = dict(base, random_seeds_used=[])
    with pytest.raises(ValueError, match="random_seeds_used"):
        exp.validate_artifact(no_seeds)

    bad_checksum = dict(base, reproducibility_checksum="")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)
