"""Tests for Exp 2907 operator hardware portfolio status card.

Spec refs: REQ-HW-063, SCENARIO-HW-063.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.hardware import operator_hardware_portfolio_status_v1 as exp2907


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_upstreams(root: Path) -> None:
    results = root / "results"
    _write_json(
        results / exp2907.EXP2898_FILENAME,
        {
            "honest_verdict": "complete: kv260_hardware_latency_transcript_recorded",
            "inference_substrate": "hardware_smoke",
            "board_transcript_path": "results/experiment_2898_kv260_transcript.log",
            "kv260_overlay_loaded": "carnot_ising_v2_n64",
            "bitstream_sha256": "a" * 64,
            "per_seed_results": [
                {"seed": 42, "per_sample_wall_clock_us_median": 24.05},
                {"seed": 137, "per_sample_wall_clock_us_median": 24.04},
                {"seed": 271, "per_sample_wall_clock_us_median": 24.01},
            ],
        },
    )
    _write_json(
        results / exp2907.EXP2899_FILENAME,
        {
            "honest_verdict": "blocked_gatemate_toolchain_missing",
            "inference_substrate": "hardware_smoke",
            "synth_succeeded": False,
            "place_and_route_succeeded": False,
            "bitstream_sha256": None,
            "preconditions_checked": [
                {"resource": "yosys", "available": True},
                {"resource": "nextpnr-gatemate", "available": False},
            ],
        },
    )
    _write_json(
        results / exp2907.EXP2900_FILENAME,
        {
            "honest_verdict": "complete: polarfire_riscv64_constraint_scorer_hash_verified",
            "inference_substrate": "hardware_smoke",
            "polarfire_arch": "riscv64",
            "scorer_output_hash_verified": True,
            "no_fpga_fabric_claim": True,
            "duration_s": 19.09,
        },
    )
    _write_json(
        results / exp2907.EXP2901_FILENAME,
        {
            "honest_verdict": "complete: thrml_import_repaired_n16_parity_passed_no_hardware_claim",
            "thrml_import_succeeded": True,
            "thrml_version_installed": "0.1.3",
            "parity_energy_delta": 2.47955e-07,
            "metadata": {
                "field_principles": {
                    "no_tsu_access_claim": True,
                    "no_hardware_acceleration_claim": True,
                }
            },
        },
    )


def test_req_hw_063_spec_anchor_exists() -> None:
    """REQ-HW-063: OpenSpec defines the portfolio card contract."""

    spec = (exp2907.REPO_ROOT / "openspec/capabilities/fpga/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-HW-063" in spec
    assert "SCENARIO-HW-063" in spec
    assert exp2907.OUTPUT_FILENAME in spec
    assert exp2907.INFERENCE_SUBSTRATE in spec


def test_scenario_hw_063_builds_concise_per_board_card(tmp_path: Path) -> None:
    """SCENARIO-HW-063: four upstream artifacts produce the operator status card."""

    _write_upstreams(tmp_path)

    artifact = exp2907.run_experiment(
        exp2907.ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            started_at=100.0,
            clock=lambda: 102.25,
        )
    )

    assert exp2907.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert exp2907.artifact_has_required_fields(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == exp2907.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["no_new_board_execution"] is True
    assert artifact["no_new_hardware_claim"] is True

    assert artifact["per_board_status"] == {
        "kv260": {
            "state": "ready_live_latency_recorded",
            "last_artifact": f"results/{exp2907.EXP2898_FILENAME}",
            "next_step": (
                "Use as KV260 baseline; add same-basis CPU comparison before speedup claims."
            ),
        },
        "gatemate": {
            "state": "blocked_gatemate_toolchain_missing",
            "last_artifact": f"results/{exp2907.EXP2899_FILENAME}",
            "next_step": (
                "Provision nextpnr-gatemate, rerun n=16 build, and do not flash until "
                "a bitstream exists."
            ),
        },
        "polarfire": {
            "state": "ready_riscv64_cpu_dispatch_verified",
            "last_artifact": f"results/{exp2907.EXP2900_FILENAME}",
            "next_step": "Treat as CPU-dispatch proof; FPGA fabric acceleration remains separate.",
        },
        "thrml": {
            "state": "ready_software_parity_no_tsu_claim",
            "last_artifact": f"results/{exp2907.EXP2901_FILENAME}",
            "next_step": "Use import/parity evidence; require TSU access before hardware claims.",
        },
    }

    expected_citations = []
    for experiment_id, filename, fields in [
        ("exp2898", exp2907.EXP2898_FILENAME, exp2907.KV260_FIELDS),
        ("exp2899", exp2907.EXP2899_FILENAME, exp2907.GATEMATE_FIELDS),
        ("exp2900", exp2907.EXP2900_FILENAME, exp2907.POLARFIRE_FIELDS),
        ("exp2901", exp2907.EXP2901_FILENAME, exp2907.THRML_FIELDS),
    ]:
        path = tmp_path / "results" / filename
        expected_citations.append(
            {
                "experiment_id": experiment_id,
                "path": f"results/{filename}",
                "fields_imported": list(fields),
                "sha256": _sha256(path),
            }
        )
    assert artifact["cited_upstream_artifacts"] == expected_citations

    saved = json.loads((tmp_path / "results" / exp2907.OUTPUT_FILENAME).read_text())
    assert saved == artifact


def test_req_hw_063_missing_or_malformed_upstreams_block_completion(tmp_path: Path) -> None:
    """REQ-HW-063: missing or malformed upstream evidence cannot produce complete."""

    missing = exp2907.run_experiment(
        exp2907.ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )

    assert missing["honest_verdict"] == "blocked_missing_exp2898_artifact"
    assert missing["per_board_status"]["kv260"] == {
        "state": "missing_upstream_artifact",
        "last_artifact": f"results/{exp2907.EXP2898_FILENAME}",
        "next_step": "Produce exp2898 before relying on the portfolio card.",
    }
    assert missing["cited_upstream_artifacts"] == []
    assert exp2907.artifact_has_required_fields(missing)

    _write_upstreams(tmp_path)
    bad_path = tmp_path / "results" / exp2907.EXP2899_FILENAME
    bad_path.write_text("not json", encoding="utf-8")
    malformed = exp2907.run_experiment(
        exp2907.ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )

    assert malformed["honest_verdict"] == "blocked_malformed_exp2899_artifact"
    assert malformed["per_board_status"]["gatemate"] == {
        "state": "malformed_upstream_artifact",
        "last_artifact": f"results/{exp2907.EXP2899_FILENAME}",
        "next_step": "Repair exp2899 JSON before relying on the portfolio card.",
    }
    assert [item["experiment_id"] for item in malformed["cited_upstream_artifacts"]] == [
        "exp2898",
        "exp2900",
        "exp2901",
    ]


def test_req_hw_063_unrecognized_board_state_needs_operator_review(tmp_path: Path) -> None:
    """REQ-HW-063: unusual upstream verdicts stay visible instead of being upgraded."""

    _write_upstreams(tmp_path)
    kv260_path = tmp_path / "results" / exp2907.EXP2898_FILENAME
    kv260 = json.loads(kv260_path.read_text(encoding="utf-8"))
    kv260["honest_verdict"] = "partial: operator interrupted run"
    kv260["board_transcript_path"] = ""
    _write_json(kv260_path, kv260)

    artifact = exp2907.run_experiment(
        exp2907.ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["per_board_status"]["kv260"] == {
        "state": "needs_operator_review",
        "last_artifact": f"results/{exp2907.EXP2898_FILENAME}",
        "next_step": "Read upstream verdict and decide whether to rerun or defer.",
    }


def test_req_hw_063_artifact_shape_validation_rejects_bad_cards(tmp_path: Path) -> None:
    """REQ-HW-063: every board row must stay exactly state/last_artifact/next_step."""

    _write_upstreams(tmp_path)
    artifact = exp2907.run_experiment(
        exp2907.ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )

    bad_missing_field = dict(artifact)
    bad_missing_field.pop("honest_verdict")
    assert not exp2907.artifact_has_required_fields(bad_missing_field)
    with pytest.raises(ValueError, match="missing required fields"):
        exp2907.validate_artifact(bad_missing_field)

    bad_board_keys = dict(artifact)
    bad_board_keys["per_board_status"] = dict(artifact["per_board_status"])
    bad_board_keys["per_board_status"]["kv260"] = {"state": "ready"}
    assert not exp2907.artifact_has_required_fields(bad_board_keys)
    with pytest.raises(ValueError, match="failed required schema"):
        exp2907.validate_artifact(bad_board_keys)
