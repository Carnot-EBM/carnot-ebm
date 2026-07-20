"""Tests for Exp5738 one-axis Rust batched backend boundary.

Spec refs: REQ-SAMPLE-5738, SCENARIO-SAMPLE-5738.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5738_one_axis_rust_batched_backend as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5738_one_axis_rust_batched_backend.py")


def _fake_ready_evidence(**_: Any) -> dict[str, Any]:
    workloads = [
        {
            "workload_id": "ferromagnetic_ring_easy_n48_fake",
            "size": 48,
            "family": "ferromagnetic_ring_easy",
            "seed": 5738,
            "exp5724_python_mean_s": 0.001,
            "exp5724_rust_mean_s": 0.002,
            "rust_lost_in_exp5724": True,
        },
        {
            "workload_id": "ferromagnetic_ring_easy_n96_fake",
            "size": 96,
            "family": "ferromagnetic_ring_easy",
            "seed": 5739,
            "exp5724_python_mean_s": 0.002,
            "exp5724_rust_mean_s": 0.004,
            "rust_lost_in_exp5724": True,
        },
    ]
    phase_rows = [
        {
            "workload_id": row["workload_id"],
            "size": row["size"],
            "phase": phase,
            "scalar_mean_s": scalar,
            "batch_mean_s": batch,
            "samples_s": [scalar],
            "measurement_repetitions": 1,
        }
        for row in workloads
        for phase, scalar, batch in (
            ("serialization", 0.0002, 0.00019),
            ("python_allocation", 0.0003, 0.00029),
            ("pyo3_crossing", 0.005, 0.001),
            ("rust_allocation", 0.0004, 0.00038),
            ("energy_update", 0.0008, 0.0008),
            ("proposal", 0.0009, 0.0009),
            ("exchange", 0.0005, 0.0005),
            ("validation", 0.0002, 0.0002),
            ("checkpoint", 0.0003, 0.0003),
            ("restart", 0.0006, 0.0006),
            ("end_to_end", 0.0092, 0.00516),
        )
    ]
    return {
        "reproduction_workloads": workloads,
        "phase_timing_receipts": phase_rows,
        "memory_phase_receipts": [
            {
                "workload_id": row["workload_id"],
                "size": row["size"],
                "phase": "pyo3_crossing",
                "peak_rss_kib": 1200 + row["size"],
                "traffic_proxy_bytes": row["size"] * row["size"] * 8,
            }
            for row in workloads
        ],
        "large_size_reversal_reproduced": True,
        "dominant_phase": {
            "phase": "pyo3_crossing",
            "mean_scalar_s": 0.005,
            "share_of_end_to_end": 0.54,
            "batch_removable": True,
        },
        "optimization_hypothesis": {
            "justified": True,
            "hypothesis": "Moving repeated per-transition PyO3 crossings behind a batch boundary preserves semantics while removing the dominant measured boundary cost.",
            "falsifiable_gate": "scalar and batch traces must match with zero mismatches",
        },
        "batch_api_contract": {
            "method": "OneAxisRustBackend.sample_batch",
            "input": "ordered sequence of independent workload mappings",
            "output": "ordered list of run_descriptor result mappings",
            "empty_batch": "returns []",
            "singleton_equivalent_to_scalar": True,
            "mixed_size_allowed": True,
            "corrupt_checkpoint_policy": "fail_closed",
            "broken_binding_policy": "exact_python_fallback",
            "two_axis_exchange": False,
        },
        "python_fallback_receipts": [
            {"case_id": "normal", "equivalent": True},
            {"case_id": "empty", "equivalent": True},
            {"case_id": "singleton", "equivalent": True},
            {"case_id": "mixed_size", "equivalent": True},
            {"case_id": "broken_binding", "equivalent": True},
        ],
        "parity_manifest": {
            "semantic_controls": [
                "normal",
                "empty",
                "singleton",
                "mixed_size",
                "corrupted_checkpoint",
                "broken_binding",
                "exception",
            ],
            "energy_trace_match": True,
            "proposal_match": True,
            "exchange_match": True,
            "checkpoint_match": True,
            "restart_match": True,
            "result_order_match": True,
        },
        "energy_trace_mismatch_count": 0,
        "proposal_mismatch_count": 0,
        "exchange_mismatch_count": 0,
        "checkpoint_mismatch_count": 0,
        "restart_mismatch_count": 0,
        "result_order_mismatch_count": 0,
        "distributional_parity_receipts": [
            {
                "workload_id": "distributional_n64_fake",
                "n_spins": 64,
                "n_samples": 10000,
                "comparison_count": 3,
                "familywise_alpha": 0.05,
                "adjusted_alpha": 0.016666666667,
                "energy_histogram_tv": 0.0,
                "passed": True,
            }
        ],
    }


def test_req_sample_5738_spec_declares_artifact_contract() -> None:
    """REQ-SAMPLE-5738: OpenSpec lists every required field and principle."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5738") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized
    for marker in (
        str(mod.RESULT_RELATIVE_PATH),
        "n=48",
        "n=96",
        "serialization, Python",
        "sample_batch",
        "SCENARIO-SAMPLE-5738",
    ):
        assert marker in section or marker in normalized


def test_scenario_sample_5738_builds_valid_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5738: justified fake evidence emits ready artifact."""

    artifact = mod.build_artifact(
        root=REPO,
        evidence_runner=_fake_ready_evidence,
        freeze_affinity=False,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    output_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(saved["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert saved["preconditions_checked"]
    assert saved["upstream_artifact_hashes"]["exp5723"]["available"] is True
    assert saved["upstream_artifact_hashes"]["exp5724"]["available"] is True
    assert saved["large_size_reversal_reproduced"] is True
    assert saved["dominant_phase"]["phase"] == "pyo3_crossing"
    assert saved["optimization_hypothesis"]["justified"] is True
    assert saved["batch_factory_receipts"]["explicit_backend_name"] == "one_axis_rust"
    assert saved["scalar_api_unchanged"] is True
    assert all(row["equivalent"] is True for row in saved["python_fallback_receipts"])
    assert saved["energy_trace_mismatch_count"] == 0
    assert saved["proposal_mismatch_count"] == 0
    assert saved["exchange_mismatch_count"] == 0
    assert saved["checkpoint_mismatch_count"] == 0
    assert saved["restart_mismatch_count"] == 0
    assert saved["result_order_mismatch_count"] == 0
    assert saved["distributional_parity_receipts"][0]["n_samples"] == 10000
    assert saved["batch_backend_ready_score"] == 1.0
    assert saved["timing_claimed"] is False
    assert saved["software_speedup_claimed"] is False
    assert saved["hardware_speedup_claimed"] is False
    assert saved["fpga_or_tsu_used"] is False
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_sample_5738_actual_small_batch_parity_receipt() -> None:
    """REQ-SAMPLE-5738: real batch smoke records scalar, fallback, and order parity."""

    evidence = mod.run_batch_semantic_parity(
        workloads=mod.reproduction_workloads(
            problem_sizes=(3, 6), topology_families=("ferromagnetic_ring_easy",)
        ),
        random_seeds=(5738, 5739),
        n_samples=2,
    )

    assert evidence["energy_trace_mismatch_count"] == 0
    assert evidence["proposal_mismatch_count"] == 0
    assert evidence["exchange_mismatch_count"] == 0
    assert evidence["checkpoint_mismatch_count"] == 0
    assert evidence["restart_mismatch_count"] == 0
    assert evidence["result_order_mismatch_count"] == 0
    assert all(row["equivalent"] is True for row in evidence["python_fallback_receipts"])
    assert evidence["parity_manifest"]["result_order_match"] is True


def test_req_sample_5738_ready_score_and_validation_fail_closed() -> None:
    """REQ-SAMPLE-5738: schema validation rejects overclaims and mismatches."""

    artifact = mod.build_artifact(
        root=REPO,
        evidence_runner=_fake_ready_evidence,
        freeze_affinity=False,
    )
    mutations = [
        ("field_principles", lambda data: data["field_principles"].__setitem__("bad", "bad")),
        (
            "large_size_reversal_reproduced",
            lambda data: data.__setitem__("large_size_reversal_reproduced", False),
        ),
        ("dominant_phase", lambda data: data.__setitem__("dominant_phase", {})),
        (
            "optimization_hypothesis",
            lambda data: data["optimization_hypothesis"].__setitem__("justified", False),
        ),
        (
            "energy_trace_mismatch_count",
            lambda data: data.__setitem__("energy_trace_mismatch_count", 1),
        ),
        (
            "distributional_parity_receipts",
            lambda data: data["distributional_parity_receipts"][0].__setitem__("n_samples", 9999),
        ),
        ("timing_claimed", lambda data: data.__setitem__("timing_claimed", True)),
        (
            "software_speedup_claimed",
            lambda data: data.__setitem__("software_speedup_claimed", True),
        ),
        (
            "hardware_speedup_claimed",
            lambda data: data.__setitem__("hardware_speedup_claimed", True),
        ),
        ("fpga_or_tsu_used", lambda data: data.__setitem__("fpga_or_tsu_used", True)),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "bad")),
        (
            "batch_backend_ready_score",
            lambda data: data.__setitem__("batch_backend_ready_score", 0.0),
        ),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        (
            "reproducibility_checksum",
            lambda data: data.__setitem__("reproducibility_checksum", "bad"),
        ),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {
            "batch_backend_ready_score",
            "honest_verdict",
            "reproducibility_checksum",
        }:
            bad["batch_backend_ready_score"] = mod.ready_score(bad)
            bad["honest_verdict"] = mod.honest_verdict(bad)
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_scenario_sample_5738_main_delegates_artifact_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-SAMPLE-5738: CLI entrypoint delegates build and write steps."""

    calls: list[tuple[str, object]] = []

    def fake_build(**kwargs: Any) -> dict[str, bool]:
        calls.append(("build", kwargs))
        return {"ok": True}

    def fake_write(root: Path, artifact: dict[str, bool]) -> Path:
        calls.append(("write", (root, artifact)))
        return Path("results/fake.json")

    monkeypatch.setattr(mod, "build_artifact", fake_build)
    monkeypatch.setattr(mod, "write_output", fake_write)

    mod.main()

    assert calls == [
        ("build", {"root": mod.REPO_ROOT}),
        ("write", (mod.REPO_ROOT, {"ok": True})),
    ]
