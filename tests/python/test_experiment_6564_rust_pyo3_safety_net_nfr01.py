"""Tests for Exp6564 Rust/PyO3 Safety-Net NFR01 benchmark.

Spec refs: REQ-BENCH-6564, SCENARIO-BENCH-6564-GATE,
SCENARIO-BENCH-6564-PARITY, SCENARIO-BENCH-6564-NFR01,
SCENARIO-BENCH-6564-ATTACKS, REQ-RUSTPY-6564,
SCENARIO-RUSTPY-6564-BATCH-ORDERED-PARITY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6564_rust_pyo3_safety_net_nfr01 as mod
from scripts import adversarial_verify


pytest.importorskip("carnot._rust")

REPO = Path(__file__).resolve().parents[2]
TESTS_RUN = [{"command": "focused-exp6564", "exit_code": 0}]


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    """REQ-BENCH-6564: build a small row-derived benchmark artifact."""

    return mod.build_artifact(
        repo_root=REPO,
        result_path=Path("/tmp/experiment_6564_test_result.json"),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        warmup_iterations=1,
        benchmark_blocks=2,
        batch_sizes=(1, 4),
        repetitions_per_block=1,
    )


def _with_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
    return payload


def test_req_bench_6564_spec_declares_benchmark_contract() -> None:
    """REQ-BENCH-6564: OpenSpec owns the NFR01 benchmark contract."""

    text = (REPO / mod.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6564") :]

    for marker in (
        "SCENARIO-BENCH-6564-GATE",
        "SCENARIO-BENCH-6564-PARITY",
        "SCENARIO-BENCH-6564-NFR01",
        "SCENARIO-BENCH-6564-ATTACKS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_bench_6564_artifact_schema_and_reducer(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6564-NFR01: terminal scores recompute from rows."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["honest_verdict"].startswith(("complete_", "blocked_"))
    assert artifact["verdict_class"] in {"positive", "null", "blocked"}
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["rust_pyo3_nfr01_ready_score"] in {0.0, 1.0}
    assert artifact["aggregate_row_recomputation"] == mod.aggregate_row_recomputation(artifact)
    assert artifact["gate_check_summary"] == mod.gate_check_summary(
        artifact["aggregate_row_recomputation"]
    )
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_bench_6564_rows_cover_parity_timing_and_allocations(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6564-PARITY: rows expose exact equality and charged work."""

    parity_rows = artifact["scalar_and_batch_parity_rows"]
    unit_rows = artifact["per_unit_rows"]
    throughput_rows = artifact["throughput_and_latency_rows"]
    allocation_rows = artifact["allocation_and_copy_rows"]
    conditions = {row["condition"] for row in parity_rows}

    assert {"supported", "abstain", "fallback", "exception", "malformed", "unsupported"} <= (
        conditions
    )
    assert all(row["python_vs_pyo3_scalar_bytes_equal"] for row in parity_rows)
    assert all(row["python_vs_pyo3_batch_bytes_equal"] for row in parity_rows)
    assert all(row["error_type_equal"] for row in parity_rows)
    assert all(row["fallback_reason_equal"] for row in parity_rows)
    assert all(row["exact_downstream_equal"] for row in parity_rows)
    assert all(row["order_equal"] for row in parity_rows)

    assert {row["implementation"] for row in unit_rows} == set(mod.IMPLEMENTATIONS)
    assert {row["batch_size"] for row in unit_rows} == {1, 4}
    assert all(row["request_byte_count"] > 0 for row in unit_rows)
    assert all(row["operations"] >= 1 for row in throughput_rows)
    assert all(row["wall_time_s"] >= 0.0 for row in throughput_rows)
    assert all(row["p99_latency_s"] >= row["p50_latency_s"] for row in throughput_rows)
    assert all(row["charged_request_bytes"] > 0 for row in allocation_rows)
    assert all(row["python_tracemalloc_peak_bytes"] >= 0 for row in allocation_rows)


def test_scenario_bench_6564_gate_contract_and_preconditions(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6564-GATE: upstream and host receipts are pinned."""

    gate = artifact["upstream_gate_receipt"]
    contract = artifact["frozen_benchmark_contract"]
    preconditions = artifact["preconditions_checked"]
    build = artifact["abi_schema_and_build_receipts"]

    assert gate["upstream_artifact_path"] == mod.UPSTREAM_RELATIVE_PATH.as_posix()
    assert gate["upstream_artifact_sha256"].startswith("sha256:")
    assert gate["workload_matrix_sha256"].startswith("sha256:")
    assert contract["randomized_condition_order_seed"] == mod.RANDOM_SEED
    assert contract["nfr01_threshold_speedup"] == 10.0
    assert contract["batch_sizes"] == [1, 4]
    assert contract["warm_up_iterations"] == 1
    assert preconditions["cpu"]["model"]
    assert preconditions["ram"]["total_kib"] >= 0
    assert preconditions["timer_resolution"]["monotonic_s"] > 0.0
    assert build["batch_schema_version"] == "carnot.safety_net.router_batch_abi.v1"
    assert build["scalar_schema_version"] == "carnot.safety_net.router_abi.v1"
    assert build["required_symbols"]["safety_net_route_batch"] is True


def test_scenario_bench_6564_adversarial_verify_substrate_floor(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-BENCH-6564: adversarial verify recognizes the no-LLM benchmark."""

    path = tmp_path / "experiment_6564.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    classification = adversarial_verify._classify_inference_substrate(artifact)
    floor = adversarial_verify.duration_floor_for_artifact(artifact)
    report = adversarial_verify.verify_artifact(path)

    assert classification["kind"] == "no_llm"
    assert classification["matched_value"] == mod.INFERENCE_SUBSTRATE
    assert floor == {
        "substrate": mod.INFERENCE_SUBSTRATE,
        "min_duration_s": adversarial_verify.NO_LLM_DECLARED_MIN_DURATION_S,
        "reason": "no_llm_declared",
    }
    assert report["flag_count"] == 0


def test_scenario_bench_6564_attack_matrix_and_verdict_classes(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6564-ATTACKS: attacks and verdict classes fail closed."""

    attacks = artifact["benchmark_attack_matrix"]
    assert attacks["all_attacks_fail_closed"] is True
    assert {
        "compiler_debug_mismatch",
        "hidden_preprocessing",
        "unequal_request_counts",
        "discarded_errors",
        "warm_cache_bias",
        "timer_granularity",
        "one_outlier_median",
        "uncharged_batch_assembly",
        "aggregate_only_speedup",
    } <= {row["attack_id"] for row in attacks["rows"]}

    positive = deepcopy(artifact)
    fastest_batch = max(
        row["batch_size"]
        for row in positive["throughput_and_latency_rows"]
        if row["implementation"] == "rust_pyo3_batch"
    )
    python_scalar = next(
        row
        for row in positive["throughput_and_latency_rows"]
        if row["implementation"] == "python_scalar" and row["batch_size"] == 1
    )
    for row in positive["throughput_and_latency_rows"]:
        if row["implementation"] == "rust_pyo3_batch" and row["batch_size"] == fastest_batch:
            row["throughput_ops_s"] = python_scalar["throughput_ops_s"] * 12.0
            row["p99_latency_s"] = min(
                row["p99_latency_s"],
                positive["frozen_benchmark_contract"]["p99_latency_bound_s"] / 2.0,
            )
    aggregate = mod.aggregate_row_recomputation(positive)
    assert aggregate["verdict_class_from_rows"] == "positive"
    assert mod._status_and_verdict(aggregate)[2] == "positive"  # noqa: SLF001
    assert mod._status_and_verdict({"verdict_class_from_rows": "partial"})[2] == "partial"  # noqa: SLF001
    assert mod._status_and_verdict({"verdict_class_from_rows": "blocked"})[2] == "blocked"  # noqa: SLF001
    assert mod._status_and_verdict({"verdict_class_from_rows": "disqualified"})[2] == "disqualified"  # noqa: SLF001

    drift = deepcopy(artifact)
    drift["scalar_and_batch_parity_rows"][0]["python_vs_pyo3_batch_bytes_equal"] = False
    drift_aggregate = mod.aggregate_row_recomputation(drift)
    assert drift_aggregate["verdict_class_from_rows"] == "disqualified"

    partial = deepcopy(artifact)
    partial["benchmark_attack_matrix"]["rows"][0]["fail_closed"] = False
    partial["benchmark_attack_matrix"]["all_attacks_fail_closed"] = False
    partial_aggregate = mod.aggregate_row_recomputation(partial)
    assert partial_aggregate["verdict_class_from_rows"] == "partial"


def test_req_bench_6564_validation_edges_and_cli(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-BENCH-6564: validation and CLI paths reject malformed artifacts."""

    assert mod.sha256_file(None) == "missing"
    assert mod.sha256_file(tmp_path / "missing.bin") == "missing"
    assert mod._read_json(tmp_path / "missing.json") == {}  # noqa: SLF001
    bad_json = tmp_path / "bad-json.json"
    bad_json.write_text("{bad json}\n", encoding="utf-8")
    assert mod._read_json(bad_json) == {}  # noqa: SLF001
    assert mod._command_version(tmp_path, ("missing-command-for-exp6564",))["available"] is False  # noqa: SLF001
    assert mod._percentile([], 0.99) == 0.0  # noqa: SLF001
    assert mod._median([]) == 0.0  # noqa: SLF001
    assert mod.benchmark_request_cases(tmp_path) == []
    assert mod.scalar_and_batch_parity_rows([], None) == []
    assert mod.benchmark_rows(
        request_cases=[],
        rust_module=None,
        batch_sizes=(1,),
        benchmark_blocks=1,
        repetitions_per_block=1,
    ) == ([], [], [])

    monkeypatch.setattr(
        mod.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("forced")),
    )
    assert mod._load_rust_module() is None  # noqa: SLF001
    monkeypatch.undo()

    class _BadGovernorPath:
        def read_text(self, encoding: str = "utf-8") -> str:
            _ = encoding
            raise OSError("forced")

        def __str__(self) -> str:
            return "/fake/governor"

    class _BadThermalZone:
        def __truediv__(self, _name: str) -> "_BadThermalZone":
            return self

        def read_text(self, encoding: str = "utf-8") -> str:
            _ = encoding
            raise OSError("forced")

        def __str__(self) -> str:
            return "/fake/thermal"

    class _FakePath:
        def __init__(self, _path: str) -> None:
            pass

        def glob(self, pattern: str) -> list[object]:
            if "cpufreq" in pattern:
                return [_BadGovernorPath()]
            return [_BadThermalZone()]

    monkeypatch.setattr(mod, "Path", _FakePath)
    assert mod._cpu_governor()["rows"][0]["error"] == "forced"  # noqa: SLF001
    assert mod._thermal_state() == {"available": False, "zones": []}  # noqa: SLF001
    monkeypatch.undo()

    monkeypatch.delattr(mod.os, "sched_getaffinity")
    assert mod._current_affinity() == []  # noqa: SLF001
    monkeypatch.undo()
    monkeypatch.setattr(mod, "_current_affinity", lambda: [])
    assert mod._set_single_core_affinity()["available"] is False  # noqa: SLF001
    monkeypatch.undo()

    blocked_relative = mod.build_artifact(
        repo_root=tmp_path,
        result_path="relative-result.json",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        benchmark_blocks=1,
        batch_sizes=(1,),
        repetitions_per_block=1,
    )
    assert blocked_relative["verdict_class"] == "blocked"

    mutations = (
        ("required field set mismatch", lambda data: data.pop("status")),
        (
            "inference_substrate mismatch",
            lambda data: data.__setitem__("inference_substrate", "wrong"),
        ),
        (
            "verifier_is_oracle must be false",
            lambda data: data.__setitem__("verifier_is_oracle", True),
        ),
        (
            "honest_verdict terminal prefix mismatch",
            lambda data: data.__setitem__("honest_verdict", "bad"),
        ),
        (
            "verdict_class outside Exp6564 enum",
            lambda data: data.__setitem__("verdict_class", "surprise"),
        ),
        (
            "field_provenance must cover required fields",
            lambda data: data.__setitem__("field_provenance", {}),
        ),
        ("ready score mismatch", lambda data: data.__setitem__("rust_pyo3_nfr01_ready_score", 1.0)),
        (
            "positive verdict requires ready score 1.0",
            lambda data: (
                data.__setitem__("verdict_class", "positive"),
                data.__setitem__("rust_pyo3_nfr01_ready_score", 0.0),
            ),
        ),
        (
            "exact parity failed",
            lambda data: data["exact_downstream_equality_receipt"].__setitem__(
                "all_exact_downstream_equal",
                False,
            ),
        ),
        (
            "benchmark attack false accept",
            lambda data: data["benchmark_attack_matrix"].__setitem__(
                "all_attacks_fail_closed",
                False,
            ),
        ),
        (
            "protected files changed",
            lambda data: data["protected_files_unchanged"].__setitem__(
                "all_protected_files_unchanged",
                False,
            ),
        ),
    )
    for expected, mutate in mutations:
        candidate = deepcopy(artifact)
        mutate(candidate)
        _with_checksum(candidate)
        assert expected in mod.validate_artifact(candidate)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)

    result = tmp_path / "experiment_6564.json"
    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--result-path",
                str(result),
                "--benchmark-blocks",
                "1",
                "--batch-sizes",
                "1,2",
                "--repetitions-per-block",
                "1",
            ]
        )
        == 0
    )
    written = json.loads(result.read_text(encoding="utf-8"))
    assert mod.validate_artifact(written) == []
    assert mod.main(["--validate", "--result-path", str(result)]) == 0

    bad = tmp_path / "bad.json"
    bad.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(bad)]) == 1

    monkeypatch.setattr(mod, "build_artifact", lambda **_kwargs: {"bad": "artifact"})
    assert mod.main(["--result-path", str(tmp_path / "invalid-build.json")]) == 1
