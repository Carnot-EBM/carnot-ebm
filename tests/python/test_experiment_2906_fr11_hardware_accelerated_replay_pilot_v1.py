"""Tests for Exp 2906 FR-11 KV260 replay dispatch pilot.

Spec: REQ-LEARN-2906,
      SCENARIO-LEARN-2906,
      SCENARIO-LEARN-2906-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import carnot.eval.fr11_hardware_accelerated_replay_pilot_v1 as pilot
from carnot.eval.fr11_hardware_accelerated_replay_pilot_v1 import (
    INFERENCE_SUBSTRATE,
    OUTPUT_FILENAME,
    REQUIRED_ARTIFACT_FIELDS,
    ExperimentConfig,
    run_experiment,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ready_upstreams(root: Path) -> None:
    results = root / "results"
    _write_json(
        results / "experiment_2882_fr11_recmem_replay_scaleup_v1.json",
        {
            "honest_verdict": "complete: CPU replay path ready",
            "recmem_replay_scaleup_ready": True,
            "n_examples": 50,
            "target_examples_met": True,
            "selected_example_ids": [f"case-{index:02d}" for index in range(50)],
            "live_llm_called": False,
            "model_weights_mutated": False,
            "run_date": "20260522",
        },
    )
    _write_json(
        results / "experiment_2887_fr11_fast_slow_memory_corrigendum_v2.json",
        {
            "honest_verdict": "complete: replay corrigendum clean",
            "fr11_scaleup_clean": True,
            "best_policy": "fast_slow_memory",
            "n_examples": 50,
            "live_llm_called": False,
            "model_weights_mutated": False,
        },
    )
    _write_json(
        results / "experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json",
        {
            "honest_verdict": "complete: kv260 board dispatch recorded",
            "inference_substrate": "hardware_smoke",
            "kv260_overlay_loaded": "carnot_ising_v2_n64",
            "kv260_uio_devices_present": ["/dev/uio0", "/dev/uio4"],
            "bitstream_sha256": "a" * 64,
            "board_transcript_path": "results/experiment_2898_kv260_transcript.log",
            "board_harness_summary": {
                "selected_uio": "/dev/uio4",
                "uio0_mmap_checked": True,
            },
            "ising_problem_spec": {"n_spins": 64, "random_seed": 42},
            "preconditions_checked": [
                {"resource": "kv260_ssh", "available": True},
                {"resource": "kv260_overlay", "available": True},
                {"resource": "kv260_uio0", "available": True},
            ],
            "per_seed_results": [
                {
                    "seed": 42,
                    "n_samples": 10000,
                    "per_sample_wall_clock_us_median": 24.05,
                },
                {
                    "seed": 137,
                    "n_samples": 10000,
                    "per_sample_wall_clock_us_median": 24.04,
                },
                {
                    "seed": 271,
                    "n_samples": 10000,
                    "per_sample_wall_clock_us_median": 24.01,
                },
            ],
        },
    )


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys |= _all_keys(child)
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys |= _all_keys(child)
        return keys
    return set()


def test_scenario_learn_2906_dispatch_path_validated_from_upstreams(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2906: live KV260 dispatch evidence validates the pilot path."""

    _write_ready_upstreams(tmp_path)

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            started_at=10.0,
            clock=lambda: 13.5,
        )
    )

    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["dispatch_path_validated"] is True
    assert artifact["pilot_only"] is True
    assert artifact["no_hardware_performance_claim"] is True
    assert artifact["duration_s"] == 3.5
    assert artifact["failed_gates"] == []
    assert artifact["dispatch_path_summary"] == {
        "cpu_replay_ready": True,
        "corrigendum_clean": True,
        "kv260_live_board_dispatch_ready": True,
    }
    assert artifact["cpu_replay_summary"]["n_examples"] == 50
    assert artifact["kv260_board_summary"]["selected_uio"] == "/dev/uio4"
    assert artifact["kv260_board_summary"]["per_seed_result_count"] == 3
    assert len(artifact["cited_upstream_artifacts"]) == 3
    for citation in artifact["cited_upstream_artifacts"]:
        assert set(citation) == {"experiment_id", "path", "fields_imported", "sha256"}
        assert len(citation["sha256"]) == 64
        assert citation["fields_imported"]

    assert not any("speedup" in key.lower() for key in _all_keys(artifact))

    saved = json.loads((tmp_path / "results" / OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact


def test_scenario_learn_2906_missing_and_not_ready_upstreams_block(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2906-BLOCKED: missing or dirty CPU replay evidence fails closed."""

    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_missing_exp2882_artifact"
    assert artifact["dispatch_path_validated"] is False
    assert artifact["cited_upstream_artifacts"] == []

    _write_ready_upstreams(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_2882_fr11_recmem_replay_scaleup_v1.json",
        {
            "recmem_replay_scaleup_ready": False,
            "n_examples": 50,
            "live_llm_called": False,
            "model_weights_mutated": False,
        },
    )
    not_ready = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert not_ready["honest_verdict"] == "blocked_exp2882_replay_not_ready"
    assert "exp2882_replay_not_ready" in not_ready["failed_gates"]

    _write_ready_upstreams(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_2887_fr11_fast_slow_memory_corrigendum_v2.json",
        {"fr11_scaleup_clean": False},
    )
    dirty_corrigendum = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert dirty_corrigendum["honest_verdict"] == "blocked_exp2887_corrigendum_not_clean"
    assert "exp2887_corrigendum_not_clean" in dirty_corrigendum["failed_gates"]


def test_req_learn_2906_kv260_gate_failures_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-2906-3/4: KV260 board gates explain blocked pilot artifacts."""

    _write_ready_upstreams(tmp_path)
    kv260_path = tmp_path / "results" / "experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
    kv260 = json.loads(kv260_path.read_text(encoding="utf-8"))

    bad_substrate = dict(kv260, inference_substrate="cpu")
    _write_json(kv260_path, bad_substrate)
    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_exp2898_not_hardware_smoke"
    assert "exp2898_not_hardware_smoke" in artifact["failed_gates"]

    no_selected_uio = dict(kv260, board_harness_summary={"uio0_mmap_checked": True})
    _write_json(kv260_path, no_selected_uio)
    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_kv260_uio_dispatch_missing"

    short_seed_rows = dict(kv260, per_seed_results=kv260["per_seed_results"][:2])
    _write_json(kv260_path, short_seed_rows)
    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_kv260_seed_results_incomplete"

    bad_precondition = dict(kv260)
    bad_precondition["preconditions_checked"] = [
        {"resource": "kv260_ssh", "available": True},
        {"resource": "kv260_uio0", "available": False},
    ]
    _write_json(kv260_path, bad_precondition)
    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_kv260_precondition_failed"
    assert artifact["dispatch_path_summary"]["kv260_live_board_dispatch_ready"] is False

    empty_preconditions = dict(kv260, preconditions_checked=[])
    _write_json(kv260_path, empty_preconditions)
    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_kv260_precondition_failed"

    malformed_preconditions = dict(kv260, preconditions_checked=["not-a-dict"])
    _write_json(kv260_path, malformed_preconditions)
    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_kv260_precondition_failed"

    nonpositive_seed_row = dict(kv260)
    nonpositive_seed_row["per_seed_results"] = [dict(row) for row in kv260["per_seed_results"]]
    nonpositive_seed_row["per_seed_results"][0]["per_sample_wall_clock_us_median"] = "bad"
    _write_json(kv260_path, nonpositive_seed_row)
    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_kv260_seed_result_nonpositive"


def test_req_learn_2906_malformed_json_and_config_helpers(tmp_path: Path) -> None:
    """REQ-LEARN-2906-1/4: malformed JSON and path helpers stay auditable."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_2882_fr11_recmem_replay_scaleup_v1.json").write_text("{broken")

    malformed = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=results),
        write=False,
    )
    assert malformed["honest_verdict"] == "blocked_malformed_exp2882_artifact"
    assert "exp2882_artifact_malformed" in malformed["failed_gates"]

    _write_ready_upstreams(tmp_path)
    (results / "experiment_2882_fr11_recmem_replay_scaleup_v1.json").write_text("[]")
    non_mapping = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=results),
        write=False,
    )
    assert non_mapping["honest_verdict"] == "blocked_malformed_exp2882_artifact"

    _write_ready_upstreams(tmp_path)
    _write_json(
        results / "experiment_2882_fr11_recmem_replay_scaleup_v1.json",
        {
            "recmem_replay_scaleup_ready": True,
            "n_examples": "not-an-int",
            "live_llm_called": False,
            "model_weights_mutated": False,
        },
    )
    bad_count = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=results),
        write=False,
    )
    assert bad_count["honest_verdict"] == "blocked_exp2882_target_too_small"

    config = ExperimentConfig(repo_root=tmp_path)
    assert config.output_dir() == tmp_path / "results"
    assert config.output_path() == tmp_path / "results" / OUTPUT_FILENAME
    assert config.start_time() > 0.0
    assert pilot._round_float(1.2345678901234) == 1.234567890123
    assert pilot._relative_path(tmp_path / "outside.json", tmp_path / "root").endswith(
        "/outside.json"
    )
