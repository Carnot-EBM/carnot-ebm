"""Tests for Exp5715 one-axis Rust/Python quality and restart parity.

Spec refs: REQ-SAMPLE-5715, SCENARIO-SAMPLE-5715.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from unittest import mock

import pytest

from carnot import experiment_5715_one_axis_tempering_rust_quality_restart as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5715_one_axis_tempering_rust_quality_restart.py")


def _small_artifact() -> dict[str, object]:
    return mod.build_artifact(
        root=REPO,
        random_seeds=mod.DEFAULT_RANDOM_SEEDS[:2],
        burn_in_sweeps=2,
        sample_sweeps=6,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )


def test_req_sample_5715_spec_declares_quality_restart_contract() -> None:
    """REQ-SAMPLE-5715: OpenSpec anchors fields, controls, gates, and no-speed scope."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5715") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-SAMPLE-5715",
        str(mod.RESULT_RELATIVE_PATH),
        "Exp 5634",
        "Exp 5714",
        "without tuning",
        "Python-to-Rust",
        "Rust-to-Python",
        "corrupt checkpoint",
        "wrong ladder",
        "stale label",
        "independent-cDLS diagnostic",
        "`two_axis_arm_count` SHALL equal `0`",
        "`timing_claimed=false`",
        "`hardware_speedup_claimed=false`",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section or marker in normalized
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_sample_5715_matched_trial_preserves_quality_and_restarts() -> None:
    """REQ-SAMPLE-5715: Rust, Python, and cross-language restarts match by seed."""

    panel = mod.frozen_panel()
    rows, checkpoints, failures = mod.run_matched_trial(
        panel,
        mod.DEFAULT_RANDOM_SEEDS[:2],
        burn_in_sweeps=2,
        sample_sweeps=6,
    )

    assert failures == []
    assert len(panel) >= 4
    assert {row.arm_id for row in rows} == set(mod.ARM_IDS)
    assert all(row.two_axis_arm is False for row in rows)
    assert all(row.corrected_kernel_transitions == 24 for row in rows)
    assert all(entry["python_to_rust_pass"] is True for entry in checkpoints)
    assert all(entry["rust_to_python_pass"] is True for entry in checkpoints)

    indexed = mod._indexed_metrics(rows)
    for key, python_metrics in indexed.items():
        instance_id, seed, arm_id = key
        if arm_id != "python_uninterrupted":
            continue
        rust_metrics = indexed[(instance_id, seed, "rust_uninterrupted")]
        py_to_rust = indexed[(instance_id, seed, "python_to_rust_restart")]
        rust_to_py = indexed[(instance_id, seed, "rust_to_python_restart")]
        for metrics in (rust_metrics, py_to_rust, rust_to_py):
            comparable = {key: value for key, value in metrics.items() if key != "arm_id"}
            expected = {key: value for key, value in python_metrics.items() if key != "arm_id"}
            assert comparable == expected

    controls = mod.corrupted_checkpoint_controls(panel[0], mod.DEFAULT_RANDOM_SEEDS[0])
    assert set(controls) == set(mod.CORRUPTED_CONTROL_IDS)
    assert all(row["failed_closed"] is True for row in controls.values())


def test_scenario_sample_5715_builds_valid_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5715: runner writes the required quality/restart JSON."""

    artifact = _small_artifact()
    output_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(saved["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert saved["upstream_gate_receipts"]["exp5634"]["ready"] is True
    assert saved["upstream_gate_receipts"]["exp5714"]["ready"] is True
    assert (
        saved["source_quality_receipt"]["experiment_id"]
        == "exp5634-temperature-exchange-cdls-quality"
    )
    assert saved["preregistered_protocol"]["no_tuning"] is True
    assert len(saved["instance_manifest"]) == len(mod.frozen_panel())
    assert set(saved["instance_hashes"]) == {item.instance_id for item in mod.frozen_panel()}
    assert saved["transition_budget_parity"]["matched_corrected_transition_budget"] is True
    assert saved["swap_schedule_parity"]["matched_language_swap_schedule"] is True
    assert saved["successful_seed_count"]["value"] == 2
    assert saved["failed_seed_reasons"] == []
    assert set(saved["exact_validity_by_arm"]) == set(mod.ARM_IDS)
    assert set(saved["energy_by_arm"]) == set(mod.ARM_IDS)
    assert set(saved["feasible_hit_rate_by_arm"]) == set(mod.ARM_IDS)
    assert set(saved["ess_by_arm"]) == set(mod.ARM_IDS)
    assert set(saved["autocorrelation_by_arm"]) == set(mod.ARM_IDS)
    assert set(saved["barrier_crossings_by_arm"]) == set(mod.ARM_IDS)
    assert set(saved["temperature_round_trips_by_arm"]) == set(mod.ARM_IDS)
    assert set(saved["solve_probability_by_arm"]) == set(mod.ARM_IDS)
    assert set(saved["target_distributions_by_arm"]) == set(mod.ARM_IDS)
    assert saved["material_regression_count"] == 0
    assert saved["checkpoint_schema_version"] == mod.CHECKPOINT_SCHEMA_VERSION
    assert saved["python_to_rust_restart_pass"] is True
    assert saved["rust_to_python_restart_pass"] is True
    assert saved["restart_suffix_metrics"]["deterministic_suffix"]["exact_match_rate"] == 1.0
    assert all(
        row["failed_closed"] is True for row in saved["corrupted_checkpoint_controls"].values()
    )
    assert saved["two_axis_arm_count"] == 0
    assert saved["timing_claimed"] is False
    assert saved["hardware_speedup_claimed"] is False
    assert saved["one_axis_rust_quality_ready_score"] == 1.0
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["random_seeds"] == list(mod.DEFAULT_RANDOM_SEEDS[:2])
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert saved["honest_verdict"].startswith("complete:")
    mod.validate_artifact(saved)


def test_req_sample_5715_validation_and_score_fail_closed() -> None:
    """REQ-SAMPLE-5715: manual promotion edits, regressions, and speed claims fail."""

    artifact = _small_artifact()
    mutations = [
        ("missing required field", lambda data: data.pop("field_principles")),
        ("field_principles", lambda data: data["field_principles"].__setitem__("x", "y")),
        (
            "inference_substrate",
            lambda data: data.__setitem__("inference_substrate", "deterministic_verifier"),
        ),
        ("two_axis_arm_count", lambda data: data.__setitem__("two_axis_arm_count", 1)),
        ("timing_claimed", lambda data: data.__setitem__("timing_claimed", True)),
        (
            "hardware_speedup_claimed",
            lambda data: data.__setitem__("hardware_speedup_claimed", True),
        ),
        (
            "one_axis_rust_quality_ready_score",
            lambda data: data.__setitem__("one_axis_rust_quality_ready_score", 0.0),
        ),
        (
            "honest_verdict mismatch",
            lambda data: data.__setitem__("honest_verdict", "blocked: stale verdict"),
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
        if expected not in {"missing required field", "reproducibility_checksum"}:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    regressed = deepcopy(artifact)
    regressed["paired_intervals"]["rust_uninterrupted_vs_python_uninterrupted"][
        "mean_energy_delta_interval_95"
    ] = [0.0, mod.FROZEN_MARGINS["mean_energy"] * 10.0]
    regressed["material_regression_count"] = mod.material_regression_count(regressed)
    regressed["one_axis_rust_quality_ready_score"] = mod.ready_score(regressed)
    assert regressed["material_regression_count"] == 1
    assert regressed["one_axis_rust_quality_ready_score"] == 0.0
    assert mod.honest_verdict(regressed).startswith("blocked:")


def test_req_sample_5715_checkpoint_parser_and_control_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-SAMPLE-5715: checkpoint schema, hash, ladder, endian, and labels are guarded."""

    panel = mod.frozen_panel()
    state = mod.initial_state_for_instance(panel[0], mod.DEFAULT_RANDOM_SEEDS[0])
    checkpoint = mod.make_checkpoint(
        instance=panel[0],
        state=state,
        implementation="python",
        direction="python_to_rust",
    )
    loaded = mod.load_checkpoint(
        checkpoint,
        expected_instance=panel[0],
        state_factory=mod.OneAxisState,
    )
    assert loaded.checkpoint() == state.checkpoint()

    bad_hash = deepcopy(checkpoint)
    bad_hash["state"]["sweep"] = int(bad_hash["state"]["sweep"]) + 1
    with pytest.raises(ValueError, match="checksum"):
        mod.load_checkpoint(bad_hash, expected_instance=panel[0], state_factory=mod.OneAxisState)

    with pytest.raises(ValueError, match="object"):
        mod.load_checkpoint([], expected_instance=panel[0], state_factory=mod.OneAxisState)

    stale_instance_hash = deepcopy(checkpoint)
    stale_instance_hash["instance_hash"] = "bad"
    stale_instance_hash["payload_checksum"] = mod.checkpoint_checksum(stale_instance_hash)
    with pytest.raises(ValueError, match="instance_hash"):
        mod.load_checkpoint(
            stale_instance_hash,
            expected_instance=panel[0],
            state_factory=mod.OneAxisState,
        )

    for mutate, message in (
        (lambda data: data.__setitem__("schema_version", "wrong"), "schema_version"),
        (lambda data: data.__setitem__("byte_order", "middle"), "byte_order"),
        (lambda data: data.__setitem__("instance_id", "stale"), "instance_id"),
        (lambda data: data.__setitem__("beta_ladder_hash", "bad"), "beta_ladder_hash"),
        (lambda data: data.__setitem__("state", {"states": [[1, -1, 1]]}), "checkpoint"),
    ):
        bad = deepcopy(checkpoint)
        mutate(bad)
        bad["payload_checksum"] = mod.checkpoint_checksum(bad)
        with pytest.raises(ValueError, match=message):
            mod.load_checkpoint(bad, expected_instance=panel[0], state_factory=mod.OneAxisState)

    assert mod._interval_95([3.0]) == [3.0, 3.0]
    assert mod._autocorrelation_time([2.0, 2.0]) == 1.0
    assert mod._raises_value_error(lambda: None) is False
    missing = mod._one_upstream_receipt(REPO / "missing.json", ready_field="ready")
    assert missing["ready"] is False
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{bad", encoding="utf-8")
    assert mod._one_upstream_receipt(invalid, ready_field="ready")["ready"] is False
    assert mod.material_regression_count({"paired_intervals": []}) == 1
    malformed_intervals = {name: [] for name in mod.GATED_COMPARISONS}
    assert mod.material_regression_count({"paired_intervals": malformed_intervals}) == len(
        mod.GATED_COMPARISONS
    )
    config = mod.config_for_instance(panel[0])
    rust_classes = mod.exp5714._rust_classes()
    with pytest.raises(ValueError, match="unknown implementation"):
        mod._core_for_implementation(config, "bad", rust_classes)
    with pytest.raises(ValueError, match="unknown implementation"):
        mod._state_for_implementation(state, "bad", rust_classes)
    assert mod._summary_values([]) == {
        "mean": None,
        "interval_95": [None, None],
        "paired_row_count": 0,
    }
    empty_distribution = mod.target_distributions_by_arm([])
    assert empty_distribution["python_uninterrupted"]["energy_histogram_counts"] == []


def test_req_sample_5715_blocked_upstream_and_replay_failures_are_reported() -> None:
    """REQ-SAMPLE-5715: stale upstreams and failed row execution keep denominators honest."""

    with mock.patch.object(mod, "_one_upstream_receipt", return_value={"ready": False}):
        receipts = mod.upstream_gate_receipts(REPO)
    assert receipts["exp5634"]["ready"] is False
    assert receipts["exp5714"]["ready"] is False

    panel = mod.frozen_panel()

    def explode(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("boom")

    with mock.patch.object(mod, "_run_one_axis_arm", side_effect=explode):
        rows, checkpoint_matrix, failures = mod.run_matched_trial(
            panel[:1],
            mod.DEFAULT_RANDOM_SEEDS[:1],
            burn_in_sweeps=1,
            sample_sweeps=1,
        )
    assert rows == []
    assert checkpoint_matrix == []
    assert failures == [
        {
            "instance_id": panel[0].instance_id,
            "seed": mod.DEFAULT_RANDOM_SEEDS[0],
            "reason": "ValueError: boom",
        }
    ]


def test_req_sample_5715_main_delegates_artifact_write() -> None:
    """SCENARIO-SAMPLE-5715: CLI entrypoint delegates build and write steps."""

    artifact = {"ok": True}
    with (
        mock.patch.object(mod, "build_artifact", return_value=artifact) as build,
        mock.patch.object(mod, "write_output") as write,
    ):
        mod.main()
    build.assert_called_once_with(root=mod.REPO_ROOT, random_seeds=mod.DEFAULT_RANDOM_SEEDS)
    write.assert_called_once_with(mod.REPO_ROOT, artifact)
