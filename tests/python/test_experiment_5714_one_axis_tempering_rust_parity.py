"""Tests for Exp5714 one-axis corrected-cDLS Rust/Python parity.

Spec refs: REQ-SAMPLE-5714, SCENARIO-SAMPLE-5714.
"""

from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from unittest import mock

import pytest

from carnot import experiment_5714_one_axis_tempering_rust_parity as mod
from carnot._rust import (
    RustOneAxisTemperingConfig,
    RustOneAxisTemperingCore,
    RustOneAxisTemperingState,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5714_one_axis_tempering_rust_parity.py")


def _rust_core() -> RustOneAxisTemperingCore:
    config = mod.default_config()
    rust_config = RustOneAxisTemperingConfig(
        config.couplings.tolist(),
        config.fields.tolist(),
        list(config.beta_ladder),
        config.proposal_std,
        config.drift_scale,
    )
    return RustOneAxisTemperingCore(rust_config)


def _rust_state(seed: int = 5714) -> RustOneAxisTemperingState:
    state = mod.default_state(seed=seed)
    return RustOneAxisTemperingState(
        [row.tolist() for row in state.states],
        list(state.labels),
        state.rng_state,
        state.sweep,
    )


def test_req_sample_5714_spec_declares_one_axis_port_contract() -> None:
    """REQ-SAMPLE-5714: OpenSpec anchors one-axis Rust parity and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5714") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-SAMPLE-5714",
        str(mod.RESULT_RELATIVE_PATH),
        "Exp 5633",
        "Exp 5634",
        "Exp 5645/5646",
        "corrected_cdls_projection_mh",
        "fixed beta-label ladder",
        "label-only adjacent swap",
        "serializable seeded state/checkpoint",
        "Existing pure-Python use SHALL continue unchanged",
        "stale-label exchange",
        "wrong-sign swap",
        "uncorrected-kernel proposals",
        "collapsed ladders",
        "corrupt state",
        "Penalty-axis exchange SHALL NOT be implemented",
        "`broken_control_rejected_score` SHALL equal exactly `1.0`",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_sample_5714_rust_binding_matches_python_deterministic_kernels() -> None:
    """REQ-SAMPLE-5714: Rust energies, proposals, decisions, and swaps match Python."""

    config = mod.default_config()
    py_core = mod.PythonOneAxisTemperingCore(config)
    rust_core = _rust_core()
    source = [1, -1, 1]
    target = [-1, -1, 1]
    beta = mod.BETA_LADDER[1]
    uniforms = [0.07, 0.61, 0.44, 0.19]

    assert rust_core.energy(source) == pytest.approx(py_core.energy(source))
    assert rust_core.proposal_log_probability(source, target, beta) == pytest.approx(
        py_core.proposal_log_probability(source, target, beta),
        abs=mod.FROZEN_TOLERANCES["proposal_log_probability"],
    )

    rust_decision = rust_core.corrected_step(source, beta, uniforms)
    py_decision = py_core.corrected_step(source, beta, uniforms)
    for key in (
        "current_energy",
        "proposed_energy",
        "proposal_log_forward",
        "proposal_log_reverse",
        "log_acceptance",
    ):
        assert rust_decision[key] == pytest.approx(py_decision[key], abs=1e-12)
    assert rust_decision["accepted"] == py_decision["accepted"]
    assert rust_decision["state"] == py_decision["state"]
    assert rust_decision["proposed_state"] == py_decision["proposed_state"]

    states = [[1, -1, 1], [-1, -1, 1], [1, 1, -1]]
    labels = [2, 0, 1]
    pair = [1, 2]
    assert rust_core.swap_log_ratio(states, labels, pair) == pytest.approx(
        py_core.swap_log_ratio(states, labels, pair),
        abs=mod.FROZEN_TOLERANCES["swap_log_ratio"],
    )
    assert rust_core.swap_decision(states, labels, pair, 0.13) == py_core.swap_decision(
        states, labels, pair, 0.13
    )


def test_req_sample_5714_scheduler_checkpoint_and_cross_language_restart_match() -> None:
    """REQ-SAMPLE-5714: scheduler parity and restart checkpoints are portable."""

    py_core = mod.PythonOneAxisTemperingCore(mod.default_config())
    rust_core = _rust_core()
    py_state = mod.default_state(seed=5715)
    rust_state = _rust_state(seed=5715)

    assert rust_core.scheduler_trace() == py_core.scheduler_trace()
    assert rust_core.target_state(rust_state) == py_core.target_state(py_state)

    rust_next = rust_core.step(rust_state)
    py_next = py_core.step(py_state)
    assert rust_next.checkpoint() == py_next.checkpoint()
    assert rust_core.target_state(rust_next) == py_core.target_state(py_next)

    restarted = RustOneAxisTemperingState.from_checkpoint(py_next.checkpoint())
    assert rust_core.step(restarted).checkpoint() == py_core.step(py_next).checkpoint()

    json.dumps(rust_next.checkpoint(), sort_keys=True)
    assert RustOneAxisTemperingState.from_checkpoint(rust_next.checkpoint()).checkpoint() == (
        rust_next.checkpoint()
    )


def test_req_sample_5714_exact_target_and_broken_controls_pass() -> None:
    """REQ-SAMPLE-5714: exact enumeration passes and preregistered controls reject."""

    parity = mod.run_parity_audit(root=REPO, random_seeds=mod.DEFAULT_RANDOM_SEEDS)

    assert parity["energy_error_max"] <= mod.FROZEN_TOLERANCES["energy"]
    assert (
        parity["proposal_probability_error_max"]
        <= mod.FROZEN_TOLERANCES["proposal_log_probability"]
    )
    assert parity["swap_log_ratio_error_max"] <= mod.FROZEN_TOLERANCES["swap_log_ratio"]
    assert parity["deterministic_decision_parity"] is True
    assert parity["scheduler_parity"] is True
    assert parity["exact_target_tv_python"] <= mod.FROZEN_TOLERANCES["exact_target_tv"]
    assert parity["exact_target_tv_rust"] <= mod.FROZEN_TOLERANCES["exact_target_tv"]
    assert parity["target_marginal_delta"] <= mod.FROZEN_TOLERANCES["target_marginal"]
    assert parity["checkpoint_roundtrip_pass"] is True
    assert parity["cross_language_restart_pass"] is True
    assert parity["python_fallback_equivalence"] is True
    assert {row["control_id"] for row in parity["broken_control_results"]} == set(
        mod.BROKEN_CONTROL_IDS
    )
    assert all(row["rejected"] is True for row in parity["broken_control_results"])
    assert parity["broken_control_rejected"] is True
    assert parity["broken_control_rejected_score"] == 1.0


def test_req_sample_5714_malformed_abi_inputs_fail_safely() -> None:
    """REQ-SAMPLE-5714: malformed Rust/PyO3 inputs raise ValueError, not UB."""

    with pytest.raises(ValueError, match="couplings must be square"):
        RustOneAxisTemperingConfig([[0.0, 0.1]], [0.0, 0.1], list(mod.BETA_LADDER))
    with pytest.raises(ValueError, match="beta_ladder"):
        RustOneAxisTemperingConfig([[0.0]], [0.0], [0.8, 0.8])
    with pytest.raises(ValueError, match="spin"):
        _rust_core().energy([1, 0, -1])
    with pytest.raises(ValueError, match="labels"):
        RustOneAxisTemperingState([[1, -1, 1], [1, -1, 1], [1, -1, 1]], [0, 0, 2], 7, 0)
    with pytest.raises(ValueError, match="checkpoint"):
        RustOneAxisTemperingState.from_checkpoint({"states": [[1, -1, 1]]})


def test_scenario_sample_5714_builds_valid_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5714: parity runner emits the required JSON artifact."""

    artifact = mod.build_artifact(
        root=REPO,
        random_seeds=mod.DEFAULT_RANDOM_SEEDS,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    output_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(saved["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert saved["openspec_requirement_ids"] == list(mod.SPEC_REFS)
    assert saved["source_promotion_receipts"]["exp5633"]["ready"] is True
    assert saved["source_promotion_receipts"]["exp5634"]["quality_mixing_ready"] is True
    assert saved["source_promotion_receipts"]["exp5645_two_axis_retired"]["ready"] is False
    assert saved["source_artifact_hashes"]["experiment_5633"] == mod.file_sha256(
        REPO / "results/experiment_5633_temperature_exchange_cdls_exact_audit.json"
    )
    assert saved["rust_source_paths"] == [path.as_posix() for path in mod.RUST_SOURCE_PATHS]
    assert saved["python_binding_paths"] == [path.as_posix() for path in mod.PYTHON_BINDING_PATHS]
    assert saved["python_reference_paths"] == [
        path.as_posix() for path in mod.PYTHON_REFERENCE_PATHS
    ]
    assert saved["compiler_and_toolchain"]["cargo_available"] is True
    assert saved["pyo3_version"] == "0.24"
    assert saved["abi_receipt"]["malformed_input_failures"] == len(mod.MALFORMED_INPUT_IDS)
    assert saved["two_axis_code_added"] is False
    assert saved["timing_claimed"] is False
    assert saved["hardware_speedup_claimed"] is False
    assert saved["one_axis_rust_parity_ready_score"] == 1.0
    assert saved["broken_control_rejected_score"] == 1.0
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_sample_5714_validation_and_fallback_fail_closed() -> None:
    """REQ-SAMPLE-5714: manual score edits and missing Rust fallback fail closed."""

    artifact = mod.build_artifact(root=REPO, random_seeds=mod.DEFAULT_RANDOM_SEEDS)
    mutations = [
        ("missing required field", lambda data: data.pop("field_principles")),
        ("field_principles", lambda data: data["field_principles"].__setitem__("x", "y")),
        (
            "openspec_requirement_ids",
            lambda data: data.__setitem__("openspec_requirement_ids", ["REQ-BAD"]),
        ),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "bad")),
        ("two_axis_code_added", lambda data: data.__setitem__("two_axis_code_added", True)),
        ("timing_claimed", lambda data: data.__setitem__("timing_claimed", True)),
        (
            "hardware_speedup_claimed",
            lambda data: data.__setitem__("hardware_speedup_claimed", True),
        ),
        (
            "one_axis_rust_parity_ready_score",
            lambda data: data.__setitem__("one_axis_rust_parity_ready_score", 0.0),
        ),
        (
            "broken_control_rejected_score",
            lambda data: data.__setitem__("broken_control_rejected_score", 0.0),
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

    mod_name = "carnot._rust_compat"
    saved = sys.modules.pop(mod_name, None)
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "carnot._rust":
            raise ImportError("no rust extension")
        return original_import(name, *args, **kwargs)

    try:
        with mock.patch("builtins.__import__", side_effect=_fake_import):
            compat = importlib.import_module(mod_name)
        assert compat.RUST_AVAILABLE is False
        assert compat.RustOneAxisTemperingCore is None
    finally:
        sys.modules.pop(mod_name, None)
        if saved is not None:
            sys.modules[mod_name] = saved


def test_req_sample_5714_python_reference_validation_branches_fail_closed() -> None:
    """REQ-SAMPLE-5714: Python reference validators reject malformed local inputs."""

    config = mod.default_config()
    core = mod.PythonOneAxisTemperingCore(config)
    valid_checkpoint = mod.default_state(seed=5714).checkpoint()
    assert mod.OneAxisState.from_checkpoint(valid_checkpoint).checkpoint() == valid_checkpoint

    bad_configs = [
        ([[0.0, 0.1]], [0.0, 0.1], mod.BETA_LADDER, 0.72, 0.17, "couplings"),
        ([[0.0]], [0.0, 0.1], mod.BETA_LADDER, 0.72, 0.17, "fields"),
        ([[float("nan")]], [0.0], mod.BETA_LADDER, 0.72, 0.17, "finite"),
        ([[0.0]], [0.0], (0.8,), 0.72, 0.17, "at least two"),
        ([[0.0]], [0.0], (0.8, float("nan")), 0.72, 0.17, "finite and positive"),
        ([[0.0]], [0.0], (0.8, 1.25), 0.0, 0.17, "proposal_std"),
        ([[0.0]], [0.0], (0.8, 1.25), 0.72, float("nan"), "drift_scale"),
    ]
    for couplings, fields, ladder, proposal_std, drift_scale, message in bad_configs:
        with pytest.raises(ValueError, match=message):
            mod.OneAxisConfig(couplings, fields, ladder, proposal_std, drift_scale)

    bad_states = [
        ([], (), 0, 0, "two-dimensional"),
        ([[1, 0, -1]], (0,), 0, 0, "spin"),
        ([[1, -1], [1, -1]], (0, 0), 0, 0, "labels"),
        ([[1, -1]], (0,), -1, 0, "rng_state"),
        ([[1, -1]], (0,), 0, -1, "sweep"),
    ]
    for states, labels, rng_state, sweep, message in bad_states:
        with pytest.raises(ValueError, match=message):
            mod.OneAxisState(states, labels, rng_state, sweep)

    with pytest.raises(ValueError, match="uniforms length"):
        core.corrected_step([1, -1, 1], mod.BETA_LADDER[0], [0.1])
    with pytest.raises(ValueError, match="uniforms"):
        core.corrected_step([1, -1, 1], mod.BETA_LADDER[0], [0.1, 0.2, 0.3, 1.0])
    with pytest.raises(ValueError, match="swap uniform"):
        core.swap_decision([[1, -1, 1], [-1, -1, 1], [1, 1, -1]], [0, 1, 2], [0, 1], 1.0)

    assert mod._acceptance_probability(-800.0) == 0.0
    assert mod._raises_value_error(lambda: None) is False
    with pytest.raises(ValueError, match="dimension"):
        mod._spin_array([1, -1], 3)
    with pytest.raises(ValueError, match="spin"):
        mod._spin_array([1, 0, -1], 3)
    with pytest.raises(ValueError, match="states"):
        mod._state_collection([[1, -1, 1]], [0], config)
    with pytest.raises(ValueError, match="spin"):
        mod._state_collection([[1, 0, 1], [1, -1, 1], [1, 1, -1]], [0, 1, 2], config)
    with pytest.raises(ValueError, match="labels"):
        mod._state_collection([[1, -1, 1], [1, -1, 1], [1, 1, -1]], [0, 0, 2], config)
    with pytest.raises(ValueError, match="exactly two"):
        mod._label_pair([0], 3)
    with pytest.raises(ValueError, match="adjacent"):
        mod._label_pair([0, 2], 3)


def test_req_sample_5714_provenance_and_replay_failures_are_reported() -> None:
    """REQ-SAMPLE-5714: stale receipts and replay mismatches fail closed."""

    hashes = {
        "experiment_5622": "h5622",
        "experiment_5633": "h5633",
        "experiment_5634": "h5634",
        "experiment_5645": "h5645",
        "experiment_5646": "h5646",
    }

    def payloads(
        *,
        source_sha: str = "source-ok",
        result_sha: str = "result-ok",
        exp5633_sha: str = "h5633",
    ) -> list[dict[str, object]]:
        return [
            {"kernel_audit_ready_score": 1.0},
            {
                "corrected_kernel_receipt": {
                    "source_path": "source.py",
                    "result_path": "result.json",
                    "source_sha256": source_sha,
                    "result_sha256": result_sha,
                },
                "replica_exchange_kernel_ready_score": 1.0,
            },
            {
                "upstream_gate_receipts": {"exp5633": {"sha256": exp5633_sha}},
                "quality_mixing_ready": True,
            },
            {"two_axis_quality_ready_score": 0.0},
            {"honest_verdict": "blocked: retired two-axis path"},
        ]

    def fake_file_sha(path: Path) -> str:
        return {"source.py": "source-ok", "result.json": "result-ok"}[Path(path).name]

    for kwargs, message in (
        ({"source_sha": "bad"}, "source hash is stale"),
        ({"result_sha": "bad"}, "artifact hash is stale"),
        ({"exp5633_sha": "bad"}, "Exp5633 receipt hash is stale"),
    ):
        with (
            mock.patch.object(mod, "_read_json", side_effect=payloads(**kwargs)),
            mock.patch.object(mod, "source_artifact_hashes", return_value=hashes),
            mock.patch.object(mod, "file_sha256", side_effect=fake_file_sha),
        ):
            with pytest.raises(ValueError, match=message):
                mod.source_promotion_receipts(REPO)

    blocked = mod.build_artifact(root=REPO, random_seeds=mod.DEFAULT_RANDOM_SEEDS)
    blocked["deterministic_decision_parity"] = False
    blocked["one_axis_rust_parity_ready_score"] = mod.ready_score(blocked)
    assert blocked["one_axis_rust_parity_ready_score"] == 0.0
    assert mod.honest_verdict(blocked).startswith("blocked:")

    class BadState:
        def __init__(self, checkpoint: dict[str, object]) -> None:
            self._checkpoint = dict(checkpoint)

        @classmethod
        def from_checkpoint(cls, checkpoint: dict[str, object]) -> "BadState":
            mutated = dict(checkpoint)
            mutated["rng_state"] = int(mutated["rng_state"]) + 1
            return cls(mutated)

        def checkpoint(self) -> dict[str, object]:
            return dict(self._checkpoint)

    class BadCore:
        def scheduler_trace(self) -> list[str]:
            return mod.PythonOneAxisTemperingCore(mod.default_config()).scheduler_trace()

        def step(self, state: BadState) -> BadState:
            mutated = state.checkpoint()
            mutated["rng_state"] = int(mutated["rng_state"]) + 1
            return BadState(mutated)

    with mock.patch.object(mod, "_rust_core_from_config", return_value=BadCore()):
        replay = mod._deterministic_replay_parity({"state": BadState}, [5714])
    assert replay == {
        "deterministic_decision_parity": False,
        "scheduler_parity": True,
        "checkpoint_roundtrip_pass": False,
        "cross_language_restart_pass": False,
    }


def test_req_sample_5714_main_delegates_artifact_write() -> None:
    """SCENARIO-SAMPLE-5714: CLI entrypoint delegates build and write steps."""

    artifact = {"ok": True}
    with (
        mock.patch.object(mod, "build_artifact", return_value=artifact) as build,
        mock.patch.object(mod, "write_output") as write,
    ):
        mod.main()
    build.assert_called_once_with(root=mod.REPO_ROOT, random_seeds=mod.DEFAULT_RANDOM_SEEDS)
    write.assert_called_once_with(mod.REPO_ROOT, artifact)
