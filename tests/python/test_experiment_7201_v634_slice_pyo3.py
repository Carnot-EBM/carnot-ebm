"""Persistent fixed-cardinality PyO3 boundary tests.

Spec: REQ-RUSTPY-7201 and SCENARIO-RUSTPY-7201-PERSISTENT-PARITY.
"""

from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
import subprocess
import sysconfig
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_7189_v633_rust_slice_parity as exp7189
from carnot import experiment_7201_v634_slice_pyo3 as exp
from carnot import experiment_7187_v633_slice_sampler as slices


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def compiled_binding() -> tuple[object, dict[str, object]]:
    """REQ-RUSTPY-7201-NO-FALLBACK builds and loads the real extension."""

    extension = exp.build_pyo3_extension(REPO)
    module, receipt = exp.load_compiled_binding(REPO, extension)
    assert receipt["python_fallback_used"] is False
    return module, receipt


@pytest.fixture(scope="session")
def rust_bridge() -> Path:
    """REQ-RUSTPY-7201-NO-SPEED-GATE retains the process baseline."""

    return exp7189.build_rust_bridge(REPO)


@pytest.fixture(scope="session")
def ready_artifact(
    compiled_binding: tuple[object, dict[str, object]], rust_bridge: Path
) -> dict[str, object]:
    """SCENARIO-RUSTPY-7201-PERSISTENT-PARITY builds frozen evidence once."""

    module, receipt = compiled_binding
    return exp.build_artifact(
        root=REPO,
        binding_module=module,
        binding_receipt=receipt,
        bridge_path=rust_bridge,
    )


def _sampler(module: object, *, n: int = 8, k: int = 2, seed: int = 718701) -> object:
    instance = slices.make_frustrated_instance(n, seed)
    return module.RustFixedCardinalitySampler(list(instance.edges), list(instance.fields), k, 2.0)


def test_req_rustpy_7201_explicit_tape_matches_python_exactly(
    compiled_binding: tuple[object, dict[str, object]],
) -> None:
    """REQ-RUSTPY-7201-REPLAY checks decisions, states, and energy deltas."""

    module, _ = compiled_binding
    instance = slices.make_frustrated_instance(8, 718701)
    initial = np.asarray([slices.enumerate_slice(8, 2)[3]], dtype=np.int8)
    tape = exp7189.make_replay_tape(8, 2, seed=718900, steps=32)
    positive = np.asarray([[row["positive_index"] for row in tape]], dtype=np.uintp)
    negative = np.asarray([[row["negative_index"] for row in tape]], dtype=np.uintp)
    uniforms = np.asarray([[row["uniform"] for row in tape]], dtype=np.float64)

    rust = _sampler(module).replay_batch(initial, positive, negative, uniforms)[0]
    python = exp7189.python_replay(instance, 2, 2.0, initial[0].tolist(), tape)

    assert rust["final_state"] == python["final_state"]
    assert [row["accepted"] for row in rust["steps"]] == [
        row["accepted"] for row in python["steps"]
    ]
    assert [row["state"] for row in rust["steps"]] == [row["state"] for row in python["steps"]]
    assert (
        max(
            abs(left["delta_energy"] - right["delta_energy"])
            for left, right in zip(rust["steps"], python["steps"], strict=True)
        )
        <= 1.0e-12
    )
    assert all(row["state"].count(1) == 2 for row in rust["steps"])


def test_req_rustpy_7201_batch_one_and_many_share_the_same_path(
    compiled_binding: tuple[object, dict[str, object]],
) -> None:
    """REQ-RUSTPY-7201-BATCH keeps scalar work inside the batch API."""

    module, _ = compiled_binding
    sampler = _sampler(module)
    states = np.asarray(slices.enumerate_slice(8, 2)[:3], dtype=np.int8)
    tape = exp7189.make_replay_tape(8, 2, seed=9, steps=5)
    positive = np.tile(np.asarray([row["positive_index"] for row in tape], dtype=np.uintp), (3, 1))
    negative = np.tile(np.asarray([row["negative_index"] for row in tape], dtype=np.uintp), (3, 1))
    uniforms = np.tile(np.asarray([row["uniform"] for row in tape], dtype=np.float64), (3, 1))

    scalar = sampler.replay_batch(states[:1], positive[:1], negative[:1], uniforms[:1])
    batch = sampler.replay_batch(states, positive, negative, uniforms)

    assert len(scalar) == 1
    assert len(batch) == 3
    assert batch[0] == scalar[0]
    assert all(row["final_state"].count(1) == 2 for row in batch)


def test_req_rustpy_7201_reuses_storage_and_owns_results(
    compiled_binding: tuple[object, dict[str, object]],
) -> None:
    """REQ-RUSTPY-7201-BUFFER keeps capacity and never borrows returned data."""

    module, _ = compiled_binding
    sampler = _sampler(module)
    states = np.asarray([slices.enumerate_slice(8, 2)[0]], dtype=np.int8)
    positive = np.zeros((1, 8), dtype=np.uintp)
    negative = np.zeros((1, 8), dtype=np.uintp)
    uniforms = np.full((1, 8), 0.5, dtype=np.float64)

    first = sampler.replay_batch(states, positive, negative, uniforms)
    before = sampler.buffer_receipt()
    saved = copy.deepcopy(first)
    states[:] = states[:, ::-1]
    positive[:] = 1
    del states, positive, negative, uniforms
    gc.collect()
    after_call = sampler.replay_batch(
        np.asarray([slices.enumerate_slice(8, 2)[0]], dtype=np.int8),
        np.zeros((1, 8), dtype=np.uintp),
        np.zeros((1, 8), dtype=np.uintp),
        np.full((1, 8), 0.5, dtype=np.float64),
    )
    after = sampler.buffer_receipt()

    assert first == saved
    assert after_call == saved
    assert after["tape_capacity"] == before["tape_capacity"]
    assert after["call_count"] == before["call_count"] + 1


@pytest.mark.parametrize(
    ("states", "positive", "negative", "uniforms", "message"),
    [
        (
            np.ones((1, 7), dtype=np.int8),
            np.zeros((1, 1), dtype=np.uintp),
            np.zeros((1, 1), dtype=np.uintp),
            np.zeros((1, 1), dtype=np.float64),
            "state width",
        ),
        (
            np.asarray([slices.enumerate_slice(8, 2)[0]], dtype=np.int8),
            np.zeros((2, 1), dtype=np.uintp),
            np.zeros((1, 1), dtype=np.uintp),
            np.zeros((1, 1), dtype=np.float64),
            "batch shape",
        ),
        (
            np.asarray([slices.enumerate_slice(8, 2)[0]], dtype=np.int8),
            np.zeros((1, 2), dtype=np.uintp),
            np.zeros((1, 1), dtype=np.uintp),
            np.zeros((1, 2), dtype=np.float64),
            "tape shape",
        ),
    ],
)
def test_req_rustpy_7201_invalid_shapes_fail_closed(
    compiled_binding: tuple[object, dict[str, object]],
    states: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    uniforms: np.ndarray,
    message: str,
) -> None:
    """REQ-RUSTPY-7201-NO-FALLBACK rejects malformed array shapes."""

    module, _ = compiled_binding
    with pytest.raises(ValueError, match=message):
        _sampler(module).replay_batch(states, positive, negative, uniforms)


def test_req_rustpy_7201_seeded_streams_and_state_serialization(
    compiled_binding: tuple[object, dict[str, object]],
) -> None:
    """REQ-RUSTPY-7201-SERIALIZATION round-trips restartable Rust state."""

    module, _ = compiled_binding
    sampler = _sampler(module)
    states = np.asarray(slices.enumerate_slice(8, 2)[:2], dtype=np.int8)
    chains = sampler.run_seeded_batch(states, [720101, 720102], 10, 40)
    encoded = sampler.serialize_state(chains[0]["final_state"])
    python_state = json.loads(encoded)
    decoded = sampler.deserialize_state(
        json.dumps(python_state, separators=(",", ":"), sort_keys=True)
    )

    assert len(chains) == 2
    assert chains[0]["final_state"]["rng_state"] != chains[1]["final_state"]["rng_state"]
    assert all(sample.count(1) == 2 for chain in chains for sample in chain["samples"])
    assert decoded == python_state == chains[0]["final_state"]
    with pytest.raises(ValueError, match="seed count"):
        sampler.run_seeded_batch(states, [1], 0, 2)
    with pytest.raises(ValueError, match="positive"):
        sampler.run_seeded_batch(states, [1, 2], 0, 0)


def test_req_rustpy_7201_preconditions_include_quarantine_and_failed_value() -> None:
    """REQ-RUSTPY-7201-NO-FALLBACK rejects quarantine before upstream use."""

    checks, hashes = exp.collect_preconditions(REPO)
    by_name = {row["check"]: row for row in checks}
    assert all(row["passed"] is True for row in checks)
    assert by_name["upstream_quarantine_flags"]["observed_value"]["quarantined"] is False
    assert by_name["upstream_known_failed_value"]["observed_value"] is False
    assert by_name["same_milestone_gate_fields"]["observed_value"] == exp.EXPECTED_TASK_CONTRACT
    assert len(hashes) == len(exp.REQUIRED_SOURCE_PATHS)

    upstream = json.loads((REPO / exp.UPSTREAM_RESULT_PATH).read_text(encoding="utf-8"))
    upstream["flagged_adversarial"] = True
    quarantine = exp.upstream_quarantine_observation(upstream, manifest_match=False)
    assert quarantine["quarantined"] is True
    assert "flagged_adversarial" in quarantine["active_flags"]


def test_scenario_rustpy_7201_artifact_is_compiled_complete_and_recomputable(
    ready_artifact: dict[str, object],
) -> None:
    """SCENARIO-RUSTPY-7201-PERSISTENT-PARITY verifies terminal evidence."""

    assert exp.validate_artifact(ready_artifact, root=REPO) == []
    assert ready_artifact["pyo3_slice_ready_score"] == 1
    assert ready_artifact["compiled_binding_receipt"]["python_fallback_used"] is False
    assert ready_artifact["MODEL_SPECS"] == []
    assert ready_artifact["model_invoked"] is False
    assert {row["phase"] for row in ready_artifact["phase_cost_rows"]} >= {
        "setup",
        "serialization",
        "process_launch",
        "kernel",
        "parsing",
    }
    assert all(
        row["error"] is None and row["abstention"] is False for row in ready_artifact["rows"]
    )
    assert len(ready_artifact["transition_rows"]) == 96
    assert all(receipt["passed"] is True for receipt in ready_artifact["e2e_receipts"])


def test_req_rustpy_7201_validator_rejects_forged_readiness(
    ready_artifact: dict[str, object],
) -> None:
    """REQ-RUSTPY-7201-NO-FALLBACK recomputes parity and binary evidence."""

    attacks = [
        (lambda item: item["transition_rows"][0].update(passed=False), "transition_rows_invalid"),
        (
            lambda item: item["compiled_binding_receipt"].update(python_fallback_used=True),
            "compiled_binding_receipt_invalid",
        ),
        (lambda item: item.update(pyo3_slice_ready_score=0), "readiness_invalid"),
        (lambda item: item["e2e_receipts"].pop(), "e2e_receipts_invalid"),
    ]
    for mutate, expected in attacks:
        changed = copy.deepcopy(ready_artifact)
        mutate(changed)
        changed["rows"] = (
            changed["transition_rows"] + changed["distribution_rows"] + changed["phase_cost_rows"]
        )
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected in exp.validate_artifact(changed)


def test_req_rustpy_7201_blocked_artifact_names_external_failure(tmp_path: Path) -> None:
    """REQ-RUSTPY-7201-NO-FALLBACK emits the exact terminal block."""

    failed = [
        {
            "check": "compiler",
            "upstream": "host_toolchain",
            "field": "cargo",
            "expected_value": True,
            "observed_value": False,
            "passed": False,
        }
    ]
    artifact = exp.build_artifact(root=REPO, preconditions=failed, source_hashes={})
    output = tmp_path / "blocked.json"
    receipt = exp.atomic_write(output, artifact)

    assert artifact["status"] == "blocked_external_precondition"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["observed_value"] is False
    assert exp.validate_artifact(artifact) == []
    assert receipt["atomic_replace"] is True
    assert exp.main(["--validate", str(output)]) == 0
    assert exp.main(["--date", "19000101"]) == 2


def test_req_rustpy_7201_binding_path_uses_python_extension_suffix(
    compiled_binding: tuple[object, dict[str, object]],
) -> None:
    """REQ-RUSTPY-7201-NO-FALLBACK binds the receipt to the loaded binary."""

    _, receipt = compiled_binding
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    assert suffix and str(receipt["loaded_module_path"]).endswith(suffix)
    assert receipt["binary_sha256"].startswith("sha256:")
    assert receipt["class_name"] == "RustFixedCardinalitySampler"


def test_req_rustpy_7201_contract_and_manifest_defensive_reads(tmp_path: Path) -> None:
    """REQ-RUSTPY-7201-NO-FALLBACK handles absent and malformed control files."""

    assert exp._task_contract(tmp_path) is None
    roadmap = tmp_path / exp.ROADMAP_PATH
    roadmap.write_text("tasks: invalid\n", encoding="utf-8")
    assert exp._task_contract(tmp_path) is None
    roadmap.write_text("tasks: [{id: another}]\n", encoding="utf-8")
    assert exp._task_contract(tmp_path) is None

    nested = {"retired": [{"experiment_ids": [1, "exp7189-old"]}]}
    assert exp._manifest_mentions_experiment(nested, "7189") is True
    assert exp._manifest_mentions_experiment([{"experiment_id": 7189}], "7189") is True
    assert exp._manifest_mentions_experiment({"reason": "7189 is prose"}, "7189") is False
    quarantine = exp.upstream_quarantine_observation({}, manifest_match=True)
    assert quarantine["quarantined"] is True
    assert quarantine["active_flags"] == ["exclusion_manifest"]


def test_req_rustpy_7201_preconditions_fail_closed_on_unreadable_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RUSTPY-7201-NO-FALLBACK records malformed upstream and manifest bytes."""

    (tmp_path / "bad-upstream.json").write_text("not-json", encoding="utf-8")
    (tmp_path / "bad-exclusion.yaml").write_text("[", encoding="utf-8")
    monkeypatch.setattr(exp, "UPSTREAM_RESULT_PATH", Path("bad-upstream.json"))
    monkeypatch.setattr(exp, "EXCLUSION_PATH", Path("bad-exclusion.yaml"))
    monkeypatch.setattr(
        exp,
        "REQUIRED_SOURCE_PATHS",
        (Path("bad-upstream.json"), Path("bad-exclusion.yaml")),
    )
    checks, hashes = exp.collect_preconditions(
        tmp_path,
        result_path=tmp_path / "out" / "result.json",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    by_name = {row["check"]: row for row in checks}
    assert by_name["upstream_artifact_bytes"]["passed"] is False
    assert by_name["driving_capability_spec"]["passed"] is False
    assert len(hashes) == 2


def test_req_rustpy_7201_build_and_loader_failures_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RUSTPY-7201-NO-FALLBACK exposes compiler, suffix, and loader errors."""

    monkeypatch.setattr(exp.sysconfig, "get_config_var", lambda _name: None)
    with pytest.raises(RuntimeError, match="suffix"):
        exp._extension_path(tmp_path)
    monkeypatch.undo()

    class FailedProcess:
        stdout = iter(["compiler output\n"])

        def wait(self, timeout: int) -> int:
            assert timeout == 600
            return 2

    monkeypatch.setattr(exp.subprocess, "Popen", lambda *args, **kwargs: FailedProcess())
    with pytest.raises(RuntimeError, match="exit 2"):
        exp._stream_process(["compiler"], root=tmp_path, operation="compiler")
    monkeypatch.undo()

    monkeypatch.setattr(exp, "_stream_process", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="library is missing"):
        exp.build_pyo3_extension(tmp_path)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("no_spec", "cannot create a loader"),
        ("wrong_path", "does not match copied path"),
        ("missing_class", "missing RustFixedCardinalitySampler"),
    ],
)
def test_req_rustpy_7201_fresh_loader_rejects_invalid_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    message: str,
) -> None:
    """REQ-RUSTPY-7201-NO-FALLBACK validates the freshly copied binary module."""

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    assert suffix
    extension = tmp_path / f"source{suffix}"
    extension.write_bytes(b"compiled-placeholder")
    monkeypatch.delitem(exp.sys.modules, "carnot._rust", raising=False)
    if mode == "no_spec":
        monkeypatch.setattr(exp.importlib.util, "spec_from_file_location", lambda *args: None)
    else:
        expected_load = tmp_path / "target" / "exp7201-pyo3-load" / extension.name
        module = ModuleType("carnot._rust")
        module.__file__ = str(extension if mode == "wrong_path" else expected_load)
        if mode != "missing_class":
            module.RustFixedCardinalitySampler = object

        class Loader:
            def exec_module(self, loaded: ModuleType) -> None:
                assert loaded is module

        fake_spec = SimpleNamespace(loader=Loader())
        monkeypatch.setattr(exp.importlib.util, "spec_from_file_location", lambda *args: fake_spec)
        monkeypatch.setattr(exp.importlib.util, "module_from_spec", lambda _spec: module)

    with pytest.raises(RuntimeError, match=message):
        exp.load_compiled_binding(tmp_path, extension)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("process_error", "profile bridge failed"),
        ("bridge_mismatch", "profile arms diverged"),
        ("python_mismatch", "Python control diverged"),
    ],
)
def test_req_rustpy_7201_profiler_rejects_arm_failures(
    monkeypatch: pytest.MonkeyPatch, mode: str, message: str
) -> None:
    """REQ-RUSTPY-7201-NO-SPEED-GATE retains failed phase comparisons."""

    class FakeSampler:
        def __init__(self, *_args: object) -> None:
            pass

        def replay_batch(self, states: np.ndarray, *_tapes: np.ndarray) -> list[dict[str, object]]:
            return [{"final_state": states[0].tolist()}]

    fake_binding = ModuleType("carnot._rust")
    fake_binding.RustFixedCardinalitySampler = FakeSampler

    class FakeProcess:
        returncode = 2 if mode == "process_error" else 0

        def communicate(self, input: bytes, timeout: int) -> tuple[bytes, bytes]:
            assert timeout == 120
            request = json.loads(input)
            final_state = request["initial_state"]
            if mode == "bridge_mismatch":
                final_state = []
            return json.dumps({"final_state": final_state}).encode(), b"failure"

    monkeypatch.setattr(exp, "PROFILE_SIZES", (32,))
    monkeypatch.setattr(exp, "PROFILE_CARDINALITIES", (2,))
    monkeypatch.setattr(exp, "PROFILE_SEEDS", (1,))
    monkeypatch.setattr(exp, "PROFILE_STEPS", 1)
    monkeypatch.setattr(exp.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    if mode == "python_mismatch":
        monkeypatch.setattr(
            exp.exp7189,
            "python_replay",
            lambda *args, **kwargs: {"final_state": []},
        )

    with pytest.raises((RuntimeError, ValueError), match=message):
        exp.profile_phase_costs(fake_binding, Path("bridge"))


def test_req_rustpy_7201_build_artifact_records_compiled_prerequisite_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RUSTPY-7201-NO-FALLBACK makes build and bridge failures terminal."""

    passed = [
        {
            "check": "synthetic",
            "upstream": "test",
            "field": "ready",
            "expected_value": True,
            "observed_value": True,
            "passed": True,
        }
    ]
    monkeypatch.setattr(
        exp,
        "build_pyo3_extension",
        lambda _root: (_ for _ in ()).throw(RuntimeError("compiler absent")),
    )
    blocked = exp.build_artifact(root=tmp_path, preconditions=passed, source_hashes={})
    assert blocked["gate_check_summary"]["observed_value"] == "compiler absent"

    module = ModuleType("carnot._rust")
    receipt = {
        "compiled": True,
        "python_fallback_used": False,
        "loaded_module_path": "binding",
    }
    blocked_bridge = exp.build_artifact(
        root=tmp_path,
        binding_module=module,
        binding_receipt=receipt,
        bridge_path=tmp_path / "missing-bridge",
        preconditions=passed,
        source_hashes={},
    )
    assert "subprocess baseline" in blocked_bridge["gate_check_summary"]["observed_value"]


def test_req_rustpy_7201_build_artifact_can_bootstrap_the_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RUSTPY-7201-NO-FALLBACK uses the built extension when none is injected."""

    passed = [
        {
            "check": "synthetic",
            "upstream": "test",
            "field": "ready",
            "expected_value": True,
            "observed_value": True,
            "passed": True,
        }
    ]
    extension = tmp_path / "binding.so"
    extension.write_bytes(b"binding")
    bridge = tmp_path / "bridge"
    bridge.write_text("bridge", encoding="utf-8")
    bridge.chmod(0o700)
    module = ModuleType("carnot._rust")
    receipt = {
        "compiled": True,
        "python_fallback_used": False,
        "loaded_module_path": str(extension),
    }
    monkeypatch.setattr(exp, "build_pyo3_extension", lambda _root: extension)
    monkeypatch.setattr(exp, "load_compiled_binding", lambda *_args: (module, receipt))
    monkeypatch.setattr(exp.exp7189, "build_rust_bridge", lambda _root: bridge)
    monkeypatch.setattr(exp, "run_transition_checks", lambda _binding: ([], {"passed": False}))
    monkeypatch.setattr(exp, "run_distribution_checks", lambda _binding: [])
    monkeypatch.setattr(
        exp,
        "profile_phase_costs",
        lambda _binding, _bridge: ([], {"supported": False}),
    )
    monkeypatch.setattr(exp, "run_e2e_receipts", lambda _binding: [])

    artifact = exp.build_artifact(
        root=tmp_path,
        preconditions=passed,
        source_hashes={},
    )
    assert artifact["compiled_binding_receipt"] == receipt
    assert artifact["pyo3_slice_ready_score"] == 0


def test_req_rustpy_7201_validator_rejects_every_terminal_mutation(
    ready_artifact: dict[str, object],
) -> None:
    """REQ-RUSTPY-7201-NO-FALLBACK rejects each forged terminal evidence family."""

    attacks = [
        (lambda item: item.pop("status"), "missing_required_fields", True),
        (lambda item: item.update(field_principles={}), "field_principles_invalid", True),
        (lambda item: item.update(run_date="19000101"), "run_date_invalid", True),
        (lambda item: item.update(duration_s=999.0), "reproducibility_checksum_mismatch", False),
        (lambda item: item.update(MODEL_SPECS=[{}]), "model_declaration_invalid", True),
        (lambda item: item.update(verifier_is_oracle=False), "verifier_authority_invalid", True),
        (lambda item: item["distribution_rows"].pop(), "distribution_rows_invalid", True),
        (lambda item: item["phase_cost_rows"].clear(), "phase_cost_rows_invalid", True),
        (lambda item: item["rows"].clear(), "rows_invalid", True),
        (
            lambda item: item.update(buffer_reuse_receipt={"passed": False}),
            "buffer_reuse_receipt_invalid",
            True,
        ),
        (lambda item: item.update(status="running"), "terminal_status_invalid", True),
        (lambda item: item.update(verdict_class="positive"), "terminal_verdict_invalid", True),
        (lambda item: item.update(honest_verdict="null"), "honest_verdict_invalid", True),
        (
            lambda item: item.update(inference_substrate_class="aggregation"),
            "substrate_class_invalid",
            True,
        ),
        (lambda item: item.update(gate_check_summary={}), "gate_summary_invalid", True),
        (
            lambda item: item["upstream_performance_null"].update(promoted=True),
            "upstream_failed_value_promoted",
            True,
        ),
        (
            lambda item: item["source_artifact_hashes"].popitem(),
            "source_artifact_hashes_invalid",
            True,
        ),
    ]
    for mutate, expected, refresh_checksum in attacks:
        changed = copy.deepcopy(ready_artifact)
        mutate(changed)
        if refresh_checksum:
            changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        root = REPO if expected == "source_artifact_hashes_invalid" else None
        assert expected in exp.validate_artifact(changed, root=root)

    blocked = exp.build_artifact(
        root=REPO,
        preconditions=[
            {
                "check": "external",
                "upstream": "producer",
                "field": "ready",
                "expected_value": True,
                "observed_value": False,
                "passed": False,
            }
        ],
        source_hashes={},
    )
    blocked["rows"] = [{}]
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    assert "blocked_state_invalid" in exp.validate_artifact(blocked)


def test_req_rustpy_7201_publication_and_cli_paths(
    ready_artifact: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RUSTPY-7201 publishes valid bytes and reports CLI failures."""

    monkeypatch.setattr(exp, "build_artifact", lambda **kwargs: copy.deepcopy(ready_artifact))
    output = tmp_path / "published.json"
    published = exp.run_experiment(root=REPO, output=output, run_date=exp.RUN_DATE)
    assert output.is_file() and published["status"] == "complete"

    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced"])
    with pytest.raises(ValueError, match="invalid Exp7201"):
        exp.run_experiment(root=REPO, output=output, run_date=exp.RUN_DATE)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    assert exp.main(["--validate", str(invalid)]) == 2
    invalid.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", str(invalid)]) == 2

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(exp, "run_experiment", lambda **kwargs: calls.append(kwargs))
    assert exp.main(["--date", exp.RUN_DATE, "--output", str(output)]) == 0
    assert calls
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda **kwargs: (_ for _ in ()).throw(subprocess.SubprocessError("failure")),
    )
    assert exp.main(["--date", exp.RUN_DATE, "--output", str(output)]) == 2
