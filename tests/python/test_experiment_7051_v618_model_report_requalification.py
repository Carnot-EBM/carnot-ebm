"""Tests for REQ-ARC-7051 and each SCENARIO-ARC-7051 contract."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7051_v618_model_report_requalification as mod


ROOT = Path(__file__).resolve().parents[2]


def _snapshot_spec(tmp_path: Path, content: bytes = b"selected-qwen-model") -> tuple[dict, Path]:
    """Create a snapshot link and its content-addressed model blob."""

    digest = hashlib.sha256(content).hexdigest()
    model_root = tmp_path / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
    blob = model_root / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(content)
    revision = "a" * 40
    filename = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    snapshot = model_root / "snapshots" / revision / filename
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(Path("../../blobs") / digest)
    spec = {
        "name": mod.MANDATED_MODEL_NAME,
        "hf_id": mod.MANDATED_MODEL_HF_ID,
        "gpu": 0,
        "gpu_uuid": "GPU-70510000-0000-0000-0000-000000000001",
        "gpu_model": "NVIDIA GeForce RTX 3090",
        "model_path": str(snapshot.absolute()),
        "model_filename": filename,
        "revision": revision,
        "quantization": "Q4_K_M",
        "model_file_hash": mod.sha256_file(snapshot),
        "server_build": "version: 9606 (9b4dae81f)",
        "resolved_via": "cached_sota_pair",
    }
    return spec, blob


def _preconditions() -> list[dict[str, Any]]:
    """Create one passing preflight row for pure artifact tests."""

    return [mod.gate_row("all_preconditions", True, True)]


def _live_evidence(spec: dict, raw_props: dict | None = None) -> dict[str, Any]:
    """Create complete owned evidence without using a model or ARC game."""

    launch = [
        "/opt/llama.cpp/llama-server",
        "-m",
        spec["model_path"],
        "-ngl",
        "999",
        "-c",
        "4096",
        "--port",
        "17051",
    ]
    diagnostics = []
    for index, prompt in enumerate(mod.DIAGNOSTIC_PROMPTS, start=1):
        start_ns = 1_010_000_000 + index * 1_000_000
        end_ns = start_ns + 500_000
        diagnostics.append(
            {
                "request_index": index,
                "request_payload": {
                    "prompt": prompt,
                    "n_predict": 1,
                    "temperature": 0.0,
                    "ignore_eos": True,
                    "seed": mod.RANDOM_SEED + index - 1,
                },
                "response": {"content": "O", "timings": {"predicted_n": 1}},
                "response_sha256": mod.sha256_json({"content": "O", "timings": {"predicted_n": 1}}),
                "requested_tokens": 1,
                "generated_tokens": 1,
                "completed": True,
                "monotonic_start_ns": start_ns,
                "monotonic_end_ns": end_ns,
                "wall_clock_start": f"2026-09-06T00:00:0{index}Z",
                "wall_clock_end": f"2026-09-06T00:00:0{index}.001000Z",
                "terminal": True,
            }
        )
    phase_rows = [
        {
            "event": "owned_live_start",
            "monotonic_ns": 1_000_000_000,
            "wall_clock": "2026-09-06T00:00:00Z",
            "terminal": True,
        },
        {
            "event": "first_token",
            "monotonic_ns": diagnostics[0]["monotonic_end_ns"],
            "wall_clock": diagnostics[0]["wall_clock_end"],
            "terminal": True,
        },
        {
            "event": "last_token",
            "monotonic_ns": diagnostics[-1]["monotonic_end_ns"],
            "wall_clock": diagnostics[-1]["wall_clock_end"],
            "terminal": True,
        },
        {
            "event": "shutdown",
            "monotonic_ns": 76_500_000_000,
            "wall_clock": "2026-09-06T00:01:15.500000Z",
            "terminal": True,
        },
    ]
    return {
        "launch_model_argument": spec["model_path"],
        "process_command_rows": [
            {
                "pid": 27051,
                "argv": launch,
                "command_line": " ".join(launch),
                "model_argument": spec["model_path"],
                "owned": True,
                "terminal": True,
            }
        ],
        "raw_server_props": raw_props
        if raw_props is not None
        else {
            "model_path": spec["model_path"],
            "model": None,
            "model_alias": spec["model_filename"],
            "build_info": "b9606-9b4dae81f",
            "default_generation_settings": {"n_ctx": 4096},
        },
        "diagnostic_request_rows": diagnostics,
        "server_process_rows": [
            {
                "pid": 27051,
                "owned": True,
                "port": 17051,
                "endpoint": "http://127.0.0.1:17051",
                "server_binary": launch[0],
                "server_binary_hash": "sha256:" + "4" * 64,
                "server_build": spec["server_build"],
                "launch_argv": launch,
                "launch_model_argument": spec["model_path"],
                "raw_server_props_sha256": mod.sha256_json(
                    raw_props
                    if raw_props is not None
                    else {
                        "model_path": spec["model_path"],
                        "model": None,
                        "model_alias": spec["model_filename"],
                        "build_info": "b9606-9b4dae81f",
                        "default_generation_settings": {"n_ctx": 4096},
                    }
                ),
                "cuda_layer_offload_confirmed": True,
                "terminal": True,
            }
        ],
        "port_lease_rows": [{"port": 17051, "owned": True, "released": True, "terminal": True}],
        "gpu_lease_rows": [
            {
                "gpu_uuid": spec["gpu_uuid"],
                "lease_id": "exp7051-lease",
                "owned": True,
                "released": True,
                "phase": "terminal_complete",
                "terminal": True,
            }
        ],
        "gpu_sample_rows": [
            {
                "pid": 27051,
                "gpu_uuid": spec["gpu_uuid"],
                "gpu_model": spec["gpu_model"],
                "pid_memory_mb": 22_000,
                "sample_time_ns": 2_000_000_000,
                "terminal": True,
            }
        ],
        "phase_clock_rows": phase_rows,
        "owned_live_interval_s": 75.5,
        "cleanup_rows": [
            {
                "owned_pid": 27051,
                "ownership_match": True,
                "signals_sent_only_to_owned_pid": True,
                "signals_sent": ["terminate"],
                "process_exit_confirmed": True,
                "process_reaped": True,
                "lease_released": True,
                "port_released": True,
                "passed": True,
                "terminal": True,
            }
        ],
    }


def _positive(tmp_path: Path, duration_s: float = 90.0) -> dict[str, Any]:
    """Build one valid positive artifact for mutation tests."""

    spec, _blob = _snapshot_spec(tmp_path)
    return mod.build_positive_artifact(
        run_date="20260906",
        duration_s=duration_s,
        preconditions=_preconditions(),
        source_artifact_hashes={"source": "sha256:" + "5" * 64},
        model_specs=[spec],
        selected_model_spec=spec,
        live_evidence=_live_evidence(spec),
    )


def test_req_arc_7051_is_declared_before_implementation() -> None:
    """REQ-ARC-7051: the behavior has a spec anchor before code changes."""

    spec = (ROOT / "openspec/capabilities/arc-agi/spec.md").read_text(encoding="utf-8")

    assert "## REQ-ARC-7051:" in spec
    assert "SCENARIO-ARC-7051-TERMINAL-CHECKSUM" in spec
    assert "SCENARIO-ARC-7051-BLOCKED-PREFLIGHT" in spec


def test_prior_duration_and_checksum_failures_are_reproduced(tmp_path: Path) -> None:
    """SCENARIO-ARC-7051-PRIOR-FAILURES rejects both Exp7039 defects."""

    too_short = _positive(tmp_path, duration_s=46.968453163979575)

    assert too_short["model_report_evidence_ready_score"] == 0
    assert too_short["verdict_class"] == "blocked"
    assert too_short["gate_check_summary"]["failed_check"] == "live_inference_duration_floor_s"
    assert too_short["gate_check_summary"]["expected_value"] == ">=60.0"
    assert too_short["gate_check_summary"]["observed_value"] == pytest.approx(46.968453163979575)

    old = json.loads(
        (ROOT / "results/experiment_7039_v617_model_report_forensics.json").read_text(
            encoding="utf-8"
        )
    )
    from carnot import experiment_7039_v617_model_report_forensics as old_mod

    assert old_mod.artifact_checksum(old) != old["reproducibility_checksum"]


def test_terminal_payload_uses_canonical_checksum_and_clean_reader(tmp_path: Path) -> None:
    """SCENARIO-ARC-7051-TERMINAL-CHECKSUM binds the final payload."""

    artifact = _positive(tmp_path)

    assert artifact["checksum_contract"] == {
        "helper": "carnot.terminal_artifacts.payload_sha256",
        "canonical_json": "sort_keys_compact_ascii",
        "excluded_fields": ["reproducibility_checksum", "checksum_recomputation_rows"],
    }
    assert artifact["reproducibility_checksum"] == mod.canonical_artifact_checksum(artifact)
    assert mod.clean_reader_checksum(json.dumps(artifact)) == artifact["reproducibility_checksum"]
    assert artifact["checksum_recomputation_rows"] == [
        {
            "reader": "clean_json_reader",
            "helper": "carnot.terminal_artifacts.payload_sha256",
            "expected_checksum": artifact["reproducibility_checksum"],
            "observed_checksum": artifact["reproducibility_checksum"],
            "passed": True,
            "terminal": True,
        }
    ]
    assert artifact["model_report_evidence_ready_score"] == 1
    assert mod.validate_artifact(artifact) == []


def test_post_hash_mutation_is_rejected(tmp_path: Path) -> None:
    """SCENARIO-ARC-7051-PRIOR-FAILURES rejects post-hash payload changes."""

    artifact = _positive(tmp_path)
    changed = deepcopy(artifact)
    changed["raw_server_props"]["model_path"] = "/changed/after/hash.gguf"

    assert mod.canonical_artifact_checksum(changed) != changed["reproducibility_checksum"]
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(changed)


def test_raw_identity_fields_stay_separate_from_resolution(tmp_path: Path) -> None:
    """SCENARIO-ARC-7051-RAW-EVIDENCE preserves report and launch facts."""

    spec, blob = _snapshot_spec(tmp_path)
    raw = {
        "model_path": spec["model_path"],
        "model": str(blob.absolute()),
        "model_alias": spec["model_filename"],
        "nested": {"unchanged": [1, 2, 3]},
    }
    original = deepcopy(raw)
    artifact = mod.build_positive_artifact(
        run_date="20260906",
        duration_s=90.0,
        preconditions=_preconditions(),
        source_artifact_hashes={},
        model_specs=[spec],
        selected_model_spec=spec,
        live_evidence=_live_evidence(spec, raw),
    )

    assert raw == original
    assert artifact["raw_server_props"] == original
    assert [row["field"] for row in artifact["raw_identity_field_rows"]] == [
        "model_path",
        "model",
        "model_alias",
    ]
    assert artifact["raw_identity_field_rows"][0]["raw_value"] == spec["model_path"]
    assert artifact["resolved_identity_field_rows"][0]["resolved_path"] == str(blob.absolute())
    assert artifact["launch_model_argument"] == spec["model_path"]
    assert {row["source_field"] for row in artifact["file_hash_rows"]} == {
        "selected_model_spec.model_path",
        "model_path",
        "model",
    }
    assert all(row["hash_matches_selected"] is True for row in artifact["file_hash_rows"])


def test_owned_interval_and_phase_clocks_are_independently_validated(tmp_path: Path) -> None:
    """SCENARIO-ARC-7051-OWNED-LIVE-INTERVAL checks token and shutdown order."""

    artifact = _positive(tmp_path)

    assert artifact["minimum_owned_live_interval_s"] == 75.0
    assert artifact["owned_live_interval_s"] == pytest.approx(75.5)
    assert [row["event"] for row in artifact["phase_clock_rows"]] == [
        "owned_live_start",
        "first_token",
        "last_token",
        "shutdown",
    ]
    assert artifact["generation_request_count"] == len(mod.DIAGNOSTIC_PROMPTS) == 2

    short = deepcopy(artifact)
    short["owned_live_interval_s"] = 74.999
    short["reproducibility_checksum"] = mod.canonical_artifact_checksum(short)
    assert "positive_evidence_invalid:owned_live_interval_floor_s" in mod.validate_artifact(short)

    reordered = deepcopy(artifact)
    reordered["phase_clock_rows"][2]["monotonic_ns"] = 1
    reordered["reproducibility_checksum"] = mod.canonical_artifact_checksum(reordered)
    assert "phase_clock_rows_invalid" in mod.validate_artifact(reordered)


class _FakeProcess:
    """Record cleanup calls so tests can prove signal ownership."""

    def __init__(self, pid: int, *, needs_kill: bool = False) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.needs_kill = needs_kill
        self.calls: list[str] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")
        self.returncode = -9

    def wait(self, timeout: float) -> int:
        self.calls.append(f"wait:{timeout}")
        if self.needs_kill and "kill" not in self.calls:
            raise mod.subprocess.TimeoutExpired("fake", timeout)
        self.returncode = 0 if "kill" not in self.calls else -9
        return self.returncode


def test_cleanup_signals_only_the_owned_process() -> None:
    """SCENARIO-ARC-7051-CLEANUP proves process ownership before signaling."""

    owned = _FakeProcess(7051, needs_kill=True)
    row = mod.cleanup_owned_process(owned, owned_pid=7051, terminate_timeout_s=0.1)

    assert owned.calls == ["terminate", "wait:0.1", "kill", "wait:0.1"]
    assert row["ownership_match"] is True
    assert row["signals_sent"] == ["terminate", "kill"]
    assert row["process_reaped"] is True

    not_owned = _FakeProcess(9999)
    rejected = mod.cleanup_owned_process(not_owned, owned_pid=7051, terminate_timeout_s=0.1)

    assert not_owned.calls == []
    assert rejected["ownership_match"] is False
    assert rejected["signals_sent"] == []
    assert rejected["process_reaped"] is False


def test_blocked_preflight_names_unattributed_server_without_stopping_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-7051-BLOCKED-PREFLIGHT preserves the foreign process."""

    monkeypatch.setattr(mod, "_load_json", lambda _path: {"readable": True})
    server_row = {
        "pid": 9911,
        "comm": "llama-server",
        "command_line": "/foreign/llama-server --port 9911",
        "attributed_to_task": False,
        "terminal": True,
    }
    monkeypatch.setattr(mod, "_unattributed_llama_server_rows", lambda: [server_row])
    monkeypatch.setattr(mod, "_source_hashes", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "_gpu_rows",
        lambda: [
            {
                "index": 0,
                "gpu_uuid": "GPU-70510000-0000-0000-0000-000000000001",
                "gpu_model": "NVIDIA GeForce RTX 3090",
                "memory_total_mb": 24576,
                "memory_free_mb": 24120,
                "utilization_pct": 0,
                "idle": True,
                "supported": True,
            }
        ],
    )
    preflight = mod.collect_preconditions(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "out.json",
        checkpoint_path=tmp_path / "checkpoints" / "state.json",
    )

    assert preflight["summary"]["failed_check"] == "no_unattributed_llama_server"
    assert preflight["summary"]["expected_value"] == []
    assert preflight["summary"]["observed_value"] == [server_row]
    assert server_row["pid"] == 9911

    artifact = mod.build_blocked_artifact(
        run_date="20260906",
        duration_s=0.1,
        preconditions=preflight["checks"],
        source_artifact_hashes={},
        model_specs=[],
        selected_model_spec=None,
        live_evidence={"server_process_rows": [server_row]},
    )
    assert artifact["model_report_evidence_ready_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "no_unattributed_llama_server"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert mod.validate_artifact(artifact) == []


def test_missing_gpu_precondition_blocks_with_exact_observation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-7051-BLOCKED-PREFLIGHT keeps resource absence exact."""

    monkeypatch.setattr(mod, "_load_json", lambda _path: {"readable": True})
    monkeypatch.setattr(mod, "_unattributed_llama_server_rows", lambda: [])
    monkeypatch.setattr(mod, "_gpu_rows", lambda: [])
    monkeypatch.setattr(mod, "_source_hashes", lambda _root: {})

    preflight = mod.collect_preconditions(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "out.json",
        checkpoint_path=tmp_path / "checkpoints" / "state.json",
    )

    assert preflight["summary"]["failed_check"] == "idle_supported_rtx3090"
    assert preflight["summary"]["expected_value"] is True
    assert preflight["summary"]["observed_value"] is False
    assert preflight["model_spec"] is None


def test_model_spec_is_selected_only_through_cached_sota_pair(tmp_path: Path) -> None:
    """REQ-ARC-7051 records the mandated hub, snapshot, hash, GPU, and build."""

    spec, _blob = _snapshot_spec(tmp_path)
    calls: list[dict[str, Any]] = []

    def cached_pair(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(dict(kwargs))
        return [{"hf_id": spec["hf_id"], "model_path": spec["model_path"]}, {}]

    selected = mod.resolve_model_spec(
        cached_pair,
        gpu_index=1,
        gpu_uuid=spec["gpu_uuid"],
        gpu_model=spec["gpu_model"],
        server_build=spec["server_build"],
    )

    assert calls == [{"gpu_indices": (1, 1)}]
    assert selected is not None
    assert selected["hf_id"] == mod.MANDATED_MODEL_HF_ID
    assert selected["revision"] == "a" * 40
    assert selected["model_filename"] == spec["model_filename"]
    assert selected["quantization"] == "Q4_K_M"
    assert selected["model_file_hash"] == spec["model_file_hash"]
    assert selected["gpu_uuid"] == spec["gpu_uuid"]
    assert selected["server_build"] == spec["server_build"]
    assert selected["resolved_via"] == "cached_sota_pair"


def test_required_fields_principles_and_terminal_mutations_fail(tmp_path: Path) -> None:
    """REQ-ARC-7051 validates every required terminal field independently."""

    artifact = _positive(tmp_path)

    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == [artifact["selected_model_spec"]]
    assert artifact["models_used"] == [mod.MANDATED_MODEL_HF_ID]
    assert artifact["cuda_layer_offload_confirmed"] is True
    assert artifact["arc_action_count"] == 0
    assert artifact["game_level_solve_claim"] is False
    assert artifact["verifier_is_oracle"] is False

    mutations = [
        (lambda value: value.pop("rows"), "required_fields_missing"),
        (lambda value: value.__setitem__("field_principles", {}), "field_principles_invalid"),
        (
            lambda value: value.__setitem__("inference_substrate", "offline"),
            "inference_substrate_invalid",
        ),
        (lambda value: value.__setitem__("arc_action_count", 1), "arc_non_action_contract_invalid"),
        (lambda value: value.__setitem__("verifier_is_oracle", True), "verifier_is_oracle_invalid"),
        (
            lambda value: value.__setitem__("minimum_owned_live_interval_s", 60),
            "minimum_owned_live_interval_invalid",
        ),
        (lambda value: value.__setitem__("checksum_contract", {}), "checksum_contract_invalid"),
        (
            lambda value: value.__setitem__("checksum_recomputation_rows", []),
            "checksum_recomputation_rows_invalid",
        ),
        (
            lambda value: value.__setitem__("model_report_evidence_ready_score", True),
            "ready_score_invalid",
        ),
        (lambda value: value.__setitem__("verdict_class", "unknown"), "verdict_prefix_invalid"),
        (lambda value: value.__setitem__("honest_verdict", "wrong"), "verdict_prefix_invalid"),
        (
            lambda value: value.__setitem__("raw_identity_field_rows", []),
            "raw_identity_rows_invalid",
        ),
        (
            lambda value: value.__setitem__("resolved_identity_field_rows", []),
            "resolved_identity_rows_invalid",
        ),
        (lambda value: value.__setitem__("file_hash_rows", []), "file_hash_rows_invalid"),
        (lambda value: value.__setitem__("consistency_rows", []), "consistency_rows_invalid"),
        (
            lambda value: value.__setitem__("diagnostic_request_rows", []),
            "positive_evidence_invalid:diagnostic_requests_complete",
        ),
        (
            lambda value: value.__setitem__("cleanup_rows", []),
            "positive_evidence_invalid:safe_cleanup",
        ),
    ]
    for mutate, expected in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        changed["reproducibility_checksum"] = mod.canonical_artifact_checksum(changed)
        assert any(error.startswith(expected) for error in mod.validate_artifact(changed)), expected

    assert mod.validate_artifact([]) == ["artifact_object_required"]


def test_writer_recomputes_after_json_round_trip(tmp_path: Path) -> None:
    """SCENARIO-ARC-7051-TERMINAL-CHECKSUM validates the published bytes."""

    artifact = _positive(tmp_path)
    target = tmp_path / "nested" / "artifact.json"

    mod.write_artifact(target, artifact)

    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert (
        mod.clean_reader_checksum(target.read_text(encoding="utf-8"))
        == loaded["reproducibility_checksum"]
    )

    bad = deepcopy(artifact)
    bad["duration_s"] = 91.0
    with pytest.raises(ValueError, match="reproducibility_checksum_mismatch"):
        mod.write_artifact(tmp_path / "bad.json", bad)


def test_model_resolution_and_reader_fail_closed(tmp_path: Path) -> None:
    """REQ-ARC-7051 rejects every unsupported model-selection observation."""

    kwargs = {
        "gpu_index": 0,
        "gpu_uuid": "GPU-70510000-0000-0000-0000-000000000001",
        "gpu_model": "NVIDIA GeForce RTX 3090",
        "server_build": "version: 9606 (9b4dae81f)",
    }
    assert mod.resolve_model_spec(lambda **_kwargs: None, **kwargs) is None
    assert mod.resolve_model_spec(lambda **_kwargs: [{"hf_id": "other/model"}], **kwargs) is None
    assert (
        mod.resolve_model_spec(
            lambda **_kwargs: [
                {"hf_id": mod.MANDATED_MODEL_HF_ID, "model_path": str(tmp_path / "missing.gguf")}
            ],
            **kwargs,
        )
        is None
    )
    ordinary_gguf = tmp_path / "ordinary.gguf"
    ordinary_gguf.write_bytes(b"not-a-snapshot")
    assert (
        mod.resolve_model_spec(
            lambda **_kwargs: [
                {"hf_id": mod.MANDATED_MODEL_HF_ID, "model_path": str(ordinary_gguf)}
            ],
            **kwargs,
        )
        is None
    )
    assert mod._snapshot_revision(ordinary_gguf, mod.MANDATED_MODEL_HF_ID) is None

    with pytest.raises(ValueError, match="JSON object"):
        mod.clean_reader_checksum("[]")


def test_missing_selected_file_and_invalid_phase_rows_are_unknown(tmp_path: Path) -> None:
    """SCENARIO-ARC-7051-RAW-EVIDENCE keeps missing path facts unknown."""

    spec, _blob = _snapshot_spec(tmp_path)
    spec["model_path"] = str(tmp_path / "removed.gguf")
    body = mod._base_artifact(
        run_date="20260906",
        duration_s=0.0,
        preconditions=_preconditions(),
        source_artifact_hashes={},
        model_specs=[spec],
        selected_model_spec=spec,
        live_evidence={},
    )
    selected_row = body["file_hash_rows"][0]
    assert selected_row["sha256"] is None
    assert selected_row["resolved_path"] is None
    assert selected_row["status"] == "unknown"

    assert mod._phase_clocks_valid({"phase_clock_rows": []}) is False
    malformed = _positive(tmp_path / "malformed")
    malformed["phase_clock_rows"][0]["monotonic_ns"] = "not-an-integer"
    assert mod._phase_clocks_valid(malformed) is False


def test_validator_rejects_each_defensive_terminal_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-7051 covers defensive validation and blocked terminal branches."""

    artifact = _positive(tmp_path)

    def make_selected_manual(value: dict[str, Any]) -> None:
        value["selected_model_spec"]["resolved_via"] = "manual"
        value["MODEL_SPECS"][0]["resolved_via"] = "manual"
        value["model_specs"][0]["resolved_via"] = "manual"

    mutations = [
        (
            lambda value: value.__setitem__("report_channel_classification", "invalid"),
            "report_channel_classification_invalid",
        ),
        (lambda value: value.__setitem__("gate_check_summary", {}), "gate_check_summary_invalid"),
        (lambda value: value.__setitem__("raw_server_props", {}), "raw_server_props_invalid"),
        (
            lambda value: value["raw_identity_field_rows"][0].__setitem__(
                "raw_value", "/different.gguf"
            ),
            "raw_identity_rows_do_not_match_props",
        ),
        (
            lambda value: value["resolved_identity_field_rows"][0].__setitem__(
                "raw_value", "/different.gguf"
            ),
            "resolved_rows_do_not_retain_raw_values",
        ),
        (
            lambda value: value["raw_identity_field_rows"][0].__setitem__(
                "evidence_status", "invalid"
            ),
            "identity_evidence_status_invalid",
        ),
        (lambda value: value.__setitem__("MODEL_SPECS", []), "model_specs_invalid"),
        (make_selected_manual, "selected_model_spec_invalid"),
        (
            lambda value: value.__setitem__("model_report_evidence_ready_score", 0),
            "positive_terminal_semantics_invalid",
        ),
    ]
    for mutate, expected in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        changed["reproducibility_checksum"] = mod.canonical_artifact_checksum(changed)
        assert expected in mod.validate_artifact(changed)

    blocked = mod.build_blocked_artifact(
        run_date="20260906",
        duration_s=0.0,
        preconditions=[mod.gate_row("resource", True, False)],
        source_artifact_hashes={},
        model_specs=[],
        selected_model_spec=None,
    )
    wrong_score = deepcopy(blocked)
    wrong_score["model_report_evidence_ready_score"] = 1
    wrong_score["reproducibility_checksum"] = mod.canonical_artifact_checksum(wrong_score)
    assert "blocked_terminal_semantics_invalid" in mod.validate_artifact(wrong_score)

    missing_failure = deepcopy(blocked)
    missing_failure["gate_check_summary"]["failed_check"] = None
    missing_failure["reproducibility_checksum"] = mod.canonical_artifact_checksum(missing_failure)
    assert "blocked_gate_failure_missing" in mod.validate_artifact(missing_failure)

    all_passed = mod.build_blocked_artifact(
        run_date="20260906",
        duration_s=0.0,
        preconditions=_preconditions(),
        source_artifact_hashes={},
        model_specs=[],
        selected_model_spec=None,
    )
    assert all_passed["gate_check_summary"]["failed_check"] == "blocked_live_capture"

    monkeypatch.setattr(mod, "clean_reader_checksum", lambda _raw: "different")
    assert "clean_reader_checksum_mismatch" in mod.validate_artifact(artifact)
    monkeypatch.setattr(
        mod,
        "clean_reader_checksum",
        lambda _raw: (_ for _ in ()).throw(ValueError("reader rejected bytes")),
    )
    assert "clean_reader_checksum_mismatch" in mod.validate_artifact(artifact)


def test_writer_rejects_changed_published_bytes_and_date_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-7051-TERMINAL-CHECKSUM covers the last publication gate."""

    artifact = _positive(tmp_path)
    calls = 0

    def validate_twice(_artifact: Any) -> list[str]:
        nonlocal calls
        calls += 1
        return [] if calls == 1 else ["published_bytes_changed"]

    monkeypatch.setattr(mod, "validate_artifact", validate_twice)
    with pytest.raises(ValueError, match="published_bytes_changed"):
        mod.write_artifact(tmp_path / "published.json", artifact)
    assert not (tmp_path / "published.json").exists()

    with pytest.raises(mod.argparse.ArgumentTypeError, match="YYYYMMDD"):
        mod._date_argument("2026-09-06")
    assert mod._date_argument("20260906") == "20260906"
