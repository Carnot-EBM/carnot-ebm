"""Tests for REQ-ARC-7032 and its repaired live-shadow scenarios."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from pathlib import Path

import pytest

from carnot.agentic.arc_eval_provenance import (
    ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2,
    ArcEvalProvenanceInput,
    build_arc_eval_provenance,
    build_arc_model_identity_receipt,
)
from carnot.experiment_7017_task_linked_compute_receipts import run_consumer_fixture
from carnot import experiment_7032_repaired_belief_shadow_live_trace as mod


ROOT = Path(__file__).resolve().parents[2]


def _snapshot_model(tmp_path: Path) -> tuple[dict, Path]:
    """Create the real Hugging Face snapshot-to-blob shape from REQ-ARC-7032."""

    content = b"exp7032-extensionless-model"
    digest = __import__("hashlib").sha256(content).hexdigest()
    model_root = tmp_path / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
    blob = model_root / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(content)
    revision = "a" * 40
    filename = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    requested = model_root / "snapshots" / revision / filename
    requested.parent.mkdir(parents=True)
    requested.symlink_to(Path("../../blobs") / digest)
    spec = {
        "name": mod.MANDATED_MODEL_NAME,
        "hf_id": mod.MANDATED_MODEL_HF_ID,
        "gpu": 1,
        "model_path": str(requested.absolute()),
        "model_filename": filename,
        "revision": revision,
        "model_file_hash": mod.sha256_file(requested),
        "resolved_via": "cached_sota_pair",
    }
    return spec, blob


def _provenance(spec: dict, blob: Path) -> dict:
    """Build one current-schema provenance row through the shared identity bridge."""

    identity = build_arc_model_identity_receipt(
        selected_model_spec=spec,
        observed_server_model_path=str(blob.absolute()),
    )
    return build_arc_eval_provenance(
        ArcEvalProvenanceInput(
            schema_version=ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2,
            inference_substrate="local_gguf_cuda",
            gpu_uuid="GPU-70320000-0000-0000-0000-000000000001",
            gpu_model="NVIDIA GeForce RTX 3090",
            cuda_device=1,
            model_repository=mod.MANDATED_MODEL_HF_ID,
            model_filename=spec["model_filename"],
            model_hash=spec["model_file_hash"],
            n_ctx=4096,
            server_binary="/opt/llama.cpp/llama-server",
            server_binary_hash="sha256:" + "2" * 64,
            server_command_hash="sha256:" + "3" * 64,
            endpoint="http://127.0.0.1:17032",
            port=17032,
            lease_id="lease-exp7032",
            lease_hash="sha256:" + "4" * 64,
            lease_issued_at="2026-09-05T00:00:00+00:00",
            lease_expires_at="2026-09-05T01:00:00+00:00",
            lease_checked_at="2026-09-05T00:30:00+00:00",
            request_count=1,
            completion_count=1,
            error_count=0,
            policy_hash="sha256:" + "5" * 64,
            factory_hash="sha256:" + "6" * 64,
            git_commit="7" * 40,
            solve_provenance="live_agent_self_discovery",
            **identity,
        )
    )


def _core_artifact(tmp_path: Path) -> tuple[dict, dict]:
    """Build the smallest valid Exp7025-core result used by the Exp7032 reducer."""

    spec, blob = _snapshot_model(tmp_path)
    provenance = _provenance(spec, blob)
    receipt, _ = run_consumer_fixture(tmp_path / "receipt", fixture_id="one-model")
    action = {"action": 1, "data": None}
    decision = {
        "query_fired": True,
        "query_count": 2,
        "evidence_hashes": ["sha256:" + "a" * 64],
        "base_actions": [action, {"action": 2, "data": None}],
        "ranked_actions": [{"action": 2, "data": None}, action],
        "control_selected_action": action,
        "counterfactual_selected_action": {"action": 2, "data": None},
        "counterfactual_ranking_changed": True,
        "abstained": False,
        "abstention_reason": None,
    }
    control = {
        "cell": "control",
        "game_id": "ls20-00000000",
        "episode_guid": "episode-7032",
        "prompt_hash": mod.sha256_json("Reply with exactly OK."),
        "candidate_actions": [action, {"action": 2, "data": None}],
        "action": action,
        "request_count": 1,
        "completion_count": 1,
        "error_count": 0,
        "arc_eval_provenance": provenance,
        "solve_provenance": "live_agent_self_discovery",
        "terminal": True,
    }
    shadow = {**deepcopy(control), "cell": "belief_shadow", "belief_decision": decision}
    identity = build_arc_model_identity_receipt(
        selected_model_spec=spec,
        observed_server_model_path=str(blob.absolute()),
    )
    launch = [
        "/opt/llama.cpp/llama-server",
        "-m",
        spec["model_path"],
        "-ngl",
        "999",
        "-c",
        "4096",
        "--port",
        "17032",
    ]
    core = {
        "preconditions_checked": [mod.gate_row("all_core_preconditions", True, True)],
        "duration_s": 65.0,
        "MODEL_SPECS": [spec],
        "models_used": [mod.MANDATED_MODEL_HF_ID],
        "model_file_hashes": {spec["model_path"]: spec["model_file_hash"]},
        "cited_upstream_artifacts": [],
        "source_artifact_hashes": {"core": "sha256:" + "8" * 64},
        "rows": [{"check": "matched_cells", "passed": True, "terminal": True}],
        "control_rows": [control],
        "shadow_rows": [shadow],
        "shadow_action_parity_rows": [
            {
                "control_action": action,
                "shadow_action": action,
                "candidate_set_match": True,
                "passed": True,
                "terminal": True,
            }
        ],
        "belief_query_rows": [
            {
                "query_fired": True,
                "query_count": 2,
                "evidence_hashes": decision["evidence_hashes"],
                "abstained": False,
                "terminal": True,
            }
        ],
        "model_execution_rows": [
            {
                "hf_id": mod.MANDATED_MODEL_HF_ID,
                "model_path": spec["model_path"],
                "model_filename": spec["model_filename"],
                "model_file_hash": spec["model_file_hash"],
                **identity,
                "n_ctx": 4096,
                "observed_server_n_ctx": 4096,
                "n_gpu_layers": 999,
                "launch_argv": launch,
                "cuda_offload": True,
                "request_count": 2,
                "completion_count": 2,
                "error_count": 0,
                "pid": 27032,
                "terminal": True,
            }
        ],
        "gpu_identity_rows": [
            {
                "index": 1,
                "gpu_uuid": "GPU-70320000-0000-0000-0000-000000000001",
                "gpu_model": "NVIDIA GeForce RTX 3090",
                "supported": True,
                "terminal": True,
            }
        ],
        "gpu_sample_rows": list(receipt["gpu_sample_rows"]),
        "server_rows": [
            {
                "pid": 27032,
                "owned": True,
                "port": 17032,
                "endpoint": "http://127.0.0.1:17032",
                "server_binary": launch[0],
                "server_command_hash": mod.sha256_json(launch),
                "model_file_hash": spec["model_file_hash"],
                "observed_server_model_path": str(blob.absolute()),
                "resolved_model_path": str(blob.absolute()),
                "n_ctx": 4096,
                "request_count": 2,
                "completion_count": 2,
                "error_count": 0,
                "terminal": True,
            }
        ],
        "lease_rows": list(receipt["lease_link_rows"]),
        "phase_receipt_rows": list(receipt["rows"]),
        "teardown_rows": [
            {
                "owned_pid": 27032,
                "process_exit_confirmed": True,
                "process_reaped": True,
                "lease_released": True,
                "port": 17032,
                "port_released": True,
                "scorecard_closed": True,
                "passed": True,
                "terminal": True,
            }
        ],
        "task_compute_receipt": receipt,
        "solve_provenance": "live_agent_self_discovery",
        "solve_registry_precheck_rows": [
            {
                "registry_hash_before": "sha256:" + "9" * 64,
                "registry_hash_after": "sha256:" + "9" * 64,
                "targeted_level": None,
                "already_reproduced_target": False,
                "terminal": True,
            }
        ],
        "belief_shadow_trace_ready_score": 1,
        "gate_check_summary": mod.gate_check_summary(
            [mod.gate_row("all_core_preconditions", True, True)]
        ),
        "verdict_class": "positive",
        "honest_verdict": "complete_positive_belief_shadow_live_trace_transport_ready",
    }
    telemetry = {
        "model_call_rows": [
            {
                "cell": cell,
                "prompt_sha256": control["prompt_hash"],
                "requested_tokens": 32,
                "generated_tokens": 2,
                "completed": True,
                "terminal": True,
            }
            for cell in ("control", "belief_shadow")
        ],
        "observation_rows": [
            {
                "cell": "control",
                "position": "pre_action",
                "observation_sha256": "sha256:" + "b" * 64,
                "levels_completed": 0,
                "terminal": True,
            },
            {
                "cell": "belief_shadow",
                "position": "pre_action",
                "observation_sha256": "sha256:" + "b" * 64,
                "levels_completed": 0,
                "terminal": True,
            },
            {
                "cell": "control",
                "position": "post_action",
                "observation_sha256": "sha256:" + "c" * 64,
                "levels_completed": 0,
                "terminal": True,
            },
        ],
    }
    return core, telemetry


def _valid_artifact(tmp_path: Path) -> dict:
    core, telemetry = _core_artifact(tmp_path)
    artifact = mod.adapt_core_artifact(
        core,
        telemetry=telemetry,
        run_date="20260905",
        checkpoint_path=tmp_path / "checkpoint.json",
        upstream_gate_rows=[mod.gate_row("exp7030_ready", 1, 1)],
        cited_upstream_artifacts=[],
        source_artifact_hashes={"module": "sha256:" + "d" * 64},
    )
    artifact["duration_s"] = 65.0
    artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
    return artifact


def test_req_arc_7032_spec_exists_before_implementation() -> None:
    """REQ-ARC-7032 and each tested scenario exist before implementation."""

    text = (ROOT / "openspec/capabilities/arc-belief-shadow-live-trace/spec.md").read_text(
        encoding="utf-8"
    )
    assert "REQ-ARC-7032" in text
    for scenario in (
        "SCENARIO-ARC-7032-PINNED-UPSTREAMS",
        "SCENARIO-ARC-7032-EXTENSIONLESS-BLOB-IDENTITY",
        "SCENARIO-ARC-7032-SHADOW-PARITY-AND-QUERY",
        "SCENARIO-ARC-7032-COMPUTE-AND-CLEANUP-RECEIPTS",
        "SCENARIO-ARC-7032-FAILS-CLOSED",
    ):
        assert scenario in text


def test_scenario_arc_7032_extensionless_blob_identity(tmp_path: Path) -> None:
    """SCENARIO-ARC-7032-EXTENSIONLESS-BLOB-IDENTITY uses the shared bridge."""

    spec, blob = _snapshot_model(tmp_path)
    calls: list[dict] = []

    def pair(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return [spec, {"hf_id": "unsloth/gemma-4-31B-it-GGUF"}]

    resolved = mod.resolve_model_spec(pair, gpu_index=1)
    assert resolved is not None
    identity = build_arc_model_identity_receipt(
        selected_model_spec=resolved,
        observed_server_model_path=str(blob.absolute()),
    )
    assert calls == [{"gpu_indices": (1, 1)}]
    assert identity["requested_model_path"].endswith(".gguf")
    assert Path(identity["observed_server_model_path"]).suffix == ""
    assert identity["resolved_model_path"] == identity["observed_server_model_path"]
    assert mod.extract_quantization(identity["requested_model_filename"]) == "Q4_K_M"


def test_scenario_arc_7032_pinned_source_hashes_detect_drift(tmp_path: Path) -> None:
    """SCENARIO-ARC-7032-PINNED-UPSTREAMS rejects one changed cited source."""

    source = tmp_path / "source.py"
    source.write_text("stable\n", encoding="utf-8")
    artifact = {"source_artifact_hashes": {"source.py": mod.sha256_file(source)}}
    assert all(row["passed"] for row in mod.pinned_source_hash_checks(tmp_path, artifact, 7030))
    source.write_text("changed\n", encoding="utf-8")
    rows = mod.pinned_source_hash_checks(tmp_path, artifact, 7030)
    assert rows == [
        mod.gate_row(
            "exp7030_source_hash:source.py",
            artifact["source_artifact_hashes"]["source.py"],
            mod.sha256_file(source),
        )
    ]
    assert rows[0]["passed"] is False


def test_scenario_arc_7032_shadow_parity_query_and_receipts(tmp_path: Path) -> None:
    """SCENARIO-ARC-7032-SHADOW-PARITY-AND-QUERY validates the complete reducer."""

    artifact = _valid_artifact(tmp_path)
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["no_autotokenizer_used"] is True
    assert artifact["live_model_invoked"] is True
    assert artifact["cuda_layer_offload_confirmed"] is True
    assert artifact["requested_model_path"].endswith(".gguf")
    assert Path(artifact["observed_server_model_path"]).suffix == ""
    assert (
        artifact["per_decision_rows"][0]["selected_action"]
        == artifact["per_decision_rows"][1]["selected_action"]
    )
    assert artifact["per_decision_rows"][1]["belief_evidence_ids"]
    assert artifact["action_parity_rows"][0]["passed"] is True
    assert artifact["belief_query_rows"][0]["query_count"] == 2
    assert artifact["cleanup_rows"][0]["passed"] is True
    assert mod.validate_artifact(artifact) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda a: a.__setitem__("requested_model_filename", "blob"), "model_identity"),
        (
            lambda a: a["action_parity_rows"][0].__setitem__(
                "shadow_action", {"action": 2, "data": None}
            ),
            "action_parity",
        ),
        (lambda a: a["belief_query_rows"][0].__setitem__("query_count", 0), "belief_query"),
        (lambda a: a["cleanup_rows"][0].__setitem__("passed", False), "cleanup"),
        (lambda a: a.__setitem__("cuda_layer_offload_confirmed", False), "cuda_offload"),
        (lambda a: a.pop("tokenizer_receipt"), "required_fields"),
    ],
)
def test_scenario_arc_7032_fails_closed(tmp_path: Path, mutation, expected: str) -> None:
    """SCENARIO-ARC-7032-FAILS-CLOSED names each damaged evidence family."""

    artifact = _valid_artifact(tmp_path)
    mutation(artifact)
    artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
    assert any(expected in error for error in mod.validate_artifact(artifact))


def test_req_arc_7032_blocked_identity_reuses_prior_verdict(tmp_path: Path) -> None:
    """REQ-ARC-7032 reuses Exp7025's exact verdict after the same identity failure."""

    checks = [
        mod.gate_row(
            "live_trace_execution",
            "successful owned live trace",
            "ValueError: invalid ARC model identity: requested filename",
        )
    ]
    artifact = mod.build_blocked_artifact(
        run_date="20260905",
        duration_s=1.0,
        preconditions=checks,
        source_artifact_hashes={},
        cited_upstream_artifacts=[],
    )
    assert artifact["honest_verdict"] == mod.PRIOR_IDENTITY_FAILURE_VERDICT
    assert artifact["belief_shadow_trace_ready_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "live_trace_execution"
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert mod.validate_artifact(artifact) == []


def test_req_arc_7032_observation_receipt_is_content_addressed() -> None:
    """REQ-ARC-7032 records an observation without exposing mutable frame objects."""

    class Frame:
        def model_dump(self, *, mode):  # type: ignore[no-untyped-def]
            assert mode == "json"
            return {
                "game_id": "ls20",
                "frame": [[[1, 2], [3, 4]]],
                "state": "NOT_FINISHED",
                "levels_completed": 0,
                "guid": "guid-7032",
                "available_actions": [1, 2],
            }

    row = mod.observation_receipt(Frame(), cell="control", position="pre_action")
    assert row["observation_sha256"].startswith("sha256:")
    assert row["frame_sha256"].startswith("sha256:")
    assert row["levels_completed"] == 0
    assert "frame" not in row
    assert row["terminal"] is True


def test_req_arc_7032_defensive_receipt_inputs_are_explicit(tmp_path: Path) -> None:
    """REQ-ARC-7032 retains deterministic evidence for alternate input shapes."""

    assert mod.extract_quantization("model.gguf") is None
    assert mod.pinned_source_hash_checks(tmp_path, {}, 7031)[0]["passed"] is False

    mapping = mod.observation_receipt(
        {"frame": [[1]], "state": "NOT_FINISHED"},
        cell="control",
        position="pre_action",
    )
    assert mapping["frame_sha256"] == mod.sha256_json([[1]])

    class State(Enum):
        READY = "READY"

    enum_row = mod.observation_receipt(
        {"state": State.READY}, cell="control", position="pre_action"
    )
    assert enum_row["state"] == "READY"
    opaque = mod.observation_receipt(object(), cell="control", position="pre_action")
    assert opaque["game_id"] is None
    assert opaque["available_actions"] == []
    assert mod._rebuild_task_receipt({}, run_date="20260905", server={}, spec={}) == {}
    assert (
        mod._decision_rows(
            {"control_rows": None, "shadow_rows": []},
            {"model_call_rows": [], "observation_rows": []},
        )
        == []
    )


def test_scenario_arc_7032_blocked_core_stays_blocked(tmp_path: Path) -> None:
    """SCENARIO-ARC-7032-FAILS-CLOSED never promotes a blocked core result."""

    failure = mod.gate_row("official_arc_access", True, False)
    artifact = mod.adapt_core_artifact(
        {
            "belief_shadow_trace_ready_score": 0,
            "duration_s": 2.0,
            "preconditions_checked": [failure],
            "MODEL_SPECS": [],
            "models_used": [],
            "model_file_hashes": {},
            "teardown_rows": [],
            "solve_registry_precheck_rows": [],
        },
        telemetry={},
        run_date="20260905",
        checkpoint_path=tmp_path / "checkpoint.json",
        upstream_gate_rows=[],
        cited_upstream_artifacts=[],
        source_artifact_hashes={},
    )
    assert artifact["honest_verdict"] == (
        "blocked_repaired_belief_shadow_live_trace:official_arc_access"
    )
    assert artifact["live_model_invoked"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_arc_7032_validator_names_all_damaged_families(tmp_path: Path) -> None:
    """SCENARIO-ARC-7032-FAILS-CLOSED covers every positive evidence family."""

    empty_errors = set(mod._positive_errors({}))
    assert {
        "duration_live_llm_floor_invalid",
        "upstream_gate_rows_invalid",
        "preconditions_invalid",
        "model_specs_invalid",
        "model_identity_invalid",
        "quantization_invalid",
        "tokenizer_receipt_invalid",
        "server_process_rows_invalid",
        "cuda_offload_invalid",
        "context_receipt_rows_invalid",
        "per_decision_rows_invalid",
        "action_parity_rows_invalid",
        "belief_query_rows_invalid",
        "observation_rows_invalid",
        "progress_rows_invalid",
        "cleanup_rows_invalid",
        "port_lease_rows_invalid",
        "gpu_lease_rows_invalid",
        "gpu_sample_rows_invalid",
        "task_compute_receipt_invalid",
        "solve_registry_precheck_rows_invalid",
        "solve_provenance_invalid",
        "live_model_invoked_invalid",
        "game_level_solve_claim_invalid",
    } <= empty_errors

    artifact = _valid_artifact(tmp_path)
    artifact["resolved_model_path"] = artifact["requested_model_path"]
    artifact["model_file_hashes"] = {}
    artifact["per_decision_rows"][0]["selected_action"] = {"action": 9, "data": None}
    artifact["per_decision_rows"][0]["pre_action_observation_sha256"] = "sha256:changed"
    artifact["per_decision_rows"][0]["model_calls"] = 0
    artifact["per_decision_rows"][0]["model_tokens_generated"] = -1
    artifact["per_decision_rows"][0]["solve_provenance"] = "development_proxy"
    artifact["phase_receipt_rows"] = []
    errors = set(mod._positive_errors(artifact))
    assert {
        "model_identity_resolved_path_invalid",
        "model_file_hashes_invalid",
        "action_parity_decisions_invalid",
        "pre_action_state_mismatch",
        "model_call_counts_invalid",
        "model_token_counts_invalid",
        "arc_eval_provenance_invalid",
        "phase_receipt_rows_invalid",
    } <= errors


def test_req_arc_7032_terminal_schema_and_atomic_write(tmp_path: Path) -> None:
    """REQ-ARC-7032 rejects bad terminal semantics and atomically writes good JSON."""

    assert mod.validate_artifact(None) == ["artifact_object_required"]
    artifact = _valid_artifact(tmp_path)
    damaged = deepcopy(artifact)
    damaged["field_principles"] = {}
    damaged["inference_substrate"] = "development_proxy"
    damaged["verifier_is_oracle"] = True
    damaged["belief_shadow_trace_ready_score"] = True
    damaged["gate_check_summary"] = {}
    damaged["verdict_class"] = "mystery"
    errors = mod.validate_artifact(damaged)
    assert "field_principles_invalid" in errors
    assert "inference_substrate_invalid" in errors
    assert "verifier_is_oracle_invalid" in errors
    assert "ready_score_invalid" in errors
    assert "gate_check_summary_invalid" in errors
    assert "verdict_class_invalid" in errors
    assert "reproducibility_checksum_mismatch" in errors

    blocked = mod.build_blocked_artifact(
        run_date="20260905",
        duration_s=1.0,
        preconditions=[mod.gate_row("cuda", True, False)],
        source_artifact_hashes={},
        cited_upstream_artifacts=[],
    )
    blocked["belief_shadow_trace_ready_score"] = 1
    blocked["reproducibility_checksum"] = mod.artifact_checksum(blocked)
    assert "blocked_terminal_semantics_invalid" in mod.validate_artifact(blocked)

    output = tmp_path / "nested" / "artifact.json"
    mod.write_artifact(output, artifact)
    assert __import__("json").loads(output.read_text(encoding="utf-8")) == artifact
    with pytest.raises(ValueError, match="invalid Exp7032 artifact"):
        mod.write_artifact(output, {"verdict_class": "bad"})
