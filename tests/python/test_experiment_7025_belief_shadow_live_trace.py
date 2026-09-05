"""Tests for REQ-ARC-7025 and its live-shadow transport scenarios."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from carnot import task_runtime_receipts
from carnot.agentic import arc_belief_shadow_live_trace as mod
from carnot.agentic.arc_eval_provenance import (
    ArcEvalProvenanceInput,
    build_arc_eval_provenance,
)
from carnot.experiment_7017_task_linked_compute_receipts import run_consumer_fixture


ROOT = Path(__file__).resolve().parents[2]


def _provenance(*, model_hash: str, requests: int = 1) -> dict:
    """Build one strict Exp7010-shaped record for a synthetic unit boundary."""

    return build_arc_eval_provenance(
        ArcEvalProvenanceInput(
            inference_substrate="local_gguf_cuda",
            gpu_uuid="GPU-70250000-0000-0000-0000-000000000001",
            gpu_model="NVIDIA GeForce RTX 3090",
            cuda_device=1,
            model_repository=mod.MANDATED_MODEL_HF_ID,
            model_filename="Qwen3.6-35B-A3B-Q4_K_M.gguf",
            model_hash=model_hash,
            n_ctx=4096,
            server_binary="/opt/llama.cpp/llama-server",
            server_binary_hash="sha256:" + "2" * 64,
            server_command_hash="sha256:" + "3" * 64,
            endpoint="http://127.0.0.1:17025",
            port=17025,
            lease_id="lease-exp7025",
            lease_hash="sha256:" + "4" * 64,
            lease_issued_at="2026-09-05T00:00:00+00:00",
            lease_expires_at="2026-09-05T01:00:00+00:00",
            lease_checked_at="2026-09-05T00:30:00+00:00",
            request_count=requests,
            completion_count=requests,
            error_count=0,
            policy_hash="sha256:" + "5" * 64,
            factory_hash="sha256:" + "6" * 64,
            git_commit="7" * 40,
            solve_provenance="live_agent_self_discovery",
        )
    )


def _valid_artifact(tmp_path: Path) -> dict:
    """Assemble a complete transport artifact from the shared receipt validators."""

    model_path = tmp_path / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model_path.write_bytes(b"qwen-exp7025-unit")
    model_hash = mod.sha256_file(model_path)
    receipt, _ = run_consumer_fixture(tmp_path / "receipt", fixture_id="one-model")
    provenance = _provenance(model_hash=model_hash)
    spec = {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": mod.MANDATED_MODEL_HF_ID,
        "gpu": 1,
        "model_path": str(model_path),
        "model_file_hash": model_hash,
        "resolved_via": "cached_sota_pair",
    }
    action = {"action": 1, "data": None}
    artifact = {
        "schema": mod.SCHEMA,
        "experiment_id": mod.EXPERIMENT_ID,
        "execution_date": mod.RUN_DATE,
        "field_principles": dict(mod.FIELD_PRINCIPLES),
        "preconditions_checked": [
            mod.gate_row("structured_gate_exp7017", 1, 1),
            mod.gate_row("structured_gate_exp7024", 1, 1),
        ],
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "duration_s": 61.0,
        "MODEL_SPECS": [spec],
        "models_used": [mod.MANDATED_MODEL_HF_ID],
        "model_file_hashes": {str(model_path): model_hash},
        "cited_upstream_artifacts": [
            {
                "experiment_id": number,
                "fields_imported": fields,
                "sha256": "sha256:" + str(number)[-1] * 64,
            }
            for number, fields in (
                (7010, ["arc_eval_provenance_contract_ready_score"]),
                (7017, ["task_compute_receipt_ready_score"]),
                (7024, ["belief_selector_live_path_ready_score"]),
            )
        ],
        "source_artifact_hashes": {"module": "sha256:" + "8" * 64},
        "rows": [{"cell_count": 2, "passed": True, "terminal": True}],
        "per_action_results": [
            {"cell": "control", "action": action, "terminal": True},
            {"cell": "belief_shadow", "action": action, "terminal": True},
        ],
        "control_rows": [
            {
                "cell": "control",
                "action": action,
                "request_count": 1,
                "completion_count": 1,
                "error_count": 0,
                "arc_eval_provenance": provenance,
                "solve_provenance": "live_agent_self_discovery",
                "terminal": True,
            }
        ],
        "shadow_rows": [
            {
                "cell": "belief_shadow",
                "action": action,
                "request_count": 1,
                "completion_count": 1,
                "error_count": 0,
                "belief_query_count": 2,
                "arc_eval_provenance": provenance,
                "solve_provenance": "live_agent_self_discovery",
                "terminal": True,
            }
        ],
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
                "abstained": False,
                "terminal": True,
            }
        ],
        "ranking_influence_rows": [
            {
                "counterfactual_ranking_changed": True,
                "action_override_applied": False,
                "terminal": True,
            }
        ],
        "abstention_rows": [],
        "model_execution_rows": [
            {
                "hf_id": mod.MANDATED_MODEL_HF_ID,
                "model_path": str(model_path),
                "model_filename": model_path.name,
                "model_file_hash": model_hash,
                "n_ctx": 4096,
                "observed_server_n_ctx": 4096,
                "n_gpu_layers": 999,
                "launch_argv": [
                    "/opt/llama.cpp/llama-server",
                    "-m",
                    str(model_path),
                    "-ngl",
                    "999",
                    "-c",
                    "4096",
                ],
                "cuda_offload": True,
                "request_count": 2,
                "completion_count": 2,
                "error_count": 0,
                "pid": 27025,
                "terminal": True,
            }
        ],
        "gpu_identity_rows": [
            {
                "index": 1,
                "gpu_uuid": "GPU-70250000-0000-0000-0000-000000000001",
                "gpu_model": "NVIDIA GeForce RTX 3090",
                "supported": True,
                "terminal": True,
            }
        ],
        "gpu_sample_rows": list(receipt["gpu_sample_rows"]),
        "server_rows": [
            {
                "pid": 27025,
                "owned": True,
                "port": 17025,
                "endpoint": "http://127.0.0.1:17025",
                "model_file_hash": model_hash,
                "n_ctx": 4096,
                "request_count": 2,
                "completion_count": 2,
                "error_count": 0,
                "terminal": True,
            }
        ],
        "lease_rows": list(receipt["lease_link_rows"]),
        "request_counter_rows": [
            {"cell": "control", "before": 0, "after": 1, "delta": 1, "terminal": True},
            {
                "cell": "belief_shadow",
                "before": 1,
                "after": 2,
                "delta": 1,
                "terminal": True,
            },
        ],
        "completion_counter_rows": [
            {
                "cell": "control",
                "requests": 1,
                "completions": 1,
                "errors": 0,
                "terminal": True,
            },
            {
                "cell": "belief_shadow",
                "requests": 1,
                "completions": 1,
                "errors": 0,
                "terminal": True,
            },
        ],
        "phase_receipt_rows": list(receipt["rows"]),
        "runner_decision_rows": [dict(receipt["runner_decision"], terminal=True)],
        "checkpoint_rows": [
            {"cell": "control", "restored": False, "terminal": True},
            {"cell": "belief_shadow", "restored": False, "terminal": True},
            {"resume_verified": True, "duplicate_requests": 0, "terminal": True},
        ],
        "teardown_rows": [
            {
                "owned_pid": 27025,
                "process_exit_confirmed": True,
                "process_reaped": True,
                "lease_released": True,
                "port_released": True,
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
        "arc_new_level_banked": 0,
        "belief_shadow_trace_ready_score": 1,
        "random_seed": mod.RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "passed": True,
            "failed_check": None,
            "expected_value": True,
            "observed_value": True,
            "checks": [],
        },
        "verifier_is_oracle": False,
        "verdict_class": "positive",
        "honest_verdict": "complete_positive_belief_shadow_live_trace_transport_ready",
    }
    artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
    return artifact


def test_req_arc_7025_spec_exists_before_implementation() -> None:
    """REQ-ARC-7025 names every transport behavior tested below."""

    text = (ROOT / "openspec/capabilities/arc-agi/spec.md").read_text(encoding="utf-8")
    assert "REQ-ARC-7025" in text
    for scenario in (
        "SCENARIO-ARC-7025-MODEL-CUDA-IDENTITY",
        "SCENARIO-ARC-7025-SHADOW-PARITY-AND-QUERY",
        "SCENARIO-ARC-7025-PROVENANCE-AND-COUNTERS",
        "SCENARIO-ARC-7025-MISSING-RECEIPT-FAILS-CLOSED",
        "SCENARIO-ARC-7025-CHECKPOINT-RESUME",
        "SCENARIO-ARC-7025-OWNED-TEARDOWN",
    ):
        assert scenario in text


def test_scenario_7025_model_is_selected_through_cached_sota_pair(tmp_path: Path) -> None:
    """SCENARIO-ARC-7025-MODEL-CUDA-IDENTITY rejects a legacy substitution."""

    blob = tmp_path / "ac0e2c1189e055faa36eff361580e79c5bd6f8e76"
    blob.write_bytes(b"qwen")
    model = tmp_path / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model.symlink_to(blob)
    calls = []

    def pair(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": mod.MANDATED_MODEL_HF_ID,
                "gpu": 1,
                "model_path": str(model),
            },
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "gpu": 1,
                "model_path": str(tmp_path / "gemma.gguf"),
            },
        ]

    resolved = mod.resolve_model_spec(pair, gpu_index=1)
    assert calls == [{"gpu_indices": (1, 1)}]
    assert resolved["hf_id"] == mod.MANDATED_MODEL_HF_ID
    assert resolved["model_file_hash"] == mod.sha256_file(model)
    assert resolved["model_path"] == str(model.absolute())
    assert Path(resolved["model_path"]).suffix == ".gguf"
    assert resolved["resolved_via"] == "cached_sota_pair"

    assert mod.resolve_model_spec(lambda **_kwargs: None, gpu_index=1) is None
    assert (
        mod.resolve_model_spec(
            lambda **_kwargs: [
                {"hf_id": "legacy/tiny-model", "model_path": str(model), "gpu": 1}
            ],
            gpu_index=1,
        )
        is None
    )
    assert (
        mod.resolve_model_spec(
            lambda **_kwargs: [
                {
                    "hf_id": mod.MANDATED_MODEL_HF_ID,
                    "model_path": str(tmp_path / "missing.gguf"),
                }
            ],
            gpu_index=1,
        )
        is None
    )


def test_scenario_7025_shadow_query_can_influence_ranking_without_action_change() -> None:
    """SCENARIO-ARC-7025-SHADOW-PARITY-AND-QUERY returns the control order."""

    candidates = [{"action": 1, "data": None}, {"action": 2, "data": None}]

    class InfluencingSelector:
        last_decision = {}

        def rank_candidates(self, _frame, rows):  # type: ignore[no-untyped-def]
            self.last_decision = {
                "query_fired": True,
                "query_count": 2,
                "ranking_changed": True,
                "abstained": False,
                "abstention_reason": None,
                "evidence_hashes": ["sha256:" + "a" * 64],
                "selected_action": {"action": 2, "data": None},
            }
            return list(reversed(rows))

    shadow = mod.ShadowOnlyBeliefSelector(InfluencingSelector())
    returned = shadow.rank_candidates(object(), candidates)

    assert returned == candidates
    assert [id(row) for row in returned] == [id(row) for row in candidates]
    assert shadow.last_decision["query_fired"] is True
    assert shadow.last_decision["counterfactual_ranking_changed"] is True
    assert shadow.last_decision["action_override_applied"] is False
    assert shadow.last_decision["control_selected_action"] == {"action": 1, "data": None}
    assert shadow.last_decision["counterfactual_selected_action"] == {
        "action": 2,
        "data": None,
    }


def test_scenario_7025_checkpoint_resume_skips_completed_cells(tmp_path: Path) -> None:
    """SCENARIO-ARC-7025-CHECKPOINT-RESUME never duplicates completed work."""

    store = mod.CellCheckpointStore(tmp_path / "checkpoint.json", manifest_hash="sha256:manifest")
    assert store.pending_cells(mod.CELL_IDS) == list(mod.CELL_IDS)
    store.save_cell("control", {"request_count": 1, "action_count": 1})

    resumed = mod.CellCheckpointStore(
        tmp_path / "checkpoint.json", manifest_hash="sha256:manifest"
    )
    assert resumed.pending_cells(mod.CELL_IDS) == ["belief_shadow"]
    assert resumed.completed_cells["control"]["request_count"] == 1
    assert resumed.resume_row()["duplicate_requests"] == 0

    with pytest.raises(ValueError, match="manifest"):
        mod.CellCheckpointStore(
            tmp_path / "checkpoint.json", manifest_hash="sha256:different"
        )
    with pytest.raises(ValueError, match="cell"):
        store.save_cell("unknown", {})

    malformed = tmp_path / "malformed-checkpoint.json"
    malformed.write_text(
        '{"manifest_hash":"sha256:manifest","completed_cells":[]}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="completed_cells"):
        mod.CellCheckpointStore(malformed, manifest_hash="sha256:manifest")


def test_scenario_7025_atomic_checkpoint_cleanup_on_publish_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-7025-CHECKPOINT-RESUME removes a failed temporary write."""

    store = mod.CellCheckpointStore(tmp_path / "checkpoint.json", manifest_hash="manifest")

    def fail_replace(_source, _target):  # type: ignore[no-untyped-def]
        raise OSError("simulated publish failure")

    monkeypatch.setattr(mod.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publish failure"):
        store.save_cell("control", {"terminal": True})
    assert list(tmp_path.iterdir()) == []


def test_scenario_7025_complete_artifact_validates(tmp_path: Path) -> None:
    """SCENARIO-ARC-7025-PROVENANCE-AND-COUNTERS accepts all shared receipts."""

    artifact = _valid_artifact(tmp_path)
    assert task_runtime_receipts.validate_task_compute_receipt(
        artifact["task_compute_receipt"]
    )["accepted"] is True
    assert mod.validate_artifact(artifact) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda a: a["MODEL_SPECS"][0].__setitem__("hf_id", "legacy/tiny"), "model_identity"),
        (
            lambda a: a["model_execution_rows"][0].__setitem__(
                "model_file_hash", "sha256:" + "0" * 64
            ),
            "model_file_hash",
        ),
        (
            lambda a: a["model_execution_rows"][0].__setitem__("cuda_offload", False),
            "cuda_offload",
        ),
        (
            lambda a: a["model_execution_rows"][0].__setitem__("launch_argv", []),
            "cuda_offload",
        ),
        (
            lambda a: a["model_execution_rows"][0].__setitem__(
                "observed_server_n_ctx", 2048
            ),
            "n_ctx",
        ),
        (lambda a: a["server_rows"][0].__setitem__("owned", False), "server"),
        (
            lambda a: a["server_rows"][0].__setitem__("completion_count", 1),
            "counter",
        ),
        (
            lambda a: a["shadow_action_parity_rows"][0].__setitem__(
                "shadow_action", {"action": 2, "data": None}
            ),
            "shadow_action_parity",
        ),
        (
            lambda a: a["belief_query_rows"][0].__setitem__("query_fired", False),
            "belief_query",
        ),
        (lambda a: a.__setitem__("phase_receipt_rows", []), "phase_receipt"),
        (
            lambda a: a["task_compute_receipt"].__setitem__("cleanup_rows", []),
            "task_compute_receipt",
        ),
        (
            lambda a: a["runner_decision_rows"][0].__setitem__(
                "runner_selected", "DualGPURunner"
            ),
            "runner",
        ),
        (lambda a: a["teardown_rows"][0].__setitem__("port_released", False), "teardown"),
        (
            lambda a: a["checkpoint_rows"][-1].__setitem__("duplicate_requests", 1),
            "checkpoint",
        ),
        (
            lambda a: a["solve_registry_precheck_rows"][0].__setitem__(
                "registry_hash_after", "sha256:" + "0" * 64
            ),
            "registry",
        ),
        (
            lambda a: a["control_rows"][0]["arc_eval_provenance"].__setitem__(
                "gpu_uuid", "wrong"
            ),
            "arc_eval_provenance",
        ),
    ],
)
def test_scenario_7025_missing_or_contradictory_receipts_fail_closed(
    tmp_path: Path, mutation, expected: str
) -> None:
    """SCENARIO-ARC-7025-MISSING-RECEIPT-FAILS-CLOSED names damaged evidence."""

    artifact = _valid_artifact(tmp_path)
    mutation(artifact)
    artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
    assert any(expected in error for error in mod.validate_artifact(artifact))


def test_scenario_7025_blocked_artifact_has_exact_gate_summary(tmp_path: Path) -> None:
    """REQ-ARC-7025 emits the exact failed precondition and no fallback evidence."""

    checks = [mod.gate_row("cuda_llama_server", True, False)]
    artifact = mod.build_blocked_artifact(
        run_date=mod.RUN_DATE,
        duration_s=0.25,
        preconditions=checks,
        source_hashes={"module": "sha256:" + "1" * 64},
    )
    assert artifact["belief_shadow_trace_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_belief_shadow_live_trace")
    assert artifact["gate_check_summary"]["failed_check"] == "cuda_llama_server"
    assert artifact["gate_check_summary"]["expected_value"] is True
    assert artifact["gate_check_summary"]["observed_value"] is False
    assert mod.validate_artifact(artifact) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda a: a.pop("rows"), "required_fields_missing"),
        (lambda a: a.__setitem__("field_principles", {}), "field_principles"),
        (lambda a: a.__setitem__("inference_substrate", "synthetic"), "inference_substrate"),
        (lambda a: a.__setitem__("verifier_is_oracle", True), "verifier_is_oracle"),
        (lambda a: a.__setitem__("verdict_class", "mystery"), "verdict_prefix"),
        (lambda a: a.__setitem__("belief_shadow_trace_ready_score", True), "ready_score"),
        (lambda a: a.__setitem__("gate_check_summary", {}), "gate_check_summary"),
        (lambda a: a.__setitem__("preconditions_checked", []), "preconditions"),
        (lambda a: Path(a["MODEL_SPECS"][0]["model_path"]).unlink(), "model_file_hash"),
        (lambda a: a["gpu_identity_rows"].clear(), "gpu_identity"),
        (lambda a: a["control_rows"].clear(), "cell_rows"),
        (lambda a: a["ranking_influence_rows"].clear(), "ranking_influence"),
        (lambda a: a["gpu_sample_rows"].clear(), "gpu_or_lease"),
        (lambda a: a.__setitem__("arc_new_level_banked", 1), "arc_new_level"),
        (lambda a: a.__setitem__("solve_provenance", "registry_replay"), "solve_provenance"),
    ],
)
def test_req_arc_7025_artifact_envelope_defenses(
    tmp_path: Path, mutation, expected: str
) -> None:
    """REQ-ARC-7025 rejects malformed terminal envelopes and absent receipt classes."""

    artifact = _valid_artifact(tmp_path)
    mutation(artifact)
    artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
    assert any(expected in error for error in mod.validate_artifact(artifact))


def test_req_arc_7025_checksum_and_terminal_consistency_defenses(tmp_path: Path) -> None:
    """REQ-ARC-7025 binds checksums and ready scores to terminal verdict semantics."""

    artifact = _valid_artifact(tmp_path)
    artifact["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(artifact)
    assert mod.validate_artifact([]) == ["artifact_object_required"]

    artifact = _valid_artifact(tmp_path)
    artifact["gate_check_summary"]["passed"] = False
    artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
    assert "positive_gate_inconsistent" in mod.validate_artifact(artifact)

    blocked = mod.build_blocked_artifact(
        run_date=mod.RUN_DATE,
        duration_s=0.1,
        preconditions=[mod.gate_row("gpu", True, False)],
        source_hashes={},
    )
    blocked["belief_shadow_trace_ready_score"] = 1
    blocked["reproducibility_checksum"] = mod.artifact_checksum(blocked)
    assert "blocked_gate_inconsistent" in mod.validate_artifact(blocked)


def test_req_arc_7025_atomic_artifact_write_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-7025 validates before publish and removes failed temporary output."""

    artifact = mod.build_blocked_artifact(
        run_date=mod.RUN_DATE,
        duration_s=0.1,
        preconditions=[mod.gate_row("gpu", True, False)],
        source_hashes={},
    )
    output = tmp_path / "artifact.json"
    mod.write_artifact(output, artifact)
    assert output.is_file()

    invalid = deepcopy(artifact)
    invalid["reproducibility_checksum"] = "invalid"
    with pytest.raises(ValueError, match="invalid Exp7025 artifact"):
        mod.write_artifact(tmp_path / "invalid.json", invalid)

    output.unlink()

    def fail_replace(_source, _target):  # type: ignore[no-untyped-def]
        raise OSError("simulated publish failure")

    monkeypatch.setattr(mod.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publish failure"):
        mod.write_artifact(output, artifact)
    assert list(tmp_path.iterdir()) == []
