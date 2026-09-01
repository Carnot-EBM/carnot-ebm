"""REQ-INFERENCE-6850 three-family scoring admission tests."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6850_three_family_scoring_admission_canary as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / exp.SPEC_RELATIVE_PATH


def _model_specs(tmp_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, hf_id in enumerate(exp.MODEL_SPECS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(b"GGUF" + hf_id.encode("utf-8"))
        tokenizer = {
            "source": "embedded_gguf_llama_cpp_vocab_only",
            "metadata_present": True,
            "loadable": True,
            "detail": "fixture native metadata",
            "tokenizer_sha256": exp.sha256_text(f"tokenizer:{hf_id}"),
        }
        rows.append(
            {
                "hf_id": hf_id,
                "family": exp.MODEL_FAMILIES[hf_id],
                "model_path": str(path),
                "model_sha256": exp.sha256_file(path),
                "model_size_bytes": path.stat().st_size,
                "tokenizer_receipt": tokenizer,
                "cached_sota_pair_called": True,
            }
        )
    return rows


def _ready_preconditions() -> dict[str, Any]:
    checks = [
        exp.gate_check("cached_sota_pair_called", True, True),
        exp.gate_check(
            "all_three_exact_gguf_artifacts", list(exp.MODEL_SPECS), list(exp.MODEL_SPECS)
        ),
        exp.gate_check("model_hashes_present", True, True),
        exp.gate_check("native_tokenizer_metadata", True, True),
        exp.gate_check("cuda_token_scoring", True, True),
        exp.gate_check("sufficient_disk", True, True),
        exp.gate_check("free_owned_ports", True, True),
        exp.gate_check("eligible_gpu", True, True),
        exp.gate_check("bounded_exclusive_gpu_lease", True, True),
    ]
    return {
        "preconditions_ready": True,
        "checks": checks,
        "blocked_reasons": [],
        "ports": [9201, 9202, 9203],
        "eligible_gpu": {
            "index": 0,
            "gpu_uuid": "GPU-fixture",
            "free_vram_mb": 24000,
            "visible_devices": "0",
        },
        "accelerator_samples": [],
        "unrelated_processes_observed": [],
    }


def _bundle(model: dict[str, Any], port: int, *, value: float = -0.25) -> dict[str, Any]:
    row = {
        "hf_id": model["hf_id"],
        "model_hash": model["model_sha256"],
        "tokenizer_hash": model["tokenizer_receipt"]["tokenizer_sha256"],
        "prompt_token_ids": [1, 2],
        "candidate_token_ids": [3, 4],
        "token_logprobs": [value, value - 0.1],
        "conditional_log_likelihood": round(value * 2 - 0.1, 6),
        "first_useful_output": "owned",
        "final_output": "owned canary",
        "latency_s": 0.25,
        "scientific_label": None,
        "supports_margin_claim": False,
        "canary_hash": "",
    }
    row["canary_hash"] = exp.canary_hash(row)
    process_receipt = {
        "pid": 7000 + port,
        "start_time_ticks": 101 + port,
        "command": ["python", "--score-worker"],
        "command_hash": exp.sha256_text(f"command:{port}"),
        "process_group_id": 7000 + port,
        "owner_pid": 6000,
        "owner_start_time_ticks": 88,
        "ownership_token_digest": exp.sha256_text(f"token:{port}"),
        "owned_by_task": True,
        "port": port,
        "gpu_uuid": "GPU-fixture",
        "visible_devices": "0",
        "model_hash": model["model_sha256"],
        "tokenizer_hash": model["tokenizer_receipt"]["tokenizer_sha256"],
    }
    return {
        "row": row,
        "process_receipt": process_receipt,
        "accelerator_samples": [
            {"phase": "before", "gpu_uuid": "GPU-fixture", "free_vram_mb": 24000},
            {"phase": "resident", "gpu_uuid": "GPU-fixture", "free_vram_mb": 4000},
            {"phase": "after", "gpu_uuid": "GPU-fixture", "free_vram_mb": 23950},
        ],
        "lease_receipt": {
            "owner": {"device_uuid": "GPU-fixture", "expected_model": model["hf_id"]},
            "lease_valid": True,
            "release": {"released": True, "phase": "terminal_complete"},
        },
        "teardown_receipt": {
            "ownership_verified": True,
            "process_exit_confirmed": True,
            "port_release_confirmed": True,
            "leak_free": True,
            "unrelated_process_kill_count_delta": 0,
        },
    }


def test_req_inference_6850_spec_declares_owned_admission_contract() -> None:
    """REQ-INFERENCE-6850 is present before implementation code."""

    section = SPEC_PATH.read_text(encoding="utf-8").split("### REQ-INFERENCE-6850:", 1)[1]
    for marker in (
        "SCENARIO-INFERENCE-6850-PROCESS-IDENTITY",
        "SCENARIO-INFERENCE-6850-PORT-AND-ORPHAN",
        "SCENARIO-INFERENCE-6850-LEASE-LOSS",
        "SCENARIO-INFERENCE-6850-CANARY",
        "SCENARIO-INFERENCE-6850-RESTART",
        "SCENARIO-INFERENCE-6850-TEARDOWN",
        exp.RESULT_RELATIVE_PATH.as_posix(),
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for hf_id in exp.MODEL_SPECS:
        assert hf_id in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_inference_6850_resolver_calls_cached_pair_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-INFERENCE-6850 calls cached_sota_pair before exact-file resolution."""

    fixtures = _model_specs(tmp_path)
    calls: list[str] = []

    def cached() -> list[dict[str, Any]]:
        calls.append("cached_sota_pair")
        return [
            {"hf_id": fixtures[0]["hf_id"], "model_path": fixtures[0]["model_path"]},
            {"hf_id": fixtures[2]["hf_id"], "model_path": fixtures[2]["model_path"]},
        ]

    def resolve(hf_id: str, quantization: str = "Q4_K_M") -> str:
        calls.append(f"resolve:{hf_id}:{quantization}")
        return next(row["model_path"] for row in fixtures if row["hf_id"] == hf_id)

    monkeypatch.setattr(exp, "cached_sota_pair", cached)
    monkeypatch.setattr(exp, "resolve_cached_gguf", resolve)
    monkeypatch.setattr(
        exp,
        "native_tokenizer_receipt",
        lambda path: {
            "source": "fixture",
            "metadata_present": True,
            "loadable": True,
            "detail": path,
            "tokenizer_sha256": exp.sha256_text(path),
        },
    )

    rows = exp.resolve_model_specs()

    assert calls[0] == "cached_sota_pair"
    assert [row["hf_id"] for row in rows] == list(exp.MODEL_SPECS)
    assert all(row["model_sha256"].startswith("sha256:") for row in rows)


def test_scenario_6850_missing_tokenizer_metadata_blocks_before_runner(tmp_path: Path) -> None:
    """SCENARIO-INFERENCE-6850-CANARY blocks missing native metadata."""

    specs = _model_specs(tmp_path)
    specs[1]["tokenizer_receipt"]["metadata_present"] = False
    preconditions = _ready_preconditions()
    preconditions["preconditions_ready"] = False
    preconditions["blocked_reasons"] = ["native_tokenizer_metadata"]
    preconditions["checks"] = [exp.gate_check("native_tokenizer_metadata", True, False)]
    calls: list[str] = []

    artifact = exp.run(
        root=REPO_ROOT,
        result_path=tmp_path / "blocked.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        model_specs=specs,
        preconditions_checked=preconditions,
        model_runner=lambda *args, **kwargs: calls.append("called"),
    )

    assert calls == []
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_check"] == "native_tokenizer_metadata"
    assert artifact["gate_check_summary"]["observed"] is False
    assert artifact["rows"] == []


def test_scenario_6850_failed_canary_and_lease_loss_fail_readiness(tmp_path: Path) -> None:
    """SCENARIO-INFERENCE-6850-LEASE-LOSS rejects non-finite or unleased rows."""

    specs = _model_specs(tmp_path)

    def runner(model: dict[str, Any], port: int, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        bundle = _bundle(model, port)
        if model["hf_id"] == exp.MODEL_SPECS[1]:
            bundle["row"]["token_logprobs"] = [-0.2, math.nan]
            bundle["row"]["canary_hash"] = exp.canary_hash(bundle["row"])
            bundle["lease_receipt"]["lease_valid"] = False
        return bundle

    artifact = exp.run(
        root=REPO_ROOT,
        result_path=tmp_path / "partial.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        model_specs=specs,
        preconditions_checked=_ready_preconditions(),
        model_runner=runner,
    )

    assert artifact["admission_canary_complete_score"] == 0
    assert artifact["three_family_scoring_admission_ready_score"] == 0
    assert artifact["scientific_effect_claimed"] is False
    assert artifact["verdict_class"] == "partial"
    assert artifact["gate_check_summary"]["failed_check"].startswith("model.")
    assert "lease_valid" in json.dumps(artifact["gate_check_summary"])


def test_scenario_6850_restart_reruns_only_incomplete_or_hash_drifted(tmp_path: Path) -> None:
    """SCENARIO-INFERENCE-6850-RESTART skips only complete verified entries."""

    specs = _model_specs(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.json"
    complete = _bundle(specs[0], 9201)
    drifted = _bundle(specs[1], 9202)
    drifted["row"]["model_hash"] = "sha256:stale"
    exp.write_checkpoint(
        checkpoint_path,
        exp.build_checkpoint_manifest([complete, drifted], model_specs=specs),
    )
    calls: list[str] = []

    def runner(model: dict[str, Any], port: int, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        calls.append(model["hf_id"])
        return _bundle(model, port)

    artifact = exp.run(
        root=REPO_ROOT,
        result_path=tmp_path / "restart.json",
        checkpoint_path=checkpoint_path,
        model_specs=specs,
        preconditions_checked=_ready_preconditions(),
        model_runner=runner,
    )

    assert calls == [exp.MODEL_SPECS[1], exp.MODEL_SPECS[2]]
    assert artifact["checkpoint_manifest"]["resumed_model_count"] == 1
    assert artifact["checkpoint_manifest"]["rerun_model_count"] == 2
    assert artifact["three_family_scoring_admission_ready_score"] == 1


def test_scenario_6850_complete_artifact_has_no_effect_or_quality_claim(tmp_path: Path) -> None:
    """SCENARIO-INFERENCE-6850-CANARY readiness uses receipts, not score direction."""

    specs = _model_specs(tmp_path)
    values = iter([-0.1, -2.0, -50.0])
    artifact = exp.run(
        root=REPO_ROOT,
        result_path=tmp_path / "ready.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        model_specs=specs,
        preconditions_checked=_ready_preconditions(),
        model_runner=lambda model, port, **kwargs: _bundle(model, port, value=next(values)),
    )

    assert artifact["admission_canary_complete_score"] == 1
    assert artifact["three_family_scoring_admission_ready_score"] == 1
    assert artifact["models_used"] == list(exp.MODEL_SPECS)
    assert artifact["scientific_effect_claimed"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert all(row["scientific_label"] is None for row in artifact["rows"])
    assert all(row["supports_margin_claim"] is False for row in artifact["rows"])
    assert all(
        row["teardown_receipt"]["port_release_confirmed"]
        for row in artifact["checkpoint_manifest"]["completed"]
    )
    assert exp.validate_artifact(artifact) == []
    assert set(artifact) <= set(artifact["field_principles"])
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_req_inference_6850_checkpoint_rejects_changed_canary_hash(tmp_path: Path) -> None:
    """REQ-INFERENCE-6850 fails closed when a completed canary changes."""

    specs = _model_specs(tmp_path)
    bundle = _bundle(specs[0], 9201)
    bundle["row"]["final_output"] = "changed after hash"

    assert "canary_hash" in exp.bundle_errors(bundle, specs[0])


def test_req_inference_6850_validation_rejects_malformed_receipts(tmp_path: Path) -> None:
    """REQ-INFERENCE-6850 fails every malformed canary and ownership receipt closed."""

    specs = _model_specs(tmp_path)
    model = specs[0]
    row = {
        "hf_id": "wrong",
        "model_hash": "sha256:wrong",
        "tokenizer_hash": "sha256:wrong",
        "candidate_token_ids": [],
        "token_logprobs": [0.0],
        "first_useful_output": "",
        "final_output": "",
        "latency_s": -1,
        "scientific_label": "quality",
        "supports_margin_claim": True,
        "canary_hash": "sha256:wrong",
    }
    bundle = {
        "row": row,
        "process_receipt": {},
        "lease_receipt": {"lease_valid": True, "release": {}},
        "teardown_receipt": {"unrelated_process_kill_count_delta": 1},
    }

    errors = exp.bundle_errors(bundle, model)

    assert {
        "hf_id",
        "model_hash",
        "tokenizer_hash",
        "candidate_token_ids",
        "token_logprob_alignment",
        "first_useful_output",
        "final_output",
        "latency_s",
        "scientific_label",
        "supports_margin_claim",
        "canary_hash",
        "process_receipt",
        "lease_release",
        "teardown_ownership_verified",
        "teardown_process_exit_confirmed",
        "teardown_port_release_confirmed",
        "teardown_leak_free",
        "unrelated_process_kill_count_delta",
    } <= set(errors)


def test_req_inference_6850_checkpoint_and_json_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-INFERENCE-6850-RESTART rejects malformed and tampered state."""

    specs = _model_specs(tmp_path)
    bad_json = tmp_path / "array.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(exp.AdmissionError, match="json_object_required"):
        exp.read_json(bad_json)

    bundle = _bundle(specs[0], 9201)
    manifest = exp.build_checkpoint_manifest([bundle], model_specs=specs)
    manifest["completed"] = ["malformed", manifest["completed"][0]]
    manifest["completed"][1]["entry_hash"] = "sha256:tampered"
    checkpoint = tmp_path / "tampered.json"
    exp.write_checkpoint(checkpoint, manifest)

    assert exp._verified_checkpoint_bundles(checkpoint, specs) == {}
    assert exp._verified_checkpoint_bundles(tmp_path / "missing.json", specs) == {}
    assert (
        exp.build_checkpoint_manifest([{"row": {"hf_id": "unknown"}}], model_specs=specs)[
            "completed"
        ]
        == []
    )


def test_req_inference_6850_run_guards_runner_and_port_contract(tmp_path: Path) -> None:
    """REQ-INFERENCE-6850 requires three ports and object-shaped runner receipts."""

    specs = _model_specs(tmp_path)
    wrong_ports = _ready_preconditions()
    wrong_ports["ports"] = [9201]
    with pytest.raises(exp.AdmissionError, match="three_ports_required"):
        exp.run(
            root=REPO_ROOT,
            model_specs=specs,
            preconditions_checked=wrong_ports,
            write=False,
        )

    with pytest.raises(exp.AdmissionError, match="model_runner_object_required"):
        exp.run(
            root=REPO_ROOT,
            checkpoint_path=tmp_path / "runner-checkpoint.json",
            model_specs=specs,
            preconditions_checked=_ready_preconditions(),
            model_runner=lambda *args, **kwargs: None,
            write=False,
        )

    blocked = _ready_preconditions()
    blocked["preconditions_ready"] = False
    blocked["blocked_reasons"] = ["fixture"]
    artifact = exp.run(
        root=REPO_ROOT,
        model_specs=specs,
        preconditions_checked=blocked,
        write=False,
    )
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT

    complete = exp.run(
        root=REPO_ROOT,
        checkpoint_path=tmp_path / "complete-checkpoint.json",
        model_specs=specs,
        preconditions_checked=_ready_preconditions(),
        model_runner=lambda model, port, **kwargs: _bundle(model, port),
        write=False,
    )
    assert complete["three_family_scoring_admission_ready_score"] == 1


def test_req_inference_6850_blank_canary_and_artifact_validator(tmp_path: Path) -> None:
    """REQ-INFERENCE-6850 reports incomplete rows and contradictory readiness."""

    model = _model_specs(tmp_path)[0]
    row = exp._blank_canary(model, error_text="failed_canary")
    assert row["error"] == "failed_canary"
    assert row["canary_hash"] == exp.canary_hash(row)

    invalid = {
        "field_principles": {},
        "inference_substrate": "wrong",
        "scientific_effect_claimed": True,
        "verifier_is_oracle": True,
        "verdict_class": "unknown",
        "honest_verdict": "positive",
        "three_family_scoring_admission_ready_score": 2,
    }
    errors = exp.validate_artifact(invalid)
    assert errors[0].startswith("missing_required_fields")
    assert {
        "field_principles",
        "inference_substrate",
        "scientific_effect_claimed",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
        "three_family_scoring_admission_ready_score",
    } <= set(errors)

    contradictory = {
        **invalid,
        "three_family_scoring_admission_ready_score": 1,
        "admission_canary_complete_score": 0,
        "gate_check_summary": {"passed": False},
    }
    errors = exp.validate_artifact(contradictory)
    assert "ready_without_complete_canaries" in errors
    assert "ready_with_failed_gate" in errors


def test_req_inference_6850_main_rejects_invalid_artifact_and_delegates_worker(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-INFERENCE-6850 returns nonzero validation and delegates worker mode."""

    monkeypatch.setattr(exp, "run", lambda **kwargs: {})
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["invalid_receipt"])
    assert exp.main(["--date", "20260901"]) == 1
    assert "invalid_receipt" in capsys.readouterr().out

    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        exp,
        "_run_score_worker",
        lambda model_path, port: calls.append((model_path, port)) or 7,
    )
    assert exp.main(["--score-worker", "--model-path", "m.gguf", "--port", "9123"]) == 7
    assert calls == [("m.gguf", 9123)]


def test_req_inference_6850_main_uses_requested_date_and_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-INFERENCE-6850 exposes the required CLI command."""

    calls: list[dict[str, Any]] = []

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        calls.append(deepcopy(kwargs))
        return {"honest_verdict": exp.BLOCKED_VERDICT}

    monkeypatch.setattr(exp, "run", fake_run)
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: [])
    output = tmp_path / "artifact.json"
    checkpoint = tmp_path / "checkpoint.json"

    assert (
        exp.main(
            [
                "--date",
                "20260901",
                "--result-path",
                str(output),
                "--checkpoint-path",
                str(checkpoint),
            ]
        )
        == 0
    )
    assert calls[0]["result_path"] == output
    assert calls[0]["checkpoint_path"] == checkpoint
    assert exp.BLOCKED_VERDICT in capsys.readouterr().out
    with pytest.raises(exp.AdmissionError, match="run_date_mismatch"):
        exp.main(["--date", "20260902"])
