"""Tests for REQ-ARC-7039 and all SCENARIO-ARC-7039 variants."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from carnot import experiment_7039_v617_model_report_forensics as mod


def _snapshot_spec(tmp_path: Path, content: bytes = b"selected-model") -> tuple[dict, Path]:
    """Create the snapshot alias and content blob needed by REQ-ARC-7039."""

    digest = hashlib.sha256(content).hexdigest()
    root = tmp_path / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
    blob = root / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(content)
    revision = "a" * 40
    filename = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    requested = root / "snapshots" / revision / filename
    requested.parent.mkdir(parents=True)
    requested.symlink_to(Path("../../blobs") / digest)
    spec = {
        "name": mod.MANDATED_MODEL_NAME,
        "hf_id": mod.MANDATED_MODEL_HF_ID,
        "gpu": 0,
        "model_path": str(requested.absolute()),
        "model_filename": filename,
        "revision": revision,
        "quantization": "Q4_K_M",
        "model_file_hash": mod.sha256_file(requested),
        "resolved_via": "cached_sota_pair",
    }
    return spec, blob


def _direct_spec(tmp_path: Path) -> tuple[dict, Path]:
    """Create a regular GGUF so the direct-file report shape is explicit."""

    path = tmp_path / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    path.write_bytes(b"direct-selected-model")
    spec = {
        "name": mod.MANDATED_MODEL_NAME,
        "hf_id": mod.MANDATED_MODEL_HF_ID,
        "gpu": 0,
        "model_path": str(path.absolute()),
        "model_filename": path.name,
        "revision": "direct-file-fixture",
        "quantization": "Q4_K_M",
        "model_file_hash": mod.sha256_file(path),
        "resolved_via": "cached_sota_pair",
    }
    return spec, path


def _live_evidence(spec: dict, raw_props: dict) -> dict:
    """Build owned synthetic process evidence for the positive artifact reducer."""

    launch = [
        "/opt/llama.cpp/llama-server",
        "-m",
        spec["model_path"],
        "-ngl",
        "999",
        "-c",
        "4096",
        "--port",
        "17039",
    ]
    return {
        "launch_model_argument": spec["model_path"],
        "process_command_rows": [
            {
                "pid": 27039,
                "argv": launch,
                "command_line": " ".join(launch),
                "model_argument": spec["model_path"],
                "owned": True,
                "terminal": True,
            }
        ],
        "raw_server_props": raw_props,
        "one_token_probe_rows": [
            {
                "request_payload": {
                    "prompt": mod.ONE_TOKEN_PROMPT,
                    "n_predict": 1,
                    "temperature": 0.0,
                    "ignore_eos": True,
                    "seed": mod.RANDOM_SEED,
                },
                "response": {"content": "O", "timings": {"predicted_n": 1}},
                "requested_tokens": 1,
                "generated_tokens": 1,
                "completed": True,
                "terminal": True,
            }
        ],
        "server_process_rows": [
            {
                "pid": 27039,
                "owned": True,
                "port": 17039,
                "endpoint": "http://127.0.0.1:17039",
                "server_binary": launch[0],
                "server_build": "version: 9999 (cafef00d)",
                "launch_argv": launch,
                "cuda_layer_offload_confirmed": True,
                "terminal": True,
            }
        ],
        "port_lease_rows": [
            {
                "port": 17039,
                "owned": True,
                "released": True,
                "terminal": True,
            }
        ],
        "gpu_lease_rows": [
            {
                "gpu_uuid": "GPU-70390000-0000-0000-0000-000000000001",
                "owned": True,
                "released": True,
                "terminal": True,
            }
        ],
        "gpu_sample_rows": [
            {
                "pid": 27039,
                "gpu_uuid": "GPU-70390000-0000-0000-0000-000000000001",
                "gpu_model": "NVIDIA GeForce RTX 3090",
                "pid_memory_mb": 22000,
                "terminal": True,
            }
        ],
        "cleanup_rows": [
            {
                "owned_pid": 27039,
                "signals_sent_only_to_owned_pid": True,
                "process_exit_confirmed": True,
                "process_reaped": True,
                "lease_released": True,
                "port_released": True,
                "passed": True,
                "terminal": True,
            }
        ],
    }


def test_snapshot_alias_keeps_raw_and_resolved_paths_distinct(tmp_path: Path) -> None:
    """SCENARIO-ARC-7039-RAW-AND-RESOLVED-STAY-DISTINCT."""

    spec, blob = _snapshot_spec(tmp_path)
    raw = {
        "model_path": spec["model_path"],
        "model": None,
        "model_alias": spec["model_filename"],
        "default_generation_settings": {"n_ctx": 4096},
    }
    original = deepcopy(raw)

    report = mod.analyze_identity_report(raw, spec)

    assert raw == original
    assert report["raw_server_props"] == original
    assert report["raw_identity_field_rows"][0]["raw_value"] == spec["model_path"]
    assert report["resolved_identity_field_rows"][0]["resolved_path"] == str(blob.absolute())
    assert report["raw_identity_field_rows"][0]["raw_value"] != str(blob.absolute())
    assert report["report_channel_classification"] == "snapshot_alias"


def test_missing_report_fields_remain_unknown(tmp_path: Path) -> None:
    """SCENARIO-ARC-7039-REPORT-SHAPE-MATRIX: missing fields stay unknown."""

    spec, _blob = _snapshot_spec(tmp_path)
    report = mod.analyze_identity_report({}, spec)

    assert report["report_channel_classification"] == "unknown"
    assert [row["field"] for row in report["raw_identity_field_rows"]] == [
        "model_path",
        "model",
        "model_alias",
    ]
    assert all(row["evidence_status"] == "unknown" for row in report["raw_identity_field_rows"])
    assert all(
        row["evidence_status"] == "unknown"
        for row in report["resolved_identity_field_rows"]
    )
    assert all(row["status"] == "unknown" for row in report["consistency_rows"])


def test_relative_report_path_is_not_made_absolute(tmp_path: Path) -> None:
    """SCENARIO-ARC-7039-REPORT-SHAPE-MATRIX: relative paths stay unknown."""

    spec, _blob = _snapshot_spec(tmp_path)
    report = mod.analyze_identity_report({"model_path": "weights/model.gguf"}, spec)
    resolved = report["resolved_identity_field_rows"][0]

    assert report["report_channel_classification"] == "unknown"
    assert resolved["raw_value"] == "weights/model.gguf"
    assert resolved["resolved_path"] is None
    assert resolved["resolution_state"] == "relative"
    assert resolved["evidence_status"] == "unknown"


def test_canonical_blob_report_is_classified_separately(tmp_path: Path) -> None:
    """SCENARIO-ARC-7039-REPORT-SHAPE-MATRIX: canonical blob report."""

    spec, blob = _snapshot_spec(tmp_path)
    report = mod.analyze_identity_report({"model_path": str(blob.absolute())}, spec)

    assert report["report_channel_classification"] == "resolved_blob"
    assert report["resolved_identity_field_rows"][0]["raw_equals_resolved"] is True
    assert report["file_hash_rows"][0]["hash_matches_selected"] is True


def test_direct_file_report_is_classified_separately(tmp_path: Path) -> None:
    """SCENARIO-ARC-7039-REPORT-SHAPE-MATRIX: direct regular GGUF report."""

    spec, direct = _direct_spec(tmp_path)
    report = mod.analyze_identity_report({"model_path": str(direct.absolute())}, spec)

    assert report["report_channel_classification"] == "direct_file"
    assert report["resolved_identity_field_rows"][0]["is_symlink"] is False
    assert report["file_hash_rows"][0]["sha256"] == spec["model_file_hash"]


def test_conflicting_report_fields_are_contradicted(tmp_path: Path) -> None:
    """SCENARIO-ARC-7039-REPORT-SHAPE-MATRIX: conflicting reachable fields."""

    spec, _blob = _snapshot_spec(tmp_path)
    other = tmp_path / "different.gguf"
    other.write_bytes(b"different-model")
    report = mod.analyze_identity_report(
        {"model_path": spec["model_path"], "model": str(other.absolute())}, spec
    )

    assert report["report_channel_classification"] == "conflicting"
    statuses = {row["field"]: row["evidence_status"] for row in report["resolved_identity_field_rows"]}
    assert statuses == {"model_path": "supported", "model": "contradicted", "model_alias": "unknown"}
    consistency = {row["check"]: row["status"] for row in report["consistency_rows"]}
    assert consistency["selected_model_hash_agreement"] == "contradicted"
    assert consistency["resolved_report_path_agreement"] == "contradicted"


def test_model_spec_must_come_from_cached_sota_pair(tmp_path: Path) -> None:
    """REQ-ARC-7039: model selection uses cached_sota_pair and records exact identity."""

    spec, _blob = _snapshot_spec(tmp_path)
    calls: list[dict] = []

    def cached_pair(**kwargs: object) -> list[dict]:
        calls.append(dict(kwargs))
        return [
            {
                "name": "other",
                "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "gpu": 0,
                "model_path": str(tmp_path / "missing.gguf"),
            },
            {
                "name": spec["name"],
                "hf_id": spec["hf_id"],
                "gpu": 0,
                "model_path": spec["model_path"],
            },
        ]

    selected = mod.resolve_model_spec(cached_pair, gpu_index=1)

    assert calls == [{"gpu_indices": (1, 1)}]
    assert selected is not None
    assert selected["hf_id"] == mod.MANDATED_MODEL_HF_ID
    assert selected["revision"] == "a" * 40
    assert selected["model_filename"] == spec["model_filename"]
    assert selected["quantization"] == "Q4_K_M"
    assert selected["model_file_hash"] == spec["model_file_hash"]
    assert selected["resolved_via"] == "cached_sota_pair"


def test_model_spec_rejects_every_non_mandated_cache_shape(tmp_path: Path) -> None:
    """REQ-ARC-7039 rejects missing, wrong, non-GGUF, and non-snapshot choices."""

    assert mod.extract_quantization("model.gguf") is None
    assert mod.resolve_model_spec(lambda **_kwargs: None, gpu_index=0) is None
    assert (
        mod.resolve_model_spec(
            lambda **_kwargs: [{"hf_id": "different/model", "model_path": "missing.gguf"}],
            gpu_index=0,
        )
        is None
    )
    assert (
        mod.resolve_model_spec(
            lambda **_kwargs: [
                {"hf_id": mod.MANDATED_MODEL_HF_ID, "model_path": str(tmp_path / "missing.gguf")}
            ],
            gpu_index=0,
        )
        is None
    )
    direct = tmp_path / "model.gguf"
    direct.write_bytes(b"not-a-snapshot")
    assert mod._snapshot_revision(direct, mod.MANDATED_MODEL_HF_ID) is None
    assert (
        mod.resolve_model_spec(
            lambda **_kwargs: [
                {"hf_id": mod.MANDATED_MODEL_HF_ID, "model_path": str(direct)}
            ],
            gpu_index=0,
        )
        is None
    )


def test_unusable_identity_values_remain_unknown(tmp_path: Path) -> None:
    """SCENARIO-ARC-7039-REPORT-SHAPE-MATRIX covers unusable raw values."""

    spec, _blob = _snapshot_spec(tmp_path)
    directory = tmp_path / "directory"
    directory.mkdir()
    report = mod.analyze_identity_report(
        {
            "model_path": str(tmp_path / "unreachable.gguf"),
            "model": directory,
            "model_alias": "",
        },
        spec,
    )

    raw_kinds = {row["field"]: row["raw_kind"] for row in report["raw_identity_field_rows"]}
    assert raw_kinds == {
        "model_path": "absolute_path",
        "model": "non_string",
        "model_alias": "blank",
    }
    assert report["resolved_identity_field_rows"][0]["resolution_state"] == "unreachable"

    directory_report = mod.analyze_identity_report({"model": str(directory.absolute())}, spec)
    assert directory_report["resolved_identity_field_rows"][1]["resolution_state"] == "not_file"
    assert directory_report["report_channel_classification"] == "unknown"


def test_blocked_artifact_keeps_exact_failed_gate(tmp_path: Path) -> None:
    """SCENARIO-ARC-7039-BLOCKED-CAPTURE records the first exact blocker."""

    checks = [
        mod.gate_row("idle_supported_rtx3090", True, True),
        mod.gate_row("cached_mandated_gguf", True, False),
    ]
    artifact = mod.build_blocked_artifact(
        run_date="20260906",
        duration_s=0.25,
        preconditions=checks,
        source_artifact_hashes={"source": "sha256:" + "1" * 64},
        model_specs=[],
        selected_model_spec=None,
    )

    assert mod.validate_artifact(artifact) == []
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["gate_check_summary"]["failed_check"] == "cached_mandated_gguf"
    assert artifact["gate_check_summary"]["expected_value"] is True
    assert artifact["gate_check_summary"]["observed_value"] is False
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["generation_request_count"] == 0
    assert artifact["arc_action_count"] == 0

    missing_spec = {
        "hf_id": mod.MANDATED_MODEL_HF_ID,
        "model_path": str(tmp_path / "gone.gguf"),
        "model_filename": "gone.gguf",
        "revision": "gone",
        "model_file_hash": "sha256:" + "0" * 64,
        "resolved_via": "cached_sota_pair",
    }
    missing_file_artifact = mod.build_blocked_artifact(
        run_date="20260906",
        duration_s=0.5,
        preconditions=checks,
        source_artifact_hashes={},
        model_specs=[missing_spec],
        selected_model_spec=missing_spec,
    )
    selected_hash_row = missing_file_artifact["file_hash_rows"][0]
    assert selected_hash_row["sha256"] is None
    assert selected_hash_row["status"] == "unknown"


def test_positive_artifact_is_complete_without_validator_claim(tmp_path: Path) -> None:
    """SCENARIO-ARC-7039-OWNED-LIVE-CAPTURE validates evidence completeness only."""

    spec, _blob = _snapshot_spec(tmp_path)
    checks = [mod.gate_row("all_preconditions", True, True)]
    evidence = _live_evidence(spec, {"model_path": spec["model_path"]})
    artifact = mod.build_positive_artifact(
        run_date="20260906",
        duration_s=65.0,
        preconditions=checks,
        source_artifact_hashes={"source": "sha256:" + "2" * 64},
        model_specs=[spec],
        selected_model_spec=spec,
        live_evidence=evidence,
    )

    assert mod.validate_artifact(artifact) == []
    assert artifact["arc_report_channel_forensics_ready_score"] == 1
    assert artifact["report_channel_classification"] == "snapshot_alias"
    assert artifact["generation_request_count"] == 1
    assert artifact["arc_action_count"] == 0
    assert artifact["game_level_solve_claim"] is False
    assert artifact["verifier_is_oracle"] is False
    assert "validator" not in artifact["honest_verdict"]
    assert artifact["raw_server_props"] == {"model_path": spec["model_path"]}

    changed = deepcopy(artifact)
    changed["raw_identity_field_rows"][0]["raw_value"] = str(tmp_path / "normalized")
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "raw_identity_rows_do_not_match_props" in mod.validate_artifact(changed)

    too_short = mod.build_positive_artifact(
        run_date="20260906",
        duration_s=59.99,
        preconditions=checks,
        source_artifact_hashes={"source": "sha256:" + "2" * 64},
        model_specs=[spec],
        selected_model_spec=spec,
        live_evidence=evidence,
    )
    assert too_short["arc_report_channel_forensics_ready_score"] == 0
    assert too_short["verdict_class"] == "blocked"
    assert too_short["gate_check_summary"]["failed_check"] == "live_inference_duration_floor_s"


def test_artifact_validator_rejects_each_terminal_contract_error(tmp_path: Path) -> None:
    """REQ-ARC-7039 makes every terminal schema error machine-detectable."""

    spec, _blob = _snapshot_spec(tmp_path)
    base = mod.build_positive_artifact(
        run_date="20260906",
        duration_s=65.0,
        preconditions=[mod.gate_row("all_preconditions", True, True)],
        source_artifact_hashes={"source": "sha256:" + "3" * 64},
        model_specs=[spec],
        selected_model_spec=spec,
        live_evidence=_live_evidence(spec, {"model_path": spec["model_path"]}),
    )

    assert mod.validate_artifact([]) == ["artifact_object_required"]
    mutations = [
        (lambda value: value.pop("models_used"), "required_fields_missing"),
        (lambda value: value.__setitem__("field_principles", {}), "field_principles_invalid"),
        (lambda value: value.__setitem__("inference_substrate", "offline"), "inference_substrate_invalid"),
        (lambda value: value.__setitem__("verifier_is_oracle", True), "verifier_is_oracle_invalid"),
        (lambda value: value.__setitem__("arc_action_count", 1), "arc_non_action_contract_invalid"),
        (lambda value: value.__setitem__("report_channel_classification", "invented"), "report_channel_classification_invalid"),
        (lambda value: value.__setitem__("gate_check_summary", {}), "gate_check_summary_invalid"),
        (lambda value: value.__setitem__("honest_verdict", "wrong"), "verdict_prefix_invalid"),
        (lambda value: value.__setitem__("arc_report_channel_forensics_ready_score", True), "ready_score_invalid"),
        (lambda value: value.__setitem__("raw_server_props", None), "raw_server_props_invalid"),
        (lambda value: value.__setitem__("raw_identity_field_rows", {}), "raw_identity_rows_invalid"),
        (lambda value: value.__setitem__("resolved_identity_field_rows", {}), "resolved_identity_rows_invalid"),
        (lambda value: value["resolved_identity_field_rows"][0].__setitem__("raw_value", "changed"), "resolved_rows_do_not_retain_raw_values"),
        (lambda value: value["raw_identity_field_rows"][0].__setitem__("evidence_status", "inferred"), "identity_evidence_status_invalid"),
        (lambda value: value.__setitem__("consistency_rows", []), "consistency_rows_invalid"),
        (lambda value: value.__setitem__("MODEL_SPECS", []), "model_specs_invalid"),
        (
            lambda value: (
                value["selected_model_spec"].__setitem__("hf_id", "wrong/model"),
                value["MODEL_SPECS"][0].__setitem__("hf_id", "wrong/model"),
                value["model_specs"][0].__setitem__("hf_id", "wrong/model"),
            ),
            "selected_model_spec_invalid",
        ),
        (lambda value: value.__setitem__("arc_report_channel_forensics_ready_score", 0), "positive_terminal_semantics_invalid"),
        (
            lambda value: (
                value.__setitem__("verdict_class", "null"),
                value.__setitem__("honest_verdict", "complete_null_report"),
            ),
            "unsupported_terminal_class",
        ),
    ]
    for mutate, expected in mutations:
        changed = deepcopy(base)
        mutate(changed)
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        assert any(error.startswith(expected) for error in mod.validate_artifact(changed)), expected

    checksum_changed = deepcopy(base)
    checksum_changed["duration_s"] = 66.0
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(checksum_changed)

    blocked = mod.build_blocked_artifact(
        run_date="20260906",
        duration_s=1.0,
        preconditions=[mod.gate_row("blocked", True, False)],
        source_artifact_hashes={},
        model_specs=[],
        selected_model_spec=None,
    )
    blocked["arc_report_channel_forensics_ready_score"] = 1
    blocked["reproducibility_checksum"] = mod.artifact_checksum(blocked)
    assert "blocked_terminal_semantics_invalid" in mod.validate_artifact(blocked)


def test_artifact_writer_validates_and_atomically_publishes(tmp_path: Path) -> None:
    """REQ-ARC-7039 publishes only a valid terminal artifact."""

    artifact = mod.build_blocked_artifact(
        run_date="20260906",
        duration_s=1.0,
        preconditions=[mod.gate_row("blocked", True, False)],
        source_artifact_hashes={},
        model_specs=[],
        selected_model_spec=None,
    )
    target = tmp_path / "nested" / "artifact.json"

    mod.write_artifact(target, artifact)

    assert json.loads(target.read_text(encoding="utf-8")) == artifact
    invalid = deepcopy(artifact)
    invalid["arc_action_count"] = 1
    with pytest.raises(ValueError, match="arc_non_action_contract_invalid"):
        mod.write_artifact(tmp_path / "invalid.json", invalid)
