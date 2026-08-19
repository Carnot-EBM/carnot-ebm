"""Tests for Exp6462 SOTA raw persistence uniqueness canary.

Spec refs: REQ-INFRA-6462, SCENARIO-INFRA-6462-1,
SCENARIO-INFRA-6462-2, SCENARIO-INFRA-6462-3,
SCENARIO-INFRA-6462-4.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6462_sota_raw_persistence_uniqueness_canary as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + "-Q4_K_M.gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((model_id + "\nExp6462 fixture weights\n").encode("utf-8"))
        paths[model_id] = path
    return paths


def _cached_pair(paths: dict[str, Path], calls: list[dict[str, Any]]):
    def resolve(
        *,
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        calls.append(
            {
                "gpu_indices": gpu_indices,
                "preferred_quant": preferred_quant,
                "model_indices": model_indices,
            }
        )
        ordered = (
            (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[2])
            if model_indices is None
            else (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[1])
        )
        return [
            {
                "name": mod.MODEL_TEMPLATE_BY_ID[model_id]["name"],
                "hf_id": model_id,
                "gpu": gpu,
                "model_path": str(paths[model_id]),
            }
            for gpu, model_id in zip(gpu_indices, ordered, strict=True)
        ]

    return resolve


def _tokenizer(path: str) -> tuple[bool, str]:
    return True, f"fixture embedded tokenizer for {Path(path).name}"


def _host_ok(
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"resource": "rtx_3090_gpu_count", "available": True, "detail": "2 fixture GPUs"},
        {"resource": "free_vram", "available": True, "detail": "fixture VRAM OK"},
        {"resource": "mandatory_model_files", "available": True, "detail": str(len(model_specs))},
        {"resource": "embedded_gguf_tokenizers", "available": True, "detail": "fixture OK"},
        {"resource": "llama_cpp_cuda_runner", "available": True, "detail": "fixture CUDA runner"},
        {"resource": "disk_space", "available": True, "detail": "fixture OK"},
        {"resource": "monotonic_clock", "available": True, "detail": "fixture OK"},
        {
            "resource": "new_output_paths",
            "available": not (data_dir / "raw_outputs").exists() and not result_path.exists(),
            "detail": "fresh fixture paths",
        },
    ]


def _fixture_event_id(
    *,
    unit_id: str,
    model_hf_id: str,
    replicate_index: int,
    seed: int,
) -> str:
    return (
        f"evt-{unit_id}-{mod.model_slug(model_hf_id)}-"
        f"{replicate_index}-{seed}"
    )


def _fixture_generation(
    *,
    model: dict[str, Any],
    unit: dict[str, Any],
    replicate_index: int,
    seed: int,
    prompt: str,
    event_id: str,
    decoding_settings: dict[str, Any],
) -> dict[str, Any]:
    raw_text = json.dumps(
        {
            "answer": "same diagnostic text",
            "unit_id": unit["unit_id"],
            "model_family": model["model_family"],
            "replicate_index": replicate_index,
        },
        sort_keys=True,
    )
    if unit["unit_id"] == "unit-00" and replicate_index in (0, 1):
        raw_text = "identical raw text allowed for diagnostic only\n"
    return {
        "raw_text": raw_text,
        "runtime_receipt": {
            "pid": 9000 + replicate_index,
            "parent_pid": 8000,
            "device_uuid": f"GPU-fixture-{model['gpu']}",
            "gpu_index": model["gpu"],
            "cuda_offload": True,
            "cpu_fallback": False,
            "completion_tokens": 12,
            "first_token_observed": True,
        },
        "timing": {
            "started_monotonic_ns": 1000 + seed,
            "ended_monotonic_ns": 2000 + seed,
            "duration_s": 1.0,
        },
        "prompt_seen_sha256": mod.sha256_text(prompt),
        "decoding_settings_seen": dict(decoding_settings),
        "event_id_seen": event_id,
    }


def _test_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = False) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    return mod.run(
        date="20260819",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "exp6462-data",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        host_preflight_func=_host_ok,
        generation_func=_fixture_generation,
        event_id_func=_fixture_event_id,
        test_exit_codes=_test_exit_codes(),
        duration_s=125.0,
        write=write,
    )


def test_req_infra_6462_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6462: OpenSpec owns the raw-persistence canary contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6462") : text.index("REQ-INFRA-6351")]
    for marker in (
        "SCENARIO-INFRA-6462-1",
        "SCENARIO-INFRA-6462-2",
        "SCENARIO-INFRA-6462-3",
        "SCENARIO-INFRA-6462-4",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "one-to-one event, path, and durable hash binding",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES
    for condition in mod.READINESS_CONDITIONS:
        assert f"raw_persistence_canary_ready_score:{condition}" in mod.FIELD_PRINCIPLES


def test_scenario_infra_6462_model_specs_use_cached_sota_and_embedded_tokenizers(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6462-1: model rows use cached SOTA and embedded tokenizers."""

    calls: list[dict[str, Any]] = []
    resolved = mod.build_model_specs(
        cached_pair_func=_cached_pair(_model_paths(tmp_path), calls),
        tokenizer_func=_tokenizer,
    )

    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": None},
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 2)},
    ]
    assert [row["hf_id"] for row in resolved["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert resolved["all_resolved"] is True
    assert resolved["autotokenizer_usage_count"] == 0
    assert all(row["tokenizer_source"] == mod.TOKENIZER_SOURCE for row in resolved["MODEL_SPECS"])


def test_scenario_infra_6462_atomic_write_and_path_receipts(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6462-2 and SCENARIO-INFRA-6462-3: bytes persist before parse."""

    artifact = _artifact(tmp_path, write=True)
    normal_rows = [
        row for row in artifact["per_unit_rows"]["rows"] if row["row_kind"] == "normal"
    ]
    written = tmp_path / mod.RESULT_RELATIVE_PATH.name

    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "success"
    assert artifact["raw_persistence_canary_ready_score"] == 1.0
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["blocked_reason"] == ""
    assert len(normal_rows) == mod.UNIT_COUNT * len(mod.MANDATED_MODEL_IDS) * mod.REPLICATES_PER_UNIT
    assert artifact["raw_text_equality_diagnostic"]["duplicate_raw_text_count"] > 0
    assert artifact["one_event_one_path_one_hash_check"]["passed"] is True
    assert artifact["nonzero_durable_byte_check"]["passed"] is True
    assert artifact["aggregate_row_recomputation"]["matches_reported"] is True
    assert artifact["cpu_fallback_count"] == 0
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    for row in normal_rows:
        raw_path = Path(row["raw_output_path"])
        assert raw_path.is_file()
        assert raw_path.stat().st_size == row["durable_byte_count"]
        assert mod.sha256_file(raw_path) == row["raw_hash"]
        assert row["raw_persisted_before_parse"] is True
        assert row["path_receipt_validation"]["accepted"] is True
        assert row["event_id"] in row["event_path_allocation_receipt"]["event_id"]
        assert row["atomic_write_receipt"]["verified_after_rename"] is True


def test_scenario_infra_6462_attacks_and_blockers_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6462-4: persistence attacks and precondition blockers close."""

    artifact = _artifact(tmp_path)
    attacks = artifact["attack_matrix"]
    attack_rows = [
        row for row in artifact["per_unit_rows"]["rows"] if row["row_kind"] == "attack"
    ]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert {row["attack_id"] for row in attack_rows} == set(mod.ATTACK_IDS)
    assert attacks["all_critical_fail_closed"] is True
    assert attacks["false_accept_count"] == 0

    bad = deepcopy(artifact)
    first = bad["per_unit_rows"]["rows"][0]
    second = next(row for row in bad["per_unit_rows"]["rows"] if row["row_kind"] == "normal" and row is not first)
    second["event_id"] = first["event_id"]
    mod.refresh_terminal_fields(bad)
    assert bad["raw_persistence_canary_ready_score"] == 0.0
    assert "duplicate_event_id" in bad["one_event_one_path_one_hash_check"]["reasons"]

    bad = deepcopy(artifact)
    bad["per_unit_rows"]["rows"][0]["durable_byte_count"] = 0
    mod.refresh_terminal_fields(bad)
    assert "zero_or_missing_durable_bytes" in bad["nonzero_durable_byte_check"]["reasons"]
    assert "nonzero durable bytes check failed" in mod.validate_artifact(bad)

    raw_present_dir = tmp_path / "raw-present"
    (raw_present_dir / "raw_outputs").mkdir(parents=True)
    calls = {"generation": 0}

    def never_generate(**_kwargs: Any) -> dict[str, Any]:
        calls["generation"] += 1
        raise AssertionError("generation must not run after blocked preconditions")

    def raw_path_blocked(**kwargs: Any) -> list[dict[str, Any]]:
        rows = _host_ok(**kwargs)
        rows[-1]["available"] = False
        rows[-1]["detail"] = "raw output path preexisted"
        return rows

    paths = _model_paths(tmp_path / "blocked-models")
    blocked = mod.run(
        date="20260819",
        result_path=tmp_path / "blocked.json",
        data_dir=raw_present_dir,
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        host_preflight_func=raw_path_blocked,
        generation_func=never_generate,
        event_id_func=_fixture_event_id,
        test_exit_codes=_test_exit_codes(),
        duration_s=0.0,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["raw_persistence_canary_ready_score"] == 0.0
    assert "new_output_paths" in blocked["blocked_reason"]
    assert blocked["honest_verdict"].startswith("blocked_")
    assert calls["generation"] == 0
    assert (tmp_path / "blocked.json").is_file()


def test_req_infra_6462_validation_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFRA-6462: defensive validation covers schema and gate edges."""

    artifact = _artifact(tmp_path / "base")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    not_object = tmp_path / "not-object.json"
    not_object.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(not_object) == {}
    missing = tmp_path / "missing.gguf"
    assert mod.sha256_file(missing) is None
    assert mod.model_slug("!!!") == "model"
    assert mod.default_event_id(
        unit_id="u",
        model_hf_id=mod.MANDATED_MODEL_IDS[0],
        replicate_index=0,
        seed=1,
    ).startswith("evt-")
    snapshot = tmp_path / "snapshots" / "rev123" / "model-plain.gguf"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("fixture", encoding="utf-8")
    assert mod._revision_from_path(snapshot) == "rev123"
    assert mod._quantization_from_path(snapshot) == "unknown"
    assert mod.inject_attack_rows([]) == []

    preexisting = tmp_path / "preexisting.txt"
    preexisting.write_text("old", encoding="utf-8")
    assert "target_preexisted" in mod.write_bytes_atomic_verified(
        preexisting,
        b"new",
        write=True,
    )["reasons"]
    assert "zero_byte_raw_output" in mod.write_bytes_atomic_verified(
        tmp_path / "dry-empty.txt",
        b"",
        write=False,
    )["reasons"]
    zero_live = mod.write_bytes_atomic_verified(tmp_path / "zero-live.txt", b"", write=True)
    assert "zero_byte_raw_output" in zero_live["reasons"]

    fsync_path = tmp_path / "fsync-fails.txt"

    def failing_fsync(_fd: int) -> None:
        raise OSError("fixture fsync failure")

    with monkeypatch.context() as mp:
        mp.setattr(mod.os, "fsync", failing_fsync)
        fsync_receipt = mod.write_bytes_atomic_verified(fsync_path, b"bytes", write=True)
    assert any(reason.startswith("file_fsync_failed") for reason in fsync_receipt["reasons"])
    assert any(reason.startswith("directory_fsync_failed") for reason in fsync_receipt["reasons"])

    with monkeypatch.context() as mp:
        mp.setattr(mod.os, "replace", lambda _src, _dst: (_ for _ in ()).throw(OSError("boom")))
        failed_write = mod.write_bytes_atomic_verified(tmp_path / "replace-fails.txt", b"bytes", write=True)
    assert any(reason.startswith("atomic_write_failed") for reason in failed_write["reasons"])

    with monkeypatch.context() as mp:
        mp.setattr(mod, "sha256_file", lambda _path: "sha256:" + "1" * 64)
        mismatch = mod.write_bytes_atomic_verified(tmp_path / "hash-mismatch.txt", b"bytes", write=True)
    assert "sha256_mismatch_after_rename" in mismatch["reasons"]

    assert mod._parse_raw_output(b"\xff")["parse_error"].startswith("unicode_decode")

    empty_specs = mod.build_model_specs(cached_pair_func=lambda **_kwargs: [], tokenizer_func=_tokenizer)
    assert empty_specs["all_resolved"] is False

    missing_specs = mod.build_model_specs(
        cached_pair_func=_cached_pair({model_id: missing for model_id in mod.MANDATED_MODEL_IDS}, []),
        tokenizer_func=_tokenizer,
    )
    assert any("model_path_missing" in reason for reason in missing_specs["blocked_reasons"])

    paths = _model_paths(tmp_path / "tokenizer-fail-models")
    bad_token_specs = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=lambda _path: (False, "fixture tokenizer failure"),
    )
    assert any(
        "embedded_tokenizer_not_loadable" in reason
        for reason in bad_token_specs["blocked_reasons"]
    )
    failed_test_receipt = mod.tests_run_receipt({mod.FULL_PYTEST_COMMAND: 3})
    assert next(
        row for row in failed_test_receipt if row["command"] == mod.FULL_PYTEST_COMMAND
    )["status"] == "failed"

    normal_rows = [row for row in deepcopy(artifact["per_unit_rows"]["rows"]) if row["row_kind"] == "normal"]
    duplicate_path = deepcopy(normal_rows)
    duplicate_path[1]["raw_output_path"] = duplicate_path[0]["raw_output_path"]
    assert "duplicate_raw_path" in mod.one_event_one_path_one_hash_check(duplicate_path)["reasons"]
    duplicate_tuple = deepcopy(normal_rows)
    for key in ("event_id", "raw_output_path", "raw_hash"):
        duplicate_tuple[1][key] = duplicate_tuple[0][key]
    assert "duplicate_event_path_hash_tuple" in mod.one_event_one_path_one_hash_check(
        duplicate_tuple
    )["reasons"]
    binding_mismatch = deepcopy(normal_rows)
    binding_mismatch[0]["event_path_allocation_receipt"]["event_id"] = "wrong"
    binding_mismatch[1]["event_path_allocation_receipt"]["final_path"] = "wrong"
    binding_mismatch[2]["atomic_write_receipt"]["sha256"] = "sha256:" + "2" * 64
    wrong_file = tmp_path / "wrong-file.txt"
    wrong_file.write_text("wrong", encoding="utf-8")
    binding_mismatch[3]["raw_output_path"] = str(wrong_file)
    assert "binding_mismatch" in mod.one_event_one_path_one_hash_check(
        binding_mismatch
    )["reasons"]

    validation_cases = [
        (lambda row: row.pop("status"), "missing required field: status"),
        (lambda row: row.update(MODEL_SPECS=[]), "MODEL_SPECS mandated ids mismatch"),
        (lambda row: row.update(models_used=["bad"]), "models_used must be empty or match mandated ids"),
        (lambda row: row.update(autotokenizer_usage_count=1), "autotokenizer_usage_count must be zero"),
        (lambda row: row.update(inference_substrate="wrong"), "inference_substrate mismatch"),
        (lambda row: row.update(verifier_is_oracle=False), "verifier_is_oracle must be true for exact byte and receipt checks"),
        (lambda row: row["per_unit_rows"].update(row_count=1), "per_unit_rows row_count mismatch"),
        (lambda row: row["per_unit_rows"].update(normal_row_count=1), "normal row count mismatch"),
        (lambda row: row["per_unit_rows"].update(attack_row_count=1), "attack row count mismatch"),
        (lambda row: row["sealed_unit_manifest"].update(unit_count=1), "sealed unit count mismatch"),
        (
            lambda row: row["one_event_one_path_one_hash_check"].update(passed=False),
            "one event/path/hash check failed",
        ),
        (
            lambda row: row["attack_matrix"].update(all_critical_fail_closed=False),
            "attack matrix must fail closed",
        ),
        (lambda row: row["attack_matrix"].update(false_accept_count=1), "ready artifact cannot accept attacks"),
        (
            lambda row: row["aggregate_row_recomputation"].update(matches_reported=False),
            "reported aggregates must recompute from rows",
        ),
        (lambda row: row.update(cpu_fallback_count=1), "cpu_fallback_count must be zero"),
        (
            lambda row: row.update(field_principles={}),
            "missing field_principles entry: status",
        ),
        (
            lambda row: row["field_principles"].pop(
                "raw_persistence_canary_ready_score:nonzero_durable_bytes"
            ),
            "missing readiness field_principles entry: nonzero_durable_bytes",
        ),
        (lambda row: row.update(field_provenance={}), "field_provenance must cover exactly required fields"),
        (lambda row: row.update(honest_verdict="bad prefix"), "honest_verdict lacks required terminal prefix"),
        (lambda row: row.update(reproducibility_checksum="sha256:bad"), "reproducibility_checksum mismatch"),
        (lambda row: row.update(raw_persistence_canary_ready_score=1.0, duration_s=0.0), "raw_persistence_canary_ready_score does not recompute"),
    ]
    for mutate, expected in validation_cases:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {
            "reproducibility_checksum mismatch",
            "raw_persistence_canary_ready_score does not recompute",
        }:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert expected in mod.validate_artifact(bad)

    model_blocked = mod.run(
        date="20260819",
        result_path=tmp_path / "model-blocked.json",
        data_dir=tmp_path / "model-blocked-data",
        cached_pair_func=lambda **_kwargs: [],
        tokenizer_func=_tokenizer,
        host_preflight_func=_host_ok,
        generation_func=_fixture_generation,
        event_id_func=_fixture_event_id,
        test_exit_codes=_test_exit_codes(),
        duration_s=0.0,
        write=False,
    )
    assert model_blocked["status"] == "blocked"
    assert "model_resolution" in model_blocked["blocked_reason"]
    mod.refresh_terminal_fields(model_blocked)
    assert model_blocked["gate_check_summary"].startswith("blocked:")

    with monkeypatch.context() as mp:
        mp.setattr(mod, "validate_artifact", lambda _payload: ["forced schema error"])
        failed = mod.run(
            date="20260819",
            result_path=tmp_path / "failed.json",
            data_dir=tmp_path / "failed-data",
            cached_pair_func=_cached_pair(_model_paths(tmp_path / "failed-models"), []),
            tokenizer_func=_tokenizer,
            host_preflight_func=_host_ok,
            generation_func=_fixture_generation,
            event_id_func=_fixture_event_id,
            test_exit_codes=_test_exit_codes(),
            duration_s=125.0,
            write=False,
        )
    assert failed["status"] == "failed_schema"
    assert failed["honest_verdict"].startswith("complete_failed_schema:")
