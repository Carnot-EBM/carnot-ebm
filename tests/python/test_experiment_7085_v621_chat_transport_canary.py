"""Tests for the bounded three-family GGUF chat transport canary.

Spec refs: REQ-INFRA-7085, REQ-VERIFY-7085, and their scenarios.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7085_v621_chat_transport_canary as mod


def _specs(tmp_path: Path) -> list[dict[str, Any]]:
    rows = []
    for model_id in mod.REQUIRED_MODEL_IDS:
        path = tmp_path / f"{model_id.rsplit('/', 1)[-1]}.gguf"
        path.write_bytes(model_id.encode("utf-8"))
        rows.append(
            {
                "name": model_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
                "hf_id": model_id,
                "model_path": str(path),
                "gpu_indices": [0, 1],
                "preferred_quant": mod.PREFERRED_QUANT,
                "tokenizer_source": "embedded_gguf",
                "remote_allowed": False,
                "headline_eligible": True,
                "resolution_method": "cached_sota_pair",
            }
        )
    return rows


def _units() -> list[dict[str, Any]]:
    return [
        {
            "unit_id": f"unit-{index}",
            "source_group_id": f"source-{index % 6}",
            "numbers": [1, 2, 3, 4, 25, 50],
            "target": 75 + index,
        }
        for index in range(12)
    ]


def _valid_raw_rows() -> list[dict[str, Any]]:
    rows = []
    raw_text = '{"operand_pair":[1,2],"operator":"+"}'
    for model_id in mod.REQUIRED_MODEL_IDS:
        for index in range(mod.UNITS_PER_MODEL):
            rows.append(
                {
                    "raw_key": f"{model_id}|unit-{index}|{mod.GENERATION_SEED}",
                    "model_id": model_id,
                    "unit_id": f"unit-{index}",
                    "seed": mod.GENERATION_SEED,
                    "execution_index": len(rows),
                    "transport_method": "create_chat_completion",
                    "chat_format": "embedded_gguf_template",
                    "chat_template_present": True,
                    "chat_template_hash": mod.sha256_text("template"),
                    "role_messages": mod.build_role_messages(
                        mod.build_proposal_prompt(_units()[index])
                    ),
                    "rendered_prompt_hash": None,
                    "rendered_prompt_hash_available": False,
                    "stop_config": deepcopy(mod.GENERATION_CONFIG["stop"]),
                    "raw_text": raw_text,
                    "raw_bytes_hex": raw_text.encode("utf-8").hex(),
                    "raw_output_hash": mod.sha256_text(raw_text),
                    "prompt_tokens": 40,
                    "completion_tokens": 14,
                    "finish_reason": "stop",
                    "timings": {"duration_s": 0.25, "backend": {}},
                    "terminal_state": "complete",
                    "exception_type": None,
                    "exception_message": None,
                    "raw_persisted_before_parse": True,
                    "parsed_at_write_time": False,
                    "labeled_at_write_time": False,
                }
            )
    return rows


def _valid_evidence(tmp_path: Path) -> dict[str, Any]:
    specs = _specs(tmp_path)
    return {
        "model_identity_rows": [
            {
                "model_id": row["hf_id"],
                "identity_matches": True,
                "tokenizer_source": "embedded_gguf",
                "chat_template_present": True,
                "chat_template_hash": mod.sha256_text("template"),
            }
            for row in specs
        ],
        "model_execution_rows": [
            {
                "model_id": model_id,
                "terminal_state": "complete",
                "raw_row_count": mod.UNITS_PER_MODEL,
                "offloaded_layers": 1,
                "used_both_gpus": True,
                "cleanup_passed": True,
                "model_load_count": 1,
                "duration_s": 1.0,
            }
            for model_id in mod.REQUIRED_MODEL_IDS
        ],
        "checkpoint_rows": [
            {
                "model_id": model_id,
                "row_count": mod.UNITS_PER_MODEL,
                "sha256": "sha256:" + "1" * 64,
                "manifest_hash": "sha256:" + "2" * 64,
            }
            for model_id in mod.REQUIRED_MODEL_IDS
        ],
        "gpu_lease_rows": [
            {
                "model_id": model_id,
                "device_uuid": f"GPU-{device}",
                "lease_id": f"{model_id}-{device}",
                "owner_preserved": True,
                "phase_history": list(mod.lease_api.COMPLETE_PHASE_SEQUENCE),
                "released": True,
                "lease_lost": False,
                "signals_sent": [],
            }
            for model_id in mod.REQUIRED_MODEL_IDS
            for device in (0, 1)
        ],
        "vram_release_rows": [
            {"model_id": model_id, "passed": True, "after_rows": []}
            for model_id in mod.REQUIRED_MODEL_IDS
        ],
        "runner_receipt": {
            "backend": "llama_cpp.Llama",
            "cuda_offload": True,
            "transport_method": "create_chat_completion",
        },
        "signals_sent": [],
        "stage_gpu_telemetry_rows": [],
        "task_gpu_telemetry_rows": [],
        "peak_vram_by_device": {"GPU-0": 1, "GPU-1": 1},
    }


def _preconditions(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "all_passed": True,
        "checks": [mod.gate_row("preflight", True, True, True)],
        "model_identity_rows": deepcopy(evidence["model_identity_rows"]),
        "runner_build_rows": [deepcopy(evidence["runner_receipt"])],
        "upstream_gate_rows": [],
        "gpu_topology": {"devices": []},
    }


def test_req_infra_7085_resolves_exact_cached_roster(tmp_path: Path) -> None:
    """REQ-INFRA-7085 rejects cache and roster substitutions."""

    paths = {
        model_id: str(tmp_path / f"{index}.gguf")
        for index, model_id in enumerate(mod.REQUIRED_MODEL_IDS)
    }
    for path in paths.values():
        Path(path).write_bytes(b"gguf")

    def pair(**_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {"hf_id": mod.REQUIRED_MODEL_IDS[0], "model_path": paths[mod.REQUIRED_MODEL_IDS[0]]},
            {"hf_id": mod.REQUIRED_MODEL_IDS[2], "model_path": paths[mod.REQUIRED_MODEL_IDS[2]]},
        ]

    specs = mod.resolve_model_specs(
        cached_pair_func=pair,
        resolver=lambda model_id, _quant: paths.get(model_id),
    )
    assert [row["hf_id"] for row in specs] == list(mod.REQUIRED_MODEL_IDS)
    assert mod.model_spec_errors(specs) == []
    assert "cached_sota_pair" in specs[0]["resolution_method"]

    broken = deepcopy(specs)
    broken[1]["model_path"] = ""
    broken.reverse()
    errors = mod.model_spec_errors(broken)
    assert "model_ids_mismatch" in errors
    assert any(error.startswith("model_path_missing:") for error in errors)


def test_req_infra_7085_freezes_matched_randomized_schedule() -> None:
    """REQ-INFRA-7085 freezes eight matched rows and one model order."""

    selected = mod.select_representative_units(_units(), count=mod.UNITS_PER_MODEL)
    schedule = mod.build_schedule(
        [
            {"hf_id": model_id, "model_path": f"/{model_id}.gguf"}
            for model_id in mod.REQUIRED_MODEL_IDS
        ],
        selected,
    )
    assert len(selected) == 8
    assert len({row["source_group_id"] for row in selected}) == 6
    assert len(schedule) == 24
    assert {row["seed"] for row in schedule} == {mod.GENERATION_SEED}
    assert {row["generation_config"]["completion_budget_tokens"] for row in schedule} == {192}
    assert mod.schedule_errors(schedule, [row["unit_id"] for row in selected]) == []
    assert mod.randomized_model_order() == mod.randomized_model_order()


def test_req_infra_7085_wrong_roles_and_stop_mismatch_fail() -> None:
    """SCENARIO-INFRA-7085-TEMPLATE-AND-ROLES rejects contract drift."""

    selected = mod.select_representative_units(_units(), count=8)
    specs = [
        {"hf_id": model_id, "model_path": f"/{model_id}.gguf"}
        for model_id in mod.REQUIRED_MODEL_IDS
    ]
    schedule = mod.build_schedule(specs, selected)
    schedule[0]["role_messages"] = [{"role": "assistant", "content": "bad"}]
    schedule[1]["generation_config"]["stop"] = ["wrong"]
    errors = mod.schedule_errors(schedule, [row["unit_id"] for row in selected])
    assert "role_message_mismatch" in errors
    assert "stop_config_mismatch" in errors


def test_req_infra_7085_worker_uses_embedded_chat_template() -> None:
    """SCENARIO-INFRA-7085-TEMPLATE-AND-ROLES bans raw completion transport."""

    class FakeLlama:
        metadata = {"tokenizer.chat_template": "{{ messages }}", "general.name": "test"}
        chat_format = "chatml"

        def create_completion(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("raw completion must never be called")

        def create_chat_completion(self, *, messages: Any, **kwargs: Any) -> dict[str, Any]:
            assert [row["role"] for row in messages] == ["system", "user"]
            assert kwargs["max_tokens"] == 192
            assert kwargs["stop"] == []
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"operand_pair":[1,2],"operator":"+"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 13},
                "timings": {"predicted_ms": 4},
            }

    row = mod.build_schedule(
        [{"hf_id": mod.REQUIRED_MODEL_IDS[0], "model_path": "/tmp/model.gguf"}],
        _units()[:1],
    )[0]
    result = mod.worker_generate_one(row, llama_instance=FakeLlama(), clock=iter([1, 5]).__next__)
    assert result["terminal_state"] == "complete"
    assert result["transport_method"] == "create_chat_completion"
    assert result["chat_template_present"] is True
    assert result["chat_template_hash"] == mod.sha256_text("{{ messages }}")
    assert bytes.fromhex(result["raw_bytes_hex"]).decode() == result["raw_text"]
    source = inspect.getsource(mod.worker_generate_one)
    assert ".create_completion(" not in source
    assert ".create_chat_completion(" in source


def test_req_infra_7085_missing_or_empty_template_fails_before_call() -> None:
    """SCENARIO-INFRA-7085-TEMPLATE-AND-ROLES requires a non-empty template."""

    class NoTemplate:
        metadata = {"tokenizer.chat_template": ""}
        called = False

        def create_chat_completion(self, **_kwargs: Any) -> Any:
            self.called = True
            raise AssertionError("must not call")

    backend = NoTemplate()
    schedule = mod.build_schedule(
        [{"hf_id": mod.REQUIRED_MODEL_IDS[0], "model_path": "/tmp/model.gguf"}],
        _units()[:1],
    )[0]
    row = mod.worker_generate_one(schedule, llama_instance=backend, clock=iter([1, 2]).__next__)
    assert row["terminal_state"] == "failed"
    assert row["exception_message"] == "missing_or_empty_embedded_chat_template"
    assert backend.called is False


def test_req_infra_7085_checkpoint_resume_is_raw_only(tmp_path: Path) -> None:
    """SCENARIO-INFRA-7085-RAW-FIRST-RESUME reuses immutable raw rows."""

    class FakeLlama:
        metadata = {"tokenizer.chat_template": "template"}
        calls = 0

        def create_chat_completion(self, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            return {
                "choices": [
                    {
                        "message": {"content": '{"operand_pair":[1,2],"operator":"+"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 10},
            }

        def close(self) -> None:
            return None

    schedule = mod.build_schedule(
        [{"hf_id": mod.REQUIRED_MODEL_IDS[0], "model_path": "/tmp/model.gguf"}],
        _units()[:2],
    )
    backend = FakeLlama()
    payload = {
        "model_id": mod.REQUIRED_MODEL_IDS[0],
        "model_path": "/tmp/model.gguf",
        "schedule_rows": schedule,
        "raw_path": str(tmp_path / "raw.jsonl"),
        "checkpoint_path": str(tmp_path / "checkpoint.json"),
        "manifest_hash": "manifest",
    }
    first = mod.worker_run_schedule(payload, llama_factory=lambda **_kwargs: backend)
    before = (tmp_path / "raw.jsonl").read_bytes()
    second = mod.worker_run_schedule(payload, llama_factory=lambda **_kwargs: backend)
    assert first["row_count"] == 2
    assert second["checkpoint_receipts"] == []
    assert backend.calls == 2
    assert (tmp_path / "raw.jsonl").read_bytes() == before
    with pytest.raises(ValueError, match="checkpoint_manifest_mismatch"):
        mod.load_checkpoint(tmp_path / "checkpoint.json", "changed")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda row: row.update(
                raw_text="", raw_bytes_hex="", raw_output_hash=mod.sha256_text("")
            ),
            "empty_output",
        ),
        (lambda row: row.update(completion_tokens=0), "zero_token_output"),
        (
            lambda row: row.update(
                raw_text="<|im_end|>{}",
                raw_output_hash=mod.sha256_text("<|im_end|>{}"),
                raw_bytes_hex=b"<|im_end|>{}".hex(),
            ),
            "leaked_control_token",
        ),
        (lambda row: row.update(finish_reason="length"), "length_limited_output"),
        (lambda row: row.update(stop_config=["wrong"]), "stop_config_mismatch"),
    ],
)
def test_req_verify_7085_output_failure_matrix(
    tmp_path: Path, mutation: Any, expected: str
) -> None:
    """SCENARIO-VERIFY-7085-OUTPUT-FAILURES rejects each transport defect."""

    rows = _valid_raw_rows()
    mutation(rows[0])
    assert expected in mod.transport_errors(rows, _valid_evidence(tmp_path))


def test_req_verify_7085_parser_is_declared_schema_only() -> None:
    """SCENARIO-VERIFY-7085-PARSE-AND-EXACT-SEPARATION uses a closed schema."""

    assert mod.parse_entrance('{"operand_pair":[1,2],"operator":"+"}') == {
        "operand_pair": [1, 2],
        "operator": "+",
    }
    assert mod.parse_entrance('prefix {"operand_pair":[1,2],"operator":"+"}') is None
    assert mod.parse_entrance('{"operand_pair":[1,2],"operator":"+","extra":1}') is None
    assert mod.parse_entrance('{"operand_pair":[1,"2"],"operator":"+"}') is None
    assert mod.parse_entrance('{"operand_pair":[1,2],"operator":"%"}') is None


def test_req_verify_7085_raw_parse_and_exact_rows_are_separate() -> None:
    """SCENARIO-VERIFY-7085-PARSE-AND-EXACT-SEPARATION preserves raw bytes."""

    raw = _valid_raw_rows()[:1]
    before = deepcopy(raw)
    views = mod.label_and_project_rows(
        raw,
        [
            {
                "unit_id": "unit-0",
                "operand_pair": [1, 2],
                "operator": "+",
                "entrance_id": "unit-0:1:2:add",
                "reachable": True,
            }
        ],
    )
    assert raw == before
    assert "legal" not in views["raw_output_rows"][0]
    assert "reachable" not in views["parse_rows"][0]
    assert "raw_text" not in views["exact_label_rows"][0]
    assert views["exact_label_rows"][0]["legal"] is True


def test_req_verify_7085_transport_rejects_identity_lease_cleanup_and_rates(
    tmp_path: Path,
) -> None:
    """REQ-INFRA-7085 lease loss and REQ-VERIFY-7085 rates fail closed."""

    rows = _valid_raw_rows()
    evidence = _valid_evidence(tmp_path)
    assert mod.transport_errors(rows, evidence) == []
    evidence["model_identity_rows"][0]["identity_matches"] = False
    evidence["gpu_lease_rows"][0]["lease_lost"] = True
    evidence["vram_release_rows"][0]["passed"] = False
    errors = mod.transport_errors(rows, evidence)
    assert "model_identity_incomplete" in errors
    assert "lease_identity_or_release_incomplete" in errors
    assert "vram_release_incomplete" in errors


def test_req_verify_7085_builds_positive_artifact_and_recomputes_mutations(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7085-AGGREGATE-RECOMPUTATION checks all projections."""

    specs = _specs(tmp_path)
    raw_rows = _valid_raw_rows()
    evidence = _valid_evidence(tmp_path)
    artifact = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=3.0,
        model_specs=specs,
        preconditions=_preconditions(evidence),
        raw_rows=raw_rows,
        entrance_rows=[],
        evidence=evidence,
        model_order_rows=mod.model_order_rows(),
    )
    assert artifact["chat_transport_ready_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["generation_invoked"] is True
    assert artifact["total_model_count"] == 3
    assert artifact["model_load_count_by_stage"] == {"generation": 3}
    assert artifact["parseable_rate_by_model"] == {
        model_id: 1.0 for model_id in mod.REQUIRED_MODEL_IDS
    }
    assert mod.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["raw_output_rows"][0]["completion_tokens"] = 0
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    errors = mod.validate_artifact(changed)
    assert "token_count_projection_mismatch" in errors
    assert "transport_readiness_mismatch" in errors


def test_req_verify_7085_blocked_artifact_is_schema_complete(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7085-BLOCKED-DIAGNOSTIC preserves the failed gate."""

    preconditions = {
        "all_passed": False,
        "checks": [mod.gate_row("all_three_cached_gguf_files", True, False, False)],
        "model_identity_rows": [],
    }
    artifact = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=0.1,
        model_specs=_specs(tmp_path),
        preconditions=preconditions,
    )
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["generation_invoked"] is False
    assert artifact["verdict_class"] == "blocked"
    assert artifact["signals_sent"] == []
    assert artifact["gate_check_summary"]["failed_check"] == "all_three_cached_gguf_files"
    assert mod.validate_artifact(artifact) == []


def test_req_infra_7085_preconditions_reject_cache_identity_and_foreign_owner(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-7085-CACHE-AND-IDENTITY blocks before generation."""

    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"entrance_fixture_ready_score": 1}), encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"gpu_lease_cold_audit_ready_score": 1}), encoding="utf-8")
    specs = _specs(tmp_path)
    specs[0]["model_path"] = ""
    devices = [
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "name": "NVIDIA GeForce RTX 3090",
            "utilization_gpu_pct": 0,
        }
        for index in (0, 1)
    ]
    result = mod.collect_preconditions(
        lease_audit_path=audit,
        expected_lease_audit_hash=mod.sha256_file(audit),
        lease_audit_validator=lambda _value: [],
        fixture_path=fixture,
        expected_fixture_hash=mod.sha256_file(fixture),
        fixture_validator=lambda _value, **_kwargs: True,
        model_specs=specs,
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        gpu_probe=lambda: {"query_ok": True, "devices": devices, "processes": [{"pid": 99}]},
        llama_probe=lambda: {"importable": True, "gpu_offload": True},
        lease_probe=lambda rows: [
            {"device_uuid": row["uuid"], "classification": "live_foreign"} for row in rows
        ],
        stop_authority_probe=lambda: {"passed": True, "observed": "clean"},
        identity_probe=lambda row: {
            "model_id": row["hf_id"],
            "identity_matches": row["hf_id"] != mod.REQUIRED_MODEL_IDS[1],
            "tokenizer_source": "embedded_gguf",
            "chat_template_present": True,
        },
    )
    failed = {row["check"] for row in result["checks"] if row["passed"] is not True}
    assert {
        "exact_model_specs",
        "all_three_cached_gguf_files",
        "model_identity_rows",
        "unattributed_gpu_processes",
        "owned_gpu_leases_available",
    } <= failed
    assert result["signals_sent"] == []


def test_req_verify_7085_artifact_validator_rejects_schema_class_and_checksum(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7085-AGGREGATE-RECOMPUTATION fails closed on metadata drift."""

    base = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=0.1,
        model_specs=_specs(tmp_path),
        preconditions={
            "all_passed": False,
            "checks": [mod.gate_row("cache", True, False, False)],
            "model_identity_rows": [],
        },
    )
    mutations = [
        (lambda row: row.pop("rows"), "missing_field:rows"),
        (lambda row: row["field_principles"].pop("rows"), "field_principles_mismatch"),
        (lambda row: row.update(schema="wrong"), "schema_mismatch"),
        (lambda row: row.update(inference_substrate="remote"), "inference_substrate_mismatch"),
        (lambda row: row.update(verifier_is_oracle=True), "verifier_is_oracle_mismatch"),
        (lambda row: row.update(chat_transport_ready_score=False), "readiness_not_bare_int"),
        (
            lambda row: row.update(reproducibility_checksum="sha256:" + "0" * 64),
            "reproducibility_checksum_mismatch",
        ),
    ]
    for mutate, expected in mutations:
        changed = deepcopy(base)
        mutate(changed)
        if expected != "reproducibility_checksum_mismatch":
            changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        assert expected in mod.validate_artifact(changed)


def test_req_infra_7085_model_schedule_and_checkpoint_defenses(tmp_path: Path) -> None:
    """REQ-INFRA-7085 covers the remaining roster, schedule, and resume defenses."""

    specs = _specs(tmp_path)
    broken = deepcopy(specs)
    broken[0].update(
        model_path="/tmp/mmproj.gguf",
        gpu_indices=[1],
        tokenizer_source="remote",
        remote_allowed=True,
        headline_eligible=False,
    )
    errors = mod.model_spec_errors(broken)
    assert {
        f"model_path_not_primary_gguf:{mod.REQUIRED_MODEL_IDS[0]}",
        f"gpu_indices_mismatch:{mod.REQUIRED_MODEL_IDS[0]}",
        f"tokenizer_source_mismatch:{mod.REQUIRED_MODEL_IDS[0]}",
        f"headline_policy_mismatch:{mod.REQUIRED_MODEL_IDS[0]}",
    } <= set(errors)

    identities = _valid_evidence(tmp_path)["model_identity_rows"]
    identities[0]["tokenizer_source"] = "remote"
    identities[1]["chat_template_hash"] = None
    identity_errors = mod.model_identity_errors(specs, identities)
    assert f"model_tokenizer_mismatch:{mod.REQUIRED_MODEL_IDS[0]}" in identity_errors
    assert f"model_chat_template_missing:{mod.REQUIRED_MODEL_IDS[1]}" in identity_errors

    with pytest.raises(ValueError, match="representative_unit_count"):
        mod.select_representative_units(_units()[:1], count=2)

    units = mod.select_representative_units(_units(), count=8)
    schedule = mod.build_schedule(specs, units)
    schedule.pop()
    schedule[0]["prompt_hash"] = "bad"
    schedule[1]["role_messages_hash"] = "bad"
    schedule[2]["seed"] = -1
    one_unit = schedule[3]["unit_id"]
    for row in schedule:
        if row["unit_id"] == one_unit and row["model_id"] == mod.REQUIRED_MODEL_IDS[1]:
            row["prompt"] += " drift"
    schedule_errors = mod.schedule_errors(schedule, [row["unit_id"] for row in units])
    assert "schedule_key_set_mismatch" in schedule_errors
    assert "prompt_hash_mismatch" in schedule_errors
    assert "role_message_hash_mismatch" in schedule_errors
    assert "generation_seed_mismatch" in schedule_errors
    assert f"cross_model_prompt_mismatch:{one_unit}" in schedule_errors

    checkpoint = tmp_path / "checkpoint.json"
    row = {"raw_key": "one", "raw_text": "x"}
    mod.checkpoint_raw_row(checkpoint, "manifest", row)
    assert mod.checkpoint_raw_row(checkpoint, "manifest", row)["written"] is False
    with pytest.raises(ValueError, match="checkpoint_row_mismatch"):
        mod.checkpoint_raw_row(checkpoint, "manifest", {"raw_key": "one", "raw_text": "y"})
    checkpoint.write_text(
        json.dumps({"manifest_hash": "manifest", "rows": [row, row]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checkpoint_duplicate_raw_key"):
        mod.load_checkpoint(checkpoint, "manifest")


def test_req_infra_7085_owned_worker_and_render_receipts() -> None:
    """REQ-INFRA-7085 records rendered prompts and closes an owned backend."""

    assert mod.render_chat_prompt_hash("", []) is None
    assert mod.render_chat_prompt_hash("{% invalid", []) is None

    class Model:
        def token_get_text(self, token: int) -> str:
            return "<eos>" if token == 2 else "<bos>"

    class FakeLlama:
        metadata = {"tokenizer.chat_template.default": "{{ bos_token }}{{ messages }}"}
        chat_format = "chat_template.default"
        _model = Model()
        closed = False

        def token_eos(self) -> int:
            return 2

        def token_bos(self) -> int:
            return 1

        def create_chat_completion(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {"content": '{"operand_pair":[1,2],"operator":"+"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 8},
            }

        def close(self) -> None:
            self.closed = True

    backend = FakeLlama()
    schedule = mod.build_schedule(
        [{"hf_id": mod.REQUIRED_MODEL_IDS[0], "model_path": "/tmp/model.gguf"}],
        _units()[:1],
    )[0]
    row = mod.worker_generate_one(
        schedule,
        llama_factory=lambda **_kwargs: backend,
        clock=iter([1, 3]).__next__,
    )
    assert row["model_close_called"] is True
    assert row["rendered_prompt_hash_available"] is True
    assert backend.closed is True


def test_req_verify_7085_all_transport_reducers_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-7085 recomputes raw, template, execution, and runner failures."""

    rows = _valid_raw_rows()
    evidence = _valid_evidence(tmp_path)
    rows.pop()
    rows[0].update(terminal_state="failed", transport_method="create_completion")
    rows[1].update(chat_template_present=False)
    rows[2]["role_messages"] = None
    evidence["model_execution_rows"][0]["model_load_count"] = 2
    evidence["checkpoint_rows"][0]["sha256"] = "bad"
    evidence["runner_receipt"] = {}
    evidence["signals_sent"] = ["SIGTERM"]
    errors = mod.transport_errors(rows, evidence)
    assert {
        "raw_row_count_or_duplicate_mismatch",
        "raw_terminal_or_hash_mismatch",
        "raw_completion_transport_detected",
        "template_receipt_invalid",
        "role_message_mismatch",
        "per_model_parseability_below_threshold",
        "model_execution_incomplete",
        "checkpoint_incomplete",
        "runner_receipt_invalid",
        "signals_sent_nonempty",
    } <= set(errors)

    role_shapes = [
        {},
        {"role_messages": [{"role": "bad", "content": "x"}, {}]},
        {"role_messages": [{"role": "system", "content": mod.SYSTEM_MESSAGE}, []]},
        {
            "role_messages": [
                {"role": "system", "content": mod.SYSTEM_MESSAGE},
                {"role": "user", "content": 3},
            ]
        },
        {
            "prompt": "expected",
            "role_messages": [
                {"role": "system", "content": mod.SYSTEM_MESSAGE},
                {"role": "user", "content": "wrong"},
            ],
        },
        {
            "role_messages_hash": "bad",
            "role_messages": [
                {"role": "system", "content": mod.SYSTEM_MESSAGE},
                {"role": "user", "content": "x"},
            ],
        },
    ]
    assert all(mod._role_messages_match(row) is False for row in role_shapes)


def test_req_verify_7085_validator_rejects_all_aggregate_drift(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7085-AGGREGATE-RECOMPUTATION covers terminal metadata."""

    evidence = _valid_evidence(tmp_path)
    positive = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=1,
        model_specs=_specs(tmp_path),
        preconditions=_preconditions(evidence),
        raw_rows=_valid_raw_rows(),
        evidence=evidence,
        model_order_rows=mod.model_order_rows(),
    )
    mutations = [
        (lambda row: row.update(run_date="wrong"), "run_date_mismatch"),
        (lambda row: row["model_specs"].pop(), "model_specs_projection_mismatch"),
        (lambda row: row["exact_label_rows"].pop(), "exact_label_projection_mismatch"),
        (lambda row: row.update(all_models_real=False), "all_models_real_mismatch"),
        (lambda row: row.update(total_model_count=2), "total_model_count_mismatch"),
        (lambda row: row.update(generation_invoked=False), "generation_invoked_mismatch"),
        (
            lambda row: row.update(inference_substrate_class="blocked_no_run"),
            "inference_substrate_class_mismatch",
        ),
        (lambda row: row.update(honest_verdict="bad"), "honest_verdict_prefix_mismatch"),
        (lambda row: row.update(verdict_class="null"), "positive_verdict_mismatch"),
        (
            lambda row: row["empty_output_rate_by_model"].update({mod.REQUIRED_MODEL_IDS[0]: 1.0}),
            "empty_output_rate_by_model_mismatch",
        ),
    ]
    for mutate, expected in mutations:
        changed = deepcopy(positive)
        mutate(changed)
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        assert expected in mod.validate_artifact(changed)

    partial = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=1,
        model_specs=_specs(tmp_path),
        preconditions=_preconditions(evidence),
        raw_rows=_valid_raw_rows()[:-1],
        evidence=evidence,
    )
    assert partial["verdict_class"] == "partial"
    partial["verdict_class"] = "null"
    partial["honest_verdict"] = "null: wrong class"
    partial["reproducibility_checksum"] = mod.artifact_checksum(partial)
    assert "failed_transport_verdict_mismatch" in mod.validate_artifact(partial)

    null_rows = _valid_raw_rows()
    null_rows[0]["completion_tokens"] = 0
    null_artifact = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=1,
        model_specs=_specs(tmp_path),
        preconditions=_preconditions(evidence),
        raw_rows=null_rows,
        evidence=evidence,
    )
    assert null_artifact["verdict_class"] == "null"

    blocked = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=1,
        model_specs=_specs(tmp_path),
        preconditions={"all_passed": False, "checks": []},
    )
    blocked["verdict_class"] = "partial"
    blocked["honest_verdict"] = "partial: wrong blocked class"
    blocked["gate_check_summary"] = {}
    blocked["reproducibility_checksum"] = mod.artifact_checksum(blocked)
    blocked_errors = mod.validate_artifact(blocked)
    assert "blocked_verdict_mismatch" in blocked_errors
    assert "blocked_gate_summary_incomplete" in blocked_errors


def test_req_infra_7085_precondition_read_failures_and_write_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFRA-7085-CACHE-AND-IDENTITY retains read and write errors."""

    monkeypatch.setattr(mod.tempfile, "mkstemp", lambda **_kwargs: (_ for _ in ()).throw(OSError()))
    assert mod._writable(tmp_path / "x") is False
    result = mod.collect_preconditions(
        lease_audit_path=tmp_path / "missing-audit",
        expected_lease_audit_hash="missing",
        fixture_path=tmp_path / "missing-fixture",
        expected_fixture_hash="missing",
        model_specs=[],
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        gpu_probe=lambda: {"query_ok": False, "devices": [], "processes": []},
        llama_probe=lambda: {"importable": False, "gpu_offload": False},
        lease_probe=lambda _rows: [],
        stop_authority_probe=lambda: {"passed": False, "observed": "bad"},
        identity_probe=lambda _row: {},
    )
    assert "read_error" in result["lease_audit"]
    assert "read_error" in result["fixture"]
    assert result["all_passed"] is False


def test_req_infra_7085_phase_projection_recomputes_leases_and_peaks(tmp_path: Path) -> None:
    """SCENARIO-INFRA-7085-LEASE-AND-CLEANUP projects phase and task telemetry."""

    owner = {"pid": 1, "pid_start_ticks": 2}
    history = [{"phase": phase} for phase in mod.lease_api.COMPLETE_PHASE_SEQUENCE]
    phase = {
        "model_id": mod.REQUIRED_MODEL_IDS[0],
        "terminal_state": "complete",
        "raw_rows": [{}] * 8,
        "offloaded_layers": 4,
        "total_layers": 5,
        "used_both_gpus": True,
        "cleanup": {"passed": True, "signals_sent": []},
        "model_load_count": 1,
        "duration_s": 2,
        "checkpoint_path": str(tmp_path / "c"),
        "checkpoint_sha256": "sha256:" + "1" * 64,
        "manifest_hash": "sha256:" + "2" * 64,
        "vram_release": {"passed": True},
        "gpu_sample_rows": [{"pid": 1}],
        "task_gpu_samples": [
            {"devices": [{"uuid": "GPU-0", "memory_used_mb": 100}]},
            {"devices": [{"uuid": "GPU-0", "memory_used_mb": 200}]},
        ],
        "gpu_lease_rows": [
            {
                "device_uuid": "GPU-0",
                "lease_id": "lease",
                "journal_after_acquisition": {"lease_id": "lease", "owner": owner},
                "journal_after_release": {
                    "lease_id": "lease",
                    "owner": owner,
                    "released": True,
                    "phase_history": history,
                },
                "release_receipt": {"lease_id": "lease", "released": True},
                "signals_sent": [],
            }
        ],
    }
    preconditions = {
        "model_identity_rows": [],
        "runner_receipt": {"backend": "llama_cpp.Llama"},
    }
    evidence = mod._phase_evidence([phase], preconditions)
    assert evidence["gpu_lease_rows"][0]["owner_preserved"] is True
    assert evidence["gpu_lease_rows"][0]["lease_lost"] is False
    assert evidence["peak_vram_by_device"] == {"GPU-0": 200}


def test_req_infra_7085_listener_ownership_receipt_is_latched() -> None:
    """SCENARIO-INFRA-7085-LEASE-AND-CLEANUP retains an observed owned listener."""

    ready = {"pid": 41, "pid_start_ticks": 73, "port": 9000}
    assert mod.latch_ready_owned(False, ready, 41, 73, 9000, [41]) is True
    assert mod.latch_ready_owned(True, ready, 41, 73, 9000, []) is True
    assert mod.latch_ready_owned(False, ready, 41, 73, 9001, [41]) is False
