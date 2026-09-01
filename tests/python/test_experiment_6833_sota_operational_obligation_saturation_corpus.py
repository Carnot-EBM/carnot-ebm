"""REQ-CONSTRAINT-6833 live saturation corpus contract tests."""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carnot import experiment_6833_sota_operational_obligation_saturation_corpus as exp
from carnot import experiment_6832_operational_obligation_saturation_fixture as fixture_api
from carnot import gpu_lease_phase_journal as lease_api


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    """Use the frozen fixture because its exact checks are the scoring authority."""

    return json.loads((exp.REPO_ROOT / exp.FIXTURE_PATH).read_bytes())


def _metadata() -> dict[str, Any]:
    return {
        "architecture": "test",
        "quantization": "Q4_K_M",
        "is_language_model": True,
        "tokenizer_metadata": {
            "model": "test-tokenizer",
            "pre": "test",
            "token_count": 32000,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "padding_token_id": 0,
            "chat_template_present": True,
        },
        "chat_template": "{{ messages }}",
    }


def _models() -> list[dict[str, Any]]:
    return [
        {
            **deepcopy(planned),
            "revision": f"revision-{index}",
            "model_path": f"/models/{planned['filename']}",
            "model_sha256": planned["expected_sha256"],
            "model_size_bytes": 20_000_000_000,
            "gguf_metadata": _metadata(),
            "decode_settings": deepcopy(exp.DECODE_SETTINGS),
        }
        for index, planned in enumerate(exp.PLANNED_MODELS)
    ]


def _receipt(model: dict[str, Any], index: int, purpose: str = "corpus") -> dict[str, Any]:
    return {
        "purpose": purpose,
        "model_id": model["hub_id"],
        "model_sha256": model["model_sha256"],
        "session_id": f"session-{purpose}-{index}",
        "command": ["llama-server", "--model", model["model_path"]],
        "pid": 9000 + index,
        "process_start_time": f"start-{index}",
        "pid_start_ticks": 100 + index,
        "port": 18000 + index,
        "physical_gpu_uuid": "GPU-test",
        "visible_devices": [0],
        "first_token_b64": base64.b64encode(b"{").decode("ascii"),
        "final_token_b64": base64.b64encode(b"}").decode("ascii"),
        "lease_owned": True,
        "lease_released": True,
        "cuda_offload": True,
        "teardown_complete": True,
        "process_absent_after_exit": True,
        "authentic": True,
    }


def _row(
    scenario: dict[str, Any],
    arm: str,
    model: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    prompt = scenario["prompts"][arm].encode()
    raw = fixture_api.canonical_bytes({"selected_action_ids": scenario["legal_action_ids"]})
    return exp.build_scored_row(
        scenario=scenario,
        arm=arm,
        prompt=prompt,
        raw_output=raw,
        model=model,
        process_receipt=receipt,
        prompt_tokens=100,
        generated_tokens=10,
        latency_s=0.25,
        checker_sha256="sha256:" + "c" * 64,
    )


def test_req_6833_spec_exists_before_implementation() -> None:
    """REQ-CONSTRAINT-6833 exists before the producer implementation."""

    text = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "## REQ-CONSTRAINT-6833:" in text
    assert "### SCENARIO-CONSTRAINT-6833-PREFLIGHT:" in text
    assert "### SCENARIO-CONSTRAINT-6833-READINESS:" in text


def test_scenario_6833_manifest_has_900_unique_equal_budget_rows(fixture: dict) -> None:
    """SCENARIO-CONSTRAINT-6833-BUDGET freezes all identities and equal settings."""

    manifest = exp.build_manifest(fixture)
    assert manifest["expected_row_count"] == 900
    assert len(manifest["expected_row_ids"]) == 900
    assert len(set(manifest["expected_row_ids"])) == 900
    assert manifest["arms"] == list(exp.ARMS)
    assert manifest["decode_settings"] == exp.DECODE_SETTINGS
    assert exp.manifest_checksum(manifest) == manifest["manifest_sha256"]


def test_scenario_6833_model_dispatch_captures_tokenizer_and_template(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6833-DISPATCH requires exact files and embedded metadata."""

    calls: list[dict[str, Any]] = []
    paths: dict[str, str] = {}
    for planned in exp.PLANNED_MODELS:
        path = tmp_path / planned["filename"]
        path.write_bytes(planned["family_id"].encode())
        paths[planned["hub_id"]] = str(path)

    def pair_resolver(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return [
            {
                "hf_id": exp.PLANNED_MODELS[0]["hub_id"],
                "model_path": paths[exp.PLANNED_MODELS[0]["hub_id"]],
            },
            {
                "hf_id": exp.PLANNED_MODELS[1]["hub_id"],
                "model_path": paths[exp.PLANNED_MODELS[1]["hub_id"]],
            },
        ]

    models = exp.resolve_model_specs(
        pair_resolver=pair_resolver,
        single_resolver=lambda hub_id, _quant: paths[hub_id],
        metadata_reader=lambda _path: _metadata(),
        file_hasher=lambda path: next(
            row["expected_sha256"]
            for row in exp.PLANNED_MODELS
            if row["filename"] == Path(path).name
        ),
    )

    assert calls == [{"gpu_indices": (0, 1), "model_indices": (0, 2)}]
    assert [row["hub_id"] for row in models] == [row["hub_id"] for row in exp.PLANNED_MODELS]
    assert all(row["gguf_metadata"]["tokenizer_metadata"]["token_count"] > 0 for row in models)
    assert all(row["gguf_metadata"]["chat_template"] for row in models)
    assert all(
        exp.model_record_errors(row, planned) == []
        for row, planned in zip(models, exp.PLANNED_MODELS, strict=True)
    )


def test_scenario_6833_request_payload_has_no_repair_or_feedback() -> None:
    """SCENARIO-CONSTRAINT-6833-BUDGET fixes equal requests with no repair path."""

    typed = exp.request_payload(
        "typed", exp.RANDOM_SEED, max_tokens=exp.DECODE_SETTINGS["max_output_tokens"]
    )
    compressed = exp.request_payload(
        "compressed", exp.RANDOM_SEED, max_tokens=exp.DECODE_SETTINGS["max_output_tokens"]
    )
    for key in ("seed", "temperature", "top_p", "top_k", "repeat_penalty", "max_tokens", "stop"):
        assert typed[key] == compressed[key]
    assert typed["stream"] is True
    assert "grammar" not in typed
    assert "response_format" not in typed
    assert exp.DECODE_SETTINGS["repair_budget"] == 0
    assert exp.DECODE_SETTINGS["retry_budget"] == 0
    assert exp.DECODE_SETTINGS["answer_feedback"] is False


def test_scenario_6833_raw_bytes_and_exact_scoring(fixture: dict) -> None:
    """SCENARIO-CONSTRAINT-6833-RAW-BYTES and -SCORING retain exact evidence."""

    scenario = fixture["scenarios"][0]
    model = _models()[0]
    receipt = _receipt(model, 0)
    row = _row(scenario, exp.ARMS[0], model, receipt)
    prompt = base64.b64decode(row["prompt_bytes_b64"], validate=True)
    raw = base64.b64decode(row["raw_output_bytes_b64"], validate=True)
    assert exp.sha256_bytes(prompt) == row["prompt_sha256"]
    assert len(prompt) == row["prompt_byte_length"]
    assert exp.sha256_bytes(raw) == row["raw_output_sha256"]
    assert len(raw) == row["raw_output_byte_length"]
    assert row["parse_status"] == {"parsed": True, "error": None}
    assert row["parsed_fields"] == {"selected_action_ids": scenario["legal_action_ids"]}
    assert row["joint_result"]["passed"] is True
    assert all(value["passed"] for value in row["obligation_results"].values())
    assert exp.validate_row(row, scenario, model, receipt) == []

    invalid = exp.build_scored_row(
        scenario=scenario,
        arm=exp.ARMS[1],
        prompt=scenario["prompts"][exp.ARMS[1]].encode(),
        raw_output=b"not-json",
        model=model,
        process_receipt=receipt,
        prompt_tokens=10,
        generated_tokens=2,
        latency_s=0.1,
        checker_sha256="sha256:" + "c" * 64,
    )
    assert invalid["parse_status"] == {"parsed": False, "error": "invalid_json"}
    assert invalid["parsed_fields"] is None
    assert invalid["obligation_results"] == {}
    assert invalid["joint_result"]["passed"] is False


def test_scenario_6833_checkpoint_restart_skips_complete_rows(
    tmp_path: Path, fixture: dict
) -> None:
    """SCENARIO-CONSTRAINT-6833-CHECKPOINT verifies hashes before exact resume."""

    manifest = exp.build_manifest(fixture)
    store = exp.RowCheckpoint(tmp_path / "rows.json", manifest["manifest_sha256"])
    scenario = fixture["scenarios"][0]
    model = _models()[0]
    row = _row(scenario, exp.ARMS[0], model, _receipt(model, 0))
    receipt = store.append_batch([row])
    reopened = exp.RowCheckpoint(tmp_path / "rows.json", manifest["manifest_sha256"])
    assert receipt["accepted_row_ids"] == [row["row_id"]]
    assert reopened.completed_ids == {row["row_id"]}
    assert row["row_id"] not in reopened.missing_ids(manifest["expected_row_ids"])
    duplicate = reopened.append_batch([row])
    assert duplicate["accepted_row_ids"] == []
    assert duplicate["duplicate_row_ids"] == [row["row_id"]]

    process_receipt = _receipt(model, 0)
    process_publish = reopened.record_process_receipt(process_receipt)
    assert process_publish["durable"] is True
    with_receipt = exp.RowCheckpoint(tmp_path / "rows.json", manifest["manifest_sha256"])
    assert with_receipt.process_receipts == [process_receipt]
    assert with_receipt.record_process_receipt(process_receipt)["duplicate"] is True

    changed = json.loads((tmp_path / "rows.json").read_bytes())
    changed["rows"][0]["raw_output_byte_length"] += 1
    (tmp_path / "rows.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint row hash"):
        exp.RowCheckpoint(tmp_path / "rows.json", manifest["manifest_sha256"])


def test_scenario_6833_checkpoint_persists_active_receipt_before_rows(
    tmp_path: Path, fixture: dict
) -> None:
    """SCENARIO-CONSTRAINT-6833-CHECKPOINT keeps process evidence across a stop."""

    manifest = exp.build_manifest(fixture)
    store = exp.RowCheckpoint(tmp_path / "rows.json", manifest["manifest_sha256"])
    model = _models()[2]
    active = _receipt(model, 2)
    active.update(
        {
            "first_token_b64": "",
            "final_token_b64": "",
            "teardown_complete": False,
            "process_absent_after_exit": False,
            "authentic": False,
            "receipt_status": "active",
        }
    )
    first = store.upsert_process_receipt(active)
    assert first["updated"] is False
    with pytest.raises(ValueError, match="no session"):
        store.upsert_process_receipt({})
    reopened = exp.RowCheckpoint(tmp_path / "rows.json", manifest["manifest_sha256"])
    assert reopened.process_receipts == [active]

    completed = deepcopy(active)
    completed.update(
        {
            "first_token_b64": base64.b64encode(b"{").decode("ascii"),
            "final_token_b64": base64.b64encode(b"}").decode("ascii"),
            "teardown_complete": True,
            "process_absent_after_exit": True,
            "authentic": True,
            "receipt_status": "complete",
        }
    )
    update = reopened.upsert_process_receipt(completed)
    assert update["updated"] is True
    assert reopened.upsert_process_receipt(completed)["duplicate"] is True
    assert exp.RowCheckpoint(tmp_path / "rows.json", manifest["manifest_sha256"]).process_receipts == [
        completed
    ]

    changed = deepcopy(completed)
    changed["pid"] += 1
    with pytest.raises(ValueError, match="process identity changed"):
        reopened.upsert_process_receipt(changed)
    changed = deepcopy(completed)
    changed["final_token_b64"] = "changed"
    with pytest.raises(ValueError, match="completed process receipt changed"):
        reopened.upsert_process_receipt(changed)
    second = deepcopy(active)
    second["session_id"] = "second-session"
    assert reopened.upsert_process_receipt(second)["updated"] is False


def test_scenario_6833_checkpoint_recovers_only_checksummed_owned_evidence(
    fixture: dict, tmp_path: Path
) -> None:
    """SCENARIO-CONSTRAINT-6833-CHECKPOINT fails closed on recovery evidence drift."""

    model = _models()[2]
    scenario = fixture["scenarios"][0]
    source_receipt = _receipt(model, 2)
    row = _row(scenario, exp.ARMS[0], model, source_receipt)
    token_digest = "sha256:" + "a" * 64
    event = {
        "phase": "preflight",
        "previous_phase": None,
        "previous_event_checksum": None,
        "monotonic_ns": 10,
        "owner_token_digest": token_digest,
        "details": {"recovery_performed": False},
    }
    event["event_checksum"] = lease_api.event_checksum(event)
    journal = {
        "schema": lease_api.SCHEMA,
        "task_id": "exp6833-corpus-gemma26",
        "owner": {
            "pid": 8000,
            "pid_start_ticks": 80,
            "executable": "/python",
            "argv_digest": "sha256:" + "b" * 64,
            "token_digest": token_digest,
        },
        "device_uuid": source_receipt["physical_gpu_uuid"],
        "expected_model": model["hub_id"],
        "acquired_monotonic_ns": 10,
        "heartbeat_monotonic_ns": 10,
        "expires_monotonic_ns": 110,
        "ttl_ns": 100,
        "phase": "preflight",
        "phase_history": [event],
        "vram_mb": {"before": 4, "resident": None, "after": None},
        "exit_evidence": {"exit_code": None, "observed_monotonic_ns": None},
        "unload_evidence": {"required": False, "observed": False, "observed_monotonic_ns": None},
        "recovery": {"performed": False, "signals_sent": []},
        "released": False,
        "released_monotonic_ns": None,
    }
    journal["checksum"] = lease_api.journal_checksum(journal)
    port = source_receipt["port"]
    server_log = (
        f"loading model '{model['model_path']}'\n"
        "device CUDA0\n"
        "offloaded 31/31 layers to GPU\n"
        f"server is listening on http://127.0.0.1:{port}\n"
        "done request: POST /v1/chat/completions 127.0.0.1 200\n"
    ).encode()
    log_path = tmp_path / "stderr.bin"
    log_path.write_bytes(server_log)
    recovered = exp.build_recovered_process_receipt(
        rows=[row],
        model=model,
        server=Path("/llama-server"),
        journal=journal,
        server_log=server_log,
        server_log_path=log_path,
        lease_release={"released": True, "recovery_performed": True},
        device_index=0,
        process_start_time="2026-09-01T01:54:40Z",
        teardown_duration_s=0.5,
        process_absent_after_exit=True,
        gpu_memory_recovered=True,
    )
    assert recovered["session_id"] == source_receipt["session_id"]
    assert recovered["authentic"] is True
    assert recovered["receipt_status"] == "recovered_complete"
    assert recovered["receipt_recovery"]["journal_sha256"].startswith("sha256:")
    assert recovered["receipt_recovery"]["server_log_sha256"].startswith("sha256:")
    assert recovered["teardown_mode"] == "stale_lease_recovery"

    common = {
        "model": model,
        "server": Path("/llama-server"),
        "journal": journal,
        "server_log": server_log,
        "server_log_path": log_path,
        "lease_release": {"released": True},
        "device_index": 0,
        "process_start_time": "2026-09-01T01:54:40Z",
        "teardown_duration_s": 0.5,
        "process_absent_after_exit": True,
        "gpu_memory_recovered": True,
    }
    with pytest.raises(ValueError, match="rows missing"):
        exp.build_recovered_process_receipt(rows=[], **common)
    invalid_journal = deepcopy(journal)
    invalid_journal["checksum"] = "wrong"
    with pytest.raises(ValueError, match="journal invalid"):
        exp.build_recovered_process_receipt(rows=[row], **{**common, "journal": invalid_journal})
    other_process = deepcopy(row)
    other_process["process_identity"]["session_id"] = "other"
    with pytest.raises(ValueError, match="process identities differ"):
        exp.build_recovered_process_receipt(rows=[row, other_process], **common)
    wrong_model = deepcopy(row)
    wrong_model["model_id"] = "wrong"
    with pytest.raises(ValueError, match="row model mismatch"):
        exp.build_recovered_process_receipt(rows=[wrong_model], **common)
    wrong_journal = deepcopy(journal)
    wrong_journal["task_id"] = "wrong"
    wrong_journal["checksum"] = lease_api.journal_checksum(wrong_journal)
    with pytest.raises(ValueError, match="journal identity mismatch"):
        exp.build_recovered_process_receipt(rows=[row], **{**common, "journal": wrong_journal})
    invalid_output = deepcopy(row)
    invalid_output["raw_output_bytes_b64"] = "%%%"
    with pytest.raises(ValueError, match="row output invalid"):
        exp.build_recovered_process_receipt(rows=[invalid_output], **common)
    empty_output = deepcopy(row)
    empty_output["raw_output_bytes_b64"] = ""
    with pytest.raises(ValueError, match="token evidence missing"):
        exp.build_recovered_process_receipt(rows=[empty_output], **common)

    changed_log = server_log.replace(str(port).encode(), b"9999")
    with pytest.raises(ValueError, match="server log identity mismatch"):
        exp.build_recovered_process_receipt(
            rows=[row],
            model=model,
            server=Path("/llama-server"),
            journal=journal,
            server_log=changed_log,
            server_log_path=log_path,
            lease_release={"released": True},
            device_index=0,
            process_start_time="2026-09-01T01:54:40Z",
            teardown_duration_s=0.5,
            process_absent_after_exit=True,
            gpu_memory_recovered=True,
        )


def test_scenario_6833_cross_model_isolation_and_teardown(fixture: dict) -> None:
    """SCENARIO-CONSTRAINT-6833-DISPATCH and -TEARDOWN reject mixed process evidence."""

    models = _models()
    receipts = [_receipt(model, index) for index, model in enumerate(models)]
    scenario = fixture["scenarios"][0]
    row = _row(scenario, exp.ARMS[0], models[0], receipts[0])
    assert exp.validate_cross_model_isolation([row], receipts) == []
    mixed = deepcopy(row)
    mixed["process_identity"]["session_id"] = receipts[1]["session_id"]
    assert "row process does not match its model receipt" in exp.validate_cross_model_isolation(
        [mixed], receipts
    )

    resumed_receipt = deepcopy(receipts[0])
    resumed_receipt["session_id"] = "resumed-session"
    resumed_row = deepcopy(row)
    resumed_row["process_identity"]["session_id"] = resumed_receipt["session_id"]
    assert (
        exp.validate_cross_model_isolation([row, resumed_row], [*receipts, resumed_receipt]) == []
    )

    fake = SimpleNamespace(returncode=None, sent=[], waited=[])
    fake.poll = lambda: fake.returncode
    fake.send_signal = lambda value: fake.sent.append(value)
    fake.wait = lambda timeout: fake.waited.append(timeout) or 0
    assert exp.terminate_owned_process(fake, timeout_s=3.0) == 0
    assert fake.sent and fake.waited == [3.0]


def test_scenario_6833_complete_artifact_readiness_ignores_accuracy(fixture: dict) -> None:
    """SCENARIO-CONSTRAINT-6833-READINESS uses evidence completeness, not accuracy."""

    manifest = exp.build_manifest(fixture)
    models = _models()
    receipts = [_receipt(model, index) for index, model in enumerate(models)]
    rows = [
        _row(scenario, arm, model, receipt)
        for model, receipt in zip(models, receipts, strict=True)
        for scenario in fixture["scenarios"]
        for arm in exp.ARMS
    ]
    rows[0] = exp.build_scored_row(
        scenario=fixture["scenarios"][0],
        arm=exp.ARMS[0],
        prompt=fixture["scenarios"][0]["prompts"][exp.ARMS[0]].encode(),
        raw_output=b"wrong",
        model=models[0],
        process_receipt=receipts[0],
        prompt_tokens=10,
        generated_tokens=1,
        latency_s=0.1,
        checker_sha256="sha256:" + "c" * 64,
    )
    artifact = exp.assemble_artifact(
        run_date="20260901",
        fixture=fixture,
        manifest=manifest,
        models=models,
        rows=rows,
        process_receipts=receipts,
        preconditions=[{"check": "all", "passed": True}],
        accelerator_samples=[{"gpu_uuid": "GPU-test"}],
        checkpoint_receipts=[{"batch": 1, "durable": True}],
        duration_s=2.0,
        phase_clocks={
            "preflight": 0.1,
            "scoring": 0.2,
            "write": 0.1,
            "verify": 0.1,
            "teardown": 0.1,
        },
    )
    assert exp.validate_artifact(artifact) == []
    assert artifact["operational_saturation_corpus_ready"] is True
    assert artifact["row_coverage"]["observed_unique"] == 900
    assert artifact["exact_scores"]["joint_passed"] == 899
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_")


def test_scenario_6833_blocked_artifact_names_exact_gate() -> None:
    """SCENARIO-CONSTRAINT-6833-PREFLIGHT emits no rows after one failed gate."""

    artifact = exp.build_blocked_artifact(
        run_date="20260901",
        failed_check="embedded_chat_template",
        expected=True,
        observed=False,
        preconditions=[{"check": "embedded_chat_template", "passed": False}],
        models=_models(),
        duration_s=1.0,
        phase_clocks={"preflight": 1.0},
    )
    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == exp.BLOCKED_STATUS
    assert artifact["per_unit_rows"] == []
    assert artifact["operational_saturation_corpus_ready"] is False
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "failed_check": "embedded_chat_template",
        "expected": True,
        "observed": False,
    }
    assert artifact["honest_verdict"] == exp.BLOCKED_STATUS


def test_req_6833_file_and_metadata_helpers_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CONSTRAINT-6833 binds file bytes, revisions, and bounded template bytes."""

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abc")
    assert exp.sha256_file(payload) == exp.sha256_bytes(b"abc")
    assert exp.sha256_file(tmp_path / "missing") == "missing"
    assert exp._revision_from_path(Path("cache/snapshots/revision/model.gguf")) == "revision"
    assert exp._revision_from_path(Path("snapshots")) == "local-unversioned"

    template = b"{{ messages }}"
    payload.write_bytes(len(template).to_bytes(8, "little") + template)
    source = {
        "field_provenance": {
            "metadata_keys": {
                "tokenizer.chat_template": {"value_offset": 0, "value_type": "string"}
            }
        }
    }
    assert (
        exp._read_metadata_string(payload, source, "tokenizer.chat_template") == template.decode()
    )
    assert exp._read_metadata_string(payload, {}, "tokenizer.chat_template") == ""
    assert exp._read_metadata_string(payload, {"field_provenance": []}, "x") == ""

    payload.write_bytes(b"short")
    assert exp._read_metadata_string(payload, source, "tokenizer.chat_template") == ""
    payload.write_bytes((0).to_bytes(8, "little"))
    assert exp._read_metadata_string(payload, source, "tokenizer.chat_template") == ""
    payload.write_bytes((17 * 1024 * 1024).to_bytes(8, "little"))
    assert exp._read_metadata_string(payload, source, "tokenizer.chat_template") == ""
    payload.write_bytes((2).to_bytes(8, "little") + b"x")
    assert exp._read_metadata_string(payload, source, "tokenizer.chat_template") == ""
    payload.write_bytes((1).to_bytes(8, "little") + b"\xff")
    assert exp._read_metadata_string(payload, source, "tokenizer.chat_template") == ""

    payload.write_bytes(len(template).to_bytes(8, "little") + template)
    parsed = {**_metadata(), **source}
    monkeypatch.setattr(exp, "read_gguf_metadata", lambda _path: parsed)
    captured = exp.read_model_metadata(payload)
    assert captured["chat_template"] == template.decode()
    assert captured["chat_template_sha256"] == exp.sha256_bytes(template)


def test_scenario_6833_missing_and_invalid_models_stay_blocked(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6833-PREFLIGHT rejects missing and malformed model records."""

    missing = exp.resolve_model_specs(
        pair_resolver=lambda **_kwargs: None,
        single_resolver=lambda _hub, _quant: None,
    )
    assert all(row["revision"] == "missing" for row in missing)
    assert "model_sha256" in exp.model_record_errors(missing[0], exp.PLANNED_MODELS[0])

    paths: dict[str, str] = {}
    for planned in exp.PLANNED_MODELS:
        path = tmp_path / planned["filename"]
        path.write_bytes(b"model")
        paths[planned["hub_id"]] = str(path)
    failed_metadata = exp.resolve_model_specs(
        pair_resolver=lambda **_kwargs: [
            {"hf_id": planned["hub_id"], "model_path": paths[planned["hub_id"]]}
            for planned in exp.PLANNED_MODELS[:2]
        ],
        single_resolver=lambda hub, _quant: paths[hub],
        metadata_reader=lambda _path: (_ for _ in ()).throw(ValueError("bad metadata")),
        file_hasher=lambda _path: "bad",
    )
    errors = exp.model_record_errors(
        {**failed_metadata[0], "model_path": "/wrong", "decode_settings": {}},
        exp.PLANNED_MODELS[0],
    )
    assert set(errors) >= {
        "model_sha256",
        "filename",
        "language_model_metadata",
        "embedded_tokenizer",
        "embedded_chat_template",
        "decode_settings",
    }


def test_scenario_6833_row_validator_reports_each_mutation(fixture: dict) -> None:
    """SCENARIO-CONSTRAINT-6833-RAW-BYTES rejects every changed row receipt."""

    scenario = fixture["scenarios"][0]
    model = _models()[0]
    receipt = _receipt(model, 0)
    row = _row(scenario, exp.ARMS[0], model, receipt)
    broken = deepcopy(row)
    broken["prompt_bytes_b64"] = "%%%"
    assert exp.validate_row(broken, scenario, model, receipt) == ["invalid row base64"]

    mutations = {
        "row identity mismatch": lambda value: value.update({"row_id": "wrong"}),
        "row model mismatch": lambda value: value.update({"model_sha256": "wrong"}),
        "row scenario mismatch": lambda value: value.update({"scenario_hash": "wrong"}),
        "prompt byte receipt mismatch": lambda value: value.update({"prompt_byte_length": -1}),
        "output byte receipt mismatch": lambda value: value.update({"raw_output_byte_length": -1}),
        "row budget mismatch": lambda value: value.update({"decode_settings": {}}),
        "row process mismatch": lambda value: value["process_identity"].update(
            {"session_id": "wrong"}
        ),
        "exact score mismatch": lambda value: value.update({"joint_result": {}}),
    }
    for message, mutation in mutations.items():
        changed = deepcopy(row)
        mutation(changed)
        assert message in exp.validate_row(changed, scenario, model, receipt)
    changed = deepcopy(row)
    changed["row_sha256"] = "wrong"
    assert "row checksum mismatch" in exp.validate_row(changed, scenario, model, receipt)
    assert exp.validate_rows([row], {}, [model], [receipt]) == [
        {"row_id": row["row_id"], "errors": ["row source missing"]}
    ]
    changed = deepcopy(row)
    changed["prompt_byte_length"] = -1
    changed["row_sha256"] = exp._row_checksum(changed)
    assert exp.validate_rows([changed], fixture, [model], [receipt])[0]["errors"]


def test_scenario_6833_checkpoint_rejects_all_identity_corruption(
    tmp_path: Path, fixture: dict
) -> None:
    """SCENARIO-CONSTRAINT-6833-CHECKPOINT rejects drift and changed completed rows."""

    manifest = exp.build_manifest(fixture)
    scenario = fixture["scenarios"][0]
    model = _models()[0]
    row = _row(scenario, exp.ARMS[0], model, _receipt(model, 0))
    path = tmp_path / "checkpoint.json"

    path.write_text(json.dumps({"manifest_sha256": "wrong", "rows": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        exp.RowCheckpoint(path, manifest["manifest_sha256"])
    path.write_text(
        json.dumps({"manifest_sha256": manifest["manifest_sha256"], "rows": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rows invalid"):
        exp.RowCheckpoint(path, manifest["manifest_sha256"])
    path.write_text(
        json.dumps({"manifest_sha256": manifest["manifest_sha256"], "rows": [row, row]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate row"):
        exp.RowCheckpoint(path, manifest["manifest_sha256"])

    path.write_text(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "rows": [row],
                "process_receipts": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="process receipts invalid"):
        exp.RowCheckpoint(path, manifest["manifest_sha256"])
    process = _receipt(model, 0)
    path.write_text(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "rows": [row],
                "process_receipts": [process, process],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate process receipt"):
        exp.RowCheckpoint(path, manifest["manifest_sha256"])

    path.unlink()
    store = exp.RowCheckpoint(path, manifest["manifest_sha256"])
    invalid = deepcopy(row)
    invalid["row_sha256"] = "wrong"
    with pytest.raises(ValueError, match="row hash"):
        store.append_batch([invalid])
    store.append_batch([row])
    changed = deepcopy(row)
    changed["latency_s"] = 5.0
    changed["row_sha256"] = exp._row_checksum(changed)
    with pytest.raises(ValueError, match="completed row changed"):
        store.append_batch([changed])
    with pytest.raises(ValueError, match="no session"):
        store.record_process_receipt({})
    store.record_process_receipt(process)
    changed_process = deepcopy(process)
    changed_process["pid"] += 1
    with pytest.raises(ValueError, match="process receipt changed"):
        store.record_process_receipt(changed_process)


def test_scenario_6833_process_helpers_cover_failure_paths(fixture: dict) -> None:
    """SCENARIO-CONSTRAINT-6833-TEARDOWN keeps timeout and isolation failures explicit."""

    models = _models()
    receipts = [_receipt(model, index) for index, model in enumerate(models)]
    receipts[1]["session_id"] = receipts[0]["session_id"]
    assert "model receipts share a process session" in exp.validate_cross_model_isolation(
        [], receipts
    )
    assert exp.terminate_owned_process(None, timeout_s=1.0) is None
    complete = SimpleNamespace(returncode=7, poll=lambda: 7)
    assert exp.terminate_owned_process(complete, timeout_s=1.0) == 7

    class TimedOut:
        returncode = None

        def __init__(self) -> None:
            self.calls = 0
            self.killed = False

        def poll(self) -> None:
            return None

        def send_signal(self, _value: int) -> None:
            return None

        def wait(self, timeout: float) -> int:
            self.calls += 1
            if self.calls == 1:
                raise exp.subprocess.TimeoutExpired("server", timeout)
            return 9

        def kill(self) -> None:
            self.killed = True

    timed_out = TimedOut()
    assert exp.terminate_owned_process(timed_out, timeout_s=0.1) == 9
    assert timed_out.killed is True
    assert exp._first_failed([{"passed": True}, {"check": "x", "passed": False}]) == {
        "check": "x",
        "passed": False,
    }
    assert exp._first_failed([{"passed": True}]) is None


def test_req_6833_partial_null_validator_and_writer_paths(
    tmp_path: Path, fixture: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CONSTRAINT-6833 validates partial, null, blocked, and atomic output paths."""

    manifest = exp.build_manifest(fixture)
    models = _models()
    receipts = [_receipt(model, index) for index, model in enumerate(models)]
    partial = exp.assemble_artifact(
        run_date="20260901",
        fixture=fixture,
        manifest=manifest,
        models=models,
        rows=[],
        process_receipts=receipts,
        preconditions=[],
        accelerator_samples=[],
        checkpoint_receipts=[],
        duration_s=1.0,
        phase_clocks={},
    )
    assert partial["status"] == exp.PARTIAL_STATUS
    assert partial["verdict_class"] == "partial"
    assert exp.validate_artifact(partial) == []

    rows = [
        exp.build_scored_row(
            scenario=scenario,
            arm=arm,
            prompt=scenario["prompts"][arm].encode(),
            raw_output=b"wrong",
            model=model,
            process_receipt=receipt,
            prompt_tokens=10,
            generated_tokens=1,
            latency_s=0.1,
            checker_sha256="sha256:" + "c" * 64,
        )
        for model, receipt in zip(models, receipts, strict=True)
        for scenario in fixture["scenarios"]
        for arm in exp.ARMS
    ]
    null = exp.assemble_artifact(
        run_date="20260901",
        fixture=fixture,
        manifest=manifest,
        models=models,
        rows=rows,
        process_receipts=receipts,
        preconditions=[],
        accelerator_samples=[],
        checkpoint_receipts=[{"durable": True}],
        duration_s=1.0,
        phase_clocks={},
    )
    assert null["operational_saturation_corpus_ready"] is True
    assert null["verdict_class"] == "null"
    assert null["honest_verdict"].startswith("complete_null")

    mutations = {
        "top-level field set mismatch": lambda value: value.pop("schema"),
        "field principles mismatch": lambda value: value["field_principles"].pop("schema"),
        "inference substrate mismatch": lambda value: value.update({"inference_substrate": "bad"}),
        "verifier_is_oracle mismatch": lambda value: value.update({"verifier_is_oracle": True}),
        "verdict class mismatch": lambda value: value.update({"verdict_class": "bad"}),
        "honest verdict prefix mismatch": lambda value: value.update({"honest_verdict": "bad"}),
        "reproducibility checksum mismatch": lambda value: value.update(
            {"reproducibility_checksum": "bad"}
        ),
        "readiness is not evidence-derived": lambda value: value.update(
            {"operational_saturation_corpus_ready": False}
        ),
        "exact scores are not row-derived": lambda value: value.update({"exact_scores": {}}),
        "gate summary mismatch": lambda value: value["gate_check_summary"].update(
            {"passed": False}
        ),
    }
    for message, mutation in mutations.items():
        changed = deepcopy(null)
        mutation(changed)
        assert message in exp.validate_artifact(changed)

    blocked = exp.build_blocked_artifact(
        run_date="20260901",
        failed_check="x",
        expected=True,
        observed=False,
        preconditions=[],
        models=models,
        duration_s=1.0,
        phase_clocks={},
    )
    blocked["per_unit_rows"] = [{}]
    blocked["operational_saturation_corpus_ready"] = True
    blocked["gate_check_summary"] = {}
    blocked["verdict_class"] = "partial"
    blocked["honest_verdict"] = "complete_partial"
    blocked["reproducibility_checksum"] = exp._artifact_checksum(blocked)
    errors = exp.validate_artifact(blocked)
    assert "blocked artifact contains headline rows" in errors
    assert "blocked gate receipt mismatch" in errors
    assert "blocked verdict mismatch" in errors

    output = tmp_path / "artifact.json"
    exp._write_json(output, {"ready": True})
    assert json.loads(output.read_bytes()) == {"ready": True}
    failure = exp.LivePhaseError("model", "failed", {"pid": 7})
    assert failure.check == "model"
    assert failure.observed == "failed"
    assert failure.receipt == {"pid": 7}
    monkeypatch.setattr(exp, "FIXTURE_PATH", Path("results/missing-exp6833-fixture.json"))
    assert exp.validate_artifact(partial) == []
