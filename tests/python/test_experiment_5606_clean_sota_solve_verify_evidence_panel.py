"""Tests for Exp5606 clean SOTA solve-versus-verify evidence panel.

Spec refs: REQ-VERIFY-5606, SCENARIO-VERIFY-5606.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5566_exact_asp_fsm_near_miss_corpus as corpus5566
from carnot import experiment_5606_clean_sota_solve_verify_evidence_panel as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5606_clean_sota_solve_verify_evidence_panel.py")


def _fake_model_specs(tmp_path: Path) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for index, hf_id in enumerate(mod.MANDATED_HEADLINE_IDS):
        stem = hf_id.rsplit("/", 1)[-1].replace("-GGUF", "")
        path = tmp_path / f"{stem}-Q4_K_M.gguf"
        path.write_bytes(f"fixture-{hf_id}".encode())
        specs.append(
            {
                "name": stem,
                "hf_id": hf_id,
                "family": mod.model_family(hf_id),
                "role": "dense" if "31B" in hf_id else "moe",
                "gpu": index % 2,
                "model_path": str(path),
                "headline_eligible": True,
            }
        )
    return mod.normalize_model_specs(specs)


def _authenticated_receipt(model_specs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "torch_cuda_available": True,
        "torch_device_count": 2,
        "llama_cpp_supports_gpu_offload": True,
        "gpu_offload_authenticated": True,
        "devices": [
            {"index": 0, "name": "NVIDIA GeForce RTX 3090"},
            {"index": 1, "name": "NVIDIA GeForce RTX 3090"},
        ],
        "model_receipts": [
            {
                "model_hf_id": spec["hf_id"],
                "model_path": spec["model_path"],
                "model_sha256": spec["model_sha256"],
                "llama_cpp_version": "0.3.33",
                "pid": 2000 + index,
                "port": None,
                "runtime_mode": "llama_cpp_python_in_process_no_http_port",
                "worker_ok": True,
                "llama_cpp_supports_gpu_offload": True,
                "torch_cuda_available": True,
                "torch_device_count": 2,
                "devices": [{"index": index % 2, "name": "NVIDIA GeForce RTX 3090"}],
                "offloaded_layer_count_from_backend_log": 31,
                "pid_gpu_memory_mb_peak": 2048,
                "gpu_utilization_pct_peak": 7,
                "gpu_offload_authenticated": True,
                "stderr_tail": "llama.cpp CUDA offloaded 31/31 layers to GPU",
            }
            for index, spec in enumerate(model_specs)
        ],
    }


def _sample_pairs() -> list[dict[str, object]]:
    rows = json.loads((REPO / corpus5566.RESULT_RELATIVE_PATH).read_text())["corpus_rows"]
    return mod.sample_independent_pairs(rows, n=mod.MIN_INDEPENDENT_INSTANCES)


def _raw_calls(model_specs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    for spec in model_specs:
        hf_id = str(spec["hf_id"])
        for phase in ("solve", *mod.ARMS):
            response = json.dumps({"phase": phase, "model": hf_id, "ok": True})
            calls.append(
                {
                    "call_id": f"{hf_id}:{phase}:fixture",
                    "task_id": f"{phase}:fixture",
                    "phase": phase,
                    "arm": phase if phase in mod.ARMS else "",
                    "model_hf_id": hf_id,
                    "prompt": f"{phase} prompt for {hf_id}",
                    "raw_response": response,
                    "seed": mod.RANDOM_SEED,
                    "stop_reason": "stop_sequence",
                    "truncation_flag": False,
                    "sampling_parameters": {"temperature": 0.0, "top_p": 1.0},
                    "llama_cpp_arguments": {"n_gpu_layers": -1, "n_ctx": 8192, "n_batch": 256},
                    "token_counts": {
                        "prompt_tokens": 5,
                        "completion_tokens": 5,
                        "total_tokens": 10,
                        "source": "fixture",
                    },
                }
            )
    return calls


def _fake_panel_result(
    pairs: Sequence[Mapping[str, object]],
    model_specs: Sequence[Mapping[str, object]],
    *,
    qwen_parser_collapse: bool = False,
) -> dict[str, object]:
    solve_records: list[dict[str, object]] = []
    verifier_records: list[dict[str, object]] = []
    raw_hashes: dict[str, str] = {}
    for model_index, spec in enumerate(model_specs):
        hf_id = str(spec["hf_id"])
        for pair_index, pair in enumerate(pairs):
            instance_id = str(pair["instance_id"])
            solve_ok = (pair_index + model_index) % 3 != 0
            solve_text = f"{hf_id}:{instance_id}:solve"
            solve_hash = mod.sha256_text(solve_text)
            raw_hashes[f"{hf_id}:solve:{instance_id}"] = solve_hash
            solve_records.append(
                {
                    "model_hf_id": hf_id,
                    "instance_id": instance_id,
                    "family": pair["family"],
                    "parser_ok": True,
                    "exact_accepted": solve_ok,
                    "latency_s": 0.2,
                    "prompt_tokens": 64,
                    "completion_tokens": 18,
                    "response_hash": solve_hash,
                    "error_type": "" if solve_ok else "solve_exact_rejected",
                }
            )
            for row_key in ("valid_row", "invalid_row"):
                row = pair[row_key]
                assert isinstance(row, Mapping)
                true_label = str(row["label"])
                for arm in mod.ARMS:
                    repeated = arm == "repeated_verdict_3x"
                    collapse = (
                        qwen_parser_collapse and hf_id == mod.QWEN_ID and arm == "discrete_verdict"
                    )
                    response_hashes = [
                        mod.sha256_text(f"{hf_id}:{instance_id}:{row['row_id']}:{arm}:{repeat}")
                        for repeat in range(3 if repeated else 1)
                    ]
                    for index, value in enumerate(response_hashes):
                        raw_hashes[f"{hf_id}:{instance_id}:{row['row_id']}:{arm}:{index}"] = value
                    verifier_records.append(
                        {
                            "model_hf_id": hf_id,
                            "instance_id": instance_id,
                            "candidate_id": row["row_id"],
                            "family": pair["family"],
                            "arm": arm,
                            "true_label": true_label,
                            "predicted_label": None if collapse else true_label,
                            "parser_ok": not collapse,
                            "latency_s": 0.1,
                            "prompt_tokens": 80,
                            "completion_tokens": 12,
                            "response_hashes": response_hashes,
                            "repeat_labels": []
                            if collapse
                            else [true_label] * len(response_hashes),
                            "error_type": "verifier_json_parse_failure" if collapse else "",
                        }
                    )
    return {
        "solve_records": solve_records,
        "verifier_records": verifier_records,
        "raw_response_hash": raw_hashes,
        "inference_duration_s": 321.0,
    }


def test_req_verify_5606_spec_declares_clean_evidence_panel_contract() -> None:
    """REQ-VERIFY-5606: OpenSpec anchors model, envelope, parser, and gate fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5606") : spec.index("### REQ-VERIFY-5605")]
    normalized = " ".join(section.split())

    assert "SCENARIO-VERIFY-5606" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(mod.EVIDENCE_ENVELOPE_RELATIVE_PATH) in section
    assert "Exp 5605 raw response evidence envelope" in section
    assert "`inference_substrate=local_gguf_llamacpp_cuda`" in section
    assert "at least 30 independent" in normalized
    assert "`maximum_parser_failure_rate` from the worst model rather than an average" in section
    assert "`<=0.05`" in section
    assert "Sub-percent differences SHALL NOT be promoted" in section
    assert "SHALL NOT modify" in section
    assert "scripts/research_conductor.py" in section
    for hf_id in mod.MANDATED_HEADLINE_IDS:
        assert hf_id in section
    for arm in mod.ARMS:
        assert arm in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5606_complete_artifact_reports_clean_three_model_panel(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5606: clean evidence unlocks complete per-model denominators."""

    specs = _fake_model_specs(tmp_path)
    pairs = _sample_pairs()
    calls = _raw_calls(specs)
    rows = mod.build_response_envelope_rows(
        raw_calls=calls,
        model_specs=specs,
        device_receipt=_authenticated_receipt(specs),
    )
    artifact = mod.build_artifact(
        model_specs=specs,
        device_receipt=_authenticated_receipt(specs),
        sampled_pairs=pairs,
        panel_result=_fake_panel_result(pairs, specs),
        evidence_rows=rows,
        evidence_envelope_path=mod.EVIDENCE_ENVELOPE_RELATIVE_PATH.as_posix(),
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
        bootstrap_iterations=64,
    )

    assert artifact["panel_complete"] is True
    assert artifact["gpu_offload_authenticated"] is True
    assert artifact["raw_response_replay_passed"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert [row["hf_id"] for row in artifact["model_specs"]] == list(mod.MANDATED_HEADLINE_IDS)
    assert set(artifact["instances_evaluated_by_model"]) == set(mod.MANDATED_HEADLINE_IDS)
    assert all(
        row["solve_instances"] == mod.MIN_INDEPENDENT_INSTANCES
        for row in artifact["instances_evaluated_by_model"].values()
    )
    assert artifact["maximum_parser_failure_rate"] == 0.0
    assert artifact["per_model_truncation_rate"] == {
        hf_id: 0.0 for hf_id in mod.MANDATED_HEADLINE_IDS
    }
    assert artifact["solve_verify_asymmetry_supported"] is True
    assert artifact["exact_oracle_agreement"]["exact_validator_is_authority"] is True
    assert artifact["exact_oracle_agreement"]["llm_judge_used"] is False
    assert (
        artifact["verify_accuracy_by_model_and_arm"][mod.QWEN_ID]["criteria_decomposition"][
            "balanced_accuracy"
        ]
        == 1.0
    )
    effect = artifact["paired_effects_and_intervals"][mod.QWEN_ID]["criteria_decomposition"]
    assert effect["paired_unit"] == "instance_id"
    assert effect["effect"] < -0.01
    assert artifact["family_heterogeneity"]["models"] == list(mod.MANDATED_HEADLINE_IDS)

    mod.validate_artifact(artifact)


def test_req_verify_5606_parser_and_truncation_ceilings_are_worst_model_gates(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5606: one failing family blocks the panel instead of being averaged away."""

    specs = _fake_model_specs(tmp_path)
    pairs = _sample_pairs()
    calls = _raw_calls(specs)
    rows = mod.build_response_envelope_rows(
        raw_calls=calls,
        model_specs=specs,
        device_receipt=_authenticated_receipt(specs),
    )
    for row in rows:
        if row["model_hf_id"] == mod.GEMMA31_ID:
            row["truncation_flag"] = True
            row["row_hash"] = mod.row_hash(row)
            break
    rows = mod.rechain_response_envelope_rows(rows)

    artifact = mod.build_artifact(
        model_specs=specs,
        device_receipt=_authenticated_receipt(specs),
        sampled_pairs=pairs,
        panel_result=_fake_panel_result(pairs, specs, qwen_parser_collapse=True),
        evidence_rows=rows,
        evidence_envelope_path=mod.EVIDENCE_ENVELOPE_RELATIVE_PATH.as_posix(),
        bootstrap_iterations=16,
    )

    assert artifact["panel_complete"] is False
    assert artifact["honest_verdict"].startswith("blocked_parser_or_truncation_ceiling_failed")
    assert artifact["per_model_parser_failure_rate"][mod.QWEN_ID] > mod.PARSER_FAILURE_CEILING
    assert (
        artifact["maximum_parser_failure_rate"]
        == artifact["per_model_parser_failure_rate"][mod.QWEN_ID]
    )
    assert artifact["per_model_truncation_rate"][mod.GEMMA31_ID] > mod.TRUNCATION_CEILING
    assert artifact["per_model_parser_failure_rate"][mod.GEMMA26_ID] == 0.0
    assert artifact["solve_verify_asymmetry_supported"] is False
    mod.validate_artifact(artifact)


def test_req_verify_5606_replay_and_cuda_authentication_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5606: tampered envelopes and CPU fallback cannot promote headline rows."""

    specs = _fake_model_specs(tmp_path)
    pairs = _sample_pairs()
    rows = mod.build_response_envelope_rows(
        raw_calls=_raw_calls(specs),
        model_specs=specs,
        device_receipt=_authenticated_receipt(specs),
    )
    corrupted = copy.deepcopy(rows)
    corrupted[0]["raw_response_payload"] = mod.encode_lossless_payload(b'{"tampered":true}')
    with pytest.raises(mod.EnvelopeReplayError):
        mod.replay_response_envelope_rows(corrupted)

    cpu_receipt = dict(_authenticated_receipt(specs))
    cpu_receipt["gpu_offload_authenticated"] = False
    cpu_receipt["model_receipts"] = [
        dict(row) | {"gpu_offload_authenticated": False, "pid_gpu_memory_mb_peak": 0}
        for row in cpu_receipt["model_receipts"]
    ]
    artifact = mod.build_artifact(
        model_specs=specs,
        device_receipt=cpu_receipt,
        sampled_pairs=pairs,
        panel_result=_fake_panel_result(pairs, specs),
        evidence_rows=rows,
        evidence_envelope_path=mod.EVIDENCE_ENVELOPE_RELATIVE_PATH.as_posix(),
        bootstrap_iterations=16,
    )

    assert artifact["gpu_offload_authenticated"] is False
    assert artifact["panel_complete"] is False
    assert artifact["honest_verdict"].startswith("blocked_no_cuda_offload_authenticated")
    overclaim = dict(artifact) | {"panel_complete": True, "honest_verdict": "complete: invalid"}
    overclaim["reproducibility_checksum"] = mod.payload_checksum(overclaim)
    with pytest.raises(ValueError, match="gpu_offload_authenticated"):
        mod.validate_artifact(overclaim)


def test_scenario_verify_5606_run_writes_artifact_and_response_ledger(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5606: injected runner writes stable JSON and replayable JSONL."""

    specs = _fake_model_specs(tmp_path)
    pairs = _sample_pairs()
    rows = mod.build_response_envelope_rows(
        raw_calls=_raw_calls(specs),
        model_specs=specs,
        device_receipt=_authenticated_receipt(specs),
    )
    result_path = tmp_path / "experiment_5606.json"
    ledger_path = tmp_path / "experiment_5606_responses.jsonl"

    artifact = mod.run(
        result_path=result_path,
        evidence_envelope_path=ledger_path,
        model_specs=specs,
        device_receipt=_authenticated_receipt(specs),
        sampled_pairs=pairs,
        panel_result=_fake_panel_result(pairs, specs),
        evidence_rows=rows,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
        bootstrap_iterations=32,
    )

    assert result_path.is_file()
    assert ledger_path.is_file()
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert len(ledger_path.read_text(encoding="utf-8").splitlines()) == len(rows)
    assert mod.replay_response_envelope_path(ledger_path)["row_count"] == len(rows)
    mod.validate_artifact(artifact)


def test_req_verify_5606_defensive_branches_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5606: defensive validators reject malformed gates and ledgers."""

    specs = _fake_model_specs(tmp_path)
    receipt = _authenticated_receipt(specs)
    rows = mod.build_response_envelope_rows(
        raw_calls=_raw_calls(specs),
        model_specs=specs,
        device_receipt=receipt,
    )

    assert mod.model_family("local/other") == "other"
    assert [row["hf_id"] for row in mod.normalize_model_specs(specs[:2])] == [
        mod.QWEN_ID,
        mod.GEMMA31_ID,
    ]
    assert mod.honest_verdict(True, False, "") == (
        "complete: clean authenticated solve-versus-verify panel without supported asymmetry; "
        "no sub-percent claims"
    )
    assert mod._model_specs_ready("bad") is False
    assert mod._model_specs_ready(specs[:2]) is False
    assert mod._full_denominators("bad") is False
    assert mod._full_denominators({mod.QWEN_ID: {"full_denominator": True}}) is False
    assert (
        mod._paired_effects([], [], {}, {}, [mod.QWEN_ID], iterations=3)[mod.QWEN_ID][mod.ARMS[0]][
            "n_bootstrap"
        ]
        == 3
    )
    assert mod._asymmetry_supported({mod.QWEN_ID: {mod.ARMS[0]: {"effect": 0.001}}}) is False
    assert (
        mod._blocked_reason(
            model_ok=False,
            gpu_ok=False,
            denominators_ok=False,
            replay_ok=False,
            parser_ok=False,
            truncation_ok=False,
        )
        == "blocked_missing_headline_gguf"
    )
    assert (
        mod._blocked_reason(
            model_ok=True,
            gpu_ok=True,
            denominators_ok=True,
            replay_ok=False,
            parser_ok=True,
            truncation_ok=True,
        )
        == "blocked_raw_response_replay_failed"
    )
    assert (
        mod._blocked_reason(
            model_ok=True,
            gpu_ok=True,
            denominators_ok=False,
            replay_ok=True,
            parser_ok=True,
            truncation_ok=True,
        )
        == "blocked_incomplete_panel_denominators"
    )
    assert mod._receipt_for_model({}, mod.QWEN_ID)["receipt_missing"] is True

    previous_bad = copy.deepcopy(rows)
    previous_bad[1]["previous_row_hash"] = "wrong"
    with pytest.raises(mod.EnvelopeReplayError, match="previous_row_hash"):
        mod.replay_response_envelope_rows(previous_bad)

    payload_shape_bad = copy.deepcopy(rows)
    payload_shape_bad[0]["raw_response_payload"] = None
    payload_shape_bad[0]["row_hash"] = mod.row_hash(payload_shape_bad[0])
    with pytest.raises(mod.EnvelopeReplayError, match="payload_decode"):
        mod.replay_response_envelope_rows(payload_shape_bad)

    prompt_hash_bad = copy.deepcopy(rows)
    prompt_hash_bad[0]["prompt_hash"] = "0" * 64
    prompt_hash_bad[0]["row_hash"] = mod.row_hash(prompt_hash_bad[0])
    with pytest.raises(mod.EnvelopeReplayError, match="prompt_hash"):
        mod.replay_response_envelope_rows(prompt_hash_bad)

    payload_hash_bad = copy.deepcopy(rows)
    payload_hash_bad[0]["raw_response_payload"] = mod.encode_lossless_payload(b'{"changed":true}')
    payload_hash_bad[0]["row_hash"] = mod.row_hash(payload_hash_bad[0])
    with pytest.raises(mod.EnvelopeReplayError, match="payload_hash"):
        mod.replay_response_envelope_rows(payload_hash_bad)
    assert mod._safe_replay(payload_hash_bad)["raw_response_replay_passed"] is False

    def assert_receipt_blocked(mutator: object) -> None:
        bad = copy.deepcopy(receipt)
        assert callable(mutator)
        mutator(bad)
        assert mod.gpu_offload_authenticated(bad, specs) is False

    assert_receipt_blocked(lambda bad: bad["model_receipts"].pop())
    assert_receipt_blocked(
        lambda bad: bad["model_receipts"][0].__setitem__("gpu_offload_authenticated", False)
    )
    assert_receipt_blocked(lambda bad: bad["model_receipts"][0].__setitem__("worker_ok", False))
    assert_receipt_blocked(
        lambda bad: bad["model_receipts"][0].__setitem__("llama_cpp_supports_gpu_offload", False)
    )
    assert_receipt_blocked(
        lambda bad: bad["model_receipts"][0].__setitem__("torch_cuda_available", False)
    )
    assert_receipt_blocked(
        lambda bad: bad["model_receipts"][0].__setitem__("torch_device_count", 0)
    )
    assert_receipt_blocked(
        lambda bad: bad["model_receipts"][0].__setitem__(
            "offloaded_layer_count_from_backend_log", 0
        )
    )
    assert_receipt_blocked(lambda bad: bad["model_receipts"][0].__setitem__("pid", 0))
    assert_receipt_blocked(lambda bad: bad["model_receipts"][0].pop("port"))
    assert_receipt_blocked(
        lambda bad: (
            bad["model_receipts"][0].__setitem__("pid_gpu_memory_mb_peak", 0),
            bad["model_receipts"][0].__setitem__("gpu_utilization_pct_peak", 0),
        )
    )
