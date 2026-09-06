"""Tests for the chat-correct three-family entrance bank.

Spec refs: REQ-INFRA-7086, REQ-VERIFY-7086, and all related scenarios.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carnot import experiment_7086_v621_three_family_entrance_bank as mod


def _specs(tmp_path: Path) -> list[dict[str, Any]]:
    rows = []
    for model_id in mod.REQUIRED_MODEL_IDS:
        path = tmp_path / f"{model_id.rsplit('/', 1)[-1]}.gguf"
        path.write_bytes(model_id.encode())
        rows.append(
            {
                "name": model_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
                "hf_id": model_id,
                "model_path": str(path),
                "gpu_indices": [0, 1],
                "preferred_quant": "Q4_K_M",
                "tokenizer_source": "embedded_gguf",
                "remote_allowed": False,
                "headline_eligible": True,
                "resolution_method": "cached_sota_pair test double",
            }
        )
    return rows


def _units(count: int = 96) -> list[dict[str, Any]]:
    return [
        {
            "unit_id": f"unit-{index:03d}",
            "source_group_id": f"group-{index // 8:02d}",
            "numbers": [1, 2, 3, 4, 5, 6],
            "target": 21,
        }
        for index in range(count)
    ]


def _entrances(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for unit in units:
        rows.extend(
            [
                {
                    "unit_id": unit["unit_id"],
                    "entrance_id": f"{unit['unit_id']}:1:2:add",
                    "operand_pair": [1, 2],
                    "operator": "+",
                    "left": 1,
                    "right": 2,
                    "result": 3,
                    "reachable": True,
                    "continuation_witness": [
                        {"left": 3, "right": 3, "operator": "+", "result": 6},
                        {"left": 4, "right": 5, "operator": "+", "result": 9},
                        {"left": 6, "right": 6, "operator": "+", "result": 12},
                        {"left": 9, "right": 12, "operator": "+", "result": 21},
                    ],
                },
                {
                    "unit_id": unit["unit_id"],
                    "entrance_id": f"{unit['unit_id']}:1:3:add",
                    "operand_pair": [1, 3],
                    "operator": "+",
                    "left": 1,
                    "right": 3,
                    "result": 4,
                    "reachable": True,
                    "continuation_witness": [],
                },
            ]
        )
    return rows


def _forced_ids(units: list[dict[str, Any]]) -> list[str]:
    return mod.select_diversity_unit_ids(units)


def _finish_raw(row: dict[str, Any], text: str) -> dict[str, Any]:
    result = deepcopy(row)
    result.update(
        {
            "transport_method": "create_chat_completion",
            "chat_format": "embedded_gguf_template",
            "chat_template_present": True,
            "chat_template_hash": mod.sha256_text("template"),
            "rendered_prompt_hash": mod.sha256_text("rendered:" + result["raw_key"]),
            "rendered_prompt_hash_available": True,
            "stop_config": [],
            "raw_text": text,
            "raw_bytes_hex": text.encode().hex(),
            "raw_output_hash": mod.sha256_text(text),
            "token_scores": [],
            "token_scores_available": False,
            "prompt_tokens": 20,
            "completion_tokens": 9,
            "prefix_token_count": 0,
            "requested_completion_budget_tokens": 192,
            "effective_completion_budget_tokens": 192,
            "finish_reason": "stop",
            "timings": {"duration_s": 0.01, "backend": {}},
            "terminal_state": "complete",
            "exception_type": None,
            "exception_message": None,
            "raw_persisted_before_parse": True,
            "parsed_at_write_time": False,
            "labeled_at_write_time": False,
            "model_close_called": False,
        }
    )
    return result


def _complete_rows(
    tmp_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    specs = _specs(tmp_path)
    units = _units()
    unit_ids = [row["unit_id"] for row in units]
    proposal_schedule = mod.build_proposal_schedule(specs, units, unit_ids)
    raw = [_finish_raw(row, '{"operand_pair":[1,2],"operator":"+"}') for row in proposal_schedule]
    proposals = mod.label_proposal_rows(raw, _entrances(units))
    forced_ids = _forced_ids(units)
    prefix_rows = mod.select_forced_prefixes(
        mod.REQUIRED_MODEL_IDS,
        units,
        _entrances(units),
        proposals,
        forced_ids,
    )
    forced_schedule = mod.build_forced_schedule(prefix_rows, specs)
    forced = []
    for row in forced_schedule:
        finished = _finish_raw(row, "[]")
        finished["prefix_token_count"] = 7
        finished["effective_completion_budget_tokens"] = 185
        forced.append(finished)
    forced = mod.label_forced_rows(forced, units)
    return specs, units, proposals, forced, unit_ids


def _preconditions(specs: list[dict[str, Any]], units: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [mod.gate_row("all_preconditions", True, True, True)]
    return {
        "all_passed": True,
        "checks": checks,
        "fixture": {"unit_rows": units},
        "fixture_hash": "sha256:" + "1" * 64,
        "lease_audit_hash": "sha256:" + "2" * 64,
        "chat_canary_hash": "sha256:" + "3" * 64,
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
        "runner_build_rows": [{"importable": True, "gpu_offload": True}],
        "runner_receipt": {
            "backend": "llama_cpp.Llama",
            "cuda_offload": True,
            "transport_method": "create_chat_completion",
        },
        "gpu_topology": {
            "devices": [
                {"index": 0, "uuid": "gpu-0", "name": "RTX 3090"},
                {"index": 1, "uuid": "gpu-1", "name": "RTX 3090"},
            ]
        },
        "upstream_gate_rows": checks,
        "signals_sent": [],
    }


def _evidence(
    specs: list[dict[str, Any]], proposal_count: int, forced_count: int
) -> dict[str, Any]:
    executions = []
    checkpoints = []
    leases = []
    releases = []
    stage_samples = []
    task_samples = []
    for phase, row_count in (("proposal", proposal_count), ("forced_prefix", forced_count)):
        for spec in specs:
            model_id = spec["hf_id"]
            executions.append(
                {
                    "model_id": model_id,
                    "phase": phase,
                    "terminal_state": "complete",
                    "raw_row_count": row_count,
                    "offloaded_layers": 49,
                    "total_layers": 49,
                    "used_both_gpus": True,
                    "cleanup_passed": True,
                    "model_load_count": 1,
                    "duration_s": 4.0,
                    "backend_stderr_hash": "sha256:" + "4" * 64,
                }
            )
            checkpoints.append(
                {
                    "model_id": model_id,
                    "phase": phase,
                    "path": f"/tmp/{phase}.json",
                    "sha256": "sha256:" + "5" * 64,
                    "manifest_hash": "sha256:" + "6" * 64,
                    "row_count": row_count,
                }
            )
            releases.append({"model_id": model_id, "phase": phase, "passed": True})
            stage_samples.append(
                {"model_id": model_id, "phase": phase, "gpu_uuid": "gpu-0", "used_mb": 9000}
            )
            task_samples.append(
                {
                    "model_id": model_id,
                    "phase": phase,
                    "devices": [
                        {"uuid": "gpu-0", "memory_used_mb": 9000},
                        {"uuid": "gpu-1", "memory_used_mb": 8000},
                    ],
                }
            )
            for device in ("gpu-0", "gpu-1"):
                leases.append(
                    {
                        "model_id": model_id,
                        "phase": phase,
                        "device_uuid": device,
                        "lease_id": f"{model_id}:{phase}:{device}",
                        "owner_preserved": True,
                        "phase_history": list(mod.COMPLETE_PHASE_SEQUENCE),
                        "released": True,
                        "lease_lost": False,
                        "signals_sent": [],
                    }
                )
    return {
        "model_execution_rows": executions,
        "checkpoint_rows": checkpoints,
        "gpu_lease_rows": leases,
        "vram_release_rows": releases,
        "stage_gpu_telemetry_rows": stage_samples,
        "task_gpu_telemetry_rows": task_samples,
        "peak_vram_by_device": {"gpu-0": 9000, "gpu-1": 8000},
        "cleanup_rows": [
            {
                "model_id": spec["hf_id"],
                "phase": phase,
                "process_owned": True,
                "owned_process_absent": True,
                "port_release_confirmed": True,
                "gpu_leases_released": True,
                "vram_release_passed": True,
                "signals_sent": [],
                "signaled_pid_owned": True,
                "passed": True,
            }
            for phase in ("proposal", "forced_prefix")
            for spec in specs
        ],
        "model_file_hash_rows": [
            {"model_id": spec["hf_id"], "path": spec["model_path"], "sha256": "sha256:" + "7" * 64}
            for spec in specs
        ],
        "runner_receipt": {
            "backend": "llama_cpp.Llama",
            "cuda_offload": True,
            "transport_method": "create_chat_completion",
        },
        "signals_sent": [],
    }


def _positive_artifact(tmp_path: Path) -> dict[str, Any]:
    specs, units, proposals, forced, unit_ids = _complete_rows(tmp_path)
    return mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=90.0,
        model_specs=specs,
        preconditions=_preconditions(specs, units),
        proposal_rows=proposals,
        forced_prefix_rows=forced,
        ordered_unit_ids=unit_ids,
        forced_unit_ids=_forced_ids(units),
        source_artifact_hashes={"manifest_hash": "sha256:" + "8" * 64},
        evidence=_evidence(specs, 96 * 4, 24),
    )


def test_req_7086_specs_precede_implementation() -> None:
    """REQ-INFRA-7086 and REQ-VERIFY-7086 exist before implementation."""

    root = Path(__file__).resolve().parents[2]
    inference = (root / "openspec/capabilities/llm-ebm-inference/spec.md").read_text()
    verification = (root / "openspec/capabilities/verification/spec.md").read_text()
    assert "REQ-INFRA-7086" in inference
    assert "SCENARIO-INFRA-7086-RAW-FIRST-RESUME" in inference
    assert "REQ-VERIFY-7086" in verification
    assert "SCENARIO-VERIFY-7086-MUTATION" in verification


def test_req_infra_7086_exact_cached_roster_and_no_fallback(tmp_path: Path) -> None:
    """SCENARIO-INFRA-7086-ROSTER-AND-TEMPLATE rejects roster substitutions."""

    pair = _specs(tmp_path)[:2]
    resolver_paths = {row["hf_id"]: row["model_path"] for row in _specs(tmp_path)}
    calls: list[dict[str, Any]] = []

    def cached_pair(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return pair

    rows = mod.resolve_model_specs(
        cached_pair_func=cached_pair,
        resolver=lambda model_id, _quant: resolver_paths[model_id],
    )
    assert calls == [{"gpu_indices": (0, 1), "model_indices": (0, 2)}]
    assert [row["hf_id"] for row in rows] == list(mod.REQUIRED_MODEL_IDS)
    assert not mod.model_spec_errors(rows)
    assert "model_path_missing:" + mod.REQUIRED_MODEL_IDS[0] in mod.model_spec_errors(
        [{**rows[0], "model_path": ""}, *rows[1:]]
    )
    assert "model_ids_mismatch" in mod.model_spec_errors(rows[::-1])
    assert "tokenizer_source_mismatch:" + rows[0]["hf_id"] in mod.model_spec_errors(
        [{**rows[0], "tokenizer_source": "transformers"}, *rows[1:]]
    )
    assert "headline_policy_mismatch:" + rows[0]["hf_id"] in mod.model_spec_errors(
        [{**rows[0], "remote_allowed": True}, *rows[1:]]
    )
    assert "model_path_not_primary_gguf:" + rows[0]["hf_id"] in mod.model_spec_errors(
        [{**rows[0], "model_path": "/tmp/mmproj.gguf"}, *rows[1:]]
    )


def test_req_infra_7086_complete_matched_schedule_and_prompt_drift(tmp_path: Path) -> None:
    """SCENARIO-INFRA-7086-PANEL-AND-RANDOMIZATION freezes all matched cells."""

    specs = _specs(tmp_path)
    units = _units()
    unit_ids = [row["unit_id"] for row in units]
    rows = mod.build_proposal_schedule(specs, units, unit_ids, seed=42)
    assert len(rows) == 3 * 96 * 4
    assert rows == mod.build_proposal_schedule(specs, units, unit_ids, seed=42)
    assert rows != mod.build_proposal_schedule(specs, units, unit_ids, seed=43)
    assert not mod.matched_schedule_errors(rows, unit_ids)
    prompt = rows[0]["prompt"]
    assert "source_group" not in prompt and "reachable" not in prompt and "witness" not in prompt
    assert rows[0]["role_messages"] == mod.build_role_messages(prompt)
    mutated = deepcopy(rows)
    mutated[0]["prompt"] += " drift"
    assert "prompt_hash_mismatch" in mod.matched_schedule_errors(mutated, unit_ids)
    mutated = deepcopy(rows)
    mutated[0]["role_messages"][0]["role"] = "user"
    assert "role_message_mismatch" in mod.matched_schedule_errors(mutated, unit_ids)
    mutated = deepcopy(rows)
    mutated[0]["generation_config"]["temperature"] = 0.9
    config_errors = mod.matched_schedule_errors(mutated, unit_ids)
    assert "generation_config_mismatch" in config_errors
    assert any(item.startswith("cross_model_budget_mismatch:") for item in config_errors)
    assert "schedule_key_set_mismatch" in mod.matched_schedule_errors(rows[:-1], unit_ids)


class _FakeLlama:
    def __init__(self, *, template: bool = True) -> None:
        self.metadata = {"tokenizer.chat_template": "{{ messages }}"} if template else {}
        self.chat_format = "embedded"
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def tokenize(self, value: bytes, add_bos: bool = False) -> list[int]:
        assert value and add_bos is False
        return [1, 2, 3]

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {"content": '{"operand_pair":[1,2],"operator":"+"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 9},
            "timings": {"predicted_per_second": 50.0},
        }

    def close(self) -> None:
        self.closed = True


def test_req_infra_7086_worker_uses_exact_chat_path_and_remaining_budget(tmp_path: Path) -> None:
    """SCENARIO-INFRA-7086-ROSTER-AND-TEMPLATE uses the approved chat call."""

    spec = _specs(tmp_path)[0]
    proposal = mod.build_proposal_schedule([spec], _units(1), ["unit-000"])[0]
    llm = _FakeLlama()
    raw = mod.worker_generate_one(proposal, llama_instance=llm, clock=iter((0, 1_000_000)).__next__)
    assert llm.calls[0]["max_tokens"] == 192
    assert llm.calls[0]["messages"] == proposal["role_messages"]
    assert llm.calls[0]["stop"] == []
    assert raw["raw_bytes_hex"] == raw["raw_text"].encode().hex()
    assert raw["token_scores"] == [] and raw["token_scores_available"] is False

    prefix = {
        "model_id": spec["hf_id"],
        "unit_id": "unit-000",
        "entrance_id": "e",
        "operand_pair": [1, 2],
        "operator": "+",
        "left": 1,
        "right": 2,
        "result": 3,
        "reachable": True,
        "initially_unselected": True,
        "prefix_json": '{"left":1,"operator":"+","result":3,"right":2}',
        "prefix_hash": mod.sha256_text("prefix"),
        "original_proposal_prompt": proposal["prompt"],
        "original_proposal_prompt_hash": proposal["prompt_hash"],
        "prompt": proposal["prompt"] + " forced",
        "prompt_hash": mod.sha256_text(proposal["prompt"] + " forced"),
    }
    forced = mod.build_forced_schedule([prefix], [spec])[0]
    forced_raw = mod.worker_generate_one(
        forced, llama_instance=llm, clock=iter((0, 1_000_000)).__next__
    )
    assert llm.calls[1]["max_tokens"] == 189
    assert forced_raw["prefix_token_count"] == 3
    assert forced_raw["requested_completion_budget_tokens"] == 192
    assert forced_raw["effective_completion_budget_tokens"] == 189

    failed = mod.worker_generate_one(
        proposal,
        llama_instance=_FakeLlama(template=False),
        clock=iter((0, 1_000_000)).__next__,
    )
    assert failed["terminal_state"] == "failed"
    assert failed["exception_message"] == "missing_or_empty_embedded_chat_template"


def test_req_infra_7086_raw_first_checkpoint_resume_and_drift(tmp_path: Path) -> None:
    """SCENARIO-INFRA-7086-RAW-FIRST-RESUME keeps durable immutable attempts."""

    spec = _specs(tmp_path)[0]
    schedule = mod.build_proposal_schedule([spec], _units(1), ["unit-000"])
    payload = {
        "model_id": spec["hf_id"],
        "model_path": spec["model_path"],
        "phase": "proposal",
        "schedule_rows": schedule,
        "raw_path": str(tmp_path / "raw.jsonl"),
        "checkpoint_path": str(tmp_path / "checkpoint.json"),
        "manifest_hash": mod.sha256_text("manifest"),
    }
    instances: list[_FakeLlama] = []

    def factory(**_kwargs: Any) -> _FakeLlama:
        instance = _FakeLlama()
        instances.append(instance)
        return instance

    first = mod.worker_run_schedule(payload, llama_factory=factory)
    assert first["row_count"] == 4
    assert len((tmp_path / "raw.jsonl").read_text().splitlines()) == 4
    second = mod.worker_run_schedule(payload, llama_factory=factory)
    assert second["row_count"] == 4
    assert len((tmp_path / "raw.jsonl").read_text().splitlines()) == 4
    assert all(instance.closed for instance in instances)
    with pytest.raises(ValueError, match="checkpoint_manifest_mismatch"):
        mod.load_checkpoint(tmp_path / "checkpoint.json", "wrong")
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    checkpoint["rows"].append(checkpoint["rows"][0])
    (tmp_path / "checkpoint.json").write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="checkpoint_duplicate_raw_key"):
        mod.load_checkpoint(tmp_path / "checkpoint.json", payload["manifest_hash"])


def test_req_infra_7086_worker_accepts_controller_payload_without_phase(tmp_path: Path) -> None:
    """SCENARIO-INFRA-7086-RAW-FIRST-RESUME covers the proven controller shape."""

    spec = _specs(tmp_path)[0]
    schedule = mod.build_proposal_schedule([spec], _units(1), ["unit-000"])
    payload = {
        "model_id": spec["hf_id"],
        "model_path": spec["model_path"],
        "schedule_rows": schedule,
        "raw_path": str(tmp_path / "raw.jsonl"),
        "checkpoint_path": str(tmp_path / "checkpoint.json"),
        "manifest_hash": mod.sha256_text("controller-manifest"),
    }
    result = mod.worker_run_schedule(payload, llama_factory=lambda **_kwargs: _FakeLlama())
    assert result["phase"] == "proposal"
    assert result["terminal_state"] == "complete"


def test_req_verify_7086_exact_labels_and_forced_prefixes_are_post_generation(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7086-RAW-AND-EXACT-SEPARATION keeps prompts label-free."""

    specs = _specs(tmp_path)
    units = _units(24)
    unit_ids = [row["unit_id"] for row in units]
    schedule = mod.build_proposal_schedule(specs, units, unit_ids)
    raw = [_finish_raw(row, '{"operand_pair":[1,2],"operator":"+"}') for row in schedule]
    raw_copy = deepcopy(raw)
    labeled = mod.label_proposal_rows(raw, _entrances(units))
    assert raw == raw_copy
    views = mod.proposal_evidence_rows(labeled, _entrances(units))
    assert all("legal" not in row and "reachable" not in row for row in views["raw_proposal_rows"])
    assert all("raw_text" not in row for row in views["exact_label_rows"])
    assert all(row["causal_witness"] is True for row in views["causal_witness_rows"])
    assert all(row["guess_without_witness"] is False for row in views["guess_without_witness_rows"])
    prefixes = mod.select_forced_prefixes(
        mod.REQUIRED_MODEL_IDS, units, _entrances(units), labeled, unit_ids
    )
    assert len(prefixes) == 72
    assert all(row["reachable"] and row["initially_unselected"] for row in prefixes)
    assert all("continuation_witness" not in row["prompt"] for row in prefixes)
    assert not mod.continuation_succeeds(units[0], prefixes[0], "not-json")
    with pytest.raises(ValueError, match="forced_diversity_unit_count"):
        mod.select_diversity_unit_ids(units[:1], count=2)
    assert (
        mod.select_forced_prefixes(
            [mod.REQUIRED_MODEL_IDS[0]],
            units,
            [row for row in _entrances(units) if row["operand_pair"] == [1, 2]],
            labeled,
            [unit_ids[0]],
            count_per_model=1,
        )
        == []
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("raw_text", "", "empty_output_rate_exceeded"),
        ("completion_tokens", 0, "zero_token_output"),
        ("raw_text", "<|eot_id|>", "leaked_control_token"),
        ("finish_reason", "length", "length_limited_output"),
    ],
)
def test_req_verify_7086_output_failures_are_family_local(
    tmp_path: Path, field: str, value: Any, error: str
) -> None:
    """SCENARIO-VERIFY-7086-OUTPUT-FAILURES rejects each transport defect."""

    specs, units, proposals, forced, unit_ids = _complete_rows(tmp_path)
    target_model = mod.REQUIRED_MODEL_IDS[0]
    for row in proposals:
        if row["model_id"] == target_model:
            row[field] = value
            if field == "raw_text":
                row["raw_bytes_hex"] = str(value).encode().hex()
                row["raw_output_hash"] = mod.sha256_text(str(value))
    errors = mod.completion_errors(
        proposal_rows=proposals,
        forced_prefix_rows=forced,
        ordered_unit_ids=unit_ids,
        forced_unit_ids=_forced_ids(units),
        unit_rows=units,
        model_specs=specs,
        identity_rows=_preconditions(specs, units)["model_identity_rows"],
        evidence=_evidence(specs, 96 * 4, 24),
    )
    assert any(item.startswith(error + ":" + target_model) for item in errors)


def test_req_infra_7086_lease_cleanup_and_telemetry_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-INFRA-7086-LEASE-LOSS-AND-CLEANUP rejects missing live receipts."""

    specs, units, proposals, forced, unit_ids = _complete_rows(tmp_path)
    base = _evidence(specs, 96 * 4, 24)
    mutations = [
        ("gpu_lease_rows", 0, "lease_identity_or_release_incomplete"),
        ("cleanup_rows", 0, "cleanup_incomplete"),
        ("model_execution_rows", 0, "model_execution_incomplete"),
        ("checkpoint_rows", 0, "checkpoint_incomplete"),
        ("vram_release_rows", 0, "vram_release_incomplete"),
        ("stage_gpu_telemetry_rows", 0, "stage_gpu_telemetry_incomplete"),
        ("task_gpu_telemetry_rows", 0, "task_gpu_telemetry_incomplete"),
    ]
    for name, index, expected in mutations:
        evidence = deepcopy(base)
        evidence[name].pop(index)
        errors = mod.completion_errors(
            proposal_rows=proposals,
            forced_prefix_rows=forced,
            ordered_unit_ids=unit_ids,
            forced_unit_ids=_forced_ids(units),
            unit_rows=units,
            model_specs=specs,
            identity_rows=_preconditions(specs, units)["model_identity_rows"],
            evidence=evidence,
        )
        assert expected in errors
    bad = deepcopy(base)
    bad["gpu_lease_rows"][0]["lease_lost"] = True
    bad["gpu_lease_rows"][0]["owner_preserved"] = False
    assert "lease_identity_or_release_incomplete" in mod.completion_errors(
        proposal_rows=proposals,
        forced_prefix_rows=forced,
        ordered_unit_ids=unit_ids,
        forced_unit_ids=_forced_ids(units),
        unit_rows=units,
        model_specs=specs,
        identity_rows=_preconditions(specs, units)["model_identity_rows"],
        evidence=bad,
    )
    assert mod.cleanup_passes(base["cleanup_rows"])
    bad_cleanup = deepcopy(base["cleanup_rows"])
    bad_cleanup[0]["signals_sent"] = ["SIGTERM"]
    bad_cleanup[0]["signaled_pid_owned"] = False
    assert not mod.cleanup_passes(bad_cleanup)


def test_req_verify_7086_completion_mutation_matrix_covers_transport_gates(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-7086 recomputes each raw, panel, runner, and prefix gate."""

    specs, units, proposals, forced, unit_ids = _complete_rows(tmp_path)
    identities = _preconditions(specs, units)["model_identity_rows"]
    complete_evidence = _evidence(specs, 96 * 4, 24)

    def errors_for(
        changed_proposals: list[dict[str, Any]] = proposals,
        changed_forced: list[dict[str, Any]] = forced,
        changed_units: list[dict[str, Any]] = units,
        changed_unit_ids: list[str] = unit_ids,
        changed_forced_ids: list[str] | None = None,
        changed_evidence: dict[str, Any] = complete_evidence,
    ) -> list[str]:
        return mod.completion_errors(
            proposal_rows=changed_proposals,
            forced_prefix_rows=changed_forced,
            ordered_unit_ids=changed_unit_ids,
            forced_unit_ids=(
                _forced_ids(units) if changed_forced_ids is None else changed_forced_ids
            ),
            unit_rows=changed_units,
            model_specs=specs,
            identity_rows=identities,
            evidence=changed_evidence,
        )

    assert "forced_prefix_key_set_mismatch" in errors_for(changed_forced=forced[:-1])
    assert "fixture_panel_mismatch" in errors_for(changed_unit_ids=unit_ids[:-1])
    assert "forced_source_group_coverage_mismatch" in errors_for(changed_forced_ids=unit_ids[:24])

    row_mutations = [
        ("raw_output_hash", "bad", "raw_terminal_or_hash_mismatch"),
        ("transport_method", "create_completion", "chat_transport_mismatch"),
        ("chat_template_present", False, "chat_template_receipt_missing"),
        ("role_messages", [], "role_message_mismatch"),
        ("prompt_hash", "bad", "prompt_hash_mismatch"),
        ("stop_config", ["stop"], "stop_config_mismatch"),
        ("generation_config", {}, "sampling_config_mismatch"),
        ("requested_completion_budget_tokens", 191, "completion_budget_mismatch"),
    ]
    for field, value, expected in row_mutations:
        changed = deepcopy(proposals)
        changed[0][field] = value
        assert expected in errors_for(changed_proposals=changed)

    changed_forced = deepcopy(forced)
    changed_forced[0]["initially_unselected"] = False
    assert "forced_prefix_contract_mismatch" in errors_for(changed_forced=changed_forced)

    evidence_mutations = [
        ("model_file_hash_rows", [], "model_file_hash_incomplete"),
        ("peak_vram_by_device", {}, "peak_vram_incomplete"),
        ("runner_receipt", {}, "runner_receipt_incomplete"),
        ("signals_sent", ["SIGTERM"], "unexpected_signal"),
    ]
    for field, value, expected in evidence_mutations:
        changed_evidence = deepcopy(complete_evidence)
        changed_evidence[field] = value
        assert expected in errors_for(changed_evidence=changed_evidence)


def test_req_infra_7086_preconditions_include_canary_and_every_blocked_path(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-7086-BLOCKED checks canary hash, score, and base gates."""

    specs = _specs(tmp_path)
    canary_path = tmp_path / "canary.json"
    canary_path.write_text(json.dumps({"chat_transport_ready_score": 1}))
    expected_hash = mod.sha256_file(canary_path)
    base = {
        "all_passed": True,
        "checks": [mod.gate_row("base", True, True, True)],
        "upstream_gate_rows": [],
    }
    result = mod.collect_preconditions(
        chat_canary_path=canary_path,
        expected_chat_canary_hash=expected_hash,
        chat_canary_validator=lambda _row: [],
        base_collector=lambda **_kwargs: deepcopy(base),
        fixture_path=tmp_path / "fixture.json",
        expected_fixture_hash="fixture",
        model_specs=specs,
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    assert result["all_passed"] is True
    assert [row["check"] for row in result["checks"][:2]] == [
        "chat_transport_source_hash",
        "chat_transport_ready_score",
    ]

    for check_index in range(len(result["checks"])):
        blocked = deepcopy(result)
        blocked["checks"][check_index]["passed"] = False
        blocked["all_passed"] = False
        artifact = mod.build_artifact(
            run_date=mod.RUN_DATE,
            duration_s=1.0,
            model_specs=specs,
            preconditions=blocked,
        )
        assert artifact["verdict_class"] == "blocked"
        assert artifact["inference_substrate_class"] == "blocked_no_run"
        assert artifact["generation_invoked"] is False
        assert (
            artifact["gate_check_summary"]["failed_check"]
            == blocked["checks"][check_index]["check"]
        )

    missing = mod.collect_preconditions(
        chat_canary_path=tmp_path / "missing.json",
        expected_chat_canary_hash=expected_hash,
        chat_canary_validator=lambda _row: [],
        base_collector=lambda **_kwargs: deepcopy(base),
        fixture_path=tmp_path / "fixture.json",
        expected_fixture_hash="fixture",
        model_specs=specs,
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    assert missing["all_passed"] is False
    assert "read_error" in missing["chat_canary"]


def test_req_verify_7086_positive_artifact_is_quality_blind_and_complete(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7086-COMPLETENESS separates transport from arithmetic quality."""

    artifact = _positive_artifact(tmp_path)
    assert artifact["entrance_proposal_bank_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["inference_substrate_class"] == "model_full_generation"
    assert artifact["generation_invoked"] is True
    assert artifact["total_model_count"] == 3
    assert artifact["model_load_count_by_stage"] == {"proposal": 3, "forced_prefix": 3}
    assert set(artifact["peak_vram_by_device"]) == {"gpu-0", "gpu-1"}
    assert len(artifact["per_source_group_rows"]) == 12
    assert all(row["continuation_success"] is False for row in artifact["forced_prefix_rows"])
    assert not mod.validate_artifact(artifact)

    specs, units, proposals, forced, unit_ids = _complete_rows(tmp_path)
    forced[0]["token_scores"] = [
        {"token": "[", "logprob": -0.1, "text_offset": 0, "top_logprobs_hash": None}
    ]
    scored = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=90.0,
        model_specs=specs,
        preconditions=_preconditions(specs, units),
        proposal_rows=proposals,
        forced_prefix_rows=forced,
        ordered_unit_ids=unit_ids,
        forced_unit_ids=_forced_ids(units),
        evidence=_evidence(specs, 96 * 4, 24),
    )
    assert any(row["arm"] == "forced_prefix" for row in scored["token_score_rows"])


def test_req_verify_7086_complete_transport_failure_builds_valid_null(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7086-OUTPUT-FAILURES preserves a complete null bank."""

    specs, units, proposals, forced, unit_ids = _complete_rows(tmp_path)
    for row in proposals:
        if row["model_id"] == mod.REQUIRED_MODEL_IDS[0]:
            row["raw_text"] = ""
            row["raw_bytes_hex"] = ""
            row["raw_output_hash"] = mod.sha256_text("")
    artifact = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=90.0,
        model_specs=specs,
        preconditions=_preconditions(specs, units),
        proposal_rows=proposals,
        forced_prefix_rows=forced,
        ordered_unit_ids=unit_ids,
        forced_unit_ids=_forced_ids(units),
        evidence=_evidence(specs, 96 * 4, 24),
    )
    assert artifact["verdict_class"] == "null"
    assert artifact["entrance_proposal_bank_complete_score"] == 0
    assert not mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("prompt_hash", "bad", "prompt_hash_mismatch"),
        ("reproducibility_checksum", "bad", "reproducibility_checksum_mismatch"),
        ("inference_substrate", "replay", "inference_substrate_mismatch"),
        ("verifier_is_oracle", True, "verifier_is_oracle_mismatch"),
        ("entrance_proposal_bank_complete_score", True, "completion_score_not_bare_int"),
        ("honest_verdict", "wrong", "honest_verdict_prefix_mismatch"),
    ],
)
def test_req_verify_7086_cold_validator_rejects_terminal_mutations(
    tmp_path: Path, field: str, value: Any, expected: str
) -> None:
    """SCENARIO-VERIFY-7086-MUTATION rejects metadata and checksum drift."""

    artifact = _positive_artifact(tmp_path)
    artifact[field] = value
    assert expected in mod.validate_artifact(artifact)


def test_req_verify_7086_validator_rejects_raw_projection_and_partial_run(tmp_path: Path) -> None:
    """REQ-VERIFY-7086 rejects raw drift and classifies incomplete launched data."""

    artifact = _positive_artifact(tmp_path)
    artifact["raw_proposal_rows"][0]["raw_output_hash"] = "bad"
    assert "raw_proposal_projection_mismatch" in mod.validate_artifact(artifact)

    specs, units, proposals, forced, unit_ids = _complete_rows(tmp_path)
    partial = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=80.0,
        model_specs=specs,
        preconditions=_preconditions(specs, units),
        proposal_rows=proposals[:-1],
        forced_prefix_rows=forced,
        ordered_unit_ids=unit_ids,
        forced_unit_ids=_forced_ids(units),
        evidence=_evidence(specs, 96 * 4, 24),
    )
    assert partial["entrance_proposal_bank_complete_score"] == 0
    assert partial["verdict_class"] == "partial"
    assert not mod.validate_artifact(partial)


def test_req_verify_7086_validator_full_mutation_matrix(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7086-MUTATION reaches every cold metadata defense."""

    original = _positive_artifact(tmp_path)
    mutations = [
        ("field_principles", {}, "field_principles_mismatch"),
        ("schema", "wrong", "schema_mismatch"),
        ("run_date", "20260907", "run_date_mismatch"),
        ("MODEL_SPECS", [], "model_specs_projection_mismatch"),
        ("all_models_real", False, "all_models_real_mismatch"),
        ("empty_output_rate_by_model", {}, "empty_output_rate_by_model_mismatch"),
        ("generation_invoked", False, "generation_invoked_mismatch"),
        ("total_model_count", 2, "total_model_count_mismatch"),
        ("inference_substrate_class", "blocked_no_run", "inference_substrate_class_mismatch"),
        ("model_load_count_by_stage", {}, "model_load_count_projection_mismatch"),
        ("per_model_duration_s", {}, "per_model_duration_projection_mismatch"),
        ("verdict_class", "invalid", "verdict_class_invalid"),
    ]
    for field, value, expected in mutations:
        artifact = deepcopy(original)
        artifact[field] = value
        assert expected in mod.validate_artifact(artifact)

    artifact = deepcopy(original)
    artifact["entrance_proposal_bank_complete_score"] = 0
    assert "completion_score_mismatch" in mod.validate_artifact(artifact)
    artifact = deepcopy(original)
    artifact["raw_proposal_rows"][0]["legal"] = True
    assert "raw_row_contains_exact_label" in mod.validate_artifact(artifact)
    artifact = deepcopy(original)
    del artifact["runner_receipt"]
    assert "missing_field:runner_receipt" in mod.validate_artifact(artifact)

    artifact = deepcopy(original)
    artifact["verdict_class"] = "null"
    artifact["honest_verdict"] = "null: wrong positive class"
    assert "positive_verdict_mismatch" in mod.validate_artifact(artifact)

    specs, units, proposals, forced, unit_ids = _complete_rows(tmp_path)
    partial = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=80.0,
        model_specs=specs,
        preconditions=_preconditions(specs, units),
        proposal_rows=proposals[:-1],
        forced_prefix_rows=forced,
        ordered_unit_ids=unit_ids,
        forced_unit_ids=_forced_ids(units),
        evidence=_evidence(specs, 96 * 4, 24),
    )
    partial["verdict_class"] = "null"
    partial["honest_verdict"] = "null: wrong partial class"
    assert "partial_verdict_mismatch" in mod.validate_artifact(partial)

    blocked = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=1.0,
        model_specs=specs,
        preconditions={
            **_preconditions(specs, units),
            "all_passed": False,
            "checks": [mod.gate_row("blocked", True, False, False)],
        },
    )
    assert not mod.validate_artifact(blocked)
    blocked["verdict_class"] = "partial"
    blocked["honest_verdict"] = "partial: wrong blocked class"
    blocked["gate_check_summary"] = {}
    blocked_errors = mod.validate_artifact(blocked)
    assert "blocked_verdict_mismatch" in blocked_errors
    assert "blocked_gate_summary_incomplete" in blocked_errors

    null_artifact = deepcopy(original)
    null_artifact["runner_receipt"] = {}
    null_artifact["entrance_proposal_bank_complete_score"] = 0
    null_artifact["verdict_class"] = "positive"
    null_artifact["honest_verdict"] = "positive: wrong null class"
    assert "null_verdict_mismatch" in mod.validate_artifact(null_artifact)


def test_req_infra_7086_identity_template_gate_and_gate_summary(tmp_path: Path) -> None:
    """REQ-INFRA-7086 rejects embedded identity or template mismatch exactly."""

    specs = _specs(tmp_path)
    good = _preconditions(specs, _units())["model_identity_rows"]
    assert not mod.model_identity_errors(specs, good)
    bad = deepcopy(good)
    bad[0]["identity_matches"] = False
    bad[1]["chat_template_present"] = False
    errors = mod.model_identity_errors(specs, bad)
    assert "model_identity_mismatch:" + specs[0]["hf_id"] in errors
    assert "model_chat_template_missing:" + specs[1]["hf_id"] in errors
    summary = mod.gate_summary(
        [
            mod.gate_row("first", 1, 1, True),
            mod.gate_row("second", 1, 0, False),
        ]
    )
    assert summary["failed_check"] == "second"
    assert summary["expected_value"] == 1 and summary["observed_value"] == 0


def test_req_infra_7086_phase_projection_preserves_runner_loads_and_peaks(tmp_path: Path) -> None:
    """REQ-INFRA-7086 projects complete task and stage telemetry."""

    specs = _specs(tmp_path)
    preconditions = _preconditions(specs, _units())
    phases = [
        {
            "model_id": row["model_id"],
            "phase": row["phase"],
            "terminal_state": row["terminal_state"],
            "raw_rows": [{}] * row["raw_row_count"],
            "offloaded_layers": row["offloaded_layers"],
            "total_layers": row["total_layers"],
            "used_both_gpus": row["used_both_gpus"],
            "cleanup": {
                "model_id": row["model_id"],
                "phase": row["phase"],
                "passed": True,
                "process_owned": True,
                "owned_process_absent": True,
                "port_release_confirmed": True,
                "gpu_leases_released": True,
                "vram_release_passed": True,
                "signals_sent": [],
                "signaled_pid_owned": True,
            },
            "model_load_count": 1,
            "duration_s": 4.0,
            "backend_stderr_hash": row["backend_stderr_hash"],
            "checkpoint_path": "/tmp/checkpoint.json",
            "checkpoint_sha256": "sha256:" + "5" * 64,
            "manifest_hash": "sha256:" + "6" * 64,
            "vram_release": {"passed": True},
            "gpu_sample_rows": [
                {"gpu_uuid": "gpu-0", "used_mb": 9000},
                {"gpu_uuid": "gpu-1", "used_mb": 8000},
            ],
            "task_gpu_samples": [
                {
                    "devices": [
                        {"uuid": "gpu-0", "memory_used_mb": 9000},
                        {"uuid": "gpu-1", "memory_used_mb": 8000},
                    ]
                }
            ],
            "gpu_lease_rows": [],
        }
        for row in _evidence(specs, 384, 24)["model_execution_rows"]
    ]
    facts = mod.phase_evidence(phases, preconditions)
    assert len(facts["model_execution_rows"]) == 6
    assert facts["peak_vram_by_device"] == {"gpu-0": 9000, "gpu-1": 8000}
    assert facts["runner_receipt"]["transport_method"] == "create_chat_completion"


def test_req_infra_7086_unattributed_resource_never_grants_signal_authority() -> None:
    """SCENARIO-INFRA-7086-BLOCKED keeps foreign-resource signals empty."""

    row = mod.unattributed_resource_gate("gpu_process", [{"pid": 91}])
    assert row["passed"] is False
    assert row["observed_value"]["signals_sent"] == []
    assert mod.randomized_phase_order(123) == mod.randomized_phase_order(123)
    assert mod.randomized_phase_order(123) != mod.randomized_phase_order(124)
