"""Focused tests for frozen-Qwen verifier-balanced strategy memory.

Spec refs: REQ-SELF-7142 and SCENARIO-SELF-7142-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7142_v627_flowbalance_memory_csl as mod


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"


def _event(index: int, split: str = "future") -> dict[str, object]:
    """Build one label-free event with the same causal surface as Exp7141."""

    prompt = f'Return only JSON for instance {index}: {{"answer":{index}}}'
    row: dict[str, object] = {
        "event_id": f"event-{index}",
        "chronology_index": index,
        "event_content_hash": mod.sha256_text(f"event-{index}"),
        "instance_id": f"instance-{index}",
        "base_id": f"base-{index // 2}",
        "constraint_family": "sat_logic" if index % 2 == 0 else "graph_coloring",
        "hardness": "medium",
        "variant_kind": "canonical",
        "split": split,
        "prompt": prompt,
    }
    return row


def _events() -> list[dict[str, object]]:
    """Return one event per split plus enough future pairs for metrics."""

    return [
        _event(0, "past"),
        _event(1, "adaptation"),
        _event(2, "future"),
        _event(3, "future"),
        _event(4, "protected_retention"),
    ]


def _model_specs(tmp_path: Path) -> list[dict[str, object]]:
    """Return a cached-pair-shaped fixture with an exact local Q4 file."""

    qwen = tmp_path / "snapshots" / "revision-1" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    qwen.parent.mkdir(parents=True)
    qwen.write_bytes(b"qwen-test-weights")
    dense = tmp_path / "gemma-4-31B-it-Q4_K_M.gguf"
    dense.write_bytes(b"dense-test-weights")
    return [
        {"name": "Qwen3.6-35B-A3B", "hf_id": QWEN, "gpu": 0, "model_path": str(qwen)},
        {
            "name": "Gemma4-31B-it",
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "gpu": 1,
            "model_path": str(dense),
        },
    ]


def _preflight(specs: list[dict[str, object]]) -> dict[str, object]:
    """Return a successful external-precondition receipt for the CPU seam."""

    path = Path(str(specs[0]["model_path"]))
    return {
        "all_passed": True,
        "checks": [mod.gate_row("test_preflight", True, True, True)],
        "model_identity_rows": [
            {
                "model_id": QWEN,
                "model_path": str(path),
                "revision": path.parent.name,
                "model_sha256": mod.sha256_path(path),
                "backend": "fake_llama_cpp_cuda",
                "device": [0, 1],
                "chat_template_source": "embedded_gguf",
            }
        ],
        "model_load_receipts": [
            {
                "request_kind": "preflight_smoke",
                "terminal_state": "complete",
                "response_hash": mod.sha256_text("smoke"),
            }
        ],
    }


def _model_call(*, event: dict[str, object], arm: str, prompt: str, seed: int, max_tokens: int) -> dict[str, object]:
    """Return deterministic model-shaped output with controlled paired effects."""

    del prompt, seed, max_tokens
    index = int(event["chronology_index"])
    success = arm != "verifier_balanced_strategy_memory" or index in {0, 2, 4}
    if arm == "no_memory":
        success = index not in {2}
    return {
        "raw_text": json.dumps({"success": success, "event": index}),
        "reasoning_text": "",
        "prompt_tokens": 32,
        "completion_tokens": 8,
        "duration_s": 0.01,
        "terminal_state": "complete",
    }


def _score(event: dict[str, object], response: dict[str, object]) -> dict[str, object]:
    """Act as the delayed exact checker without inspecting the model prompt."""

    del event
    parsed = json.loads(str(response["raw_text"]))
    return {
        "exact_success": bool(parsed["success"]),
        "parse_success": True,
        "failed_constraint_classes": [] if parsed["success"] else ["fixture_failure"],
    }


def _complete_artifact(tmp_path: Path) -> dict[str, object]:
    """Run the injected seam so validation tests start from complete rows."""

    specs = _model_specs(tmp_path)
    source = tmp_path / "source.json"
    source.write_text('{"csl_event_stream_ready_score":1}', encoding="utf-8")
    return mod.run_experiment(
        run_date="20260908",
        result_path=tmp_path / "result.json",
        transaction_root=tmp_path / "transactions",
        source_path=source,
        events=_events(),
        model_specs=mod.resolve_model_specs(cached_pair_func=lambda **_: specs),
        preflight_func=lambda **_: _preflight(specs),
        model_call=_model_call,
        exact_score=_score,
        expected_event_count=5,
    )


def test_req_self_7142_model_spec_resolves_cached_qwen_q4(tmp_path: Path) -> None:
    """REQ-SELF-7142 binds the run to cached Qwen Q4 and embedded chat."""

    specs = mod.resolve_model_specs(cached_pair_func=lambda **_: _model_specs(tmp_path))
    assert specs[0]["hf_id"] == QWEN
    assert specs[0]["quantization"] == "Q4_K_M"
    assert specs[0]["chat_template_source"] == "embedded_gguf"
    assert specs[0]["revision"] == "revision-1"
    assert specs[0]["model_sha256"] == mod.sha256_path(Path(specs[0]["model_path"]))
    assert specs[0]["remote_allowed"] is False
    with pytest.raises(mod.PreconditionError, match="cached_qwen_q4"):
        mod.resolve_model_specs(cached_pair_func=lambda **_: [])


def test_scenario_self_7142_initialize_blocks_before_preflight(tmp_path: Path) -> None:
    """SCENARIO-SELF-7142-INITIALIZE writes all fields before a failed gate."""

    result = tmp_path / "blocked.json"
    initialized = mod.initialize_artifact(result, "20260908")
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(initialized)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(initialized["field_principles"])
    called = False

    def forbidden_preflight(**_: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("preflight must not follow a failed source gate")

    source = tmp_path / "source.json"
    source.write_text('{"csl_event_stream_ready_score":0}', encoding="utf-8")
    artifact = mod.run_experiment(
        run_date="20260908",
        result_path=result,
        transaction_root=tmp_path / "transactions",
        source_path=source,
        events=[],
        model_specs=[deepcopy(mod.MODEL_SPECS[0])],
        preflight_func=forbidden_preflight,
        model_call=_model_call,
        exact_score=_score,
        expected_event_count=5,
    )
    assert called is False
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["failed_check"] == (
        "experiment_7141_v627_csl_event_stream.csl_event_stream_ready_score"
    )
    assert artifact["gate_check_summary"]["expected_value"] == 1
    assert artifact["gate_check_summary"]["observed_value"] == 0
    assert mod.validate_artifact(artifact, expected_event_count=5) == []


def test_scenario_self_7142_budget_and_reveal_are_frozen() -> None:
    """SCENARIO-SELF-7142-BUDGET and -REVEAL reject hindsight and drift."""

    event = _event(2)
    memories = {arm: "different guidance" for arm in mod.ARMS}
    prompt_rows = mod.build_prompt_rows(event, memories)
    assert {row["prompt_bytes"] for row in prompt_rows} == {mod.PROMPT_BUDGET_BYTES}
    assert {row["memory_bytes"] for row in prompt_rows} == {mod.MEMORY_BUDGET_BYTES}
    assert {row["max_tokens"] for row in prompt_rows} == {mod.MAX_GENERATION_TOKENS}
    assert len({row["seed"] for row in prompt_rows}) == 1
    with pytest.raises(mod.ProtocolError, match="outcome_leakage"):
        mod.build_prompt_rows({**event, "exact_success": True}, memories)
    responses = {
        arm: {"raw_text": arm, "terminal_state": "complete"} for arm in mod.ARMS
    }
    receipts = mod.seal_action_receipts(event, prompt_rows, responses)
    assert len(receipts) == 4
    assert mod.reveal_outcomes(event, responses, receipts, _score)[0]["event_id"] == "event-2"
    with pytest.raises(mod.ProtocolError, match="all_action_receipts_required"):
        mod.reveal_outcomes(event, responses, receipts[:-1], _score)


def test_scenario_self_7142_flowbalance_signed_rule() -> None:
    """SCENARIO-SELF-7142-FLOWBALANCE retains, reverses, or abstains."""

    candidate = {"strategy": "check every clause", "family": "sat_logic"}
    positive = mod.flowbalance_update(candidate, balanced_success=True, baseline_success=False)
    negative = mod.flowbalance_update(candidate, balanced_success=False, baseline_success=True)
    tie = mod.flowbalance_update(candidate, balanced_success=True, baseline_success=True)
    missing = mod.flowbalance_update(candidate, balanced_success=None, baseline_success=True)
    assert positive["advantage"] == 1 and positive["direction"] == "retain"
    assert negative["advantage"] == -1 and negative["direction"] == "reverse"
    assert negative["record"]["strategy"].startswith("REVERSE:")
    assert tie["write"] is False and tie["direction"] == "abstain"
    assert missing["write"] is False and missing["preference"] == "missing"


def test_scenario_self_7142_transaction_conflict_rollback_and_refresh(tmp_path: Path) -> None:
    """SCENARIO-SELF-7142-TRANSACTION and -REFRESH keep state atomic."""

    store = mod.TransactionalMemory(tmp_path / "store", "balanced", max_records=1)
    first = mod.sign_memory_record(
        arm="verifier_balanced_strategy_memory",
        event=_event(0, "adaptation"),
        payload={"strategy": "first", "family": "sat_logic"},
        admitted_for_event_index=1,
        direction="retain",
    )
    receipt = store.commit(first, current_event_index=1)
    assert receipt["terminal_state"] == "committed"
    assert mod.signature_valid(first)
    forged = deepcopy(first)
    forged["payload"]["strategy"] = "forged"
    assert mod.signature_valid(forged) is False
    second = mod.sign_memory_record(
        arm="verifier_balanced_strategy_memory",
        event=_event(1, "adaptation"),
        payload={"strategy": "second", "family": "sat_logic"},
        admitted_for_event_index=2,
        direction="retain",
    )
    conflict = store.commit(second, current_event_index=2)
    assert conflict["conflict"]["resolution"] == "replace_older_same_family"
    before = store.state_hash()
    failed = mod.sign_memory_record(
        arm="verifier_balanced_strategy_memory",
        event=_event(2, "future"),
        payload={"strategy": "third", "family": "graph_coloring"},
        admitted_for_event_index=3,
        direction="reverse",
    )
    rollback = store.commit(failed, current_event_index=3, force_failure=True)
    assert rollback["terminal_state"] == "rolled_back"
    assert rollback["parent_restored"] is True
    assert store.state_hash() == before
    visible = store.retrieve("sat_logic", current_event_index=3)
    assert visible[0]["source_event_index"] < 3
    refresh = store.refresh(boundary_event_index=3)
    assert refresh["record_count_after"] <= 1
    with pytest.raises(mod.ProtocolError, match="same_event_write"):
        store.commit(failed, current_event_index=2)


def test_scenario_self_7142_reduce_and_verdict_are_row_derived(tmp_path: Path) -> None:
    """SCENARIO-SELF-7142-REDUCE and -VERDICT separate completion from uplift."""

    artifact = _complete_artifact(tmp_path)
    assert artifact["flowbalance_memory_csl_complete_score"] == 1
    assert len(artifact["event_rows"]) == 5 * len(mod.ARMS)
    assert len(artifact["action_receipt_rows"]) == 5 * len(mod.ARMS)
    assert len(artifact["outcome_reveal_rows"]) == 5 * len(mod.ARMS)
    assert artifact["model_weights_changed"] is False
    assert artifact["flowbalance_training_reproduction"] is False
    assert artifact["verdict_class"] in {"positive", "null"}
    assert artifact["honest_verdict"].startswith(artifact["verdict_class"])
    assert len(artifact["paired_interval_rows"]) == len(mod.ARMS) - 1
    assert {row["arm"] for row in artifact["future_success_rows"]} == set(mod.ARMS)
    assert mod.validate_artifact(artifact, expected_event_count=5) == []


def test_req_self_7142_validator_rejects_receipt_budget_and_verdict_drift(tmp_path: Path) -> None:
    """REQ-SELF-7142 cold validation recomputes the safety-critical claims."""

    artifact = _complete_artifact(tmp_path)

    def errors(change: object) -> list[str]:
        changed = deepcopy(artifact)
        change(changed)
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        return mod.validate_artifact(changed, expected_event_count=5)

    assert "action_receipt_hash_mismatch" in errors(
        lambda value: value["action_receipt_rows"][0].__setitem__("receipt_hash", "bad")
    )
    assert "prompt_budget_mismatch" in errors(
        lambda value: value["event_rows"][0].__setitem__("prompt_bytes", 1)
    )
    assert "transaction_not_terminal" in errors(
        lambda value: value["transaction_rows"][0].__setitem__("terminal_state", "prepared")
    )
    assert "completion_score_mismatch" in errors(
        lambda value: value.__setitem__("flowbalance_memory_csl_complete_score", 0)
    )
    assert "model_weights_changed" in errors(
        lambda value: value.__setitem__("model_weights_changed", True)
    )
    assert "verdict_class_mismatch" in errors(
        lambda value: value.__setitem__("verdict_class", "circular_positive")
    )


def test_req_self_7142_atomic_writer_and_checksum(tmp_path: Path) -> None:
    """REQ-SELF-7142 publishes only complete JSON and detects later mutation."""

    path = tmp_path / "nested" / "artifact.json"
    mod.write_json_atomic(path, {"complete": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": True}
    artifact = _complete_artifact(tmp_path / "run")
    assert artifact["reproducibility_checksum"] == mod.artifact_checksum(artifact)
    artifact["duration_s"] = 99.0
    assert artifact["reproducibility_checksum"] != mod.artifact_checksum(artifact)
