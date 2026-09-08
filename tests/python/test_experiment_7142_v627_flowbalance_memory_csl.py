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


def _model_call(
    *, event: dict[str, object], arm: str, prompt: str, seed: int, max_tokens: int
) -> dict[str, object]:
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
    responses = {arm: {"raw_text": arm, "terminal_state": "complete"} for arm in mod.ARMS}
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


def test_scenario_self_7142_row_consistency_replays_receipts_updates_and_metrics(
    tmp_path: Path,
) -> None:
    """SCENARIO-SELF-7142-REDUCE cold-replays action, outcome, update, and metric rows."""

    artifact = _complete_artifact(tmp_path)

    def errors(change: object) -> list[str]:
        changed = deepcopy(artifact)
        change(changed)
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        return mod.validate_artifact(changed, expected_event_count=5)

    assert "event_action_link_mismatch" in errors(
        lambda value: value["event_rows"][0].__setitem__("response_hash", "bad")
    )
    assert "outcome_receipt_hash_mismatch" in errors(
        lambda value: value["outcome_reveal_rows"][0].__setitem__("outcome_receipt_hash", "bad")
    )
    assert "event_outcome_link_mismatch" in errors(
        lambda value: value["event_rows"][0].__setitem__("exact_success", None)
    )
    assert "advantage_rule_mismatch" in errors(
        lambda value: value["advantage_rows"][0].__setitem__("advantage", 99)
    )
    assert "reduced_metric_rows_mismatch" in errors(
        lambda value: value["future_success_rows"][0].__setitem__("exact_success_count", 99)
    )
    assert "future_uplift_score_mismatch" in errors(
        lambda value: value.__setitem__(
            "future_uplift_supported_score",
            1 - int(value["future_uplift_supported_score"]),
        )
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


def test_req_self_7142_protocol_edges_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-SELF-7142 covers malformed sources, budgets, signatures, and atomic cleanup."""

    assert mod.sha256_path(tmp_path / "missing") is None
    assert mod._paired_interval([]) == (None, None, None)
    assert mod._paired_interval([1]) == (1.0, 1.0, 1.0)
    fitted = mod._fit_bytes("x" * 7 + "é", 8)
    assert len(fitted.encode("utf-8")) == 8
    assert mod._nested_keys([{"exact_success": True}]) >= {"exact_success"}

    event = _event(0)
    with pytest.raises(mod.ProtocolError, match="arm_memory_roster_mismatch"):
        mod.build_prompt_rows(event, {"no_memory": ""})
    prompts = mod.build_prompt_rows(event, {arm: "" for arm in mod.ARMS})
    responses = {arm: {"raw_text": "{}", "terminal_state": "complete"} for arm in mod.ARMS}
    with pytest.raises(mod.ProtocolError, match="all_action_receipts_required"):
        mod.seal_action_receipts(event, prompts[:-1], responses)
    receipts = mod.seal_action_receipts(event, prompts, responses)
    receipts[0]["receipt_hash"] = "bad"
    with pytest.raises(mod.ProtocolError, match="action_receipt_invalid"):
        mod.reveal_outcomes(event, responses, receipts, _score)

    store = mod.TransactionalMemory(tmp_path / "store", "edges")
    signed = mod.sign_memory_record(
        arm="delayed_procedural_memory",
        event=event,
        payload={"family": "sat_logic", "strategy": "check"},
        admitted_for_event_index=1,
        direction="retain",
    )
    forged = deepcopy(signed)
    forged["signature"] = "bad"
    with pytest.raises(mod.ProtocolError, match="memory_signature_invalid"):
        store.commit(forged, current_event_index=1)
    with pytest.raises(mod.ProtocolError, match="admission_index_mismatch"):
        store.commit(signed, current_event_index=2)

    wrong_name = tmp_path / "weights.gguf"
    wrong_name.write_bytes(b"weights")
    with pytest.raises(mod.PreconditionError, match="cached_qwen_q4"):
        mod.resolve_model_specs(
            cached_pair_func=lambda **_: [{"hf_id": QWEN, "model_path": str(wrong_name)}]
        )

    with pytest.raises(mod.PreconditionError, match="exists"):
        mod._load_source(tmp_path / "absent.json")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(mod.PreconditionError, match="valid_json"):
        mod._load_source(malformed)
    wrong_type = tmp_path / "wrong-type.json"
    wrong_type.write_text("[]", encoding="utf-8")
    with pytest.raises(mod.PreconditionError, match="valid_json"):
        mod._load_source(wrong_type)

    destination = tmp_path / "atomic" / "result.json"
    destination.parent.mkdir()

    def fail_replace(_self: Path, _target: Path) -> Path:
        raise OSError("forced replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="forced replace failure"):
        mod.write_json_atomic(destination, {"complete": True})
    assert list(destination.parent.iterdir()) == []


def test_scenario_self_7142_preflight_failures_stay_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-SELF-7142-INITIALIZE records each source and preflight failure exactly."""

    source = tmp_path / "source.json"
    source.write_text('{"csl_event_stream_ready_score":1}', encoding="utf-8")
    specs = _model_specs(tmp_path)

    def run_case(name: str, **kwargs: object) -> dict[str, object]:
        return mod.run_experiment(
            run_date="20260908",
            result_path=tmp_path / f"{name}.json",
            transaction_root=tmp_path / f"{name}-transactions",
            source_path=source,
            model_specs=specs,
            preflight_func=lambda **_: {"all_passed": True, "checks": []},
            model_call=_model_call,
            exact_score=_score,
            expected_event_count=5,
            **kwargs,
        )

    assert (
        run_case("count", events=_events()[:-1])["gate_check_summary"]["failed_check"]
        == "frozen_event_count"
    )
    moved = _events()
    moved[0]["chronology_index"] = 4
    assert (
        run_case("order", events=moved)["gate_check_summary"]["failed_check"]
        == "frozen_event_chronology"
    )

    failed = mod.run_experiment(
        run_date="20260908",
        result_path=tmp_path / "preflight.json",
        transaction_root=tmp_path / "preflight-transactions",
        source_path=source,
        events=_events(),
        model_specs=specs,
        preflight_func=lambda **_: {"all_passed": False, "checks": []},
        model_call=_model_call,
        exact_score=_score,
        expected_event_count=5,
    )
    assert failed["gate_check_summary"]["failed_check"] == "preflight"

    monkeypatch.setattr(mod, "resolve_model_specs", lambda: specs)
    missing_calls = mod.run_experiment(
        run_date="20260908",
        result_path=tmp_path / "calls.json",
        transaction_root=tmp_path / "calls-transactions",
        source_path=source,
        events=_events(),
        model_specs=None,
        preflight_func=lambda **_: {"all_passed": True, "checks": []},
        expected_event_count=5,
    )
    assert missing_calls["gate_check_summary"]["failed_check"] == "live_model_and_exact_checker"


def test_scenario_self_7142_session_cleanup_and_internal_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-SELF-7142-TRANSACTION closes live state and disqualifies an invalid reduction."""

    specs = _model_specs(tmp_path)
    source = tmp_path / "source.json"
    source.write_text('{"csl_event_stream_ready_score":1}', encoding="utf-8")

    class Session:
        closed = False

        def close(self) -> None:
            self.closed = True

    session = Session()
    original_validate = mod.validate_artifact
    monkeypatch.setattr(mod, "validate_artifact", lambda *_args, **_kwargs: ["forced_failure"])
    artifact = mod.run_experiment(
        run_date="20260908",
        result_path=tmp_path / "result.json",
        transaction_root=tmp_path / "transactions",
        source_path=source,
        events=_events(),
        model_specs=specs,
        preflight_func=lambda **_: {**_preflight(specs), "_session": session},
        model_call=_model_call,
        exact_score=_score,
        expected_event_count=5,
    )
    monkeypatch.setattr(mod, "validate_artifact", original_validate)
    assert session.closed is True
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["gate_check_summary"]["failed_check"] == "artifact_validation"


def test_req_self_7142_validator_reports_all_safety_classes(tmp_path: Path) -> None:
    """REQ-SELF-7142 validation detects schema, chronology, metric, and terminal drift."""

    artifact = _complete_artifact(tmp_path)
    changed = deepcopy(artifact)
    del changed["cost_rows"]
    changed["field_principles"]["cost_rows"] = ""
    changed["reproducibility_checksum"] = "bad"
    changed["verdict_class"] = "unknown"
    changed["honest_verdict"] = "wrong"
    changed["flowbalance_training_reproduction"] = True
    changed["rows"] = []
    changed["event_rows"] = changed["event_rows"][:-1]
    changed["action_receipt_rows"] = changed["action_receipt_rows"][:-1]
    changed["outcome_reveal_rows"] = changed["outcome_reveal_rows"][:-1]
    changed["event_rows"][0]["memory_bytes"] = 1
    changed["event_rows"][0]["max_tokens"] = 1
    changed["event_rows"][0]["memory_record_ids"] = [
        changed["signature_rows"][0]["record"]["record_id"]
    ]
    changed["signature_rows"][0]["signature_valid"] = False
    changed["rollback_rows"][0]["parent_restored"] = False
    changed["refresh_rows"][0]["record_count_after"] = mod.MAX_MEMORY_RECORDS + 1
    changed["paired_interval_rows"] = []
    changed["future_success_rows"] = []
    errors = mod.validate_artifact(changed, expected_event_count=5)
    assert {
        "required_field_missing:cost_rows",
        "field_principle_missing:cost_rows",
        "reproducibility_checksum_mismatch",
        "verdict_class_invalid",
        "honest_verdict_class_mismatch",
        "flowbalance_training_reproduction",
        "rows_event_rows_mismatch",
        "event_row_count_mismatch",
        "action_receipt_count_mismatch",
        "outcome_receipt_count_mismatch",
        "memory_budget_mismatch",
        "generation_budget_mismatch",
        "future_memory_visible",
        "signature_invalid",
        "rollback_parent_not_restored",
        "refresh_bound_or_chronology_mismatch",
        "completion_score_mismatch",
        "paired_interval_count_mismatch",
        "future_success_arm_roster_mismatch",
        "verdict_class_mismatch",
        "gate_completion_mismatch",
    } <= set(errors)

    blocked = mod.initialize_artifact(tmp_path / "blocked.json", "20260908")
    blocked["inference_substrate_class"] = "wrong"
    blocked["gate_check_summary"] = {}
    blocked["flowbalance_memory_csl_complete_score"] = 1
    blocked["reproducibility_checksum"] = mod.artifact_checksum(blocked)
    assert {
        "blocked_substrate_class_mismatch",
        "blocked_gate_diagnostic_missing",
        "blocked_gate_values_missing",
        "blocked_completion_nonzero",
    } <= set(mod.validate_artifact(blocked, expected_event_count=5))
