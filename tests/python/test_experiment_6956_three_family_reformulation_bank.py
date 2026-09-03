"""REQ-VERIFY-6956 tests for the frozen three-family proposal bank."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6956_three_family_reformulation_bank as mod


def _formulation(name: str) -> dict:
    return {
        "schema_version": "carnot.bounded_optimization_formulation.v1",
        "variables": [
            {
                "name": name,
                "kind": "integer",
                "universe": [-1, 0, 1, 2],
                "domain": {"lower": "0", "upper": "1"},
            }
        ],
        "constraints": [{"terms": {name: "1"}, "op": "<=", "rhs": "1"}],
        "objective": {
            "direction": "min",
            "expression": {"kind": "linear", "terms": {name: "1"}, "constant": "0"},
        },
    }


def _pair(pair_id: str = "0-3-0", family: str = "bounded_integer_linear") -> dict:
    return {
        "pair_id": pair_id,
        "family": family,
        "source_formulation": _formulation("x"),
        "target_formulation": _formulation("y"),
    }


def _mapping_payload() -> dict:
    return {
        "mapping": {
            "schema_version": "carnot.reformulation_mapping.v1",
            "variables": [{"source": "x", "target": "y", "scale": "1", "offset": "0"}],
            "domain_clauses": [
                {
                    "source": "x",
                    "target": "y",
                    "source_lower": "0",
                    "source_upper": "1",
                    "target_lower": "0",
                    "target_upper": "1",
                }
            ],
            "objective": {
                "source_direction": "min",
                "target_direction": "min",
                "scale": "1",
                "offset": "0",
            },
            "claimed_relation": "equivalent",
        },
        "confidence": 0.75,
        "rationale": "The affine identity preserves the public domains.",
    }


def _model(hf_id: str = "fixture/model-GGUF", family: str = "fixture_family") -> dict:
    return {
        "hf_id": hf_id,
        "family": family,
        "name": family,
        "quantization": "Q4_K_M",
        "model_path": "/models/fixture.gguf",
        "model_sha256": "sha256:" + "1" * 64,
    }


def _plan() -> dict:
    return mod.freeze_plan(
        [_pair()], model_specs=[_model()], prompt_variants=[mod.PROMPT_VARIANTS[0]]
    )


def _bindings() -> tuple[dict[str, str], dict[str, str]]:
    return ({"fixture/model-GGUF": "sha256:" + "1" * 64}, {"fixture/model-GGUF": "sha256:t"})


def _complete_lifecycle() -> list[dict]:
    return [
        {
            "hf_id": "fixture/model-GGUF",
            "load_start_ns": 1,
            "load_end_ns": 2,
            "close_start_ns": 3,
            "close_end_ns": 4,
            "model_closed": True,
            "cuda_context_released": True,
            "cuda_offload_authenticated": True,
            "process_exit_confirmed": True,
            "process_reaped": True,
            "model_sha256": "sha256:" + "1" * 64,
        }
    ]


def _checkpoint_for(plan: dict, tmp_path: Path) -> tuple[Path, dict]:
    model_hashes, tokenizers = _bindings()
    checkpoint = mod.new_checkpoint(plan, model_hashes=model_hashes, tokenizer_bindings=tokenizers)
    path = tmp_path / "checkpoint.json"
    mod.write_checkpoint(path, checkpoint)
    return path, checkpoint


def _raw_result(raw_text: str | None = None) -> dict:
    text = raw_text if raw_text is not None else json.dumps(_mapping_payload())
    return {
        "raw_text": text,
        "output_token_ids": [1, 2, 3],
        "prompt_token_ids": [4, 5],
        "latency_s": 0.25,
        "call_status": "complete",
        "failure_reason": None,
        "runtime_receipt": {"child_pid": 123, "gpu_uuids": ["GPU-a", "GPU-b"]},
    }


def test_frozen_design_has_exact_162_attempt_budget() -> None:
    """REQ-VERIFY-6956 freezes 18 x 3 x 3 unique attempts."""

    fixture = json.loads(mod.FIXTURE_ARTIFACT_PATH.read_text(encoding="utf-8"))
    public_pairs = mod.load_public_pairs(fixture)
    plan = mod.freeze_plan(public_pairs)
    assert [row["pair_id"] for row in public_pairs] == list(mod.HELD_OUT_PAIR_IDS)
    assert len({row["family"] for row in public_pairs}) == 3
    assert len(plan["attempts"]) == mod.EXPECTED_ATTEMPT_COUNT == 162
    assert len({row["attempt_key"] for row in plan["attempts"]}) == 162
    assert all(row["prompt_sha256"].startswith("sha256:") for row in plan["attempts"])


def test_resume_raw_stage_without_duplicate_inference(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6956-RECOVERY parses durable raw bytes once."""

    plan = _plan()
    path, checkpoint = _checkpoint_for(plan, tmp_path)
    attempt = plan["attempts"][0]
    checkpoint = mod.persist_raw_attempt(checkpoint, attempt, _raw_result())
    mod.write_checkpoint(path, checkpoint)
    calls: list[str] = []

    resumed = mod.resume_attempts(
        plan,
        checkpoint_path=path,
        model_hashes=_bindings()[0],
        tokenizer_bindings=_bindings()[1],
        infer=lambda row: calls.append(row["attempt_key"]),
    )
    assert calls == []
    assert len(resumed["attempt_rows"]) == 1
    assert resumed["attempt_rows"][0]["terminal"] is True
    again = mod.resume_attempts(
        plan,
        checkpoint_path=path,
        model_hashes=_bindings()[0],
        tokenizer_bindings=_bindings()[1],
        infer=lambda row: calls.append(row["attempt_key"]),
    )
    assert len(again["attempt_rows"]) == 1
    assert calls == []


def test_malformed_json_remains_terminal_and_unrepaired(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6956-TERMINAL retains malformed candidate bytes."""

    plan = _plan()
    path, _checkpoint = _checkpoint_for(plan, tmp_path)
    resumed = mod.resume_attempts(
        plan,
        checkpoint_path=path,
        model_hashes=_bindings()[0],
        tokenizer_bindings=_bindings()[1],
        infer=lambda _row: _raw_result("```json\n{bad}\n```"),
    )
    row = resumed["attempt_rows"][0]
    assert row["raw_text"] == "```json\n{bad}\n```"
    assert row["parse"]["json_valid"] is False
    assert row["parse"]["schema_valid"] is False
    assert row["parse"]["failure_reason"] == "malformed_json"


def test_timeout_is_a_terminal_attempt(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6956-TERMINAL keeps timeout in the denominator."""

    plan = _plan()
    path, _checkpoint = _checkpoint_for(plan, tmp_path)

    def timeout(_row: dict) -> dict:
        raise TimeoutError("bounded model call expired")

    resumed = mod.resume_attempts(
        plan,
        checkpoint_path=path,
        model_hashes=_bindings()[0],
        tokenizer_bindings=_bindings()[1],
        infer=timeout,
    )
    row = resumed["attempt_rows"][0]
    assert row["terminal"] is True
    assert row["call_status"] == "timeout"
    assert row["failure_reason"] == "timeout"
    assert row["parse"]["failure_reason"] == "timeout"


def test_tokenizer_mismatch_and_model_hash_drift_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6956-BINDING rejects reuse under either drift."""

    plan = _plan()
    path, _checkpoint = _checkpoint_for(plan, tmp_path)
    with pytest.raises(mod.BankError, match="tokenizer_binding_drift"):
        mod.load_checkpoint(
            path,
            plan=plan,
            model_hashes=_bindings()[0],
            tokenizer_bindings={"fixture/model-GGUF": "sha256:changed"},
        )
    with pytest.raises(mod.BankError, match="model_hash_drift"):
        mod.load_checkpoint(
            path,
            plan=plan,
            model_hashes={"fixture/model-GGUF": "sha256:" + "2" * 64},
            tokenizer_bindings=_bindings()[1],
        )


def test_checkpoint_malformed_json_and_checksum_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-6956 atomic recovery does not trust a corrupt checkpoint."""

    path = tmp_path / "checkpoint.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(mod.BankError, match="checkpoint_malformed_json"):
        mod.load_checkpoint(
            path,
            plan=_plan(),
            model_hashes=_bindings()[0],
            tokenizer_bindings=_bindings()[1],
        )
    path, checkpoint = _checkpoint_for(_plan(), tmp_path)
    checkpoint["checkpoint_sha256"] = "sha256:bad"
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(mod.BankError, match="checkpoint_checksum_invalid"):
        mod.load_checkpoint(
            path,
            plan=_plan(),
            model_hashes=_bindings()[0],
            tokenizer_bindings=_bindings()[1],
        )


def test_hidden_labels_and_prior_candidates_are_rejected() -> None:
    """SCENARIO-VERIFY-6956-ISOLATION audits the exact prompt bytes."""

    prompt = mod.build_prompt(_pair(), mod.PROMPT_VARIANTS[0])
    assert mod.audit_prompt(prompt) == []
    assert "expected_label" not in prompt
    assert "feasibility_witness" not in prompt
    assert mod.audit_prompt(prompt + '\n"expected_label":"equivalent"') == [
        "forbidden_prompt_key:expected_label"
    ]
    assert mod.audit_prompt(prompt + "\nOther candidate: {}") == [
        "forbidden_prompt_phrase:other candidate"
    ]


def test_schema_rejects_extra_fields_without_repair() -> None:
    """REQ-VERIFY-6956 records strict schema validity, not repaired validity."""

    payload = _mapping_payload()
    payload["mapping"]["invented"] = True
    parsed = mod.parse_candidate(json.dumps(payload), _pair())
    assert parsed["json_valid"] is True
    assert parsed["schema_valid"] is False
    assert parsed["failure_reason"] == "mapping_keys"
    assert parsed["parsed_candidate"]["mapping"]["invented"] is True


def test_copied_candidates_are_reported_but_retained() -> None:
    """SCENARIO-VERIFY-6956-DIVERSITY preserves copied candidate rows."""

    raw = json.dumps(_mapping_payload(), sort_keys=True)
    rows = [
        {"attempt_key": "a", "raw_text": raw, "raw_sha256": mod.sha256_text(raw)},
        {"attempt_key": "b", "raw_text": raw, "raw_sha256": mod.sha256_text(raw)},
        {"attempt_key": "c", "raw_text": "", "raw_sha256": mod.sha256_text("")},
    ]
    diversity = mod.diversity_rows(rows)
    copied = next(row for row in diversity if row["duplicate_nonempty"])
    assert copied["attempt_keys"] == ["a", "b"]
    assert sum(row["attempt_count"] for row in diversity) == 3


def test_process_teardown_and_overlap_are_required() -> None:
    """SCENARIO-VERIFY-6956-LIFECYCLE requires closed sequential workers."""

    good = _complete_lifecycle()
    assert mod.lifecycle_errors(good, [_model()]) == []
    broken = deepcopy(good)
    broken[0]["process_reaped"] = False
    assert "process_not_reaped:fixture/model-GGUF" in mod.lifecycle_errors(broken, [_model()])
    second = {**deepcopy(good[0]), "hf_id": "fixture/model-2-GGUF", "load_start_ns": 3}
    assert "model_lifecycle_overlap" in mod.lifecycle_errors(
        [good[0], second], [_model(), _model("fixture/model-2-GGUF", "fixture_family_2")]
    )


def test_completion_allows_parse_failure_but_rejects_missing_attempt() -> None:
    """REQ-VERIFY-6956 completion measures durable coverage, not correctness."""

    plan = _plan()
    attempt = plan["attempts"][0]
    raw = mod.persist_raw_attempt(
        mod.new_checkpoint(plan, model_hashes=_bindings()[0], tokenizer_bindings=_bindings()[1]),
        attempt,
        _raw_result("not json"),
    )
    terminal = mod.finalize_pending_attempt(raw, attempt["attempt_key"])
    assert mod.completion_errors(plan, terminal["attempt_rows"], _complete_lifecycle()) == []
    assert mod.completion_errors(plan, [], _complete_lifecycle()) == ["attempt_key_roster_mismatch"]


def test_aggregate_mismatch_is_rejected(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6956-AGGREGATES rejects contradictory summaries."""

    plan = _plan()
    path, _checkpoint = _checkpoint_for(plan, tmp_path)
    checkpoint = mod.resume_attempts(
        plan,
        checkpoint_path=path,
        model_hashes=_bindings()[0],
        tokenizer_bindings=_bindings()[1],
        infer=lambda _row: _raw_result(),
    )
    artifact = mod.build_artifact(
        run_date="20260903",
        duration_s=1.0,
        preconditions_checked=[],
        plan=plan,
        checkpoint=checkpoint,
        model_specs=[_model()],
        lifecycle_rows=_complete_lifecycle(),
        source_artifact_hashes={"fixture": "sha256:" + "3" * 64},
    )
    assert mod.validate_artifact(artifact) == []
    artifact["prompt_variant_rows"][0]["attempt_count"] = 999
    assert "prompt_variant_rows_mismatch" in mod.validate_artifact(artifact)


def test_blocked_artifact_is_schema_complete() -> None:
    """REQ-VERIFY-6956 failed preconditions still produce the full contract."""

    failed = [mod.gate_check("fixture_ready", 1, 0)]
    artifact = mod.blocked_artifact("20260903", failed, duration_s=0.1)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["reformulation_bank_complete_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_three_family_reformulation_bank"
    assert artifact["gate_check_summary"] == [
        {"failed_check": "fixture_ready", "expected_value": 1, "observed_value": 0}
    ]


def test_public_pair_loader_rejects_missing_and_unbalanced_fixture_rows() -> None:
    """REQ-VERIFY-6956 freezes both the held-out roster and family balance."""

    fixture = json.loads(mod.FIXTURE_ARTIFACT_PATH.read_text(encoding="utf-8"))
    missing = deepcopy(fixture)
    missing["formulation_rows"] = [
        row
        for row in missing["formulation_rows"]
        if not (row["pair_id"] == mod.HELD_OUT_PAIR_IDS[0] and row["side"] == "source")
    ]
    with pytest.raises(mod.BankError, match="held_out_pair_unavailable"):
        mod.load_public_pairs(missing)
    unbalanced = deepcopy(fixture)
    next(row for row in unbalanced["pair_rows"] if row["pair_id"] == mod.HELD_OUT_PAIR_IDS[0])[
        "family"
    ] = "unexpected_family"
    with pytest.raises(mod.BankError, match="held_out_family_balance"):
        mod.load_public_pairs(unbalanced)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda value: value["mapping"].__setitem__("schema_version", "v0"),
            "mapping_schema_version",
        ),
        (lambda value: value["mapping"].__setitem__("variables", None), "variable_rows"),
        (lambda value: value["mapping"].__setitem__("domain_clauses", None), "domain_clause_rows"),
        (lambda value: value["mapping"].__setitem__("objective", None), "objective_keys"),
        (
            lambda value: value["mapping"]["variables"][0].__setitem__("source", "missing"),
            "source_variable_roster",
        ),
        (
            lambda value: value["mapping"]["variables"][0].__setitem__("target", "missing"),
            "target_variable_roster",
        ),
        (
            lambda value: value["mapping"]["domain_clauses"][0].__setitem__("source", "missing"),
            "source_domain_roster",
        ),
        (
            lambda value: value["mapping"]["domain_clauses"][0].__setitem__("target", "missing"),
            "target_domain_roster",
        ),
        (
            lambda value: value["mapping"]["variables"][0].__setitem__("scale", 1),
            "rational_string_required",
        ),
        (
            lambda value: value["mapping"]["objective"].__setitem__("source_direction", "up"),
            "objective_direction",
        ),
        (
            lambda value: value["mapping"].__setitem__("claimed_relation", "unknown"),
            "claimed_relation",
        ),
    ],
)
def test_mapping_schema_rejection_codes_are_stable(mutation, reason: str) -> None:
    """REQ-VERIFY-6956 keeps every structural rejection deterministic."""

    payload = _mapping_payload()
    mutation(payload)
    assert mod.parse_candidate(json.dumps(payload), _pair())["failure_reason"] == reason


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("", "empty_output"),
        ("[]", "response_object_required"),
        (json.dumps({"confidence": 0.5}), "response_keys"),
        (json.dumps({**_mapping_payload(), "extra": True}), "response_keys"),
        (json.dumps({**_mapping_payload(), "confidence": True}), "confidence_range"),
        (json.dumps({**_mapping_payload(), "confidence": 2}), "confidence_range"),
        (json.dumps({**_mapping_payload(), "rationale": "x" * 241}), "rationale_length"),
    ],
)
def test_candidate_envelope_rejections_are_terminal(payload: str, reason: str) -> None:
    """SCENARIO-VERIFY-6956-TERMINAL covers every strict envelope failure."""

    assert mod.parse_candidate(payload, _pair())["failure_reason"] == reason


def test_checkpoint_rejects_schema_plan_and_duplicate_key_drift(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6956-BINDING covers every durable manifest binding."""

    plan = _plan()
    path, checkpoint = _checkpoint_for(plan, tmp_path)
    for field, value, reason in (
        ("schema_version", "old", "checkpoint_schema_mismatch"),
        ("plan_sha256", "sha256:changed", "plan_hash_drift"),
    ):
        changed = deepcopy(checkpoint)
        changed[field] = value
        mod.write_checkpoint(path, changed)
        with pytest.raises(mod.BankError, match=reason):
            mod.load_checkpoint(
                path,
                plan=plan,
                model_hashes=_bindings()[0],
                tokenizer_bindings=_bindings()[1],
            )
    duplicated = mod.persist_raw_attempt(checkpoint, plan["attempts"][0], _raw_result())
    duplicated["attempt_rows"].append(deepcopy(duplicated["attempt_rows"][0]))
    mod.write_checkpoint(path, duplicated)
    with pytest.raises(mod.BankError, match="checkpoint_duplicate_attempt_key"):
        mod.load_checkpoint(
            path,
            plan=plan,
            model_hashes=_bindings()[0],
            tokenizer_bindings=_bindings()[1],
        )


def test_raw_and_parse_stage_guards_fail_closed() -> None:
    """SCENARIO-VERIFY-6956-RECOVERY refuses ambiguous stage transitions."""

    plan = _plan()
    checkpoint = mod.new_checkpoint(
        plan, model_hashes=_bindings()[0], tokenizer_bindings=_bindings()[1]
    )
    attempt = plan["attempts"][0]
    raw = mod.persist_raw_attempt(checkpoint, attempt, _raw_result())
    with pytest.raises(mod.BankError, match="duplicate_attempt_key"):
        mod.persist_raw_attempt(raw, attempt, _raw_result())
    with pytest.raises(mod.BankError, match="attempt_not_found"):
        mod.finalize_pending_attempt(raw, "absent")
    terminal = mod.finalize_pending_attempt(raw, attempt["attempt_key"])
    assert mod.finalize_pending_attempt(terminal, attempt["attempt_key"]) == terminal
    broken = deepcopy(raw)
    broken["attempt_rows"][0]["raw_durable"] = False
    with pytest.raises(mod.BankError, match="raw_stage_not_durable"):
        mod.finalize_pending_attempt(broken, attempt["attempt_key"])


@pytest.mark.parametrize("result", [None, RuntimeError("runner broke")])
def test_resume_converts_invalid_runner_results_to_terminal_failure(
    tmp_path: Path, result: object
) -> None:
    """SCENARIO-VERIFY-6956-TERMINAL retains invalid runner outcomes."""

    plan = _plan()
    path, _checkpoint = _checkpoint_for(plan, tmp_path)

    def infer(_row: dict):
        if isinstance(result, Exception):
            raise result
        return result

    resumed = mod.resume_attempts(
        plan,
        checkpoint_path=path,
        model_hashes=_bindings()[0],
        tokenizer_bindings=_bindings()[1],
        infer=infer,
    )
    row = resumed["attempt_rows"][0]
    assert row["terminal"] is True
    assert row["call_status"] == "runner_failure"
    assert row["parse"]["failure_reason"].startswith(("TypeError", "RuntimeError"))


def test_all_lifecycle_and_attempt_integrity_failures_are_named() -> None:
    """SCENARIO-VERIFY-6956-LIFECYCLE makes each closure claim falsifiable."""

    plan = _plan()
    attempt = plan["attempts"][0]
    checkpoint = mod.persist_raw_attempt(
        mod.new_checkpoint(plan, model_hashes=_bindings()[0], tokenizer_bindings=_bindings()[1]),
        attempt,
        _raw_result(),
    )
    terminal = mod.finalize_pending_attempt(checkpoint, attempt["attempt_key"])
    row = terminal["attempt_rows"][0]
    row["terminal"] = False
    row["raw_durable"] = False
    row["hidden_label_isolation_errors"] = ["leak"]
    lifecycle = deepcopy(_complete_lifecycle()[0])
    lifecycle.update(
        {
            "model_sha256": "sha256:changed",
            "cuda_offload_authenticated": False,
            "model_closed": False,
            "cuda_context_released": False,
            "process_exit_confirmed": False,
            "process_reaped": False,
        }
    )
    errors = mod.completion_errors(plan, [row], [lifecycle])
    assert {
        "nonterminal_attempt",
        "raw_stage_not_durable",
        "hidden_label_isolation_failure",
        "lifecycle_model_hash_drift:fixture/model-GGUF",
        "cuda_offload_not_authenticated:fixture/model-GGUF",
        "model_not_closed:fixture/model-GGUF",
        "cuda_context_not_released:fixture/model-GGUF",
        "process_exit_not_confirmed:fixture/model-GGUF",
        "process_not_reaped:fixture/model-GGUF",
    } <= set(errors)
    invalid_interval = deepcopy(_complete_lifecycle())
    invalid_interval.append({**deepcopy(invalid_interval[0]), "hf_id": "fixture/model-2-GGUF"})
    invalid_interval[1].pop("load_start_ns")
    assert "model_lifecycle_interval_invalid" in mod.lifecycle_errors(
        invalid_interval, [_model(), _model("fixture/model-2-GGUF", "fixture_family_2")]
    )
    assert "model_lifecycle_roster_or_order" in mod.lifecycle_errors([], [_model()])


def test_artifact_validator_names_contract_and_prefix_mutations(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6956-AGGREGATES covers non-aggregate artifact claims too."""

    plan = _plan()
    path, _checkpoint = _checkpoint_for(plan, tmp_path)
    checkpoint = mod.resume_attempts(
        plan,
        checkpoint_path=path,
        model_hashes=_bindings()[0],
        tokenizer_bindings=_bindings()[1],
        infer=lambda _row: _raw_result(),
    )
    artifact = mod.build_artifact(
        run_date="20260903",
        duration_s=1.0,
        preconditions_checked=[],
        plan=plan,
        checkpoint=checkpoint,
        model_specs=[_model()],
        lifecycle_rows=_complete_lifecycle(),
        source_artifact_hashes={},
    )
    missing = deepcopy(artifact)
    missing.pop("rows")
    assert mod.validate_artifact(missing)[0].startswith("missing_required_fields:")
    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    assert "field_principles_incomplete" in mod.validate_artifact(bad_principles)
    bad_score = deepcopy(artifact)
    bad_score["reformulation_bank_complete_score"] = 0
    assert "completion_score_mismatch" in mod.validate_artifact(bad_score)
    bad_receipt = deepcopy(artifact)
    bad_receipt["task_runtime_receipt"]["task_id"] = "copied"
    assert "task_runtime_receipt_hash_mismatch" in mod.validate_artifact(bad_receipt)
    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = True
    assert "verifier_is_oracle_must_be_false" in mod.validate_artifact(bad_oracle)
    for verdict_class, bad_prefix in (
        ("null", "partial_wrong"),
        ("blocked", "complete_wrong"),
        ("partial", "complete_wrong"),
    ):
        changed = deepcopy(artifact)
        changed["verdict_class"] = verdict_class
        changed["honest_verdict"] = bad_prefix
        assert "honest_verdict_prefix_mismatch" in mod.validate_artifact(changed)


def test_json_reader_rejects_missing_malformed_and_nonobject(tmp_path: Path) -> None:
    """REQ-VERIFY-6956 preflight preserves malformed-source failures."""

    with pytest.raises(mod.BankError, match="json_unavailable"):
        mod._read_json(tmp_path / "missing.json")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(mod.BankError, match="json_unavailable"):
        mod._read_json(malformed)
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    with pytest.raises(mod.BankError, match="json_object_required"):
        mod._read_json(sequence)
    valid = tmp_path / "valid.json"
    valid.write_text('{"ready":true}', encoding="utf-8")
    assert mod._read_json(valid) == {"ready": True}
