"""Tests for the queue-regulated prospective self-learning experiment.

Spec refs: REQ-LEARN-6962 and SCENARIO-LEARN-6962-*.
"""

from __future__ import annotations

from copy import deepcopy
import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_6962_queue_regulated_self_learning as exp


@pytest.fixture
def event() -> dict:
    """Provide one sealed event with a prior-only retrieval surface."""

    return {
        "event_id": "EVT6961-010",
        "ordinal": 10,
        "model_family": "qwen3.6_moe",
        "problem_family": "boolean_cardinality",
        "split": "evaluation",
        "prompt_hash": "sha256:prompt",
        "exact_outcome": "equivalent",
        "exact_success": True,
        "certificate_id": "EVT6961-010-CERT",
        "available_after_ordinal": 10,
        "token_budget": 196,
        "eligible_prior_certificate_ids": ["CERT-A", "CERT-B"],
        "prohibited_current_certificate_id": "EVT6961-010-CERT",
        "prohibited_future_certificate_ids": ["EVT6961-011-CERT"],
        "conflict_class": None,
        "control_class": "standard",
        "reusable_factor_ids": ["variable_bijection"],
        "arm_prompt_payloads": {
            "no_memory": {"event_id": "EVT6961-010", "retrieved_memory": []},
            "fifo": {"event_id": "EVT6961-010", "retrieved_memory": []},
            "queue": {"event_id": "EVT6961-010", "retrieved_memory": []},
        },
    }


def _certificate(
    *,
    certificate_id: str = "CERT-A",
    event_id: str = "EVT6961-001",
    ordinal: int = 1,
    problem_family: str = "boolean_cardinality",
    exact_success: bool = True,
    authority: str = exp.EXACT_AUTHORITY,
    confidence: float = 0.9,
    rationale: str = "model says it is correct",
) -> dict:
    raw_output = '{"relation": "equivalent"}'
    certificate = {
        "certificate_id": certificate_id,
        "event_id": event_id,
        "event_ordinal": ordinal,
        "available_after_ordinal": ordinal,
        "problem_family": problem_family,
        "model_id": exp.MODEL_SPECS[0]["hf_id"],
        "prompt_hash": "sha256:prompt",
        "raw_output_hash": exp.sha256_text(raw_output),
        "exact_outcome": "equivalent",
        "exact_success": exact_success,
        "authority": authority,
        "confidence": confidence,
        "rationale": rationale,
        "parent_store_hash": exp.sha256_json(exp.empty_store_state()),
        "reusable_factor_ids": ["variable_bijection"],
    }
    certificate["certificate_hash"] = exp.certificate_hash(certificate)
    return certificate


def _complete_artifact(tmp_path: Path) -> dict:
    rows = [
        {
            "model_id": model["hf_id"],
            "arm": arm,
            "event_id": "EVT6961-010",
            "ordinal": 10,
            "split": "evaluation",
            "exact_success": arm == "debt_queue",
            "retention_success": True,
            "process_id": 1000 + model_index * 10 + arm_index,
            "store_path": str(tmp_path / f"{model_index}-{arm_index}.json"),
        }
        for model_index, model in enumerate(exp.MODEL_SPECS)
        for arm_index, arm in enumerate(exp.ARMS)
    ]
    artifact = exp.empty_artifact("20260904")
    artifact.update(
        {
            "rows": rows,
            "event_rows": rows,
            "model_rows": [
                {
                    "model_id": model["hf_id"],
                    "model_hash_before": f"sha256:{model_index:064d}",
                    "model_hash_after": f"sha256:{model_index:064d}",
                    "headline_eligible": True,
                }
                for model_index, model in enumerate(exp.MODEL_SPECS)
            ],
            "arm_rows": exp.recompute_arm_rows(rows),
            "model_hashes_before": {
                model["hf_id"]: f"sha256:{model_index:064d}"
                for model_index, model in enumerate(exp.MODEL_SPECS)
            },
            "model_hashes_after": {
                model["hf_id"]: f"sha256:{model_index:064d}"
                for model_index, model in enumerate(exp.MODEL_SPECS)
            },
            "queue_learning_run_complete_score": 1,
            "queue_learning_positive_score": 0,
            "verdict_class": "null",
            "honest_verdict": "complete_null_queue_accuracy_gain_not_certified",
            "gate_check_summary": exp.gate_summary([]),
        }
    )
    artifact["reproducibility_checksum"] = exp.payload_checksum(artifact)
    return artifact


def _sealed_event(event_id: str, ordinal: int, *, exact_outcome: str = "equivalent") -> dict:
    """Build a small answer-sealed event for injected worker tests."""

    prior_ids = [f"E{index}-CERT" for index in range(ordinal)]
    return {
        "event_id": event_id,
        "certificate_id": f"{event_id}-CERT",
        "ordinal": ordinal,
        "split": "train" if ordinal == 0 else "evaluation",
        "problem_family": "boolean_cardinality",
        "prompt_hash": f"sha256:prompt-{ordinal}",
        "prompt_payload": {"event_id": event_id, "visible": [ordinal]},
        "token_budget": 8,
        "exact_outcome": exact_outcome,
        "exact_success": True,
        "eligible_prior_certificate_ids": prior_ids,
        "prohibited_current_certificate_id": f"{event_id}-CERT",
        "prohibited_future_certificate_ids": [],
        "reusable_factor_ids": ["cardinality_preservation"],
        "conflict_class": None,
        "control_class": "standard",
    }


def test_model_specs_use_only_required_cached_gguf_pair() -> None:
    """REQ-LEARN-6962: headline model IDs are frozen without HF tokenizers."""

    assert [row["hf_id"] for row in exp.MODEL_SPECS] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]
    assert all(row["headline_eligible"] for row in exp.MODEL_SPECS)
    source = Path(exp.__file__).read_text(encoding="utf-8")
    assert "AutoTokenizer" not in source


def test_arm_isolation_and_resume_key_recovery(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6962-ARM-ISOLATION: stores and process IDs differ."""

    rows = [
        {
            "model_id": exp.MODEL_SPECS[0]["hf_id"],
            "arm": arm,
            "process_id": 10 + index,
            "store_path": str(tmp_path / f"{arm}.json"),
        }
        for index, arm in enumerate(exp.ARMS)
    ]
    assert exp.arm_isolation_errors(rows) == []
    rows[1]["process_id"] = rows[0]["process_id"]
    rows[1]["store_path"] = rows[0]["store_path"]
    assert set(exp.arm_isolation_errors(rows)) == {
        "process_reused_across_arms",
        "store_reused_across_arms",
    }

    planned = ["m|a|0", "m|a|1", "m|a|2"]
    assert exp.next_missing_key(planned, {"m|a|0", "m|a|1"}) == "m|a|2"
    assert exp.next_missing_key(planned, set(planned)) is None


def test_future_label_and_certificate_rejection(event: dict) -> None:
    """SCENARIO-LEARN-6962-FUTURE-LABEL: future data fails closed."""

    valid = _certificate()
    future = _certificate(certificate_id="EVT6961-011-CERT", ordinal=11)
    assert exp.retrieval_eligibility(valid, event) == (True, "eligible_prior_exact_certificate")
    assert exp.retrieval_eligibility(future, event) == (False, "future_certificate")
    leaking = deepcopy(event)
    leaking["arm_prompt_payloads"]["fifo"]["exact_outcome"] = "equivalent"
    assert exp.future_label_isolation_errors(leaking) == [
        "exact_outcome_visible_before_certification"
    ]


def test_raw_output_is_durable_before_outcome_and_write(tmp_path: Path, event: dict) -> None:
    """SCENARIO-LEARN-6962-WRITE-ORDER: durable output precedes authority."""

    store = exp.TransactionalMemoryStore(tmp_path / "memory.json", capacity=2)
    row = exp.run_event_transaction(
        event=event,
        model_id=exp.MODEL_SPECS[0]["hf_id"],
        arm="fifo",
        store=store,
        raw_output='{"relation": "equivalent"}',
        exact_outcome="equivalent",
        exact_success=True,
        retrieved_ids=[],
    )
    assert row["raw_output_durable_step"] < row["exact_outcome_visible_step"]
    assert row["exact_outcome_visible_step"] < row["write_step"]
    assert store.active_ids() == ["EVT6961-010-CERT"]


def test_forged_certificate_and_confidence_admission_rejection(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6962-FORGED-CERTIFICATE and confidence authority."""

    store = exp.TransactionalMemoryStore(tmp_path / "memory.json", capacity=2)
    valid = _certificate()
    forged = deepcopy(valid)
    forged["authority"] = "model_self_report"
    assert exp.admission_decision(valid, "fifo", store.state, exp.FROZEN_POLICY)["admit"]
    assert not exp.admission_decision(forged, "fifo", store.state, exp.FROZEN_POLICY)["admit"]

    low_confidence = _certificate(confidence=0.01, rationale="uncertain")
    high_confidence = _certificate(confidence=0.99, rationale="certain")
    low = exp.admission_decision(low_confidence, "debt_queue", store.state, exp.FROZEN_POLICY)
    high = exp.admission_decision(high_confidence, "debt_queue", store.state, exp.FROZEN_POLICY)
    assert {key: low[key] for key in ("admit", "reason", "objective")} == {
        key: high[key] for key in ("admit", "reason", "objective")
    }


def test_queue_underflow_and_duplicate_debt() -> None:
    """SCENCHARGE: underflow and duplicate causal debt are rejected."""

    # SCENARIO-LEARN-6962-QUEUE-UNDERFLOW
    transition = exp.apply_debt_transition(
        previous_balance=0,
        event_id="EVT6961-010",
        causal_source_ids=[],
        cause="exact_success",
        requested_service=3,
        charged_pairs=set(),
    )
    assert transition["service"] == 0
    assert transition["balance"] == 0

    # SCENARIO-LEARN-6962-DUPLICATE-DEBT
    duplicate = exp.apply_debt_transition(
        previous_balance=1,
        event_id="EVT6961-011",
        causal_source_ids=["CERT-A", "CERT-A", "CERT-B"],
        cause="exact_failure",
        requested_service=0,
        charged_pairs={"EVT6961-011|CERT-A"},
    )
    assert duplicate["arrival"] == 1
    assert duplicate["causal_source_ids"] == ["CERT-B"]
    assert duplicate["balance"] == 2


def test_poison_contradiction_tombstone_and_retrieval(tmp_path: Path, event: dict) -> None:
    """SCENARIO-LEARN-6962-POISON and C-LEARN-6962-CONTRADICT-SAFETY."""

    store = exp.TransactionalMemoryStore(tmp_path / "memory.json", capacity=2)
    valid = _certificate()
    store.admit(valid)
    poison = _certificate(certificate_id="CERT-P", exact_success=False)
    assert not exp.admission_decision(poison, "debt_queue", store.state, exp.FROZEN_POLICY)["admit"]

    transition = exp.apply_debt_transition(
        previous_balance=0,
        event_id=event["event_id"],
        causal_source_ids=[valid["certificate_id"]],
        cause="contradiction",
        requested_service=1,
        charged_pairs=set(),
    )
    assert transition["arrival"] == 1
    assert transition["service"] == 0
    assert transition["balance"] == 1

    # SCENARIO-LEARN-6962-TOMBSTONE
    store.tombstone(valid["certificate_id"], "delayed_copy_poison")
    assert store.active_ids() == []
    assert store.state["tombstones"][valid["certificate_id"]]["reason"] == "delayed_copy_poison"
    assert exp.select_retrieval(store.state, event, 3) == []


def test_fifo_capacity_restart_and_rollback(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6962-RESTART and rollback preserve exact bytes."""

    path = tmp_path / "memory.json"
    store = exp.TransactionalMemoryStore(path, capacity=2)
    first = _certificate(certificate_id="CERT-A", ordinal=1)
    second = _certificate(certificate_id="CERT-B", ordinal=2)
    third = _certificate(certificate_id="CERT-C", ordinal=3)
    store.admit(first)
    store.admit(second)
    parent_bytes = path.read_bytes()
    store.admit(third)
    assert store.active_ids() == ["CERT-B", "CERT-C"]
    assert store.state["tombstones"]["CERT-A"]["reason"] == "fifo_capacity_eviction"

    restored = exp.TransactionalMemoryStore(path, capacity=2)
    assert restored.store_hash() == store.store_hash()
    assert restored.state == store.state

    # SCENARIO-LEARN-6962-ROLLBACK
    restored.rollback(parent_bytes)
    assert path.read_bytes() == parent_bytes
    assert restored.active_ids() == ["CERT-A", "CERT-B"]


def test_transaction_failure_restores_parent_bytes(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6962-ROLLBACK: failed atomic writes restore the parent."""

    store = exp.TransactionalMemoryStore(tmp_path / "memory.json", capacity=2)
    store.admit(_certificate())
    parent = store.path.read_bytes()
    with pytest.raises(exp.TransactionError):
        store.apply_transaction(
            lambda state: state["active"].append({"certificate_id": "BROKEN"}),
            fail_before_commit=True,
        )
    assert store.path.read_bytes() == parent


def test_model_hash_drift_and_aggregate_mismatch(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6962-MODEL-HASH and aggregate mismatch fail closed."""

    artifact = _complete_artifact(tmp_path)
    assert exp.validate_artifact(artifact) == []

    drift = deepcopy(artifact)
    model_id = exp.MODEL_SPECS[0]["hf_id"]
    drift["model_hashes_after"][model_id] = "sha256:" + "f" * 64
    assert "model_hash_drift" in exp.validate_artifact(drift)

    # SCENARIO-LEARN-6962-AGGREGATE
    mismatch = deepcopy(artifact)
    mismatch["arm_rows"][0]["exact_accuracy"] = 1.0
    mismatch["queue_learning_positive_score"] = 1
    errors = exp.validate_artifact(mismatch)
    assert "aggregate_mismatch" in errors
    assert "positive_score_invalid" in errors


def test_required_fields_and_blocked_gate_summary() -> None:
    """REQ-LEARN-6962: blocked artifacts remain complete and actionable."""

    checks = [exp.gate_check("cached_gargantua", True, False)]
    artifact = exp.build_blocked_artifact(
        run_date="20260904",
        checks=checks,
        source_hashes={"event_sequence": "sha256:source"},
        duration_s=0.2,
    )
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_principles"])
    assert artifact["gate_check_summary"] == {
        "checks": checks,
        "failed_check": "cached_gargantua",
        "expected_value": True,
        "observed_value": False,
        "passed": False,
    }
    assert artifact["queue_learning_run_complete_score"] == 0
    assert artifact["queue_learning_positive_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_queue_regulated_self_learning"
    assert exp.validate_artifact(artifact) == []


def test_recovery_checkpoint_is_idempotent(tmp_path: Path) -> None:
    """REQ-LEARN-6962: duplicate recovery keys do not repeat work."""

    checkpoint = exp.EventCheckpoint(tmp_path / "checkpoint.json")
    assert checkpoint.record("model|arm|event", {"result": 1})
    first_bytes = checkpoint.path.read_bytes()
    assert not checkpoint.record("model|arm|event", {"result": 2})
    assert checkpoint.path.read_bytes() == first_bytes
    assert checkpoint.completed_keys() == {"model|arm|event"}


def test_pairing_gate_requires_accuracy_ci_and_retention() -> None:
    """REQ-LEARN-6962: lower debt alone cannot create a positive result."""

    rows = [
        {
            "model_id": exp.MODEL_SPECS[0]["hf_id"],
            "event_id": f"E{index}",
            "split": "evaluation",
            "arm": arm,
            "exact_success": success,
            "retention_success": True,
        }
        for index, (no_memory, fifo, queue) in enumerate(
            [(False, False, True), (False, False, True), (False, False, True), (False, False, True)]
        )
        for arm, success in zip(exp.ARMS, (no_memory, fifo, queue), strict=True)
    ]
    paired, confidence = exp.compute_paired_metrics(rows)
    gates = exp.evaluate_positive_gate(
        paired_rows=paired,
        confidence_rows=confidence,
        retention_rows=[{"arm": "debt_queue", "retained": True}],
        future_rows=[{"passed": True}],
        restart_rows=[{"passed": True}],
        rollback_rows=[{"passed": True}],
    )
    assert gates["positive"]

    confidence[0]["ci95_lower"] = 0.0
    assert not exp.evaluate_positive_gate(
        paired_rows=paired,
        confidence_rows=confidence,
        retention_rows=[{"arm": "debt_queue", "retained": True}],
        future_rows=[{"passed": True}],
        restart_rows=[{"passed": True}],
        rollback_rows=[{"passed": True}],
    )["positive"]


def test_main_writes_only_to_requested_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-LEARN-6962: the CLI writes a complete blocked result on gate failure."""

    source = tmp_path / "event-sequence.json"
    source.write_text(
        json.dumps(
            {
                "certified_event_sequence_ready_score": 0,
                "sealed_checkpoint_path": "missing.json",
                "sealed_checkpoint_sha256": None,
                "event_rows": [],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    monkeypatch.setattr(exp, "resolve_model_specs", lambda: [dict(row) for row in exp.MODEL_SPECS])
    code = exp.main(
        [
            "--date",
            "20260904",
            "--event-sequence",
            str(source),
            "--output",
            str(output),
            "--workspace",
            str(tmp_path),
        ]
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "certified_event_sequence_ready_score"


def test_certificate_store_and_retrieval_rejection_branches(tmp_path: Path, event: dict) -> None:
    """REQ-LEARN-6962: invalid, duplicate, removed, and unrelated records stay out."""

    store = exp.TransactionalMemoryStore(tmp_path / "memory.json", capacity=2)
    invalid = _certificate()
    invalid["certificate_hash"] = "sha256:forged"
    with pytest.raises(ValueError, match="externally exact"):
        store.admit(invalid)

    valid = _certificate()
    assert store.admit(valid) is None
    assert store.admit(valid) is None
    assert not store.tombstone("NOT-A-CERT", "absent")
    store.tombstone("CERT-A", "removed")
    assert store.admit(valid) is None

    current = _certificate(certificate_id=event["prohibited_current_certificate_id"], ordinal=1)
    prohibited = _certificate(certificate_id="EVT6961-011-CERT", ordinal=1)
    unrelated = _certificate(certificate_id="CERT-Z", ordinal=1)
    forged = _certificate(certificate_id="CERT-B", ordinal=1)
    forged["certificate_hash"] = "sha256:bad"
    assert exp.retrieval_eligibility(current, event)[1] == "current_certificate"
    assert exp.retrieval_eligibility(prohibited, event)[1] == "future_certificate"
    assert exp.retrieval_eligibility(unrelated, event)[1] == "not_in_sealed_eligibility_set"
    assert exp.retrieval_eligibility(forged, event)[1] == "invalid_exact_certificate"


def test_retrieval_policy_debt_threshold_fifo_and_empty_limit(event: dict) -> None:
    """REQ-LEARN-6962: retrieval uses frozen debt, family, age, and count."""

    first = _certificate(certificate_id="CERT-A", ordinal=1)
    second = _certificate(
        certificate_id="CERT-B", ordinal=2, problem_family="bounded_integer_linear"
    )
    state = exp.empty_store_state()
    state["active"] = [first, second]
    assert [row["certificate_id"] for row in exp.select_retrieval(state, event, 3)] == ["CERT-A"]
    assert [row["certificate_id"] for row in exp.select_retrieval(state, event, 3, arm="fifo")] == [
        "CERT-B",
        "CERT-A",
    ]
    assert exp.select_retrieval(state, event, -1, arm="fifo") == []
    state["debt_balance"] = exp.FROZEN_POLICY.debt_threshold
    assert exp.select_retrieval(state, event, 3) == []


def test_admission_unknown_arm_risk_rejection_and_proposal_parser(tmp_path: Path) -> None:
    """REQ-LEARN-6962: exact metadata controls queue rejection and parsing."""

    store = exp.TransactionalMemoryStore(tmp_path / "memory.json")
    valid = _certificate()
    assert exp.admission_decision(valid, "no_memory", store.state, exp.FROZEN_POLICY)["reason"] == (
        "no_memory_policy"
    )
    assert exp.admission_decision(valid, "unknown", store.state, exp.FROZEN_POLICY)["reason"] == (
        "unknown_arm"
    )
    risky = deepcopy(valid)
    risky["certified_risk_flags"] = ["contradiction"]
    risky["certificate_hash"] = exp.certificate_hash(risky)
    store.state["debt_balance"] = exp.FROZEN_POLICY.debt_threshold - 1
    assert not exp.admission_decision(risky, "debt_queue", store.state, exp.FROZEN_POLICY)["admit"]
    assert exp._proposal_relation("not equivalent") == "non_equivalent"
    assert exp._proposal_relation("the pair is equivalent") == "equivalent"
    assert exp._proposal_relation("unknown") is None


def test_debt_store_conflict_transaction_and_checkpoint_restore(
    tmp_path: Path, event: dict
) -> None:
    """SCENARIO-LEARN-6962-CONTRADICTION: debt persists and resume reloads rows."""

    store = exp.TransactionalMemoryStore(tmp_path / "memory.json")
    conflicting = deepcopy(event)
    conflicting["conflict_class"] = "objective_direction_conflict"
    row = exp.run_event_transaction(
        event=conflicting,
        model_id=exp.MODEL_SPECS[0]["hf_id"],
        arm="debt_queue",
        store=store,
        raw_output="unknown",
        exact_outcome="equivalent",
        exact_success=False,
        retrieved_ids=["CERT-A"],
    )
    assert row["debt_transition"]["cause"] == "contradiction"
    assert store.state["debt_balance"] == 1

    checkpoint = exp.EventCheckpoint(tmp_path / "events.json")
    checkpoint.record("one", row)
    restored = exp.EventCheckpoint(checkpoint.path)
    assert restored.rows() == [row]


def test_hash_json_reader_metric_empty_and_validation_error_branches(tmp_path: Path) -> None:
    """REQ-LEARN-6962: malformed inputs and inconsistent metadata fail closed."""

    assert exp.sha256_file(tmp_path / "absent") is None
    malformed = tmp_path / "list.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        exp._read_json(malformed)
    paired, confidence = exp.compute_paired_metrics([])
    assert paired[0]["paired_event_count"] == 0
    assert confidence[0]["ci95_lower"] == -1.0
    one_row = [
        {
            "model_id": "m",
            "event_id": "e",
            "split": "evaluation",
            "arm": "debt_queue",
            "exact_success": True,
        },
        {
            "model_id": "m",
            "event_id": "e",
            "split": "evaluation",
            "arm": "fifo",
            "exact_success": False,
        },
    ]
    _, confidence = exp.compute_paired_metrics(one_row)
    assert next(row for row in confidence if row["comparator"] == "fifo")["ci95_lower"] == 1.0
    assert exp.recompute_arm_rows([])[0]["retention_rate"] == 0.0
    assert exp._float_equal("same", "same")

    bad = exp.empty_artifact("20260904")
    bad.pop("rows")
    bad.update(
        {
            "field_principles": {},
            "inference_substrate": "wrong",
            "continuous_self_learning_task": False,
            "learning_tier": 3,
            "no_model_weight_mutation": False,
            "verifier_is_oracle": True,
            "verdict_class": "mystery",
            "honest_verdict": "wrong",
            "model_rows": [
                {"model_id": "m", "process_id": 1, "store_path": "x"},
                {"model_id": "m", "process_id": 1, "store_path": "x"},
            ],
            "arm_rows": [],
            "event_rows": one_row,
            "reproducibility_checksum": "sha256:wrong",
        }
    )
    errors = set(exp.validate_artifact(bad))
    assert {
        "required_fields_missing",
        "field_principles_incomplete",
        "inference_substrate_mismatch",
        "continuous_self_learning_task_invalid",
        "learning_tier_invalid",
        "model_weight_mutation_declared",
        "verifier_is_oracle_invalid",
        "verdict_class_invalid",
        "honest_verdict_prefix_mismatch",
        "process_reused_across_arms",
        "store_reused_across_arms",
        "aggregate_mismatch",
        "reproducibility_checksum_mismatch",
    } <= errors

    blocked = exp.build_blocked_artifact(
        run_date="20260904", checks=[], source_hashes={}, duration_s=0
    )
    blocked.update(
        {
            "honest_verdict": "wrong",
            "queue_learning_run_complete_score": 1,
            "queue_learning_positive_score": 1,
        }
    )
    blocked_errors = set(exp.validate_artifact(blocked))
    assert {
        "blocked_verdict_invalid",
        "blocked_run_complete_invalid",
        "blocked_positive_invalid",
        "positive_score_invalid",
    } <= blocked_errors


def test_resolver_and_preconditions_cover_success_and_tool_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-LEARN-6962: local model and resource preconditions record both sides."""

    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"small-test-model")
    checkpoint = tmp_path / "sealed.json"
    checkpoint.write_text("{}", encoding="utf-8")
    source_path = tmp_path / "source.json"
    source_path.write_text("{}", encoding="utf-8")
    source = {
        "certified_event_sequence_ready_score": 1,
        "sealed_checkpoint_path": str(checkpoint),
        "sealed_checkpoint_sha256": exp.sha256_file(checkpoint),
    }
    monkeypatch.setattr(exp, "resolve_cached_gguf", lambda model_id: str(model_file))
    assert all(row["model_path"] == str(model_file) for row in exp.resolve_model_specs())
    monkeypatch.setattr(exp, "gguf_tokenizer_loadable", lambda path: (True, f"ok:{path}"))
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="0, GPU A\n1, GPU B\n"),
    )
    monkeypatch.setattr(
        __import__("carnot.experiment_6961_certified_event_sequence", fromlist=["x"]),
        "sequence_conformance_errors",
        lambda value: [],
    )
    specs = [{**row, "model_path": str(model_file)} for row in exp.MODEL_SPECS]
    checks, hashes = exp.collect_preconditions(
        source=source,
        source_path=source_path,
        workspace=tmp_path,
        model_specs=specs,
    )
    assert all(row["passed"] for row in checks)
    assert hashes[f"model:{exp.MODEL_SPECS[0]['hf_id']}"]["sha256"] == exp.sha256_file(model_file)

    monkeypatch.setattr(
        exp.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no tool"))
    )
    checks, _ = exp.collect_preconditions(
        source=source,
        source_path=source_path,
        workspace=tmp_path,
        model_specs=specs,
    )
    assert not next(row for row in checks if row["check"] == "authenticated_cuda_offload")["passed"]


def test_gpu_probe_loader_generator_and_prompt_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-LEARN-6962: runtime helpers bind the frozen local inference surface."""

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"{exp.os.getpid()}, 1234\n"),
    )
    assert exp._gpu_process_memory(exp.os.getpid()) == 1234
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"{exp.os.getpid()}, not-a-number\n"),
    )
    assert exp._gpu_process_memory(exp.os.getpid()) == 0
    monkeypatch.setattr(
        exp.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError())
    )
    assert exp._gpu_process_memory(exp.os.getpid()) == 0

    calls = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def create_chat_completion(self, **kwargs):
            calls.update(kwargs)
            return {
                "choices": [{"message": {"content": '{"relation":"equivalent"}'}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }

    monkeypatch.setattr(__import__("llama_cpp"), "Llama", FakeLlama)
    llm = exp._load_llama(str(tmp_path / "model.gguf"), 1, exp.FROZEN_POLICY)
    assert calls["n_gpu_layers"] == -1
    assert exp.os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert exp._generate(llm, "prompt", 8, exp.FROZEN_POLICY) == (
        '{"relation":"equivalent"}',
        7,
        3,
    )
    prompt, prompt_hash = exp.render_prompt({"prompt_payload": {"visible": True}}, [_certificate()])
    assert "CERT-A" in prompt
    assert prompt_hash == exp.sha256_text(prompt)


def test_injected_workers_resume_safety_projection_and_complete_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-LEARN-6962: fake local inference exercises the full worker reduction."""

    source = {
        "event_rows": [_sealed_event("E0", 0), _sealed_event("E1", 1)],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"frozen-model")

    def factory(model_path, gpu, policy):
        assert model_path == str(model_file)
        assert gpu in (0, 1)
        assert policy == exp.FROZEN_POLICY
        return object()

    def generate(model, prompt, max_tokens, policy):
        assert model is not None and "Return JSON" in prompt
        assert max_tokens == 8 and policy.temperature == 0
        return '{"relation":"equivalent"}', 10, 2

    summaries = []
    for model_index, model in enumerate(exp.MODEL_SPECS):
        for arm_index, arm in enumerate(exp.ARMS):
            spec = {**model, "model_path": str(model_file)}
            summary = exp.run_worker(
                source_path=source_path,
                model_spec=spec,
                arm=arm,
                workspace=tmp_path / "workers",
                generator_factory=factory,
                generate_fn=generate,
            )
            summary["process_id"] = 100 + model_index * 10 + arm_index
            summaries.append(summary)
    resumed = exp.run_worker(
        source_path=source_path,
        model_spec={**exp.MODEL_SPECS[0], "model_path": str(model_file)},
        arm="fifo",
        workspace=tmp_path / "workers",
        generator_factory=factory,
        generate_fn=generate,
    )
    assert len(resumed["event_rows"]) == 2

    monkeypatch.setattr(
        exp,
        "_fresh_restore_probe",
        lambda path: {"passed": True, "store_hash": exp.sha256_file(path)},
    )
    safety = exp.run_safety_cases(summaries, source, tmp_path)
    assert all(row["passed"] for values in safety.values() for row in values)
    projections = exp._project_event_rows(summaries[0]["event_rows"])
    assert projections["checkpoint_rows"][0]["completed_key"].endswith("|E0")

    model_hash = exp.sha256_file(model_file)
    source_hashes = {
        **{f"model:{model['hf_id']}": {"sha256": model_hash} for model in exp.MODEL_SPECS},
        "event_sequence": exp.sha256_file(source_path),
    }
    artifact = exp.build_complete_artifact(
        run_date="20260904",
        duration_s=1.0,
        checks=[],
        source_hashes=source_hashes,
        model_specs=[{**row, "model_path": str(model_file)} for row in exp.MODEL_SPECS],
        model_rows=summaries,
        source=source,
        workspace=tmp_path,
    )
    assert artifact["queue_learning_run_complete_score"] == 1
    assert artifact["queue_learning_positive_score"] == 0
    assert exp.validate_artifact(artifact) == []


def test_process_helpers_run_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-LEARN-6962-ARM-ISOLATION: process boundaries fail closed."""

    summary = tmp_path / "summary.json"
    summary.write_text('{"ok": true}', encoding="utf-8")
    monkeypatch.setattr(exp, "_read_json", lambda path: {"ok": True})
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr="", stdout='{"passed":true}'),
    )
    assert exp._run_worker_process(
        source_path=tmp_path / "source.json",
        model_spec={**exp.MODEL_SPECS[0], "model_path": "model.gguf"},
        arm="fifo",
        workspace=tmp_path,
    ) == {"ok": True}
    assert exp._fresh_restore_probe(tmp_path / "store.json")["passed"]

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="failed", stdout=""),
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        exp._run_worker_process(
            source_path=tmp_path / "source.json",
            model_spec={**exp.MODEL_SPECS[0], "model_path": "model.gguf"},
            arm="fifo",
            workspace=tmp_path,
        )
    assert not exp._fresh_restore_probe(tmp_path / "store.json")["passed"]


def test_run_existing_missing_worker_failure_and_disqualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-LEARN-6962: orchestration resumes and preserves every terminal failure."""

    missing_output = tmp_path / "missing-output.json"
    missing = exp.run(
        run_date="20260904",
        source_path=tmp_path / "missing-source.json",
        output_path=missing_output,
        workspace=tmp_path,
    )
    assert missing["verdict_class"] == "blocked"

    existing = _complete_artifact(tmp_path)
    existing_output = tmp_path / "existing.json"
    existing_output.write_text(json.dumps(existing), encoding="utf-8")
    assert (
        exp.run(
            run_date="20260904",
            source_path=tmp_path / "unused.json",
            output_path=existing_output,
            workspace=tmp_path,
        )
        == existing
    )

    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps({"event_rows": []}), encoding="utf-8")
    monkeypatch.setattr(exp, "resolve_model_specs", lambda: [dict(row) for row in exp.MODEL_SPECS])
    monkeypatch.setattr(exp, "collect_preconditions", lambda **kwargs: ([], {}))
    failed_output = tmp_path / "failed.json"
    failed = exp.run(
        run_date="20260904",
        source_path=source_path,
        output_path=failed_output,
        workspace=tmp_path,
        worker_runner=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("worker crash")),
    )
    assert failed["gate_check_summary"]["failed_check"] == "all_model_arm_workers"

    monkeypatch.setattr(
        exp, "build_complete_artifact", lambda **kwargs: _complete_artifact(tmp_path)
    )
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["forced_mismatch"])
    disqualified = exp.run(
        run_date="20260904",
        source_path=source_path,
        output_path=tmp_path / "disqualified.json",
        workspace=tmp_path,
        worker_runner=lambda **kwargs: {},
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["gate_check_summary"]["failed_check"] == "artifact_validation"


def test_main_restore_worker_and_public_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-LEARN-6962: all command modes use explicit paths and terminal rows."""

    store = exp.TransactionalMemoryStore(tmp_path / "store.json")
    assert exp.main(["--restore-probe", str(store.path)]) == 0
    with pytest.raises(SystemExit, match="worker mode requires"):
        exp.main(["--worker"])

    monkeypatch.setattr(exp, "run_worker", lambda **kwargs: {"worker": "complete"})
    worker_output = tmp_path / "worker.json"
    assert (
        exp.main(
            [
                "--worker",
                "--model-spec",
                json.dumps(exp.MODEL_SPECS[0]),
                "--arm",
                "fifo",
                "--worker-output",
                str(worker_output),
                "--workspace",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert json.loads(worker_output.read_text(encoding="utf-8"))["worker"] == "complete"

    artifact = exp.build_blocked_artifact(
        run_date="20260904", checks=[], source_hashes={}, duration_s=0
    )
    monkeypatch.setattr(exp, "run", lambda **kwargs: artifact)
    assert exp.main(["--output", str(tmp_path / "public.json")]) == 0


def test_safe_prompt_and_precondition_exception_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-LEARN-6962: unavailable guards report failure instead of raising."""

    assert exp.future_label_isolation_errors({"arm_prompt_payloads": [{"safe": True}]}) == []
    source_path = tmp_path / "source.json"
    source_path.write_text("{}", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("nvidia-smi absent")),
    )
    monkeypatch.setattr(
        exp,
        "TransactionalMemoryStore",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("store unavailable")),
    )
    exact_module = __import__("carnot.experiment_6961_certified_event_sequence", fromlist=["x"])
    monkeypatch.setattr(
        exact_module,
        "sequence_conformance_errors",
        lambda value: (_ for _ in ()).throw(ValueError("certifier unavailable")),
    )
    checks, _ = exp.collect_preconditions(
        source={},
        source_path=source_path,
        workspace=tmp_path,
        model_specs=[],
    )
    assert not next(row for row in checks if row["check"] == "transactional_stores")["passed"]
    assert not next(row for row in checks if row["check"] == "exact_certifier_access")["passed"]


def test_worker_refuses_unattributed_cuda_offload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-LEARN-6962: a live worker needs PID-linked CUDA memory evidence."""

    source_path = tmp_path / "source.json"
    source_path.write_text('{"event_rows": []}', encoding="utf-8")
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    monkeypatch.setattr(exp, "_load_llama", lambda *args: object())
    monkeypatch.setattr(exp, "_gpu_process_memory", lambda pid: 0)
    with pytest.raises(RuntimeError, match="PID-linked"):
        exp.run_worker(
            source_path=source_path,
            model_spec={**exp.MODEL_SPECS[0], "model_path": str(model_path)},
            arm="fifo",
            workspace=tmp_path,
            generator_factory=exp._load_llama,
        )
