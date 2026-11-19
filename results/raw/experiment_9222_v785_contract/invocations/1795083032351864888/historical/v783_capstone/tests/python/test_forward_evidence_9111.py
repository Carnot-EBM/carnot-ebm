"""REQ-REPORT-9111-V777 / SCENARIO-REPORT-9111-V777: evidence is operation-specific."""

from copy import deepcopy
from typing import Any

import pytest

from carnot.reporting import current_work_receipt as receipt
from scripts import adversarial_verify as av


def panel() -> dict[str, Any]:
    """Use original-style sibling identities so cardinality cannot prove itself."""
    rows = [
        dict(
            unit_id=f"q{q}-{s}",
            qid=f"q{q}",
            group_id=f"g{q}",
            role="CAL" if q < 50 else "TEST",
            status="completed",
        )
        for q in range(150)
        for s in range(3)
    ]
    return dict(
        rows=rows,
        test_predictions=[r.copy() for r in rows if r["role"] == "TEST"],
        qid_rows=[
            dict(qid=f"q{q}", group_id=f"g{q}", denominator=3, half_denominator=1)
            for q in range(50, 150)
        ],
        structural_count_evidence="qid-panel-v1",
        effective_independent_groups=150,
        denominator=300,
        half_denominator=150,
    )


def invocation(**counts: Any) -> dict[str, Any]:
    """A declared class alone must not turn a request into execution."""
    return dict(
        inference_substrate="frozen_gguf_forward_scoring",
        inference_substrate_class="model_load_no_generation",
        duration_s=61,
        model_invocation_counts=counts,
    )


@pytest.mark.parametrize(
    "counts",
    [
        dict(
            model_loads_attempted=1,
            model_loads_completed=1,
            forward_calls=900,
            generation_calls_attempted=0,
            generation_calls_completed=0,
        ),
        dict(
            model_loads_attempted=1,
            model_loads_completed=0,
            model_loads_failed=1,
            generation_calls_attempted=0,
            generation_calls_completed=0,
        ),
        dict(
            model_loads_attempted=1,
            model_loads_completed=1,
            forward_calls=2,
            generation_calls_attempted=1,
            generation_calls_completed=1,
        ),
        dict(
            model_loads_attempted=1,
            model_loads_completed=1,
            forward_calls=400,
            generation_calls_attempted=20,
            generation_calls_completed=20,
        ),
    ],
)
def test_operation_specific_receipts(counts: dict[str, int]) -> None:
    """REQ-REPORT-9111-V777: sampling zeros do not erase load or forward work."""
    value = invocation(**counts)
    classification = av._classify_current_task_inference_claim(value)
    assert classification["state"] == av.CLAIM_STATE_LIVE
    flags: list[Any] = []
    av.check_substrate_class(value, flags)
    assert not flags


@pytest.mark.parametrize(
    "counts",
    [
        dict(model_loads_attempted=0, model_loads_completed=0, forward_calls=1),
        dict(model_loads_attempted=1, model_loads_completed=0, forward_calls=1),
        dict(
            model_loads_attempted=1,
            model_loads_completed=1,
            generation_calls_attempted=0,
            generation_calls_completed=1,
        ),
        dict(model_loads_attempted=1, model_loads_completed=2),
        dict(generation_calls_attempted=1, generation_calls_completed=2),
        dict(forward_calls_attempted=1, forward_calls_completed=2),
    ],
)
def test_conflicting_positive_counters(counts: dict[str, int]) -> None:
    """REQ-REPORT-9111-V777: contradictory execution remains disqualified."""
    assert av._classify_current_task_inference_claim(invocation(**counts))["state"] == (
        av.CLAIM_STATE_CONTRADICTORY
    )


def test_no_load_and_historical_scope() -> None:
    """REQ-REPORT-9111-V777: requests and cited receipts cannot claim current work."""
    value = dict(
        MODEL_SPECS=[dict(hf_id="requested-only")],
        model_invoked=False,
        inference_substrate="deterministic_no_model_load",
        historical_receipts=invocation(model_loads_attempted=1, forward_calls=5),
    )
    claim = av._classify_current_task_inference_claim(value)
    assert claim["state"] == av.CLAIM_STATE_NON_LIVE
    assert claim["external_evidence"] and not claim["live_evidence"]
    value["current_work_receipts"] = [invocation(model_loads_attempted=1)]
    assert av._classify_current_task_inference_claim(value)["state"] == (
        av.CLAIM_STATE_CONTRADICTORY
    )


@pytest.mark.parametrize("kind", ["model_bounded_generation", "model_full_generation"])
def test_generation_class_requires_sampling(kind: str) -> None:
    """REQ-REPORT-9111-V777: a load cannot excuse an explicit but unperformed generation claim."""
    value = invocation(
        model_loads_attempted=1,
        model_loads_completed=1,
        generation_calls_attempted=0,
        generation_calls_completed=0,
    )
    value["inference_substrate_class"] = kind
    flags: list[Any] = []
    av.check_substrate_class(value, flags)
    assert any(f.severity == "critical" for f in flags)


def test_malformed_class_stays_a_validation_error() -> None:
    """REQ-REPORT-9111-V777: malformed declarations must not crash the shared reader."""
    value = invocation(model_loads_attempted=1)
    value["inference_substrate_class"] = {}
    flags: list[Any] = []
    av.check_substrate_class(value, flags)
    assert any(f.kind == av.SUBSTRATE_CLASS_MISMATCH_KIND for f in flags)
    assert av._typed_invocation_evidence(value)[0]
    current = {"current_receipt": {"scope": "current", "model_invoked": True}}
    assert av._typed_invocation_evidence(current)[0][0]["scope"] == "current_task"


def test_forward_event_ledger() -> None:
    """REQ-REPORT-9111-V777: forwards have their own attempted and completed counts."""
    events = [
        dict(
            scope="current",
            transport="owned_runtime",
            run_id="forward",
            owner_pid=7,
            call_id="f",
            operation="forward",
            state=state,
            monotonic_ns=i,
        )
        for i, state in enumerate(["attempted", "completed"])
    ]
    value = receipt.build_current_work_receipt(
        run_id="forward",
        owner_pid=7,
        events=events,
        inference_substrate="forward_scoring",
        inference_substrate_details={},
        inference_substrate_class="model_load_no_generation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=2,
    )
    assert value["invocation_counts"]["forward_calls_completed"] == 1
    assert value["invocation_counts"]["generation_calls_completed"] == 0
    assert value["model_invoked"] is True
    assert receipt.validate_current_work_receipt(value) == []


def test_empty_ledger_preserves_historical_counter_shape() -> None:
    """REQ-REPORT-9111-V777: new forward support cannot alter immutable zero-load dictionaries."""
    assert not any(key.startswith("forward_calls_") for key in receipt.ZERO_INVOCATION_COUNTS)
    counts, errors = receipt._reduce_events([], run_id="old", owner_pid=7)
    assert not errors and counts == receipt.ZERO_INVOCATION_COUNTS


def test_roster_derived_equality() -> None:
    """REQ-REPORT-9111-V777: two independently reduced integer counts may coincide."""
    flags: list[Any] = []
    av.check_tautology(panel(), flags)
    assert not flags


@pytest.mark.parametrize(
    "change", ["unbacked", "noninteger", "qid_denominator", "duplicate", "independent_metrics"]
)
def test_structural_proof_rejects_unsupported_equality(change: str) -> None:
    """SCENARIO-REPORT-9111-V777: field suffixes cannot substitute for row evidence."""
    value = deepcopy(panel())
    if change == "unbacked":
        value.pop("rows")
    elif change == "noninteger":
        value["effective_independent_groups"] = 150.0
    elif change == "qid_denominator":
        value["qid_rows"][0]["denominator"] = 4
    elif change == "duplicate":
        value["rows"][0] = value["rows"][1]
    else:
        value.update(accuracy=0.7654321, auroc=0.7654321)
    flags: list[Any] = []
    av.check_tautology(value, flags)
    assert any(f.severity == "critical" for f in flags)


def test_semantic_control_panel_is_complete() -> None:
    """REQ-REPORT-9111-V777: every frozen acceptance and rejection control runs."""
    from carnot.reporting.forward_evidence_9111 import semantic_control_rows

    rows = semantic_control_rows()
    assert len(rows) >= 12 and all(row["passed"] for row in rows)
    assert {
        "no_load_refusal",
        "failed_attempted_load",
        "zero_sampling_forward",
        "bounded_generation",
        "full_generation",
        "positive_counter_conflict",
        "historical_receipt",
        "current_receipt",
    } <= {r["unit_id"] for r in rows}


def test_legacy_receipt_and_partial_forward_counter_refusal(tmp_path: Any) -> None:
    """SCENARIO-REPORT-9111-V777: backward compatibility cannot omit actual forwards."""
    value = receipt.build_current_work_receipt(
        run_id="legacy",
        owner_pid=1,
        events=[],
        inference_substrate="no_model_load",
        inference_substrate_details={},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=1,
    )
    value["invocation_counts"] = {
        k: v for k, v in value["invocation_counts"].items() if not k.startswith("forward_calls_")
    }
    assert receipt.validate_current_work_receipt(value, root=tmp_path) == []
    value["invocation_counts"]["forward_calls_attempted"] = 1
    assert "invocation_counts_mismatch" in receipt.validate_current_work_receipt(value)


@pytest.mark.parametrize(
    "change",
    ["bounds", "identity", "role", "alias", "predictions", "qid_rows", "qid_missing", "shape"],
)
def test_malformed_panel_is_critical(change: str) -> None:
    """REQ-REPORT-9111-V777: proof validation fails closed at every input boundary."""
    value = panel()
    if change == "bounds":
        value["rows"] = []
    elif change == "identity":
        value["rows"][0]["qid"] = False
    elif change == "role":
        value["rows"][0]["role"] = "TEST"
    elif change == "alias":
        for r in value["rows"]:
            if r["group_id"] == "g0":
                r["qid"] = "q1"
    elif change == "predictions":
        value["test_predictions"].pop()
    elif change == "qid_rows":
        value["qid_rows"][0]["group_id"] = "unknown"
    elif change == "qid_missing":
        value["qid_rows"].pop()
    else:
        value["rows"] = [None]
    flags: list[Any] = []
    av.check_tautology(value, flags)
    assert any(f.kind == "STRUCTURAL_COUNT_EVIDENCE" for f in flags)
