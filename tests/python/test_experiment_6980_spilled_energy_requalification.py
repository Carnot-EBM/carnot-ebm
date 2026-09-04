"""Tests for the final span-localized spilled-energy requalification.

Spec refs: REQ-VERIFY-6980 and SCENARIO-VERIFY-6980-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6980_spilled_energy_requalification as exp


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verification/spec.md"


def _trace(
    step: int,
    *,
    selected: float,
    logsumexp: float,
    entropy: float = 0.2,
    top_probability: float = 0.7,
    token_id: int | None = None,
) -> dict[str, object]:
    emitted = step + 10 if token_id is None else token_id
    return {
        "attempt_key": "model|pair|direct",
        "phase_id": "certificate",
        "phase_step_index": step,
        "attempt_step_index": step,
        "emitted_token_id": emitted,
        "selected_token_logit": selected,
        "full_vocabulary_logsumexp": logsumexp,
        "selected_token_logprob": selected - logsumexp,
        "entropy": entropy,
        "top_probability": top_probability,
        "full_vocabulary_size": 100,
        "finite_logit_count": 100,
        "full_vocabulary_logits_sha256": f"sha256:{step:064x}",
    }


def _span(step: int, start: int, end: int, *, token_id: int | None = None) -> dict[str, object]:
    emitted = step + 10 if token_id is None else token_id
    return {
        "attempt_key": "model|pair|direct",
        "phase_id": "certificate",
        "phase_step_index": step,
        "attempt_step_index": step,
        "emitted_token_id": emitted,
        "phase_byte_start": start,
        "phase_byte_end": end,
        "attempt_byte_start": start,
        "attempt_byte_end": end,
    }


def _metric_row(
    pair_id: str,
    error: bool,
    spilled: float,
    *,
    model: str = "model-a",
    schedule: str = "direct",
    error_class: str | None = None,
) -> dict[str, object]:
    return {
        "attempt_key": f"{model}|{pair_id}|{schedule}|{error}|{spilled}",
        "hf_id": model,
        "pair_id": pair_id,
        "split": "heldout",
        "schedule_id": schedule,
        "formulation_family": "family-a",
        "error_label": int(error),
        "error_class": error_class or ("domain_correspondence" if error else "correct"),
        "eligible": True,
        "abstention_reason": None,
        "signals": {
            "spilled_energy": spilled,
            "marginalized_energy": spilled + 0.1,
            "entropy": spilled - 0.1,
            "top_probability": 1.0 - spilled,
        },
        "terminal": True,
    }


def test_req_verify_6980_spec_anchors_fields_and_scenarios() -> None:
    """REQ-VERIFY-6980 owns each required field and named behavior."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6980") :]
    for name in (
        "PRECONDITIONS",
        "FORMULAS",
        "SPAN",
        "CALIBRATION",
        "DEGENERATE",
        "BOOTSTRAP",
        "RETIREMENT",
    ):
        assert f"SCENARIO-VERIFY-6980-{name}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field in {"schema", "experiment_id", "run_date"}


def test_scenario_verify_6980_paper_formulas_use_adjacent_steps() -> None:
    """SCENARIO-VERIFY-6980-FORMULAS implements the paper at temperature one."""

    current = _trace(4, selected=2.5, logsumexp=3.0, entropy=0.4, top_probability=0.6)
    following = _trace(5, selected=1.0, logsumexp=4.25)
    values = exp.compute_token_signals(current, following)

    assert values["token_energy"] == pytest.approx(-2.5)
    assert values["marginalized_energy"] == pytest.approx(-4.25)
    assert values["spilled_energy"] == pytest.approx(1.75)
    assert values["entropy"] == pytest.approx(0.4)
    assert values["top_probability"] == pytest.approx(0.6)
    assert values["selected_token_logprob_reproduced"] == pytest.approx(-0.5)
    assert values["selected_token_logprob_residual"] == pytest.approx(0.0)


def test_req_verify_6980_formula_validation_fails_closed() -> None:
    """REQ-VERIFY-6980 rejects nonadjacent, incomplete, and irreproducible traces."""

    current = _trace(0, selected=1.0, logsumexp=2.0)
    following = _trace(1, selected=1.0, logsumexp=3.0)
    mutations = (
        (lambda row: row.__setitem__("selected_token_logprob", -0.5), "logprob"),
        (lambda row: row.__setitem__("finite_logit_count", 101), "vocabulary"),
        (lambda row: row.__setitem__("entropy", float("nan")), "finite"),
        (lambda row: row.pop("top_probability"), "missing"),
    )
    for mutate, reason in mutations:
        changed = deepcopy(current)
        mutate(changed)
        with pytest.raises(exp.TraceError, match=reason):
            exp.compute_token_signals(changed, following)

    nonadjacent = deepcopy(following)
    nonadjacent["phase_step_index"] = 2
    with pytest.raises(exp.TraceError, match="adjacent"):
        exp.compute_token_signals(current, nonadjacent)
    wrong_phase = deepcopy(following)
    wrong_phase["phase_id"] = "other"
    with pytest.raises(exp.TraceError, match="phase"):
        exp.compute_token_signals(current, wrong_phase)
    wrong_attempt = deepcopy(following)
    wrong_attempt["attempt_key"] = "other"
    with pytest.raises(exp.TraceError, match="attempt"):
        exp.compute_token_signals(current, wrong_attempt)


def test_scenario_verify_6980_mapping_span_is_exact_utf8_bytes() -> None:
    """SCENARIO-VERIFY-6980-SPAN locates semantic mapping values, not framing bytes."""

    raw = json.dumps(
        {
            "note": "é",
            "objective_map": {"direction": "same", "scale": "1", "offset": "0"},
            "variable_map": [{"source": "x", "target": "y", "scale": "1", "offset": "0"}],
        },
        ensure_ascii=False,
    )
    start, end = exp.extract_mapping_span(raw)
    mapped = raw.encode("utf-8")[start:end].decode("utf-8")

    assert mapped.startswith('{"direction"')
    assert mapped.endswith("]")
    assert '"variable_map"' in mapped
    assert '"objective_map"' in raw.encode("utf-8")[:start].decode("utf-8")
    assert exp.extract_mapping_span("{bad") is None
    assert exp.extract_mapping_span("[]") is None
    assert exp.extract_mapping_span('{"variable_map": []}') is None


@pytest.mark.parametrize(
    "raw",
    (
        "",
        "{",
        "{bad}",
        "{1: 2}",
        '{"a" 1}',
        '{"a": bad}',
        '{"a": 1',
        '{"a": 1 x}',
        '{"a": 1} trailing',
    ),
)
def test_req_verify_6980_json_span_scanner_rejects_ambiguous_framing(raw: str) -> None:
    """REQ-VERIFY-6980 rejects syntax that cannot own exact byte offsets."""

    assert exp._top_level_value_spans(raw) == {}


def test_req_verify_6980_mapping_span_requires_both_value_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6980 does not infer a mapping span after scanner failure."""

    raw = '{"objective_map": {}, "variable_map": []}'
    monkeypatch.setattr(exp, "_top_level_value_spans", lambda _raw: {})
    assert exp.extract_mapping_span(raw) is None


def test_scenario_verify_6980_multi_token_span_uses_mean_pooling() -> None:
    """SCENARIO-VERIFY-6980-SPAN means all four signals over mapping tokens."""

    traces = [
        _trace(0, selected=0.0, logsumexp=1.0, entropy=0.1, top_probability=0.8),
        _trace(1, selected=1.0, logsumexp=2.0, entropy=0.2, top_probability=0.7),
        _trace(2, selected=2.0, logsumexp=5.0, entropy=0.3, top_probability=0.6),
        _trace(3, selected=3.0, logsumexp=6.0, entropy=0.4, top_probability=0.5),
    ]
    spans = [_span(0, 0, 1), _span(1, 1, 2), _span(2, 2, 3), _span(3, 3, 4)]
    reduced = exp.pool_token_span(
        attempt_key="model|pair|direct",
        phase_id="certificate",
        mapping_span=(1, 3),
        trace_rows=traces,
        token_span_rows=spans,
    )

    assert reduced["eligible"] is True
    assert reduced["overlapping_token_count"] == 2
    assert reduced["signals"] == pytest.approx(
        {
            "spilled_energy": 4.0,
            "marginalized_energy": -5.5,
            "entropy": 0.25,
            "top_probability": 0.65,
        }
    )
    assert len(reduced["token_results"]) == 2
    assert reduced["trace_reproduced"] is True


def test_scenario_verify_6980_missing_step_abstains_complete_span() -> None:
    """SCENARIO-VERIFY-6980-SPAN does not average around a missing boundary."""

    traces = [
        _trace(0, selected=0.0, logsumexp=1.0),
        _trace(1, selected=1.0, logsumexp=2.0),
    ]
    spans = [_span(0, 0, 1), _span(1, 1, 2)]
    missing = exp.pool_token_span(
        attempt_key="model|pair|direct",
        phase_id="certificate",
        mapping_span=(0, 2),
        trace_rows=traces,
        token_span_rows=spans,
    )
    empty = exp.pool_token_span(
        attempt_key="model|pair|direct",
        phase_id="certificate",
        mapping_span=(3, 4),
        trace_rows=traces,
        token_span_rows=spans,
    )

    assert missing["eligible"] is False
    assert missing["abstention_reason"] == "missing_adjacent_step"
    assert missing["signals"] == {signal: None for signal in exp.SIGNALS}
    assert empty["abstention_reason"] == "mapping_span_has_no_token_overlap"

    bad_spans = deepcopy(spans)
    bad_spans[0]["emitted_token_id"] = -1
    mismatch = exp.pool_token_span(
        attempt_key="model|pair|direct",
        phase_id="certificate",
        mapping_span=(0, 1),
        trace_rows=traces,
        token_span_rows=bad_spans,
    )
    assert mismatch["abstention_reason"] == "trace_span_token_mismatch"

    invalid = exp.pool_token_span(
        attempt_key="model|pair|direct",
        phase_id="certificate",
        mapping_span=(2, 1),
        trace_rows=traces,
        token_span_rows=spans,
    )
    assert invalid["abstention_reason"] == "mapping_span_invalid"

    duplicate = exp.pool_token_span(
        attempt_key="model|pair|direct",
        phase_id="certificate",
        mapping_span=(0, 1),
        trace_rows=[traces[0], traces[0], traces[1]],
        token_span_rows=spans,
    )
    assert duplicate["abstention_reason"] == "duplicate_phase_step"

    missing_current = exp.pool_token_span(
        attempt_key="model|pair|direct",
        phase_id="certificate",
        mapping_span=(0, 1),
        trace_rows=[traces[1]],
        token_span_rows=spans,
    )
    assert missing_current["abstention_reason"] == "missing_current_step"

    bad_logprob = deepcopy(traces)
    bad_logprob[0]["selected_token_logprob"] = 100.0
    irreproducible = exp.pool_token_span(
        attempt_key="model|pair|direct",
        phase_id="certificate",
        mapping_span=(0, 1),
        trace_rows=bad_logprob,
        token_span_rows=spans,
    )
    assert irreproducible["abstention_reason"] == "selected_logprob_not_reproduced"


def test_req_verify_6980_candidate_reducer_preserves_span_failures() -> None:
    """REQ-VERIFY-6980 keeps parsed candidates with missing span evidence terminal."""

    base = {
        "attempt_key": "a",
        "parser_diagnostic": {"constraintir_shape_valid": True},
        "candidate_raw_text": '{"objective_map": {}, "variable_map": []}',
        "candidate_phase_id": None,
    }
    missing_phase = exp.reduce_candidate_span(base)
    assert missing_phase["abstention_reason"] == "candidate_phase_missing"

    no_mapping = exp.reduce_candidate_span(
        {**base, "candidate_phase_id": "certificate", "candidate_raw_text": "{}"}
    )
    assert no_mapping["abstention_reason"] == "mapping_span_missing"


def test_scenario_verify_6980_calibration_policy_is_split_isolated() -> None:
    """SCENARIO-VERIFY-6980-CALIBRATION forbids held-out policy selection."""

    rows = [
        {**_metric_row("c1", False, 0.1), "split": "calibration"},
        {**_metric_row("c2", False, 0.2), "split": "calibration"},
        {**_metric_row("c3", True, 0.8), "split": "calibration"},
        {**_metric_row("c4", True, 0.9), "split": "calibration"},
    ]
    first, policy_hash = exp.select_calibration_policies(rows)
    second, second_hash = exp.select_calibration_policies(list(reversed(rows)))

    assert first == second
    assert policy_hash == second_hash
    assert all(row["selection_split"] == "calibration" for row in first)
    spilled = next(row for row in first if row["signal"] == "spilled_energy")
    top = next(row for row in first if row["signal"] == "top_probability")
    assert spilled["direction_sign"] == 1
    assert spilled["calibration_balanced_accuracy"] == pytest.approx(1.0)
    assert top["direction_sign"] == -1
    assert top["calibration_balanced_accuracy"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="calibration_rows_only"):
        exp.select_calibration_policies(rows + [_metric_row("h", True, 0.5)])
    abstained = deepcopy(rows)
    abstained[0].update({"eligible": False, "abstention_reason": "missing", "signals": {}})
    selected, _ = exp.select_calibration_policies(abstained)
    assert all(row["eligible_count"] == 3 for row in selected)

    no_eligible = deepcopy(rows)
    for row in no_eligible:
        row.update({"eligible": False, "signals": {}})
    empty_policies, _ = exp.select_calibration_policies(no_eligible)
    assert all(row["threshold"] is None for row in empty_policies)


def test_req_verify_6980_policy_selection_and_indexes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6980 rejects missing signal policies and undefined selection."""

    assert exp._balanced_accuracy([], []) is None
    rows = [{**_metric_row("c", True, 0.5), "split": "calibration"}]
    monkeypatch.setattr(exp, "_balanced_accuracy", lambda _labels, _predictions: None)
    with pytest.raises(ValueError, match="selection_failed"):
        exp.select_calibration_policies(rows)

    with pytest.raises(ValueError, match="roster"):
        exp._policy_index([{"signal": "spilled_energy"}])
    assert exp.binary_auroc([1], [0.5]) is None
    assert exp.average_precision([1], [0.5]) is None


def test_scenario_verify_6980_label_vault_requires_frozen_policy() -> None:
    """SCENARIO-VERIFY-6980-CALIBRATION opens held-out labels once and after freeze."""

    source = [
        {"attempt_key": "c", "split": "calibration", "exact_semantic_success": False},
        {"attempt_key": "h", "split": "heldout", "exact_semantic_success": True},
    ]
    vault = exp.ExactOutcomeVault(source)
    assert vault.open("calibration") == {"c": 1}
    with pytest.raises(ValueError, match="formula_config_hash_required"):
        vault.open("heldout")
    frozen_hash = "sha256:" + "1" * 64
    assert vault.open("heldout", formula_config_hash=frozen_hash) == {"h": 0}
    with pytest.raises(ValueError, match="labels_already_opened"):
        vault.open("heldout", formula_config_hash=frozen_hash)
    assert [row["opening_sequence"] for row in vault.opening_rows] == [1, 2]

    invalid = exp.ExactOutcomeVault(
        [{"attempt_key": "x", "split": "calibration", "exact_semantic_success": None}]
    )
    with pytest.raises(ValueError, match="terminal_exact_labels"):
        invalid.open("calibration")

    with pytest.raises(ValueError, match="opened_outcome_missing"):
        exp._attach_outcomes([{"attempt_key": "missing"}], {}, {})


def test_req_verify_6980_metrics_and_degenerate_classes() -> None:
    """SCENARIO-VERIFY-6980-DEGENERATE leaves undefined metrics null."""

    rows = [
        _metric_row("p1", False, 0.1),
        _metric_row("p1", True, 0.9),
        _metric_row("p2", False, 0.2),
        _metric_row("p2", True, 0.8),
    ]
    policy = {"direction_sign": 1, "threshold": 0.8}
    summary = exp.metric_summary(rows, signal="spilled_energy", policy=policy)

    assert summary["auroc"] == pytest.approx(1.0)
    assert summary["auprc"] == pytest.approx(1.0)
    assert summary["calibration_error"] == pytest.approx(0.0)
    assert summary["paired_ranking_accuracy"] == pytest.approx(1.0)
    assert summary["coverage"] == pytest.approx(1.0)
    assert summary["degenerate_class"] is False

    ties = [_metric_row("p", False, 0.5), _metric_row("p", True, 0.5)]
    tied = exp.metric_summary(ties, signal="spilled_energy", policy=policy)
    assert tied["auroc"] == pytest.approx(0.5)
    assert tied["paired_ranking_accuracy"] == pytest.approx(0.5)

    degenerate = exp.metric_summary(
        [row for row in rows if row["error_label"] == 1],
        signal="spilled_energy",
        policy=policy,
    )
    assert degenerate["auroc"] is None
    assert degenerate["auprc"] is None
    assert degenerate["paired_ranking_accuracy"] is None
    assert degenerate["degenerate_class"] is True


def test_scenario_verify_6980_pair_bootstrap_is_deterministic() -> None:
    """SCENARIO-VERIFY-6980-BOOTSTRAP resamples whole pair clusters."""

    rows = [
        _metric_row("p1", False, 0.1),
        _metric_row("p1", True, 0.9),
        _metric_row("p2", False, 0.4),
        _metric_row("p2", True, 0.6),
        _metric_row("p3", False, 0.8),
        _metric_row("p3", True, 0.2),
    ]
    policy = {"direction_sign": 1, "threshold": 0.5}
    first = exp.bootstrap_metric_interval(
        rows,
        signal="spilled_energy",
        policy=policy,
        metric="auroc",
        seed=6980,
        resamples=200,
    )
    second = exp.bootstrap_metric_interval(
        list(reversed(rows)),
        signal="spilled_energy",
        policy=policy,
        metric="auroc",
        seed=6980,
        resamples=200,
    )

    assert first == second
    assert first["bootstrap_unit"] == "pair_id"
    assert first["requested_resamples"] == 200
    assert first["eligible_resamples"] == 200
    assert first["lower"] <= first["point_estimate"] <= first["upper"]

    control = exp.bootstrap_control_delta(
        rows,
        control_signal="entropy",
        policies={signal: policy for signal in exp.SIGNALS},
        seed=6980,
        resamples=200,
    )
    assert control["control_signal"] == "entropy"
    assert control["paired_row_count"] == len(rows)
    assert control["bootstrap_unit"] == "pair_id"

    no_pairs = exp.bootstrap_metric_interval(
        [rows[0]],
        signal="spilled_energy",
        policy=policy,
        metric="auroc",
        seed=1,
        resamples=10,
    )
    assert no_pairs["point_estimate"] is None
    assert no_pairs["eligible_resamples"] == 0
    assert no_pairs["lower"] is None


def test_req_verify_6980_preconditions_name_failures() -> None:
    """SCENARIO-VERIFY-6980-PRECONDITIONS reports exact expected and observed values."""

    inputs = exp.load_inputs(REPO)
    checks = exp.check_preconditions(inputs)
    assert all(row["passed"] for row in checks)

    changed = dict(inputs)
    changed["bank"] = {**inputs["bank"], "candidate_bank_complete_score": 0}
    failures = exp.gate_summary(exp.check_preconditions(changed))
    assert any(row["failed_check"] == "candidate_bank_complete_score" for row in failures)
    assert all({"failed_check", "expected_value", "observed_value"} <= set(row) for row in failures)

    mismatched = deepcopy(inputs["certification"]["per_candidate_rows"][0])
    mismatched["raw_sha256"] = "sha256:changed"
    changed_certification = {
        **inputs["certification"],
        "per_candidate_rows": [
            mismatched,
            *inputs["certification"]["per_candidate_rows"][1:],
        ],
    }
    changed = {**inputs, "certification": changed_certification}
    failures = exp.gate_summary(exp.check_preconditions(changed))
    assert any(row["failed_check"] == "candidate_hash_agreement" for row in failures)


def test_req_verify_6980_precondition_helpers_reject_malformed_rosters(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6980-PRECONDITIONS keeps malformed input shapes blocked."""

    array_path = tmp_path / "array.json"
    array_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp._read_object(array_path)

    assert exp._attempt_index({"per_attempt_rows": {}}) == {}
    assert (
        exp._attempt_index({"per_attempt_rows": [{"attempt_key": "x"}, {"attempt_key": "x"}]}) == {}
    )
    assert exp._certification_index({"per_candidate_rows": {}}) == {}
    assert (
        exp._certification_index(
            {"per_candidate_rows": [{"attempt_key": "x"}, {"attempt_key": "x"}]}
        )
        == {}
    )
    assert exp._trace_fields_complete({}) is False

    malformed_inputs = {
        "bank": [],
        "certification": [],
        "source_file_hashes": {"bank": None, "certification": None, "prior": None},
    }
    assert exp.gate_summary(exp.check_preconditions(malformed_inputs))

    missing_phase_attempt = {
        "a": {
            "parser_diagnostic": {"constraintir_shape_valid": True},
            "candidate_raw_text": '{"objective_map": {}, "variable_map": []}',
            "candidate_phase_id": "certificate",
            "phase_outputs": [],
        }
    }
    assert exp._mapping_span_observation(missing_phase_attempt) == {
        "parsed_candidate_count": 1,
        "valid_mapping_span_count": 0,
    }
    bad_span_attempt = deepcopy(missing_phase_attempt)
    bad_span_attempt["a"]["phase_outputs"] = [
        {"phase_id": "certificate", "raw_text": bad_span_attempt["a"]["candidate_raw_text"]}
    ]
    assert exp._mapping_span_observation(bad_span_attempt)["valid_mapping_span_count"] == 0


@pytest.mark.parametrize(
    ("updates", "expected"),
    (
        ({"parse_outcome": "rejected"}, "parse"),
        ({"schema_outcome": "rejected"}, "schema"),
        ({"domain_correspondence_outcome": "failed"}, "domain_correspondence"),
        ({"objective_direction_outcome": "failed"}, "objective_direction"),
        ({"objective_order_outcome": "failed"}, "objective_order"),
        ({"satisfiability_outcome": "failed"}, "satisfiability"),
        ({"optimum_outcome": "failed"}, "optimum"),
        ({"solution_space_equivalence_outcome": "failed"}, "solution_space"),
        ({"timeout_outcome": {"z3": True}}, "timeout"),
        ({"unknown_outcome": {"z3": True}}, "unknown"),
        ({"exception_outcome": {"z3": True}}, "exception"),
        ({}, "other_exact_failure"),
    ),
)
def test_req_verify_6980_error_classes_are_explicit(updates: dict, expected: str) -> None:
    """REQ-VERIFY-6980 retains each exact error mechanism as a separate class."""

    assert exp.classify_error({"exact_semantic_success": False, **updates}) == expected
    assert exp.classify_error({"exact_semantic_success": True}) == "correct"


def test_scenario_verify_6980_prior_null_retires_once() -> None:
    """SCENARIO-VERIFY-6980-RETIREMENT closes the repeated null lineage."""

    overall = {"auroc": 0.64}
    interval = {"lower": 0.51, "upper": 0.75}
    comparisons = [
        {
            "hf_id": model,
            "control_signal": control,
            "delta_lower": -0.01,
        }
        for model in exp.MODEL_FAMILIES
        for control in exp.CONTROL_SIGNALS
    ]
    verdict = exp.decide_requalification(
        evaluation_complete_score=1,
        overall_spilled=overall,
        spilled_interval=interval,
        per_model_control_comparisons=comparisons,
    )

    assert verdict["spilled_energy_requalified_score"] == 0
    assert verdict["verdict_class"] == "null"
    assert verdict["honest_verdict"].startswith("complete_")
    assert verdict["retirement_recommendation"]["retire_if_same_verdict"] is True
    assert verdict["retirement_recommendation"]["propose_retry"] is False

    positive = deepcopy(comparisons)
    for row in positive:
        if row["hf_id"] in exp.MODEL_FAMILIES[:2]:
            row["delta_lower"] = 0.01
    qualified = exp.decide_requalification(
        evaluation_complete_score=1,
        overall_spilled={"auroc": 0.7},
        spilled_interval={"lower": 0.6, "upper": 0.8},
        per_model_control_comparisons=positive,
    )
    assert qualified["spilled_energy_requalified_score"] == 1
    assert qualified["verdict_class"] == "positive"
    assert qualified["retirement_recommendation"]["retire"] is False


@pytest.fixture(scope="module")
def frozen_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """REQ-VERIFY-6980 runs the frozen integration path once in temporary storage."""

    output = tmp_path_factory.mktemp("exp6980") / "artifact.json"
    artifact = exp.run(date="20260904", repo_root=REPO, output_path=output)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    return artifact


def test_req_verify_6980_frozen_artifact_is_complete(frozen_artifact: dict[str, object]) -> None:
    """REQ-VERIFY-6980 emits one terminal span row for every frozen candidate."""

    artifact = frozen_artifact
    assert len(artifact["rows"]) == 108
    assert artifact["rows"] == artifact["per_span_results"]
    assert len(artifact["trace_reproduction_rows"]) == 108
    assert artifact["spilled_energy_evaluation_complete_score"] == 1
    assert type(artifact["spilled_energy_requalified_score"]) is int
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert all(row["terminal"] for row in artifact["rows"])
    assert all(row["passed"] for row in artifact["preconditions_checked"])
    assert artifact["formula_config"]["pooling"] == "arithmetic_mean"
    assert artifact["formula_config"]["temperature"] == 1.0


def test_req_verify_6980_validator_rejects_derived_drift(
    frozen_artifact: dict[str, object],
) -> None:
    """REQ-VERIFY-6980 makes per-span evidence authoritative over summaries."""

    mutations = (
        (lambda row: row.pop("rows"), "required_fields_missing"),
        (lambda row: row["rows"].clear(), "rows_projection_mismatch"),
        (
            lambda row: row.__setitem__("spilled_energy_evaluation_complete_score", True),
            "not_bare_int",
        ),
        (lambda row: row.__setitem__("inference_substrate", "other"), "substrate"),
        (lambda row: row.__setitem__("verifier_is_oracle", True), "oracle"),
        (lambda row: row["field_principles"].pop("rows"), "field_principles"),
    )
    for mutate, reason in mutations:
        changed = deepcopy(frozen_artifact)
        mutate(changed)
        changed["reproducibility_checksum"] = exp.payload_checksum(changed)
        with pytest.raises(ValueError, match=reason):
            exp.validate_artifact(changed)

    changed = deepcopy(frozen_artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    with pytest.raises(ValueError, match="checksum"):
        exp.validate_artifact(changed)


def test_req_verify_6980_validator_rejects_blocked_and_terminal_contract_drift(
    frozen_artifact: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6980 validates blocked receipts and every derived terminal surface."""

    blocked = deepcopy(frozen_artifact)
    blocked.update(
        {
            "verdict_class": "blocked",
            "honest_verdict": "wrong",
            "gate_check_summary": [{"failed_check": "x"}],
        }
    )
    blocked["reproducibility_checksum"] = exp.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked_verdict"):
        exp.validate_artifact(blocked)
    blocked["honest_verdict"] = "blocked_spilled_energy_requalification"
    blocked["gate_check_summary"] = []
    blocked["reproducibility_checksum"] = exp.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked_gate_summary"):
        exp.validate_artifact(blocked)

    invalid_class = deepcopy(frozen_artifact)
    invalid_class["verdict_class"] = "other"
    invalid_class["reproducibility_checksum"] = exp.payload_checksum(invalid_class)
    with pytest.raises(ValueError, match="verdict_class"):
        exp.validate_artifact(invalid_class)

    short = deepcopy(frozen_artifact)
    short["rows"].pop()
    short["per_span_results"] = deepcopy(short["rows"])
    short["reproducibility_checksum"] = exp.payload_checksum(short)
    with pytest.raises(ValueError, match="span_row_count"):
        exp.validate_artifact(short)

    trace_drift = deepcopy(frozen_artifact)
    trace_drift["trace_reproduction_rows"].clear()
    trace_drift["reproducibility_checksum"] = exp.payload_checksum(trace_drift)
    with pytest.raises(ValueError, match="trace_reproduction"):
        exp.validate_artifact(trace_drift)

    config_drift = deepcopy(frozen_artifact)
    config_drift["formula_config"]["pooling"] = "max"
    config_drift["reproducibility_checksum"] = exp.payload_checksum(config_drift)
    with pytest.raises(ValueError, match="formula_config_hash"):
        exp.validate_artifact(config_drift)

    saved_surfaces = {
        name: deepcopy(frozen_artifact[name])
        for name in (
            "heldout_metric_rows",
            "per_model_metric_rows",
            "per_schedule_metric_rows",
            "per_formulation_metric_rows",
            "per_error_class_metric_rows",
        )
    }
    saved_bootstrap = deepcopy(frozen_artifact["bootstrap_interval_rows"])
    saved_controls = deepcopy(frozen_artifact["control_comparison_rows"])
    monkeypatch.setattr(exp, "build_metric_surfaces", lambda _rows, _policies: saved_surfaces)
    monkeypatch.setattr(exp, "build_bootstrap_rows", lambda _rows, _policies: saved_bootstrap)
    monkeypatch.setattr(exp, "build_control_comparisons", lambda _rows, _policies: saved_controls)

    mutations = (
        ("heldout_metric_rows", lambda row: row["heldout_metric_rows"].clear(), "metric_rows"),
        (
            "bootstrap_interval_rows",
            lambda row: row["bootstrap_interval_rows"].clear(),
            "bootstrap_interval",
        ),
        (
            "control_comparison_rows",
            lambda row: row["control_comparison_rows"].clear(),
            "control_comparison",
        ),
        ("coverage_rows", lambda row: row["coverage_rows"].clear(), "coverage_rows"),
        ("abstention_rows", lambda row: row["abstention_rows"].clear(), "abstention_rows"),
        (
            "spilled_energy_evaluation_complete_score",
            lambda row: row.__setitem__("spilled_energy_evaluation_complete_score", 0),
            "evaluation_complete",
        ),
        (
            "spilled_energy_requalified_score",
            lambda row: row.__setitem__(
                "spilled_energy_requalified_score",
                1 - row["spilled_energy_requalified_score"],
            ),
            "decision_mismatch",
        ),
        (
            "gate_check_summary",
            lambda row: row["gate_check_summary"].append({"failed_check": "late"}),
            "completed_gate_summary",
        ),
    )
    for _field, mutate, reason in mutations:
        changed = deepcopy(frozen_artifact)
        mutate(changed)
        changed["reproducibility_checksum"] = exp.payload_checksum(changed)
        with pytest.raises(ValueError, match=reason):
            exp.validate_artifact(changed)


def test_req_verify_6980_absent_sources_write_blocked_contract(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6980-PRECONDITIONS keeps an unreadable run schema-complete."""

    output = tmp_path / "blocked.json"
    artifact = exp.run(date="20260904", repo_root=tmp_path, output_path=output)

    exp.validate_artifact(artifact)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_spilled_energy_requalification"
    assert artifact["spilled_energy_evaluation_complete_score"] == 0
    assert artifact["spilled_energy_requalified_score"] == 0
    assert artifact["gate_check_summary"]
    assert output.is_file()


def test_req_verify_6980_main_reports_terminal_scores(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6980 exposes the required command without adding model behavior."""

    output = tmp_path / "artifact.json"
    monkeypatch.setattr(
        exp,
        "run",
        lambda **_kwargs: {
            "verdict_class": "null",
            "honest_verdict": "complete_null_spilled_energy_requalification_retired",
            "spilled_energy_evaluation_complete_score": 1,
            "spilled_energy_requalified_score": 0,
        },
    )
    assert exp.main(["--date", "20260904", "--output", str(output)]) == 0
    reported = json.loads(capsys.readouterr().out)
    assert reported["artifact"] == str(output)
    assert reported["verdict_class"] == "null"
