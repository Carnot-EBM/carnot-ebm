"""Tests for REQ-VERIFY-7462 and SCENARIO-VERIFY-7462-*.

These tests cover only the new option protocol. They use scripted score buffers,
so they make no model-inference or scientific-efficacy claim.
"""

from __future__ import annotations

from copy import deepcopy
import math

import pytest

from carnot import experiment_7462_v654_option_protocol as exp


def _tokenizer(text: str, *, add_bos: bool) -> list[int]:
    """Tokenize labels as one token while keeping prompt boundaries observable."""

    labels = {" A": 11, " B": 17}
    prefix: list[int] = [1] if add_bos else []
    if text in labels:
        return prefix + [labels[text]]
    for label, token_id in labels.items():
        if text.endswith(label):
            return _tokenizer(text[: -len(label)], add_bos=add_bos) + [token_id]
    return prefix + [2 + (ord(character) % 7) for character in text]


def _score_rows(prompt_length: int, left: float = 4.0, right: float = 1.0) -> list[list[float]]:
    """Return a prompt surface plus an unused uniform buffer row."""

    rows = [[0.0] * 24 for _ in range(prompt_length + 3)]
    rows[prompt_length - 1][11] = left
    rows[prompt_length - 1][17] = right
    return rows


def _readout(order: tuple[str, str], left: float = 4.0, right: float = 1.0) -> dict[str, object]:
    prompt = exp.build_option_prompt("Complete source.", "Complete response.", order)
    tokens = _tokenizer(prompt, add_bos=True)
    return exp.read_option_logits(
        "Complete source.",
        "Complete response.",
        order,
        tokenize=_tokenizer,
        score_rows=_score_rows(len(tokens), left, right),
    )


def _row(
    row_key: str,
    group_id: str,
    role: str,
    source: str,
    *,
    corpus: str = "ragtruth",
    label: int = 1,
    ambiguous: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    predictor = {
        "row_key": row_key,
        "group_id": group_id,
        "corpus": corpus,
        "role": role,
        "source_text": source,
        "response_text": f"Response for {group_id}.",
        "source_features": {"forbidden_old_feature": 0.5},
    }
    evaluator = {
        "row_key": row_key,
        "group_id": group_id,
        "corpus": corpus,
        "role": role,
        "source_id": f"source-{group_id}",
        "response_id": f"response-{group_id}",
        "official_split": "test" if role in {"internal_test", "online"} else "train",
        "label": label,
        "annotation_labels": ["Questionable"] if ambiguous else [],
        "ambiguous": ambiguous,
        "label_policy": "human_source_support",
    }
    return predictor, evaluator


def test_req_verify_7462_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7462 and all six scenarios exist before implementation."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "### REQ-VERIFY-7462:" in text
    for name in ("LOGITS", "COHORT", "SUPPORT", "PLAN", "MINIMA", "NO-MODEL", "E2E"):
        assert f"SCENARIO-VERIFY-7462-{name}" in text


def test_scenario_verify_7462_logits_uses_last_evaluated_position_and_raw_values() -> None:
    """SCENARIO-VERIFY-7462-LOGITS catches the scores[-1] unused-buffer defect."""

    result = _readout(exp.OPTION_IDS)
    assert result["last_evaluated_prompt_position"] == result["prompt_token_count"] - 1
    assert result["score_buffer_rows"] > result["prompt_token_count"]
    assert result["raw_logits_by_option_id"] == {
        "supported": 4.0,
        "contains_unsupported": 1.0,
    }
    probabilities = result["probabilities_by_option_id"]
    assert probabilities["supported"] > probabilities["contains_unsupported"]
    assert math.isclose(sum(probabilities.values()), 1.0)


def test_scenario_verify_7462_logits_rejects_token_and_surface_mutations() -> None:
    """SCENARIO-VERIFY-7462-LOGITS rejects six known interface mutations."""

    prompt = exp.build_option_prompt("S", "R", exp.OPTION_IDS)
    prompt_count = len(_tokenizer(prompt, add_bos=True))
    with pytest.raises(exp.OptionProtocolError, match="option_ids"):
        exp.build_option_prompt("S", "R", ("supported", "supported"))

    def missing(text: str, *, add_bos: bool) -> list[int]:
        return [] if text == " B" else _tokenizer(text, add_bos=add_bos)

    with pytest.raises(exp.OptionProtocolError, match="single_token"):
        exp.read_option_logits("S", "R", exp.OPTION_IDS, tokenize=missing, score_rows=[])

    def duplicate(text: str, *, add_bos: bool) -> list[int]:
        if text in {" A", " B"}:
            return ([1] if add_bos else []) + [11]
        if text.endswith(" B"):
            return duplicate(text[:-2], add_bos=add_bos) + [11]
        return _tokenizer(text, add_bos=add_bos)

    with pytest.raises(exp.OptionProtocolError, match="duplicate_label_token"):
        exp.read_option_logits("S", "R", exp.OPTION_IDS, tokenize=duplicate, score_rows=[])

    broken_boundary = lambda text, add_bos: (  # noqa: E731
        _tokenizer(text, add_bos=add_bos) + ([23] if len(text) > 2 and text.endswith(" A") else [])
    )
    with pytest.raises(exp.OptionProtocolError, match="prompt_label_boundary"):
        exp.read_option_logits("S", "R", exp.OPTION_IDS, tokenize=broken_boundary, score_rows=[])

    uniform = _score_rows(prompt_count, 2.0, 2.0)
    with pytest.raises(exp.OptionProtocolError, match="uniform_logits"):
        exp.read_option_logits("S", "R", exp.OPTION_IDS, tokenize=_tokenizer, score_rows=uniform)

    nonfinite = _score_rows(prompt_count)
    nonfinite[prompt_count - 1][17] = float("nan")
    with pytest.raises(exp.OptionProtocolError, match="nonfinite_logit"):
        exp.read_option_logits("S", "R", exp.OPTION_IDS, tokenize=_tokenizer, score_rows=nonfinite)

    with pytest.raises(exp.OptionProtocolError, match="score_buffer_short"):
        exp.read_option_logits(
            "S", "R", exp.OPTION_IDS, tokenize=_tokenizer, score_rows=[[0.0] * 24]
        )


def test_scenario_verify_7462_logits_remaps_orders_and_catches_option_swap() -> None:
    """SCENARIO-VERIFY-7462-LOGITS averages by stable ID and rejects ID swaps."""

    original = _readout(exp.OPTION_IDS, 3.0, 1.0)
    reversed_readout = _readout(tuple(reversed(exp.OPTION_IDS)), 1.5, 2.5)
    combined = exp.average_order_readouts(original, reversed_readout)
    assert set(combined) == set(exp.OPTION_IDS)
    assert combined["supported"] == pytest.approx(
        (
            original["probabilities_by_option_id"]["supported"]
            + reversed_readout["probabilities_by_option_id"]["supported"]
        )
        / 2
    )

    swapped = deepcopy(reversed_readout)
    swapped["option_ids_in_prompt"] = list(exp.OPTION_IDS)
    with pytest.raises(exp.OptionProtocolError, match="option_order_pair"):
        exp.average_order_readouts(original, swapped)

    missing = deepcopy(reversed_readout)
    del missing["probabilities_by_option_id"]["supported"]
    with pytest.raises(exp.OptionProtocolError, match="probability_ids"):
        exp.average_order_readouts(original, missing)


def test_scenario_verify_7462_cohort_preserves_roles_and_filters_duplicates() -> None:
    """SCENARIO-VERIFY-7462-COHORT selects online rows without role leakage."""

    old_pairs = [
        _row("t", "g-train", "training", "Train source."),
        _row("c", "g-cal", "calibration_tuning", "Calibration source."),
        _row("i", "g-test", "internal_test", "Internal source."),
        _row("e", "g-external", "external", "External source.", corpus="faithbench"),
    ]
    candidate_pairs = [
        _row("dup", "g-duplicate", "internal_test", "Internal source."),
        _row("o2", "g-online-2", "internal_test", "Online source two.", label=0),
        _row("o1", "g-online-1", "internal_test", "Online source one."),
    ]
    frozen = exp.freeze_cohort(
        [row[0] for row in old_pairs],
        [row[1] for row in old_pairs],
        [row[0] for row in candidate_pairs],
        [row[1] for row in candidate_pairs],
        online_cap=2,
    )
    assert frozen["counts"] == {
        "training": 1,
        "calibration_tuning": 1,
        "internal_test": 1,
        "online": 2,
        "external": 1,
    }
    assert len(frozen["predictors"]) == len(frozen["evaluators"]) == 6
    assert all("source_features" not in row for row in frozen["predictors"])
    assert sum(row["role"] == "online" for row in frozen["predictors"]) == 2
    assert any(
        row["reason"] == "excluded_previous_role_source_hash" for row in frozen["exclusions"]
    )
    role_by_source: dict[str, set[str]] = {}
    for row in frozen["groups"]:
        role_by_source.setdefault(row["source_hash"], set()).add(row["role"])
    assert all(len(roles) == 1 for roles in role_by_source.values())

    duplicated_old = [*old_pairs, _row("t2", "g-train-2", "training", "Train source.")]
    with pytest.raises(exp.OptionProtocolError, match="cross_role_duplicate"):
        exp.freeze_cohort(
            [row[0] for row in duplicated_old],
            [row[1] for row in duplicated_old],
            [],
            [],
        )


def test_scenario_verify_7462_support_and_comparison_plan_are_frozen() -> None:
    """SCENARIO-VERIFY-7462-SUPPORT/PLAN keep labels and costs prespecified."""

    supported = _row("s", "g-s", "training", "S", label=1)[1]
    unsupported = _row("u", "g-u", "training", "U", label=0)[1]
    contested = _row("q", "g-q", "external", "Q", label=0, ambiguous=True)[1]
    assert exp.annotation_disposition(supported) == "supported"
    assert exp.annotation_disposition(unsupported) == "contains_unsupported"
    assert exp.annotation_disposition(contested) == "uncertain_or_contested"
    with pytest.raises(exp.OptionProtocolError, match="support_label"):
        exp.annotation_disposition({"label": 3, "ambiguous": False})

    plan = exp.comparison_plan()
    assert plan["fit_seeds"] == [65_401, 65_402, 65_403, 65_404, 65_405]
    assert plan["bootstrap"] == {"draws": 10_000, "unit": "source_group"}
    assert plan["familywise_alpha"] == 0.05
    assert plan["primary_outcome"] == "paired_multiclass_brier"
    assert plan["calibration_controls"]["reserved_groups"] == 40
    assert plan["decision_costs"] == {
        "false_accept": 10.0,
        "false_reject": 1.0,
        "escalate": 0.2,
        "false_accept_sensitivity": [5.0, 20.0],
        "operator_approved_deployment_policy": False,
    }
    assert plan["threshold_fit_role"] == "calibration_tuning_only"


def test_scenario_verify_7462_minima_and_no_model_receipt() -> None:
    """SCENARIO-VERIFY-7462-MINIMA/NO-MODEL separate readiness from benefit."""

    counts = {
        "training": 180,
        "calibration_tuning": 60,
        "internal_test": 60,
        "online": 160,
        "external": 74,
    }
    gates = exp.protocol_gates(
        counts,
        interface_passed=True,
        cohort_disjoint=True,
        predictor_isolated=True,
        validation_passed=True,
    )
    reduced = exp.reduce_protocol(gates, flagged_adversarial=False)
    assert reduced["option_protocol_ready_score"] == 1
    assert reduced["confirmatory_minima_met"] is True
    assert reduced["verdict_class"] == "null"

    short = exp.protocol_gates(
        {**counts, "online": 119},
        interface_passed=True,
        cohort_disjoint=True,
        predictor_isolated=True,
        validation_passed=True,
    )
    short_reduction = exp.reduce_protocol(short, flagged_adversarial=False)
    assert short_reduction["option_protocol_ready_score"] == 1
    assert short_reduction["confirmatory_minima_met"] is False
    assert short_reduction["verdict_class"] == "null"

    failed = deepcopy(gates)
    next(row for row in failed if row["check"] == "raw_logit_interface")["passed"] = False
    assert exp.reduce_protocol(failed, flagged_adversarial=False)["verdict_class"] == "disqualified"
    assert exp.reduce_protocol(gates, flagged_adversarial=True)["verdict_class"] == "disqualified"

    assert exp.MODEL_SPECS == []
    assert exp.MODEL_SPECS_LOWER == []
    assert set(exp.INVOCATION_COUNTS) == {
        f"{operation}_{state}"
        for operation in ("model_loads", "forward_calls", "generation_calls")
        for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    assert set(exp.INVOCATION_COUNTS.values()) == {0}
    assert exp.SMALL_EBM_TRAINING == {
        "attempted": False,
        "fit_attempts": 0,
        "fit_completions": 0,
        "fit_failures": 0,
        "duration_s": 0.0,
        "receipt_class": "small_ebm_training",
        "deferred_to": "post_capture_numeric_fit",
    }


def test_artifact_checksum_and_required_fields_are_stable() -> None:
    """REQ-VERIFY-7462 binds its terminal schema without wrapping bare scores."""

    artifact = {field: None for field in exp.REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "schema": exp.SCHEMA,
            "experiment_id": exp.EXPERIMENT_ID,
            "milestone": exp.MILESTONE,
            "run_date": exp.RUN_DATE,
            "MODEL_SPECS": [],
            "model_specs": [],
            "model_invoked": False,
            "invocation_counts": deepcopy(exp.INVOCATION_COUNTS),
            "inference_substrate_class": "no_model_load",
            "execution_venue": "host",
            "option_protocol_ready_score": 1,
            "promotion_score": 0,
            "flagged_adversarial": False,
            "field_principles": {field: "why" for field in exp.REQUIRED_ARTIFACT_FIELDS},
            "reproducibility_checksum": None,
        }
    )
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert exp.artifact_checksum(artifact) == artifact["reproducibility_checksum"]
    assert exp.validate_artifact_shape(artifact) == []
    artifact["model_invoked"] = True
    assert "current_model_contract_invalid" in exp.validate_artifact_shape(artifact)
    assert exp.validate_artifact_shape({})[0].startswith("missing_field:")
