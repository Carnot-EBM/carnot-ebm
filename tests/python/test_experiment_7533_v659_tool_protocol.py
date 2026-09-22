"""Tests for the V659 label-blind tool protocol.

Spec refs: REQ-VERIFY-7533 and SCENARIO-VERIFY-7533-*.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from carnot.experiment_7533_v659_tool_protocol import (
    EMPTY_EVIDENCE_MARKER,
    ROLE_COUNTS,
    ToolProtocolError,
    allocate_roles,
    annotation_binary_label,
    build_components,
    build_feature_vector,
    build_intervention_manifest,
    build_tool_prompt,
    choose_action,
    qualify_guard_rows,
    read_protocol,
    reload_protocol_shards,
    seal_protocol_shards,
    semantic_unsupported_probability,
)


def _row(
    name: str,
    *,
    split: str = "train",
    instance: str | None = None,
    tool_type: str = "read_file",
    context: str | None = None,
    answer: str | None = None,
) -> dict[str, object]:
    return {
        "official_split": split,
        "dataset": "lettucedetect-tool-output",
        "context": context or f"context {name}",
        "question": f"question {name}",
        "answer": answer or f"answer {name}",
        "metadata": {
            "instance_id": instance or f"instance-{name}",
            "tool_type": tool_type,
            "is_hallucinated": name.endswith("bad"),
            "injector_model": "forbidden-model",
        },
    }


def _candidate(index: int, *, split: str) -> dict[str, object]:
    return {
        "component_hash": f"sha256:{index:064x}",
        "context_hash": f"sha256:{index + 1000:064x}",
        "answer_hash": f"sha256:{index + 2000:064x}",
        "context": f"context {index}",
        "question": f"question {index}",
        "answer": f"answer {index}",
        "tool_type": "read_file",
        "official_split": split,
        "context_token_count": 8 + index % 12,
        "selection_rank": f"sha256:{index + 3000:064x}",
    }


def test_scenario_verify_7533_components_join_transitive_duplicates() -> None:
    """REQ-VERIFY-7533 keeps connected duplicates in one component."""

    rows = [
        _row("a", instance="one", context="shared"),
        _row("b", instance="two", context="shared", answer="shared answer"),
        _row("c", instance="two", context="other", answer="shared answer"),
        _row("d"),
    ]
    candidates, exclusions = build_components(rows, exposure_context_hashes=set())
    assert len(candidates) == 2
    assert sorted(row["component_size"] for row in candidates) == [1, 3]
    assert exclusions == []
    assert not {
        "instance_id",
        "dataset",
        "is_hallucinated",
        "injector_model",
    } & set(candidates[0])


def test_scenario_verify_7533_components_exclude_split_overlap_and_exposure() -> None:
    """SCENARIO-VERIFY-7533-COMPONENTS excludes tainted components."""

    cross = [
        _row("a", split="train", instance="same"),
        _row("b", split="test", instance="same"),
    ]
    clean, exclusions = build_components(cross, exposure_context_hashes=set())
    assert clean == []
    assert exclusions[0]["reason"] == "cross_official_split_component"

    context_hash = build_components([_row("c")], exposure_context_hashes=set())[0][0][
        "context_hash"
    ]
    clean, exclusions = build_components([_row("c")], exposure_context_hashes={str(context_hash)})
    assert clean == []
    assert exclusions[0]["reason"] == "previous_source_exposure"


def test_scenario_verify_7533_components_are_label_blind() -> None:
    """REQ-VERIFY-7533 ignores outcomes and injector identity in selection."""

    first = _row("x")
    second = deepcopy(first)
    metadata = second["metadata"]
    assert isinstance(metadata, dict)
    metadata["is_hallucinated"] = not bool(metadata["is_hallucinated"])
    metadata["injector_model"] = "different"
    assert build_components([first], exposure_context_hashes=set()) == build_components(
        [second], exposure_context_hashes=set()
    )


def test_scenario_verify_7533_roles_are_exact_and_disjoint() -> None:
    """SCENARIO-VERIFY-7533-ROLES freezes 400 train and 80 test groups."""

    train = [_candidate(index, split="train") for index in range(400)]
    test = [_candidate(index + 10_000, split="test") for index in range(80)]
    frozen = allocate_roles(train, test)
    counts = {role: sum(row["role"] == role for row in frozen) for role in ROLE_COUNTS}
    assert counts == ROLE_COUNTS
    assert len({row["component_hash"] for row in frozen}) == 480
    assert all(
        row["official_split"] == ("test" if row["role"] == "test" else "train") for row in frozen
    )


@pytest.mark.parametrize(
    ("train_count", "test_count", "message"),
    [(399, 80, "train_capacity:399"), (400, 79, "test_capacity:79")],
)
def test_scenario_verify_7533_roles_fail_without_capacity(
    train_count: int, test_count: int, message: str
) -> None:
    """REQ-VERIFY-7533 never lowers the planned sample."""

    train = [_candidate(index, split="train") for index in range(train_count)]
    test = [_candidate(index + 10_000, split="test") for index in range(test_count)]
    with pytest.raises(ToolProtocolError, match=message):
        allocate_roles(train, test)


def test_scenario_verify_7533_prompts_use_same_role_tool_donors() -> None:
    """SCENARIO-VERIFY-7533-PROMPTS freezes six complete requests per group."""

    rows = []
    for index in range(4):
        row = _candidate(index, split="train")
        row["role"] = "fit" if index < 2 else "online"
        rows.append(row)
    manifest = build_intervention_manifest(rows, token_count=len, n_ctx=4096)
    assert len(manifest) == 4
    by_hash = {row["component_hash"]: row for row in rows}
    for entry in manifest:
        donor = by_hash[entry["donor_component_hash"]]
        target = by_hash[entry["component_hash"]]
        assert donor["role"] == target["role"]
        assert donor["tool_type"] == target["tool_type"]
        assert donor["component_hash"] != target["component_hash"]
        assert len(entry["requests"]) == 6
        assert entry["condition_sources"]["absent"] == EMPTY_EVIDENCE_MARKER
        assert all(request["prompt_token_count"] <= 4096 for request in entry["requests"])


def test_scenario_verify_7533_prompts_reject_donor_and_length_failures() -> None:
    """REQ-VERIFY-7533 excludes singleton strata and overlength prompts."""

    only = _candidate(1, split="train")
    only["role"] = "fit"
    with pytest.raises(ToolProtocolError, match="donor_missing"):
        build_intervention_manifest([only], token_count=len, n_ctx=4096)

    peer = _candidate(2, split="train")
    peer["role"] = "fit"
    with pytest.raises(ToolProtocolError, match="prompt_overlength"):
        build_intervention_manifest([only, peer], token_count=lambda _: 4097, n_ctx=4096)


def _logit_row(condition: str, order: list[str], left: float, right: float) -> dict[str, object]:
    return {
        "condition": condition,
        "option_order": order,
        "display_logits": {" A": left, " B": right},
    }


def test_scenario_verify_7533_readout_remaps_orders_and_clips() -> None:
    """SCENARIO-VERIFY-7533-READOUT preserves semantic option meaning."""

    supported_first = ["supported", "contains_unsupported"]
    unsupported_first = list(reversed(supported_first))
    assert semantic_unsupported_probability(
        _logit_row("original", supported_first, -20.0, 20.0)
    ) == pytest.approx(1.0)
    assert semantic_unsupported_probability(
        _logit_row("original", unsupported_first, 20.0, -20.0)
    ) == pytest.approx(1.0)
    rows = []
    for condition in ("original", "absent", "mismatched"):
        rows.append(_logit_row(condition, supported_first, -20.0, 20.0))
        rows.append(_logit_row(condition, unsupported_first, 20.0, -20.0))
    assert build_feature_vector(rows) == {
        "supported_first": [12.0, 12.0, 12.0],
        "unsupported_first": [12.0, 12.0, 12.0],
    }


def test_scenario_verify_7533_readout_rejects_missing_and_invalid_cells() -> None:
    """REQ-VERIFY-7533 keeps unchanged strict readout guards."""

    order = ["supported", "contains_unsupported"]
    with pytest.raises(ToolProtocolError, match="readout_cells_invalid"):
        build_feature_vector([_logit_row("original", order, 0.0, 1.0)])
    invalid = _logit_row("original", ["wrong", "supported"], 0.0, 1.0)
    with pytest.raises(ToolProtocolError, match="readout_mapping_invalid"):
        semantic_unsupported_probability(invalid)


def test_scenario_verify_7533_annotation_binary_label_validates_spans() -> None:
    """SCENARIO-VERIFY-7533-READERS uses evaluator-only valid spans."""

    answer = "tool output"
    assert annotation_binary_label(answer, []) == 0
    assert (
        annotation_binary_label(
            answer,
            [{"start": 0, "end": 4, "label": "hallucination", "category": "tool"}],
        )
        == 1
    )
    for labels in (
        [{"start": -1, "end": 2, "label": "hallucination"}],
        [{"start": 4, "end": 4, "label": "hallucination"}],
        [{"start": 0, "end": 99, "label": "hallucination"}],
        [{"start": "0", "end": 2, "label": "hallucination"}],
    ):
        with pytest.raises(ToolProtocolError, match="annotation_span_invalid"):
            annotation_binary_label(answer, labels)


def _predictor(role: str, index: int) -> dict[str, object]:
    return {
        "component_hash": f"sha256:{index:064x}",
        "role": role,
        "tool_type": "read_file",
        "context": f"context {index}",
        "question": f"question {index}",
        "answer": f"answer {index}",
    }


def test_scenario_verify_7533_readers_seal_label_access() -> None:
    """SCENARIO-VERIFY-7533-READERS requires freeze and delay boundaries."""

    predictors = [
        _predictor("fit", 1),
        _predictor("tune", 2),
        _predictor("policy", 3),
        _predictor("test", 4),
        _predictor("online", 5),
    ]
    labels = [
        {"component_hash": row["component_hash"], "role": row["role"], "label": index % 2}
        for index, row in enumerate(predictors)
    ]
    capture = read_protocol(predictors, labels, mode="capture")
    assert len(capture["rows"]) == 5
    assert all("label" not in row for row in capture["rows"])
    fit = read_protocol(predictors, labels, mode="fit")
    assert {row["role"] for row in fit["rows"]} == {"fit", "tune"}
    assert all("label" in row for row in fit["rows"])
    freeze = "sha256:frozen"
    policy = read_protocol(
        predictors, labels, mode="policy", observed_freeze=freeze, expected_freeze=freeze
    )
    assert policy["rows"][0]["role"] == "policy"
    evaluate = read_protocol(
        predictors, labels, mode="evaluate", observed_freeze=freeze, expected_freeze=freeze
    )
    assert evaluate["rows"][0]["role"] == "test"
    online = read_protocol(predictors, labels, mode="online", online_release_count=0)
    assert "label" not in online["rows"][0]
    released = read_protocol(predictors, labels, mode="online", online_release_count=1)
    assert "label" in released["rows"][0]


def test_scenario_verify_7533_readers_reject_leakage_overlap_and_missing_labels() -> None:
    """REQ-VERIFY-7533 fails closed on metadata leakage and malformed joins."""

    predictor = _predictor("fit", 1)
    leaked = {**predictor, "instance_id": "secret"}
    label = {"component_hash": predictor["component_hash"], "role": "fit", "label": 1}
    with pytest.raises(ToolProtocolError, match="predictor_field_forbidden"):
        read_protocol([leaked], [label], mode="capture")
    with pytest.raises(ToolProtocolError, match="freeze_hash_required"):
        read_protocol([predictor], [label], mode="policy")
    with pytest.raises(ToolProtocolError, match="authorized_label_missing"):
        read_protocol([predictor], [], mode="fit")
    with pytest.raises(ToolProtocolError, match="reader_mode_invalid"):
        read_protocol([predictor], [label], mode="unknown")


def test_scenario_verify_7533_guards_and_cost_policy_are_strict() -> None:
    """SCENARIO-VERIFY-7533-GUARDS preserves nulls and rejects missing data."""

    rows = [
        {"component_hash": "a", "label": 0, "candidate": 0.25, "control": 0.25},
        {"component_hash": "b", "label": 1, "candidate": 0.75, "control": 0.75},
    ]
    result = qualify_guard_rows(rows, expected_ids={"a", "b"})
    assert result["paired_brier_delta"] == 0.0
    assert result["honest_no_headroom_annotation"] == "identical_arm_predictions"
    with pytest.raises(ToolProtocolError, match="guard_row_missing"):
        qualify_guard_rows(rows[:1], expected_ids={"a", "b"})
    assert choose_action(0.01) == "accept"
    assert choose_action(0.5) == "escalate"
    assert choose_action(0.99) == "reject"
    assert choose_action(0.04) == "escalate"
    with pytest.raises(ToolProtocolError, match="probability_invalid"):
        choose_action(float("nan"))


def test_req_verify_7533_rejects_malformed_public_rows_and_options() -> None:
    """REQ-VERIFY-7533 fails closed on malformed public-only inputs."""

    string_metadata = _row("string")
    string_metadata["metadata"] = '{"instance_id":"i","tool_type":"read_file"}'
    assert len(build_components([string_metadata], exposure_context_hashes=set())[0]) == 1
    for metadata in ("{", None):
        row = _row("bad")
        row["metadata"] = metadata
        with pytest.raises(ToolProtocolError, match="metadata_invalid"):
            build_components([row], exposure_context_hashes=set())
    incomplete = _row("incomplete")
    incomplete["question"] = None
    candidates, exclusions = build_components([incomplete], exposure_context_hashes=set())
    assert candidates == []
    assert exclusions[0]["reason"] == "complete_question_missing"
    malformed = _row("malformed")
    malformed["question"] = 3
    with pytest.raises(ToolProtocolError, match="public_row_incomplete"):
        build_components([malformed], exposure_context_hashes=set())
    wrong_dataset = _row("dataset")
    wrong_dataset["dataset"] = "other"
    with pytest.raises(ToolProtocolError, match="dataset_invalid"):
        build_components([wrong_dataset], exposure_context_hashes=set())
    with pytest.raises(ToolProtocolError, match="option_order_invalid"):
        build_tool_prompt("c", "q", "a", ["wrong", "supported"])


def test_req_verify_7533_role_allocation_rejects_split_and_overlap() -> None:
    """REQ-VERIFY-7533 rejects split drift and component reuse."""

    train = [_candidate(index, split="train") for index in range(400)]
    test = [_candidate(index + 10_000, split="test") for index in range(80)]
    wrong_train = deepcopy(train)
    wrong_train[0]["official_split"] = "test"
    with pytest.raises(ToolProtocolError, match="official_split_role_mismatch"):
        allocate_roles(wrong_train, test)
    wrong_test = deepcopy(test)
    wrong_test[0]["official_split"] = "train"
    with pytest.raises(ToolProtocolError, match="official_split_role_mismatch"):
        allocate_roles(train, wrong_test)
    duplicate_test = deepcopy(test)
    duplicate_test[0]["component_hash"] = train[0]["component_hash"]
    with pytest.raises(ToolProtocolError, match="component_role_overlap"):
        allocate_roles(train, duplicate_test)


def test_req_verify_7533_readout_and_span_defenses_cover_bad_shapes() -> None:
    """REQ-VERIFY-7533 rejects nonfinite logits, bad cells, and bad span containers."""

    order = ["supported", "contains_unsupported"]
    missing_display = _logit_row("original", order, 0.0, 1.0)
    missing_display["display_logits"] = {" A": 0.0}
    with pytest.raises(ToolProtocolError, match="readout_mapping_invalid"):
        semantic_unsupported_probability(missing_display)
    with pytest.raises(ToolProtocolError, match="readout_nonfinite"):
        semantic_unsupported_probability(_logit_row("original", order, 0.0, float("nan")))
    assert semantic_unsupported_probability(_logit_row("original", order, 1.0, 0.0)) < 0.5
    with pytest.raises(ToolProtocolError, match="readout_cells_invalid"):
        build_feature_vector([_logit_row("wrong", order, 0.0, 1.0)])
    for spans in (None, ["not-a-span"]):
        with pytest.raises(ToolProtocolError, match="annotation_span_invalid"):
            annotation_binary_label("answer", spans)


def test_req_verify_7533_reader_rejects_duplicate_labels_and_freeze_drift() -> None:
    """REQ-VERIFY-7533 rejects ambiguous evaluator joins and changed freeze bytes."""

    predictor = _predictor("policy", 1)
    label = {"component_hash": predictor["component_hash"], "role": "policy", "label": 1}
    with pytest.raises(ToolProtocolError, match="evaluator_identity_duplicate"):
        read_protocol([predictor], [label, deepcopy(label)], mode="capture")
    with pytest.raises(ToolProtocolError, match="freeze_hash_mismatch"):
        read_protocol(
            [predictor],
            [label],
            mode="policy",
            observed_freeze="sha256:one",
            expected_freeze="sha256:two",
        )


def test_req_verify_7533_guard_rejects_missing_and_invalid_values() -> None:
    """SCENARIO-VERIFY-7533-GUARDS rejects null and nonprobability operands."""

    with pytest.raises(ToolProtocolError, match="guard_value_missing"):
        qualify_guard_rows(
            [{"component_hash": "a", "label": 0, "candidate": None, "control": 0.2}],
            expected_ids={"a"},
        )
    with pytest.raises(ToolProtocolError, match="guard_value_invalid"):
        qualify_guard_rows(
            [{"component_hash": "a", "label": 0, "candidate": 2.0, "control": 0.2}],
            expected_ids={"a"},
        )


def test_scenario_verify_7533_sealed_shards_rehash_and_chunk(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7533-READERS seals and independently rehashes each shard."""

    predictors = [_predictor(role, index) for index, role in enumerate(ROLE_COUNTS, 1)]
    evaluators = [
        {"component_hash": row["component_hash"], "role": row["role"], "label": index % 2}
        for index, row in enumerate(predictors)
    ]
    large = "x" * (7 * 1024 * 1024)
    interventions = [
        {"component_hash": "one", "payload": large},
        {"component_hash": "two", "payload": large},
    ]
    manifest = seal_protocol_shards(tmp_path / "raw", predictors, evaluators, interventions)
    assert {name for name in manifest if name.startswith("interventions_")} == {
        "interventions_000",
        "interventions_001",
    }
    reduction = reload_protocol_shards(manifest)
    assert reduction["passed"] is True
    bad_hash = deepcopy(manifest)
    bad_hash["predictor"]["sha256"] = "sha256:wrong"
    with pytest.raises(ToolProtocolError, match="sidecar_hash_mismatch"):
        reload_protocol_shards(bad_hash)
    bad_rows = deepcopy(manifest)
    bad_rows["predictor"]["rows"] = 999
    with pytest.raises(ToolProtocolError, match="sidecar_row_mismatch"):
        reload_protocol_shards(bad_rows)
