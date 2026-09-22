"""Tests for the V658 source-removal and mismatch protocol.

Spec refs: REQ-VERIFY-7517 and SCENARIO-VERIFY-7517-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7517_v658_source_protocol as protocol


def _source(index: int, family: str = "QA") -> dict[str, object]:
    return {
        "source_id": f"source-{index}",
        "source_info": f"Evidence passage {index} with grounded facts.",
        "task_type": family,
    }


def _response(index: int, *, label: bool = False) -> dict[str, object]:
    return {
        "id": f"response-{index}",
        "source_id": f"source-{index}",
        "response": f"Answer {index}.",
        "split": "train",
        "quality": "good",
        "labels": [{"start": 0, "end": 3}] if label else [],
        "model": "evaluator-only-generator",
    }


def _candidate(index: int, family: str = "QA") -> dict[str, object]:
    source = f"Evidence passage {index} with grounded facts."
    return {
        "group_id": f"group-{index}",
        "source_family": family,
        "source_text": source,
        "response_text": f"Answer {index}.",
        "source_hash": protocol.normalized_text_hash(source),
        "source_sha256": protocol.sha256_text(source),
        "response_hash": protocol.sha256_text(f"Answer {index}."),
        "source_token_count": 8 + index % 5,
    }


def _roster() -> list[dict[str, object]]:
    candidates = [
        _candidate(index, ("QA", "Summary", "Data2txt")[index % 3]) for index in range(480)
    ]
    return protocol.freeze_roles(candidates)


def test_selection_is_label_blind_fresh_and_official_train_only() -> None:
    """REQ-VERIFY-7517 / SCENARIO-VERIFY-7517-FRESHNESS."""

    sources = [_source(index, "QA" if index % 2 else "Summary") for index in range(6)]
    responses = [_response(index, label=bool(index % 2)) for index in range(6)]
    responses.extend(
        [
            {**_response(2), "id": "alternate", "response": "Alternate."},
            {**_response(4), "id": "test-row", "split": "test"},
            {**_response(5), "id": "bad-row", "quality": "truncated"},
        ]
    )
    exposed_hash = protocol.normalized_text_hash(str(sources[0]["source_info"]))
    exposure = protocol.ExposureUnion(
        source_ids=frozenset({"source-1"}),
        normalized_source_hashes=frozenset({exposed_hash}),
        exact_source_hashes=frozenset(),
        response_hashes=frozenset(),
    )
    selected = protocol.select_fresh_groups(
        sources,
        responses,
        exposure,
        requested=3,
        token_count=lambda text: len(text.split()),
    )
    mutated = deepcopy(responses)
    for row in mutated:
        row["labels"] = ["mutated-private-label"]
        row["model"] = "mutated-private-generator"
    selected_after_mutation = protocol.select_fresh_groups(
        sources,
        mutated,
        exposure,
        requested=3,
        token_count=lambda text: len(text.split()),
    )
    assert selected == selected_after_mutation
    assert len(selected) == 3
    assert {row["source_hash"] for row in selected}.isdisjoint(
        {
            protocol.normalized_text_hash(str(sources[0]["source_info"])),
            protocol.normalized_text_hash(str(sources[1]["source_info"])),
        }
    )
    assert all(set(row) <= set(protocol.PREDICTOR_ALLOWLIST) for row in selected)


def test_selection_blocks_instead_of_reusing_exposed_or_overlength_groups() -> None:
    """REQ-VERIFY-7517 / SCENARIO-VERIFY-7517-FRESHNESS."""

    sources = [_source(index) for index in range(3)]
    responses = [_response(index) for index in range(3)]
    exposure = protocol.ExposureUnion(
        source_ids=frozenset({"source-0", "source-1"}),
        normalized_source_hashes=frozenset(),
        exact_source_hashes=frozenset(),
        response_hashes=frozenset(),
    )
    selected = protocol.select_fresh_groups(
        sources,
        responses,
        exposure,
        requested=2,
        token_count=lambda text: 3_000 if "passage 2" in text else 10,
    )
    assert selected == []


def test_roles_are_exact_deterministic_and_family_aware() -> None:
    """REQ-VERIFY-7517 / SCENARIO-VERIFY-7517-ROLES."""

    first = _roster()
    second = protocol.freeze_roles(
        list(
            reversed(
                [
                    _candidate(index, ("QA", "Summary", "Data2txt")[index % 3])
                    for index in range(480)
                ]
            )
        )
    )
    assert first == second
    assert protocol.role_counts(first) == protocol.ROLE_COUNTS
    for role in protocol.ROLE_COUNTS:
        assert {row["source_family"] for row in first if row["role"] == role} == {
            "QA",
            "Summary",
            "Data2txt",
        }
    duplicate = deepcopy(first)
    duplicate[1]["source_hash"] = duplicate[0]["source_hash"]
    with pytest.raises(protocol.SourceProtocolError, match="normalized_source_duplicate"):
        protocol.freeze_roles(duplicate)


def test_interventions_use_same_role_family_nearest_bin_and_six_prompts() -> None:
    """REQ-VERIFY-7517 / SCENARIO-VERIFY-7517-INTERVENTIONS."""

    rows = [
        {**_candidate(0), "role": "training", "source_token_count": 10},
        {**_candidate(1), "role": "training", "source_token_count": 12},
        {**_candidate(2), "role": "training", "source_token_count": 90},
        {**_candidate(3, "Summary"), "role": "training", "source_token_count": 11},
        {**_candidate(6, "Summary"), "role": "training", "source_token_count": 14},
        {**_candidate(4), "role": "test", "source_token_count": 11},
        {**_candidate(5), "role": "test", "source_token_count": 13},
    ]
    manifest = protocol.build_intervention_manifest(rows, token_count=lambda text: len(text))
    by_group = {row["group_id"]: row for row in manifest}
    assert by_group["group-0"]["donor_group_id"] == "group-1"
    for row in manifest:
        donor = by_group[row["donor_group_id"]]
        assert donor["role"] == row["role"]
        assert donor["source_family"] == row["source_family"]
        assert donor["source_hash"] != row["source_hash"]
        assert len(row["requests"]) == 6
        assert {request["condition"] for request in row["requests"]} == {
            "original",
            "absent",
            "mismatched",
        }
        assert all(request["prompt_token_count"] <= 2_048 for request in row["requests"])
        absent = [request for request in row["requests"] if request["condition"] == "absent"]
        assert all(protocol.EMPTY_EVIDENCE_MARKER in request["prompt"] for request in absent)


def test_intervention_rejects_missing_donor_duplicate_and_overlength() -> None:
    """REQ-VERIFY-7517 / SCENARIO-VERIFY-7517-INTERVENTIONS."""

    one = [{**_candidate(0), "role": "training"}]
    with pytest.raises(protocol.SourceProtocolError, match="donor_missing"):
        protocol.build_intervention_manifest(one, token_count=lambda text: len(text))
    duplicate = [
        {**_candidate(0), "role": "training"},
        {**_candidate(1), "role": "training"},
    ]
    duplicate[1]["source_hash"] = duplicate[0]["source_hash"]
    with pytest.raises(protocol.SourceProtocolError, match="donor_missing"):
        protocol.build_intervention_manifest(duplicate, token_count=lambda text: len(text))
    rows = [
        {**_candidate(0), "role": "training"},
        {**_candidate(1), "role": "training"},
    ]
    with pytest.raises(protocol.SourceProtocolError, match="prompt_overlength"):
        protocol.build_intervention_manifest(rows, token_count=lambda text: 2_049)


def _readout(condition: str, order: tuple[str, str], logits: tuple[float, float]) -> dict:
    return {
        "condition": condition,
        "option_order": list(order),
        "display_logits": {" A": logits[0], " B": logits[1]},
    }


def test_option_orders_map_to_semantic_probability_and_clipped_views() -> None:
    """REQ-VERIFY-7517 / SCENARIO-VERIFY-7517-READOUT."""

    normal = ("supported", "contains_unsupported")
    reversed_order = tuple(reversed(normal))
    assert protocol.semantic_unsupported_probability(_readout("original", normal, (0.0, 2.0))) == (
        pytest.approx(0.8807970779778823)
    )
    assert protocol.semantic_unsupported_probability(
        _readout("original", reversed_order, (2.0, 0.0))
    ) == pytest.approx(0.8807970779778823)
    rows = [
        _readout("original", normal, (-100.0, 100.0)),
        _readout("absent", normal, (0.0, 0.0)),
        _readout("mismatched", normal, (2.0, 0.0)),
        _readout("original", reversed_order, (100.0, -100.0)),
        _readout("absent", reversed_order, (0.0, 0.0)),
        _readout("mismatched", reversed_order, (0.0, 2.0)),
    ]
    views = protocol.build_three_vector_views(rows)
    assert views["supported_first"] == pytest.approx([12.0, 0.0, -2.0])
    assert views["unsupported_first"] == pytest.approx([12.0, 0.0, -2.0])
    assert set(views) == {"supported_first", "unsupported_first"}
    with pytest.raises(protocol.SourceProtocolError, match="readout_cells_invalid"):
        protocol.build_three_vector_views(rows[:-1])


def test_reader_access_is_role_separated_and_freeze_bound() -> None:
    """REQ-VERIFY-7517 / SCENARIO-VERIFY-7517-ACCESS."""

    roles = list(protocol.ROLE_COUNTS)
    predictors = [
        {
            **_candidate(index),
            "role": role,
            "arrival_order": index if role == "online" else None,
        }
        for index, role in enumerate(roles)
    ]
    evaluators = [
        {"group_id": row["group_id"], "role": row["role"], "label": index % 2}
        for index, row in enumerate(predictors)
    ]
    fit = protocol.read_protocol(predictors, evaluators, mode="fit")
    assert {row["role"] for row in fit["rows"]} == {"training", "calibration_tuning"}
    assert all("label" in row for row in fit["rows"])
    predict = protocol.read_protocol(predictors, evaluators, mode="predict")
    assert len(predict["rows"]) == len(predictors)
    assert all("label" not in row for row in predict["rows"])
    with pytest.raises(protocol.SourceProtocolError, match="freeze_hash_required"):
        protocol.read_protocol(predictors, evaluators, mode="policy")
    policy = protocol.read_protocol(
        predictors,
        evaluators,
        mode="policy",
        observed_freeze="sha256:fit",
        expected_freeze="sha256:fit",
    )
    assert [row["role"] for row in policy["rows"]] == ["calibration_policy"]
    evaluate = protocol.read_protocol(
        predictors,
        evaluators,
        mode="evaluate",
        observed_freeze="sha256:predictions",
        expected_freeze="sha256:predictions",
    )
    assert [row["role"] for row in evaluate["rows"]] == ["test"]
    online = protocol.read_protocol(predictors, evaluators, mode="online", online_release_count=1)
    assert [row["role"] for row in online["rows"]] == ["online"]
    assert online["rows"][0]["label"] in {0, 1}


def test_reader_mutation_and_source_identity_leak_guards() -> None:
    """REQ-VERIFY-7517 / SCENARIO-VERIFY-7517-ACCESS/READOUT."""

    predictor = {**_candidate(0), "role": "training"}
    evaluator = {"group_id": "group-0", "role": "training", "label": 1}
    before = protocol.read_protocol([predictor], [evaluator], mode="predict")
    evaluator["label"] = 0
    assert protocol.read_protocol([predictor], [evaluator], mode="predict") == before
    leaked = {**predictor, "source_id": "private-source-id"}
    with pytest.raises(protocol.SourceProtocolError, match="predictor_field_forbidden"):
        protocol.read_protocol([leaked], [evaluator], mode="predict")
    with pytest.raises(protocol.SourceProtocolError, match="freeze_hash_mismatch"):
        protocol.read_protocol(
            [predictor],
            [evaluator],
            mode="evaluate",
            observed_freeze="sha256:changed",
            expected_freeze="sha256:predictions",
        )


def test_strict_guards_keep_absolute_metrics_and_no_headroom_annotations() -> None:
    """REQ-VERIFY-7517 / SCENARIO-VERIFY-7517-GUARDS."""

    identical = [
        {"group_id": "a", "label": 0, "candidate": 0.2, "control": 0.2},
        {"group_id": "b", "label": 1, "candidate": 0.8, "control": 0.8},
    ]
    reduced = protocol.qualify_guard_rows(identical, expected_ids={"a", "b"})
    assert reduced["absolute_arm_metrics"]["candidate_brier"] == pytest.approx(0.04)
    assert reduced["absolute_arm_metrics"]["control_brier"] == pytest.approx(0.04)
    assert reduced["headroom_present"] is False
    assert reduced["honest_no_headroom_annotation"] == "identical_arm_predictions"
    mixed = deepcopy(identical)
    mixed[1]["candidate"] = 0.9
    assert protocol.qualify_guard_rows(mixed, expected_ids={"a", "b"})["headroom_present"] is True
    all_null = deepcopy(identical)
    for row in all_null:
        row["candidate"] = None
        row["control"] = None
    with pytest.raises(protocol.SourceProtocolError, match="guard_value_missing"):
        protocol.qualify_guard_rows(all_null, expected_ids={"a", "b"})
    with pytest.raises(protocol.SourceProtocolError, match="guard_row_missing"):
        protocol.qualify_guard_rows(identical[:1], expected_ids={"a", "b"})


def test_blocked_artifact_is_schema_complete_and_mutation_detecting(tmp_path: Path) -> None:
    """REQ-VERIFY-7517 / SCENARIO-VERIFY-7517-FRESHNESS/NO-MODEL."""

    inventory = tmp_path / "exposure_inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "carnot.exp7517.exposure.v1",
                "candidate_official_training_groups": 2_515,
                "exposed_candidate_groups": 2_515,
                "fresh_eligible_groups": 0,
                "files": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = protocol.build_blocked_artifact_for_test(inventory)
    assert artifact["honest_verdict"] == "complete_blocked_fresh_source_inventory"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["source_protocol_ready_score"] == 0
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["sample_size_budget"] == {
        "planned": 480,
        "attempted": 0,
        "completed": 0,
        "excluded": 2_515,
        "failed": 0,
        "censored": 0,
        "unstarted": 480,
    }
    assert protocol.validate_artifact(artifact, root=tmp_path, require_terminal=False) == []
    changed = deepcopy(artifact)
    changed["source_protocol_ready_score"] = 1
    assert "blocked_readiness_mismatch" in protocol.validate_artifact(
        changed, root=tmp_path, require_terminal=False
    )
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in protocol.validate_artifact(
        changed, root=tmp_path, require_terminal=False
    )


def test_exposure_sidecar_independent_reduction(tmp_path: Path) -> None:
    """REQ-VERIFY-7517 / SCENARIO-VERIFY-7517-E2E."""

    sidecar = tmp_path / "exposure.json"
    protocol.write_exposure_sidecar(
        sidecar,
        files=[
            {
                "path": "results/raw/prior/predictors.jsonl",
                "sha256": "sha256:prior",
                "source_id_count": 2,
                "normalized_source_hash_count": 2,
            }
        ],
        candidate_count=3,
        exposed_count=2,
        fresh_count=1,
    )
    reduction = protocol.reduce_exposure_sidecar(sidecar)
    assert reduction["fresh_eligible_groups"] == 1
    assert reduction["accounting_passed"] is True
    mutated = json.loads(sidecar.read_text(encoding="utf-8"))
    mutated["fresh_eligible_groups"] = 2
    sidecar.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(protocol.SourceProtocolError, match="exposure_accounting_mismatch"):
        protocol.reduce_exposure_sidecar(sidecar)


def test_selection_readout_reader_and_guard_error_matrix(tmp_path: Path) -> None:
    """REQ-VERIFY-7517 keeps malformed public and native evidence fail-closed."""

    empty = protocol.ExposureUnion(frozenset(), frozenset(), frozenset(), frozenset())
    assert (
        protocol._public_candidate({"source_info": None}, {"response": "x"}, lambda text: len(text))
        is None
    )
    sources = [
        {"source_id": "source-0", "source_info": {"fact": "zero"}, "task_type": "QA"},
        {"source_id": "source-1", "source_info": ["one"], "task_type": "QA"},
        {"source_id": "source-2", "source_info": None, "task_type": "QA"},
        {"source_id": "source-3", "source_info": "three", "task_type": "QA"},
    ]
    responses = [
        _response(0),
        {**_response(1), "response": 7},
        {**_response(2), "byte_complete": False},
        {**_response(3), "byte_complete": False},
    ]
    selected = protocol.select_fresh_groups(
        sources, responses, empty, requested=None, token_count=lambda text: len(text.split())
    )
    assert len(selected) == 1
    blocked_response = protocol.ExposureUnion(
        frozenset(),
        frozenset(),
        frozenset(),
        frozenset({selected[0]["response_hash"]}),
    )
    assert (
        protocol.select_fresh_groups(
            sources,
            responses,
            blocked_response,
            requested=1,
            token_count=lambda text: len(text.split()),
        )
        == []
    )
    with pytest.raises(protocol.SourceProtocolError, match="fresh_inventory_count"):
        protocol.freeze_roles([_candidate(0)])

    normal = ("supported", "contains_unsupported")
    assert protocol.semantic_unsupported_probability(
        _readout("original", normal, (2.0, 0.0))
    ) == pytest.approx(0.11920292202211755)
    for malformed, message in [
        ({"option_order": ["bad", "ids"], "display_logits": {" A": 0, " B": 1}}, "mapping"),
        ({"option_order": list(normal), "display_logits": {" A": 0}}, "mapping"),
        (
            {"option_order": list(normal), "display_logits": {" A": 0, " B": float("nan")}},
            "nonfinite",
        ),
    ]:
        with pytest.raises(protocol.SourceProtocolError, match=f"readout_{message}"):
            protocol.semantic_unsupported_probability(malformed)
    complete_rows = [
        _readout(condition, order, (0.0, 1.0))
        for order in protocol.OPTION_ORDERS
        for condition in ("original", "absent", "mismatched")
    ]
    with pytest.raises(protocol.SourceProtocolError, match="readout_cells_invalid"):
        protocol.build_three_vector_views([*complete_rows, deepcopy(complete_rows[0])])

    predictor = {**_candidate(0), "role": "training"}
    evaluator = {"group_id": "group-0", "role": "training", "label": 1}
    with pytest.raises(protocol.SourceProtocolError, match="evaluator_identity_duplicate"):
        protocol.read_protocol([predictor], [evaluator, evaluator], mode="fit")
    with pytest.raises(protocol.SourceProtocolError, match="reader_mode_invalid"):
        protocol.read_protocol([predictor], [evaluator], mode="invalid")
    with pytest.raises(protocol.SourceProtocolError, match="authorized_label_missing"):
        protocol.read_protocol([predictor], [], mode="fit")
    invalid_guard = [{"group_id": "x", "label": 1, "candidate": 2.0, "control": 0.5}]
    with pytest.raises(protocol.SourceProtocolError, match="guard_value_invalid"):
        protocol.qualify_guard_rows(invalid_guard, expected_ids={"x"})

    array_sidecar = tmp_path / "array.json"
    array_sidecar.write_text("[]", encoding="utf-8")
    with pytest.raises(protocol.SourceProtocolError, match="exposure_object_invalid"):
        protocol.reduce_exposure_sidecar(array_sidecar)
    invalid_count = tmp_path / "invalid-count.json"
    payload: dict[str, object] = {
        "candidate_official_training_groups": True,
        "exposed_candidate_groups": 0,
        "fresh_eligible_groups": 1,
        "files": [],
        "schema": "carnot.exp7517.exposure.v1",
    }
    payload["inventory_sha256"] = protocol.canonical_hash(payload)
    invalid_count.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(protocol.SourceProtocolError, match="exposure_count_invalid"):
        protocol.reduce_exposure_sidecar(invalid_count)
    valid = tmp_path / "valid.json"
    protocol.write_exposure_sidecar(
        valid, files=[], candidate_count=1, exposed_count=0, fresh_count=1
    )
    changed = json.loads(valid.read_text(encoding="utf-8"))
    changed["files"] = [{"changed": True}]
    valid.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(protocol.SourceProtocolError, match="exposure_checksum_mismatch"):
        protocol.reduce_exposure_sidecar(valid)


def test_blocked_classification_and_artifact_validation_matrix(tmp_path: Path) -> None:
    """REQ-VERIFY-7517 retains exact block causes and cold-reader mutations."""

    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "candidate_official_training_groups": 2_515,
                "exposed_candidate_groups": 2_515,
                "fresh_eligible_groups": 0,
            }
        ),
        encoding="utf-8",
    )
    receipt = {
        "path": inventory.name,
        "sha256": protocol.sha256_file(inventory),
        "bytes": inventory.stat().st_size,
    }
    failed_precondition = [
        {
            "check": "resource_exists",
            "path": "missing.json",
            "field": "exists",
            "expected": True,
            "observed": False,
            "passed": False,
        }
    ]
    blocked = protocol._build_blocked_artifact(
        inventory,
        root=tmp_path,
        preconditions=failed_precondition,
        source_hashes=[receipt],
        validation_receipts=[],
        validation_passed=True,
        duration_s=0.1,
        phase_spans=[],
    )
    assert blocked["honest_verdict"] == "complete_blocked_prerequisite_missing_or_mismatched"
    assert blocked["gate_check_summary"]["first_failure"] == {
        "check": "precondition:resource_exists",
        "upstream": "missing.json",
        "field": "exists",
        "path": "missing.json",
        "expected": True,
        "observed": False,
        "op": "eq",
    }
    passing_precondition = deepcopy(failed_precondition)
    passing_precondition[0].update({"observed": True, "passed": True})
    incomplete = protocol._build_blocked_artifact(
        inventory,
        root=tmp_path,
        preconditions=passing_precondition,
        source_hashes=[receipt],
        validation_receipts=[],
        validation_passed=True,
        duration_s=0.1,
        phase_spans=[],
        exposure_complete=False,
    )
    assert incomplete["honest_verdict"] == "complete_blocked_exposure_audit_incomplete"
    assert protocol._path_label(inventory, tmp_path / "different-root") == str(inventory.resolve())

    artifact = protocol.build_blocked_artifact_for_test(inventory)
    mutations = [
        (lambda value: value.pop("schema"), "required_fields_missing"),
        (lambda value: value.__setitem__("schema", "wrong"), "terminal_identity_mismatch"),
        (lambda value: value.__setitem__("MODEL_SPECS", [{}]), "model_specs_nonempty"),
        (
            lambda value: value.__setitem__("model_invoked", True),
            "current_model_accounting_mismatch",
        ),
        (
            lambda value: value.__setitem__("inference_substrate", "wrong"),
            "inference_substrate_mismatch",
        ),
        (lambda value: value.__setitem__("honest_verdict", "blocked"), "blocked_verdict_mismatch"),
        (lambda value: value.__setitem__("rows", [{}]), "blocked_rows_present"),
        (
            lambda value: value["exposure_audit"].__setitem__("inventory_sha256", "wrong"),
            "exposure_inventory_hash_mismatch",
        ),
        (
            lambda value: value["exposure_audit"].__setitem__("official_test_rows_opened", 1),
            "official_test_touched",
        ),
        (lambda value: value.__setitem__("exposure_audit", None), "exposure_audit_invalid"),
        (
            lambda value: value.__setitem__("gate_check_summary", {"first_failure": None}),
            "gate_summary_mismatch",
        ),
        (
            lambda value: value["field_principles"].pop("schema"),
            "field_principles_incomplete",
        ),
    ]
    for mutate, expected in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert expected in protocol.validate_artifact(
            changed, root=tmp_path, require_terminal=False
        )
    assert "terminal_validation_incomplete" in protocol.validate_artifact(
        artifact, root=tmp_path, require_terminal=True
    )
