"""Behavior tests for REQ-AUTO-7449 and SCENARIO-AUTO-7449-*.

The tests cover label-blind grouping, evaluator isolation, immutable release
authentication, complete-input representations, and terminal reduction.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from carnot import experiment_7449_v653_source_protocol as exp
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS


def _ragtruth_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return duplicate, sibling, and cross-role RAGTruth fixture rows."""

    sources = [
        {"source_id": "a", "task_type": "Summary", "source_info": "Alpha source."},
        {"source_id": "a2", "task_type": "Summary", "source_info": " Alpha  source. "},
        {"source_id": "b", "task_type": "Summary", "source_info": "Beta source."},
        {"source_id": "c", "task_type": "Summary", "source_info": "Gamma source."},
        {"source_id": "d", "task_type": "Summary", "source_info": "Delta source."},
        {"source_id": "x", "task_type": "Summary", "source_info": "Shared source."},
        {"source_id": "x2", "task_type": "Summary", "source_info": "shared source."},
    ]
    responses = [
        {
            "id": "a-1",
            "source_id": "a",
            "response": "Alpha response one.",
            "split": "train",
            "quality": "good",
            "labels": [],
        },
        {
            "id": "a-2",
            "source_id": "a2",
            "response": "Alpha response two.",
            "split": "train",
            "quality": "good",
            "labels": [],
        },
        {
            "id": "b-1",
            "source_id": "b",
            "response": "Beta response.",
            "split": "train",
            "quality": "good",
            "labels": [{"label_type": "Evident Conflict", "implicit_true": False}],
        },
        {
            "id": "c-1",
            "source_id": "c",
            "response": "Gamma response.",
            "split": "train",
            "quality": "incorrect_refusal",
            "labels": [],
        },
        {
            "id": "d-1",
            "source_id": "d",
            "response": "Delta response.",
            "split": "test",
            "quality": "good",
            "labels": [],
        },
        {
            "id": "x-1",
            "source_id": "x",
            "response": "Shared train response.",
            "split": "train",
            "quality": "good",
            "labels": [],
        },
        {
            "id": "x-2",
            "source_id": "x2",
            "response": "Shared test response.",
            "split": "test",
            "quality": "good",
            "labels": [],
        },
    ]
    return sources, responses


def _faith_sample(source: str, sample_id: int, labels: list[str]) -> dict[str, object]:
    """Build one public-release-shaped FaithBench sample without forbidden metadata."""

    return {
        "sample_id": sample_id,
        "source": source,
        "summary": f"Summary {sample_id}.",
        "annotations": [
            {
                "label": labels,
                "note": "must not leave evaluator projection",
                "annotator_name": "private name",
                "summary_span": "Summary",
            }
        ]
        if labels
        else [],
        "metadata": {"summarizer": "forbidden", "hhemv1": 0.5},
    }


def test_req_auto_7449_spec_precedes_implementation() -> None:
    """REQ-AUTO-7449 has all six scenarios before implementation exists."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "### REQ-AUTO-7449:" in text
    for number in range(1, 7):
        assert f"SCENARIO-AUTO-7449-{number:02d}" in text


def test_scenario_auto_7449_01_groups_roles_and_shortages_without_labels() -> None:
    """SCENARIO-AUTO-7449-01 preserves official roles and never cross-fills."""

    sources, responses = _ragtruth_fixture()
    panel = exp.select_ragtruth_panel(
        sources,
        responses,
        caps={"training": 1, "calibration_tuning": 1, "internal_test": 2},
    )
    predictors = panel["predictors"]
    evaluators = panel["evaluators"]
    counts = panel["realized_group_counts"]
    assert counts == {"training": 1, "calibration_tuning": 1, "internal_test": 1}
    assert len(predictors) == len(evaluators) == 3
    assert len({row["group_id"] for row in predictors}) == 3
    assert {row["role"] for row in predictors} == {
        "training",
        "calibration_tuning",
        "internal_test",
    }
    assert all("label" not in row and "annotations" not in row for row in predictors)
    assert any(
        row["reason"] == "excluded_cross_official_role_duplicate" for row in panel["dispositions"]
    )
    assert any(
        row["reason"] == "excluded_quality_incorrect_refusal" for row in panel["dispositions"]
    )

    permuted = deepcopy(responses)
    for row in permuted:
        row["labels"] = [] if row["labels"] else [{"label_type": "mutated"}]
    changed = exp.select_ragtruth_panel(
        sources,
        permuted,
        caps={"training": 1, "calibration_tuning": 1, "internal_test": 2},
    )
    assert changed["predictors"] == predictors
    assert [row["response_id"] for row in changed["evaluators"]] == [
        row["response_id"] for row in evaluators
    ]

    capped = exp.select_ragtruth_panel(
        sources,
        responses,
        caps={"training": 0, "calibration_tuning": 0, "internal_test": 0},
    )
    assert any(row["reason"] == "excluded_role_cap" for row in capped["dispositions"])


def test_scenario_auto_7449_01_rejects_malformed_release_rows() -> None:
    """SCENARIO-AUTO-7449-01 rejects ambiguous identities and malformed roles."""

    sources, responses = _ragtruth_fixture()
    with pytest.raises(exp.CorpusInvalid, match="source identity"):
        exp.select_ragtruth_panel([sources[0], sources[0]], responses)
    with pytest.raises(exp.CorpusInvalid, match="response identity"):
        exp.select_ragtruth_panel(sources, [responses[0], responses[0]])
    missing_source = [{**responses[0], "source_id": "missing"}]
    with pytest.raises(exp.CorpusInvalid, match="response source missing"):
        exp.select_ragtruth_panel(sources, missing_source)
    bad_split = [{**responses[0], "split": "validation"}]
    with pytest.raises(exp.CorpusInvalid, match="official split"):
        exp.select_ragtruth_panel(sources, bad_split)
    bad_text = [{**responses[0], "response": ""}]
    with pytest.raises(exp.CorpusInvalid, match="response text"):
        exp.select_ragtruth_panel(sources, bad_text)
    with pytest.raises(exp.CorpusInvalid, match="annotation list"):
        exp._annotation_types(None)
    with pytest.raises(exp.CorpusInvalid, match="annotation object"):
        exp._annotation_types(["not-an-object"])


def test_scenario_auto_7449_02_faithbench_aggregation_and_isolation() -> None:
    """SCENARIO-AUTO-7449-02 retains ambiguity but strips forbidden fields."""

    samples = [
        _faith_sample("One source.", 1, ["Benign", "Unwanted.Intrinsic"]),
        _faith_sample("One source.", 2, ["Benign", "Questionable"]),
        _faith_sample("Two source.", 3, ["Questionable"]),
        _faith_sample("Local duplicate.", 4, []),
    ]
    blocked = {exp.normalized_source_hash("Local duplicate.")}
    panel = exp.select_faithbench_panel(samples, local_source_hashes=blocked, cap=10)
    assert panel["realized_group_counts"] == {"external": 2}
    assert len(panel["predictors"]) == len(panel["evaluators"]) == 2
    assert any(row["ambiguous"] is True for row in panel["evaluators"])
    assert {row["label"] for row in panel["evaluators"]} <= {0, 1}
    forbidden = {"note", "metadata", "summarizer", "hhemv1", "annotator_name"}
    serialized_predictors = json.dumps(panel["predictors"], sort_keys=True)
    serialized_evaluators = json.dumps(panel["evaluators"], sort_keys=True)
    assert all(field not in serialized_predictors for field in forbidden)
    assert all(field not in serialized_evaluators for field in forbidden)
    assert any(row["reason"] == "excluded_local_corpus_duplicate" for row in panel["dispositions"])
    assert exp.official_faithbench_label(samples[0]) == (0, True)
    assert exp.official_faithbench_label(_faith_sample("Clean.", 5, [])) == (1, False)
    with pytest.raises(exp.CorpusInvalid, match="unknown FaithBench label"):
        exp.official_faithbench_label(_faith_sample("Bad.", 6, ["Unknown"]))

    capped = exp.select_faithbench_panel(samples, local_source_hashes=set(), cap=1)
    assert any(row["reason"] == "excluded_external_cap" for row in capped["dispositions"])
    with pytest.raises(exp.CorpusInvalid, match="annotations list"):
        exp.official_faithbench_label({"annotations": None})
    with pytest.raises(exp.CorpusInvalid, match="annotation shape"):
        exp.official_faithbench_label({"annotations": [{}]})
    with pytest.raises(exp.CorpusInvalid, match="text invalid"):
        exp.select_faithbench_panel(
            [{"sample_id": 7, "source": "", "summary": "summary"}],
            local_source_hashes=set(),
        )


def test_scenario_auto_7449_03_label_permutation_leaves_inputs_identical() -> None:
    """SCENARIO-AUTO-7449-03 proves evaluator labels cannot alter inputs."""

    sources, responses = _ragtruth_fixture()
    panel = exp.select_ragtruth_panel(
        sources,
        responses,
        caps={"training": 1, "calibration_tuning": 1, "internal_test": 1},
    )
    receipt = exp.label_permutation_invariance(panel["predictors"], panel["evaluators"])
    assert receipt["passed"] is True
    assert receipt["prompt_bytes_unchanged"] is True
    assert receipt["feature_bytes_unchanged"] is True
    assert receipt["eligibility_unchanged"] is True
    assert receipt["selection_unchanged"] is True
    with pytest.raises(exp.CorpusInvalid, match="keys differ"):
        exp.label_permutation_invariance(panel["predictors"], panel["evaluators"][:-1])


def test_scenario_auto_7449_04_representation_views_are_complete_and_neutral() -> None:
    """SCENARIO-AUTO-7449-04 freezes complete text without a yes/no request."""

    predictor = {
        "row_key": "row-1",
        "group_id": "group-1",
        "corpus": "ragtruth",
        "role": "training",
        "source_text": "Complete source evidence.",
        "response_text": "Complete response.",
        "source_features": {name: 0.0 for name in exp.SOURCE_FEATURE_NAMES},
    }
    response = exp.representation_bytes(predictor, "response")
    paired = exp.representation_bytes(predictor, "source_response")
    assert response == b"Complete response."
    assert paired == b"SOURCE\nComplete source evidence.\n\nRESPONSE\nComplete response."
    assert b"yes" not in paired.lower() and b"no" not in paired.lower()
    with pytest.raises(ValueError, match="representation view"):
        exp.representation_bytes(predictor, "unknown")

    eligibility = exp.planned_eligibility_rows([predictor])
    assert len(eligibility) == 2
    assert all(row["token_count"] is None for row in eligibility)
    assert all(row["status"] == "unstarted" for row in eligibility)
    assert all(row["complete_input_required"] is True for row in eligibility)


def test_scenario_auto_7449_05_comparison_plan_has_matched_controls() -> None:
    """SCENARIO-AUTO-7449-05 freezes capacity, controls, seeds, and correction."""

    plan = exp.comparison_plan()
    assert plan["pooling"] == "final_layer_last_token"
    assert plan["projection"]["dimensions"] == 32
    assert len(plan["fit_seeds"]) == 5
    assert set(plan["arms"]) == {
        "source_response_gibbs",
        "source_response_logistic",
        "response_only_gibbs",
        "source_shuffled_gibbs",
        "old_lexical_features",
        "training_prevalence",
    }
    assert plan["primary_metrics"] == ["external_brier", "external_log_loss"]
    assert plan["bootstrap"] == {
        "unit": "source_group",
        "draws": 10_000,
        "seed": exp.BOOTSTRAP_SEED,
        "multiplicity": "holm",
    }
    assert plan["minimum_groups"] == {
        "training": 150,
        "calibration_tuning": 40,
        "internal_test": 40,
        "external": 60,
    }


def test_release_asset_authentication_rejects_drift(tmp_path: Path) -> None:
    """REQ-AUTO-7449 authenticates bytes with the pinned Git blob identity."""

    payload = b"public release bytes\n"
    git_hash = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    path = tmp_path / "LICENSE"
    path.write_bytes(payload)
    receipt = exp.authenticate_release_file(
        path,
        relative="LICENSE",
        expected_size=len(payload),
        expected_git_sha1=git_hash,
    )
    assert receipt["size_bytes"] == len(payload)
    assert receipt["sha256"].startswith("sha256:")
    assert receipt["git_blob_sha1"] == git_hash
    path.write_bytes(payload + b"changed")
    with pytest.raises(exp.SourceBlocked, match="size mismatch"):
        exp.authenticate_release_file(
            path,
            relative="LICENSE",
            expected_size=len(payload),
            expected_git_sha1=git_hash,
        )
    with pytest.raises(exp.SourceBlocked, match="file missing"):
        exp.authenticate_release_file(
            tmp_path / "absent",
            relative="LICENSE",
            expected_size=len(payload),
            expected_git_sha1=git_hash,
        )


def test_protocol_shards_reload_and_detect_tampering(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7449-02/03 bind separate predictor and evaluator bytes."""

    sources, responses = _ragtruth_fixture()
    panel = exp.select_ragtruth_panel(
        sources,
        responses,
        caps={"training": 1, "calibration_tuning": 1, "internal_test": 1},
    )
    manifest = exp.seal_protocol_shards(
        tmp_path,
        panel["predictors"],
        panel["evaluators"],
        panel["dispositions"],
    )
    loaded = exp.reload_protocol_shards(tmp_path)
    assert loaded["manifest"] == manifest
    assert loaded["predictors"] == panel["predictors"]
    assert loaded["evaluators"] == panel["evaluators"]
    predictor_path = tmp_path / manifest["shards"][0]["path"]
    predictor_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(exp.CorpusInvalid, match="shard hash mismatch"):
        exp.reload_protocol_shards(tmp_path)

    with pytest.raises(exp.CorpusInvalid, match="predictor field"):
        exp.seal_protocol_shards(tmp_path / "bad-predictor", [{}], [], [])
    with pytest.raises(exp.CorpusInvalid, match="evaluator field"):
        exp.seal_protocol_shards(tmp_path / "bad-evaluator", panel["predictors"], [{}], [])


def test_scenario_auto_7449_06_terminal_reduction_separates_readiness() -> None:
    """SCENARIO-AUTO-7449-06 makes readiness protocol-only and promotion zero."""

    counts = {
        "training": 180,
        "calibration_tuning": 60,
        "internal_test": 60,
        "external": 100,
    }
    gates = exp.protocol_gates(
        counts,
        release_authenticated=True,
        evaluator_isolated=True,
        invariance_passed=True,
        validation_passed=True,
    )
    reduced = exp.reduce_protocol(gates, flagged_adversarial=False)
    assert reduced == {
        "source_protocol_ready_score": 1,
        "confirmatory_minima_met": True,
        "status": "complete",
        "honest_verdict": "complete_null_source_protocol_ready_no_predictive_measurement",
        "verdict_class": "null",
        "promotion_score": 0,
    }
    short = exp.protocol_gates(
        {**counts, "external": 59},
        release_authenticated=True,
        evaluator_isolated=True,
        invariance_passed=True,
        validation_passed=True,
    )
    short_reduced = exp.reduce_protocol(short, flagged_adversarial=False)
    assert short_reduced["source_protocol_ready_score"] == 1
    assert short_reduced["confirmatory_minima_met"] is False
    assert short_reduced["verdict_class"] == "null"

    failed = deepcopy(gates)
    next(row for row in failed if row["check"] == "release_authenticated")["passed"] = False
    invalid = exp.reduce_protocol(failed, flagged_adversarial=False)
    assert invalid["source_protocol_ready_score"] == 0
    assert invalid["verdict_class"] == "disqualified"
    flagged = exp.reduce_protocol(gates, flagged_adversarial=True)
    assert flagged["verdict_class"] == "disqualified"


def test_model_contract_and_artifact_checksum_are_cold_replayable() -> None:
    """REQ-AUTO-7449 keeps current inference at zero and binds stable evidence."""

    assert exp.MODEL_SPECS == []
    assert exp.INVOCATION_COUNTS == ZERO_INVOCATION_COUNTS
    artifact = {
        "schema": exp.SCHEMA,
        "experiment_id": exp.EXPERIMENT_ID,
        "milestone": exp.MILESTONE,
        "run_date": exp.RUN_DATE,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "source_protocol_ready_score": 1,
        "promotion_score": 0,
        "flagged_adversarial": False,
        "reproducibility_checksum": None,
    }
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert exp.artifact_checksum(artifact) == artifact["reproducibility_checksum"]
    artifact["promotion_score"] = 1
    assert exp.artifact_checksum(artifact) != artifact["reproducibility_checksum"]
