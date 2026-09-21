"""Tests for the lossless V656 response-window protocol.

Spec refs: REQ-VERIFY-7491, SCENARIO-VERIFY-7491-ROLES,
SCENARIO-VERIFY-7491-WINDOWS, SCENARIO-VERIFY-7491-PROMPTS,
SCENARIO-VERIFY-7491-BUDGET, SCENARIO-VERIFY-7491-NO-MODEL, and
SCENARIO-VERIFY-7491-E2E.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7491_v656_window_protocol as protocol
from carnot.experiment_7462_v654_option_protocol import OPTION_IDS, build_option_prompt


@pytest.mark.parametrize(
    "response",
    [
        "Not supported. It remains qualified.",
        "She denied it! He referred to that pronoun?",
        "On 2026-09-21, the value stayed 4.2.",
        "Café déjà vu。次の文です！",
        "Repeat. Repeat. Repeat. Repeat.",
        "One sentence without terminal punctuation",
        "",
    ],
)
def test_lossless_windows_partition_exact_utf8_bytes(response: str) -> None:
    """REQ-VERIFY-7491 / SCENARIO-VERIFY-7491-WINDOWS."""

    windows = protocol.build_lossless_windows(response)
    payload = response.encode("utf-8")

    assert 1 <= len(windows) <= 3
    assert windows[0]["byte_start"] == 0
    assert windows[-1]["byte_end"] == len(payload)
    assert all(left["byte_end"] == right["byte_start"] for left, right in zip(windows, windows[1:]))
    assert b"".join(payload[row["byte_start"] : row["byte_end"]] for row in windows) == payload
    assert all(row["window_version"] == protocol.WINDOW_VERSION for row in windows)


def test_one_long_sentence_stays_whole_and_bad_byte_boundary_is_rejected() -> None:
    """REQ-VERIFY-7491 keeps a sentence whole and rejects Unicode byte splits."""

    response = "é" + (" long" * 500)
    assert protocol.build_lossless_windows(response) == [
        {
            "window_index": 0,
            "byte_start": 0,
            "byte_end": len(response.encode("utf-8")),
            "sentence_count": 1,
            "window_sha256": protocol.sha256_bytes(response.encode("utf-8")),
            "sentence_version": protocol.SENTENCE_VERSION,
            "window_version": protocol.WINDOW_VERSION,
        }
    ]
    with pytest.raises(protocol.WindowProtocolError, match="focus_boundary_not_utf8"):
        protocol.mark_focus(response, 1, len(response.encode("utf-8")))
    with pytest.raises(protocol.WindowProtocolError, match="focus_boundary_not_utf8"):
        protocol.mark_focus(response, -1, len(response.encode("utf-8")))


def test_sentence_closers_remain_with_the_complete_sentence() -> None:
    """REQ-VERIFY-7491 retains closing punctuation and following whitespace."""

    response = "First sentence!\u201d  Second sentence."
    windows = protocol.build_lossless_windows(response)
    first = response.encode("utf-8")[windows[0]["byte_start"] : windows[0]["byte_end"]]
    assert first.decode("utf-8") == "First sentence!\u201d  "


def test_focused_prompt_keeps_one_full_response_and_whole_prompt_is_unchanged() -> None:
    """REQ-VERIFY-7491 / SCENARIO-VERIFY-7491-PROMPTS."""

    source = "The source says Café opened in 2020."
    response = "It did not close. It is open today."
    window = protocol.build_lossless_windows(response)[0]
    focused = protocol.build_focused_prompt(source, response, OPTION_IDS, window)

    assert protocol.build_whole_prompt(source, response, OPTION_IDS) == build_option_prompt(
        source, response, OPTION_IDS
    )
    assert focused["source_occurrences"] == 1
    assert focused["response_recovered"] is True
    assert protocol.remove_focus_markers(focused["marked_response"]) == response
    assert focused["prompt"].count(protocol.FOCUS_START_PREFIX) == 1
    assert focused["prompt"].count(protocol.FOCUS_END) == 1
    assert "group_id" not in focused["prompt"]
    assert "annotation" not in focused["prompt"].lower()

    corrupted = focused["marked_response"].replace(protocol.FOCUS_END, "", 1)
    with pytest.raises(protocol.WindowProtocolError, match="focus_markers_invalid"):
        protocol.remove_focus_markers(corrupted)


def _fit_predictor(index: int, role: str) -> dict[str, object]:
    source = f"Fit source {index}."
    response = f"Fit response {index}."
    return {
        "row_key": f"fit-row-{index}",
        "group_id": f"fit-group-{index}",
        "corpus": "ragtruth",
        "role": role,
        "source_family": "QA",
        "source_text": source,
        "response_text": response,
        "source_hash": protocol.sha256_text(source.casefold()),
        "response_hash": protocol.sha256_text(response),
    }


def _candidate(index: int) -> dict[str, object]:
    source = f"Candidate source {index}."
    response = f"Candidate response {index}."
    return {
        "row_key": f"candidate-row-{index}",
        "group_id": f"candidate-group-{index}",
        "source_alias": f"source-{index}",
        "response_alias": f"response-{index}",
        "official_split": "train",
        "source_family": "Summary",
        "source_text": source,
        "response_text": response,
        "source_hash": protocol.sha256_text(source.casefold()),
        "response_hash": protocol.sha256_text(response),
    }


def test_roles_are_frozen_without_labels_and_evaluator_stays_separate() -> None:
    """REQ-VERIFY-7491 / SCENARIO-VERIFY-7491-ROLES."""

    fit = [_fit_predictor(0, "training"), _fit_predictor(1, "calibration_tuning")]
    candidates = [_candidate(index) for index in range(7)]
    exclusions = {
        "source_hashes": {str(candidates[0]["source_hash"])},
        "group_aliases": {"candidate-group-1"},
        "source_aliases": {"source-2"},
    }
    frozen = protocol.freeze_roles(
        fit,
        candidates,
        exclusions,
        role_caps={"test": 2, "online": 2},
        split_seed=656001,
    )
    reversed_frozen = protocol.freeze_roles(
        fit,
        list(reversed(candidates)),
        exclusions,
        role_caps={"test": 2, "online": 2},
        split_seed=656001,
    )

    assert frozen == reversed_frozen
    assert [row["role"] for row in frozen[:2]] == ["training", "calibration_tuning"]
    assert sum(row["role"] == "test" for row in frozen) == 2
    assert sum(row["role"] == "online" for row in frozen) == 2
    assert all(not (set(row) & protocol.FORBIDDEN_PREDICTOR_FIELDS) for row in frozen)

    evaluator_by_group = {
        str(row["group_id"]): {
            "group_id": row["group_id"],
            "label": index % 2,
            "annotations": [f"private-{index}"],
            "source_id": f"source-{index}",
            "response_id": f"response-{index}",
            "response_generator_identity": "private-generator",
        }
        for index, row in enumerate(frozen)
    }
    evaluators = protocol.freeze_evaluator_store(frozen, evaluator_by_group)
    assert len(evaluators) == len(frozen)
    assert {row["label"] for row in evaluators if row["role"] == "test"} == {0, 1}
    assert all("label" not in row for row in frozen)
    with pytest.raises(protocol.WindowProtocolError, match="evaluator_group_missing"):
        protocol.freeze_evaluator_store(frozen, {})
    with pytest.raises(protocol.WindowProtocolError, match="insufficient_role_candidates"):
        protocol.freeze_roles(
            fit,
            [],
            exclusions,
            role_caps={"test": 1, "online": 1},
            split_seed=656001,
        )


def test_request_manifest_enforces_prompt_and_forward_ceilings_before_labels() -> None:
    """REQ-VERIFY-7491 / SCENARIO-VERIFY-7491-BUDGET."""

    groups = [
        _fit_predictor(0, "training"),
        _fit_predictor(1, "calibration_tuning"),
        {**_fit_predictor(2, "test"), "response_text": "One. Two. Three. Four."},
        _fit_predictor(3, "online"),
    ]
    plan = protocol.build_request_plan(
        groups,
        token_count=lambda text: len(text.encode("utf-8")),
        token_ceiling=20_000,
        control_group_limit=1,
    )

    main_by_group: dict[str, int] = {}
    for row in plan["requests"]:
        if row["arm"] != "source_derangement_control":
            main_by_group[str(row["group_id"])] = main_by_group.get(str(row["group_id"]), 0) + 1
        assert row["gold_label"] is None
        assert row["prompt_token_count"] <= 20_000
    assert all(count <= 8 and count % 2 == 0 for count in main_by_group.values())
    assert plan["budgets"]["fit"]["planned_forwards"] <= plan["budgets"]["fit"]["ceiling"]
    assert (
        plan["budgets"]["evaluation"]["planned_forwards"]
        <= plan["budgets"]["evaluation"]["ceiling"]
    )
    controls = [row for row in plan["requests"] if row["arm"] == "source_derangement_control"]
    assert len(controls) == 2
    assert all(row["source_group_id"] != row["group_id"] for row in controls)

    overlength = protocol.build_request_plan(
        groups,
        token_count=lambda text: 2_049 if protocol.FOCUS_START_PREFIX in text else 10,
        token_ceiling=2_048,
        control_group_limit=0,
    )
    assert all(row["eligible"] is False for row in overlength["group_rows"])
    assert all(row["disposition"] == "excluded_overlength" for row in overlength["group_rows"])

    with pytest.raises(protocol.WindowProtocolError, match="derangement_donor_missing"):
        protocol.build_request_plan(
            [_fit_predictor(0, "calibration_tuning")],
            token_count=len,
            token_ceiling=20_000,
            control_group_limit=1,
        )


def test_protocol_shards_replay_and_reject_byte_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-7491 independently reduces hash-bound per-unit rows."""

    predictors = [_fit_predictor(0, "training"), _fit_predictor(1, "test")]
    evaluators = [
        {"group_id": "fit-group-0", "role": "training", "label": 1},
        {"group_id": "fit-group-1", "role": "test", "label": 0},
    ]
    plan = protocol.build_request_plan(
        predictors,
        token_count=lambda text: len(text),
        token_ceiling=10_000,
        control_group_limit=0,
    )
    manifest = protocol.seal_protocol_shards(
        tmp_path,
        predictors=predictors,
        evaluators=evaluators,
        group_rows=plan["group_rows"],
        window_rows=plan["window_rows"],
        request_rows=plan["requests"],
        exposure_rows=[{"path": "fixture", "status": "searched"}],
    )
    replay = protocol.reload_protocol_shards(tmp_path, manifest)
    assert replay["role_counts"] == {"training": 1, "test": 1}
    assert replay["lossless_windows"] is True
    assert replay["prompt_isolation"] is True

    request_path = tmp_path / "requests.jsonl"
    request_path.write_text(request_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(protocol.WindowProtocolError, match="raw_shard_hash_mismatch"):
        protocol.reload_protocol_shards(tmp_path, manifest)


def test_artifact_fixture_has_no_model_calls_and_cold_validation(tmp_path: Path) -> None:
    """REQ-VERIFY-7491 / SCENARIO-VERIFY-7491-NO-MODEL and E2E."""

    artifact = protocol.build_artifact_for_test(tmp_path)
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert not any(artifact["invocation_counts"].values())
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["small_ebm_training"]["fit_attempts"] == 0
    assert artifact["window_protocol_ready_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert set(artifact["field_principles"]) == set(artifact)
    assert protocol.validate_artifact(artifact, root=tmp_path, require_terminal=False) == []

    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["invented-model"]
    assert "current_model_provenance_invalid" in protocol.validate_artifact(
        changed, root=tmp_path, require_terminal=False
    )
    changed = deepcopy(artifact)
    changed["rows"][0]["eligible"] = False
    assert "independent_reduction_mismatch" in protocol.validate_artifact(
        changed, root=tmp_path, require_terminal=False
    )
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in protocol.validate_artifact(
        changed, root=tmp_path, require_terminal=False
    )


def test_gate_summary_names_every_failure_with_a_principle() -> None:
    """REQ-VERIFY-7491 requires falsifiable typed gates and diagnosed failures."""

    gates = protocol.build_acceptance_gates(
        preconditions_ok=True,
        freshness_complete=False,
        exact_role_counts=False,
        minima_met=False,
        class_support_met=False,
        windows_lossless=True,
        prompts_isolated=True,
        budgets_valid=True,
        validation_passed=False,
    )
    summary = protocol.gate_check_summary(gates)
    assert summary["passed"] is False
    assert summary["first_failure"]["check"] == "freshness_search_complete"
    assert all(row["principle"] for row in gates)
    assert all(
        {"check", "upstream", "field_path", "expected", "observed", "op"} <= set(row)
        for row in summary["failed_checks"]
    )


def test_jsonl_loader_rejects_non_objects(tmp_path: Path) -> None:
    """REQ-VERIFY-7491 rejects malformed raw evidence instead of skipping it."""

    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps([1, 2, 3]) + "\n", encoding="utf-8")
    with pytest.raises(protocol.WindowProtocolError, match="jsonl_row_invalid"):
        protocol.load_jsonl(path)

    path.write_text("\n" + json.dumps({"ok": True}) + "\n", encoding="utf-8")
    assert protocol.load_jsonl(path) == [{"ok": True}]


def test_cold_validator_covers_absolute_paths_and_terminal_failures(tmp_path: Path) -> None:
    """REQ-VERIFY-7491 reports malformed raw and incomplete terminal evidence."""

    artifact = protocol.build_artifact_for_test(tmp_path)
    artifact["role_manifest"]["raw_directory"] = "."
    artifact["reproducibility_checksum"] = protocol.artifact_checksum(artifact)
    assert protocol.validate_artifact(artifact, root=tmp_path, require_terminal=False) == []
    artifact["role_manifest"]["raw_directory"] = str(tmp_path.resolve())
    artifact["field_principles"].pop("schema")
    artifact["reproducibility_checksum"] = protocol.artifact_checksum(artifact)
    errors = protocol.validate_artifact(artifact, root=tmp_path, require_terminal=True)
    assert "field_principles_incomplete" in errors
    assert "terminal_validation_incomplete" in errors

    artifact["role_manifest"]["raw_shards"] = {}
    artifact["reproducibility_checksum"] = protocol.artifact_checksum(artifact)
    assert "independent_reduction_failed" in protocol.validate_artifact(
        artifact, root=tmp_path, require_terminal=False
    )
