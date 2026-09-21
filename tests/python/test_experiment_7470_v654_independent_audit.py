"""Tests for REQ-REPORT-7470 and SCENARIO-REPORT-7470-*.

The tests use small numeric fixtures for absent decision branches. They use the
real immutable Exp7467 sidecars for the available extraction branch.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time

import pytest

from carnot import experiment_7470_v654_independent_audit as audit


def _decision_fixture() -> tuple[list[dict[str, object]], dict[str, object]]:
    checkpoint: dict[str, object] = {
        "feature_names": ["support", "missing"],
        "weights": {
            "accept": [1.0, 0.0],
            "reject": [-1.0, 0.0],
            "escalate": [0.0, 0.0],
        },
        "bias": {"accept": 0.0, "reject": 0.0, "escalate": 0.0},
    }
    checkpoint["checkpoint_sha256"] = audit.canonical_hash(checkpoint)
    rows: list[dict[str, object]] = []
    for index, (support, label) in enumerate(((1.0, "accept"), (-1.0, "reject"))):
        canonical_logits = {
            "accept": support,
            "reject": -support,
            "escalate": 0.0,
        }
        option_order = ["reject", "escalate", "accept"]
        stored = audit.softmax_by_option(
            [canonical_logits[name] for name in option_order], option_order
        )
        rows.append(
            {
                "unit_id": f"decision-{index}",
                "group_id": f"group-{index}",
                "role": "evaluation",
                "option_order": option_order,
                "raw_logits": [canonical_logits[name] for name in option_order],
                "stored_probabilities": stored,
                "features": {"support": support, "missing": 0.0},
                "label": label,
                "eligible": True,
                "source_swap_control": True,
                "all_escalate_cost": 1.0,
            }
        )
    return rows, checkpoint


def _update_fixture() -> tuple[list[dict[str, object]], dict[str, object]]:
    checkpoint: dict[str, object] = {
        "seed": 17,
        "weights": [0.0, 0.0],
        "updates": 0,
        "learning_rate": 0.25,
        "frozen_control": False,
    }
    checkpoint["checkpoint_sha256"] = audit.canonical_hash(checkpoint)
    before = audit.residual_state_hash(checkpoint)
    probability = 0.5
    next_weights = [0.125, 0.0]
    after = audit.residual_state_hash({"seed": 17, "weights": next_weights, "updates": 1})
    events: list[dict[str, object]] = [
        {
            "type": "prediction",
            "event_id": "p1",
            "timestamp_ns": 10,
            "seed": 17,
            "group_id": "g1",
            "arm": "residual",
            "features": [1.0, 0.0],
            "probability": probability,
            "state_hash_before": before,
        },
        {
            "type": "feedback",
            "event_id": "f1",
            "prediction_event_id": "p1",
            "timestamp_ns": 20,
            "seed": 17,
            "label": 1,
            "audit_selected": True,
            "audit_probability": 0.5,
            "state_hash_before": before,
            "state_hash_after": after,
        },
        {
            "type": "retention",
            "event_id": "r1",
            "timestamp_ns": 30,
            "seed": 17,
            "features": [1.0, 0.0],
            "probability": 1.0 / (1.0 + math.exp(-0.125)),
            "state_hash": after,
        },
    ]
    return events, checkpoint


def test_principle_values_do_not_unwrap_normal_mappings() -> None:
    """REQ-REPORT-7470: only explicit principle wrappers unwrap."""

    ordinary = {"value": 3, "other": "evidence"}
    wrapped = {"principle": "why", "value": 4}
    assert audit.principle_value(ordinary) is ordinary
    assert audit.principle_value(wrapped) == 4


def test_producer_location_is_exact_and_branch_local(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7470-BRANCHES: alternate names need exact identity."""

    results = tmp_path / "results"
    results.mkdir()
    wrong = results / "experiment_7466_wrong_name.json"
    wrong.write_text(
        json.dumps(
            {
                "experiment_id": audit.PRODUCERS["typed_decision"].experiment_id,
                "milestone": "wrong",
            }
        )
    )
    slot = audit.locate_producer(tmp_path, audit.PRODUCERS["typed_decision"])
    assert slot["availability"] == "missing"
    exact = results / "conductor_pre_gate_7466.json"
    exact.write_text(
        json.dumps(
            {
                "experiment_id": audit.PRODUCERS["typed_decision"].experiment_id,
                "milestone": audit.MILESTONE,
                "verdict_class": "blocked",
                "flagged_adversarial": False,
                "blocked_at_layer": "conductor_pre_gate",
            }
        )
    )
    slot = audit.locate_producer(tmp_path, audit.PRODUCERS["typed_decision"])
    assert slot["availability"] == "pre_gate"
    assert slot["path"] == "results/conductor_pre_gate_7466.json"
    external = tmp_path / "external.json"
    external.write_text("{}")
    nested_root = tmp_path / "nested"
    nested_root.mkdir()
    assert audit._reference(nested_root, external, "test")["path"] == str(external.resolve())
    exact.unlink()
    declared = tmp_path / audit.PRODUCERS["typed_decision"].path
    declared.write_text(json.dumps({"experiment_id": "wrong", "milestone": "wrong"}))
    assert (
        audit.locate_producer(tmp_path, audit.PRODUCERS["typed_decision"])["availability"]
        == "invalid"
    )


def test_decision_reducer_remaps_options_and_rejects_leakage() -> None:
    """SCENARIO-REPORT-7470-DECISIONS: recompute fit-free probabilities."""

    rows, checkpoint = _decision_fixture()
    reduced = audit.audit_decision_rows(rows, checkpoint)
    assert reduced["errors"] == []
    assert reduced["groups"] == 2
    assert reduced["full_source_eligible"] is True
    assert reduced["mean_brier"] < 0.3

    remapped = deepcopy(rows)
    remapped[0]["option_order"] = ["accept", "escalate", "reject"]
    assert (
        "option_mapping_mismatch:decision-0"
        in audit.audit_decision_rows(remapped, checkpoint)["errors"]
    )

    leaked = deepcopy(rows)
    leaked[0]["features"] = {"support": 1.0, "missing": 0.0, "gold_label": 1.0}
    assert (
        "label_leakage:decision-0:gold_label"
        in audit.audit_decision_rows(leaked, checkpoint)["errors"]
    )

    fabricated = audit.verify_declared_improvement(rows, declared=-0.9)
    assert fabricated == ["fabricated_improvement"]


def test_scalar_update_replay_detects_causal_and_checkpoint_changes() -> None:
    """SCENARIO-REPORT-7470-UPDATES: replay saved state without producer code."""

    events, checkpoint = _update_fixture()
    replay = audit.replay_residual_updates(events, checkpoint)
    assert replay["errors"] == []
    assert replay["updates_replayed"] == 1
    assert replay["retention_predictions_checked"] == 1

    delayed = deepcopy(events)
    delayed[1]["timestamp_ns"] = 9
    assert (
        "label_before_prediction:f1" in audit.replay_residual_updates(delayed, checkpoint)["errors"]
    )

    changed = deepcopy(checkpoint)
    changed["weights"] = [0.1, 0.0]
    assert "checkpoint_hash_mismatch" in audit.replay_residual_updates(events, changed)["errors"]

    duplicated = [*deepcopy(events), deepcopy(events[1])]
    assert (
        "duplicate_feedback:p1" in audit.replay_residual_updates(duplicated, checkpoint)["errors"]
    )


def test_real_extraction_replay_keeps_all_108_dispositions() -> None:
    """SCENARIO-REPORT-7470-EXTRACTION: reparse every available raw reply."""

    producer = json.loads((audit.REPO_ROOT / audit.PRODUCERS["extraction"].path).read_text())
    reduced = audit.audit_extraction_branch(audit.REPO_ROOT, producer)
    assert reduced["errors"] == []
    assert len(reduced["rows"]) == 108
    assert reduced["counts"] == {
        "planned": 108,
        "attempted": 12,
        "completed": 7,
        "failed": 5,
        "censored": 0,
        "unstarted": 96,
    }
    assert reduced["factual_content_counts"] == {"span": 1, "verbatim": 2}
    assert reduced["correct_empty_controls"] == 4
    assert reduced["natural_annotation_uncertainty"] is True
    assert reduced["constructed_exact_pairs"] == 12


def test_mutations_and_artifact_validation_fail_closed() -> None:
    """SCENARIO-REPORT-7470-MUTATIONS/ARTIFACT: six controls and checksum bind."""

    mutations = audit.run_mutation_controls()
    assert [row["mutation"] for row in mutations] == list(audit.REQUIRED_MUTATIONS)
    assert all(row["caught"] for row in mutations)

    artifact = audit.build_artifact_for_test()
    assert audit.validate_artifact(artifact) == []
    assert artifact["independent_audit_complete_score"] == 1
    assert [row["verdict_class"] for row in artifact["branch_rows"]] == [
        "blocked",
        "blocked",
        "null",
    ]
    mutated = deepcopy(artifact)
    mutated["branch_rows"][0]["availability"] = "available"
    assert "reproducibility_checksum_mismatch" in audit.validate_artifact(mutated)


def test_parser_rejects_missing_failure_row_and_bad_literal() -> None:
    """SCENARIO-REPORT-7470-EXTRACTION: failures and literal spans are required."""

    producer = json.loads((audit.REPO_ROOT / audit.PRODUCERS["extraction"].path).read_text())
    rows = producer["development_rows"] + producer["extraction_rows"]
    assert audit.validate_extraction_completeness(rows[:-1]) == [
        "planned_extraction_row_count_mismatch"
    ]
    row = {
        "call_id": "literal",
        "arm": "verbatim",
        "capture_phase": "development",
        "canary_kind": "factual",
        "attempted": True,
        "terminal_state": "response",
        "paragraph": "A literal fact.",
        "raw_reply": '{"claims":["invented fact"]}',
    }
    decoded = audit.parse_extraction_row(row, row)
    assert decoded["disposition"] == "malformed"
    assert decoded["errors"] == ["nonliteral_claim"]


@pytest.mark.parametrize(
    ("events_change", "expected"),
    [
        (lambda rows: rows[1].update(audit_selected=False), "unaudited_update:f1"),
        (lambda rows: rows[1].update(seed=18), "seed_pooling:f1"),
        (lambda rows: rows[1].update(state_hash_before="stale"), "stale_state:f1"),
    ],
)
def test_update_replay_rejects_other_causal_violations(events_change, expected: str) -> None:
    """SCENARIO-REPORT-7470-UPDATES: causal guards remain independent."""

    events, checkpoint = _update_fixture()
    events_change(events)
    assert expected in audit.replay_residual_updates(events, checkpoint)["errors"]


def test_numeric_readers_cover_fail_closed_shapes() -> None:
    """REQ-REPORT-7470: malformed decision evidence names each failed check."""

    with pytest.raises(ValueError, match="option_mapping_invalid"):
        audit.softmax_by_option([1.0], ["accept"])
    with pytest.raises(ValueError, match="raw_logit_invalid"):
        audit.softmax_by_option([math.inf, 0.0, 1.0], audit.OPTIONS)

    rows, checkpoint = _decision_fixture()
    checkpoint["checkpoint_sha256"] = "changed"
    rows[1]["group_id"] = rows[0]["group_id"]
    rows[1]["stored_probabilities"] = {}
    rows[1]["label"] = "unknown"
    rows[1]["source_swap_control"] = False
    rows[1]["all_escalate_cost"] = None
    errors = audit.audit_decision_rows(rows, checkpoint)["errors"]
    assert "checkpoint_hash_mismatch" in errors
    assert "group_separation:decision-1" in errors
    assert "probability_mismatch:decision-1" in errors
    assert "label_invalid:decision-1" in errors

    missing_controls, valid_checkpoint = _decision_fixture()
    missing_controls[0]["source_swap_control"] = False
    missing_controls[0]["all_escalate_cost"] = None
    control_errors = audit.audit_decision_rows(missing_controls, valid_checkpoint)["errors"]
    assert "source_swap_control_missing:decision-0" in control_errors
    assert "all_escalate_baseline_missing:decision-0" in control_errors

    no_features = deepcopy(rows[:1])
    no_features[0]["features"] = None
    assert (
        "features_invalid:decision-0"
        in audit.audit_decision_rows(no_features, checkpoint)["errors"]
    )
    bad_checkpoint = deepcopy(checkpoint)
    bad_checkpoint.pop("feature_names")
    assert any(
        error.startswith("decision_shape:decision-0:checkpoint_shape_invalid")
        for error in audit.audit_decision_rows(_decision_fixture()[0][:1], bad_checkpoint)["errors"]
    )
    assert audit._sigmoid(-1.0) < 0.5


def test_update_reader_names_every_remaining_error_exit() -> None:
    """SCENARIO-REPORT-7470-UPDATES: malformed causal events fail closed."""

    events, checkpoint = _update_fixture()
    events[0].update(seed=18, state_hash_before="stale", probability=0.7)
    events[1].update(frozen=True, state_hash_after="bad")
    checkpoint["frozen_control"] = True
    checkpoint["checkpoint_sha256"] = audit.canonical_hash(
        {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    )
    errors = audit.replay_residual_updates(events, checkpoint)["errors"]
    assert "seed_pooling:p1" in errors
    assert "stale_state:p1" in errors
    assert "prediction_probability_mismatch:p1" in errors
    assert "nonfrozen_control:f1" in errors
    assert "state_hash_mismatch:f1" in errors

    missing = deepcopy(events[1])
    missing.update(event_id="missing", prediction_event_id="absent")
    invalid = deepcopy(_update_fixture()[0])
    invalid.extend(
        [
            missing,
            {
                "type": "feedback",
                "event_id": "bad-label",
                "prediction_event_id": "p1",
                "timestamp_ns": 30,
                "seed": 17,
                "label": 2,
                "audit_selected": True,
                "audit_probability": 0.5,
                "state_hash_before": invalid[1]["state_hash_after"],
            },
            {
                "type": "retention",
                "event_id": "ret",
                "features": [-1.0, 0.0],
                "probability": 0.9,
                "state_hash": "bad",
            },
            {"type": "unknown", "event_id": "unknown"},
        ]
    )
    errors = audit.replay_residual_updates(invalid, _update_fixture()[1])["errors"]
    assert "feedback_prediction_missing:missing" in errors
    assert "label_invalid:bad-label" in errors
    assert "retention_state_mismatch:ret" in errors
    assert "retention_prediction_mismatch:ret" in errors
    assert "event_type_invalid:unknown" in errors


def test_extraction_parser_rejects_all_malformed_shapes() -> None:
    """SCENARIO-REPORT-7470-EXTRACTION: strict literal parsing has named exits."""

    unstarted_meta = {
        "call_id": "u",
        "attempted": False,
        "terminal_state": "unstarted",
        "raw_request_sha256": audit.canonical_hash({}),
        "raw_response_sha256": audit.canonical_hash({}),
        "raw_reply_sha256": audit.canonical_hash(""),
    }
    bad_unstarted = {
        "call_id": "other",
        "attempted": False,
        "terminal_state": "response",
        "raw_reply": "x",
        "raw_request": {"changed": True},
        "raw_response": {"changed": True},
    }
    errors = audit.parse_extraction_row(unstarted_meta, bad_unstarted)["errors"]
    assert "raw_call_id_mismatch" in errors
    assert "raw_terminal_state_mismatch" in errors
    assert "raw_request_hash_mismatch" in errors
    assert "raw_response_hash_mismatch" in errors
    assert "raw_reply_hash_mismatch" in errors
    assert "unstarted_payload_invalid" in errors

    base = {
        "call_id": "x",
        "arm": "span",
        "capture_phase": "development",
        "canary_kind": "factual",
        "attempted": True,
        "terminal_state": "response",
        "paragraph": "abc",
    }
    malformed = audit.parse_extraction_row(base, {**base, "raw_reply": "{"})
    assert {"malformed_json", "claims_shape"} <= set(malformed["errors"])
    span_shape = audit.parse_extraction_row(
        base, {**base, "raw_reply": '{"claims":[[0], [true, 2]]}'}
    )
    assert span_shape["errors"] == ["span_shape"]
    verbatim = {**base, "arm": "verbatim", "paragraph": "a a"}
    bad_verbatim = audit.parse_extraction_row(
        verbatim, {**verbatim, "raw_reply": '{"claims":[1,"a","a"]}'}
    )
    assert {"verbatim_shape", "ambiguous_claim"} <= set(bad_verbatim["errors"])
    unknown = {**base, "arm": "unknown"}
    assert audit.parse_extraction_row(unknown, {**unknown, "raw_reply": '{"claims":[]}'})[
        "errors"
    ] == ["arm_invalid"]
    empty = audit.parse_extraction_row(base, {**base, "raw_reply": '{"claims":[]}'})
    assert empty["disposition"] == "empty"


def test_sidecar_and_extraction_audit_error_paths(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-REPORT-7470-EXTRACTION: sidecar and summary drift are visible."""

    directory = tmp_path / "sidecars"
    directory.mkdir()
    first = directory / "sha256-bad.json"
    first.write_text(json.dumps({"call_id": "duplicate"}))
    second = directory / "sha256-worse.json"
    second.write_text(json.dumps({"call_id": "duplicate"}))
    values, _references, errors = audit._load_hash_named(directory, tmp_path)
    assert list(values) == ["duplicate"]
    assert sum("sidecar_filename_hash_mismatch" in error for error in errors) == 2
    assert any("sidecar_call_id_invalid" in error for error in errors)
    assert audit.audit_extraction_branch(tmp_path, {})["errors"][-1] == (
        "producer_extraction_rows_invalid"
    )

    dev_meta = {
        "call_id": "dev",
        "unit_id": "dev",
        "arm": "verbatim",
        "capture_phase": "development",
        "canary_kind": "factual",
        "attempted": True,
        "terminal_state": "response",
        "paragraph": "fact",
        "development_disposition": "wrong",
        "censored": False,
    }
    dev_raw = {**dev_meta, "raw_reply": '{"claims":["fact"]}'}
    eval_meta = {
        "call_id": "eval",
        "unit_id": "eval",
        "arm": "span",
        "capture_phase": "evaluation",
        "attempted": False,
        "terminal_state": "unstarted",
        "raw_reply": "",
        "disposition": "wrong",
        "censored": False,
        "extra": "producer",
    }
    eval_raw = {key: value for key, value in eval_meta.items() if key != "extra"}

    def fake_load(directory_path: Path, _root: Path):
        if directory_path.name == "responses":
            return {"dev": dev_raw}, [], []
        return {"eval": eval_raw}, [], []

    monkeypatch.setattr(audit, "_load_hash_named", fake_load)
    producer = {
        "development_rows": [dev_meta],
        "extraction_rows": [eval_meta],
        "semantic_pair_rows": [
            {
                "pair_id": "bad",
                "authority": "natural",
                "span_qualifier_retained": False,
                "verbatim_qualifier_retained": True,
                "paired_delta": 0,
            }
        ],
    }
    errors = audit.audit_extraction_branch(tmp_path, producer)["errors"]
    assert "planned_extraction_row_count_mismatch" in errors
    assert "development_disposition_mismatch:dev" in errors
    assert "evaluation_sidecar_row_mismatch:eval" in errors
    assert "evaluation_disposition_mismatch:eval" in errors
    assert "constructed_pair_count_mismatch" in errors
    assert "constructed_authority_invalid:bad" in errors
    assert "constructed_delta_mismatch:bad" in errors

    def missing_development(directory_path: Path, _root: Path):
        if directory_path.name == "responses":
            return {}, [], []
        return {"eval": eval_raw}, [], []

    monkeypatch.setattr(audit, "_load_hash_named", missing_development)
    assert (
        "development_sidecar_missing:dev"
        in audit.audit_extraction_branch(tmp_path, producer)["errors"]
    )

    def missing_evaluation(directory_path: Path, _root: Path):
        if directory_path.name == "responses":
            return {"dev": dev_raw}, [], []
        return {}, [], []

    monkeypatch.setattr(audit, "_load_hash_named", missing_evaluation)
    assert (
        "evaluation_sidecar_missing:eval"
        in audit.audit_extraction_branch(tmp_path, producer)["errors"]
    )


def test_audit_sources_covers_pre_gate_invalid_and_available_branches(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-REPORT-7470-BRANCHES: all availability classes terminate."""

    slots = {
        "typed_decision": {
            "branch": "typed_decision",
            "availability": "pre_gate",
            "path": "typed.json",
            "sha256": "sha256:typed",
            "artifact": {"verdict_class": "blocked", "flagged_adversarial": False},
        },
        "residual_learning": {
            "branch": "residual_learning",
            "availability": "invalid",
            "path": "residual.json",
            "sha256": "sha256:residual",
            "artifact": {},
        },
        "extraction": {
            "branch": "extraction",
            "availability": "available",
            "path": "extract.json",
            "sha256": "sha256:extract",
            "artifact": {"verdict_class": "null", "flagged_adversarial": "bad"},
        },
    }
    monkeypatch.setattr(audit, "locate_producer", lambda _root, spec: slots[spec.branch])
    first = audit.audit_sources(tmp_path)
    assert [row["verdict_class"] for row in first["branch_rows"]] == [
        "blocked",
        "disqualified",
        "disqualified",
    ]

    decision_rows, decision_checkpoint = _decision_fixture()
    update_events, update_checkpoint = _update_fixture()
    slots["typed_decision"] = {
        "branch": "typed_decision",
        "availability": "available",
        "path": "typed.json",
        "sha256": "sha256:typed",
        "artifact": {
            "verdict_class": "null",
            "flagged_adversarial": False,
            "rows": decision_rows,
            "frozen_checkpoints": [decision_checkpoint],
        },
    }
    slots["residual_learning"] = {
        "branch": "residual_learning",
        "availability": "available",
        "path": "residual.json",
        "sha256": "sha256:residual",
        "artifact": {
            "verdict_class": "null",
            "flagged_adversarial": False,
            "causal_event_rows": update_events,
            "initial_checkpoint": update_checkpoint,
        },
    }
    slots["extraction"]["artifact"] = {"verdict_class": "null", "flagged_adversarial": False}
    monkeypatch.setattr(
        audit,
        "audit_extraction_branch",
        lambda _root, _producer: {
            "errors": [],
            "rows": [],
            "sidecars": [],
            "counts": {},
        },
    )
    second = audit.audit_sources(tmp_path)
    assert [row["verdict_class"] for row in second["branch_rows"]] == [
        "null",
        "null",
        "null",
    ]
    slots["residual_learning"]["artifact"].pop("causal_event_rows")
    assert audit.audit_sources(tmp_path)["branch_rows"][1]["verdict_class"] == "disqualified"


def test_artifact_error_contract_and_fresh_readers(tmp_path: Path, monkeypatch, capsys) -> None:
    """SCENARIO-REPORT-7470-ARTIFACT: every protected projection fails closed."""

    artifact = audit.build_artifact_for_test()
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact))
    assert audit.cold_replay(path) == []
    assert audit.cold_replay(tmp_path / "missing.json") == ["candidate_json_invalid"]
    assert audit._gate_summary([{"check": "ok", "passed": True}])["all_passed"] is True

    bad = deepcopy(artifact)
    bad.pop("schema")
    bad.update(
        {
            "experiment_id": "wrong",
            "milestone": "wrong",
            "run_date": "wrong",
            "MODEL_SPECS": ["model"],
            "model_specs": ["model"],
            "model_invoked": True,
            "invocation_counts": {},
            "inference_substrate": "generation",
            "inference_substrate_class": "generation",
            "execution_venue": "gpu",
            "branch_rows": [],
            "mutation_rows": [],
            "rows": [],
            "validation_receipts": [],
            "independent_audit_complete_score": 2,
            "promotion_score": 1,
        }
    )
    errors = audit.validate_artifact(bad)
    assert "required_field_missing:schema" in errors
    assert "artifact_identity_mismatch" in errors
    assert "artifact_execution_contract_mismatch" in errors
    assert "model_specs_not_empty" in errors
    assert "current_invocation_not_zero" in errors
    assert "inference_substrate_invalid" in errors
    assert "execution_declaration_invalid" in errors
    assert "branch_rows_invalid" in errors
    assert "mutation_rows_invalid" in errors
    assert "sample_budget_row_mismatch" in errors
    assert "required_validation_failed" in errors
    assert "independent_audit_complete_score_invalid" in errors
    assert "promotion_forbidden" in errors

    uncaught = deepcopy(artifact)
    uncaught["mutation_rows"][0]["caught"] = False
    assert "mutation_not_caught" in audit.validate_artifact(uncaught)
    monkeypatch.setattr(
        audit,
        "audit_sources",
        lambda _root: {
            "branch_rows": artifact["branch_rows"],
            "rows": artifact["rows"],
            "independent_update_replay": artifact["independent_update_replay"],
        },
    )
    assert audit.independent_replay(path, root=tmp_path) == []
    changed = deepcopy(artifact)
    changed["branch_rows"] = []
    changed["rows"] = []
    changed["independent_update_replay"] = {}
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed))
    assert audit.independent_replay(changed_path, root=tmp_path) == [
        "branch_reduction_mismatch",
        "row_reduction_mismatch",
        "update_replay_mismatch",
    ]
    assert audit.independent_replay(tmp_path / "absent.json") == ["candidate_json_invalid"]
    audit._progress(time.monotonic(), "test", "event", completed_units=1)
    assert "completed_units=1" in capsys.readouterr().out
