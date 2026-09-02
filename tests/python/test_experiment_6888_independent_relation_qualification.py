"""Tests for the independent held relation reducer.

Spec refs: REQ-VERIFY-6888 and SCENARIO-VERIFY-6888-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6888_independent_relation_qualification as exp


def _source(
    *,
    arm_split: str = "calibration",
    fixture_id: str = "graph_coloring_00",
    source_text: str = "Node n0 has color red.",
) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "group_id": "relation_group_00",
        "family": "graph_coloring",
        "split": arm_split,
        "source_order": 0,
        "source_text": source_text,
        "source_text_hash": exp.sha256_text(source_text),
    }


def _parse_relation(
    *,
    start: int = 5,
    end: int = 7,
    status: str = "accepted",
    predicate: str = "has_color",
    polarity: str = "positive",
) -> dict[str, Any]:
    return {
        "line_index": 0,
        "raw_line": f"REL\t{start}\t{end}\t{predicate}\t18\t21\t{polarity}",
        "status": status,
        "reason": "accepted" if status == "accepted" else "fixture_reason",
        "subject": {"start_utf8": start, "end_utf8": end, "text": "n0"},
        "predicate": predicate,
        "object": {"start_utf8": 18, "end_utf8": 21, "text": "red"},
        "polarity": polarity,
        "normalized_tuple": ["n0", predicate, "red", polarity],
    }


def _cell(
    *,
    arm: str = "arm:rule",
    split: str = "calibration",
    parse_rows: list[dict[str, Any]] | None = None,
    raw_output: str = "REL\t5\t7\thas_color\t18\t21\tpositive",
    fixture_id: str = "graph_coloring_00",
) -> dict[str, Any]:
    return {
        "cell_identity": f"{arm}::{fixture_id}",
        "arm": arm,
        "fixture_id": fixture_id,
        "group_id": "relation_group_00",
        "family": "graph_coloring",
        "split": split,
        "source_text_hash": exp.sha256_text("Node n0 has color red."),
        "prompt_sha256": exp.sha256_text("safe prompt"),
        "raw_output": raw_output,
        "raw_output_sha256": exp.sha256_text(raw_output),
        "parse_rows": parse_rows if parse_rows is not None else [_parse_relation()],
        "terminal": True,
        "timed_out": False,
        "truncated": False,
        "stop_reason": "complete",
    }


def _formal(*, expected_case: str = "valid") -> dict[str, Any]:
    program = "1 {gc_0_red; gc_0_blue} 1.\ngc_0_red.\n"
    return {
        "fixture_id": "graph_coloring_00",
        "group_id": "relation_group_00",
        "expected_case": expected_case,
        "asp_program": program,
        "answer_sets": [["gc_0_red"]],
        "zero_energy_states": [["gc_0_red"]],
        "solver_receipt": {
            "name_version": "clingo test",
            "error": None,
            "answer_sets_hash": exp.sha256_json([["gc_0_red"]]),
        },
    }


def _vocabulary() -> list[dict[str, Any]]:
    return [
        {
            "family": "graph_coloring",
            "subject_id": "node_0",
            "predicate": "has_color",
            "object_id": "red",
            "polarity": "positive",
            "normalized_tuple": ["node_0", "has_color", "red", "positive"],
            "asp_atom": "gc_0_red",
        },
        {
            "family": "graph_coloring",
            "subject_id": "node_0",
            "predicate": "has_color",
            "object_id": "red",
            "polarity": "negative",
            "normalized_tuple": ["node_0", "has_color", "red", "negative"],
            "asp_atom": "gc_0_not_red",
        },
    ]


def _solver(program: Any) -> list[list[str]]:
    return program and exp.asp_energy.solve_with_clingo(program)


def _score(
    cells: list[dict[str, Any]],
    *,
    formal: dict[str, Any] | None = None,
    solver: Any = _solver,
) -> dict[str, Any]:
    return exp.score_partition(
        cells=cells,
        sources=[_source(arm_split=str(cells[0]["split"]))],
        formal_rows=[formal or _formal()],
        vocabulary=_vocabulary(),
        arms=sorted({str(row["arm"]) for row in cells}),
        solver=solver,
        solver_timeout_s=0.5,
    )


def test_spec_declares_req_verify_6888_and_all_scenarios() -> None:
    """REQ-VERIFY-6888 declares the reducer contract before implementation."""

    text = Path("openspec/capabilities/constraint-verification/spec.md").read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6888") :]
    for scenario in (
        "SCENARIO-VERIFY-6888-PRECONDITIONS",
        "SCENARIO-VERIFY-6888-SEAL",
        "SCENARIO-VERIFY-6888-SPANS",
        "SCENARIO-VERIFY-6888-TUPLES",
        "SCENARIO-VERIFY-6888-DENOMINATORS",
        "SCENARIO-VERIFY-6888-SOLVER",
        "SCENARIO-VERIFY-6888-POOLING",
        "SCENARIO-VERIFY-6888-IDENTITY",
        "SCENARIO-VERIFY-6888-REPLAY",
    ):
        assert scenario in section


def test_artifact_drift_and_missing_cells_block_before_held_open() -> None:
    """SCENARIO-VERIFY-6888-PRECONDITIONS rejects drift and incomplete cells."""

    sources = [_source()]
    cells = [_cell()]
    clean = exp.validate_acquisition_matrix(cells, sources, ["arm:rule"])
    assert clean["passed"] is True

    drifted = exp.check_exact_hashes({"exp6887": "sha256:changed"}, {"exp6887": "sha256:frozen"})
    assert drifted["passed"] is False
    assert drifted["failed_check"] == "artifact_hash:exp6887"

    missing = exp.validate_acquisition_matrix([], sources, ["arm:rule"])
    assert missing["passed"] is False
    assert missing["failed_check"] == "complete_raw_cells"
    invalid = _cell()
    invalid["terminal"] = False
    content = exp.validate_acquisition_matrix([invalid], sources, ["arm:rule"])
    assert content["failed_check"] == "terminal_cell_content"

    nested = exp.gate_check("nested", {"arms": {"b", "a"}}, {"arms": {"a", "b"}})
    assert nested == {
        "check": "nested",
        "expected": {"arms": ["a", "b"]},
        "observed": {"arms": ["a", "b"]},
        "passed": True,
    }
    assert exp._jsonable_gate_value([("x",)]) == [["x"]]


def test_arm_identity_duplicate_cell_and_fixture_identity_fail_closed() -> None:
    """SCENARIO-VERIFY-6888-IDENTITY requires exact arm and record IDs."""

    sources = [_source()]
    substituted = exp.validate_acquisition_matrix([_cell(arm="arm:other")], sources, ["arm:rule"])
    assert substituted["passed"] is False
    assert substituted["failed_check"] == "arm_identity"

    duplicated = exp.validate_acquisition_matrix(
        [_cell(), deepcopy(_cell())], sources, ["arm:rule"]
    )
    assert duplicated["passed"] is False
    assert duplicated["failed_check"] == "unique_cell_identity"

    unmatched = exp.validate_acquisition_matrix(
        [_cell(fixture_id="graph_coloring_99")], sources, ["arm:rule"]
    )
    assert unmatched["passed"] is False
    assert unmatched["failed_check"] == "fixture_identity"


def test_held_sidecar_opens_once_and_formal_leakage_is_rejected(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6888-SEAL opens once and detects direct formal leakage."""

    payload = {"schema": exp.SIDECAR_SCHEMA, "split": "held", "rows": [_formal()]}
    path = tmp_path / "sealed_held.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    reader = exp.HeldSidecarReader(path, exp.sha256_file(path))
    opened = reader.open_once()
    assert opened == payload
    assert reader.open_count == 1
    with pytest.raises(exp.QualificationError, match="held_sidecar_opened_more_than_once"):
        reader.open_once()

    safe = {"prompt_manifest": {"template": "extract relations"}, "rows": [_cell()]}
    assert exp.detect_held_leakage(safe, payload) == []
    assert exp.detect_held_leakage(safe, {"rows": [None]}) == []
    leaked = deepcopy(safe)
    leaked["rows"][0]["raw_output"] = _formal()["asp_program"]
    rows = exp.detect_held_leakage(leaked, payload)
    assert rows[0]["leak_type"] == "asp_program"


def test_offset_mismatch_fails_span_credit_but_duplicate_gets_one_tuple_credit() -> None:
    """SCENARIO-VERIFY-6888-SPANS and TUPLES recompute spans and de-duplicate."""

    exact = _parse_relation()
    duplicate = deepcopy(exact)
    duplicate["line_index"] = 1
    shifted = _parse_relation(start=4, end=6)
    report = _score([_cell(parse_rows=[exact, duplicate, shifted])])
    metric = report["tuple_metric_rows"][0]
    span = report["span_metric_rows"][0]
    assert (metric["true_positive"], metric["false_positive"], metric["false_negative"]) == (
        1,
        0,
        0,
    )
    assert metric["duplicate_proposal_count"] == 2
    assert span["true_positive"] == 1
    assert span["offset_mismatch_count"] == 1
    assert any(row["outcome"] == "duplicate_no_credit" for row in report["rows"])
    assert any(row["outcome"] == "offset_mismatch" for row in report["rows"])


def test_abstentions_and_malformed_cells_remain_in_all_denominators() -> None:
    """SCENARIO-VERIFY-6888-DENOMINATORS preserves nulls and failed cells."""

    empty = _cell(
        arm="arm:empty",
        raw_output="",
        parse_rows=[{"line_index": 0, "raw_line": "", "status": "empty", "reason": "empty"}],
    )
    malformed = _cell(
        arm="arm:bad",
        raw_output="bad",
        parse_rows=[{"line_index": 0, "raw_line": "bad", "status": "malformed", "reason": "bad"}],
    )
    report = _score([empty, malformed])
    coverage = {row["arm"]: row for row in report["parse_coverage_rows"]}
    abstention = {row["arm"]: row for row in report["abstention_rows"]}
    tuples = {row["arm"]: row for row in report["tuple_metric_rows"]}
    assert coverage["arm:empty"]["cell_count"] == 1
    assert coverage["arm:bad"]["cell_count"] == 1
    assert coverage["arm:empty"]["parse_coverage"] == 0.0
    assert abstention["arm:empty"]["abstention_count"] == 1
    assert tuples["arm:empty"]["precision"] is None
    assert tuples["arm:bad"]["false_negative"] == 1


def test_unsupported_atom_and_solver_timeout_are_explicit_failures() -> None:
    """SCENARIO-VERIFY-6888-SOLVER fails closed on mapping and solver errors."""

    unsupported = _parse_relation(predicate="unknown_relation")
    report = _score([_cell(parse_rows=[unsupported])])
    assert report["asp_compilation_rows"][0]["status"] == "unsupported_atom"
    assert report["solver_parity_rows"][0]["parity"] is False

    def timeout_solver(_program: Any) -> list[list[str]]:
        raise TimeoutError("bounded timeout")

    timed = _score([_cell()], solver=timeout_solver)
    assert timed["solver_parity_rows"][0]["status"] == "timeout"
    assert timed["independent_solver_receipts"]["timeout_count"] == 1


def test_calibration_freezes_thresholds_and_family_pooling_cannot_hide_failure() -> None:
    """SCENARIO-VERIFY-6888-POOLING uses calibration and minimum subgroup rows."""

    calibration = {
        "span_metric_rows": [{"arm": "arm:a", "f1": 0.8}],
        "tuple_metric_rows": [{"arm": "arm:a", "precision": 0.75, "recall": 0.9, "f1": 0.81}],
        "parse_coverage_rows": [{"arm": "arm:a", "parse_coverage": 0.85}],
        "family_rows": [
            {"arm": "arm:a", "family": "easy", "f1": 0.9},
            {"arm": "arm:a", "family": "hard", "f1": 0.7},
        ],
        "perturbation_rows": [{"arm": "arm:a", "perturbation": "base", "f1": 0.65}],
        "solver_parity_rows": [{"arm": "arm:a", "parity": True}],
    }
    thresholds = exp.freeze_thresholds(calibration)
    assert thresholds["reference_arm"] == "arm:a"
    assert thresholds["minimum_family_floor"] == 0.7
    assert thresholds["minimum_perturbation_floor"] == 0.65
    assert thresholds["exact_semantic_parity"] == 1.0

    held = deepcopy(calibration)
    held["family_rows"][0]["f1"] = 0.99
    held["family_rows"][1]["f1"] = 0.69
    held["tuple_metric_rows"][0].update({"precision": 0.99, "recall": 0.99})
    eligible = exp.evaluate_eligible_arms(held, thresholds, {"arm:a": 100})
    assert eligible["eligible_arm_rows"][0]["passed"] is False
    assert "family_floor" in eligible["eligible_arm_rows"][0]["failed_thresholds"]
    assert eligible["relation_qualification_ready_score"] == 0


def test_aggregate_row_disagreement_and_positive_verdict_are_invalid() -> None:
    """SCENARIO-VERIFY-6888-REPLAY rejects aggregate drift and positive class."""

    artifact = exp.blocked_artifact(
        date="20260902",
        duration_s=0.1,
        checks=[exp.gate_check("fixture", True, False)],
        source_artifact_hashes={},
        sealed_sidecar_hashes={},
    )
    assert exp.validate_artifact(artifact) == []
    drifted = deepcopy(artifact)
    drifted["reported_vs_recomputed_metrics"] = {
        "reported_ready_score": 1,
        "recomputed_ready_score": 0,
        "agreement": False,
    }
    drifted["relation_qualification_ready_score"] = 1
    assert "aggregate_vs_row_disagreement" in exp.validate_artifact(drifted)
    drifted["verdict_class"] = "positive"
    assert "verdict_class" in exp.validate_artifact(drifted)


def test_build_artifact_opens_held_after_freeze_and_counts_qualified_events(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6888 runs the fresh reduction without model inference or repair."""

    calibration_payload = {
        "schema": exp.SIDECAR_SCHEMA,
        "split": "calibration",
        "rows": [_formal()],
    }
    held_payload = {
        "schema": exp.SIDECAR_SCHEMA,
        "split": "held",
        "rows": [_formal()],
    }
    held_path = tmp_path / "held.json"
    held_path.write_text(json.dumps(held_payload), encoding="utf-8")
    reader = exp.HeldSidecarReader(held_path, exp.sha256_file(held_path))
    proposal = {
        "prompt_manifest": {"template": "extract relations"},
        "rows": [
            _cell(split="calibration"),
            {**_cell(split="held"), "cell_identity": "arm:rule::graph_coloring_00"},
        ],
    }
    sources = [_source(), _source(arm_split="held")]
    artifact = exp.reduce_frozen_outputs(
        date="20260902",
        proposal_artifact=proposal,
        sources=sources,
        calibration_sidecar=calibration_payload,
        held_reader=reader,
        vocabulary=_vocabulary(),
        arms=["arm:rule"],
        event_count_by_arm={"arm:rule": 90},
        source_artifact_hashes={"exp6887": "sha256:test"},
        sealed_sidecar_hashes={"held": exp.sha256_file(held_path)},
        solver=_solver,
        solver_timeout_s=0.5,
        duration_s=0.2,
    )
    assert reader.open_count == 1
    assert artifact["frozen_thresholds"]["reference_arm"] == "arm:rule"
    assert artifact["relation_qualification_ready_score"] == 1
    assert artifact["qualified_relation_event_count"] == 90
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["held_leakage_count"] == 0
    assert exp.validate_artifact(artifact) == []
    aggregate_drift = deepcopy(artifact)
    aggregate_drift["eligible_arm_rows"][0]["qualified_event_count"] = 0
    aggregate_drift["reproducibility_checksum"] = exp._artifact_checksum(aggregate_drift)
    assert "aggregate_vs_row_disagreement" in exp.validate_artifact(aggregate_drift)


def test_sidecar_and_mapping_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-6888 rejects sealed, family, offset, and row-shape drift."""

    path = tmp_path / "held.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(exp.QualificationError, match="held_sidecar_hash_drift"):
        exp.HeldSidecarReader(path, "sha256:wrong").open_once()
    with pytest.raises(exp.QualificationError, match="held_sidecar_identity"):
        exp.HeldSidecarReader(path, exp.sha256_file(path)).open_once()

    empty = {"schema": exp.SIDECAR_SCHEMA, "split": "held", "rows": []}
    path.write_text(json.dumps(empty), encoding="utf-8")
    with pytest.raises(exp.QualificationError, match="held_sidecar_rows"):
        exp.HeldSidecarReader(path, exp.sha256_file(path)).open_once()
    with pytest.raises(exp.QualificationError, match="unsupported_family"):
        exp._surface_ids("unknown_family_0")

    assert exp._span_matches("é", {"start_utf8": True, "end_utf8": 2}, "é") is False
    assert exp._span_matches("é", {"start_utf8": 1, "end_utf8": 1}, "é") is False
    assert exp._span_matches("é", {"start_utf8": 0, "end_utf8": 3}, "é") is False
    assert exp._span_matches("é", {"start_utf8": 0, "end_utf8": 1}, "é") is False
    malformed = exp._map_proposal("graph_coloring_00", "Node n0 has color red.", {}, _vocabulary())
    assert malformed["reason"] == "malformed_relation"
    assert exp._metrics(0, 0, 0)["f1"] == 1.0


def test_unmatched_rows_and_compile_errors_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-6888-SOLVER preserves unmatched and compile failures."""

    with pytest.raises(exp.QualificationError, match="unmatched_fixture"):
        exp.score_partition(
            cells=[_cell()],
            sources=[],
            formal_rows=[_formal()],
            vocabulary=_vocabulary(),
            arms=["arm:rule"],
            solver=_solver,
            solver_timeout_s=0.5,
        )

    malformed = _cell(parse_rows=[None])
    malformed_report = _score([malformed])
    assert malformed_report["rows"][0]["outcome"] == "malformed"

    def compile_error(*_args: Any, **_kwargs: Any) -> Any:
        raise exp.asp_energy.UnsupportedASPSyntax("aggregate", "bad")

    monkeypatch.setattr(exp.asp_energy, "compile_program", compile_error)
    report = _score([_cell()])
    assert report["asp_compilation_rows"][0]["status"] == "compile_error"
    assert report["solver_parity_rows"][0]["status"] == "not_run_compile_error"


def test_freeze_and_reducer_order_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-6888 requires calibration evidence before any held open."""

    with pytest.raises(exp.QualificationError, match="calibration_metrics_missing"):
        exp.freeze_thresholds({})

    held = {"schema": exp.SIDECAR_SCHEMA, "split": "held", "rows": [_formal()]}
    held_path = tmp_path / "held.json"
    held_path.write_text(json.dumps(held), encoding="utf-8")
    reader = exp.HeldSidecarReader(held_path, exp.sha256_file(held_path))
    reader.open_once()
    with pytest.raises(exp.QualificationError, match="held_opened_before_threshold_freeze"):
        exp.reduce_frozen_outputs(
            date="20260902",
            proposal_artifact={"rows": [_cell()]},
            sources=[_source()],
            calibration_sidecar={
                "schema": exp.SIDECAR_SCHEMA,
                "split": "calibration",
                "rows": [_formal()],
            },
            held_reader=reader,
            vocabulary=_vocabulary(),
            arms=["arm:rule"],
            event_count_by_arm={"arm:rule": 1},
            source_artifact_hashes={},
            sealed_sidecar_hashes={},
            solver=_solver,
            solver_timeout_s=0.5,
            duration_s=0.1,
        )


def _reduce_variant(
    tmp_path: Path,
    *,
    held_cell: dict[str, Any],
) -> dict[str, Any]:
    held = {"schema": exp.SIDECAR_SCHEMA, "split": "held", "rows": [_formal()]}
    held_path = tmp_path / (held_cell["arm"].replace(":", "_") + ".json")
    held_path.write_text(json.dumps(held), encoding="utf-8")
    return exp.reduce_frozen_outputs(
        date="20260902",
        proposal_artifact={
            "prompt_manifest": {"template": "extract relations"},
            "rows": [_cell(), held_cell],
        },
        sources=[_source(), _source(arm_split="held")],
        calibration_sidecar={
            "schema": exp.SIDECAR_SCHEMA,
            "split": "calibration",
            "rows": [_formal()],
        },
        held_reader=exp.HeldSidecarReader(held_path, exp.sha256_file(held_path)),
        vocabulary=_vocabulary(),
        arms=["arm:rule"],
        event_count_by_arm={"arm:rule": 90},
        source_artifact_hashes={},
        sealed_sidecar_hashes={},
        solver=_solver,
        solver_timeout_s=0.5,
        duration_s=0.1,
    )


def test_leakage_disqualifies_and_failed_held_metrics_return_null(tmp_path: Path) -> None:
    """REQ-VERIFY-6888 keeps disqualified and null verdicts terminal."""

    leaked = _cell(split="held", raw_output=_formal()["asp_program"])
    disqualified = _reduce_variant(tmp_path, held_cell=leaked)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["relation_qualification_ready_score"] == 0

    malformed = _cell(
        split="held",
        raw_output="bad",
        parse_rows=[{"status": "malformed", "reason": "bad"}],
    )
    null = _reduce_variant(tmp_path, held_cell=malformed)
    assert null["verdict_class"] == "null"
    assert null["honest_verdict"] == "complete_null_no_relation_arm_passed_frozen_thresholds"


def test_artifact_validator_reports_each_schema_boundary() -> None:
    """SCENARIO-VERIFY-6888-REPLAY reports every terminal schema defect."""

    clean = exp.blocked_artifact(
        date="20260902",
        duration_s=0.1,
        checks=[exp.gate_check("fixture", True, False)],
        source_artifact_hashes={},
        sealed_sidecar_hashes={},
    )
    variants: list[tuple[dict[str, Any], str]] = []
    missing = deepcopy(clean)
    missing.pop("rows")
    variants.append((missing, "required_fields"))
    no_principles = deepcopy(clean)
    no_principles["field_principles"] = {}
    variants.append((no_principles, "field_principles"))
    bad_substrate = deepcopy(clean)
    bad_substrate["inference_substrate"] = "model_inference"
    variants.append((bad_substrate, "inference_substrate"))
    no_oracle = deepcopy(clean)
    no_oracle["verifier_is_oracle"] = False
    variants.append((no_oracle, "verifier_is_oracle"))
    unfinished = deepcopy(clean)
    unfinished["honest_verdict"] = "blocked"
    variants.append((unfinished, "honest_verdict"))
    no_report = deepcopy(clean)
    no_report["reported_vs_recomputed_metrics"] = None
    variants.append((no_report, "reported_vs_recomputed_metrics"))
    bad_row = deepcopy(clean)
    bad_row["rows"] = [{"arm": "a"}]
    variants.append((bad_row, "row_schema"))
    checksum = deepcopy(clean)
    checksum["reproducibility_checksum"] = "sha256:bad"
    variants.append((checksum, "reproducibility_checksum"))
    for artifact, expected in variants:
        assert any(error.startswith(expected) for error in exp.validate_artifact(artifact))


def _configure_run_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    calibration_identity_valid: bool = True,
    held_identity_valid: bool = True,
) -> None:
    paths = {
        "exp6274": Path("compiler.json"),
        "exp6886": Path("fixture.json"),
        "exp6887": Path("proposal.json"),
        "compiler": Path("asp_energy.py"),
    }
    source_rows = [
        _source(),
        {
            **_source(
                arm_split="held",
                fixture_id="graph_coloring_01",
                source_text="Node n1 has color red.",
            ),
            "group_id": "relation_group_01",
        },
    ]
    held_parse = _parse_relation()
    held_parse["subject"] = {"start_utf8": 5, "end_utf8": 7, "text": "n1"}
    held_parse["normalized_tuple"] = ["n1", "has_color", "red", "positive"]
    held_cell = _cell(split="held", parse_rows=[held_parse], fixture_id="graph_coloring_01")
    held_cell["group_id"] = "relation_group_01"
    held_cell["source_text_hash"] = source_rows[1]["source_text_hash"]
    held_cell["cell_identity"] = "arm:rule::graph_coloring_01"

    proposal = {
        "relation_corpus_complete_score": 1,
        "held_sidecar_access_count": 0,
        "prompt_manifest": {"template": "extract relations"},
        "rows": [_cell(), held_cell],
    }
    vocabulary = [
        *_vocabulary(),
        {
            "family": "graph_coloring",
            "subject_id": "node_1",
            "predicate": "has_color",
            "object_id": "red",
            "polarity": "positive",
            "normalized_tuple": ["node_1", "has_color", "red", "positive"],
            "asp_atom": "gc_1_red",
        },
    ]
    fixture = {
        "relation_fixture_ready_score": 1,
        "closed_vocabulary_manifest": {"entries": vocabulary},
    }
    compiler = {"asp_energy_semantic_ready_score": 1.0, "parity_failure_count": 0}
    formal_0 = _formal()
    formal_1 = {
        **_formal(),
        "fixture_id": "graph_coloring_01",
        "group_id": "relation_group_01",
        "asp_program": "1 {gc_1_red; gc_1_blue} 1.\ngc_1_red.\n",
        "answer_sets": [["gc_1_red"]],
        "zero_energy_states": [["gc_1_red"]],
    }
    calibration = {
        "schema": exp.SIDECAR_SCHEMA if calibration_identity_valid else "bad",
        "split": "calibration",
        "rows": [formal_0],
    }
    held = {
        "schema": exp.SIDECAR_SCHEMA if held_identity_valid else "bad",
        "split": "held",
        "rows": [formal_1],
    }
    payloads = {"exp6274": compiler, "exp6886": fixture, "exp6887": proposal}
    for name, payload in payloads.items():
        (tmp_path / paths[name]).write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / paths["compiler"]).write_text("# compiler fixture\n", encoding="utf-8")
    calibration_path = tmp_path / "calibration.json"
    held_path = tmp_path / "held.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    held_path.write_text(json.dumps(held), encoding="utf-8")

    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp, "EXP6274_PATH", paths["exp6274"])
    monkeypatch.setattr(exp, "EXP6886_PATH", paths["exp6886"])
    monkeypatch.setattr(exp, "EXP6887_PATH", paths["exp6887"])
    monkeypatch.setattr(exp, "COMPILER_PATH", paths["compiler"])
    monkeypatch.setattr(exp, "CALIBRATION_SIDECAR_PATH", calibration_path)
    monkeypatch.setattr(exp, "HELD_SIDECAR_PATH", held_path)
    monkeypatch.setattr(exp, "PROPOSAL_ARMS", ("arm:rule",))
    monkeypatch.setattr(exp, "build_frozen_source_records", lambda: source_rows)
    expected = {name: exp.sha256_file(tmp_path / path) for name, path in paths.items()}
    expected["calibration_sidecar"] = exp.sha256_file(calibration_path)
    expected["held_sidecar"] = exp.sha256_file(held_path)
    monkeypatch.setattr(exp, "EXPECTED_HASHES", expected)


def test_run_experiment_writes_complete_and_blocked_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-VERIFY-6888 writes stable terminal output for pass and precondition block."""

    _configure_run_inputs(monkeypatch, tmp_path)
    output = tmp_path / "complete.json"
    artifact = exp.run_experiment(date="20260902", output_path=output)
    assert artifact["verdict_class"] == "circular_positive"
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    checked = artifact["preconditions_checked"]["gate_check_summary"]
    assert checked["passed"] is True
    assert any(row["check"] == "relation_corpus_complete_score" for row in checked["checks"])
    assert any(
        row["check"] == "artifact_hash:exp6887" for row in artifact["gate_check_summary"]["checks"]
    )

    exp.EXPECTED_HASHES["exp6274"] = "sha256:drift"
    blocked = exp.run_experiment(date="20260902", output_path=tmp_path / "blocked.json")
    assert blocked["honest_verdict"] == "complete_blocked_independent_relation_qualification"


def test_run_experiment_blocks_bad_sidecars_and_rejects_invalid_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-VERIFY-6888 blocks sidecar identity drift and refuses invalid terminals."""

    _configure_run_inputs(monkeypatch, tmp_path, calibration_identity_valid=False)
    calibration = exp.run_experiment(date="20260902", output_path=tmp_path / "cal_bad.json")
    assert calibration["gate_check_summary"]["failed_check"] == "calibration_sidecar_identity"

    monkeypatch.undo()
    held_root = tmp_path / "held_case"
    held_root.mkdir()
    _configure_run_inputs(monkeypatch, held_root, held_identity_valid=False)
    held = exp.run_experiment(date="20260902", output_path=held_root / "held_bad.json")
    assert held["gate_check_summary"]["failed_check"] == "held_sidecar"

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced"])
    with pytest.raises(exp.QualificationError, match="artifact_validation:forced"):
        exp.run_experiment(date="20260902", output_path=held_root / "invalid.json")


def test_main_prints_terminal_artifact(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    """REQ-VERIFY-6888 exposes the fresh reducer through its required CLI."""

    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda **_kwargs: {"honest_verdict": "complete_null_fixture"},
    )
    assert exp.main(["--date", "20260902", "--output", "/tmp/exp6888-test.json"]) == 0
    assert "complete_null_fixture" in capsys.readouterr().out
