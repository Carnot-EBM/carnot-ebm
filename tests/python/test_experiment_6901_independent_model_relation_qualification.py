"""Tests for independent model-only relation qualification.

Spec refs: REQ-VERIFY-6901 and SCENARIO-VERIFY-6901-*.
"""

from __future__ import annotations

from copy import deepcopy
import base64
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6901_independent_model_relation_qualification as exp


MODEL_ARM = "gguf:model-a"
RULE_ARM = "rule:anchored_lexical_v1"


def _source(*, split: str = "calibration") -> dict[str, Any]:
    text = "Node n0 has color red."
    return {
        "fixture_id": "graph_coloring_00",
        "group_id": "relation_group_00",
        "family": "graph_coloring",
        "split": split,
        "source_order": 0,
        "source_text": text,
        "source_text_hash": exp.sha256_text(text),
    }


def _relation(
    *,
    start: int = 5,
    end: int = 7,
    predicate: str = "has_color",
    status: str = "accepted",
) -> dict[str, Any]:
    return {
        "line_index": 0,
        "raw_line": "relation",
        "status": status,
        "reason": "accepted" if status == "accepted" else "fixture_reason",
        "subject": {"start_utf8": start, "end_utf8": end, "text": "n0"},
        "predicate": predicate,
        "object": {"start_utf8": 18, "end_utf8": 21, "text": "red"},
        "polarity": "positive",
    }


def _cell(
    *,
    arm: str = MODEL_ARM,
    split: str = "calibration",
    seed: int | None = 7,
    parse_rows: list[Any] | None = None,
    timed_out: bool = False,
) -> dict[str, Any]:
    model_id = arm.removeprefix("gguf:") if arm.startswith("gguf:") else None
    identity_seed = str(seed) if arm.startswith("gguf:") else "deterministic"
    raw = "relation"
    return {
        "cell_identity": f"{model_id or arm}::{identity_seed}::graph_coloring_00",
        "arm": arm,
        "hf_id": model_id,
        "model_family": "gguf" if model_id else "lexical_rule",
        "seed": seed if model_id else None,
        "fixture_id": "graph_coloring_00",
        "group_id": "relation_group_00",
        "family": "graph_coloring",
        "split": split,
        "source_text_hash": exp.sha256_text("Node n0 has color red."),
        "raw_output": raw,
        "raw_output_sha256": exp.sha256_text(raw),
        "parse_rows": [_relation()] if parse_rows is None else parse_rows,
        "terminal": True,
        "timed_out": timed_out,
        "truncated": False,
        "stop_reason": "complete",
    }


def _formal(*, expected_case: str = "valid") -> dict[str, Any]:
    return {
        "fixture_id": "graph_coloring_00",
        "group_id": "relation_group_00",
        "expected_case": expected_case,
        "asp_program": "1 {gc_0_red; gc_0_blue} 1.\ngc_0_red.\n",
        "answer_sets": [["gc_0_red"]],
        "zero_energy_states": [["gc_0_red"]],
        "solver_receipt": {"name_version": "clingo test", "error": None},
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
        }
    ]


def _solver(_program: Any) -> list[list[str]]:
    return [["gc_0_red"]]


def _score(cells: list[dict[str, Any]], solver: Any = _solver) -> dict[str, Any]:
    split = str(cells[0]["split"])
    return exp.score_partition(
        cells=cells,
        sources=[_source(split=split)],
        formal_rows=[_formal()],
        vocabulary=_vocabulary(),
        arms=sorted({str(row["arm"]) for row in cells}),
        solver=solver,
        solver_timeout_s=0.1,
    )


def test_req_verify_6901_spec_declares_all_failure_boundaries() -> None:
    """REQ-VERIFY-6901 owns the contract before implementation exists."""

    text = Path("openspec/capabilities/constraint-verification/spec.md").read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6901") :]
    for suffix in (
        "ADMISSION",
        "SEAL",
        "IDENTITY",
        "SPANS",
        "TUPLES",
        "DENOMINATORS",
        "SOLVER",
        "POOLING",
        "MODEL-ONLY",
        "REPLAY",
    ):
        assert f"SCENARIO-VERIFY-6901-{suffix}" in section


def test_source_drift_flagged_admission_and_missing_cells_block() -> None:
    """SCENARIO-VERIFY-6901-ADMISSION rejects drift, flags, and missing cells."""

    drift = exp.check_exact_hashes({"exp6900": "changed"}, {"exp6900": "frozen"})
    assert drift["passed"] is False
    assert drift["failed_check"] == "artifact_hash:exp6900"

    rows = exp.adversarial_admission_rows(
        {"flagged_adversarial": True},
        {"loaded": True, "flags": [{"kind": "TAUTOLOGY", "severity": "critical"}]},
    )
    assert [row["passed"] for row in rows] == [False, False, True]
    assert rows[1]["observed"] == 1

    sources = [_source()]
    assert exp.validate_acquisition_matrix([_cell()], sources, [MODEL_ARM], [7])["passed"]
    missing = exp.validate_acquisition_matrix([], sources, [MODEL_ARM], [7])
    assert missing["passed"] is False
    assert missing["failed_check"] == "complete_terminal_cells"
    nonterminal = _cell()
    nonterminal["terminal"] = False
    assert (
        exp.validate_acquisition_matrix([nonterminal], sources, [MODEL_ARM], [7])["failed_check"]
        == "terminal_cell_content"
    )


def test_record_arm_model_and_seed_identity_must_match() -> None:
    """SCENARIO-VERIFY-6901-IDENTITY rejects substituted model and seed IDs."""

    sources = [_source()]
    wrong_model = _cell()
    wrong_model["hf_id"] = "model-b"
    report = exp.validate_acquisition_matrix([wrong_model], sources, [MODEL_ARM], [7])
    assert report["passed"] is False
    assert report["failed_check"] == "record_arm_model_seed_identity"

    wrong_seed = _cell()
    wrong_seed["seed"] = 8
    report = exp.validate_acquisition_matrix([wrong_seed], sources, [MODEL_ARM], [7])
    assert report["failed_check"] == "record_arm_model_seed_identity"


def test_held_sidecar_opens_once_and_formal_leakage_is_refused(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6901-SEAL opens held authority once and rejects leakage."""

    payload = {"schema": exp.SIDECAR_SCHEMA, "split": "held", "rows": [_formal()]}
    path = tmp_path / "held.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    reader = exp.HeldSidecarReader(path, exp.sha256_file(path))
    assert reader.open_once() == payload
    with pytest.raises(exp.QualificationError, match="held_sidecar_opened_more_than_once"):
        reader.open_once()

    proposal = {"prompt_manifest": {"text": "extract"}, "cell_manifest": [_cell()]}
    assert exp.detect_held_leakage(proposal, payload) == []
    proposal["prompt_manifest"]["text"] = _formal()["asp_program"]
    assert exp.detect_held_leakage(proposal, payload)[0]["leak_type"] == "asp_program"
    encoded = deepcopy(proposal)
    encoded["prompt_manifest"] = {"nested": ["safe"]}
    encoded["cell_manifest"][0].pop("raw_output")
    encoded["cell_manifest"][0]["raw_output_b64"] = base64.b64encode(
        _formal()["asp_program"].encode()
    ).decode()
    assert exp.detect_held_leakage(encoded, payload)
    encoded["cell_manifest"][0]["raw_output_b64"] = "not-base64"
    assert exp._raw_output(encoded["cell_manifest"][0]) == ""


def test_offsets_duplicates_and_exact_checks_remain_per_relation() -> None:
    """SCENARIO-VERIFY-6901-SPANS and TUPLES recompute spans and duplicate credit."""

    exact = _relation()
    duplicate = deepcopy(exact)
    duplicate["line_index"] = 1
    shifted = _relation(start=4, end=6)
    report = _score([_cell(parse_rows=[exact, duplicate, shifted])])
    tuples = report["tuple_metric_rows"][0]
    spans = report["span_metric_rows"][0]
    assert tuples["true_positive"] == 1
    assert tuples["duplicate_proposal_count"] == 2
    assert spans["true_positive"] == 1
    assert spans["offset_mismatch_count"] == 1
    assert all(
        {"arm", "record_id", "relation", "family", "perturbation", "exact_check"} <= set(row)
        for row in report["rows"]
    )
    assert report["solver_parity_rows"][0]["exact_validity"] is True

    no_headroom = _formal(expected_case="abstain")
    no_headroom["asp_program"] = "1 {gc_0_red; gc_0_blue} 1.\n"
    no_headroom["answer_sets"] = [["gc_0_blue"], ["gc_0_red"]]
    report = exp.score_partition(
        cells=[_cell(parse_rows=[])],
        sources=[_source()],
        formal_rows=[no_headroom],
        vocabulary=_vocabulary(),
        arms=[MODEL_ARM],
        solver=lambda _program: [["gc_0_blue"], ["gc_0_red"]],
        solver_timeout_s=0.1,
    )
    assert report["rows"][0]["outcome"] == "missing_proposal"


def test_abstention_malformed_and_missing_proposals_stay_in_denominators() -> None:
    """SCENARIO-VERIFY-6901-DENOMINATORS preserves every failed proposal cell."""

    empty = _cell(
        arm=RULE_ARM,
        seed=None,
        parse_rows=[{"status": "empty", "reason": "empty"}],
    )
    malformed = _cell(parse_rows=[None])
    report = _score([empty, malformed])
    coverage = {row["arm"]: row for row in report["parse_coverage_rows"]}
    abstention = {row["arm"]: row for row in report["abstention_rows"]}
    assert coverage[MODEL_ARM]["cell_count"] == 1
    assert coverage[RULE_ARM]["parse_coverage"] == 0.0
    assert abstention[RULE_ARM]["abstention_cost"] == 1.0
    assert any(row["outcome"] == "malformed" for row in report["rows"])
    assert report["completeness_blind_spot_rows"]


def test_unsupported_atoms_and_solver_timeouts_fail_exact_validity() -> None:
    """SCENARIO-VERIFY-6901-SOLVER keeps unsupported atoms and timeout receipts."""

    unsupported = _score([_cell(parse_rows=[_relation(predicate="unknown")])])
    assert unsupported["asp_compilation_rows"][0]["status"] == "unsupported_atom"
    assert unsupported["solver_parity_rows"][0]["exact_validity"] is False

    def timeout(_program: Any) -> list[list[str]]:
        raise TimeoutError("bounded timeout")

    timed = _score([_cell(timed_out=True)], solver=timeout)
    assert timed["solver_parity_rows"][0]["status"] == "timeout"
    assert timed["independent_solver_receipts"]["timeout_count"] == 1


def test_calibration_pooling_abstention_and_rule_only_readiness() -> None:
    """SCENARIO-VERIFY-6901-POOLING and MODEL-ONLY keep weak slices and rules out."""

    calibration = {
        "span_metric_rows": [{"arm": RULE_ARM, "f1": 0.8}],
        "tuple_metric_rows": [{"arm": RULE_ARM, "precision": 0.75, "recall": 0.9, "f1": 0.81}],
        "parse_coverage_rows": [{"arm": RULE_ARM, "parse_coverage": 0.85}],
        "abstention_rows": [{"arm": RULE_ARM, "abstention_cost": 0.15}],
        "family_rows": [
            {"arm": RULE_ARM, "family": "easy", "f1": 0.9},
            {"arm": RULE_ARM, "family": "hard", "f1": 0.7},
        ],
        "perturbation_rows": [{"arm": RULE_ARM, "perturbation": "base", "f1": 0.65}],
        "solver_parity_rows": [{"arm": RULE_ARM, "exact_validity": True}],
    }
    thresholds = exp.freeze_thresholds(calibration)
    assert thresholds["minimum_family_floor"] == 0.7
    assert thresholds["minimum_perturbation_floor"] == 0.65
    assert thresholds["maximum_abstention_cost"] == 0.15
    with pytest.raises(exp.QualificationError, match="calibration_metrics_missing"):
        exp.freeze_thresholds({})

    held = deepcopy(calibration)
    eligibility = exp.evaluate_model_readiness(
        held, thresholds, {RULE_ARM: 200}, model_arms=[MODEL_ARM]
    )
    assert eligibility["rule_control_rows"][0]["threshold_passed"] is True
    assert eligibility["model_eligible_arm_rows"] == []
    assert eligibility["qualified_model_relation_event_count"] == 0
    assert eligibility["model_relation_qualification_ready_score"] == 0

    model_held = deepcopy(held)
    for rows in model_held.values():
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    row["arm"] = MODEL_ARM
    model_held["family_rows"].append({"arm": MODEL_ARM, "family": "weak", "f1": 0.69})
    weak = exp.evaluate_model_readiness(
        model_held, thresholds, {MODEL_ARM: 100}, model_arms=[MODEL_ARM]
    )
    assert "family_floor" in weak["model_eligible_arm_rows"][0]["failed_thresholds"]
    model_held["family_rows"].pop()
    model_held["abstention_rows"] = []
    insufficient = exp.evaluate_model_readiness(
        model_held, thresholds, {MODEL_ARM: 0}, model_arms=[MODEL_ARM]
    )
    failures = insufficient["model_eligible_arm_rows"][0]["failed_thresholds"]
    assert "abstention_cost" in failures
    assert "minimum_exact_admitted_events" in failures


def test_clean_reduction_counts_only_exact_admitted_model_events(tmp_path: Path) -> None:
    """REQ-VERIFY-6901 opens held after calibration and applies the 90-event model floor."""

    held = {"schema": exp.SIDECAR_SCHEMA, "split": "held", "rows": [_formal()]}
    held_path = tmp_path / "held.json"
    held_path.write_text(json.dumps(held), encoding="utf-8")
    proposal = {
        "prompt_manifest": {"text": "extract relations"},
        "cell_manifest": [_cell(), _cell(split="held")],
    }
    artifact = exp.reduce_frozen_outputs(
        date="20260902",
        proposal_artifact=proposal,
        sources=[_source(), _source(split="held")],
        calibration_sidecar={
            "schema": exp.SIDECAR_SCHEMA,
            "split": "calibration",
            "rows": [_formal()],
        },
        held_reader=exp.HeldSidecarReader(held_path, exp.sha256_file(held_path)),
        vocabulary=_vocabulary(),
        arms=[MODEL_ARM],
        model_arms=[MODEL_ARM],
        source_artifact_hashes={"exp6900": {"sha256": "test"}},
        sealed_sidecar_hashes={"held": {"sha256": exp.sha256_file(held_path)}},
        adversarial_admission_rows=[],
        solver=_solver,
        solver_timeout_s=0.1,
        duration_s=0.1,
        minimum_model_events=1,
    )
    assert artifact["model_relation_qualification_ready_score"] == 1
    assert artifact["qualified_model_relation_event_count"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["held_leakage_count"] == 0
    assert exp.validate_artifact(artifact) == []

    leaked_proposal = deepcopy(proposal)
    leaked_proposal["prompt_manifest"]["text"] = _formal()["asp_program"]
    disqualified = exp.reduce_frozen_outputs(
        date="20260902",
        proposal_artifact=leaked_proposal,
        sources=[_source(), _source(split="held")],
        calibration_sidecar={
            "schema": exp.SIDECAR_SCHEMA,
            "split": "calibration",
            "rows": [_formal()],
        },
        held_reader=exp.HeldSidecarReader(held_path, exp.sha256_file(held_path)),
        vocabulary=_vocabulary(),
        arms=[MODEL_ARM],
        model_arms=[MODEL_ARM],
        source_artifact_hashes={},
        sealed_sidecar_hashes={},
        adversarial_admission_rows=[],
        solver=_solver,
        solver_timeout_s=0.1,
        duration_s=0.1,
        minimum_model_events=1,
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["model_eligible_arm_rows"][0]["passed"] is False

    opened_reader = exp.HeldSidecarReader(held_path, exp.sha256_file(held_path))
    opened_reader.open_once()
    with pytest.raises(exp.QualificationError, match="held_opened_before_threshold_freeze"):
        exp.reduce_frozen_outputs(
            date="20260902",
            proposal_artifact=proposal,
            sources=[_source(), _source(split="held")],
            calibration_sidecar={
                "schema": exp.SIDECAR_SCHEMA,
                "split": "calibration",
                "rows": [_formal()],
            },
            held_reader=opened_reader,
            vocabulary=_vocabulary(),
            arms=[MODEL_ARM],
            model_arms=[MODEL_ARM],
            source_artifact_hashes={},
            sealed_sidecar_hashes={},
            adversarial_admission_rows=[],
            solver=_solver,
            solver_timeout_s=0.1,
            duration_s=0.1,
        )


def test_blocked_artifact_is_complete_and_aggregate_drift_is_rejected() -> None:
    """SCENARIO-VERIFY-6901-REPLAY rejects aggregate-versus-row disagreement."""

    artifact = exp.blocked_artifact(
        date="20260902",
        duration_s=0.1,
        checks=[exp.gate_check("unflagged_exp6900", False, True)],
        source_artifact_hashes={},
        sealed_sidecar_hashes={},
        adversarial_admission_rows=[],
    )
    assert artifact["honest_verdict"] == (
        "complete_blocked_independent_model_relation_qualification"
    )
    assert artifact["preconditions_checked"]["held_sidecar_open_count"] == 0
    assert exp.validate_artifact(artifact) == []

    drifted = deepcopy(artifact)
    drifted["reported_vs_recomputed_metrics"]["reported_ready_score"] = 1
    drifted["reproducibility_checksum"] = exp._artifact_checksum(drifted)
    assert "aggregate_vs_row_disagreement" in exp.validate_artifact(drifted)
    positive = deepcopy(artifact)
    positive["verdict_class"] = "positive"
    positive["reproducibility_checksum"] = exp._artifact_checksum(positive)
    assert "verdict_class" in exp.validate_artifact(positive)

    variants: list[tuple[dict[str, Any], str]] = []
    missing = deepcopy(artifact)
    missing.pop("rows")
    variants.append((missing, "required_fields"))
    no_principles = deepcopy(artifact)
    no_principles["field_principles"] = {}
    variants.append((no_principles, "field_principles"))
    substrate = deepcopy(artifact)
    substrate["inference_substrate"] = "live_llm_inference"
    variants.append((substrate, "inference_substrate"))
    oracle = deepcopy(artifact)
    oracle["verifier_is_oracle"] = False
    variants.append((oracle, "verifier_is_oracle"))
    verdict = deepcopy(artifact)
    verdict["honest_verdict"] = "blocked"
    variants.append((verdict, "honest_verdict"))
    report = deepcopy(artifact)
    report["reported_vs_recomputed_metrics"] = None
    variants.append((report, "reported_vs_recomputed_metrics"))
    row = deepcopy(artifact)
    row["rows"] = [{"arm": MODEL_ARM}]
    variants.append((row, "row_schema"))
    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "bad"
    variants.append((checksum, "reproducibility_checksum"))
    for candidate, expected in variants:
        assert any(error.startswith(expected) for error in exp.validate_artifact(candidate))

    recomputed = deepcopy(artifact)
    recomputed["verdict_class"] = "null"
    recomputed["frozen_thresholds"] = {
        "minimum_span_f1": 0.0,
        "minimum_tuple_precision": 0.0,
        "minimum_tuple_recall": 0.0,
        "minimum_parse_coverage": 0.0,
        "minimum_exact_validity": 0.0,
        "minimum_family_floor": 0.0,
        "minimum_perturbation_floor": 0.0,
        "maximum_abstention_cost": 1.0,
        "minimum_exact_admitted_events": 1,
    }
    recomputed["tuple_metric_rows"] = [{"arm": MODEL_ARM, "precision": 1.0, "recall": 1.0}]
    recomputed["model_relation_qualification_ready_score"] = 1
    recomputed["qualified_model_relation_event_count"] = 1
    recomputed["model_eligible_arm_rows"] = [{"arm": MODEL_ARM}]
    recomputed["reported_vs_recomputed_metrics"] = {
        "reported_ready_score": 1,
        "recomputed_ready_score": 1,
        "reported_qualified_event_count": 1,
        "recomputed_qualified_event_count": 1,
        "agreement": True,
    }
    recomputed["reproducibility_checksum"] = exp._artifact_checksum(recomputed)
    assert "aggregate_vs_row_disagreement" in exp.validate_artifact(recomputed)


def test_real_flagged_source_writes_blocked_artifact_without_held_open(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6901-ADMISSION re-runs admission in the fresh reducer."""

    output = tmp_path / "experiment_6901.json"
    artifact = exp.run_experiment(date="20260902", output_path=output)
    assert output.is_file()
    assert artifact["verdict_class"] == "blocked"
    assert artifact["model_relation_qualification_ready_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] in {
        "unflagged_exp6900",
        "fresh_adversarial_critical_count",
    }
    assert artifact["preconditions_checked"]["held_sidecar_open_count"] == 0


def test_run_clean_branch_and_cli_entrypoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-VERIFY-6901 supports a clean future source and the required CLI."""

    source_paths = {
        "exp6274": Path("compiler-artifact.json"),
        "exp6886": Path("fixture.json"),
        "exp6900": Path("proposal.json"),
        "compiler": Path("asp_energy.py"),
    }
    calibration_path = tmp_path / "calibration.json"
    held_path = tmp_path / "held.json"
    calibration_path.write_text(
        json.dumps({"schema": exp.SIDECAR_SCHEMA, "split": "calibration", "rows": [_formal()]}),
        encoding="utf-8",
    )
    held_path.write_text(
        json.dumps({"schema": exp.SIDECAR_SCHEMA, "split": "held", "rows": [_formal()]}),
        encoding="utf-8",
    )
    proposal = {
        "relation_corpus_complete_score": 1,
        "flagged_adversarial": False,
        "source_artifact_hashes": {"test": "bound"},
        "prompt_manifest": {"text": "extract"},
        "cell_manifest": [_cell()],
    }
    payloads = {
        "exp6274": {"asp_energy_semantic_ready_score": 1.0, "parity_failure_count": 0},
        "exp6886": {
            "relation_fixture_ready_score": 1,
            "closed_vocabulary_manifest": {"entries": _vocabulary()},
        },
        "exp6900": proposal,
    }
    for name, payload in payloads.items():
        (tmp_path / source_paths[name]).write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / source_paths["compiler"]).write_text("compiler", encoding="utf-8")
    expected = {name: exp.sha256_file(tmp_path / path) for name, path in source_paths.items()}
    expected.update(
        {
            "calibration_sidecar": exp.sha256_file(calibration_path),
            "held_sidecar": exp.sha256_file(held_path),
        }
    )
    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp, "EXP6274_PATH", source_paths["exp6274"])
    monkeypatch.setattr(exp, "EXP6886_PATH", source_paths["exp6886"])
    monkeypatch.setattr(exp, "EXP6900_PATH", source_paths["exp6900"])
    monkeypatch.setattr(exp, "COMPILER_PATH", source_paths["compiler"])
    monkeypatch.setattr(exp, "CALIBRATION_SIDECAR_PATH", calibration_path)
    monkeypatch.setattr(exp, "HELD_SIDECAR_PATH", held_path)
    monkeypatch.setattr(exp, "EXPECTED_HASHES", expected)
    monkeypatch.setattr(exp, "EXPECTED_EXP6900_SOURCE_HASHES", proposal["source_artifact_hashes"])
    monkeypatch.setattr(exp, "PROPOSAL_ARMS", (MODEL_ARM,))
    monkeypatch.setattr(exp, "MODEL_ARMS", (MODEL_ARM,))
    monkeypatch.setattr(exp, "GGUF_SEEDS", (7,))
    monkeypatch.setattr(exp.acquisition, "reconstruct_source_records", lambda: [_source()])
    monkeypatch.setattr(
        exp,
        "verify_artifact",
        lambda _path: {"loaded": True, "flags": [], "gate_version": "test"},
    )
    monkeypatch.setattr(exp.asp_energy, "solver_name_version", lambda: "clingo test")
    monkeypatch.setattr(exp.asp_energy, "solve_with_clingo", _solver)
    output = tmp_path / "clean.json"
    artifact = exp.run_experiment(date="20260902", output_path=output)
    assert artifact["verdict_class"] == "null"
    assert artifact["preconditions_checked"]["held_sidecar_open_count"] == 1

    real_run = exp.run_experiment
    monkeypatch.setattr(exp, "run_experiment", lambda **_kwargs: artifact)
    assert exp.main(["--date", "20260902", "--output", str(output)]) == 0
    assert "fresh_process_sealed_relation_reduction_no_llm" in capsys.readouterr().out

    monkeypatch.setattr(exp, "run_experiment", real_run)
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced"])
    with pytest.raises(exp.QualificationError, match="artifact_validation:forced"):
        exp.run_experiment(date="20260902", output_path=output)
