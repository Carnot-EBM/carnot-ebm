"""Tests for relation ASP and isomorphic qualification.

Spec refs: REQ-CONSTRAINT-6914 and SCENARIO-CONSTRAINT-6914-*.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6914_relation_asp_isomorphic_qualification as exp


MODEL_ARM = "gguf:model-a"


def _vocabulary() -> list[dict[str, Any]]:
    rows = []
    for ordinal in (0, 1):
        for polarity, atom in (
            ("positive", f"gc_{ordinal}_red"),
            ("negative", f"gc_{ordinal}_not_red"),
        ):
            rows.append(
                {
                    "family": "graph_coloring",
                    "subject_id": f"node_{ordinal}",
                    "predicate": "has_color",
                    "object_id": "red",
                    "polarity": polarity,
                    "normalized_tuple": [
                        f"node_{ordinal}",
                        "has_color",
                        "red",
                        polarity,
                    ],
                    "asp_atom": atom,
                }
            )
    return rows


def _formal(ordinal: int = 0) -> dict[str, Any]:
    return {
        "fixture_id": f"graph_coloring_{ordinal:02d}",
        "group_id": f"relation_group_{ordinal:02d}",
        "expected_case": "valid",
        "asp_program": (
            f"1 {{gc_{ordinal}_red; gc_{ordinal}_blue}} 1.\n"
            f":- gc_{ordinal}_red, gc_{ordinal}_not_red.\n"
            f"gc_{ordinal}_red.\n"
        ),
        "answer_sets": [[f"gc_{ordinal}_red"]],
        "zero_energy_states": [[f"gc_{ordinal}_red"]],
        "solver_receipt": {"name_version": "clingo test", "error": None},
    }


def _cell(*, parse_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    relation = {
        "line_index": 0,
        "status": "accepted",
        "reason": "accepted",
        "subject": {"start_utf8": 5, "end_utf8": 7, "text": "n0"},
        "predicate": "has_color",
        "object": {"start_utf8": 18, "end_utf8": 21, "text": "red"},
        "polarity": "positive",
        "normalized_tuple": ["n0", "has_color", "red", "positive"],
    }
    return {
        "cell_identity": "model-a::7::graph_coloring_00",
        "arm": MODEL_ARM,
        "hf_id": "model-a",
        "model_family": "test",
        "seed": 7,
        "fixture_id": "graph_coloring_00",
        "group_id": "relation_group_00",
        "family": "graph_coloring",
        "split": "calibration",
        "parse_rows": [relation] if parse_rows is None else parse_rows,
        "raw_output": "REL fixture output",
        "terminal": True,
        "timed_out": False,
        "truncated": False,
    }


def _build_artifact(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "date": "20260903",
        "cells": [_cell()],
        "formal_rows": [_formal(0), _formal(1)],
        "vocabulary": _vocabulary(),
        "source_artifact_hashes": {"exp6912": {"observed_sha256": "sha256:test"}},
        "sealed_sidecar_hashes": {
            "calibration": {"sha256": "sha256:cal"},
            "held": {"sha256": "sha256:held"},
        },
        "precondition_checks": [exp.gate_check("public_inputs", True, True)],
        "sealed_access_rows": [
            {"sidecar": "calibration", "open_count": 1, "hash_match": True},
            {"sidecar": "held", "open_count": 1, "hash_match": True},
        ],
        "held_leakage_rows": [],
        "duration_s": 1.25,
    }
    values.update(overrides)
    return exp.build_artifact(**values)


def test_req_constraint_6914_spec_declares_every_failure_boundary() -> None:
    """REQ-CONSTRAINT-6914 owns the contract before implementation."""

    text = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-CONSTRAINT-6914") :]
    for suffix in (
        "PRECONDITIONS",
        "VOCABULARY",
        "SOLVERS",
        "RENAMING",
        "PARAPHRASE",
        "REVERSAL",
        "CONTRADICTION",
        "OMISSION",
        "RESTRUCTURING",
        "SEAL",
        "AGGREGATES",
        "READINESS",
    ):
        assert f"SCENARIO-CONSTRAINT-6914-{suffix}" in section


def test_compiler_drift_blocks_and_blocked_receipt_is_complete() -> None:
    """SCENARIO-CONSTRAINT-6914-PRECONDITIONS blocks public drift before labels."""

    artifacts = {
        "exp6274": {"asp_energy_semantic_ready_score": 1.0, "parity_failure_count": 0},
        "exp6886": {"relation_fixture_ready_score": 1},
        "exp6900": {"relation_corpus_complete_score": 1, "held_sidecar_access_count": 0},
        "exp6912": {"clean_relation_corpus_ready_score": 1, "replayed_cell_count": 1400},
    }
    observed = {name: f"sha256:{name}" for name in exp.EXPECTED_PUBLIC_HASHES}
    expected = dict(observed)
    expected["compiler"] = "sha256:qualified"
    summary = exp.public_precondition_summary(
        observed,
        artifacts,
        expected_hashes=expected,
        solver_name="clingo test",
    )
    assert summary["passed"] is False
    assert summary["failed_check"] == "public_hash:compiler"
    artifact = exp.blocked_artifact(
        date="20260903",
        checks=summary["checks"],
        source_artifact_hashes={},
        sealed_sidecar_hashes={},
        duration_s=0.1,
    )
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["preconditions_checked"]["sealed_sidecar_open_count"] == 0
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact) <= set(artifact["field_principles"])


def test_vocabulary_escape_is_explicit_and_nonparseable_cells_remain() -> None:
    """SCENARIO-CONSTRAINT-6914-VOCABULARY rejects escaped atoms without filtering."""

    accepted = exp.map_relation_tuple(["node_0", "has_color", "red", "positive"], _vocabulary())
    escaped = exp.map_relation_tuple(["node_0", "owns", "red", "positive"], _vocabulary())
    assert accepted == {"accepted": True, "asp_atom": "gc_0_red", "reason": "mapped"}
    assert escaped["accepted"] is False
    assert escaped["reason"] == "unsupported_atom"

    artifact = _build_artifact(
        cells=[_cell(parse_rows=[{"status": "malformed", "reason": "bad_fields"}])]
    )
    assert len(artifact["rows"]) == len(exp.PERTURBATIONS)
    assert artifact["proposal_coverage_by_arm"][0]["denominator"] == 1
    assert artifact["proposal_coverage_by_arm"][0]["numerator"] == 0


def test_stable_model_mismatch_and_shared_bug_are_separate() -> None:
    """SCENARIO-CONSTRAINT-6914-SOLVERS separates parity from sealed agreement."""

    theory_atoms = {
        "gc_0_red",
        "gc_0_blue",
        "gc_0_not_red",
        "gc_1_red",
        "gc_1_blue",
        "gc_1_not_red",
    }

    def red(_program: str, _program_id: str) -> dict[str, Any]:
        return {"models": [["gc_0_red"]], "receipt": {"engine": "red"}}

    def blue(_program: str, _program_id: str) -> dict[str, Any]:
        return {"models": [["gc_0_blue"]], "receipt": {"engine": "blue"}}

    mismatch = exp.evaluate_cell(
        _cell(), _formal(), _vocabulary(), theory_atoms, primary_engine=red, independent_engine=blue
    )[0]
    assert mismatch["solver_parity"] is False
    assert mismatch["expected_effect_met"] is True

    shared_bug = exp.evaluate_cell(
        _cell(),
        _formal(),
        _vocabulary(),
        theory_atoms,
        primary_engine=blue,
        independent_engine=blue,
    )[0]
    assert shared_bug["solver_parity"] is True
    assert shared_bug["expected_effect_met"] is False


def test_noninjective_entity_renaming_fails_closed() -> None:
    """SCENARIO-CONSTRAINT-6914-RENAMING rejects a many-to-one atom map."""

    with pytest.raises(exp.QualificationError, match="non_injective_renaming"):
        exp.rename_program("a.\nb.\n", {"a": "x", "b": "x"})
    renamed = exp.rename_program("a.\nb :- a.\n", {"a": "x", "b": "y"})
    assert renamed == "x.\ny :- x.\n"


def test_semantic_changing_paraphrase_is_not_canonicalized() -> None:
    """SCENARIO-CONSTRAINT-6914-PARAPHRASE accepts only frozen aliases."""

    original = ["node_0", "has_color", "red", "positive"]
    paraphrased = exp.build_paraphrase_tuple(original)
    assert paraphrased[1] == "is colored"
    assert exp.canonicalize_paraphrase(paraphrased) == original
    changed = ["node_0", "rejects", "red", "positive"]
    with pytest.raises(exp.QualificationError, match="semantic_changing_paraphrase"):
        exp.canonicalize_paraphrase(changed)


@pytest.mark.parametrize(
    ("perturbation", "base", "candidate"),
    [
        (
            "reversal",
            {"tuples": [["node_0", "has_color", "red", "positive"]]},
            {
                "tuples": [["node_0", "has_color", "red", "positive"]],
                "exact_atom_valid": True,
            },
        ),
        (
            "contradiction",
            {"atoms": ["gc_0_red"], "satisfiable": True},
            {"atoms": ["gc_0_red"], "satisfiable": True},
        ),
        (
            "omission",
            {"atoms": ["gc_0_red"], "satisfiable": True},
            {"atoms": ["gc_0_red"], "satisfiable": True},
        ),
    ],
)
def test_directional_perturbation_no_op_fails(
    perturbation: str, base: dict[str, Any], candidate: dict[str, Any]
) -> None:
    """SCENARIO-CONSTRAINT-6914-REVERSAL, CONTRADICTION, and OMISSION reject no-ops."""

    result = exp.validate_pair(perturbation, base, candidate)
    assert result["applicable"] is True
    assert result["met"] is False


def test_restructuring_count_shortcut_and_no_op_fail() -> None:
    """SCENARIO-CONSTRAINT-6914-RESTRUCTURING rejects count decisions and no-ops."""

    base = {
        "satisfiable": True,
        "model_count": 1,
        "projected_models": [["gc_0_red"]],
        "exact_outcome_decision": "qualified",
    }
    no_op = {**base}
    assert exp.validate_pair("solution_space_restructuring", base, no_op)["met"] is False
    shortcut = {
        **base,
        "model_count": 2,
        "exact_outcome_decision": "disqualified",
    }
    result = exp.validate_pair("solution_space_restructuring", base, shortcut)
    assert result["met"] is False
    assert result["shortcut_detected"] is True
    valid = {**shortcut, "exact_outcome_decision": "qualified"}
    assert exp.validate_pair("solution_space_restructuring", base, valid)["met"] is True


def test_held_leakage_and_sidecar_second_open_fail(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6914-SEAL detects formal leakage and one-open drift."""

    sidecar = {
        "schema": exp.SIDECAR_SCHEMA,
        "split": "held",
        "rows": [_formal()],
    }
    leaked = {
        "prompt_manifest": {},
        "cell_manifest": [{"cell_identity": "held-cell", "raw_output": _formal()["asp_program"]}],
    }
    rows = exp.detect_held_leakage(leaked, sidecar)
    assert rows[0]["leak_kind"] == "asp_program"

    path = tmp_path / "held.json"
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    reader = exp.SealedSidecarReader(path, exp.sha256_file(path), expected_split="held")
    assert reader.open_once()["split"] == "held"
    with pytest.raises(exp.QualificationError, match="opened_more_than_once"):
        reader.open_once()


def test_artifact_has_terminal_pairs_independent_receipts_and_replay() -> None:
    """REQ-CONSTRAINT-6914 emits every cell pair and keeps all metrics separate."""

    artifact = _build_artifact()
    assert len(artifact["rows"]) == len(exp.PERTURBATIONS)
    assert {row["perturbation"] for row in artifact["rows"]} == set(exp.PERTURBATIONS)
    assert all(row["terminal"] is True for row in artifact["rows"])
    assert all("expected_effect" in row and "observed_effect" in row for row in artifact["rows"])
    assert len(artifact["primary_solver_rows"]) == len(exp.PERTURBATIONS)
    assert len(artifact["independent_solver_rows"]) == len(exp.PERTURBATIONS)
    assert artifact["solver_disagreement_count"] == 0
    assert artifact["held_leakage_count"] == 0
    assert artifact["model_inference_call_count"] == 0
    assert artifact["asp_isomorphic_shard_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["reported_vs_recomputed_metrics"]["agreement"] is True
    assert exp.validate_artifact(artifact) == []


def test_aggregate_disagreement_forces_readiness_to_zero() -> None:
    """SCENARIO-CONSTRAINT-6914-AGGREGATES rejects a changed reported summary."""

    artifact = _build_artifact()
    changed = deepcopy(artifact)
    changed["arm_summary_rows"][0]["terminal_row_count"] += 1
    changed["asp_isomorphic_shard_ready_score"] = 1
    errors = exp.validate_artifact(changed)
    assert "aggregate_disagreement" in errors
    replay = exp.replay_reported_metrics(changed)
    assert replay["agreement"] is False
    assert replay["recomputed_ready_score"] == 0


def test_helpers_cover_transform_and_hash_failures(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6914 keeps every deterministic transform fail-closed."""

    relation = ["node_0", "has_color", "red", "positive"]
    assert exp.reverse_tuple(relation) == ["red", "has_color", "node_0", "positive"]
    assert exp.contradiction_atoms(["gc_0_red"], _vocabulary(), "graph_coloring_00") == [
        "gc_0_not_red",
        "gc_0_red",
    ]
    assert exp.omit_relation_atoms(["gc_0_red"]) == []
    assert exp.restructure_program("gc_0_red.\n", "gc_1_red").endswith("0 {gc_1_red} 1.\n")
    with pytest.raises(exp.QualificationError, match="restructuring_no_op"):
        exp.restructure_program("gc_0_red.\n", "gc_0_red")
    with pytest.raises(exp.QualificationError, match="unknown_pair"):
        exp.validate_pair("mystery", {}, {})

    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    reader = exp.SealedSidecarReader(bad, exp.sha256_file(bad), expected_split="held")
    with pytest.raises(exp.QualificationError, match="sidecar_identity"):
        reader.open_once()
    reader = exp.SealedSidecarReader(bad, "sha256:wrong", expected_split="held")
    with pytest.raises(exp.QualificationError, match="sidecar_hash_drift"):
        reader.open_once()

    encoded = base64.b64encode(b"sealed text").decode("ascii")
    assert exp.proposal_strings({"raw_output_b64": encoded}) == ["sealed text"]
    assert exp.proposal_strings({"raw_output_b64": "not-base64"}) == []

    assert exp.gate_check("nested", {"values": {2, 1}}, {"values": {1, 2}})["passed"] is True
    with pytest.raises(exp.QualificationError, match="unsupported_fixture"):
        exp._fixture_parts("unknown_00")
    with pytest.raises(exp.QualificationError, match="invalid_tuple_arity"):
        exp.reverse_tuple(["too", "short"])
    with pytest.raises(exp.QualificationError, match="unsupported_paraphrase_source"):
        exp.build_paraphrase_tuple(["node_0", "owns", "red", "positive"])
    with pytest.raises(exp.QualificationError, match="empty_renaming"):
        exp.rename_program("a.\n", {})
    with pytest.raises(exp.QualificationError, match="entity_renaming_no_op"):
        exp.rename_program("a.\n", {"b": "c"})
    with pytest.raises(exp.QualificationError, match="engine_models_not_list"):
        exp._canonical_models(None)


def test_internal_fail_closed_edges_and_disqualified_artifact(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-6914-VOCABULARY keeps internal edge failures terminal."""

    row = exp._stub_row(
        _cell(),
        "base",
        tuples=[["node_0", "has_color", "red", "positive"]],
        atoms=["gc_0_red"],
        parseable_count=1,
        nonparseable_count=0,
        exact_atom_valid=True,
    )
    evaluated = exp._evaluate_program(
        row=row,
        program_text="escaped_atom.\n",
        allowed_atoms={"gc_0_red"},
        primary_engine=exp.primary_exact_engine,
        independent_engine=exp.independent_exact_engine,
    )
    assert evaluated["compilation_status"] == "unsupported_atom"
    with pytest.raises(exp.QualificationError, match="ambiguous_fixture_vocabulary"):
        exp._semantic_tuple({}, [])
    with pytest.raises(exp.QualificationError, match="contradiction_vocabulary"):
        exp.contradiction_atoms([], _vocabulary()[:1], "graph_coloring_00")
    with pytest.raises(exp.QualificationError, match="entity_renaming_target_missing"):
        exp._entity_mapping("gc_0_red.\n", "graph_coloring_00", {"gc_0_red"})
    with pytest.raises(exp.QualificationError, match="entity_renaming_escape"):
        exp._entity_mapping(
            "gc_0_red.\ngc_0_blue.\n",
            "graph_coloring_00",
            {"gc_0_red", "gc_0_blue", "gc_1_red"},
        )

    bad_relation = deepcopy(_cell()["parse_rows"][0])
    bad_relation["predicate"] = "owns"
    rows = exp.evaluate_cell(
        _cell(parse_rows=[bad_relation]),
        _formal(),
        _vocabulary(),
        {"gc_0_red", "gc_0_blue", "gc_0_not_red", "gc_1_red"},
    )
    assert rows[0]["exact_outcome_decision"] == "unsupported_atom"
    assert all(row["compilation_status"] == "not_run_base_ineligible" for row in rows[1:])

    disqualified = _build_artifact(
        held_leakage_rows=[{"leak_kind": "asp_program", "location": "test"}]
    )
    assert disqualified["asp_isomorphic_shard_ready_score"] == 0
    assert disqualified["verdict_class"] == "disqualified"

    nested_leak = {
        "prompt_manifest": {"nested": ["prefix " + _formal()["asp_program"]]},
        "cell_manifest": ["not-a-cell", {"cell_identity": "empty"}],
    }
    held = {"rows": ["not-formal", {**_formal(), "answer_sets": [], "solver_receipt": {}}]}
    assert exp.detect_held_leakage(nested_leak, held)[0]["leak_kind"] == "asp_program"

    non_object = tmp_path / "public.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(exp.QualificationError, match="public_artifact_not_object"):
        exp._read_object(non_object)
    assert exp._safe_hash(tmp_path / "missing") == "missing"


def test_validator_reports_each_terminal_boundary() -> None:
    """SCENARIO-CONSTRAINT-6914-AGGREGATES rejects every terminal schema escape."""

    artifact = _build_artifact()
    mutations = [
        ("missing", lambda value: value.pop("rows"), "missing_fields:rows"),
        (
            "principles",
            lambda value: value.__setitem__("field_principles", {}),
            "field_principles",
        ),
        (
            "substrate",
            lambda value: value.__setitem__("inference_substrate", "wrong"),
            "inference_substrate",
        ),
        (
            "oracle",
            lambda value: value.__setitem__("verifier_is_oracle", False),
            "verifier_is_oracle",
        ),
        (
            "class",
            lambda value: value.__setitem__("verdict_class", "positive"),
            "positive_verdict_forbidden",
        ),
        (
            "verdict",
            lambda value: value.__setitem__("honest_verdict", "unfinished"),
            "honest_verdict",
        ),
        (
            "inference",
            lambda value: value.__setitem__("model_inference_call_count", 1),
            "model_inference_call_count",
        ),
        (
            "rows",
            lambda value: value["rows"].pop(),
            "terminal_row_count",
        ),
        (
            "solver",
            lambda value: value.__setitem__("solver_disagreement_count", 1),
            "solver_disagreement_count",
        ),
        (
            "leak",
            lambda value: value.__setitem__("held_leakage_count", 1),
            "held_leakage_count",
        ),
        (
            "ready",
            lambda value: value.__setitem__("asp_isomorphic_shard_ready_score", 0),
            "readiness_disagreement",
        ),
    ]
    for _name, mutate, expected in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert expected in exp.validate_artifact(changed)
    assert (
        exp.validate_artifact(
            exp.blocked_artifact(
                date="20260903",
                checks=[],
                source_artifact_hashes={},
                sealed_sidecar_hashes={},
                duration_s=0,
            )
        )
        == []
    )


def test_build_rejects_missing_formal_row() -> None:
    """SCENARIO-CONSTRAINT-6914-PRECONDITIONS rejects an unmatched fixture."""

    with pytest.raises(exp.QualificationError, match="missing_formal_row"):
        _build_artifact(formal_rows=[_formal(1)])


def test_run_writes_real_artifact_and_public_failure_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CONSTRAINT-6914 runs the sealed protocol and its blocked public branch."""

    output = tmp_path / "real.json"
    artifact = exp.run_experiment(date="20260903", output_path=output)
    assert artifact["asp_isomorphic_shard_ready_score"] == 1
    assert (
        json.loads(output.read_text(encoding="utf-8"))["reproducibility_checksum"]
        == artifact["reproducibility_checksum"]
    )

    blocked_output = tmp_path / "blocked.json"
    blocked = exp.run_experiment(date="20260903", output_path=blocked_output, root=tmp_path)
    assert blocked["honest_verdict"] == exp.BLOCKED_VERDICT
    assert blocked["preconditions_checked"]["sealed_sidecar_open_count"] == 0

    def fail_open(_reader: Any) -> dict[str, Any]:
        raise exp.QualificationError("forced_sidecar_failure")

    monkeypatch.setattr(exp.SealedSidecarReader, "open_once", fail_open)
    sidecar_blocked = exp.run_experiment(
        date="20260903", output_path=tmp_path / "sidecar-blocked.json"
    )
    assert sidecar_blocked["gate_check_summary"]["failed_check"] == "sealed_sidecars"


def test_main_forwards_required_date_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CONSTRAINT-6914 exposes the required command-line entrypoint."""

    output = tmp_path / "result.json"
    calls: list[tuple[str, Path]] = []

    def fake_run(*, date: str, output_path: Path) -> dict[str, Any]:
        calls.append((date, output_path))
        return {}

    monkeypatch.setattr(exp, "run_experiment", fake_run)
    assert exp.main(["--date", "20260903", "--output", str(output)]) == 0
    assert calls == [("20260903", output)]
