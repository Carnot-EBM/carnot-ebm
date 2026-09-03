"""Tests for the Exp6915 qualified relation event-bank merge.

Spec refs: REQ-CONSTRAINT-6915 and SCENARIO-CONSTRAINT-6915-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from carnot import experiment_6915_qualified_relation_event_bank as exp


MODEL_A = "model/a"
MODEL_B = "model/b"
PERTURBATIONS = (
    "base",
    "entity_renaming",
    "relation_paraphrase",
    "relation_reversal",
    "contradiction_injection",
    "relation_omission",
    "solution_space_restructuring",
)


def _source_row(
    *,
    model: str = MODEL_A,
    model_family: str = "family_a",
    seed: int | None = 11,
    fixture_id: str = "graph_coloring_00",
    group_id: str = "group_0",
    source_ok: bool = True,
    arm: str | None = None,
) -> dict[str, Any]:
    actual_arm = arm or f"gguf:{model}"
    deterministic = not actual_arm.startswith("gguf:")
    actual_seed = None if deterministic else seed
    seed_part = "deterministic" if deterministic else str(actual_seed)
    cell_identity = f"{model}::{seed_part}::{fixture_id}"
    passed = bool(source_ok)
    return {
        "row_type": "terminal_source_tuple_decision",
        "cell_identity": cell_identity,
        "arm": actual_arm,
        "model_id": model,
        "model_family": model_family,
        "seed": actual_seed,
        "seed_label": seed_part,
        "fixture_id": fixture_id,
        "group_id": group_id,
        "family": "graph_coloring",
        "split": "held",
        "terminal": True,
        "source_text_hash": "sha256:source",
        "source_bytes_b64": "c291cmNl",
        "raw_output_check": {"passed": passed},
        "parser_check": {"passed": passed},
        "source_offset_check": {"passed": passed},
        "source_byte_identity_check": {"passed": passed},
        "tuple_type_check": {"passed": passed},
        "entity_anchor_check": {"passed": passed},
        "relation_direction_check": {"passed": passed},
        "omission_check": {"passed": passed},
        "duplicate_check": {"passed": passed},
        "abstention_check": {"passed": passed},
        "parsed_tuple_fields": [["node_0", "has_color", "red", "positive"]],
        "source_grounded_correct": not passed,
        "source_grounding_label": "intentionally_inverted_shard_label",
    }


def _effect(models: list[list[str]]) -> dict[str, Any]:
    return {
        "model_count": len(models),
        "models": models,
        "models_sha256": exp.sha256_json(models),
        "satisfiable": bool(models),
    }


def _semantic_row(source: dict[str, Any], perturbation: str) -> dict[str, Any]:
    base = _effect([["a"]])
    observed = deepcopy(base)
    expected: dict[str, Any] = {"kind": "sealed_exact_models", **deepcopy(base)}
    atoms = ["a"]
    program_atoms = ["a"]
    projected_models = [["a"]]
    compilation_status = "compiled"
    program: str | None = "a.\n"
    exact_atom_valid = True
    solver_parity: bool | None = True
    primary_receipt: dict[str, Any] = {"status": "complete", "program_sha256": "sha256:p"}
    independent_receipt: dict[str, Any] = {
        "status": "complete",
        "program_sha256": "sha256:p",
    }
    unsupported_atoms: list[Any] = []
    tuples = [["node_0", "has_color", "red", "positive"]]
    isomorphic_invariant: bool | None = None
    auxiliary_atom: str | None = None

    if perturbation == "entity_renaming":
        observed = _effect([["b"]])
        expected = {"kind": "renamed_sealed_models", **deepcopy(observed)}
        atoms = ["b"]
        program_atoms = ["b"]
        isomorphic_invariant = True
    elif perturbation == "relation_paraphrase":
        expected = {"kind": "paraphrased_sealed_models", **deepcopy(base)}
        isomorphic_invariant = True
    elif perturbation == "relation_reversal":
        expected = {"kind": "directional_vocabulary_rejection"}
        observed = {"status": "not_run_directional_rejection", **_effect([])}
        observed["satisfiable"] = None
        observed["model_count"] = None
        compilation_status = "unsupported_atom"
        program = None
        atoms = []
        program_atoms = []
        exact_atom_valid = False
        solver_parity = None
        primary_receipt = {"status": "not_run"}
        independent_receipt = {"status": "not_run"}
        unsupported_atoms = [["red", "has_color", "node_0", "positive"]]
        tuples = [["red", "has_color", "node_0", "positive"]]
        projected_models = []
    elif perturbation == "contradiction_injection":
        observed = _effect([])
        expected = {"kind": "contradiction", "satisfiable": False}
        atoms = ["a", "not_a"]
        program_atoms = ["a", "not_a"]
        projected_models = []
    elif perturbation == "relation_omission":
        observed = _effect([[]])
        expected = {"kind": "all_proposal_relation_facts_removed"}
        atoms = []
        program_atoms = ["a"]
        projected_models = [[]]
        tuples = []
    elif perturbation == "solution_space_restructuring":
        observed = _effect([["a"], ["a", "aux"]])
        expected = {"kind": "same_projected_models_and_satisfiability_different_model_count"}
        program_atoms = ["a", "aux"]
        projected_models = [["a"]]
        auxiliary_atom = "aux"

    row = {
        "row_id": f"{source['cell_identity']}::{perturbation}",
        "cell_identity": source["cell_identity"],
        "perturbation": perturbation,
        "arm": source["arm"],
        "model_id": source["model_id"],
        "seed": source["seed"] if source["seed"] is not None else "deterministic",
        "fixture_id": source["fixture_id"],
        "family": source["family"],
        "split": source["split"],
        "terminal": True,
        "compilation_status": compilation_status,
        "program": program,
        "program_sha256": "sha256:p" if program is not None else None,
        "atoms": atoms,
        "program_atoms": program_atoms,
        "unsupported_atoms": unsupported_atoms,
        "exact_atom_valid": exact_atom_valid,
        "primary_effect": deepcopy(observed),
        "independent_effect": deepcopy(observed),
        "observed_effect": deepcopy(observed),
        "expected_effect": expected,
        "primary_solver_receipt": primary_receipt,
        "independent_solver_receipt": independent_receipt,
        "solver_parity": solver_parity,
        "expected_effect_met": True,
        "transform_valid": True,
        "pair_requirement_met": True,
        "pair_applicable": perturbation != "base",
        "isomorphic_invariant": isomorphic_invariant,
        "projected_models": projected_models,
        "tuples": tuples,
        "exact_outcome_decision": "disqualified",
        "shortcut_detected": False,
    }
    if auxiliary_atom is not None:
        row["auxiliary_atom"] = auxiliary_atom
    return row


def _semantic_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [_semantic_row(source, perturbation) for perturbation in PERTURBATIONS]


def _passed_preconditions() -> dict[str, Any]:
    return exp.gate_summary([exp.gate_check("synthetic_inputs", True, True)])


def _artifact(
    sources: list[dict[str, Any]],
    semantics: list[dict[str, Any]],
    *,
    required_model_families: tuple[str, ...] = ("family_a",),
    required_families: tuple[str, ...] = ("graph_coloring",),
    minimum_events: int = 1,
    minimum_per_model_family: int = 1,
) -> dict[str, Any]:
    return exp.build_artifact(
        date="20260903",
        duration_s=0.25,
        source_rows=sources,
        semantic_rows=semantics,
        source_artifact_hashes={"exp6913": {}, "exp6914": {}},
        preconditions_checked=_passed_preconditions(),
        expected_cell_ids={row["cell_identity"] for row in sources},
        perturbations=PERTURBATIONS,
        required_model_families=required_model_families,
        required_families=required_families,
        minimum_events=minimum_events,
        minimum_per_model_family=minimum_per_model_family,
        fresh_adversarial_rows=[],
    )


def test_req_constraint_6915_spec_declares_requested_failure_boundaries() -> None:
    """REQ-CONSTRAINT-6915 owns every requested merge failure."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-CONSTRAINT-6915") :]
    for suffix in (
        "PRECONDITIONS",
        "JOIN",
        "CONTROLS",
        "ELIGIBILITY",
        "POOLING",
        "ROWS",
        "AGGREGATES",
        "READINESS",
    ):
        assert f"SCENARIO-CONSTRAINT-6915-{suffix}" in section


def test_missing_join_rows_and_duplicate_ids_are_explicit() -> None:
    """SCENARIO-CONSTRAINT-6915-JOIN preserves missing and duplicate rows."""

    source = _source_row()
    semantic = _semantic_rows(source)
    missing = exp.build_join_evidence(
        [source], semantic[:-1], {source["cell_identity"]}, PERTURBATIONS
    )
    assert missing["missing_join_rows"] == [
        {
            "cell_identity": source["cell_identity"],
            "perturbation_id": "solution_space_restructuring",
            "source_occurrence_count": 1,
            "semantic_occurrence_count": 0,
            "reason": "missing_semantic_join_row",
        }
    ]
    duplicated = exp.build_join_evidence(
        [source, deepcopy(source)],
        [*semantic, deepcopy(semantic[0])],
        {source["cell_identity"]},
        PERTURBATIONS,
    )
    kinds = {row["reason"] for row in duplicated["duplicate_join_rows"]}
    assert kinds == {"duplicate_source_cell_id", "duplicate_semantic_join_id"}
    assert all(row["decision"] == "rejected" for row in duplicated["join_rows"])


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("cross_model", "identity_mismatch:model_id"),
        ("cross_seed", "identity_mismatch:seed"),
        ("perturbation", "perturbation_identity_mismatch"),
    ],
)
def test_cross_identity_and_perturbation_mismatches_reject(mutation: str, reason: str) -> None:
    """SCENARIO-CONSTRAINT-6915-JOIN rejects cross-cell substitutions."""

    source = _source_row()
    semantic = _semantic_rows(source)
    target = semantic[0]
    if mutation == "cross_model":
        target["model_id"] = MODEL_B
    elif mutation == "cross_seed":
        target["seed"] = 99
    else:
        target["row_id"] = f"{source['cell_identity']}::relation_omission"
    evidence = exp.build_join_evidence([source], semantic, {source["cell_identity"]}, PERTURBATIONS)
    base = next(row for row in evidence["join_rows"] if row["perturbation_id"] == "base")
    assert reason in base["reasons"]
    assert base["decision"] == "rejected"


def test_control_substitution_never_increases_the_model_count() -> None:
    """SCENARIO-CONSTRAINT-6915-CONTROLS keeps both control classes separate."""

    model = _source_row(source_ok=False)
    enoki = _source_row(
        model="enoki:pinned_openie_encoder",
        model_family="enoki",
        arm=exp.ENOKI_ARM,
    )
    rule = _source_row(
        model="rule:anchored_lexical_v1",
        model_family="lexical_rule",
        arm=exp.RULE_ARM,
    )
    sources = [model, enoki, rule]
    semantics = [row for source in sources for row in _semantic_rows(source)]
    artifact = _artifact(sources, semantics)
    assert artifact["qualified_model_relation_event_count"] == 0
    assert len(artifact["enoki_control_rows"]) == 1
    assert len(artifact["rule_control_rows"]) == 1
    assert artifact["control_substitution_count"] == 0
    assert artifact["qualified_relation_event_bank_ready_score"] == 0


def test_family_pooling_cannot_hide_a_failing_model_family() -> None:
    """SCENARIO-CONSTRAINT-6915-POOLING requires each model family."""

    strong = _source_row()
    weak = _source_row(model=MODEL_B, model_family="family_b", source_ok=False)
    sources = [strong, weak]
    semantics = [row for source in sources for row in _semantic_rows(source)]
    artifact = _artifact(
        sources,
        semantics,
        required_model_families=("family_a", "family_b"),
    )
    assert artifact["qualified_model_relation_event_count"] == 1
    summaries = {row["model_family"]: row for row in artifact["model_summary_rows"]}
    assert summaries["family_a"]["admitted_event_count"] == 1
    assert summaries["family_b"]["admitted_event_count"] == 0
    failed = {row["check"] for row in artifact["gate_check_summary"]["checks"] if not row["passed"]}
    assert "minimum_events:model_family:family_b" in failed
    assert artifact["qualified_relation_event_bank_ready_score"] == 0


def test_eligibility_is_recomputed_instead_of_importing_shard_labels() -> None:
    """SCENARIO-CONSTRAINT-6915-ELIGIBILITY ignores inverted shard decisions."""

    source = _source_row()
    semantic = _semantic_rows(source)
    admitted = _artifact([source], semantic)
    assert admitted["admitted_event_rows"][0]["cell_identity"] == source["cell_identity"]

    invalid = deepcopy(source)
    invalid["source_grounded_correct"] = True
    invalid["tuple_type_check"]["passed"] = False
    rejected = _artifact([invalid], _semantic_rows(invalid))
    assert rejected["admitted_event_rows"] == []
    assert "tuple_type_invalid" in rejected["rejected_event_rows"][0]["reasons"]


def test_base_unsupported_and_solver_disagreement_reject_the_cell() -> None:
    """REQ-CONSTRAINT-6915 rejects unsupported or solver-disagreed components."""

    source = _source_row()
    unsupported = _semantic_rows(source)
    unsupported[0]["compilation_status"] = "unsupported_atom"
    unsupported[0]["unsupported_atoms"] = [["escaped"]]
    artifact = _artifact([source], unsupported)
    assert "unsupported_component:base" in artifact["rejected_event_rows"][0]["reasons"]

    disagreed = _semantic_rows(source)
    disagreed[0]["independent_effect"] = _effect([])
    artifact = _artifact([source], disagreed)
    assert "solver_parity_failed:base" in artifact["rejected_event_rows"][0]["reasons"]


def test_row_omission_and_aggregate_disagreement_fail_validation() -> None:
    """SCENARIO-CONSTRAINT-6915-ROWS and AGGREGATES replay all rows."""

    admitted_source = _source_row()
    rejected_source = _source_row(
        model=MODEL_A,
        seed=12,
        fixture_id="graph_coloring_01",
        source_ok=False,
    )
    sources = [admitted_source, rejected_source]
    semantics = [row for source in sources for row in _semantic_rows(source)]
    artifact = _artifact(sources, semantics)
    assert exp.validate_artifact(artifact) == []

    omitted = deepcopy(artifact)
    omitted["rows"].pop()
    omitted["reproducibility_checksum"] = exp.artifact_checksum(omitted)
    assert "row_coverage" in exp.validate_artifact(omitted)

    disagreed = deepcopy(artifact)
    disagreed["model_summary_rows"][0]["admitted_event_count"] += 1
    disagreed["reproducibility_checksum"] = exp.artifact_checksum(disagreed)
    assert "aggregate_disagreement" in exp.validate_artifact(disagreed)


def test_preconditions_require_gates_hashes_manifests_and_clean_reports() -> None:
    """SCENARIO-CONSTRAINT-6915-PRECONDITIONS fails every unsafe boundary."""

    source = _source_row()
    semantics = _semantic_rows(source)
    source_shard = {"source_tuple_shard_ready_score": 1, "rows": [source]}
    semantic_shard = {"asp_isomorphic_shard_ready_score": 1, "rows": semantics}
    clean_reports = {
        "exp6913": {"loaded": True, "flags": []},
        "exp6914": {"loaded": True, "flags": []},
    }
    summary = exp.validate_preconditions(
        source_shard=source_shard,
        semantic_shard=semantic_shard,
        observed_hashes={"exp6913": "sha256:a", "exp6914": "sha256:b"},
        expected_hashes={"exp6913": "sha256:a", "exp6914": "sha256:b"},
        expected_cell_ids={source["cell_identity"]},
        perturbations=PERTURBATIONS,
        fresh_reports=clean_reports,
    )
    assert summary["passed"] is True

    unsafe_source = deepcopy(source_shard)
    unsafe_source["source_tuple_shard_ready_score"] = 0
    unsafe_semantic = deepcopy(semantic_shard)
    unsafe_semantic["rows"].pop()
    flagged = deepcopy(clean_reports)
    flagged["exp6914"]["flags"] = [{"severity": "critical", "kind": "TEST", "detail": "synthetic"}]
    failed = exp.validate_preconditions(
        source_shard=unsafe_source,
        semantic_shard=unsafe_semantic,
        observed_hashes={"exp6913": "sha256:drift", "exp6914": "sha256:b"},
        expected_hashes={"exp6913": "sha256:a", "exp6914": "sha256:b"},
        expected_cell_ids={source["cell_identity"]},
        perturbations=PERTURBATIONS,
        fresh_reports=flagged,
    )
    names = {row["check"] for row in failed["checks"] if not row["passed"]}
    assert {
        "source_tuple_shard_ready_score",
        "source_hash:exp6913",
        "semantic_join_identity_manifest",
        "fresh_adversarial_critical_count:exp6914",
    } <= names


def test_blocked_artifact_has_the_complete_required_schema() -> None:
    """SCENARIO-CONSTRAINT-6915-PRECONDITIONS emits a complete blocked receipt."""

    failed = exp.gate_summary([exp.gate_check("source_hash:exp6913", "sha256:a", "missing")])
    artifact = exp.blocked_artifact(
        date="20260903",
        duration_s=0.1,
        source_artifact_hashes={},
        preconditions_checked=failed,
        fresh_adversarial_rows=[],
    )
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"] == "complete_blocked_qualified_relation_event_bank"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "source_hash:exp6913"
    assert exp.validate_artifact(artifact) == []


def test_ready_bank_uses_only_admitted_model_rows_and_has_headroom() -> None:
    """SCENARIO-CONSTRAINT-6915-READINESS emits a circular-positive bank receipt."""

    admitted = _source_row()
    rejected = _source_row(model=MODEL_A, seed=12, fixture_id="graph_coloring_01", source_ok=False)
    sources = [admitted, rejected]
    semantics = [row for source in sources for row in _semantic_rows(source)]
    artifact = _artifact(sources, semantics)
    assert artifact["qualified_model_relation_event_count"] == 1
    assert artifact["qualified_model_relation_event_count"] == len(artifact["admitted_event_rows"])
    assert artifact["source_group_headroom_rows"][0]["positive_headroom"] is True
    assert artifact["qualified_relation_event_bank_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert exp.validate_artifact(artifact) == []


def test_run_writes_blocked_receipt_and_wrapper_exposes_cli(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6915 writes only the requested path and exposes its command."""

    source = _source_row()
    semantic = _semantic_rows(source)
    source_path = tmp_path / "source.json"
    semantic_path = tmp_path / "semantic.json"
    source_path.write_text(
        json.dumps({"source_tuple_shard_ready_score": 1, "rows": [source]}),
        encoding="utf-8",
    )
    semantic_path.write_text(
        json.dumps({"asp_isomorphic_shard_ready_score": 1, "rows": semantic}),
        encoding="utf-8",
    )
    output = tmp_path / "artifact.json"
    artifact = exp.run(
        date="20260903",
        root=tmp_path,
        output_path=output,
        source_paths={"exp6913": Path("source.json"), "exp6914": Path("semantic.json")},
        expected_hashes={"exp6913": "sha256:wrong", "exp6914": "sha256:wrong"},
        expected_cell_ids={source["cell_identity"]},
        perturbations=PERTURBATIONS,
        verify_fn=lambda _path: {"loaded": True, "flags": []},
    )
    assert output.is_file()
    assert artifact["status"] == "blocked"
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    command = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/experiment_6915_qualified_relation_event_bank.py",
            "--help",
        ],
        cwd=exp.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert command.returncode == 0
    assert "--date" in command.stdout


def test_manifest_and_defensive_component_failures_remain_explicit() -> None:
    """REQ-CONSTRAINT-6915 exposes malformed components and incomplete cells."""

    identities = exp.expected_cell_identities()
    assert len(identities) == exp.EXPECTED_CELL_COUNT
    assert "unsloth/Qwen3.6-35B-A3B-GGUF::6899::graph_coloring_00" in identities
    assert "rule:anchored_lexical_v1::deterministic::scheduling_19" in identities

    source = _source_row()
    source["flagged_adversarial"] = True
    source["terminal"] = False
    assert {"source_component_flagged", "source_row_nonterminal"} <= set(
        exp._source_reasons(source)
    )
    assert exp._effect_projection(None) is None
    assert exp._expected_effect_matches({"expected_effect": None}, "base") is False
    assert (
        exp._expected_effect_matches(
            {"expected_effect": {}, "observed_effect": {}}, "unknown_perturbation"
        )
        is False
    )

    base = _semantic_row(_source_row(), "base")
    base.update(
        {
            "flagged_adversarial": True,
            "terminal": False,
            "transform_valid": False,
            "exact_atom_valid": False,
            "expected_effect_met": False,
        }
    )
    base_reasons = set(exp._semantic_reasons(base, "base"))
    assert {
        "semantic_component_flagged:base",
        "semantic_row_nonterminal:base",
        "transform_invalid:base",
        "exact_atom_invalid:base",
        "perturbation_behavior_failed:base",
    } <= base_reasons

    reversal = _semantic_row(_source_row(), "relation_reversal")
    reversal["compilation_status"] = "compiled"
    assert "perturbation_behavior_failed:relation_reversal" in exp._semantic_reasons(
        reversal, "relation_reversal"
    )

    missing = exp.build_join_evidence([], [], {"missing::1::fixture"}, PERTURBATIONS)
    assert missing["missing_join_rows"][0]["reason"] == "missing_source_join_row"
    assert "missing_source_join_row" in missing["join_rows"][0]["reasons"]
    incomplete = exp.derive_eligibility_rows(missing["join_rows"][:1], PERTURBATIONS)
    assert "incomplete_perturbation_join" in incomplete[0]["reasons"]


def test_validator_reports_each_terminal_integrity_failure() -> None:
    """SCENARIO-CONSTRAINT-6915-AGGREGATES rejects every terminal inconsistency."""

    admitted = _source_row()
    rejected = _source_row(model=MODEL_A, seed=12, fixture_id="graph_coloring_01", source_ok=False)
    artifact = _artifact(
        [admitted, rejected],
        [row for source in (admitted, rejected) for row in _semantic_rows(source)],
    )

    mutations: list[tuple[str, Any]] = [
        ("missing_fields:", lambda value: value.pop("random_seed")),
        ("field_principles", lambda value: value.update(field_principles={})),
        (
            "inference_substrate",
            lambda value: value.update(inference_substrate="not_the_frozen_substrate"),
        ),
        ("verifier_is_oracle", lambda value: value.update(verifier_is_oracle=False)),
        ("verdict_class", lambda value: value.update(verdict_class="positive")),
        ("positive_verdict_forbidden", lambda value: value.update(verdict_class="positive")),
        ("honest_verdict", lambda value: value.update(honest_verdict="nonterminal")),
        (
            "event_count_disagreement",
            lambda value: value.update(
                qualified_model_relation_event_count=value["qualified_model_relation_event_count"]
                + 1
            ),
        ),
        (
            "eligibility_inversion",
            lambda value: value["eligibility_rows"][0].update(eligible=False),
        ),
    ]
    for expected_error, mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert any(error.startswith(expected_error) for error in exp.validate_artifact(changed))

    checksum_drift = deepcopy(artifact)
    checksum_drift["reproducibility_checksum"] = "sha256:drift"
    assert "reproducibility_checksum" in exp.validate_artifact(checksum_drift)


def test_io_verifier_and_cli_failure_paths_are_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CONSTRAINT-6915 blocks read/verifier errors and validates before writing."""

    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="source_artifact_must_be_object"):
        exp._read_object(list_path)
    assert exp._safe_hash(tmp_path / "absent.json") == "missing"

    failed_report = exp._safe_verify(
        list_path, lambda _path: (_ for _ in ()).throw(RuntimeError("synthetic verifier error"))
    )
    assert failed_report["loaded"] is False
    assert "synthetic verifier error" in failed_report["error"]

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{", encoding="utf-8")
    output = tmp_path / "blocked.json"
    artifact = exp.run(
        date="20260903",
        root=tmp_path,
        output_path=output,
        source_paths={"exp6913": Path("missing.json"), "exp6914": Path("invalid.json")},
        expected_hashes={"exp6913": "sha256:a", "exp6914": "sha256:b"},
        expected_cell_ids=set(),
        verify_fn=lambda _path: {"loaded": True, "flags": []},
    )
    assert artifact["status"] == "blocked"
    failed_checks = set(artifact["preconditions_checked"]["failed_checks"])
    assert {"source_readable:exp6913", "source_readable:exp6914"} <= failed_checks

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["synthetic_failure"])
    with pytest.raises(RuntimeError, match="artifact_validation:synthetic_failure"):
        exp.run(
            date="20260903",
            root=tmp_path,
            output_path=tmp_path / "must_not_exist.json",
            source_paths={"exp6913": Path("missing.json"), "exp6914": Path("invalid.json")},
            expected_hashes={"exp6913": "sha256:a", "exp6914": "sha256:b"},
            expected_cell_ids=set(),
            verify_fn=lambda _path: {"loaded": True, "flags": []},
        )


def test_current_verifier_loader_and_main_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CONSTRAINT-6915 loads the current verifier and exposes deterministic CLI args."""

    blocked = exp.blocked_artifact(
        date="20260903",
        duration_s=0.1,
        source_artifact_hashes={},
        preconditions_checked=exp.gate_summary([exp.gate_check("synthetic", True, False)]),
        fresh_adversarial_rows=[],
    )
    artifact_path = tmp_path / "artifact.json"
    exp._write_json(artifact_path, blocked)
    assert isinstance(exp._load_current_verifier(str(artifact_path)), dict)

    monkeypatch.setattr(exp.importlib.util, "spec_from_file_location", lambda *_args: None)
    with pytest.raises(RuntimeError, match="adversarial_verifier_import_failed"):
        exp._load_current_verifier(str(artifact_path))

    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        exp,
        "run",
        lambda *, date, output_path: calls.append((date, output_path)),
    )
    assert exp.main(["--date", "20260903", "--output", str(tmp_path / "out.json")]) == 0
    assert calls == [("20260903", tmp_path / "out.json")]
