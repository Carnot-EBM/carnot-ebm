"""Tests for Exp6288 partial atom evidence adapter.

Spec refs: REQ-CONSTRAINT-6288,
SCENARIO-CONSTRAINT-6288-EXTRACT-FAIL-CLOSED,
SCENARIO-CONSTRAINT-6288-ORACLE-AFTER-EXTRACTION,
SCENARIO-CONSTRAINT-6288-WARM-CONTROLS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6288_partial_atom_evidence_adapter as exp6288


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"


def test_req_constraint_6288_spec_declares_adapter_contract() -> None:
    """REQ-CONSTRAINT-6288: OpenSpec anchors the adapter contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-CONSTRAINT-6288") :]

    for marker in (
        "SCENARIO-CONSTRAINT-6288-EXTRACT-FAIL-CLOSED",
        "SCENARIO-CONSTRAINT-6288-ORACLE-AFTER-EXTRACTION",
        "SCENARIO-CONSTRAINT-6288-WARM-CONTROLS",
        exp6288.RESULT_RELATIVE_PATH.as_posix(),
        "unsafe_evidence_acceptance_count",
        "source_model_weight_mutation_count",
        "positive warm-start delta",
    ):
        assert marker in section
    for field in exp6288.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_constraint_6288_extracts_positive_negative_unknown_atoms() -> None:
    """REQ-CONSTRAINT-6288: explicit atoms become positive, negative, or unknown."""

    evidence = exp6288.extract_partial_atom_evidence(
        "ANSWER: a. The label b is not selected.",
        ("a", "b", "c"),
        generated_token_count=8,
        row_id="unit",
    )

    assert evidence["accepted"] is True
    assert evidence["positive_atoms"] == ["a"]
    assert evidence["negative_atoms"] == ["b"]
    assert evidence["unknown_atoms"] == ["c"]
    assert evidence["rejection_reasons"] == []


def test_scenario_constraint_6288_atom_aliases_and_negation_scope() -> None:
    """SCENARIO-CONSTRAINT-6288-EXTRACT-FAIL-CLOSED: aliases and scope are strict."""

    alias = exp6288.extract_partial_atom_evidence(
        "Select color v1 blue, and do not select color-v2-red.",
        ("color_v1_blue", "color_v2_red", "color_v3_green"),
        generated_token_count=9,
        row_id="alias",
    )
    assert alias["accepted"] is True
    assert alias["positive_atoms"] == ["color_v1_blue"]
    assert alias["negative_atoms"] == ["color_v2_red"]
    assert alias["unknown_atoms"] == ["color_v3_green"]

    scoped = exp6288.extract_partial_atom_evidence(
        "a is not selected. b is selected.",
        ("a", "b"),
        generated_token_count=6,
        row_id="scope",
    )
    assert scoped["negative_atoms"] == ["a"]
    assert scoped["positive_atoms"] == ["b"]


@pytest.mark.parametrize(
    ("text", "tokens", "atoms", "reason"),
    (
        ("a is selected and a is not selected.", 7, ("a", "b"), "contradictory_evidence"),
        ("ANSWER: a, d", 3, ("a", "b"), "foreign_atom"),
        ("a may not be b.", 5, ("a", "b"), "ambiguous_negation"),
        ("", 1, ("a",), "empty_output"),
        ("ANSWER: a", 0, ("a",), "zero_token_row"),
    ),
)
def test_scenario_constraint_6288_rejects_unsafe_text(
    text: str,
    tokens: int,
    atoms: tuple[str, ...],
    reason: str,
) -> None:
    """SCENARIO-CONSTRAINT-6288-EXTRACT-FAIL-CLOSED: unsafe text fails closed."""

    evidence = exp6288.extract_partial_atom_evidence(
        text,
        atoms,
        generated_token_count=tokens,
        row_id="unsafe",
    )

    assert evidence["accepted"] is False
    assert reason in evidence["rejection_reasons"]


def test_scenario_constraint_6288_sidecar_support_runs_after_extraction() -> None:
    """SCENARIO-CONSTRAINT-6288-ORACLE-AFTER-EXTRACTION: support is post hoc."""

    evidence = exp6288.extract_partial_atom_evidence(
        "ANSWER: a, c.",
        ("a", "b", "c"),
        generated_token_count=4,
        row_id="oracle-after",
    )
    assert evidence["accepted"] is True
    assert evidence["sidecar_checked_after_extraction"] is False

    supported = exp6288.check_evidence_support(
        evidence,
        exact_answer_sets=[["a", "c"], ["b"]],
    )
    assert supported["supported"] is True
    assert supported["supporting_completion"] == ["a", "c"]
    assert supported["positive_correct_count"] == 2
    assert supported["positive_evidence_count"] == 2

    unsupported = exp6288.check_evidence_support(
        evidence,
        exact_answer_sets=[["b"]],
    )
    assert unsupported["supported"] is False
    assert "unsupported_by_exact_sidecar" in unsupported["rejection_reasons"]


def test_scenario_constraint_6288_fixed_budget_controls_share_budgets() -> None:
    """SCENARIO-CONSTRAINT-6288-WARM-CONTROLS: starts use matched budgets."""

    table = exp6288.table_from_program("1 { a; b } 1.\n", "unit_choice")
    evidence = exp6288.extract_partial_atom_evidence(
        "ANSWER: a.",
        table.atoms,
        generated_token_count=2,
        row_id="warm",
    )
    support = exp6288.check_evidence_support(evidence, exact_answer_sets=[["a"], ["b"]])
    starts = exp6288.compare_refinement_starts(
        table,
        evidence,
        support,
        row_id="warm",
        seed=6288,
    )

    assert set(starts) == {"evidence_warm", "blank", "random"}
    budgets = {arm["step_budget"] for arm in starts.values()} | {
        arm["restart_budget"] for arm in starts.values()
    }
    assert exp6288.OPTIMIZER_STEPS in budgets
    assert exp6288.RESTART_BUDGET in budgets
    assert starts["evidence_warm"]["known_true"] == ["a"]
    assert starts["blank"]["known_true"] == []
    assert starts["random"]["seed"] == 6288


def test_req_constraint_6288_artifact_schema_precision_and_controls(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6288: terminal artifact carries required fields and gates."""

    result_path = tmp_path / exp6288.RESULT_RELATIVE_PATH.name
    artifact = exp6288.run(
        date="20260810",
        result_path=result_path,
        duration_s=1.25,
        test_exit_codes={
            exp6288.RUN_COMMAND: 0,
            ".venv/bin/pytest tests/python/test_experiment_6288_partial_atom_evidence_adapter.py -q --no-cov -n 0": 0,
        },
        write=True,
    )

    assert result_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert set(exp6288.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp6288.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["status"] == "complete"
    assert artifact["partial_atom_evidence_adapter_ready_score"] == 1.0
    assert artifact["unsafe_evidence_acceptance_count"] == 0
    assert type(artifact["unsafe_evidence_acceptance_count"]) is int
    assert artifact["source_model_weight_mutation_count"] == 0
    assert type(artifact["source_model_weight_mutation_count"]) is int
    assert artifact["inference_substrate"] == exp6288.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["evidence_leakage_controls"]["extractor_reads_exact_sidecar"] is False
    assert artifact["evidence_leakage_controls"]["foreign_atom_control"]["rejected"] is True
    assert artifact["evidence_leakage_controls"]["label_leakage_control"]["passed"] is True
    assert artifact["accepted_and_rejected_row_counts"]["accepted_rows"] >= 1
    assert artifact["accepted_and_rejected_row_counts"]["rejected_rows"] >= 1
    assert artifact["accepted_and_rejected_row_counts"][
        "represented_model_families_with_acceptance"
    ]
    assert (
        artifact["continuous_refinement_results"]["fixed_budgets"]["steps"]
        == exp6288.OPTIMIZER_STEPS
    )
    assert artifact["exact_completion_results"]["accepted_exact_completion_count"] >= 1
    assert (
        artifact["cold_exact_completion_controls"]["blank"]["budget"]
        == exp6288.EXACT_COMPLETION_BUDGET
    )
    assert artifact["reproducibility_checksum"] == exp6288.payload_checksum(artifact)


def test_req_constraint_6288_validate_artifact_fails_closed(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6288: validator rejects false readiness and drift."""

    artifact = exp6288.run(
        date="20260810",
        result_path=tmp_path / "artifact.json",
        duration_s=0.5,
        write=False,
    )

    missing = deepcopy(artifact)
    missing.pop("adapter_source_paths_and_hashes")
    missing["reproducibility_checksum"] = exp6288.payload_checksum(missing)
    with pytest.raises(ValueError, match="adapter_source_paths_and_hashes"):
        exp6288.validate_artifact(missing)

    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = False
    bad_oracle["reproducibility_checksum"] = exp6288.payload_checksum(bad_oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        exp6288.validate_artifact(bad_oracle)

    bad_score = deepcopy(artifact)
    bad_score["unsafe_evidence_acceptance_count"] = 1
    bad_score["partial_atom_evidence_adapter_ready_score"] = 1.0
    bad_score["reproducibility_checksum"] = exp6288.payload_checksum(bad_score)
    with pytest.raises(ValueError, match="ready_score"):
        exp6288.validate_artifact(bad_score)

    bad_prefix = deepcopy(artifact)
    bad_prefix["honest_verdict"] = "ready"
    bad_prefix["reproducibility_checksum"] = exp6288.payload_checksum(bad_prefix)
    with pytest.raises(ValueError, match="honest_verdict"):
        exp6288.validate_artifact(bad_prefix)

    blocked = deepcopy(artifact)
    blocked["status"] = "blocked"
    blocked["evidence_precision_coverage_and_sample_sizes_by_model_family_and_fixture_family"] = (
        deepcopy(
            blocked[
                "evidence_precision_coverage_and_sample_sizes_by_model_family_and_fixture_family"
            ]
        )
    )
    blocked["evidence_precision_coverage_and_sample_sizes_by_model_family_and_fixture_family"][
        "overall"
    ]["positive_precision"] = 0.0
    blocked["partial_atom_evidence_adapter_ready_score"] = 0.0
    blocked["honest_verdict"] = exp6288._honest_verdict("blocked")
    blocked["reproducibility_checksum"] = exp6288.payload_checksum(blocked)
    exp6288.validate_artifact(blocked)


def test_req_constraint_6288_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6288: helper edge cases stay deterministic."""

    missing_raw = exp6288._extract_row("missing", {}, None, ("a",))
    assert missing_raw["accepted"] is False
    assert "missing_raw_sample" in missing_raw["rejection_reasons"]

    assert exp6288._cold_exact_completion_controls({})["blank"]["fixture_count"] == 0
    assert exp6288.model_family("unsloth/Qwen3.6-35B-A3B-GGUF") == "qwen3_6_35b_a3b"
    assert exp6288.model_family("other/model") == "unknown"
    assert exp6288._frozen_vocabularies({"bad": []}) == {}
    assert exp6288._honest_verdict("blocked").startswith("blocked:")

    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('\n{"ok": true}\n[]\n', encoding="utf-8")
    assert exp6288._read_jsonl(jsonl) == [{"ok": True}]

    root = tmp_path
    raw_dir = root / exp6288.RAW_DIR_RELATIVE_PATH
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "bad.raw.jsonl"
    raw_path.write_text(
        "\n".join(
            [
                json.dumps({"seed": "bad"}),
                json.dumps({"seed": 7, "model_hf_id": "m", "task_id": "t"}),
            ]
        ),
        encoding="utf-8",
    )
    lookup, receipts = exp6288._load_raw_sources(root)
    assert ("m", "t", 7) in lookup
    assert receipts["raw_outputs_by_model"]["bad.raw.jsonl"]["row_count"] == 2


def test_req_constraint_6288_cli_writes_requested_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CONSTRAINT-6288: CLI writes the requested terminal artifact."""

    result_path = tmp_path / "experiment_6288.json"
    assert exp6288.main(["--date", "20260810", "--result-path", str(result_path)]) == 0
    emitted = json.loads(capsys.readouterr().out)

    assert emitted["result"] == str(result_path)
    assert emitted["status"] == "complete"
    assert result_path.exists()
