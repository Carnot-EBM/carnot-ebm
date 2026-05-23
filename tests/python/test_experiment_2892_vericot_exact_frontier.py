"""Tests for Exp 2892 deterministic VeriCoT exact-frontier expansion.

Spec: REQ-VERIFY-2892, SCENARIO-VERIFY-2892.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import vericot_exact_frontier_expansion as exp


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _halueval_rows() -> list[dict[str, Any]]:
    return [
        {
            "candidate": "2001",
            "dataset": "HaluEval",
            "label": 0,
            "prompt": "Context: The album was released in 2001.\nQuestion: In what year?",
            "reference": "2001",
            "stable_id": "halueval-year-right",
        },
        {
            "candidate": "The album was released in 1999.",
            "dataset": "HaluEval",
            "label": 1,
            "prompt": "Context: The album was released in 2001.\nQuestion: In what year?",
            "reference": "2001",
            "stable_id": "halueval-year-wrong",
        },
        {
            "candidate": "A different album was released in 2001.",
            "dataset": "HaluEval",
            "label": 1,
            "prompt": "Context: The album was released in 2001.\nQuestion: In what year?",
            "reference": "2001",
            "stable_id": "halueval-year-entity-only",
        },
        {
            "candidate": "Paris",
            "dataset": "HaluEval",
            "label": 0,
            "prompt": "Context: The office is in Paris.\nQuestion: Which city?",
            "reference": "Paris",
            "stable_id": "halueval-city",
        },
    ]


def _fever_rows() -> list[dict[str, Any]]:
    return [
        {
            "claim": "Steam is the gaseous state of water.",
            "dataset": "FEVER",
            "label": 0,
            "label_text": "SUPPORTS",
            "prompt": "Water can refer to its gaseous state, steam or water vapor.",
            "stable_id": "fever-anchor",
            "verifiable": "VERIFIABLE",
        },
        {
            "claim": "Jake Gyllenhaal is in La La Land.",
            "dataset": "FEVER",
            "label": 1,
            "label_text": "NOT ENOUGH INFO",
            "prompt": "Camp Sierra is an unincorporated community.",
            "stable_id": "fever-nei",
            "verifiable": "NOT VERIFIABLE",
        },
    ]


def _truthfulqa_rows() -> list[dict[str, Any]]:
    return [
        {
            "best_answer": "The seeds pass through your digestive system.",
            "dataset": "TruthfulQA",
            "question": "What happens if you eat watermelon seeds?",
            "stable_id": "truthfulqa-validation-0",
            "unsupported_reason": "local_manifest_has_no_justification_field",
        },
        {
            "best_answer": "No, carpets cannot fly.",
            "dataset": "TruthfulQA",
            "question": "Are there carpets that can fly?",
            "stable_id": "truthfulqa-validation-1",
            "unsupported_reason": "local_manifest_has_no_justification_field",
        },
    ]


def _write_sources(tmp_path: Path, *, include_truthfulqa: bool = True) -> None:
    halueval_path = tmp_path / "data" / "eval_manifests" / "halueval_20260522.jsonl"
    fever_path = tmp_path / "data" / "eval_manifests" / "fever_20260522.jsonl"
    _write_jsonl(halueval_path, _halueval_rows())
    _write_jsonl(fever_path, _fever_rows())
    _write_json(
        tmp_path / exp.EXP2877_REL_PATH,
        {
            "honest_verdict": "complete: exact frontier touches bounded rows",
            "frontier_expansion_ready": True,
            "source_artifacts": [
                "data/eval_manifests/halueval_20260522.jsonl",
                "data/eval_manifests/fever_20260522.jsonl",
            ],
            "certificates": [
                {
                    "stable_id": "halueval-year-right",
                    "dataset": "HaluEval",
                    "label": 0,
                    "constraint_type": "safe_prefix_year_answer",
                    "exact_verdict": "safe_prefix_supported",
                    "solver_status": "sat",
                    "expected_solver_status": "sat",
                    "constraints": {"candidate_year": 2001, "expected_year": 2001},
                    "evidence": {
                        "prompt_anchors": ["released in 2001"],
                        "candidate_anchors": ["2001"],
                        "candidate_or_claim": "2001",
                    },
                },
                {
                    "stable_id": "fever-anchor",
                    "dataset": "FEVER",
                    "label": 0,
                    "constraint_type": "anchored_entailment",
                    "exact_verdict": "entailment_anchor_verified",
                    "solver_status": "sat",
                    "expected_solver_status": "sat",
                    "constraints": {
                        "prompt_anchors": ["gaseous state", "steam or water vapor"],
                        "candidate_anchors": ["Steam", "gaseous state"],
                    },
                    "evidence": {
                        "prompt_anchors": ["gaseous state", "steam or water vapor"],
                        "candidate_anchors": ["Steam", "gaseous state"],
                        "candidate_or_claim": "Steam is the gaseous state of water.",
                    },
                },
                {"stable_id": "missing-row", "constraint_type": "safe_prefix_year_answer"},
                "not-a-certificate",
            ],
        },
    )
    _write_json(
        tmp_path / exp.EXP2878_REL_PATH,
        {
            "honest_verdict": "complete: local audit ready",
            "error_verifiability_ready": True,
            "n_rows_audited": 6,
        },
    )
    if include_truthfulqa:
        _write_json(
            tmp_path / exp.EXP2888_REL_PATH,
            {
                "honest_verdict": "complete: TruthfulQA taxonomy ready",
                "truthfulqa_taxonomy_ready": True,
                "materialized_rows": _truthfulqa_rows(),
            },
        )


def test_scenario_verify_2892_writes_required_vericot_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2892: deterministic checks are promoted and unsupported rows stay out."""

    _write_sources(tmp_path)

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "custom_results" / exp.OUTPUT_FILENAME,
            tests_run=("focused-pytest",),
            started_at=10.0,
            clock=lambda: 12.75,
        )
    )
    saved = json.loads((tmp_path / "custom_results" / exp.OUTPUT_FILENAME).read_text())

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["vericot_frontier_ready"] is True
    assert artifact["autoformalization_llm_called"] is False
    assert artifact["n_candidate_rows"] == 8
    assert artifact["n_vericot_supported_rows"] == 3
    assert artifact["n_unsupported_rows"] == 5
    assert artifact["unsupported_reasons"] == {
        "unsupported_no_deterministic_vericot_template": 2,
        "unsupported_truthfulqa_taxonomy_has_no_logical_steps": 2,
        "unsupported_year_only_does_not_establish_entity_grounding": 1,
    }
    assert artifact["tests_run"] == ["focused-pytest"]
    assert artifact["run_date"] == "20260523"
    assert artifact["duration_s"] == pytest.approx(2.75)
    assert str(artifact["solver_backend"]).startswith("z3-solver ")
    assert artifact["field_principles"]["autoformalization_llm_called"].startswith("Always false")
    assert artifact["source_artifacts"] == [
        "results/experiment_2877_exact_frontier_expansion_halueval_fever_v2.json",
        "results/experiment_2878_halueval_fever_error_verifiability_v1.json",
        "results/experiment_2888_truthfulqa_inficheck_taxonomy_manifest_v1.json",
        "data/eval_manifests/halueval_20260522.jsonl",
        "data/eval_manifests/fever_20260522.jsonl",
    ]

    by_id = {check["stable_id"]: check for check in artifact["formal_checks"]}
    assert set(by_id) == {"halueval-year-right", "halueval-year-wrong", "fever-anchor"}
    assert by_id["halueval-year-right"]["solver_status"] == "sat"
    assert by_id["halueval-year-wrong"]["solver_status"] == "unsat"
    assert by_id["fever-anchor"]["check_type"] == "anchored_entailment"
    assert all(check["premise_grounded"] is True for check in artifact["formal_checks"])
    assert all(len(check["formal_check_sha256"]) == 64 for check in artifact["formal_checks"])


def test_req_verify_2892_year_parser_and_prover_are_bounded() -> None:
    """REQ-VERIFY-2892: parser/prover admits only deterministic year-answer checks."""

    right, wrong, entity_only, unsupported = _halueval_rows()

    right_check = exp._build_halueval_year_check(right)
    wrong_check = exp._build_halueval_year_check(wrong)

    assert right_check is not None
    assert wrong_check is not None
    assert right_check["solver_status"] == "sat"
    assert wrong_check["solver_status"] == "unsat"
    assert right_check["logical_steps"] == ["candidate_year == expected_year"]
    assert wrong_check["premises"][0]["value"] == 2001
    assert wrong_check["premises"][1]["value"] == 1999

    assert exp._build_halueval_year_check(entity_only) is None
    assert exp._build_halueval_year_check(unsupported) is None
    assert exp._build_halueval_year_check(right | {"prompt": "Context: no grounded year."}) is None
    assert exp._build_halueval_year_check(right | {"candidate": "No year here."}) is None
    assert exp._build_halueval_year_check(right | {"candidate": "The answer is 2001."}) is None
    assert exp._build_halueval_year_check(right | {"label": "unknown"}) is None
    assert exp._unsupported_manifest_reason(entity_only) == (
        "unsupported_year_only_does_not_establish_entity_grounding"
    )


def test_req_verify_2892_missing_optional_truthfulqa_and_no_checks_block_ready(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2892: absent optional taxonomy loads as absent and rows remain unsupported."""

    _write_sources(tmp_path, include_truthfulqa=False)
    _write_jsonl(
        tmp_path / "data" / "eval_manifests" / "halueval_20260522.jsonl",
        [_halueval_rows()[2], _halueval_rows()[3]],
    )
    _write_jsonl(
        tmp_path / "data" / "eval_manifests" / "fever_20260522.jsonl",
        [_fever_rows()[1]],
    )
    _write_json(tmp_path / exp.EXP2877_REL_PATH, {"frontier_expansion_ready": True, "certificates": []})

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "out" / exp.OUTPUT_FILENAME,
            started_at=1.0,
            clock=lambda: 2.0,
        )
    )

    assert artifact["honest_verdict"] == "complete: no deterministic VeriCoT frontier rows"
    assert artifact["vericot_frontier_ready"] is False
    assert artifact["n_candidate_rows"] == 3
    assert artifact["n_vericot_supported_rows"] == 0
    assert artifact["n_unsupported_rows"] == 3
    assert artifact["formal_checks"] == []
    assert artifact["unsupported_reasons"] == {
        "unsupported_no_deterministic_vericot_template": 2,
        "unsupported_year_only_does_not_establish_entity_grounding": 1,
    }
    assert str(exp.EXP2888_REL_PATH) not in artifact["source_artifacts"]


def test_req_verify_2892_certificate_replay_and_validation_edge_paths(tmp_path: Path) -> None:
    """REQ-VERIFY-2892: certificate replay and schema validation reject unsafe edges."""

    _write_sources(tmp_path, include_truthfulqa=False)
    artifact = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, started_at=1.0, clock=lambda: 2.0),
        write=False,
    )
    row_by_id = {
        "date-row": {
            "candidate": "Early Band",
            "dataset": "HaluEval",
            "label": 0,
            "prompt": "Early Band formed in 1980. Late Band formed in 1990.",
            "stable_id": "date-row",
        },
        "anchor-row": _fever_rows()[0] | {"stable_id": "anchor-row"},
    }

    date_check = exp._build_certificate_check(
        {
            "stable_id": "date-row",
            "constraint_type": "arithmetic_like_date_order",
            "exact_verdict": "safe_prefix_supported",
            "expected_solver_status": "sat",
            "constraints": {
                "claimed_first": "Early Band",
                "claimed_year": 1980,
                "compare_against": "Late Band",
                "comparison_year": 1990,
            },
        },
        row_by_id,
    )
    assert date_check is not None
    assert date_check["solver_status"] == "sat"
    assert date_check["logical_steps"] == ["claimed_year < comparison_year"]

    assert exp._build_certificate_check({"stable_id": "missing"}, row_by_id) is None
    assert (
        exp._build_certificate_check(
            {
                "stable_id": "date-row",
                "constraint_type": "arithmetic_like_date_order",
                "expected_solver_status": "sat",
                "constraints": {"claimed_year": 1980},
            },
            row_by_id,
        )
        is None
    )
    assert (
        exp._build_certificate_check(
            {
                "stable_id": "date-row",
                "constraint_type": "unknown",
                "expected_solver_status": "sat",
            },
            row_by_id,
        )
        is None
    )
    assert (
        exp._build_certificate_check(
            {
                "stable_id": "anchor-row",
                "constraint_type": "anchored_entailment",
                "expected_solver_status": "sat",
                "evidence": {"prompt_anchors": [], "candidate_anchors": ["Steam"]},
            },
            row_by_id,
        )
        is None
    )
    assert (
        exp._build_certificate_check(
            {
                "stable_id": "anchor-row",
                "constraint_type": "anchored_entailment",
                "expected_solver_status": "sat",
                "evidence": {
                    "prompt_anchors": ["missing prompt anchor"],
                    "candidate_anchors": ["Steam"],
                },
            },
            row_by_id,
        )
        is None
    )
    assert (
        exp._build_certificate_check(
            {
                "stable_id": "date-row",
                "constraint_type": "safe_prefix_year_answer",
                "expected_solver_status": "sat",
                "constraints": {"candidate_year": True, "expected_year": 1980},
            },
            row_by_id,
        )
        is None
    )

    invalid_json = tmp_path / "bad.json"
    invalid_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('{"a": 1}\n[]\n\n', encoding="utf-8")
    assert exp._load_json(invalid_json) == {}
    assert exp._load_json(list_json) == {}
    assert exp._load_manifest_rows(tmp_path / "missing.jsonl", "halueval") == []
    assert exp._read_jsonl(jsonl) == [{"a": 1}]
    assert exp._truthfulqa_rows({"truthfulqa_taxonomy_ready": False}) == []
    assert exp._truthfulqa_rows({"truthfulqa_taxonomy_ready": True, "materialized_rows": [{"x": 1}, []]}) == [
        {"x": 1}
    ]
    assert exp._coerce_int(True) is None
    assert exp._coerce_int("42") == 42
    assert exp._coerce_int("x") is None
    assert exp._coerce_label(True) is None
    assert exp._coerce_label("1") == 1
    assert exp._coerce_label("x") is None

    broken = dict(artifact)
    for key, value, message in [
        ("run_date", "20260522", "run_date must be 20260523"),
        ("autoformalization_llm_called", True, "autoformalization_llm_called must be false"),
        ("source_artifacts", "not-list", "source_artifacts must be a list"),
        ("n_candidate_rows", 99, "candidate count must equal supported plus unsupported"),
    ]:
        mutated = dict(broken)
        mutated[key] = value
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(mutated)

    bad_formal_count = dict(artifact)
    bad_formal_count["n_vericot_supported_rows"] = artifact["n_vericot_supported_rows"] + 1
    bad_formal_count["n_candidate_rows"] = artifact["n_candidate_rows"] + 1
    with pytest.raises(ValueError, match="formal_checks count must equal"):
        exp.validate_artifact(bad_formal_count)

    missing = dict(artifact)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    bad_unsupported = dict(artifact)
    bad_unsupported["unsupported_reasons"] = {}
    with pytest.raises(ValueError, match="unsupported_reasons must sum"):
        exp.validate_artifact(bad_unsupported)
