"""Tests for Exp 5137 solver-verified formulation selector.

Spec refs: REQ-INFER-SOTA-033,
SCENARIO-INFER-SOTA-033-SELECTOR,
SCENARIO-INFER-SOTA-033-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5136_receipt_structured_pool_v2_v471 as pool_mod
from carnot import experiment_5137_solver_verified_formulation_selector_v471 as mod
from scripts import experiment_5137_solver_verified_formulation_selector_v471 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/llm-ebm-inference/spec.md"


def _fake_specs() -> list[dict[str, object]]:
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "loader": "llama.cpp",
            "model_path": "/models/qwen3.6-35b-a3b-q4.gguf",
        },
        {
            "name": "Gemma4-31B-it",
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "gpu": None,
            "loader": "llama.cpp",
            "model_path": "/models/gemma-4-31b-q4.gguf",
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "gpu": 1,
            "loader": "llama.cpp",
            "model_path": "/models/gemma-4-26b-a4b-q4.gguf",
        },
    ]


def _write_ready_upstream(root: Path) -> list[dict[str, Any]]:
    rows, receipts = pool_mod.build_pool_rows(
        pool_mod.build_task_bank(), _fake_specs(), run_date="20260702"
    )
    pool_path = root / pool_mod.POOL_RELATIVE_PATH
    pool_mod.write_jsonl(pool_path, rows)
    artifact = {
        "experiment_id": pool_mod.EXPERIMENT_ID,
        "milestone": pool_mod.MILESTONE,
        "honest_verdict": pool_mod.SUCCESS_VERDICT,
        "inference_substrate": pool_mod.INFERENCE_SUBSTRATE,
        "duration_s": 143.366125,
        "MODEL_SPECS": _fake_specs(),
        "model_specs": _fake_specs(),
        "structured_pool_v2_clean": True,
        "pool_path": pool_mod.POOL_RELATIVE_PATH,
        "pool_sha256": pool_mod.sha256_file(pool_path),
        "pool_n": len(rows),
        "receipt_records": receipts,
        "receipt_record_count": len(receipts),
        "fover_scope_used": False,
        "conductor_modified": False,
        "tests_run": ["fixture"],
    }
    pool_mod.write_json(root / pool_mod.RESULT_RELATIVE_PATH, artifact)
    return rows


def test_req_infer_sota_033_spec_declares_selector_contract() -> None:
    """REQ-INFER-SOTA-033: OpenSpec declares the Exp 5137 selector contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-INFER-SOTA-033")
    end = spec.index("### REQ-INFER-SOTA-032", start)
    section = spec[start:end]

    assert "SCENARIO-INFER-SOTA-033-SELECTOR" in section
    assert "SCENARIO-INFER-SOTA-033-BLOCKED" in section
    assert mod.EXPERIMENT_ID in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.PMC_RECORDS_RELATIVE_PATH in section
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_solver_records_restore_feasibility_without_hiding_original_quality() -> None:
    """SCENARIO-INFER-SOTA-033-SELECTOR: solver feedback is exact and auditable."""

    tasks = mod.select_formulation_tasks(pool_mod.build_task_bank(), per_family=1)
    rows, _ = pool_mod.build_pool_rows(tasks, _fake_specs(), run_date="20260702")
    direct_answers = mod.direct_answer_rows_by_task(rows)
    records = mod.build_pmc_records(tasks, _fake_specs())
    evaluation = mod.evaluate_selector(tasks, records, direct_answers)

    assert {task["family"] for task in tasks} == set(mod.EXACT_FORMULATION_TASK_FAMILIES)
    assert len(records) == len(tasks) * len(_fake_specs()) * len(mod.FORMULATION_FAMILIES)
    assert all(record["exact_post_check_passed"] for record in records)
    assert any(record["feasibility_restored"] for record in records)
    assert (
        evaluation["original_model_quality"]["exact_correct_rate"] < evaluation["feasibility_rate"]
    )
    assert evaluation["feasibility_rate"] == 1.0
    assert evaluation["wrong_label_count"] == 0
    assert evaluation["selector_delta_vs_best_static"] == 0.0
    assert evaluation["delta_ci95"] == [0.0, 0.0]
    assert evaluation["formulation_selector_ready"] is False
    assert evaluation["baseline_metrics"]["static_hand_formulation"]["accuracy_at_1"] == 1.0
    assert evaluation["baseline_metrics"]["direct_answer"]["accuracy_at_1"] < 1.0
    assert evaluation["solve_effort_delta"]["delta_units"] > 0


def test_write_artifact_emits_required_schema_and_pmc_records(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-033: writer emits selector metrics and PMC evidence."""

    _write_ready_upstream(tmp_path)

    artifact = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["honest_verdict"] == mod.SUCCESS_NOT_READY_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(143.366125)
    assert artifact["MODEL_SPECS"] == _fake_specs()
    assert artifact["model_specs"] == artifact["MODEL_SPECS"]
    assert artifact["upstream_pool_artifact"] == pool_mod.RESULT_RELATIVE_PATH
    assert artifact["formulation_families"] == list(mod.FORMULATION_FAMILIES)
    assert artifact["pmc_records_path"] == mod.PMC_RECORDS_RELATIVE_PATH
    assert artifact["solver_backend"] == mod.SOLVER_BACKEND
    assert artifact["feasibility_restoration_used"] is True
    assert artifact["selector_delta_vs_best_static"] == 0.0
    assert artifact["wrong_label_count"] == 0
    assert artifact["formulation_selector_ready"] is False
    assert artifact["fover_scope_used"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["tests_run"] == ["focused"]
    assert set(artifact["family_holdout_behavior"]) == set(mod.EXACT_FORMULATION_TASK_FAMILIES)

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    pmc_path = tmp_path / mod.PMC_RECORDS_RELATIVE_PATH
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert len(mod.read_jsonl(pmc_path)) == artifact["pmc_record_count"]
    assert artifact["pmc_records_sha256"] == mod.sha256_file(pmc_path)


def test_dirty_structured_pool_gate_blocks_without_pmc_rows(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-033-BLOCKED: dirty upstream gate fails closed."""

    pool_mod.write_json(
        tmp_path / pool_mod.RESULT_RELATIVE_PATH,
        {
            "experiment_id": pool_mod.EXPERIMENT_ID,
            "structured_pool_v2_clean": False,
            "pool_path": pool_mod.POOL_RELATIVE_PATH,
            "MODEL_SPECS": _fake_specs(),
            "duration_s": 143.0,
        },
    )

    artifact = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.BLOCKED_POOL_VERDICT
    assert artifact["formulation_selector_ready"] is False
    assert artifact["pmc_records_path"] is None
    assert artifact["pmc_record_count"] == 0
    assert not (tmp_path / mod.PMC_RECORDS_RELATIVE_PATH).exists()
    assert artifact["preconditions_checked"]["structured_pool_v2_clean"] is False


def test_missing_rows_validation_and_cli_edges_for_req_infer_sota_033(
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-033: missing rows, validation, and CLI paths stay deterministic."""

    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.read_jsonl(tmp_path / "missing.jsonl") == []
    missing = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    assert missing["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert "missing upstream" in missing["preconditions_checked"]["upstream_error"]

    upstream_path = tmp_path / pool_mod.RESULT_RELATIVE_PATH
    upstream_path.parent.mkdir(parents=True, exist_ok=True)
    upstream_path.write_text("{", encoding="utf-8")
    malformed = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    assert malformed["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert "JSONDecodeError" in malformed["preconditions_checked"]["upstream_error"]

    upstream_path.write_text("[]", encoding="utf-8")
    non_object = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    assert non_object["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert "not a JSON object" in non_object["preconditions_checked"]["upstream_error"]

    _write_ready_upstream(tmp_path)
    (tmp_path / pool_mod.POOL_RELATIVE_PATH).unlink()
    missing_rows = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    assert missing_rows["honest_verdict"] == mod.BLOCKED_ROWS_VERDICT

    rows = _write_ready_upstream(tmp_path)
    pool_mod.write_json(
        tmp_path / pool_mod.RESULT_RELATIVE_PATH,
        {
            "experiment_id": pool_mod.EXPERIMENT_ID,
            "structured_pool_v2_clean": True,
            "pool_path": pool_mod.POOL_RELATIVE_PATH,
            "MODEL_SPECS": _fake_specs()[:2],
            "duration_s": 143.0,
        },
    )
    blocked_model = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    assert rows
    assert blocked_model["honest_verdict"] == mod.BLOCKED_MODEL_VERDICT

    _write_ready_upstream(tmp_path)
    artifact = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(
            {key: value for key, value in artifact.items() if key != "duration_s"}
        )
    with pytest.raises(ValueError, match="wrong labels"):
        mod.validate_artifact(artifact | {"wrong_label_count": 1})
    with pytest.raises(ValueError, match="ready gate"):
        mod.validate_artifact(
            artifact | {"formulation_selector_ready": True, "selector_delta_vs_best_static": 0.0}
        )

    assert script_mod.main(["--root", str(tmp_path), "--date", "20260702"]) == 0
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_private_edge_helpers_for_req_infer_sota_033() -> None:
    """REQ-INFER-SOTA-033: parser and CI helper edge cases remain deterministic."""

    tasks = {str(task["family"]): task for task in pool_mod.build_task_bank()}

    assert mod._as_int_list("not-list") is None
    assert mod._as_int_list([True]) is None
    assert mod._as_str_list("not-list") is None
    assert mod._feasible_without_optimality(tasks["travel_budget"], "not-list") is False
    assert mod._feasible_without_optimality(tasks["travel_budget"], ["missing"]) is False
    assert mod._feasible_without_optimality(tasks["code_property"], "not-list") is False
    assert mod._feasible_without_optimality(tasks["or_allocation"], "not-list") is False
    assert mod._feasible_without_optimality(tasks["or_allocation"], [-1, 0, 0]) is False
    assert mod._paired_delta_ci95([], []) == [0.0, 0.0]
    assert mod._paired_delta_ci95([True, True, False], [False, False, False]) == [0.0, 1.0]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact | {"experiment_id": "bad"}, "experiment_id"),
        (lambda artifact: artifact | {"milestone": "2026.07.470"}, "milestone"),
        (lambda artifact: artifact | {"honest_verdict": "bad"}, "honest_verdict"),
        (lambda artifact: artifact | {"inference_substrate": "bad"}, "substrate"),
        (lambda artifact: artifact | {"model_specs": []}, "model_specs"),
        (lambda artifact: artifact | {"fover_scope_used": True}, "fover_scope_used"),
        (lambda artifact: artifact | {"conductor_modified": True}, "conductor_modified"),
        (lambda artifact: artifact | {"tests_run": []}, "tests_run"),
        (
            lambda artifact: (
                artifact
                | {
                    "MODEL_SPECS": artifact["MODEL_SPECS"][:2],
                    "model_specs": artifact["MODEL_SPECS"][:2],
                }
            ),
            "mandated",
        ),
        (lambda artifact: artifact | {"pmc_records_path": "bad.jsonl"}, "pmc_records_path"),
        (lambda artifact: artifact | {"pmc_record_count": 0}, "PMC records"),
        (lambda artifact: artifact | {"pmc_records_sha256": None}, "hash PMC"),
        (lambda artifact: artifact | {"solver_backend": "bad"}, "solver backend"),
        (lambda artifact: artifact | {"formulation_families": []}, "formulation families"),
    ],
)
def test_artifact_validator_rejects_schema_drift_for_req_infer_sota_033(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    """REQ-INFER-SOTA-033: complete artifacts reject schema and gate drift."""

    _write_ready_upstream(tmp_path)
    artifact = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact | {"pmc_records_path": mod.PMC_RECORDS_RELATIVE_PATH}, "PMC"),
        (lambda artifact: artifact | {"pmc_record_count": 1}, "pmc_record_count"),
    ],
)
def test_blocked_artifact_validator_rejects_fabricated_pmc_for_req_infer_sota_033(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    """SCENARIO-INFER-SOTA-033-BLOCKED: blocked artifacts cannot carry PMC evidence."""

    blocked = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(blocked))
