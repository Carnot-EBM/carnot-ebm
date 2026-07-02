"""Tests for Exp 5138 exact-validator energy-guided decoding gate.

Spec refs: REQ-PIPELINE-5138,
SCENARIO-PIPELINE-5138.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5136_receipt_structured_pool_v2_v471 as pool_mod
from carnot import experiment_5138_ets_ebd_guided_decoding_v471 as mod
from scripts import experiment_5138_ets_ebd_guided_decoding_v471 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/pipeline/spec.md"


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


def _write_ready_upstream(root: Path, rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    source_rows = rows
    if source_rows is None:
        tasks = [
            task
            for task in pool_mod.build_task_bank()
            if str(task["family"]) in mod.GUIDABLE_FAMILIES
        ][:9]
        source_rows, _ = pool_mod.build_pool_rows(tasks, _fake_specs(), run_date="20260702")
    pool_path = root / pool_mod.POOL_RELATIVE_PATH
    pool_mod.write_jsonl(pool_path, source_rows)
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
        "pool_n": len(source_rows),
        "fover_scope_used": False,
        "conductor_modified": False,
        "tests_run": ["fixture"],
    }
    pool_mod.write_json(root / pool_mod.RESULT_RELATIVE_PATH, artifact)
    return source_rows


def test_req_pipeline_5138_spec_declares_guided_decoding_gate() -> None:
    """REQ-PIPELINE-5138: OpenSpec declares the Exp 5138 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-PIPELINE-5138")
    end = spec.index("### REQ-PIPELINE-1677", start)
    section = spec[start:end]

    assert "SCENARIO-PIPELINE-5138" in section
    assert mod.EXPERIMENT_ID in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_clean_pool_without_stepwise_telemetry_blocks_guided_claim(tmp_path: Path) -> None:
    """SCENARIO-PIPELINE-5138: complete-candidate pool cannot masquerade as guided decoding."""

    rows = _write_ready_upstream(tmp_path)

    artifact = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["honest_verdict"] == mod.BLOCKED_TELEMETRY_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(143.366125)
    assert artifact["MODEL_SPECS"] == _fake_specs()
    assert artifact["model_specs"] == artifact["MODEL_SPECS"]
    assert artifact["upstream_pool_artifact"] == pool_mod.RESULT_RELATIVE_PATH
    assert artifact["selected_task_families"] == list(mod.GUIDABLE_FAMILIES)
    assert artifact["selected_task_count"] == len(rows)
    assert artifact["exact_validator_authority"]["authority_intact"] is True
    assert artifact["exact_validator_authority"]["llm_judge_used_as_ground_truth"] is False
    assert artifact["controls_differentiated"] is False
    assert artifact["guided_decoding_ready"] is False
    assert artifact["guided_decoding_delta"] is None
    assert artifact["delta_ci95"] == [None, None]
    assert artifact["violation_rate_delta"] is None
    assert artifact["rerank_only_control"]["arm"] == "best_of_n_reranking"
    assert artifact["control_metrics"]["unguided_generation"]["task_count"] == len(rows)
    assert artifact["control_metrics"]["best_of_n_reranking"]["validator_calls"] == len(rows) * 4
    assert artifact["token_nfe_accounting"]["guided_decoding"]["executed"] is False
    assert artifact["logprob_or_blocker_evidence"]["has_required_stepwise_telemetry"] is False
    assert artifact["logprob_or_blocker_evidence"]["candidate_pool_has_only_completed_outputs"] is True
    assert artifact["conductor_modified"] is False
    assert artifact["tests_run"] == ["focused"]

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact


def test_dirty_or_missing_upstream_blocks_before_controls(tmp_path: Path) -> None:
    """SCENARIO-PIPELINE-5138: closed Exp 5136 gate writes a terminal blocked artifact."""

    missing = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    mod.validate_artifact(missing)
    assert missing["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert missing["control_metrics"] == {}
    assert "missing upstream" in missing["preconditions_checked"]["upstream_error"]

    pool_mod.write_json(
        tmp_path / pool_mod.RESULT_RELATIVE_PATH,
        {
            "experiment_id": pool_mod.EXPERIMENT_ID,
            "structured_pool_v2_clean": False,
            "pool_path": pool_mod.POOL_RELATIVE_PATH,
            "MODEL_SPECS": _fake_specs(),
            "duration_s": 12.0,
        },
    )
    dirty = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    mod.validate_artifact(dirty)
    assert dirty["honest_verdict"] == mod.BLOCKED_POOL_VERDICT
    assert dirty["preconditions_checked"]["structured_pool_v2_clean"] is False
    assert dirty["guided_decoding_ready"] is False


def test_rows_model_specs_and_cli_edges_for_req_pipeline_5138(tmp_path: Path) -> None:
    """REQ-PIPELINE-5138: rows, model provenance, and CLI paths stay deterministic."""

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

    _write_ready_upstream(tmp_path)
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
    assert blocked_model["honest_verdict"] == mod.BLOCKED_MODEL_VERDICT

    _write_ready_upstream(tmp_path)
    artifact = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(
            {key: value for key, value in artifact.items() if key != "duration_s"}
        )
    with pytest.raises(ValueError, match="guided_decoding_ready"):
        mod.validate_artifact(artifact | {"guided_decoding_ready": True})

    assert script_mod.main(["--root", str(tmp_path), "--date", "20260702"]) == 0
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact | {"experiment_id": "bad"}, "experiment_id"),
        (lambda artifact: artifact | {"milestone": "2026.07.470"}, "milestone"),
        (lambda artifact: artifact | {"honest_verdict": "bad"}, "honest_verdict"),
        (lambda artifact: artifact | {"inference_substrate": "bad"}, "substrate"),
        (lambda artifact: artifact | {"MODEL_SPECS": []}, "mandated"),
        (lambda artifact: artifact | {"model_specs": []}, "mandated"),
        (lambda artifact: artifact | {"upstream_pool_artifact": "bad.json"}, "upstream"),
        (lambda artifact: artifact | {"conductor_modified": True}, "conductor_modified"),
        (lambda artifact: artifact | {"tests_run": []}, "tests_run"),
        (lambda artifact: artifact | {"exact_validator_authority": {}}, "authority"),
        (lambda artifact: artifact | {"controls_differentiated": True}, "controls"),
        (lambda artifact: artifact | {"rerank_only_control": {}}, "rerank"),
        (lambda artifact: artifact | {"token_nfe_accounting": {}}, "token_nfe"),
        (lambda artifact: artifact | {"guided_decoding_delta": 0.0}, "guided decoding delta"),
        (lambda artifact: artifact | {"delta_ci95": [0.0, 0.0]}, "guided decoding CI"),
        (lambda artifact: artifact | {"violation_rate_delta": 0.0}, "violation-rate delta"),
        (
            lambda artifact: artifact | {"logprob_or_blocker_evidence": {}},
            "blocker evidence",
        ),
    ],
)
def test_artifact_validator_rejects_schema_drift_for_req_pipeline_5138(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    """REQ-PIPELINE-5138: artifact validation rejects schema and gate drift."""

    _write_ready_upstream(tmp_path)
    artifact = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_control_accounting_and_telemetry_helpers_for_req_pipeline_5138() -> None:
    """REQ-PIPELINE-5138: control metrics and telemetry checks are explicit."""

    rows = [
        {
            "task_id": "graph_coloring_000",
            "family": "graph_coloring",
            "validator": "graph_coloring",
            "candidates": [
                {"candidate_id": "a", "raw_response": '{"answer": [0]}', "correct": False},
                {"candidate_id": "b", "raw_response": '{"answer": [1]}', "correct": True},
            ],
        },
        {
            "task_id": "travel_budget_000",
            "family": "travel_budget",
            "validator": "travel_budget",
            "candidates": [
                {"candidate_id": "c", "raw_response": '{"answer": []}', "correct": False},
                {
                    "candidate_id": "d",
                    "raw_response": '{"answer": ["museum"]}',
                    "correct": False,
                },
            ],
        },
    ]

    selected = mod.select_guidable_rows(rows, per_family=2)
    capped = mod.select_guidable_rows(
        [
            *rows,
            {"task_id": "ignored", "family": "knights_knaves", "candidates": []},
            {"task_id": "graph_coloring_001", "family": "graph_coloring", "candidates": []},
        ],
        per_family=1,
    )
    controls = mod.evaluate_controls(selected)
    telemetry = mod.inspect_stepwise_telemetry(selected)

    assert [row["task_id"] for row in selected] == ["graph_coloring_000", "travel_budget_000"]
    assert [row["task_id"] for row in capped] == ["graph_coloring_000", "travel_budget_000"]
    assert controls["unguided_generation"]["exact_validator_success"] == 0.0
    assert controls["best_of_n_reranking"]["exact_validator_success"] == 0.5
    assert controls["fixed_token_reranking"]["token_budget_per_task"] == mod.FIXED_TOKEN_BUDGET
    assert controls["guided_decoding"]["blocked_reason"] == mod.STEPWISE_TELEMETRY_BLOCKER
    assert telemetry["has_required_stepwise_telemetry"] is False
    assert telemetry["rows_with_token_logprobs"] == 0
    assert telemetry["candidate_pool_has_only_completed_outputs"] is True
    assert mod._paired_delta_ci95([], []) == [0.0, 0.0]
    assert mod._paired_delta_ci95([True, True], [False, True]) == [-0.19, 1.0]
    assert mod._estimated_tokens("") == 1

    malformed_controls = mod.evaluate_controls(
        [
            {"task_id": "empty", "family": "graph_coloring", "candidates": []},
            {"task_id": "bad", "family": "travel_budget", "candidates": "not-list"},
        ]
    )
    assert malformed_controls["best_of_n_reranking"]["exact_validator_success"] == 0.0
    assert malformed_controls["fixed_token_reranking"]["validator_calls"] == 0
