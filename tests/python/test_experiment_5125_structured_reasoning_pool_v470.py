"""Tests for Exp 5125 structured reasoning candidate pool.

Spec refs: REQ-INFER-SOTA-030,
SCENARIO-INFER-SOTA-030-POOL,
SCENARIO-INFER-SOTA-030-BLOCKED.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5125_structured_reasoning_pool_v470 as mod
from scripts import experiment_5125_structured_reasoning_pool_v470 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/llm-ebm-inference/spec.md"


def _fake_pair(**_: object) -> list[dict[str, object]]:
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": "/models/qwen.gguf",
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "gpu": 1,
            "model_path": "/models/gemma.gguf",
        },
    ]


def _clean_upstream() -> dict[str, object]:
    return {
        "experiment_id": "exp5124-clean-sota-runtime-provenance-v470",
        "sota_runtime_clean": True,
        "MODEL_SPECS": _fake_pair(),
    }


def _write_upstream(root: Path, payload: dict[str, object] | None = None) -> None:
    path = root / mod.UPSTREAM_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload if payload is not None else _clean_upstream()), encoding="utf-8"
    )


def test_req_infer_sota_030_spec_declares_pool_contract() -> None:
    """REQ-INFER-SOTA-030: OpenSpec declares the non-FoVer structured pool."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-INFER-SOTA-030")
    end = spec.index("### REQ-INFER-018", start)
    section = spec[start:end]

    assert "SCENARIO-INFER-SOTA-030-POOL" in section
    assert "SCENARIO-INFER-SOTA-030-BLOCKED" in section
    assert mod.EXPERIMENT_ID in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.POOL_RELATIVE_PATH in section
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_task_bank_validators_accept_gold_and_reject_wrong_answers() -> None:
    """SCENARIO-INFER-SOTA-030-POOL: deterministic validators define ground truth."""

    tasks = mod.build_task_bank()
    families = {task["family"] for task in tasks}

    assert len(tasks) == 96
    assert families == {"code_property", "graph_coloring", "knights_knaves", "travel_budget"}
    assert not any("fover" in json.dumps(task).lower() for task in tasks)

    seen_families: set[str] = set()
    for task in tasks:
        if task["family"] in seen_families:
            continue
        seen_families.add(str(task["family"]))
        correct = json.dumps({"answer": mod.correct_answer(task)})
        wrong = json.dumps({"answer": mod.wrong_answer(task, variant=0)})

        correct_score = mod.score_candidate(task, correct)
        wrong_score = mod.score_candidate(task, wrong)

        assert correct_score["parse_ok"] is True
        assert correct_score["correct"] is True
        assert wrong_score["parse_ok"] is True
        assert wrong_score["correct"] is False

    assert seen_families == families


def test_pool_metrics_show_non_saturated_headroom_and_parse_coverage() -> None:
    """SCENARIO-INFER-SOTA-030-POOL: oracle headroom beats the cheap first candidate."""

    tasks = mod.build_task_bank()
    specs = mod.resolve_model_specs(_clean_upstream(), cached_pair_fn=_fake_pair)
    rows = mod.build_pool_rows(tasks, specs)
    metrics = mod.compute_pool_metrics(rows)

    assert len(rows) == 96
    assert all(len(row["candidates"]) == mod.CANDIDATES_PER_ITEM for row in rows)
    assert 0.80 <= metrics["oracle_at_k"] < 0.95
    assert 0.20 <= metrics["cheap_baseline_at_1"] < 0.40
    assert metrics["oracle_at_k"] > metrics["cheap_baseline_at_1"]
    assert metrics["parse_coverage"] >= mod.PARSE_COVERAGE_GATE
    assert 0.0 < metrics["duplicate_rate"] < 0.05
    assert 0.0 < metrics["copy_rate"] < 0.05
    assert all(value["headroom"] > 0 for value in metrics["task_family_headroom"].values())

    candidate_model_paths = {
        candidate["model_path"] for row in rows for candidate in row["candidates"]
    }
    assert candidate_model_paths == {"/models/qwen.gguf", "/models/gemma.gguf"}


def test_write_artifact_emits_schema_valid_pool_for_req_infer_sota_030(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-030: writer emits artifact plus hashed pool data path."""

    _write_upstream(tmp_path)

    artifact = mod.write_artifact(
        root=tmp_path,
        duration_s=1.25,
        run_date="20260701",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["honest_verdict"] == mod.SUCCESS_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["structured_pool_ready"] is True
    assert artifact["pool_n"] == 96
    assert artifact["candidates_per_item"] == mod.CANDIDATES_PER_ITEM
    assert artifact["verifier_is_oracle"] is False
    assert artifact["fover_scope_used"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["tests_run"] == ["focused"]

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    pool_path = tmp_path / mod.POOL_RELATIVE_PATH
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert pool_path.exists()
    assert artifact["pool_sha256"] == mod.sha256_file(pool_path)
    assert len(mod.read_jsonl(pool_path)) == 96


def test_dirty_exp5124_gate_blocks_without_candidate_rows(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-030-BLOCKED: dirty upstream runtime gate blocks the pool."""

    _write_upstream(
        tmp_path,
        {
            "experiment_id": "exp5124-clean-sota-runtime-provenance-v470",
            "sota_runtime_clean": False,
            "MODEL_SPECS": _fake_pair(),
        },
    )

    artifact = mod.write_artifact(
        root=tmp_path,
        duration_s=0.5,
        run_date="20260701",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.BLOCKED_GATE_VERDICT
    assert artifact["structured_pool_ready"] is False
    assert artifact["pool_n"] == 0
    assert artifact["pool_path"] is None
    assert artifact["pool_sha256"] is None
    assert not (tmp_path / mod.POOL_RELATIVE_PATH).exists()
    assert artifact["preconditions_checked"]["exp5124_sota_runtime_clean"] is False


def test_missing_mandated_model_path_blocks_for_req_infer_sota_030(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-030: MODEL_SPECS must include a mandated local GGUF path."""

    _write_upstream(tmp_path, _clean_upstream() | {"MODEL_SPECS": []})

    artifact = mod.write_artifact(
        root=tmp_path,
        duration_s=0.5,
        run_date="20260701",
        tests_run=["focused"],
        cached_pair_fn=lambda **_: [],
    )

    assert artifact["honest_verdict"] == mod.BLOCKED_MODEL_VERDICT
    assert artifact["structured_pool_ready"] is False
    assert artifact["pool_n"] == 0
    mod.validate_artifact(artifact)


def test_malformed_inputs_and_private_guards_for_req_infer_sota_030(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFER-SOTA-030: parser and exact-validator error branches stay deterministic."""

    tasks = {task["family"]: task for task in mod.build_task_bank()}

    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.read_jsonl(tmp_path / "missing.jsonl") == []
    assert mod._read_json(tmp_path / "missing.json") is None

    with pytest.raises(ValueError, match="unknown statement kind"):
        mod._statement_truth({"kind": "unknown"}, {"A": True})
    assert (
        mod._statement_text({"kind": "same_as", "left": "A", "right": "B"})
        == "A and B are the same type"
    )
    with pytest.raises(ValueError, match="unknown statement kind"):
        mod._statement_text({"kind": "unknown"})
    with pytest.raises(ValueError, match="unknown task family"):
        mod.wrong_answer({"family": "unknown"}, variant=0)

    graph_score = mod.score_candidate(tasks["graph_coloring"], json.dumps({"answer": [True]}))
    knight_score = mod.score_candidate(
        tasks["knights_knaves"],
        json.dumps({"answer": {"A": "maybe", "B": "knight", "C": "knave"}}),
    )
    missing_answer_score = mod.score_candidate(
        tasks["code_property"], json.dumps({"not_answer": []})
    )
    travel_score = mod.score_candidate(tasks["travel_budget"], json.dumps({"answer": ["missing"]}))

    assert graph_score["parse_ok"] is True
    assert graph_score["correct"] is False
    assert knight_score["parse_ok"] is True
    assert knight_score["correct"] is False
    assert missing_answer_score["parse_ok"] is False
    assert "answer key" in missing_answer_score["error"]
    assert travel_score["parse_ok"] is True
    assert travel_score["correct"] is False

    fallback_calls = {"count": 0}

    def no_unique_assignments(names: object, statements: object) -> list[dict[str, bool]]:
        fallback_calls["count"] += 1
        return []

    monkeypatch.setattr(mod, "_valid_knights_assignments", no_unique_assignments)
    fallback_task = mod._knights_task(0)
    assert fallback_calls["count"] > 0
    assert fallback_task["family"] == "knights_knaves"
    assert len(fallback_task["constraints"]["statements"]) == 3

    specs = mod.resolve_model_specs(
        {"MODEL_SPECS": [17, _fake_pair()[0]]},
        cached_pair_fn=lambda **_: [],
    )
    assert specs == [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": "/models/qwen.gguf",
            "loader": "llama.cpp",
            "from_cached_sota_pair": False,
        }
    ]


def test_upstream_missing_or_malformed_blocks_for_req_infer_sota_030(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-030-BLOCKED: unreadable Exp 5124 evidence fails closed."""

    missing_artifact = mod.write_artifact(
        root=tmp_path,
        duration_s=0.5,
        run_date="20260701",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
    )
    assert missing_artifact["honest_verdict"] == mod.BLOCKED_GATE_VERDICT
    assert missing_artifact["preconditions_checked"]["upstream_error"] == (
        "missing upstream Exp 5124 artifact"
    )

    upstream_path = tmp_path / mod.UPSTREAM_RELATIVE_PATH
    upstream_path.parent.mkdir(parents=True, exist_ok=True)
    upstream_path.write_text("{bad json", encoding="utf-8")
    malformed_artifact = mod.write_artifact(
        root=tmp_path,
        duration_s=0.5,
        run_date="20260701",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
    )
    assert malformed_artifact["honest_verdict"] == mod.BLOCKED_GATE_VERDICT
    assert "JSONDecodeError" in malformed_artifact["preconditions_checked"]["upstream_error"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {key: value for key, value in artifact.items() if key != "pool_n"},
            "missing",
        ),
        (lambda artifact: artifact | {"experiment_id": "bad"}, "experiment_id"),
        (lambda artifact: artifact | {"milestone": "2026.07.469"}, "milestone"),
        (lambda artifact: artifact | {"honest_verdict": "bad"}, "honest_verdict"),
        (lambda artifact: artifact | {"inference_substrate": "live_llm_inference"}, "substrate"),
        (lambda artifact: artifact | {"verifier_is_oracle": True}, "verifier_is_oracle"),
        (lambda artifact: artifact | {"fover_scope_used": True}, "fover_scope_used"),
        (lambda artifact: artifact | {"conductor_modified": True}, "conductor_modified"),
        (lambda artifact: artifact | {"pool_n": 79}, "pool_n"),
        (lambda artifact: artifact | {"candidates_per_item": 5}, "candidates_per_item"),
        (lambda artifact: artifact | {"pool_path": None}, "pool_path"),
        (lambda artifact: artifact | {"parse_coverage": 0.5}, "structured_pool_ready"),
        (lambda artifact: artifact | {"cheap_baseline_at_1": artifact["oracle_at_k"]}, "headroom"),
        (lambda artifact: artifact | {"honest_verdict": "blocked_fake_ready"}, "ready artifact"),
        (lambda artifact: artifact | {"MODEL_SPECS": []}, "MODEL_SPECS"),
        (lambda artifact: artifact | {"tests_run": []}, "tests_run"),
    ],
)
def test_artifact_validator_rejects_schema_or_gate_drift_for_req_infer_sota_030(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    """REQ-INFER-SOTA-030: artifact validation preserves the downstream gate contract."""

    _write_upstream(tmp_path)
    artifact = mod.write_artifact(
        root=tmp_path,
        duration_s=1.25,
        run_date="20260701",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
    )

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(copy.deepcopy(artifact)))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact | {"honest_verdict": "complete_fake"}, "not-ready"),
        (lambda artifact: artifact | {"pool_n": 1}, "pool_n"),
    ],
)
def test_blocked_artifact_validator_rejects_drift_for_req_infer_sota_030(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    """SCENARIO-INFER-SOTA-030-BLOCKED: blocked artifacts cannot look complete."""

    artifact = mod.write_artifact(
        root=tmp_path,
        duration_s=0.5,
        run_date="20260701",
        tests_run=["focused"],
        cached_pair_fn=_fake_pair,
    )

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(copy.deepcopy(artifact)))


def test_script_main_delegates_to_tested_module_for_req_infer_sota_030(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-030: CLI wrapper calls the module main."""

    _write_upstream(tmp_path)
    output = tmp_path / mod.RESULT_RELATIVE_PATH

    exit_code = script_mod.main(
        [
            "--date",
            "20260701",
            "--root",
            str(tmp_path),
            "--duration-override",
            "1.25",
        ],
        cached_pair_fn=_fake_pair,
    )

    assert exit_code == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["structured_pool_ready"] is True


def test_main_measures_duration_when_no_override_for_req_infer_sota_030(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-030: CLI path records measured duration when no override is supplied."""

    _write_upstream(tmp_path)
    exit_code = script_mod.main(
        ["--date", "20260701", "--root", str(tmp_path)], cached_pair_fn=_fake_pair
    )

    assert exit_code == 0
    artifact = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert artifact["duration_s"] > 0
    assert artifact["structured_pool_ready"] is True
