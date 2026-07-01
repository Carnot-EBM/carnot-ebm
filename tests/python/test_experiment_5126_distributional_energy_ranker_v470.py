"""Tests for Exp 5126 distributional-energy ranker.

Spec refs: REQ-INFER-SOTA-031,
SCENARIO-INFER-SOTA-031-RANKER,
SCENARIO-INFER-SOTA-031-BLOCKED.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5125_structured_reasoning_pool_v470 as pool_mod
from carnot import experiment_5126_distributional_energy_ranker_v470 as mod
from scripts import experiment_5126_distributional_energy_ranker_v470 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/llm-ebm-inference/spec.md"


def _fake_specs() -> list[dict[str, object]]:
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": "/models/qwen.gguf",
            "loader": "llama.cpp",
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "gpu": 1,
            "model_path": "/models/gemma.gguf",
            "loader": "llama.cpp",
        },
    ]


def _write_ready_pool(root: Path) -> list[dict[str, object]]:
    specs = _fake_specs()
    rows = pool_mod.build_pool_rows(pool_mod.build_task_bank(), specs)
    pool_path = root / pool_mod.POOL_RELATIVE_PATH
    pool_mod.write_jsonl(pool_path, rows)
    artifact = {
        "experiment_id": pool_mod.EXPERIMENT_ID,
        "milestone": pool_mod.MILESTONE,
        "honest_verdict": pool_mod.SUCCESS_VERDICT,
        "inference_substrate": pool_mod.INFERENCE_SUBSTRATE,
        "duration_s": 1.0,
        "MODEL_SPECS": specs,
        "structured_pool_ready": True,
        "pool_path": pool_mod.POOL_RELATIVE_PATH,
        "pool_sha256": pool_mod.sha256_file(pool_path),
        "pool_n": len(rows),
        "tests_run": ["fixture"],
    }
    pool_mod.write_json(root / pool_mod.RESULT_RELATIVE_PATH, artifact)
    return rows


def test_req_infer_sota_031_spec_declares_ranker_contract() -> None:
    """REQ-INFER-SOTA-031: OpenSpec declares the Exp 5126 ranker contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-INFER-SOTA-031")
    end = spec.index("### REQ-INFER-018", start)
    section = spec[start:end]

    assert "SCENARIO-INFER-SOTA-031-RANKER" in section
    assert "SCENARIO-INFER-SOTA-031-BLOCKED" in section
    assert mod.EXPERIMENT_ID in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_structured_pool_gate_hard_blocks_for_req_infer_sota_031(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-031-BLOCKED: closed upstream gate writes no metrics."""

    pool_mod.write_json(
        tmp_path / pool_mod.RESULT_RELATIVE_PATH,
        {
            "experiment_id": pool_mod.EXPERIMENT_ID,
            "structured_pool_ready": False,
            "pool_path": pool_mod.POOL_RELATIVE_PATH,
            "MODEL_SPECS": _fake_specs(),
        },
    )

    artifact = mod.write_artifact(
        root=tmp_path,
        duration_s=0.25,
        run_date="20260701",
        tests_run=["focused"],
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.BLOCKED_POOL_VERDICT
    assert artifact["ranker_ready_for_audit"] is False
    assert artifact["distributional_energy_delta"] == 0.0
    assert artifact["strongest_cheap_baseline"] is None
    assert artifact["preconditions_checked"]["structured_pool_ready"] is False


def test_split_and_energy_are_grouped_and_oracle_distinct(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-031-RANKER: splits prevent item leakage and decompose energy."""

    rows = _write_ready_pool(tmp_path)
    bundle = mod.load_structured_pool(root=tmp_path)
    splits = mod.split_rows_by_family_and_item(bundle.rows)
    split_ids = {
        split: {str(row["task_id"]) for row in split_rows} for split, split_rows in splits.items()
    }

    assert bundle.source_artifact["structured_pool_ready"] is True
    assert bundle.source_pool_path == pool_mod.POOL_RELATIVE_PATH
    assert split_ids["train"].isdisjoint(split_ids["calibration"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["calibration"].isdisjoint(split_ids["test"])
    assert {row["family"] for row in splits["test"]} == {
        "code_property",
        "graph_coloring",
        "knights_knaves",
        "travel_budget",
    }

    task_lookup = mod.build_task_lookup()
    model = mod.fit_quality_model(mod.build_candidate_examples(splits["train"], task_lookup))
    row = next(row for row in rows if any(candidate["correct"] for candidate in row["candidates"]))
    good = next(candidate for candidate in row["candidates"] if candidate["correct"])
    bad = next(candidate for candidate in row["candidates"] if not candidate["correct"])

    good_penalty = mod.deterministic_constraint_penalty(row, good, task_lookup)
    bad_penalty = mod.deterministic_constraint_penalty(row, bad, task_lookup)
    good_score = mod.score_decomposed_energy(row, good, task_lookup, model)
    bad_score = mod.score_decomposed_energy(row, bad, task_lookup, model)
    mutated = copy.deepcopy(good)
    mutated["model_hf_id"] = "shortcut/model"
    mutated["model_path"] = "/shortcut/model.gguf"
    mutated_score = mod.score_decomposed_energy(row, mutated, task_lookup, model)

    assert good_penalty["total_penalty"] == 0.0
    assert bad_penalty["total_penalty"] > 0.0
    assert good_score["mean_energy"] < bad_score["mean_energy"]
    assert mutated_score == good_score


def test_write_artifact_reports_required_metrics_and_controls(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-031: artifact reports baselines, controls, CI, and audit gate."""

    _write_ready_pool(tmp_path)

    artifact = mod.write_artifact(
        root=tmp_path,
        duration_s=1.5,
        run_date="20260701",
        tests_run=["focused"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["honest_verdict"] in {
        mod.SUCCESS_NOT_READY_VERDICT,
        mod.SUCCESS_READY_VERDICT,
    }
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["source_pool_path"] == pool_mod.POOL_RELATIVE_PATH
    assert artifact["MODEL_SPECS"] == _fake_specs()
    assert artifact["verifier_is_oracle"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["tests_run"] == ["focused"]
    assert artifact["ranker_metrics"]["accuracy_at_1"] >= 0.0
    assert artifact["ranker_metrics"]["abstention_rate"] > 0.0
    assert artifact["ranker_metrics"]["auroc"] is not None
    assert artifact["strongest_cheap_baseline"]["name"] in artifact["baseline_metrics"]
    assert len(artifact["delta_ci95"]) == 2
    assert artifact["label_shuffle_result"]["passed"] is True
    assert artifact["duplicate_control_result"]["passed"] is True
    assert artifact["model_identity_shortcut_check"]["passed"] is True
    assert set(artifact["family_holdout_results"]) == {
        "code_property",
        "graph_coloring",
        "knights_knaves",
        "travel_budget",
    }
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact


def test_validation_and_cli_edges_for_req_infer_sota_031(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-031: validation and script wrapper reject malformed results."""

    rows = _write_ready_pool(tmp_path)
    artifact = mod.write_artifact(
        root=tmp_path,
        duration_s=0.75,
        run_date="20260701",
        tests_run=["focused"],
    )

    assert mod.read_jsonl(tmp_path / "missing.jsonl") == []
    missing_root = tmp_path / "missing-root"
    missing_root.mkdir()
    missing_artifact, missing_error = mod._pool_artifact(missing_root)
    assert missing_artifact is None
    assert "missing Exp 5125" in str(missing_error)

    malformed_root = tmp_path / "malformed-root"
    (malformed_root / "results").mkdir(parents=True)
    (malformed_root / pool_mod.RESULT_RELATIVE_PATH).write_text("{", encoding="utf-8")
    malformed_artifact, malformed_error = mod._pool_artifact(malformed_root)
    assert malformed_artifact is None
    assert str(malformed_error).startswith("JSONDecodeError")

    not_ready_root = tmp_path / "not-ready-root"
    pool_mod.write_json(
        not_ready_root / pool_mod.RESULT_RELATIVE_PATH,
        {"structured_pool_ready": False, "pool_path": pool_mod.POOL_RELATIVE_PATH},
    )
    with pytest.raises(ValueError, match="structured_pool_ready"):
        mod.load_structured_pool(root=not_ready_root)

    missing_rows_root = tmp_path / "missing-rows-root"
    pool_mod.write_json(
        missing_rows_root / pool_mod.RESULT_RELATIVE_PATH,
        {
            "structured_pool_ready": True,
            "pool_path": pool_mod.POOL_RELATIVE_PATH,
            "MODEL_SPECS": _fake_specs(),
        },
    )
    missing_rows_artifact = mod.build_artifact(
        root=missing_rows_root,
        duration_s=0.5,
        run_date="20260701",
        tests_run=["focused"],
    )
    assert missing_rows_artifact["honest_verdict"] == mod.BLOCKED_ROWS_VERDICT

    task_lookup = mod.build_task_lookup()
    graph_row = next(row for row in rows if row["family"] == "graph_coloring")
    graph_candidate = dict(graph_row["candidates"][0])
    graph_candidate["raw_response"] = json.dumps({"answer": [True]})
    graph_bool_penalty = mod.deterministic_constraint_penalty(
        graph_row, graph_candidate, task_lookup
    )
    graph_candidate["raw_response"] = json.dumps({"answer": [0]})
    graph_short_penalty = mod.deterministic_constraint_penalty(
        graph_row, graph_candidate, task_lookup
    )
    knights_row = next(row for row in rows if row["family"] == "knights_knaves")
    knights_candidate = dict(knights_row["candidates"][0])
    knights_candidate["raw_response"] = json.dumps(
        {"answer": {"A": "maybe", "B": "knight", "C": "knave"}}
    )
    knights_penalty = mod.deterministic_constraint_penalty(
        knights_row, knights_candidate, task_lookup
    )
    missing_answer_candidate = dict(graph_row["candidates"][0])
    missing_answer_candidate["raw_response"] = json.dumps({"not_answer": []})
    missing_answer_penalty = mod.deterministic_constraint_penalty(
        graph_row, missing_answer_candidate, task_lookup
    )

    assert graph_bool_penalty["components"]["answer_type"] == 10.0
    assert graph_short_penalty["components"]["edge_conflicts"] > 0.0
    assert knights_penalty["components"]["answer_type"] == 10.0
    assert missing_answer_penalty["components"]["parse"] == 100.0
    assert mod._auroc([True, True], [0.5, 0.7]) is None

    missing = dict(artifact)
    missing.pop("MODEL_SPECS")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(missing)

    bad_experiment = dict(artifact, experiment_id="wrong")
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(bad_experiment)

    bad_milestone = dict(artifact, milestone="wrong")
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(bad_milestone)

    bad_substrate = dict(artifact, inference_substrate="wrong")
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_verdict = dict(artifact, honest_verdict="maybe")
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(bad_verdict)

    bad_oracle = dict(artifact, verifier_is_oracle=True)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_oracle)

    bad_conductor = dict(artifact, conductor_modified=True)
    with pytest.raises(ValueError, match="conductor_modified"):
        mod.validate_artifact(bad_conductor)

    bad_tests = dict(artifact, tests_run=[])
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(bad_tests)

    bad_ready = dict(artifact, ranker_ready_for_audit=True, delta_ci95=[0.0, 0.1])
    with pytest.raises(ValueError, match="delta CI95"):
        mod.validate_artifact(bad_ready)

    bad_baseline = dict(artifact, strongest_cheap_baseline=None)
    with pytest.raises(ValueError, match="strongest_cheap_baseline"):
        mod.validate_artifact(bad_baseline)

    assert script_mod.main(["--root", str(tmp_path), "--date", "20260701"]) == 0
    rerun = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert rerun["experiment_id"] == mod.EXPERIMENT_ID
