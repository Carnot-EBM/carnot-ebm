"""Tests for Exp 4022 decentralization gated by Exp 4012.

Spec refs: REQ-PHASE4-031, SCENARIO-PHASE4-031.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from carnot.agentic.arc_decentralization_gated import (
    BRANCH_A,
    BRANCH_B,
    REQUIRED_ARTIFACT_FIELDS,
    artifact_schema_errors,
    build_blocked_artifact,
    build_decentralization_artifact,
    choose_branch,
    harvest_distillation_traces,
    tiny_sanity_finetune,
    write_artifact,
    write_jsonl,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
sys.path.insert(0, str(REPO / "scripts" / "experiments"))


def _exp4012_no_lift() -> dict[str, object]:
    return {
        "honest_verdict": "complete: gap4_local_bestofn_cov0.2581_pass20.4516_below_codex",
        "local_beats_vote": False,
        "coverage_gain_vs_3attempt": 0.0,
        "local_demo_perfect_coverage_bestofn": 0.2581,
        "local_gated_pass2": 0.4516,
        "k_samples_per_task": 8,
        "inference_substrate": "live_local_gguf_llama_cpp_best_of_n",
    }


def _pool_payload() -> dict[str, object]:
    return {
        "entries": [
            {
                "task": "T1",
                "demos": [
                    {"input": [[1]], "output": [[2]]},
                    {"input": [[3]], "output": [[4]]},
                ],
                "test_input": [[5]],
            },
            {
                "task": "T2",
                "demos": [{"input": [[1]], "output": [[1]]}],
                "test_input": [[9]],
            },
        ]
    }


def _program_payload() -> dict[str, object]:
    return {
        "programs": [
            {
                "task": "T1",
                "code": "def transform(grid):\n    return grid + 1\n",
                "demo_fit": 1.0,
                "demo_perfect": True,
                "pred_grid": [[6]],
                "pred_hash": "abc",
                "codex_seconds": 3.0,
            },
            {
                "task": "T1",
                "code": "def transform(grid):\n    return grid\n",
                "demo_fit": 0.5,
                "demo_perfect": False,
                "pred_grid": [[5]],
                "pred_hash": "bad",
            },
            {
                "task": "T2",
                "code": "import os\ndef transform(grid):\n    return grid\n",
                "demo_fit": 1.0,
                "demo_perfect": True,
                "pred_grid": [[9]],
                "pred_hash": "unsafe",
            },
        ]
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_pool(path: Path, payload: dict[str, object]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def test_req_phase4_031_spec_declares_exp4022_contract() -> None:
    """REQ-PHASE4-031: OpenSpec declares the gate and required artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-031" in spec
    assert "SCENARIO-PHASE4-031" in spec
    assert "experiment_4022_decentralization_gated.json" in spec
    assert "blocked_exp4012_result_unavailable" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_phase4_031_branch_is_data_driven_by_exp4012_numbers() -> None:
    """SCENARIO-PHASE4-031: no local lift takes branch B; material lift takes A."""

    branch, cited = choose_branch(_exp4012_no_lift())
    positive = dict(_exp4012_no_lift(), local_beats_vote=True, coverage_gain_vs_3attempt=0.08)
    positive_branch, positive_cited = choose_branch(positive)

    assert branch == "B_distill_feasibility"
    assert cited["local_beats_vote"] is False
    assert cited["coverage_gain_vs_3attempt"] == 0.0
    assert cited["local_demo_perfect_coverage_bestofn"] == 0.2581
    assert positive_branch == "A_scale"
    assert positive_cited["local_beats_vote"] is True


def test_req_phase4_031_blocked_artifact_does_not_fabricate_branch() -> None:
    """REQ-PHASE4-031: absent or blocked Exp4012 writes the explicit blocked verdict."""

    artifact = build_blocked_artifact("blocked_exp4012_result_unavailable", duration_s=0.01)

    assert artifact["honest_verdict"] == "blocked_exp4012_result_unavailable"
    assert artifact["branch_taken"] == "blocked"
    assert artifact["exp4012_result_cited"] == "unavailable"
    assert artifact_schema_errors(artifact) == []


def test_scenario_phase4_031_harvests_only_clean_execution_validated_traces(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-031: corpus rows are demo-perfect, execution-validated, and safe."""

    program_path = _write_json(tmp_path / "programs.json", _program_payload())
    pool_path = _write_pool(tmp_path / "pool.json.gz", _pool_payload())

    traces, report = harvest_distillation_traces([program_path], [pool_path])

    assert len(traces) == 1
    assert traces[0]["task"] == "T1"
    assert "Demo 1 INPUT" in traces[0]["instruction"]
    assert "TEST INPUT" in traces[0]["instruction"]
    assert traces[0]["response"].startswith("```python\n")
    assert traces[0]["verifier_certification"]["demo_perfect"] is True
    assert report["n_clean_traces"] == 1
    assert report["rejection_reasons"]["not_demo_perfect_or_not_executed"] == 1
    assert report["rejection_reasons"]["unsafe_code"] == 1


def test_scenario_phase4_031_tiny_sanity_finetune_is_bounded_and_honest() -> None:
    """SCENARIO-PHASE4-031: tiny sanity training lowers subset token loss only."""

    traces = [
        {"response": "```python\ndef transform(grid):\n    return grid + 1\n```"},
        {"response": "```python\ndef transform(grid):\n    return grid.copy()\n```"},
    ]

    result = tiny_sanity_finetune(traces, subset_size=2)

    assert result["ran"] is True
    assert result["subset_size"] == 2
    assert result["loss_delta"] > 0.0
    assert result["full_llm_finetune"] is False


def test_req_phase4_031_artifact_schema_and_write_path(tmp_path: Path) -> None:
    """REQ-PHASE4-031: branch-B artifacts expose the required bare fields."""

    traces = [
        {
            "task": "T1",
            "instruction": "Demo 1 INPUT\n[[1]]\nDemo 1 OUTPUT\n[[2]]",
            "response": "```python\ndef transform(grid):\n    return grid + 1\n```",
            "verifier_certification": {"demo_perfect": True, "demo_fit": 1.0},
            "quality": {"code_chars": 42, "hardcoded_grid_suspect": False},
        }
    ]
    quality = {
        "n_programs_scanned": 1,
        "n_execution_validated": 1,
        "n_clean_traces": 1,
        "rejection_reasons": {},
        "median_code_chars": 42,
        "hardcoded_grid_suspect_count": 0,
    }
    sanity = tiny_sanity_finetune(traces, subset_size=1)
    artifact = build_decentralization_artifact(
        _exp4012_no_lift(),
        branch_taken="B_distill_feasibility",
        corpus_report=quality,
        sanity_finetune=sanity,
        corpus_path=tmp_path / "corpus.jsonl",
        duration_s=0.02,
    )
    output = write_artifact(artifact, tmp_path / "experiment_4022_decentralization_gated.json")

    assert artifact["honest_verdict"].startswith("complete: B_distill_feasibility")
    assert artifact["branch_taken"] == "B_distill_feasibility"
    assert artifact["exp4012_result_cited"]["coverage_gain_vs_3attempt"] == 0.0
    assert "stronger local base" in artifact["decentralization_next_step"]
    assert artifact_schema_errors(artifact) == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    bad = dict(artifact)
    bad["branch_taken"] = 12
    bad["decentralization_next_step"] = []
    bad["inference_substrate"] = None
    errors = artifact_schema_errors(bad)

    assert any("branch_taken" in err for err in errors)
    assert any("decentralization_next_step" in err for err in errors)
    assert any("inference_substrate" in err for err in errors)


def test_scenario_phase4_031_runner_writes_result_and_corpus(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-PHASE4-031: script runner takes branch B from Exp4012 and writes artifacts."""

    import experiment_4022_decentralization_gated as exp

    exp4012_path = _write_json(tmp_path / "exp4012.json", _exp4012_no_lift())
    program_path = _write_json(tmp_path / "programs.json", _program_payload())
    pool_path = _write_pool(tmp_path / "pool.json.gz", _pool_payload())
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "EXP4012_RESULT", exp4012_path)
    monkeypatch.setattr(exp, "PROGRAM_ARTIFACTS", [program_path])
    monkeypatch.setattr(exp, "POOL_ARTIFACTS", [pool_path])

    artifact = exp.run(write=True)
    result_path = tmp_path / "results" / exp.RESULT_NAME
    corpus_path = tmp_path / "results" / exp.CORPUS_NAME

    assert artifact["branch_taken"] == "B_distill_feasibility"
    assert artifact["distillation_corpus"]["n_clean_traces"] == 1
    assert result_path.exists()
    assert corpus_path.exists()
    assert len(corpus_path.read_text(encoding="utf-8").strip().splitlines()) == 1

    blocked_path = _write_json(
        tmp_path / "blocked4012.json",
        {"honest_verdict": "blocked_local_gguf_not_cached"},
    )
    monkeypatch.setattr(exp, "EXP4012_RESULT", blocked_path)
    blocked = exp.run(write=False)

    assert blocked["honest_verdict"] == "blocked_exp4012_result_unavailable"
    assert blocked["branch_taken"] == "blocked"


def test_req_phase4_031_jsonl_writer_is_stable(tmp_path: Path) -> None:
    """REQ-PHASE4-031: harvested corpus is written as deterministic JSONL."""

    path = write_jsonl([{"b": 2, "a": 1}], tmp_path / "corpus.jsonl")

    assert path.read_text(encoding="utf-8") == '{"a": 1, "b": 2}\n'


def test_req_phase4_031_malformed_exp4012_values_and_json_are_explicit(tmp_path: Path) -> None:
    """REQ-PHASE4-031: malformed numbers do not fabricate a positive branch."""

    malformed = {
        "local_beats_vote": False,
        "coverage_gain_vs_3attempt": "bad",
        "local_demo_perfect_coverage_bestofn": None,
        "local_gated_pass2": "bad",
        "k_samples_per_task": "bad",
    }
    branch, cited = choose_branch(malformed)
    scalar_path = tmp_path / "scalar.json"
    scalar_path.write_text("[1, 2, 3]", encoding="utf-8")

    assert branch == BRANCH_B
    assert cited["coverage_gain_vs_3attempt"] == 0.0
    assert cited["k_samples_per_task"] == 0
    try:
        from carnot.agentic.arc_decentralization_gated import load_json

        load_json(scalar_path)
    except ValueError as exc:
        assert "did not contain a JSON object" in str(exc)
    else:  # pragma: no cover - test guard
        raise AssertionError("load_json accepted a scalar payload")


def test_scenario_phase4_031_harvester_handles_missing_malformed_duplicate_and_limit(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-031: defensive harvest paths are counted rather than claimed clean."""

    pool_json = _write_json(tmp_path / "pool.json", _pool_payload())
    bad_pool = tmp_path / "bad_pool.json"
    bad_pool.write_text("{", encoding="utf-8")
    missing_program = tmp_path / "missing_program.json"
    malformed_program = tmp_path / "malformed_program.json"
    malformed_program.write_text("{", encoding="utf-8")
    duplicate_program = {
        "programs": [
            {
                "task": "T1",
                "code": "def transform(grid):\n    return grid + 1\n",
                "demo_fit": 1.0,
                "demo_perfect": True,
                "pred_grid": [[6]],
                "pred_hash": "abc",
            },
            {
                "task": "T1",
                "code": "def transform(grid):\n    return grid + 1\n",
                "demo_fit": 1.0,
                "demo_perfect": True,
                "pred_grid": [[6]],
                "pred_hash": "abc",
            },
            {
                "task": "UNKNOWN",
                "code": "def transform(grid):\n    return grid\n",
                "demo_fit": 1.0,
                "demo_perfect": True,
                "pred_grid": [[6]],
                "pred_hash": "abc",
            },
            "not a dict",
        ]
    }
    program_path = _write_json(tmp_path / "programs.json", duplicate_program)

    traces, report = harvest_distillation_traces(
        [missing_program, malformed_program, program_path],
        [tmp_path / "missing_pool.json.gz", bad_pool, pool_json],
    )
    limited, limited_report = harvest_distillation_traces([program_path], [pool_json], max_traces=1)

    assert len(traces) == 1
    assert report["rejection_reasons"]["duplicate_trace"] == 1
    assert report["rejection_reasons"]["missing_pool_entry"] == 1
    assert len(limited) == 1
    assert limited_report["n_programs_scanned"] == 1


def test_req_phase4_031_instruction_tolerates_nonlist_demos_and_hardcoded_quality(tmp_path: Path) -> None:
    """REQ-PHASE4-031: prompt construction and quality reporting stay deterministic."""

    pool_path = _write_json(
        tmp_path / "pool.json",
        {"entries": [{"task": "T1", "demos": "not-list", "test_input": [[0]]}]},
    )
    hardcoded_code = "def transform(grid):\n    return arr([" + ",".join("[1]" for _ in range(121)) + "])\n"
    program_path = _write_json(
        tmp_path / "programs.json",
        {
            "programs": [
                {
                    "task": "T1",
                    "code": hardcoded_code,
                    "demo_fit": 1.0,
                    "demo_perfect": True,
                    "pred_grid": [[1]],
                    "pred_hash": "hash",
                }
            ]
        },
    )

    traces, report = harvest_distillation_traces([program_path], [pool_path])

    assert traces[0]["quality"]["hardcoded_grid_suspect"] is True
    assert traces[0]["quality"]["quality_score"] == 0.75
    assert "Demo 1 INPUT" not in traces[0]["instruction"]
    assert report["hardcoded_grid_suspect_count"] == 1
    assert report["generic_trace_ratio"] == 0.0


def test_scenario_phase4_031_empty_sanity_and_branch_a_artifact_are_honest(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-031: no corpus and branch-A states do not claim a full train."""

    empty = tiny_sanity_finetune([], subset_size=4)
    no_clean_artifact = build_decentralization_artifact(
        _exp4012_no_lift(),
        branch_taken=BRANCH_B,
        corpus_report={"n_clean_traces": 0},
        sanity_finetune=empty,
        corpus_path=tmp_path / "corpus.jsonl",
        duration_s=0.01,
    )
    positive = dict(_exp4012_no_lift(), local_beats_vote=True, coverage_gain_vs_3attempt=0.08)
    branch_a = build_decentralization_artifact(
        positive,
        branch_taken=BRANCH_A,
        corpus_report={"n_clean_traces": 0},
        sanity_finetune=empty,
        corpus_path=tmp_path / "corpus.jsonl",
        duration_s=0.01,
    )

    assert empty["ran"] is False
    assert "regenerate verifier-certified traces" in no_clean_artifact["decentralization_next_step"]
    assert branch_a["honest_verdict"].startswith("complete: A_scale")
    assert branch_a["local_support_diagnostic"] == "latent_support_possible"
    assert "bounded same-pool scaling" in branch_a["decentralization_next_step"]
    assert artifact_schema_errors(branch_a) == []


def test_req_phase4_031_schema_reports_missing_bad_verdict_and_citation_errors() -> None:
    """REQ-PHASE4-031: schema validation covers missing and inconsistent fields."""

    missing_errors = artifact_schema_errors({})
    bad = {
        "honest_verdict": "done",
        "branch_taken": BRANCH_B,
        "exp4012_result_cited": "unavailable",
        "decentralization_next_step": "next",
        "inference_substrate": "substrate",
    }
    blocked_bad = {
        "honest_verdict": "blocked_exp4012_result_unavailable",
        "branch_taken": "blocked",
        "exp4012_result_cited": {"local_beats_vote": False},
        "decentralization_next_step": "next",
        "inference_substrate": "substrate",
    }

    bad_errors = artifact_schema_errors(bad)
    blocked_errors = artifact_schema_errors(blocked_bad)

    assert any("missing required field honest_verdict" in err for err in missing_errors)
    assert any("honest_verdict" in err for err in bad_errors)
    assert any("exp4012_result_cited" in err for err in bad_errors)
    assert any("blocked artifacts" in err for err in blocked_errors)
