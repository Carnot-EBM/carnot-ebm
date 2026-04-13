"""Tests for Experiment 246: solver-semantic live benchmark runner.

Verifies cohort reuse, artifact schema stability, checkpoint compatibility,
and route-summary aggregation without executing any live model inference.

Spec: REQ-VERIFY-058, REQ-VERIFY-059
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def load_module() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_246_solver_semantic_live.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_246_solver_semantic_live",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def _write_exp235_artifact(repo: Path, cases: list[dict[str, Any]] | None = None) -> Path:
    """Write a minimal Exp 235 artifact into repo/results/."""
    if cases is None:
        cases = [
            {
                "case_id": "gsm8k-1",
                "question": "Q1",
                "ground_truth": 4,
                "task_slice": "live_gsm8k_semantic_failure",
                "prompt_seeds": {"baseline": 11, "verify_only": 11, "verify_repair": 11},
            },
            {
                "case_id": "gsm8k-2",
                "question": "Q2",
                "ground_truth": 9,
                "task_slice": "live_gsm8k_semantic_failure",
                "prompt_seeds": {"baseline": 22, "verify_only": 22, "verify_repair": 22},
            },
        ]
    results_dir = repo / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "experiment": 235,
        "benchmark": "gsm8k_semantic",
        "run_date": "20260413",
        "metadata": {
            "sample_seed": 218,
            "sample_size": len(cases),
        },
        "cohort": {
            "case_count": len(cases),
            "case_ids": [c["case_id"] for c in cases],
            "cases": cases,
        },
        "statistics": {},
    }
    path = results_dir / "experiment_235_results.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_exp221_artifact(repo: Path, cases: list[dict[str, Any]] | None = None) -> Path:
    """Write a minimal Exp 221 artifact into repo/results/."""
    if cases is None:
        cases = [
            {
                "case_id": "exp211-code-score-1",
                "example_id": "exp211-code-score-1",
                "source_family": "code_typed_properties",
                "task_slice": "code_typed_properties",
                "constraint_types": ["signature"],
                "prompt": "Write dedupe(lst)",
                "expected_answer_schema": {"type": "python_function", "name": "dedupe"},
                "gold_atomic_constraints": [
                    {"constraint_id": "c1", "type": "function_name", "value": "dedupe", "target": "function_name"},
                ],
                "prompt_seeds": {"baseline": 33, "verify_only": 33, "verify_repair": 33},
            },
            {
                "case_id": "exp211-instruction-bullets-1",
                "example_id": "exp211-instruction-bullets-1",
                "source_family": "instruction_following",
                "task_slice": "instruction_surface_only",
                "constraint_types": ["count_exact"],
                "prompt": "List 3 items",
                "expected_answer_schema": {"type": "bullet_list"},
                "gold_atomic_constraints": [
                    {"constraint_id": "c2", "type": "count_exact", "value": 3, "target": "bullet_count"},
                ],
                "prompt_seeds": {"baseline": 44, "verify_only": 44, "verify_repair": 44},
            },
        ]
    results_dir = repo / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "experiment": 221,
        "benchmark": "constraint_ir",
        "run_date": "20260412",
        "metadata": {
            "sample_seed": 218,
            "sample_size": len(cases),
        },
        "cohort": {
            "case_count": len(cases),
            "case_ids": [c["case_id"] for c in cases],
            "cases": cases,
        },
        "statistics": {},
    }
    path = results_dir / "experiment_221_results.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Cohort reuse tests  (REQ-VERIFY-058, SCENARIO-VERIFY-025)
# ---------------------------------------------------------------------------


# REQ-VERIFY-058: cohort from Exp 235 is loaded verbatim
def test_load_exp235_cohort_returns_cases_and_meta(tmp_path: Path) -> None:
    module = load_module()
    repo = _make_repo(tmp_path)
    artifact_path = _write_exp235_artifact(repo)

    cohort, meta = module.load_gsm8k_cohort(artifact_path)

    assert len(cohort) == 2
    assert cohort[0]["case_id"] == "gsm8k-1"
    assert cohort[1]["case_id"] == "gsm8k-2"
    assert meta["source_artifact"].endswith("experiment_235_results.json")
    assert meta["benchmark"] == "gsm8k_semantic"
    assert meta["case_count"] == 2


# REQ-VERIFY-058: cohort from Exp 221 is loaded verbatim
def test_load_constraint_ir_cohort_returns_cases_and_meta(tmp_path: Path) -> None:
    module = load_module()
    repo = _make_repo(tmp_path)
    artifact_path = _write_exp221_artifact(repo)

    cohort, meta = module.load_constraint_ir_cohort(artifact_path)

    assert len(cohort) == 2
    assert cohort[0]["case_id"] == "exp211-code-score-1"
    assert meta["source_artifact"].endswith("experiment_221_results.json")
    assert meta["benchmark"] == "constraint_ir"
    assert meta["case_count"] == 2


# REQ-VERIFY-058: wrong-benchmark artifact raises ValueError
def test_load_gsm8k_cohort_rejects_wrong_benchmark(tmp_path: Path) -> None:
    module = load_module()
    repo = _make_repo(tmp_path)
    # Write a constraint_ir artifact under the exp235 path
    path = _write_exp221_artifact(repo)
    renamed = path.parent / "experiment_235_results.json"
    path.rename(renamed)

    with pytest.raises(ValueError, match="gsm8k_semantic"):
        module.load_gsm8k_cohort(renamed)


def test_load_constraint_ir_cohort_rejects_wrong_benchmark(tmp_path: Path) -> None:
    module = load_module()
    repo = _make_repo(tmp_path)
    path = _write_exp235_artifact(repo)
    renamed = path.parent / "experiment_221_results.json"
    path.rename(renamed)

    with pytest.raises(ValueError, match="constraint_ir"):
        module.load_constraint_ir_cohort(renamed)


# ---------------------------------------------------------------------------
# Checkpoint compatibility tests  (SCENARIO-VERIFY-025)
# ---------------------------------------------------------------------------


# REQ-VERIFY-058: checkpoint path encodes benchmark/model/mode
def test_checkpoint_path_slug_format(tmp_path: Path) -> None:
    module = load_module()
    ckpt_dir = tmp_path / "checkpoints"

    path = module.checkpoint_path(
        ckpt_dir,
        benchmark="gsm8k_semantic",
        model_name="Qwen3.5-0.8B",
        mode="verify_only",
    )

    assert path.parent == ckpt_dir
    assert "gsm8k_semantic" in path.name
    assert "qwen3" in path.name.lower() or "0_8b" in path.name.lower()
    assert "verify_only" in path.name
    assert path.suffix == ".json"


# REQ-VERIFY-058: fresh checkpoint when file missing
def test_load_checkpoint_fresh_when_missing(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "missing.json"
    case_ids = ["case-a", "case-b"]

    result = module.load_checkpoint(path, case_ids)

    assert result["case_ids"] == case_ids
    assert result["results_by_case"] == {}


# REQ-VERIFY-058: checkpoint invalidated when case_ids differ
def test_load_checkpoint_invalidates_on_case_id_mismatch(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "ckpt.json"
    old_ids = ["case-x", "case-y"]
    new_ids = ["case-a", "case-b"]

    # Write a checkpoint with the old case list
    path.write_text(
        json.dumps({"case_ids": old_ids, "results_by_case": {"case-x": {"result": 1}}}),
        encoding="utf-8",
    )

    result = module.load_checkpoint(path, new_ids)

    # Stale checkpoint discarded; fresh state returned
    assert result["case_ids"] == new_ids
    assert result["results_by_case"] == {}


# REQ-VERIFY-058: save + reload checkpoint round-trips correctly
def test_save_and_reload_checkpoint_roundtrips(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "sub" / "ckpt.json"
    case_ids = ["case-a", "case-b"]
    payload = {
        "benchmark": "gsm8k_semantic",
        "model_name": "Qwen3.5-0.8B",
        "mode": "baseline",
        "case_ids": case_ids,
        "results_by_case": {"case-a": {"correct": True}},
    }

    module.save_checkpoint(path, payload)
    loaded = module.load_checkpoint(path, case_ids)

    assert loaded["results_by_case"]["case-a"]["correct"] is True
    assert loaded["case_ids"] == case_ids


# REQ-VERIFY-058: run_mode skips already-completed cases
def test_run_mode_skips_completed_cases(tmp_path: Path) -> None:
    module = load_module()
    ckpt_dir = tmp_path / "checkpoints"
    cases = [
        {"case_id": "a", "prompt_seeds": {"baseline": 1, "verify_only": 1, "verify_repair": 1}},
        {"case_id": "b", "prompt_seeds": {"baseline": 2, "verify_only": 2, "verify_repair": 2}},
    ]

    # Pre-seed checkpoint with case "a" already done
    ckpt_path = module.checkpoint_path(
        ckpt_dir, benchmark="gsm8k_semantic", model_name="Qwen3.5-0.8B", mode="baseline"
    )
    module.save_checkpoint(
        ckpt_path,
        {
            "benchmark": "gsm8k_semantic",
            "model_name": "Qwen3.5-0.8B",
            "mode": "baseline",
            "case_ids": ["a", "b"],
            "results_by_case": {"a": {"case_id": "a", "mode": "baseline", "correct": True}},
        },
    )

    executed: list[str] = []

    def execute_case(case: dict[str, Any]) -> dict[str, Any]:
        executed.append(str(case["case_id"]))
        return {"correct": False, "formal_claims": []}

    results = module.run_mode(
        benchmark="gsm8k_semantic",
        model_name="Qwen3.5-0.8B",
        mode="baseline",
        cases=cases,
        checkpoint_dir=ckpt_dir,
        execute_case=execute_case,
    )

    # Only case "b" should have been executed (case "a" was already in checkpoint)
    assert executed == ["b"]
    assert len(results) == 2
    # Case "a" preserved from checkpoint
    assert results[0]["correct"] is True
    # Case "b" from execute_case
    assert results[1]["correct"] is False


# REQ-VERIFY-058: run_mode preserves original case order
def test_run_mode_preserves_case_order(tmp_path: Path) -> None:
    module = load_module()
    ckpt_dir = tmp_path / "checkpoints"
    cases = [
        {"case_id": "z", "prompt_seeds": {"baseline": 10, "verify_only": 10, "verify_repair": 10}},
        {"case_id": "a", "prompt_seeds": {"baseline": 20, "verify_only": 20, "verify_repair": 20}},
    ]

    call_order: list[str] = []

    def execute_case(case: dict[str, Any]) -> dict[str, Any]:
        call_order.append(str(case["case_id"]))
        return {"correct": True, "formal_claims": []}

    results = module.run_mode(
        benchmark="gsm8k_semantic",
        model_name="Qwen3.5-0.8B",
        mode="baseline",
        cases=cases,
        checkpoint_dir=ckpt_dir,
        execute_case=execute_case,
    )

    assert [r["case_id"] for r in results] == ["z", "a"]


# ---------------------------------------------------------------------------
# Route-summary aggregation tests  (REQ-VERIFY-058)
# ---------------------------------------------------------------------------


def _make_claim(
    *,
    route: str = "arithmetic",
    verdict: str = "supported",
    failure_detail: str | None = None,
) -> dict[str, Any]:
    return {
        "claim_id": "c1",
        "route": route,
        "verdict": verdict,
        "failure_detail": failure_detail,
    }


# REQ-VERIFY-058: empty claim list produces zero-counts summary
def test_build_route_summary_empty() -> None:
    module = load_module()
    summary = module.build_route_summary([])

    assert summary["total_claims"] == 0
    assert summary["abstain_rate"] == 0.0
    assert summary["by_route"] == {}
    assert summary["by_verdict"]["supported"] == 0
    assert summary["by_verdict"]["violated"] == 0
    assert summary["by_verdict"]["abstain"] == 0


# REQ-VERIFY-058: counts are correctly aggregated across routes and verdicts
def test_build_route_summary_aggregates_routes_and_verdicts() -> None:
    module = load_module()
    claims = [
        _make_claim(route="arithmetic", verdict="supported"),
        _make_claim(route="arithmetic", verdict="violated", failure_detail="claimed=5 but got 6"),
        _make_claim(route="comparison", verdict="supported"),
        _make_claim(route="abstain", verdict="abstain"),
        _make_claim(route="set_membership", verdict="abstain"),
    ]
    summary = module.build_route_summary(claims)

    assert summary["total_claims"] == 5
    assert summary["by_route"]["arithmetic"] == 2
    assert summary["by_route"]["comparison"] == 1
    assert summary["by_route"]["abstain"] == 1
    assert summary["by_route"]["set_membership"] == 1
    assert summary["by_verdict"]["supported"] == 2
    assert summary["by_verdict"]["violated"] == 1
    assert summary["by_verdict"]["abstain"] == 2
    assert summary["abstain_rate"] == pytest.approx(2 / 5, abs=1e-6)


# REQ-VERIFY-058: abstain_rate is a fraction, not a count
def test_build_route_summary_abstain_rate_is_fraction() -> None:
    module = load_module()
    claims = [
        _make_claim(route="arithmetic", verdict="supported"),
        _make_claim(route="abstain", verdict="abstain"),
        _make_claim(route="abstain", verdict="abstain"),
    ]
    summary = module.build_route_summary(claims)

    assert 0.0 <= summary["abstain_rate"] <= 1.0
    assert summary["abstain_rate"] == pytest.approx(2 / 3, abs=1e-6)


# REQ-VERIFY-058: route_summary keys are stable (Exp 247 compatibility)
def test_build_route_summary_has_stable_keys() -> None:
    module = load_module()
    summary = module.build_route_summary([])

    assert set(summary.keys()) == {"by_route", "by_verdict", "total_claims", "abstain_rate"}
    assert set(summary["by_verdict"].keys()) == {"supported", "violated", "abstain"}


# ---------------------------------------------------------------------------
# Formal claim extraction tests  (REQ-VERIFY-058)
# ---------------------------------------------------------------------------


# REQ-VERIFY-058: arithmetic equations extracted from GSM8K response text
def test_extract_formal_claims_gsm8k_arithmetic() -> None:
    module = load_module()
    case: dict[str, Any] = {
        "case_id": "gsm8k-1",
        "question": "Q",
        "ground_truth": 12,
        "task_slice": "live_gsm8k_semantic_failure",
        "prompt_seeds": {"baseline": 1, "verify_only": 1, "verify_repair": 1},
    }
    response = "First I compute 3 + 4 = 7. Then 7 + 5 = 12."

    claims = module.extract_formal_claims_from_response(response, case=case, benchmark="gsm8k_semantic")

    assert len(claims) >= 2
    routes = {c["candidate_solver_route"] for c in claims}
    assert "arithmetic" in routes
    for claim in claims:
        assert claim["formalization_status"] == "formalized"
        assert "operands" in claim
        assert len(claim["operands"]) == 3


# REQ-VERIFY-058: no arithmetic in response → empty claims for gsm8k
def test_extract_formal_claims_gsm8k_no_equations() -> None:
    module = load_module()
    case: dict[str, Any] = {
        "case_id": "gsm8k-2",
        "question": "Q",
        "ground_truth": 5,
        "task_slice": "live_gsm8k_semantic_failure",
        "prompt_seeds": {"baseline": 1, "verify_only": 1, "verify_repair": 1},
    }
    claims = module.extract_formal_claims_from_response(
        "The answer is five.", case=case, benchmark="gsm8k_semantic"
    )

    assert claims == []


# REQ-VERIFY-058: count_exact constraint maps to cardinality route
def test_extract_formal_claims_constraint_ir_count_exact() -> None:
    module = load_module()
    case: dict[str, Any] = {
        "case_id": "exp211-bullets-1",
        "gold_atomic_constraints": [
            {"constraint_id": "c1", "type": "count_exact", "value": 3, "target": "bullet_count"},
        ],
        "expected_answer_schema": {"type": "bullet_list"},
        "task_slice": "instruction_surface_only",
    }

    claims = module.extract_formal_claims_from_response(
        "- item a\n- item b\n- item c", case=case, benchmark="constraint_ir"
    )

    assert len(claims) == 1
    assert claims[0]["candidate_solver_route"] == "cardinality"
    assert claims[0]["formalization_status"] == "formalized"


# REQ-VERIFY-058: must_include_token constraint maps to set_membership route
def test_extract_formal_claims_constraint_ir_must_include() -> None:
    module = load_module()
    case: dict[str, Any] = {
        "case_id": "exp211-include-1",
        "gold_atomic_constraints": [
            {
                "constraint_id": "c2",
                "type": "must_include_token",
                "value": "hello",
                "target": "response",
            },
        ],
        "expected_answer_schema": {"type": "text"},
        "task_slice": "instruction_surface_only",
    }

    claims = module.extract_formal_claims_from_response(
        "hello world", case=case, benchmark="constraint_ir"
    )

    assert len(claims) == 1
    assert claims[0]["candidate_solver_route"] == "set_membership"
    assert "hello" in claims[0]["bound_variables"]


# REQ-VERIFY-058: unformalizable constraint types produce no claims
def test_extract_formal_claims_constraint_ir_unformalizable() -> None:
    module = load_module()
    case: dict[str, Any] = {
        "case_id": "exp211-tone-1",
        "gold_atomic_constraints": [
            {"constraint_id": "c3", "type": "tone", "value": "formal", "target": "response"},
        ],
        "expected_answer_schema": {"type": "text"},
        "task_slice": "instruction_surface_only",
    }

    claims = module.extract_formal_claims_from_response(
        "Sure, no problem!", case=case, benchmark="constraint_ir"
    )

    # tone constraints are not formalizable into solver routes
    assert claims == []


# ---------------------------------------------------------------------------
# Artifact schema tests  (REQ-VERIFY-058)
# ---------------------------------------------------------------------------


def _make_stub_cohort(n: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "case_id": f"case-{i}",
            "prompt_seeds": {"baseline": i, "verify_only": i, "verify_repair": i},
        }
        for i in range(n)
    ]


def _make_stub_paired_runs(
    benchmark: str,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runs = []
    for model in [
        {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B"},
        {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it"},
    ]:
        for mode in ("baseline", "verify_only", "verify_repair"):
            runs.append(
                {
                    "benchmark": benchmark,
                    "model_name": model["name"],
                    "hf_id": model["hf_id"],
                    "mode": mode,
                    "cases": [
                        {
                            "case_id": c["case_id"],
                            "mode": mode,
                            "correct": True,
                            "formal_claims": [],
                        }
                        for c in cases
                    ],
                    "summary": {"n_cases": len(cases)},
                }
            )
    return runs


def _make_stub_statistics() -> dict[str, Any]:
    return {
        "Qwen3.5-0.8B": {
            "baseline": {"n_cases": 2, "accuracy": 0.5},
            "verify_only": {"n_cases": 2, "n_flagged": 0},
            "verify_repair": {"n_cases": 2, "repair_yield": 0.0},
        },
        "Gemma4-E4B-it": {
            "baseline": {"n_cases": 2, "accuracy": 0.5},
            "verify_only": {"n_cases": 2, "n_flagged": 0},
            "verify_repair": {"n_cases": 2, "repair_yield": 0.0},
        },
    }


def _make_stub_route_summary() -> dict[str, Any]:
    return {
        "by_route": {"arithmetic": 2},
        "by_verdict": {"supported": 1, "violated": 1, "abstain": 0},
        "total_claims": 2,
        "abstain_rate": 0.0,
    }


# REQ-VERIFY-058: artifact schema has required top-level keys
def test_build_artifact_payload_schema_keys(tmp_path: Path) -> None:
    module = load_module()
    repo = _make_repo(tmp_path)
    output_path = repo / "results" / "experiment_246_results.json"
    ckpt_dir = repo / "results" / "checkpoints" / "experiment_246"

    gsm8k_cases = _make_stub_cohort(2)
    constraint_ir_cases = _make_stub_cohort(2)

    payload = module.build_artifact_payload(
        output_path=output_path,
        gsm8k_cohort=gsm8k_cases,
        gsm8k_cohort_meta={"source_artifact": "results/experiment_235_results.json", "benchmark": "gsm8k_semantic", "case_count": 2},
        constraint_ir_cohort=constraint_ir_cases,
        constraint_ir_cohort_meta={"source_artifact": "results/experiment_221_results.json", "benchmark": "constraint_ir", "case_count": 2},
        gsm8k_paired_runs=_make_stub_paired_runs("gsm8k_semantic", gsm8k_cases),
        constraint_ir_paired_runs=_make_stub_paired_runs("constraint_ir", constraint_ir_cases),
        gsm8k_route_summary=_make_stub_route_summary(),
        constraint_ir_route_summary=_make_stub_route_summary(),
        gsm8k_statistics=_make_stub_statistics(),
        constraint_ir_statistics=_make_stub_statistics(),
        started_at="2026-04-13T00:00:00Z",
        finished_at="2026-04-13T00:01:00Z",
        runtime_seconds=60.0,
        checkpoint_dir=ckpt_dir,
        max_repairs=3,
        inference_mode="simulated",
    )

    # Top-level structure
    assert payload["experiment"] == 246
    assert payload["run_date"] == "20260413"
    assert "schema" in payload
    assert "metadata" in payload
    assert "benchmarks" in payload

    # Both benchmark slices present
    benchmarks = payload["benchmarks"]
    assert "gsm8k_semantic" in benchmarks
    assert "constraint_ir" in benchmarks

    # Each benchmark block has required keys
    for key in ("gsm8k_semantic", "constraint_ir"):
        block = benchmarks[key]
        assert "cohort" in block
        assert "paired_runs" in block
        assert "route_summary" in block
        assert "statistics" in block


# REQ-VERIFY-058: run_date is always 20260413
def test_build_artifact_payload_run_date(tmp_path: Path) -> None:
    module = load_module()
    repo = _make_repo(tmp_path)
    output_path = repo / "results" / "experiment_246_results.json"
    ckpt_dir = repo / "results" / "checkpoints" / "experiment_246"

    gsm8k_cases = _make_stub_cohort(1)
    constraint_ir_cases = _make_stub_cohort(1)

    payload = module.build_artifact_payload(
        output_path=output_path,
        gsm8k_cohort=gsm8k_cases,
        gsm8k_cohort_meta={"source_artifact": "results/experiment_235_results.json", "benchmark": "gsm8k_semantic", "case_count": 1},
        constraint_ir_cohort=constraint_ir_cases,
        constraint_ir_cohort_meta={"source_artifact": "results/experiment_221_results.json", "benchmark": "constraint_ir", "case_count": 1},
        gsm8k_paired_runs=_make_stub_paired_runs("gsm8k_semantic", gsm8k_cases),
        constraint_ir_paired_runs=_make_stub_paired_runs("constraint_ir", constraint_ir_cases),
        gsm8k_route_summary=_make_stub_route_summary(),
        constraint_ir_route_summary=_make_stub_route_summary(),
        gsm8k_statistics=_make_stub_statistics(),
        constraint_ir_statistics=_make_stub_statistics(),
        started_at="2026-04-13T00:00:00Z",
        finished_at="2026-04-13T00:01:00Z",
        runtime_seconds=60.0,
        checkpoint_dir=ckpt_dir,
        max_repairs=3,
        inference_mode="simulated",
    )

    assert payload["run_date"] == "20260413"


# REQ-VERIFY-058: artifact schema artifact type string is stable
def test_build_artifact_payload_schema_artifact_string(tmp_path: Path) -> None:
    module = load_module()
    repo = _make_repo(tmp_path)
    output_path = repo / "results" / "experiment_246_results.json"
    ckpt_dir = repo / "results" / "checkpoints" / "experiment_246"

    gsm8k_cases = _make_stub_cohort(1)
    constraint_ir_cases = _make_stub_cohort(1)

    payload = module.build_artifact_payload(
        output_path=output_path,
        gsm8k_cohort=gsm8k_cases,
        gsm8k_cohort_meta={"source_artifact": "results/experiment_235_results.json", "benchmark": "gsm8k_semantic", "case_count": 1},
        constraint_ir_cohort=constraint_ir_cases,
        constraint_ir_cohort_meta={"source_artifact": "results/experiment_221_results.json", "benchmark": "constraint_ir", "case_count": 1},
        gsm8k_paired_runs=_make_stub_paired_runs("gsm8k_semantic", gsm8k_cases),
        constraint_ir_paired_runs=_make_stub_paired_runs("constraint_ir", constraint_ir_cases),
        gsm8k_route_summary=_make_stub_route_summary(),
        constraint_ir_route_summary=_make_stub_route_summary(),
        gsm8k_statistics=_make_stub_statistics(),
        constraint_ir_statistics=_make_stub_statistics(),
        started_at="2026-04-13T00:00:00Z",
        finished_at="2026-04-13T00:01:00Z",
        runtime_seconds=60.0,
        checkpoint_dir=ckpt_dir,
        max_repairs=3,
        inference_mode="simulated",
    )

    assert payload["schema"]["artifact"] == module.SCHEMA_ARTIFACT
    assert "gsm8k_semantic" in payload["schema"]["benchmark_slices"]
    assert "constraint_ir" in payload["schema"]["benchmark_slices"]


# REQ-VERIFY-058: cohort block preserves case_ids and source_artifact
def test_build_artifact_payload_cohort_preserves_ids(tmp_path: Path) -> None:
    module = load_module()
    repo = _make_repo(tmp_path)
    output_path = repo / "results" / "experiment_246_results.json"
    ckpt_dir = repo / "results" / "checkpoints" / "experiment_246"

    gsm8k_cases = [
        {"case_id": "gsm8k-1", "prompt_seeds": {"baseline": 1, "verify_only": 1, "verify_repair": 1}},
        {"case_id": "gsm8k-2", "prompt_seeds": {"baseline": 2, "verify_only": 2, "verify_repair": 2}},
    ]
    constraint_ir_cases = [
        {"case_id": "exp211-1", "prompt_seeds": {"baseline": 3, "verify_only": 3, "verify_repair": 3}},
    ]

    payload = module.build_artifact_payload(
        output_path=output_path,
        gsm8k_cohort=gsm8k_cases,
        gsm8k_cohort_meta={"source_artifact": "results/experiment_235_results.json", "benchmark": "gsm8k_semantic", "case_count": 2},
        constraint_ir_cohort=constraint_ir_cases,
        constraint_ir_cohort_meta={"source_artifact": "results/experiment_221_results.json", "benchmark": "constraint_ir", "case_count": 1},
        gsm8k_paired_runs=_make_stub_paired_runs("gsm8k_semantic", gsm8k_cases),
        constraint_ir_paired_runs=_make_stub_paired_runs("constraint_ir", constraint_ir_cases),
        gsm8k_route_summary=_make_stub_route_summary(),
        constraint_ir_route_summary=_make_stub_route_summary(),
        gsm8k_statistics=_make_stub_statistics(),
        constraint_ir_statistics=_make_stub_statistics(),
        started_at="2026-04-13T00:00:00Z",
        finished_at="2026-04-13T00:01:00Z",
        runtime_seconds=60.0,
        checkpoint_dir=ckpt_dir,
        max_repairs=3,
        inference_mode="simulated",
    )

    gsm8k_cohort_block = payload["benchmarks"]["gsm8k_semantic"]["cohort"]
    assert gsm8k_cohort_block["case_ids"] == ["gsm8k-1", "gsm8k-2"]
    assert gsm8k_cohort_block["case_count"] == 2
    assert "experiment_235_results.json" in gsm8k_cohort_block["source_artifact"]

    ir_cohort_block = payload["benchmarks"]["constraint_ir"]["cohort"]
    assert ir_cohort_block["case_ids"] == ["exp211-1"]
    assert "experiment_221_results.json" in ir_cohort_block["source_artifact"]


# REQ-VERIFY-058: artifact is JSON-serializable
def test_build_artifact_payload_is_json_serializable(tmp_path: Path) -> None:
    module = load_module()
    repo = _make_repo(tmp_path)
    output_path = repo / "results" / "experiment_246_results.json"
    ckpt_dir = repo / "results" / "checkpoints" / "experiment_246"

    gsm8k_cases = _make_stub_cohort(2)
    constraint_ir_cases = _make_stub_cohort(2)

    payload = module.build_artifact_payload(
        output_path=output_path,
        gsm8k_cohort=gsm8k_cases,
        gsm8k_cohort_meta={"source_artifact": "results/experiment_235_results.json", "benchmark": "gsm8k_semantic", "case_count": 2},
        constraint_ir_cohort=constraint_ir_cases,
        constraint_ir_cohort_meta={"source_artifact": "results/experiment_221_results.json", "benchmark": "constraint_ir", "case_count": 2},
        gsm8k_paired_runs=_make_stub_paired_runs("gsm8k_semantic", gsm8k_cases),
        constraint_ir_paired_runs=_make_stub_paired_runs("constraint_ir", constraint_ir_cases),
        gsm8k_route_summary=_make_stub_route_summary(),
        constraint_ir_route_summary=_make_stub_route_summary(),
        gsm8k_statistics=_make_stub_statistics(),
        constraint_ir_statistics=_make_stub_statistics(),
        started_at="2026-04-13T00:00:00Z",
        finished_at="2026-04-13T00:01:00Z",
        runtime_seconds=60.0,
        checkpoint_dir=ckpt_dir,
        max_repairs=3,
        inference_mode="simulated",
    )

    # Must not raise
    serialized = json.dumps(payload)
    restored = json.loads(serialized)
    assert restored["experiment"] == 246


# REQ-VERIFY-058: write_artifact creates file and parent dirs
def test_write_artifact_creates_file(tmp_path: Path) -> None:
    module = load_module()
    output_path = tmp_path / "deep" / "nested" / "exp246.json"

    module.write_artifact(output_path, {"experiment": 246, "run_date": "20260413"})

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["experiment"] == 246


# ---------------------------------------------------------------------------
# CLI / constants tests
# ---------------------------------------------------------------------------


# REQ-VERIFY-058: constants are correctly set
def test_module_constants() -> None:
    module = load_module()

    assert module.RUN_DATE == "20260413"
    assert module.EXPERIMENT == 246
    assert module.SCHEMA_ARTIFACT == "carnot.solver_semantic_live.v1"


# REQ-VERIFY-058: model specs use the two required models
def test_model_specs_are_fixed() -> None:
    module = load_module()

    names = [m["name"] for m in module.MODEL_SPECS]
    hf_ids = [m["hf_id"] for m in module.MODEL_SPECS]

    assert names == ["Qwen3.5-0.8B", "Gemma4-E4B-it"]
    assert hf_ids == ["Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it"]


# REQ-VERIFY-058: MODE_ORDER is the standard three-mode sequence
def test_mode_order_is_fixed() -> None:
    module = load_module()

    assert module.MODE_ORDER == ("baseline", "verify_only", "verify_repair")


# REQ-VERIFY-058: CLI parser exposes required flags
def test_build_parser_has_required_flags() -> None:
    module = load_module()
    parser = module.build_parser()

    # Smoke-test with defaults
    args = parser.parse_args([])

    assert args.max_repairs == module.DEFAULT_MAX_REPAIRS
    assert args.output == module.default_output_path()
    assert args.checkpoint_dir == module.default_checkpoint_dir()


# REQ-VERIFY-058: default_output_path returns experiment_246_results.json
def test_default_output_path_contains_246() -> None:
    module = load_module()
    path = module.default_output_path()

    assert "246" in path.name
    assert path.suffix == ".json"


# REQ-VERIFY-058: default_checkpoint_dir contains experiment_246
def test_default_checkpoint_dir_contains_246() -> None:
    module = load_module()
    ckpt_dir = module.default_checkpoint_dir()

    assert "246" in str(ckpt_dir)


# REQ-VERIFY-058: summarize_benchmark_runs produces correct baseline accuracy
def test_summarize_benchmark_runs_baseline_accuracy() -> None:
    module = load_module()

    baseline_runs = [
        {"case_id": "a", "correct": True, "formal_claims": [], "latency_seconds": 1.0, "prompt_tokens": 10, "response_tokens": 5, "total_tokens": 15},
        {"case_id": "b", "correct": False, "formal_claims": [], "latency_seconds": 1.2, "prompt_tokens": 10, "response_tokens": 5, "total_tokens": 15},
        {"case_id": "c", "correct": True, "formal_claims": [], "latency_seconds": 0.8, "prompt_tokens": 10, "response_tokens": 5, "total_tokens": 15},
        {"case_id": "d", "correct": False, "formal_claims": [], "latency_seconds": 1.5, "prompt_tokens": 10, "response_tokens": 5, "total_tokens": 15},
    ]
    verify_only_runs = [
        {"case_id": "a", "flagged": False, "accepted_correct": True, "formal_claims": [], "latency_seconds": 0.5, "prompt_tokens": 5, "response_tokens": 3, "total_tokens": 8},
        {"case_id": "b", "flagged": True, "accepted_correct": False, "formal_claims": [], "latency_seconds": 0.6, "prompt_tokens": 5, "response_tokens": 3, "total_tokens": 8},
        {"case_id": "c", "flagged": False, "accepted_correct": True, "formal_claims": [], "latency_seconds": 0.4, "prompt_tokens": 5, "response_tokens": 3, "total_tokens": 8},
        {"case_id": "d", "flagged": True, "accepted_correct": False, "formal_claims": [], "latency_seconds": 0.7, "prompt_tokens": 5, "response_tokens": 3, "total_tokens": 8},
    ]
    verify_repair_runs = [
        {"case_id": "a", "correct": True, "repaired": False, "n_repairs": 0, "formal_claims": [], "latency_seconds": 0.0, "prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0},
        {"case_id": "b", "correct": True, "repaired": True, "n_repairs": 1, "formal_claims": [], "latency_seconds": 1.0, "prompt_tokens": 20, "response_tokens": 10, "total_tokens": 30},
        {"case_id": "c", "correct": True, "repaired": False, "n_repairs": 0, "formal_claims": [], "latency_seconds": 0.0, "prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0},
        {"case_id": "d", "correct": False, "repaired": False, "n_repairs": 1, "formal_claims": [], "latency_seconds": 0.9, "prompt_tokens": 20, "response_tokens": 10, "total_tokens": 30},
    ]

    stats = module.summarize_benchmark_runs(
        baseline_runs=baseline_runs,
        verify_only_runs=verify_only_runs,
        verify_repair_runs=verify_repair_runs,
    )

    assert stats["baseline"]["accuracy"] == pytest.approx(0.5)
    assert stats["verify_only"]["n_flagged"] == 2
    assert stats["verify_repair"]["n_repaired"] == 1
    assert stats["verify_repair"]["repair_yield"] == pytest.approx(0.5)  # 1 repaired / 2 wrong
    assert "paired_deltas" in stats
