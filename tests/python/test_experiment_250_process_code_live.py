"""Tests for Experiment 250: process-aware live HumanEval benchmark.

Covers artifact schema shape, cohort reuse from Exp 238, comparison summary
generation, process-integrity flag recording, and right-for-wrong-reasons
counting.  No live model loading — all generation is stubbed.

Spec: REQ-CODE-028, REQ-CODE-029, REQ-CODE-030,
      REQ-VERIFY-061, REQ-VERIFY-062
SCENARIO-CODE-026, SCENARIO-CODE-027, SCENARIO-CODE-028,
SCENARIO-VERIFY-065, SCENARIO-VERIFY-066, SCENARIO-VERIFY-067,
SCENARIO-VERIFY-068, SCENARIO-VERIFY-069
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    module_path = REPO_ROOT / "scripts" / "experiment_250_process_code_live.py"
    python_dir = str(REPO_ROOT / "python")
    removed = False
    if python_dir in sys.path:
        sys.path.remove(python_dir)
        removed = True
    spec = importlib.util.spec_from_file_location(
        "experiment_250_process_code_live", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if removed:
            sys.path.insert(0, python_dir)
    return module


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------


def make_case(case_id: str, *, dataset_idx: int = 0) -> dict[str, Any]:
    """Return a minimal HumanEval-style case compatible with the Exp 250 schema."""
    seed = 1000 + dataset_idx
    return {
        "case_id": case_id,
        "dataset_idx": dataset_idx,
        "task_id": f"HumanEval/{dataset_idx}",
        "prompt": f"def fn_{dataset_idx}(x: int) -> int:\n    pass\n",
        "test": "def check(candidate):\n    assert candidate(1) == 2\n",
        "entry_point": f"fn_{dataset_idx}",
        "sample_position": dataset_idx + 1,
        "prompt_seeds": {"baseline": seed, "verify_only": seed, "verify_repair": seed},
    }


def make_process_flags(
    *,
    process_valid: bool = True,
    right_for_wrong_reasons: bool = False,
    defects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "process_valid": process_valid,
        "outcome_correct": True,
        "right_for_wrong_reasons": right_for_wrong_reasons,
        "defects": defects or [],
        "process_label": "clean" if process_valid else "right_answer_wrong_process",
        "run_date": "20260413",
    }


def make_case_result(
    case: dict[str, Any],
    *,
    official_passed: bool = True,
    pbt_accepted: bool = True,
    spec_accepted: bool = True,
    process_accepted: bool = True,
    right_for_wrong_reasons: bool = False,
    repair_accepted: bool = True,
    n_repairs: int = 0,
) -> dict[str, Any]:
    """Build a minimal per-case result matching the Exp 250 schema."""
    pf = make_process_flags(
        process_valid=process_accepted,
        right_for_wrong_reasons=right_for_wrong_reasons,
        defects=(
            [{"kind": "outcome_correct_process_invalid", "detail": "rfwr", "step_id": None, "evidence": {}}]
            if right_for_wrong_reasons
            else []
        ),
    )
    return {
        "case_id": str(case["case_id"]),
        "dataset_idx": int(case["dataset_idx"]),
        "task_id": str(case["task_id"]),
        "entry_point": str(case["entry_point"]),
        "baseline": {
            "official_passed": official_passed,
            "body": "    return x + 1",
            "candidate_code": f"{case['prompt']}    return x + 1\n",
        },
        "official_tests_verify_only": {"accepted": official_passed},
        "pbt_verify_only": {
            "accepted": pbt_accepted,
            "harness_passing_rejected_by_pbt": official_passed and not pbt_accepted,
        },
        "spec_aware_verify_only": {
            "accepted": spec_accepted,
            "harness_passing_rejected_by_specs": pbt_accepted and not spec_accepted,
        },
        "process_aware_verify_only": {
            "accepted": process_accepted,
            "right_for_wrong_reasons": right_for_wrong_reasons,
        },
        "verify_repair": {
            "accepted": repair_accepted,
            "official_passed": official_passed,
            "repaired": repair_accepted and n_repairs > 0,
            "n_repairs": n_repairs,
            "final_body": "    return x + 1",
            "final_code": f"{case['prompt']}    return x + 1\n",
        },
        "process_flags": {"baseline": dict(pf), "final": dict(pf), "history": [dict(pf)]},
        "history": [],
    }


def make_reference_artifact(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a minimal reference artifact resembling the Exp 238 output."""
    return {
        "experiment": 238,
        "benchmark": "humaneval_dual_model_spec",
        "run_date": "20260413",
        "schema": {},
        "metadata": {},
        "cohort": {
            "case_count": len(cases),
            "case_ids": [c["case_id"] for c in cases],
            "task_ids": [c["task_id"] for c in cases],
            "cases": [dict(c) for c in cases],
            "shared_with_reference_artifact": True,
        },
        "model_runs": {},
        "comparison": {},
        "blockers": [],
        "run_status": "complete",
    }


# ---------------------------------------------------------------------------
# Module smoke / constants
# ---------------------------------------------------------------------------


def test_constants():
    """Module constants match the task specification."""
    module = load_module()
    assert module.RUN_DATE == "20260413"
    assert module.EXPERIMENT_ID == 250
    assert module.DEFAULT_MAX_REPAIRS == 3
    assert module.DEFAULT_MAX_NEW_TOKENS == 220
    assert module.DEFAULT_PBT_MAX_EXAMPLES == 64
    assert "process_integrity" in module.VERIFIER_STACK
    assert len(module.MODEL_SPECS) == 2
    names = {m["name"] for m in module.MODEL_SPECS}
    assert "Qwen3.5-0.8B" in names
    assert "Gemma4-E4B-it" in names


def test_default_paths():
    """Default paths embed the experiment number."""
    module = load_module()
    assert "250" in str(module.default_output_path())
    assert "250" in str(module.default_checkpoint_dir())
    assert "238" in str(module.default_reference_artifact_path())


def test_utc_now():
    module = load_module()
    ts = module.utc_now()
    assert "T" in ts and ts.endswith("Z")


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def test_build_parser_defaults():
    """Parser exposes all expected flags with correct defaults."""
    module = load_module()
    parser = module.build_parser()
    args = parser.parse_args([])
    assert args.max_repairs == 3
    assert args.max_new_tokens == 220
    assert args.pbt_max_examples == 64
    assert args.bootstrap_samples == 10_000
    assert args.checkpoint_interval == 10
    assert args.output == module.default_output_path()
    assert args.checkpoint_dir == module.default_checkpoint_dir()


# ---------------------------------------------------------------------------
# SCENARIO-CODE-026: cohort reuse from Exp 238 reference artifact
# ---------------------------------------------------------------------------


def test_load_shared_cohort_from_reference_artifact(tmp_path: Path) -> None:
    """SCENARIO-CODE-026: load_shared_cohort reads cases from the Exp 238 artifact."""
    module = load_module()
    cases = [make_case(f"humaneval-{i}", dataset_idx=i) for i in range(3)]
    artifact = make_reference_artifact(cases)
    artifact_path = tmp_path / "exp238.json"
    artifact_path.write_text(json.dumps(artifact) + "\n", encoding="utf-8")

    loaded_cases, meta = module.load_shared_cohort(artifact_path)
    assert len(loaded_cases) == 3
    assert [c["case_id"] for c in loaded_cases] == [c["case_id"] for c in cases]
    assert meta["source_experiment"] == 238
    assert meta["reference_experiment"] == 238
    assert meta["case_count"] == 3


def test_load_shared_cohort_rejects_mismatched_seeds(tmp_path: Path) -> None:
    """load_shared_cohort raises when prompt seeds are mismatched."""
    module = load_module()
    bad_case = {
        "case_id": "humaneval-0",
        "dataset_idx": 0,
        "task_id": "HumanEval/0",
        "prompt": "def fn(): pass\n",
        "test": "def check(candidate): pass",
        "entry_point": "fn",
        "sample_position": 1,
        "prompt_seeds": {"baseline": 1, "verify_only": 1, "verify_repair": 99},  # mismatch
    }
    artifact = make_reference_artifact([bad_case])
    artifact_path = tmp_path / "exp238_bad.json"
    artifact_path.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatched prompt seeds"):
        module.load_shared_cohort(artifact_path)


def test_load_shared_cohort_rejects_missing_cases(tmp_path: Path) -> None:
    """load_shared_cohort raises on empty cohort."""
    module = load_module()
    artifact = make_reference_artifact([])
    artifact_path = tmp_path / "exp238_empty.json"
    artifact_path.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing valid cohort cases"):
        module.load_shared_cohort(artifact_path)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    """save_checkpoint / load_checkpoint are inverse operations."""
    module = load_module()
    ckpt = tmp_path / "model.json"
    payload = {
        "case_ids": ["humaneval-0", "humaneval-1"],
        "results_by_case": {"humaneval-0": {"case_id": "humaneval-0"}},
    }
    module.save_checkpoint(ckpt, payload)
    loaded = module.load_checkpoint(ckpt, ["humaneval-0", "humaneval-1"])
    assert loaded["case_ids"] == ["humaneval-0", "humaneval-1"]
    assert "humaneval-0" in loaded["results_by_case"]


def test_checkpoint_stale_cohort_returns_fresh(tmp_path: Path) -> None:
    """load_checkpoint returns a fresh dict when cohort ids changed."""
    module = load_module()
    ckpt = tmp_path / "model.json"
    payload = {
        "case_ids": ["humaneval-OLD"],
        "results_by_case": {"humaneval-OLD": {}},
    }
    module.save_checkpoint(ckpt, payload)
    loaded = module.load_checkpoint(ckpt, ["humaneval-NEW"])
    assert loaded["results_by_case"] == {}
    assert loaded["case_ids"] == ["humaneval-NEW"]


def test_checkpoint_path_naming():
    """checkpoint_path encodes the model name safely."""
    module = load_module()
    path = module.checkpoint_path("/tmp/ckpts", "Qwen3.5-0.8B")
    assert "qwen3" in str(path).lower()
    assert "exp250" in str(path).lower()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def test_build_generation_prompt_contains_case_prompt():
    module = load_module()
    case = make_case("humaneval-0")
    prompt = module.build_generation_prompt(case)
    assert case["prompt"] in prompt
    assert "expert Python programmer" in prompt
    assert "ONLY the function body" in prompt


def test_build_repair_prompt_identical_format_across_models():
    """Same case + evaluation → same repair prompt regardless of model name.

    Spec: REQ-CODE-028 (identical repair prompt format across models)
    """
    module = load_module()
    case = make_case("humaneval-0")
    evaluation: dict[str, Any] = {
        "official_tests": {"passed": False, "error_type": "AssertionError", "error_message": "fail", "stdout": ""},
        "instrumentation": {"constraint_feedback": [], "dynamic_violations": []},
        "pbt": {"verified": False, "n_failures": 1, "violations": ["sorted order violated"], "repair_feedback": "sort output"},
        "explicit_specs": {"n_violations": 0, "violations": [], "repair_hints": []},
        "process_integrity": {"process_valid": False, "defects": [
            {"kind": "outcome_correct_process_invalid", "detail": "rfwr", "evidence": {}}
        ]},
    }
    # Both models call the same build_repair_prompt — model name is not an argument.
    p1 = module.build_repair_prompt(case, previous_body="    return x", evaluation=evaluation, repair_idx=0)
    p2 = module.build_repair_prompt(case, previous_body="    return x", evaluation=evaluation, repair_idx=0)
    assert p1 == p2
    assert "repair attempt 1" in p1
    assert "Process-integrity findings" in p1  # Exp 250 addition


def test_build_repair_prompt_no_process_section_when_clean():
    """No process-integrity section emitted when process is valid with no defects."""
    module = load_module()
    case = make_case("humaneval-0")
    evaluation: dict[str, Any] = {
        "official_tests": {"passed": False, "error_type": "AssertionError", "error_message": "fail", "stdout": ""},
        "instrumentation": {"constraint_feedback": [], "dynamic_violations": []},
        "pbt": {"verified": True, "n_failures": 0, "violations": [], "repair_feedback": ""},
        "explicit_specs": {"n_violations": 0, "violations": [], "repair_hints": []},
        "process_integrity": {"process_valid": True, "defects": []},
    }
    prompt = module.build_repair_prompt(case, previous_body="    return x", evaluation=evaluation, repair_idx=0)
    assert "Process-integrity findings" not in prompt


# ---------------------------------------------------------------------------
# Process corpus row and process check helpers
# ---------------------------------------------------------------------------


def test_derive_process_label_clean():
    module = load_module()
    label = module._derive_process_label(True, True, 0)
    assert label == "clean"


def test_derive_process_label_right_for_wrong_reasons_pbt():
    module = load_module()
    label = module._derive_process_label(True, False, 0)
    assert label == "right_answer_wrong_process"


def test_derive_process_label_right_for_wrong_reasons_spec():
    module = load_module()
    label = module._derive_process_label(True, True, 3)
    assert label == "right_answer_wrong_process"


def test_derive_process_label_wrong_answer_partial():
    module = load_module()
    label = module._derive_process_label(False, True, 0)
    assert label == "wrong_answer_partially_sound_process"


def test_derive_process_label_wrong_answer_wrong():
    module = load_module()
    label = module._derive_process_label(False, False, 2)
    assert label == "wrong_answer_wrong_process"


def make_minimal_evaluation(
    *,
    official_passed: bool = True,
    pbt_verified: bool = True,
    n_pbt_failures: int = 0,
    n_spec_violations: int = 0,
) -> dict[str, Any]:
    return {
        "official_tests": {"passed": official_passed, "error_type": "", "error_message": "", "stdout": ""},
        "instrumentation": {},
        "pbt": {"verified": pbt_verified, "n_failures": n_pbt_failures, "violations": [], "repair_feedback": ""},
        "explicit_specs": {"n_violations": n_spec_violations, "violations": [], "repair_hints": []},
    }


def test_build_process_corpus_row_clean():
    module = load_module()
    ev = make_minimal_evaluation()
    row = module._build_process_corpus_row(ev)
    assert row["outcome_label"] == "correct"
    assert row["process_label"] == "clean"
    assert row["process_evidence"]["n_unsupported_claims"] == 0
    assert row["process_evidence"]["verifier_verdict"] == "abstain"
    assert row["process_evidence"]["max_premise_support"] == 1.0
    assert "repair_context" not in row


def test_build_process_corpus_row_rfwr():
    module = load_module()
    ev = make_minimal_evaluation(official_passed=True, pbt_verified=True, n_spec_violations=2)
    row = module._build_process_corpus_row(ev)
    assert row["process_label"] == "right_answer_wrong_process"
    assert row["process_evidence"]["n_unsupported_claims"] == 2
    assert row["process_evidence"]["verifier_verdict"] == "violated"


def test_build_process_corpus_row_with_repair_context():
    module = load_module()
    ev = make_minimal_evaluation(official_passed=True, pbt_verified=True)
    row = module._build_process_corpus_row(ev, prior_outcome="incorrect")
    assert row["repair_context"]["prior_outcome"] == "incorrect"


def test_run_process_check_clean_has_no_rfwr():
    """SCENARIO-VERIFY-065: clean evaluation → process_valid True, no defects."""
    module = load_module()
    ev = make_minimal_evaluation()
    result = module._run_process_check(ev)
    assert result["process_valid"] is True
    assert result["right_for_wrong_reasons"] is False
    assert result["defects"] == []


def test_run_process_check_rfwr_detected():
    """SCENARIO-VERIFY-066: official passes but spec violations → rfwr detected."""
    module = load_module()
    ev = make_minimal_evaluation(official_passed=True, pbt_verified=True, n_spec_violations=3)
    result = module._run_process_check(ev)
    # process_label = right_answer_wrong_process → OUTCOME_CORRECT_PROCESS_INVALID defect
    assert result["right_for_wrong_reasons"] is True
    defect_kinds = [d["kind"] for d in result["defects"]]
    assert "outcome_correct_process_invalid" in defect_kinds


def test_run_process_check_repair_stall_detected():
    """SCENARIO-VERIFY-068: prior=incorrect, current=incorrect → repair_stall detected."""
    module = load_module()
    ev = make_minimal_evaluation(official_passed=False, pbt_verified=False, n_pbt_failures=2)
    result = module._run_process_check(ev, prior_outcome="incorrect")
    defect_kinds = [d["kind"] for d in result["defects"]]
    assert "repair_stall" in defect_kinds


def test_run_process_check_repair_regression_detected():
    """SCENARIO-VERIFY-069: prior=correct, current=incorrect → repair_regression detected."""
    module = load_module()
    ev = make_minimal_evaluation(official_passed=False, pbt_verified=False)
    result = module._run_process_check(ev, prior_outcome="correct")
    defect_kinds = [d["kind"] for d in result["defects"]]
    assert "repair_regression" in defect_kinds


# ---------------------------------------------------------------------------
# Stage flags
# ---------------------------------------------------------------------------


def test_stage_flags_all_true():
    module = load_module()
    case = make_case("humaneval-0")
    result = make_case_result(case, official_passed=True, pbt_accepted=True, spec_accepted=True,
                              process_accepted=True, repair_accepted=True)
    flags = module._stage_flags(result)
    assert flags["baseline"] is True
    assert flags["official_tests_verify_only"] is True
    assert flags["pbt_verify_only"] is True
    assert flags["spec_aware_verify_only"] is True
    assert flags["process_aware_verify_only"] is True
    assert flags["verify_repair"] is True


def test_stage_flags_process_rejects_spec_passing():
    """process_aware_verify_only can be False even when spec_aware is True."""
    module = load_module()
    case = make_case("humaneval-0")
    result = make_case_result(case, official_passed=True, pbt_accepted=True, spec_accepted=True,
                              process_accepted=False, repair_accepted=False)
    flags = module._stage_flags(result)
    assert flags["spec_aware_verify_only"] is True
    assert flags["process_aware_verify_only"] is False


# ---------------------------------------------------------------------------
# SCENARIO-CODE-027: process integrity stats in summarize_model_results
# ---------------------------------------------------------------------------


def test_summarize_model_results_empty():
    module = load_module()
    summary = module.summarize_model_results([], n_bootstrap=100, seed=1)
    assert "stages" in summary
    assert "process_integrity" in summary
    assert summary["process_integrity"]["right_for_wrong_reasons_count"] == 0
    assert "process_aware_verify_only" in summary["stages"]


def test_summarize_model_results_counts_rfwr():
    """right_for_wrong_reasons_count reflects actual RFWR cases.

    Spec: REQ-VERIFY-062, SCENARIO-VERIFY-066
    """
    module = load_module()
    cases_in = [make_case(f"humaneval-{i}", dataset_idx=i) for i in range(4)]
    results = [
        make_case_result(cases_in[0], right_for_wrong_reasons=True, process_accepted=False),
        make_case_result(cases_in[1], right_for_wrong_reasons=True, process_accepted=False),
        make_case_result(cases_in[2], right_for_wrong_reasons=False, process_accepted=True),
        make_case_result(cases_in[3], right_for_wrong_reasons=False, process_accepted=True),
    ]
    summary = module.summarize_model_results(results, n_bootstrap=100, seed=1)
    pi = summary["process_integrity"]
    assert pi["right_for_wrong_reasons_count"] == 2
    assert pi["total_cases"] == 4
    # stage counter
    stage = summary["stages"]["process_aware_verify_only"]
    assert stage["right_for_wrong_reasons"] == 2


def test_summarize_model_results_process_stage_accepted_pass_at_1():
    """process_aware_verify_only accepted_pass_at_1 counts only clean cases."""
    module = load_module()
    cases_in = [make_case(f"humaneval-{i}", dataset_idx=i) for i in range(4)]
    results = [
        make_case_result(cases_in[0], process_accepted=True),
        make_case_result(cases_in[1], process_accepted=True),
        make_case_result(cases_in[2], process_accepted=False, repair_accepted=False),
        make_case_result(cases_in[3], process_accepted=False, repair_accepted=False),
    ]
    summary = module.summarize_model_results(results, n_bootstrap=100, seed=1)
    pct = summary["stages"]["process_aware_verify_only"]["accepted_pass_at_1"]
    assert abs(pct - 0.5) < 0.05  # bootstrap CI may shift slightly


def test_summarize_model_results_defect_kind_counts():
    """defect_kind_counts aggregates defect kinds across all baseline process flags."""
    module = load_module()
    cases_in = [make_case(f"humaneval-{i}", dataset_idx=i) for i in range(2)]
    rfwr_defect: dict[str, Any] = {
        "kind": "outcome_correct_process_invalid",
        "detail": "rfwr",
        "step_id": None,
        "evidence": {},
    }
    results = [
        make_case_result(cases_in[0], right_for_wrong_reasons=True, process_accepted=False),
        make_case_result(cases_in[1]),
    ]
    # Inject actual defects into process_flags.baseline for case 0.
    results[0]["process_flags"]["baseline"]["defects"] = [rfwr_defect]
    summary = module.summarize_model_results(results, n_bootstrap=100, seed=1)
    dkc = summary["process_integrity"]["defect_kind_counts"]
    assert dkc.get("outcome_correct_process_invalid", 0) >= 1


def test_summarize_model_results_added_rejections_over_spec():
    """added_rejections_over_spec tallies cases where spec passes but process rejects."""
    module = load_module()
    cases_in = [make_case(f"humaneval-{i}", dataset_idx=i) for i in range(3)]
    results = [
        # spec passes, process rejects
        make_case_result(cases_in[0], spec_accepted=True, process_accepted=False, repair_accepted=False),
        # both pass
        make_case_result(cases_in[1], spec_accepted=True, process_accepted=True),
        # spec rejects
        make_case_result(cases_in[2], pbt_accepted=True, spec_accepted=False, process_accepted=False, repair_accepted=False),
    ]
    summary = module.summarize_model_results(results, n_bootstrap=100, seed=1)
    assert summary["stages"]["process_aware_verify_only"]["added_rejections_over_spec"] == 1


# ---------------------------------------------------------------------------
# SCENARIO-CODE-027: comparison summary (Gemma-vs-Qwen)
# ---------------------------------------------------------------------------


def test_build_comparison_summary_empty_returns_zero_block():
    """Empty model_runs → zero comparison block with process_aware_verify_only keys."""
    module = load_module()
    comparison = module.build_comparison_summary({}, n_bootstrap=100, seed=1, repair_budget=3)
    assert comparison["paired_case_count"] == 0
    assert "process_aware_verify_only" in comparison["stage_deltas"]
    assert "process_aware_verify_only" in comparison["stage_outcomes"]


def test_build_comparison_summary_paired_cases():
    """SCENARIO-CODE-027: comparison block includes process_aware_verify_only deltas.

    Spec: REQ-CODE-029
    """
    module = load_module()
    cases_in = [make_case(f"humaneval-{i}", dataset_idx=i) for i in range(3)]

    gemma_results = [
        make_case_result(cases_in[0], process_accepted=True),
        make_case_result(cases_in[1], process_accepted=False, repair_accepted=False),
        make_case_result(cases_in[2], process_accepted=True),
    ]
    qwen_results = [
        make_case_result(cases_in[0], process_accepted=False, repair_accepted=False),
        make_case_result(cases_in[1], process_accepted=True),
        make_case_result(cases_in[2], process_accepted=True),
    ]

    model_runs = {
        "Gemma4-E4B-it": {"per_problem_results": gemma_results},
        "Qwen3.5-0.8B": {"per_problem_results": qwen_results},
    }
    comparison = module.build_comparison_summary(
        model_runs, n_bootstrap=200, seed=7, repair_budget=3
    )
    assert comparison["paired_case_count"] == 3
    assert "process_aware_verify_only" in comparison["stage_deltas"]
    proc_stage = comparison["stage_outcomes"]["process_aware_verify_only"]
    # gemma wins case 0, qwen wins case 1, both win case 2
    assert proc_stage["gemma_only"] == 1
    assert proc_stage["qwen_only"] == 1
    assert proc_stage["both"] == 1
    assert proc_stage["neither"] == 0
    assert "methodology_note" in comparison


# ---------------------------------------------------------------------------
# SCENARIO-CODE-028: partial run preserves completed work and records blockers
# ---------------------------------------------------------------------------


def test_build_artifact_payload_schema(tmp_path: Path) -> None:
    """SCENARIO-CODE-028 / REQ-CODE-030: artifact schema shape validation."""
    module = load_module()
    cases_in = [make_case(f"humaneval-{i}", dataset_idx=i) for i in range(2)]
    results = [make_case_result(c) for c in cases_in]

    model_runs = {
        "Qwen3.5-0.8B": {
            "per_problem_results": results,
            "run_status": "complete",
            "blockers": [],
            "statistics": module.summarize_model_results(results, n_bootstrap=100, seed=1),
        }
    }
    comparison = module.build_comparison_summary(model_runs, n_bootstrap=100, seed=1, repair_budget=3)
    payload = module.build_artifact_payload(
        output_path=tmp_path / "out.json",
        cohort=cases_in,
        cohort_meta={
            "source_artifact": "results/experiment_238_results.json",
            "source_experiment": 238,
            "reference_experiment": 238,
            "reference_run_date": "20260413",
            "case_count": 2,
        },
        model_runs=model_runs,
        comparison=comparison,
        blockers=[],
        started_at="2026-04-13T00:00:00Z",
        finished_at="2026-04-13T00:01:00Z",
        runtime_seconds=60.0,
        checkpoint_dir=tmp_path / "ckpts",
        max_repairs=3,
        pbt_max_examples=64,
        bootstrap_samples=100,
        run_status="complete",
    )

    # Top-level required fields.
    for field in ("experiment", "benchmark", "run_date", "schema", "metadata",
                   "cohort", "model_runs", "comparison", "blockers", "run_status"):
        assert field in payload, f"Missing field: {field}"

    assert payload["experiment"] == 250
    assert payload["run_date"] == "20260413"
    assert payload["benchmark"] == "humaneval_dual_model_process"

    # Cohort metadata.
    cohort_block = payload["cohort"]
    assert cohort_block["reference_experiment"] == 238
    assert cohort_block["case_count"] == 2
    assert len(cohort_block["cases"]) == 2

    # Metadata includes Exp 238 as a source artifact.
    src = payload["metadata"]["source_artifacts"]
    assert any("238" in str(s) for s in src)


def test_build_artifact_payload_partial_run_status(tmp_path: Path) -> None:
    """Partial run → run_status is 'partial', blocker recorded."""
    module = load_module()
    payload = module.build_artifact_payload(
        output_path=tmp_path / "out.json",
        cohort=[],
        cohort_meta={"source_artifact": "", "source_experiment": 238,
                     "reference_experiment": 238, "reference_run_date": "", "case_count": 0},
        model_runs={},
        comparison={},
        blockers=[{"model_name": "Qwen3.5-0.8B", "stage": "model_load", "error": "OOM"}],
        started_at="2026-04-13T00:00:00Z",
        finished_at="2026-04-13T00:01:00Z",
        runtime_seconds=60.0,
        checkpoint_dir=tmp_path / "ckpts",
        max_repairs=3,
        pbt_max_examples=64,
        bootstrap_samples=100,
        run_status="partial",
    )
    assert payload["run_status"] == "partial"
    assert len(payload["blockers"]) == 1
    assert payload["blockers"][0]["error"] == "OOM"


# ---------------------------------------------------------------------------
# write_artifact roundtrip
# ---------------------------------------------------------------------------


def test_write_artifact_roundtrip(tmp_path: Path) -> None:
    module = load_module()
    payload = {"experiment": 250, "run_date": "20260413", "data": [1, 2, 3]}
    out = tmp_path / "out.json"
    module.write_artifact(out, payload)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == payload


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-067: process-integrity stats across process_flags.history
# ---------------------------------------------------------------------------


def test_process_integrity_stats_history_ignored_in_defect_counts():
    """_process_integrity_stats only reads baseline defects, not history entries."""
    module = load_module()
    case = make_case("humaneval-0")
    result = make_case_result(case, right_for_wrong_reasons=False)
    # Artificially add a repair-phase defect into history (not baseline).
    result["process_flags"]["history"].append({
        "process_valid": False,
        "right_for_wrong_reasons": False,
        "defects": [{"kind": "repair_stall", "detail": "stall", "step_id": None, "evidence": {}}],
        "outcome_correct": False,
        "process_label": "wrong_answer_wrong_process",
        "run_date": "20260413",
    })
    stats = module._process_integrity_stats([result])
    # repair_stall is in history, not baseline — must NOT appear in defect_kind_counts.
    assert "repair_stall" not in stats["defect_kind_counts"]


# ---------------------------------------------------------------------------
# Verifier stack label in comparison
# ---------------------------------------------------------------------------


def test_verifier_stack_includes_process_integrity():
    """Comparison summary carries the full verifier stack including process_integrity."""
    module = load_module()
    comparison = module.build_comparison_summary({}, n_bootstrap=100, seed=1, repair_budget=3)
    assert "process_integrity" in comparison["shared_verifier_stack"]
