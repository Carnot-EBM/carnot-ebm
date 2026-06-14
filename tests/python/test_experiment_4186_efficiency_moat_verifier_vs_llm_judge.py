"""Tests for Exp 4186 verifier efficiency moat versus LLM-as-judge.

Spec refs: REQ-VERIFY-4186, SCENARIO-VERIFY-4186.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import efficiency_moat_verifier_vs_llm_judge_4186 as mod


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _code_rows() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "HumanEval/0",
            "baseline_passed": False,
            "repair_passed": True,
            "extracted_constraints": 2,
        },
        {
            "task_id": "HumanEval/1",
            "baseline_passed": True,
            "repair_passed": True,
            "extracted_constraints": 0,
        },
        {
            "task_id": "HumanEval/2",
            "baseline_passed": False,
            "repair_passed": True,
            "extracted_constraints": 1,
        },
        {
            "task_id": "HumanEval/3",
            "baseline_passed": False,
            "repair_passed": False,
            "extracted_constraints": 3,
        },
    ]


def _selection(task_id: str, chosen_index: int, row: dict[str, Any], latency: float, tokens: int) -> dict[str, Any]:
    correct = bool(row["baseline_passed"] if chosen_index == 0 else row["repair_passed"])
    return {
        "task_id": task_id,
        "chosen_index": chosen_index,
        "chosen_role": "baseline" if chosen_index == 0 else "repair",
        "chosen_correct": correct,
        "cost": {
            "chosen_index": chosen_index,
            "latency_s": latency,
            "prompt_tokens": tokens - 2,
            "completion_tokens": 2,
            "total_tokens": tokens,
            "raw_output": str(chosen_index),
        },
    }


def _make_repo(
    root: Path,
    *,
    headroom: float = 0.50,
    ready: bool = True,
    domain: str = "code",
    include_all_judge_tasks: bool = True,
) -> Path:
    source = root / "results" / "experiment_1999_code_verification_humaneval.json"
    rows = _code_rows()
    _write_json(source, {"honest_verdict": "complete: fixture", "results": rows})

    choices = [1, 0, 0, 0]
    latencies = [2.0, 3.0, 4.0, 1.0]
    tokens = [100, 110, 90, 100]
    selections = [
        _selection(row["task_id"], choice, row, latency, token_count)
        for row, choice, latency, token_count in zip(rows, choices, latencies, tokens, strict=True)
    ]
    if not include_all_judge_tasks:
        selections = selections[:-1]
    judge_costs = [selection["cost"] for selection in selections]
    total_tokens = sum(int(cost["total_tokens"]) for cost in judge_costs)
    total_latency = sum(float(cost["latency_s"]) for cost in judge_costs)
    n_calls = len(judge_costs)
    _write_json(
        root / "results" / "experiment_4185_headroom_recensus_llm_judge_harness.json",
        {
            "honest_verdict": "complete: fixture",
            "max_selectable_headroom": float(headroom),
            "headroom_present_domain": domain if headroom >= mod.HEADROOM_THRESHOLD else "",
            "llm_judge_ready": ready,
            "judge_cost_meter": {
                "mean_judge_latency_s": total_latency / n_calls if n_calls else 0.0,
                "mean_judge_tokens": total_tokens / n_calls if n_calls else 0.0,
                "mean_prompt_tokens": 98.0,
                "mean_completion_tokens": 2.0,
                "n_calls": n_calls,
            },
            "model_specs": {
                "selected_judge": {
                    "name": "Gemma4-26B-A4B-it",
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "model_path": "/models/gemma-4-26b.gguf",
                },
                "prompt_version": "exp4185-best-candidate-index-v1",
                "loader": "llama_cpp.Llama",
            },
            "judge_pass1_smoke": {
                "n_candidate_sets": n_calls,
                "pass1_accuracy": sum(bool(sel["chosen_correct"]) for sel in selections) / n_calls
                if n_calls
                else 0.0,
                "selections": selections,
            },
            "per_domain_headroom": {
                "code": {
                    "oracle_at_k": 0.75,
                    "sc_vote_pass1": 0.25,
                    "selectable_headroom": 0.50,
                    "artifact_flags": {"source": str(source)},
                }
            },
            "random_seed": 4185,
            "reproducibility_checksum": "a" * 64,
            "field_principles": {},
            "spec_refs": ["REQ-VERIFY-4185", "SCENARIO-VERIFY-4185"],
            "inference_substrate": "fixture",
            "duration_s": 1.0,
        },
    )
    _write_json(
        root / "results" / "experiment_4176_vstar_selector_model.json",
        {
            "model_type": "logistic_regression",
            "feature_names": list(mod.FEATURE_NAMES),
            "intercept": 0.0,
            "coefficients": [4.0, 0.0, 0.0, 0.0],
            "random_seed": 4176,
            "reproducibility_checksum": "b" * 64,
            "spec_refs": ["REQ-VERIFY-4176", "SCENARIO-VERIFY-4176"],
        },
    )
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "adversarial_verify.py").write_text(
        "import json, sys\nprint(json.dumps({'flagged': [], 'path': sys.argv[-1]}))\n",
        encoding="utf-8",
    )
    return root


def test_req_4186_spec_declares_four_arm_efficiency_contract() -> None:
    """REQ-VERIFY-4186: OpenSpec declares the runner, fields, and principles."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4186",
        "SCENARIO-VERIFY-4186",
        "python/carnot/reporting/efficiency_moat_verifier_vs_llm_judge_4186.py",
        "results/experiment_4186_efficiency_moat_verifier_vs_llm_judge.py",
        "results/experiment_4186_efficiency_moat_verifier_vs_llm_judge.json",
        "complete_efficiency_moat_deferred_no_headroom_or_no_judge",
        "verifier_efficiency_win",
        "accuracy_parity_vs_judge",
        "cost_ratio_vs_judge",
        "REAL cost",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4186_headroom_present_reports_accuracy_and_real_cost(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4186: verifier, judge, vote, and oracle are compared with real costs."""

    root = _make_repo(tmp_path)

    artifact = mod.run(root, random_seed=mod.RANDOM_SEED, bootstrap_resamples=300)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["positive_control_confirmed"] is True
    assert artifact["verifier_efficiency_win"] is True
    assert artifact["accuracy_parity_vs_judge"]["arm_a_pass1"] == pytest.approx(0.75)
    assert artifact["accuracy_parity_vs_judge"]["arm_j_pass1"] == pytest.approx(0.50)
    assert artifact["accuracy_parity_vs_judge"]["delta"] == pytest.approx(0.25)
    assert artifact["accuracy_parity_vs_judge"]["ci95"][1] >= 0.0
    assert artifact["cost_ratio_vs_judge"]["arm_j_total_wall_clock_s"] == pytest.approx(10.0)
    assert artifact["cost_ratio_vs_judge"]["arm_j_total_tokens"] == 400
    assert artifact["cost_ratio_vs_judge"]["arm_a_total_tokens"] == 0
    assert artifact["cost_ratio_vs_judge"]["tokens"] == 0.0
    assert artifact["cost_ratio_vs_judge"]["wall_clock"] <= 0.1
    assert artifact["arms"]["arm_b_sc_vote"]["pass1"] == pytest.approx(0.25)
    assert artifact["arms"]["oracle"]["pass_at_k"] == pytest.approx(0.75)
    assert artifact["model_specs"]["selected_judge"]["hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == ["REQ-VERIFY-4186", "SCENARIO-VERIFY-4186"]
    assert artifact["acceptance_gate"] is True
    assert artifact["adversarial_verify"]["returncode"] == 0

    written = json.loads((root / mod.OUTPUT_REL).read_text(encoding="utf-8"))
    assert written == artifact


def test_scenario_4186_defers_without_headroom_or_ready_judge(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4186: A1 gate failures stop before scoring."""

    low_headroom = mod.run(_make_repo(tmp_path / "low", headroom=0.05))
    mod.validate_artifact(low_headroom)
    assert low_headroom["honest_verdict"] == "complete_efficiency_moat_deferred_no_headroom_or_no_judge"
    assert low_headroom["positive_control_confirmed"] is False
    assert low_headroom["verifier_efficiency_win"] is False
    assert low_headroom["accuracy_parity_vs_judge"]["status"] == "deferred_no_headroom_or_no_judge"
    assert low_headroom["cost_ratio_vs_judge"]["status"] == "deferred_no_headroom_or_no_judge"
    assert low_headroom["acceptance_gate"] is True
    assert low_headroom["adversarial_verify"]["status"] == "not_run"

    no_judge = mod.run(_make_repo(tmp_path / "no-judge", ready=False))
    assert no_judge["honest_verdict"] == "complete_efficiency_moat_deferred_no_headroom_or_no_judge"
    assert no_judge["acceptance_gate"] is True


def test_req_4186_validation_and_blocked_paths_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-4186: malformed inputs and invalid artifacts fail loudly."""

    assert mod._finite_float(True, 0.7) == 0.7
    assert mod._finite_float("bad", 0.3) == 0.3
    assert mod._bootstrap_ci([], random_seed=mod.RANDOM_SEED, resamples=10) == [0.0, 0.0]
    assert mod._base_task_id("HumanEval/0#repeat1") == "HumanEval/0"
    assert mod._ratio(1.0, 0.0) is None

    missing = mod.run(tmp_path / "missing")
    assert missing["honest_verdict"] == "complete_efficiency_moat_deferred_no_headroom_or_no_judge"
    assert missing["acceptance_gate"] is True

    unsupported = mod.run(_make_repo(tmp_path / "unsupported", domain="math"))
    assert unsupported["honest_verdict"] == "blocked_unsupported_headroom_domain_math"
    assert unsupported["acceptance_gate"] is False

    missing_judge = mod.run(_make_repo(tmp_path / "missing-judge", include_all_judge_tasks=False))
    assert missing_judge["honest_verdict"] == "blocked_missing_judge_selections"
    assert missing_judge["acceptance_gate"] is False

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(mod.BlockedRun, match="blocked_malformed_json_artifact"):
        mod._read_json_object(bad_json)

    valid = mod._empty_artifact(
        "complete_efficiency_moat_deferred_no_headroom_or_no_judge",
        "deferred_no_headroom_or_no_judge",
        mod.RANDOM_SEED,
        0.1,
    )
    mod.validate_artifact(valid)
    for payload, message in (
        ({k: v for k, v in valid.items() if k != "honest_verdict"}, "missing required"),
        ({**valid, "honest_verdict": "missing prefix"}, "terminal-prefixed"),
        ({**valid, "verifier_efficiency_win": 1}, "bare bool"),
        ({**valid, "positive_control_confirmed": 0}, "bare bool"),
        ({**valid, "random_seed": True}, "bare int"),
        ({**valid, "reproducibility_checksum": ""}, "checksum"),
        ({**valid, "field_principles": {}}, "field_principles"),
        ({**valid, "spec_refs": []}, "spec_refs"),
        ({**valid, "accuracy_parity_vs_judge": []}, "must be an object"),
        ({**valid, "cost_ratio_vs_judge": []}, "must be an object"),
        ({**valid, "model_specs": []}, "must be an object"),
    ):
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)
