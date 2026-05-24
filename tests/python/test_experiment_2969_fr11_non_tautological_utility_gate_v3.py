"""Tests for Exp 2969 non-tautological FR-11 utility gate.

Spec: REQ-LEARN-2969, SCENARIO-LEARN-2969,
SCENARIO-LEARN-2969-ROLLBACK, SCENARIO-LEARN-2969-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fr11_non_tautological_utility_gate_v3 as exp


SPEC_PATH = exp.REPO_ROOT / "openspec/capabilities/self-learning/spec.md"
REQUIRED_FIELDS = {
    "honest_verdict",
    "continuous_self_learning_task",
    "non_tautological_self_learning_ready",
    "source_artifacts",
    "split_checksums",
    "leakage_check_passed",
    "replay_policies_compared",
    "frozen_heldout_utility",
    "random_replay_heldout_utility",
    "prior_utility_gated_heldout_utility",
    "new_heldout_utility",
    "heldout_utility_delta_vs_random",
    "negative_control_delta",
    "forgetting_guard_passed",
    "rollback_triggered",
    "update_rule",
    "model_specs_if_live_llm_used",
    "inference_substrate",
    "duration_s",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(
    stable_id: str,
    mode: str,
    sample_index: int,
    *,
    passed: bool = False,
    syntax_success: bool = False,
    parser_status: str | None = None,
) -> dict[str, Any]:
    return {
        "stable_id": stable_id,
        "task_id": f"MBPP:{stable_id}",
        "sample_id": f"MBPP:{stable_id}:fixture",
        "mode": mode,
        "sample_index": sample_index,
        "random_seed": 2969 + sample_index,
        "passed": passed,
        "syntax_success": syntax_success,
        "parser_status": parser_status or ("parsed" if syntax_success else "syntax_error"),
        "verifier_accepted": passed,
        "false_accept": False,
        "test_status": "passed" if passed else "not_run",
        "candidate_manifest_sha256": hashlib.sha256(
            f"{stable_id}:{mode}:{sample_index}".encode()
        ).hexdigest(),
    }


def _logic_item(item_id: str, category: str, *, answer_correct: bool = False) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "failure_category": category,
        "answer_correct": answer_correct,
        "solver_formula_correct": category == "solver_verified_correct",
        "z3_executed": category in {"solver_verified_correct", "wrong_formula"},
        "raw_output_sha256": hashlib.sha256(item_id.encode()).hexdigest(),
    }


def _write_ready_upstreams(root: Path) -> None:
    _write_json(
        root / exp.EXP2954_REL_PATH,
        {
            "honest_verdict": "complete: utility_gated_replay_improved_heldout_without_forgetting",
            "continuous_self_learning_task": True,
            "self_learning_utility_artifact_ready": True,
            "heldout_utility_baseline": 0.24,
            "heldout_utility_after": 0.35,
            "heldout_utility_delta": 0.11,
            "forgetting_guard_passed": True,
            "rollback_triggered": False,
            "final_replay_weights": {
                "syntax_repair": 0.34,
                "runtime_repair": 0.08,
                "extraction_repair": 0.36,
                "verified_pass": 0.22,
            },
            "replay_policies_compared": [],
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
    )
    _write_json(
        root / exp.EXP2952_REL_PATH,
        {
            "honest_verdict": "complete: taxonomy-guided repair delta passed",
            "baseline_pass_at_1": 0.0,
            "repair_pass_at_1": 0.25,
            "candidate_manifest_sha256": "fixture-candidates",
            "candidate_evaluations": [
                _candidate("mbpp-11", "baseline_no_taxonomy", 0),
                _candidate("mbpp-14", "baseline_no_taxonomy", 1),
                _candidate("mbpp-16", "baseline_no_taxonomy", 2, syntax_success=True),
                _candidate("mbpp-12", "baseline_no_taxonomy", 3, passed=True, syntax_success=True),
                _candidate("mbpp-11", "taxonomy_guided", 4),
                _candidate("mbpp-11", "taxonomy_guided", 5, passed=True, syntax_success=True),
                _candidate("mbpp-14", "taxonomy_guided", 6),
                _candidate("mbpp-16", "taxonomy_guided", 7),
                _candidate("mbpp-12", "taxonomy_guided", 8, passed=True, syntax_success=True),
            ],
            "inference_substrate": "live_llm_inference",
        },
    )
    _write_json(
        root / exp.EXP2959_REL_PATH,
        {
            "honest_verdict": "complete: local SOTA logic proposals accepted_or_rejected_by_z3",
            "answer_accuracy": 0.083333,
            "failure_categories": {"unparseable": 2, "wrong_formula": 1, "solver_verified_correct": 1},
            "formalization_manifest_sha256": "logic-fixture",
            "per_item_results": [
                _logic_item("logic-001", "unparseable"),
                _logic_item("logic-002", "wrong_formula", answer_correct=True),
                _logic_item("logic-003", "solver_verified_correct", answer_correct=True),
                _logic_item("logic-004", "unparseable"),
            ],
            "inference_substrate": "live_llm_inference",
        },
    )
    _write_json(
        root / exp.EXP2960_REL_PATH,
        {
            "honest_verdict": "complete: matrix_v12_ready=true",
            "matrix_v12_ready": True,
            "self_learning_delta_summary": {
                "artifact_ready": True,
                "heldout_utility_after": 0.35,
                "heldout_utility_delta": 0.11,
                "forgetting_guard_passed": True,
            },
            "code_repair_delta_summary": {
                "taxonomy_repair_delta_pass": True,
                "repair_pass_at_1": 0.25,
                "false_accept_delta": -0.125,
            },
            "matrix_rows": [
                {
                    "row_id": "exp2953_threshold_policy",
                    "source_experiment_id": "exp2953",
                    "row_class": "clean",
                    "summary": {
                        "threshold_policy_ready": True,
                        "selected_default_threshold": 1.0,
                        "expected_ppv_at_default": 0.8888888888888888,
                        "expected_recall_at_default": 1.0,
                    },
                }
            ],
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
    )


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        started_at=10.0,
        clock=lambda: 14.25,
        tests_run=("focused-pytest",),
    )


def test_req_learn_2969_spec_anchor_exists() -> None:
    """REQ-LEARN-2969: OpenSpec declares the non-tautological replay contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-2969" in spec
    assert "SCENARIO-LEARN-2969" in spec
    assert "SCENARIO-LEARN-2969-ROLLBACK" in spec
    assert "SCENARIO-LEARN-2969-BLOCKED" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="aggregation_from_upstream_artifacts"' in spec


def test_scenario_learn_2969_builds_ready_non_tautological_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-2969: disjoint evidence gates readiness."""

    _write_ready_upstreams(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text())

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["non_tautological_self_learning_ready"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert artifact["model_specs_if_live_llm_used"] == []

    assert artifact["leakage_check_passed"] is True
    assert artifact["split_checksums"]["train_ids_sha256"]
    assert artifact["split_checksums"]["reward_source_sha256"] != artifact["split_checksums"][
        "heldout_target_sha256"
    ]
    assert artifact["split_checksums"]["overlap_counts"] == {
        "train_vs_replay": 0,
        "train_vs_heldout": 0,
        "replay_vs_heldout": 0,
        "reward_vs_heldout": 0,
    }

    assert artifact["new_heldout_utility"] > artifact["random_replay_heldout_utility"]
    assert artifact["new_heldout_utility"] > artifact["frozen_heldout_utility"]
    assert artifact["heldout_utility_delta_vs_random"] > 0.0
    assert artifact["negative_control_delta"] <= 0.0
    assert artifact["forgetting_guard_passed"] is True
    assert artifact["rollback_triggered"] is False

    policies = {policy["policy_name"]: policy for policy in artifact["replay_policies_compared"]}
    assert set(policies) == {
        "frozen_baseline",
        "random_replay",
        "prior_278_utility_gated_replay",
        "negative_control_uninformative",
        "non_tautological_utility_gated_replay",
    }
    assert policies["non_tautological_utility_gated_replay"]["accepted"] is True
    assert policies["negative_control_uninformative"]["accepted"] is False

    source_by_id = {source["experiment_id"]: source for source in artifact["source_artifacts"]}
    assert set(source_by_id) == {"exp2952", "exp2954", "exp2959", "exp2960"}
    assert source_by_id["exp2952"]["sha256"] == _sha256(tmp_path / exp.EXP2952_REL_PATH)
    assert source_by_id["exp2960"]["role"] == "matrix_v12_guard_and_summary"


def test_scenario_learn_2969_rollback_on_guard_degradation() -> None:
    """SCENARIO-LEARN-2969-ROLLBACK: any degraded guard rolls back the update."""

    heldout = (
        exp.EvidenceExample(
            item_id="heldout-syntax",
            domain="code",
            split="heldout",
            taxonomy="syntax_repair",
            reward_signal=0.0,
            utility_signal=1.0,
            guard_signal=0.0,
            source_id="heldout",
        ),
    )
    guards = (
        exp.EvidenceExample(
            item_id="guard-code",
            domain="code",
            split="guard",
            taxonomy="verified_pass",
            reward_signal=0.0,
            utility_signal=0.0,
            guard_signal=1.0,
            source_id="guard-code",
        ),
        exp.EvidenceExample(
            item_id="guard-logic",
            domain="logic",
            split="guard",
            taxonomy="logic_guard",
            reward_signal=0.0,
            utility_signal=0.0,
            guard_signal=1.0,
            source_id="guard-logic",
        ),
        exp.EvidenceExample(
            item_id="guard-threshold",
            domain="threshold_policy",
            split="guard",
            taxonomy="threshold_policy",
            reward_signal=0.0,
            utility_signal=0.0,
            guard_signal=1.0,
            source_id="guard-threshold",
        ),
    )
    baseline = {
        "syntax_repair": 0.25,
        "verified_pass": 0.25,
        "logic_guard": 0.25,
        "threshold_policy": 0.25,
    }
    candidate = {
        "syntax_repair": 0.85,
        "verified_pass": 0.05,
        "logic_guard": 0.05,
        "threshold_policy": 0.05,
    }

    decision = exp.evaluate_policy_update(
        baseline_weights=baseline,
        candidate_weights=candidate,
        heldout_examples=heldout,
        guard_examples=guards,
    )

    assert decision["utility_improved"] is True
    assert decision["forgetting_guard_passed"] is False
    assert decision["rollback_triggered"] is True
    assert decision["accepted_weights"] == exp.normalize_weights(baseline)
    assert decision["guard_metrics_after"]["stable_code"] < decision["guard_metrics_before"][
        "stable_code"
    ]
    assert decision["guard_metrics_after"]["stable_logic"] < decision["guard_metrics_before"][
        "stable_logic"
    ]
    assert decision["guard_metrics_after"]["threshold_policy"] < decision["guard_metrics_before"][
        "threshold_policy"
    ]


def test_scenario_learn_2969_blocked_missing_sources_and_fields(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2969-BLOCKED: missing evidence fails closed."""

    missing = exp.build_artifact(_config(tmp_path))

    assert missing["honest_verdict"] == "blocked_missing_upstream_artifact"
    assert missing["non_tautological_self_learning_ready"] is False
    assert missing["leakage_check_passed"] is False
    assert missing["missing_fields"] == [
        "source:exp2954",
        "source:exp2952",
        "source:exp2959",
        "source:exp2960",
    ]
    assert REQUIRED_FIELDS <= set(missing)

    _write_ready_upstreams(tmp_path)
    exp2952 = json.loads((tmp_path / exp.EXP2952_REL_PATH).read_text())
    exp2952.pop("candidate_evaluations")
    _write_json(tmp_path / exp.EXP2952_REL_PATH, exp2952)

    malformed = exp.build_artifact(_config(tmp_path))

    assert malformed["honest_verdict"] == "blocked_missing_required_fields"
    assert "exp2952:candidate_evaluations" in malformed["missing_fields"]
    assert malformed["replay_policies_compared"] == []


def test_req_learn_2969_helpers_detect_leakage_and_negative_control() -> None:
    """REQ-LEARN-2969-2/4: checksums and negative controls are deterministic."""

    examples = (
        exp.EvidenceExample("train-a", "code", "train", "syntax_repair", 1.0, 0.0, 0.0, "r1"),
        exp.EvidenceExample("replay-a", "code", "replay", "runtime_repair", 0.5, 0.0, 0.0, "r2"),
        exp.EvidenceExample("held-a", "code", "heldout", "syntax_repair", 0.0, 1.0, 0.0, "h1"),
    )

    checks = exp.compute_split_checksums(examples)
    assert checks["overlap_counts"]["reward_vs_heldout"] == 0
    assert exp.leakage_check(checks) is True

    leaked = (
        *examples,
        exp.EvidenceExample("held-a", "code", "train", "syntax_repair", 1.0, 0.0, 0.0, "h1"),
    )
    leaked_checks = exp.compute_split_checksums(leaked)
    assert leaked_checks["overlap_counts"]["train_vs_heldout"] == 1
    assert leaked_checks["overlap_counts"]["reward_vs_heldout"] == 1
    assert exp.leakage_check(leaked_checks) is False

    random_weights = exp.normalize_weights({"syntax_repair": 1.0, "runtime_repair": 1.0})
    negative_weights = exp.negative_control_weights(("syntax_repair", "runtime_repair"))
    heldout = (examples[2],)

    assert negative_weights == random_weights
    assert exp.policy_utility(negative_weights, heldout) - exp.policy_utility(
        random_weights,
        heldout,
    ) == pytest.approx(0.0)


def test_req_learn_2969_defensive_paths_are_conservative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-2969: defensive branches fail closed without live inference."""

    _write_ready_upstreams(tmp_path)
    matrix = json.loads((tmp_path / exp.EXP2960_REL_PATH).read_text())
    matrix["matrix_rows"] = [
        {"row_id": "unrelated"},
        {"row_id": "exp2953_threshold_policy", "summary": {"threshold_policy_ready": False}},
    ]
    _write_json(tmp_path / exp.EXP2960_REL_PATH, matrix)

    insufficient = exp.build_artifact(_config(tmp_path))

    assert insufficient["honest_verdict"] == "blocked_insufficient_disjoint_slices"
    assert "guard:threshold_policy" in insufficient["missing_fields"]
    assert insufficient["leakage_check_passed"] is True

    assert exp.ExperimentConfig(clock=lambda: 1.5).start_time() == pytest.approx(1.5)
    with pytest.raises(ValueError, match="positive"):
        exp.normalize_weights({"syntax_repair": 0.0})

    assert exp.frozen_baseline_weights(("syntax_repair",)) == {"syntax_repair": 1.0}
    assert exp.prior_utility_gated_weights(
        {"candidate_replay_weights": {"syntax_repair": 2.0}},
        ("syntax_repair", "runtime_repair"),
    ) == {"syntax_repair": 1.0}
    assert exp.prior_utility_gated_weights({}, ("syntax_repair", "runtime_repair")) == {
        "runtime_repair": 0.5,
        "syntax_repair": 0.5,
    }
    assert exp.target_weights_from_reward(
        (),
        baseline_weights={"syntax_repair": 1.0},
        guard_taxonomies=(),
    ) == {"syntax_repair": 1.0}
    assert exp.policy_utility({"syntax_repair": 1.0}, ()) == 0.0
    assert exp.observed_taxonomies((), {}) == exp.TAXONOMY_ORDER

    assert exp.read_json_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp.read_json_object(malformed) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp.read_json_object(list_json) == {}

    assert exp._code_examples("not rows") == ()
    assert exp._code_examples([{"mode": "taxonomy_guided"}]) == ()
    assert exp._threshold_guard_examples({"matrix_rows": matrix["matrix_rows"]}) == ()
    assert exp._code_taxonomy({"parser_status": "extraction_failed"}) == "extraction_repair"
    assert exp._code_taxonomy({"syntax_success": True}) == "runtime_repair"
    assert exp.forgetting_guard_metrics({"syntax_repair": 1.0}, ()) == {
        "stable_code": 1.0,
        "stable_logic": 1.0,
        "threshold_policy": 1.0,
    }

    empty_checks = exp.compute_split_checksums(())
    errors = exp._slice_errors((), empty_checks)
    assert errors == [
        "split:train",
        "split:replay",
        "split:heldout",
        "split:guard",
        "split:leakage_check",
        "guard:code",
        "guard:logic",
        "guard:threshold_policy",
    ]
    assert "exp2954:self_learning_utility_artifact_ready_true" in exp._missing_required_fields(
        {
            "exp2954": {"self_learning_utility_artifact_ready": False, "heldout_utility_after": 0.0, "final_replay_weights": {}},
            "exp2952": {"candidate_evaluations": [], "baseline_pass_at_1": 0.0, "repair_pass_at_1": 0.0},
            "exp2959": {"per_item_results": [], "failure_categories": {}, "formalization_manifest_sha256": "x"},
            "exp2960": {"matrix_v12_ready": False, "matrix_rows": [], "self_learning_delta_summary": {}},
        }
    )
    assert exp._verdict(False, True) == (
        "complete: non_tautological_candidate_rolled_back_by_forgetting_guard"
    )
    assert exp._verdict(False, False) == "complete: non_tautological_self_learning_not_ready"
    assert exp._sequence("abc") == ()

    monkeypatch.setattr(exp, "write_artifact", lambda: {})
    assert exp.main() == 0
