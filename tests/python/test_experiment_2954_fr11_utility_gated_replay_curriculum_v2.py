"""Tests for Exp 2954 FR-11 utility-gated replay curriculum.

Spec: REQ-LEARN-2954, SCENARIO-LEARN-2954,
SCENARIO-LEARN-2954-ROLLBACK.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import utility_gated_replay_curriculum_v2 as exp


SPEC_PATH = exp.REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(stable_id: str, row_status: str, seed: int) -> dict[str, Any]:
    passed = row_status == "candidate_passed"
    syntax_success = row_status in {"candidate_passed", "candidate_failed"}
    extraction_success = row_status != "candidate_extraction_failed"
    return {
        "stable_id": stable_id,
        "row_status": row_status,
        "random_seed": seed,
        "passed": passed,
        "syntax_success": syntax_success,
        "runtime_success": passed,
        "extraction_success": extraction_success,
        "error_type": None if passed else row_status,
    }


def _write_ready_upstreams(root: Path) -> None:
    protocol_rel = Path("results/experiment_2946_nested_exp2910_protocol.json")
    _write_json(
        root / exp.EXP2947_REL_PATH,
        {
            "honest_verdict": "complete: nonuniform_continuation_replay_curriculum_piloted",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "replay_count_distribution": {
                "structural_memory_bootstrap": 26,
                "process_reward_replay": 14,
                "continuation_boundary_replay": 10,
                "retention_guard_replay": 14,
            },
        },
    )
    _write_json(
        root / exp.EXP2946_REL_PATH,
        {
            "honest_verdict": "complete: retain continuation executed",
            "inference_substrate": "live_llm_inference",
            "protocol_artifact_path": protocol_rel.as_posix(),
            "pass_at_1": 0.06,
            "pass_at_k": 0.16,
        },
    )
    _write_json(
        root / exp.EXP2940_REL_PATH,
        {
            "honest_verdict": "complete: verifier provides meaningful information",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "code_status_energy_values": [0.0, 1.0, 2.0, 3.0],
            "max_f1_operating_point": {"threshold": 1.0, "ppv": 0.9, "recall": 1.0},
            "code_status_energy_definition": {
                "0.0": "extracted, syntax-valid, runtime-clean candidate",
                "1.0": "extracted and syntax-valid candidate with runtime failure",
                "2.0": "extracted candidate with syntax failure",
                "3.0": "candidate extraction failed",
            },
        },
    )
    _write_json(
        root / protocol_rel,
        {
            "candidate_results": [
                _candidate("task-0", "candidate_syntax_failed", 10),
                _candidate("task-1", "candidate_extraction_failed", 11),
                _candidate("task-2", "candidate_extraction_failed", 12),
                _candidate("task-3", "candidate_extraction_failed", 13),
                _candidate("task-4", "candidate_passed", 14),
            ],
            "per_task_results": [],
        },
    )


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        started_at=20.0,
        clock=lambda: 23.5,
        tests_run=("focused-pytest",),
    )


def test_req_learn_2954_spec_anchor_exists() -> None:
    """REQ-LEARN-2954: OpenSpec anchors the utility-gated replay artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-2954" in spec
    assert "SCENARIO-LEARN-2954" in spec
    assert "SCENARIO-LEARN-2954-ROLLBACK" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert "aggregation_from_upstream_artifacts" in spec


def test_scenario_learn_2954_accepts_utility_positive_nonforgetting_update(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-2954: utility improves and the stable guard is retained."""

    _write_ready_upstreams(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text())

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["self_learning_utility_artifact_ready"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["model_weights_mutated"] is False
    assert artifact["live_model_invoked"] is False

    assert artifact["heldout_utility_after"] > artifact["heldout_utility_baseline"]
    assert artifact["heldout_utility_delta"] > 0.0
    assert artifact["self_learning_utility_positive"] is True
    assert artifact["forgetting_guard_passed"] is True
    assert artifact["rollback_triggered"] is False
    assert artifact["forgetting_guard_metric_after"] >= artifact["forgetting_guard_metric_before"]

    policy_names = {policy["policy_name"] for policy in artifact["replay_policies_compared"]}
    assert policy_names == {"flat_replay", "nonuniform_replay_exp2947", "utility_gated_replay"}
    assert artifact["slice_manifest"]["train_replay"]["candidate_count"] == 3
    assert artifact["slice_manifest"]["heldout_utility"]["candidate_count"] == 1
    assert artifact["slice_manifest"]["forgetting_guard"]["candidate_count"] == 1
    assert "future design context" in artifact["soft_radial_projection_note"]

    source_by_id = {source["experiment_id"]: source for source in artifact["source_artifacts"]}
    assert set(source_by_id) == {"exp2940", "exp2946", "exp2946_protocol", "exp2947"}
    assert source_by_id["exp2940"]["sha256"] == _sha256(tmp_path / exp.EXP2940_REL_PATH)
    assert source_by_id["exp2946_protocol"]["role"] == "repair_taxonomy_rows"


def test_scenario_learn_2954_rollback_when_forgetting_guard_degrades() -> None:
    """SCENARIO-LEARN-2954-ROLLBACK: guard degradation restores baseline weights."""

    heldout = (
        exp.ReplayExample(
            stable_id="heldout",
            split="heldout_utility",
            taxonomy="extraction_repair",
            status_energy=3.0,
            utility_signal=1.0,
            forgetting_signal=0.0,
            row_status="candidate_extraction_failed",
            random_seed=1,
        ),
    )
    guard = (
        exp.ReplayExample(
            stable_id="guard",
            split="forgetting_guard",
            taxonomy="verified_pass",
            status_energy=0.0,
            utility_signal=0.1,
            forgetting_signal=1.0,
            row_status="candidate_passed",
            random_seed=2,
        ),
    )
    baseline = {"verified_pass": 0.6, "extraction_repair": 0.4}
    candidate = {"verified_pass": 0.1, "extraction_repair": 0.9}

    decision = exp.evaluate_policy_update(
        baseline_weights=baseline,
        candidate_weights=candidate,
        heldout_examples=heldout,
        guard_examples=guard,
    )

    assert decision["utility_improved"] is True
    assert decision["forgetting_guard_passed"] is False
    assert decision["rollback_triggered"] is True
    assert decision["accepted_weights"] == baseline
    assert decision["heldout_utility_after"] > decision["heldout_utility_baseline"]
    assert decision["forgetting_guard_metric_after"] < decision["forgetting_guard_metric_before"]


def test_req_learn_2954_blocks_missing_or_empty_upstreams(tmp_path: Path) -> None:
    """REQ-LEARN-2954-1/2: missing sources and empty slices fail closed."""

    missing = exp.build_artifact(_config(tmp_path))

    assert missing["honest_verdict"] == "blocked_missing_upstream_artifact"
    assert missing["self_learning_utility_artifact_ready"] is False
    assert missing["missing_fields"] == ["source:exp2947", "source:exp2946", "source:exp2940"]

    _write_ready_upstreams(tmp_path)
    protocol_path = tmp_path / "results/experiment_2946_nested_exp2910_protocol.json"
    protocol_path.unlink()

    no_protocol = exp.build_artifact(_config(tmp_path))

    assert no_protocol["honest_verdict"] == "blocked_missing_exp2946_protocol_artifact"
    assert no_protocol["missing_fields"] == ["source:exp2946_protocol"]

    _write_ready_upstreams(tmp_path)
    protocol = json.loads(protocol_path.read_text())
    protocol["candidate_results"] = []
    _write_json(protocol_path, protocol)

    empty = exp.build_artifact(_config(tmp_path))

    assert empty["honest_verdict"] == "blocked_empty_replay_slices"
    assert empty["self_learning_utility_artifact_ready"] is False
    assert empty["replay_policies_compared"] == []


def test_req_learn_2954_weight_helpers_are_deterministic() -> None:
    """REQ-LEARN-2954-3/4: weights normalize and reject invalid input."""

    assert exp.normalize_weights({"b": 3.0, "a": 1.0}) == {"a": 0.25, "b": 0.75}
    with pytest.raises(ValueError, match="positive"):
        exp.normalize_weights({"a": 0.0})

    examples = (
        exp.ReplayExample("a", "train_replay", "syntax_repair", 2.0, 0.5, 0.0, "x", 1),
        exp.ReplayExample("b", "train_replay", "extraction_repair", 3.0, 1.0, 0.0, "y", 2),
    )
    target = exp.target_weights_from_training(
        examples,
        baseline_weights={"syntax_repair": 0.4, "extraction_repair": 0.4, "verified_pass": 0.2},
        guard_taxonomies=("verified_pass",),
    )

    assert target["verified_pass"] == pytest.approx(0.2)
    assert target["extraction_repair"] > target["syntax_repair"]
    assert sum(target.values()) == pytest.approx(1.0)


def test_req_learn_2954_edge_cases_stay_deterministic() -> None:
    """REQ-LEARN-2954-2/4: defensive paths return stable, conservative values."""

    assert exp.replay_examples_from_rows("not-a-sequence", {}) == ()
    assert exp.nonuniform_weights_from_exp2947(
        {"replay_count_distribution": {}},
        ("syntax_repair", "verified_pass"),
    ) == {"syntax_repair": 0.5, "verified_pass": 0.5}

    rows = [
        _candidate("task-0", "candidate_passed", 1),
        {"stable_id": "task-1", "row_status": "unknown_status", "random_seed": 2},
    ]
    examples = exp.replay_examples_from_rows(
        rows,
        {"max_f1_operating_point": {"ppv": "not-a-number"}},
    )

    assert len(examples) == 1
    assert examples[0].utility_signal == pytest.approx(0.1)

    baseline = {"verified_pass": 0.7, "syntax_repair": 0.3}
    assert exp.target_weights_from_training(
        (examples[0],),
        baseline_weights=baseline,
        guard_taxonomies=("verified_pass",),
    ) == exp.normalize_weights(baseline)

    zero_signal = exp.ReplayExample("z", "heldout_utility", "syntax_repair", 2.0, 0.0, 0.0, "x", 3)
    assert exp.policy_utility({"syntax_repair": 1.0}, (zero_signal,)) == 0.0
    assert exp.forgetting_guard_metric({"syntax_repair": 1.0}, (zero_signal,)) == 1.0
    assert exp._verdict(False, True) == "complete: utility_candidate_rolled_back_by_forgetting_guard"
    assert exp._verdict(False, False) == "complete: utility_gated_replay_no_positive_heldout_gain"
