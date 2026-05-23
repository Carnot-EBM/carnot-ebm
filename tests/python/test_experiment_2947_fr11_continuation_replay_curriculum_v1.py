"""Tests for Exp 2947 FR-11 continuation replay curriculum.

Spec: REQ-LEARN-2947,
      SCENARIO-LEARN-2947,
      SCENARIO-LEARN-2947-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import fr11_continuation_replay_curriculum_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ready_upstreams(root: Path) -> None:
    results = root / "results"
    _write_json(
        results / exp.EXP2918_FILENAME,
        {
            "honest_verdict": "complete: verifier_process_rewards_updated_replay_scheduler",
            "online_self_learning_ready": True,
            "replay_scheduler_updated": True,
            "delta_energy_proxy": 0.2,
            "pdi_proxy": 0.4,
            "forgetting_rate": 0.0,
            "replay_corpus_summary": {
                "code_rows": 32,
                "hardware_rows": 8,
                "prior_fr11_rows": 6,
                "held_out_prior_rows": 2,
            },
        },
    )
    _write_json(
        results / exp.EXP2933_FILENAME,
        {
            "honest_verdict": "complete: kan_rbf_importance_self_learning_passed",
            "kan_cl_self_learning_ready": True,
            "utility_delta_vs_replay_only": 0.5,
            "energy_proxy_delta": 0.4,
            "forgetting_rate": 0.0,
            "updated_knot_or_rbf_count": 12,
            "dataset_manifest": {
                "constraint_count": 3,
                "train_example_count": 72,
                "holdout_example_count": 48,
            },
        },
    )
    _write_json(
        results / exp.EXP2942_FILENAME,
        {
            "honest_verdict": "complete: kv260_fixed_n64_latency_profile_recorded",
            "inference_substrate": "hardware_smoke",
            "measured_n_values": [64],
            "unsupported_n_values": [128, 256, 512, 1024],
            "per_n_results": [{"n": 64, "per_sample_us_median": 25.0}],
            "bitstream_supports_variable_n": False,
        },
    )


def test_req_learn_2947_spec_anchor_exists() -> None:
    """REQ-LEARN-2947: OpenSpec anchors the curriculum artifact schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-2947" in spec
    assert "SCENARIO-LEARN-2947" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert "curriculum_schedule_used" in spec
    assert "replay_count_distribution" in spec


def test_scenario_learn_2947_curriculum_replaces_flat_uniform(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2947: ready upstreams produce non-uniform replay counts."""

    _write_ready_upstreams(tmp_path)

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            replay_budget=64,
            started_at=10.0,
            clock=lambda: 12.25,
        )
    )

    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["curriculum_schedule_used"] == "fr11_continuation_curriculum_v1"
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["replay_budget"] == 64
    assert artifact["flat_uniform_sampling_used"] is False

    distribution = artifact["replay_count_distribution"]
    assert sum(distribution.values()) == 64
    assert set(distribution) == {
        "structural_memory_bootstrap",
        "process_reward_replay",
        "continuation_boundary_replay",
        "retention_guard_replay",
    }
    assert len(set(distribution.values())) > 1
    assert distribution["structural_memory_bootstrap"] > distribution["continuation_boundary_replay"]

    assert len(artifact["cited_upstream_artifacts"]) == 3
    for citation in artifact["cited_upstream_artifacts"]:
        assert set(citation) == {"experiment_id", "path", "fields_imported", "sha256"}
        assert len(citation["sha256"]) == 64
        assert citation["fields_imported"]

    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact


def test_scenario_learn_2947_dirty_upstreams_block_curriculum(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2947-BLOCKED: missing or unready upstreams fail closed."""

    missing = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )

    assert missing["honest_verdict"] == "blocked_missing_exp2918_artifact"
    assert missing["replay_count_distribution"] == {}
    assert missing["curriculum_schedule_used"] == "blocked"
    assert missing["cited_upstream_artifacts"] == []

    _write_ready_upstreams(tmp_path)
    _write_json(
        tmp_path / "results" / exp.EXP2933_FILENAME,
        {"kan_cl_self_learning_ready": False},
    )

    unready = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )

    assert unready["honest_verdict"] == "blocked_exp2933_not_ready"
    assert "exp2933_not_ready" in unready["failed_gates"]

    (tmp_path / "results" / exp.EXP2918_FILENAME).write_text("{broken", encoding="utf-8")
    malformed = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )

    assert malformed["honest_verdict"] == "blocked_malformed_exp2918_artifact"
    assert "exp2918_artifact_malformed" in malformed["failed_gates"]

    (tmp_path / "results" / exp.EXP2918_FILENAME).write_text("[]", encoding="utf-8")
    non_mapping = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )

    assert non_mapping["honest_verdict"] == "blocked_malformed_exp2918_artifact"


def test_req_learn_2947_allocation_is_deterministic_and_validated(tmp_path: Path) -> None:
    """REQ-LEARN-2947-2/3: replay allocation is deterministic and validates inputs."""

    first = exp.allocate_replay_counts({"a": 3.0, "b": 1.0, "c": 1.0}, replay_budget=10)
    second = exp.allocate_replay_counts({"c": 1.0, "b": 1.0, "a": 3.0}, replay_budget=10)

    assert first == second
    assert first == {"a": 5, "b": 3, "c": 2}

    with pytest.raises(ValueError, match="replay_budget"):
        exp.allocate_replay_counts({"a": 1.0, "b": 1.0}, replay_budget=1)
    with pytest.raises(ValueError, match="positive curriculum score"):
        exp.allocate_replay_counts({"a": 0.0}, replay_budget=4)

    _write_ready_upstreams(tmp_path)
    exp2918_path = tmp_path / "results" / exp.EXP2918_FILENAME
    exp2918 = json.loads(exp2918_path.read_text(encoding="utf-8"))
    exp2918["pdi_proxy"] = "not-a-number"
    _write_json(exp2918_path, exp2918)

    artifact = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["curriculum_signal_scores"]["process_reward_replay"] == pytest.approx(0.2)

    config = exp.ExperimentConfig(repo_root=tmp_path)
    assert config.output_dir() == tmp_path / "results"
    assert config.output_path() == tmp_path / "results" / exp.OUTPUT_FILENAME
    assert config.start_time() > 0.0
