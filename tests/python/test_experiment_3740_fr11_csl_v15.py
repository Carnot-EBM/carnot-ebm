"""Tests for Exp 3740 FR-11 continuous self-learning v15.

Spec: REQ-LEARN-3740, SCENARIO-LEARN-3740.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.fr11 import continuous_self_learning_v15 as exp3740
from carnot.fr11.continuous_self_learning_v15 import (
    EMPTY_VERDICT,
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    SUCCESS_VERDICT_PREFIX,
    StabilizerEfficacyTracker,
    TrainingChunk,
    build_artifact,
    build_tracker_from_chunks,
    efficacy_table,
    load_training_chunks,
    recommended_recipe,
    validate_artifact,
    write_artifact,
)


SPEC_PATH = Path("openspec/capabilities/self-learning/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _fixture_root(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    _write_json(
        results / "experiment_3734_fix_harness_and_bounded_train_chunk1.json",
        {
            "stabilizers_applied": (
                "replay_buffer, langevin_noise, random_alpha, "
                "random_descent_steps, grad_clip, kl_cd_fix"
            ),
            "nan_or_divergence_events": False,
            "ebt_loss_curve": [0.99, 1.14],
            "cumulative_steps_trained": 2,
        },
    )
    _write_json(
        results / "experiment_3735_bounded_train_chunk2_resume.json",
        {
            "stabilizers_applied": "none",
            "nan_or_divergence_events": False,
            "ebt_converged": False,
            "ebt_loss_curve": [],
            "cumulative_steps_trained": 0,
        },
    )
    _write_json(
        results / "experiment_3728_bounded_checkpointed_train_ebt_and_ar.json",
        {
            "stabilizers_applied": "none",
            "nan_or_divergence_events": False,
            "ebt_converged": False,
            "ebt_loss_curve": [],
            "cumulative_steps_trained": 0,
        },
    )
    _write_json(
        results / "experiment_3737_ebt_generation_smoke.json",
        {"stabilizers_applied": "grad_clip", "nan_or_divergence_events": True},
    )
    return tmp_path


def test_req_learn_3740_spec_declares_contract() -> None:
    """REQ-LEARN-3740: OpenSpec declares the v15 counter-tracker contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3740" in spec
    assert "SCENARIO-LEARN-3740" in spec
    assert "stabilizer_efficacy_table" in spec
    assert INFERENCE_SUBSTRATE in spec


def test_scenario_learn_3740_counter_updates_and_recipe_are_observation_driven() -> None:
    """SCENARIO-LEARN-3740: recipe follows real enabled/disabled counters."""

    chunks = [
        TrainingChunk(
            source_path="a.json",
            experiment_id=1,
            stabilizers=("grad_clip", "kl_cd_fix"),
            nan_or_divergence_events=False,
            ebt_converged=True,
            cumulative_steps_trained=2,
        ),
        TrainingChunk(
            source_path="b.json",
            experiment_id=2,
            stabilizers=(),
            nan_or_divergence_events=True,
            ebt_converged=False,
            cumulative_steps_trained=2,
        ),
        TrainingChunk(
            source_path="c.json",
            experiment_id=3,
            stabilizers=("grad_clip",),
            nan_or_divergence_events=False,
            ebt_converged=True,
            cumulative_steps_trained=2,
        ),
    ]

    tracker, observed = build_tracker_from_chunks(chunks)
    table_rows = efficacy_table(tracker, observed)
    table = {row["stabilizer"]: row for row in table_rows}
    recipe = recommended_recipe(table_rows, n_chunks_observed=len(chunks))

    assert observed == ("grad_clip", "kl_cd_fix")
    assert table["grad_clip"]["enabled_total"] == 2
    assert table["grad_clip"]["enabled_no_divergence"] == 2
    assert table["grad_clip"]["disabled_total"] == 1
    assert table["grad_clip"]["disabled_no_divergence"] == 0
    assert table["grad_clip"]["no_divergence_rate_delta_enabled_minus_disabled"] == 1.0
    assert table["kl_cd_fix"]["enabled_total"] == 1
    assert table["kl_cd_fix"]["disabled_total"] == 2
    assert table["kl_cd_fix"]["disabled_no_divergence"] == 1
    assert table["kl_cd_fix"]["no_divergence_rate_delta_enabled_minus_disabled"] == 0.5
    assert recipe["stabilizers"] == ["grad_clip"]
    assert recipe["is_preliminary_heuristic"] is True


def test_req_learn_3740_discovers_real_shaped_chunks_and_persists_state(tmp_path: Path) -> None:
    """REQ-LEARN-3740-1/3/4/5: artifact writes counters, recipe, and state."""

    root = _fixture_root(tmp_path)
    chunks = load_training_chunks(root / "results")

    assert [chunk.experiment_id for chunk in chunks] == [3728, 3734, 3735]
    assert chunks[1].stabilizers == (
        "grad_clip",
        "kl_cd_fix",
        "langevin_noise",
        "random_alpha",
        "random_descent_steps",
        "replay_buffer",
    )

    output = write_artifact(
        root,
        output_path=Path("results/out.json"),
        state_path=Path("results/state.json"),
        started_s=10.0,
        now_s=10.5,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    state = json.loads((root / "results/state.json").read_text(encoding="utf-8"))

    validate_artifact(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith(SUCCESS_VERDICT_PREFIX)
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["n_chunks_observed"] == 3
    assert artifact["tracker_state_persisted"] is True
    assert artifact["recommended_recipe"]["stabilizers"] == [
        "grad_clip",
        "kl_cd_fix",
        "langevin_noise",
        "random_alpha",
        "random_descent_steps",
        "replay_buffer",
    ]
    assert state["n_chunks_observed"] == 3
    assert state["stats"]["grad_clip"]["enabled_total"] == 1
    assert state["stats"]["grad_clip"]["disabled_total"] == 2
    assert "GGUF" not in json.dumps(artifact)
    assert "CUDA" not in json.dumps(artifact)
    assert "cuda" not in json.dumps(artifact)


def test_req_learn_3740_empty_fallback_persists_empty_tracker(tmp_path: Path) -> None:
    """REQ-LEARN-3740-2: absent diagnostics initialize an empty tracker."""

    output = write_artifact(
        tmp_path,
        output_path=Path("results/empty.json"),
        state_path=Path("results/empty_state.json"),
        started_s=1.0,
        now_s=1.0,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "results/empty_state.json").read_text(encoding="utf-8"))

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == EMPTY_VERDICT
    assert artifact["n_chunks_observed"] == 0
    assert artifact["stabilizer_efficacy_table"] == []
    assert artifact["recommended_recipe"]["stabilizers"] == []
    assert artifact["tracker_state_persisted"] is True
    assert state["stats"] == {}


def test_req_learn_3740_validation_and_parser_edges(tmp_path: Path) -> None:
    """REQ-LEARN-3740-3/6: parsing and schema guards reject poisoned output."""

    assert exp3740.parse_stabilizers("none") == ()
    assert exp3740.parse_stabilizers([" grad_clip ", "none", "grad_clip"]) == (
        "grad_clip",
    )
    assert exp3740.parse_stabilizers(None) == ()

    tracker = StabilizerEfficacyTracker()
    tracker.record_chunk(("grad_clip",), no_divergence=True, observed_stabilizers=("grad_clip",))
    tracker.record_chunk((), no_divergence=False, observed_stabilizers=("grad_clip",))
    payload = tracker.to_json()
    restored = StabilizerEfficacyTracker.from_json(payload)
    assert restored.to_json() == payload

    artifact = build_artifact(tmp_path / "empty-root", started_s=0.0, now_s=0.1)
    validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required artifact fields"):
        bad = dict(artifact)
        bad.pop("honest_verdict")
        validate_artifact(bad)
    with pytest.raises(ValueError, match="inference_substrate"):
        validate_artifact(dict(artifact, inference_substrate="live_llm_inference"))
    with pytest.raises(ValueError, match="forbidden inference marker"):
        validate_artifact(dict(artifact, source_training_chunks=["CUDA"]))


def test_req_learn_3740_guard_edges_for_state_loading_and_schema(tmp_path: Path) -> None:
    """REQ-LEARN-3740-2/3/6: guard branches stay explicit and deterministic."""

    with pytest.raises(ValueError, match="unsupported stabilizer tracker state version"):
        StabilizerEfficacyTracker.from_json({"version": 2, "stats": {}})
    with pytest.raises(ValueError, match="stats must be a mapping"):
        StabilizerEfficacyTracker.from_json({"version": 1, "stats": []})
    with pytest.raises(ValueError, match="stat entry must be a mapping"):
        StabilizerEfficacyTracker.from_json({"version": 1, "stats": {"grad_clip": []}})
    assert exp3740.parse_stabilizers(7) == ()
    assert exp3740.StabilizerStats().enabled_rate == 0.0

    results = tmp_path / "results"
    (results / "experiment_3699_malformed.json").parent.mkdir(parents=True, exist_ok=True)
    (results / "experiment_3699_malformed.json").write_text("{}\n{}\n", encoding="utf-8")
    _write_json(results / "experiment_3700_missing_tuple.json", {"model_specs": {"ebt_model": "x"}})
    _write_json(
        results / "experiment_3701_custom_name.json",
        {
            "model_specs": {"ebt_model": "tiny"},
            "stabilizers_applied": ["grad_clip"],
            "nan_or_divergence_events": "false",
            "ebt_converged": "true",
        },
    )
    loaded = load_training_chunks(results)
    assert [chunk.experiment_id for chunk in loaded] == [3701]
    assert loaded[0].cumulative_steps_trained is None
    assert loaded[0].ebt_converged is True

    bad_results = tmp_path / "bad-results"
    _write_json(
        bad_results / "experiment_3702_custom_name.json",
        {
            "model_specs": {"ebt_model": "tiny"},
            "stabilizers_applied": "grad_clip",
            "nan_or_divergence_events": "maybe",
        },
    )
    with pytest.raises(ValueError, match="expected boolean diagnostic value"):
        load_training_chunks(bad_results)

    artifact = build_artifact(tmp_path / "empty-root", started_s=0.0, now_s=0.1)
    with pytest.raises(ValueError, match="field_principles"):
        validate_artifact(dict(artifact, field_principles=None))
    with pytest.raises(ValueError, match="missing field principles"):
        validate_artifact(dict(artifact, field_principles={}))
    with pytest.raises(ValueError, match="tier1_counter_update"):
        validate_artifact(dict(artifact, tier1_counter_update="model_retrain"))
    with pytest.raises(ValueError, match="tracker_state_persisted"):
        validate_artifact(dict(artifact, tracker_state_persisted="true"))
    with pytest.raises(ValueError, match="is_preliminary_heuristic"):
        validate_artifact(dict(artifact, is_preliminary_heuristic=False))
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(dict(artifact, duration_s="fast"))
    with pytest.raises(ValueError, match="stabilizer_efficacy_table"):
        validate_artifact(dict(artifact, stabilizer_efficacy_table={}))
    with pytest.raises(ValueError, match="recommended_recipe"):
        validate_artifact(dict(artifact, recommended_recipe={}))
    with pytest.raises(ValueError, match="empty tracker verdict"):
        validate_artifact(dict(artifact, honest_verdict="complete: wrong_empty"))
    poisoned_empty = dict(
        artifact,
        recommended_recipe=dict(artifact["recommended_recipe"], stabilizers=["grad_clip"]),
    )
    with pytest.raises(ValueError, match="must not recommend"):
        validate_artifact(poisoned_empty)

    non_empty = build_artifact(_fixture_root(tmp_path / "non-empty"), started_s=0.0, now_s=0.1)
    with pytest.raises(ValueError, match="v15 success prefix"):
        validate_artifact(dict(non_empty, honest_verdict="complete: wrong_non_empty"))
    assert exp3740._relative_path(Path("/outside-carnot-v15-state.json"), tmp_path) == (  # noqa: SLF001
        "/outside-carnot-v15-state.json"
    )
