"""Tests for Exp 3033 FR-11 nonforgetting negative-control stress.

Spec refs: REQ-LEARN-3033, SCENARIO-LEARN-3033,
SCENARIO-LEARN-3033-BOUNDED, SCENARIO-LEARN-3033-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import beaver_style_validator_frontier_certificate_v1 as exp3018
from carnot.eval import dvi_verifier_feedback_self_learning_controller_v1 as exp3020
from carnot.eval import fr11_feasibility_channel_de_tautology_diagnostic_v1 as exp3019
from carnot.eval import fr11_heldout_dvi_replay_v2 as exp3032
from carnot.eval import fr11_nonforgetting_negative_control_stress_v1 as exp
from carnot.eval import nsvif_instruction_validator_tree_expansion_v1 as exp3017


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "experiment_3033_fr11_nonforgetting_negative_control_stress_v1.py"
)


def _exp3017_config(tmp_path: Path) -> exp3017.ExperimentConfig:
    return exp3017.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp3017.ARTIFACT_FILENAME,
        manifest_path=tmp_path / exp3017.VALIDATOR_MANIFEST_REL_PATH,
        z3_transcript_dir=tmp_path / exp3017.Z3_TRANSCRIPT_REL_DIR,
        runtime_transcript_dir=tmp_path / exp3017.RUNTIME_TRANSCRIPT_REL_DIR,
        started_at=10.0,
        clock=lambda: 12.0,
    )


def _exp3018_config(tmp_path: Path) -> exp3018.ExperimentConfig:
    return exp3018.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp3018.ARTIFACT_FILENAME,
        certificate_manifest_path=tmp_path / exp3018.CERTIFICATE_MANIFEST_REL_PATH,
        transcript_dir=tmp_path / exp3018.TRANSCRIPT_REL_DIR,
        source_artifact_path=tmp_path / "results" / exp3017.ARTIFACT_FILENAME,
        source_manifest_path=tmp_path / exp3017.VALIDATOR_MANIFEST_REL_PATH,
        started_at=20.0,
        clock=lambda: 23.0,
    )


def _write_exp3007_minimal(tmp_path: Path) -> None:
    artifact = {
        "artifact": "experiment_3007_fr11_attractor_trace_memory_stability_v1",
        "trace_memory_stability_ready": True,
        "continuous_self_learning_task": True,
        "independent_self_learning_boundary_preserved": True,
        "forgetting_guard_passed": True,
        "drift_guard_passed": True,
        "negative_control_rejected": True,
        "negative_control_report": {
            "accepted_control_ids": [],
            "control_heldout_deltas": {
                "contradicted_constraint": 0.0,
                "irrelevant_trace": 0.0,
                "shuffled_validator_label": 0.0,
            },
            "negative_control_rejected": True,
        },
        "accepted_memory_ids": ["trace-symbolization", "trace-schema"],
        "heldout_baseline_score": 0.5,
        "heldout_final_score": 1.0,
        "heldout_delta": 0.5,
        "native_attractor_model_claim_made": False,
        "self_reported_memory_utility_counted": False,
        "promotion_metric_names": ["exact_heldout_verifier_score"],
    }
    path = tmp_path / exp3019.EXP3007_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp3019_config(tmp_path: Path) -> exp3019.ExperimentConfig:
    return exp3019.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp3019.ARTIFACT_FILENAME,
        diagnostic_table_path=tmp_path / exp3019.DIAGNOSTIC_TABLE_REL_PATH,
        source_certificate_artifact_path=tmp_path / "results" / exp3018.ARTIFACT_FILENAME,
        source_certificate_manifest_path=tmp_path / exp3018.CERTIFICATE_MANIFEST_REL_PATH,
        source_validator_manifest_path=tmp_path / exp3017.VALIDATOR_MANIFEST_REL_PATH,
        exp3007_artifact_path=tmp_path / exp3019.EXP3007_REL_PATH,
        started_at=30.0,
        clock=lambda: 34.0,
    )


def _exp3020_config(tmp_path: Path) -> exp3020.ExperimentConfig:
    return exp3020.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp3020.ARTIFACT_FILENAME,
        controller_config_path=tmp_path / exp3020.CONTROLLER_CONFIG_REL_PATH,
        replay_transcript_path=tmp_path / exp3020.REPLAY_TRANSCRIPT_REL_PATH,
        exp3017_artifact_path=tmp_path / "results" / exp3017.ARTIFACT_FILENAME,
        exp3017_manifest_path=tmp_path / exp3017.VALIDATOR_MANIFEST_REL_PATH,
        exp3018_artifact_path=tmp_path / "results" / exp3018.ARTIFACT_FILENAME,
        exp3018_manifest_path=tmp_path / exp3018.CERTIFICATE_MANIFEST_REL_PATH,
        exp3019_artifact_path=tmp_path / "results" / exp3019.ARTIFACT_FILENAME,
        exp3019_table_path=tmp_path / exp3019.DIAGNOSTIC_TABLE_REL_PATH,
        exp3007_artifact_path=tmp_path / exp3020.EXP3007_REL_PATH,
        started_at=40.0,
        clock=lambda: 45.0,
    )


def _exp3032_config(tmp_path: Path) -> exp3032.ExperimentConfig:
    return exp3032.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp3032.ARTIFACT_FILENAME,
        heldout_replay_path=tmp_path / exp3032.HELDOUT_REPLAY_REL_PATH,
        exp3019_artifact_path=tmp_path / "results" / exp3019.ARTIFACT_FILENAME,
        exp3019_table_path=tmp_path / exp3019.DIAGNOSTIC_TABLE_REL_PATH,
        exp3020_artifact_path=tmp_path / "results" / exp3020.ARTIFACT_FILENAME,
        controller_config_path=tmp_path / exp3020.CONTROLLER_CONFIG_REL_PATH,
        replay_transcript_path=tmp_path / exp3020.REPLAY_TRANSCRIPT_REL_PATH,
        exp3017_manifest_path=tmp_path / exp3017.VALIDATOR_MANIFEST_REL_PATH,
        exp3018_manifest_path=tmp_path / exp3018.CERTIFICATE_MANIFEST_REL_PATH,
        started_at=50.0,
        clock=lambda: 56.0,
    )


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        stress_report_path=tmp_path / exp.STRESS_REPORT_REL_PATH,
        exp3032_artifact_path=tmp_path / "results" / exp3032.ARTIFACT_FILENAME,
        heldout_replay_path=tmp_path / exp3032.HELDOUT_REPLAY_REL_PATH,
        exp3020_artifact_path=tmp_path / "results" / exp3020.ARTIFACT_FILENAME,
        controller_config_path=tmp_path / exp3020.CONTROLLER_CONFIG_REL_PATH,
        replay_transcript_path=tmp_path / exp3020.REPLAY_TRANSCRIPT_REL_PATH,
        started_at=60.0,
        clock=lambda: 67.0,
        tests_run=("focused-req-3033",),
    )


def _write_sources(tmp_path: Path) -> None:
    exp3017.run_experiment(_exp3017_config(tmp_path))
    exp3018.run_experiment(_exp3018_config(tmp_path))
    _write_exp3007_minimal(tmp_path)
    exp3019.run_experiment(_exp3019_config(tmp_path))
    exp3020.run_experiment(_exp3020_config(tmp_path))
    exp3032.run_experiment(_exp3032_config(tmp_path))


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_req_learn_3033_spec_and_template_script_anchor_exists() -> None:
    """REQ-LEARN-3033: Exp 3033 is OpenSpec anchored and template-runnable."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3033" in spec
    assert "SCENARIO-LEARN-3033" in spec
    assert "SCENARIO-LEARN-3033-BOUNDED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "fr11_self_learning_promotable" in spec
    assert "prior_retention_delta" in spec
    assert SCRIPT_PATH.exists()
    assert "ExperimentTemplate" in script


def test_scenario_learn_3033_writes_promotable_stress_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3033: held-out feedback improves without prior forgetting."""

    _write_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))
    stress_rows = _jsonl_rows(tmp_path / exp.STRESS_REPORT_REL_PATH)

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["fr11_nonforgetting_stress_ready"] is True
    assert artifact["fr11_self_learning_promotable"] is True
    assert artifact["prior_retention_delta"] >= -artifact["retention_tolerance"]
    assert artifact["heldout_delta_after_update"] > 0.0
    assert artifact["shuffled_control_delta"] <= 0.0
    assert artifact["adversarial_irrelevant_control_delta"] <= 0.0
    assert artifact["no_feedback_delta"] == 0.0
    assert artifact["drift_failures"] == []
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["duration_s"] == pytest.approx(7.0)
    assert artifact["tests_run"] == ["focused-req-3033"]
    assert artifact["promotion_decision"] == "controller_only_promotable"

    substrate = artifact["inference_substrate"]
    assert substrate["live_llm_inference"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["controller_weight_update"] is True
    assert substrate["mode"] == "cached_feedback_controller_stress"

    assert artifact["kan_locality_probe_available"] is True
    assert artifact["kan_locality_report"]["updates_concentrate_on_local_features"] is True
    assert artifact["source_trace_counts"]["prior_exact_trace_count"] == 32
    assert artifact["source_trace_counts"]["heldout_trace_count"] == 8
    assert {row["section"] for row in stress_rows} == {
        "prior_retention",
        "heldout_update",
        "negative_controls",
    }

    exp.validate_artifact(artifact)


def test_req_learn_3033_trace_splits_controls_and_locality_probe(tmp_path: Path) -> None:
    """REQ-LEARN-3033-2/4/5: traces are split and controls reject leakage."""

    _write_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    prior = exp.build_prior_exact_traces(sources)
    heldout = exp.build_heldout_feedback_traces(sources)
    stress = exp.stress_controller(prior, heldout, sources.controller_config)
    locality = exp.kan_locality_probe()

    assert prior
    assert heldout
    assert {trace.expected_feedback for trace in prior} == {True, False}
    assert {trace.expected_feedback for trace in heldout} == {True, False}
    assert {trace.row_id for trace in prior}.isdisjoint({trace.row_id for trace in heldout})
    assert all(trace.exact_machine_checked for trace in prior)
    assert all(not (exp.PROHIBITED_FEATURE_NAMES & set(trace.features)) for trace in prior)
    assert all(not (exp.PROHIBITED_FEATURE_NAMES & set(trace.features)) for trace in heldout)

    assert stress["heldout_delta_after_update"] > 0.0
    assert stress["shuffled_control_delta"] <= 0.0
    assert stress["adversarial_irrelevant_control_delta"] <= 0.0
    assert stress["no_feedback_delta"] == 0.0
    assert stress["prior_retention_delta"] >= -exp.RETENTION_TOLERANCE
    assert stress["drift_failures"] == []

    assert locality["available"] is True
    assert locality["updates_concentrate_on_local_features"] is True
    assert locality["mean_active_fraction"] < 0.5


def test_scenario_learn_3033_bounded_when_feedback_forgets_prior_traces(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3033-BOUNDED: retention failures deny promotion."""

    _write_sources(tmp_path)
    cfg = _config(tmp_path)
    flipped_rows = []
    for row in _jsonl_rows(cfg.resolved_heldout_replay_path()):
        row["expected_label"] = not bool(row["expected_label"])
        flipped_rows.append(row)
    cfg.resolved_heldout_replay_path().write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in flipped_rows),
        encoding="utf-8",
    )

    artifact = exp.run_experiment(cfg)

    assert artifact["fr11_nonforgetting_stress_ready"] is True
    assert artifact["fr11_self_learning_promotable"] is False
    assert artifact["prior_retention_delta"] < -artifact["retention_tolerance"]
    assert artifact["drift_failures"]
    assert any(
        "prior_retention_below_tolerance" in failure for failure in artifact["drift_failures"]
    )
    assert artifact["honest_verdict"].startswith("complete_")
    exp.validate_artifact(artifact)


def test_scenario_learn_3033_blocked_artifact_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3033-BLOCKED: missing source evidence writes zeroed fields."""

    artifact = exp.run_experiment(_config(tmp_path))

    assert artifact["fr11_nonforgetting_stress_ready"] is False
    assert artifact["fr11_self_learning_promotable"] is False
    assert artifact["prior_retention_delta"] == 0.0
    assert artifact["heldout_delta_after_update"] == 0.0
    assert artifact["shuffled_control_delta"] == 0.0
    assert artifact["no_feedback_delta"] == 0.0
    assert artifact["drift_failures"] == ["exp3032_artifact_missing_or_empty"]
    assert artifact["kan_locality_probe_available"] is False
    assert artifact["honest_verdict"].startswith("blocked_")
    assert (tmp_path / "results" / exp.ARTIFACT_FILENAME).is_file()
    assert not (tmp_path / exp.STRESS_REPORT_REL_PATH).exists()
    exp.validate_artifact(artifact)


def test_req_learn_3033_validation_rejects_inconsistent_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-3033-5: validation enforces promotion and substrate gates."""

    _write_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete_incomplete"})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(artifact | {"inference_substrate": "cached"})
    with pytest.raises(ValueError, match="model_weight_training"):
        exp.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"model_weight_training": True}
            }
        )
    with pytest.raises(ValueError, match="heldout_delta_after_update"):
        exp.validate_artifact(artifact | {"heldout_delta_after_update": 0.0})
    with pytest.raises(ValueError, match="shuffled_control_delta"):
        exp.validate_artifact(artifact | {"shuffled_control_delta": 0.1})
    with pytest.raises(ValueError, match="no_feedback_delta"):
        exp.validate_artifact(artifact | {"no_feedback_delta": 0.1})
    with pytest.raises(ValueError, match="prior_retention_delta"):
        exp.validate_artifact(
            artifact
            | {
                "prior_retention_delta": -0.2,
                "drift_failures": [],
            }
        )
    with pytest.raises(ValueError, match="drift_failures"):
        exp.validate_artifact(artifact | {"drift_failures": ["unexpected"]})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready: wrong"})
    with pytest.raises(ValueError, match="blocked prefix"):
        exp.validate_artifact(
            artifact
            | {
                "fr11_nonforgetting_stress_ready": False,
                "fr11_self_learning_promotable": False,
                "honest_verdict": "complete_wrong_for_blocked",
            }
        )


def test_req_learn_3033_source_and_trace_blockers_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-3033-1/2: source and trace blockers fail closed by name."""

    ready_source_payload = {
        "exp3032_artifact": {"fr11_heldout_replay_ready": True},
        "heldout_replay_rows": ({"row": "present"},),
        "exp3020_artifact": {
            "verifier_feedback_controller_ready": True,
            "native_llm_training_claim_made": False,
        },
        "controller_config": {"final_weights": {"evidence::all_authoritative_checks_passed": 1.0}},
        "replay_transcript_rows": ({"row": "present"},),
    }
    ready_sources = exp.SourceBundle(**ready_source_payload)
    assert exp.precondition_blocker(ready_sources) is None

    cases = [
        (ready_source_payload | {"exp3032_artifact": {}}, "exp3032_artifact_missing_or_empty"),
        (
            ready_source_payload | {"exp3032_artifact": {"_malformed": True}},
            "exp3032_artifact_malformed",
        ),
        (
            ready_source_payload | {"exp3032_artifact": {"fr11_heldout_replay_ready": False}},
            "exp3032_heldout_replay_not_ready",
        ),
        (ready_source_payload | {"heldout_replay_rows": ()}, "exp3032_heldout_replay_missing"),
        (ready_source_payload | {"exp3020_artifact": {}}, "exp3020_artifact_missing_or_empty"),
        (
            ready_source_payload | {"exp3020_artifact": {"_malformed": True}},
            "exp3020_artifact_malformed",
        ),
        (
            ready_source_payload
            | {"exp3020_artifact": {"verifier_feedback_controller_ready": False}},
            "exp3020_controller_not_ready",
        ),
        (
            ready_source_payload
            | {
                "exp3020_artifact": {
                    "verifier_feedback_controller_ready": True,
                    "native_llm_training_claim_made": True,
                }
            },
            "exp3020_native_llm_training_claimed",
        ),
        (
            ready_source_payload | {"controller_config": {"final_weights": {}}},
            "exp3020_controller_weights_missing",
        ),
        (
            ready_source_payload | {"replay_transcript_rows": ()},
            "exp3020_replay_transcript_missing",
        ),
    ]
    for payload, expected in cases:
        assert exp.precondition_blocker(exp.SourceBundle(**payload)) == expected

    true_trace = exp.FeedbackTrace("t", "source", "i", "r", True, True, ("evidence::x",))
    false_trace = exp.FeedbackTrace("f", "source", "i", "r2", False, True, ("evidence::y",))
    assert (
        exp.trace_precondition_blocker([true_trace, false_trace], [true_trace, false_trace]) is None
    )
    assert (
        exp.trace_precondition_blocker([], [true_trace, false_trace])
        == "prior_exact_traces_missing"
    )
    assert (
        exp.trace_precondition_blocker([true_trace, false_trace], [])
        == "heldout_feedback_traces_missing"
    )
    assert (
        exp.trace_precondition_blocker([true_trace], [true_trace, false_trace])
        == "prior_exact_trace_labels_unbalanced"
    )
    assert (
        exp.trace_precondition_blocker([true_trace, false_trace], [true_trace])
        == "heldout_feedback_trace_labels_unbalanced"
    )

    cfg = _config(tmp_path)
    cfg.resolved_exp3032_artifact_path().parent.mkdir(parents=True, exist_ok=True)
    cfg.resolved_exp3032_artifact_path().write_text(
        json.dumps({"fr11_heldout_replay_ready": True}),
        encoding="utf-8",
    )
    cfg.resolved_heldout_replay_path().parent.mkdir(parents=True, exist_ok=True)
    cfg.resolved_heldout_replay_path().write_text(
        json.dumps(
            {
                "trace_id": "heldout::one",
                "row_id": "one",
                "item_id": "one",
                "exact_claim_id": "sha",
                "expected_authorities": ["runtime"],
                "expected_label": True,
                "checker_features": ["evidence::all_authoritative_checks_passed"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg.resolved_exp3020_artifact_path().write_text(
        json.dumps(
            {
                "verifier_feedback_controller_ready": True,
                "native_llm_training_claim_made": False,
                "controller_summary": {
                    "final_weights": {"evidence::all_authoritative_checks_passed": 1.0}
                },
            }
        ),
        encoding="utf-8",
    )
    cfg.resolved_controller_config_path().parent.mkdir(parents=True, exist_ok=True)
    cfg.resolved_controller_config_path().write_text(
        json.dumps({"final_weights": {"evidence::all_authoritative_checks_passed": 1.0}}),
        encoding="utf-8",
    )
    cfg.resolved_replay_transcript_path().write_text(
        json.dumps(
            {
                "partition": "forgetting_guard",
                "exact_machine_checked": True,
                "exact_feedback": True,
                "features": ["memory_guard::trace"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trace_blocked = exp.run_experiment(cfg)
    assert trace_blocked["blocked_reason"] == "prior_exact_traces_missing"


def test_req_learn_3033_defensive_helpers_and_control_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3033-3/4: helper branches keep defensive behavior covered."""

    sources = exp.SourceBundle(
        exp3032_artifact={"fr11_heldout_replay_ready": True},
        heldout_replay_rows=(
            {"checker_features": [], "exact_claim_id": "sha", "expected_authorities": ["a"]},
            {
                "trace_id": "h",
                "row_id": "h",
                "item_id": "h",
                "checker_features": ["evidence::all_authoritative_checks_passed"],
                "exact_claim_id": "sha",
                "expected_authorities": ["a"],
                "expected_label": True,
            },
        ),
        exp3020_artifact={
            "verifier_feedback_controller_ready": True,
            "native_llm_training_claim_made": False,
        },
        controller_config={"final_weights": {"evidence::x": 1.0}},
        replay_transcript_rows=(
            {"partition": "forgetting_guard", "features": ["memory_guard::x"]},
            {"partition": "train", "exact_machine_checked": False, "features": ["evidence::x"]},
            {"partition": "train", "exact_machine_checked": True, "features": []},
            {
                "partition": "train",
                "row_id": "p",
                "item_id": "p",
                "exact_machine_checked": True,
                "exact_feedback": True,
                "features": ["evidence::x"],
            },
        ),
    )
    assert [trace.row_id for trace in exp.build_prior_exact_traces(sources)] == ["p"]
    assert [trace.row_id for trace in exp.build_heldout_feedback_traces(sources)] == ["h"]

    exact = exp.FeedbackTrace("exact", "s", "i", "r", True, True, ("evidence::x",))
    skipped = exp.FeedbackTrace("skip", "s", "i", "r2", True, False, ("evidence::x",))
    outside = exp.FeedbackTrace("outside", "s", "i", "r3", True, True, ("control::x",))
    updated = exp.apply_feedback_updates({}, [exact, skipped, outside], {})
    assert updated == {"evidence::x": 0.25}
    assert exp.retention_score({}, []) == 0.0
    assert exp.mean_signed_margin({}, []) == 0.0
    assert exp.shuffled_feedback_traces([]) == []

    forgotten = exp.FeedbackTrace("forgotten", "s", "i", "forgotten", True, True, ("evidence::x",))
    prohibited = exp.FeedbackTrace("bad", "s", "i", "bad", True, True, ("row_id",))
    failures = exp.drift_failures_for(
        [forgotten, prohibited, outside],
        [outside],
        {},
        before_weights={"evidence::x": 1.0},
        after_weights={"evidence::x": -1.0},
        prior_retention_delta=-1.0,
        heldout_delta_after_update=0.0,
        shuffled_control_delta=0.1,
        adversarial_irrelevant_control_delta=0.2,
        no_feedback_delta=0.3,
        retention_tolerance=0.05,
    )
    assert "prior_trace_forgotten:forgotten" in failures
    assert any("heldout_delta_not_positive" in failure for failure in failures)
    assert any("shuffled_control_improved" in failure for failure in failures)
    assert any("adversarial_irrelevant_control_improved" in failure for failure in failures)
    assert any("no_feedback_replay_moved" in failure for failure in failures)
    assert any("prohibited_feature:bad:row_id" in failure for failure in failures)
    assert any("feature_outside_controller_boundary" in failure for failure in failures)

    monkeypatch.setattr(
        exp,
        "KAN_HELPERS_IMPORTER",
        lambda: (_ for _ in ()).throw(ImportError("not installed")),
    )
    assert exp.kan_locality_probe()["available"] is False
    monkeypatch.setattr(exp, "KAN_HELPERS_IMPORTER", lambda: object())
    assert exp.kan_locality_probe()["available"] is False

    malformed_json = tmp_path / "bad.json"
    malformed_json.write_text("{", encoding="utf-8")
    assert exp._read_json(malformed_json) == {"_malformed": True}
    malformed_jsonl = tmp_path / "bad.jsonl"
    malformed_jsonl.write_text("{\n", encoding="utf-8")
    assert exp._read_jsonl(malformed_jsonl) == [{"_malformed": True}]
    nonmapping_jsonl = tmp_path / "nonmapping.jsonl"
    nonmapping_jsonl.write_text("[]\n", encoding="utf-8")
    assert exp._read_jsonl(nonmapping_jsonl) == [{"_malformed": True}]
    assert exp._relative_to(tmp_path, Path("/definitely/outside/carnot")) == Path(
        "/definitely/outside/carnot"
    )


def test_req_learn_3033_validation_covers_remaining_gate_errors(tmp_path: Path) -> None:
    """REQ-LEARN-3033-5: validation reports each remaining gate by field."""

    _write_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))

    with pytest.raises(ValueError, match="live_llm_inference"):
        exp.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"live_llm_inference": True}
            }
        )
    with pytest.raises(ValueError, match="blocked artifacts cannot be promotable"):
        exp.validate_artifact(
            artifact
            | {
                "fr11_nonforgetting_stress_ready": False,
                "fr11_self_learning_promotable": True,
                "honest_verdict": "blocked_wrong",
            }
        )
    with pytest.raises(ValueError, match="adversarial_irrelevant_control_delta"):
        exp.validate_artifact(artifact | {"adversarial_irrelevant_control_delta": 0.1})
