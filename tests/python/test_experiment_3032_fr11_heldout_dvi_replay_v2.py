"""Tests for Exp 3032 FR-11 held-out DVI replay.

Spec refs: REQ-LEARN-3032, SCENARIO-LEARN-3032,
SCENARIO-LEARN-3032-BOUNDED, SCENARIO-LEARN-3032-BLOCKED.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from carnot.eval import beaver_style_validator_frontier_certificate_v1 as exp3018
from carnot.eval import dvi_verifier_feedback_self_learning_controller_v1 as exp3020
from carnot.eval import fr11_feasibility_channel_de_tautology_diagnostic_v1 as exp3019
from carnot.eval import fr11_heldout_dvi_replay_v2 as exp
from carnot.eval import nsvif_instruction_validator_tree_expansion_v1 as exp3017


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_3032_fr11_heldout_dvi_replay_v2.py"


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


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        heldout_replay_path=tmp_path / exp.HELDOUT_REPLAY_REL_PATH,
        exp3019_artifact_path=tmp_path / "results" / exp3019.ARTIFACT_FILENAME,
        exp3019_table_path=tmp_path / exp3019.DIAGNOSTIC_TABLE_REL_PATH,
        exp3020_artifact_path=tmp_path / "results" / exp3020.ARTIFACT_FILENAME,
        controller_config_path=tmp_path / exp3020.CONTROLLER_CONFIG_REL_PATH,
        replay_transcript_path=tmp_path / exp3020.REPLAY_TRANSCRIPT_REL_PATH,
        exp3017_manifest_path=tmp_path / exp3017.VALIDATOR_MANIFEST_REL_PATH,
        exp3018_manifest_path=tmp_path / exp3018.CERTIFICATE_MANIFEST_REL_PATH,
        started_at=50.0,
        clock=lambda: 56.0,
        tests_run=("focused-req-3032",),
    )


def _write_sources(tmp_path: Path) -> None:
    exp3017.run_experiment(_exp3017_config(tmp_path))
    exp3018.run_experiment(_exp3018_config(tmp_path))
    _write_exp3007_minimal(tmp_path)
    exp3019.run_experiment(_exp3019_config(tmp_path))
    exp3020.run_experiment(_exp3020_config(tmp_path))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_req_learn_3032_spec_and_template_script_anchor_exists() -> None:
    """REQ-LEARN-3032: Exp 3032 is OpenSpec anchored and template-runnable."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3032" in spec
    assert "SCENARIO-LEARN-3032" in spec
    assert "SCENARIO-LEARN-3032-BOUNDED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "fr11_heldout_replay_ready" in spec
    assert "shuffled_feedback_delta" in spec
    assert SCRIPT_PATH.exists()
    assert "ExperimentTemplate" in script


def test_scenario_learn_3032_writes_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3032: held-out exact replay clears promotion gates."""

    _write_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))
    replay_rows = _load_jsonl(tmp_path / exp.HELDOUT_REPLAY_REL_PATH)

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["fr11_heldout_replay_ready"] is True
    assert artifact["continuous_self_learning_tested"] is True
    assert artifact["heldout_trace_count"] == 8
    assert artifact["feasible_infeasible_auc_delta"] > 0.0
    assert artifact["shuffled_feedback_delta"] <= 0.0
    assert artifact["false_positive_delta"] <= 0.0
    assert artifact["false_negative_delta"] <= 0.0
    assert artifact["tautology_risk_cleared"] is True
    assert artifact["information_asymmetry_enforced"] is True
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["duration_s"] == pytest.approx(6.0)
    assert artifact["tests_run"] == ["focused-req-3032"]

    substrate = artifact["inference_substrate"]
    assert substrate["live_llm_inference"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["mode"] == "cached_exact_trace_replay"

    assert replay_rows
    for row in replay_rows:
        assert row["row_id"] not in artifact["controller_update_row_ids"]
        assert not (exp.PROHIBITED_CHECKER_FIELDS & set(row))
        assert not (exp.PROHIBITED_CHECKER_FIELDS & set(row["checker_features"]))

    exp.validate_artifact(artifact)


def test_req_learn_3032_role_split_and_negative_controls(tmp_path: Path) -> None:
    """REQ-LEARN-3032-2/3/4: role split and controls are non-tautological."""

    _write_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    traces = exp.build_heldout_traces(sources)
    report = exp.evaluate_heldout_replay(traces, sources.controller_config)

    assert traces
    assert {trace.expected_label for trace in traces} == {True, False}
    assert all(trace.row_id not in sources.controller_update_row_ids for trace in traces)
    assert all(trace.expected_authorities for trace in traces)
    assert all(
        "semantic_boundary_non_authoritative" not in trace.expected_authorities
        for trace in traces
    )
    assert all(not (exp.PROHIBITED_CHECKER_FIELDS & set(trace.checker_features)) for trace in traces)
    assert exp.information_asymmetry_enforced(traces, sources) is True

    assert report["baseline_auc"] == pytest.approx(0.5)
    assert report["controller_auc"] > report["baseline_auc"]
    assert report["shuffled_feedback_delta"] <= 0.0
    assert report["withheld_authority_delta"] <= 0.0
    assert report["tautology_exposure"]["training_row_overlap_count"] == 0
    assert report["tautology_exposure"]["prohibited_checker_field_count"] == 0
    assert report["tautology_exposure"]["source_tautology_flag_observed"] is True
    assert report["tautology_exposure"]["risk_cleared"] is True


def test_scenario_learn_3032_bounded_completion_when_tautology_exposed(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3032-BOUNDED: complete artifact can deny promotion."""

    _write_sources(tmp_path)
    cfg = _config(tmp_path)
    controller = json.loads(cfg.resolved_controller_config_path().read_text(encoding="utf-8"))
    controller["final_weights"] = {"heldout_success_label": 1.0}
    cfg.resolved_controller_config_path().write_text(
        json.dumps(controller, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact = exp.run_experiment(cfg)

    assert artifact["fr11_heldout_replay_ready"] is False
    assert artifact["tautology_risk_cleared"] is False
    assert artifact["tautology_exposure"]["prohibited_controller_weight_count"] == 1
    assert artifact["honest_verdict"].startswith("complete_")
    exp.validate_artifact(artifact)


def test_scenario_learn_3032_blocked_artifact_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3032-BLOCKED: missing evidence writes zeroed gates."""

    artifact = exp.run_experiment(_config(tmp_path))

    assert artifact["fr11_heldout_replay_ready"] is False
    assert artifact["continuous_self_learning_tested"] is True
    assert artifact["heldout_trace_count"] == 0
    assert artifact["feasible_infeasible_auc_delta"] == 0.0
    assert artifact["shuffled_feedback_delta"] == 0.0
    assert artifact["false_positive_delta"] == 0.0
    assert artifact["false_negative_delta"] == 0.0
    assert artifact["tautology_risk_cleared"] is False
    assert artifact["information_asymmetry_enforced"] is False
    assert artifact["honest_verdict"].startswith("blocked_")
    assert (tmp_path / "results" / exp.ARTIFACT_FILENAME).is_file()
    assert not (tmp_path / exp.HELDOUT_REPLAY_REL_PATH).exists()
    exp.validate_artifact(artifact)


def test_req_learn_3032_validation_rejects_inconsistent_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-3032-5: terminal validation enforces promotion gates."""

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
                "inference_substrate": {
                    "mode": "cached_exact_trace_replay",
                    "live_llm_inference": False,
                    "model_weight_training": True,
                }
            }
        )
    with pytest.raises(ValueError, match="tautology_risk_cleared"):
        exp.validate_artifact(artifact | {"tautology_risk_cleared": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready: wrong"})
    with pytest.raises(ValueError, match="blocked prefix"):
        exp.validate_artifact(
            artifact
            | {
                "fr11_heldout_replay_ready": False,
                "honest_verdict": "wrong_prefix",
            }
        )

    ready_cases = [
        ("continuous_self_learning_tested", False, "continuous_self_learning_tested"),
        ("heldout_trace_count", 0, "heldout_trace_count"),
        ("feasible_infeasible_auc_delta", 0.0, "feasible_infeasible_auc_delta"),
        ("shuffled_feedback_delta", 0.1, "shuffled_feedback_delta"),
        ("false_positive_delta", 0.1, "false_positive_delta"),
        ("false_negative_delta", 0.1, "false_negative_delta"),
        ("information_asymmetry_enforced", False, "information_asymmetry_enforced"),
    ]
    for field, value, message in ready_cases:
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(artifact | {field: value})

    with pytest.raises(ValueError, match="live_llm_inference"):
        exp.validate_artifact(
            artifact
            | {
                "inference_substrate": {
                    "mode": "cached_exact_trace_replay",
                    "live_llm_inference": True,
                    "model_weight_training": False,
                }
            }
        )

    blocked_bad = exp.run_experiment(_config(tmp_path / "blocked"))
    with pytest.raises(ValueError, match="blocked artifacts"):
        exp.validate_artifact(blocked_bad | {"heldout_trace_count": 1})


def test_req_learn_3032_precondition_blockers_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-3032-1: malformed or missing source evidence fails closed."""

    _write_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))

    cases = [
        (replace(sources, exp3019_artifact={"_malformed": True}), "exp3019_artifact_malformed"),
        (
            replace(sources, exp3019_artifact={"feasibility_channel_diagnostic_ready": False}),
            "exp3019_not_terminal_ready",
        ),
        (replace(sources, exp3019_rows=()), "exp3019_diagnostic_table_missing"),
        (replace(sources, exp3020_artifact={}), "exp3020_artifact_missing_or_empty"),
        (replace(sources, exp3020_artifact={"_malformed": True}), "exp3020_artifact_malformed"),
        (
            replace(sources, exp3020_artifact={"verifier_feedback_controller_ready": False}),
            "exp3020_controller_not_ready",
        ),
        (
            replace(sources, controller_config={"final_weights": {}}),
            "exp3020_controller_weights_missing",
        ),
        (replace(sources, replay_transcript_rows=()), "exp3020_replay_transcript_missing"),
        (replace(sources, exp3017_rows=()), "exp3017_validator_manifest_missing"),
        (replace(sources, exp3018_rows=()), "exp3018_certificate_manifest_missing"),
    ]
    for mutated, expected in cases:
        assert exp.precondition_blocker(mutated) == expected


def test_req_learn_3032_skip_paths_and_asymmetry_failures(tmp_path: Path) -> None:
    """REQ-LEARN-3032-2/3: held-out construction rejects invalid checker inputs."""

    _write_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    update_row_id = next(iter(sources.controller_update_row_ids))
    extra_rows = (
        {
            "heldout_partition": True,
            "row_type": "non_prefix_closed_node",
            "row_id": "skip-non-candidate",
            "feasibility_class": "feasible",
        },
        {
            "heldout_partition": True,
            "row_type": "candidate_frontier",
            "row_id": update_row_id,
            "feasibility_class": "feasible",
        },
        {
            "heldout_partition": True,
            "row_type": "candidate_frontier",
            "row_id": "skip-unknown-class",
            "feasibility_class": "unresolved",
        },
    )
    mutated = replace(sources, exp3019_rows=sources.exp3019_rows + extra_rows)
    traces = exp.build_heldout_traces(mutated)

    assert "skip-non-candidate" not in {trace.row_id for trace in traces}
    assert "skip-unknown-class" not in {trace.row_id for trace in traces}
    assert exp.information_asymmetry_enforced([], sources) is False

    trace = traces[0]
    assert (
        exp.information_asymmetry_enforced(
            [replace(trace, row_id=update_row_id)],
            sources,
        )
        is False
    )
    assert (
        exp.information_asymmetry_enforced(
            [replace(trace, expected_authorities=())],
            sources,
        )
        is False
    )
    assert (
        exp.information_asymmetry_enforced(
            [replace(trace, checker_features=("candidate_role",))],
            sources,
        )
        is False
    )


def test_req_learn_3032_metric_and_io_defense_helpers(tmp_path: Path) -> None:
    """REQ-LEARN-3032-4: metric and parser edge cases are deterministic."""

    assert exp.mann_whitney_auc([], [1.0]) == 0.0
    assert exp._label_from_feasibility_class("unresolved") is None
    assert exp._shuffled_feedback_weights({}) == {}

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert exp._read_json(bad_json) == {"_malformed": True}

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("{", encoding="utf-8")
    assert exp._read_jsonl(bad_jsonl) == [{"_malformed": True}]

    non_mapping_jsonl = tmp_path / "non-mapping.jsonl"
    non_mapping_jsonl.write_text("[]\n", encoding="utf-8")
    assert exp._read_jsonl(non_mapping_jsonl) == [{"_malformed": True}]

    assert exp._relative_to(tmp_path, Path("/definitely/not/under/tmp")) == Path(
        "/definitely/not/under/tmp"
    )
