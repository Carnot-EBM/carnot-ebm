"""Tests for Exp 3020 DVI verifier-feedback self-learning controller.

Spec refs: REQ-LEARN-3020, SCENARIO-LEARN-3020,
SCENARIO-LEARN-3020-BLOCKED.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from carnot.eval import beaver_style_validator_frontier_certificate_v1 as exp3018
from carnot.eval import dvi_verifier_feedback_self_learning_controller_v1 as exp
from carnot.eval import fr11_feasibility_channel_de_tautology_diagnostic_v1 as exp3019
from carnot.eval import nsvif_instruction_validator_tree_expansion_v1 as exp3017


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "experiment_3020_dvi_verifier_feedback_self_learning_controller_v1.py"
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


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        controller_config_path=tmp_path / exp.CONTROLLER_CONFIG_REL_PATH,
        replay_transcript_path=tmp_path / exp.REPLAY_TRANSCRIPT_REL_PATH,
        exp3017_artifact_path=tmp_path / "results" / exp3017.ARTIFACT_FILENAME,
        exp3017_manifest_path=tmp_path / exp3017.VALIDATOR_MANIFEST_REL_PATH,
        exp3018_artifact_path=tmp_path / "results" / exp3018.ARTIFACT_FILENAME,
        exp3018_manifest_path=tmp_path / exp3018.CERTIFICATE_MANIFEST_REL_PATH,
        exp3019_artifact_path=tmp_path / "results" / exp3019.ARTIFACT_FILENAME,
        exp3019_table_path=tmp_path / exp3019.DIAGNOSTIC_TABLE_REL_PATH,
        exp3007_artifact_path=tmp_path / exp.EXP3007_REL_PATH,
        started_at=40.0,
        clock=lambda: 45.0,
        tests_run=("focused-req-3020",),
    )


def _write_exp3007_minimal(tmp_path: Path) -> dict[str, object]:
    artifact: dict[str, object] = {
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
    path = tmp_path / exp.EXP3007_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _write_sources(tmp_path: Path) -> None:
    exp3017.run_experiment(_exp3017_config(tmp_path))
    exp3018.run_experiment(_exp3018_config(tmp_path))
    _write_exp3007_minimal(tmp_path)
    exp3019.run_experiment(_exp3019_config(tmp_path))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_req_learn_3020_spec_and_template_script_anchor_exists() -> None:
    """REQ-LEARN-3020: Exp 3020 is OpenSpec anchored and template-runnable."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3020" in spec
    assert "SCENARIO-LEARN-3020" in spec
    assert "SCENARIO-LEARN-3020-BLOCKED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "verifier_feedback_controller_ready" in spec
    assert "native_llm_training_claim_made=false" in spec
    assert SCRIPT_PATH.exists()
    assert "ExperimentTemplate" in script


def test_scenario_learn_3020_writes_config_transcript_and_ready_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3020: cached verifier feedback improves held-out utility."""

    _write_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))
    config = json.loads((tmp_path / exp.CONTROLLER_CONFIG_REL_PATH).read_text(encoding="utf-8"))
    transcript = _load_jsonl(tmp_path / exp.REPLAY_TRANSCRIPT_REL_PATH)

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["verifier_feedback_controller_ready"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["independent_self_learning_boundary_preserved"] is True
    assert artifact["controller_config_path"] == str(exp.CONTROLLER_CONFIG_REL_PATH)
    assert artifact["replay_transcript_path"] == str(exp.REPLAY_TRANSCRIPT_REL_PATH)
    assert artifact["n_replay_items"] >= 40
    assert artifact["heldout_delta"] > 0.0
    assert artifact["negative_control_delta"] <= 0.0
    assert artifact["forgetting_guard_passed"] is True
    assert artifact["drift_guard_passed"] is True
    assert artifact["tautology_risk_flag"] is False
    assert artifact["native_llm_training_claim_made"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == pytest.approx(5.0)
    assert artifact["tests_run"] == ["focused-req-3020"]

    assert config["controller_type"] == "bounded_verifier_feedback_router"
    assert config["native_llm_training_claim_made"] is False
    assert set(config["prohibited_feature_names"]) == exp.PROHIBITED_FEATURE_NAMES
    assert set(config["update_metric_names"]).isdisjoint(config["independent_metric_names"])

    assert transcript
    accepted = [row for row in transcript if row["update_accepted"]]
    assert accepted
    for row in accepted:
        assert row["exact_machine_checked"] is True
        assert row["after_heldout_score"] > row["before_heldout_score"]
        assert row["after_forgetting_score"] >= row["before_forgetting_score"]
        assert not (exp.PROHIBITED_FEATURE_NAMES & set(row["features"]))

    baselines = artifact["control_comparison"]
    assert baselines["no_learning_delta"] == 0.0
    assert baselines["replay_only_delta"] == 0.0
    assert baselines["random_update_delta"] <= 0.0
    assert artifact["heldout_delta"] > baselines["random_update_delta"]

    exp.validate_artifact(artifact)


def test_req_learn_3020_replay_items_are_exact_and_boundary_safe(tmp_path: Path) -> None:
    """REQ-LEARN-3020-2/4: replay features exclude labels and accepted updates gate."""

    _write_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    items = exp.build_replay_set(sources)
    rows_with_unknown_status = sources.exp3018_rows + (
        {"row_type": "candidate_frontier", "certificate_status": "unresolved"},
    )
    assert exp.build_replay_set(replace(sources, exp3018_rows=rows_with_unknown_status)) == items
    report = exp.train_controller(items, exp.default_controller_config())

    assert {item.source_experiment for item in items} == {"exp3018", "exp3007"}
    assert {"train", "heldout", "forgetting_guard"} <= {item.partition for item in items}
    assert all(item.machine_checked for item in items if item.partition != "negative_control")
    assert all(not (exp.PROHIBITED_FEATURE_NAMES & set(item.features)) for item in items)
    assert report["final_score"] > report["baseline_score"]
    assert report["heldout_delta"] > 0.0
    assert report["forgetting_guard_passed"] is True
    assert report["drift_guard_passed"] is True
    assert all(
        row["after_heldout_score"] > row["before_heldout_score"]
        for row in report["transcript_rows"]
        if row["update_accepted"]
    )
    assert exp.evaluate_utility({}, []) == 0.0

    no_allowed = exp.ReplayItem(
        "bad-empty-config",
        "control",
        "bad",
        "bad",
        "negative_control",
        True,
        True,
        ("evidence::all_authoritative_checks_passed",),
    )
    bad_feature = replace(no_allowed, replay_id="bad-prohibited", features=("row_id",))
    bad_prefix = replace(no_allowed, replay_id="bad-prefix", features=("control::off_domain",))
    assert exp.item_drift_guard(no_allowed, {"allowed_feature_prefixes": []}) is False
    assert exp.item_drift_guard(bad_feature, exp.default_controller_config()) is False
    assert exp.item_drift_guard(bad_prefix, exp.default_controller_config()) is False


def test_scenario_learn_3020_blocked_artifact_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3020-BLOCKED: missing source evidence writes zeroed gates."""

    artifact = exp.run_experiment(_config(tmp_path))

    assert artifact["verifier_feedback_controller_ready"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["independent_self_learning_boundary_preserved"] is False
    assert artifact["controller_config_path"] == str(exp.CONTROLLER_CONFIG_REL_PATH)
    assert artifact["n_replay_items"] == 0
    assert artifact["heldout_delta"] == 0.0
    assert artifact["negative_control_delta"] == 0.0
    assert artifact["forgetting_guard_passed"] is False
    assert artifact["drift_guard_passed"] is False
    assert artifact["tautology_risk_flag"] is False
    assert artifact["native_llm_training_claim_made"] is False
    assert artifact["honest_verdict"].startswith("blocked_")
    assert (tmp_path / "results" / exp.ARTIFACT_FILENAME).is_file()
    assert not (tmp_path / exp.CONTROLLER_CONFIG_REL_PATH).exists()
    assert not (tmp_path / exp.REPLAY_TRANSCRIPT_REL_PATH).exists()

    exp.validate_artifact(artifact)


def test_req_learn_3020_validation_and_source_blockers_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-3020-1/5: validation rejects unsafe terminal artifacts."""

    _write_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="native_llm_training_claim_made"):
        exp.validate_artifact(artifact | {"native_llm_training_claim_made": True})
    with pytest.raises(ValueError, match="tautology_risk_flag"):
        exp.validate_artifact(artifact | {"tautology_risk_flag": True})
    with pytest.raises(ValueError, match="heldout_delta"):
        exp.validate_artifact(artifact | {"heldout_delta": 0.0})
    with pytest.raises(ValueError, match="negative_control_delta"):
        exp.validate_artifact(artifact | {"negative_control_delta": 0.1})
    with pytest.raises(ValueError, match="continuous_self_learning_task"):
        exp.validate_artifact(artifact | {"continuous_self_learning_task": False})
    with pytest.raises(ValueError, match="independent self-learning boundary"):
        exp.validate_artifact(artifact | {"independent_self_learning_boundary_preserved": False})
    with pytest.raises(ValueError, match="n_replay_items"):
        exp.validate_artifact(artifact | {"n_replay_items": 0})
    with pytest.raises(ValueError, match="forgetting_guard_passed"):
        exp.validate_artifact(artifact | {"forgetting_guard_passed": False})
    with pytest.raises(ValueError, match="drift_guard_passed"):
        exp.validate_artifact(artifact | {"drift_guard_passed": False})
    with pytest.raises(ValueError, match="controller_config_path"):
        exp.validate_artifact(artifact | {"controller_config_path": ""})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready: wrong"})
    with pytest.raises(ValueError, match="blocked prefix"):
        exp.validate_artifact(
            artifact
            | {
                "verifier_feedback_controller_ready": False,
                "honest_verdict": "complete_wrong_for_blocked",
            }
        )

    cfg = _config(tmp_path)
    exp3017_path = cfg.resolved_exp3017_artifact_path()
    exp3017_path.write_text("{", encoding="utf-8")
    malformed = exp.run_experiment(cfg)
    assert malformed["blocked_reason"] == "exp3017_artifact_malformed"

    exp3017_path.write_text(json.dumps({"instruction_validator_tree_ready": False}), encoding="utf-8")
    not_ready = exp.run_experiment(cfg)
    assert not_ready["blocked_reason"] == "exp3017_not_ready"

    exp3017_path.write_text(json.dumps({"instruction_validator_tree_ready": True}), encoding="utf-8")
    (tmp_path / exp3017.VALIDATOR_MANIFEST_REL_PATH).unlink()
    missing_manifest = exp.run_experiment(cfg)
    assert missing_manifest["blocked_reason"] == "exp3017_manifest_missing"


def test_req_learn_3020_source_blocker_matrix_and_malformed_jsonl(tmp_path: Path) -> None:
    """REQ-LEARN-3020-1: every upstream readiness gate fails closed distinctly."""

    _write_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))

    assert exp.precondition_blocker(replace(sources, exp3018_artifact={})) == (
        "exp3018_artifact_missing_or_empty"
    )
    assert exp.precondition_blocker(replace(sources, exp3018_artifact={"_malformed": True})) == (
        "exp3018_artifact_malformed"
    )
    assert exp.precondition_blocker(
        replace(sources, exp3018_artifact={"frontier_certificate_ready": False})
    ) == "exp3018_not_ready"
    assert exp.precondition_blocker(
        replace(
            sources,
            exp3018_artifact={"frontier_certificate_ready": True, "live_llm_evidence_used": True},
        )
    ) == "exp3018_live_llm_evidence_contaminated"
    assert exp.precondition_blocker(replace(sources, exp3018_rows=())) == "exp3018_manifest_missing"

    assert exp.precondition_blocker(replace(sources, exp3019_artifact={})) == (
        "exp3019_artifact_missing_or_empty"
    )
    assert exp.precondition_blocker(replace(sources, exp3019_artifact={"_malformed": True})) == (
        "exp3019_artifact_malformed"
    )
    assert exp.precondition_blocker(
        replace(sources, exp3019_artifact={"feasibility_channel_diagnostic_ready": False})
    ) == "exp3019_not_ready"
    assert exp.precondition_blocker(
        replace(
            sources,
            exp3019_artifact={
                "feasibility_channel_diagnostic_ready": True,
                "reused_label_as_feature": True,
            },
        )
    ) == "exp3019_reused_label_as_feature"
    assert exp.precondition_blocker(replace(sources, exp3019_rows=())) == "exp3019_table_missing"

    assert exp.precondition_blocker(replace(sources, exp3007_artifact={})) == (
        "exp3007_artifact_missing_or_empty"
    )
    assert exp.precondition_blocker(replace(sources, exp3007_artifact={"_malformed": True})) == (
        "exp3007_artifact_malformed"
    )
    assert exp.precondition_blocker(
        replace(sources, exp3007_artifact={"trace_memory_stability_ready": False})
    ) == "exp3007_not_ready"
    assert exp.precondition_blocker(
        replace(
            sources,
            exp3007_artifact={
                "trace_memory_stability_ready": True,
                "forgetting_guard_passed": False,
            },
        )
    ) == "exp3007_forgetting_guard_failed"
    assert exp.precondition_blocker(
        replace(
            sources,
            exp3007_artifact={
                "trace_memory_stability_ready": True,
                "forgetting_guard_passed": True,
                "drift_guard_passed": False,
            },
        )
    ) == "exp3007_drift_guard_failed"

    malformed_path = tmp_path / "bad.jsonl"
    malformed_path.write_text("{", encoding="utf-8")
    assert exp._read_jsonl(malformed_path) == [{"_malformed": True}]
    scalar_path = tmp_path / "scalar.jsonl"
    scalar_path.write_text("1\n", encoding="utf-8")
    assert exp._read_jsonl(scalar_path) == [{"_malformed": True}]
    assert exp._relative_to(tmp_path, Path("/outside/controller.json")) == Path(
        "/outside/controller.json"
    )
