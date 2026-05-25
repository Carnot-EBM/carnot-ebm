"""Tests for Exp 3019 FR-11 feasibility-channel de-tautology diagnostic.

Spec refs: REQ-VERIFY-3019, SCENARIO-VERIFY-3019,
SCENARIO-VERIFY-3019-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import beaver_style_validator_frontier_certificate_v1 as exp3018
from carnot.eval import fr11_feasibility_channel_de_tautology_diagnostic_v1 as exp
from carnot.eval import nsvif_instruction_validator_tree_expansion_v1 as exp3017


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "experiment_3019_fr11_feasibility_channel_de_tautology_diagnostic_v1.py"
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


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        diagnostic_table_path=tmp_path / exp.DIAGNOSTIC_TABLE_REL_PATH,
        source_certificate_artifact_path=tmp_path / "results" / exp3018.ARTIFACT_FILENAME,
        source_certificate_manifest_path=tmp_path / exp3018.CERTIFICATE_MANIFEST_REL_PATH,
        source_validator_manifest_path=tmp_path / exp3017.VALIDATOR_MANIFEST_REL_PATH,
        exp3007_artifact_path=tmp_path / exp.EXP3007_REL_PATH,
        started_at=30.0,
        clock=lambda: 34.0,
        tests_run=("focused-req-3019",),
    )


def _write_exp3007_minimal(tmp_path: Path) -> dict[str, object]:
    artifact = {
        "artifact": "experiment_3007_fr11_attractor_trace_memory_stability_v1",
        "trace_memory_stability_ready": True,
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
        "accepted_memory_ids": ["trace-a", "trace-b"],
        "replay_cycles": [
            {"cycle": 0, "accepted_memory_ids": [], "heldout_score": 0.5},
            {"cycle": 1, "accepted_memory_ids": ["trace-a", "trace-b"], "heldout_score": 1.0},
        ],
        "heldout_baseline_score": 0.5,
        "heldout_final_score": 1.0,
        "native_attractor_model_claim_made": False,
        "self_reported_memory_utility_counted": False,
        "promotion_metric_names": ["exact_heldout_verifier_score"],
    }
    path = tmp_path / exp.EXP3007_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def _write_sources(tmp_path: Path) -> None:
    exp3017.run_experiment(_exp3017_config(tmp_path))
    exp3018.run_experiment(_exp3018_config(tmp_path))
    _write_exp3007_minimal(tmp_path)


def _load_table(tmp_path: Path, artifact: dict[str, object]) -> list[dict[str, object]]:
    table_path = tmp_path / str(artifact["diagnostic_table_path"])
    return [
        json.loads(line)
        for line in table_path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_req_verify_3019_spec_and_template_script_anchor_exists() -> None:
    """REQ-VERIFY-3019: Exp 3019 is OpenSpec anchored and template-runnable."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3019" in spec
    assert "SCENARIO-VERIFY-3019" in spec
    assert "SCENARIO-VERIFY-3019-BLOCKED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "feasibility_channel_diagnostic_ready" in spec
    assert "tautology_risk_flag" in spec
    assert SCRIPT_PATH.exists()
    assert "ExperimentTemplate" in script


def test_scenario_verify_3019_writes_diagnostic_table_and_required_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3019: exact rows become inspectable feasibility evidence."""

    _write_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))
    rows = _load_table(tmp_path, artifact)

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["feasibility_channel_diagnostic_ready"] is True
    assert artifact["diagnostic_table_path"] == str(exp.DIAGNOSTIC_TABLE_REL_PATH)
    assert artifact["n_rows"] == len(rows)
    assert artifact["feasible_infeasible_auc"] >= 0.95
    assert artifact["negative_control_rejection_rate"] == pytest.approx(1.0)
    assert artifact["heldout_metric_correlation"] > 0.80
    assert artifact["tautology_risk_flag"] is True
    assert artifact["reused_label_as_feature"] is False
    assert artifact["native_dsp_claim_made"] is False
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["duration_s"] == pytest.approx(4.0)
    assert artifact["tests_run"] == ["focused-req-3019"]

    classes = {row["feasibility_class"] for row in rows}
    assert {"feasible", "violating", "unresolved", "non_prefix", "negative_control"} <= classes
    assert artifact["class_counts"]["feasible"] == exp3017.MIN_INSTRUCTION_ITEMS
    assert artifact["class_counts"]["violating"] == exp3017.MIN_INSTRUCTION_ITEMS
    assert artifact["class_counts"]["negative_control"] == 3

    banned = {"certificate_status", "candidate_role", "heldout_success_label"}
    assert not (banned & set(artifact["score_feature_names"]))
    for row in rows:
        assert not (banned & set(row["feature_values"]))
        assert 0.0 <= float(row["feasibility_score"]) <= 1.0
        assert row["native_dsp_claim_made"] is False

    heldout_rows = [row for row in rows if row["heldout_partition"]]
    assert heldout_rows
    assert {row["heldout_success_label"] for row in heldout_rows} == {True, False}

    exp.validate_artifact(artifact)


def test_req_verify_3019_features_score_exact_rows_without_label_reuse(tmp_path: Path) -> None:
    """REQ-VERIFY-3019: feature scoring avoids status/role labels as inputs."""

    _write_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    rows = exp.build_diagnostic_rows(_config(tmp_path), sources)
    safe = next(row for row in rows if row["certificate_status"] == "certified_safe")
    violating = next(row for row in rows if row["certificate_status"] == "certified_violating")

    assert safe["feasibility_score"] > exp.FEASIBILITY_PROMOTION_THRESHOLD
    assert violating["feasibility_score"] < exp.FEASIBILITY_PROMOTION_THRESHOLD
    assert safe["feature_values"]["failure_count"] == 0
    assert violating["feature_values"]["failure_count"] > 0
    assert exp.reused_label_as_feature(exp.SCORE_FEATURE_NAMES) is False
    assert exp.reused_label_as_feature(("failure_count", "certificate_status")) is True


def test_scenario_verify_3019_blocked_artifact_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3019-BLOCKED: missing exact evidence writes zeroed gates."""

    artifact = exp.run_experiment(_config(tmp_path))

    assert artifact["feasibility_channel_diagnostic_ready"] is False
    assert artifact["n_rows"] == 0
    assert artifact["feasible_infeasible_auc"] == 0.0
    assert artifact["negative_control_rejection_rate"] == 0.0
    assert artifact["heldout_metric_correlation"] == 0.0
    assert artifact["tautology_risk_flag"] is False
    assert artifact["reused_label_as_feature"] is False
    assert artifact["native_dsp_claim_made"] is False
    assert artifact["honest_verdict"].startswith("blocked_")
    assert (tmp_path / "results" / exp.ARTIFACT_FILENAME).is_file()
    assert not (tmp_path / exp.DIAGNOSTIC_TABLE_REL_PATH).exists()

    exp.validate_artifact(artifact)


def test_req_verify_3019_validation_and_source_blockers_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-3019: terminal validation rejects leakage and malformed sources."""

    _write_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete_incomplete"})
    with pytest.raises(ValueError, match="reused_label_as_feature"):
        exp.validate_artifact(artifact | {"reused_label_as_feature": True})
    with pytest.raises(ValueError, match="native_dsp_claim_made"):
        exp.validate_artifact(artifact | {"native_dsp_claim_made": True})
    with pytest.raises(ValueError, match="n_rows"):
        exp.validate_artifact(artifact | {"n_rows": 0})
    with pytest.raises(ValueError, match="diagnostic_table_path"):
        exp.validate_artifact(artifact | {"diagnostic_table_path": ""})
    with pytest.raises(ValueError, match="range"):
        exp.validate_artifact(artifact | {"feasible_infeasible_auc": 1.5})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready: wrong"})
    with pytest.raises(ValueError, match="blocked prefix"):
        exp.validate_artifact(
            artifact
            | {
                "feasibility_channel_diagnostic_ready": False,
                "honest_verdict": "complete_wrong_for_blocked",
            }
        )

    cfg = _config(tmp_path)
    cert_artifact = cfg.resolved_source_certificate_artifact_path()
    cert_artifact.write_text("{", encoding="utf-8")
    malformed = exp.run_experiment(cfg)
    assert malformed["blocked_reason"] == "exp3018_artifact_malformed"

    cert_artifact.write_text(
        json.dumps({"frontier_certificate_ready": False}),
        encoding="utf-8",
    )
    unready = exp.run_experiment(cfg)
    assert unready["blocked_reason"] == "exp3018_not_ready"

    cert_artifact.write_text(
        json.dumps({"frontier_certificate_ready": True}),
        encoding="utf-8",
    )
    (tmp_path / exp3018.CERTIFICATE_MANIFEST_REL_PATH).unlink()
    missing_manifest = exp.run_experiment(cfg)
    assert missing_manifest["blocked_reason"] == "exp3018_certificate_manifest_missing"

    ready_cert = {"frontier_certificate_ready": True}
    ready_3007 = {
        "trace_memory_stability_ready": True,
        "negative_control_report": {"control_heldout_deltas": {"irrelevant_trace": 0.0}},
    }
    assert (
        exp.precondition_blocker(exp.SourceBundle(ready_cert, ({"row": 1},), (), ready_3007))
        == "exp3017_validator_manifest_missing"
    )
    assert (
        exp.precondition_blocker(exp.SourceBundle(ready_cert, ({"row": 1},), ({"row": 1},), {}))
        == "exp3007_artifact_missing_or_empty"
    )
    assert (
        exp.precondition_blocker(
            exp.SourceBundle(ready_cert, ({"row": 1},), ({"row": 1},), {"_malformed": True})
        )
        == "exp3007_artifact_malformed"
    )
    assert (
        exp.precondition_blocker(
            exp.SourceBundle(
                ready_cert,
                ({"row": 1},),
                ({"row": 1},),
                {"trace_memory_stability_ready": False},
            )
        )
        == "exp3007_not_ready"
    )
    assert (
        exp.precondition_blocker(
            exp.SourceBundle(
                ready_cert,
                ({"row": 1},),
                ({"row": 1},),
                {"trace_memory_stability_ready": True, "negative_control_report": {}},
            )
        )
        == "exp3007_negative_controls_missing"
    )


def test_req_verify_3019_metric_helpers_cover_degenerate_edges() -> None:
    """REQ-VERIFY-3019: metric helpers fail closed on small or constant samples."""

    assert exp.mann_whitney_auc([], []) == 0.0
    assert exp.mann_whitney_auc([0.5], [0.5]) == 0.5
    assert exp.pearson_correlation([1.0], [1.0]) == 0.0
    assert exp.pearson_correlation([0.2, 0.2], [0.0, 1.0]) == 0.0
    assert exp.pearson_correlation([0.0, 1.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert exp._tautology_risk_reason(False, 0.0).startswith("no tautology")
    assert exp._heldout_label({"item_id": "if-3017-005"}, None) is None
    assert exp._heldout_label({"item_id": "if-3017-005", "candidate_role": "other"}, {}) is None

    controls = [
        {
            "feasibility_class": "negative_control",
            "feasibility_score": 0.2,
            "negative_control_delta": 0.0,
            "negative_control_accepted": False,
        },
        {
            "feasibility_class": "negative_control",
            "feasibility_score": 0.8,
            "negative_control_delta": 0.1,
            "negative_control_accepted": True,
        },
    ]
    assert exp.negative_control_rejection_rate(controls) == pytest.approx(0.5)
    assert exp.negative_control_rejection_rate([]) == 0.0


def test_req_verify_3019_cli_runs_against_explicit_cached_sources(tmp_path: Path) -> None:
    """REQ-VERIFY-3019: the module CLI writes the same schema from explicit paths."""

    _write_sources(tmp_path)
    output = tmp_path / "results" / "cli-3019.json"
    table = tmp_path / "results" / "cli-3019-table.jsonl"

    exit_code = exp.main(
        [
            "--output",
            str(output),
            "--diagnostic-table",
            str(table),
            "--source-certificate-artifact",
            str(tmp_path / "results" / exp3018.ARTIFACT_FILENAME),
            "--source-certificate-manifest",
            str(tmp_path / exp3018.CERTIFICATE_MANIFEST_REL_PATH),
            "--source-validator-manifest",
            str(tmp_path / exp3017.VALIDATOR_MANIFEST_REL_PATH),
            "--exp3007-artifact",
            str(tmp_path / exp.EXP3007_REL_PATH),
        ]
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert loaded["feasibility_channel_diagnostic_ready"] is True
    assert table.is_file()
