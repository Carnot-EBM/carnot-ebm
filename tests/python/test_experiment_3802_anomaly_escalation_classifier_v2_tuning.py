"""Tests for Exp 3802 tuned anomaly-escalation classifier validation.

Spec refs: REQ-AUTO-017, SCENARIO-AUTO-014.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_3802_anomaly_escalation_classifier_v2_tuning as exp3802


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")


def test_req_auto_017_spec_anchor_exists() -> None:
    """REQ-AUTO-017: OpenSpec declares the v2 tuning contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-AUTO-017" in spec
    assert "SCENARIO-AUTO-014" in spec
    assert "false escalation" in spec
    assert "frame-violating recall 1.0" in spec
    assert "MUST NOT modify the conductor" in spec


def test_scenario_auto_014_artifact_reports_tuned_metrics() -> None:
    """SCENARIO-AUTO-014: v2 validation supports advisory hook wiring."""

    conductor_path = ROOT / "scripts/research_conductor.py"
    before_conductor = conductor_path.read_text(encoding="utf-8")
    artifact = exp3802.build_artifact(ROOT, started_s=10.0, now_s=12.25)

    exp3802.validate_artifact(artifact)

    assert set(exp3802.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3802.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == (
        "complete: anomaly_escalation_v2_tuned_false_escalation_"
        "0.833333_to_0.000000_frame_violating_recall_1.0_"
        "never_relaxes_verification_supports_wiring_in_true_conductor_unmodified"
    )
    assert artifact["inference_substrate"] == exp3802.INFERENCE_SUBSTRATE
    assert artifact["false_escalation_rate_before"] == pytest.approx(0.833333)
    assert artifact["false_escalation_rate_after"] == pytest.approx(0.0)
    assert artifact["false_escalation_rate_after"] <= 0.2
    assert artifact["frame_violating_recall_after"] == pytest.approx(1.0)
    assert artifact["confusion_matrix_after"]["clean_bounded_negative"] == {
        "clean_bounded_negative": 12,
        "clean_positive": 0,
        "frame_violating_anomaly": 0,
    }
    assert artifact["confusion_matrix_after"]["frame_violating_anomaly"][
        "frame_violating_anomaly"
    ] == 2
    assert artifact["n_validation_artifacts"] == 32
    assert artifact["never_relaxes_verification"] is True
    assert artifact["supports_wiring_in"] is True
    assert artifact["tests_assert_real_behavior"] is True
    assert artifact["random_seed"] == 3802
    assert artifact["duration_s"] == 2.25
    assert artifact["validation_sample_path"] == str(
        (ROOT / exp3802.EXP3791_VALIDATION_REL_PATH).resolve()
    )
    assert artifact["reproducibility_checksum"] == exp3802.payload_checksum(artifact)
    assert "results/experiment_3791_anomaly_escalation_classifier_validation.json" in (
        artifact["cited_upstream_artifacts"]
    )
    assert "results/thesis_a_p1_discrete_search_v2.json" in (
        artifact["cited_upstream_artifacts"]
    )
    assert artifact["misflagged_clean_bounded_negatives_before"][0]["rule_fired"] == (
        "negative_missing_expected_metadata"
    )
    assert conductor_path.read_text(encoding="utf-8") == before_conductor

    encoded = json.dumps(artifact, sort_keys=True)
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded


def test_scenario_auto_014_validation_rows_pin_real_behavior() -> None:
    """SCENARIO-AUTO-014: anti-poison rows include a real clean and anomaly case."""

    artifact = exp3802.build_artifact(ROOT, started_s=1.0, now_s=1.5)
    by_source = {row["source"]: row for row in artifact["validation_rows_after"]}

    clean = by_source["results/thesis_a_part_b_matched_compute.json"]
    anomaly = by_source["results/thesis_a_p1_discrete_search_v2.json"]

    assert clean["expected_label"] == "clean_bounded_negative"
    assert clean["predicted_label"] == "clean_bounded_negative"
    assert clean["recommendation"] == "standard_auto_reconcile"
    assert anomaly["expected_label"] == "frame_violating_anomaly"
    assert anomaly["predicted_label"] == "frame_violating_anomaly"
    assert anomaly["recommendation"] == "halt_pruning_escalate_to_human"
    assert anomaly["verification_relaxation_recommended"] is False


def test_req_auto_017_preconditions_block_without_fabrication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-AUTO-017: missing interpreter or sample writes honest blocked artifacts."""

    monkeypatch.setattr(exp3802.sys, "executable", "/usr/bin/python")
    bad_interpreter = exp3802.build_artifact(ROOT, started_s=1.0, now_s=1.2)
    exp3802.validate_artifact(bad_interpreter)
    assert bad_interpreter["honest_verdict"] == "blocked_interpreter"
    assert bad_interpreter["n_validation_artifacts"] == 0

    monkeypatch.setattr(exp3802.sys, "executable", str(ROOT / ".venv/bin/python"))
    missing_sample = exp3802.build_artifact(tmp_path, started_s=1.0, now_s=1.2)
    exp3802.validate_artifact(missing_sample)
    assert missing_sample["honest_verdict"] == "blocked_validation_sample_missing"
    assert missing_sample["cited_upstream_artifacts"] == []
    assert missing_sample["supports_wiring_in"] is False

    validation_path = tmp_path / exp3802.EXP3791_VALIDATION_REL_PATH
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text("[]", encoding="utf-8")
    malformed = exp3802.build_artifact(tmp_path, started_s=1.0, now_s=1.2)
    exp3802.validate_artifact(malformed)
    assert malformed["honest_verdict"] == "blocked_validation_sample_malformed"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact.pop("honest_verdict"), "missing required"),
        (
            lambda artifact: artifact.update(false_escalation_rate_before=0.1),
            "false_escalation_rate_before",
        ),
        (
            lambda artifact: artifact.update(false_escalation_rate_after=0.25),
            "false_escalation_rate_after",
        ),
        (
            lambda artifact: artifact.update(frame_violating_recall_after=0.5),
            "frame_violating_recall_after",
        ),
        (lambda artifact: artifact.update(never_relaxes_verification=False), "never"),
        (lambda artifact: artifact.update(supports_wiring_in=False), "supports"),
        (lambda artifact: artifact.update(tests_assert_real_behavior=False), "tests"),
        (lambda artifact: artifact.update(random_seed=3791), "random_seed"),
        (lambda artifact: artifact.update(extra_marker="CUDA"), "live-compute"),
        (lambda artifact: artifact.update(reproducibility_checksum="bad"), "checksum"),
    ],
)
def test_req_auto_017_validate_rejects_invalid_complete_artifacts(
    mutate,
    message: str,
) -> None:
    """REQ-AUTO-017: validation rejects fabricated or regressed metrics."""

    artifact = exp3802.build_artifact(ROOT, started_s=10.0, now_s=12.25)
    mutate(artifact)

    with pytest.raises(ValueError, match=message):
        exp3802.validate_artifact(artifact)


def test_req_auto_017_write_artifact_and_main_print_stable_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-AUTO-017: writer and CLI emit the validated tuned artifact."""

    artifact = exp3802.build_artifact(ROOT, started_s=10.0, now_s=12.25)
    monkeypatch.setattr(exp3802, "build_artifact", lambda root=ROOT: dict(artifact))

    path = exp3802.write_artifact(tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / exp3802.OUTPUT_REL_PATH
    assert saved == artifact

    monkeypatch.setattr(exp3802, "write_artifact", lambda root=ROOT: path)
    rc = exp3802.main()
    printed = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert printed == artifact
