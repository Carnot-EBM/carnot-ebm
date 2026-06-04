"""Tests for Exp 3791 anomaly-escalation classifier historical validation.

Spec refs: REQ-AUTO-016, SCENARIO-AUTO-013.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_3791_anomaly_escalation_classifier_validation as exp3791


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")


def test_req_auto_016_spec_anchor_exists() -> None:
    """REQ-AUTO-016: OpenSpec declares the historical validation contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-AUTO-016" in spec
    assert "SCENARIO-AUTO-013" in spec
    assert "experiment_3791_anomaly_escalation_classifier_validation.py" in spec
    assert "false-escalation rate" in spec
    assert "MUST NOT modify" in spec


def test_scenario_auto_013_labeled_sample_uses_real_historical_corpus() -> None:
    """SCENARIO-AUTO-013: sample covers retros, positives, negatives, and P1 anomalies."""

    sample = exp3791.load_labeled_sample(ROOT)

    assert len(sample) >= 30
    assert {row.expected_label for row in sample} == {
        "clean_bounded_negative",
        "frame_violating_anomaly",
        "clean_positive",
    }
    sources = {row.source for row in sample}
    assert "results/thesis_a_p1_discrete_search.json" in sources
    assert "results/thesis_a_p1_discrete_search_v2.json" in sources
    assert "results/experiment_3766_thesis_a_definitive_reconcile.json" in sources
    assert any(source.startswith("results/operational_retro_") for source in sources)
    assert all((ROOT / source).exists() for source in sources)


def test_scenario_auto_013_real_classifier_behavior_controls() -> None:
    """SCENARIO-AUTO-013: anti-poison test asserts the shipped classifier behavior."""

    results = exp3791.classify_labeled_sample(exp3791.load_labeled_sample(ROOT))
    by_source = {row["source"]: row for row in results}

    assert by_source["results/experiment_3779_abstention_operating_point_product_wiring.json"][
        "predicted_label"
    ] == "clean_positive"
    assert by_source["results/thesis_a_p1_discrete_search_v2.json"][
        "predicted_label"
    ] == "frame_violating_anomaly"
    assert by_source["results/experiment_3766_thesis_a_definitive_reconcile.json"][
        "expected_label"
    ] == "clean_bounded_negative"
    assert by_source["results/experiment_3766_thesis_a_definitive_reconcile.json"][
        "predicted_label"
    ] == "frame_violating_anomaly"


def test_scenario_auto_013_artifact_reports_confusion_and_advisory_recommendation() -> None:
    """SCENARIO-AUTO-013: artifact quantifies false escalations and recall."""

    conductor_path = ROOT / exp3791.CONDUCTOR_REL_PATH
    before_conductor = conductor_path.read_text(encoding="utf-8")
    artifact = exp3791.build_artifact(ROOT, started_s=10.0, now_s=12.5)

    exp3791.validate_artifact(artifact)

    assert set(exp3791.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3791.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == (
        "complete: anomaly_escalation_validated_n32_false_escalation_rate_"
        "0.833333_frame_violating_recall_1.000000_never_relaxes_verification_"
        "conductor_unmodified"
    )
    assert artifact["inference_substrate"] == exp3791.INFERENCE_SUBSTRATE
    assert artifact["n_validation_artifacts"] == 32
    assert artifact["confusion_matrix"]["clean_bounded_negative"][
        "frame_violating_anomaly"
    ] == 10
    assert artifact["confusion_matrix"]["clean_bounded_negative"]["clean_positive"] == 2
    assert artifact["confusion_matrix"]["frame_violating_anomaly"][
        "frame_violating_anomaly"
    ] == 2
    assert artifact["false_escalation_rate"] == pytest.approx(10 / 12)
    assert artifact["frame_violating_recall"] == pytest.approx(1.0)
    assert artifact["never_relaxes_verification"] is True
    assert artifact["supports_wiring_in"] is False
    assert artifact["tests_assert_real_behavior"] is True
    assert artifact["conductor_unmodified"] is True
    assert artifact["duration_s"] == 2.5
    assert artifact["random_seed"] == 3791
    assert artifact["reproducibility_checksum"] == exp3791.payload_checksum(artifact)
    assert artifact["cited_upstream_artifacts"] == sorted(
        row["source"] for row in artifact["validation_rows"]
    )
    assert conductor_path.read_text(encoding="utf-8") == before_conductor

    encoded = json.dumps(artifact, sort_keys=True)
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded


def test_req_auto_016_missing_corpus_blocks_without_fabricating(tmp_path: Path) -> None:
    """REQ-AUTO-016: absent historical corpus writes blocked_corpus_missing."""

    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").write_text("", encoding="utf-8")

    artifact = exp3791.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    exp3791.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_corpus_missing"
    assert artifact["n_validation_artifacts"] == 0
    assert artifact["confusion_matrix"] == {}
    assert artifact["cited_upstream_artifacts"] == []
    assert artifact["supports_wiring_in"] is False
    assert artifact["tests_assert_real_behavior"] is True


def test_req_auto_016_precondition_and_sample_loader_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-AUTO-016: precondition and corpus loader failures fail closed."""

    not_object = tmp_path / "not_object.json"
    not_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        exp3791._read_json_object(not_object)

    monkeypatch.setattr(
        exp3791,
        "SAMPLE_SPECS",
        (("results/missing.json", "clean_positive", "missing fixture"),),
    )
    with pytest.raises(FileNotFoundError):
        exp3791.load_labeled_sample(ROOT)

    monkeypatch.setattr(exp3791, "load_labeled_sample", lambda root: (_ for _ in ()).throw(FileNotFoundError()))
    missing = exp3791.build_artifact(ROOT, started_s=1.0, now_s=1.2)
    assert missing["honest_verdict"] == "blocked_corpus_missing"

    monkeypatch.setattr(exp3791, "load_labeled_sample", lambda root: (_ for _ in ()).throw(ValueError()))
    malformed = exp3791.build_artifact(ROOT, started_s=1.0, now_s=1.2)
    assert malformed["honest_verdict"] == "blocked_corpus_malformed"

    monkeypatch.setattr(exp3791.sys, "executable", "/usr/bin/python")
    bad_interpreter = exp3791.build_artifact(ROOT, started_s=1.0, now_s=1.2)
    assert bad_interpreter["honest_verdict"] == "blocked_interpreter"

    assert exp3791._rate(1, 0) == 0.0
    assert exp3791._conductor_text(tmp_path) is None
    assert exp3791.recommendations_never_relax(
        [{"verification_relaxation_recommended": True}]
    ) is False
    assert exp3791.recommendations_never_relax(
        [
            {
                "verification_relaxation_recommended": False,
                "predicted_label": "frame_violating_anomaly",
                "recommendation": "lower_bar",
            }
        ]
    ) is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact.pop("honest_verdict"), "missing required"),
        (lambda artifact: artifact.update(inference_substrate="live_model"), "inference_substrate"),
        (lambda artifact: artifact.update(n_validation_artifacts=29), "at least 30"),
        (lambda artifact: artifact.update(confusion_matrix=[]), "confusion_matrix"),
        (lambda artifact: artifact.update(confusion_matrix={}), "confusion_matrix"),
        (lambda artifact: artifact.update(false_escalation_rate="bad"), "false_escalation_rate"),
        (lambda artifact: artifact.update(false_escalation_rate=-0.1), "false_escalation_rate"),
        (lambda artifact: artifact.update(frame_violating_recall=1.5), "frame_violating_recall"),
        (lambda artifact: artifact.update(never_relaxes_verification=False), "never_relaxes_verification"),
        (lambda artifact: artifact.update(tests_assert_real_behavior=False), "tests_assert_real_behavior"),
        (lambda artifact: artifact.update(random_seed=0), "random_seed"),
        (lambda artifact: artifact.update(supports_wiring_in="false"), "supports_wiring_in"),
        (lambda artifact: artifact.update(cited_upstream_artifacts="bad"), "cited_upstream_artifacts"),
        (lambda artifact: artifact.update(cited_upstream_artifacts=[]), "cited_upstream_artifacts"),
        (lambda artifact: artifact.update(duration_s=float("nan")), "duration_s"),
        (lambda artifact: artifact.update(field_principles=[]), "field_principles"),
        (lambda artifact: artifact.update(field_principles={}), "missing field principles"),
        (lambda artifact: artifact.update(n_validation_artifacts="32"), "n_validation_artifacts"),
        (lambda artifact: artifact.update(honest_verdict="complete: wrong"), "honest_verdict"),
        (lambda artifact: artifact.update(reproducibility_checksum="bad"), "checksum"),
        (lambda artifact: artifact.update(extra_marker="GGUF"), "live-compute markers"),
    ],
)
def test_req_auto_016_validate_artifact_rejects_schema_violations(
    mutate,
    message: str,
) -> None:
    """REQ-AUTO-016: validation enforces load-bearing artifact invariants."""

    artifact = exp3791.build_artifact(ROOT, started_s=10.0, now_s=12.5)
    mutate(artifact)

    with pytest.raises(ValueError, match=message):
        exp3791.validate_artifact(artifact)


def test_req_auto_016_validate_rejects_blocked_artifact_with_rows() -> None:
    """REQ-AUTO-016: blocked artifacts cannot claim validation rows."""

    artifact = exp3791.build_artifact(ROOT, started_s=10.0, now_s=12.5)
    artifact["honest_verdict"] = "blocked_corpus_missing"
    artifact["n_validation_artifacts"] = 1
    artifact["reproducibility_checksum"] = exp3791.payload_checksum(artifact)

    with pytest.raises(ValueError, match="blocked artifacts"):
        exp3791.validate_artifact(artifact)


def test_req_auto_016_write_artifact_and_main_print_stable_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-AUTO-016: writer and CLI emit the validated JSON artifact."""

    artifact = exp3791.build_artifact(ROOT, started_s=10.0, now_s=12.5)
    monkeypatch.setattr(exp3791, "build_artifact", lambda root: dict(artifact))

    path = exp3791.write_artifact(tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / exp3791.OUTPUT_REL_PATH
    assert saved == artifact

    monkeypatch.setattr(exp3791, "write_artifact", lambda root=exp3791.REPO_ROOT: path)
    rc = exp3791.main()
    printed = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert printed == artifact
