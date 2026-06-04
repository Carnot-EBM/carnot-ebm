"""Tests for Exp 3809 recommend-only anomaly escalation advisory wiring.

Spec refs: REQ-AUTO-018, SCENARIO-AUTO-015.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.autoresearch.anomaly_escalation_advisory import classify_negative
from scripts import experiment_3809_anomaly_escalation_advisory_hook as exp3809


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
PROPOSAL_PATH = Path("openspec/change-proposals/anomaly-escalation-conductor-hook.md")


def _artifact(path: str) -> dict[str, object]:
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_req_auto_018_spec_anchor_exists() -> None:
    """REQ-AUTO-018: OpenSpec declares the advisory wrapper contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-AUTO-018" in spec
    assert "SCENARIO-AUTO-015" in spec
    assert "classify_negative" in spec
    assert "recommend-only" in spec
    assert "MUST NOT relax verification" in spec
    assert "leave the actual conductor unmodified" in spec


def test_scenario_auto_015_classify_negative_real_clean_and_anomaly() -> None:
    """SCENARIO-AUTO-015: real clean negative and anomaly rows are not poison tests."""

    clean = classify_negative(_artifact("results/thesis_a_part_b_matched_compute.json"))
    anomaly = classify_negative(_artifact("results/thesis_a_p1_discrete_search_v2.json"))

    assert clean == {
        "recommendation": "auto_reconcile",
        "reason": "verdict text matches tuned bounded-negative verdict text",
        "frame_violation": False,
    }
    assert anomaly["recommendation"] == "escalate_to_human"
    assert "positive control failure" in anomaly["reason"]
    assert anomaly["frame_violation"] is True
    assert set(clean) == {"recommendation", "reason", "frame_violation"}
    assert set(anomaly) == {"recommendation", "reason", "frame_violation"}


def test_req_auto_018_classify_negative_accepts_verdict_string() -> None:
    """REQ-AUTO-018: verdict-only inputs still use the tuned classifier path."""

    result = classify_negative(
        "complete: bounded_kill_gate_honest_negative_no_improvement"
    )

    assert result["recommendation"] == "auto_reconcile"
    assert result["frame_violation"] is False
    assert "bounded-negative" in result["reason"]


def test_scenario_auto_015_artifact_reports_offline_replay_and_boundaries() -> None:
    """SCENARIO-AUTO-015: Exp 3809 reports replay metrics and conductor boundary."""

    conductor_path = ROOT / exp3809.CONDUCTOR_REL_PATH
    before_conductor = conductor_path.read_text(encoding="utf-8")
    artifact = exp3809.build_artifact(ROOT, started_s=10.0, now_s=12.75)

    exp3809.validate_artifact(artifact)

    assert set(exp3809.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3809.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == (
        "complete: anomaly_escalation_advisory_hook_wired_recommend_only_"
        "replay_false_escalation_0.000000_frame_violating_recall_1.0_"
        "never_relaxes_verification_conductor_unmodified_integration_proposal_emitted"
    )
    assert artifact["inference_substrate"] == exp3809.INFERENCE_SUBSTRATE
    assert artifact["advisory_module_added"] is True
    assert artifact["offline_replay_false_escalation_rate"] == pytest.approx(0.0)
    assert artifact["offline_replay_false_escalation_rate"] <= 0.2
    assert artifact["offline_replay_frame_violating_recall"] == pytest.approx(1.0)
    assert artifact["never_relaxes_verification"] is True
    assert artifact["conductor_unmodified"] is True
    assert artifact["integration_proposal_emitted"] is True
    assert artifact["n_replay_negatives"] == 32
    assert artifact["tests_assert_real_behavior"] is True
    assert artifact["random_seed"] == 3809
    assert artifact["duration_s"] == 2.75
    assert artifact["reproducibility_checksum"] == exp3809.payload_checksum(artifact)
    assert artifact["confusion_matrix"]["clean_bounded_negative"] == {
        "auto_reconcile": 12,
        "escalate_to_human": 0,
    }
    assert artifact["confusion_matrix"]["frame_violating_anomaly"] == {
        "auto_reconcile": 0,
        "escalate_to_human": 2,
    }
    assert artifact["replay_rows"][0]["recommendation"] == "auto_reconcile"
    assert "results/experiment_3802_anomaly_escalation_classifier_v2_tuning.json" in (
        artifact["cited_upstream_artifacts"]
    )
    assert str((ROOT / exp3809.EXP3802_TUNED_REL_PATH).resolve()) == artifact[
        "tuned_classifier_artifact_path"
    ]
    assert conductor_path.read_text(encoding="utf-8") == before_conductor

    encoded = json.dumps(artifact, sort_keys=True)
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "live-model" not in encoded


def test_scenario_auto_015_proposal_describes_operator_applied_hook() -> None:
    """SCENARIO-AUTO-015: integration is proposal-only, not conductor mutation."""

    artifact = exp3809.build_artifact(ROOT, started_s=1.0, now_s=1.5)
    proposal = (ROOT / PROPOSAL_PATH).read_text(encoding="utf-8")

    assert artifact["integration_proposal_emitted"] is True
    assert "carnot.autoresearch.anomaly_escalation_advisory" in proposal
    assert "classify_negative" in proposal
    assert "operator applies" in proposal
    assert "recommend-only" in proposal
    assert "scripts/research_conductor.py" in proposal
    assert "MUST NOT auto-relax verification" in proposal


def test_req_auto_018_preconditions_block_without_fabricating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-AUTO-018: missing interpreter or tuned artifact writes blocked values."""

    monkeypatch.setattr(exp3809.sys, "executable", "/usr/bin/python")
    bad_interpreter = exp3809.build_artifact(ROOT, started_s=1.0, now_s=1.25)
    exp3809.validate_artifact(bad_interpreter)
    assert bad_interpreter["honest_verdict"] == "blocked_interpreter"
    assert bad_interpreter["advisory_module_added"] is False
    assert bad_interpreter["n_replay_negatives"] == 0

    monkeypatch.setattr(exp3809.sys, "executable", str(ROOT / ".venv/bin/python"))
    missing_tuned = exp3809.build_artifact(tmp_path, started_s=1.0, now_s=1.25)
    exp3809.validate_artifact(missing_tuned)
    assert missing_tuned["honest_verdict"] == "blocked_tuned_classifier_missing"
    assert missing_tuned["cited_upstream_artifacts"] == []
    assert missing_tuned["integration_proposal_emitted"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact.pop("honest_verdict"), "missing required"),
        (lambda artifact: artifact.update(inference_substrate="live-model"), "inference"),
        (lambda artifact: artifact.update(advisory_module_added=False), "advisory"),
        (
            lambda artifact: artifact.update(offline_replay_false_escalation_rate=0.25),
            "false_escalation",
        ),
        (
            lambda artifact: artifact.update(offline_replay_frame_violating_recall=0.5),
            "frame_violating",
        ),
        (lambda artifact: artifact.update(never_relaxes_verification=False), "never"),
        (lambda artifact: artifact.update(conductor_unmodified=False), "conductor"),
        (lambda artifact: artifact.update(integration_proposal_emitted=False), "proposal"),
        (lambda artifact: artifact.update(n_replay_negatives=29), "n_replay"),
        (lambda artifact: artifact.update(tests_assert_real_behavior=False), "tests"),
        (lambda artifact: artifact.update(cited_upstream_artifacts=[]), "cited"),
        (lambda artifact: artifact.update(random_seed=0), "random_seed"),
        (lambda artifact: artifact.update(duration_s=float("nan")), "duration_s"),
        (lambda artifact: artifact.update(extra_marker="CUDA"), "live-compute"),
        (lambda artifact: artifact.update(reproducibility_checksum="bad"), "checksum"),
    ],
)
def test_req_auto_018_validate_rejects_invalid_complete_artifacts(
    mutate,
    message: str,
) -> None:
    """REQ-AUTO-018: validation rejects fabricated or regressed advisory claims."""

    artifact = exp3809.build_artifact(ROOT, started_s=10.0, now_s=12.75)
    mutate(artifact)

    with pytest.raises(ValueError, match=message):
        exp3809.validate_artifact(artifact)


def test_req_auto_018_write_artifact_and_main_print_stable_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-AUTO-018: writer and CLI emit the validated advisory artifact."""

    artifact = exp3809.build_artifact(ROOT, started_s=10.0, now_s=12.75)
    monkeypatch.setattr(exp3809, "build_artifact", lambda root=ROOT: dict(artifact))

    path = exp3809.write_artifact(tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / exp3809.OUTPUT_REL_PATH
    assert saved == artifact

    monkeypatch.setattr(exp3809, "write_artifact", lambda root=exp3809.REPO_ROOT: path)
    rc = exp3809.main()
    printed = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert printed == artifact
