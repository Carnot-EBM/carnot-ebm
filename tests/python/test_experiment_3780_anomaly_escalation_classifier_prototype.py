"""Tests for Exp 3780 anomaly-escalation classifier prototype.

Spec refs: REQ-AUTO-015, SCENARIO-AUTO-012.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_3780_anomaly_escalation_classifier_prototype as exp3780


ROOT = Path(__file__).resolve().parents[2]


def test_scenario_auto_012_exp3780_writes_required_artifact_and_proposal(
    tmp_path: Path,
) -> None:
    """SCENARIO-AUTO-012: prototype artifact records recommend-only behavior."""

    output_path = tmp_path / "results/experiment_3780_anomaly_escalation_classifier_prototype.json"
    proposal_path = tmp_path / "openspec/change-proposals/anomaly-escalation-conductor-hook.md"
    conductor_path = tmp_path / "scripts/research_conductor.py"
    conductor_path.parent.mkdir(parents=True)
    conductor_path.write_text("# conductor stays untouched\n", encoding="utf-8")
    before_conductor = conductor_path.read_text(encoding="utf-8")

    artifact = exp3780.run(
        ROOT,
        output_path=output_path,
        proposal_path=proposal_path,
        conductor_path=conductor_path,
        started_s=100.0,
        now_s=100.25,
    )

    stored = json.loads(output_path.read_text(encoding="utf-8"))
    proposal = proposal_path.read_text(encoding="utf-8")

    assert stored == artifact
    assert artifact["honest_verdict"] == exp3780.TERMINAL_VERDICT
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert set(exp3780.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3780.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["classifier_shipped"] is True
    assert artifact["classifier_only_recommends"] is True
    assert artifact["change_proposal_written"] is True
    assert artifact["never_relaxes_verification"] is True
    assert artifact["tests_assert_real_behavior"] is True
    assert artifact["n_test_artifacts_classified"] >= 3
    assert artifact["random_seed"] == exp3780.RANDOM_SEED
    assert artifact["duration_s"] == 0.25
    assert artifact["reproducibility_checksum"] == exp3780.payload_checksum(artifact)
    assert "clean_bounded_negative" in artifact["clean_vs_anomaly_examples"]
    assert "frame_violating_anomaly" in artifact["clean_vs_anomaly_examples"]
    assert artifact["clean_vs_anomaly_examples"]["frame_violating_anomaly"][
        "recommendation"
    ] == "halt_pruning_escalate_to_human"
    assert "advisory" in proposal
    assert "MUST NOT auto-relax verification" in proposal
    assert conductor_path.read_text(encoding="utf-8") == before_conductor

    encoded = json.dumps(artifact, sort_keys=True)
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded


def test_req_auto_015_validate_artifact_rejects_missing_recommend_only_flag() -> None:
    """REQ-AUTO-015: artifact validation enforces the load-bearing caveat."""

    artifact = exp3780.build_artifact(
        root=ROOT,
        proposal_path=Path("openspec/change-proposals/anomaly-escalation-conductor-hook.md"),
        started_s=1.0,
        now_s=1.25,
    )
    artifact["classifier_only_recommends"] = False

    try:
        exp3780.validate_artifact(artifact)
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - assertion branch
        message = ""

    assert "classifier_only_recommends" in message


def test_req_auto_015_read_json_requires_object(tmp_path: Path) -> None:
    """REQ-AUTO-015: example artifact loading rejects non-object JSON."""

    path = tmp_path / "not_object.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        exp3780._read_json(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact.pop("honest_verdict"), "missing required artifact fields"),
        (lambda artifact: artifact.update(honest_verdict="complete: wrong"), "honest_verdict"),
        (lambda artifact: artifact.update(inference_substrate="live_llm_inference"), "inference_substrate"),
        (lambda artifact: artifact.update(clean_vs_anomaly_examples=[]), "clean_vs_anomaly_examples"),
        (
            lambda artifact: artifact["clean_vs_anomaly_examples"].pop("frame_violating_anomaly"),
            "frame_violating_anomaly",
        ),
        (lambda artifact: artifact.update(reproducibility_checksum="bad"), "checksum"),
        (lambda artifact: artifact.update(extra_marker="GGUF"), "live-compute markers"),
    ],
)
def test_req_auto_015_validate_artifact_rejects_schema_violations(
    mutate,
    message: str,
) -> None:
    """REQ-AUTO-015: validation enforces every load-bearing artifact invariant."""

    artifact = exp3780.build_artifact(
        root=ROOT,
        proposal_path=ROOT / exp3780.PROPOSAL_REL_PATH,
        started_s=1.0,
        now_s=1.25,
    )
    mutate(artifact)

    with pytest.raises(ValueError, match=message):
        exp3780.validate_artifact(artifact)


def test_req_auto_015_run_detects_conductor_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-AUTO-015: the prototype refuses to proceed if conductor content changes."""

    conductor_path = tmp_path / "scripts/research_conductor.py"
    conductor_path.parent.mkdir(parents=True)
    conductor_path.write_text("before\n", encoding="utf-8")
    proposal_path = tmp_path / "openspec/change-proposals/anomaly.md"

    def mutating_proposal(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(exp3780.PROPOSAL_TEXT, encoding="utf-8")
        conductor_path.write_text("after\n", encoding="utf-8")
        return path

    monkeypatch.setattr(exp3780, "write_change_proposal", mutating_proposal)

    with pytest.raises(RuntimeError, match="research_conductor.py changed"):
        exp3780.run(
            ROOT,
            output_path=tmp_path / "results/out.json",
            proposal_path=proposal_path,
            conductor_path=conductor_path,
            started_s=1.0,
            now_s=1.25,
        )


def test_req_auto_015_main_prints_artifact(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-AUTO-015: script entrypoint prints the generated artifact."""

    monkeypatch.setattr(exp3780, "run", lambda: {"honest_verdict": exp3780.TERMINAL_VERDICT})

    rc = exp3780.main()
    output = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert output == {"honest_verdict": exp3780.TERMINAL_VERDICT}
