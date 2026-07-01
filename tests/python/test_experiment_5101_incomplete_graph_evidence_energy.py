"""Tests for Exp 5101 incomplete graph evidence energy.

Spec refs: REQ-VERIFY-5101, SCENARIO-VERIFY-5101.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5101_incomplete_graph_evidence_energy as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_verify_5101_spec_declares_graph_energy_contract() -> None:
    """REQ-VERIFY-5101: OpenSpec anchors the module, artifact, fields, and substrate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5101",
        "SCENARIO-VERIFY-5101",
        "python/carnot/experiment_5101_incomplete_graph_evidence_energy.py",
        "results/experiment_5101_incomplete_graph_evidence_energy_v468.json",
        mod.INFERENCE_SUBSTRATE,
        mod.SUCCESS_VERDICT,
        mod.NO_WIN_VERDICT,
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_verify_5101_fixture_has_hidden_evidence_and_exact_labels() -> None:
    """SCENARIO-VERIFY-5101: synthetic graph authority creates all four label classes."""

    fixture = mod.build_graph_fixture(hidden_edge_rate=0.4, perturbation=0.0)
    rows = [row.as_dict() for row in mod.score_claims(fixture)]
    labels = {row["exact_label"] for row in rows}

    assert fixture.observed_edges
    assert fixture.hidden_edges
    assert labels == {
        "supported",
        "contradicted",
        "unsupported_true",
        "unsupported_false",
    }
    assert any(row["support_paths"] for row in rows if row["exact_label"] == "supported")
    assert any(row["contradiction_edges"] for row in rows if row["exact_label"] == "contradicted")
    assert all(row["inference_authority"] == "synthetic_graph_exact_labels" for row in rows)
    assert all(set(mod.FEATURE_NAMES) <= set(row["features"]) for row in rows)


def test_req_verify_5101_energy_separates_support_retention_and_rejection() -> None:
    """REQ-VERIFY-5101: energy decisions do not collapse unsupported true and contradiction."""

    evaluation = mod.run_evaluation()
    thresholds = evaluation["energy_thresholds"]
    heldout = [row for row in evaluation["claim_rows"] if row["split"] == "heldout"]
    by_label = {
        label: [row for row in heldout if row["exact_label"] == label]
        for label in (
            "supported",
            "contradicted",
            "unsupported_true",
            "unsupported_false",
        )
    }

    assert all(by_label.values())
    assert all(row["decision"] == "accept" for row in by_label["supported"])
    assert all(row["decision"] == "reject" for row in by_label["contradicted"])
    assert all(row["decision"] == "retain" for row in by_label["unsupported_true"])
    assert all(row["decision"] == "reject" for row in by_label["unsupported_false"])
    assert max(row["energy"] for row in by_label["supported"]) <= thresholds["accept_below"]
    assert min(row["energy"] for row in by_label["unsupported_true"]) > thresholds["accept_below"]
    assert max(row["energy"] for row in by_label["unsupported_true"]) < thresholds["reject_at_or_above"]
    assert min(row["energy"] for row in by_label["contradicted"]) >= thresholds["reject_at_or_above"]


def test_scenario_verify_5101_thresholds_are_tuned_without_heldout() -> None:
    """SCENARIO-VERIFY-5101: train/dev tune thresholds, heldout reports final rates."""

    evaluation = mod.run_evaluation()

    assert evaluation["threshold_tuning"]["splits_used"] == ["train", "dev"]
    assert evaluation["threshold_tuning"]["heldout_used_for_tuning"] is False
    assert all(row["split_scope"] == "dev" for row in evaluation["slack_sweep"])
    assert evaluation["heldout_metrics_by_label"]["supported"]["accept_rate"] == pytest.approx(1.0)
    assert evaluation["heldout_metrics_by_label"]["contradicted"]["reject_rate"] == pytest.approx(1.0)
    assert evaluation["heldout_metrics_by_label"]["unsupported_true"]["retain_rate"] == pytest.approx(
        1.0
    )
    assert evaluation["heldout_metrics_by_label"]["unsupported_false"]["reject_rate"] == pytest.approx(
        1.0
    )


def test_req_verify_5101_artifact_contains_required_fields_and_principles(tmp_path: Path) -> None:
    """REQ-VERIFY-5101: artifact emits required schema fields and principle notes."""

    artifact = mod.write_artifact(root=tmp_path)
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert loaded == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith(mod.SUCCESS_VERDICT)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["hidden_edge_rate"] == pytest.approx(mod.PRIMARY_HIDDEN_EDGE_RATE)
    assert artifact["hidden_edge_rates_evaluated"] == list(mod.HIDDEN_EDGE_RATES)
    assert artifact["supported_accept_rate"] == pytest.approx(1.0)
    assert artifact["contradiction_reject_rate"] == pytest.approx(1.0)
    assert artifact["unsupported_retained_rate"] == pytest.approx(1.0)
    assert artifact["unsupported_false_reject_rate"] == pytest.approx(1.0)
    assert artifact["flagged_adversarial"] is False
    assert "llm" not in artifact["inference_substrate"].lower()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "optimistic", "honest_verdict"),
        ("duration_s", -1.0, "duration_s"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("graph_fixture_hash", "short", "graph_fixture_hash"),
        ("supported_accept_rate", 2.0, "supported_accept_rate"),
        ("contradiction_reject_rate", -0.1, "contradiction_reject_rate"),
        ("unsupported_retained_rate", 2.0, "unsupported_retained_rate"),
        ("unsupported_false_reject_rate", -0.1, "unsupported_false_reject_rate"),
        ("flagged_adversarial", "false", "flagged_adversarial"),
    ],
)
def test_req_verify_5101_validate_artifact_rejects_schema_violations(
    field: str,
    value: object,
    message: str,
) -> None:
    """REQ-VERIFY-5101: malformed terminal artifacts fail closed."""

    artifact = mod.run()
    artifact[field] = value

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda artifact: artifact.pop("energy_thresholds"), "missing required fields"),
        (lambda artifact: artifact.update({"energy_thresholds": {"accept_below": 0.9}}), "energy_thresholds"),
        (lambda artifact: artifact.update({"slack_sweep": []}), "slack_sweep"),
        (
            lambda artifact: artifact.update(
                {"stability_under_perturbation": {"passed": False}}
            ),
            "stability_under_perturbation",
        ),
        (lambda artifact: artifact.update({"field_principles": {}}), "field_principles"),
    ],
)
def test_req_verify_5101_validate_artifact_rejects_consistency_violations(
    mutator: object,
    message: str,
) -> None:
    """REQ-VERIFY-5101: coherent-looking but inconsistent artifacts fail closed."""

    artifact = mod.run()
    mutator(artifact)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


def test_req_verify_5101_main_writes_default_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5101: CLI entrypoint writes the configured result path."""

    monkeypatch.setenv("CARNOT_EXP5101_ROOT", str(tmp_path))

    assert mod.main() == 0
    payload = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["result_path"] == mod.RESULT_RELATIVE_PATH


def test_deliverable_file_validates_for_req_verify_5101() -> None:
    """SCENARIO-VERIFY-5101: checked-in deliverable satisfies the terminal schema."""

    assert ARTIFACT_PATH.exists()
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith(mod.SUCCESS_VERDICT)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
