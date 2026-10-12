"""REQ-REPORT-8443: rejected producer qualification preserves the observed zero."""

from contextlib import nullcontext

from carnot.reporting import v727_historical_inputs as h


def test_failed_historical_readiness(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8443-FAMILIES: authenticated zero is distinct from absence."""
    original = dict(
        path=str(tmp_path / "primary"),
        snapshot_path=str(tmp_path / "copy"),
        sha256="sha256:" + h.PINS[8431],
    )
    refs = [
        dict(snapshot_path=str(tmp_path / "design")),
        dict(snapshot_path=str(tmp_path / "active")),
    ]
    monkeypatch.setattr(
        h, "sha256_file", lambda p: h.DESIGN_PIN if p.name == "design" else h.ACTIVE_PIN
    )
    monkeypatch.setattr(h.cap, "load", lambda ref: dict(head_ready_score=0))
    monkeypatch.setattr(h, "frozen_inputs", lambda refs, scratch: nullcontext())
    (tmp_path / "active").write_text("fixture")
    monkeypatch.setattr(h.yaml, "safe_load", lambda raw: dict(tasks=[dict(id="exp8431-fixture")]))
    monkeypatch.setattr(h.cap, "outcome", lambda *args: dict(eligible=True))
    result = h.qualify(8431, original, refs, tmp_path / "scratch")
    assert not result["qualified"]
    failure = result["gate_check_summary"][0]
    assert failure["observed"] == 0
    assert failure["reason"] == "original_producer_replay_and_readiness"


def test_service_without_v726_authority(tmp_path):
    """SCENARIO-REPORT-8443-FAMILIES: V724 service survives absent V726 custody."""
    source = h.ROOT / "results/experiment_8407_v724_python_transaction_cost.json"
    primary = h.cap.freeze(source, tmp_path / "primary")
    result = h.qualify(8407, primary, [], tmp_path / "scratch")
    assert result["qualified"], result["gate_check_summary"]
    assert result["field"] == "python_cost_ready_score"
