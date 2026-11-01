"""REQ-REPORT-8706, REQ-VERIFY-8706: numerical state and durable restart controls."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.verify import spline_lifecycle_8706 as n


def fixture():
    """Small constructed coefficients isolate arithmetic from natural learning claims."""
    head = dict(
        arm="spline",
        parameters=[0.01] * 109,
        decay_scale=1.0,
        temperature=1.0,
        scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9),
    )
    return dict(
        head=head,
        head_index=0,
        sequence="alternating",
        labels=[0, 1],
        vectors=[dict(x=[0.4] * 9)],
        boundary=False,
    )


def test_restart_and_operation_costs(tmp_path):
    """SCENARIO-REPORT-8706-LIFECYCLE: identical commits acknowledge identical work."""
    job = fixture()
    a = n.drive(job, tmp_path / "full", 0, 8, True)
    n.drive(job, tmp_path / "split", 0, 4, False)
    b = n.drive(job, tmp_path / "split", 4, 8, False)
    assert a["checkpoint"]["state"] == b["checkpoint"]["state"]
    assert a["checkpoint"]["ack"] == b["checkpoint"]["ack"]
    assert a["counts"]["overflow"] == a["counts"]["saturation"] == 0
    rows = [json.loads(x) for x in Path(a["rows_path"]).read_text().splitlines()]
    assert len(rows) == 8 and all(r["passed"] for r in rows)
    assert all(r["local_writes"] <= 37 for r in rows)
    assert all(r["host_snapshot_coefficient_copies"] == 109 for r in rows)
    assert a["costs"]["host_serialization"]["calls"] > 0
    assert a["costs"]["durable_storage"]["calls"] > 0
    assert n.envelope(a)["whole_format_bound_exceeds_signed64"]


def test_boundary_and_rejections(tmp_path):
    """SCENARIO-REPORT-8706-LIFECYCLE: artificial decay states expose dense writes."""
    job = dict(fixture(), boundary=True, boundary_offset=-1)
    value = n.drive(job, tmp_path / "boundary", 0, 4, True)
    rows = [json.loads(x) for x in Path(value["rows_path"]).read_text().splitlines()]
    assert rows[0]["renormalization_writes"] == 109
    assert rows[0]["boundary_continuity_error"] <= 0.5 / (1 << 24)
    with pytest.raises(ValueError, match="restart_identity"):
        n.drive(fixture(), tmp_path / "boundary", 1, 4, False)
    f = n.numeric.Fixed(32, 24)
    with pytest.raises(OverflowError, match="signed64"):
        f.checked(1 << 63)
    bad = deepcopy(fixture())
    bad["head"]["parameters"] = [100.0] * 109
    failed = n.drive(bad, tmp_path / "bad", 0, 2, True)
    assert not failed["passed"] and failed["first_failure"] is not None


def test_durable_identity_overflow_and_fallback(tmp_path, monkeypatch):
    """REQ-VERIFY-8706: stored bytes and rejected widened products cannot be hidden."""
    job = fixture()
    n.drive(job, tmp_path / "resume", 0, 2, False)
    (tmp_path / "resume" / "checkpoint.bin").write_bytes(b"bad")
    with pytest.raises(ValueError, match="restart_serialization"):
        n.drive(job, tmp_path / "resume", 2, 4, False)

    def overflow(*args):
        raise OverflowError("signed64")

    with monkeypatch.context() as m:
        m.setattr(n.numeric, "step", overflow)
        result = n.drive(job, tmp_path / "overflow", 0, 1, True)
    assert not result["passed"] and result["first_failure"]["error"] == "signed64"
    near = fixture()
    near["head"]["parameters"] = [0.0] * 109
    near["vectors"] = [dict(x=[0.2] + [0.4] * 8)]
    result = n.drive(near, tmp_path / "threshold", 0, 1, True)
    row = json.loads(Path(result["rows_path"]).read_text())
    assert row["fallback"] and row["decision_parity"]
