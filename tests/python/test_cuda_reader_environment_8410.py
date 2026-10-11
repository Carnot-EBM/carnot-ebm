"""REQ-VERIFY-8410: toolkit discovery hints cannot authorize driver execution."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from carnot.verify import cuda_reader_8410 as q


@pytest.mark.parametrize("hint", ["CUDA_HOME", "CUDA_PATH"])
@pytest.mark.parametrize(
    "before,after", [(None, "/opt/cuda"), ("/opt/cuda", None), ("/old", "/new")]
)
def test_toolkit_hint_preserves_observation_without_execution_delta(hint, before, after):
    """SCENARIO-VERIFY-8410-TOOLKIT-HINTS: discovery metadata stays visible without admission."""
    previous = {"environment": {hint: before} if before is not None else {}}
    current = {"environment": {hint: after} if after is not None else {}}
    row = next(r for r in q.compare(previous, current) if r["field"] == "environment")
    assert row["previous"] == previous["environment"]
    assert row["current"] == current["environment"]
    assert row["comparable"] and row["missing_reason"] is None
    assert not row["changed"]


@pytest.mark.parametrize(
    "variable",
    [
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
    ],
)
def test_execution_environment_delta_still_admitted(variable):
    """SCENARIO-VERIFY-8410-TOOLKIT-HINTS: visibility and loader changes retain admission."""
    previous = {"environment": {variable: "before", "CUDA_PATH": "/old"}}
    current = {"environment": {variable: "after", "CUDA_PATH": "/new"}}
    row = next(r for r in q.compare(previous, current) if r["field"] == "environment")
    assert row["comparable"] and row["changed"]


def test_missing_environment_stays_unknown():
    """SCENARIO-VERIFY-8410-TOOLKIT-HINTS: absent environment evidence cannot authorize work."""
    for previous, current in [(None, {}), ({}, None), (None, None)]:
        row = next(
            r
            for r in q.compare({"environment": previous}, {"environment": current})
            if r["field"] == "environment"
        )
        assert not row["comparable"] and not row["changed"]
        assert row["missing_reason"]


def test_conductor_environment_does_not_repeat_cached_probe(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8410-TOOLKIT-HINTS: absent shell hints keep cached CUDA failure closed."""
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    probe = MagicMock(side_effect=AssertionError("toolkit hint removal cannot authorize a probe"))
    monkeypatch.setattr(q, "probe", probe)
    work = q.measure(q.ROOT, tmp_path)
    assert not work["failures"]
    assert "CUDA_PATH" in work["cached"]["previous"]["environment"]
    assert "CUDA_PATH" not in work["current"]["environment"]
    result = q.reduction(work, passed=True)
    assert result["runtime_reader_ready_score"] == 1
    assert result["runtime_changed_score"] == result["cuda_context_ready_score"] == 0
    assert result["current_device_probe_count"] == 0
    probe.assert_not_called()
    changed = deepcopy(work["current"])
    changed["kernel_driver"] += "changed"
    assert any(r["changed"] for r in q.compare(work["cached"]["previous"], changed))
