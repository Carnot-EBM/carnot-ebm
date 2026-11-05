"""REQ-REPORT-8848: exercise the parent heartbeat without extending its deadline."""

from pathlib import Path

from carnot.reporting import memory_size_cost_8848 as m


def test_empty_poll_keeps_actual_child_failure(tmp_path: Path, monkeypatch, capsys):
    """SCENARIO-REPORT-8848-CONTRACT: an empty poll reports elapsed work, never success."""
    original_select = m.selectors.DefaultSelector.select
    calls = 0

    def select(selector, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return []
        return original_select(selector, timeout)

    monkeypatch.setattr(m.selectors.DefaultSelector, "select", select)
    result = m.process(tmp_path / "absent-request.json", 0, 60)
    assert calls > 1 and result["status"] == "failed" and not result["timed_out"]
    assert "phase=child_elapsed_" in capsys.readouterr().out
    assert Path(result["log"]["path"]).is_file()
