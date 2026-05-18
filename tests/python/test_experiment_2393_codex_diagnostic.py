"""Tests for the exp2393 codex-CLI diagnostic.

We exercise the pure-Python building blocks with synthetic inputs so the test
suite never spawns a real codex subprocess. The integration end of the module
(`run_minimal_codex_test`, `main`) is exercised indirectly via the artifact
write path with codex stubbed out.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

# Load the experiment module directly from its scripts/ path. scripts/ is not a
# proper Python package, so importing by file path keeps the test independent
# of sys.path tweaks elsewhere in the repo.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "experiment_2393_codex_diagnostic.py"
)
_spec = importlib.util.spec_from_file_location("exp2393_codex_diag", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
diag = importlib.util.module_from_spec(_spec)
sys.modules["exp2393_codex_diag"] = diag
_spec.loader.exec_module(diag)


def test_needle_is_exactly_sixty_chars():
    """Width invariant: the conductor truncates at output[:60]."""
    assert len(diag.EXPECTED_ECHO_NEEDLE) == 60


def test_classify_failure_signature_matches_postamble_tail():
    """The classifier flags a tail that contains the truncation needle."""
    tail = (
        "preamble noise\n"
        "If you finish the real work inside 10 minutes, that is correct and "
        "expected -- exit promptly."
    )
    result = diag.classify_failure_signature(tail)
    assert result["matched_echo"] is True
    assert "transient_backend_error" in result["hypothesis"]
    assert "non-zero" in result["explanation"]


def test_classify_failure_signature_misses_unrelated_tail():
    """Random task output that does not contain the postamble is not flagged."""
    tail = "tokens used\n42,000\nDone: results/experiment_999.json written."
    result = diag.classify_failure_signature(tail)
    assert result["matched_echo"] is False
    assert result["hypothesis"] == "non_echo_failure"


def test_find_failure_window_extracts_first_and_last_timestamps():
    """Window detection picks the earliest and latest matching log lines."""
    lines = [
        "| 2026-05-18 10:42 UTC | OK row | OK | passing",
        (
            "| 2026-05-18 10:44 UTC | Phase 6 | FAIL | Codex CLI error: "
            + diag.EXPECTED_ECHO_NEEDLE
        ),
        "| 2026-05-18 10:46 UTC | unrelated | OK | passing",
        (
            "| 2026-05-18 14:03 UTC | Phase 3 | FAIL | Codex CLI error: "
            + diag.EXPECTED_ECHO_NEEDLE
        ),
    ]
    first, last = diag.find_failure_window(iter(lines))
    assert first == "2026-05-18 10:44 UTC"
    assert last == "2026-05-18 14:03 UTC"


def test_find_failure_window_returns_none_when_no_match():
    """No matching rows -> no window."""
    lines = ["| 2026-05-18 10:42 UTC | OK row | OK | passing"]
    first, last = diag.find_failure_window(iter(lines))
    assert first is None and last is None


def test_probe_codex_version_returns_none_when_missing(monkeypatch):
    """Absent binary -> None, no exception."""
    monkeypatch.setattr(diag.shutil, "which", lambda _bin: None)
    assert diag.probe_codex_version("does-not-exist") is None


def test_probe_codex_version_extracts_first_line(monkeypatch):
    """When the binary exists, return its first stdout line."""
    monkeypatch.setattr(diag.shutil, "which", lambda _bin: "/usr/bin/codex")

    class _FakeCompleted:
        stdout = "codex-cli 0.130.0\n"
        stderr = ""

    def _fake_run(*_args, **_kwargs):
        return _FakeCompleted()

    monkeypatch.setattr(diag.subprocess, "run", _fake_run)
    assert diag.probe_codex_version() == "codex-cli 0.130.0"


def test_probe_codex_version_handles_timeout(monkeypatch):
    """Timeout / OSError from the version probe degrade to None."""
    monkeypatch.setattr(diag.shutil, "which", lambda _bin: "/usr/bin/codex")

    def _raise(*_args, **_kwargs):
        raise diag.subprocess.TimeoutExpired(cmd="codex", timeout=10)

    monkeypatch.setattr(diag.subprocess, "run", _raise)
    assert diag.probe_codex_version() is None


def test_run_minimal_codex_test_returns_none_when_missing(monkeypatch):
    """Absent binary path returns the sentinel (None, [])."""
    monkeypatch.setattr(diag.shutil, "which", lambda _bin: None)
    exit_code, lines = diag.run_minimal_codex_test("not-a-real-bin")
    assert exit_code is None and lines == []


def test_run_minimal_codex_test_returns_exit_and_lines(monkeypatch):
    """Healthy codex returns (0, stdout split into lines, capped at 5)."""
    monkeypatch.setattr(diag.shutil, "which", lambda _bin: "/usr/bin/codex")

    class _FakeCompleted:
        returncode = 0
        stdout = "line1\nline2\nline3\nline4\nline5\nline6_dropped\n"

    monkeypatch.setattr(
        diag.subprocess, "run", lambda *a, **kw: _FakeCompleted()
    )
    exit_code, lines = diag.run_minimal_codex_test()
    assert exit_code == 0
    assert lines == ["line1", "line2", "line3", "line4", "line5"]


def test_run_minimal_codex_test_handles_timeout(monkeypatch):
    """Timeout from the live invocation reports (None, []) rather than raising."""
    monkeypatch.setattr(diag.shutil, "which", lambda _bin: "/usr/bin/codex")

    def _raise(*_a, **_kw):
        raise diag.subprocess.TimeoutExpired(cmd="codex", timeout=120)

    monkeypatch.setattr(diag.subprocess, "run", _raise)
    exit_code, lines = diag.run_minimal_codex_test()
    assert exit_code is None and lines == []


def test_build_artifact_healthy_path(tmp_path):
    """All schema fields present; verdict carries terminal prefix; repaired=True."""
    log = tmp_path / "conductor-log.md"
    log.write_text(
        "| 2026-05-18 10:44 UTC | Phase 6 | FAIL | Codex CLI error: "
        + diag.EXPECTED_ECHO_NEEDLE
        + "\n"
        "| 2026-05-18 14:03 UTC | Phase 3 | FAIL | Codex CLI error: "
        + diag.EXPECTED_ECHO_NEEDLE
        + "\n",
        encoding="utf-8",
    )
    artifact = diag.build_artifact(
        codex_version="codex-cli 0.130.0",
        minimal_test_exit=0,
        minimal_test_stdout_first_lines=["alive"],
        conductor_log_path=log,
        duration_s=3.14,
        started_at="2026-05-18T15:18:23Z",
    )
    required = {
        "honest_verdict",
        "infrastructure_repaired",
        "codex_cli_version",
        "root_cause_diagnosed",
        "root_cause_summary",
        "repair_action_taken",
        "minimal_codex_test_result",
        "preconditions_checked",
        "duration_s",
    }
    assert required.issubset(artifact.keys())
    assert artifact["honest_verdict"].startswith("complete: ")
    assert artifact["infrastructure_repaired"] is True
    assert artifact["codex_cli_version"] == "codex-cli 0.130.0"
    assert artifact["observed_failure_window"]["first_failure_utc"] == "2026-05-18 10:44 UTC"
    assert artifact["observed_failure_window"]["last_failure_utc"] == "2026-05-18 14:03 UTC"


def test_build_artifact_blocked_when_codex_missing(tmp_path):
    """No codex available -> verdict uses the blocked_ prefix, repaired=False."""
    artifact = diag.build_artifact(
        codex_version=None,
        minimal_test_exit=None,
        minimal_test_stdout_first_lines=[],
        conductor_log_path=None,
        duration_s=0.5,
        started_at="2026-05-18T15:18:23Z",
    )
    assert artifact["infrastructure_repaired"] is False
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["codex_cli_version"] is None
    assert artifact["preconditions_checked"][0]["available"] is False


def test_main_writes_a_well_formed_artifact(tmp_path, monkeypatch):
    """End-to-end: main() runs with codex stubbed and emits a parseable JSON."""
    monkeypatch.setattr(diag, "probe_codex_version", lambda: "codex-cli 0.130.0")
    monkeypatch.setattr(
        diag, "run_minimal_codex_test", lambda: (0, ["alive"])
    )
    deliverable = tmp_path / "results" / "experiment_2393_codex_diagnostic.json"
    fake_root = tmp_path
    (fake_root / "ops").mkdir(parents=True)
    (fake_root / "ops" / "conductor-log.md").write_text("", encoding="utf-8")
    artifact = diag.main(project_root=fake_root, deliverable=deliverable)
    assert deliverable.exists()
    on_disk = json.loads(deliverable.read_text(encoding="utf-8"))
    assert on_disk == artifact
    assert on_disk["honest_verdict"].startswith("complete: ")


def test_main_default_module_entrypoint_does_not_error(tmp_path, monkeypatch):
    """Calling main() at the module level (with stubs) is non-destructive."""
    monkeypatch.setattr(diag, "probe_codex_version", lambda: None)
    monkeypatch.setattr(diag, "run_minimal_codex_test", lambda: (None, []))
    deliverable = tmp_path / "results" / "experiment_2393_codex_diagnostic.json"
    artifact = diag.main(project_root=tmp_path, deliverable=deliverable)
    assert artifact["infrastructure_repaired"] is False
