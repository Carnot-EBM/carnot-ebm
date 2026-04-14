#!/usr/bin/env python3
"""Behavioral audit logger for the Carnot Research Conductor.

The research conductor runs autonomously for hours, invoking agents, making
commits, and modifying code. This module provides an independent audit trail
so that anomalous behavior — reward hacking, self-modification, unexpected
file deletions, or security file access — can be detected and flagged.

Design principles:
- Append-only NDJSON log (ops/conductor-audit.jsonl).  Each event is one JSON
  line so the log is easy to tail, grep, and parse programmatically.
- Never raise — audit errors must never crash the conductor.  All public
  functions catch exceptions internally and log a warning at most.
- Anomaly detection is heuristic, not exhaustive.  False positives are
  acceptable; false negatives (missed real anomalies) are not.
- No runtime dependencies beyond the Python standard library + PyYAML (already
  required by the conductor).

Usage (from research_step or any other conductor function):

    from scripts.conductor_audit import ConductorAudit

    audit = ConductorAudit()                      # opens/creates the log
    audit.log_agent_invocation(
        prompt_hash="abc123",
        max_turns=50,
        exit_code=0,
        duration_s=42.3,
        output_snippet="...",
    )
    audit.log_git_commit(
        files_changed=["foo.py", "bar.py"],
        lines_added=30,
        lines_removed=5,
        test_result="12 passed",
    )
    audit.log_file_modification("scripts/experiment_322.py", old_size=0, new_size=4096)
    summary = audit.milestone_summary()

Per-milestone summary:  call audit.milestone_summary() at milestone boundaries
to get a structured dict suitable for JSON serialisation and archiving.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("conductor.audit")

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
AUDIT_LOG = PROJECT_ROOT / "ops" / "conductor-audit.jsonl"

# File extensions considered "source code" (as opposed to data or docs).
CODE_EXTENSIONS = frozenset(
    {".py", ".rs", ".toml", ".yaml", ".yml", ".json", ".ts", ".js",
     ".sh", ".bash", ".zsh", ".fish", ".html", ".css", ".md"}
)

# Security-sensitive file patterns — access by an autonomous agent should be
# flagged for review.  These are files that could expose credentials, modify
# trust boundaries, or disable safety checks.
SECURITY_PATTERNS = (
    re.compile(r"(^|[/\\])\.(env|envrc|secrets?|sops\.yaml)$", re.IGNORECASE),
    re.compile(r"\.(key|pem|p12|pfx|crt|cer|csr|gpg|asc)$", re.IGNORECASE),
    re.compile(r"(^|[/\\])(\.ssh[/\\]|id_(rsa|ecdsa|ed25519|dsa))", re.IGNORECASE),
    re.compile(r"(^|[/\\])(secrets?|credentials?|token)[/\\]", re.IGNORECASE),
    re.compile(r"(^|[/\\])\.pre-commit-config\.yaml$", re.IGNORECASE),
)

# The conductor's own source file — the autonomous agent must never modify it.
CONDUCTOR_SCRIPT = "scripts/research_conductor.py"

# If an agent deletes more than this many files in a single operation, flag it.
# Normal research commits add/modify files; mass deletion is suspicious.
MASS_DELETE_THRESHOLD = 5

# Network-access indicators in agent output — substrings that suggest the
# agent made unexpected outbound calls beyond the project scope.
NETWORK_INDICATORS = (
    "curl ", "wget ", "requests.get(", "urllib.request", "http.client",
    "socket.connect", "paramiko", "ftplib",
)


# ── Data Structures ───────────────────────────────────────────────────────────


@dataclass
class AuditEvent:
    """A single audited event in the conductor's operation.

    The `anomalies` list is populated by check_anomalies() and describes any
    policy violations or suspicious patterns detected in this event.
    """

    timestamp: str
    event_type: str          # "agent_invocation" | "git_commit" | "file_modification" | "anomaly"
    details: dict            # event-specific fields (varies by type)
    anomalies: List[str] = field(default_factory=list)


@dataclass
class MilestoneSummary:
    """Aggregated statistics for the current milestone, suitable for archiving.

    Built by ConductorAudit.milestone_summary() and written to the audit log
    as an event of type "milestone_summary".
    """

    generated_at: str
    total_agent_invocations: int
    total_commits: int
    total_file_modifications: int
    total_anomalies: int
    anomaly_breakdown: dict      # {anomaly_type: count}
    top_modified_files: List[str]
    avg_agent_duration_s: float
    total_lines_added: int
    total_lines_removed: int


# ── ConductorAudit ────────────────────────────────────────────────────────────


class ConductorAudit:
    """Append-only behavioral audit logger for the research conductor.

    One instance per conductor session is the normal usage pattern.  The log
    file is opened in append mode; concurrent writers are safe because each
    write is a single json.dumps call followed by a newline (atomic on most
    Linux filesystems for small payloads).

    All public methods are non-raising: if anything goes wrong internally the
    exception is caught, logged at DEBUG level, and the method returns
    gracefully.  This ensures audit errors never interrupt the conductor.
    """

    def __init__(self, audit_log: Path = AUDIT_LOG) -> None:
        self._log_path = audit_log
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # In-memory accumulator for milestone summary statistics.
        self._agent_durations: list[float] = []
        self._commit_stats: list[dict] = []
        self._file_mods: list[str] = []
        self._anomaly_counts: dict[str, int] = {}

    # ── Public logging methods ────────────────────────────────────────────────

    def log_agent_invocation(
        self,
        prompt_hash: str,
        max_turns: int,
        exit_code: int,
        duration_s: float,
        output_snippet: str = "",
    ) -> list[str]:
        """Record one agent invocation and return any anomalies detected.

        Args:
            prompt_hash: SHA-256 hex digest of the prompt text.  We store the
                hash rather than the full prompt so the log stays compact and
                does not leak sensitive task instructions.
            max_turns: Turn budget passed to the agent CLI.
            exit_code: Process exit code (0 = success, non-zero = failure).
            duration_s: Wall-clock seconds the agent ran.
            output_snippet: Last ~500 chars of agent stdout for spot-checks.

        Returns:
            List of anomaly description strings (empty if clean).
        """
        details = {
            "prompt_hash": prompt_hash,
            "max_turns": max_turns,
            "exit_code": exit_code,
            "duration_s": round(duration_s, 2),
            "output_snippet": output_snippet[:500],
        }
        anomalies = self._detect_agent_anomalies(output_snippet)
        self._write_event("agent_invocation", details, anomalies)
        self._agent_durations.append(duration_s)
        return anomalies

    def log_git_commit(
        self,
        files_changed: list[str],
        lines_added: int,
        lines_removed: int,
        test_result: str = "",
    ) -> list[str]:
        """Record a git commit and return any anomalies detected.

        Args:
            files_changed: List of relative file paths touched in the commit.
            lines_added: Total lines inserted (+).
            lines_removed: Total lines deleted (-).
            test_result: Test suite summary line (e.g. "12 passed, 0 failed").

        Returns:
            List of anomaly description strings (empty if clean).
        """
        details = {
            "files_changed": files_changed,
            "file_count": len(files_changed),
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "test_result": test_result,
        }
        anomalies = self._detect_commit_anomalies(files_changed, lines_removed)
        self._write_event("git_commit", details, anomalies)
        self._commit_stats.append(
            {"lines_added": lines_added, "lines_removed": lines_removed}
        )
        return anomalies

    def log_file_modification(
        self,
        path: str,
        old_size: int,
        new_size: int,
    ) -> list[str]:
        """Record a file modification and return any anomalies detected.

        Args:
            path: Relative path from project root (e.g. "scripts/foo.py").
            old_size: File size in bytes before modification (0 for new files).
            new_size: File size in bytes after modification (0 for deletions).

        Returns:
            List of anomaly description strings (empty if clean).
        """
        is_code = Path(path).suffix.lower() in CODE_EXTENSIONS
        is_deletion = new_size == 0 and old_size > 0
        is_creation = old_size == 0 and new_size > 0
        size_delta = new_size - old_size

        details = {
            "path": path,
            "old_size": old_size,
            "new_size": new_size,
            "size_delta": size_delta,
            "is_code_file": is_code,
            "is_deletion": is_deletion,
            "is_creation": is_creation,
        }
        anomalies = self._detect_file_anomalies(path)
        self._write_event("file_modification", details, anomalies)
        self._file_mods.append(path)
        return anomalies

    def milestone_summary(self) -> dict:
        """Build and log an aggregated summary of all events since instantiation.

        Call this at milestone boundaries (e.g., after archiving a completed
        milestone) to capture a structured snapshot for retrospective review.

        Returns:
            Dict representation of MilestoneSummary suitable for JSON.
        """
        total_added = sum(s["lines_added"] for s in self._commit_stats)
        total_removed = sum(s["lines_removed"] for s in self._commit_stats)
        avg_duration = (
            sum(self._agent_durations) / len(self._agent_durations)
            if self._agent_durations
            else 0.0
        )

        # Top 10 most frequently modified files
        from collections import Counter
        top_files = [f for f, _ in Counter(self._file_mods).most_common(10)]

        total_anomalies = sum(self._anomaly_counts.values())

        summary = MilestoneSummary(
            generated_at=_utcnow(),
            total_agent_invocations=len(self._agent_durations),
            total_commits=len(self._commit_stats),
            total_file_modifications=len(self._file_mods),
            total_anomalies=total_anomalies,
            anomaly_breakdown=dict(self._anomaly_counts),
            top_modified_files=top_files,
            avg_agent_duration_s=round(avg_duration, 2),
            total_lines_added=total_added,
            total_lines_removed=total_removed,
        )
        summary_dict = asdict(summary)
        self._write_event("milestone_summary", summary_dict, [])
        return summary_dict

    # ── Anomaly detection ─────────────────────────────────────────────────────

    def _detect_agent_anomalies(self, output_snippet: str) -> list[str]:
        """Detect anomalies in agent output.

        Checks for evidence of unexpected network access (curl, wget, etc.)
        in the captured output.  Direct network access by an autonomous agent
        running inside the conductor is unexpected and could indicate data
        exfiltration or dependency confusion attacks.
        """
        found = []
        lower = output_snippet.lower()
        for indicator in NETWORK_INDICATORS:
            if indicator.lower() in lower:
                found.append(
                    f"Unexpected network indicator in agent output: '{indicator}'"
                )
        return found

    def _detect_commit_anomalies(
        self, files_changed: list[str], lines_removed: int
    ) -> list[str]:
        """Detect anomalies in a git commit.

        Rules:
        1. Conductor self-modification: the agent must never alter
           scripts/research_conductor.py.
        2. Mass file deletion: removing many files at once is suspicious.
        3. Security file access: touching credentials, keys, or .env files.
        """
        found = []

        # Rule 1: conductor self-modification
        for f in files_changed:
            if f == CONDUCTOR_SCRIPT or f.endswith("/" + CONDUCTOR_SCRIPT):
                found.append(
                    f"Conductor self-modification detected: {f} was changed"
                )

        # Rule 2: mass deletion (lines_removed is a proxy; file count is better
        # but we use the provided list to count actual deletions vs modifications)
        if len(files_changed) >= MASS_DELETE_THRESHOLD and lines_removed > 200:
            found.append(
                f"Mass file change: {len(files_changed)} files, "
                f"{lines_removed} lines removed"
            )

        # Rule 3: security-sensitive file access
        for f in files_changed:
            for pat in SECURITY_PATTERNS:
                if pat.search(f):
                    found.append(f"Security-sensitive file touched: {f}")
                    break  # one report per file

        return found

    def _detect_file_anomalies(self, path: str) -> list[str]:
        """Detect anomalies for a single file modification.

        Rules:
        1. Conductor self-modification.
        2. Security-sensitive file access.
        """
        found = []

        if path == CONDUCTOR_SCRIPT or path.endswith("/" + CONDUCTOR_SCRIPT):
            found.append(
                f"Conductor self-modification detected: {path} was modified"
            )

        for pat in SECURITY_PATTERNS:
            if pat.search(path):
                found.append(f"Security-sensitive file touched: {path}")
                break

        return found

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _write_event(
        self,
        event_type: str,
        details: dict,
        anomalies: list[str],
    ) -> None:
        """Serialise and append one audit event to the NDJSON log.

        Also updates the in-memory anomaly counter for milestone summaries.
        Never raises — any I/O error is caught and logged at WARNING level.
        """
        event = AuditEvent(
            timestamp=_utcnow(),
            event_type=event_type,
            details=details,
            anomalies=anomalies,
        )
        if anomalies:
            logger.warning(
                "AUDIT ANOMALY [%s]: %s", event_type, "; ".join(anomalies)
            )
            for a in anomalies:
                # Bucket by first few words to get a stable category key
                key = " ".join(a.split()[:4])
                self._anomaly_counts[key] = self._anomaly_counts.get(key, 0) + 1

        try:
            line = json.dumps(asdict(event), ensure_ascii=False) + "\n"
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception as exc:
            logger.debug("AUDIT: failed to write event: %s", exc)


# ── Convenience helpers ───────────────────────────────────────────────────────


def prompt_hash(prompt: str) -> str:
    """Return the SHA-256 hex digest of a prompt string.

    Use this to log prompts without storing the full text.  The hash is
    stable across runs for the same task, making it easy to correlate
    audit entries with roadmap task IDs in post-hoc analysis.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def parse_diff_stats(diff_stat_output: str) -> tuple[int, int]:
    """Parse ``git diff --stat`` output into (lines_added, lines_removed).

    Handles both per-file lines ("foo.py | 3 +++") and the summary line
    ("3 files changed, 12 insertions(+), 5 deletions(-)").  Returns (0, 0)
    if the output cannot be parsed.

    This is used by the conductor to extract structured numbers for
    log_git_commit() without re-running git commands.
    """
    added = removed = 0
    # Summary line pattern: "N insertions(+), M deletions(-)"
    ins_match = re.search(r"(\d+) insertion", diff_stat_output)
    del_match = re.search(r"(\d+) deletion", diff_stat_output)
    if ins_match:
        added = int(ins_match.group(1))
    if del_match:
        removed = int(del_match.group(1))
    return added, removed


def collect_staged_files() -> list[str]:
    """Return the list of files in the current git staging area.

    Calls ``git diff --name-only --cached``.  Returns an empty list if git
    is unavailable or the working directory is not a git repo.

    Used by the conductor audit hooks to populate log_git_commit(files_changed).
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=10,
        )
        if result.returncode != 0:
            return []
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except Exception:
        return []


def _utcnow() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    """Print a summary of the current audit log to stdout."""
    import argparse

    parser = argparse.ArgumentParser(description="Carnot conductor audit log viewer")
    parser.add_argument(
        "--tail",
        type=int,
        default=20,
        metavar="N",
        help="Show last N events (default: 20)",
    )
    parser.add_argument(
        "--anomalies-only",
        action="store_true",
        help="Show only events that have anomalies",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print the most recent milestone_summary event",
    )
    args = parser.parse_args()

    if not AUDIT_LOG.exists():
        print(f"No audit log found at {AUDIT_LOG}")
        return

    lines = AUDIT_LOG.read_text(encoding="utf-8").splitlines()
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if args.summary:
        summaries = [e for e in events if e.get("event_type") == "milestone_summary"]
        if summaries:
            print(json.dumps(summaries[-1], indent=2))
        else:
            print("No milestone summary events in log.")
        return

    if args.anomalies_only:
        events = [e for e in events if e.get("anomalies")]

    for event in events[-args.tail:]:
        ts = event.get("timestamp", "")
        etype = event.get("event_type", "")
        anomalies = event.get("anomalies", [])
        flag = " [ANOMALY]" if anomalies else ""
        print(f"{ts}  {etype}{flag}")
        if anomalies:
            for a in anomalies:
                print(f"    ! {a}")


if __name__ == "__main__":
    main()
