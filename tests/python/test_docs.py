"""Tests for the documentation UI and public reporting copy.

REQ-DOCUI-001: Premium Aesthetic
REQ-DOCUI-002: Interactive Micro-animations
REQ-REPORT-003: README Provenance Disclosure
REQ-REPORT-004: Report and Landing-Page Disclosure
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_repo_file(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _current_provenance_counts() -> dict[str, int]:
    counts = {
        "live_gpu": 0,
        "simulated": 0,
        "unverified": 0,
        "software_simulation": 0,
    }
    for path in (REPO_ROOT / "results").glob("experiment_*_results.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        statistics = payload.get("statistics", {})
        mode = (
            payload.get("inference_mode")
            or metadata.get("inference_mode")
            or statistics.get("inference_mode")
        )
        if mode == "live_gpu":
            counts["live_gpu"] += 1
        elif mode in {"simulated", "simulation"}:
            counts["simulated"] += 1
        elif mode == "software_simulation":
            counts["software_simulation"] += 1
        else:
            counts["unverified"] += 1
    return counts


def _current_experiment_count() -> int:
    status = _read_repo_file("ops/status.md")
    match = re.search(r"\*\*Last Updated:\*\*.*?—\s+(\d+)\s+EXPERIMENTS", status)
    assert match is not None
    return int(match.group(1))


def _current_milestone_count() -> int:
    payload = yaml.safe_load(_read_repo_file("research-complete.yaml"))
    return len(payload["milestones"])


def test_docs_have_premium_aesthetic():
    """Verify the landing page keeps the required premium UI affordances.

    REQ-DOCUI-001, REQ-DOCUI-002
    """
    docs_path = REPO_ROOT / "docs" / "index.html"
    assert docs_path.exists()
    content = docs_path.read_text(encoding="utf-8")

    # REQ-DOCUI-001
    assert "backdrop-filter: blur" in content, "Missing glassmorphism"
    assert "rgba(255" in content, "Missing soft borders or shadows"

    # REQ-DOCUI-002
    assert "fade-in-up" in content, "Missing micro-interactions"
    assert "@keyframes fadeInUp" in content, "Missing animations"


def test_public_docs_disclose_current_provenance_inventory() -> None:
    """REQ-REPORT-003, REQ-REPORT-004: public docs expose current provenance counts."""
    counts = _current_provenance_counts()
    expected_snippets = [
        f"{counts['live_gpu']} live GPU artifacts",
        f"{counts['simulated']} simulated artifacts",
        f"{counts['unverified']} unverified artifacts",
        f"{counts['software_simulation']} software-model artifact",
    ]

    readme = _read_repo_file("README.md")
    report = _read_repo_file("docs/technical-report.md")
    index = _read_repo_file("docs/index.html")

    for snippet in expected_snippets:
        assert snippet in readme
        assert snippet in report
        assert snippet in index


def test_public_docs_cover_latest_pbt_and_fpga_reporting() -> None:
    """REQ-REPORT-003, REQ-REPORT-004: docs mention the latest PBT and FPGA updates."""
    exp_count = _current_experiment_count()
    milestone_count = _current_milestone_count()
    readme = _read_repo_file("README.md")
    report = _read_repo_file("docs/technical-report.md")
    report_html = _read_repo_file("docs/technical-report.html")
    index = _read_repo_file("docs/index.html")

    assert "## PBT Verification" in readme
    assert "Exp 226" in readme
    assert "Exp 227" in readme
    assert "[FPGA Ising design](docs/fpga-ising-design.md)" in readme
    assert "software simulation" in readme

    assert "Property-Based Code Verification at Scale" in report
    assert "Exp 227" in report
    assert "Experiment 228" in report
    assert "software simulation" in report

    assert f"{exp_count} completed experiments" in readme
    assert f"{milestone_count} completed milestones" in readme
    assert f"{exp_count} completed experiments" in index
    assert f"{milestone_count} completed milestones" in index
    assert (
        f"{exp_count} Completed Experiments Across {milestone_count} Research Milestones" in report
    )
    assert (
        f"{exp_count} Completed Experiments Across {milestone_count} Research Milestones"
        in report_html
    )

    assert "Exp 227" in index
    assert "software-model" in index

    assert "VERIFY-030" in report
    assert "VERIFY-031" in report
    assert "verify_code_with_pbt" in readme
    assert "Experiment 228" in report_html
