"""Shared focused pytest smoke gate for ARC solve preconditions.

Spec refs: REQ-REPORT-4475, SCENARIO-REPORT-4475-SMOKE.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TIMEOUT_S = 120
DEFAULT_SUMMARY_CHARS = 2000


def build_pytest_command(selector: str, *, root: Path | str = REPO_ROOT) -> list[str]:
    """REQ-REPORT-4475: build the no-coverage focused pytest smoke command."""

    repo = Path(root)
    return [
        str(repo / ".venv" / "bin" / "pytest"),
        "-k",
        str(selector),
        "-q",
        "--no-cov",
    ]


def arc_precondition_smoke(
    selector: str,
    *,
    root: Path | str = REPO_ROOT,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    summary_chars: int = DEFAULT_SUMMARY_CHARS,
) -> tuple[bool, str]:
    """Run a focused pytest smoke gate and return ``(green, summary)``."""

    repo = Path(root)
    command = build_pytest_command(selector, root=repo)
    try:
        completed = subprocess.run(
            command,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - exact subprocess failures are platform-bound.
        return False, f"{type(exc).__name__}: {exc}"[-max(1, int(summary_chars)) :]
    summary = str(completed.stdout or "")[-max(1, int(summary_chars)) :]
    return completed.returncode == 0, summary
