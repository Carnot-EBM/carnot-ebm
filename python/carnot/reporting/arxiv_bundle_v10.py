"""Build the Exp 1270 gated arXiv bundle-v10 artifact.

This module is intentionally narrow because arXiv packaging is an operational
gate, not a research experiment. It records exactly what local tooling was
available, runs one compile/package path, and refuses to invent either a PDF or
an arXiv submission event when the machine cannot produce one.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_ARXIV_DIR = REPO_ROOT / "docs" / "arxiv-paper"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1270_arxiv_bundle_v10_gated.json"
DEFAULT_BUNDLE_PATH = DEFAULT_RESULTS_DIR / "carnot-arxiv-v10-20260504.tar.gz"

EXPERIMENT = "1270_arxiv_bundle_v10_gated"
SCHEMA = "arxiv_bundle_v10_gated_v1"
RUN_DATE = "20260504"
GATE_FILENAME = "experiment_1269_paper_v6_critical_fixes_v2.json"
COMMAND_NAMES = ("tectonic", "latexmk", "make")
MAKEFILE_NAMES = ("Makefile", "makefile", "GNUmakefile")
SUBMISSION_RECEIPT_GLOB = "arxiv_submission_receipt*.json"

CommandRunner = Callable[[list[str], Path, int], Any]
Which = Callable[[str], str | None]


def _relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """Write the durable placeholder required before any build attempt.

    The conductor can be interrupted at any point. A concrete in-progress JSON
    makes the interruption auditable instead of leaving a missing deliverable
    that looks identical to "the task was never started."
    """

    return _write_json(
        Path(out_path),
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "status": "in_progress",
            "pdf_compiled": False,
            "bundle_path": None,
            "arxiv_submitted": False,
            "honest_verdict": "in_progress",
        },
    )


def default_command_runner(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run one local build command and capture logs for the artifact.

    The caller chooses the command after tool discovery; this helper only
    performs the subprocess call with bounded runtime and captured output so
    failures can be reported honestly in JSON.
    """

    return subprocess.run(
        cmd,
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )


def load_critical_gate(results_dir: Path | str = DEFAULT_RESULTS_DIR) -> dict[str, Any]:
    """Load Exp 1269, the prerequisite that proves paper-critical fixes landed."""

    path = Path(results_dir) / GATE_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def critical_gate_satisfied(gate: dict[str, Any]) -> bool:
    """Return whether the paper may proceed to bundling.

    The threshold is intentionally numeric instead of verdict-string-only so
    equivalent future exp1269 artifacts still satisfy the gate when they record
    at least five critical issue fixes.
    """

    fixed = gate.get("critical_issues_fixed", 0)
    return isinstance(fixed, int | float) and fixed >= 5


def discover_make_targets(arxiv_dir: Path | str = DEFAULT_ARXIV_DIR) -> list[str]:
    """Read Makefile target names from the paper directory, if a Makefile exists."""

    arxiv_path = Path(arxiv_dir)
    for name in MAKEFILE_NAMES:
        makefile = arxiv_path / name
        if makefile.exists():
            targets: list[str] = []
            for line in makefile.read_text(encoding="utf-8").splitlines():
                if line.startswith(("\t", " ")) or ":" not in line:
                    continue
                target = line.split(":", 1)[0].strip()
                if target and not target.startswith("."):
                    targets.append(target)
            return targets
    return []


def _choose_make_command(make_targets: list[str]) -> list[str]:
    for target in ("package", "bundle", "pdf", "all"):
        if target in make_targets:
            return ["make", target]
    return []


def discover_build_command(
    arxiv_dir: Path | str = DEFAULT_ARXIV_DIR,
    *,
    which: Which = shutil.which,
) -> tuple[list[str], dict[str, bool], list[str]]:
    """Pick the narrowest local compile/package command in spec order."""

    arxiv_path = Path(arxiv_dir)
    tools = {name: bool(which(name)) for name in COMMAND_NAMES}
    make_targets = discover_make_targets(arxiv_path)
    if tools["tectonic"]:
        return ["tectonic", "main.tex"], tools, make_targets
    if tools["latexmk"]:
        return ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], tools, make_targets
    if tools["make"]:
        make_command = _choose_make_command(make_targets)
        if make_command:
            return make_command, tools, make_targets
    return [], tools, make_targets


def local_submission_receipt_exists(results_dir: Path | str = DEFAULT_RESULTS_DIR) -> bool:
    """Check for a checked-in local receipt before ever setting arxiv_submitted."""

    return any(path.is_file() for path in Path(results_dir).glob(SUBMISSION_RECEIPT_GLOB))


def build_bundle(
    *,
    project_root: Path | str = REPO_ROOT,
    arxiv_dir: Path | str = DEFAULT_ARXIV_DIR,
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
) -> str:
    """Create the source tarball arXiv expects from the paper directory.

    The tarball stores paper files at the archive root because arXiv unpacks a
    submitted archive and runs TeX from that root. Existing historical tarballs
    in the directory are intentionally ignored to avoid recursive bundles.
    """

    root = Path(project_root)
    arxiv_path = Path(arxiv_dir)
    target = Path(bundle_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz") as tf:
        for rel in ("main.tex", "main.pdf", "carnot.bib", "README_ARXIV.txt"):
            src = arxiv_path / rel
            if src.exists():
                tf.add(src, arcname=rel)
        figures_dir = arxiv_path / "figures"
        if figures_dir.exists():
            for src in sorted(figures_dir.iterdir()):
                if src.suffix.lower() in {".pdf", ".png"}:
                    tf.add(src, arcname=f"figures/{src.name}")
    return _relative_path(target, root)


def _tail(text: object, limit: int = 1200) -> str:
    value = str(text or "")
    return value[-limit:]


def _blocked_artifact(
    *,
    status_reason: str,
    gate: dict[str, Any],
    arxiv_dir: Path,
    results_dir: Path,
    project_root: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "status": "blocked",
        "critical_issues_fixed": gate.get("critical_issues_fixed", 0),
        "source_dir": _relative_path(arxiv_dir, project_root),
        "pdf_path": _relative_path(arxiv_dir / "main.pdf", project_root),
        "pdf_compiled": False,
        "bundle_path": None,
        "arxiv_submitted": local_submission_receipt_exists(results_dir),
        "honest_verdict": status_reason,
    }
    if extra:
        artifact.update(extra)
    return artifact


def run(
    *,
    project_root: Path | str = REPO_ROOT,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    arxiv_dir: Path | str = DEFAULT_ARXIV_DIR,
    out_path: Path | str = DEFAULT_OUT_PATH,
    bundle_path: Path | str | None = None,
    which: Which = shutil.which,
    command_runner: CommandRunner = default_command_runner,
    timeout: int = 300,
) -> dict[str, Any]:
    """Run the gated bundle workflow and write the final Exp 1270 artifact."""

    root = Path(project_root)
    results_path = Path(results_dir)
    arxiv_path = Path(arxiv_dir)
    output = Path(out_path)
    bundle_target = Path(bundle_path) if bundle_path is not None else root / "results" / DEFAULT_BUNDLE_PATH.name
    write_in_progress_artifact(output)

    gate = load_critical_gate(results_path)
    if not critical_gate_satisfied(gate):
        return _write_json(
            output,
            _blocked_artifact(
                status_reason="blocked_exp1269_gate_not_satisfied",
                gate=gate,
                arxiv_dir=arxiv_path,
                results_dir=results_path,
                project_root=root,
            ),
        )

    command, tools, make_targets = discover_build_command(arxiv_path, which=which)
    if not command:
        missing = [name for name in COMMAND_NAMES if not tools[name]]
        return _write_json(
            output,
            _blocked_artifact(
                status_reason="blocked_missing_local_tex_tooling",
                gate=gate,
                arxiv_dir=arxiv_path,
                results_dir=results_path,
                project_root=root,
                extra={
                    "available_tools": tools,
                    "make_targets": make_targets,
                    "missing_tool": missing,
                },
            ),
        )

    result = command_runner(command, arxiv_path, timeout)
    pdf_path = arxiv_path / "main.pdf"
    pdf_compiled = result.returncode == 0 and pdf_path.exists()
    compile_fields = {
        "available_tools": tools,
        "make_targets": make_targets,
        "compile_command": command,
        "compile_returncode": result.returncode,
        "compile_stdout_tail": _tail(result.stdout),
        "compile_stderr_tail": _tail(result.stderr),
    }
    if not pdf_compiled:
        return _write_json(
            output,
            _blocked_artifact(
                status_reason="blocked_compile_failed",
                gate=gate,
                arxiv_dir=arxiv_path,
                results_dir=results_path,
                project_root=root,
                extra=compile_fields,
            ),
        )

    bundle_rel = build_bundle(project_root=root, arxiv_dir=arxiv_path, bundle_path=bundle_target)
    bundle_abs = root / bundle_rel
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "status": "complete",
        "critical_issues_fixed": gate.get("critical_issues_fixed", 0),
        "source_dir": _relative_path(arxiv_path, root),
        "pdf_path": _relative_path(pdf_path, root),
        "pdf_compiled": True,
        "bundle_path": bundle_rel,
        "bundle_size_bytes": bundle_abs.stat().st_size,
        "arxiv_submitted": local_submission_receipt_exists(results_path),
        "honest_verdict": "arxiv_bundle_v10_compiled_upload_pending",
        **compile_fields,
    }
    return _write_json(output, artifact)
