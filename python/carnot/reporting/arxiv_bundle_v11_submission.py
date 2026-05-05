"""Build the Exp 1380 audited arXiv bundle-v11 submission artifact.

This runner is deliberately operational and narrow. It trusts Exp 1379 as the
paper-integrity gate, compiles the audited TeX source with local tooling, and
packages only the active source files needed by arXiv. It does not invent an
upload event when the machine lacks a non-interactive arXiv submission command.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1380_arxiv_bundle_v11_submission.json"
DEFAULT_BUNDLE_PATH = DEFAULT_RESULTS_DIR / "arxiv_bundle_v11.tar.gz"
DEFAULT_AUDIT_PATH = DEFAULT_RESULTS_DIR / "experiment_1379_paper_integrity_audit_v2.json"

EXPERIMENT = "1380_arxiv_bundle_v11_submission"
SCHEMA = "arxiv_bundle_v11_submission_v1"
RUN_DATE = "20260505"
FIGURE_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg")
TEX_TOOL_NAMES = ("pdflatex", "xelatex", "bibtex", "tectonic")
UPLOAD_COMMAND = "arxiv-upload"

CommandRunner = Callable[[list[str], Path, int], Any]
Which = Callable[[str], str | None]


def _relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def _base_artifact(status: str) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "status": status,
        "paper_file_found": False,
        "latex_compile_success": False,
        "bundle_file_path": None,
        "bundle_size_bytes": 0,
        "figures_included": [],
        "submission_attempted": False,
        "submission_result": None,
        "arxiv_id_if_submitted": None,
        "remaining_blocker": None,
        "honest_verdict": status,
    }


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """Write the interruption-safe placeholder before any gate or build work.

    A partial conductor run should leave a real artifact explaining where it
    stopped. Without this first write, an interrupted bundle attempt is
    indistinguishable from a task that was never started.
    """

    return _write_json(Path(out_path), _base_artifact("in_progress"))


def default_command_runner(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run a local command with captured output for honest artifact logging."""

    return subprocess.run(
        cmd,
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )


def load_audit(audit_path: Path | str = DEFAULT_AUDIT_PATH) -> dict[str, Any]:
    path = Path(audit_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_paper_paths(root: Path, audit: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    audit_path = audit.get("paper_file_path")
    if isinstance(audit_path, str) and audit_path:
        path = Path(audit_path)
        candidates.append(path if path.is_absolute() else root / path)
    candidates.extend(
        [root / "docs" / "paper" / "main.tex", root / "docs" / "arxiv-paper" / "main.tex"]
    )
    return candidates


def find_paper_file(project_root: Path | str, audit: dict[str, Any]) -> tuple[Path, bool]:
    """Find the audited paper source, falling back to the documented defaults."""

    root = Path(project_root)
    for candidate in _candidate_paper_paths(root, audit):
        if candidate.exists():
            return candidate, True
    return _candidate_paper_paths(root, audit)[0], False


def parse_active_figures(tex_source: str) -> list[str]:
    """Return figure paths referenced by ``\\includegraphics`` in paper order.

    Only referenced figures are packaged. That keeps stale exploratory files in
    ``figures/`` from silently becoming part of the arXiv source archive.
    """

    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    seen: set[str] = set()
    figures: list[str] = []
    for match in pattern.finditer(tex_source):
        figure = match.group(1).strip()
        if figure not in seen:
            seen.add(figure)
            figures.append(figure)
    return figures


def _resolve_figure_path(paper_dir: Path, figure_ref: str) -> Path:
    direct = paper_dir / figure_ref
    if direct.suffix:
        return direct
    for suffix in FIGURE_EXTENSIONS:
        candidate = direct.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return direct.with_suffix(".pdf")


def collect_active_figure_paths(paper_file: Path) -> tuple[list[Path], list[str]]:
    tex_source = paper_file.read_text(encoding="utf-8")
    refs = parse_active_figures(tex_source)
    paths = [_resolve_figure_path(paper_file.parent, ref) for ref in refs]
    missing = [str(path.relative_to(paper_file.parent)) for path in paths if not path.exists()]
    return paths, missing


def _figure_name_from_audit_label(label: object) -> str | None:
    if not isinstance(label, str) or not label:
        return None
    first = label.split("/", 1)[0].strip()
    return Path(first).name if first else None


def live_gpu_figures_required_by_audit(audit: dict[str, Any]) -> list[str]:
    """Extract the figure filenames Exp 1379 explicitly marked live-GPU-backed."""

    names: list[str] = []
    for entry in audit.get("figures_with_live_provenance", []):
        name = _figure_name_from_audit_label(
            entry.get("figure") if isinstance(entry, dict) else None
        )
        if name and name not in names:
            names.append(name)
    return names


def placeholder_or_simulated_active_figure_blockers(
    audit: dict[str, Any],
    included_names: list[str],
) -> list[str]:
    """Flag active figures that the audit itself calls placeholder or simulated."""

    included = set(included_names)
    blockers: list[str] = []
    for entry in audit.get("figures_needing_verification", []):
        if not isinstance(entry, dict):
            continue
        name = _figure_name_from_audit_label(entry.get("figure"))
        reason = str(entry.get("reason", "")).lower()
        if name in included and ("placeholder" in reason or "simulated" in reason):
            blockers.append(f"{name}: {entry.get('reason')}")
    return blockers


def discover_latex_plan(
    *,
    which: Which = shutil.which,
) -> tuple[list[list[str]], dict[str, bool], str | None]:
    """Choose a local TeX build plan, preferring the arXiv-style engines first."""

    tools = {name: bool(which(name)) for name in TEX_TOOL_NAMES}
    if tools["pdflatex"]:
        plan = [["pdflatex", "-interaction=nonstopmode", "main.tex"]]
        if tools["bibtex"]:
            plan.append(["bibtex", "main"])
        plan.extend([["pdflatex", "-interaction=nonstopmode", "main.tex"]] * 2)
        return plan, tools, "pdflatex"
    if tools["xelatex"]:
        plan = [["xelatex", "-interaction=nonstopmode", "main.tex"]]
        if tools["bibtex"]:
            plan.append(["bibtex", "main"])
        plan.extend([["xelatex", "-interaction=nonstopmode", "main.tex"]] * 2)
        return plan, tools, "xelatex"
    if tools["tectonic"]:
        return [["tectonic", "--keep-intermediates", "main.tex"]], tools, "tectonic"
    return [], tools, None


def _tail(text: object, limit: int = 1600) -> str:
    value = str(text or "")
    return value[-limit:]


def compile_latex(
    paper_dir: Path,
    *,
    which: Which = shutil.which,
    command_runner: CommandRunner = default_command_runner,
    timeout: int = 300,
) -> dict[str, Any]:
    """Compile the paper and preserve enough command output to diagnose failure."""

    plan, tools, engine = discover_latex_plan(which=which)
    if not plan:
        return {
            "success": False,
            "engine": None,
            "available_tex_tools": tools,
            "commands": [],
            "blocker": "missing_local_tex_tooling: pdflatex, xelatex, and tectonic unavailable",
        }

    command_results: list[dict[str, Any]] = []
    for cmd in plan:
        result = command_runner(cmd, paper_dir, timeout)
        command_results.append(
            {
                "command": cmd,
                "returncode": result.returncode,
                "stdout_tail": _tail(getattr(result, "stdout", "")),
                "stderr_tail": _tail(getattr(result, "stderr", "")),
            }
        )
        if result.returncode != 0:
            return {
                "success": False,
                "engine": engine,
                "available_tex_tools": tools,
                "commands": command_results,
                "blocker": f"latex_compile_failed: {cmd[0]} returned {result.returncode}",
            }

    pdf_exists = (paper_dir / "main.pdf").exists()
    return {
        "success": pdf_exists,
        "engine": engine,
        "available_tex_tools": tools,
        "commands": command_results,
        "blocker": None if pdf_exists else "latex_compile_failed: main.pdf was not produced",
    }


def build_bundle(
    *,
    project_root: Path | str,
    paper_file: Path,
    figure_paths: list[Path],
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
) -> tuple[str, int]:
    """Create the arXiv source archive without stale or unused figure files."""

    root = Path(project_root)
    target = Path(bundle_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    paper_dir = paper_file.parent
    with tarfile.open(target, "w:gz") as tf:
        tf.add(paper_file, arcname=paper_file.name)
        for rel in ("carnot.bib", "main.bbl"):
            src = paper_dir / rel
            if src.exists():
                tf.add(src, arcname=rel)
        for figure in figure_paths:
            tf.add(figure, arcname=f"figures/{figure.name}")
    return _relative_path(target, root), target.stat().st_size


def _parse_arxiv_id(text: str) -> str | None:
    match = re.search(r"(?:arXiv:)?(\d{4}\.\d{4,5})(?:v\d+)?", text)
    return match.group(1) if match else None


def attempt_submission(
    bundle_path: Path,
    *,
    which: Which = shutil.which,
    command_runner: CommandRunner = default_command_runner,
    timeout: int = 300,
) -> dict[str, Any]:
    """Attempt upload only when a local non-interactive CLI is actually present."""

    if not which(UPLOAD_COMMAND):
        return {
            "submission_attempted": False,
            "submission_result": (
                "not_attempted_arxiv_upload_cli_missing; manual submission: upload "
                f"{bundle_path} at https://arxiv.org/submit, inspect AutoTeX output, "
                "select cs.LG primary with cs.AI/cs.NE/quant-ph as applicable, and submit."
            ),
            "arxiv_id_if_submitted": None,
            "submission_command": None,
        }

    result = command_runner([UPLOAD_COMMAND, str(bundle_path)], bundle_path.parent, timeout)
    combined_output = f"{getattr(result, 'stdout', '')}\n{getattr(result, 'stderr', '')}"
    arxiv_id = _parse_arxiv_id(combined_output)
    if result.returncode == 0:
        return {
            "submission_attempted": True,
            "submission_result": "submitted"
            if arxiv_id
            else "upload_command_succeeded_no_arxiv_id_seen",
            "arxiv_id_if_submitted": arxiv_id,
            "submission_command": [UPLOAD_COMMAND, str(bundle_path)],
            "submission_stdout_tail": _tail(getattr(result, "stdout", "")),
            "submission_stderr_tail": _tail(getattr(result, "stderr", "")),
        }
    return {
        "submission_attempted": True,
        "submission_result": f"submission_failed_returncode_{result.returncode}",
        "arxiv_id_if_submitted": arxiv_id,
        "submission_command": [UPLOAD_COMMAND, str(bundle_path)],
        "submission_stdout_tail": _tail(getattr(result, "stdout", "")),
        "submission_stderr_tail": _tail(getattr(result, "stderr", "")),
    }


def _blocked_artifact(
    blocker: str,
    *,
    paper_file_found: bool = False,
    figures_included: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = _base_artifact("blocked")
    artifact.update(
        {
            "paper_file_found": paper_file_found,
            "figures_included": figures_included or [],
            "remaining_blocker": blocker,
            "submission_result": "not_attempted_blocked_before_submission",
            "honest_verdict": "blocked",
        }
    )
    if extra:
        artifact.update(extra)
    return artifact


def run(
    *,
    project_root: Path | str = REPO_ROOT,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    audit_path: Path | str | None = None,
    out_path: Path | str = DEFAULT_OUT_PATH,
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
    which: Which = shutil.which,
    command_runner: CommandRunner = default_command_runner,
    timeout: int = 300,
) -> dict[str, Any]:
    """Run the Exp 1380 workflow and write the final bundle/submission artifact."""

    root = Path(project_root)
    results = Path(results_dir)
    output = Path(out_path)
    bundle_target = Path(bundle_path)
    audit_file = Path(audit_path) if audit_path is not None else results / DEFAULT_AUDIT_PATH.name
    write_in_progress_artifact(output)

    audit = load_audit(audit_file)
    if not audit:
        return _write_json(output, _blocked_artifact("missing_exp1379_audit_artifact"))
    if audit.get("arxiv_submission_ready") is not True:
        return _write_json(
            output,
            _blocked_artifact(
                "exp1379_arxiv_submission_ready_false",
                extra={"exp1379_remaining_blockers": audit.get("remaining_blockers", [])},
            ),
        )

    paper_file, paper_found = find_paper_file(root, audit)
    if not paper_found:
        return _write_json(output, _blocked_artifact("audited_paper_file_missing"))

    figure_paths, missing_figures = collect_active_figure_paths(paper_file)
    figures_included = [path.name for path in figure_paths if path.exists()]
    if missing_figures:
        return _write_json(
            output,
            _blocked_artifact(
                f"active_paper_figures_missing: {missing_figures}",
                paper_file_found=True,
                figures_included=figures_included,
            ),
        )

    live_required = live_gpu_figures_required_by_audit(audit)
    missing_live = [name for name in live_required if name not in figures_included]
    figure_blockers = placeholder_or_simulated_active_figure_blockers(audit, figures_included)
    if missing_live or figure_blockers:
        return _write_json(
            output,
            _blocked_artifact(
                f"figure_provenance_blocker: missing_live={missing_live}; blockers={figure_blockers}",
                paper_file_found=True,
                figures_included=figures_included,
                extra={"live_gpu_figures_required": live_required},
            ),
        )

    compile_result = compile_latex(
        paper_file.parent,
        which=which,
        command_runner=command_runner,
        timeout=timeout,
    )
    compile_extra = {
        "paper_file_path": _relative_path(paper_file, root),
        "latex_engine": compile_result.get("engine"),
        "available_tex_tools": compile_result.get("available_tex_tools"),
        "compile_commands": compile_result.get("commands"),
    }
    if not compile_result["success"]:
        return _write_json(
            output,
            _blocked_artifact(
                str(compile_result["blocker"]),
                paper_file_found=True,
                figures_included=figures_included,
                extra=compile_extra,
            ),
        )

    bundle_rel, bundle_size = build_bundle(
        project_root=root,
        paper_file=paper_file,
        figure_paths=figure_paths,
        bundle_path=bundle_target,
    )
    submission = attempt_submission(
        root / bundle_rel,
        which=which,
        command_runner=command_runner,
        timeout=timeout,
    )

    artifact = _base_artifact("complete")
    artifact.update(
        {
            "paper_file_found": True,
            "paper_file_path": _relative_path(paper_file, root),
            "latex_compile_success": True,
            "bundle_file_path": bundle_rel,
            "bundle_size_bytes": bundle_size,
            "figures_included": figures_included,
            "submission_attempted": submission["submission_attempted"],
            "submission_result": submission["submission_result"],
            "arxiv_id_if_submitted": submission["arxiv_id_if_submitted"],
            "remaining_blocker": None,
            "honest_verdict": (
                "arxiv_submitted"
                if submission["arxiv_id_if_submitted"]
                else "submission_ready_archive_created_manual_upload_required"
            ),
            "live_gpu_figures_required": live_required,
            "non_placeholder_active_figures_verified": True,
            **compile_extra,
            **submission,
        }
    )
    return _write_json(output, artifact)


def main() -> int:
    artifact = run()
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "latex_compile_success": artifact["latex_compile_success"],
                "bundle_file_path": artifact["bundle_file_path"],
                "submission_result": artifact["submission_result"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
