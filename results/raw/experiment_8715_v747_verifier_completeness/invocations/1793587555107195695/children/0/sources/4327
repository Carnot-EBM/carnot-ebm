"""Build the Exp 2553 arXiv package v3 readiness artifact.

This module is deliberately limited to local readiness checks. It compiles the
checked-in paper, evaluates the four publication gates under the .245 Gate 3
definition, and writes an operator checklist. It does not contain any code path
that can log in to arXiv, upload a bundle, or submit the paper.

Spec refs: REQ-PUBLISH-030, SCENARIO-PUBLISH-030.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_REL_PATH = Path("docs/arxiv-paper/main.tex")
EXP2544_REL_PATH = Path("results/experiment_2544_phase4_option_b.json")
EXP2536_REL_PATH = Path("results/experiment_2536_latex_compile_fix.json")
EXP2552_REL_PATH = Path("results/experiment_2552_paper_writethrough.json")
EXP2441_REL_PATH = Path("results/experiment_2441_phase1_ship_gate_completion_v5.json")
EXP2479_REL_PATH = Path("results/experiment_2479_paper_integrity_fix.json")
EXP2498_REL_PATH = Path("results/experiment_2498_auroc_adversarial_v2_group_cond.json")
OUTPUT_REL_PATH = Path("results/experiment_2553_arxiv_package_v3.json")

ABSTRACT_LIMIT_WORDS = 250
BEST_AUROC = 0.9750
TEX_ENGINES = ("tectonic", "pdflatex")
TERMINAL_PREFIXES = ("complete:", "blocked_", "blocked:")

CommandRunner = Callable[[list[str], Path, int], subprocess.CompletedProcess[str]]
Which = Callable[[str], str | None]

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": "Terminal-prefix required.",
    "arxiv_ready": (
        "True if all 4 gates satisfied. This is the PRIMARY output of the entire "
        ".240-.245 research track."
    ),
    "submission_package_ready": (
        "True if mechanically ready for operator submission. Distinct from arxiv_ready "
        "(mechanical vs gate-based)."
    ),
    "gate_3_phase4_resolved": (
        "True if phase4_validated_any OR phase4_honest_negative_documented. Redefined "
        "gate for Option B path."
    ),
    "latex_compile_success": "Must be True -- arXiv's LaTeX compiler is strict.",
    "abstract_word_count": "Must be <= 250 for arXiv.",
    "operator_submission_checklist": (
        "Concrete operator action items before submission -- prevents accidental "
        "submission without review."
    ),
    "preconditions_checked": "Records which resources were verified.",
    "duration_s": "Wall-clock measurement.",
}

OPERATOR_SUBMISSION_CHECKLIST = [
    "[ ] Review the 4/delta citation in \u00a73 (arXiv:2512.02080)",
    (
        "[ ] Confirm \u00a74 Phase 4 honest negative accurately represents experiments "
        "exp2486/exp2508/exp2519/exp2532"
    ),
    "[ ] Confirm best_AUROC=0.9750 in abstract and results section",
    "[ ] Confirm author list and affiliation are correct",
    "[ ] Submit to arXiv at arxiv.org/submit (category: cs.AI, cs.LG)",
]


def read_json(path: Path) -> Mapping[str, Any]:
    """Return a JSON object from a local artifact, or an empty object on failure."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _precondition(resource: str, check: str, available: bool, **extra: Any) -> dict[str, Any]:
    record = {"resource": resource, "check": check, "available": available}
    record.update(extra)
    return record


def count_abstract_words(tex_text: str) -> int | str:
    """Count words inside the LaTeX abstract environment using the requested split.

    The milestone asks for the simple arXiv-preflight count rather than a detex
    transform: locate ``\\begin{abstract}``, take text through
    ``\\end{abstract}``, and split on whitespace. That deliberately mirrors the
    terminal one-liner in the task so the artifact is easy to audit.
    """

    abstract = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", tex_text, re.DOTALL)
    return len(abstract.group(1).split()) if abstract else "not_found"


def _abstract_within_limit(word_count: int | str) -> bool:
    return isinstance(word_count, int) and word_count <= ABSTRACT_LIMIT_WORDS


def discover_tex_engine(*, which: Which = shutil.which) -> tuple[str | None, dict[str, bool]]:
    """Find the local TeX engine, preferring tectonic over pdflatex."""

    availability = {engine: bool(which(engine)) for engine in TEX_ENGINES}
    for engine in TEX_ENGINES:
        if availability[engine]:
            return engine, availability
    return None, availability


def default_command_runner(
    cmd: list[str], cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    """Run a local command and capture enough output for the readiness artifact."""

    return subprocess.run(
        cmd,
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )


def _compile_command(engine: str) -> list[str]:
    if engine == "tectonic":
        return ["tectonic", "main.tex"]
    if engine == "pdflatex":
        return ["pdflatex", "-interaction=nonstopmode", "main.tex"]
    raise ValueError(f"unsupported TeX engine: {engine}")


def _tail(text: object, limit: int = 1200) -> str:
    return str(text or "")[-limit:]


def compile_paper(
    paper_path: Path,
    *,
    engine: str | None,
    command_runner: CommandRunner = default_command_runner,
    timeout: int = 300,
) -> dict[str, Any]:
    """Compile ``main.tex`` and report the exact local command outcome."""

    if engine is None:
        return {
            "success": False,
            "engine": None,
            "command": None,
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "missing_local_tex_tooling: tectonic and pdflatex unavailable",
        }

    command = _compile_command(engine)
    result = command_runner(command, paper_path.parent, timeout)
    return {
        "success": result.returncode == 0,
        "engine": engine,
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": _tail(getattr(result, "stdout", "")),
        "stderr_tail": _tail(getattr(result, "stderr", "")),
    }


def _gate_1_phase1_ship(exp2441: Mapping[str, Any]) -> bool:
    return exp2441.get("phase1_ship_gate_met") is True


def _gate_2_audit(exp2479: Mapping[str, Any]) -> bool:
    return exp2479.get("audit_passed_after_fix") is True


def _gate_4_auroc(exp2498: Mapping[str, Any]) -> bool:
    return (
        exp2498.get("auroc_adversarially_verified") is True
        and round(float(exp2498.get("group_conditional_auroc_replicated", 0.0)), 4) == BEST_AUROC
    )


def _base_blocked_artifact(
    *,
    honest_verdict: str,
    preconditions_checked: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "honest_verdict": honest_verdict,
        "arxiv_ready": False,
        "submission_package_ready": False,
        "gate_3_phase4_resolved": False,
        "latex_compile_success": False,
        "abstract_word_count": "not_found",
        "operator_submission_checklist": OPERATOR_SUBMISSION_CHECKLIST,
        "preconditions_checked": preconditions_checked,
        "duration_s": duration_s,
        "field_principles": FIELD_PRINCIPLES,
        "acceptance_gates": [
            {
                "condition": "arxiv_ready == true",
                "principle": (
                    "The primary milestone success criterion. If all 4 gates met "
                    "including the redefined gate-3, the paper is ready for operator "
                    "submission."
                ),
                "passed": False,
            }
        ],
        "submission_attempted": False,
        "credentialed_submission_attempted": False,
    }


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    started_epoch: float | None = None,
    now_epoch: float | None = None,
    which: Which = shutil.which,
    command_runner: CommandRunner = default_command_runner,
) -> dict[str, Any]:
    """Build the Exp 2553 readiness artifact from local checked-in evidence."""

    started = time.time() if started_epoch is None else started_epoch
    root = Path(root)
    paper_path = root / PAPER_REL_PATH
    exp2544_path = root / EXP2544_REL_PATH
    exp2536_path = root / EXP2536_REL_PATH
    exp2552_path = root / EXP2552_REL_PATH
    exp2441_path = root / EXP2441_REL_PATH
    exp2479_path = root / EXP2479_REL_PATH
    exp2498_path = root / EXP2498_REL_PATH

    engine, tex_availability = discover_tex_engine(which=which)
    preconditions = [
        _precondition(
            str(paper_path),
            f"ls {paper_path}",
            paper_path.is_file(),
        ),
        _precondition("tectonic", "command -v tectonic", tex_availability["tectonic"]),
        _precondition("pdflatex", "command -v pdflatex", tex_availability["pdflatex"]),
        _precondition(
            str(exp2544_path),
            "read phase4_honest_negative_documented",
            exp2544_path.is_file(),
        ),
        _precondition(
            str(exp2536_path),
            "read prior latex compile and abstract artifact",
            exp2536_path.is_file(),
        ),
        _precondition(
            str(exp2552_path),
            "read paper write-through artifact",
            exp2552_path.is_file(),
        ),
        _precondition(
            str(exp2441_path), "verify Phase 1 ship gate artifact", exp2441_path.is_file()
        ),
        _precondition(
            str(exp2479_path), "verify integrity audit fix artifact", exp2479_path.is_file()
        ),
        _precondition(
            str(exp2498_path), "verify AUROC adversarial artifact", exp2498_path.is_file()
        ),
    ]

    if not paper_path.is_file():
        finished = time.time() if now_epoch is None else now_epoch
        return _base_blocked_artifact(
            honest_verdict="blocked_paper_not_found",
            preconditions_checked=preconditions,
            duration_s=round(max(0.0, finished - started), 6),
        )

    paper_text = paper_path.read_text(encoding="utf-8")
    exp2544 = read_json(exp2544_path)
    exp2536 = read_json(exp2536_path)
    exp2552 = read_json(exp2552_path)
    exp2441 = read_json(exp2441_path)
    exp2479 = read_json(exp2479_path)
    exp2498 = read_json(exp2498_path)

    phase4_honest_negative_documented = exp2544.get("phase4_honest_negative_documented") is True
    phase4_validated_any = exp2544.get("phase4_validated_any") is True
    gate_3_phase4_resolved = phase4_validated_any or phase4_honest_negative_documented
    abstract_word_count = count_abstract_words(paper_text)
    compile_status = compile_paper(
        paper_path,
        engine=engine,
        command_runner=command_runner,
    )
    latex_compile_success = compile_status["success"] is True

    gate_1_phase1_ship = _gate_1_phase1_ship(exp2441)
    gate_2_audit = _gate_2_audit(exp2479)
    gate_4_auroc_adversarially_verified = _gate_4_auroc(exp2498)
    abstract_word_count_lte_250 = _abstract_within_limit(abstract_word_count)
    arxiv_ready = (
        latex_compile_success
        and abstract_word_count_lte_250
        and gate_1_phase1_ship
        and gate_2_audit
        and gate_3_phase4_resolved
        and gate_4_auroc_adversarially_verified
    )
    submission_package_ready = arxiv_ready
    honest_prefix = "complete:" if arxiv_ready else "blocked:"

    finished = time.time() if now_epoch is None else now_epoch
    duration_s = round(max(0.0, finished - started), 6)

    return {
        "honest_verdict": (
            f"{honest_prefix} arxiv_ready={arxiv_ready}; "
            f"latex_compile_success={latex_compile_success}; "
            f"abstract_word_count={abstract_word_count}; "
            f"gate_3_phase4_resolved={gate_3_phase4_resolved}"
        ),
        "arxiv_ready": arxiv_ready,
        "submission_package_ready": submission_package_ready,
        "gate_3_phase4_resolved": gate_3_phase4_resolved,
        "latex_compile_success": latex_compile_success,
        "abstract_word_count": abstract_word_count,
        "operator_submission_checklist": OPERATOR_SUBMISSION_CHECKLIST,
        "preconditions_checked": preconditions,
        "duration_s": duration_s,
        "arxiv_gates": {
            "gate_1_phase1_ship": gate_1_phase1_ship,
            "gate_2_audit": gate_2_audit,
            "gate_3_phase4_resolved": gate_3_phase4_resolved,
            "gate_4_auroc_adversarially_verified": gate_4_auroc_adversarially_verified,
        },
        "mechanical_gates": {
            "latex_compile_success": latex_compile_success,
            "abstract_word_count_lte_250": abstract_word_count_lte_250,
        },
        "gate_3_definition": (
            "phase4_resolved = phase4_validated_any OR phase4_honest_negative_documented"
        ),
        "phase4_validated_any": phase4_validated_any,
        "phase4_honest_negative_documented": phase4_honest_negative_documented,
        "best_AUROC": BEST_AUROC,
        "best_AUROC_source_artifact": str(EXP2498_REL_PATH),
        "latex_engine_used": compile_status["engine"],
        "latex_compile": compile_status,
        "source_artifacts": {
            "exp2536_latex_compile_success": exp2536.get("latex_compile_success"),
            "exp2536_abstract_word_count": exp2536.get("abstract_word_count"),
            "exp2552_paper_updated": exp2552.get("paper_updated"),
        },
        "field_principles": FIELD_PRINCIPLES,
        "acceptance_gates": [
            {
                "condition": "arxiv_ready == true",
                "principle": (
                    "The primary milestone success criterion. If all 4 gates met "
                    "including the redefined gate-3, the paper is ready for operator "
                    "submission."
                ),
                "passed": arxiv_ready,
            }
        ],
        "submission_attempted": False,
        "credentialed_submission_attempted": False,
        "files_modified": [
            "docs/arxiv-paper/main.pdf",
            str(OUTPUT_REL_PATH),
            "python/carnot/reporting/arxiv_package_v3_2553.py",
            "tests/python/test_arxiv_package_v3_2553.py",
            "openspec/capabilities/publication/spec.md",
        ],
    }


def main() -> int:
    started_env = os.environ.get("CARNOT_EXP2553_START_EPOCH")
    started_epoch = float(started_env) if started_env else None
    artifact = build_artifact(REPO_ROOT, started_epoch=started_epoch)
    out_path = REPO_ROOT / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if artifact["arxiv_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
