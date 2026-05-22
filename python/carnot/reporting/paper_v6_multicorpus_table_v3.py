"""Exp 2841 paper-v6 multi-corpus table integration.

The v3 integrator reads the post-.268 dual-condition artifacts, updates the
paper from those local sources, and refuses to promote blocked corpus rows into
publication-ready claims. This keeps the operator review package separate from
external submission.

Spec traces: REQ-PUBLISH-034, SCENARIO-PUBLISH-034, SCENARIO-PUBLISH-034B.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_REL_PATH = Path("docs/arxiv-paper/main.tex")
OUTPUT_FILENAME = "experiment_2841_paper_v6_multicorpus_table_v3.json"
OUTPUT_REL_PATH = Path("results") / OUTPUT_FILENAME

ARTIFACT_FILES = {
    "Runtime": "results/experiment_2836_sota_runtime_preflight.json",
    "FoVer": "results/experiment_2837_fover_memory_leakage_v3.json",
    "MBPP": "results/experiment_2838_mbpp_dual_condition_v3.json",
    "HumanEval": "results/experiment_2839_humaneval_dual_condition_v3.json",
    "TruthfulQA": "results/experiment_2840_truthfulqa_dual_condition_v4.json",
    "Matrix": "results/experiment_2840_cross_corpus_verifier_matrix_v3.json",
}

CORPUS_ORDER = ("FoVer", "MBPP", "HumanEval", "TruthfulQA")
CORPUS_N = {
    "FoVer": 1000,
    "MBPP": 100,
    "HumanEval": 164,
    "TruthfulQA": 200,
}
PEER_BASELINES = {
    "FoVer": "HIVE 0.924",
    "MBPP": "peer baseline not established",
    "HumanEval": "peer baseline not established",
    "TruthfulQA": "GPT-3 MC1 $\\sim$28\\%",
}

SECTION_START = "\\begin{table}[h]\n\\centering\n\\caption{Multi-Corpus Dual-Condition Evaluation"
SECTION_END = "\nSection~\\ref{sec:bounds} corrects"

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix.",
    "paper_v6_compile_success": "Did pdflatex succeed.",
    "corpora_in_table": "Names corpora present in the dual-condition results table.",
    "submission_package_ready": (
        "True if operator-ready; task NEVER submits per Operator-Only rule."
    ),
    "arxiv_ready_v8": "True only when every corpus row is measured and pdflatex passes.",
    "duration_s": "Real wall time for source integration and pdflatex.",
}

CommandRunner = Callable[[list[str], Path, int], subprocess.CompletedProcess[str]]


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifacts(root: Path) -> dict[str, dict[str, object]]:
    """Load the Exp 2836 runtime, Exp 2837-2840 corpus, and Exp 2840 matrix artifacts."""

    return {
        name: _load_json(root / relative_path) for name, relative_path in ARTIFACT_FILES.items()
    }


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _unmeasured_reason(verdict: object) -> str:
    text = str(verdict or "")
    if text.startswith("blocked_"):
        return "blocked " + text.split(":", 1)[0].removeprefix("blocked_").replace("_", " ")
    if text.startswith("blocked"):
        return text.split(":", 1)[0].replace("_", " ")
    return "not reported"


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _format_auroc(mean: float | None, std: float | None, verdict: object) -> str:
    if mean is None:
        return f"\\emph{{unmeasured ({_unmeasured_reason(verdict)})}}"
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def _format_delta(delta: float | None, verdict: object) -> str:
    if delta is None:
        return f"\\emph{{unmeasured ({_unmeasured_reason(verdict)})}}"
    return f"{delta:+.3f}"


def build_rows(
    artifacts: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Convert source corpus artifacts into table rows without inventing values."""

    rows: list[dict[str, object]] = []
    for corpus in CORPUS_ORDER:
        artifact = artifacts[corpus]
        production = _float_or_none(artifact.get("condition_a_production_auroc_mean"))
        production_std = _float_or_none(artifact.get("condition_a_production_auroc_std"))
        architecture = _float_or_none(artifact.get("condition_b_architecture_only_auroc_mean"))
        architecture_std = _float_or_none(artifact.get("condition_b_architecture_only_auroc_std"))
        learning_delta = _float_or_none(artifact.get("learning_contribution"))
        verdict = artifact.get("honest_verdict")
        rows.append(
            {
                "corpus": corpus,
                "n": CORPUS_N[corpus],
                "architecture_only": architecture,
                "architecture_only_std": architecture_std,
                "production": production,
                "production_std": production_std,
                "learning_delta": learning_delta,
                "peer": PEER_BASELINES[corpus],
                "honest_verdict": verdict,
                "measured": production is not None and architecture is not None,
            }
        )
    return rows


def _render_table(rows: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Multi-Corpus Dual-Condition Evaluation (exp2836--exp2840)}",
        r"\label{tab:multi_corpus}",
        r"\begin{tabular}{l c c c c l}",
        r"\toprule",
        (
            r"Corpus & N & Architecture-only AUROC & Production AUROC & "
            r"Learning $\Delta$ & Peer baseline \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        verdict = row["honest_verdict"]
        lines.append(
            " & ".join(
                [
                    str(row["corpus"]),
                    str(row["n"]),
                    _format_auroc(row["architecture_only"], row["architecture_only_std"], verdict),
                    _format_auroc(row["production"], row["production_std"], verdict),
                    _format_delta(row["learning_delta"], verdict),
                    str(row["peer"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _format_per_verifier_contributions(values: object) -> str:
    if not isinstance(values, Mapping) or not values:
        return "No per-verifier learning-contribution rows were measured."
    items = sorted((str(name), float(delta)) for name, delta in values.items())
    return ", ".join(f"{_latex_escape(name)} ({delta:+.3f})" for name, delta in items)


def _render_self_learning_disclosure(fover: Mapping[str, object]) -> str:
    learning = _float_or_none(fover.get("learning_contribution"))
    if learning is None:
        contribution_sentence = (
            "The FoVer dual-condition source reports "
            "\\texttt{learning\\_contribution is unmeasured} because the run "
            f"ended at {_unmeasured_reason(fover.get('honest_verdict'))}."
        )
    else:
        contribution_sentence = (
            "The FoVer dual-condition source reports "
            "\\texttt{learning\\_contribution = "
            f"{learning:.3f}"
            "}, computed as production AUROC minus architecture-only AUROC "
            "on the same FoVer subset."
        )

    per_verifier = _format_per_verifier_contributions(
        fover.get("per_verifier_learning_contribution")
    )
    return "\n".join(
        [
            r"\subsection{Self-Learning Contribution Disclosure}",
            r"\label{sec:self_learning_disclosure}",
            (
                f"{contribution_sentence} The per-verifier contribution field "
                f"reports: {per_verifier} This disclosure keeps accumulated "
                "FR-11 state separate from the architecture-only baseline."
            ),
        ]
    )


def _format_category(matrix: Mapping[str, object], key: str, label: str) -> str:
    values = matrix.get(key, [])
    rendered = "none"
    if isinstance(values, list) and values:
        rendered = ", ".join(_latex_escape(value) for value in values)
    return f"{label}: {rendered}"


def _render_per_verifier_breakdown(matrix: Mapping[str, object]) -> str:
    categories = "; ".join(
        [
            _format_category(matrix, "architecture_transfer_verifiers", "architecture-transfer"),
            _format_category(matrix, "memory_augmented_verifiers", "memory-augmented"),
            _format_category(matrix, "corpus_specific_verifiers", "corpus-specific"),
            _format_category(matrix, "low_signal_verifiers", "low-signal"),
        ]
    )
    matrix_size = len(matrix.get("verifier_corpus_dual_matrix", {}))
    diversity_gap = matrix.get("diversity_gap_on_non_fover")
    methodology = _latex_escape(matrix.get("methodology_note", "not reported"))
    return "\n".join(
        [
            r"\subsection{Per-verifier Breakdown}",
            r"\label{sec:per_verifier_breakdown}",
            (
                f"Exp 2840 provides the per-verifier cross-corpus matrix used by "
                f"this table. It contains {matrix_size} measured verifier rows; "
                f"{categories}. The non-FoVer diversity-gap flag is "
                f"\\texttt{{{str(diversity_gap).lower()}}}. Methodology note: "
                f"{methodology}"
            ),
        ]
    )


def render_section_block(artifacts: Mapping[str, Mapping[str, object]]) -> str:
    """Render the complete paper-v6 Section 5 table and disclosure subsections."""

    rows = build_rows(artifacts)
    return "\n\n".join(
        [
            _render_table(rows),
            _render_self_learning_disclosure(artifacts["FoVer"]),
            _render_per_verifier_breakdown(artifacts["Matrix"]),
        ]
    )


def update_paper_text(tex_text: str, section_block: str) -> str:
    """Replace the existing multi-corpus table block in the active paper source."""

    start = tex_text.find(SECTION_START)
    if start < 0:  # pragma: no cover - defensive guard for unexpected paper drift.
        raise ValueError("multi-corpus table block start not found")
    end = tex_text.find(SECTION_END, start)
    if end < 0:  # pragma: no cover - defensive guard for unexpected paper drift.
        raise ValueError("multi-corpus table block end not found")
    return tex_text[:start] + section_block + "\n" + tex_text[end:]


def default_command_runner(
    cmd: list[str], cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:  # pragma: no cover - exercised by real run.
    return subprocess.run(
        cmd,
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )


def _tail(value: object, limit: int = 1200) -> str:
    return str(value or "")[-limit:]


def compile_paper(
    paper_path: Path,
    *,
    command_runner: CommandRunner = default_command_runner,
    timeout: int = 300,
) -> dict[str, object]:
    """Compile the active paper once with pdflatex and summarize the outcome."""

    command = ["pdflatex", "-interaction=nonstopmode", "main.tex"]
    result = command_runner(command, paper_path.parent, timeout)
    return {
        "success": result.returncode == 0,
        "engine": "pdflatex",
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": _tail(getattr(result, "stdout", "")),
        "stderr_tail": _tail(getattr(result, "stderr", "")),
    }


def _source_statuses(
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    statuses: dict[str, dict[str, object]] = {}
    for name, artifact in artifacts.items():
        status = {
            "path": ARTIFACT_FILES[name],
            "honest_verdict": artifact.get("honest_verdict"),
        }
        if name in CORPUS_ORDER:
            status.update(
                {
                    "production_auroc": artifact.get("condition_a_production_auroc_mean"),
                    "architecture_only_auroc": artifact.get(
                        "condition_b_architecture_only_auroc_mean"
                    ),
                    "learning_contribution": artifact.get("learning_contribution"),
                }
            )
        if name == "Runtime":
            status.update(
                {
                    "sota_runtime_ready": artifact.get("sota_runtime_ready"),
                    "selected_python": artifact.get("selected_python"),
                }
            )
        statuses[name] = status
    return statuses


def build_artifact(
    artifacts: Mapping[str, Mapping[str, object]],
    *,
    paper_v6_compile_success: bool,
    compile_result: Mapping[str, object],
    duration_s: float,
) -> dict[str, object]:
    """Build the Exp 2841 terminal artifact from source artifacts and compile status."""

    rows = build_rows(artifacts)
    all_measured = all(bool(row["measured"]) for row in rows)
    submission_ready = paper_v6_compile_success and all_measured
    if submission_ready:
        verdict = (
            "complete: exp2836-2840 dual-condition results integrated into "
            "paper-v6 section 5; arxiv_ready_v8=true for operator review"
        )
    elif not paper_v6_compile_success:
        verdict = (
            "complete: exp2836-2840 table integrated but pdflatex failed; arxiv_ready_v8=false"
        )
    else:
        verdict = (
            "complete: exp2836-2840 artifacts integrated honestly; at least one "
            "dual-condition AUROC remains unmeasured, so arxiv_ready_v8=false"
        )

    return {
        "honest_verdict": verdict,
        "paper_v6_compile_success": paper_v6_compile_success,
        "corpora_in_table": list(CORPUS_ORDER),
        "submission_package_ready": submission_ready,
        "arxiv_ready_v8": submission_ready,
        "duration_s": float(duration_s),
        "all_dual_condition_auroc_measured": all_measured,
        "table_rows": rows,
        "source_artifacts": _source_statuses(artifacts),
        "compile_result": dict(compile_result),
        "paper_path": PAPER_REL_PATH.as_posix(),
        "submission_attempted": False,
        "credentialed_submission_attempted": False,
        "operator_only_external_publication": True,
        "field_principles": FIELD_PRINCIPLES,
    }


def write_artifact(root: Path, artifact: Mapping[str, object]) -> None:
    output_path = root / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def run(
    root: Path = REPO_ROOT,
    *,
    command_runner: CommandRunner = default_command_runner,
    clock: Callable[[], float] = time.time,
) -> dict[str, object]:
    """Update paper-v6, compile it with pdflatex, and write the Exp 2841 artifact."""

    started = clock()
    root = Path(root)
    paper_path = root / PAPER_REL_PATH
    artifacts = load_artifacts(root)
    section_block = render_section_block(artifacts)
    paper_path.write_text(
        update_paper_text(paper_path.read_text(encoding="utf-8"), section_block),
        encoding="utf-8",
    )
    compile_result = compile_paper(paper_path, command_runner=command_runner)
    artifact = build_artifact(
        artifacts,
        paper_v6_compile_success=bool(compile_result["success"]),
        compile_result=compile_result,
        duration_s=clock() - started,
    )
    write_artifact(root, artifact)
    return artifact


def main() -> None:  # pragma: no cover
    run(REPO_ROOT)


if __name__ == "__main__":  # pragma: no cover
    main()
