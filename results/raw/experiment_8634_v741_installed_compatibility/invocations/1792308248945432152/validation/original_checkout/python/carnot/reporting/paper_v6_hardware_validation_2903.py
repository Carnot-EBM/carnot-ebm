"""Stage the Exp 2903 paper-v6 hardware-validation subsection.

Spec refs: REQ-PUBLISH-035, SCENARIO-PUBLISH-035, SCENARIO-PUBLISH-035B.

This module reads the completed KV260 latency artifact from Exp 2898 and turns
only those measured fields into a standalone LaTeX subsection. It does not edit
``main.tex`` because external publication integration is operator-owned.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from statistics import median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
SCHEMA = "carnot.paper_v6_hardware_validation.v1"
ARTIFACT = "experiment_2903_paper_v6_hardware_validation_section_v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP2898_REL_PATH = Path(
    "results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
)
OUTPUT_REL_PATH = Path("results/experiment_2903_paper_v6_hardware_validation_section_v1.json")
SNIPPET_REL_PATH = Path("docs/arxiv-paper/sections/hardware-validation-v1.tex")
PAPER_REL_PATH = Path("docs/arxiv-paper/main.tex")

BOARD_NAME = "Xilinx Kria KV260"
FIELDS_IMPORTED = [
    "honest_verdict",
    "inference_substrate",
    "bitstream_sha256",
    "bitstream_sha256_source",
    "kv260_overlay_loaded",
    "ising_problem_spec.n_spins",
    "preconditions_checked",
    "per_seed_results",
    "sample_count_sweep_results",
]


def read_json(path: Path) -> dict[str, Any]:
    """Return a JSON object or ``{}`` when the source cannot support a citation.

    Paper-facing aggregation should fail closed. A missing, malformed, or
    non-object upstream artifact is not evidence, so callers get an empty object
    and record the blocked source explicitly.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def validate_upstream(payload: dict[str, Any]) -> list[str]:
    """REQ-PUBLISH-035: verify Exp 2898 is citable before paper text is staged."""

    if not payload:
        return ["exp2898_artifact_missing_or_malformed"]

    reasons: list[str] = []
    if not _terminal_success(payload.get("honest_verdict")):
        reasons.append("honest_verdict_not_complete_or_success")
    if payload.get("inference_substrate") != "hardware_smoke":
        reasons.append("inference_substrate_not_hardware_smoke")
    if not _preconditions_available(payload):
        reasons.append("preconditions_not_all_available")

    rows = payload.get("per_seed_results")
    if not isinstance(rows, list) or not rows:
        reasons.append("per_seed_results_missing")
    elif any(not _positive_latency_row(row) for row in rows):
        reasons.append("per_seed_results_have_nonpositive_latency")

    reasons.extend(_sample_failures(payload))
    reasons.extend(_acceptance_gate_failures(payload))
    return reasons


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """Build the Exp 2903 artifact without writing files."""

    artifact, _snippet = build_outputs(root, started_s=started_s, now_s=now_s)
    return artifact


def build_outputs(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Build the JSON artifact and optional LaTeX snippet from local evidence."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    source_path = root_path / EXP2898_REL_PATH
    payload = read_json(source_path)
    blocked_reasons = validate_upstream(payload)
    duration_s = (time.perf_counter() if now_s is None else now_s) - started
    cited = _cited_upstream_artifacts(source_path)

    if blocked_reasons:
        return (
            _base_artifact(
                honest_verdict="blocked: exp2898_not_citable_for_paper_v6_hardware_validation",
                duration_s=duration_s,
                cited_upstream_artifacts=cited,
                blocked_reasons=blocked_reasons,
            ),
            None,
        )

    summary = _latency_summary(payload)
    snippet = render_snippet(summary)
    artifact = _base_artifact(
        honest_verdict="complete: paper_v6_hardware_validation_section_staged_from_exp2898",
        duration_s=duration_s,
        cited_upstream_artifacts=cited,
        blocked_reasons=[],
        summary=summary,
    )
    return artifact, snippet


def write_outputs(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    snippet_path: Path | str = SNIPPET_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write the Exp 2903 artifact and, when valid, the standalone snippet."""

    root_path = Path(root)
    out_path = _resolve(root_path, Path(output_path))
    tex_path = _resolve(root_path, Path(snippet_path))
    artifact, snippet = build_outputs(root_path, started_s=started_s, now_s=now_s)

    if snippet is not None:
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(snippet, encoding="utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def render_snippet(summary: dict[str, Any]) -> str:
    """Render the standalone LaTeX subsection from source-derived values."""

    rows = summary["per_seed_latencies"]
    lines = [
        r"\subsection{Hardware Validation}",
        r"\label{sec:hardware-validation}",
        (
            "Experiment 2898 provides a board-level hardware-smoke latency "
            f"measurement on a {_latex_escape(summary['board'])}. The loaded overlay was "
            f"\\texttt{{{_latex_escape(summary['overlay_name'])}}}, with bitstream SHA-256 "
            f"\\texttt{{{_latex_escape(summary['bitstream_sha256'])}}} "
            f"from \\texttt{{{_latex_escape(summary['bitstream_sha256_source'])}}}. "
            f"The uploaded Ising problem used \\texttt{{n\\_spins={summary['n_spins']}}}."
        ),
        "",
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \begin{tabular}{r r r r}",
        r"    \toprule",
        r"    Seed & Samples & Median latency ($\mu s$) & p95 latency ($\mu s$) \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            f"{row['seed']} & {row['n_samples']} & "
            f"{_format_us(row['median_us'])} & {_format_us(row['p95_us'])} \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            (
                r"  \caption{KV260 per-seed board-level latency from Exp 2898. "
                r"The cited aggregate p50 is the median of the per-seed medians "
                r"and the cited p95 is the largest per-seed p95.}"
            ),
            r"  \label{tab:kv260-hardware-validation-v1}",
            r"\end{table}",
            "",
            (
                f"Across these three seeds, the cited aggregate p50 is "
                f"{_format_us(summary['cited_p50_us'])}\\,$\\mu s$ and the cited p95 is "
                f"{_format_us(summary['cited_p95_us'])}\\,$\\mu s$. "
                "No same-basis CPU baseline has been measured yet; this section therefore "
                "makes no CPU comparison and no FPGA speedup claim."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _base_artifact(
    *,
    honest_verdict: str,
    duration_s: float,
    cited_upstream_artifacts: list[dict[str, Any]],
    blocked_reasons: list[str],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summary or {}
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "latex_snippet_path": SNIPPET_REL_PATH.as_posix(),
        "kv260_latency_cited_p50_us": float(summary.get("cited_p50_us", 0.0)),
        "kv260_latency_cited_p95_us": float(summary.get("cited_p95_us", 0.0)),
        "bitstream_sha256_cited": str(summary.get("bitstream_sha256", "")),
        "cited_upstream_artifacts": cited_upstream_artifacts,
        "blocked_reasons": blocked_reasons,
        "kv260_board": summary.get("board"),
        "kv260_overlay_name": summary.get("overlay_name"),
        "n_spins": summary.get("n_spins"),
        "per_seed_latency_rows": summary.get("per_seed_latencies", []),
        "snippet_written": not blocked_reasons,
        "main_tex_modified": False,
        "operator_only_external_publication": True,
        "run_date": RUN_DATE,
        "duration_s": round(max(0.0, duration_s), 6),
    }


def _latency_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "seed": int(row["seed"]),
            "n_samples": int(row["n_samples"]),
            "median_us": round(float(row["per_sample_wall_clock_us_median"]), 6),
            "p95_us": round(float(row["per_sample_wall_clock_us_p95"]), 6),
        }
        for row in sorted(payload["per_seed_results"], key=lambda item: int(item["seed"]))
    ]
    problem = payload.get("ising_problem_spec") if isinstance(payload.get("ising_problem_spec"), dict) else {}
    medians = [row["median_us"] for row in rows]
    p95s = [row["p95_us"] for row in rows]
    return {
        "board": BOARD_NAME,
        "overlay_name": str(payload["kv260_overlay_loaded"]),
        "bitstream_sha256": str(payload["bitstream_sha256"]),
        "bitstream_sha256_source": str(payload.get("bitstream_sha256_source", "board firmware")),
        "n_spins": int(problem.get("n_spins", 0)),
        "per_seed_latencies": rows,
        "cited_p50_us": round(float(median(medians)), 6),
        "cited_p95_us": round(max(p95s), 6),
    }


def _cited_upstream_artifacts(source_path: Path) -> list[dict[str, Any]]:
    if not source_path.is_file():
        return []
    return [
        {
            "experiment_id": "exp2898",
            "artifact_path": EXP2898_REL_PATH.as_posix(),
            "fields_imported": list(FIELDS_IMPORTED),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
    ]


def _terminal_success(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().startswith(("complete:", "success:"))


def _preconditions_available(payload: dict[str, Any]) -> bool:
    checked = payload.get("preconditions_checked")
    return isinstance(checked, list) and bool(checked) and all(
        isinstance(item, dict) and item.get("available") is True for item in checked
    )


def _positive_latency_row(row: object) -> bool:
    if not isinstance(row, dict):
        return False
    return (
        float(row.get("per_sample_wall_clock_us_median") or 0.0) > 0.0
        and float(row.get("per_sample_wall_clock_us_p95") or 0.0) > 0.0
    )


def _sample_failures(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    rows = payload.get("sample_count_sweep_results", [])
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            failed = int(row.get("failed_samples") or 0) if isinstance(row, dict) else 0
            if failed > 0:
                failures.append(f"sample_count_sweep_results[{index}].failed_samples={failed}")
    return failures


def _acceptance_gate_failures(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in ("acceptance_gates", "acceptance_gate_results", "gate_results"):
        entries = payload.get(key)
        if isinstance(entries, list):
            iterable = [(str(index), entry) for index, entry in enumerate(entries)]
        elif isinstance(entries, dict):
            iterable = [(str(name), entry) for name, entry in entries.items()]
        else:
            iterable = []
        failures.extend(f"{key}[{name}] failed" for name, entry in iterable if _gate_failed(entry))
    return failures


def _gate_failed(entry: object) -> bool:
    status = str(entry.get("status", "")).lower() if isinstance(entry, dict) else ""
    verdict = str(entry.get("verdict", "")).lower() if isinstance(entry, dict) else ""
    return isinstance(entry, dict) and (
        entry.get("passed") is False
        or entry.get("success") is False
        or status in {"failed", "failure", "error", "rejected", "blocked"}
        or verdict.startswith(("failed", "blocked", "error"))
    )


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


def _format_us(value: object) -> str:
    return f"{float(value):.2f}"


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> None:  # pragma: no cover - CLI convenience wrapper.
    print(write_outputs())


if __name__ == "__main__":  # pragma: no cover - CLI convenience wrapper.
    main()
