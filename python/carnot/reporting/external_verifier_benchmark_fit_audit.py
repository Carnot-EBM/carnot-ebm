"""Build the Exp 1465 external verifier benchmark fit audit artifact.

This module keeps the .112 scope-reduction task deliberately small.  It does
not run VNN-COMP, clone benchmark suites, or create a new runner.  It records
which external verifier comparison is the best next fit for Carnot's current
evidence chain and writes a reviewer-facing markdown note plus the terminal
JSON artifact.

Spec: REQ-REPORT-046, SCENARIO-REPORT-046.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260507"
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
SCHEMA = "external_verifier_benchmark_fit_audit_v1"
EXPERIMENT = "1465_external_verifier_benchmark_fit_audit"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1465_external_verifier_benchmark_fit_audit.json"
)
DEFAULT_DECISION_NOTE_PATH = (
    REPO_ROOT / "docs" / "research-notes" / "external_verifier_benchmark_fit.md"
)

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "benchmarks_reviewed",
    "benchmark_decision_table_path",
    "benchmark_adoption_decision",
    "adopted_benchmark",
    "deferred_benchmarks",
    "retired_benchmarks",
    "next_minimal_benchmark_task",
    "honest_verdict",
}

REQUIRED_BENCHMARKS_REVIEWED = [
    "VNNLIB/VNN-COMP",
    "BEAVER-style deterministic bounds",
    "smaller existing benchmark",
]

EXTERNAL_SOURCES_REVIEWED = [
    {
        "name": "VNN-COMP official site",
        "url": "https://vnn-comp.github.io/",
        "fit_signal": (
            "standardized neural-network verification competition with ONNX "
            "networks and VNN-LIB specs; benchmark proposers provide ONNX "
            "networks and VNN-LIB specifications."
        ),
    },
    {
        "name": "VNN-LIB official standard page",
        "url": "https://www.vnnlib.org/",
        "fit_signal": (
            "VNN-LIB 2.0 and official parsers are available, and ONNX is the "
            "model format VNN-LIB relies on."
        ),
    },
    {
        "name": "VNNLIB benchmarks repository",
        "url": "https://github.com/vnnlib/benchmarks",
        "fit_signal": (
            "benchmarks are organized around fully connected, convolutional, "
            "and residual ONNX network families with expected-result CSVs."
        ),
    },
    {
        "name": "BEAVER OpenReview/arXiv",
        "url": "https://openreview.net/forum?id=xO3efBXHM9",
        "fit_signal": (
            "deterministic probability bounds for prefix-closed semantic "
            "constraints on LLM outputs map directly onto Carnot's existing "
            "BEAVER-lite certificate tier."
        ),
    },
]


@dataclass(frozen=True)
class BenchmarkDecision:
    benchmark: str
    decision: str
    rationale: str
    fit_risks: str
    next_or_reopen_condition: str
    source_basis: str


DEFAULT_DECISIONS = [
    BenchmarkDecision(
        benchmark="VNNLIB/VNN-COMP",
        decision="defer",
        rationale=(
            "Credible external neural-network verification standard, but the "
            "current Carnot comparison target is LLM-output semantic bounds. "
            "VNN-COMP expects ONNX networks plus VNN-LIB properties, while "
            "Carnot has not yet exported the relevant verifier comparison as "
            "a small ONNX property instance."
        ),
        fit_risks=(
            "A broad VNN-COMP runner would expand .112 scope, mix neural-network "
            "robustness verification with LLM semantic-output verification, and "
            "risk producing an integration artifact instead of a fit decision."
        ),
        next_or_reopen_condition=(
            "Reopen after Carnot has a single checked-in ONNX verifier or energy "
            "network plus one VNN-LIB property that represents a real Carnot "
            "claim boundary."
        ),
        source_basis="VNN-COMP official site, VNN-LIB official standard page",
    ),
    BenchmarkDecision(
        benchmark="BEAVER-style deterministic bounds",
        decision="adopt",
        rationale=(
            "Best immediate fit: BEAVER's prefix-closed semantic-constraint "
            "bounds match Carnot's certificate and false-acceptance-bound need, "
            "and the repo already contains BEAVER-lite bounder tests and "
            "artifacts that can support a tiny smoke comparison."
        ),
        fit_risks=(
            "The next task must remain a bounded smoke check over existing "
            "BEAVER-lite code.  It must not claim full BEAVER reproduction, "
            "secure-code coverage, privacy coverage, or broad LLM safety bounds."
        ),
        next_or_reopen_condition=(
            "Adopt only one minimal BEAVER-lite external-bounds smoke task with "
            "three deterministic arithmetic prompts and explicit mock/live "
            "logprob provenance."
        ),
        source_basis=(
            "BEAVER OpenReview/arXiv, python/carnot/verify/beaver_lite.py, "
            "tests/python/test_beaver_lite.py"
        ),
    ),
    BenchmarkDecision(
        benchmark="smaller existing benchmark",
        decision="defer",
        rationale=(
            "Existing local micro-benchmarks are useful regressions, but they "
            "do not provide an external verifier comparison by themselves.  "
            "They should support the adopted BEAVER-style task only after the "
            "minimal bound artifact exists."
        ),
        fit_risks=(
            "Adopting a generic local benchmark would blur the external-verifier "
            "comparison question and could repeat scope-expansion patterns that "
            ".112 is explicitly reducing."
        ),
        next_or_reopen_condition=(
            "Reconsider after the BEAVER-lite smoke produces a terminal artifact "
            "and the next comparison needs a regression corpus rather than a "
            "new external method."
        ),
        source_basis=(
            "openspec/capabilities/benchmarks/spec.md and existing BEAVER-lite "
            "artifact/test coverage"
        ),
    ),
]


def default_decisions() -> list[BenchmarkDecision]:
    """Return the fixed Exp 1465 fit decisions for the reviewed families."""

    return list(DEFAULT_DECISIONS)


def write_in_progress_artifact(out_path: str | Path = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-046: write the durable startup artifact before the audit."""

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "spec": ["REQ-REPORT-046", "SCENARIO-REPORT-046"],
        "status": "in_progress",
        "benchmarks_reviewed": [],
        "benchmark_decision_table_path": _relative_path(DEFAULT_DECISION_NOTE_PATH),
        "benchmark_adoption_decision": "pending",
        "adopted_benchmark": None,
        "deferred_benchmarks": [],
        "retired_benchmarks": [],
        "next_minimal_benchmark_task": None,
        "honest_verdict": "in_progress",
    }
    return _write_json(Path(out_path), artifact)


def next_minimal_beaver_task() -> dict[str, Any]:
    """Return the single allowed follow-up task for the adopted benchmark.

    The task is intentionally a smoke comparison, not a new benchmark suite.
    It reuses existing BEAVER-lite inputs and records whether the reported
    probability bounds are sound under mock or live logprob provenance.
    """

    return {
        "task_id": "exp_next_beaver_lite_external_bounds_smoke",
        "benchmark_family": "BEAVER-style deterministic bounds",
        "inputs": [
            "python/carnot/verify/beaver_lite.py",
            "tests/python/test_beaver_lite.py",
            "tests/python/test_beaver_lite_live_logprobs.py",
            "results/experiment_1142_beaver_lite_certificate_tier.json",
            "results/experiment_1158_beaver_lite_live_logprobs.json",
        ],
        "expected_artifact_fields": [
            "status",
            "benchmark_family",
            "questions_evaluated",
            "prefix_closed_constraint",
            "unsafe_mass_bound",
            "empirical_violation_rate",
            "bound_is_sound",
            "mock_or_live_logprobs",
            "external_fit_verdict",
            "honest_verdict",
        ],
        "e2e_check": (
            "run the existing BEAVER-lite bounder over three deterministic "
            "GSM8K-style arithmetic prompts and assert every reported unsafe "
            "mass bound is in [0, 1], sound, and labeled mock_or_live_logprobs"
        ),
        "scope_limit": (
            "No VNN-COMP runner, no broad BEAVER reproduction, no fresh LLM "
            "benchmark; this is a terminal smoke artifact only."
        ),
    }


def build_artifact(
    *,
    decisions: list[BenchmarkDecision],
    decision_table_path: str,
) -> dict[str, Any]:
    """Build the terminal fit-audit artifact from the reviewed decisions."""

    adopted = [row.benchmark for row in decisions if row.decision == "adopt"]
    deferred = [row.benchmark for row in decisions if row.decision == "defer"]
    retired = [row.benchmark for row in decisions if row.decision == "retire"]
    task: dict[str, Any] | None = next_minimal_beaver_task() if adopted else None
    adopted_benchmark = (
        "BEAVER-style deterministic bounds via existing BEAVER-lite smoke"
        if adopted == ["BEAVER-style deterministic bounds"]
        else (adopted[0] if adopted else None)
    )

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "spec": ["REQ-REPORT-046", "SCENARIO-REPORT-046"],
        "status": "complete",
        "benchmarks_reviewed": [row.benchmark for row in decisions],
        "benchmark_decision_table_path": decision_table_path,
        "benchmark_adoption_decision": _benchmark_adoption_decision(
            adopted=adopted,
            deferred=deferred,
            retired=retired,
        ),
        "adopted_benchmark": adopted_benchmark,
        "deferred_benchmarks": deferred,
        "retired_benchmarks": retired,
        "next_minimal_benchmark_task": task,
        "honest_verdict": _honest_verdict(adopted=adopted, deferred=deferred, retired=retired),
        "decision_rows": [asdict(row) for row in decisions],
        "external_sources_reviewed": EXTERNAL_SOURCES_REVIEWED,
        "source_files_checked": [
            "CODEX.md",
            "CLAUDE.md",
            "research-references.md",
            "docs/research-notes/comparator_cite_retire_audit.md",
            "openspec/capabilities/benchmarks/spec.md",
            "openspec/capabilities/verification/spec.md",
            "openspec/capabilities/verifiable-reasoning/spec.md",
            "tests/python/test_beaver_lite.py",
            "tests/python/test_beaver_lite_live_logprobs.py",
            "_bmad/prd.md",
            "ops/e2e-test-plan.md",
        ],
        "scope_note": (
            "This is a fit audit only.  It adopts one minimal BEAVER-style "
            "bounds smoke because that is already aligned with Carnot's "
            "certificate code and .112 scope reduction; it does not implement "
            "a VNN-COMP or broad external benchmark runner."
        ),
    }
    validate_artifact(artifact)
    return artifact


def render_decision_note(
    decisions: list[BenchmarkDecision],
    next_minimal_task: Mapping[str, Any] | None,
) -> str:
    """Render the markdown decision note for reviewer and conductor use."""

    lines = [
        "# External Verifier Benchmark Fit Audit",
        "",
        f"Run date: `{RUN_DATE}`",
        "",
        "## Decision Table",
        "",
        "| Benchmark Family | Decision | Rationale | Fit Risks | Next/Reopen Condition |",
        "|---|---|---|---|---|",
    ]
    for row in decisions:
        lines.append(
            "| {benchmark} | {decision} | {rationale} | {fit_risks} | {condition} |".format(
                benchmark=_md_cell(row.benchmark),
                decision=_md_cell(row.decision),
                rationale=_md_cell(row.rationale),
                fit_risks=_md_cell(row.fit_risks),
                condition=_md_cell(row.next_or_reopen_condition),
            )
        )

    lines.extend(
        [
            "",
            "## External Sources Reviewed",
            "",
        ]
    )
    for source in EXTERNAL_SOURCES_REVIEWED:
        lines.append(f"- `{source['name']}`: {source['url']} - {source['fit_signal']}")

    lines.extend(
        [
            "",
            "## Next Minimal Benchmark Task",
            "",
        ]
    )
    if next_minimal_task is None:
        lines.append(
            "No benchmark family was adopted, so no future benchmark task is defined."
        )
    else:
        lines.extend(
            [
                f"- `task_id`: {next_minimal_task['task_id']}",
                f"- `benchmark_family`: {next_minimal_task['benchmark_family']}",
                "- `inputs`: " + ", ".join(f"`{item}`" for item in next_minimal_task["inputs"]),
                "- `expected_artifact_fields`: "
                + ", ".join(f"`{item}`" for item in next_minimal_task["expected_artifact_fields"]),
                f"- `e2e_check`: {next_minimal_task['e2e_check']}",
                f"- `scope_limit`: {next_minimal_task['scope_limit']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Honest Verdict",
            "",
            (
                "Adopt BEAVER-style deterministic bounds for one future "
                "BEAVER-lite smoke comparison; defer VNNLIB/VNN-COMP and the "
                "generic smaller existing benchmark option until they have a "
                "tighter Carnot-specific acceptance object."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 1465 schema and one-adoption gate."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete")
    reviewed = artifact["benchmarks_reviewed"]
    if reviewed != REQUIRED_BENCHMARKS_REVIEWED:
        raise ValueError(f"unexpected benchmarks_reviewed: {reviewed}")
    adopted = artifact["adopted_benchmark"]
    if adopted and not artifact["next_minimal_benchmark_task"]:
        raise ValueError("adopted benchmark requires next_minimal_benchmark_task")
    if adopted and len([row for row in artifact["decision_rows"] if row["decision"] == "adopt"]) != 1:
        raise ValueError("exactly one benchmark family may be adopted")
    if not artifact["benchmark_decision_table_path"]:
        raise ValueError("benchmark_decision_table_path must be set")


def run(
    *,
    root: str | Path = REPO_ROOT,
    out_path: str | Path = DEFAULT_OUT_PATH,
    decision_note_path: str | Path = DEFAULT_DECISION_NOTE_PATH,
    decisions: list[BenchmarkDecision] | None = None,
) -> dict[str, Any]:
    """Write the Exp 1465 note and terminal artifact without broad benchmarks."""

    root_path = Path(root)
    output = Path(out_path)
    note_path = Path(decision_note_path)
    rows = default_decisions() if decisions is None else list(decisions)

    write_in_progress_artifact(output)
    artifact = build_artifact(
        decisions=rows,
        decision_table_path=_relative_path(note_path, root_path),
    )
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        render_decision_note(rows, artifact["next_minimal_benchmark_task"]),
        encoding="utf-8",
    )
    return _write_json(output, artifact)


def _benchmark_adoption_decision(
    *,
    adopted: list[str],
    deferred: list[str],
    retired: list[str],
) -> str:
    if adopted == ["BEAVER-style deterministic bounds"] and deferred == [
        "VNNLIB/VNN-COMP",
        "smaller existing benchmark",
    ] and not retired:
        return (
            "adopt_beaver_style_deterministic_bounds_smoke; "
            "defer_vnnlib_vnncomp_and_smaller_existing_benchmark"
        )
    if not adopted:
        return "no_adoption_all_deferred_or_retired"
    return "custom_decision_review_required"


def _honest_verdict(*, adopted: list[str], deferred: list[str], retired: list[str]) -> str:
    if adopted == ["BEAVER-style deterministic bounds"] and len(deferred) == 2 and not retired:
        return "adopt_one_minimal_beaver_bounds_smoke"
    if not adopted:
        return "no_external_benchmark_adopted"
    return "manual_review_required"


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _relative_path(path: Path, root: Path = REPO_ROOT) -> str:
    return os.path.relpath(path, root)


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


__all__ = [
    "BenchmarkDecision",
    "REQUIRED_ARTIFACT_FIELDS",
    "REQUIRED_BENCHMARKS_REVIEWED",
    "build_artifact",
    "default_decisions",
    "next_minimal_beaver_task",
    "render_decision_note",
    "run",
    "validate_artifact",
    "write_in_progress_artifact",
]
