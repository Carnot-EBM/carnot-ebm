"""Compile the Exp 2389 paper-v6 real-data results table.

The paper-v6 results section needs a single reviewer-facing table that is
traceable to local experiment artifacts. This module reads the expected `.231`
and `.232` JSON files when they are present, records missing files explicitly,
adds the two external AUROC baselines as non-local comparison rows, writes the
markdown table, and emits the terminal Exp 2389 deliverable.

Spec refs: REQ-REPORT-2389, SCENARIO-REPORT-2389.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260518"
OUTPUT_FILENAME = "experiment_2389_paperv6_table.json"
DEFAULT_OUT_PATH = REPO_ROOT / "results" / OUTPUT_FILENAME
TABLE_REL_PATH = Path("docs/paper_v6_results_table.md")
DEFAULT_TABLE_PATH = REPO_ROOT / TABLE_REL_PATH
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
SCHEMA = "carnot.paper_v6_results_table.v1"
EXPERIMENT = "2389_paperv6_results_table"
HALLUSCAN_AUROC = 0.88
HIVE_EXTERNAL_AUROC = 0.9236

EXPECTED_SOURCE_ARTIFACTS: tuple[dict[str, str], ...] = (
    {
        "source_id": "exp2351",
        "path": "results/experiment_2351_semantic_energy_real.json",
        "milestone": "2026.05.230 carry-forward",
    },
    {
        "source_id": "exp2365",
        "path": "results/experiment_2365_fst_live_gen.json",
        "milestone": "2026.05.231",
    },
    {
        "source_id": "exp2366",
        "path": "results/experiment_2366_nsvif_verge_real.json",
        "milestone": "2026.05.231",
    },
    {
        "source_id": "exp2368",
        "path": "results/experiment_2368_laab_k17.json",
        "milestone": "2026.05.231",
    },
    {
        "source_id": "exp2369",
        "path": "results/experiment_2369_spilled_energy_k18.json",
        "milestone": "2026.05.231",
    },
    {
        "source_id": "exp2372",
        "path": "results/experiment_2372_kv260_rtl_fix.json",
        "milestone": "2026.05.231",
    },
    {
        "source_id": "exp2374",
        "path": "results/experiment_2374_kancl_hard_domains.json",
        "milestone": "2026.05.231",
    },
    {
        "source_id": "exp2375",
        "path": "results/experiment_2375_fr11_real_csl.json",
        "milestone": "2026.05.231",
    },
    {
        "source_id": "exp2379",
        "path": "results/experiment_2379_halt_tier0j.json",
        "milestone": "2026.05.232",
    },
    {
        "source_id": "exp2380",
        "path": "results/experiment_2380_hive_ensemble.json",
        "milestone": "2026.05.232",
    },
    {
        "source_id": "exp2382",
        "path": "results/experiment_2382_fst_live_path_ab.json",
        "milestone": "2026.05.232",
    },
    {
        "source_id": "exp2384",
        "path": "results/experiment_2384_kv260_yosys.json",
        "milestone": "2026.05.232",
    },
)


@dataclass(frozen=True)
class MetricDefinition:
    """Configuration for extracting one table row from one source artifact."""

    metric_name: str
    source_id: str
    value_fields: tuple[str, ...]
    n_fields: tuple[str, ...]
    methodology_when_missing: str
    value_kind: str = "scalar"
    baseline_label: str = "none"
    baseline_value: float | None = None


METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        metric_name="Tier 0g SemanticEnergy AUROC",
        source_id="exp2351",
        value_fields=("semantic_energy_real_auroc",),
        n_fields=("source_rows_usable", "n_real_examples", "n_examples", "n_eval_examples"),
        methodology_when_missing="cached live GGUF top-k logprob telemetry",
        value_kind="auroc",
        baseline_label="HalluScan AUROC 0.88",
        baseline_value=HALLUSCAN_AUROC,
    ),
    MetricDefinition(
        metric_name="Tier 0h LaaB AUROC",
        source_id="exp2368",
        value_fields=("laab_k17_auroc", "laab_auroc", "auroc"),
        n_fields=("n_real_examples", "n_eval_examples", "n_examples", "source_rows_usable"),
        methodology_when_missing="LaaB assertion-alignment NLI on real model outputs",
        value_kind="auroc",
        baseline_label="HalluScan AUROC 0.88",
        baseline_value=HALLUSCAN_AUROC,
    ),
    MetricDefinition(
        metric_name="Tier 0i SpilledEnergy k=18 AUROC",
        source_id="exp2369",
        value_fields=("spilled_energy_k18_auroc", "spilled_energy_auroc", "auroc"),
        n_fields=("n_real_examples", "n_eval_examples", "n_examples", "source_rows_usable"),
        methodology_when_missing="SpilledEnergy k=18 logprob-compatible verifier on real outputs",
        value_kind="auroc",
        baseline_label="HalluScan AUROC 0.88",
        baseline_value=HALLUSCAN_AUROC,
    ),
    MetricDefinition(
        metric_name="Tier 0j HALT AUROC",
        source_id="exp2379",
        value_fields=("halt_k19j_auroc", "halt_tier0j_auroc", "halt_auroc", "auroc"),
        n_fields=("n_real_examples", "n_eval_examples", "n_examples", "source_rows_usable"),
        methodology_when_missing="HALT latent-probe verifier on real telemetry outputs",
        value_kind="auroc",
        baseline_label="HalluScan AUROC 0.88",
        baseline_value=HALLUSCAN_AUROC,
    ),
    MetricDefinition(
        metric_name="HIVE 4-verifier ensemble AUROC",
        source_id="exp2380",
        value_fields=("ensemble_auroc_4verifier", "hive_ensemble_auroc", "auroc"),
        n_fields=("n_real_examples", "n_eval_examples", "n_examples", "source_rows_usable"),
        methodology_when_missing="4-verifier ensemble over Tier 0g, 0h, 0i, and 0j",
        value_kind="auroc",
        baseline_label="HIVE external AUROC 0.9236",
        baseline_value=HIVE_EXTERNAL_AUROC,
    ),
    MetricDefinition(
        metric_name="NSVIF verification pass rate",
        source_id="exp2366",
        value_fields=(
            "verification_pass_rate",
            "nsvif_real_verification_pass_rate",
            "nsvif_verification_pass_rate",
        ),
        n_fields=("n_real_examples", "n_eval_examples", "n_examples", "n_outputs"),
        methodology_when_missing="NSVIF Z3 verification over real Qwen3.6-35B outputs",
    ),
    MetricDefinition(
        metric_name="VERGE repair success rate",
        source_id="exp2366",
        value_fields=(
            "repair_success_rate",
            "verge_real_repair_success_rate",
            "verge_repair_success_rate",
        ),
        n_fields=("n_real_examples", "n_eval_examples", "n_examples", "n_outputs"),
        methodology_when_missing="VERGE repair over NSVIF-detected real-output violations",
    ),
    MetricDefinition(
        metric_name="KV260 RTL lint errors",
        source_id="exp2372",
        value_fields=("lint_errors_count", "lint_errors_count_after_fix"),
        n_fields=(),
        methodology_when_missing="Verilator lint on KV260 Ising RTL after reserved-keyword fix",
    ),
    MetricDefinition(
        metric_name="KV260 Yosys synthesis succeeded",
        source_id="exp2384",
        value_fields=("synthesis_succeeded",),
        n_fields=(),
        methodology_when_missing="Yosys generic or Xilinx synthesis of lint-clean KV260 RTL",
    ),
    MetricDefinition(
        metric_name="KAN-CL hard-domain forgetting reduction pct",
        source_id="exp2374",
        value_fields=("kancl_hard_forgetting_reduction_pct", "forgetting_reduction_pct"),
        n_fields=("n_examples", "n_eval_examples", "n_domains", "n_tasks"),
        methodology_when_missing="KAN-CL B-spline continual-learning stress test on hard domains",
    ),
    MetricDefinition(
        metric_name="FST cached-telemetry validation",
        source_id="exp2365",
        value_fields=("fst_live_validated",),
        n_fields=(
            "n_real_examples",
            "n_eval_examples",
            "n_examples",
            "n_live_traces",
            "source_rows_usable",
        ),
        methodology_when_missing="FST PATH C cached telemetry validation on real model outputs",
    ),
    MetricDefinition(
        metric_name="FST live inference completed",
        source_id="exp2382",
        value_fields=("live_inference_completed",),
        n_fields=("n_live_traces", "n_real_examples", "n_eval_examples", "n_examples"),
        methodology_when_missing="FST PATH A/B live llama.cpp or transformers inference attempt",
    ),
    MetricDefinition(
        metric_name="FR-11 cross-domain retention rate",
        source_id="exp2375",
        value_fields=("cross_domain_retention_rate", "fr11_cross_domain_retention_rate"),
        n_fields=("n_real_examples", "n_eval_examples", "n_examples", "n_domains", "n_outputs"),
        methodology_when_missing="FR-11 fast/slow cross-domain retention on real GGUF outputs",
    ),
)

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "n_paper_ready_results",
    "best_auroc_achieved",
    "hallscan_gap",
    "results_table_written",
    "n_missing_results",
    "duration_s",
}


def collect_source_status(root: str | Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """Load each expected source artifact and record available/missing status."""

    root_path = Path(root)
    status: dict[str, dict[str, Any]] = {}
    for source in EXPECTED_SOURCE_ARTIFACTS:
        rel_path = source["path"]
        path = root_path / rel_path
        row: dict[str, Any] = {
            "source_id": source["source_id"],
            "path": rel_path,
            "milestone": source["milestone"],
            "available": path.is_file(),
            "status": "missing",
        }
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                row.update({"status": "parse_error", "parse_error": str(exc), "payload": None})
            else:
                row.update(
                    {
                        "status": "available",
                        "payload": payload,
                        "honest_verdict": payload.get("honest_verdict"),
                    }
                )
        status[source["source_id"]] = row
    return status


def collect_metric_rows(
    root: str | Path = REPO_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return paper-v6 metric rows plus source-artifact availability records."""

    source_status = collect_source_status(root)
    rows = [_extract_metric_row(defn, source_status) for defn in METRIC_DEFINITIONS]
    rows.extend(_external_baseline_rows())
    return rows, source_status


def find_metric(rows: Sequence[Mapping[str, Any]], metric_name: str) -> Mapping[str, Any]:
    """Find one metric row by exact display name."""

    for row in rows:
        if row.get("metric_name") == metric_name:
            return row
    raise KeyError(metric_name)


def compute_summary(
    rows: Sequence[Mapping[str, Any]],
    source_status: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute paper-ready count, missing-artifact count, and AUROC gap summary."""

    local_auroc_values = [
        float(row["metric_value"])
        for row in rows
        if row.get("value_kind") == "auroc"
        and row.get("source_status") == "available"
        and not row.get("is_external_baseline")
        and _is_number(row.get("metric_value"))
    ]
    best_auroc = max(local_auroc_values) if local_auroc_values else None
    missing_sources = [row for row in source_status.values() if row["status"] != "available"]
    missing_232 = [row for row in missing_sources if str(row.get("milestone")) == "2026.05.232"]
    return {
        "n_paper_ready_results": sum(
            1
            for row in rows
            if row.get("paper_ready") is True and not row.get("is_external_baseline")
        ),
        "n_missing_results": len(missing_sources),
        "n_missing_232_results": len(missing_232),
        "best_auroc_achieved": _round_metric(best_auroc),
        "hallscan_gap": _round_metric(HALLUSCAN_AUROC - best_auroc)
        if best_auroc is not None
        else None,
    }


def render_markdown_table(
    rows: Sequence[Mapping[str, Any]],
    source_status: Mapping[str, Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> str:
    """Render the reviewer-facing paper-v6 results-table markdown document."""

    lines = [
        "# Paper v6 Real-Data Results Table",
        "",
        f"Run date: `{RUN_DATE}`",
        "",
        "## Summary",
        "",
        f"- Paper-ready local results: `{summary['n_paper_ready_results']}`",
        f"- Missing expected source artifacts: `{summary['n_missing_results']}`",
        f"- Missing `.232` source artifacts: `{summary['n_missing_232_results']}`",
        f"- Best Carnot Tier 0 AUROC achieved: `{_format_value(summary['best_auroc_achieved'])}`",
        f"- HalluScan gap (`0.88 - best_auroc_achieved`): `{_format_value(summary['hallscan_gap'])}`",
        "",
        "## Results Table",
        "",
        "| metric_name | value | n_examples | paper_ready | external_baseline | gap_to_baseline |",
        "|---|---:|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {metric_name} | {value} | {n_examples} | {paper_ready} | {baseline} | {gap} |".format(
                metric_name=_md_cell(str(row["metric_name"])),
                value=_format_value(row.get("metric_value")),
                n_examples=_format_value(row.get("n_examples")),
                paper_ready=str(bool(row["paper_ready"])).lower(),
                baseline=_md_cell(str(row["external_baseline"])),
                gap=_format_value(row.get("gap_to_baseline")),
            )
        )
    lines.extend(
        [
            "",
            "## Methodology Notes",
            "",
            "| metric_name | source_artifact | source_status | methodology_note | adversarial_cleared |",
            "|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {metric_name} | {source} | {status} | {note} | {cleared} |".format(
                metric_name=_md_cell(str(row["metric_name"])),
                source=_md_cell(str(row["source_artifact"])),
                status=_md_cell(str(row["source_status"])),
                note=_md_cell(str(row["methodology_note"])),
                cleared=str(bool(row["adversarial_cleared"])).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Source Artifact Availability",
            "",
            "| source_id | path | milestone | status |",
            "|---|---|---|---|",
        ]
    )
    for source_id in sorted(source_status):
        row = source_status[source_id]
        lines.append(
            "| {source_id} | {path} | {milestone} | {status} |".format(
                source_id=_md_cell(source_id),
                path=_md_cell(str(row["path"])),
                milestone=_md_cell(str(row["milestone"])),
                status=_md_cell(str(row["status"])),
            )
        )
    return "\n".join(lines) + "\n"


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_status: Mapping[str, Mapping[str, Any]],
    table_path: str,
    results_table_written: bool,
    duration_s: float,
) -> dict[str, Any]:
    """Build and validate the terminal Exp 2389 deliverable payload."""

    summary = compute_summary(rows, source_status)
    source_status_for_json = {
        source_id: {key: value for key, value in row.items() if key not in {"payload"}}
        for source_id, row in source_status.items()
    }
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "spec": ["REQ-REPORT-2389", "SCENARIO-REPORT-2389"],
        "status": "complete",
        "results_table_path": table_path,
        "results_table_written": results_table_written,
        "results_table": [dict(row) for row in rows],
        "source_artifact_status": source_status_for_json,
        "external_baselines": {
            "halluscan_auroc": HALLUSCAN_AUROC,
            "halluscan_source": "arXiv:2605.02443",
            "hive_external_auroc": HIVE_EXTERNAL_AUROC,
            "hive_external_source": "arXiv:2604.26139",
        },
        "field_principles": {
            "honest_verdict": "Terminal-prefix required. complete: with n_paper_ready_results.",
            "n_paper_ready_results": "Count of results suitable for paper-v6 citation (n>=30, adversarial-cleared).",
            "best_auroc_achieved": "Highest AUROC across all Carnot Tier 0 verifiers. Tracks gap closure.",
            "hallscan_gap": "0.88 - best_auroc_achieved. Tracks competitive baseline gap.",
            "results_table_written": "True if docs/paper_v6_results_table.md was created/updated.",
            "n_missing_results": "Count of expected source results not yet available. Honest accounting.",
            "duration_s": "Guards against fabrication.",
        },
        "acceptance_gates": {
            "results_table_written": results_table_written is True,
        },
        "n_paper_ready_results": summary["n_paper_ready_results"],
        "n_missing_results": summary["n_missing_results"],
        "n_missing_232_results": summary["n_missing_232_results"],
        "best_auroc_achieved": summary["best_auroc_achieved"],
        "hallscan_gap": summary["hallscan_gap"],
        "duration_s": duration_s,
        "honest_verdict": (
            "complete: n_paper_ready_results={n}; best_auroc_achieved={best}; "
            "hallscan_gap={gap}; n_missing_results={missing}"
        ).format(
            n=summary["n_paper_ready_results"],
            best=_format_value(summary["best_auroc_achieved"]),
            gap=_format_value(summary["hallscan_gap"]),
            missing=summary["n_missing_results"],
        ),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 2389 schema invariants."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete")
    if artifact["results_table_written"] is not True:
        raise ValueError("results_table_written must be true")
    if not str(artifact["honest_verdict"]).startswith("complete: n_paper_ready_results="):
        raise ValueError("honest_verdict must include terminal complete prefix and count")
    if artifact["duration_s"] < 0:
        raise ValueError("duration_s must be non-negative")
    if artifact["hallscan_gap"] is not None and artifact["best_auroc_achieved"] is None:
        raise ValueError("hallscan_gap cannot exist without best_auroc_achieved")


def run(
    *,
    root: str | Path = REPO_ROOT,
    out_path: str | Path = DEFAULT_OUT_PATH,
    table_path: str | Path | None = None,
    duration_override_s: float | None = None,
) -> dict[str, Any]:
    """Compile rows, write markdown, and emit the terminal Exp 2389 artifact."""

    start = time.perf_counter()
    root_path = Path(root)
    output_path = Path(out_path)
    markdown_path = Path(table_path) if table_path is not None else root_path / TABLE_REL_PATH

    rows, source_status = collect_metric_rows(root_path)
    summary = compute_summary(rows, source_status)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown_table(rows, source_status, summary), encoding="utf-8")

    duration_s = (
        float(duration_override_s)
        if duration_override_s is not None
        else round(max(time.perf_counter() - start, 0.0), 6)
    )
    artifact = build_artifact(
        rows=rows,
        source_status=source_status,
        table_path=_relative_path(markdown_path, root_path),
        results_table_written=markdown_path.is_file(),
        duration_s=duration_s,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _write_json(output_path, artifact)


def _extract_metric_row(
    defn: MetricDefinition,
    source_status: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source = source_status[defn.source_id]
    payload = source.get("payload") if source.get("status") == "available" else None
    value = (
        _first_nested_value(payload, defn.value_fields) if isinstance(payload, Mapping) else None
    )
    n_examples = _first_nested_int(payload, defn.n_fields) if isinstance(payload, Mapping) else None
    adversarial_cleared = bool(
        payload is not None and value is not None and not _contains_implausible_perfect(payload)
    )
    paper_ready = bool(n_examples is not None and n_examples >= 30 and adversarial_cleared)
    gap = (
        _round_metric(float(defn.baseline_value) - float(value))
        if defn.baseline_value is not None and _is_number(value)
        else None
    )
    return {
        "metric_name": defn.metric_name,
        "metric_value": _json_metric(value),
        "n_examples": n_examples,
        "methodology_note": _methodology_note(defn, payload, n_examples),
        "adversarial_cleared": adversarial_cleared,
        "paper_ready": paper_ready,
        "external_baseline": defn.baseline_label,
        "gap_to_baseline": gap,
        "source_artifact": source["path"],
        "source_status": source["status"],
        "source_id": defn.source_id,
        "value_kind": defn.value_kind,
        "is_external_baseline": False,
    }


def _external_baseline_rows() -> list[dict[str, Any]]:
    return [
        {
            "metric_name": "HalluScan external baseline AUROC",
            "metric_value": HALLUSCAN_AUROC,
            "n_examples": None,
            "methodology_note": "Published external HalluScan baseline from arXiv:2605.02443; not re-evaluated locally.",
            "adversarial_cleared": True,
            "paper_ready": False,
            "external_baseline": "external baseline",
            "gap_to_baseline": None,
            "source_artifact": "arXiv:2605.02443",
            "source_status": "external",
            "source_id": "halluscan_external",
            "value_kind": "auroc",
            "is_external_baseline": True,
        },
        {
            "metric_name": "HIVE external baseline AUROC",
            "metric_value": HIVE_EXTERNAL_AUROC,
            "n_examples": None,
            "methodology_note": "Published external HIVE baseline from arXiv:2604.26139; not re-evaluated locally.",
            "adversarial_cleared": True,
            "paper_ready": False,
            "external_baseline": "external baseline",
            "gap_to_baseline": None,
            "source_artifact": "arXiv:2604.26139",
            "source_status": "external",
            "source_id": "hive_external",
            "value_kind": "auroc",
            "is_external_baseline": True,
        },
    ]


def _methodology_note(
    defn: MetricDefinition,
    payload: Mapping[str, Any] | None,
    n_examples: int | None,
) -> str:
    if payload is None:
        return f"missing source artifact; planned method: {defn.methodology_when_missing}"
    if defn.source_id == "exp2351":
        n_text = n_examples if n_examples is not None else "unknown"
        model_names = payload.get("model_names") or payload.get("hf_ids") or ["Qwen3.6-35B-A3B"]
        logit_source = payload.get("logit_source", "cached top-k logprobs")
        return (
            f"{n_text} usable cached live {', '.join(map(str, model_names))} telemetry rows; "
            f"AUROC computed from {logit_source}."
        )
    for key in ("methodology_note", "evaluation_design", "precondition_cache_check", "title"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if n_examples is not None:
        return f"{defn.methodology_when_missing}; n_examples={n_examples}"
    return defn.methodology_when_missing


def _first_nested_value(payload: Mapping[str, Any] | None, fields: Sequence[str]) -> Any | None:
    if payload is None:
        return None
    for field in fields:
        found = _find_nested_key(payload, field)
        if found is not None:
            return found
    return None


def _first_nested_int(payload: Mapping[str, Any] | None, fields: Sequence[str]) -> int | None:
    value = _first_nested_value(payload, fields)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _find_nested_key(value: Any, target_key: str) -> Any | None:
    if isinstance(value, Mapping):
        if target_key in value:
            return value[target_key]
        for child in value.values():
            found = _find_nested_key(child, target_key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_nested_key(child, target_key)
            if found is not None:
                return found
    return None


def _contains_implausible_perfect(payload: Mapping[str, Any]) -> bool:
    return "IMPLAUSIBLE_PERFECT" in json.dumps(payload, sort_keys=True)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _json_metric(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if _is_number(value):
        return _round_metric(float(value))
    return value


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return str(value)


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    serializable = deepcopy(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return serializable


if __name__ == "__main__":
    run()
