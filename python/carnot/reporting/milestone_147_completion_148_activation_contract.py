"""Build the Exp 1890 `.147` completion to `.148` activation contract.

Spec: REQ-REPORT-1890, SCENARIO-REPORT-1890.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260512"
PREDECESSOR_MILESTONE = "2026.05.147"
TARGET_MILESTONE = "2026.05.148"
EXPERIMENT = "1890_147_completion_148_activation_contract"
SCHEMA = "carnot.milestone_147_completion_148_activation_contract.v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1890_147_completion_148_activation_contract.json"
)

SOURCE_FILES = {
    "exp1876": "experiment_1876_146_completion_147_gate_contract.json",
    "exp1877": "experiment_1877_artifact_contract_normalization.json",
    "exp1878": "experiment_1878_roce_validator_tree.json",
    "exp1879": "experiment_1879_beaver_lite_bounds.json",
    "exp1889": "experiment_1889_milestone_147_retro.json",
    "operational_retro_147": "operational_retro_2026_05_147.json",
}

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "honest_verdict",
    "milestone_147_archived",
    "validator_tree_ready",
    "beaver_bounds_ready",
    "live_sota_blocked_missing_models",
    "telemetry_missing_terminal_artifact",
    "fr11_ledger_missing_terminal_artifact",
    "hardware_accounting_missing_terminal_artifact",
    "same_title_compute_dedupe_required",
    "next_gate_contract_ready",
    "tests_run",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-1890: persist a started marker before source evidence reads."""

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "milestone": TARGET_MILESTONE,
        "predecessor_milestone": PREDECESSOR_MILESTONE,
        "status": "in_progress",
        "honest_verdict": "in_progress",
        "milestone_147_archived": False,
        "validator_tree_ready": False,
        "beaver_bounds_ready": False,
        "live_sota_blocked_missing_models": False,
        "telemetry_missing_terminal_artifact": False,
        "fr11_ledger_missing_terminal_artifact": False,
        "hardware_accounting_missing_terminal_artifact": False,
        "same_title_compute_dedupe_required": False,
        "next_gate_contract_ready": False,
        "tests_run": [],
    }
    return _write_json(Path(out_path), artifact)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_text(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _load_sources(results_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    sources: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for source_id, filename in SOURCE_FILES.items():
        payload = _read_json(results_dir / filename)
        if payload is None:
            missing.append(source_id)
        else:
            sources[source_id] = payload
    return sources, missing


def _source_path(source_id: str) -> str:
    return f"results/{SOURCE_FILES[source_id]}"


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status") or "").lower()


def _is_complete(payload: Mapping[str, Any]) -> bool:
    return _status(payload) in {"complete", "completed", "success"}


def _extract_conductor_entries(conductor_log_text: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    pattern = re.compile(
        r"\|\s*([^|]+)\|\s*Exp\s+(18(?:7[6-9]|8[0-9])):([^|]+)\|\s*([^|]+)\|\s*(.*)"
    )
    for raw_line in conductor_log_text.splitlines():
        match = pattern.search(raw_line)
        if match:
            exp_id = f"exp{match.group(2)}"
            entries[exp_id] = {
                "timestamp": match.group(1).strip(),
                "experiment_id": exp_id,
                "title": match.group(3).strip(),
                "status": match.group(4).strip(),
                "details": match.group(5).strip().rstrip("|").strip(),
            }
    return entries


def _blocked_scope(retro_payload: Mapping[str, Any], exp_id: str) -> dict[str, Any]:
    for row in retro_payload.get("blocked_scopes", []):
        if isinstance(row, Mapping) and row.get("experiment_id") == exp_id:
            return dict(row)
    return {}


def _title_for_duplicate_check(value: object) -> str:
    raw = str(value).strip()
    match = re.search(r"Exp\s+\d+:\s*[^|]+", raw)
    if match:
        return re.sub(r"\s+", " ", match.group(0)).strip()
    return raw


def _duplicate_compute_bound_titles(slowest_rows: object) -> list[str]:
    if not isinstance(slowest_rows, Sequence) or isinstance(slowest_rows, str):
        return []
    titles = [
        _title_for_duplicate_check(row.get("experiment"))
        for row in slowest_rows
        if isinstance(row, Mapping) and row.get("compute_bound") is True
    ]
    counts = Counter(titles)
    seen: set[str] = set()
    duplicates: list[str] = []
    for title in titles:
        if counts[title] > 1 and title not in seen:
            seen.add(title)
            duplicates.append(title)
    return duplicates


def _text_has_all(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def _source_artifacts_checked(
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, bool]]:
    return {_source_path(source_id): {"exists": source_id in sources} for source_id in SOURCE_FILES}


def _ready_substrate(
    exp1878: Mapping[str, Any],
    exp1879: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "validator_tree": {
            "source": _source_path("exp1878"),
            "status": exp1878.get("status", "missing"),
            "honest_verdict": exp1878.get("honest_verdict", "missing_artifact"),
            "validator_tree_compiler_ready": exp1878.get("validator_tree_compiler_ready") is True,
            "zero_false_accepts": exp1878.get("zero_false_accepts"),
            "constraint_coverage_rate": exp1878.get("constraint_coverage_rate"),
        },
        "beaver_lite_bounds": {
            "source": _source_path("exp1879"),
            "status": exp1879.get("status", "missing"),
            "honest_verdict": exp1879.get("honest_verdict", "missing_artifact"),
            "beaver_lite_bounds_ready": exp1879.get("beaver_lite_bounds_ready") is True,
            "deterministic_coverage_bound": exp1879.get("deterministic_coverage_bound"),
            "residual_risk_bound": exp1879.get("residual_risk_bound"),
            "acceptance_authority_unchanged": exp1879.get("acceptance_authority_unchanged"),
        },
    }


def _operational_speedups(
    operational_retro: Mapping[str, Any],
    duplicate_titles: Sequence[str],
) -> list[dict[str, Any]]:
    estimated_pct = operational_retro.get("estimated_time_savings_pct")
    return [
        {
            "name": "same_title_compute_bound_terminal_state_dedupe",
            "source": _source_path("operational_retro_147"),
            "estimated_time_savings_pct": estimated_pct,
            "duplicate_compute_bound_titles": list(duplicate_titles),
            "tracking_field": "same_title_compute_dedupe_required",
        },
        {
            "name": "gpu_model_count_telemetry",
            "source": _source_path("operational_retro_147"),
            "estimated_time_savings_pct": estimated_pct,
            "tracking_fields": ["model_count", "parallel_model_count", "gpu_utilization_spans"],
        },
    ]


def build_artifact(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: Sequence[str],
    conductor_log_text: str,
    status_text: str,
    changelog_text: str,
    roadmap_text: str,
    roadmap_doc_text: str,
    tests_run: Sequence[str],
) -> dict[str, Any]:
    """REQ-REPORT-1890: derive `.148` activation gates from `.147` evidence."""

    exp1878 = sources.get("exp1878", {})
    exp1879 = sources.get("exp1879", {})
    exp1889 = sources.get("exp1889", {})
    operational_retro = sources.get("operational_retro_147", {})
    gate_readiness = exp1889.get("gate_readiness", {})
    prompt_gate = gate_readiness.get("prompt_to_validator", {})
    telemetry_gate = gate_readiness.get("telemetry", {})
    fr11_gate = gate_readiness.get("fr11", {})
    hardware_gate = gate_readiness.get("hardware_accounting", {})

    milestone_147_archived = (
        _is_complete(exp1889)
        and exp1889.get("milestone_147_retro_complete") is True
        and exp1889.get("completed_task_count") == 5
        and exp1889.get("blocked_task_count") == 9
    )
    validator_tree_ready = (
        _is_complete(exp1878) and exp1878.get("validator_tree_compiler_ready") is True
    )
    beaver_bounds_ready = _is_complete(exp1879) and exp1879.get("beaver_lite_bounds_ready") is True

    missing_models = list(prompt_gate.get("missing_models") or [])
    live_sota_blocked_missing_models = prompt_gate.get("live_sota_ready") is False and bool(
        missing_models
    )
    telemetry_scope = _blocked_scope(exp1889, "exp1881")
    fr11_scope = _blocked_scope(exp1889, "exp1884")
    hardware_scope = _blocked_scope(exp1889, "exp1887")
    telemetry_missing_terminal_artifact = (
        telemetry_gate.get("source_status") == "missing"
        or telemetry_scope.get("artifact_missing") is True
    )
    fr11_ledger_missing_terminal_artifact = (
        fr11_gate.get("ledger_status") == "missing" or fr11_scope.get("artifact_missing") is True
    )
    hardware_accounting_missing_terminal_artifact = (
        hardware_gate.get("accounting_status") == "missing"
        or hardware_scope.get("artifact_missing") is True
    )

    duplicate_titles = _duplicate_compute_bound_titles(
        operational_retro.get("slowest_experiments", [])
    )
    operational_text = "\n".join(
        str(item)
        for item in (
            *operational_retro.get("improvements_suggested", []),
            *operational_retro.get("top_3_highest_leverage_actions", []),
            status_text,
            changelog_text,
        )
    )
    same_title_compute_dedupe_required = (
        operational_retro.get("estimated_time_savings_pct") == 11
        and bool(duplicate_titles)
        and _text_has_all(operational_text, ["same-title", "gpu"])
        and ("model-count" in operational_text.lower() or "model_count" in operational_text.lower())
    )

    roadmap_contract_ready = _text_has_all(
        roadmap_text + "\n" + roadmap_doc_text,
        [
            "experiment_1890_147_completion_148_activation_contract.json",
            "validator_tree_ready",
            "beaver_bounds_ready",
        ],
    )
    field_values = {
        "milestone_147_archived": milestone_147_archived,
        "validator_tree_ready": validator_tree_ready,
        "beaver_bounds_ready": beaver_bounds_ready,
        "live_sota_blocked_missing_models": live_sota_blocked_missing_models,
        "telemetry_missing_terminal_artifact": telemetry_missing_terminal_artifact,
        "fr11_ledger_missing_terminal_artifact": fr11_ledger_missing_terminal_artifact,
        "hardware_accounting_missing_terminal_artifact": hardware_accounting_missing_terminal_artifact,
        "same_title_compute_dedupe_required": same_title_compute_dedupe_required,
        "roadmap_contract_ready": roadmap_contract_ready,
    }
    blocked_reasons = [field for field, ready in field_values.items() if not ready]
    next_gate_contract_ready = not blocked_reasons and not missing_source_ids
    if missing_source_ids:
        blocked_reasons.extend(f"missing:{source_id}" for source_id in missing_source_ids)

    status = "complete" if next_gate_contract_ready else "blocked"
    honest_verdict = (
        "complete: milestone_147_archived_148_activation_contract_ready_"
        "validator_beaver_ready_live_sota_telemetry_fr11_hardware_blocked"
        if status == "complete"
        else "blocked: " + "; ".join(blocked_reasons)
    )

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "milestone": TARGET_MILESTONE,
        "predecessor_milestone": PREDECESSOR_MILESTONE,
        "status": status,
        "honest_verdict": honest_verdict,
        "milestone_147_archived": milestone_147_archived,
        "validator_tree_ready": validator_tree_ready,
        "beaver_bounds_ready": beaver_bounds_ready,
        "live_sota_blocked_missing_models": live_sota_blocked_missing_models,
        "telemetry_missing_terminal_artifact": telemetry_missing_terminal_artifact,
        "fr11_ledger_missing_terminal_artifact": fr11_ledger_missing_terminal_artifact,
        "hardware_accounting_missing_terminal_artifact": hardware_accounting_missing_terminal_artifact,
        "same_title_compute_dedupe_required": same_title_compute_dedupe_required,
        "next_gate_contract_ready": next_gate_contract_ready,
        "ready_substrate": _ready_substrate(exp1878, exp1879),
        "blocked_gates": {
            "live_sota": {
                "blocked": live_sota_blocked_missing_models,
                "blocking_field": prompt_gate.get("blocking_field"),
                "missing_models": missing_models,
            },
            "telemetry": {
                "missing_terminal_artifact": telemetry_missing_terminal_artifact,
                "blocking_field": telemetry_gate.get("blocking_field"),
                "source_status": telemetry_gate.get("source_status", "missing"),
                "blocked_scope": telemetry_scope,
            },
            "fr11_ledger": {
                "missing_terminal_artifact": fr11_ledger_missing_terminal_artifact,
                "blocking_field": fr11_gate.get("blocking_field"),
                "ledger_status": fr11_gate.get("ledger_status", "missing"),
                "blocked_scope": fr11_scope,
            },
            "hardware_accounting": {
                "missing_terminal_artifact": hardware_accounting_missing_terminal_artifact,
                "blocking_field": hardware_gate.get("blocking_field"),
                "accounting_status": hardware_gate.get("accounting_status", "missing"),
                "blocked_scope": hardware_scope,
            },
        },
        "operational_speedups_to_track": _operational_speedups(operational_retro, duplicate_titles),
        "gate_recommendations": exp1889.get("next_gate_recommendations", {}),
        "blocked_reasons": blocked_reasons,
        "missing_artifacts": {
            source_id: _source_path(source_id) for source_id in sorted(missing_source_ids)
        },
        "source_artifacts_checked": _source_artifacts_checked(sources),
        "conductor_events_147": _extract_conductor_entries(conductor_log_text),
        "tests_run": list(tests_run),
    }


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    tests_run: Sequence[str] = (),
) -> dict[str, Any]:
    """SCENARIO-REPORT-1890: write the `.148` activation contract artifact."""

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)
    sources, missing_source_ids = _load_sources(root_path / "results")
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=missing_source_ids,
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        status_text=_read_text(root_path / "ops" / "status.md"),
        changelog_text=_read_text(root_path / "ops" / "changelog.md"),
        roadmap_text=_read_text(root_path / "research-roadmap.yaml"),
        roadmap_doc_text=_read_text(
            root_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
        ),
        tests_run=tests_run,
    )
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
