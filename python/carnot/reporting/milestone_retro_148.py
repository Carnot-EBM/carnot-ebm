"""Build the Exp 1903 milestone .148 retrospective artifact.

Spec: REQ-REPORT-1903, SCENARIO-REPORT-1903.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260512"
MILESTONE = "2026.05.148"
EXPERIMENT = "1903_milestone_148_retro"
SCHEMA = "carnot.milestone_148_retro.v1"

DEFAULT_OUT_PATH = REPO_ROOT / "results" / "experiment_1903_milestone_148_retro.json"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "honest_verdict",
    "milestone_148_retro_complete",
    "completed_task_count",
    "blocked_task_count",
    "failed_task_count",
    "same_title_compute_dedupe_result",
    "next_gate_recommendations",
    "tests_run",
}

SOURCE_FILES = {
    "exp1890": "experiment_1890_147_completion_148_activation_contract.json",
    "exp1891": "experiment_1891_sota_gguf_cache_runtime_preflight.json",
    "exp1892": "experiment_1892_terminal_low_cost_telemetry_adapter.json",
    "exp1893": "experiment_1893_live_sota_roce_validator_eval_v2.json",
    "exp1894": "experiment_1894_dccd_roce_repair_v2.json",
    "exp1895": "experiment_1895_residual_drift_validator_ledger.json",
    "exp1896": "experiment_1896_fr11_validator_tree_promotion_ledger_v2.json",
    "exp1897": "experiment_1897_routing_without_forgetting_fr11_audit.json",
    "exp1898": "experiment_1898_sota_fr11_promotion_smoke_v2.json",
    "exp1899": "experiment_1899_gem_consformer_validator_graph_preconditioner_v2.json",
    "exp1900": "experiment_1900_fpga_s2kan_ising_resource_accounting_v2.json",
    "exp1901": "experiment_1901_pbit_pdit_ising_sampler_accounting.json",
    "exp1902": "experiment_1902_integrated_trisota_e2e_v2.json",
}

TASK_IDS = tuple(f"exp{experiment_id}" for experiment_id in range(1890, 1903))


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-1903: persist a started marker before scoring source evidence."""

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "milestone": MILESTONE,
        "status": "in_progress",
        "honest_verdict": "in_progress",
        "milestone_148_retro_complete": False,
        "completed_task_count": 0,
        "blocked_task_count": 0,
        "failed_task_count": 0,
        "same_title_compute_dedupe_result": {},
        "next_gate_recommendations": {},
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


def load_available_sources(results_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    sources: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for exp_id, filename in SOURCE_FILES.items():
        payload = _read_json(results_dir / filename)
        if payload is None:
            missing.append(exp_id)
        else:
            sources[exp_id] = payload
    return sources, missing


def _source_path(exp_id: str) -> str:
    return f"results/{SOURCE_FILES[exp_id]}"


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status") or "").lower()


def _is_complete(payload: Mapping[str, Any]) -> bool:
    return _status(payload) in {"complete", "completed", "success"}


def _is_structured_block(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("schema") == "blocked_gate_check_v1"
        or payload.get("blocked_at_layer") == "conductor_pre_gate"
        or _status(payload) in {"blocked", "gate_block", "gate_blocked"}
    )


def _extract_conductor_entries(conductor_log_text: str) -> dict[str, list[dict[str, str]]]:
    entries: dict[str, list[dict[str, str]]] = {}
    pattern = re.compile(
        r"\|\s*([^|]+)\|\s*Exp\s+(18(?:9[0-9])|190[0-2]):([^|]+)\|\s*([^|]+)\|\s*(.*)"
    )
    for raw_line in conductor_log_text.splitlines():
        match = pattern.search(raw_line)
        if match:
            exp_id = f"exp{match.group(2)}"
            entries.setdefault(exp_id, []).append(
                {
                    "timestamp": match.group(1).strip(),
                    "experiment_id": exp_id,
                    "title": match.group(3).strip(),
                    "status": match.group(4).strip(),
                    "details": match.group(5).strip().rstrip("|").strip(),
                }
            )
    return entries


def _entries_text(entries: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(
        f"{entry.get('status', '')} {entry.get('details', '')}" for entry in entries
    ).lower()


def _has_upstream_retired(entries: Sequence[Mapping[str, str]]) -> bool:
    return "pre-emptive skip: upstream retired" in _entries_text(entries)


def _has_failure_signal(entries: Sequence[Mapping[str, str]]) -> bool:
    text = _entries_text(entries)
    statuses = {str(entry.get("status", "")).upper() for entry in entries}
    return (
        bool({"FAIL", "SKIP"} & statuses)
        or "pre-tests failing" in text
        or "codex cli error" in text
    )


def _classify_task(
    exp_id: str,
    payload: Mapping[str, Any],
    entries: Sequence[Mapping[str, str]],
    artifact_missing: bool,
) -> dict[str, Any]:
    if _is_complete(payload):
        classification = "completed"
    elif _is_structured_block(payload):
        classification = "blocked"
    elif _has_upstream_retired(entries):
        classification = "retired"
    elif _has_failure_signal(entries) or artifact_missing:
        classification = "failed"
    else:
        classification = "blocked"

    latest = entries[-1] if entries else {}
    reason = str(payload.get("honest_verdict") or payload.get("gate_check_summary") or "") or str(
        latest.get("details") or "source artifact missing"
    )
    expected_gate_skip = classification in {"blocked", "retired"}
    return {
        "experiment_id": exp_id,
        "deliverable": _source_path(exp_id),
        "classification": classification,
        "status": payload.get("status") or latest.get("status") or "missing",
        "title": payload.get("title") or latest.get("title") or exp_id,
        "artifact_exists": not artifact_missing,
        "expected_gate_skip": expected_gate_skip,
        "unexpected_missing_artifact": artifact_missing and not expected_gate_skip,
        "reason": reason,
    }


def _terminal_artifact_presence(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp1891 = sources.get("exp1891", {})
    exp1892 = sources.get("exp1892", {})
    exp1896 = sources.get("exp1896", {})
    exp1897 = sources.get("exp1897", {})
    exp1898 = sources.get("exp1898", {})
    exp1900 = sources.get("exp1900", {})
    exp1901 = sources.get("exp1901", {})
    return {
        "sota_cache_runtime": {
            "exists": bool(exp1891),
            "cache_all_available": exp1891.get("cache_all_available"),
            "runtime_smoke_ready": exp1891.get("runtime_smoke_ready"),
            "model_count": exp1891.get("model_count"),
            "parallel_model_count": exp1891.get("parallel_model_count"),
        },
        "telemetry": {
            "exists": bool(exp1892),
            "telemetry_adapter_ready": exp1892.get("telemetry_adapter_ready") is True,
        },
        "fr11": {
            "promotion_ledger_exists": bool(exp1896),
            "routing_audit_exists": bool(exp1897),
            "sota_smoke_exists": bool(exp1898),
            "promotion_gate_passed": exp1896.get("promotion_gate_passed") is True,
            "fr11_sota_self_learning_ready": exp1898.get("fr11_sota_self_learning_ready") is True,
        },
        "hardware_accounting": {
            "primary_exists": bool(exp1900),
            "pbit_blocked_artifact_exists": bool(exp1901),
            "fpga_decomposition_accounting_ready": exp1900.get(
                "fpga_decomposition_accounting_ready"
            )
            is True,
            "pbit_pdit_accounting_ready": exp1901.get("pbit_pdit_accounting_ready") is True,
        },
    }


def _same_title_compute_dedupe_result(
    activation_contract: Mapping[str, Any],
    presence: Mapping[str, Any],
) -> dict[str, Any]:
    speedups = activation_contract.get("operational_speedups_to_track", [])
    prior_target_pct = None
    duplicate_titles: list[str] = []
    for row in speedups if isinstance(speedups, Sequence) else []:
        if (
            isinstance(row, Mapping)
            and row.get("name") == "same_title_compute_bound_terminal_state_dedupe"
        ):
            prior_target_pct = row.get("estimated_time_savings_pct")
            duplicate_titles = list(row.get("duplicate_compute_bound_titles") or [])

    sota_presence = presence["sota_cache_runtime"]
    gpu_model_count_telemetry_produced = (
        sota_presence.get("model_count") is not None
        and sota_presence.get("parallel_model_count") is not None
    )
    return {
        "prior_target_pct": prior_target_pct,
        "prior_duplicate_compute_bound_titles": duplicate_titles,
        "same_title_compute_dedupe_required_in_activation": activation_contract.get(
            "same_title_compute_dedupe_required"
        )
        is True,
        "gpu_model_count_telemetry_produced": gpu_model_count_telemetry_produced,
        "observed_speedup_pct": None,
        "expected_speedup_proven": False,
        "improved_over_147": "partial_gate_skips_prevented_live_eval_rerun",
        "reason": (
            "No Exp 1891 terminal cache/runtime artifact exists, so the 11 percent "
            "target and GPU/model-count telemetry were not proven even though "
            "downstream live-eval reruns were gate-skipped."
        ),
    }


def _next_gate_recommendations(
    activation_contract: Mapping[str, Any],
    presence: Mapping[str, Any],
) -> dict[str, Any]:
    inherited = dict(activation_contract.get("gate_recommendations") or {})
    inherited.setdefault(
        "activation_contract",
        {
            "action": "Recover Exp 1890 activation evidence before planning downstream gates.",
            "required_fields": ["next_gate_contract_ready", "same_title_compute_dedupe_required"],
        },
    )
    inherited["terminal_sota_cache_runtime_preflight"] = {
        "action": (
            "Run the cache/runtime preflight first and write a terminal blocked artifact "
            "when mandated GGUFs or smoke tests are unavailable."
        ),
        "required_fields": [
            "cache_all_available",
            "cache_any_available",
            "runtime_smoke_ready",
            "missing_models",
            "model_count",
            "parallel_model_count",
        ],
        "current_ready": presence["sota_cache_runtime"]["exists"]
        and presence["sota_cache_runtime"].get("runtime_smoke_ready") is True,
    }
    inherited["structured_gate_skip_artifacts"] = {
        "action": (
            "Emit schema-complete blocked artifacts for pre-emptive upstream-retired "
            "gate skips so retros do not need to infer them solely from the conductor log."
        ),
        "required_fields": ["status", "honest_verdict", "blocked_at_layer", "gates_evaluated"],
    }
    inherited["retro_always_runs"] = {
        "action": "Keep the milestone retrospective ungated and allow it to run after upstream failures.",
        "required_fields": [
            "milestone_retro_complete",
            "completed_task_count",
            "failed_task_count",
        ],
    }
    return inherited


def build_artifact(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: Sequence[str],
    conductor_log_text: str,
    tests_run: Sequence[str],
) -> dict[str, Any]:
    """SCENARIO-REPORT-1903: classify `.148` evidence without rerunning tasks."""

    entries = _extract_conductor_entries(conductor_log_text)
    missing_set = set(missing_source_ids)
    task_rows = [
        _classify_task(
            exp_id,
            sources.get(exp_id, {}),
            entries.get(exp_id, []),
            exp_id in missing_set,
        )
        for exp_id in TASK_IDS
    ]
    completed_tasks = [row for row in task_rows if row["classification"] == "completed"]
    blocked_tasks = [row for row in task_rows if row["classification"] == "blocked"]
    retired_tasks = [row for row in task_rows if row["classification"] == "retired"]
    failed_tasks = [row for row in task_rows if row["classification"] == "failed"]
    expected_gate_skips = [row for row in task_rows if row["expected_gate_skip"]]
    unexpected_missing = [row for row in task_rows if row["unexpected_missing_artifact"]]

    presence = _terminal_artifact_presence(sources)
    activation_contract = sources.get("exp1890", {})
    dedupe_result = _same_title_compute_dedupe_result(activation_contract, presence)
    completed_count = len(completed_tasks)
    blocked_count = len(blocked_tasks)
    retired_count = len(retired_tasks)
    failed_count = len(failed_tasks)

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "milestone": MILESTONE,
        "status": "complete",
        "honest_verdict": (
            "complete: milestone_148_retro_filed_"
            f"{completed_count}_completed_{blocked_count}_blocked_"
            f"{retired_count}_retired_{failed_count}_failed_"
            "sota_gap_unresolved_speedup_not_proven"
        ),
        "milestone_148_retro_complete": True,
        "completed_task_count": completed_count,
        "blocked_task_count": blocked_count,
        "retired_task_count": retired_count,
        "failed_task_count": failed_count,
        "gate_blocked_task_count": blocked_count + retired_count,
        "task_outcomes": task_rows,
        "completed_tasks": completed_tasks,
        "blocked_tasks": blocked_tasks,
        "retired_tasks": retired_tasks,
        "failed_tasks": failed_tasks,
        "expected_structured_gate_skips": expected_gate_skips,
        "unexpected_missing_artifact_failures": unexpected_missing,
        "terminal_artifact_presence": presence,
        "sota_cache_runtime_gap_resolved": (
            presence["sota_cache_runtime"]["exists"]
            and presence["sota_cache_runtime"].get("cache_all_available") is True
            and presence["sota_cache_runtime"].get("runtime_smoke_ready") is True
        ),
        "same_title_compute_dedupe_result": dedupe_result,
        "next_gate_recommendations": _next_gate_recommendations(activation_contract, presence),
        "available_artifacts": {
            exp_id: _source_path(exp_id) for exp_id in TASK_IDS if exp_id in sources
        },
        "missing_artifacts": {
            exp_id: _source_path(exp_id) for exp_id in TASK_IDS if exp_id in missing_set
        },
        "conductor_events_148": entries,
        "tests_run": list(tests_run),
    }


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    tests_run: Sequence[str] = (),
) -> dict[str, Any]:
    """REQ-REPORT-1903: write the terminal milestone `.148` retro JSON artifact."""

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)
    sources, missing_source_ids = load_available_sources(root_path / "results")
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=missing_source_ids,
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        tests_run=tests_run,
    )
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
