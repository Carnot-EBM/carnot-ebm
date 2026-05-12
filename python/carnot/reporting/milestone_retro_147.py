"""Build the Exp 1889 milestone .147 retrospective artifact.

Spec: REQ-REPORT-1889, SCENARIO-REPORT-1889.
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
MILESTONE = "2026.05.147"
EXPERIMENT = "1889_milestone_147_retro"
SCHEMA = "carnot.milestone_147_retro.v1"

DEFAULT_OUT_PATH = REPO_ROOT / "results" / "experiment_1889_milestone_147_retro.json"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "honest_verdict",
    "milestone_147_retro_complete",
    "completed_task_count",
    "blocked_task_count",
    "next_gate_recommendations",
    "tests_run",
}

SOURCE_FILES = {
    "exp1876": "experiment_1876_146_completion_147_gate_contract.json",
    "exp1877": "experiment_1877_artifact_contract_normalization.json",
    "exp1878": "experiment_1878_roce_validator_tree.json",
    "exp1879": "experiment_1879_beaver_lite_bounds.json",
    "exp1880": "experiment_1880_sota_roce_validator_eval.json",
    "exp1881": "experiment_1881_low_cost_hallucination_telemetry.json",
    "exp1882": "experiment_1882_dccd_roce_repair.json",
    "exp1883": "experiment_1883_hiled_live_logprob_smoke.json",
    "exp1884": "experiment_1884_fr11_cerce_cnsp_ledger.json",
    "exp1885": "experiment_1885_sota_fr11_promotion_gate.json",
    "exp1886": "experiment_1886_gem_consformer_preconditioner.json",
    "exp1887": "experiment_1887_fpga_s2kan_ising_accounting.json",
    "exp1888": "experiment_1888_integrated_trisota_e2e.json",
}

TASK_IDS = tuple(f"exp{experiment_id}" for experiment_id in range(1876, 1890))


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-1889: persist an auditable started marker before reads."""

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "milestone": MILESTONE,
        "status": "in_progress",
        "honest_verdict": "in_progress",
        "milestone_147_retro_complete": False,
        "completed_task_count": 0,
        "blocked_task_count": 0,
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


def _load_sources(results_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    sources: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for exp_id, filename in SOURCE_FILES.items():
        payload = _read_json(results_dir / filename)
        if payload is None:
            missing.append(exp_id)
        else:
            sources[exp_id] = payload
    return sources, missing


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status") or "").lower()


def _verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("honest_verdict") or "missing_artifact")


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


def _source_path(exp_id: str) -> str:
    return f"results/{SOURCE_FILES[exp_id]}"


def _completed_scopes(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "experiment_id": exp_id,
            "path": _source_path(exp_id),
            "status": payload.get("status"),
            "honest_verdict": _verdict(payload),
        }
        for exp_id, payload in sources.items()
        if exp_id in SOURCE_FILES and _is_complete(payload)
    ]
    rows.append(
        {
            "experiment_id": "exp1889",
            "path": "results/experiment_1889_milestone_147_retro.json",
            "status": "complete",
            "honest_verdict": "retrospective_filed",
        }
    )
    return sorted(rows, key=lambda row: row["experiment_id"])


def _blocked_scopes(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
    conductor_entries: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp_id in TASK_IDS:
        if exp_id == "exp1889":
            continue
        payload = sources.get(exp_id, {})
        if payload and _is_complete(payload):
            continue
        log_entry = conductor_entries.get(exp_id, {})
        rows.append(
            {
                "experiment_id": exp_id,
                "path": _source_path(exp_id),
                "status": payload.get("status") or log_entry.get("status") or "missing",
                "honest_verdict": _verdict(payload),
                "conductor_status": log_entry.get("status"),
                "blocked_reason": payload.get("honest_verdict")
                or log_entry.get("details")
                or "source artifact missing",
                "artifact_missing": exp_id in missing_source_ids,
            }
        )
    return rows


def _retired_scopes(blocked_scopes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": str(row["experiment_id"]),
            "retirement_reason": str(row["blocked_reason"]),
            "rerun_without_changed_root_cause": False,
        }
        for row in blocked_scopes
        if "upstream retired" in str(row.get("blocked_reason", "")).lower()
    ]


def _recommendations() -> dict[str, dict[str, Any]]:
    return {
        "prompt_to_validator": {
            "required_fields": [
                "validator_tree_compiler_ready",
                "beaver_lite_bounds_ready",
                "sota_roce_eval_ready",
                "inference_mode",
                "missing_models",
            ],
            "recommendation": (
                "Gate headline prompt-to-validator claims on live SOTA readiness, not only "
                "compiled validator fixtures."
            ),
        },
        "telemetry": {
            "required_fields": [
                "telemetry_adapter_ready",
                "phi_first_available",
                "spilled_energy_available",
                "acceptance_authority_unchanged",
            ],
            "recommendation": "Telemetry remains an analysis signal until a terminal adapter artifact exists.",
        },
        "fr11": {
            "required_fields": [
                "promotion_gate_passed",
                "utility_delta",
                "fr11_sota_self_learning_ready",
                "nonforgetting_rate",
            ],
            "recommendation": "FR-11 promotion must gate on positive utility and retention before SOTA use.",
        },
        "hardware_accounting": {
            "required_fields": [
                "sampler_preconditioning_ready",
                "fpga_decomposition_accounting_ready",
                "kv260_no_synthesis_claim",
                "hardware_execution_claim",
            ],
            "recommendation": "Hardware-accounting work needs no-synthesis estimates before any board claim.",
        },
    }


def _gate_readiness(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    exp1878 = sources.get("exp1878", {})
    exp1879 = sources.get("exp1879", {})
    exp1880 = sources.get("exp1880", {})
    exp1881 = sources.get("exp1881", {})
    exp1884 = sources.get("exp1884", {})
    exp1885 = sources.get("exp1885", {})
    exp1886 = sources.get("exp1886", {})
    exp1887 = sources.get("exp1887", {})

    validator_ready = exp1878.get("validator_tree_compiler_ready") is True
    bounds_ready = exp1879.get("beaver_lite_bounds_ready") is True
    live_ready = exp1880.get("sota_roce_eval_ready") is True
    contract_ready = validator_ready and bounds_ready
    prompt_blocker = (
        "validator_tree_compiler_ready"
        if not validator_ready
        else "beaver_lite_bounds_ready"
        if not bounds_ready
        else "sota_roce_eval_ready"
        if not live_ready
        else None
    )

    telemetry_ready = exp1881.get("telemetry_adapter_ready") is True
    exp1884_promoted = (
        exp1884.get("promotion_gate_passed") is True
        and isinstance(exp1884.get("utility_delta"), int | float)
        and exp1884.get("utility_delta", 0.0) > 0.0
    )
    fr11_ready = exp1885.get("fr11_sota_self_learning_ready") is True and exp1884_promoted
    hardware_ready = (
        exp1886.get("sampler_preconditioning_ready") is True
        and exp1887.get("fpga_decomposition_accounting_ready") is True
        and exp1887.get("kv260_no_synthesis_claim") is True
        and exp1887.get("hardware_execution_claim") is False
    )

    return {
        "prompt_to_validator": {
            "ready_for_next_milestone": contract_ready and live_ready,
            "contract_ready": contract_ready,
            "live_sota_ready": live_ready,
            "blocking_field": prompt_blocker,
            "missing_models": exp1880.get("missing_models", []),
        },
        "telemetry": {
            "ready_for_next_milestone": telemetry_ready,
            "blocking_field": None if telemetry_ready else "telemetry_adapter_ready",
            "source_status": exp1881.get("status", "missing"),
        },
        "fr11": {
            "ready_for_next_milestone": fr11_ready,
            "blocking_field": None if fr11_ready else "promotion_gate_passed",
            "ledger_status": exp1884.get("status", "missing"),
            "sota_gate_status": exp1885.get("status", "missing"),
        },
        "hardware_accounting": {
            "ready_for_next_milestone": hardware_ready,
            "blocking_field": None if hardware_ready else "fpga_decomposition_accounting_ready",
            "preconditioner_status": exp1886.get("status", "missing"),
            "accounting_status": exp1887.get("status", "missing"),
        },
    }


def _follow_up_scopes(gate_readiness: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "scope": scope,
            "blocking_field": str(values.get("blocking_field")),
            "action": "produce_terminal_gate_evidence_before_reuse",
        }
        for scope, values in gate_readiness.items()
        if values.get("ready_for_next_milestone") is not True
    ]


def build_artifact(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: Sequence[str],
    conductor_log_text: str,
    tests_run: Sequence[str],
) -> dict[str, Any]:
    """REQ-REPORT-1889: reconcile `.147` source evidence into one closeout artifact."""

    missing_ids = set(missing_source_ids)
    conductor_entries = _extract_conductor_entries(conductor_log_text)
    completed = _completed_scopes(sources)
    blocked = _blocked_scopes(sources, missing_ids, conductor_entries)
    readiness = _gate_readiness(sources)
    completed_count = len(completed)
    blocked_count = len(blocked)

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "milestone": MILESTONE,
        "status": "complete",
        "honest_verdict": (
            f"complete: milestone_147_retro_filed_{completed_count}_completed_"
            f"{blocked_count}_blocked_prompt_validator_partial_telemetry_fr11_hardware_not_ready"
        ),
        "milestone_147_retro_complete": True,
        "completed_task_count": completed_count,
        "blocked_task_count": blocked_count,
        "completed_scopes": completed,
        "blocked_scopes": blocked,
        "retired_scopes": _retired_scopes(blocked),
        "follow_up_scopes": _follow_up_scopes(readiness),
        "gate_readiness": readiness,
        "next_gate_recommendations": _recommendations(),
        "missing_artifacts": {exp_id: _source_path(exp_id) for exp_id in sorted(missing_ids)},
        "source_artifacts_checked": {
            exp_id: {"path": _source_path(exp_id), "exists": exp_id in sources}
            for exp_id in SOURCE_FILES
        },
        "conductor_events_147": conductor_entries,
        "tests_run": list(tests_run),
    }


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    tests_run: Sequence[str] = (),
) -> dict[str, Any]:
    """Run the .147 milestone retrospective generation."""

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)
    sources, missing_source_ids = _load_sources(root_path / "results")
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=missing_source_ids,
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        tests_run=tests_run,
    )
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
