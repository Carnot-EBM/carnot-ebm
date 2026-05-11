"""Build the Exp 1876 `.146` completion ledger and `.147` gate contract.

Spec: REQ-REPORT-1876, SCENARIO-REPORT-1876.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260511"
PREDECESSOR_MILESTONE = "2026.05.146"
TARGET_MILESTONE = "2026.05.147"
EXPERIMENT = "1876_146_completion_147_gate_contract"
SCHEMA = "carnot.milestone_146_completion_147_gate_contract.v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1876_146_completion_147_gate_contract.json"
)

SOURCE_FILES = {
    "exp1864": "experiment_1864_roce.json",
    "exp1868": "experiment_1868_ltlzinc.json",
    "exp1869": "experiment_1869_hiled.json",
    "exp1871": "experiment_1871_s2kan_rust.json",
    "exp1872": "experiment_1872_ising_consensus.json",
}

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "honest_verdict",
    "milestone_146_archived",
    "artifact_schema_contract_ready",
    "prior_failure_carryforward_ready",
    "gate_contract_ready",
    "blocked_scope_summary",
}

STANDARD_RESULT_FIELDS = ("status", "honest_verdict")

BLOCKED_SCOPE_ROOT_CAUSES = {
    "exp1866": "doomed_rerun_prior_failures_missing",
    "exp1867": "upstream_retired_exp1866",
    "exp1873": "doomed_rerun_prior_failures_missing",
    "exp1874": "gemini_cli_or_pretest_infrastructure",
    "exp1875": "gemini_cli_or_pretest_infrastructure",
}

MISSING_GATE_FIELD_RERUN_POLICY = (
    "do_not_rerun_downstream_gate_until_upstream_artifact_has_standard_fields"
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-1876: persist a started marker before source evidence reads."""

    artifact: dict[str, Any] = {field: False for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "in_progress",
            "milestone": TARGET_MILESTONE,
            "predecessor_milestone": PREDECESSOR_MILESTONE,
            "honest_verdict": "in_progress",
            "blocked_scope_summary": [],
        }
    )
    return _write_json(Path(out_path), artifact)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _load_sources(results_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    loaded: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for exp_id, filename in SOURCE_FILES.items():
        payload = _read_json(results_dir / filename)
        if payload is None:
            missing.append(f"results/{filename}")
        else:
            loaded[exp_id] = payload
    return loaded, missing


def _source_inputs_read(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, bool]]:
    return {
        f"results/{filename}": {"exists": exp_id in sources}
        for exp_id, filename in SOURCE_FILES.items()
    }


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status") or "").lower()


def _missing_standard_fields(payload: Mapping[str, Any]) -> list[str]:
    return sorted(field for field in STANDARD_RESULT_FIELDS if field not in payload)


def _metric_summary(exp_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if exp_id == "exp1864":
        return {
            "dataset_size": payload.get("dataset_size"),
            "success_rate": payload.get("success_rate"),
            "successes": payload.get("successes"),
        }
    if exp_id == "exp1868":
        return {
            "cerce_ledger_ready": payload.get("cerce_ledger_ready"),
            "promotion_gate_passed": payload.get("promotion_gate_passed"),
            "replay_retention_rate": payload.get("replay_retention_rate"),
            "cerce_nonforgetting_rate": payload.get("cerce_nonforgetting_rate"),
        }
    if exp_id == "exp1869":
        return {
            "constraint_enforcement_rate": payload.get("constraint_enforcement_rate"),
            "efficiency_gains_ms": payload.get("efficiency_gains_ms"),
            "hiled_enabled": payload.get("hiled_enabled"),
        }
    if exp_id == "exp1871":
        return {"module": payload.get("module")}
    if exp_id == "exp1872":
        return {"min_energy": payload.get("min_energy")}
    return {}


def _classify_sources(
    sources: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    schema_complete: list[dict[str, Any]] = []
    malformed_actionable: list[dict[str, Any]] = []
    status_normalization: list[dict[str, Any]] = []

    for exp_id in SOURCE_FILES:
        payload = sources.get(exp_id)
        if payload is None:
            continue
        missing = _missing_standard_fields(payload)
        row = {
            "experiment_id": exp_id,
            "artifact_path": f"results/{SOURCE_FILES[exp_id]}",
            "metrics": _metric_summary(exp_id, payload),
        }
        if not missing and _status(payload) == "complete":
            schema_complete.append(row | {"status": payload.get("status")})
        elif not missing and _status(payload) == "completed":
            status_normalization.append(
                row
                | {
                    "status": payload.get("status"),
                    "normalization_needed": "status_value_should_be_complete",
                }
            )
        else:
            malformed_actionable.append(
                row
                | {
                    "missing_standard_fields": missing,
                    "normalization_needed": "wrap_raw_evidence_with_standard_result_fields",
                }
            )
    return schema_complete, malformed_actionable, status_normalization


def _extract_conductor_entries(conductor_log_text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for raw_line in conductor_log_text.splitlines():
        match = re.search(r"\|\s*([^|]+)\|\s*Exp\s+(18(?:6[4-9]|7[0-5])):([^|]+)\|\s*([^|]+)\|\s*(.*)", raw_line)
        if not match:
            continue
        entries.append(
            {
                "timestamp": match.group(1).strip(),
                "experiment_id": f"exp{match.group(2)}",
                "title": match.group(3).strip(),
                "status": match.group(4).strip(),
                "details": match.group(5).strip().rstrip("|").strip(),
            }
        )
    return entries


def _missing_gate_field_blocks(entries: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        if entry.get("status") != "GATE_BLOCK":
            continue
        details = str(entry.get("details") or "")
        gate_match = re.search(r"(exp\d+)[\w-]*\.(\w+)\s+\(actual=Non?e?", details)
        if not gate_match:
            continue
        key = (str(entry.get("experiment_id") or ""), gate_match.group(1), gate_match.group(2))
        if key in seen:
            continue
        seen.add(key)
        blocks.append(
            {
                "blocked_experiment_id": key[0],
                "upstream_experiment_id": gate_match.group(1),
                "missing_field": gate_match.group(2),
                "gate_status": "missing_upstream_field",
                "rerun_policy": MISSING_GATE_FIELD_RERUN_POLICY,
            }
        )
    return blocks


def _blocked_scope_summary(entries: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for entry in entries:
        exp_id = str(entry.get("experiment_id") or "")
        if exp_id not in BLOCKED_SCOPE_ROOT_CAUSES or exp_id in seen:
            continue
        seen.add(exp_id)
        rows.append(
            {
                "experiment_id": exp_id,
                "terminal_status": entry.get("status"),
                "root_cause": BLOCKED_SCOPE_ROOT_CAUSES[exp_id],
                "rerun_allowed_without_changed_root_cause": False,
                "details": entry.get("details"),
            }
        )
    return rows


def _context_has_all(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def _protected_files_clean(root: Path) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", "research-roadmap.yaml", "scripts/research_conductor.py"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def build_artifact(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_paths: Sequence[str],
    conductor_log_text: str,
    roadmap_text: str,
    roadmap_doc_text: str,
    changelog_text: str,
    protected_files_unchanged: bool,
) -> dict[str, Any]:
    """REQ-REPORT-1876: build the terminal `.146` ledger and `.147` gate contract."""

    schema_complete, malformed_actionable, status_normalization = _classify_sources(sources)
    conductor_entries = _extract_conductor_entries(conductor_log_text)
    missing_gate_blocks = _missing_gate_field_blocks(conductor_entries)
    blocked_scopes = _blocked_scope_summary(conductor_entries)

    expected_log_ids = {"exp1864", "exp1868", "exp1869", "exp1871", "exp1872"}
    log_ids = {entry["experiment_id"] for entry in conductor_entries}
    milestone_146_archived = bool(
        not missing_source_paths
        and expected_log_ids <= log_ids
        and (schema_complete or malformed_actionable or status_normalization)
    )

    artifact_schema_contract_ready = bool(
        {"exp1864", "exp1869"}
        <= {row["experiment_id"] for row in malformed_actionable}
        and {
            ("exp1865", "exp1864", "status"),
            ("exp1870", "exp1869", "status"),
        }
        <= {
            (
                row["blocked_experiment_id"],
                row["upstream_experiment_id"],
                row["missing_field"],
            )
            for row in missing_gate_blocks
        }
        and _context_has_all(
            roadmap_text + "\n" + roadmap_doc_text,
            ["artifact_schema_contract_ready", "normalize malformed"],
        )
    )

    carryforward_tokens = [
        "exp1865",
        "exp1870",
        "exp1866",
        "exp1867",
        "exp1873",
        "exp1874",
        "exp1875",
    ]
    prior_failure_carryforward_ready = bool(
        blocked_scopes
        and _context_has_all(roadmap_text, carryforward_tokens)
        and all(row["rerun_allowed_without_changed_root_cause"] is False for row in blocked_scopes)
    )
    gate_contract_ready = bool(
        artifact_schema_contract_ready
        and prior_failure_carryforward_ready
        and "gate_contract_ready" in roadmap_text
    )

    blocked_reasons: list[str] = []
    if missing_source_paths:
        blocked_reasons.append("listed source artifacts are missing")
    if not conductor_entries:
        blocked_reasons.append("conductor log entries for Exp 1864-1875 are missing")
    if not artifact_schema_contract_ready:
        blocked_reasons.append("artifact schema contract is not ready")
    if not prior_failure_carryforward_ready:
        blocked_reasons.append("prior failure carryforward is incomplete")
    if not protected_files_unchanged:
        blocked_reasons.append("protected files changed")

    status = "complete" if not blocked_reasons and gate_contract_ready else "blocked"
    changelog_mentions = sorted(
        exp_id for exp_id in SOURCE_FILES if exp_id.replace("exp", "Exp ") in changelog_text
    )

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "milestone": TARGET_MILESTONE,
        "predecessor_milestone": PREDECESSOR_MILESTONE,
        "status": status,
        "milestone_146_archived": milestone_146_archived and status == "complete",
        "artifact_schema_contract_ready": artifact_schema_contract_ready and status == "complete",
        "prior_failure_carryforward_ready": prior_failure_carryforward_ready and status == "complete",
        "gate_contract_ready": gate_contract_ready and status == "complete",
        "schema_complete_evidence": schema_complete,
        "malformed_actionable_evidence": malformed_actionable,
        "usable_with_status_normalization": status_normalization,
        "missing_gate_field_blocks": missing_gate_blocks,
        "blocked_scope_summary": blocked_scopes,
        "blocked_reasons": blocked_reasons,
        "source_inputs_read": _source_inputs_read(sources),
        "conductor_entries_1864_1875": conductor_entries,
        "changelog_source_mentions": changelog_mentions,
        "research_roadmap_yaml_modified": not protected_files_unchanged,
        "scripts_research_conductor_modified": not protected_files_unchanged,
        "tests_run": [],
    }
    if status == "complete":
        artifact["honest_verdict"] = "complete: milestone_146_archived_147_gate_contract_ready"
    else:
        artifact["honest_verdict"] = "blocked: " + "; ".join(blocked_reasons)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    protected_files_unchanged: bool | None = None,
) -> dict[str, Any]:
    """SCENARIO-REPORT-1876: write the in-progress and terminal JSON artifacts."""

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)
    sources, missing = _load_sources(root_path / "results")
    protected_clean = (
        _protected_files_clean(root_path)
        if protected_files_unchanged is None
        else protected_files_unchanged
    )
    artifact = build_artifact(
        sources=sources,
        missing_source_paths=missing,
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        roadmap_text=_read_text(root_path / "research-roadmap.yaml"),
        roadmap_doc_text=_read_text(
            root_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
        ),
        changelog_text=_read_text(root_path / "ops" / "changelog.md"),
        protected_files_unchanged=protected_clean,
    )
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
