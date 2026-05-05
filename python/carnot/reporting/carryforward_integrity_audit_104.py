"""Build the Exp 1351 `.104` carry-forward artifact integrity audit.

Spec: REQ-REPORT-030, SCENARIO-REPORT-030.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260505"
MILESTONE = "2026.04.104"
EXPERIMENT = "1351_104_carryforward_artifact_integrity_audit"
SCHEMA = "carnot.carryforward_integrity_audit_104.v1"
DEFAULT_OUT_PATH = REPO_ROOT / "results" / "experiment_1351_104_carryforward_artifact_integrity_audit.json"

SOURCE_FILES = {
    1337: "experiment_1337_environment_gate_disk_pretest_stale_skeleton_audit.json",
    1338: "experiment_1338_exp1325_skeleton_and_gate_state_finalizer.json",
    1339: "experiment_1339_xgrammar2_tagdispatch_certificate_grammar_dryrun.json",
    1340: "experiment_1340_trigger_before_constrain_certificate_v6_sota.json",
    1341: "experiment_1341_halluguard_certificate_failure_split.json",
    1342: "experiment_1342_chopchop_nsvif_semantic_validator_gated.json",
    1343: "experiment_1343_margin_aware_beaver_cactus_scheduler.json",
    1344: "experiment_1344_continuous_self_learning_failure_type_memory_policy.json",
    1345: "experiment_1345_dvi_certificate_tail_v3_gated.json",
    1346: "experiment_1346_grpo_vprm_v13_gated_micro_audit.json",
    1347: "experiment_1347_thrml_compatibility_parity_audit.json",
    1348: "experiment_1348_pbit_update_dynamics_dual_bram_packet_v2.json",
    1349: "experiment_1349_ebt_citation_kona_parity_gap_audit.json",
    1350: "experiment_1350_milestone_104_retro_carryforward.json",
}

CONTEXT_FILES = (
    "research-roadmap.yaml",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "ops/conductor-log.md",
    "ops/changelog.md",
    "ops/status.md",
    "research-complete.yaml",
    "research-references.md",
    "_bmad/traceability.md",
)

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "artifact_paths_checked",
    "missing_artifacts",
    "stale_or_blocked_artifacts",
    "gates_open",
    "gates_closed",
    "prior_failure_requirements",
    "docs_reconciliation_needed",
    "terminal_certificate_required",
    "honest_verdict",
}

TERMINAL_STATUSES = {"complete", "blocked", "failed"}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-030: write the bootstrap marker before reading source evidence."""

    return _write_json(
        Path(out_path),
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "milestone": MILESTONE,
            "status": "in_progress",
        },
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status", "")).lower()


def _honest_verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("honest_verdict", ""))


def _is_terminal(payload: Mapping[str, Any] | None) -> bool:
    return bool(payload) and _status(payload or {}) in TERMINAL_STATUSES and bool(
        _honest_verdict(payload or {})
    )


def _has_valid_terminal_certificate(payload: Mapping[str, Any] | None) -> bool:
    if not _is_terminal(payload):
        return False
    return (
        isinstance((payload or {}).get("certificate_parse_rate"), int | float)
        or bool((payload or {}).get("terminal_blocker"))
        or (payload or {}).get("certificate_branch_retired_with_evidence") is True
    )


def _missing_artifacts(missing_source_ids: set[int]) -> list[dict[str, str]]:
    return [
        {
            "experiment_id": f"exp{exp_id}",
            "path": f"results/{SOURCE_FILES[exp_id]}",
            "reason": "expected .104/.104-retro source artifact not found",
        }
        for exp_id in SOURCE_FILES
        if exp_id in missing_source_ids
    ]


def _stale_or_blocked_artifacts(sources: Mapping[int, Mapping[str, Any]]) -> list[dict[str, str]]:
    classified: list[dict[str, str]] = []
    for exp_id in SOURCE_FILES:
        payload = sources.get(exp_id)
        if payload is None or _status(payload) not in {"blocked", "in_progress"}:
            continue
        classified.append(
            {
                "experiment_id": f"exp{exp_id}",
                "path": f"results/{SOURCE_FILES[exp_id]}",
                "status": _status(payload),
                "reason": str(payload.get("gate_check_summary") or _honest_verdict(payload)),
            }
        )
    return classified


def _gate_entry(gate: str, evidence: str) -> dict[str, str]:
    return {"gate": gate, "evidence": evidence}


def _gate_state(
    sources: Mapping[int, Mapping[str, Any]],
    missing_source_ids: set[int],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    gates_open: list[dict[str, str]] = []
    gates_closed: list[dict[str, str]] = []

    if sources.get(1337, {}).get("environment_ready") is True:
        gates_open.append(_gate_entry("environment_gate", "exp1337 environment_ready=true"))
    if _is_terminal(sources.get(1338)):
        gates_open.append(_gate_entry("exp1325_gate_state_finalized", "exp1338 terminal finalizer"))
    if sources.get(1339, {}).get("dynamic_grammar_ready") is True:
        gates_open.append(_gate_entry("dynamic_grammar", "exp1339 dynamic_grammar_ready=true"))
    if _is_terminal(sources.get(1341)):
        gates_open.append(_gate_entry("failure_taxonomy", "exp1341 terminal diagnostic split"))
    if _is_terminal(sources.get(1344)):
        gates_open.append(_gate_entry("self_learning_replay", "exp1344 terminal replay evidence"))
    if all(_is_terminal(sources.get(exp_id)) for exp_id in (1347, 1348, 1349)):
        gates_open.append(_gate_entry("hardware_and_external_mapping", "exp1347-exp1349 terminal audits"))

    certificate_ready = 1340 not in missing_source_ids and _has_valid_terminal_certificate(
        sources.get(1340)
    )
    semantic_ready = (
        certificate_ready
        and 1342 not in missing_source_ids
        and _is_terminal(sources.get(1342))
        and isinstance(sources.get(1342, {}).get("validator_execution_pass_rate"), int | float)
    )
    dvi_ready = semantic_ready and _is_terminal(sources.get(1345))
    grpo_ready = dvi_ready and _is_terminal(sources.get(1346))

    if certificate_ready:
        gates_open.append(_gate_entry("terminal_certificate_parse_gate", "exp1340 terminal evidence"))
    else:
        gates_closed.append(
            _gate_entry("terminal_certificate_parse_gate", "missing valid terminal exp1340 artifact")
        )

    if semantic_ready:
        gates_open.append(_gate_entry("semantic_validator", "exp1342 terminal validator evidence"))
    else:
        gates_closed.append(
            _gate_entry("semantic_validator", "closed until terminal exp1340 and exp1342 evidence")
        )

    if semantic_ready and _status(sources.get(1343, {})) == "complete":
        gates_open.append(_gate_entry("scheduler", "exp1343 complete after semantic validation"))
    else:
        gates_closed.append(_gate_entry("scheduler", "closed because semantic/scheduler evidence is absent or blocked"))

    if dvi_ready:
        gates_open.append(_gate_entry("dvi_certificate_tail", "exp1345 terminal DVI evidence"))
    else:
        gates_closed.append(_gate_entry("dvi_certificate_tail", "closed until parse and semantic gates pass"))

    if grpo_ready:
        gates_open.append(_gate_entry("grpo_vprm", "exp1346 terminal GRPO/VPRM evidence"))
    else:
        gates_closed.append(_gate_entry("grpo_vprm", "closed until DVI lossless evidence exists"))

    return gates_open, gates_closed


def _strip_yaml_scalar(value: str) -> object:
    cleaned = value.strip().strip('"').strip("'")
    if cleaned == "true":
        return True
    if cleaned == "false":
        return False
    return cleaned


def extract_prior_failure_requirements(roadmap_text: str) -> list[dict[str, Any]]:
    """REQ-REPORT-030: collect the prior-failure citations required by `.105` tasks."""

    requirements: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in roadmap_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- experiment_id:"):
            if current:
                requirements.append(current)
            current = {"experiment_id": _strip_yaml_scalar(stripped.split(":", 1)[1])}
            continue
        if current is None or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key in {"verdict", "addressed_by", "retire_if_same_verdict"}:
            current[key] = _strip_yaml_scalar(value)
    if current:
        requirements.append(current)
    return requirements


def _path_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for exp_id, filename in SOURCE_FILES.items():
        rel_path = f"results/{filename}"
        records.append(
            {"experiment_id": f"exp{exp_id}", "path": rel_path, "exists": (root / rel_path).exists()}
        )
    records.extend({"path": rel_path, "exists": (root / rel_path).exists()} for rel_path in CONTEXT_FILES)
    return records


def _load_sources(root: Path) -> tuple[dict[int, dict[str, Any]], set[int]]:
    sources: dict[int, dict[str, Any]] = {}
    missing: set[int] = set()
    for exp_id, filename in SOURCE_FILES.items():
        payload = _read_json(root / "results" / filename)
        if payload is None:
            missing.add(exp_id)
        else:
            sources[exp_id] = payload
    return sources, missing


def _docs_reconciliation_needed(root: Path) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if "experiment_1351_104_carryforward_artifact_integrity_audit" not in _read_text(
        root / "ops" / "changelog.md"
    ):
        reasons.append("ops/changelog.md has no exp1351 entry")
    if "experiment_1351_104_carryforward_artifact_integrity_audit" not in _read_text(
        root / "ops" / "status.md"
    ):
        reasons.append("ops/status.md has no exp1351 handoff entry")
    if "REQ-REPORT-030" not in _read_text(
        root / "openspec" / "capabilities" / "research-reporting" / "spec.md"
    ):
        reasons.append("openspec research-reporting spec lacks REQ-REPORT-030")
    if "REQ-REPORT-030" not in _read_text(root / "_bmad" / "traceability.md"):
        reasons.append("_bmad/traceability.md has no REQ-REPORT-030 trace")
    return bool(reasons), reasons


def build_artifact(
    *,
    sources: Mapping[int, Mapping[str, Any]],
    missing_source_ids: set[int],
    artifact_path_records: Sequence[Mapping[str, Any]],
    prior_failure_requirements: Sequence[Mapping[str, Any]],
    docs_reconciliation_needed: bool,
    docs_reconciliation_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """REQ-REPORT-030: build the terminal handoff audit from observed source files."""

    gates_open, gates_closed = _gate_state(sources, missing_source_ids)
    terminal_certificate_required = not _has_valid_terminal_certificate(sources.get(1340))
    honest_verdict = (
        "handoff_state_missing_exp1340_terminal_certificate_semantic_scheduler_dvi_grpo_closed"
        if terminal_certificate_required
        else "handoff_state_terminal_certificate_present_downstream_gates_follow_source_evidence"
    )
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "milestone": MILESTONE,
        "artifact_metadata": {
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "source_experiments": [f"exp{exp_id}" for exp_id in SOURCE_FILES],
        },
        "status": "complete",
        "artifact_paths_checked": [dict(record) for record in artifact_path_records],
        "missing_artifacts": _missing_artifacts(missing_source_ids),
        "stale_or_blocked_artifacts": _stale_or_blocked_artifacts(sources),
        "gates_open": gates_open,
        "gates_closed": gates_closed,
        "prior_failure_requirements": [dict(item) for item in prior_failure_requirements],
        "docs_reconciliation_needed": docs_reconciliation_needed,
        "docs_reconciliation_reasons": list(docs_reconciliation_reasons),
        "terminal_certificate_required": terminal_certificate_required,
        "honest_verdict": honest_verdict,
    }


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-030: write bootstrap, read context, and persist the final artifact."""

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)
    sources, missing_source_ids = _load_sources(root_path)
    docs_needed, docs_reasons = _docs_reconciliation_needed(root_path)
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=missing_source_ids,
        artifact_path_records=_path_records(root_path),
        prior_failure_requirements=extract_prior_failure_requirements(
            _read_text(root_path / "research-roadmap.yaml")
        ),
        docs_reconciliation_needed=docs_needed,
        docs_reconciliation_reasons=docs_reasons,
    )
    return _write_json(out, artifact)
