"""Exp 5349: archive .487 and emit the .488 transition artifact.

Spec refs: REQ-REPORT-5349, SCENARIO-REPORT-5349,
SCENARIO-REPORT-5349-BLOCKED-NEXT-ROADMAP.

This module is a transition receipt. It reads the .487 capstone and the .488
roadmap preconditions, then writes one JSON artifact. It deliberately reports
missing or mismatched roadmap-next state instead of repairing it, because this
task is meant to preserve what happened without hand-editing the active roadmap
or conductor.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5349_archive_487_activate_488.json")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5348_capstone_v487.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5349_archive_487_activate_488"
EXPERIMENT_ID = "exp5349-archive-487-activate-488"
ARCHIVED_MILESTONE = "2026.07.487"
ACTIVATED_MILESTONE = "2026.07.488"
SCHEMA = "carnot.experiment_5349_archive_487_activate_488.v1"
RANDOM_SEED = 5349
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked_")

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Stable id ties the artifact to this roadmap task.",
    "milestone": "Prevents cross-milestone evidence laundering.",
    "status": "Lets the conductor distinguish completed, blocked, and partial work.",
    "honest_verdict": (
        "Terminal prefix `complete:` or `blocked_` prevents ambiguous reconciliation."
    ),
    "inference_substrate": "This is aggregation from upstream artifacts, not fresh inference.",
    "archived_milestone": "Records the source milestone being closed.",
    "activated_milestone": "Records the destination milestone being staged.",
    "v487_capstone_verdict": "Carries forward the actual capstone truth.",
    "roadmap_next_present": (
        "Bare boolean guard against activation without a pre-staged roadmap."
    ),
    "milestone_doc_present": "Bare boolean guard against roadmap/doc drift.",
    "active_roadmap_modified": (
        "Bare boolean must remain false because activation is conductor-owned."
    ),
    "conductor_modified": "Bare boolean must remain false by operator instruction.",
    "cited_upstream_artifacts": (
        "Makes the transition auditable without re-running experiments."
    ),
    "preconditions_checked": (
        "Records which files/resources existed before writing the artifact."
    ),
}

CAPSTONE_PROVES_PRINCIPLE = (
    "Summarizes only fields present in the .487 capstone so this transition cannot turn "
    "flagged, bounded, no-speedup, no-quality, or blocked evidence into a stronger claim."
)

WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "archived_milestone",
    "activated_milestone",
    "v487_capstone_verdict",
    "cited_upstream_artifacts",
    "preconditions_checked",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "spec_refs",
    "result_path",
    "field_principles",
    "duration_s",
    "random_seed",
    "v487_capstone_proves",
    "failed_preconditions",
    "honest_verdict",
    "inference_substrate",
    "archived_milestone",
    "activated_milestone",
    "v487_capstone_verdict",
    "roadmap_next_present",
    "milestone_doc_present",
    "active_roadmap_modified",
    "conductor_modified",
    "cited_upstream_artifacts",
    "preconditions_checked",
    "reproducibility_checksum",
)

SPEC_REFS = [
    "REQ-REPORT-5349",
    "SCENARIO-REPORT-5349",
    "SCENARIO-REPORT-5349-BLOCKED-NEXT-ROADMAP",
]

SOURCE_CONTEXT_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/conductor-log.md"),
    CAPSTONE_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)


def value_of(value: Any) -> Any:
    """Return the machine value from a principle-wrapped or bare artifact field."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _principled(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def path_sha256(path: Path) -> str | None:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, {"exists": True, "loadable": False, "error": "malformed_json"}
    if not isinstance(parsed, dict):
        return {}, {"exists": True, "loadable": False, "error": "not_json_object"}
    return parsed, {
        "exists": True,
        "loadable": True,
        "error": None,
        "sha256": path_sha256(path),
    }


def yaml_milestone(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, Mapping):
        return None
    milestone = parsed.get("milestone")
    return str(milestone) if milestone is not None else None


def git_path_modified(root: Path, relative_path: Path) -> bool:
    if not (root / ".git").exists():
        return False
    result = subprocess.run(
        ("git", "status", "--short", "--", str(relative_path)),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _modification_status(
    root: Path,
    relative_path: Path,
    overrides: Mapping[Path | str, bool] | None,
) -> bool:
    if overrides is None:
        return git_path_modified(root, relative_path)
    if relative_path in overrides:
        return bool(overrides[relative_path])
    return bool(overrides.get(str(relative_path), git_path_modified(root, relative_path)))


def document_contains_milestone(path: Path, milestone: str) -> bool:
    return path.exists() and milestone in path.read_text(encoding="utf-8", errors="replace")


def _as_list(value: Any) -> list[Any]:
    raw = value_of(value)
    return list(raw) if isinstance(raw, list) else []


def _as_map(value: Any) -> JsonDict:
    raw = value_of(value)
    return dict(raw) if isinstance(raw, Mapping) else {}


def capstone_verdict(capstone: JsonMap) -> str:
    verdict = value_of(capstone.get("honest_verdict", ""))
    return verdict if isinstance(verdict, str) else str(verdict)


def capstone_truth_summary(capstone: JsonMap) -> JsonDict:
    recommendation = _as_map(capstone.get("next_milestone_recommendation"))
    gates = []
    for row in _as_list(capstone.get("gate_table")):
        if not isinstance(row, Mapping):
            continue
        gates.append(
            {
                "gate": row.get("gate"),
                "ready": row.get("ready"),
                "classification": row.get("classification"),
                "claim_boundary": row.get("claim_boundary"),
                "source_artifacts": row.get("source_artifacts"),
            }
        )
    return {
        "capstone_milestone": value_of(capstone.get("milestone")),
        "capstone_status": value_of(capstone.get("status")),
        "capstone_honest_verdict": capstone_verdict(capstone),
        "capstone_inference_substrate": value_of(capstone.get("inference_substrate")),
        "runtime_clean": value_of(capstone.get("runtime_clean")),
        "structured_output_protocol_ready": value_of(
            capstone.get("structured_output_protocol_ready")
        ),
        "bounded_sota_quality_usable": value_of(capstone.get("bounded_sota_quality_usable")),
        "utility_memory_ready": value_of(capstone.get("utility_memory_ready")),
        "bounded_compressor_ready": value_of(capstone.get("bounded_compressor_ready")),
        "self_learning_scaleup_ready": value_of(capstone.get("self_learning_scaleup_ready")),
        "qstr_fixture_ready": value_of(capstone.get("qstr_fixture_ready")),
        "solver_guidance_ready": value_of(capstone.get("solver_guidance_ready")),
        "kan_constraint_bridge_ready": value_of(capstone.get("kan_constraint_bridge_ready")),
        "internal_energy_corrigendum_clean": value_of(
            capstone.get("internal_energy_corrigendum_clean")
        ),
        "hardware_speedup_claim": value_of(capstone.get("hardware_speedup_claim")),
        "capstone_active_roadmap_modified": value_of(capstone.get("active_roadmap_modified")),
        "capstone_conductor_modified": value_of(capstone.get("conductor_modified")),
        "gate_claim_boundaries": gates,
        "do_not_claim": recommendation.get("do_not_claim"),
    }


def capstone_failures(capstone: JsonMap) -> list[str]:
    if not capstone:
        return ["capstone_missing_or_unloadable"]
    summary = capstone_truth_summary(capstone)
    failures: list[str] = []
    if summary.get("capstone_milestone") != ARCHIVED_MILESTONE:
        failures.append(
            f"capstone_milestone_expected_{ARCHIVED_MILESTONE}_observed_"
            f"{summary.get('capstone_milestone')}"
        )
    if summary.get("capstone_status") != "complete":
        failures.append(f"capstone_status_expected_complete_observed_{summary.get('capstone_status')}")
    if not capstone_verdict(capstone).startswith(TERMINAL_PREFIXES):
        failures.append("capstone_honest_verdict_missing_terminal_prefix")
    return failures


def cited_upstream_artifacts(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for relative_path in SOURCE_CONTEXT_PATHS:
        path = root / relative_path
        rows.append(
            {
                "relative_path": str(relative_path),
                "exists": path.exists(),
                "sha256": path_sha256(path),
                "read_only": True,
            }
        )
    return rows


def precondition_summary(
    *,
    root: Path,
    capstone_meta: JsonMap,
    active_roadmap_modified: bool,
    conductor_modified: bool,
) -> JsonDict:
    milestone_doc_path = root / VNEXT_RELATIVE_PATH
    roadmap_next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    active_milestone = yaml_milestone(root / ROADMAP_RELATIVE_PATH)
    next_milestone = yaml_milestone(roadmap_next_path)
    milestone_doc_names_488 = document_contains_milestone(
        milestone_doc_path, ACTIVATED_MILESTONE
    )
    return {
        "capstone_present": capstone_meta.get("exists") is True,
        "capstone_loadable": capstone_meta.get("loadable") is True,
        "milestone_doc_present": milestone_doc_path.exists(),
        "milestone_doc_names_activated_milestone": milestone_doc_names_488,
        "active_roadmap_present": (root / ROADMAP_RELATIVE_PATH).exists(),
        "active_roadmap_milestone": active_milestone,
        "roadmap_next_present": roadmap_next_path.exists(),
        "roadmap_next_milestone": next_milestone,
        "roadmap_next_names_activated_milestone": next_milestone == ACTIVATED_MILESTONE,
        "active_roadmap_modified": active_roadmap_modified,
        "conductor_modified": conductor_modified,
        "no_active_roadmap_overwrite_performed": not active_roadmap_modified,
        "no_conductor_edit_performed": not conductor_modified,
        "ops_reconciliation_delegated": True,
        "checked_paths": [
            str(CAPSTONE_RELATIVE_PATH),
            str(VNEXT_RELATIVE_PATH),
            str(ROADMAP_NEXT_RELATIVE_PATH),
            str(ROADMAP_RELATIVE_PATH),
            str(CONDUCTOR_RELATIVE_PATH),
        ],
    }


def failed_preconditions(capstone_failure_rows: Sequence[str], preconditions: JsonMap) -> list[str]:
    failures = list(capstone_failure_rows)
    if preconditions.get("milestone_doc_names_activated_milestone") is not True:
        failures.append(f"milestone_doc_missing_or_mismatch_{ACTIVATED_MILESTONE}")
    if preconditions.get("roadmap_next_present") is not True:
        failures.append("roadmap_next_missing")
    elif preconditions.get("roadmap_next_names_activated_milestone") is not True:
        failures.append(
            f"roadmap_next_milestone_expected_{ACTIVATED_MILESTONE}_observed_"
            f"{preconditions.get('roadmap_next_milestone')}"
        )
    if preconditions.get("active_roadmap_present") is not True:
        failures.append("active_roadmap_missing")
    if preconditions.get("active_roadmap_modified") is True:
        failures.append("active_roadmap_modified")
    if preconditions.get("conductor_modified") is True:
        failures.append("conductor_modified")
    return failures


def build_honest_verdict(status: str, roadmap_next_present: bool) -> str:
    if status == "complete":
        return (
            "complete: .487 capstone truth recorded and .488 roadmap-next plus milestone "
            "doc observed; roadmap_next_present=true; no active roadmap overwrite or "
            "conductor edit performed."
        )
    return (
        "blocked_archive_487_activate_488: transition preconditions failed; "
        f"roadmap_next_present={str(roadmap_next_present).lower()}; no active roadmap "
        "overwrite or conductor edit was performed by this task."
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260707",
    duration_s: float | None = None,
    modification_status: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    start = time.perf_counter()
    capstone, capstone_meta = read_json_mapping(root / CAPSTONE_RELATIVE_PATH)
    active_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_status)
    conductor_dirty = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_status)
    preconditions = precondition_summary(
        root=root,
        capstone_meta=capstone_meta,
        active_roadmap_modified=active_modified,
        conductor_modified=conductor_dirty,
    )
    failures = failed_preconditions(capstone_failures(capstone), preconditions)
    status = "complete" if not failures else "blocked"
    roadmap_next_present = bool(preconditions["roadmap_next_present"])
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": _principled("experiment_id", EXPERIMENT_ID),
        "milestone": _principled("milestone", ACTIVATED_MILESTONE),
        "status": _principled("status", status),
        "run_date": run_date,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start), 6
        ),
        "random_seed": RANDOM_SEED,
        "v487_capstone_proves": {
            "principle": CAPSTONE_PROVES_PRINCIPLE,
            "value": capstone_truth_summary(capstone) if capstone else {},
        },
        "failed_preconditions": failures,
        "honest_verdict": _principled(
            "honest_verdict", build_honest_verdict(status, roadmap_next_present)
        ),
        "inference_substrate": _principled("inference_substrate", INFERENCE_SUBSTRATE),
        "archived_milestone": _principled("archived_milestone", ARCHIVED_MILESTONE),
        "activated_milestone": _principled("activated_milestone", ACTIVATED_MILESTONE),
        "v487_capstone_verdict": _principled("v487_capstone_verdict", capstone_verdict(capstone)),
        "roadmap_next_present": roadmap_next_present,
        "milestone_doc_present": bool(preconditions["milestone_doc_present"]),
        "active_roadmap_modified": active_modified,
        "conductor_modified": conductor_dirty,
        "cited_upstream_artifacts": _principled(
            "cited_upstream_artifacts", cited_upstream_artifacts(root)
        ),
        "preconditions_checked": _principled("preconditions_checked", preconditions),
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(payload: JsonMap) -> None:
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if payload.get("schema") != SCHEMA:
        raise ValueError("schema mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    for field in WRAPPED_FIELDS:
        wrapped = payload[field]
        if not isinstance(wrapped, Mapping) or wrapped.get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"{field} principle mismatch")
        if "value" not in wrapped:
            raise ValueError(f"{field} missing value")
    if value_of(payload["experiment_id"]) != EXPERIMENT_ID:
        raise ValueError("experiment_id mismatch")
    if value_of(payload["milestone"]) != ACTIVATED_MILESTONE:
        raise ValueError("milestone mismatch")
    verdict = value_of(payload["honest_verdict"])
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with complete: or blocked_")
    if value_of(payload["inference_substrate"]) != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if value_of(payload["archived_milestone"]) != ARCHIVED_MILESTONE:
        raise ValueError("archived_milestone mismatch")
    if value_of(payload["activated_milestone"]) != ACTIVATED_MILESTONE:
        raise ValueError("activated_milestone mismatch")
    for field in (
        "roadmap_next_present",
        "milestone_doc_present",
        "active_roadmap_modified",
        "conductor_modified",
    ):
        if not isinstance(payload[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    if payload["active_roadmap_modified"] is not False:
        raise ValueError("active_roadmap_modified must be bare false")
    if payload["conductor_modified"] is not False:
        raise ValueError("conductor_modified must be bare false")
    preconditions = value_of(payload["preconditions_checked"])
    if not isinstance(preconditions, Mapping):
        raise ValueError("preconditions_checked must wrap a mapping")
    cited = value_of(payload["cited_upstream_artifacts"])
    if not isinstance(cited, list):
        raise ValueError("cited_upstream_artifacts must wrap a list")
    failures = payload["failed_preconditions"]
    status = value_of(payload["status"])
    if status == "complete" and failures:
        raise ValueError("complete status cannot carry failed preconditions")
    if status == "blocked" and not failures:
        raise ValueError("blocked status must carry failed preconditions")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260707",
    duration_s: float | None = None,
) -> Path:
    artifact = build_artifact(root=root, run_date=run_date, duration_s=duration_s)
    out_path = root / RESULT_RELATIVE_PATH
    write_json(out_path, artifact)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-date", default="20260707")
    args = parser.parse_args(argv)
    print(run(root=args.root, run_date=args.run_date))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
