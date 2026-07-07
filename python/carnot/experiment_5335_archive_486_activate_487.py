"""Exp 5335: archive .486 and emit the .487 transition artifact.

Spec refs: REQ-REPORT-5335, SCENARIO-REPORT-5335,
SCENARIO-REPORT-5335-BLOCKED-NEXT-ROADMAP.

This module is a transition receipt. It reads the .486 capstone and the .487
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
RESULT_RELATIVE_PATH = Path("results/experiment_5335_archive_486_activate_487.json")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5334_capstone_v486.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5335_archive_486_activate_487"
EXPERIMENT_ID = "exp5335-archive-486-activate-487"
ARCHIVED_MILESTONE = "2026.07.486"
ACTIVATED_MILESTONE = "2026.07.487"
SCHEMA = "carnot.experiment_5335_archive_486_activate_487.v1"
RANDOM_SEED = 5335
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked_")

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": (
        "Identifies Exp5335 as the .486-to-.487 transition receipt rather than a "
        "roadmap activator or new research result."
    ),
    "milestone": (
        "Binds this receipt to target milestone 2026.07.487 so downstream gates cannot "
        "confuse transition scope with the archived .486 capstone."
    ),
    "status": "Machine-readable terminal state derived from explicit transition preconditions.",
    "honest_verdict": (
        "Must start with complete: or blocked_ and preserve missing roadmap-next, flagged, "
        "bounded, and no-claim evidence without laundering it."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts because Exp5335 reads local artifacts and "
        "roadmap metadata without model, solver, or hardware execution."
    ),
    "archived_milestone": "Records the completed milestone whose capstone truth is being carried forward.",
    "activated_milestone": (
        "Records the next milestone that pre-staged roadmap files must name before "
        "activation is ready."
    ),
    "v486_capstone_verdict": (
        "Carries forward the capstone's own terminal verdict without converting it into "
        "broader quality, hardware, or certificate claims."
    ),
    "cited_upstream_artifacts": (
        "Lists every upstream artifact or roadmap file used so the transition claim is "
        "traceable to local evidence."
    ),
    "preconditions_checked": (
        "Records each capstone, roadmap-next, milestone-doc, active-roadmap, and conductor "
        "no-edit check that determines completion or blockage."
    ),
}

CAPSTONE_PROVES_PRINCIPLE = (
    "Summarizes only fields present in the .486 capstone so this transition cannot turn "
    "flagged, bounded, no-speedup, or no-headline-quality evidence into a stronger claim."
)

WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)
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
    "v486_capstone_proves",
    "failed_preconditions",
    "honest_verdict",
    "inference_substrate",
    "archived_milestone",
    "activated_milestone",
    "v486_capstone_verdict",
    "roadmap_next_present",
    "milestone_doc_present",
    "active_roadmap_modified",
    "conductor_modified",
    "cited_upstream_artifacts",
    "preconditions_checked",
    "reproducibility_checksum",
)

SPEC_REFS = [
    "REQ-REPORT-5335",
    "SCENARIO-REPORT-5335",
    "SCENARIO-REPORT-5335-BLOCKED-NEXT-ROADMAP",
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
                "source_experiments": row.get("source_experiments"),
            }
        )
    return {
        "capstone_milestone": value_of(capstone.get("milestone")),
        "capstone_status": value_of(capstone.get("status")),
        "capstone_honest_verdict": capstone_verdict(capstone),
        "runtime_stable": value_of(capstone.get("runtime_stable")),
        "sota_quality_measured": value_of(capstone.get("sota_quality_measured")),
        "rewrite_state_ready": value_of(capstone.get("rewrite_state_ready")),
        "smt_corrigendum_clean": value_of(capstone.get("smt_corrigendum_clean")),
        "context_lifecycle_ready": value_of(capstone.get("context_lifecycle_ready")),
        "certificate_self_learning_ready": value_of(
            capstone.get("certificate_self_learning_ready")
        ),
        "internal_signal_path_open": value_of(capstone.get("internal_signal_path_open")),
        "kan_localization_ready": value_of(capstone.get("kan_localization_ready")),
        "hardware_speedup_claim": value_of(capstone.get("hardware_speedup_claim")),
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
    milestone_doc_names_487 = document_contains_milestone(
        milestone_doc_path, ACTIVATED_MILESTONE
    )
    return {
        "capstone_present": capstone_meta.get("exists") is True,
        "capstone_loadable": capstone_meta.get("loadable") is True,
        "milestone_doc_present": milestone_doc_path.exists(),
        "milestone_doc_names_activated_milestone": milestone_doc_names_487,
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
            "complete: .486 capstone truth recorded and .487 roadmap-next plus milestone "
            "doc observed; roadmap_next_present=true; no active roadmap overwrite or "
            "conductor edit performed."
        )
    return (
        "blocked_archive_486_activate_487: transition preconditions failed; "
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
        "v486_capstone_proves": {
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
        "v486_capstone_verdict": _principled("v486_capstone_verdict", capstone_verdict(capstone)),
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
    for field in ("roadmap_next_present", "milestone_doc_present"):
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
