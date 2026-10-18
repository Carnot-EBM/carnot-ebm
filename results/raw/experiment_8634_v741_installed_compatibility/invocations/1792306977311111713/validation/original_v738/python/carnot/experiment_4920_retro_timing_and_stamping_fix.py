"""Experiment 4920: ship retro timing mtime fallback and runtime-stamp audit.

Spec refs: REQ-REPORT-4920, SCENARIO-REPORT-4920,
SCENARIO-REPORT-4920-STAMPING-AUDIT,
SCENARIO-REPORT-4920-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_ROOT = _REPO_ROOT / "python"
if str(_PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(_PYTHON_ROOT))  # pragma: no cover - direct script guard.

from carnot.reporting import retro_timing_mtime_fallback as timing
from carnot.reporting import runtime_stamping


JsonDict = dict[str, Any]
REPO_ROOT = _REPO_ROOT
EXPERIMENT = "experiment_4920_retro_timing_and_stamping_fix"
EXPERIMENT_ID = 4920
SCHEMA = "carnot.exp4920.retro_timing_and_stamping_fix.v1"
RANDOM_SEED = 20260628
MILESTONE = "2026.06.452"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_4920_retro_timing_and_stamping_fix.json")
RETRO_REL_PATH = Path("results/operational_retro_2026_06_452.json")
MTIME_FALLBACK_MODULE_REL_PATH = Path("python/carnot/reporting/retro_timing_mtime_fallback.py")
STAMPING_HELPER_REL_PATH = Path("python/carnot/reporting/runtime_stamping.py")
WIRING_PROPOSAL_REL_PATH = Path("docs/retro_timing_mtime_fallback_wiring_proposal_4920.md")
RESEARCH_CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")
V452_ARM_REL_PATHS = (
    Path("results/experiment_4902_archive_451_activate_452.json"),
    Path("results/experiment_4903_env_grounded_location_pruned_search.json"),
    Path("results/experiment_4904_latent_action_interface.json"),
    Path("results/experiment_4905_levelup_attempt.json"),
    Path("results/experiment_4906_self_play_verifier_checkpoint.json"),
    Path("results/experiment_4907_heldout_first_win_readiness.json"),
    Path("results/experiment_4908_env_grounded_search_audit.json"),
    Path("results/experiment_4909_submission_package_harden.json"),
    Path("results/experiment_4910_kv260_continuity.json"),
    Path("results/experiment_4911_sota_ingestion_v453_frontier.json"),
    Path("results/experiment_4912_capstone_v452.json"),
)

SPEC_REFS = [
    "REQ-REPORT-4920",
    "SCENARIO-REPORT-4920",
    "SCENARIO-REPORT-4920-STAMPING-AUDIT",
    "SCENARIO-REPORT-4920-BLOCKED-PRECONDITION",
]
TERMINAL_PREFIXES = ("success_", "blocked_")

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success_retro_timing_mtime_fallback_and_stamping_shipped."
        )
    },
    "mtime_fallback_window": {
        "principle": (
            "the .452 {n_arms, window_start, window_end, wall_minutes, "
            "compute_bound_count} reconstructed from results/ mtimes -- the "
            "verifiable fix for the ~79-milestone false-zero."
        )
    },
    "mtime_fallback_module_path": {
        "principle": (
            "the standalone module path (operator wires it into the conductor's retro "
            "prompt-assembly)."
        )
    },
    "stamping_helper_path": {
        "principle": (
            "the write-time duration_s/inference_substrate/compute_bound stamping helper."
        )
    },
    "stamping_audit_missing_duration": {
        "principle": (
            "the list of .452 arms missing duration_s (>=exp4905, exp4906) -- closes "
            "the disk-archaeology gap."
        )
    },
    "wiring_proposal_path": {
        "principle": (
            "the proposed conductor call site (operator wires; the experiment does NOT "
            "edit research_conductor.py)."
        )
    },
    "research_conductor_modified": {
        "principle": (
            "false -- Public Documentation Discipline: the conductor is operator-wired, "
            "not autonomously edited."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (reads retro + arm mtimes; 0.0001s "
            "floor)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records retro + arm-artifact presence; a missing input emits blocked_."
        )
    },
}

REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "milestone",
    "stamping_audit",
    "wiring_proposal_written",
    "research_conductor_path",
    "research_conductor_before_sha256",
    "research_conductor_after_sha256",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)


def duration_from(started_s: float | None, now_s: float | None) -> float:
    """Return a positive duration with the aggregation floor."""

    started = time.perf_counter() if started_s is None else float(started_s)
    finished = time.perf_counter() if now_s is None else float(now_s)
    return round(max(runtime_stamping.MIN_DURATION_S, finished - started), 6)


def file_sha256(path: Path) -> str:
    """Return a file SHA-256 digest, or an empty string when absent."""

    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return a deterministic checksum over the deliverable payload."""

    filtered = {
        key: value for key, value in payload.items() if key != "reproducibility_checksum"
    }
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _empty_audit() -> JsonDict:
    return {
        "scanned_count": 0,
        "missing_by_field": {
            "duration_s": [],
            "inference_substrate": [],
            "compute_bound": [],
        },
        "missing_any": [],
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    """Record the `.452` retro and arm artifact presence checks."""

    root_path = Path(root)
    arm_records = [
        {"path": str(rel_path), "present": (root_path / rel_path).exists()}
        for rel_path in V452_ARM_REL_PATHS
    ]
    missing_arms = [record["path"] for record in arm_records if record["present"] is not True]
    retro_present = (root_path / RETRO_REL_PATH).exists()
    return {
        "retro_artifact": str(RETRO_REL_PATH),
        "retro_present": retro_present,
        "arm_artifacts": arm_records,
        "missing_arm_artifacts": missing_arms,
        "ok": retro_present and not missing_arms,
    }


def _blocked_verdict(preconditions_checked: Mapping[str, Any]) -> str:
    if preconditions_checked.get("retro_present") is not True:
        return "blocked_missing_operational_retro"
    if preconditions_checked.get("missing_arm_artifacts"):
        return "blocked_missing_v452_arm_artifact"
    return ""


def write_wiring_proposal(root: Path | str, window: Mapping[str, Any], audit: Mapping[str, Any]) -> None:
    """Write the operator-facing conductor wiring proposal."""

    root_path = Path(root)
    missing_duration = [
        str(row.get("path", ""))
        for row in dict(audit.get("missing_by_field", {})).get("duration_s", [])
    ]
    text = f"""# Exp 4920 Retro Timing Mtime Fallback Wiring Proposal

## Proposed Call Site

Operator wiring belongs in `scripts/research_conductor.py`, in the operational
retro prompt-assembly path that builds the TIMING DATA block. After the existing
milestone-scoped commit detector returns its experiment/wall-minute/compute
counts, and before the retro prompt interpolates that TIMING DATA block, call
`carnot.reporting.retro_timing_mtime_fallback.mtime_fallback_window(results_dir,
milestone)`.
The operator should assemble that fallback subsection into the retro prompt
beside the detector output so both sources stay auditable.

Use the fallback only when the detector reports a false-zero shape such as
`0 experiments / 0 wall-minutes / 0 compute-bound` while milestone result
artifacts exist. The fallback output should populate an explicit
`artifact_mtime_fallback` subsection rather than overwriting the locked detector
fields.

## `.452` Evidence

- Reconstructed arms: {window.get("n_arms")}
- Reconstructed window: {window.get("window_start")} to {window.get("window_end")}
- Wall minutes: {window.get("wall_minutes")}
- Compute-bound count from legacy GPU backend evidence: {window.get("compute_bound_count")}
- Duration backfill list: {", ".join(missing_duration)}

## Public Documentation Discipline

This experiment ships the standalone module and proposal only. It does not edit
`scripts/research_conductor.py`; the operator wires the call site.
"""
    target = root_path / WIRING_PROPOSAL_REL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def build_artifact(
    *,
    root: Path,
    honest_verdict: str,
    preconditions_checked: Mapping[str, Any],
    mtime_fallback_window: Mapping[str, Any],
    stamping_audit: Mapping[str, Any],
    wiring_proposal_written: bool,
    research_conductor_before_sha256: str,
    research_conductor_after_sha256: str,
    duration_s: float,
) -> JsonDict:
    """Build the Exp 4920 deliverable artifact."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": str(OUTPUT_REL_PATH),
        "milestone": MILESTONE,
        "honest_verdict": honest_verdict,
        "mtime_fallback_window": dict(mtime_fallback_window),
        "mtime_fallback_module_path": str(MTIME_FALLBACK_MODULE_REL_PATH),
        "stamping_helper_path": str(STAMPING_HELPER_REL_PATH),
        "stamping_audit": dict(stamping_audit),
        "stamping_audit_missing_duration": list(
            dict(stamping_audit.get("missing_by_field", {})).get("duration_s", [])
        ),
        "wiring_proposal_path": str(WIRING_PROPOSAL_REL_PATH),
        "wiring_proposal_written": wiring_proposal_written,
        "research_conductor_path": str(RESEARCH_CONDUCTOR_REL_PATH),
        "research_conductor_before_sha256": research_conductor_before_sha256,
        "research_conductor_after_sha256": research_conductor_after_sha256,
        "research_conductor_modified": (
            research_conductor_before_sha256 != research_conductor_after_sha256
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(runtime_stamping.MIN_DURATION_S, float(duration_s)), 6),
        "random_seed": RANDOM_SEED,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def run(
    *,
    root: Path | str = REPO_ROOT,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Run Exp 4920 and write the deliverable JSON."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    preconditions = check_preconditions(root_path)
    blocker = _blocked_verdict(preconditions)
    conductor_path = root_path / RESEARCH_CONDUCTOR_REL_PATH
    conductor_before = file_sha256(conductor_path)
    if blocker:
        artifact = build_artifact(
            root=root_path,
            honest_verdict=blocker,
            preconditions_checked=preconditions,
            mtime_fallback_window=timing.reconstruct_mtime_window(MILESTONE, []),
            stamping_audit=_empty_audit(),
            wiring_proposal_written=False,
            research_conductor_before_sha256=conductor_before,
            research_conductor_after_sha256=file_sha256(conductor_path),
            duration_s=duration_from(started, now_s),
        )
        write_payload(root_path / OUTPUT_REL_PATH, artifact)
        return artifact

    window = timing.mtime_fallback_window(root_path / "results", MILESTONE)
    audit = runtime_stamping.audit_runtime_stamps(
        root_path / rel_path for rel_path in V452_ARM_REL_PATHS
    )
    write_wiring_proposal(root_path, window, audit)
    conductor_after = file_sha256(conductor_path)
    artifact = build_artifact(
        root=root_path,
        honest_verdict="success_retro_timing_mtime_fallback_and_stamping_shipped",
        preconditions_checked=preconditions,
        mtime_fallback_window=window,
        stamping_audit=audit,
        wiring_proposal_written=True,
        research_conductor_before_sha256=conductor_before,
        research_conductor_after_sha256=conductor_after,
        duration_s=duration_from(started, now_s),
    )
    write_payload(root_path / OUTPUT_REL_PATH, artifact)
    return artifact


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _duration_missing_ids(payload: Mapping[str, Any]) -> set[int]:
    rows = payload.get("stamping_audit_missing_duration")
    if not isinstance(rows, list):
        return set()
    return {
        int(row.get("experiment_id"))
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("experiment_id"), int)
    }


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema-contract errors for the Exp 4920 deliverable."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    principles = _mapping(payload.get("field_principles"))
    for field, principle in FIELD_PRINCIPLES.items():
        if _mapping(principles.get(field)).get("principle") != principle["principle"]:
            errors.append(f"missing_principle:{field}")
    if payload.get("research_conductor_modified") is not False:
        errors.append("invalid_research_conductor_modified")
    if isinstance(verdict, str) and verdict.startswith("success_"):
        window = _mapping(payload.get("mtime_fallback_window"))
        if (
            window.get("n_arms", 0) < 11
            or float(window.get("wall_minutes") or 0.0) <= 0.0
            or window.get("compute_bound_count", 0) < 3
        ):
            errors.append("invalid_mtime_fallback_window")
        if not {4905, 4906}.issubset(_duration_missing_ids(payload)):
            errors.append("invalid_stamping_audit_missing_duration")
    if not _is_sha256(payload.get("reproducibility_checksum")):
        errors.append("invalid_reproducibility_checksum")
    return errors


def main(root: Path | str = REPO_ROOT) -> int:
    """Run the workflow and print the deliverable path."""

    artifact = run(root=Path(root))
    errors = validate_artifact(artifact)
    print(Path(root) / OUTPUT_REL_PATH)
    if errors:
        print("validation_errors=" + ",".join(errors))
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - direct CLI entrypoint.
    raise SystemExit(main())
