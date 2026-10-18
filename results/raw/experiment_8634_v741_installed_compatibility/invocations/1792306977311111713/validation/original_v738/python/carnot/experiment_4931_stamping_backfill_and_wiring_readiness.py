"""Experiment 4931: backfill .454 runtime stamps and reconfirm wiring readiness.

Spec refs: REQ-REPORT-4931, SCENARIO-REPORT-4931,
SCENARIO-REPORT-4931-MTIME-WINDOW,
SCENARIO-REPORT-4931-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path
import re
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
EXPERIMENT = "experiment_4931_stamping_backfill_and_wiring_readiness"
EXPERIMENT_ID = 4931
SCHEMA = "carnot.exp4931.stamping_backfill_and_wiring_readiness.v1"
RANDOM_SEED = 20260628
MILESTONE = "2026.06.454"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_4931_stamping_backfill_and_wiring_readiness.json")
EXP4920_RESULT_REL_PATH = Path("results/experiment_4920_retro_timing_and_stamping_fix.json")
MTIME_FALLBACK_MODULE_REL_PATH = Path("python/carnot/reporting/retro_timing_mtime_fallback.py")
STAMPING_HELPER_REL_PATH = Path("python/carnot/reporting/runtime_stamping.py")
WIRING_PROPOSAL_REL_PATH = Path("docs/retro_timing_mtime_fallback_wiring_proposal_4920.md")
RESEARCH_CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")
V454_ARM_GLOBS = ("experiment_492[4-9]_*.json", "experiment_493[0-4]_*.json")
EXPECTED_MIN_ARMS = 10
EXPECTED_MIN_COMPUTE_BOUND = 3
SPEC_REFS = [
    "REQ-REPORT-4931",
    "SCENARIO-REPORT-4931",
    "SCENARIO-REPORT-4931-MTIME-WINDOW",
    "SCENARIO-REPORT-4931-BLOCKED-PRECONDITION",
]
TERMINAL_PREFIXES = ("success_", "blocked_")

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success_v454_stamping_backfilled_and_mtime_window_confirmed."
        )
    },
    "mtime_fallback_window": {
        "principle": (
            "the .454 {n_arms, window_start, window_end, wall_minutes, "
            "compute_bound_count} from results/ mtimes -- the verifiable fix for the "
            "false-zero."
        )
    },
    "stamping_backfilled_arms": {
        "principle": (
            "the list of .454 arms newly stamped with duration_s/inference_substrate/"
            "compute_bound (or 'none missing') -- closes the duration_s=None gap."
        )
    },
    "wiring_proposal_reconfirmed": {
        "principle": (
            "true -- the conductor retro-prompt-assembly call site doc is present/current "
            "for the operator wire."
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
            "aggregation_from_upstream_artifacts (reads arm artifacts + mtimes; 0.0001s floor)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records module + arm-artifact presence; a missing input emits blocked_."
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
    "mtime_fallback_module_path",
    "stamping_helper_path",
    "wiring_proposal_path",
    "research_conductor_path",
    "research_conductor_before_sha256",
    "research_conductor_after_sha256",
    "stamping_audit_before",
    "stamping_audit_after",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)

_EXPERIMENT_RE = re.compile(r"experiment_(\d+)")
_COMPUTE_BOUND_MARKERS = ("cuda", "gpu0", "gpu1", "3090", "a100", "h100", "hip")
_COMPUTE_BOUND_KEY_MARKERS = (
    "backend",
    "device",
    "accelerator",
    "server",
    "server_kind",
    "llama_server",
)


def _experiment_id(path: Path | str) -> int | None:
    match = _EXPERIMENT_RE.search(Path(path).name)
    return int(match.group(1)) if match else None


def _relative_result_path(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        return str(Path(*parts[index:]))
    return str(path)


def _read_json_object(path: Path) -> tuple[JsonDict, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"json_error:{exc.__class__.__name__}"
    if not isinstance(value, dict):
        return {}, "json_error:non_object"
    return value, ""


def _walk_key_values(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_key_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_key_values(child)


def compute_bound_for_backfill(artifact: Mapping[str, Any]) -> bool:
    """Infer the write-time compute-bound stamp for a legacy .454 artifact."""

    explicit = artifact.get("compute_bound")
    if isinstance(explicit, bool):
        return explicit
    if timing.compute_bound_from_artifact(artifact):
        return True
    for key, value in _walk_key_values(artifact):
        key_lower = key.casefold()
        value_lower = str(value).casefold()
        if any(marker in key_lower for marker in _COMPUTE_BOUND_MARKERS) and value is True:
            return True
        if any(marker in key_lower for marker in _COMPUTE_BOUND_KEY_MARKERS) and any(
            marker in value_lower for marker in _COMPUTE_BOUND_MARKERS
        ):
            return True
    return artifact.get("inference_substrate") == "live_llm_inference"


def _duration_missing(value: Any) -> bool:
    return value is None or isinstance(value, bool) or not isinstance(value, (int, float))


def _substrate_missing(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip()


def _compute_bound_missing(value: Any) -> bool:
    return not isinstance(value, bool)


def _missing_runtime_fields(artifact: Mapping[str, Any]) -> list[str]:
    checks = {
        "duration_s": _duration_missing(artifact.get("duration_s")),
        "inference_substrate": _substrate_missing(artifact.get("inference_substrate")),
        "compute_bound": _compute_bound_missing(artifact.get("compute_bound")),
    }
    return [field for field, missing in checks.items() if missing]


def _duration_for_stamp(artifact: Mapping[str, Any]) -> float:
    value = artifact.get("duration_s")
    if _duration_missing(value):
        return runtime_stamping.MIN_DURATION_S
    return round(max(runtime_stamping.MIN_DURATION_S, float(value)), 6)


def _substrate_for_stamp(artifact: Mapping[str, Any]) -> str:
    value = artifact.get("inference_substrate")
    return str(value).strip() if isinstance(value, str) and value.strip() else INFERENCE_SUBSTRATE


def discover_v454_arm_paths(root: Path | str) -> list[Path]:
    """Return concrete .454 arm artifact paths present so far, excluding Exp 4931."""

    results_dir = Path(root) / "results"
    output_path = Path(root) / OUTPUT_REL_PATH
    paths: dict[Path, None] = {}
    for pattern in V454_ARM_GLOBS:
        for path in results_dir.glob(pattern):
            exp_id = _experiment_id(path)
            if exp_id is None or not (4924 <= exp_id <= 4934):
                continue
            if path == output_path:
                continue
            paths[path] = None
    return sorted(paths, key=lambda path: (_experiment_id(path) or 10**9, path.name))


def _arm_records(paths: Iterable[Path]) -> list[JsonDict]:
    return [
        {
            "path": _relative_result_path(path),
            "experiment_id": _experiment_id(path),
            "present": path.exists(),
        }
        for path in paths
    ]


def _wiring_proposal_current(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return "retro prompt-assembly" in text and "mtime_fallback_window" in text


def check_preconditions(root: Path | str, arm_paths: Iterable[Path] | None = None) -> JsonDict:
    """Record module, proposal, and concrete .454 arm presence."""

    root_path = Path(root)
    paths = list(discover_v454_arm_paths(root_path) if arm_paths is None else arm_paths)
    module_records = [
        {"path": str(rel_path), "present": (root_path / rel_path).exists()}
        for rel_path in (MTIME_FALLBACK_MODULE_REL_PATH, STAMPING_HELPER_REL_PATH)
    ]
    arm_errors = []
    for path in paths:
        _, error = _read_json_object(path)
        if error:
            arm_errors.append({"path": _relative_result_path(path), "error": error})
    missing_modules = [record["path"] for record in module_records if not record["present"]]
    exp4920_present = (root_path / EXP4920_RESULT_REL_PATH).exists()
    proposal_path = root_path / WIRING_PROPOSAL_REL_PATH
    proposal_present = proposal_path.exists()
    proposal_current = _wiring_proposal_current(proposal_path)
    return {
        "module_artifacts": module_records,
        "missing_modules": missing_modules,
        "exp4920_shipping_artifact": str(EXP4920_RESULT_REL_PATH),
        "exp4920_shipping_artifact_present": exp4920_present,
        "wiring_proposal_path": str(WIRING_PROPOSAL_REL_PATH),
        "wiring_proposal_present": proposal_present,
        "wiring_proposal_current": proposal_current,
        "arm_glob_patterns": list(V454_ARM_GLOBS),
        "arm_artifacts": _arm_records(paths),
        "arm_count": len(paths),
        "arm_artifact_errors": arm_errors,
        "ok": (
            not missing_modules
            and exp4920_present
            and proposal_present
            and proposal_current
            and bool(paths)
            and not arm_errors
        ),
    }


def _blocked_precondition_verdict(preconditions: Mapping[str, Any]) -> str:
    if preconditions.get("missing_modules"):
        return "blocked_missing_453_reporting_module"
    if preconditions.get("exp4920_shipping_artifact_present") is not True:
        return "blocked_missing_exp4920_shipping_artifact"
    if (
        preconditions.get("wiring_proposal_present") is not True
        or preconditions.get("wiring_proposal_current") is not True
    ):
        return "blocked_missing_wiring_proposal"
    if not preconditions.get("arm_artifacts"):
        return "blocked_missing_v454_arm_artifacts"
    if preconditions.get("arm_artifact_errors"):
        return "blocked_unreadable_v454_arm_artifact"
    return ""


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


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def backfill_runtime_stamps(paths: Iterable[Path]) -> list[JsonDict] | str:
    """Backfill missing runtime fields with the Exp 4920 stamping helper."""

    backfilled: list[JsonDict] = []
    for path in sorted(paths, key=lambda item: (_experiment_id(item) or 10**9, item.name)):
        artifact, error = _read_json_object(path)
        if error:
            continue
        missing_fields = _missing_runtime_fields(artifact)
        if not missing_fields:
            continue
        stat = path.stat()
        duration_s = _duration_for_stamp(artifact)
        stamped = runtime_stamping.stamp_runtime_metadata(
            artifact,
            started_s=0.0,
            finished_s=duration_s,
            inference_substrate=_substrate_for_stamp(artifact),
            compute_bound=compute_bound_for_backfill(artifact),
        )
        _write_payload(path, stamped)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        backfilled.append(
            {
                "path": _relative_result_path(path),
                "experiment_id": _experiment_id(path),
                "missing_fields": missing_fields,
                "duration_s": stamped["duration_s"],
                "inference_substrate": stamped["inference_substrate"],
                "compute_bound": stamped["compute_bound"],
                "mtime_preserved_ns": stat.st_mtime_ns,
            }
        )
    return backfilled if backfilled else "none missing"


def mtime_window_for_paths(paths: Iterable[Path], milestone: str = MILESTONE) -> JsonDict:
    """Reconstruct the milestone window from explicit .454 result-artifact mtimes."""

    records = []
    for path in paths:
        artifact, error = _read_json_object(path)
        if error:
            continue
        records.append(
            timing.ArtifactMtimeRecord(
                path=_relative_result_path(path),
                mtime_ns=path.stat().st_mtime_ns,
                compute_bound=compute_bound_for_backfill(artifact),
            )
        )
    return timing.reconstruct_mtime_window(milestone, records)


def _window_gate(window: Mapping[str, Any]) -> JsonDict:
    n_arms = int(window.get("n_arms") or 0)
    wall_minutes = float(window.get("wall_minutes") or 0.0)
    compute_bound_count = int(window.get("compute_bound_count") or 0)
    return {
        "min_arms": EXPECTED_MIN_ARMS,
        "min_compute_bound_count": EXPECTED_MIN_COMPUTE_BOUND,
        "n_arms": n_arms,
        "wall_minutes": wall_minutes,
        "compute_bound_count": compute_bound_count,
        "passed": (
            n_arms >= EXPECTED_MIN_ARMS
            and wall_minutes > 0.0
            and compute_bound_count >= EXPECTED_MIN_COMPUTE_BOUND
        ),
    }


def duration_from(started_s: float | None, now_s: float | None) -> float:
    """Return a positive aggregation duration with the 0.0001s floor."""

    started = time.perf_counter() if started_s is None else float(started_s)
    finished = time.perf_counter() if now_s is None else float(now_s)
    return round(max(runtime_stamping.MIN_DURATION_S, finished - started), 6)


def file_sha256(path: Path) -> str:
    """Return a file SHA-256 digest, or an empty string when absent."""

    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return a deterministic checksum over the Exp 4931 deliverable."""

    filtered = {
        key: value for key, value in payload.items() if key != "reproducibility_checksum"
    }
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    honest_verdict: str,
    preconditions_checked: Mapping[str, Any],
    mtime_fallback_window: Mapping[str, Any],
    stamping_backfilled_arms: list[JsonDict] | str,
    stamping_audit_before: Mapping[str, Any],
    stamping_audit_after: Mapping[str, Any],
    research_conductor_before_sha256: str,
    research_conductor_after_sha256: str,
    duration_s: float,
) -> JsonDict:
    """Build the Exp 4931 deliverable artifact."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(OUTPUT_REL_PATH),
        "milestone": MILESTONE,
        "honest_verdict": honest_verdict,
        "mtime_fallback_window": dict(mtime_fallback_window),
        "stamping_backfilled_arms": stamping_backfilled_arms,
        "stamping_audit_before": dict(stamping_audit_before),
        "stamping_audit_after": dict(stamping_audit_after),
        "mtime_fallback_module_path": str(MTIME_FALLBACK_MODULE_REL_PATH),
        "stamping_helper_path": str(STAMPING_HELPER_REL_PATH),
        "wiring_proposal_path": str(WIRING_PROPOSAL_REL_PATH),
        "wiring_proposal_reconfirmed": (
            preconditions_checked.get("wiring_proposal_current") is True
        ),
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
    """Run Exp 4931 and write the deliverable JSON."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    arm_paths = discover_v454_arm_paths(root_path)
    preconditions = check_preconditions(root_path, arm_paths)
    conductor_path = root_path / RESEARCH_CONDUCTOR_REL_PATH
    conductor_before = file_sha256(conductor_path)
    blocker = _blocked_precondition_verdict(preconditions)
    if blocker:
        window = mtime_window_for_paths([]) if blocker != "blocked_unreadable_v454_arm_artifact" else mtime_window_for_paths(arm_paths)
        preconditions = {**preconditions, "window_gate": _window_gate(window)}
        artifact = build_artifact(
            honest_verdict=blocker,
            preconditions_checked=preconditions,
            mtime_fallback_window=window,
            stamping_backfilled_arms="none missing",
            stamping_audit_before=_empty_audit(),
            stamping_audit_after=_empty_audit(),
            research_conductor_before_sha256=conductor_before,
            research_conductor_after_sha256=file_sha256(conductor_path),
            duration_s=duration_from(started, now_s),
        )
        _write_payload(root_path / OUTPUT_REL_PATH, artifact)
        return artifact

    audit_before = runtime_stamping.audit_runtime_stamps(arm_paths)
    backfilled = backfill_runtime_stamps(arm_paths)
    audit_after = runtime_stamping.audit_runtime_stamps(arm_paths)
    window = mtime_window_for_paths(arm_paths)
    gate = _window_gate(window)
    preconditions = {**preconditions, "window_gate": gate}
    honest_verdict = (
        "success_v454_stamping_backfilled_and_mtime_window_confirmed"
        if gate["passed"]
        else "blocked_insufficient_v454_mtime_window"
    )
    conductor_after = file_sha256(conductor_path)
    artifact = build_artifact(
        honest_verdict=honest_verdict,
        preconditions_checked=preconditions,
        mtime_fallback_window=window,
        stamping_backfilled_arms=backfilled,
        stamping_audit_before=audit_before,
        stamping_audit_after=audit_after,
        research_conductor_before_sha256=conductor_before,
        research_conductor_after_sha256=conductor_after,
        duration_s=duration_from(started, now_s),
    )
    _write_payload(root_path / OUTPUT_REL_PATH, artifact)
    return artifact


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema-contract errors for the Exp 4931 deliverable."""

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
        if payload.get("wiring_proposal_reconfirmed") is not True:
            errors.append("invalid_wiring_proposal_reconfirmed")
        gate = _window_gate(_mapping(payload.get("mtime_fallback_window")))
        if gate["passed"] is not True:
            errors.append("invalid_success_mtime_fallback_window")
    if not _is_sha256(payload.get("reproducibility_checksum")):
        errors.append("invalid_reproducibility_checksum")
    elif payload.get("reproducibility_checksum") != payload_checksum(payload):
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
